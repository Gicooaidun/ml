import argparse
import os
import glob
import sys
import traceback
import geopandas as gpd
import numpy as np
from zipfile import ZipFile
import pickle
from os.path import join
from shutil import rmtree
import torch.nn as nn
from affine import Affine
from pyproj import Transformer
import torch
sys.path.insert(1, '/scratch2/biomass_estimation/code/patches')
from helper_patches import *
from create_patches import *
import h5py

def setup_parser():
    parser = argparse.ArgumentParser(description="inference")
    parser.add_argument("--path_icesat", type=str, default="/scratch2/biomass_estimation/code/notebook/cropped_mosaic_no_nan")
    parser.add_argument("--tilenames", type=str, default="/scratch2/biomass_estimation/code/ml/inference/tile_names_inference.txt")
    parser.add_argument("--path_shp", type=str, default=os.path.join('/scratch2', 'biomass_estimation', 'code', 'notebook', 'S2_tiles_Siberia_polybox', 'S2_tiles_Siberia_all.geojson'))
    parser.add_argument("--path_s2", type=str, default="/scratch3/Siberia")
    parser.add_argument("--norm_path", type=str, default="/scratch2/biomass_estimation/code/ml/data/normalization_values.pkl")
    parser.add_argument("--model_path", type=str, default="/scratch2/biomass_estimation/code/ml/models")

    args = parser.parse_args()
    return args.path_shp, args.tilenames, args.path_s2, args.path_icesat, args.norm_path, args.model_path

class SimpleFCN(nn.Module):
    def __init__(self,
                 in_features=18,
                 channel_dims = (16, 32, 64, 128, 64, 32, 16),
                 num_outputs=1,
                 kernel_size=3,
                 stride=1):
        """
        A simple fully convolutional neural network.
        """
        super(SimpleFCN, self).__init__()
        self.relu = nn.ReLU(inplace = True)
        layers = list()
        for i in range(len(channel_dims)):
            in_channels = in_features if i == 0 else channel_dims[i-1]
            layers.append(nn.Conv2d(in_channels=in_channels, 
                                    out_channels=channel_dims[i], 
                                    kernel_size=kernel_size, stride=stride, padding=1))
            layers.append(nn.BatchNorm2d(num_features=channel_dims[i]))
            layers.append(self.relu)
        # print(layers)
        self.conv_layers = nn.Sequential(*layers)
        
        self.conv_output = nn.Conv2d(in_channels=channel_dims[-1], out_channels=num_outputs, kernel_size=1,
                                     stride=1, padding=0, bias=True)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.conv_output(x)

        return x
    
class EnsembleModel(nn.Module):
    def __init__(self, models):
        super(EnsembleModel, self).__init__()
        self.models = models

    def forward(self, x):
        if torch.cuda.is_available():
            self.models = [model.cuda() for model in self.models]
        outputs = [model(x) for model in self.models]
        mean = torch.mean(torch.stack(outputs), dim=0)
        std = torch.std(torch.stack(outputs), dim=0)
        return mean, std
    
def encode_lat_lon(lat, lon) :
    """
    Encode the latitude and longitude into sin/cosine values. We use a simple WRAP positional encoding, as 
    Mac Aodha et al. (2019).

    Args:
    - lat (float): the latitude
    - lon (float): the longitude

    Returns:
    - (lat_cos, lat_sin, lon_cos, lon_sin) (tuple): the sin/cosine values for the latitude and longitude
    """

    # The latitude goes from -90 to 90
    lat_cos, lat_sin = np.cos(np.pi * lat / 90), np.sin(np.pi * lat / 90)
    # The longitude goes from -180 to 180
    lon_cos, lon_sin = np.cos(np.pi * lon / 180), np.sin(np.pi * lon / 180)

    # Now we put everything in the [0,1] range
    lat_cos, lat_sin = (lat_cos + 1) / 2, (lat_sin + 1) / 2
    lon_cos, lon_sin = (lon_cos + 1) / 2, (lon_sin + 1) / 2

    return lat_cos, lat_sin, lon_cos, lon_sin


def encode_coords(central_lat, central_lon, patch_size, resolution = 10) :
    """ 
    This function computes the latitude and longitude of a patch, from the latitude and longitude of its central pixel.
    It then encodes these values into sin/cosine values, and scales the results to [0,1].

    Args:
    - central_lat (float): the latitude of the central pixel
    - central_lon (float): the longitude of the central pixel
    - patch_size (tuple): the size of the patch
    - resolution (int): the resolution of the patch

    Returns:
    - (lat_cos, lat_sin, lon_cos, lon_sin) (tuple): the sin/cosine values for the latitude and longitude
    """

    # Initialize arrays to store latitude and longitude coordinates

    i_indices, j_indices = np.indices(patch_size)

    # Calculate the distance offset in meters for each pixel
    offset_lat = (i_indices - patch_size[0] // 2) * resolution
    offset_lon = (j_indices - patch_size[1] // 2) * resolution

    # Calculate the latitude and longitude for each pixel
    latitudes = central_lat + (offset_lat / 6371000) * (180 / np.pi)
    longitudes = central_lon + (offset_lon / 6371000) * (180 / np.pi) / np.cos(central_lat * np.pi / 180)

    lat_cos, lat_sin, lon_cos, lon_sin = encode_lat_lon(latitudes, longitudes)

    return lat_cos, lat_sin, lon_cos, lon_sin


def normalize_data(data, norm_values, norm_strat, nodata_value = None) :
    """
    Normalize the data, according to various strategies:
    - mean_std: subtract the mean and divide by the standard deviation
    - pct: subtract the 1st percentile and divide by the 99th percentile
    - min_max: subtract the minimum and divide by the maximum

    Args:
    - data (np.array): the data to normalize
    - norm_values (dict): the normalization values
    - norm_strat (str): the normalization strategy

    Returns:
    - normalized_data (np.array): the normalized data
    """

    if norm_strat == 'mean_std' :
        mean, std = norm_values['mean'], norm_values['std']
        if nodata_value is not None :
            data = np.where(data == nodata_value, 0, (data - mean) / std)
        else : data = (data - mean) / std

    elif norm_strat == 'pct' :
        p1, p99 = norm_values['p1'], norm_values['p99']
        if nodata_value is not None :
            data = np.where(data == nodata_value, 0, (data - p1) / (p99 - p1))
        else :
            data = (data - p1) / (p99 - p1)
        data = np.clip(data, 0, 1)

    elif norm_strat == 'min_max' :
        min_val, max_val = norm_values['min'], norm_values['max']
        if nodata_value is not None :
            data = np.where(data == nodata_value, 0, (data - min_val) / (max_val - min_val))
        else:
            data = (data - min_val) / (max_val - min_val)
    
    else: 
        raise ValueError(f'Normalization strategy `{norm_strat}` is not valid.')

    return data


def normalize_bands(bands_data, norm_values, order, norm_strat, nodata_value = None) :
    """
    This function normalizes the bands data using the normalization values and strategy.

    Args:
    - bands_data (np.array): the bands data to normalize
    - norm_values (dict): the normalization values
    - order (list): the order of the bands
    - norm_strat (str): the normalization strategy
    - nodata_value (int/float): the nodata value

    Returns:
    - bands_data (np.array): the normalized bands data
    """
    normalized = {}
    for i, band in enumerate(order) :
        if band != 'SCL' and band != 'transform':
            print('normalizing ', band)
            band_norm = norm_values[band]
            # print(band_norm)
            # print(bands_data[band].shape)
            normalized[band] = normalize_data(bands_data[band], band_norm, norm_strat, nodata_value)
    
    return normalized

def load_files(path_shp, tilenames, path_s2, path_icesat):
    # Read the Sentinel-2 grid shapefile
    grid_df = gpd.read_file(path_shp, engine = 'pyogrio')

    # List all S2 tiles and their geometries
    tile_names, tile_geoms = list_s2_tiles(tilenames, grid_df, path_s2)

    s2_processed_bands = []
    s2_transforms = []
    s2_crs = []
    icesat_raws = []

    for s2_prod in tile_names:
        print(f'>> Extracting patches for product {s2_prod}.')
        # Get the filename that includes s2_prod in the path_s2 folder
        # print(glob.glob(f"{path_s2}/*{s2_prod}*.zip"))
        # exit()
        all_corresponding_s2_paths = glob.glob(f"{path_s2}/*{s2_prod}*")
        for total_s2_path in all_corresponding_s2_paths:
            # Extract the folder and the filename separately
            s2_folder_path, s2_file_name = os.path.split(total_s2_path)

            print(f'>> Found {total_s2_path}.')

            # Unzip the S2 L2A product if it hasn't been done
            total_unzipped_path = total_s2_path[:-4] + '.SAFE'
            if not os.path.exists(total_unzipped_path):
                try:
                    with ZipFile(total_s2_path, 'r') as zip_ref:
                        zip_ref.extractall(path_s2)
                except Exception as e:
                    print(f'>> Could not unzip {s2_prod}.')
                    print(e)
                    continue

            # Reproject and upsample the S2 bands            
            try: 
                transform, upsampling_shape, processed_bands, crs_2, bounds = process_S2_tile(product=s2_file_name[:-4], path_s2 = s2_folder_path)
            except IndexError:
                s2_folder_path = os.path.join(path_s2, 'scratch2', 'gsialelli', 'S2_L2A', 'Siberia')
                total_unzipped_path = os.path.join(s2_folder_path, s2_file_name[:-4] + '.SAFE')
                try: 
                    transform, upsampling_shape, processed_bands, crs_2, bounds = process_S2_tile(product=s2_file_name[:-4], path_s2 = s2_folder_path)
                except Exception as e:
                    print(f'>> Could not process product {s2_prod}.')
                    print(traceback.format_exc())
                    continue
            
            
            s2_processed_bands.append(processed_bands)
            s2_transforms.append(transform)
            s2_crs.append(crs_2)
        icesat_raw = load_BM_data(path_bm=path_icesat, tile_name=s2_prod)
        icesat_raws.append(icesat_raw)

        # Remove the unzipped S2 product
        rmtree(total_unzipped_path)
    
    return s2_processed_bands, s2_transforms, s2_crs, icesat_raws

def normalize(processed_bands, icesat_raw):
    icesat_order = []
    icesat_norm = []
    s2_order = []
    s2_bands_dict = []
    s2_indices = []

    with open(norm_path, mode = 'rb') as f:
                norm_values = pickle.load(f)
    # print("processed_bands keys ", processed_bands.keys())
    # print("norm_values[S2_bands] keys", norm_values['S2_bands'].keys()) #we don't have SCL in the normalization values so not in the model...?
    # print(norm_values.keys())

    norm_strat = "mean_std"
    for i in range(len(processed_bands)):
        icesat_order.append(sorted(list(icesat_raw[i].keys())))
        icesat_norm.append(normalize_bands(icesat_raw[i], norm_values['BM'], icesat_order[i], norm_strat, nodata_value = -9999.0))

        s2_order.append(sorted(list(processed_bands[i].keys())))
        s2_bands_dict.append(normalize_bands(processed_bands[i], norm_values['S2_bands'], s2_order[i], norm_strat, nodata_value = 0))
        s2_indices.append([s2_order[i].index(band) for band in s2_bands_dict[i]])
        return icesat_order, icesat_norm, s2_order, s2_bands_dict, s2_indices



if __name__ == "__main__":
    #get arguments
    path_shp, tilenames, path_s2, path_icesat, norm_path, model_path = setup_parser()
    #load s2 and icesat data
    s2_processed_bands, s2_transforms, s2_crs, icesat_raws = load_files(path_shp, tilenames, path_s2, path_icesat)
    #normalize the data with the normalization values
    icesat_order, icesat_norm, s2_order, s2_bands_dict, s2_indices = normalize(s2_processed_bands, icesat_raws)

    #make np arrays from dicts
    s2_bands = []
    for i in range(len(s2_bands_dict)):
        temp = np.stack([s2_bands_dict[i][key] for key in s2_bands_dict[i].keys()], axis=-1)
        temp = temp[:, :, s2_indices[i]]
        s2_bands.append(temp)

    upsampled_icesat = []
    for i in range(len(icesat_norm)):
        icesat = {}
        icesat['bm'] = upsampling_with_nans(icesat_norm[i]['bm'], s2_bands_dict[i]['B01'].shape, -9999, 3)
        icesat['std'] = upsampling_with_nans(icesat_norm[i]['std'], s2_bands_dict[i]['B01'].shape, -9999, 3)
        upsampled_icesat.append(icesat)

    sub_models = []
    for i in range(5):
        temp_model = SimpleFCN()
        temp_model.load_state_dict(torch.load(os.path.join(model_path, f'early_stopping_sub_ensemble_model_{i}.pth')))
        temp_model.eval()
        sub_models.append(temp_model)
    
    model = EnsembleModel(sub_models)   


    outputs = []
    
    for t in range(len(s2_bands)):
        fwd = Affine.from_gdal(s2_transforms[t][2], s2_transforms[t][0], s2_transforms[t][1], s2_transforms[t][5], s2_transforms[t][3], s2_transforms[t][4])
        coordinate_transformer = Transformer.from_crs(s2_crs[t], 'epsg:4326')
        outputs.append(np.zeros((icesat['bm'].shape[0], icesat['bm'].shape[1], 2)))
        print(icesat['bm'].shape)
        for i in range(7, icesat['bm'].shape[0], 15):
            for j in range(7, icesat['bm'].shape[1], 15):
                        
                data = []
                s2_temp = s2_bands[t][i-7:i+8, j-7:j+8,:]
                data.extend([s2_temp])                

                lat1, lon1 = fwd * (i, j)
                #print(lat1, lon1)
                lat2, lon2 = coordinate_transformer.transform(lat1, lon1)
                #print(lat2, lon2)
                lat_cos, lat_sin, lon_cos, lon_sin = encode_coords(lat2, lon2, (15, 15))
                data.extend([lat_cos[..., np.newaxis], lat_sin[..., np.newaxis], lon_cos[..., np.newaxis], lon_sin[..., np.newaxis]])

                icesat_temp_bm = upsampled_icesat[t]['bm'][i-7:i+8, j-7:j+8, np.newaxis]
                icesat_temp_std = upsampled_icesat[t]['std'][i-7:i+8, j-7:j+8, np.newaxis]
                data.extend([icesat_temp_bm, icesat_temp_std])
                # for i in range(len(data)):
                #     print(data[i].shape)
                # print(len(data))
                # Concatenate the data together
                data = torch.from_numpy(np.concatenate(data, axis = -1).swapaxes(-1, 0)).to(torch.float)
                if torch.cuda.is_available():
                    data = data.cuda()
                outputs[t][i-7:i+8, j-7:j+8, 0] = model(data.unsqueeze(0))[0].detach().cpu().numpy().squeeze()
                outputs[t][i-7:i+8, j-7:j+8, 1] = model(data.unsqueeze(0))[1].detach().cpu().numpy().squeeze()

    print("saving outputs")
    # Save the outputs as a .npy file
    np.save('ml/inference/inference_ensemble_all_s2.npy', outputs)
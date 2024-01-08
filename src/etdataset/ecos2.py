#!/usr/bin/env python

# -*- coding: utf-8 -*-
#
# Copyright: (c) 2023 CESBIO / Centre National d'Etudes Spatiales / Université Paul Sabatier (UT3)
#
"""
Function to create dataset from ECOSTRESS and Sentinel2
"""
import affine
import geopandas as gpd
import pandas as pd
import numpy as np
import rasterio as rio
import xarray as xr

from sensorsio import sentinel2, utils, ecostress_coll2
from sensorsio.ecostress_coll2 import Ecostress

from etdataset.logging import LoggerManager
from etdataset.indices import compute_ndvi, compute_lai_from_ndvi

logger = LoggerManager.get_logger(__name__)


ecos2_name_mapping = {
    'B2': 'blue',
    'B3': 'green',
    'B4': 'red',
    'B6': 'red_edge',
    'B8': 'nir',
    'B8A': 'nir2',
    'B11': 'swir1',
    'B12': 'swir2',
    'LST'  : 'lst',
    'EmisWB': 'emis',
}


def compute_albedo(data: xr.Dataset) -> None:  
    """
    Compute albedo
    Bonafoni and al., Albedo Retrieval From Sentinel-2 by New Narrow-to-Broadband Conversion Coefficients, 
    IEEE Geoscience and Remote Sensing Letters, 2020
    """
    data['albedo'] = 0.2266 * data.blue + 0.1236 * data.green + 0.1573 * data.red + \
                     0.3417 * data.nir + 0.1170 * data.swir1 + 0.0338 * data.swir2


def compute_seli(data: xr.Dataset) -> None:  
    """
    Compute SeLI
    Pasqualotto, N.; Delegido, J.; Van Wittenberghe, S.; Rinaldi, M.; Moreno, J. Multi-Crop 
    Green LAI Estimation with a New Simple Sentinel-2 LAI Index (SeLI). Sensors 2019, 19, 904.
    """
    #data['seli'] = np.clip((data.nir2 - data.red_edge) / 
    #                       (data.nir2 + data.red_edge + 1e-9),
    #                       -1,1) #.transpose('y','x')
    data['seli'] =(data.nir2 - data.red_edge) / (data.nir2 + data.red_edge + 1e-9)


def compute_lai_from_seli(data: xr.Dataset) -> None:  
    """
    Compute LAI
    Pasqualotto, N.; Delegido, J.; Van Wittenberghe, S.; Rinaldi, M.; Moreno, J. Multi-Crop 
    Green LAI Estimation with a New Simple Sentinel-2 LAI Index (SeLI). Sensors 2019, 19, 904.
    """
    data['lai'] = 5.405 * data["seli"] - 0.114


def create_dataset(s2_path: str,
                   eco_path:str,
                   resolution:int = 57) -> xr.Dataset:
    """
    Create dataset from sentinel2 and ecostress products
    """
    logger.debug(f"Sentinel2 path: {s2_path}")
    logger.debug(f"Ecostress path: {eco_path}")

    # Create an instance of ecostress from the product path
    eco_ds = ecostress_coll2.Ecostress(eco_path)
    
    # Read ecostress product
    eco_xr = eco_ds.read_as_xarray([
            Ecostress.LST,Ecostress.EMIS
        ],
                                   resolution=resolution,
                                   algorithm=rio.enums.Resampling.cubic)

    # Filter QA from ecostress
    eco_xr = eco_xr.where(eco_xr.QC != 0, np.nan)

    # Drop time dimension
    eco_xr = eco_xr.isel(t=0, drop=True)
    
    # Create an instance of Sentinel2 from the product path
    s2_ds = sentinel2.Sentinel2(s2_path)
    
    # Read sentinel2 data (optical bands)
    s2_xr = s2_ds.read_as_xarray([
            sentinel2.Sentinel2.B2,
            sentinel2.Sentinel2.B3,
            sentinel2.Sentinel2.B4,
            sentinel2.Sentinel2.B6,
            sentinel2.Sentinel2.B8,
            sentinel2.Sentinel2.B8A,
            sentinel2.Sentinel2.B11,
            sentinel2.Sentinel2.B12
        ],
                            resolution=resolution,
                            crs=eco_ds.crs, # Use same projection as ecostress product
                            bounds=eco_ds.bounds, # Use bb of Ecostress product
                            algorithm=rio.enums.Resampling.average)

    # Filter pixels
    # https://labo.obs-mip.fr/multitemp/sentinel-2/theias-sentinel-2-l2a-product-format/#English
    #s2_xr = s2_xr.where(s2_xr.MG2 == 0, np.nan) #no data
    #s2_xr = s2_xr.where(s2_xr.MG2 == 1, np.nan) #saturated
    #s2_xr = s2_xr.where(s2_xr.MG2 == 3, np.nan) #cloud shadows
    #s2_xr = s2_xr.where(s2_xr.MG2 == 6, np.nan) #water
    #s2_xr = s2_xr.where(s2_xr.MG2 == 8, np.nan) #clouds
    #s2_xr = s2_xr.where(s2_xr.MG2 == 9, np.nan) #clouds
    #s2_xr = s2_xr.where(s2_xr.MG2 == 10, np.nan) #clouds

    # Drop time dimension
    s2_xr = s2_xr.isel(t=0, drop=True)

    # Merge xarray
    merged_xr = xr.merge((eco_xr, s2_xr))

    # Add metedata
    # Add transform
    merged_xr.attrs['transform'] = affine.Affine(resolution, 0.0, eco_ds.bounds.left, 0.0, -resolution, eco_ds.bounds.top)
    # Add capteur name
    merged_xr.attrs['name'] = "EcoS2"
    # Add acquisition date
    merged_xr.attrs['date'] = eco_ds.date
    # Add tile id
    merged_xr.attrs['tile_id'] = eco_ds.tile

    # Apply name mapping
    merged_xr = merged_xr.rename_vars(ecos2_name_mapping)
    # Remove any variable not listeed in name_mapping
    for var in merged_xr.data_vars.keys():
        if var not in ecos2_name_mapping.values():
            merged_xr = merged_xr.drop(var)

    # Compute indices
    compute_seli(merged_xr)
    compute_ndvi(merged_xr)
    compute_lai_from_seli(merged_xr)

    # Compute albedo
    compute_albedo(merged_xr)

    return merged_xr
    




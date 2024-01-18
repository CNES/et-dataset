#!/usr/bin/env python

# -*- coding: utf-8 -*-
#
# Copyright: (c) 2023 CESBIO / Centre National d'Etudes Spatiales / Université Paul Sabatier (UT3)
#
"""
Function to create dataset from ECOSTRESS and Landsat8
"""
import affine
import geopandas as gpd
import pandas as pd
import numpy as np
import rasterio as rio
import xarray as xr

from sensorsio import landsat, utils, ecostress_v2
from sensorsio.ecostress_v2 import EcostressV2

from etdataset.logging import LoggerManager
from etdataset.vegetation_indices import compute_ndvi, compute_lai_from_ndvi
from etdataset.ls8 import compute_albedo

logger = LoggerManager.get_logger(__name__)


ecols8_name_mapping = {
    'SR_B2': 'blue',
    'SR_B3': 'green',
    'SR_B4': 'red',
    'SR_B5': 'nir',
    'SR_B6': 'swir1',
    'SR_B7': 'swir2',
    'LST'  : 'lst',
    'EmisWB': 'emis',
}


def create_dataset(ls8_path: str,
                   eco_path:str,
                   resolution:int = 57) -> xr.Dataset:
    """
    Create dataset from landsat and ecostress products
    """
    logger.debug(f"Landsat path: {ls8_path}")
    logger.debug(f"Ecostress path: {eco_path}")

    # Create an instance of ecostress from the product path
    eco_ds = ecostress_coll2.EcostressV2(eco_path)
    
    # Read ecostress product
    eco_xr = eco_ds.read_as_xarray([
            EcostressV2.LST,EcostressV2.EMIS
        ],
                                   resolution=resolution,
                                   algorithm=rio.enums.Resampling.cubic)

    # Filter QA from ecostress
    eco_xr = eco_xr.where(eco_xr.QC != 0, np.nan)

    # Drop time dimension
    eco_xr = eco_xr.isel(t=0, drop=True)
    
    # Create an instance of Landsat8 from the product path
    ls8_ds = landsat.Landsat(ls8_path)
    
    # Read landsat data (optical bands)
    ls8_xr = ls8_ds.read_as_xarray([
            landsat.Landsat.B2,
            landsat.Landsat.B3, landsat.Landsat.B4, landsat.Landsat.B5,
            landsat.Landsat.B6, landsat.Landsat.B7
        ],
                                   resolution=resolution,
                                   crs=eco_ds.crs, # Use same projection as ecostress product
                                   bounds=eco_ds.bounds, # Use bb of Ecostress product
                                   algorithm=rio.enums.Resampling.average)

    # Filter QA from ls8
    clear_pixels_mask = utils.extract_bitmask(ls8_xr.QA_PIXEL.values, 6)
    not_filled_mask = ~utils.extract_bitmask(ls8_xr.QA_PIXEL.values, 0)
    not_water_mask = ~utils.extract_bitmask(ls8_xr.QA_PIXEL.values, 7)
    ls8_xr = ls8_xr.where(
        np.logical_and(np.logical_and(clear_pixels_mask, not_filled_mask),
                       not_water_mask), np.nan)


    # Drop time dimension
    ls8_xr = ls8_xr.isel(t=0, drop=True)

    # Merge xarray
    merged_xr = xr.merge((eco_xr, ls8_xr))

    # Add metedata
    # Add transform
    merged_xr.attrs['transform'] = affine.Affine(resolution, 0.0, eco_ds.bounds.left, 0.0, -resolution, eco_ds.bounds.top)
    # Add capteur name
    merged_xr.attrs['name'] = "EcoLs8"
    # Add acquisition date
    merged_xr.attrs['date'] = eco_ds.date
    # Add tile id
    merged_xr.attrs['tile_id'] = eco_ds.tile

    # Apply name mapping
    merged_xr = merged_xr.rename_vars(ecols8_name_mapping)
    # Remove any variable not listeed in name_mapping
    for var in merged_xr.data_vars.keys():
        if var not in ecols8_name_mapping.values():
            merged_xr = merged_xr.drop(var)

    # Compute NDVI
    compute_ndvi(merged_xr)
    # Compute LAI with coefficients from 
    # PADILLA, F. L. M., MAAS, S. J., GONZÁLEZ-DUGO, M. P., et al. 
    # Monitoring regional wheat yield in Southern Spain using the GRAMI model and satellite imagery. 
    # Field Crops Research, 2012, vol. 130, p. 145-154.
    compute_lai_from_ndvi(merged_xr, 0.04, 4.91)

    # Compute albedo
    compute_albedo(merged_xr)

    return merged_xr
    




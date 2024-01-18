#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright: (c) 2023 CESBIO / Centre National d'Etudes Spatiales / Université Paul Sabatier (UT3)
#
"""
Landsat8
"""
import affine
import geopandas as gpd
import pandas as pd
import numpy as np
import rasterio as rio
import xarray as xr

from sensorsio import landsat, mgrs, utils

from etdataset.logging import LoggerManager
from etdataset.common import create_polygon
from etdataset.vegetation_indices import compute_ndvi, compute_lai_from_ndvi

logger = LoggerManager.get_logger(__name__)


ls8_name_mapping = {
    'ST_B10': 'lst',
    'SR_B2': 'blue',
    'SR_B3': 'green',
    'SR_B4': 'red',
    'SR_B5': 'nir',
    'SR_B6': 'swir1',
    'SR_B7': 'swir2',
    'ST_QA': 'lst_err',
    'ST_EMIS': 'emis',
    'ST_EMSD': 'emis_err',
}


def compute_albedo(data: xr.Dataset) -> None:  
    """
    Compute albedo
    Liang, S. Narrowband to Broadband Conversions of Land Surface Albedo I: Algorithms. Remote Sens. Environ. 2001, 76, 213–238.
    """
    data['albedo'] = 0.356 * data.blue + 0.130 * data.red + 0.373 * data.nir + 0.085 * data.swir1 + 0.072 * data.swir2 - 0.0018


def create_dataset(product_path: str,
                   tile_id: str,
                   resolution:int = 57) -> xr.Dataset:
    """
    Create dataset from landsat product
    """
    # Create an instance of Landsat8 from the product path
    ls8_ds = landsat.Landsat(product_path)
    
    # Get bounding box for MRGS tile
    bb = mgrs.get_bbox_mgrs_tile(tile_id, False)
    bb = utils.bb_transform(mgrs.get_crs_mgrs_tile(tile_id),
                            ls8_ds.crs,
                            bb)
   
    # Read landsat data 
    # Every bands in the product is sampled at 30m 
    # RGB: B4, B3, B2
    # NIR: B5
    # SWIR: B6,B7
    # LST: B10
    ls8_xr = ls8_ds.read_as_xarray([
            landsat.Landsat.B10, landsat.Landsat.B1, landsat.Landsat.B2,
            landsat.Landsat.B3, landsat.Landsat.B4, landsat.Landsat.B5,
            landsat.Landsat.B6, landsat.Landsat.B7, landsat.Landsat.ST_EMIS,
            landsat.Landsat.ST_EMSD, landsat.Landsat.ST_TRAD,
            landsat.Landsat.ST_QA, landsat.Landsat.ST_URAD,
            landsat.Landsat.ST_DRAD, landsat.Landsat.ST_ATRAN
        ],
                                       resolution=resolution,
                                       crs=ls8_ds.crs,
                                       bounds=bb,
                                       algorithm=rio.enums.Resampling.average)
    # Add transform
    ls8_xr.attrs['transform'] = affine.Affine(resolution, 0.0, bb.left, 0.0, -resolution, bb.top)
    # Add capteur name
    ls8_xr.attrs['name'] = "Landsat"
    # Add acquisition date
    ls8_xr.attrs['date'] = ls8_ds.date
    # Add tile id
    ls8_xr.attrs['tile_id'] = tile_id

    # Filter QA from ls8
    clear_pixels_mask = utils.extract_bitmask(ls8_xr.QA_PIXEL.values, 6)
    not_filled_mask = ~utils.extract_bitmask(ls8_xr.QA_PIXEL.values, 0)
    not_water_mask = ~utils.extract_bitmask(ls8_xr.QA_PIXEL.values, 7)
    ls8_xr = ls8_xr.where(
        np.logical_and(np.logical_and(clear_pixels_mask, not_filled_mask),
                       not_water_mask), np.nan)

    # Apply name mapping
    ls8_xr= ls8_xr.rename_vars(ls8_name_mapping)
    # Remove any variable not listeed in name_mapping
    for var in ls8_xr.data_vars.keys():
        if var not in ls8_name_mapping.values():
            ls8_xr = ls8_xr.drop(var)

    # keep track of acquisition time
    acquisition_time = ls8_xr.t.values[0]

    # Drop time dimension
    ls8_xr = ls8_xr.isel(t=0, drop=True)

    # Compute NDVI
    compute_ndvi(ls8_xr)
    # Compute LAI with coefficients from 
    # PADILLA, F. L. M., MAAS, S. J., GONZÁLEZ-DUGO, M. P., et al. 
    # Monitoring regional wheat yield in Southern Spain using the GRAMI model and satellite imagery. Field Crops Research, 2012, vol. 130, p. 145-154.
    compute_lai_from_ndvi(ls8_xr, 0.04, 4.91)

    # Compute albedo
    compute_albedo(ls8_xr)

    return ls8_xr
    


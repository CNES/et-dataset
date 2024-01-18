#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright: (c) 2023 CESBIO / Centre National d'Etudes Spatiales / Université Paul Sabatier (UT3)
#
"""
Common function
"""
import os
import numpy as np
import rasterio as rio
import xarray as xr

from rasterio.enums import ColorInterp

from shapely.geometry import Point, Polygon


BANDS=["red","blue","green","nir","lst","emis","ndvi","albedo","lai"]

def create_polygon(ul_lat: float, 
                   ul_long: float, 
                   ur_lat: float, 
                   ur_long: float, 
                   ll_lat: float, 
                   ll_long: float, 
                   lr_lat: float, 
                   lr_long: float) -> Polygon:
    ul = Point(ul_long,ul_lat)
    ur = Point(ur_long,ur_lat)
    ll = Point(ll_long,ll_lat)
    lr = Point(lr_long,lr_lat)
    points = [ul, ur, lr, ll]
    return Polygon([i for i in points])

def write_dataset(xrds: xr.Dataset,
                  bands: list[str]  = BANDS,
                  dir: str = os.getcwd()):
    row = xrds.dims["y"]
    col = xrds.dims["x"]
    if bands is None:
        bands = [i for i in xrds.data_vars]
    filename = f"{xrds.attrs['name']}_{xrds.attrs['date']:%Y%m%d}_{xrds.attrs['tile_id']}.tif"
    with rio.open(
        os.path.join(dir,filename),
        mode="w+",
        driver="GTiff",
        width=col,
        height=row,
        count=len(bands),
        dtype=rio.dtypes.float32,
        nodata=np.nan,
        crs=xrds.attrs['crs'],
        transform=xrds.attrs['transform'],
        ) as source_ds:
            source_ds.colorinterp = [ColorInterp.gray for _ in bands]
            for id, band in enumerate(bands,start=1):
                source_ds.write_band(id, xrds[band].data)
                source_ds.set_band_description(id, band)

def write_band(xrds: xr.Dataset,
               band: str,
               dir: str = os.getcwd()):
    row = xrds.dims["y"]
    col = xrds.dims["x"]
    filename = f"{xrds.attrs['name']}_{xrds.attrs['date']:%Y%m%d}_{xrds.attrs['tile_id']}_{band}.tif"
    try:
        dtype = xrds[band].dtype
    except KeyError:
        raise Exception(f"Band {band} not in dataset")
    with rio.open(
        os.path.join(dir,filename),
        mode="w+",
        driver="GTiff",
        width=col,
        height=row,
        count=1,
        dtype=dtype,
        nodata=np.nan,
        crs=xrds.attrs['crs'],
        transform=xrds.attrs['transform'],
        ) as source_ds:
            source_ds.write_band(1, xrds[band].data)
            source_ds.set_band_description(1, band)                

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
import pandas as pd
import rasterio as rio
import xarray as xr
from rasterio.enums import ColorInterp
from scipy.io import savemat

BANDS = ["red", "blue", "green", "nir", "lst", "emis", "ndvi", "albedo", "lai"]


def write_dataset(
    xrds: xr.Dataset, bands: list[str] = BANDS, directory: str = os.getcwd()
):
    row = xrds.dims["y"]
    col = xrds.dims["x"]
    if bands is None:
        bands = [i for i in xrds.data_vars]
    if xrds.attrs["vis"] == xrds.attrs["tir"]:
        filename = f"{xrds.attrs['vis']}_{xrds.attrs['vis_date']:%Y%m%d}"
    else:
        filename = (
            f"{xrds.attrs['vis']}_{xrds.attrs['vis_date']:%Y%m%d}_"
            f"{xrds.attrs['tir']}_{xrds.attrs['tir_date']:%Y%m%d}"
        )
    filename += f"_{xrds.attrs['tile']}.tif"
    with rio.open(
        os.path.join(directory, filename),
        mode="w+",
        driver="GTiff",
        width=col,
        height=row,
        count=len(bands),
        dtype=rio.dtypes.float32,
        nodata=np.nan,
        crs=xrds.attrs["crs"],
        transform=xrds.attrs["transform"],
    ) as source_ds:
        source_ds.colorinterp = [ColorInterp.gray for _ in bands]
        for id, band in enumerate(bands, start=1):
            source_ds.write_band(id, xrds[band].data)
            source_ds.set_band_description(id, band)


def write_band(xrds: xr.Dataset, band: str, directory: str = os.getcwd()):
    row = xrds.dims["y"]
    col = xrds.dims["x"]
    if xrds.attrs["vis"] == xrds.attrs["tir"]:
        filename = f"{xrds.attrs['vis']}_{xrds.attrs['vis_date']:%Y%m%d}"
    else:
        filename = (
            f"{xrds.attrs['vis']}_{xrds.attrs['vis_date']:%Y%m%d}_"
            f"{xrds.attrs['tir']}_{xrds.attrs['tir_date']:%Y%m%d}"
        )
    filename += f"_{xrds.attrs['tile']}.tif"
    try:
        dtype = xrds[band].dtype
    except KeyError:
        raise Exception(f"Band {band} not in dataset")
    with rio.open(
        os.path.join(directory, filename),
        mode="w+",
        driver="GTiff",
        width=col,
        height=row,
        count=1,
        dtype=dtype,
        nodata=np.nan,
        crs=xrds.attrs["crs"],
        transform=xrds.attrs["transform"],
    ) as source_ds:
        source_ds.write_band(1, xrds[band].data)
        source_ds.set_band_description(1, band)


def export_matlab(
    xrds: xr.Dataset, bands: list[str] = BANDS, directory: str = os.getcwd()
):
    if bands is None:
        bands = [i for i in xrds.data_vars]
    if xrds.attrs["vis"] == xrds.attrs["tir"]:
        suffix = f"{xrds.attrs['vis']}_{xrds.attrs['vis_date']:%Y%m%d}"
    else:
        suffix = (
            f"{xrds.attrs['vis']}_{xrds.attrs['vis_date']:%Y%m%d}_"
            f"{xrds.attrs['tir']}_{xrds.attrs['tir_date']:%Y%m%d}"
        )
    suffix += f"_{xrds.attrs['tile']}"

    for band in bands:
        file_path = os.path.join(directory, suffix + f"_{band}.mat")
        mdic = {"data": xrds[band].data, "label": band}
        savemat(file_path, mdic)


def write_matches(res: pd.DataFrame, output: str = "matches.csv") -> None:
    res.to_csv(output)


def write_results(res: pd.DataFrame, output: str = "results.csv") -> None:
    res.to_csv(output)

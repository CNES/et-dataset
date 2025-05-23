# Copyright: (c) 2023 CESBIO / Centre National d'Etudes Spatiales /
#            Université Paul Sabatier (UT3)
#
"""
Common function
"""

import os

import numpy as np
import pandas as pd
import rasterio as rio
import rioxarray  # noqa # Import to activate rio attributes
import xarray as xr
from rasterio.enums import ColorInterp
from scipy.io import savemat


def get_row_col(xrds: xr.Dataset) -> tuple[int, int]:
    """
    Get number of rows and columns from a dataset
    """
    if len(xrds.data_vars) == 0:
        msg = "Datset empty"
        raise ValueError(msg)
    # Get col/row
    da = next(iter(xrds.data_vars.values()))
    dim_names = da.dims
    shape = da.shape
    dim_map = dict(zip(dim_names, shape, strict=False))
    row_names = ["y", "lat", "latitude"]
    col_names = ["x", "lon", "longitude"]
    coords = xrds.coords
    row = next((dim_map[name] for name in row_names if name in coords), None)
    col = next((dim_map[name] for name in col_names if name in coords), None)
    if row is None or col is None:
        msg = "Unable to get rows and columns from dataset"
        raise ValueError(msg)
    return row, col


def write_dataset(
    xrds: xr.Dataset,
    bands: list[str] | None = None,
    directory: str = os.getcwd(),
    separate=False,
):
    """
    Write dataset in one file or in separated files
    """
    row, col = get_row_col(xrds)
    if bands is None:
        bands = list(xrds.data_vars)
    else:
        bands = [i for i in bands if i in xrds.data_vars]
    if xrds.attrs["vis"] == xrds.attrs["tir"]:
        filename = f"{xrds.attrs['vis']}_{xrds.attrs['vis_date']:%Y%m%d}"
    else:
        filename = (
            f"{xrds.attrs['vis']}_{xrds.attrs['vis_date']:%Y%m%d}_"
            f"{xrds.attrs['tir']}_{xrds.attrs['tir_date']:%Y%m%d}"
        )
    # Get projection
    if xrds.attrs.get("crs", None) is not None:
        crs = xrds.attrs["crs"]
        transform = xrds.attrs["transform"]
    elif hasattr(xrds, "rio"):
        crs = xrds.rio.crs
        transform = xrds.rio.transform()
    else:
        raise AttributeError("No CRS is defined")
    if not separate:
        if xrds.attrs.get("tile", None) is not None:
            filename += f"_{xrds.attrs['tile']}"
        filename += ".tif"
        with rio.open(
            os.path.join(directory, filename),
            mode="w+",
            driver="GTiff",
            width=col,
            height=row,
            count=len(bands),
            dtype=rio.dtypes.float32,
            nodata=np.nan,
            crs=crs,
            transform=transform,
        ) as source_ds:
            source_ds.colorinterp = [ColorInterp.gray for _ in bands]
            for i, band in enumerate(bands, start=1):
                source_ds.write_band(i, xrds[band].data)
                source_ds.set_band_description(i, band)
    else:
        if xrds.attrs.get("tile", None) is not None:
            filename += f"_{xrds.attrs['tile']}"
        os.makedirs(os.path.join(directory, filename), exist_ok=True)
        for band in bands:
            with rio.open(
                os.path.join(directory, filename, filename + f"_{band}.tif"),
                mode="w+",
                driver="GTiff",
                width=col,
                height=row,
                count=1,
                dtype=rio.dtypes.float32,
                nodata=np.nan,
                crs=crs,
                transform=transform,
            ) as source_ds:
                source_ds.colorinterp = [ColorInterp.gray]
                source_ds.write_band(1, xrds[band].data)
                source_ds.set_band_description(1, band)


def write_band(xrds: xr.Dataset, band: str, directory: str = os.getcwd()):
    row, col = get_row_col(xrds)
    if xrds.attrs["vis"] == xrds.attrs["tir"]:
        filename = f"{xrds.attrs['vis']}_{xrds.attrs['vis_date']:%Y%m%d}"
    else:
        filename = (
            f"{xrds.attrs['vis']}_{xrds.attrs['vis_date']:%Y%m%d}_"
            f"{xrds.attrs['tir']}_{xrds.attrs['tir_date']:%Y%m%d}"
        )
    if xrds.attrs.get("tile", None) is not None:
        filename += f"_{xrds.attrs['tile']}"
    filename += ".tif"
    try:
        dtype = xrds[band].dtype
    except KeyError:
        raise ValueError(f"Band {band} not in dataset")
    # Get projection
    if xrds.attrs.get("crs", None) is not None:
        crs = xrds.attrs["crs"]
        transform = xrds.attrs["transform"]
    elif hasattr(xrds, "rio"):
        crs = xrds.rio.crs
        transform = xrds.rio.transform()
    else:
        raise AttributeError("No CRS is defined")
    with rio.open(
        os.path.join(directory, filename),
        mode="w+",
        driver="GTiff",
        width=col,
        height=row,
        count=1,
        dtype=dtype,
        nodata=np.nan,
        crs=crs,
        transform=transform,
    ) as source_ds:
        source_ds.write_band(1, xrds[band].data)
        source_ds.set_band_description(1, band)


def export_matlab(
    xrds: xr.Dataset,
    bands: list[str] | None = None,
    directory: str = os.getcwd(),
):
    if bands is None:
        bands = list(xrds.data_vars)
    else:
        bands = [i for i in bands if i in xrds.data_vars]
    if xrds.attrs["vis"] == xrds.attrs["tir"]:
        suffix = f"{xrds.attrs['vis']}_{xrds.attrs['vis_date']:%Y%m%d}"
    else:
        suffix = (
            f"{xrds.attrs['vis']}_{xrds.attrs['vis_date']:%Y%m%d}_"
            f"{xrds.attrs['tir']}_{xrds.attrs['tir_date']:%Y%m%d}"
        )
    if xrds.attrs.get("tile", None) is not None:
        suffix += f"_{xrds.attrs['tile']}"

    for band in bands:
        file_path = os.path.join(directory, suffix + f"_{band}.mat")
        mdic = {"data": xrds[band].data, "label": band}
        savemat(file_path, mdic)


def write_matches(res: pd.DataFrame, output: str = "matches.csv") -> None:
    if len(res) > 0:
        res.to_csv(output, index=False)


def write_results(res: pd.DataFrame, output: str = "results.csv") -> None:
    if len(res) > 0:
        res.to_csv(output, index=False)


def write_to_tif(xrds: xr.Dataset, filename=str):
    """
    Write data to tif
    """
    row, col = get_row_col(xrds)
    bands = list(xrds.data_vars)
    # Get projection
    if xrds.attrs.get("crs", None) is not None:
        crs = xrds.attrs["crs"]
        transform = xrds.attrs["transform"]
    elif hasattr(xrds, "rio"):
        crs = xrds.rio.crs
        transform = xrds.rio.transform()
    else:
        raise AttributeError("No CRS is defined")
    with rio.open(
        filename,
        mode="w+",
        driver="GTiff",
        width=col,
        height=row,
        count=len(bands),
        dtype=rio.dtypes.float32,
        nodata=np.nan,
        crs=crs,
        transform=transform,
    ) as source_ds:
        source_ds.colorinterp = [ColorInterp.gray for _ in bands]
        for i, band in enumerate(bands, start=1):
            source_ds.write_band(i, xrds[band].data)
            source_ds.set_band_description(i, band)

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

    Parameters
    ----------
    xrds: xr.Dataset
        Dataset

    Returns
    -------
    row,col: tuple[int,int]
        number of rows and columns
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
    Write the dataset to a TIFF file or to separated TIFF files

    Parameters
    ----------
    xrds: xr.Dataset
        Dataset
    bands: list[str]
        List of bands to write
    directory: str
        Output directory path
    separate: bool
        Write one or multiple files
    """
    row, col = get_row_col(xrds)
    if bands is None:
        bands = [str(b) for b in xrds.data_vars]
    else:
        bands = [i for i in bands if i in xrds.data_vars]
    if (
        xrds.attrs.get("vis", None) is not None
        and xrds.attrs.get("tir", None) is not None
    ):
        if xrds.attrs["vis"] == xrds.attrs["tir"]:
            filename = (
                f"{xrds.attrs['vis']}_"
                f"{xrds.attrs['vis_date']:%Y%m%d}"
                f"T{xrds.attrs['vis_time']:%H%M%S%z}"
            )
        else:
            filename = (
                f"{xrds.attrs['vis']}_"
                f"{xrds.attrs['vis_date']:%Y%m%d}"
                f"T{xrds.attrs['vis_time']:%H%M%S%z}_"
                f"{xrds.attrs['tir']}_"
                f"{xrds.attrs['tir_date']:%Y%m%d}"
                f"T{xrds.attrs['tir_time']:%H%M%S%z}"
            )
    elif (
        xrds.attrs.get("vis", None) is not None
        and xrds.attrs.get("tir", None) is None
    ):
        filename = (
            f"{xrds.attrs['vis']}_"
            f"{xrds.attrs['vis_date']:%Y%m%d}"
            f"T{xrds.attrs['vis_time']:%H%M%S%z}"
        )
    elif (
        xrds.attrs.get("vis", None) is None
        and xrds.attrs.get("tir", None) is not None
    ):
        filename = (
            f"{xrds.attrs['tir']}_"
            f"{xrds.attrs['tir_date']:%Y%m%d}"
            f"T{xrds.attrs['tir_time']:%H%M%S%z}"
        )
    else:
        filename = "data"
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
    """
    Write one band from a dataset to a TIFF file

    Parameters
    ----------
    xrds: xr.Dataset
        Dataset
    band: str
        Band to write
    directory: str
        Output directory path
    """
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
    """
    Write the dataset to a matlab file

    Parameters
    ----------
    xrds: xr.Dataset
        Dataset
    bands: list[str]
        List of bands to write
    directory: str
        Output directory path
    """
    if bands is None:
        bands = [str(b) for b in xrds.data_vars]
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
    """
    Write matches file

    Parameters
    ----------
    res: pd.DataFrame
        Matches results
    output: str
        Output file
    """
    if len(res) > 0:
        res.to_csv(output, index=False)


def write_results(res: pd.DataFrame, output: str = "results.csv") -> None:
    """
    Write results file

    Parameters
    ----------
    res: pd.DataFrame
        Results
    output: str
        Output file
    """
    if len(res) > 0:
        res.to_csv(output, index=False)


def write_to_tif(
    xrds: xr.Dataset, bands: list[str] | None = None, filename=str
):
    """
    Write the dataset to TIFF file

    Parameters
    ----------
    xrds: xr.Dataset
        Dataset
    filename: str
        File path
    """
    row, col = get_row_col(xrds)
    if bands is None:
        bands = [str(b) for b in xrds.data_vars]
    else:
        bands = [i for i in bands if i in xrds.data_vars]
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


def write_to_netcdf(xrds: xr.Dataset, filename=str):
    """
    Write the dataset to NETCDF file

    Parameters
    ----------
    xrds: xr.Dataset
        Dataset
    filename: str
        File path
    """
    # Create encoding dictionary
    for variable in list(xrds.keys()):
        xrds[variable].encoding.update(
            {
                "dtype": "f4",  # float32
                "_FillValue": np.float32(np.nan),
                # TODO: check if compression affects reading speed
                "zlib": True,
                "complevel": 1,  # level 1 is low compression but fast
            }
        )

    # Get projection
    if xrds.attrs.get("crs", None) is not None:
        crs = xrds.attrs["crs"]
    elif hasattr(xrds, "rio"):
        crs = xrds.rio.crs
    xrds.attrs.clear()
    # Write crs
    xrds = xrds.rio.write_crs(crs)  # write crs if necessary
    # Save to netCDF4
    xrds.to_netcdf(filename)


def write_dataset_to_netcdf(
    xrds: xr.Dataset,
    directory: str = os.getcwd(),
):
    """
    Write to NETCDF format
    """
    if (
        xrds.attrs.get("vis", None) is not None
        and xrds.attrs.get("tir", None) is not None
    ):
        if xrds.attrs["vis"] == xrds.attrs["tir"]:
            filename = (
                f"{xrds.attrs['vis']}_"
                f"{xrds.attrs['vis_date']:%Y%m%d}"
                f"T{xrds.attrs['vis_time']:%H%M%S%z}"
            )
        else:
            filename = (
                f"{xrds.attrs['vis']}_"
                f"{xrds.attrs['vis_date']:%Y%m%d}"
                f"T{xrds.attrs['vis_time']:%H%M%S%z}_"
                f"{xrds.attrs['tir']}_"
                f"{xrds.attrs['tir_date']:%Y%m%d}"
                f"T{xrds.attrs['tir_time']:%H%M%S%z}"
            )
    elif (
        xrds.attrs.get("vis", None) is not None
        and xrds.attrs.get("tir", None) is None
    ):
        filename = (
            f"{xrds.attrs['vis']}_"
            f"{xrds.attrs['vis_date']:%Y%m%d}"
            f"T{xrds.attrs['vis_time']:%H%M%S%z}"
        )
    elif (
        xrds.attrs.get("vis", None) is None
        and xrds.attrs.get("tir", None) is not None
    ):
        filename = (
            f"{xrds.attrs['tir']}_"
            f"{xrds.attrs['tir_date']:%Y%m%d}"
            f"T{xrds.attrs['tir_time']:%H%M%S%z}"
        )
    else:
        filename = "data"
    # Get projection
    if xrds.attrs.get("crs", None) is not None:
        crs = xrds.attrs["crs"]
    elif hasattr(xrds, "rio"):
        crs = xrds.rio.crs
    else:
        raise AttributeError("No CRS is defined")
    if xrds.attrs.get("tile", None) is not None:
        filename += f"_{xrds.attrs['tile']}"
    filename += ".nc"
    # Create encoding dictionary
    for variable in list(xrds.keys()):
        xrds[variable].encoding.update(
            {
                "dtype": "f4",  # float32
                "_FillValue": np.float32(np.nan),
                # TODO: check if compression affects reading speed
                "zlib": True,
                "complevel": 1,  # level 1 is low compression but fast
            }
        )
    xrds.attrs.clear()
    # Write crs
    xrds = xrds.rio.write_crs(crs)  # write crs if necessary
    # Save to netCDF4
    xrds.to_netcdf(os.path.join(directory, filename))


def write_daily_radiation(data: xr.Dataset, path: str | None = None) -> None:
    """
    Description
    -----------
    Create .tif file from a daily radiation dataset

    Parameters
    ----------
    data: xr.Dataset
        Daily radaition dataset
    path: str
        Directory path to store the .tif file
    """
    if path is None:
        path = os.getcwd()
    ts_path = os.path.join(path, "timeseries/daily_radiation")
    os.makedirs(ts_path, exist_ok=True)
    date = data.attrs["vis_date"]
    filename = os.path.join(ts_path, f"radiation_{date.strftime('%Y%m%d')}.tif")
    row, col = get_row_col(data)
    band = "daily_radiation"
    with rio.open(
        filename,
        mode="w+",
        driver="GTiff",
        width=col,
        height=row,
        count=1,
        dtype=rio.dtypes.float32,
        nodata=np.nan,
        crs=data.rio.crs,
        transform=data.rio.transform(),
    ) as source_ds:
        source_ds.colorinterp = [ColorInterp.gray]
        source_ds.write_band(1, data[band].data)
        source_ds.set_band_description(1, band)


def write_et_single_date(data: xr.Dataset, path: str | None = None) -> None:
    """
    Description
    -----------
    Create .tif file from a daily evapotranspiration dataset with flags

    Parameters
    ----------
    data: xr.Dataset
        Daily radaition dataset
    path: str
        Directory to store the .tif file
    """
    if path is None:
        path = os.getcwd()
    ts_path = os.path.join(path, "timeseries/et")
    os.makedirs(ts_path, exist_ok=True)
    date = data.attrs["vis_date"]
    file_name = os.path.join(
        ts_path, f"et_single_date_{date.strftime('%Y%m%d')}.tif"
    )
    row, col = get_row_col(data)
    bands = list(data.data_vars)
    with rio.open(
        file_name,
        mode="w+",
        driver="GTiff",
        width=col,
        height=row,
        count=2,
        dtype=rio.dtypes.float32,
        nodata=np.nan,
        crs=data.rio.crs,
        transform=data.rio.transform(),
    ) as source_ds:
        for i, band in enumerate(bands, start=1):
            source_ds.write_band(i, data[band].data)
            source_ds.set_band_description(i, band)


def write_daily_explanatory(data: xr.Dataset, path: str | None = None) -> None:
    """
    Write .tif file from a daily explanatory variable dataset

    Parameters
    ----------
    data: xr.Dataset
        Daily explanatory variables dataset
    path: str
        Directory path to store the .tif file
    """
    if path is None:
        path = os.getcwd()
    ts_path = os.path.join(path, "explanatory_variables")
    os.makedirs(ts_path, exist_ok=True)
    date = data.attrs["vis_date"]
    file_name = os.path.join(
        ts_path, f"explanatory_{date.strftime('%Y%m%d')}.tif"
    )
    row, col = get_row_col(data)
    bands = list(data.data_vars)
    with rio.open(
        file_name,
        mode="w+",
        driver="GTiff",
        width=col,
        height=row,
        count=4,
        dtype=rio.dtypes.float32,
        nodata=np.nan,
        crs=data.rio.crs,
        transform=data.rio.transform(),
    ) as source_ds:
        for i, band in enumerate(bands, start=1):
            source_ds.write_band(i, data[band].data)
            source_ds.set_band_description(i, band)

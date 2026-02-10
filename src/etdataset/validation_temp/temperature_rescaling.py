# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales

"""
Module for temperature rescaling
"""

import datetime as dt
import os
from collections.abc import Generator

import pandas as pd
import rasterio as rio
import xarray as xr
from pyproj import CRS

from etdataset.dem import compute_egm96_height
from etdataset.era5 import (
    ERA5Dataset,
    ERA5Var,
    read,
    rescale_temperature_with_lapserate,
)
from etdataset.icos import StationConfig, kelvin_to_celsius
from etdataset.interpolation import create_grid_dataset
from etdataset.logging import LoggerManager
from etdataset.utils import work_area_from_coord_point

logger = LoggerManager.get_logger(__name__)
G_CST = 9.80665


#########################################
##                                     ##
##                                     ##
##           Prepare Inputs            ##
##                                     ##
##                                     ##
#########################################


def generate_dates(
    start_date: dt.date,
    end_date: dt.date,
    day_step: int = 1,
) -> Generator[dt.date, None, None]:
    """
    Description
    -----------
    Generate dates between two dates with a given step.

    This function yields one date at a time, starting from 'start_date'
    up to and including 'end_date', incremented by 'day_step' days.

    Parameters
    ----------
    start_date : dt.date
        First date to generate
    end_date : dt.date
        Last date to generate
    day_step : int, optional
        Number of days between generated dates

    Yields
    ------
    dt.date
        A date in the specified range.
    """
    cur_date = start_date
    while cur_date <= end_date:
        yield cur_date
        cur_date += dt.timedelta(days=day_step)


# TO DO : TYPE
def prepare_temperature_inputs(
    data: xr.Dataset,
    era5_data: xr.Dataset,
    dataset: ERA5Dataset = ERA5Dataset.ERA5,
) -> tuple[xr.DataArray | None, xr.DataArray | None, xr.Dataset]:
    """
    Description
    -----------
    Prepare DEM, ERA5 DEM and copy of dataset for temperature rescaling.

    Parameters
    ----------
    data: xr.Dataset
        Data (e.g : grid or dem)
    era5_data : xr.Dataset
        Era5 data for one day
    dataset: ERA5Dataset
        ERA5 Dataset used for download

    Return
    -----------
    dem : xr.DataArray
        dem
    era5_dem : xr.DataArray
        ERA5 dem
    updated_data : xr.Dataset
        Updated data
    """
    dem = data.get("height", None)
    crs = data.rio.crs

    if dem is not None:
        dem = dem.rio.write_crs(crs)

    logger.info(f"Dem = {dem}")

    # Compute ERA5 DEM
    if (
        dataset == ERA5Dataset.ERA5
        and ERA5Var.GEOPOTENTIAL.key in era5_data.data_vars
    ):
        era5_dem = era5_data[ERA5Var.GEOPOTENTIAL.key] / G_CST
        logger.warning(
            "DEM is missing in ERA5 data: No variables 'height' in the dataset"
        )
    elif dataset == ERA5Dataset.ERA5LAND:
        era5_dem = compute_egm96_height(era5_data)
        logger.debug("Compute EGM96 height")
    else:
        era5_dem = None

    updated_data = data.copy()

    return dem, era5_dem, updated_data


def read_era5_file(
    date: dt.date,
    dataset: ERA5Dataset = ERA5Dataset.ERA5,
    path: str | None = None,
) -> xr.Dataset:
    """
    Description
    -----------
    For a given date and dataset of ERA5, read the file and return a dataset.

    Parameters
    -----------
    datetime : dt.datetime
        A date
    dataset : ERA5Dataset
        Dataset
    path : str
        Path to ERA5 file

    Returns
    -------
    era5_xrds : xr.Dataset
    """
    if path is None:
        path = os.getcwd()
    product_path = os.path.join(
        path,
        "ERA5_data",
        f"download_{dataset.key}_{date.isoformat()}.zip",
    )
    logger.info(f"Product path: {product_path}")
    if not os.path.isfile(product_path):
        raise OSError(f"ERA5 data not found: {product_path}")

    era5_xrds = read(product=product_path)
    return era5_xrds


def generate_hours(
    hour_start: int = 0,
    hour_end: int = 22,
    hour_step: int = 2,
) -> list[dt.time]:
    """
    Description
    -----------


    Parameters
    -----------
    hour_start : int
        Dataset to filter
    hour_end : int
        List of datetimes
    hour_step : int
        Name of the datetime column of the dataset

    Returns
    -------
    list[dt.time]
    """
    if hour_step <= 0:
        raise ValueError("Hour step must be positive")

    hours = range(hour_start, hour_end + 1, hour_step)

    return [dt.time(hour=h) for h in hours]


def filter_dataset_by_hours(
    ds: xr.Dataset,
    date: dt.date,
    times: list[dt.time],
    name_column: str = "time",
) -> xr.Dataset:
    """
    Description
    -----------
    Filter a dataset by a given list of hours

    Parameters
    -----------
    ds : xr.Dataset
        Dataset to filter
    list_dt : list[dt.datetime]
        List of datetimes
    name_column : str
        Name of the datetime column of the dataset

    Returns
    -------
    xr.Dataset
        Filtered dataset
    """
    list_dt = [dt.datetime.combine(date, t) for t in times]
    if name_column not in ds.coords:
        raise ValueError(f"'{name_column}' is not a coordinate in the dataset")
    ds = ds.copy()
    ds[name_column] = pd.to_datetime(ds[name_column].values)
    return ds.sel({name_column: list_dt})


def create_era5_sub_dataset(era5_xrds: xr.Dataset) -> xr.Dataset:
    """
    Description
    -----------
    Create a subset of an ERA5 xarray Dataset containing only selected variables
    and rename them to standardized names

    It renames 't2m' and 'd2m' to 'ta' and 'tdp'.

    Parameters
    ----------
    era5_xrds : xr.Dataset
        The original ERA5 dataset containing multiple data variables.

    Returns
    -------
    era5_sub : xr.Dataset
        A new xarray Dataset
    """
    era5_sub = era5_xrds[["t2m", "d2m"]]
    era5_sub = era5_sub.rename(
        {
            "t2m": "ta",
            "d2m": "tdp",
        }
    )
    return era5_sub


def get_era5_grid(
    bounds: rio.coords.BoundingBox, crs: CRS, resolution: float
) -> xr.Dataset:
    """
    Description
    -----------
    Get a ERA5 grid for interpolation, given bounds and special resolution

    Parameters
    ----------
    bounds : rio.coords.BoundingBox
        Bounds of the grid
    crs : CRS
        CRS of the grid
    resolution : float
        Resolution of the grid

    Return
    -----------
    grid : xr.Dataset
        Grid for ERA5 data
    """
    grid = create_grid_dataset(bounds, crs, resolution)
    grid["height"] = grid["grid"].copy()
    return grid


#########################################
##                                     ##
##                                     ##
##      Temperature rescaling          ##
##                                     ##
##                                     ##
#########################################


# to do
def temperature_rescaling_constant_lapse_rate(
    updated_data: xr.Dataset,
    dem: xr.DataArray | None,
    era5_data: xr.Dataset,
    era5_dem: xr.DataArray | None,
    lr_ta: float = -0.0065,
    lr_tdp: float = -0.0052,
) -> xr.Dataset:
    """
    Description
    -----------
    Rescale ERA5 temperature variables using a constant lapse rate.

    ERA5 temperatures are originally computed at the ERA5 elevation level.

    -> In a first step, the temperatures are adjusted to a fixed reference
    elevation using a constant lapse rate.


    -> In a second step, if a target DEM is provided, the temperatures are
    further rescaled from the reference elevation to the DEM elevation.

    The resulting temperature variables are added to the updated dataset.

    Parameters
    ----------
    updated_data: xr.Dataset
        Updated Data
    dem : xr.DataArray
        dem
    era5_data: xr.Dataset
        ERA5 Data
    era5_dem: xr.DataArray
        ERA5 Data
    lr_ta: float
        Lapse rate for air temperature
    lr_tdp : float
        Lapse rate for dew point temperature

    Return
    -----------
    updated_data : xr.Dataset
        Updated data
    """
    # Air temperature

    rescaled = rescale_temperature_with_lapserate(
        dem=dem,
        era5_data=era5_data.get(ERA5Var.TEMPERATURE.key, None),
        era5_dem=era5_dem,
        lapse_rate=lr_ta,
        key="ta",
        description="2m air temperature",
    )
    if rescaled is not None:
        updated_data["ta"] = rescaled
        logger.debug("Add temperature: OK")

    # Dewpoint temperature
    rescaled = rescale_temperature_with_lapserate(
        dem=dem,
        era5_data=era5_data.get(ERA5Var.DEWPOINT_TEMPERATURE.key, None),
        era5_dem=era5_dem,
        lapse_rate=lr_tdp,
        key="tdp",
        description="dewpoint temperature",
    )
    if rescaled is not None:
        updated_data["tdp"] = rescaled
        logger.debug("Add dewpoint temperature: OK")

    return updated_data


def get_xy_dims(ds: xr.Dataset) -> dict[str, str]:
    """
    Description
    -----------
    Determine the names of the spatial dimensions for interpolation in an
    xarray Dataset

    This function looks the dataset's dimensions.
    It supports datasets with dimensions
    named either ("x", "y") or ("latitude", "longitude")

    Parameters
    ----------
    ds : xr.Dataset

    Returns
    -------
    dict[str, str]
        A dictionary mapping "x" and "y" to the corresponding dimension names
        in the dataset.
        e.g. {"x": "x", "y": "y"} or `{"x": "latitude", "y": "longitude"}

    """
    if "x" in ds.dims and "y" in ds.dims:
        return {"x": "x", "y": "y"}
    if "latitude" in ds.dims and "longitude" in ds.dims:
        return {"x": "longitude", "y": "latitude"}
    raise ValueError("Unable to detect spatial dimensions")


# TO DO : TYPE
def get_ta_td_celsius_at_location(
    data: xr.Dataset,
    cfg: StationConfig,
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Description
    -----------
    Get Air temperature and dewpoint temperature at a specified location
    from data

    Parameters
    ----------
     data: xr.Dataset
        Data
     x : float
        Coord x (longitude or utm)
     y : float
        Coord y  (latitude or utm)

    Return
    -----------
    ta, td : tuple[xr.DataArray,xr.DataArray]
        Air temperature and dewpoint temperature at station location
    """
    # Determine the names of the spatial dimensions in the dataset
    dims = get_xy_dims(data)

    # If the dataset uses projected coordinates (x/y)
    if dims["x"] == "x" and dims["y"] == "y":
        x, y = (
            work_area_from_coord_point(
                cfg.lat, cfg.lon, 0, 0, CRS.from_epsg(4326)
            )["utm"][0].left,
            work_area_from_coord_point(
                cfg.lat, cfg.lon, 0, 0, CRS.from_epsg(4326)
            )["utm"][0].bottom,
        )

    # If the dataset uses geographic coordinates (longitude/latitude)
    if dims["x"] == "longitude" and dims["y"] == "latitude":
        x = cfg.lon
        y = cfg.lat

    # Interpolate air temperature and dewpoint temperature at the specified
    # location

    ta, td = (
        kelvin_to_celsius(
            data["ta"].interp({dims["x"]: x, dims["y"]: y}, method="nearest")
        ),
        kelvin_to_celsius(
            data["tdp"].interp({dims["x"]: x, dims["y"]: y}, method="nearest")
        ),
    )
    return ta, td


#########################################
##                                     ##
##                                     ##
##            Save results             ##
##                                     ##
##                                     ##
#########################################


def save_ta_td_csv(
    ds: xr.Dataset,
    cfg: StationConfig,
    path: str | None = None,
    name_dir: str | None = None,
):
    """
    Description
    -----------
    Save Air Temperature (TA) and Dewpoint Temperature (TDP) timeseries into a
    csv file

    Parameters
    ----------
    ds: xr.Dataset
        Data to save
    cfg : StationConfig
        Current station configuration
    path: str
        Path to save th csv file
    name_dir : str
        Name of the directory

    """
    if name_dir is None:
        name_dir = "csv"

    if path is None:
        file_path = os.path.abspath(os.path.join(os.getcwd(), name_dir))
    else:
        file_path = os.path.abspath(os.path.join(path, name_dir))

    os.makedirs(file_path, exist_ok=True)

    file_name = os.path.join(file_path, f"{cfg.id}_csv.csv")

    df = ds.to_dataframe().reset_index()
    df = df[["time", "ta", "tdp"]]

    fichier_exist = os.path.exists(file_name)

    df.to_csv(file_name, mode="a", header=not fichier_exist, index=False)

# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales
import datetime as dt
import os

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
from etdataset.icos import (
    kelvin_to_celsius,
)
from etdataset.interpolation import create_grid_dataset
from etdataset.logging import LoggerManager

logger = LoggerManager.get_logger(__name__)
G_CST = 9.80665


def generate_datetime(
    start_date: dt.date,
    end_date: dt.date,
    day_step: int = 1,
    hour_start: int = 0,
    hour_end: int = 22,
    hour_step: int = 2,
) -> list[dt.datetime]:
    """
    Description
    -----------
    Generate a list of datetimes for a given start date, end date,
    and hour start and end.

    Parameters
    -----------
    start_date : dt.date
        Start date
    end_date : dt.date
        End date
    day_step : int
        Step of day
    hour_start : int
        Start hour
    hour_end : int
        End hour
    hour_step : int
        Step of hour

    Returns
    -------
    datetimes : list[dt.datetime]
        List of datetimes
    """
    datetimes = []
    cur_date = start_date
    while cur_date <= end_date:
        daily_times = [
            dt.datetime.combine(cur_date, dt.time(hour=h))
            for h in range(hour_start, hour_end + 1, hour_step)
        ]
        datetimes.extend(daily_times)
        cur_date += dt.timedelta(days=day_step)
    return datetimes


def read_era5_file(
    datetime: dt.datetime,
    dataset: ERA5Dataset = ERA5Dataset.ERA5,
    path: str | None = None,
):
    date = datetime.date()
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


def read_era5_files(
    datetimes: list[dt.datetime], path: str | None = None
) -> xr.Dataset:
    unique_dates = sorted({d.date() for d in datetimes})
    # One day
    if len(unique_dates) == 1:
        ds = read_era5_file(
            dt.datetime.combine(unique_dates[0], dt.time()),
            path=path,
        )
        return ds.sel(time=datetimes)

    # many days
    datasets = [
        read_era5_file(dt.datetime.combine(d, dt.time()), path=path)
        for d in unique_dates
    ]

    ds = xr.concat(datasets, dim="time", coords="minimal", compat="override")

    return ds.sel(time=datetimes)


def filter_dataset_by_datetimes(
    ds: xr.Dataset, list_dt: list[dt.datetime], name_column: str = "time"
) -> xr.Dataset:
    """
    Description
    -----------
    Filter a dataset by a given list of datetimes

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
    if name_column not in ds.coords:
        raise ValueError(f"'{name_column}' is not a coordinate in the dataset")

    return ds.sel({name_column: list_dt})


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


def temperature_rescaling_constant_lapse_rate(
    updated_data: xr.Dataset,
    dem: xr.DataArray | None,
    era5_data: xr.Dataset,
    era5_dem: xr.DataArray | None,
    variables: list[ERA5Var] | None,
) -> xr.Dataset:
    """
    Description
    -----------
    Apply constant lapse rate corrections and add temperature variables.

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
    variables: list[ERA5Var]
        List of variables to add

    Return
    -----------
    updated_data : xr.Dataset
        Updated data
    """
    if variables is None:
        logger.warning("No variables specify")
        return updated_data

    # Air temperature
    if ERA5Var.TEMPERATURE in variables:
        rescaled = rescale_temperature_with_lapserate(
            dem=dem,
            era5_data=era5_data.get(ERA5Var.TEMPERATURE.key, None),
            era5_dem=era5_dem,
            lapse_rate=-0.0065,
            key="ta",
            description="2m air temperature",
        )
        if rescaled is not None:
            updated_data["ta"] = rescaled
            logger.debug("Add temperature: OK")

    # Dewpoint temperature
    if ERA5Var.DEWPOINT_TEMPERATURE in variables:
        rescaled = rescale_temperature_with_lapserate(
            dem=dem,
            era5_data=era5_data.get(ERA5Var.DEWPOINT_TEMPERATURE.key, None),
            era5_dem=era5_dem,
            lapse_rate=-0.0052,
            key="tdp",
            description="dewpoint temperature",
        )
        if rescaled is not None:
            updated_data["tdp"] = rescaled
            logger.debug("Add dewpoint temperature: OK")

    return updated_data


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


# TO DO : TYPE
def get_ta_td_celsius_at_location(
    data: xr.Dataset,
    x: float,
    y: float,
) -> tuple[xr.DataArray, xr.DataArray]:
    """
     Description
    -----------
    Get Air temperature and dewpoint temperature at a location from data

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
    ta, td = (
        kelvin_to_celsius(
            data["ta"].interp({"x": x, "y": y}, method="nearest")
        ),
        kelvin_to_celsius(
            data["tdp"].interp({"x": x, "y": y}, method="nearest")
        ),
    )
    return ta, td

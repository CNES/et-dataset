# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales

"""
Module for temperature rescaling
"""

import datetime as dt
import os
import zipfile
from collections.abc import Generator

import numpy as np
import numpy.typing as npt
import pandas as pd
import rasterio as rio
import xarray as xr
from pyproj import CRS
from rasterio.warp import transform_bounds
from sklearn.metrics import r2_score

from etdataset.era5 import (
    ERA5Dataset,
    ERA5pressureVar,
    ERA5Var,
    get_era5_dem,
    get_era5land_dem,
    read,
    rescale_temperature_with_lapserate,
)
from etdataset.icos import (
    StationConfig,
    celsius_to_kelvin,
    compute_dewpoint_temp,
    kelvin_to_celsius,
)
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


def normalize_longitude_latitude(
    ds: xr.Dataset,
    lon_name: str = "longitude",
    lat_name: str = "latitude",
    target: str = "-180_180",
) -> xr.Dataset:
    """
    Normalize longitude coordinates of an xarray Dataset to a desired range.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset containing longitude and latitude coordinates.
    lon_name : str
        Name of the longitude coordinate
    lat_name : str
        Name of the latitude coordinate
    target : str
        Target longitude convention:
        - "-180_180": longitudes in [-180, 180]
        - "0_360": longitudes in [0, 360]
    Returns
    -------
    xr.Dataset
        Dataset with normalized longitude coordinates and sorted by longitude.

    """
    lon = ds[lon_name]
    lat = ds[lat_name]

    lon_min = float(lon.min())
    lon_max = float(lon.max())
    lat_min = float(lat.min())
    lat_max = float(lat.max())
    if target == "-180_180":
        # If longitudes exceed 180, we assume they are in [0, 360]
        if lon_max > 180:
            lon = ((lon + 180) % 360) - 180
            logger.warning(
                f"Longitudes detected in 0-360 range ({lon_min} to {lon_max})"
            )
    elif target == "0_360":
        # If longitudes contain negative values, assume [-180, 180]
        if lon_min < 0:
            logger.warning(
                f"Converting longitude from -180-180 to 0-360 "
                f"({lon_min:.2f} to {lon_max:.2f})"
            )
            lon = lon % 360
    else:
        raise ValueError("Wanted must be [-180_180] or [0_360]")

    if lat_min < -90 or lat_max > 90:
        raise ValueError("Unvalid latitude")

    ds = ds.assign_coords({lon_name: lon})
    ds = ds.sortby(lon_name)
    return ds


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

    # Compute ERA5 DEM
    if dataset == ERA5Dataset.ERA5:
        era5_dem = get_era5_dem()
        era5_dem = xr.DataArray(
            era5_dem.data,
            dims=("latitude", "longitude"),
            coords={
                "latitude": era5_data.latitude,
                "longitude": era5_data.longitude,
            },
        ).rio.write_crs(CRS(4326))
        era5_dem_ds = era5_dem.to_dataset(name="dem")
        era5_dem_ds = normalize_longitude_latitude(era5_dem_ds)
        era5_dem = era5_dem_ds["dem"]
        logger.warning(
            "DEM is missing in ERA5 data: No variables 'height' in the dataset"
        )
    elif dataset == ERA5Dataset.ERA5LAND:
        era5_dem = get_era5land_dem()
        era5_dem = xr.DataArray(
            era5_dem.data,
            dims=("latitude", "longitude"),
            coords={
                "latitude": era5_data.latitude,
                "longitude": era5_data.longitude,
            },
        ).rio.write_crs(CRS(4326))
        era5_dem_ds = era5_dem.to_dataset(name="dem")
        era5_dem_ds = normalize_longitude_latitude(era5_dem_ds)
        era5_dem = era5_dem_ds["dem"]

    elif (
        dataset == ERA5Dataset.ERA5PRESSURE
        and ERA5pressureVar.GEOPOTENTIAL.key in era5_data.data_vars
    ):
        era5_dem = era5_data[ERA5pressureVar.GEOPOTENTIAL.key] / G_CST
        era5_dem = era5_dem.squeeze("pressure_level")
        era5_dem = era5_dem.rio.write_crs(CRS(4326))
        era5_dem_ds = era5_dem.to_dataset(name="dem")
        era5_dem_ds = normalize_longitude_latitude(era5_dem_ds)
        era5_dem = era5_dem_ds["dem"]

        logger.warning(
            "DEM is missing in ERA5 data: No variables 'height' in the dataset"
        )
    else:
        era5_dem = None

    updated_data = data.copy()

    return dem, era5_dem, updated_data


def read_era5_file(
    date: dt.date,
    dataset: ERA5Dataset = ERA5Dataset.ERA5,
    path: str | None = None,
) -> xr.Dataset | None:
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

    logger.info("Product path: %s", product_path)

    if not os.path.isfile(product_path):
        logger.warning("ERA5 data not found: %s", product_path)
        return None

    try:
        era5_xrds = read(product=product_path)
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        logger.warning(
            "Skipping unreadable ERA5 file %s | Error: %s",
            product_path,
            exc,
        )
        return None
    else:
        return era5_xrds


def generate_hours(
    hour_start: int = 0,
    hour_end: int = 22,
    hour_step: int = 2,
) -> list[dt.time]:
    """
    Description
    -----------
    Generate hours between two hours with a given step.

    Parameters
    -----------
    hour_start : int
        Start hour
    hour_end : int
        End hour
    hour_step : int
        Hour step

    Returns
    -------
    list[dt.time]
    """
    if hour_step <= 0:
        raise ValueError("Hour step must be positive")

    hours = range(hour_start, hour_end + 1, hour_step)

    return [dt.time(hour=h) for h in hours]


# FILTERS


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


def filter_dataset_by_location(
    ds: xr.Dataset,
    latitude: float,
    longitude: float,
) -> xr.Dataset:
    """
    Filter Dataset to keep only the specified latitude and longitude

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset.
    latitude : float
        Latitude's location to keep
    longitude : float
        Longitude's location to keep

    Returns
    -------
    xr.Dataset
        Dataset filtered to include only the requested location

    """

    return ds.sel(latitude=latitude, longitude=longitude, method="nearest")


# @overload
# def filter_dataset_by_roi(
#     ds: xr.DataArray, roi_bbox: rio.coords.BoundingBox, roi_crs: CRS
# ) -> xr.DataArray: ...
# @overload
# def filter_dataset_by_roi(
#     ds: xr.Dataset, roi_bbox: rio.coords.BoundingBox, roi_crs: CRS
# ) -> xr.Dataset: ...


def filter_dataset_by_roi(
    ds: xr.Dataset | xr.DataArray,
    roi_bbox: rio.coords.BoundingBox,
    roi_crs: CRS,
    target_crs: str = "EPSG:4326",
) -> xr.Dataset | xr.DataArray:
    """
    Filter Dataset to keep only data inside ROI.

    Handles ROI in:
    - lat/lon (EPSG:4326)
    - UTM or any other CRS
    Parameters
    ----------
    ds : xr.Dataset
        Input dataset.
    roi_bbox : rio.BoundingBox
        Bounding box of the ROI
    roi_crs : CRS
        CRS of the bounding box

    Returns
    -------
    xr.Dataset
        Dataset filtered to include only the ROI
    """

    if ds.rio.crs is None:
        ds = ds.rio.write_crs(target_crs)
    dataset_crs = ds.rio.crs
    roi_crs = CRS.from_user_input(roi_crs)

    if dataset_crs is None:
        raise ValueError("Dataset does not have defined CRS")

    if roi_crs != dataset_crs:
        min_lon, min_lat, max_lon, max_lat = transform_bounds(
            roi_crs,
            dataset_crs,
            roi_bbox.left,
            roi_bbox.bottom,
            roi_bbox.right,
            roi_bbox.top,
        )
    else:
        min_lon = roi_bbox.left
        max_lon = roi_bbox.right
        min_lat = roi_bbox.bottom
        max_lat = roi_bbox.top

    if ds.longitude.max() > 180:
        min_lon, max_lon = np.mod([min_lon, max_lon], 360)

    latitudes = ds.latitude.values

    if latitudes[0] > latitudes[-1]:
        lat_slice = slice(max_lat, min_lat)
    else:
        lat_slice = slice(min_lat, max_lat)
    ds = ds.rio.write_crs(target_crs, inplace=False)
    return ds.sel(
        latitude=lat_slice,
        longitude=slice(min_lon, max_lon),
    )


def filter_dataset_by_pressure_levels(
    ds: xr.Dataset,
    pressure_levels: str | list[str],
    pressure_dim: str = "pressure_level",
) -> xr.Dataset:
    """
    Filter an xarray Dataset to retain only the specified pressure level(s).

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset.
    pressure_levels : str or list of str
        Pressure level(s) to keep. Values must match the dataset's
        pressure coordinate labels (e.g., "500" or ["1000", "850", "500"]).
    pressure_dim : str, default "pressure_level"
        Name of the pressure dimension or coordinate in the dataset.
    Returns
    -------
    xr.Dataset
        Dataset filtered to include only the requested pressure level(s).

    Raises
    ------
    ValueError
        If the specified pressure dimension or coordinate does not exist.
    """

    if pressure_dim not in ds.dims and pressure_dim not in ds.coords:
        raise ValueError(
            f"The dimension/coordinate '{pressure_dim}' does not exist in the\
              dataset."
        )

    if isinstance(pressure_levels, str):
        pressure_levels = [pressure_levels]

    pressure_levels_sorted = sorted(
        pressure_levels,
        key=lambda x: float(x),
        reverse=True,  # highest pressure first
    )

    ds_sel = ds.sel({pressure_dim: pressure_levels_sorted}, method="nearest")

    ds_sel = ds_sel.sortby(ds_sel[pressure_dim], ascending=False)
    ds_sel = ds_sel.isel(
        {pressure_dim: ~ds_sel[pressure_dim].to_index().duplicated()}
    )
    return ds_sel


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
            "d2m": "td",
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
##       Dewpoint Temperature          ##
##                                     ##
##                                     ##
#########################################
def get_saturation_vapor_pressure(
    t: npt.ArrayLike, a: float = 611.21, b: float = 17.502, c: float = 240.97
) -> npt.NDArray:
    """
    Description
    -----------
    Compute saturation vapor pressure (Pa) at a temperature t (°C):

                es = a * exp(b*t/(c+t))

    from "A Meteorological Distribution System for High-Resolution
    Terrestrial Modeling (MicroMet)", Journal of Hydrometeorology,
    by G.E. Liston, K. Elder

    - For water : a = 611.21 Pa, b = 17.502, c = 240.97°C
    - For ice : a = 611.15 Pa, b = 22.452, c = 272.55°C

    Parameters
    ----------
    t : npt.ArrayLike
        Temperature (in °C)
    a : float
    b : float
    c : float

    Return
    -----------
    npt.NDArray
    Saturation vapor pressure at temperature t
    """
    return a * np.exp(b * np.array(t) / (c + np.array(t)))


def compute_vapor_pressure(rh: npt.ArrayLike, es: npt.ArrayLike) -> npt.NDArray:
    """
    Description
    -----------
    Compute actual vapor pressure (Pa):

                RH = 100 * e / es
    it gives:
                e = RH * es / 100

    from "A Meteorological Distribution System for High-Resolution
    Terrestrial Modeling (MicroMet)", Journal of Hydrometeorology,
    by G.E. Liston, K. Elder

    Parameters
    ----------
    rh : npt.ArrayLike
        Relative humidity (in %)
    es : npt.ArrayLike
        Saturation vapor pressure (Pa)

    Return
    -----------
    npt.NDArray
    Vapor pressure (Pa)
    """
    return np.array(rh) * np.array(es) / 100


def compute_dewpoint_temp_from_e(
    e: npt.ArrayLike, a: float = 611.21, b: float = 17.502, c: float = 240.97
):
    """
    Description
    -----------
    Compute dewpoint temperature from vapor pressure (e):

                td = c * ln(e/a)/(b - ln(e/a))

    from "A Meteorological Distribution System for High-Resolution
    Terrestrial Modeling (MicroMet)", Journal of Hydrometeorology,
    by G.E. Liston, K. Elder

    - For water : a = 611.21 Pa, b = 17.502, c = 240.97°C
    - For ice : a = 611.15 Pa, b = 22.452, c = 272.55°C

    Parameters
    ----------
    e: npt.ArrayLike
        Vapor pressure (Pa)
    a : float
    b : float
    c : float

    Return
    -----------
    npt.NDArray
    Dewpoint temperature (in °C)
    """
    num = c * np.log(np.array(e) / a)
    den = b - np.log(np.array(e) / a)
    return num / den


#########################################
##                                     ##
##                                     ##
##      Temperature rescaling          ##
##                                     ##
##                                     ##
#########################################


def temperature_rescaling_constant_lapse_rate(
    updated_data: xr.Dataset,
    dem: xr.DataArray | None,
    era5_data: xr.Dataset,
    era5_dem: xr.DataArray | None,
    lr_ta: npt.ArrayLike | float = -0.0065,
    lr_tdp: npt.ArrayLike | float = -0.0052,
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

        if data[dims["x"]].max() > 180:
            # x = (x + 360) % 360
            x = ((x + 180) % 360) - 180
    # Interpolate air temperature and dewpoint temperature at the specified
    # location

    ta, td = (
        xr.DataArray(
            kelvin_to_celsius(
                data["ta"].interp(
                    {dims["x"]: x, dims["y"]: y}, method="nearest"
                )
            )
        ),
        xr.DataArray(
            kelvin_to_celsius(
                data["td"].interp(
                    {dims["x"]: x, dims["y"]: y}, method="nearest"
                )
            )
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


##########################################################################
#################  Variable lapse rate ###################################
##########################################################################

# MONTHLY LAPSE RATE
LAPSE_RATE_BY_MONTH = {
    1: -0.0044,
    2: -0.0059,
    3: -0.0071,
    4: -0.0078,
    5: -0.0081,
    6: -0.0082,
    7: -0.0081,
    8: -0.0081,
    9: -0.0077,
    10: -0.0068,
    11: -0.0065,
    12: -0.0047,
}
VAPOR_PRESSURE_BY_MONTH = {
    1: 0.00041,
    2: 0.00042,
    3: 0.0004,
    4: 0.00039,
    5: 0.00038,
    6: 0.00036,
    7: 0.00033,
    8: 0.00033,
    9: 0.00036,
    10: 0.00037,
    11: 0.00040,
    12: 0.00040,
}


def get_lapse_rate_monthly(dt: dt.date) -> float:
    return LAPSE_RATE_BY_MONTH[dt.month]


def get_vapor_pressure_monthly(dt: dt.date) -> float:
    return VAPOR_PRESSURE_BY_MONTH[dt.month]


def compute_dewpoint_lr(
    coeff: float | npt.ArrayLike, b: float = 17.502, c: float = 240.97
) -> float | npt.NDArray[np.float64]:
    arr = np.asarray(coeff, dtype=float)
    result = -arr * c / b

    if arr.ndim == 0:
        return float(result)
    return result


def vapor_pressure_model(z, em0, am, z0):
    return em0 * np.exp(-am * (z - z0))


# LAPSE RATE EVERY DAY
def compute_lapse_rate_from_2_levels_roi(
    ds: xr.Dataset,
    temperature_var: str = "ta",
    height_var: str = "z",
    level_dim: str = "pressure_level",
) -> xr.DataArray:
    """
    Compute time-dependent lapse rate between two pressure levels.

    The lapse rate is computed as:
        (T_high - T_low) / (Z_high - Z_low)

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing exactly two pressure levels.
        Must include temperature and height variables.
    temperature_var : str
        Name of temperature variable.
    height_var : str
        Name of height variable.
    level_dim : str
        Name of pressure level dimension.

    Returns
    -------
    xr.DataArray
        Lapse rate as a function of time.
    """

    if ds.sizes[level_dim] != 2:
        raise ValueError("Dataset must contain exactly two pressure levels.")

    t = ds[temperature_var]
    z = ds[height_var] / G_CST

    # sort by height
    order = z.mean(("latitude", "longitude")).argsort()

    t = t.isel({level_dim: order})
    z = z.isel({level_dim: order})

    # level index 0 = low, 1 = high
    t_low = t.isel({level_dim: 0})
    t_high = t.isel({level_dim: 1})

    z_low = z.isel({level_dim: 0})
    z_high = z.isel({level_dim: 1})

    lapse_rate = (t_high - t_low) / (z_high - z_low)

    lapse_rate.name = "lapse_rate"
    lapse_rate.attrs["units"] = "°C m-1"
    lapse_rate.attrs["description"] = (
        "Temperature lapse rate between two pressure levels"
    )

    return lapse_rate


def compute_lapse_rate_from_2_levels(
    ds: xr.Dataset,
    temperature_var: str = "t",
    height_var: str = "z",
    level_dim: str = "pressure_level",
) -> xr.DataArray:
    """
    Compute time-dependent lapse rate between two pressure levels.

    The lapse rate is computed as:
        (T_high - T_low) / (Z_high - Z_low)

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing exactly two pressure levels.
        Must include temperature and height variables.
    temperature_var : str
        Name of temperature variable.
    height_var : str
        Name of height variable.
    level_dim : str
        Name of pressure level dimension.

    Returns
    -------
    xr.DataArray
        Lapse rate as a function of time.
    """

    if ds.sizes[level_dim] != 2:
        raise ValueError("Dataset must contain exactly two pressure levels.")

    t = ds[temperature_var]
    z = ds[height_var] / G_CST

    # level index 0 = low, 1 = high
    t_low = t.isel({level_dim: 0})
    t_high = t.isel({level_dim: 1})

    z_low = z.isel({level_dim: 0})
    z_high = z.isel({level_dim: 1})

    lapse_rate = (t_high - t_low) / (z_high - z_low)

    lapse_rate.name = "lapse_rate"
    lapse_rate.attrs["units"] = "°C m-1"
    lapse_rate.attrs["description"] = (
        "Temperature lapse rate between two pressure levels"
    )

    return lapse_rate


def interpolate_temperature_variant(
    src_temp: xr.DataArray,
    src_dem: xr.DataArray,
    dst_dem: xr.DataArray,
    lapse_rate: xr.DataArray,
) -> xr.DataArray:
    """
    Interpolate temperature on a new DEM using a potentially time- and
    space-varying lapse rate.

    Parameters
    ----------
    src_temp : xr.DataArray
        Temperature on source grid (dims: time, lat, lon)
    src_dem : xr.DataArray
        Elevation on source grid (dims: lat, lon)
    dst_dem : xr.DataArray
        Elevation on destination DEM (dims: x, y )
    lapse_rate : float or xr.DataArray
        Lapse rate (can be scalar or dims=(time, lat, lon))

    Returns
    -------
    dem_temp : xr.DataArray
        Temperature interpolated on the destination DEM (dims: time, lat, lon)
    """
    # Compute reference temperature at src_ref = 0 (or any reference level)
    # Broadcast automatically src_dem to match lapse_rate
    logger.info(f"src_dem ={src_dem}")
    logger.info(f"src_temp ={src_temp}")
    logger.info(f"lapse_rate ={lapse_rate}")
    # lapse_rate_bis = lapse_rate.rio.reproject_match(
    #     src_temp, resampling=rio.enums.Resampling.nearest
    # )
    ref_temp = src_temp - lapse_rate * src_dem
    # ref_temp = ref_temp.transpose("time", "latitude", "longitude")

    # logger.info(f"CRS DE REF_TEMP :{ref_temp.rio.crs}")
    # logger.info(f"CRS DE SRC_TEMP :{src_temp.rio.crs}")
    # logger.info(f"CRS DE SRC_DEM :{src_dem.rio.crs}")
    # logger.info(f"CRS DE LAPSE_RATE :{lapse_rate.rio.crs}")

    logger.info(f"ref_temp ={ref_temp}")

    # logger.info(f"dst_dem ={dst_dem}")
    # logger.info(f"CRS DE REF_TEMP :{ref_temp.rio.crs}")
    # logger.info(f"CRS DE DST_DEM :{dst_dem.rio.crs}")

    projected_temp = ref_temp.rio.reproject_match(
        dst_dem,
        resampling=rio.enums.Resampling.bilinear,
        # rio.enums.Resampling.bilinear,
        # rio.enums.Resampling.nearest,
    )
    logger.info(f"projected_temp = {projected_temp}")
    if (
        {"latitude", "longitude"} <= set(lapse_rate.dims)
        and lapse_rate.sizes["latitude"] > 1
        and lapse_rate.sizes["longitude"] > 1
    ):
        lr = lapse_rate.rio.reproject_match(
            dst_dem, resampling=rio.enums.Resampling.bilinear
        )
    else:
        lr = lapse_rate
    logger.info(f"lr reprojection  ={lr}")

    # Adjust to actual DEM elevation
    dem_temp = projected_temp + lr * dst_dem

    # Keep attributes
    dem_temp.attrs.update(src_temp.attrs)
    return dem_temp


def rescale_temperature_with_variable_lapserate(
    dem: xr.DataArray | None,
    era5_data: xr.DataArray | None,
    era5_dem: xr.DataArray | None,
    lapse_rate: xr.DataArray,
    key: str,
    description: str,
) -> xr.DataArray | None:
    """
    Rescale temperature using a time-variable lapse rate.
    """

    if era5_data is None:
        logger.warning(
            f"Skip {description} interpolation because data is missing"
        )
        return None

    if dem is None or era5_dem is None:
        logger.warning(
            f"Skip {description} interpolation because DEM is missing"
        )
        return None

    data = interpolate_temperature_variant(
        src_temp=era5_data,
        src_dem=era5_dem,
        dst_dem=dem,
        lapse_rate=lapse_rate,
    )

    data.attrs.update(
        {
            "standard_name": key,
            "long_name": description,
            "units": "K",
            "description": description,
            "lapse_rate_type": "time-variable",
        }
    )
    return data


def add_dewpoint_to_ds_roi(
    ds: xr.Dataset,
    temp_var: str = "t",
    rh_var: str = "r",
    output_var: str = "tdp",
) -> xr.Dataset:
    """
    Compute the dew point temperature from temperature and
    relative humidity and add it to an xarray Dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset containing temperature and relative humidity.
    temp_var : str, default "t"
        Name of the air temperature variable (expected in Kelvin).
    rh_var : str, default "r"
        Name of the relative humidity variable (in %).
    output_var : str, default "tdp"
        Name of the dew point variable to be added to the dataset.

    Returns
    -------
    xr.Dataset
        A new dataset with the dew point temperature added
        (in Kelvin).
    """
    ds = ds.copy()
    t = ds[temp_var]
    rh = ds[rh_var]
    rh = xr.where(rh <= 0, np.nan, rh)
    # Convert temperature to Celsius
    t_c = kelvin_to_celsius(t)

    # Compute dew point in Celsius
    td_c = compute_dewpoint_temp(t_c, rh)

    # Convert back to Kelvin
    td_k = celsius_to_kelvin(td_c)

    ds[output_var] = (t.dims, td_k)
    ds[output_var].attrs.update(
        {"units": "K", "long_name": "Dew point temperature"}
    )
    ds = ds[["t", "tdp"]]
    ds = ds.rename({"t": "ta"})
    return ds


def add_dewpoint_to_ds(
    ds: xr.Dataset,
    temp_var: str = "t",
    rh_var: str = "r",
    output_var: str = "td",
) -> xr.Dataset:
    """
    Compute the dew point temperature from temperature and
    relative humidity and add it to an xarray Dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset containing temperature and relative humidity.
    temp_var : str, default "t"
        Name of the air temperature variable (expected in Kelvin).
    rh_var : str, default "r"
        Name of the relative humidity variable (in %).
    output_var : str, default "tdp"
        Name of the dew point variable to be added to the dataset.

    Returns
    -------
    xr.Dataset
        A new dataset with the dew point temperature added
        (in Kelvin).
    """

    ds = ds.copy()
    t = ds[temp_var]
    rh = ds[rh_var]
    rh = xr.where(rh <= 0, np.nan, rh)
    # Convert temperature to Celsius
    t_c = kelvin_to_celsius(t)

    # Compute dew point in Celsius
    td_c = compute_dewpoint_temp(t_c, rh)

    # Convert back to Kelvin
    td_k = celsius_to_kelvin(td_c)

    ds[output_var] = (t.dims, td_k)
    ds[output_var].attrs.update(
        {"units": "K", "long_name": "Dew point temperature"}
    )
    return ds


def temperature_rescaling_variable_lapse_rate(
    updated_data: xr.Dataset,
    dataset: ERA5Dataset,
    dem: xr.DataArray | None,
    era5_data: xr.Dataset,
    era5_dem: xr.DataArray | None,
    lr_ta: xr.DataArray,
    lr_tdp: xr.DataArray,
) -> xr.Dataset:
    """
    Description
    -----------
    Rescale ERA5 temperature variables using a variable lapse rate.

    The lapse rate can depend on time (and only time).

    -> In a first step, the temperatures are adjusted to a fixed reference
    elevation using a constant lapse rate.


    -> In a second step, if a target DEM is provided, the temperatures are
    further rescaled from the reference elevation to the DEM elevation.

    The resulting temperature variables are added to the updated dataset.

    Parameters
    ----------
    updated_data: xr.Dataset
        Updated Data
    dem : xr.DataArray | None
        dem
    era5_data: xr.Dataset
        ERA5 Data
    era5_dem: xr.DataArray | None
        ERA5 Data
    lr_ta: xr.
    lr_tdp :

    Return
    -----------
    updated_data : xr.Dataset
        Updated data
    """
    ta_data: xr.DataArray | None = None

    if dataset in (ERA5Dataset.ERA5, ERA5Dataset.ERA5LAND):
        ta_data = era5_data.get(ERA5Var.TEMPERATURE.key)

    elif dataset == ERA5Dataset.ERA5PRESSURE:
        ta_data = era5_data.get(ERA5pressureVar.TEMPERATURE.key)
        if ta_data is not None and "pressure_level" in ta_data.dims:
            if ta_data.sizes["pressure_level"] != 1:
                raise ValueError(
                    "Cannot compute temperature: ERA5PRESSURE dataset "
                    "contains multiple pressure levels. "
                    "Select a single pressure level first."
                )
            ta_data = ta_data.squeeze("pressure_level")

    if ta_data is not None:
        rescaled = rescale_temperature_with_variable_lapserate(
            dem=dem,
            era5_data=ta_data,
            era5_dem=era5_dem,
            lapse_rate=lr_ta,
            key="ta",
            description="air temperature",
        )

        if rescaled is not None:
            updated_data["ta"] = rescaled
            logger.debug("Add temperature (variable lapse rate): OK")

    tdp_data: xr.DataArray | None = None

    # Dew point temperature
    if dataset in (ERA5Dataset.ERA5, ERA5Dataset.ERA5LAND):
        tdp_data = era5_data.get(ERA5Var.DEWPOINT_TEMPERATURE.key)

    elif dataset == ERA5Dataset.ERA5PRESSURE:
        temperature = era5_data.get("t")
        rh = era5_data.get("r")
        td = era5_data.get("td")

        # If dewpoint already present
        if td is not None:
            tdp_data = td
            if "pressure_level" in tdp_data.dims:
                if tdp_data.sizes["pressure_level"] != 1:
                    raise ValueError(
                        "Cannot compute dewpoint: multiple pressure levelsfound"
                    )
                tdp_data = tdp_data.squeeze("pressure_level")
        elif temperature is not None and rh is not None:
            if (
                "pressure_level" in temperature.dims
                and temperature.sizes["pressure_level"] != 1
            ):
                raise ValueError(
                    "Cannot compute dewpoint: ERA5PRESSURE dataset contains "
                    "multiple pressure levels. Select a single level first."
                )

            era5_data = add_dewpoint_to_ds(era5_data)
            tdp_data = era5_data["td"]
            tdp_data = tdp_data.squeeze("pressure_level")

    if tdp_data is not None:
        rescaled = rescale_temperature_with_variable_lapserate(
            dem=dem,
            era5_data=tdp_data,
            era5_dem=era5_dem,
            lapse_rate=lr_tdp,
            key="tdp",
            description="dewpoint temperature",
        )

        if rescaled is not None:
            updated_data["tdp"] = rescaled
            logger.debug("Add dewpoint temperature (variable lapse rate): OK")
    else:
        logger.debug("No dewpoint temperature available in dataset : skipped")

    return updated_data


def compute_ah(
    z: xr.DataArray, e: xr.DataArray
) -> tuple[float, float, float, float]:
    """
    Compute the vertical coefficient 'ah' from the exponential model:

        e(z) = e(z0) * exp(-ah * (z - z0))

    where:
        - 'z0' is the elevation of the reference location,
        - 'z' is the elevation at other vertical levels,
        - 'ah' is an hourly-dependent coefficient,
        - 'e(z)' is the water vapor pressure at height z.

    Method
    ------
    The exponential relationship is linearized by taking the natural logarithm:

        ln(e(z)) = ln(e(z0)) - ah * (z - z0)

    which can be written in linear regression form:

        y = b + a * x

    with:
        x = z - z0
        y = ln(e(z))
        b = ln(e(z0))
        a = -ah

    A first-order polynomial fit (np.polyfit) is applied to (x, y)
    to estimate the slope 'a' and intercept 'b'. The coefficient 'ah' is
    obtained as:

        ah = -a

    The fit is evaluated using the coefficient of
    determination (R2) computed between observed and predicted
    ln(e).

    Parameters
    ----------
    z : xr.DataArray
        DataArray containing geopotentials
    e : xr.DataArray
        DataArray containing water vaport pressure

    Returns
    -------
    ah : float
        Vertical exponential  coefficient (km-1).

    r2 : float
        Coefficient of determination of the linear fit in log-space.
    """

    # Convert geopotential height to geometric height in km

    # z / G_CST converts geopotential to meters, then /1000 converts to km
    z_data = z / G_CST

    # Convert vapor pressure from Pa to kPa (optional, does not affect slope)
    e_data = e

    # Create a mask to keep only valid data: finite z, finite e, and e > 0
    mask = np.isfinite(z_data) & np.isfinite(e_data) & (e_data > 0)

    # Skip this time step if there are less than 2 valid points
    if np.sum(mask) < 2:
        return np.nan, np.nan, np.nan, np.nan

    # Keep only valid values
    z_clean = z_data[mask]
    e_clean = e_data[mask]

    # Reference height (z0) at the first valid level
    # This will be used as the base for the exponential
    z0 = z_clean[0]

    # Linearize the exponential relation:
    # e(z) = e(z0) * exp(-ah * (z - z0))
    # Taking natural log:
    # ln(e(z)) = ln(e(z0)) - ah * (z - z0)
    # Which can be expressed in standard linear regression form:
    # y = a*x + b
    # y = ln(e(z))
    # x = z - z0
    # a = -ah
    # b = ln(e(z0))

    x = z_clean - z0
    y = np.log(e_clean)

    # linear regression on (x, y)
    a, b = np.polyfit(x, y, 1)

    # Predicted values for computing R2
    y_pred = a * x + b
    r2 = r2_score(y, y_pred)

    return -a, np.exp(b), float(z0), r2

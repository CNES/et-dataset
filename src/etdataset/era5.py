#
# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales /
#            Université Paul Sabatier (UT3)
#
"""
Module for reading ERA5 and ERA5-land
"""

import datetime as dt
import os
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import datetime, time
from enum import Enum
from functools import lru_cache
from pathlib import Path
from time import sleep

import cdsapi
import pandas as pd
import rasterio as rio
import xarray as xr
from pyproj import CRS

from etdataset.logging import LoggerManager

logger = LoggerManager.get_logger(__name__)

# Gravitational constant
G_CST = 9.80665


class ERA5Exception(Exception):
    """
    Exception for ERA5
    """


@dataclass
class DatasetInfo:
    """Class for dataset info"""

    key: str
    label: str
    variables: list[str]


class ERA5Dataset(DatasetInfo, Enum):
    """
    ERA5 dataset
    """

    ERA5 = (
        "era5",
        "reanalysis-era5-single-levels",
        [
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "2m_dewpoint_temperature",
            "2m_temperature",
            "surface_solar_radiation_downward_clear_sky",
            "surface_solar_radiation_downwards",
            "surface_thermal_radiation_downward_clear_sky",
            "surface_thermal_radiation_downwards",
            "total_column_ozone",
            "total_column_water",
            "total_precipitation",
            "total_column_water_vapour",
        ],
    )
    ERA5LAND = (
        "era5land",
        "reanalysis-era5-land",
        [
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "2m_dewpoint_temperature",
            "2m_temperature",
            "surface_solar_radiation_downwards",
            "surface_thermal_radiation_downwards",
            "total_precipitation",
            "total_evaporation",
        ],
    )


@dataclass
class ERA5DataInfo:
    """Class for describing ERA5 data"""

    key: str
    label: str
    unit: str


class ERA5Var(ERA5DataInfo, Enum):
    """
    ERA5 variables
    """

    DEWPOINT_TEMPERATURE = ("d2m", "2m dewpoint temperature", "K")
    TEMPERATURE = ("t2m", "2m temperature", "K")
    GEOPOTENTIAL = ("z", "Geopotential", "m2 s-2")
    HEIGHT = ("h", "Geopotential height", "m")
    SURFACE_PRESSURE = ("sp", "Surface pressure", "Pa")
    SURFACE_SOLAR_RADIATION_DOWNWARD_CLEAR_SKY = (
        "ssrdc",
        "Surface solar radiation downward, clear sky",
        "J m-2",
    )
    SURFACE_SOLAR_RADIATION_DOWNWARD = (
        "ssrd",
        "Surface solar radiation downwards",
        "J m-2",
    )
    SURFACE_THERMAL_RADIATION_DOWNWARD_CLEAR_SKY = (
        "strdc",
        "Surface thermal radiation downward, clear sky",
        "J m-2",
    )
    SURFACE_THERMAL_RADIATION_DOWNWARD = (
        "strd",
        "Surface thermal radiation downwards",
        "J m-2",
    )
    TOTAL_COLUMN_OZONE = ("tco3", "Total column ozone", "kg m-2")
    TOTAL_COLUMN_WATER = ("tcw", "Total column water", "kg m-2")
    TOTAL_COLUMN_WATER_VAPOR = ("tcwv", "Total column water vapour", "kg m-2")
    TOTAL_PRECIPITATION = ("tp", "Total precipitation", "m")
    U_WIND = ("u10", "10m u-component of wind", "m s-1")
    V_WIND = ("v10", "10m v-component of wind", "m s-1")
    TOTAL_EVAPORATION = ("e", "Total evaporation", "m")

    @classmethod
    def from_key(cls, key):
        """
        Create enum from a key value
        """
        for value in cls:
            if value.key == key:
                return value
        raise ValueError(f"No variable found with key {key}")

    @classmethod
    def _missing_(cls, value):
        """
        Overload the missing method to call from_key method
        if enum is instanciated with a string
        """
        if isinstance(value, str):
            return cls.from_key(value)
        return super()._missing_(value)


class DataVar(ERA5DataInfo, Enum):
    """
    ERA5 variables
    """

    DEWPOINT_TEMPERATURE = ("tp", "2m dewpoint temperature", "K")
    TEMPERATURE = ("ta", "2m air temperature", "K")
    SURFACE_SOLAR_RADIATION_DOWNWARD_CLEAR_SKY = (
        "rsd",
        "Surface solar radiation downward, clear sky",
        "J m-2",
    )
    SURFACE_SOLAR_RADIATION_DOWNWARD = (
        "rsd",
        "Surface solar radiation downwards",
        "J m-2",
    )
    SURFACE_THERMAL_RADIATION_DOWNWARD_CLEAR_SKY = (
        "rld",
        "Surface thermal radiation downward, clear sky",
        "J m-2",
    )
    SURFACE_THERMAL_RADIATION_DOWNWARD = (
        "rld",
        "Surface thermal radiation downwards",
        "J m-2",
    )
    TOTAL_COLUMN_OZONE = ("tco3", "Total column ozone", "kg m-2")
    TOTAL_COLUMN_WATER_VAPOR = ("tcwv", "Total column water vapour", "kg m-2")
    TOTAL_EVAPORATION = ("e", "Total evaporation", "m")

    @classmethod
    def from_key(cls, key):
        """
        Create enum from a key value
        """
        for value in cls:
            if value.key == key:
                return value
        raise ValueError(f"No variable found with key {key}")

    @classmethod
    def _missing_(cls, value):
        """
        Overload the missing method to call from_key method
        if enum is instanciated with a string
        """
        if isinstance(value, str):
            return cls.from_key(value)
        return super()._missing_(value)


def read(product: str) -> xr.Dataset:
    """
    Description
    -----------
    Read ERA5 product

    Parameter
    ---------
    product: str
        Path to ERA5 product

    Return
    ------
    data: xr.Dataset
        Data
    """
    if zipfile.is_zipfile(product):
        # Read archive content
        with zipfile.ZipFile(product, "r") as zip_file:
            file_list = zip_file.namelist()
            datasets: list[xr.Dataset] = [
                xr.open_dataset(zip_file.open(filename))  # type: ignore
                for filename in file_list
            ]
        # Merge
        return (
            xr.merge(datasets)
            .rio.write_crs(CRS("4236"))
            .rename({"valid_time": "time"})
            .drop_vars("number")
        )
    return xr.open_dataset(product)


@lru_cache
def get_era5_dem() -> xr.DataArray:
    """
    Get DEM for ERA5

    Returns
    -------
    dem: xr.DataArray
        ERA5 DEM
    """
    data = xr.open_dataarray(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "data",
            "geopotential_era5.nc",
        )
    )
    return data / G_CST


@lru_cache
def get_era5land_dem() -> xr.DataArray:
    """
    Get DEM for ERA5Land

    Returns
    -------
    dem: xr.DataArray
        ERA5 DEM
    """
    data = xr.open_dataarray(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "data",
            "geopotential_era5land.zarr",
        )
    )
    return data / G_CST


def interpolate_time(
    data: xr.Dataset, date: datetime, variables: list[ERA5Var] | None = None
) -> xr.Dataset:
    """
    Description
    -----------
    Compute a linear time interpolation on data
    at a specific date for a list of variables.

    Parameters
    ----------
    data: xr.Dataset
        Dataset all the variables and the dates
    date: dt.datetime
        Date at which interpolation is computed
    variables: list[ERA5Var]
        List of Variables to consider for the interpolation

    Return
    ------
    interpolated_data: xr.Dataset
        Dataset interpolated
    """
    # Check variable
    if variables is None:
        vars_str = list(data.data_vars)
    else:
        vars_str = []
        for var in variables:
            if var.key not in list(data.data_vars):
                raise KeyError(f"Variable {var} not found")
            vars_str.append(var.key)
    # Get all available dates for time interpolation
    dates_str: list[str] = sorted(
        [
            dt.strftime("%Y-%m-%d %H:%M:%S")
            for dt in data.coords["time"].data.astype("M8[ms]").astype("O")
            if date.date() == dt.date()
        ]
    )
    if len(dates_str) == 0:
        raise KeyError(f"No date can be used for interpolation at {date}")
    # Select data used for interpolation
    selected = data[vars_str].sel(time=dates_str)
    # Interpolate for the acquisition time
    date_str = date.strftime("%Y-%m-%d %H:%M:%S")
    if selected.sizes["time"] == 1:
        logger.warning("No interpolation, only one timestamp available")
        return selected.isel(time=0, drop=True)
    return selected.interp(time=date_str, method="linear").drop_vars("time")


def interpolate_temperature(
    src_temp: xr.DataArray,
    src_dem: xr.DataArray,
    dst_dem: xr.DataArray,
    lapse_rate: float,
) -> xr.DataArray:
    """
    Description
    -----------
    This method interpolates temperature data on a new grid
    by taking into account altitude.

    First, the ERA5 temperature is adjusted to a common level,
    using the formula  Tref = Tin + rate x (Zref - Zin).
    Zout is a reference elevation taken as Zout = 0.
    The reference-level temperature is then projected
    from the original geographic coordinate system (i.e. WGS84 for ERA5)
    onto the projection coordinate system of the destination DEM
    using bilinear interpolation.
    The DEM data and lapse rate are then used to adjust the
    reference-level gridded temperature to the elevations provided
    by the DEM, using  Tdem = Tref + rate x (Zdem - Zref), where
    where Tref is now the gridded temperature at the reference elevation Zref,
    and Tdem is the gridded temperature at the elevation of the DEM Zdem.

    Parameters
    ----------
    src_temp: xr.DataArray
        Temperature (reference grid)
    src_dem: xr.DataArray
        Elevation associated to the temperature (reference grid)
    dst_dem: xr.DataArray
        Elevation associated to the temperature (projection grid)
    lapse_rate: float
        Gradient of temperature per unit of elevation

    Return
    ------
    dem_temp: xr.DataArray
        Temperature projected on the new DEM
    """
    # Compute temperature at reference elevation
    ref_temp = src_temp - lapse_rate * src_dem
    # Project into the coordinate system of the destination DEM
    projected_temp = ref_temp.rio.reproject_match(
        dst_dem,
        resampling=rio.enums.Resampling.bilinear,
    )
    # Adjust temperatures to DEM elevation
    dem_temp = projected_temp + lapse_rate * dst_dem
    return dem_temp


def interpolate_ozone(
    data: xr.DataArray,
    dem: xr.DataArray,
) -> xr.DataArray:
    """
    Description
    -----------
    This method interpolates ozone on a new grid.

    The reference-level ozone is projected
    from the original geographic coordinate system (i.e. WGS84 for ERA5)
    onto the projection coordinate system of the destination DEM
    using bilinear interpolation.

    No elevation correction is performed.

    Parameters
    ----------
    data: xr.DataArray
        Data to project
    dem:  xr.DataArray
        Grid used for the projection

    Return
    ------
    projected: xr.DataArray
        Data projected
    """
    return data.rio.reproject_match(
        dem,
        resampling=rio.enums.Resampling.bilinear,
    )


def interpolate_evaporation(
    data: xr.DataArray,
    dem: xr.DataArray,
) -> xr.DataArray:
    """
    Description
    -----------
    This method interpolates evaporation on a new grid.

    The reference-level evaporation is projected
    from the original geographic coordinate system (i.e. WGS84 for ERA5)
    onto the projection coordinate system of the destination DEM
    using bilinear interpolation.

    No elevation correction is performed.

    Parameters
    ----------
    data: xr.DataArray
        Data to project
    dem:  xr.DataArray
        Grid used for the projection

    Return
    ------
    projected: xr.DataArray
        Data projected
    """
    return data.rio.reproject_match(
        dem,
        resampling=rio.enums.Resampling.bilinear,
    )


def interpolate_tcvw(
    data: xr.DataArray,
    dem: xr.DataArray,
) -> xr.DataArray:
    """
    Description
    -----------
    This method interpolates total column
    vapor water on a new grid.

    The reference-level tcvw is projected
    from the original geographic coordinate system (i.e. WGS84 for ERA5)
    onto the projection coordinate system of the destination DEM
    using bilinear interpolation.

    No elevation correction is performed.

    Parameters
    ----------
    data: xr.DataArray
        Data to project
    dem:  xr.DataArray
        Grid used for the projection

    Return
    ------
    projected: xr.DataArray
        Data projected
    """
    return data.rio.reproject_match(
        dem,
        resampling=rio.enums.Resampling.bilinear,
    )


def interpolate_radiation(
    data: xr.DataArray,
    dem: xr.DataArray,
) -> xr.DataArray:
    """
    Description
    -----------
    This method interpolates radiation data on a new grid.

    The reference-level ozone is projected
    from the original geographic coordinate system (i.e. WGS84 for ERA5)
    onto the projection coordinate system of the destination DEM
    using bilinear interpolation.

    No slope/aspect correction is performed.

    Parameters
    ----------
    data: xr.DataArray
        Data to project
    dem:  xr.DataArray
        Grid used for the projection

    Return
    ------
    projected: xr.DataArray
        Data projected
    """
    # Project into the coordinate system of the destination DEM
    return data.rio.reproject_match(
        dem,
        resampling=rio.enums.Resampling.nearest,
    )


def _download(dataset: str, request: dict, target: str) -> None:
    """
    Description
    -----------
    Download data from ERA5

    Parameters
    ----------
    dataset: str
        Dataset name
    request: dict
        Data request for download
    target: str
        Target filename
    """
    creds = Path(Path.home(), Path(".cdsapirc"))
    if not creds.is_file():
        raise FileNotFoundError("Credentials for CDS are missing.")

    client = cdsapi.Client(timeout=600, wait_until_complete=False, delete=False)
    result = client.retrieve(dataset, request)
    delta_sleep = 30

    while True:
        result.update()
        reply = result.reply

        if reply["state"] == "completed":
            break
        if reply["state"] in ("queued", "running"):
            sleep(delta_sleep)
        elif reply["state"] in ("failed",):
            result.error(f"Message: {reply['error'].get('message')}")
            result.error(f"Reason:  {reply['error'].get('reason')}")
            for n in (
                reply.get("error", {})
                .get("context", {})
                .get("traceback", "")
                .split("\n")
            ):
                if n.strip() == "":
                    break
                result.error(f"  {n}")
            raise ERA5Exception(
                f"{reply['error'].get('message')}  "
                f"{reply['error'].get('reason')}"
            )
    result.download(target)


def download(
    date: datetime,
    dataset: ERA5Dataset = ERA5Dataset.ERA5,
    variables: list[str] | None = None,
    path: str | None = None,
) -> None:
    """
    Description
    -----------
    Download data from a ERA5 dataset

    Parameters
    ----------
    date: dt.datetime
        Date
    dataset: ERA5Dataset
        ERA5 Dataset used for download
    variables: list[str]
        List of product to download
    path: str
        Directory path to store data
    """
    if path is None:
        path = os.getcwd()
    # Create a directory MSG products
    era5_path = os.path.join(path, "ERA5_data")
    os.makedirs(era5_path, exist_ok=True)
    # Filename
    filename = os.path.join(
        era5_path,
        f"download_{dataset.key}_{date.strftime('%Y-%m-%d')}.zip",
    )
    # Skip download if file already exists
    if os.path.exists(filename):
        logger.info(f"File {filename} already exits. Skip download.")
        return
    # Dataset
    if variables is None:
        variables = dataset.variables
    request = {
        "product_type": "reanalysis",
        "variable": variables,
        "year": date.year,
        "month": date.month,
        "day": date.day,
        "time": [
            "01:00",
            "02:00",
            "03:00",
            "04:00",
            "05:00",
            "06:00",
            "07:00",
            "08:00",
            "09:00",
            "10:00",
            "11:00",
            "12:00",
            "13:00",
            "14:00",
            "15:00",
            "16:00",
            "17:00",
            "18:00",
            "19:00",
            "20:00",
            "21:00",
            "22:00",
            "23:00",
        ],
        "data_format": "netcdf",
        "download_format": "zip",
    }
    # Download
    _download(dataset.label, request, filename)


def rescale_temperature_with_lapserate(
    dem: xr.DataArray | None,
    era5_data=xr.DataArray | None,
    era5_dem=xr.DataArray | None,
    lapse_rate=float,
    key=str,
    description=str,
) -> xr.DataArray | None:
    """
    Resacle temperature using constant lapse rate

    Parameters
    ----------
    dem: xr.DataArray
        DEM used to rescale data
    era5_data: xr.DataArray
        Data to rescaled
    era5_dem: xr.DataArray
        DEM corresponding to data to rescaled
    lapse_rate: float
        Lapse rate
    key: str
        Variable name
    description: str
        Variable description

    Returns
    -------
    data: xr.DataArray
        Rescaled data
    """
    if era5_data is None:
        msg = f"Skip {description} interpolation because  data is missing"
        logger.warning(msg)
        return None
    if dem is None:
        logger.warning("Skip temperature interpolation because DEM is missing")
        return None
    data = interpolate_temperature(
        src_temp=era5_data,
        src_dem=era5_dem,
        dst_dem=dem,
        lapse_rate=lapse_rate,
    )
    data.attrs["standard_name"] = key
    data.attrs["long_name"] = description
    data.attrs["name"] = key
    data.attrs["unit"] = "K"
    data.attrs["description"] = description
    return data


def rescale_radiation(
    dem: xr.DataArray,
    era5_data=xr.DataArray | None,
    key=str,
    description=str,
) -> xr.DataArray | None:
    """
    Rescale radiation using constant lapse rate

    Parameters
    ----------
    dem: xr.DataArray
        DEM or grid used to rescale data
    era5_data: xr.DataArray
        Data to rescaled
    key: str
        Variable name
    description: str
        Variable description

    Returns
    -------
    data: xr.DataArray
        Rescaled data
    """
    radiation_factor = 3600.0
    if era5_data is None:
        logger.warning(
            f"Skip {description} interpolation because data is missing"
        )
        return None
    data = (
        interpolate_radiation(
            data=era5_data,
            dem=dem,
        )
        / radiation_factor
    )
    data.attrs["standard_name"] = key
    data.attrs["long_name"] = description
    data.attrs["name"] = key
    data.attrs["unit"] = "W.m-2"
    data.attrs["description"] = description
    return data


def add(
    data: xr.Dataset,
    dataset: ERA5Dataset = ERA5Dataset.ERA5,
    variables: list[ERA5Var] | None = None,
    path: str | None = None,
) -> xr.Dataset:
    """
    Description
    -----------
    Add ERA5 data to the dataset

    Parameters
    ----------
    data: xr.Dataset
        Data
    dataset: ERA5Dataset
        ERA5 Dataset used for download
    variables: list[ERA5Var]
        List of variables to add
    path: str
        Directory where ERA5 data have been downloaded data
    """
    ##############
    # Check inputs
    ##############
    if data.attrs.get("vis_date", None) is None:
        raise ValueError("Vis date attribute is missing in dataset")
    if data.attrs.get("vis_time", None) is None:
        raise ValueError("Vis time attribute is missing in dataset")
    if len(data.data_vars) == 0:
        raise ValueError("Dataset is empty")
    # Extract data
    date = dt.datetime.combine(data.attrs["vis_date"], data.attrs["vis_time"])
    if data.rio.crs is None:
        if data.attrs.get("crs") is not None:
            crs = data.attrs["crs"]
            data = data.rio.write_crs(crs)
        raise ValueError("crs attribute is missing in dataset")
    crs = data.rio.crs
    dem = data.get("height", None)
    if dem is not None:
        dem = dem.rio.write_crs(crs)
    temp_dir = None
    # Configure variables if necessary
    if variables is None:
        if dataset == ERA5Dataset.ERA5:
            variables = [
                ERA5Var.TEMPERATURE,
                ERA5Var.DEWPOINT_TEMPERATURE,
                ERA5Var.SURFACE_SOLAR_RADIATION_DOWNWARD_CLEAR_SKY,
                ERA5Var.SURFACE_THERMAL_RADIATION_DOWNWARD_CLEAR_SKY,
            ]
        else:
            variables = [
                ERA5Var.TEMPERATURE,
                ERA5Var.DEWPOINT_TEMPERATURE,
                ERA5Var.SURFACE_SOLAR_RADIATION_DOWNWARD,
                ERA5Var.SURFACE_THERMAL_RADIATION_DOWNWARD,
            ]
    # Download data if necessary
    if path is None:
        # Download data
        logger.debug("Download ERA5 data")
        # Create a temp directory
        temp_dir = tempfile.TemporaryDirectory()
        path = temp_dir.name
        logger.debug(f"Temp dir: {path}")
        # Download files
        download(date=date, dataset=dataset, path=path)
    logger.debug("Check data")
    ############################
    # Prepare ERA5/ERA5Land data
    ############################
    # ERA5/ERA5Land data path
    product_path = os.path.join(
        path,
        "ERA5_data",
        f"download_{dataset.key}_{date.strftime('%Y-%m-%d')}.zip",
    )
    logger.debug(f"Product path: {product_path}")
    if not os.path.isfile(product_path):
        raise OSError(f"ERA5 data not found: {product_path}")
    # Read data
    era5_xrds = read(product=product_path)
    era5_xrds = era5_xrds[[var.key for var in variables]]
    logger.debug("Read ERA5 product:OK")
    logger.debug(f"Variables : {list(era5_xrds.data_vars)}")
    # Process accumulated variables for ERA5land dataset
    if (
        dataset == ERA5Dataset.ERA5LAND
        and ERA5Var.SURFACE_SOLAR_RADIATION_DOWNWARD.key in era5_xrds.data_vars
        and ERA5Var.SURFACE_THERMAL_RADIATION_DOWNWARD.key
        in era5_xrds.data_vars
    ):
        era5_xrds[ERA5Var.SURFACE_SOLAR_RADIATION_DOWNWARD.key] = era5_xrds[
            ERA5Var.SURFACE_SOLAR_RADIATION_DOWNWARD.key
        ].diff(dim="time")
        era5_xrds[ERA5Var.SURFACE_THERMAL_RADIATION_DOWNWARD.key] = era5_xrds[
            ERA5Var.SURFACE_THERMAL_RADIATION_DOWNWARD.key
        ].diff(dim="time")
    # Interpolate time
    era5_xrds = interpolate_time(data=era5_xrds, date=date)
    logger.debug("Interpolate ERA5 product:OK")
    # Add DEM
    if dataset == ERA5Dataset.ERA5:
        era5_dem = get_era5_dem()
        era5_dem = xr.DataArray(
            era5_dem.data,
            dims=era5_xrds.dims,
            coords=era5_xrds.coords,
        ).rio.write_crs(CRS(4326))
    elif dataset == ERA5Dataset.ERA5LAND:
        era5_dem = get_era5land_dem()
        era5_dem = xr.DataArray(
            era5_dem.data,
            dims=era5_xrds.dims,
            coords=era5_xrds.coords,
        ).rio.write_crs(CRS(4326))
    else:
        era5_dem = None
    ##########
    # Add data
    ##########
    # Copy data
    updated_data = data.copy()
    updated_data.attrs = data.attrs.copy()
    #############
    # Temperature
    #############
    # Add temperature
    if ERA5Var.TEMPERATURE in variables:
        rescaled = rescale_temperature_with_lapserate(
            dem=dem,
            era5_data=era5_xrds.get(ERA5Var.TEMPERATURE.key, None),
            era5_dem=era5_dem,
            lapse_rate=-0.0065,
            key="ta",
            description="2m air temperature",
        )
        if rescaled is not None:
            updated_data["ta"] = rescaled
            logger.debug("Add temperature:OK")
    # Add dewpoint temperature temperature
    if ERA5Var.DEWPOINT_TEMPERATURE in variables:
        rescaled = rescale_temperature_with_lapserate(
            dem=dem,
            era5_data=era5_xrds.get(ERA5Var.DEWPOINT_TEMPERATURE.key, None),
            era5_dem=era5_dem,
            lapse_rate=-0.0052,
            key="tdp",
            description="dewpoint temperature",
        )
        if rescaled is not None:
            updated_data["tdp"] = rescaled
            logger.debug("Add dewpoint temperature:OK")
    ###########
    # Radiation
    ###########
    # Add solar radiation
    if ERA5Var.SURFACE_SOLAR_RADIATION_DOWNWARD in variables:
        dst = next(iter(data.data_vars.values()))
        rescaled = rescale_radiation(
            dem=dst.rio.write_crs(crs),  # transfer crs attribute
            era5_data=era5_xrds.get(
                ERA5Var.SURFACE_SOLAR_RADIATION_DOWNWARD.key, None
            ),
            key="rsd",
            description="shortwave downwelling radiation",
        )
        if rescaled is not None:
            updated_data[f"rsd_{dataset.key}"] = rescaled
            logger.debug("Add solar radiation:OK")
    # Add solar radiation
    if ERA5Var.SURFACE_SOLAR_RADIATION_DOWNWARD_CLEAR_SKY in variables:
        dst = next(iter(data.data_vars.values()))
        rescaled = rescale_radiation(
            dem=dst.rio.write_crs(crs),  # transfer crs attribute
            era5_data=era5_xrds.get(
                ERA5Var.SURFACE_SOLAR_RADIATION_DOWNWARD_CLEAR_SKY.key, None
            ),
            key="rsd",
            description="shortwave downwelling radiation (clear sky)",
        )
        if rescaled is not None:
            updated_data[f"rsd_{dataset.key}"] = rescaled
            logger.debug("Add solar radiation (clear sky):OK")
    # Add solar radiation
    if ERA5Var.SURFACE_THERMAL_RADIATION_DOWNWARD in variables:
        dst = next(iter(data.data_vars.values()))
        rescaled = rescale_radiation(
            dem=dst.rio.write_crs(crs),  # transfer crs attribute
            era5_data=era5_xrds.get(
                ERA5Var.SURFACE_THERMAL_RADIATION_DOWNWARD.key, None
            ),
            key="rld",
            description="longwave downwelling radiation",
        )
        if rescaled is not None:
            updated_data[f"rld_{dataset.key}"] = rescaled
            logger.debug("Add thermal radiation:OK")
    # Add solar radiation
    if ERA5Var.SURFACE_THERMAL_RADIATION_DOWNWARD_CLEAR_SKY in variables:
        dst = next(iter(data.data_vars.values()))
        rescaled = rescale_radiation(
            dem=dst.rio.write_crs(crs),  # transfer crs attribute
            era5_data=era5_xrds.get(
                ERA5Var.SURFACE_THERMAL_RADIATION_DOWNWARD_CLEAR_SKY.key, None
            ),
            key="rld",
            description="longwave downwelling radiation (clear sky)",
        )
        if rescaled is not None:
            updated_data[f"rld_{dataset.key}"] = rescaled
            logger.debug("Add thermal radiation (clear sky):OK")
    # Add ozone
    if (
        ERA5Var.TOTAL_COLUMN_OZONE in variables
        and ERA5Var.TOTAL_COLUMN_OZONE.key in era5_xrds.data_vars
    ):
        dst = next(iter(data.data_vars.values()))
        updated_data[f"tco3_{dataset.key}"] = interpolate_ozone(
            data=era5_xrds[ERA5Var.TOTAL_COLUMN_OZONE.key],
            dem=dst.rio.write_crs(crs),  # transfer crs attribute
        )
        updated_data[f"tco3_{dataset.key}"].attrs["standard_name"] = "tco3"
        updated_data[f"tco3_{dataset.key}"].attrs["long_name"] = (
            "Total column ozone"
        )
        updated_data[f"tco3_{dataset.key}"].attrs["name"] = "tco3"
        updated_data[f"tco3_{dataset.key}"].attrs["unit"] = "kg.m-2"
        updated_data[f"tco3_{dataset.key}"].attrs["description"] = (
            "Total column ozone"
        )
        logger.debug("Add total_column_ozone :OK")
    # Clean
    if temp_dir is not None:
        temp_dir.cleanup()  # Manually delete the directory
    return updated_data


def add_et(
    data: xr.Dataset,
    path: str | None = None,
) -> xr.Dataset:
    """
    Description
    -----------
    Add only the ERA5-Land "Total evaporation" product to the dataset

    Parameters
    ----------
    data: xr.Dataset
        Data
    path: str
        Directory where ERA5-Land data have been downloaded data
    """
    ##############
    # Check inputs
    ##############
    dataset = ERA5Dataset.ERA5LAND
    if data.attrs.get("vis_date", None) is None:
        raise ValueError("Vis date attribute is missing in dataset")
    if len(data.data_vars) == 0:
        raise ValueError("Dataset is empty")
    # Extract data
    date = dt.datetime.combine(data.attrs["vis_date"], time(12, 0))
    if data.rio.crs is None:
        if data.attrs.get("crs") is not None:
            crs = data.attrs["crs"]
            data = data.rio.write_crs(crs)
        raise ValueError("crs attribute is missing in dataset")
    crs = data.rio.crs
    temp_dir = None
    # Download data if necessary
    if path is None:
        # Download data
        logger.debug("Download ERA5 data")
        # Create a temp directory
        temp_dir = tempfile.TemporaryDirectory()
        path = temp_dir.name
        logger.debug(f"Temp dir: {path}")
        # Download files
        download(date=date, dataset=dataset, path=path)
    logger.debug("Check data")
    ##############
    # Prepare data
    ###############
    # ERA5-Land data path
    product_path = os.path.join(
        path,
        "ERA5_data",
        f"download_{dataset.key}_{date.strftime('%Y-%m-%d')}.zip",
    )
    logger.debug(f"Product path: {product_path}")
    if not os.path.isfile(product_path):
        raise OSError(f"ERA5 data not found: {product_path}")
    # Read data
    era5_xrds = read(product=product_path)
    logger.debug("Read ERA5 product:OK")
    # Process "total_evaporation" variable
    dataset = ERA5Dataset.ERA5LAND
    era5_xrds[ERA5Var.TOTAL_EVAPORATION.key] = era5_xrds[
        ERA5Var.TOTAL_EVAPORATION.key
    ].min(dim="time", skipna=False)  # Min selected due to daily accumulation
    ##########
    # Add data
    ##########
    # Copy data
    updated_data = data.copy()
    updated_data.attrs = data.attrs.copy()
    #############
    # Total Evaporation
    #############
    # Add evaporation
    dst = next(iter(data.data_vars.values()))
    updated_data[f"e_{dataset.key}"] = interpolate_evaporation(
        data=era5_xrds[ERA5Var.TOTAL_EVAPORATION.key],
        dem=dst.rio.write_crs(crs),  # transfer crs attribute
    )
    updated_data[f"e_{dataset.key}"] = (
        -updated_data[f"e_{dataset.key}"] * 1000
    )  # Convert m to mm, invert sign for upward flux
    updated_data[f"e_{dataset.key}"].attrs["standard_name"] = "e"
    updated_data[f"e_{dataset.key}"].attrs["long_name"] = "Total evaporation"
    updated_data[f"e_{dataset.key}"].attrs["name"] = "e"
    updated_data[f"e_{dataset.key}"].attrs["unit"] = "mm"
    updated_data[f"e_{dataset.key}"].attrs["description"] = "Total evaporation"
    logger.debug("Add total_evaporation :OK")
    # Clean
    if temp_dir is not None:
        temp_dir.cleanup()  # Manually delete the directory
    return updated_data


def download_date_by_date(
    date1: dt.datetime,
    date2: dt.datetime,
    variables: list | None = None,
    output: str | None = None,
) -> pd.DatetimeIndex:
    """
    Description
    -----------
    Download ERA5-Land product day by day
    from a start and end date

    Parameters
    ----------
    date1: dt.datetime
        Start date
    date2: dt.datetime
        End date
    variables: list[str]
        List of ERA5-Land product to download
        (default: all ERA5-Land products)
    output: str
        Directory path to store data

    Return
    ------
    time: pd.DatetimeIndex
        list of dates beetween start and end date
    """
    # Check
    if variables is None:
        variables = ERA5Dataset.ERA5LAND.variables
    else:
        for v in variables:
            if v not in ERA5Dataset.ERA5LAND.variables:
                raise ERA5Exception(f"Error: {v} is not in ERA5LAND")
    # Run
    time = xr.date_range(date1, freq="1D", end=date2)
    for t in time:
        download(
            t.to_pydatetime(), ERA5Dataset.ERA5LAND, variables, path=output
        )
    return time


def create_daily_et_dataset(
    date: dt.datetime, grid: xr.Dataset, path: str | None = None
) -> xr.Dataset:
    """
    Description
    -----------
    - Read the "Total evaporation" product
    (contained in the ERA5-Land .zip repertory) as dataset,
    projecting data on a given ROI
    - Add a "flags" variable (1 for nan value, 0 otherwise)

    Parameters
    ----------
    date: datetime.datetime
        Date
    grid: xr.Dataset
        grid centered on the ROI
    path: str
        Directory path where ERA5-Land data have been downloaded

    Return
    ------
    dst_rad: xr.Dataset
        Daily evapotranspiration dataset with flags
    """
    # Read as dataset
    if path is None:
        path = os.getcwd()
    grid.attrs["vis_date"] = date.date()
    dst = add_et(grid, path=path)
    # Add "flags" variable
    dst["flags"] = dst[f"e_{ERA5Dataset.ERA5LAND.key}"].isnull().astype(int)
    dst = dst.rename_vars({f"e_{ERA5Dataset.ERA5LAND.key}": "et"})
    return dst[["et", "flags"]]

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
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path
from time import sleep

import cdsapi
import pandas as pd
import rasterio as rio
import xarray as xr
from pyproj import CRS

from etdataset.dewpoint_temp import add_dewpoint
from etdataset.era5_type import (
    ERA5Dataset,
    ERA5Exception,
    ERA5pressureVar,
    ERA5Var,
)
from etdataset.logging import LoggerManager
from etdataset.temperature import (
    METHOD_REQUIREMENTS,
    NAME_MAP,
    RescalTempMethod,
    TempVariable,
    crop_ds,
    method_const_lr,
    method_interp_lr_hybrid,
    method_interp_profile_surf,
    method_lr_profile_alt_dep,
    method_lr_profile_fixed_levels,
    method_monthly_lr,
    method_vertical_interp,
    normalize_variables,
    rename_var_ds,
)
from etdataset.utils import (
    normalize_longitude_latitude,
)

logger = LoggerManager.get_logger(__name__)

# Gravitational constant
G_CST = 9.80665


def read(product: str) -> xr.Dataset:
    """
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


def _download(dataset: str, request: dict, target: str) -> None:
    """
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
    date: dt.datetime,
    dataset: ERA5Dataset,
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
    variables: list[str] | None
        List of product to download
    path: str | None
        Directory path to store data
    """
    if path is None:
        path = os.getcwd()
    # Create a directory MSG products
    era5_path = path
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
    if dataset == ERA5Dataset.ERA5PRESSURE:
        request = {
            "product_type": "reanalysis",
            "variable": variables,
            "year": date.year,
            "month": date.month,
            "day": date.day,
            "pressure_level": [
                "700",
                "725",
                "750",
                "775",
                "800",
                "825",
                "850",
                "875",
                "900",
                "925",
                "950",
                "975",
                "1000",
            ],
            "time": [
                "00:00",
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
    else:
        request = {
            "product_type": "reanalysis",
            "variable": variables,
            "year": date.year,
            "month": date.month,
            "day": date.day,
            "time": [
                "00:00",
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


def interpolate_time(
    data: xr.Dataset, date: dt.datetime, variables: list[ERA5Var] | None = None
) -> xr.Dataset:
    """
    Compute a linear time interpolation on data
    at a specific date for a list of variables.

    Parameters
    ----------
    data: xr.Dataset
        Dataset all the variables and the dates
    date: dt.datetime
        Date at which interpolation is computed
    variables: list[ERA5Var] | None
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


def interpolate_ozone(
    data: xr.DataArray,
    dem: xr.DataArray,
) -> xr.DataArray:
    """
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


def interpolate_tcvw(
    data: xr.DataArray,
    dem: xr.DataArray,
) -> xr.DataArray:
    """
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


def rescale_radiation(
    dem: xr.DataArray,
    era5_data: xr.DataArray | None,
    key: str,
    description: str,
) -> xr.DataArray | None:
    """
    Rescale radiation using constant lapse rate

    Parameters
    ----------
    dem: xr.DataArray
        DEM or grid used to rescale data
    era5_data: xr.DataArray | None
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
    data.attrs.clear()
    data.attrs["standard_name"] = key
    data.attrs["long_name"] = description
    data.attrs["name"] = key
    data.attrs["unit"] = "W.m-2"
    data.attrs["description"] = description
    return data


def prepare_data(
    data: xr.Dataset,
    dataset: ERA5Dataset = ERA5Dataset.ERA5,
    variables: list[ERA5Var] | None = None,
    path: str | None = None,
    temp_method: RescalTempMethod = RescalTempMethod.CONST_LR,
) -> tuple[xr.Dataset, xr.Dataset | None, xr.DataArray, xr.DataArray | None]:
    """
    Prepare ERA5 datas to be added to the dataset

    Parameters
    ----------
    data: xr.Dataset
        Data
    dataset: ERA5Dataset
        ERA5 Dataset used for download
    variables: list[ERA5Var] | None
        List of variables to prepare
    path: str | None
        Directory where ERA5 data have been downloaded data
    temp_method : RescalTempMethod
        Temperature rescaling method used

    Returns
    -------
    era5_surface : xr.Dataset
        ERA5 data prepared
    era5_pressure : xr.Dataset | None
        ERA5 pressure prepared
    era5_dem :  xr.DataArray
        ERA5 DEM prepared
    era5_dem_pressure : xr.DataArray | None
        ERA5 pressure DEM prepared
    """
    ###################### CHECK INPUTS #######################################
    if data.attrs.get("vis_date", None) is None:
        raise ValueError("Vis date attribute is missing in dataset")

    if data.attrs.get("vis_time", None) is None:
        raise ValueError("Vis time attribute is missing in dataset")

    if len(data.data_vars) == 0:
        raise ValueError("Dataset is empty")
    # Extract data
    date = dt.datetime.combine(data.attrs["vis_date"], data.attrs["vis_time"])

    ######################## SET VARIABLES ###################################
    # Configure ERA5 variables if necessary
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

    # Variables for temperature rescaling
    temp_variables = [
        temp_var
        for era5_var, temp_var in [
            (ERA5Var.TEMPERATURE, TempVariable.TA),
            (ERA5Var.DEWPOINT_TEMPERATURE, TempVariable.TD),
        ]
        if era5_var in variables
    ]
    temp_variables_set = normalize_variables(temp_variables)

    ################# TEMPERATURE METHOD REQUIREMENTS #########################
    req = METHOD_REQUIREMENTS.get(temp_method)
    if req is None:
        raise ValueError(f"Unsupported method: {temp_method}")

    ######################## EXTRACT DEM ######################################
    # Check CRS
    if data.rio.crs is None:
        crs = data.attrs.get("crs")
        if crs is None:
            raise ValueError("crs attribute is missing in dataset")
        data = data.rio.write_crs(crs)

    crs = data.rio.crs

    # DEM
    height = data.get("height", None)

    if height is None:
        raise ValueError("DEM (height) is missing in dataset")

    dem = height.rio.write_crs(crs)

    ###################### GET ERA5 DATA ####################################
    temp_dir = None
    # Download data if necessary
    if path is None:
        logger.debug("Download only required ERA5 data")
        # Create a temp directory
        temp_dir = tempfile.TemporaryDirectory()
        path = temp_dir.name
        logger.debug(f"Temp dir: {path}")
        # Download files
        download(date=date, dataset=dataset, path=path)

        ## Get ERA5 pressure level
        if req["pressure"]:
            logger.debug("Downloading ERA5 pressure dataset")
            download(date=date, dataset=ERA5Dataset.ERA5PRESSURE, path=path)

    ###################### READ ERA5 DATA ####################################
    # ERA5/ERA5Land data path
    product_path = os.path.join(
        path,
        f"download_{dataset.key}_{date.strftime('%Y-%m-%d')}.zip",
    )
    logger.debug(f"Product path: {product_path}")
    if not os.path.isfile(product_path):
        raise OSError(f"ERA5 data not found: {product_path}")

    # ERA5 pressure data path
    if req["pressure"]:
        pressure_product_path = os.path.join(
            path,
            f"download_era5_pressure_{date.strftime('%Y-%m-%d')}.zip",
        )
        logger.debug(f"Product path: {pressure_product_path}")
        if not os.path.isfile(pressure_product_path):
            raise OSError(
                f"ERA5 pressure data not found: {pressure_product_path}"
            )
    # Warning for temperature rescaling input datas
    if not req["surface"]:
        logger.warning(
            f"ERA5 data not used for rescaling temperature method {temp_method}"
        )
    if not req["pressure"]:
        logger.warning(
            f"ERA5 pressure not used for rescaling temperature method "
            f"{temp_method}"
        )

    # Read data:
    ## Read ERA5/ERA5Land data
    era5_xrds = read(product=product_path)
    logger.debug("Read ERA5 product:OK")
    era5_xrds = era5_xrds[[var.key for var in variables]]
    logger.debug(f"Variables : {list(era5_xrds.data_vars)}")

    ## Read ERA5 pressure data
    era5_pressure_xrds = None
    if req["pressure"]:
        era5_pressure_xrds = read(product=pressure_product_path)
    logger.debug("Read ERA5 pressure product:OK")

    ###################### PREPARE ERA5 DATA ###################################
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

    ## Prepare ERA5/ERA5Land data
    era5_surface = None
    era5_xrds = interpolate_time(data=era5_xrds, date=date)
    era5_xrds_ = normalize_longitude_latitude(era5_xrds)
    era5_surface = rename_var_ds(era5_xrds_, NAME_MAP)

    if era5_surface is not None:
        era5_surface = era5_surface.rio.write_crs("EPSG:4326")
        era5_surface = crop_ds(era5_surface, dem)

    ## Prepare ERA5 pressure data
    era5_pressure = None
    if req["pressure"] and era5_pressure_xrds is not None:
        era5_pressure_xrds = interpolate_time(
            data=era5_pressure_xrds, date=date
        )
        era5_pressure_xrds_ = normalize_longitude_latitude(era5_pressure_xrds)

        if TempVariable.TD in temp_variables_set:
            era5_pressure_xrds_ = add_dewpoint(era5_pressure_xrds_)

        era5_pressure = rename_var_ds(era5_pressure_xrds_, NAME_MAP)

        if era5_pressure is not None:
            era5_pressure = era5_pressure.rio.write_crs("EPSG:4326")
            era5_pressure = crop_ds(era5_pressure, dem)

    ###################### BUILD ERA5 DEM ######################################
    ## HEIGHT ERA5 SURFACE
    if dataset == ERA5Dataset.ERA5:
        era5_dem_init = get_era5_dem()
    elif dataset == ERA5Dataset.ERA5LAND:
        era5_dem_init = get_era5land_dem()
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    era5_dem = xr.DataArray(
        era5_dem_init.data,
        dims=era5_xrds.dims,
        coords=era5_xrds.coords,
    ).rio.write_crs(CRS(4326))

    # Prepare ERA5 DEM
    # era5_dem_ = era5_dem.to_dataset(name="dem")
    # era5_dem_ = normalize_longitude_latitude(era5_dem_)
    # era5_dem = era5_dem_["dem"]
    # era5_dem = era5_dem.rio.write_crs("EPSG:4326")
    # era5_dem = crop_ds(era5_dem, dem)
    era5_dem = era5_dem.to_dataset(name="dem")
    era5_dem = normalize_longitude_latitude(era5_dem)
    era5_dem = era5_dem.rio.write_crs("EPSG:4326")

    era5_dem = crop_ds(era5_dem, dem)
    era5_dem = era5_dem["dem"]

    ## HEIGHT ERA5 PRESSURE
    era5_dem_pressure = None
    if req["pressure"] and era5_pressure is not None:
        era5_dem_pressure = (
            era5_pressure[ERA5pressureVar.GEOPOTENTIAL.key] / G_CST
        )
        era5_dem_pressure = era5_dem_pressure.rio.write_crs("EPSG:4326")
        if era5_dem_pressure is not None:
            era5_dem_pressure_ = era5_dem_pressure.to_dataset(name="dem")
            era5_dem_pressure_ = crop_ds(era5_dem_pressure_, dem)
            era5_dem_pressure = era5_dem_pressure_["dem"]

    return era5_surface, era5_pressure, era5_dem, era5_dem_pressure


def add_temp(
    data: xr.Dataset,
    path: str | None = None,
    dataset: ERA5Dataset = ERA5Dataset.ERA5,
    variables: TempVariable | Iterable[TempVariable] | None = None,
    method: RescalTempMethod = RescalTempMethod.CONST_LR,
    interp_type: rio.enums.Resampling = rio.enums.Resampling.cubic_spline,
):
    """
    Add temperature variables to a dataset using ERA5/ERA5-Land data

    This function retrieves the required ERA5 datasets (surface and/or pressure)
    depending on the selected rescaling method, processes them, and applies
    a temperature rescaling algorithm

    Parameters
    ----------
    data: xr.Dataset
        Input dataset containing at least:
        - 'height' variable (DEM)
        - 'vis_date' and 'vis_time' attributes
        - CRS information
    path: str | None
        Path to a directory containing ERA5 data files
    dataset: ERA5Dataset
        Surface dataset to use (ERA5 or ERA5-Land)
    variables: TempVariable | Iterable[TempVariable] | None,
        Temperature variables to compute (e.g., TA, TD)
    method: RescalTempMethod
        Temperature rescaling method to apply
    interp_type: rio.enums.Resampling
        Method used for resampling
        By default, "cubic_spline"

    Returns
    -------
    xr.Dataset
        Dataset with added/rescaled temperature variables.
    """

    logger.info(f"ADD TEMP | method={method}")

    ###################### METHOD REQUIREMENTS ################################
    req = METHOD_REQUIREMENTS.get(method)
    if req is None:
        raise ValueError(f"Unsupported method: {method}")

    ###################### CHECK INPUTS #######################################
    if data.attrs.get("vis_date", None) is None:
        raise ValueError("Vis date attribute is missing in dataset")

    if data.attrs.get("vis_time", None) is None:
        raise ValueError("Vis time attribute is missing in dataset")

    if len(data.data_vars) == 0:
        raise ValueError("Dataset is empty")

    date = dt.datetime.combine(data.attrs["vis_date"], data.attrs["vis_time"])

    ###################### GET ERA5 DATA ####################################
    # Download data
    if path is None:
        logger.debug("Download only required ERA5 data")
        temp_dir = tempfile.TemporaryDirectory()
        path = temp_dir.name
        logger.debug(f"Temp dir: {path}")

        ## Get ERA5 single level
        if req["surface"]:
            logger.debug(f"Downloading surface dataset: {dataset.key}")
            download(date=date, dataset=dataset, path=path)

        ## Get ERA5 pressure level
        if req["pressure"]:
            logger.debug("Downloading ERA5 pressure dataset")
            download(date=date, dataset=ERA5Dataset.ERA5PRESSURE, path=path)

    # Product path data

    ## ERA5 single level path data
    surface_product_path = os.path.join(
        path,
        f"download_{dataset.key}_{date.strftime('%Y-%m-%d')}.zip",
    )

    ## ERA5 pressure levels path data
    pressure_product_path = os.path.join(
        path,
        f"download_era5_pressure_{date.strftime('%Y-%m-%d')}.zip",
    )

    logger.debug(f"Surface path: {surface_product_path}")
    logger.debug(f"Pressure path: {pressure_product_path}")

    # Check file
    use_surface = req["surface"] and os.path.isfile(surface_product_path)
    use_pressure = req["pressure"] and os.path.isfile(pressure_product_path)

    if req["surface"] and not use_surface:
        raise ValueError(
            f"Missing surface file ({dataset.key}) for method {method}"
        )

    if not use_surface:
        logger.warning(f"ERA5 surface not used for method {method}")

    if req["pressure"] and not use_pressure:
        raise ValueError(f"Missing ERA5 pressure file for method {method}")

    if not use_pressure:
        logger.warning(f"ERA5 pressure not used for method {method}")

    ######################## EXTRACT DEM ######################################
    # DEM
    if data.rio.crs is None:
        if data.attrs.get("crs") is not None:
            crs = data.attrs["crs"]
            data = data.rio.write_crs(crs)
        raise ValueError("crs attribute is missing in dataset")

    crs = data.rio.crs

    height = data.get("height", None)

    if height is None:
        raise ValueError("DEM (height) is missing in dataset")

    dem = height.rio.write_crs(crs)

    ###################### VARIABLES ##########################################
    variables_set = normalize_variables(variables)

    ###################### READ ERA5 FILES ####################################
    era5_surface = None
    era5_pressure = None

    # SURFACE DATA
    if use_surface and surface_product_path is not None:
        era5_surface_data = read(product=surface_product_path)

        if era5_surface_data is not None:
            era5_surface_data = interpolate_time(
                data=era5_surface_data, date=date
            )
            era5_surface = normalize_longitude_latitude(era5_surface_data)
            era5_surface = rename_var_ds(era5_surface, NAME_MAP)

            if era5_surface is not None:
                era5_surface = era5_surface.rio.write_crs("EPSG:4326")
                era5_surface = crop_ds(era5_surface, dem)
        else:
            logger.warning("ERA5 surface read failed")

    # PRESSURE DATA
    if use_pressure and pressure_product_path is not None:
        era5_pressure_data = read(product=pressure_product_path)

        if era5_pressure_data is not None:
            era5_pressure_data = interpolate_time(
                data=era5_pressure_data, date=date
            )
            era5_pressure = normalize_longitude_latitude(era5_pressure_data)

            if TempVariable.TD in variables_set:
                era5_pressure = add_dewpoint(era5_pressure)

            era5_pressure = rename_var_ds(era5_pressure, NAME_MAP)

            if era5_pressure is not None:
                era5_pressure = era5_pressure.rio.write_crs("EPSG:4326")
                era5_pressure = crop_ds(era5_pressure, dem)
        else:
            logger.warning("ERA5 pressure read failed")

    ###################### BUILD ERA5 DEM ######################################
    ## HEIGHT ERA5 SURFACE
    era5_dem_surface = None
    if era5_surface is not None:
        if dataset == ERA5Dataset.ERA5:
            era5_dem_surface_init = get_era5_dem()
        elif dataset == ERA5Dataset.ERA5LAND:
            era5_dem_surface_init = get_era5land_dem()
        era5_dem_surface = xr.DataArray(
            era5_dem_surface_init.data,
            dims=("latitude", "longitude"),
            coords={
                "latitude": era5_surface_data.latitude,
                "longitude": era5_surface_data.longitude,
            },
        ).rio.write_crs(CRS(4326))
        era5_dem_surface_ = era5_dem_surface.to_dataset(name="dem")
        era5_dem_surface_ = normalize_longitude_latitude(era5_dem_surface_)
        era5_dem_surface = era5_dem_surface_["dem"]
        era5_dem_surface = era5_dem_surface.rio.write_crs("EPSG:4326")
        era5_dem_surface = crop_ds(era5_dem_surface, dem)

    ## HEIGHT ERA5 PRESSURE
    era5_dem_pressure = None
    if era5_pressure is not None:
        era5_dem_pressure = (
            era5_pressure[ERA5pressureVar.GEOPOTENTIAL.key] / G_CST
        )
        era5_dem_pressure = era5_dem_pressure.rio.write_crs("EPSG:4326")
        if era5_dem_pressure is not None:
            era5_dem_pressure_ = era5_dem_pressure.to_dataset(name="dem")
            era5_dem_pressure_ = crop_ds(era5_dem_pressure_, dem)
            era5_dem_pressure = era5_dem_pressure_["dem"]

    ##################### PROCESSING METHODS ###################################
    if method == RescalTempMethod.CONST_LR:
        if era5_surface is None or era5_dem_surface is None:
            raise ValueError(
                "Missing ERA5 surface or DEM surface for CONST_LR method"
            )

        rescaled_data = method_const_lr(
            data=data,
            variables_set=variables_set,
            dem=dem,
            era5_surface=era5_surface,
            era5_dem_surface=era5_dem_surface,
            interp_type=interp_type,
        )

    if method == RescalTempMethod.MONTHLY_LR:
        if era5_surface is None or era5_dem_surface is None:
            raise ValueError(
                "Missing ERA5 surface or DEM surface for MONTHLY_LR method"
            )
        rescaled_data = method_monthly_lr(
            data=data,
            date=date,
            variables_set=variables_set,
            dem=dem,
            era5_surface=era5_surface,
            era5_dem_surface=era5_dem_surface,
            interp_type=interp_type,
        )

    if method == RescalTempMethod.LR_PROFILE_FIXED_LEVELS:
        if (
            era5_surface is None
            or era5_dem_surface is None
            or era5_pressure is None
            or era5_dem_pressure is None
        ):
            raise ValueError(
                "Missing ERA5 datas for LR_PROFILE_FIXED_LEVELS method"
            )
        rescaled_data = method_lr_profile_fixed_levels(
            data=data,
            variables_set=variables_set,
            dem=dem,
            era5_surface=era5_surface,
            era5_dem_surface=era5_dem_surface,
            era5_pressure=era5_pressure,
            era5_dem_pressure=era5_dem_pressure,
            interp_type=interp_type,
        )
    if method == RescalTempMethod.LR_PROFILE_ALT_DEP:
        if (
            era5_surface is None
            or era5_dem_surface is None
            or era5_pressure is None
            or era5_dem_pressure is None
        ):
            raise ValueError("Missing ERA5 datas for LR_PROFILE_ALT_DEP method")
        rescaled_data = method_lr_profile_alt_dep(
            data=data,
            variables_set=variables_set,
            dem=dem,
            era5_surface=era5_surface,
            era5_dem_surface=era5_dem_surface,
            era5_pressure=era5_pressure,
            era5_dem_pressure=era5_dem_pressure,
            interp_type=interp_type,
        )

    if method == RescalTempMethod.VERTICAL_INTERP:
        if era5_pressure is None or era5_dem_pressure is None:
            raise ValueError(
                "Missing ERA5 pressure or DEM pressure for VERTICAL_INTERP "
                r"\method"
            )
        rescaled_data = method_vertical_interp(
            data=data,
            variables_set=variables_set,
            dem=dem,
            era5_pressure=era5_pressure,
            era5_dem_pressure=era5_dem_pressure,
            interp_type=interp_type,
            dz=100,
        )

    if method == RescalTempMethod.INTERP_PROFILE_SURF:
        if (
            era5_surface is None
            or era5_dem_surface is None
            or era5_pressure is None
            or era5_dem_pressure is None
        ):
            raise ValueError("Missing ERA5 datas for INTERP_PROFILE_SURF")
        rescaled_data = method_interp_profile_surf(
            data=data,
            variables_set=variables_set,
            dem=dem,
            era5_surface=era5_surface,
            era5_dem_surface=era5_dem_surface,
            era5_pressure=era5_pressure,
            era5_dem_pressure=era5_dem_pressure,
            interp_type=interp_type,
            dz=100,
        )

    if method == RescalTempMethod.INTERP_LR_HYBRID:
        if (
            era5_surface is None
            or era5_dem_surface is None
            or era5_pressure is None
            or era5_dem_pressure is None
        ):
            raise ValueError("Missing ERA5 datas for INTERP_LR_HYBRID method")
        rescaled_data = method_interp_lr_hybrid(
            data=data,
            variables_set=variables_set,
            dem=dem,
            era5_surface=era5_surface,
            era5_dem_surface=era5_dem_surface,
            era5_pressure=era5_pressure,
            era5_dem_pressure=era5_dem_pressure,
            interp_type=interp_type,
            dz=100,
        )
    return rescaled_data


def add(
    data: xr.Dataset,
    dataset: ERA5Dataset = ERA5Dataset.ERA5,
    variables: list[ERA5Var] | None = None,
    path: str | None = None,
    temp_method: RescalTempMethod = RescalTempMethod.CONST_LR,
    interp_type: rio.enums.Resampling = rio.enums.Resampling.cubic_spline,
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
    variables: list[ERA5Var] | None
        List of variables to add
    path: str | None
        Directory where ERA5 data have been downloaded data
    temp_method : RescalTempMethod
        Temperature rescaling method to apply
    interp_type: rio.enums.Resampling
        Method used for resampling

    Returns
    -------
    updated_data: xr.Dataset
        Updated data
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
        f"download_{dataset.key}_{date.strftime('%Y-%m-%d')}.zip",
    )
    logger.debug(f"Product path: {product_path}")
    if not os.path.isfile(product_path):
        raise OSError(f"ERA5 data not found: {product_path}")
    # Read data
    era5_xrds = read(product=product_path)
    logger.debug("Read ERA5 product:OK")
    era5_xrds = era5_xrds[[var.key for var in variables]]
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
    list_var = [
        temp_var
        for era5_var, temp_var in [
            (ERA5Var.TEMPERATURE, TempVariable.TA),
            (ERA5Var.DEWPOINT_TEMPERATURE, TempVariable.TD),
        ]
        if era5_var in variables
    ]
    if list_var:
        updated_data = add_temp(
            data=data,
            dataset=ERA5Dataset.ERA5,
            variables=list_var,
            method=temp_method,
            interp_type=interp_type,
            path=path,
        )
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
        updated_data[f"tco3_{dataset.key}"].attrs.clear()
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


def download_date_by_date(
    date1: dt.datetime,
    date2: dt.datetime,
    dataset: ERA5Dataset,
    variables: list[str] | None = None,
    output: str | None = None,
) -> pd.DatetimeIndex:
    """
    Download ERA5 products day by day for a given dataset
    between two dates.

    Parameters
    ----------
    date1 : datetime.datetime
        Start date
    date2 : datetime.datetime
        End date
    dataset : ERA5Dataset
        ERA5 dataset class to use (e.g. ERA5Dataset.ERA5,
        ERA5Dataset.ERA5LAND, ERA5Dataset.ERA5PRESSURE).
    variables : list[str] | None
        List of variables to download. If None, all variables
        available in the dataset are downloaded.
    output : str | None
        Directory path to store data
    Returns
    -------
    time : pd.DatetimeIndex
        List of dates between start and end date
    """

    if variables is None:
        # Use all available variables from the dataset
        variables = dataset.variables
    else:
        for v in variables:
            if v not in dataset.variables:
                raise ERA5Exception(f"Error: {v} is not available in {dataset}")

    time = xr.date_range(date1, freq="1D", end=date2)
    for t in time:
        download(
            t.to_pydatetime(),
            dataset,
            variables,
            path=output,
        )

    return time

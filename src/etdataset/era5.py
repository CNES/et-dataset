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

import pandas as pd
import rasterio as rio
import xarray as xr
from pyproj import CRS

from etdataset.era5_type import ERA5Dataset, ERA5Exception, ERA5Var
from etdataset.era5_utils import download, get_era5_dem, get_era5land_dem, read
from etdataset.logging import LoggerManager
from etdataset.temperature import (
    RescalTempMethod,
    TempVariable,
    add_temp,
)

logger = LoggerManager.get_logger(__name__)

# Gravitational constant
G_CST = 9.80665


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

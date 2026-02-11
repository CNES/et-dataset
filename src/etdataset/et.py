#
# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales /
#            Université Paul Sabatier (UT3)
#
"""
Module for preparing ET data time series
"""

import datetime as dt
import os
import tempfile

import rasterio as rio
import xarray as xr

from etdataset.era5 import ERA5Dataset, ERA5Var, download, read
from etdataset.logging import LoggerManager

logger = LoggerManager.get_logger(__name__)

# Gravitational constant
G_CST = 9.80665


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


def interpolate_explanatory(
    data: xr.DataArray,
    dem: xr.DataArray,
) -> xr.DataArray:
    """
    Description
    -----------
    This method interpolates explanatory variable data on a new grid.

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
    date = dt.datetime.combine(data.attrs["vis_date"], dt.time(12, 0))
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
    era5_xrds = era5_xrds[[ERA5Var.TOTAL_EVAPORATION.key]]
    logger.debug("Read ERA5 product:OK")
    # Process "total_evaporation" variable
    era5_xrds[ERA5Var.TOTAL_EVAPORATION.key] = era5_xrds[
        ERA5Var.TOTAL_EVAPORATION.key
    ].min(dim="time", skipna=False)  # Min selected due to daily accumulation
    logger.debug("Process ERA5 product:OK")
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
    updated_data[f"et_{dataset.key}"] = interpolate_evaporation(
        data=era5_xrds[ERA5Var.TOTAL_EVAPORATION.key],
        dem=dst.rio.write_crs(crs),  # transfer crs attribute
    )
    updated_data[f"et_{dataset.key}"] = (
        -updated_data[f"et_{dataset.key}"] * 1000
    )  # Convert m to mm, invert sign for upward flux
    updated_data[f"et_{dataset.key}"].attrs["standard_name"] = "et"
    updated_data[f"et_{dataset.key}"].attrs["long_name"] = "Evapotranspiration"
    updated_data[f"et_{dataset.key}"].attrs["name"] = "et"
    updated_data[f"et_{dataset.key}"].attrs["unit"] = "mm"
    updated_data[f"et_{dataset.key}"].attrs["description"] = (
        "Evapotranspiration"
    )
    logger.debug("Add total_evaporation :OK")
    # Clean
    if temp_dir is not None:
        temp_dir.cleanup()  # Manually delete the directory
    return updated_data


def add_explanatory(
    data: xr.Dataset,
    path: str | None = None,
) -> xr.Dataset:
    """
    Description
    -----------
    Add the ERA5-Land regression model explanatory variables to the dataset.

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
    date = dt.datetime.combine(data.attrs["vis_date"], dt.time(12, 0))
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
    era5_xrds = era5_xrds[
        [
            ERA5Var.TOTAL_PRECIPITATION.key,
            ERA5Var.SURFACE_RUNOFF.key,
            ERA5Var.SKIN_RESERVOIR_CONTENT.key,
            ERA5Var.SOIL_WATER_LEVEL1.key,
        ]
    ]
    logger.debug("Read ERA5 product:OK")
    # Process "total_precipitation" variable
    era5_xrds[ERA5Var.TOTAL_PRECIPITATION.key] = era5_xrds[
        ERA5Var.TOTAL_PRECIPITATION.key
    ].max(dim="time", skipna=False)  # Max selected due to daily accumulation
    # Process "surface_runoff" variable
    era5_xrds[ERA5Var.SURFACE_RUNOFF.key] = era5_xrds[
        ERA5Var.SURFACE_RUNOFF.key
    ].max(dim="time", skipna=False)  # Max selected due to daily accumulation
    # Process "skin reservoir content" variable
    era5_xrds[ERA5Var.SKIN_RESERVOIR_CONTENT.key] = era5_xrds[
        ERA5Var.SKIN_RESERVOIR_CONTENT.key
    ].max(dim="time", skipna=False)
    # Process "skin reservoir content" variable
    era5_xrds[ERA5Var.SOIL_WATER_LEVEL1.key] = era5_xrds[
        ERA5Var.SOIL_WATER_LEVEL1.key
    ].mean(dim="time", skipna=False)
    ##########
    # Add data
    ##########
    # Copy data
    updated_data = data.copy()
    updated_data.attrs = data.attrs.copy()
    #############
    # Total Precipitation
    #############
    # Add precipitation
    dst = next(iter(data.data_vars.values()))
    updated_data[f"tp_{dataset.key}"] = interpolate_explanatory(
        data=era5_xrds[ERA5Var.TOTAL_PRECIPITATION.key],
        dem=dst.rio.write_crs(crs),  # transfer crs attribute
    )
    updated_data[f"tp_{dataset.key}"] = (
        updated_data[f"tp_{dataset.key}"] * 1000
    )  # Convert m to mm
    updated_data[f"tp_{dataset.key}"].attrs["standard_name"] = "tp"
    updated_data[f"tp_{dataset.key}"].attrs["long_name"] = "Total precipitation"
    updated_data[f"tp_{dataset.key}"].attrs["name"] = "tp"
    updated_data[f"tp_{dataset.key}"].attrs["unit"] = "mm"
    updated_data[f"tp_{dataset.key}"].attrs["description"] = (
        "Total precipitation"
    )
    logger.debug("Add total_precipitation :OK")
    #############
    # Surface Runoff
    #############
    # Add runoff
    dst = next(iter(data.data_vars.values()))
    updated_data[f"sro_{dataset.key}"] = interpolate_explanatory(
        data=era5_xrds[ERA5Var.SURFACE_RUNOFF.key],
        dem=dst.rio.write_crs(crs),  # transfer crs attribute
    )
    updated_data[f"sro_{dataset.key}"] = (
        updated_data[f"sro_{dataset.key}"] * 1000
    )  # Convert m to mm
    updated_data[f"sro_{dataset.key}"].attrs["standard_name"] = "sro"
    updated_data[f"sro_{dataset.key}"].attrs["long_name"] = "Surface runoff"
    updated_data[f"sro_{dataset.key}"].attrs["name"] = "sro"
    updated_data[f"sro_{dataset.key}"].attrs["unit"] = "mm"
    updated_data[f"sro_{dataset.key}"].attrs["description"] = "Surface Runoff"
    logger.debug("Add surface_runoff :OK")
    #############
    # Skin Reservoir Content
    #############
    # Add skin reservoir content
    dst = next(iter(data.data_vars.values()))
    updated_data[f"src_{dataset.key}"] = interpolate_explanatory(
        data=era5_xrds[ERA5Var.SKIN_RESERVOIR_CONTENT.key],
        dem=dst.rio.write_crs(crs),  # transfer crs attribute
    )
    updated_data[f"src_{dataset.key}"] = (
        updated_data[f"src_{dataset.key}"] * 1000
    )  # Convert m to mm
    updated_data[f"src_{dataset.key}"].attrs["standard_name"] = "src"
    updated_data[f"src_{dataset.key}"].attrs["long_name"] = (
        "Skin reservoir content"
    )
    updated_data[f"src_{dataset.key}"].attrs["name"] = "src"
    updated_data[f"src_{dataset.key}"].attrs["unit"] = "mm"
    updated_data[f"src_{dataset.key}"].attrs["description"] = (
        "Skin reservoir content"
    )
    #############
    # Soil Water Level 1
    #############
    # Add soil water
    dst = next(iter(data.data_vars.values()))
    updated_data[f"swvl1_{dataset.key}"] = interpolate_explanatory(
        data=era5_xrds[ERA5Var.SOIL_WATER_LEVEL1.key],
        dem=dst.rio.write_crs(crs),  # transfer crs attribute
    )
    updated_data[f"swvl1_{dataset.key}"].attrs["standard_name"] = "swvl1"
    updated_data[f"swvl1_{dataset.key}"].attrs["long_name"] = (
        "Soil water level 1"
    )
    updated_data[f"swvl1_{dataset.key}"].attrs["name"] = "swvl1"
    updated_data[f"swvl1_{dataset.key}"].attrs["unit"] = "m-3 m3"
    updated_data[f"swvl1_{dataset.key}"].attrs["description"] = (
        "Soil water level 1"
    )
    logger.debug("Add volumetric_soil_water_layer_1 :OK")
    # Clean
    if temp_dir is not None:
        temp_dir.cleanup()  # Manually delete the directory
    return updated_data


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
    dst["flags"] = dst[f"et_{ERA5Dataset.ERA5LAND.key}"].isnull().astype(int)
    dst = dst.rename_vars({f"et_{ERA5Dataset.ERA5LAND.key}": "et"})
    return dst[["et", "flags"]]


def create_daily_explanatory_dataset(
    date: dt.datetime, grid: xr.Dataset, path: str | None = None
) -> xr.Dataset:
    """
    Description
    -----------
    Read the regression model explanatory products
    (contained in the ERA5-Land .zip repertory) as dataset,
    projecting data on a given ROI

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
    xr.Dataset
        Daily explanatory variable products
    """
    # Read as dataset
    if path is None:
        path = os.getcwd()
    grid.attrs["vis_date"] = date.date()
    dst = add_explanatory(grid, path=path)
    # Add "flags" variable
    dst = dst.rename_vars(
        {
            f"tp_{ERA5Dataset.ERA5LAND.key}": "tp",
            f"sro_{ERA5Dataset.ERA5LAND.key}": "sro",
            f"src_{ERA5Dataset.ERA5LAND.key}": "src",
            f"swvl1_{ERA5Dataset.ERA5LAND.key}": "sw",
        }
    )
    return dst[["tp", "sro", "src", "sw"]]

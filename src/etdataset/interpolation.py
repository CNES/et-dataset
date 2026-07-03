#
# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales /
#            Université Paul Sabatier (UT3)
#
"""
Module for reading ERA5 and ERA5-land
"""

from datetime import datetime

import numpy as np
import rasterio as rio
import rioxarray  # noqa # Use to activate rioxarray accessors
import xarray as xr
from pyproj import CRS

from etdataset.logging import LoggerManager

logger = LoggerManager.get_logger(__name__)


def interpolate_time(
    data: xr.Dataset, date: datetime, variables: list[str] | None = None
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
            if var not in list(data.data_vars):
                raise KeyError(f"Variable {var} not found")
            vars_str.append(var)
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


def create_grid_array(
    bounds: rio.coords.BoundingBox,
    crs: CRS,
    resolution: float,
) -> xr.DataArray:
    """
    Create a grid
    """
    # X coords
    width = int(np.ceil((bounds.right - bounds.left) / resolution))
    start = bounds.left + 0.5 * resolution
    # Calculate the stop value
    stop = start + resolution * (width - 1)
    xcoords: np.ndarray = np.linspace(
        start,
        stop,
        width,
    )
    # Y coords
    height = int(np.ceil((bounds.top - bounds.bottom) / resolution))
    start = bounds.bottom + 0.5 * resolution
    # Calculate the stop value
    stop = start + resolution * (height - 1)
    ycoords: np.ndarray = np.linspace(
        start,
        stop,
        height,
    )
    grid = xr.DataArray(
        data=np.ones((height, width)).astype(int),
        dims=["y", "x"],
        coords={"x": xcoords, "y": ycoords},
    )
    return grid.rio.write_crs(crs)


def create_grid_dataset(
    bounds: rio.coords.BoundingBox,
    crs: CRS,
    resolution: float,
) -> xr.Dataset:
    """
    Create a grid
    """
    # X coords
    width = int(np.ceil((bounds.right - bounds.left) / resolution))
    start = bounds.left + 0.5 * resolution
    # Calculate the stop value
    stop = start + resolution * (width - 1)
    xcoords: np.ndarray = np.linspace(
        start,
        stop,
        width,
    )
    # Y coords
    height = int(np.ceil((bounds.top - bounds.bottom) / resolution))
    start = bounds.bottom + 0.5 * resolution
    # Calculate the stop value
    stop = start + resolution * (height - 1)
    ycoords: np.ndarray = np.linspace(
        start,
        stop,
        height,
    )
    grid = xr.Dataset(
        data_vars={"grid": (("y", "x"), np.ones((height, width)).astype(int))},
        coords={"x": xcoords, "y": ycoords},
    )
    return grid.rio.write_crs(crs)


def interpolate_on_grid(
    data: xr.Dataset | xr.DataArray,
    grid: xr.DataArray,
    algorithm: rio.enums.Resampling = rio.enums.Resampling.bilinear,
) -> xr.Dataset | xr.DataArray:
    """
    Method for spatial interpolation on a grid

    :param data: Dataset or DataArray to to be spatially interpolated
    :param grid: Grid used for interplation
    :param algorithm: Algorithm used for resampling
    :return: xr.Dataset or xr.DataArray
    """
    return data.rio.reproject_match(
        grid,
        resampling=algorithm,
    )


def reproject_era5_grid(
    data: xr.DataArray, dst_data: xr.DataArray
) -> xr.DataArray:
    """
    Description
    -----------
    Reproject an ERA5 grid (in EPSG:4326) to match the grid/CRS of a
    destination DataArray (in EPSG:32631), using nearest-neighbor resampling.

    Parameters
    ----------
    data: Source ERA5 DataArray with 'latitude'/'longitude' dimensions,
        assumed to be in EPSG:4326.
    dst_data: Destination DataArray whose grid/CRS (EPSG:32631) the
        source data will be reprojected to match.

    Returns
    -------
    The source data reprojected onto the destination grid.
    """

    data = data.rename({"latitude": "y", "longitude": "x"})
    data = data.rio.write_crs("EPSG:4326")
    data = data.rio.write_nodata(np.nan)
    dst_data = dst_data.rio.write_crs("EPSG:32631")
    # Reproject the source data to match the destination grid
    res = data.rio.reproject_match(
        dst_data, resampling=rio.enums.Resampling.nearest
    )

    return res


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

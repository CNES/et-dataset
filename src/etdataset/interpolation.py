#
# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales /
#            Université Paul Sabatier (UT3)
#
"""
Module for reading ERA5 and ERA5-land
"""

import sys
from datetime import datetime

import numpy as np
import psutil  # type: ignore
import rasterio as rio
import xarray as xr
from pyproj import CRS
from rasterio.warp import reproject
from sensorsio.utils import bb_transform

from etdataset.logging import LoggerManager

logger = LoggerManager.get_logger(__name__)


def interpolate_time(
    data: xr.Dataset, variables: list[str], date: datetime
) -> xr.Dataset:
    """
    Interpolate data for a variable at the date

    :param data: Dataset all the variables and the dates
    :param vars: List of Variables to consider for the interpolation
    :param date: Date at which interpolation is computed
    :return: xr.Dataset
    """
    # Check variable
    for var in variables:
        if var not in list(data.data_vars):
            raise KeyError(f"Variable {var} not found")
    # Get all available dates for time interpolation
    dates_str: list[str] = sorted(
        [
            dt.strftime("%Y-%m-%d %H:%M:%S")
            for dt in data.coords["t"].data.astype("M8[ms]").astype("O")
            if date.date() == dt.date()
        ]
    )
    if len(dates_str) == 0:
        raise KeyError(f"No date can be used for interpolation at {date}")
    # Select data used for interpolation
    selected = data[variables].sel(t=dates_str)
    # Interpolate for the acquisition time
    date_str = date.strftime("%Y-%m-%d %H:%M:%S")
    if selected.sizes["t"] == 1:
        logger.warning("No interpolation, only one timestamp available")
        return selected.isel(t=0, drop=True)
    return selected.interp(t=date_str, method="linear")


def interpolate(
    data: xr.Dataset | xr.DataArray,
    crs: str | None = None,
    resolution: float | None = None,
    bounds: rio.coords.BoundingBox | None = None,
    algorithm: rio.enums.Resampling = rio.enums.Resampling.cubic,
    dtype: np.dtype = np.dtype("float32"),  # noqa
) -> xr.Dataset | xr.DataArray:
    """
    Method for spatial interpolation

    :param data: Dataset or DataArray to to be spatially interpolated
    :param crs: Requested CRS
    :param resolution: Requested resolution
    :param bounds: Requested bounding box
    :param algorithm: Algorithm used for resampling
    :param dtype: Type of data
    :return: xr.Dataset or xr.DataArray
    """
    # Init
    need_to_reproject = False
    assert (data.dims) != 2  # noqa
    try:
        src_crs = data.crs
        src_transform = data.transform
        src_bounds = data.bounds
        src_resolutionx = data.resolution["x"]
        src_resolutiony = data.resolution["y"]
    except KeyError:
        raise AttributeError("Data without projection metadata")
    nb_vars = len(data.data_vars) if isinstance(data, xr.Dataset) else 1

    # Projection
    dst_crs = crs if crs is not None else src_crs
    if src_crs == dst_crs:
        # Same CRS
        if resolution is None:
            dst_resolutionx = src_resolutionx
            dst_resolutiony = src_resolutiony

        if bounds is not None and src_bounds != bounds:
            # If we change bounds
            need_to_reproject = True
            dst_bounds = rio.coords.BoundingBox(*bounds)
        else:
            dst_bounds = rio.coords.BoundingBox(*src_bounds)

        # If we change resolution
        if resolution is not None and (
            src_resolutionx != resolution or src_resolutiony != resolution
        ):
            need_to_reproject = True
            dst_resolutionx = resolution
            dst_resolutiony = resolution

    else:
        # Different CRS
        need_to_reproject = True

        if bounds is not None and src_bounds != bounds:
            # If we change bounds
            dst_bounds = rio.coords.BoundingBox(*bounds)
        else:
            dst_bounds = bb_transform(src_crs, dst_crs, src_bounds)

        if resolution is None:
            dst_transform = rio.transform.from_bounds(
                *dst_bounds, data.sizes["x"], data.sizes["y"]
            )
            dst_resolutionx = abs(dst_transform[0])
            dst_resolutiony = abs(dst_transform[4])
        else:
            dst_resolutionx = resolution
            dst_resolutiony = resolution

    left = dst_bounds.left + dst_resolutionx / 2
    right = dst_bounds.right - dst_resolutionx / 2
    dst_sizex = int(np.rint((right - left) / dst_resolutionx)) + 1
    top = dst_bounds.top - dst_resolutiony / 2
    bottom = dst_bounds.bottom + dst_resolutiony / 2
    dst_sizey = int(np.rint((top - bottom) / dst_resolutiony)) + 1
    dst_transform = rio.transform.from_bounds(*dst_bounds, dst_sizex, dst_sizey)

    # Safeguard to compute if there is enough memory available
    available_memory = (
        psutil.virtual_memory().available / 1024 / 1024 / 1024
    )  # in Gb
    requested_memory = (
        (sys.getsizeof(dtype) * nb_vars * dst_sizex * dst_sizey)
        / 1024
        / 1024
        / 1024
    )
    if requested_memory > available_memory:
        raise MemoryError(
            "Not enough memory to process the "
            "dataset (requested memory = "
            f"{requested_memory} / available memory "
            f"= {available_memory}"
        )

    xcoords: np.ndarray = np.linspace(
        dst_bounds.left + 0.5 * dst_resolutionx,
        dst_bounds.left - 0.5 * dst_resolutionx + dst_sizex * dst_resolutionx,
        dst_sizex,
    )

    ycoords: np.ndarray = np.linspace(
        dst_bounds.top - 0.5 * dst_resolutiony,
        dst_bounds.top + 0.5 * dst_resolutiony - dst_sizey * dst_resolutiony,
        dst_sizey,
    )

    if isinstance(data, xr.DataArray):
        if need_to_reproject:
            src_data = data.to_numpy()
            dst_data = np.zeros((dst_sizey, dst_sizex), dtype)
            reproject(
                src_data,
                dst_data,
                src_transform=src_transform,
                src_crs=src_crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=algorithm,
            )
        else:
            dst_data = data.to_numpy()

        return xr.DataArray(
            dst_data,
            coords=[ycoords, xcoords],
            dims=["y", "x"],
            attrs={
                "crs": dst_crs,
                "bounds": dst_bounds,
                "transform": dst_transform,
                "resolution": {"x": dst_resolutionx, "y": dst_resolutiony},
            },
        )
    if isinstance(data, xr.Dataset):
        if need_to_reproject:
            values: dict[str, tuple[list[str], np.ndarray]] = {}
            for var in data.data_vars:
                src_data = data[var].to_numpy()
                dst_data = np.zeros((dst_sizey, dst_sizex), dtype)
                reproject(
                    src_data,
                    dst_data,
                    src_transform=src_transform,
                    src_crs=src_crs,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    resampling=algorithm,
                )
                values[str(var)] = (["y", "x"], dst_data)
        else:
            values = {
                str(var): (["y", "x"], data[var].to_numpy())
                for var in data.data_vars
            }

        return xr.Dataset(
            values,
            coords={"x": xcoords, "y": ycoords},
            attrs={
                "crs": dst_crs,
                "bounds": dst_bounds,
                "transform": dst_transform,
                "resolution": {"x": dst_resolutionx, "y": dst_resolutiony},
            },
        )
    raise TypeError(
        "Interpolation can only be performed on Dataset or DataArray"
    )


def create_grid(
    bounds: rio.coords.BoundingBox,
    crs: CRS,
    resolution: float,
    transform: rio.Affine,
    shape: tuple[int, int],
) -> xr.DataArray:
    """
    Create a grid
    """
    data = np.ones(shape)
    xcoords: np.ndarray = np.linspace(
        bounds.left + 0.5 * resolution,
        bounds.right - 0.5 * resolution,
        shape[1],
    )
    ycoords: np.ndarray = np.linspace(
        bounds.top - 0.5 * resolution,
        bounds.bottom + 0.5 * resolution,
        shape[0],
    )
    return xr.DataArray(
        data=data,
        dims=["y", "x"],
        coords={"x": xcoords, "y": ycoords},
        attrs={
            "bounds": bounds,
            "resolution": resolution,
            "transform": transform,
            "crs": crs,
        },
    )


def interpolate_on_grid(
    data: xr.Dataset | xr.DataArray,
    grid: xr.DataArray,
    algorithm: rio.enums.Resampling = rio.enums.Resampling.bilinear,
    dtype: np.dtype = np.dtype("float32"),  # noqa
) -> xr.Dataset | xr.DataArray:
    """
    Method for spatial interpolation on a grid

    :param data: Dataset or DataArray to to be spatially interpolated
    :param grid: Grid used for interplation
    :param algorithm: Algorithm used for resampling
    :param dtype: Type of data
    :return: xr.Dataset or xr.DataArray
    """
    # Parameters for projection
    src_transform = data.transform
    src_crs = data.crs
    dst_transform = grid.transform
    dst_crs = grid.crs
    dst_sizey = grid.sizes["y"]
    dst_sizex = grid.sizes["x"]

    if isinstance(data, xr.DataArray):
        src_data = data.to_numpy()
        dst_data = np.zeros((dst_sizey, dst_sizex), dtype)
        reproject(
            src_data,
            dst_data,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=algorithm,
        )
        return xr.DataArray(
            dst_data,
            coords=[grid.coords["y"], grid.coords["x"]],
            dims=["y", "x"],
            attrs={
                "crs": dst_crs,
                "bounds": grid.bounds,
                "transform": dst_transform,
                "resolution": grid.resolution,
            },
        )
    if isinstance(data, xr.Dataset):
        values: dict[str, tuple[list[str], np.ndarray]] = {}
        for var in data.data_vars:
            src_data = data[var].to_numpy()
            dst_data = np.zeros((dst_sizey, dst_sizex), dtype)
            reproject(
                src_data,
                dst_data,
                src_transform=src_transform,
                src_crs=src_crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=algorithm,
            )
            values[str(var)] = (["y", "x"], dst_data)

        return xr.Dataset(
            values,
            coords={"x": grid.coords["x"], "y": grid.coords["y"]},
            attrs={
                "crs": dst_crs,
                "bounds": grid.bounds,
                "transform": dst_transform,
                "resolution": grid.resolution,
            },
        )
    raise TypeError(
        "Interpolation can only be performed on Dataset or DataArray"
    )

    return data


def interpolate_temperature(
    src_temp: xr.DataArray,
    src_dem: xr.DataArray,
    dst_dem: xr.DataArray,
    lapse_rate: float,
    algorithm: rio.enums.Resampling = rio.enums.Resampling.bilinear,
    dtype: np.dtype = np.dtype("float32"),  # noqa
) -> xr.DataArray:
    """
    Interpolation of temperature

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

    :param src_temp: Temperature
    :param src_dem: Elevation associated to temperature
    :param dst_dem: Elevation used for the projection
    :param lapse_rate: Gradient of temperature per unit of elevation
    :param algorithm: Algorithm used for resampling
    :param dtype: Type of data
    :return: xr.DataArray
    """
    # Compute temperature at reference elevation
    ref_temp = src_temp - lapse_rate * src_dem
    ref_temp.attrs = src_temp.attrs.copy()
    # Project into the coordinate system of the destination DEM
    ref_temp = interpolate_on_grid(
        data=ref_temp,  # type: ignore
        grid=dst_dem,
        algorithm=algorithm,
        dtype=dtype,
    )
    # Adjust temperatures to DEM elevation
    dem_temp = ref_temp + lapse_rate * dst_dem
    dem_temp.attrs = dst_dem.attrs.copy()
    return dem_temp

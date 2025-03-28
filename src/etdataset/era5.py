#!/usr/bin/env python
#
# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales / Université Paul Sabatier (UT3)
#
"""
Module for reading ERA5 and ERA5-land
"""

import os
import sys
import zipfile
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from time import sleep
from typing import Any, cast

import cdsapi
import numpy as np
import psutil  # type: ignore
import rasterio as rio
import xarray as xr
from pyproj import CRS
from rasterio.warp import reproject, transform_bounds
from sensorsio.utils import bb_transform

from .logging import LoggerManager

logger = LoggerManager.get_logger(__name__)


class ERA5Exception(Exception):
    """
    Exception for ERA5
    """


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


class ERA5Data:
    """ERA5 data model"""

    def __init__(
        self,
        filename: str,
    ) -> None:
        """
        Init method: Get metadata from ERA5 file
        It is assumed that the resolution is identical in longitude and latitude.
        """
        self.filename = filename
        # Data in WGS 84
        self.crs = CRS.from_epsg(4326).to_string()
        # Get available variables
        self.__available_variables = []
        with (
            xr.open_dataset(zipfile.ZipFile(self.filename).open("data.nc"))  # type: ignore
            if zipfile.is_zipfile(self.filename)
            else xr.open_dataset(self.filename)
        ) as ds:
            for var in ds.data_vars:
                self.__available_variables.append(ERA5Var.from_key(var))
            self.__available_dates = [date.strftime("%Y-%m-%d %H:%M:%S") for date in ds.indexes["valid_time"]]
            lon: np.ndarray = self._trunc(ds.longitude.data, 6)
            lat: np.ndarray = self._trunc(ds.latitude.data, 6)
            self.resolution = float(self._trunc(lon[1] - lon[0], 6))
            west, south, east, north = (
                np.min(lon) - self.resolution / 2,
                np.min(lat) - self.resolution / 2,
                np.max(lon) + self.resolution / 2,
                np.max(lat) + self.resolution / 2,
            )
            self.transform = rio.transform.from_origin(float(west), float(north), self.resolution, self.resolution)
            self.bounds = rio.coords.BoundingBox(float(west), float(south), float(east), float(north))

    def __repr__(self):
        return f"ERA5 data file={self.filename}"

    def _trunc(self, values: np.ndarray, decs: int = 0) -> np.ndarray:
        """
        Truncate decimal digits of an array
        """
        return cast(np.ndarray, np.trunc(values * 10**decs) / (10**decs))

    def get_available_variables(self) -> list[ERA5Var]:
        """
        Return the variables avalaible in the product
        """
        return [var.key for var in self.__available_variables]

    def get_available_dates(self) -> list[str]:
        """
        Return the date avalaible in the product
        """
        return self.__available_dates

    def read_as_numpy(
        self,
        variables: list[ERA5Var] | None = None,
        dates: list[str] | None = None,
    ) -> tuple[
        np.ndarray[Any, Any],
        np.ndarray[Any, Any],
        np.ndarray[Any, Any],
        np.ndarray[Any, Any],
        str,
    ]:
        """
        Read data from ERA5 products as a numpy ndarray.

        :param variables: The list of variables to read
        :param dates: The list of dates to read
        :return: The image pixels as a np.ndarray of shape [variables, width, height],
                 The time coords as a np.array of shape [dates],
                 The x coords as a np.ndarray of shape [width],
                 the y coords as a np.ndarray of shape [height],
                 the crs as a string
        """
        # Safeguard
        if variables is None:
            variables = self.__available_variables
        assert len(variables) > 0
        for var in variables:
            if var.key not in self.get_available_variables():
                raise ValueError(
                    f"Error variable {var.key} not found in the product ({self.get_available_variables()})"
                )

        datasets = []

        with (
            xr.open_dataset(
                zipfile.ZipFile(self.filename).open("data.nc")  # type: ignore
            )
            if zipfile.is_zipfile(self.filename)
            else xr.open_dataset(self.filename)
        ) as ds:
            time: np.ndarray = np.array([], dtype=np.datetime64)
            if dates is not None:
                for date in dates:
                    try:
                        time = np.append(time, ds.sel(valid_time=date).valid_time.data)
                    except KeyError as exc:
                        raise KeyError(f"Date {date} not found in the dataset") from exc
            else:
                time = ds.valid_time.data

            # Safeguard to compute if there is enough memory available
            available_memory = psutil.virtual_memory().available / 1024 / 1024 / 1024  # in Gb
            requested_memory = (
                (sys.getsizeof(np.float32) * len(variables) * len(time) * ds.sizes["longitude"] * ds.sizes["latitude"])
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
            if dates is None:
                datasets = [ds[var.key].to_numpy() for var in variables]
            else:
                for var in variables:
                    sub_datasets = []
                    for date in dates:
                        sub_data = ds.sel(valid_time=date)[var.key].to_numpy()
                        if sub_data.ndim == 2:
                            sub_data = np.expand_dims(sub_data, axis=0)
                        sub_datasets.append(sub_data)
                    datasets.append(np.concatenate([data for data in sub_datasets], axis=0))

        np_stack: np.ndarray = np.stack([data for data in datasets], axis=0)

        xcoords: np.ndarray = ds.coords["longitude"].data

        ycoords: np.ndarray = ds.coords["latitude"].data

        return np_stack, time, xcoords, ycoords, self.crs

    def read(
        self,
        variables: list[ERA5Var] | None = None,
        dates: list[str] | None = None,
    ) -> xr.Dataset:
        """
        Read data from ERA5 products as xarray dataset.

        :param variables: The list of variables to read
        :param dates: The list of dates to read
        :return: xr.Dataset
        """
        if variables is None:
            variables = self.__available_variables

        np_arr, time, xcoords, ycoords, crs = self.read_as_numpy(variables, dates)

        data = {}
        for i, var in enumerate(variables):
            data[var.key] = (["t", "y", "x"], np_arr[i])

        return xr.Dataset(
            data,
            coords={"t": time, "x": xcoords, "y": ycoords},
            attrs={
                "crs": crs,
                "transform": self.transform,
                "bounds": self.bounds,
                "resolution": {"x": self.resolution, "y": self.resolution},
            },
        )

    def get(self, var: ERA5Var, date: datetime) -> xr.DataArray:
        """
        This method:
          - Extract one specific variable from ERA5 product
          - Perform a time interpolation
          - Return an xarray DataArray with the corresponding data

        :param var: Variable to extract
        :param date: Date to interpolate
        :return: xr.DataArray
        """
        factor = 1.0
        search = var
        if var == ERA5Var.HEIGHT:
            search = ERA5Var.GEOPOTENTIAL
            factor = 1 / 9.80665
        xrds = self.read([search], [date.date().strftime("%Y-%m-%d")])
        xrds = interpolate_time(xrds, [search], date)
        xarr = xrds[search.key].copy() * factor
        xarr.attrs = xrds.attrs.copy()
        return xarr


def interpolate_time(data: xr.Dataset, vars: list[ERA5Var], date: datetime) -> xr.Dataset:
    """
    Interpolate data for a variable at the date

    :param data: Dataset all the variables and the dates
    :param vars: List of Variables to consider for the interpolation
    :param date: Date at which interpolation is computed
    :return: xr.Dataset
    """
    # Check variable
    vars_str: list[str] = []
    for var in vars:
        if var.key not in list(data.data_vars):
            raise KeyError(f"Variable {var} not found")
        vars_str.append(var.key)
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
    selected = data[vars_str].sel(t=dates_str)
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
    dtype: np.dtype = np.dtype("float32"),
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
    assert (data.dims) != 2
    try:
        src_crs = data.crs
        src_transform = data.transform
        src_bounds = data.bounds
        src_resolutionx = data.resolution["x"]
        src_resolutiony = data.resolution["y"]
    except KeyError:
        raise ERA5Exception("Data without projection metadata")
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
        if resolution is not None and (src_resolutionx != resolution or src_resolutiony != resolution):
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
            dst_transform = rio.transform.from_bounds(*dst_bounds, data.sizes["x"], data.sizes["y"])
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
    available_memory = psutil.virtual_memory().available / 1024 / 1024 / 1024  # in Gb
    requested_memory = (sys.getsizeof(dtype) * nb_vars * dst_sizex * dst_sizey) / 1024 / 1024 / 1024
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
            values = {str(var): (["y", "x"], data[var].to_numpy()) for var in data.data_vars}

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
    raise ERA5Exception("Interpolation can only be performed on Dataset or DataArray")


def interpolate_on_grid(
    data: xr.Dataset | xr.DataArray,
    grid: xr.DataArray,
    algorithm: rio.enums.Resampling = rio.enums.Resampling.bilinear,
    dtype: np.dtype = np.dtype("float32"),
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
    raise ERA5Exception("Interpolation can only be performed on Dataset or DataArray")


def interpolate_temperature(
    src_temp: xr.DataArray,
    src_dem: xr.DataArray,
    dst_dem: xr.DataArray,
    lapse_rate: float,
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
    :return: xr.DataArray
    """
    # Compute temperature at reference elevation
    ref_temp = src_temp - lapse_rate * src_dem
    ref_temp.attrs = src_temp.attrs.copy()
    # Project into the coordinate system of the destination DEM
    ref_temp = interpolate_on_grid(
        data=ref_temp,  # type: ignore
        grid=dst_dem,
    )
    # Adjust temperatures to DEM elevation
    dem_temp = ref_temp + lapse_rate * dst_dem
    dem_temp.attrs = dst_dem.attrs.copy()
    return dem_temp


def interpolate_ozone(
    data: xr.DataArray,
    dem: xr.DataArray,
) -> xr.DataArray:
    """
    Interpolation of ozone

    The reference-level ozone is projected
    from the original geographic coordinate system (i.e. WGS84 for ERA5)
    onto the projection coordinate system of the destination DEM
    using bilinear interpolation.

    No elevation correction is performed.

    :param data: Ozone
    :param dem: Elevation used for the projection
    :return: xr.DataArray
    """
    # Project into the coordinate system of the destination DEM
    ozone = interpolate_on_grid(
        data=data,  # type: ignore
        grid=dem,
    )
    ozone.attrs = dem.attrs.copy()
    return ozone  # type: ignore


def interpolate_radiation(
    data: xr.DataArray,
    dem: xr.DataArray,
) -> xr.DataArray:
    """
    Interpolation of radiation

    The reference-level ozone is projected
    from the original geographic coordinate system (i.e. WGS84 for ERA5)
    onto the projection coordinate system of the destination DEM
    using bilinear interpolation.

    No elevation correction is performed.

    :param data: Radiation
    :param dem: Elevation used for the projection
    :return: xr.DataArray
    """
    # Project into the coordinate system of the destination DEM
    radiation = interpolate_on_grid(
        data=data,  # type: ignore
        grid=dem,
    )
    radiation.attrs = dem.attrs.copy()
    return radiation  # type: ignore


def _download(dataset: str, request: dict, target: str) -> None:
    """
    Download from ERA5
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
            for n in reply.get("error", {}).get("context", {}).get("traceback", "").split("\n"):
                if n.strip() == "":
                    break
                result.error(f"  {n}")
            raise ERA5Exception(f"{reply['error'].get('message')}  {reply['error'].get('reason')}")
    result.download(target)


def download_era5(
    date: datetime,
    roi_bbox: rio.coords.BoundingBox,
    roi_crs: CRS,
    path: str = os.getcwd(),
) -> None:
    """
    Download ERA5

    :param date: Date for the data
    :param roi_bbox: Bounds of the ROI
    :param roi_crs: CRS of the ROI
    :param path: Directory path to store data
    """
    # Convert bounding box to lat/lon
    west, south, east, north = transform_bounds(roi_crs, CRS(4326), *roi_bbox)
    west = (west - 0.1) if (west - 0.1) > -180 else -180
    south = (south - 0.1) if (south - 0.1) > -90 else -90
    east = (east + 0.1) if (east + 0.1) < 180 else 180
    north = (north + 0.1) if (north + 0.1) < 90 else 90
    # Filename
    filename = os.path.join(
        path,
        f"download_era5_{date.date().strftime('%Y-%m-%d')}_{north:.2f}_{west:.2f}_{south:.2f}_{east:.2f}.nc",
    )
    # Dataset ERA5
    dataset = "reanalysis-era5-single-levels"
    # Request
    request = {
        "product_type": "reanalysis",
        "variable": [
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
            "geopotential",
        ],
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
        "area": [north, west, south, east],
        "data_format": "netcdf",
        "download_format": "unarchived",
    }
    # Download
    _download(dataset, request, filename)


def download_era5land(
    date: datetime,
    roi_bbox: rio.coords.BoundingBox,
    roi_crs: CRS,
    path: str = os.getcwd(),
) -> None:
    """
    Download ERA5 Land

    :param date: Date for the data
    :param roi_bbox: Bounds of the ROI
    :param roi_crs: CRS of the ROI
    :param path: Directory path to store data
    """
    # Convert bounding box to lat/lon
    west, south, east, north = transform_bounds(roi_crs, CRS(4326), *roi_bbox)
    west = (west - 0.1) if (west - 0.1) > -180 else -180
    south = (south - 0.1) if (south - 0.1) > -90 else -90
    east = (east + 0.1) if (east + 0.1) < 180 else 180
    north = (north + 0.1) if (north + 0.1) < 90 else 90
    # Filename
    filename = os.path.join(
        path,
        f"download_era5land_{date.date().strftime('%Y-%m-%d')}_{north:.2f}_{west:.2f}_{south:.2f}_{east:.2f}.nc",
    )
    # Dataset
    dataset = "reanalysis-era5-land"
    # Request
    request = {
        "variable": [
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "2m_dewpoint_temperature",
            "2m_temperature",
            "surface_solar_radiation_downwards",
            "surface_thermal_radiation_downwards",
            "total_precipitation",
        ],
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
        "area": [north, west, south, east],
        "data_format": "netcdf",
        "download_format": "unarchived",
    }
    # Download
    _download(dataset, request, filename)

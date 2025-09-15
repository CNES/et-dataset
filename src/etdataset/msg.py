#
# Copyright: (c) 2025 CESBIO / Centre National d'Etudes Spatiales /
#             Université Paul Sabatier (UT3)
#
"""
Module for downloading and reading MSG product from LSA-SAF
"""

from __future__ import annotations

import bisect
import datetime as dt
import os
import tempfile
from dataclasses import dataclass
from enum import Enum

import dask.array as da
import numpy as np
import pandas as pd
import rasterio as rio
import requests
import xarray as xr
from psutil import cpu_count
from pyproj import CRS
from sensorsio import utils

from etdataset.logging import LoggerManager
from etdataset.utils import mask_bits

logger = LoggerManager.get_logger(__name__)


class MSGException(Exception):
    """
    Class Exception for handling MSG product
    """


@dataclass
class SatelliteInfo:
    """Class for satellite info"""

    key: str
    label: str


class MSGSatellite(SatelliteInfo, Enum):
    MSG = ("MSG", "MSG")
    MSGIODC = ("MSG-IODC", "IODC")


@dataclass
class FormatInfo:
    """Class for satellite info"""

    key: str
    label: str
    extension: str


class MSGFormat(FormatInfo, Enum):
    HDF5 = ("HDF5", "HDF5", "")
    NETCDF = ("NETCDF", "NETCDF4", ".nc")


@dataclass
class ProductInfo:
    """Class for product info"""

    key: str
    short: str
    freq: str


@dataclass
class MSGDataInfo:
    """Class for describing MSG data"""

    key: str
    label: str
    unit: str


class MSGVar(MSGDataInfo, Enum):
    """
    MSG variables
    """

    SURFACE_SOLAR_RADIATION_DOWNWARD = (
        "DSSF_TOT",
        "Surface solar radiation downward",
        "W.m-2",
    )
    SURFACE_THERMAL_RADIATION_DOWNWARD = (
        "DSLF",
        "Surface thermal radiation downward",
        "W.m-2",
    )
    FRACTION_DIFFUSE = (
        "FRACTION_DIFFUSE",
        "Fraction diffuse",
        "-",
    )
    QA = (
        "quality_flag",
        "Quality flags",
        "-",
    )
    DAILY_SURFACE_SOLAR_RADIATION_DOWNWARD = (
        "DSSF",
        "Daily surface solar radiation downward",
        "J.m-2",
    )

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


class MSGProduct(ProductInfo, Enum):
    SURFACE_SOLAR_RADIATION_DOWNWARD = ("MDSSFTD", "MDSSFTD", "15min")
    SURFACE_THERMAL_RADIATION_DOWNWARD = ("MDSLF", "DSLF", "30min")
    DAILY_SURFACE_SOLAR_RADIATION_DOWNWARD = ("MDIDSSF", "DIDSSF", "1day")


def get_url(
    satellite: MSGSatellite,
    product: MSGProduct,
    date: dt.datetime,
    fmt: MSGFormat,
):
    """
    Description
    -----------
    Get URL file for a product from LSA-SAF

    Parameters
    ----------
    satellite: MSGSatellite
        Satellite used
    product: MSGProduct
        Type of product to download
    date: datetime.datetime
        Date
    path: str
        Directory to download data
    fmt: MSGFormat
        Format of the data
    """
    if os.environ.get("LSASAF_USER", None) is None:
        raise ValueError("LSASAF_USER not provided")
    if os.environ.get("LSASAF_PASSWORD", None) is None:
        raise ValueError("LSASAF_PASSWORD not provided")
    login = os.environ["LSASAF_USER"]
    password = os.environ["LSASAF_PASSWORD"]
    if product.value == MSGProduct.DAILY_SURFACE_SOLAR_RADIATION_DOWNWARD.value:
        date = dt.datetime(date.year, date.month, date.day, 0, 0)
    else:
        old_date = date
        date = find_previous_date(product, old_date)
        if date != old_date:
            logger.warning("Use closest previous product date: {date}")
    filename = (
        f"{fmt.label}_LSASAF_{satellite.key}_{product.short}_{satellite.label}"
        f"-Disk_{date.strftime('%Y%m%d%H%M')}{fmt.extension}"
    )
    return (
        f"https://{login}:{password}@datalsasaf.lsasvcs.ipma.pt/PRODUCTS/{satellite.key}/{product.key}/"
        f"{fmt.key}/{date.year}/{str(date.month).zfill(2)}/{str(date.day).zfill(2)}/{filename}"
    )


def _download(
    satellite: MSGSatellite,
    product: MSGProduct,
    date: dt.datetime,
    fmt: MSGFormat,
    path: str,
) -> None:
    """
    Description
    -----------
    Download a MSG product from LSA-SAF

    Parameters
    ----------
    satellite: MSGSatellite
        Satellite used
    product: MSGProduct
        Type of product to download
    date: datetime.datetime
        Date
    path: str
        Directory to download data
    fmt: MSGFormat
        Format of the data
    """
    if product.value == MSGProduct.DAILY_SURFACE_SOLAR_RADIATION_DOWNWARD.value:
        date = dt.datetime(date.year, date.month, date.day, 0, 0)
    filename = (
        f"{fmt.label}_LSASAF_{satellite.key}_{product.short}_{satellite.label}"
        f"-Disk_{date.strftime('%Y%m%d%H%M')}{fmt.extension}"
    )
    # Do not download if the file already exists
    if os.path.exists(os.path.join(path, filename)):
        logger.info(
            f"File {os.path.join(path, filename)} already exists. "
            "Skip download."
        )
        return

    url = (
        f"https://datalsasaf.lsasvcs.ipma.pt/PRODUCTS/{satellite.key}/"
        f"{product.key}/{fmt.key}/{date.year}/{str(date.month).zfill(2)}/"
        f"{str(date.day).zfill(2)}/{filename}"
    )

    logger.info(f"Download url: {url}")

    response = requests.get(
        url, auth=(os.environ["LSASAF_USER"], os.environ["LSASAF_PASSWORD"])
    )

    if response.status_code == 200:
        with open(os.path.join(path, filename), "wb") as file:
            file.write(response.content)
        logger.debug(f"File {filename} downloaded successfully")
    else:
        logger.error(f"Failed to download file {filename}")


def get_filename(
    satellite: MSGSatellite,
    product: MSGProduct,
    date: dt.datetime,
    fmt: MSGFormat,
) -> str:
    """
    Description
    -----------
    Get filename of a MSG product from LSA-SAF

    Parameters
    ----------
    satellite: MSGSatellite
        Satellite used
    product: MSGProduct
        Type of product to download
    date: datetime.datetime
        Date
    fmt: MSGFormat
        Format of the data
    """
    if product.value == MSGProduct.DAILY_SURFACE_SOLAR_RADIATION_DOWNWARD.value:
        date = dt.datetime(date.year, date.month, date.day, 0, 0)
    return (
        f"{fmt.label}_LSASAF_{satellite.key}_{product.short}_{satellite.label}"
        f"-Disk_{date.strftime('%Y%m%d%H%M')}{fmt.extension}"
    )


def get_satellite(latlon_bbox: rio.coords.BoundingBox):
    """
    Get best satellite
    """
    if latlon_bbox.left > 30:
        return MSGSatellite.MSGIODC
    return MSGSatellite.MSG


def find_nearest_date(product: MSGProduct, pivot: dt.datetime) -> dt.datetime:
    """
    Find the closest nearest product date
    """
    time = pd.date_range(
        pivot.strftime("%Y-%m-%d"),
        freq=product.freq,
        end=(pivot + pd.Timedelta("1day")).strftime("%Y-%m-%d"),
    )
    return min(time, key=lambda x: abs(x - pivot))


def find_previous_date(product: MSGProduct, pivot: dt.datetime) -> dt.datetime:
    """
    Find the closest previous product date
    """
    time = pd.date_range(
        pivot.strftime("%Y-%m-%d"),
        freq=product.freq,
        end=(pivot + pd.Timedelta("1day")).strftime("%Y-%m-%d"),
    )
    # Use bisect_right to find the index
    idx = bisect.bisect_right(time, pivot)

    # Check if we have a valid previous date
    if idx == 0:
        return time[0]
    return time[idx - 1]


def download_file(
    satellite: MSGSatellite,
    product: MSGProduct,
    date: dt.datetime,
    path: str | None = None,
    fmt: MSGFormat = MSGFormat.NETCDF,
) -> None:
    """
    Description
    -----------
    Download one MSG product file

    Parameters
    ----------
    satellite: MSGSatellite
        Satellite used
    product: MSGProduct
        Type of product to download
    date: datetime.datetime
        Date
    path: str
        Directory to download data
    fmt: MSGFormat
        Format of the data
    """
    if os.environ.get("LSASAF_USER", None) is None:
        raise ValueError("LSASAF_USER not provided")
    if os.environ.get("LSASAF_PASSWORD", None) is None:
        raise ValueError("LSASAF_PASSWORD not provided")
    if path is None:
        path = os.getcwd()
    if product.value == MSGProduct.DAILY_SURFACE_SOLAR_RADIATION_DOWNWARD.value:
        _download(
            satellite=satellite,
            product=product,
            date=dt.datetime(date.year, date.month, date.day, 0, 0),
            fmt=fmt,
            path=path,
        )
    else:
        previous = find_previous_date(product, date)
        if date != previous:
            logger.warning(
                "Use closest previous product date: "
                f"{previous.strftime('%Y-%m-%d %H:%M')}"
            )
        _download(
            satellite=satellite,
            product=product,
            date=previous,
            fmt=fmt,
            path=path,
        )


def download(
    date: dt.datetime,
    latlon_bbox: rio.coords.BoundingBox,
    path: str | None = None,
) -> None:
    """
    Description
    -----------
    Download data for a specific datetime

    Parameters
    ----------
    date: datetime.datetime
        Date
    latlon_bbox: rio.coords.BoundingBox
        Bounding box in lat/lon coordinates
    path: str
        Directory to download data (Default current directory)
    """
    if os.environ.get("LSASAF_USER", None) is None:
        raise ValueError("LSASAF_USER not provided")
    if os.environ.get("LSASAF_PASSWORD", None) is None:
        raise ValueError("LSASAF_PASSWORD not provided")
    if path is None:
        path = os.getcwd()
    # Create a directory MSG products
    msg_path = os.path.join(path, "MSG_data")
    os.makedirs(msg_path, exist_ok=True)
    # Choose satellite
    satellite = get_satellite(latlon_bbox)
    # Create path
    product_path = os.path.join(
        msg_path, f"{satellite.key}_{date.strftime('%Y-%m-%d')}"
    )
    os.makedirs(product_path, exist_ok=True)
    # For daily DSSF
    product = MSGProduct.DAILY_SURFACE_SOLAR_RADIATION_DOWNWARD
    _download(
        satellite=satellite,
        product=product,
        date=dt.datetime(date.year, date.month, date.day, 0, 0),
        fmt=MSGFormat.NETCDF,
        path=product_path,
    )
    # If date argument is a date
    if type(date) is dt.date:
        for product in [
            MSGProduct.SURFACE_SOLAR_RADIATION_DOWNWARD,
            MSGProduct.SURFACE_THERMAL_RADIATION_DOWNWARD,
        ]:
            time = pd.date_range(
                date.strftime("%Y-%m-%d"),
                freq=product.freq,
                end=(date + pd.Timedelta("1day")).strftime("%Y-%m-%d"),
            )
            for t in time:
                _download(
                    satellite=satellite,
                    product=product,
                    date=t,
                    fmt=MSGFormat.NETCDF,
                    path=product_path,
                )
    elif type(date) is dt.datetime:
        for product in [
            MSGProduct.SURFACE_SOLAR_RADIATION_DOWNWARD,
            MSGProduct.SURFACE_THERMAL_RADIATION_DOWNWARD,
        ]:
            previous = find_previous_date(product, date)
            # TODO: Correct type
            for t in [  # type: ignore
                previous - pd.Timedelta(product.freq),
                previous,
                previous + pd.Timedelta(product.freq),
            ]:
                _download(
                    satellite=satellite,
                    product=product,
                    date=t,
                    fmt=MSGFormat.NETCDF,
                    path=product_path,
                )
    else:
        raise MSGException(f"Unknown format for date {date}")


def add_daily_data(data: xr.Dataset, product: str) -> xr.Dataset:
    """
    Description
    -----------
    Add daily MSG data to the dataset

    Parameters
    ----------
    data: xr.Dataset
        Data
    product: str
        Path to daily product
    """
    # Ressource management
    mem_limit = int(
        np.ceil(
            len(data.x)
            * len(data.y)
            * np.dtype(np.float32).itemsize
            / (1024**2)
        )
        * 1.15
    )
    try:
        nb_threads = min(
            [cpu_count(logical=True), len(os.sched_getaffinity(0))]
        )
    except:  # noqa
        nb_threads = min(
            [cpu_count(logical=True)]
        )  # os.sched_getaffinity won't work on windows

    # Reproject

    # Read data
    daily_xrds = (
        xr.open_dataset(product)
        .set_coords("crs")
        .squeeze("time")
        .drop_vars("time")
    )
    daily_xrds = daily_xrds.rio.write_crs(CRS(4326))
    # Reproject data
    projected = (
        daily_xrds[[MSGVar.DAILY_SURFACE_SOLAR_RADIATION_DOWNWARD.key]]
        .rio.reproject_match(
            data,
            resampling=rio.enums.Resampling.nearest,
            num_threads=nb_threads,
            warp_mem_limit=mem_limit,
        )
        .drop_vars("crs")
        .rename(
            {MSGVar.DAILY_SURFACE_SOLAR_RADIATION_DOWNWARD.key: "daily_msg"}
        )
    )
    projected["daily_msg"].attrs.clear()
    projected["daily_msg"].attrs.update({"long_name": "DAILY DSSF"})
    # Add attributes
    projected["daily_msg"].attrs["standard_name"] = "rld"
    projected["daily_msg"].attrs["long_name"] = "Longwave downwelling radiation"
    projected["daily_msg"].attrs["name"] = "rld"
    projected["daily_msg"].attrs["unit"] = "W.m-2"
    projected["daily_msg"].attrs["description"] = (
        "Longwave downwelling radiation"
    )
    return data.merge(projected)


def add_solar_data(
    data: xr.Dataset, date: dt.datetime, products: list[str]
) -> xr.Dataset:
    """
    Description
    -----------
    Add MSG solar data to the dataset

    Parameters
    ----------
    data: xr.Dataset
        Data
    date: xr.Dataset
        Datetime
    products: list[str]
        List of paths to solar products
    """
    # Ressource management
    mem_limit = int(
        np.ceil(
            len(data.x)
            * len(data.y)
            * len(products)
            * 2  # Number of variables
            * np.dtype(np.float32).itemsize
            / (1024**2)
        )
        * 1.15
    )
    try:
        nb_threads = min(
            [cpu_count(logical=True), len(os.sched_getaffinity(0))]
        )
    except:  # noqa
        nb_threads = min(
            [cpu_count(logical=True)]
        )  # os.sched_getaffinity won't work on windows
    # Open multiple files in a single dataset
    xrds = xr.open_mfdataset(
        products,
        combine="nested",
        concat_dim="time",
        chunks={"time": 1, "lat": -1, "lon": -1},
    )
    xrds = xrds.drop_vars("crs")
    # Set chunk size
    chunk_size_interp = {"time": -1, "lat": 500, "lon": 500}
    chunk_size_calc = {"time": 1, "lat": 3201, "lon": 3201}
    # Correct time index if missing file
    full_time_index = pd.date_range(
        start=xrds.time.values[0],
        end=xrds.time.values[-1],
        freq=MSGProduct.SURFACE_SOLAR_RADIATION_DOWNWARD.freq,
    )
    if len(full_time_index) != len(xrds.time):
        xrds = (
            xrds.reindex(time=full_time_index, fill_value=np.nan)
            .chunk(chunks=chunk_size_interp)
            .interpolate_na(dim="time", method="linear")
            .chunk(chunks=chunk_size_calc)
        )
    else:
        xrds = xrds.chunk(chunks=chunk_size_calc)
    # Get number of lines
    N = len(xrds.lat)
    # Create time bias aray
    t_bias = da.empty(
        shape=(N, len(xrds.lon)),
        chunks=(chunk_size_calc["lat"], chunk_size_calc["lon"]),
        dtype=np.float32,
    )
    i = da.arange(N, dtype=np.float32)
    t_bias[:] = (12 * (i[:, None] / 3712) + 3) / 15
    # Copy original dataset
    corrected_ds = xrds.copy()
    # Apply time correction
    corrected_ds[MSGVar.SURFACE_SOLAR_RADIATION_DOWNWARD.key] = (
        xrds[MSGVar.SURFACE_SOLAR_RADIATION_DOWNWARD.key].shift(time=1)
        * (1 - t_bias)
        + xrds[MSGVar.SURFACE_SOLAR_RADIATION_DOWNWARD.key] * t_bias
    )
    # Drop the first date
    corrected_ds = corrected_ds.isel(time=slice(1, len(corrected_ds.time)))
    # Mask
    not_ocean = ~mask_bits(corrected_ds[MSGVar.QA.key], pos=0, mask="00")
    not_space = ~mask_bits(corrected_ds[MSGVar.QA.key], pos=0, mask="10")
    # clear = ~mask_bits(corrected_ds["quality_flag"], pos=2, mask="001")
    corrected_ds[MSGVar.SURFACE_SOLAR_RADIATION_DOWNWARD.key] = corrected_ds[
        MSGVar.SURFACE_SOLAR_RADIATION_DOWNWARD.key
    ].where(not_ocean & not_space)
    corrected_ds[MSGVar.FRACTION_DIFFUSE.key] = corrected_ds[
        MSGVar.FRACTION_DIFFUSE.key
    ].where(not_ocean & not_space)
    corrected_ds = corrected_ds.drop_vars(MSGVar.QA.key)
    # Interpolate
    interp_ds = (
        corrected_ds.interp(time=date, method="linear")
        .drop_vars("time")
        .compute()
    )
    interp_ds = interp_ds.rio.write_crs(CRS(4326))
    projected = (
        interp_ds.rio.reproject_match(
            data,
            resampling=rio.enums.Resampling.nearest,
            num_threads=nb_threads,
            warp_mem_limit=mem_limit,
        )
        .drop_vars("crs")
        .rename(
            {
                MSGVar.SURFACE_SOLAR_RADIATION_DOWNWARD.key: "rsd_msg",
                MSGVar.FRACTION_DIFFUSE.key: "fdiff_msg",
            }
        )
    )
    # Add attributes
    projected["rsd_msg"].attrs.clear()
    projected["rsd_msg"].attrs["standard_name"] = "rsd"
    projected["rsd_msg"].attrs["long_name"] = "Shortwave downwelling radiation"
    projected["rsd_msg"].attrs["name"] = "rsd"
    projected["rsd_msg"].attrs["unit"] = "W.m-2"
    projected["rsd_msg"].attrs["description"] = (
        "Shortwave downwelling radiation"
    )
    projected["fdiff_msg"].attrs.clear()
    projected["fdiff_msg"].attrs["standard_name"] = "fdiff"
    projected["fdiff_msg"].attrs["long_name"] = (
        "Diffuse fraction for shortwave downwelling radiation"
    )
    projected["fdiff_msg"].attrs["name"] = "fdiff"
    projected["fdiff_msg"].attrs["unit"] = "-"
    projected["fdiff_msg"].attrs["description"] = (
        "Diffuse fraction for shortwave downwelling radiation"
    )
    return data.merge(projected)


def add_thermal_data(
    data: xr.Dataset, date: dt.datetime, products: list[str]
) -> xr.Dataset:
    """
    Description
    -----------
    Add MSG thermal data to the dataset

    Parameters
    ----------
    data: xr.Dataset
        Data
    date: xr.Dataset
        Datetime
    products: list[str]
        List of paths to thermal products
    """
    # Ressource management
    mem_limit = int(
        np.ceil(
            len(data.x)
            * len(data.y)
            * len(products)
            * np.dtype(np.float32).itemsize
            / (1024**2)
        )
        * 1.15
    )
    try:
        nb_threads = min(
            [cpu_count(logical=True), len(os.sched_getaffinity(0))]
        )
    except:  # noqa
        nb_threads = min(
            [cpu_count(logical=True)]
        )  # os.sched_getaffinity won't work on windows
    # Open multiple files in a single dataset
    xrds = xr.open_mfdataset(
        products,
        combine="nested",
        concat_dim="time",
        chunks={"time": 1, "lat": -1, "lon": -1},
    )
    xrds = xrds.drop_vars("crs")
    # Set chunk size
    chunk_size_interp = {"time": -1, "lat": 500, "lon": 500}
    chunk_size_calc = {"time": 1, "lat": 3201, "lon": 3201}
    # Correct time index if missing file
    full_time_index = pd.date_range(
        start=xrds.time.values[0],
        end=xrds.time.values[-1],
        freq=MSGProduct.SURFACE_THERMAL_RADIATION_DOWNWARD.freq,
    )
    if len(full_time_index) != len(xrds.time):
        xrds = (
            xrds.reindex(time=full_time_index, fill_value=np.nan)
            .chunk(chunks=chunk_size_interp)
            .interpolate_na(dim="time", method="linear")
            .chunk(chunks=chunk_size_calc)
        )
    else:
        xrds = xrds.chunk(chunks=chunk_size_calc)
    # Copy original dataset
    corrected_ds = xrds.copy()
    # Mask
    not_ocean = ~mask_bits(corrected_ds[MSGVar.QA.key], pos=0, mask="00")
    not_space = ~mask_bits(corrected_ds[MSGVar.QA.key], pos=0, mask="10")
    # clear = ~mask_bits(corrected_ds[MSGVar.QA.value], pos=2, mask="001")
    corrected_ds[MSGVar.SURFACE_THERMAL_RADIATION_DOWNWARD.key] = corrected_ds[
        MSGVar.SURFACE_THERMAL_RADIATION_DOWNWARD.key
    ].where(not_ocean & not_space)
    corrected_ds = corrected_ds.drop_vars(MSGVar.QA.key)
    # Interpolate
    interp_ds = (
        corrected_ds.interp(time=date, method="linear")
        .drop_vars("time")
        .compute()
    )
    interp_ds = interp_ds.rio.write_crs(CRS(4326))
    projected = (
        interp_ds.rio.reproject_match(
            data,
            resampling=rio.enums.Resampling.nearest,
            num_threads=nb_threads,
            warp_mem_limit=mem_limit,
        )
        .drop_vars("crs")
        .rename({MSGVar.SURFACE_THERMAL_RADIATION_DOWNWARD.key: "rld_msg"})
    )
    # Add attributes
    projected["rld_msg"].attrs.clear()
    projected["rld_msg"].attrs["standard_name"] = "rld"
    projected["rld_msg"].attrs["long_name"] = "Longwave downwelling radiation"
    projected["rld_msg"].attrs["name"] = "rld"
    projected["rld_msg"].attrs["unit"] = "W.m-2"
    projected["rld_msg"].attrs["description"] = "Longwave downwelling radiation"
    return data.merge(projected)


def add(
    data: xr.Dataset,
    path: str | None = None,
) -> xr.Dataset:
    """
    Description
    -----------
    Add MSG data to the dataset

    Parameters
    ----------
    data: xr.Dataset
        Data
    path: str
        Directory where MSG data have been downloaded data
    """
    # Check inputs
    if data.attrs.get("vis_date", None) is None:
        raise ValueError("Vis date attribute is missing in dataset")
    if data.attrs.get("vis_time", None) is None:
        raise ValueError("Vis time attribute is missing in dataset")
    date = dt.datetime.combine(data.attrs["vis_date"], data.attrs["vis_time"])
    if data.rio.crs is None:
        if data.attrs.get("crs") is not None:
            crs = data.attrs["crs"]
            data = data.rio.write_crs(crs)
        raise ValueError("crs attribute is missing in dataset")
    crs = data.rio.crs
    bounds = rio.coords.BoundingBox(*data.rio.bounds())
    latlon_bounds = utils.bb_transform(
        source_crs=str(crs), target_crs="EPSG:4326", bounding_box=bounds
    )
    temp_dir = None
    if path is None:
        # Download data
        logger.debug("Download MSG data")
        # Create a temp directory
        temp_dir = tempfile.TemporaryDirectory()
        path = temp_dir.name
        logger.debug(f"Temp dir: {path}")
        # Download files
        download(date=date, latlon_bbox=latlon_bounds, path=path)
    # Check data
    logger.debug("Check data")
    # Choose satellite
    satellite = get_satellite(latlon_bounds)
    # Data path
    product_path = os.path.join(
        path, "MSG_data", f"{satellite.key}_{date.strftime('%Y-%m-%d')}"
    )
    logger.debug(f"Product path: {product_path}")
    if not os.path.isdir(product_path):
        raise OSError(f"Data path not found: {product_path}")
    # File list
    product = MSGProduct.DAILY_SURFACE_SOLAR_RADIATION_DOWNWARD
    filename = get_filename(
        satellite=satellite,
        product=product,
        date=dt.datetime(date.year, date.month, date.day, 0, 0),
        fmt=MSGFormat.NETCDF,
    )
    daily_product = os.path.join(product_path, filename)
    logger.debug(f"Daily product path: {daily_product}")
    if not os.path.isfile(daily_product):
        raise OSError(f"File not found: {daily_product}")
    solar_product = []
    product = MSGProduct.SURFACE_SOLAR_RADIATION_DOWNWARD
    previous = find_previous_date(product, date)
    # TODO: Correct type
    for t in [  # type: ignore
        previous - pd.Timedelta(product.freq),
        previous,
        previous + pd.Timedelta(product.freq),
    ]:
        solar_product.append(  # noqa
            os.path.join(
                product_path,
                get_filename(
                    satellite=satellite,
                    product=product,
                    date=t,
                    fmt=MSGFormat.NETCDF,
                ),
            )
        )
    logger.debug(f"Solar product paths: {solar_product}")
    for p in solar_product:
        if not os.path.isfile(p):
            raise OSError(f"File not found: {p}")
    thermal_product = []
    product = MSGProduct.SURFACE_THERMAL_RADIATION_DOWNWARD
    previous = find_previous_date(product, date)
    # TODO: Correct type
    for t in [  # type: ignore
        previous - pd.Timedelta(product.freq),
        previous,
        previous + pd.Timedelta(product.freq),
    ]:
        thermal_product.append(  # noqa
            os.path.join(
                product_path,
                get_filename(
                    satellite=satellite,
                    product=product,
                    date=t,
                    fmt=MSGFormat.NETCDF,
                ),
            )
        )
    logger.debug(f"Thermal product paths: {thermal_product}")
    for p in thermal_product:
        if not os.path.isfile(p):
            raise OSError(f"File not found: {p}")
    # Add data
    updated_data = add_daily_data(data, daily_product)
    logger.debug("Add daily data: OK")
    updated_data = add_solar_data(updated_data, date, solar_product)
    logger.debug("Add solar data: OK")
    updated_data = add_thermal_data(updated_data, date, thermal_product)
    logger.debug("Add thermal data: OK")
    # Clean
    if temp_dir is not None:
        temp_dir.cleanup()  # Manually delete the directory
    return updated_data

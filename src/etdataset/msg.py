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
from dataclasses import dataclass
from enum import Enum

import pandas as pd
import rasterio as rio
import requests

# from etdataset.interpolation import create_grid, interpolate_on_grid
# from etdataset.io import read_metadata
from etdataset.logging import LoggerManager

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
        "W m-2",
    )
    SURFACE_THERMAL_RADIATION_DOWNWARD = (
        "DLSF_TOT",
        "Surface thermal radiation downward",
        "W m-2",
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
    DAILY_SURFACE_SOLAR_RADIATION_DOWNWARD = ("MDIDSSF", "DIDSSF", "day")


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
    url = (
        f"https://datalsasaf.lsasvcs.ipma.pt/PRODUCTS/{satellite.key}/{product.key}/"
        f"{fmt.key}/{date.year}/{str(date.month).zfill(2)}/{str(date.day).zfill(2)}/{filename}"
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


def download_msg_file(
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


def download_msg(
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
    satellite = MSGSatellite.MSG
    if latlon_bbox.xmin > 30:
        satellite = MSGSatellite.MSGIODC
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
        time = pd.date_range(
            date.strftime("%Y-%m-%d"),
            freq=product.freq,
            end=(date + pd.Timedelta("1day")).strftime("%Y-%m-%d"),
        )
        for t in time:
            for product in [
                MSGProduct.SURFACE_SOLAR_RADIATION_DOWNWARD,
                MSGProduct.SURFACE_THERMAL_RADIATION_DOWNWARD,
            ]:
                _download(
                    satellite=satellite,
                    product=product,
                    date=t,
                    fmt=MSGFormat.NETCDF,
                    path=product_path,
                )
    elif type(date) is dt.datetime:
        time = pd.date_range(
            date.strftime("%Y-%m-%d"),
            freq=product.freq,
            end=(date + pd.Timedelta("1day")).strftime("%Y-%m-%d"),
        )
        previous = find_previous_date(product, date)
        # TODO: Correct type
        for t in [  # type: ignore
            previous - pd.Timedelta(product.freq),
            previous,
            previous + pd.Timedelta(product.freq),
        ]:
            for product in [
                MSGProduct.SURFACE_SOLAR_RADIATION_DOWNWARD,
                MSGProduct.SURFACE_THERMAL_RADIATION_DOWNWARD,
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

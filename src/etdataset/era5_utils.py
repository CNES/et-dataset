"""
Module for reading ERA5 and ERA5-land
"""

import datetime as dt
import os
import zipfile
from functools import lru_cache
from pathlib import Path
from time import sleep

import cdsapi
import xarray as xr
from pyproj import CRS

from etdataset.era5_type import ERA5Dataset, ERA5Exception
from etdataset.logging import LoggerManager

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

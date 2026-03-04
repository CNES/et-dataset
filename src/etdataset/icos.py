#
# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales /
#            Université Paul Sabatier (UT3)
#

"""
Module for ICOS data
"""

import os
import zipfile
from dataclasses import dataclass
from typing import overload

import geopandas as gpd
import numpy as np
import numpy.typing as npt
import pandas as pd
import requests
import xarray as xr
from icoscp_core.icos import meta
from pyproj import CRS
from shapely.geometry import Point

from etdataset.logging import LoggerManager

logger = LoggerManager.get_logger(__name__)

#########################################################
##                                                     ##
##                                                     ##
##                    ICOS STATIONS                    ##
##                                                     ##
##                                                     ##
#########################################################


@dataclass
class StationConfig:
    id: str
    name: str
    country: str
    lat: float
    lon: float
    elev: float


#####################################
##                                 ##
##                                 ##
##   Get available ICOS datas      ##
##                                 ##
##                                 ##
#####################################


def get_csv_with_valid_icos_stations(update: bool = False):
    """
    Description
    ----------
    Return the file path that contains all ICOS ecosystem stations (ES)
    that have available Meteo L2 dataset.
    If the file is not available or an update has been requested,
    fetch all ICOS ecosystem stations (ES) that have available Meteo L2 dataset,
    extract their metadata, save the list into a CSV file.

    Returns
    -------
    csv_path : str
        Path to the CSV file containing stations information.
    """
    csv_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "data",
        "stations_meteoL2.csv",
    )

    if not os.path.isfile(csv_path) or update:
        icos_stations = meta.list_stations()
        ecosystem_stations = [
            s for s in icos_stations if s.type_uri.endswith("/ES")
        ]
        datatypes = meta.list_datatypes()
        meteo_l2_filter = [
            d.uri
            for d in datatypes
            if "Meteo" in d.uri
            and "L2" in getattr(d, "label", "")
            and "Meteosens" not in d.uri
        ]
        list_valid_station = []
        for s in ecosystem_stations:
            data_objects = meta.list_data_objects(
                station=s.uri,
                datatype=meteo_l2_filter,
                order_by={"prop": "submTime", "descending": True},
            )
            if data_objects:
                list_valid_station.append(
                    {
                        "id": s.id,
                        "name": s.name,
                        "country": s.country_code,
                        "lat": s.lat,
                        "lon": s.lon,
                        "elev": s.elevation,
                    }
                )
        df = pd.DataFrame(list_valid_station)
        df.sort_values(
            by=["country", "name"], ascending=[True, False], inplace=True
        )
        df.to_csv(csv_path, index=False)
        logger.info(
            f"All valid stations and their metadata are stored in the csv file\
            : {csv_path}"
        )
    return csv_path


def get_station_list():
    """ """
    csv_path = get_csv_with_valid_icos_stations()
    data = pd.read_csv(csv_path, index_col="id")
    return list(data.index)


def filter_stations_by_country_code(country_code: str):
    """
    Description
    ----------
    Filter stations from a CSV file based on their country code.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file
    country_code : str
        Country code used to filter the stations (e.g., "FR", "DE").

    Returns
    -------
    data_filtered: pd.DataFrame
        A DataFrame containing only the stations filtered
    """
    csv_path = get_csv_with_valid_icos_stations()
    data = pd.read_csv(csv_path)
    data_filtered = data[data["country"] == country_code]
    return data_filtered


def filter_valid_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Description
    -----------
    Filter invalid RH and TA datas in station's data

    Parameters
    -----------
    data : pd.DtaFrame

    Return
    -----------
    filtered_data : pd.DataFrame
        Filtered dataframe
    """

    data["TIMESTAMP_START"] = pd.to_datetime(
        data["TIMESTAMP_START"], format="%Y%m%d%H%M"
    )

    # Select only valid mesures
    filtered_data = data[
        (data["TA"] != -9999) & (data["RH"] != -9999)
    ].reset_index(drop=True)
    return filtered_data


#####################################
##                                 ##
##                                 ##
##    Get metadatas of a station   ##
##                                 ##
##                                 ##
#####################################


def get_stations_config(id_station: str) -> StationConfig:
    """
    Description
    ----------
    Load station configurations from a CSV file.

    For each row, this function creates a StationConfig entry with:
        - id of the station
        - name of the station
        - country of the station
        - geographic coordinates
        - elevation

    Parameters
    ----------
    id_station : str
        IDs of ICOS station

    Returns
    -------
    StationConfig
        A dictionary mapping station IDs to their corresponding StationConfig.
    """
    csv_path = get_csv_with_valid_icos_stations()
    df = pd.read_csv(csv_path, index_col="id")
    try:
        station = df.loc[id_station]
    except KeyError:
        raise ValueError("Station id is unknown: {id_station}")
    # convert to StationConfig
    return StationConfig(id=id_station, **station)


def get_station_location(
    stations: str | list[str],
) -> gpd.GeoDataFrame:
    """
    Description
    ----------
    Create a geopandas DataFrame of all the given
    ICOS stations

    Parameters
    ----------
    stations : str | list[str]
        the given ICOS stations
    Returns
    -------
    gdf : GeoDataFrame
    """
    if isinstance(stations, str):
        stations = [stations]
    csv_path = get_csv_with_valid_icos_stations()
    df = pd.read_csv(csv_path, usecols=["id", "name", "lat", "lon", "elev"])
    df = df[df["id"].isin(stations)]
    geometry = [
        Point(row["lon"], row["lat"], row["elev"]) for _, row in df.iterrows()
    ]

    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=CRS.from_epsg(4326))

    return gdf


#####################################
##                                 ##
##                                 ##
##   Download icos stations file   ##
##                                 ##
##                                 ##
#####################################


def _download_file(obj, path: str, id_station: str, cookies: dict):
    """
    Description
    ----------
    Downloads a file associated with an ICOS object and saves it to the
    specified directory.

    Parameters
    ----------
    obj : DataObject
        ICOS file metadata object containing at least the attributes
        "filename" (file name) and "uri" (download endpoint).
    path : str
        Local directory where the file should be saved.
    id_station : str
        Id of the station to which the file belongs.
    cookies : dict
        Dictionary of HTTP cookies required for authentication during the
        download request.

    Returns
    -------
    str or None
        The absolute path to the downloaded file, or the path of the
        pre-existing file if it already exists. Returns "None" if the
        download request fails.

    """
    filename = f"ICOSETC_{id_station}_METEO_L2.zip"
    download_dir = os.path.abspath(path)
    os.makedirs(download_dir, exist_ok=True)

    file_path = os.path.join(download_dir, filename)
    # Do not download if the file already exists

    if os.path.exists(file_path):
        logger.info(f"File {file_path} already exists. Skip download")
        return file_path

    # get the url for downloading
    hash_id = obj.uri.split("/")[-1]
    url = f"https://data.icos-cp.eu/objects/{hash_id}"
    logger.info(f"Download url =  {url}")
    response = requests.get(url, cookies=cookies)

    if response.status_code == 200:
        os.makedirs(path, exist_ok=True)
        with open(file_path, "wb") as file:
            file.write(response.content)
        logger.info(
            f"File of the station {id_station} :"
            f"{filename} downloaded successfully in {path}"
        )
        return file_path
    logger.error(
        f"Failed to download file of the station {id_station}: {filename} with the status code : {response.status_code}"  # noqa: E501
    )
    return None


def download_file(
    data_objects,
    cookies: dict,
    id_station: str,
    path: str | None = None,
):
    """
    Description
    ----------
    Downloads all the files associated with ICOS objects and unzip them in
    the download folder.

    Parameters
    ----------
    data_objects : DataObjectList
        ICOS file metadata objects.
    cookies : dict
        Dictionary of HTTP cookies.
    station_id : str
        Id of the station.
    path : str
        Local directory where the file should be saved.

    Returns
    -------
    str or None

    """
    if cookies is None:
        raise ValueError("The autentification token is not provided")

    filename = f"ICOSETC_{id_station}_METEO_L2.zip"

    if path is None:
        download_folder = os.path.join(os.getcwd(), "ICOS")
    else:
        download_folder = os.path.abspath(os.path.join(path, "ICOS"))

    os.makedirs(download_folder, exist_ok=True)

    for obj in data_objects:
        downloaded_file = _download_file(
            obj, download_folder, id_station, cookies
        )
        if downloaded_file is None:
            continue
        # Unzip the downloaded file
        if downloaded_file.lower().endswith(".zip"):
            extract_folder = os.path.join(
                download_folder, os.path.splitext(filename)[0]
            )
            if not os.path.exists(extract_folder) or not os.listdir(
                extract_folder
            ):
                os.makedirs(extract_folder, exist_ok=True)
                try:
                    with zipfile.ZipFile(downloaded_file, "r") as zip_ref:
                        zip_ref.extractall(extract_folder)
                    logger.info(
                        f"{id_station} station file unzipped in : {extract_folder}"  # noqa: E501
                    )
                    for f in os.listdir(extract_folder):
                        if f.lower().endswith(".csv"):
                            old_csv_path = os.path.join(extract_folder, f)
                            filename_csv = f"ICOSETC_{id_station}_METEO_L2.csv"
                            csv_path = os.path.join(
                                extract_folder, filename_csv
                            )
                            os.rename(old_csv_path, csv_path)
                            logger.info(f"CSV file renamed to: {csv_path}")
                            break

                except zipfile.BadZipFile:
                    logger.exception(f"Non valid zip file: {downloaded_file}")
            else:
                logger.info(
                    f"{id_station} station file already extracted : {extract_folder}"  # noqa: E501
                )


def download_icos_station(
    stations: list[str] | str = "all",
    output: str | None = None,
):
    """
    Description
    ----------
    Downloads all files associated with the specified ICOS stations.

    It uses the provided authentication token as a cookie for access to the ICOS
    Carbon Portal.

    Parameters
    ----------
    auth_token : str
        ICOS authentication token (HTTP cookie). You can obtain this by
        creating an ICOS account. Tokens are refreshed every 28 hours via
        the API token section of your account.
    ids : list of str
        List of station IDs for which the files
        should be downloaded.

    Returns
    -------
    None
    """
    if os.environ.get("ICOS_API_TOKEN", None) is None:
        raise ValueError("ICOS_API_TOKEN is not provided")
    token = os.environ["ICOS_API_TOKEN"]
    cookies = {"cpauthToken": token}
    csv_path = get_csv_with_valid_icos_stations()
    df = pd.read_csv(csv_path)
    valid_stations_list = df["id"].tolist()

    if stations == "all":
        ids = valid_stations_list
    elif isinstance(stations, str) and stations != "all":
        ids = [stations]
    elif isinstance(stations, list) and stations != "all":
        ids = stations

    invalid_stations = [s for s in ids if s not in valid_stations_list]
    if invalid_stations:
        raise ValueError(f"Invalid station ID given: {invalid_stations}.")
    # Iterates over a list of ICOS station IDs
    for station_id in ids:
        station_uri = (
            f"http://meta.icos-cp.eu/resources/stations/ES_{station_id}"
        )
        # Retrieves available meteorological L2 data objects for each station
        datatypes = meta.list_datatypes()
        meteo_l2_filter = [
            d.uri
            for d in datatypes
            if "Meteo" in d.uri
            and "L2" in getattr(d, "label", "")
            and "Meteosens" not in d.uri
        ]

        data_objects = meta.list_data_objects(
            station=station_uri,
            datatype=meteo_l2_filter,
            order_by={"prop": "submTime", "descending": True},
        )
        " Download all the files"
        download_file(data_objects, cookies, station_id, output)
    logger.info(f"All the stations are downloaded {ids}")


#####################################
##                                 ##
##                                 ##
##   Read icos stations csv file   ##
##                                 ##
##                                 ##
#####################################


def read_csv_data(cfg: StationConfig, path: str | None = None) -> pd.DataFrame:
    """
    Description
    -----------
    Read station's csv

    Parameters
    -----------
    cfg : StationConfig
        Station configuration
    path : str
        base directory

    Return
    -----------
    data : pd.DataFrame
        Filtered dataframe
    """
    if path is None:
        path = os.getcwd()
    csv_path = os.path.join(
        path,
        "ICOS",
        f"ICOSETC_{cfg.id}_METEO_L2",
        f"ICOSETC_{cfg.id}_METEO_L2.csv",
    )

    usecols = [
        "TIMESTAMP_START",
        "TA",
        "RH",
    ]

    data = pd.read_csv(csv_path, usecols=usecols)
    return data


#####################################
##                                 ##
##                                 ##
##         Save ICOS data          ##
##                                 ##
##                                 ##
#####################################


def create_geopckg_from_gdf(gdf: gpd.GeoDataFrame, path: str | None = None):
    """
    Description
    ----------
    Create a geopackage of a GeoDataFrame
    ICOS stations

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame of stations
    path : str
        Path where to csv the pckg
    Returns
    -------
    """
    if path is None:
        file_path = os.path.join(os.getcwd(), "pckg")
    else:
        file_path = os.path.abspath(os.path.join(path, "pckg"))

    os.makedirs(file_path, exist_ok=True)
    pckg_name = "stations_package.gpkg"

    pckg_path = os.path.join(file_path, pckg_name)

    gdf.to_file(pckg_path, layer="stations", driver="GPKG")
    logger.info(f"Stations package save : {pckg_path}")


def save_station_data(
    cfg: StationConfig,
    data: pd.DataFrame,
    out_dir: str = "icos_data",
):
    """
    Description
    -----------
    Save ICOS station data with dew point temperature to CSV.

    Parameters
    ----------
    cfg : StationConfig
        Station configuration
    data : pd.DataFrame
        DataFrame with TIMESTAMP_START, TA, RH
    td : np.ndarray
        Dew point temperature array
    """

    folder = os.path.join(os.getcwd(), out_dir)
    os.makedirs(folder, exist_ok=True)
    rename_map = {
        "TIMESTAMP_START": "time",
        "TA": "ta",
        "TD": "tdp",
        "RH": "rh",
    }
    data = data.rename(
        columns={k: v for k, v in rename_map.items() if k in data.columns}
    )
    # data = data.rename(columns={"TIMESTAMP_START": "time"})
    csv_path = os.path.join(folder, f"{cfg.id}_data.csv")
    data.to_csv(csv_path, index=False)

    return csv_path


#########################################
##                                     ##
##                                     ##
## Calculations related to temperature ##
##                                     ##
##                                     ##
#########################################


def compute_dewpoint_temp(
    ta: npt.ArrayLike, rh: npt.ArrayLike, f: float = 243.04, d: float = 17.625
) -> npt.NDArray:
    """
    Description
    -----------
    Compute dew point temperature Tp from air temperature Ta (°C) and
    relative humidity RH (%):

            RH = 100 * exp[d*Td/(Td+f)-d*Ta/(Ta+f)]

            it gives:

            Td = f*(I + d*Ta/(Ta+f))/(d-I-d*Ta/(Ta+f))

            with I = ln(RH/100)

    from "The Relationship between Relative Humidity and the Dewpoint
    Temperature in Moist Air: A Simple Conversion and Applications"
    by Mark G. Lawrence

    Parameters
    -----------
    ta : ntp.ArrayLike
        Air temperature from ICOS
    rh : ntp.ArrayLike
        Relative humidity from ICOS

    Return
    -----------
    tp : ntp.NDArray
        Dew point temperature
    """
    ta = np.array(ta)
    rh = np.array(rh)
    L = np.log(rh / 100)
    gamma = L + d * ta / (ta + f)

    num = f * gamma
    den = d - gamma
    tp = num / den

    return tp


@overload
def kelvin_to_celsius(kelvin: xr.DataArray) -> xr.DataArray: ...
@overload
def kelvin_to_celsius(kelvin: npt.ArrayLike) -> npt.NDArray: ...


def kelvin_to_celsius(
    kelvin: xr.DataArray | npt.ArrayLike,
) -> xr.DataArray | npt.NDArray:
    """
    Description
    -----------
    Compute the temperature in celsius from a temperature in kelvin
    """
    if isinstance(kelvin, xr.DataArray):
        return kelvin - 273.15
    return np.array(kelvin) - 273.15


@overload
def celsius_to_kelvin(celsius: xr.DataArray) -> xr.DataArray: ...
@overload
def celsius_to_kelvin(celsius: npt.ArrayLike) -> npt.NDArray: ...


def celsius_to_kelvin(
    celsius: xr.DataArray | npt.ArrayLike,
) -> xr.DataArray | npt.NDArray:
    """
    Description
    -----------
    Compute the temperature in kelvin from a temperature in celsius
    """
    if isinstance(celsius, xr.DataArray):
        return celsius + 273.15
    return np.array(celsius) + 273.15

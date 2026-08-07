#
# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales /
#            Université Paul Sabatier (UT3)
#

"""
Module for Fluxnet data
"""

import glob
import os
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from pyproj import CRS
from shapely.geometry import Point

from etdataset.dewpoint_temp import compute_dewpoint_temp
from etdataset.icos import StationConfig
from etdataset.logging import LoggerManager

logger = LoggerManager.get_logger(__name__)

#########################################################
##                                                     ##
##                                                     ##
##                 Fluxnet STATIONS                    ##
##                                                     ##
##                                                     ##
#########################################################

COLUMN_RENAME = {
    "site_id": "id",
    "site_name": "name",
    "location_lat": "lat",
    "location_long": "lon",
}

FLUXNET_COLUMNS = {
    "TIMESTAMP_START": "time",
    "TA_F": "ta_gapfilled",
    "TA_F_QC": "ta_gapfilled_qc",
    "TA_F_MDS": "ta",
    "TA_F_MDS_QC": "ta_qc",
    "RH": "rh",
}
DATA_DIR = Path(__file__).parent / "data"
STATIONS_CSV = DATA_DIR / "fluxnet_shuttle.csv"


def get_fluxnet_stations_list():
    """Load the stations CSV and return a list of all station site IDs"""
    csv_path = STATIONS_CSV
    data = pd.read_csv(csv_path, index_col="site_id")
    return list(data.index)


def load_fluxnet_stations_csv() -> pd.DataFrame:
    """Load the stations CSV as a pandas DataFrame with renamed columns"""
    df = pd.read_csv(STATIONS_CSV)
    df = df.rename(columns=COLUMN_RENAME)
    return df


#####################################
##                                 ##
##                                 ##
##    Get metadata of a station   ##
##                                 ##
##                                 ##
#####################################


def get_fluxnet_stations_config(id_station: str) -> StationConfig:
    """
    Load station configurations from a CSV file

    For each row, this function creates a StationConfig entry with:
        - id of the station
        - name of the station
        - country of the station
        - geographic coordinates
        - elevation

    Parameters
    ----------
    id_station : str
        IDs of Fluxnet station

    Returns
    -------
    StationConfig
        A dictionary mapping station IDs to their corresponding StationConfig
    """
    df = load_fluxnet_stations_csv()
    df = df.set_index("id")
    try:
        station = df.loc[id_station]
    except KeyError:
        raise ValueError(f"Station id is unknown: {id_station}")
    # convert to StationConfig
    return StationConfig(
        id=id_station,
        name=str(station["name"]),
        country=str(station["data_hub"]),
        lat=float(station["lat"]),
        lon=float(station["lon"]),
        elev=float(station["elev"]) if "elev" in station.index else 0.0,
    )


def get_fluxnet_station_location(
    stations: str | list[str],
) -> gpd.GeoDataFrame:
    """
    Create a geopandas DataFrame of all the given
    Fluxnet stations

    Parameters
    ----------
    stations : str | list[str]
        the given Fluxnet stations
    Returns
    -------
    gdf : GeoDataFrame
    """
    if isinstance(stations, str):
        stations = [stations]
    df = load_fluxnet_stations_csv()
    df = df[df["id"].isin(stations)]
    geometry = [Point(row["lon"], row["lat"]) for _, row in df.iterrows()]

    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=CRS.from_epsg(4326))

    return gdf


def filter_fluxnet_stations_by_sources(src_code: str):
    """
    Filter stations from a CSV file based on their source code

    Parameters
    ----------
    src_code : str
        Source code used to filter the stations (e.g., "AmeriFlux", "ICOS")

    Returns
    -------
    data_filtered: pd.DataFrame
        A DataFrame containing only the stations filtered
    """
    df = load_fluxnet_stations_csv()
    data_filtered = df[df["data_hub"] == src_code]
    return data_filtered


def filter_fluxnet_stations_by_years(
    first_year: int, last_year: int
) -> pd.DataFrame:
    """
    Filter stations from a CSV file based on their data year range

    Parameters
    ----------
    first_year : int
        Start year to filter the stations (e.g., 2022)
    last_year : int
        End year to filter the stations (e.g., 2023)

    Returns
    -------
    data_filtered: pd.DataFrame
        A DataFrame containing only the stations filtered
    """
    df = load_fluxnet_stations_csv()
    data_filtered = df[
        (df["first_year"] <= first_year) & (df["last_year"] >= last_year)
    ]
    return data_filtered


#####################################
##                                 ##
##                                 ##
##      Handle Fluxnet EC data     ##
##                                 ##
##                                 ##
#####################################


def download_zip(
    download_link: str, file_path: str, id_station: str
) -> str | None:
    """
    Download a zip file from a URL and save it to a local path

    Parameters
    ----------
    download_link : str
        URL to download the file from
    file_path : str
        Local path where the file will be saved
    id_station : str
        Station ID (used for logging)

    Returns
    -------
    str | None
        The file path if download succeeded
    """
    logger.info(f"Download url = {download_link}")
    try:
        response = requests.get(download_link)
        if response.status_code == 200:
            with open(file_path, "wb") as f:
                f.write(response.content)
            logger.info(f"Station {id_station}: downloaded to {file_path}")
            return file_path
    except Exception:
        logger.exception(
            f"Error during the downloading of station {id_station}"
        )
        return None
    else:
        logger.error(
            f"Failed to download station {id_station}: {file_path} "
            f"(status code: {response.status_code})"
        )
        return None


def download_fluxnet_data(
    stations: str | list[str] = "all",
    csv_path: Path | str | None = None,
    output_dir: str | None = None,
) -> None:
    """
    Download, save, and extract Fluxnet station ZIP archives based on a
    metadata CSV

    Parameters
    ----------
    stations : str | list[str]
        Station ID or list of station IDs to download. If "all", downloads all
        valid stations found in the database.
    csv_path : Path | str | None
        Path to the local CSV file containing stations metadata. If None,
        defaults to global STATIONS_CSV.
    output_dir : str | None, default "fluxnet_data"
        Directory where downloaded ZIP files and extracted folders will be saved
    """
    # Stations
    valid_stations_list = get_fluxnet_stations_list()
    if stations == "all":
        ids = valid_stations_list
    elif isinstance(stations, str):
        ids = [stations]
    else:  # list
        ids = stations

    invalid_stations = [s for s in ids if s not in valid_stations_list]
    if invalid_stations:
        raise ValueError(f"Invalid station ID given: {invalid_stations}.")

    if csv_path is None:
        csv_path = STATIONS_CSV
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file is not found: {csv_path}")

    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), "fluxnet_data")
    else:
        output_dir = os.path.abspath(os.path.join(output_dir, "fluxnet_data"))

    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    df = df.rename(columns=COLUMN_RENAME)
    if "id" not in df.columns or "download_link" not in df.columns:
        raise KeyError("CSV file must contain 'id' and 'download_link'")

    # Filter metadata to keep only the requested stations
    df_filtered = df[df["id"].isin(ids)]
    if df_filtered.empty:
        logger.info("Stations not found")
        return

    os.makedirs(output_dir, exist_ok=True)

    ## Process downloads and extractions
    for _, row in df_filtered.iterrows():
        id_station = row["id"]
        download_link = row["download_link"]

        # Skip stations missing a valid download link
        if pd.isna(download_link):
            logger.warning(
                f"No download link for station {id_station}. Ignored."
            )
            continue

        filename = (
            row["filename"]
            if "filename" in row and pd.notna(row["filename"])
            else (
                f"{row['data_hub']}_{id_station}_"
                f"{row['first_year']}_{row['last_year']}.zip"
            )
        )
        file_path = os.path.join(output_dir, filename)

        downloaded_file = download_zip(download_link, file_path, id_station)
        if downloaded_file is None:
            continue

        # Unzip the downloaded file
        if downloaded_file.lower().endswith(".zip"):
            extract_folder = os.path.join(
                output_dir, os.path.splitext(filename)[0]
            )
            if not os.path.exists(extract_folder) or not os.listdir(
                extract_folder
            ):
                os.makedirs(extract_folder, exist_ok=True)
                try:
                    with zipfile.ZipFile(downloaded_file, "r") as zip_ref:
                        zip_ref.extractall(extract_folder)
                    logger.info(
                        f"{id_station} station file unzipped "
                        f"in: {extract_folder}"
                    )
                except zipfile.BadZipFile:
                    logger.exception(f"Non valid zip file: {downloaded_file}")
            else:
                logger.info(
                    f"{id_station} station files already "
                    f"extracted: {extract_folder}"
                )


def get_fluxnet_archive(
    station_id: str | list[str] = "all", data_dir: str = "fluxnet_data"
) -> str | None:
    """
    Find the HH (half hourly) CSV file path for a given station ID in the
    fluxnet_data folder

    Parameters
    ----------
    station_id : str
        Station ID (e.g. 'FR-Pue')
    fluxnet_dir : str
        Path to the fluxnet_data directory

    Returns
    -------
    str | None
        Path to the HH CSV file, or None if not found.
    """
    # Look for matching station directories
    folder_pattern = os.path.join(data_dir, f"*_{station_id}_*")
    folders = [f for f in glob.glob(folder_pattern) if os.path.isdir(f)]

    if not folders:
        logger.warning(
            f"No folder found for station {station_id} in {data_dir}"
        )
        return None

    station_folder = folders[0]

    # Search for the specific HH (half hourly) CSV file inside the folder
    csv_pattern = os.path.join(station_folder, "*_FLUXNET_FLUXMET_HH_*.csv")
    csv_files = glob.glob(csv_pattern)

    if not csv_files:
        logger.warning(f"No HH CSV found in folder {station_folder}")
        return None

    return csv_files[0]


def get_fluxnet_archives(
    stations: str | list[str] = "all",
    fluxnet_dir: str = "fluxnet_data",
) -> dict[str, str | None]:
    """
    Find HH (half-hourly) CSV file paths for one or multiple stations

    Parameters
    ----------
    stations : str | list[str]
        A station ID, a list of station IDs, or 'all'
    fluxnet_dir : str
        Path to the fluxnet_data directory

    Returns
    -------
    dict[str, str | None]
        Mapping of station_id -> CSV path
    """
    if stations == "all":
        # get station IDs from all zips present in the directory
        zips = glob.glob(os.path.join(fluxnet_dir, "*.zip"))
        ids = []
        for z in zips:
            parts = os.path.basename(z).split("_")
            if len(parts) >= 2:
                ids.append(parts[1])  # data_hub_STATIONID_...
    elif isinstance(stations, str):
        ids = [stations]
    else:
        ids = stations

    return {
        station_id: get_fluxnet_archive(station_id, fluxnet_dir)
        for station_id in ids
    }


def read_fluxnet_data(file_path: str) -> pd.DataFrame:
    """
    Read a Fluxnet CSV file and compute dewpoint temperature

    Parameters
    ----------
    file_path : str
        Path to the target Fluxnet CSV data file

    Returns
    -------
    pd.DataFrame
    """
    df = pd.read_csv(file_path, usecols=list(FLUXNET_COLUMNS.keys()))
    df = df.rename(columns=FLUXNET_COLUMNS)
    df = df.replace(-9999, np.nan)

    df["time"] = pd.to_datetime(df["time"], format="%Y%m%d%H%M")  # start
    # df["end"] = pd.to_datetime(df["end"], format="%Y%m%d%H%M")

    # Compute dewpoint temperature
    df["tdp_gpfilled"] = compute_dewpoint_temp(df["ta_gapfilled"], df["rh"])
    df["tdp"] = compute_dewpoint_temp(df["ta"], df["rh"])
    return df


def save_fluxnet_station(
    cfg: StationConfig, df: pd.DataFrame, output_dir: str = "fluxnet_processed"
) -> str:
    """
    Save the processed station DataFrame to a CSV file

    Parameters
    ----------
    cfg : StationConfig
        Configuration holding the station metadata (e.g., ID)
    df : pd.DataFrame
        The processed data to be saved
    output_dir : str, default "fluxnet_processed"
        Directory where the output file will be saved

    Returns
    -------
    str : path
    """
    if os.path.isabs(output_dir):
        folder = output_dir
    else:
        folder = os.path.join(os.getcwd(), output_dir)

    os.makedirs(folder, exist_ok=True)
    csv_path = os.path.join(folder, f"{cfg.id}_processed.csv")
    df.to_csv(csv_path, index=False)
    return csv_path

import datetime as dt
import os
import zipfile
from dataclasses import dataclass
from enum import Enum

import geopandas as gpd
import numpy as np
import numpy.typing as npt
import pandas as pd
import pyproj
import rasterio as rio
import requests
import rioxarray
import xarray as xr
from icoscp_core.icos import meta
from pyproj import CRS
from rasterio.transform import from_origin
from rioxarray.merge import merge_arrays
from shapely.geometry import Point, box

from etdataset.dem import compute_slope_aspect
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


def get_csv_with_valid_icos_stations(update: bool = False):  # noqa: FBT001
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


#####################################
##                                 ##
##                                 ##
##   DOWNLOAD ICOS STATIONS FILE   ##
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
    save : bool = False
        save or not the geodataframe into a geopackage file
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


# TO DO
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


def kelvin_to_celsius(kelvin: npt.ArrayLike) -> npt.NDArray:
    """
    Description
    -----------
    Compute the temperature in celsius from a temperature in kelvin

    """
    return np.array(kelvin) - 273.15


def save_station_data(
    cfg: StationConfig,
    data: pd.DataFrame,
    out_dir: str = "icos_data",
):
    """
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

    data = data.rename(columns={"TIMESTAMP_START": "time"})
    csv_path = os.path.join(folder, f"{cfg.id}_data.csv")
    data.to_csv(csv_path, index=False)

    return csv_path


#############################################################################################
@dataclass
class DatasetInfo:
    """Class for dataset info"""

    key: str
    label: str
    variables: list[str]


class ICOSDataset(DatasetInfo, Enum):
    """
    ICOS datasets
    """

    METEO = (
        "meteo",
        "ICOS METEO level 2",
        ["TA", "RH"],
    )


@dataclass
class ICOSDataInfo:
    """Class for describing ICOS data"""

    key: str
    label: str
    unit: str


class ICOSVar(ICOSDataInfo, Enum):
    """
    ICOS variables
    """

    AIR_TEMPERATURE = ("TA", "2m Air temparture", "°C")
    RELATIVE_HUMIDITY = ("RH", "2m Relative humidity", "%")
    DEWPOINT_TEMPERATURE = ("Td", "Dewpoint temperature", "°C")

    @staticmethod
    def compute_dewpoint_temp(
        ta: pd.Series, rh: pd.Series, f: float = 243.04, d: float = 17.625
    ) -> pd.Series:
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
        ta : pd.Series
            Air temperature from ICOS
        rh : pd.Series
            Relative humidity from ICOS

        Return
        -----------
        tp : pd.Series
            Dew point temperature
        """

        L = np.log(rh / 100)
        gamma = L + d * ta / (ta + f)

        num = f * gamma
        den = d - gamma
        tp = num / den

        return tp

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


# def kelvin_to_celsius(kelvin: float | np.ndarray) -> float | np.ndarray:
#    """
#    Description
#    -----------
#    Compute the temperature in celsius from a temperature in kelvin
#
#    """
#    return kelvin - 273.15


def add_time_attrs(ds: xr.Dataset, date: dt.datetime) -> xr.Dataset:
    """
    Description
    -----------
    Add time and date attributes to the data
    """
    ds.attrs.update(
        {
            "vis_date": date.date(),
            "vis_time": date.time(),
            "tir_date": date.date(),
            "tir_time": date.time(),
        }
    )
    return ds


def create_xr_point_dataset(
    time, ta: float | np.ndarray, td: float | np.ndarray
) -> xr.Dataset:
    """
    Description
    -----------
    Creates a single-point xarray Dataset with temperature and dew point.

    Parameters
    ----------
    time : datetime like
        Timestamp for the data point.
    ta : np.ndarray
        Air temperature value corresponding to the given time.
    td : np.ndarray
        Dew point temperature value corresponding to the given time.

    Returns
    -------
    xr.Dataset
    """
    return xr.Dataset(
        data_vars={"ta": ("time", [ta]), "td": ("time", [td])},
        coords={"time": [time]},
    )


class ICOSStation:
    def __init__(
        self,
        stations_cfg: StationConfig,
    ):
        self.id = stations_cfg.id
        self.name = stations_cfg.name
        self.csv_path = f"ICOS/ICOSETC_{stations_cfg.id}_METEO_L2/ICOSETC_{stations_cfg.id}_METEO_L2.csv"  # noqa: E501
        self.latitude = stations_cfg.lat
        self.longitude = stations_cfg.lon
        self.elevation = stations_cfg.elev
        self.crs = CRS("EPSG:4326")
        self.roi_bounds: rio.coords.BoundingBox | None = None
        self.data: pd.DataFrame

        # Among all the available mesures, select only Ta and RH
        usecols = [
            "TIMESTAMP_START",
            "TIMESTAMP_END",
            ICOSVar.AIR_TEMPERATURE.key,
            ICOSVar.RELATIVE_HUMIDITY.key,
        ]
        self.data = pd.read_csv(self.csv_path, usecols=usecols)

        self.data["TIMESTAMP_START"] = pd.to_datetime(
            self.data["TIMESTAMP_START"], format="%Y%m%d%H%M"
        )

        self.data["TIMESTAMP_END"] = pd.to_datetime(
            self.data["TIMESTAMP_END"], format="%Y%m%d%H%M"
        )

        # Select only valid mesures
        self.data = self.data[
            (self.data[ICOSVar.AIR_TEMPERATURE.key] != -9999)
            & (self.data[ICOSVar.RELATIVE_HUMIDITY.key] != -9999)
        ]

    def date_hour_filter(
        self,
        start_date: dt.date,
        end_date: dt.date,
        hour_start: int = 0,
        hour_end: int = 22,
        hour_step: int = 2,
    ) -> pd.DataFrame:
        """
        Description
        -----------
        Filter ICOS data for a given date range and selected hours

        Parameters
        -----------
        start_date : dt.date
            The start date of the filtering period
        end_date : dt.date
            The end date of the filtering period
        hour_start : int
            The beginning of the selected hour interval (inclusive)
        hour_end : int
            The end of the selected hour interval (inclusive)
        hour_step : int
            The step between hours to select

        Return
        -----------
        filtered_df : pd.DataFrame
            The subset of the ICOS dataset filtered by the given date range
            and hour interval.

        """
        mask_date = (self.data["TIMESTAMP_START"].dt.date >= start_date) & (
            self.data["TIMESTAMP_START"].dt.date <= end_date
        )

        mask_hours = (
            (self.data["TIMESTAMP_START"].dt.hour >= hour_start)
            & (self.data["TIMESTAMP_START"].dt.hour <= hour_end)
            & (self.data["TIMESTAMP_START"].dt.hour % hour_step == 0)
            & (self.data["TIMESTAMP_START"].dt.minute == 0)
        )
        filtered_df = self.data[mask_date & mask_hours]
        return filtered_df

    def get_ta_rh_td(self, filtered_df: pd.DataFrame):
        """
        Description
        -----------
        Extracts air temperature, relative humidity, and computes dew point
        temperature.

        Parameters
        ----------
        filtered_df : pd.DataFrame
            Filtered ICOS DataFrame containing.

        Returns
        -------
        ta : pd.Series
            Air temperature values extracted from the DataFrame.
        rh : pd.Series
            Relative humidity values extracted from the DataFrame.
        td :
            Dew point temperature values computed from 'ta' and 'rh'.
        """
        ta = filtered_df[ICOSVar.AIR_TEMPERATURE.key]
        rh = filtered_df[ICOSVar.RELATIVE_HUMIDITY.key]
        td = ICOSVar.compute_dewpoint_temp(ta, rh)
        return ta, rh, td

    def to_xarray(self, filtered_df: pd.DataFrame) -> xr.Dataset:
        """
        Description
        -----------
        Convert a filtered DataFrame into a structured xarray Dataset.

        Parameters
        -----------
        filtered_df : pd.DataFrame
            Filtered DataFrame

        Returns
        -----------
        xr.Dataset: An xarray Dataset containing:
            - data_vars:
                - "ta": air temperature, dimension ("time",)
                - "rh": relative humidity, dimension ("time",)
                - "td": dew point temperature, dimension ("time",)
            - coords:
                - "time": start timestamps
                - "time_end": end timestamps
                - "elevation": elevation (constant for all entries)
                - "latitude": latitude (constant)
                - "longitude": longitude (constant)
            - attrs:
                - "crs": coordinate reference system
        """
        ta, rh, td = self.get_ta_rh_td(filtered_df)

        n = len(filtered_df)
        return xr.Dataset(
            data_vars={
                "ta": ("time", ta),
                "rh": ("time", rh),
                "td": ("time", td),
            },
            coords={
                "time": filtered_df["TIMESTAMP_START"].values,
                "time_end": ("time", filtered_df["TIMESTAMP_END"].values),
                "elevation": ("time", np.full(n, self.elevation)),
                "latitude": ("time", np.full(n, self.latitude)),
                "longitude": ("time", np.full(n, self.longitude)),
            },
            attrs={"crs": str(self.crs)},
        )

    def _get_utm_crs(self) -> pyproj.CRS:
        """
        Description
        -----------
        Determines the appropriate UTM CRS based on the station's latitude and
        longitude.

        It creates an Area of Interest (AOI) around the point and
        retrieves the corresponding UTM CRS metadata.

        Returns
        -------
        pyproj.CRS
            The UTM crs
        """
        # Query the PROJ database to obtain UTM CRS info that matches the point
        info_utm = pyproj.database.query_utm_crs_info(
            datum_name="WGS 84",
            area_of_interest=pyproj.aoi.AreaOfInterest(
                west_lon_degree=self.longitude,
                south_lat_degree=self.latitude,
                east_lon_degree=self.longitude,
                north_lat_degree=self.latitude,
            ),  # Thearea of interest reduces the CRS selection to exactly the
            # UTM zone covering this location.
        )[0]
        return pyproj.CRS.from_epsg(info_utm.code)

    def create_bbox(
        self, width_m: float, height_m: float
    ) -> rio.coords.BoundingBox:
        """
        Description
        ----------
        Creates a rectangular bounding box around the station's geographic
        location with a specified width and height (in meters), returned UTM
        coordinates and geographic (lat/lon) coordinates.

        Parameters
        ----------
        width_m : float
            width of the bounding box in meters.
        height_m : float
            height of the bounding box in meters.

        Returns
        -------
        bbox_utm : rio.coords.BoundingBox
            Bounding box in UTM coordinates.
        utm_crs : pyproj.CRS
            The UTM CRS used for the transformation.
        bbox_lat_lon : rio.coords.BoundingBox
            Bounding box transformed back to geographic coordinates
            (same CRS as self.crs).
        """
        utm_crs = self._get_utm_crs()

        # For the UTM bounding box
        transformer_to_utm = pyproj.Transformer.from_crs(
            self.crs, utm_crs, always_xy=True
        )
        x, y = transformer_to_utm.transform(self.longitude, self.latitude)

        half_w, half_h = width_m / 2, height_m / 2
        bbox_utm = rio.coords.BoundingBox(
            left=x - half_w, bottom=y - half_h, right=x + half_w, top=y + half_h
        )
        # For the lat/lon bounding box
        transformer_to_lat_lon = pyproj.Transformer.from_crs(
            utm_crs, self.crs, always_xy=True
        )
        lon_min, lat_min = transformer_to_lat_lon.transform(
            bbox_utm.left, bbox_utm.bottom
        )
        lon_max, lat_max = transformer_to_lat_lon.transform(
            bbox_utm.right, bbox_utm.top
        )
        bbox_lat_lon = rio.coords.BoundingBox(
            left=lon_min, bottom=lat_min, right=lon_max, top=lat_max
        )

        return bbox_utm, utm_crs, bbox_lat_lon

    def work_area_from_coord_station(self, w: float, h: float):
        """
        Description
        ----------
        Get the working area around the station with UTM coordinates and
        geographic coordinates.

        Parameters
        ----------
        w : float
            width of the bounding box in meters.
        h : float
            height of the bounding box in meters.

        Returns
        -------
        dict:
            - "utm" : UTM coordinates and CRS
            - "lat/lon" : geographic coordinates and CRS
        """
        bbox_utm, utm_crs, bbox_lat_lon = self.create_bbox(w, h)
        return {
            "utm": (bbox_utm, utm_crs),
            "lat/lon": (bbox_lat_lon, self.crs),
        }


#########################################################
##                                                     ##
##                                                     ##
##      VALIDATION (ERA5, ERA5 DOWNSCALED, ICOS)       ##
##                      Ta and Tdp                     ##
##                                                     ##
##                                                     ##
#########################################################


def get_dem_from_roi(  # Not used with the cluster
    roi_bbox: rio.coords.BoundingBox,
    roi_crs: CRS,
    base_dir: str,
    resolution: float = 60,
) -> xr.Dataset:
    """
    Description
    -----------
    Read several tiles for DEM Copernicus based on a ROI.
    Then, resample them at a specific resolution and
    compute slope et aspect

    Parameters
    ----------
    roi_bbox: roi.coords.BoundingBox
        ROI bounding box
    roi_crs: pyproj.CRS
        ROI CRS
    resolution: str, deflaut=60
        DEM spatial resolution
    base_dir: str
        Path to the DEM directory
        Required to set MNT_PATH environment variable

    Returns
    -------
    xarr: xarray.Dataset
    """
    # Convert to shapely box
    roi_geom = box(roi_bbox.left, roi_bbox.bottom, roi_bbox.right, roi_bbox.top)
    file_names = [
        os.path.join(base_dir, f)
        for f in os.listdir(base_dir)
        if f.endswith(".tif")
    ]
    # Compute transform
    left = roi_bbox.left
    top = roi_bbox.top
    res_x = resolution  # pixel width (e.g., in meters or degrees)
    res_y = resolution  # pixel height (positive, will be negated internally)
    transform = from_origin(left, top, res_x, res_y)
    datasets = []
    for tile in file_names:
        # Reda file
        ds = rioxarray.open_rasterio(tile, masked=True)
        # Reproject tile
        grid = ds.rio.reproject(  # type: ignore
            dst_crs=roi_crs,
            resampling=rio.enums.Resampling.cubic,
            transform=transform,
        )
        # Clip
        grid = grid.rio.clip([roi_geom], roi_crs, drop=True)
        datasets.append(grid)
    elevation = merge_arrays(datasets)
    elevation.name = "height"
    dem = elevation.to_dataset().squeeze(dim="band").drop_vars("band")
    slope, aspect = compute_slope_aspect(dem["height"], resolution)
    dem["slope"] = (("y", "x"), slope)
    dem["aspect"] = (("y", "x"), aspect)
    # Set variable attributes
    dem["height"].attrs.clear()
    dem["height"].attrs["standard_name"] = "height"
    dem["height"].attrs["long_name"] = "height"
    dem["height"].attrs["name"] = "height"
    dem["height"].attrs["unit"] = "m"
    dem["height"].attrs["description"] = "Height"
    dem["slope"].attrs["standard_name"] = "slope"
    dem["slope"].attrs["long_name"] = "slope"
    dem["slope"].attrs["name"] = "slope"
    dem["slope"].attrs["unit"] = "degree"
    dem["slope"].attrs["description"] = "Slope"
    dem["aspect"].attrs["standard_name"] = "aspect"
    dem["aspect"].attrs["long_name"] = "aspect"
    dem["aspect"].attrs["name"] = "aspect"
    dem["aspect"].attrs["unit"] = "degree"
    dem["aspect"].attrs["description"] = "Aspect"
    # Transform to rioxarray
    dem = dem.rio.write_crs(roi_crs)
    bounds = rio.coords.BoundingBox(*dem.rio.bounds())
    logger.debug(f"DEM bbox: {bounds}")
    # Clean rio attributes
    dem = dem.drop_vars("spatial_ref", errors="ignore")
    dem.attrs["crs"] = roi_crs
    dem.attrs["transform"] = transform
    dem.attrs["bounds"] = bounds
    return dem


def remove_dates_from_csv(csv_path: str, dates_to_remove):
    """
    Description
    -----------
    Removes the data (Ta and Tdp of ERA5 and ERA5 resampled) of the given
    dates in the csv of a station.

    Parameters
    ----------
    csv_path: str
        path to the csv of the station
    dates_to_remove : list[str]
        list of dates to removed in the csv
    """
    df = pd.read_csv(csv_path, parse_dates=["time"])

    dates_to_remove = pd.to_datetime(dates_to_remove)

    df = df[~df["time"].isin(dates_to_remove)]

    df.to_csv(csv_path, index=False)
    # print(f"Removed dates: {len(dates_to_remove)}")

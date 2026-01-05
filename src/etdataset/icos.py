import datetime as dt
import os
import zipfile
from dataclasses import dataclass
from enum import Enum
from multiprocessing import Process
from typing import TypedDict

import matplotlib.pyplot as plt
import numpy as np
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
from shapely.geometry import box
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)

from etdataset.dem import compute_slope_aspect
from etdataset.era5 import ERA5Dataset, ERA5Var, add, read
from etdataset.interpolation import create_grid_dataset
from etdataset.logging import LoggerManager

logger = LoggerManager.get_logger(__name__)

G_CST = 9.80665

#########################################################
##                                                     ##
##                                                     ##
##                    ICOS STATIONS                    ##
##                                                     ##
##                                                     ##
#########################################################


class StationConfig(TypedDict):
    name: str
    id: str
    lat: float
    lon: float
    elevation: float
    crs: CRS
    country_code: str


def get_csv_with_valid_icos_stations():
    """
    Description
    ----------
    Fetch all ICOS ecosystem stations (ES) that have available Meteo L2 dataset,
    extract their metadata, save the list into a CSV file.
    Run it only once (just to get the csv file)

    Returns
    -------
    csv_path : str
        Path to the CSV file containing stations informations.
    """
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
    for _, s in enumerate(ecosystem_stations, 1):
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
    csv_path = "stations_meteoL2.csv"
    df.to_csv(csv_path, index=False)
    logger.info(
        f"All valid stations and their metadata are stored in the csv file\
          : {csv_path}"
    )
    return csv_path


def filter_stations_by_country_code(csv_path: str, country_code: str):
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
    data = pd.read_csv(csv_path)
    data_filtered = data[data["country"] == country_code]
    return data_filtered


def load_stations_config(csv_path: str) -> dict[str, StationConfig]:
    """
    Description
    ----------
    Load station configurations from a CSV file.

    For each row, this function creates a StationConfig entry with:
        - id of the station
        - geographic coordinates
        - elevation
        - CRS fixed to EPSG:4326 (WGS84)

    Parameters
    ----------
    csv_path : str
        Path to the CSV file containing stations informations.

    Returns
    -------
    starions_cfg : dict[str, StationConfig]
        A dictionary mapping station IDs to their corresponding StationConfig.
    """
    df = pd.read_csv(csv_path)
    stations_cfg: dict[str, StationConfig] = {}
    for _, row in df.iterrows():
        stations_cfg[row["id"]] = StationConfig(
            id=row["id"],
            name=row["name"],
            lat=row["lat"],
            lon=row["lon"],
            elevation=row["elev"],
            crs=CRS("4326"),
            country_code=row["country"],
        )
    return stations_cfg


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
    filename = obj.filename

    # Do not download if the file already exists
    if os.path.exists(os.path.join(path, filename)):
        logger.info(
            f"File {os.path.join(path, filename)} already exists. Skip download"
        )
        return os.path.join(path, filename)

    # get the url for downloading
    hash_id = obj.uri.split("/")[-1]
    url = f"https://data.icos-cp.eu/objects/{hash_id}"
    logger.info(f"Download url =  {url}")

    response = requests.get(url, cookies=cookies)

    if response.status_code == 200:
        with open(os.path.join(path, filename), "wb") as file:
            file.write(response.content)
        logger.debug(
            f"File of the station {id_station} :"
            f"{filename} downloaded successfully"
        )
        return os.path.join(path, filename)
    logger.error(
        f"Failed to download file of the station {id_station}: {filename} withthe status code : {response.status_code}"  # noqa: E501
    )
    return None


def download_file(
    data_objects, cookies: dict, station_id: str, path: str | None = None
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
    if path is None:
        path = os.getcwd()
    download_folder = os.path.join(path, "ICOS")
    os.makedirs(download_folder, exist_ok=True)

    for obj in data_objects:
        downloaded_file = _download_file(
            obj, download_folder, station_id, cookies
        )
        if downloaded_file is None:
            continue
        # Unzip the downloaded file
        if downloaded_file.lower().endswith(".zip"):
            extract_folder = os.path.join(
                download_folder, os.path.splitext(obj.filename)[0]
            )
            if not os.path.exists(extract_folder) or not os.listdir(
                extract_folder
            ):
                os.makedirs(extract_folder, exist_ok=True)
                try:
                    with zipfile.ZipFile(downloaded_file, "r") as zip_ref:
                        zip_ref.extractall(extract_folder)
                    logger.info(
                        f"{station_id} station file unzipped in : {extract_folder}"  # noqa: E501
                    )
                except zipfile.BadZipFile:
                    logger.exception(f"Non valid zip file: {downloaded_file}")
            else:
                logger.info(
                    f"{station_id} station file already extracted : {extract_folder}"  # noqa: E501
                )


def download_icos_station(auth_token: str, ids: list):
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
    if auth_token is None:
        raise ValueError("Authentification token is not provided")

    cookies = {"cpauthToken": auth_token}
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
        download_file(data_objects, cookies, station_id)
    logger.info(f"All the stations are downloaded {ids}")


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
    ) -> np.ndarray:
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
        ta : np.ndarray
            Air temperature from ICOS
        rh : np.ndarray
            Relative humidity from ICOS

        Return
        -----------
        tp : np.ndarray
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


def kelvin_to_celsius(kelvin: float | np.ndarray) -> float | np.ndarray:
    """
    Description
    -----------
    Compute the temperature in celsius from a temperature in kelvin

    """
    return kelvin - 273.15


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
        station_id: str,
        stations_cfg: dict,
    ):
        if station_id not in stations_cfg:
            raise KeyError(f"Station '{station_id}' not find.")

        cfg = stations_cfg[station_id]
        self.name = station_id
        self.csv_path = f"ICOS/ICOSETC_{station_id}_METEO_L2/ICOSETC_{station_id}_METEO_L2.csv"  # noqa: E501
        self.latitude = cfg["lat"]
        self.longitude = cfg["lon"]
        self.elevation = cfg["elevation"]
        self.crs = CRS(cfg["crs"])
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
##                  VALIDATION (ERA5)                  ##
##                      lapse_rate                     ##
##                                                     ##
##                                                     ##
#########################################################


def _compute_lapse_rate(records: list) -> float:
    """
    Description
    -----------
    Computes the vertical temperature lapse rate by selecting the two
    atmospheric levels closest to the station's altitude.

    Parameters
    ----------
    records : list
        List of atmospheric records, where each element is a dictionary
        containing at least:
        - 'z'  : altitude of the level (m)
        - 'T'  : temperature at the level (K)
        - 'dz' : absolute difference between the level altitude and the
                station altitude (m)
    Returns
    -------
    lr: float
         Computed vertical temperature lapse rate (K/m).
    """
    df = pd.DataFrame(records).sort_values("z")

    # Sort by altitude difference (dz) to identify
    # the levels closest to the reference altitude
    df_sorted = df.sort_values("dz")

    # Select the level closest to the reference altitude
    df_ref = df_sorted.iloc[0]
    z_ref = df_ref["z"]  # reference altitude
    t_ref = df_ref["T"]  # temperature at reference altitude

    # Select the second closest level
    df_second = df_sorted.iloc[1]
    z_second = df_second["z"]  # second level altitude
    t_second = df_second["T"]

    # Compute the vertical temperature gradient (lapse rate)
    lr = (t_ref - t_second) / (z_ref - z_second)
    return lr


def compute_lapse_rate(
    station: ICOSStation,
    output: str,
    output_csv: str,
    start_date: dt.date,
    end_date: dt.date,
):
    """
    Description
    -----------
    Compute daily vertical lapse rates for air temperature and dew point
    temperature at a given ICOS station using ERA5 pressure-level data.
    e.g : "Elevation Correction of ERA5 Reanalysis Temperature over the
    Qilian Mountains of China", Peng Zhao and Lihui Qian.

    For each day in the specified period, temperature, relative humidity and
    geopotential are extracted at multiple pressure levels, interpolated to the
    station location. Geopotential are converted to altitude.

    The lapse rate is then computed using the two atmospheric levels closest
    to the station elevation.

    Parameters
    -----------
    station : ICOSStation
        ICOS station object containing latitude, longitude, elevation,
        and station name.
    output : str
        Path to the directory containing downloaded ERA5 pressure-level files.
    output_csv : str
        Path to the directory where the output CSV file will be written.
    start_date : dt.datetime
        Start date (inclusive).
    end_date : dt.datetime
        End date (inclusive).

    Returns
    --------
    Results are written to a CSV file containing daily lapse rates for
    air temperature and dew point temperature.
    """

    # Get the station's data
    lat_sta = station.latitude
    lon_sta = station.longitude
    z_sta = station.elevation

    # Loop over the requested time period
    cur_date = start_date
    lapse_records = []
    while cur_date <= end_date:
        filename = f"download_era5_pressure_{cur_date.isoformat()}.zip"
        logger.info(
            f"Current file : download_era5_pressure_{cur_date.isoformat()}.zip "
        )
        logger.info(f"elevation station : {z_sta}")
        filepath = os.path.join(output, "ERA5_data", filename)
        era5_xrds = read(product=filepath)
        records_ta = []
        records_tdp = []
        # Pressure levels used to estimate the vertical gradient
        pressure_levels = [800, 825, 850, 875, 900, 925, 950, 975, 1000]
        for p in pressure_levels:
            logger.info(f"pressure level :{p}")
            # Interpolate geopotential height at station location
            height = (
                era5_xrds["z"]
                .sel(
                    time=f"{cur_date.isoformat()}T12:00",
                    pressure_level=p,
                )
                .interp(
                    latitude=lat_sta,
                    longitude=lon_sta,
                    method="linear",
                )
                .item()
            )
            # Interpolate air temperature at station location
            air_temp = (
                era5_xrds["t"]
                .sel(time=f"{cur_date.isoformat()}T12:00", pressure_level=p)
                .interp(
                    latitude=lat_sta,
                    longitude=lon_sta,
                    method="linear",
                )
                .item()
            )
            # Interpolate relative humidity at station location
            rel_hum = (
                era5_xrds["r"]
                .sel(time=f"{cur_date.isoformat()}T12:00", pressure_level=p)
                .interp(latitude=lat_sta, longitude=lon_sta)
                .item()
            )
            # Compute dew point temperature
            dp_temp = ICOSVar.compute_dewpoint_temp(
                ta=pd.Series([air_temp]),
                rh=pd.Series([rel_hum]),
            )[0]

            z = height / G_CST  # Convert geopotential to geometric height
            records_ta.append({"z": z, "T": air_temp, "dz": abs(z - z_sta)})
            records_tdp.append({"z": z, "T": dp_temp, "dz": abs(z - z_sta)})

        # Compute lapse rates using the two levels closest to station height
        lapse_rate_ta = _compute_lapse_rate(records_ta)
        lapse_rate_tdp = _compute_lapse_rate(records_tdp)

        lapse_records.append(
            {
                "time": cur_date,
                "station": station.name,
                "lapse_rate_ta": lapse_rate_ta,
                "lapse_rate_tdp": lapse_rate_tdp,
            }
        )
        cur_date += dt.timedelta(days=1)

    os.makedirs(output_csv, exist_ok=True)
    out_path = os.path.join(output_csv, f"{station.name}_lapse_rate.csv")
    df = pd.DataFrame(lapse_records)
    df.to_csv(out_path)


def run_station_process_lr(
    station_id: str,
    cfg: dict[str, StationConfig],
    output: str,
    start_date: dt.date,
    end_date: dt.date,
    out_csv: str,
):
    if station_id not in cfg:
        logger.warning(
            f" Station {station_id} not found in stations_dict. skipping."
        )
    # Initialize station object
    station = ICOSStation(station_id, cfg)
    logger.info(f"\n Processing station: {station.name} ({station_id})")
    try:
        compute_lapse_rate(
            station=station,
            output=output,
            output_csv=out_csv,
            start_date=start_date,
            end_date=end_date,
        )
        logger.info(f"{station_id} finished")
    except Exception:
        logger.exception("Error")


def generate_lapse_rate_for_stations_multiprocess(
    station_ids,
    cfg,
    output,
    start_date,
    end_date,
    out_csv,
):
    """
    Description
    ----------
    Launch the generation of time series (ICOS, ERA5, downscaled ERA5)
    for stations in parallel using multiprocessing.

    Each station is processed in a separate subprocess, calling the function
    'run_station_process_lr'

    Parameters
    ----------
    station_ids: list
        List of station identifiers
    cfg: dict
        Configuration station
    output: str
        Path to the ERA5 directory
    start_date: dt.date
        Start date
    end_date: dt.date
        End date
    out_csv: str
        Directory where the station time series CSV files will be written.
    """
    procs = []
    # Spawn one process per station
    for sid in station_ids:
        p = Process(
            target=run_station_process_lr,
            args=(
                sid,
                cfg,
                output,
                start_date,
                end_date,
                out_csv,
            ),
        )
        procs.append(p)
        p.start()
    # Block until all station processes complete
    for p in procs:
        p.join()


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


def process_era5_point(
    date: dt.datetime,
    station: ICOSStation,
    lat: float,
    lon: float,
    output,
    grid_res=0.25,
    resampled=False,
    dem_dir="rasters_COP30/DE-Geb",
    dem_res=60,
) -> xr.Dataset:
    """
    Description
    -----------
    Extract temperature (Ta) and dewpoint temperature (Td) from ERA5 or
    resampled ERA5 data at the location of an ICOS station for a given date.


    -> When 'resampled=True', the function builds a high-resolution DEM around
    the station, adds ERA5 variables at this resolution, and extracts the
    corresponding point values.

    -> When 'resampled=False', ERA5 data are extracted directly from a coarse
    grid defined around the station coordinates.

    Parameters
    -----------
    date: dt.datetime
        The date for which ERA5 data are extracted.
    station : ICOSStation
        ICOS station object containing metadata and coordinate utilities.
    lat: float
        Latitude of the point where ERA5 data will be extracted
        (used when 'resampled=False').
    lon: float
        Longitude of the point where ERA5 data will be extracted
        (used when 'resampled=False').
    output:
        Path where ERA5 data are.
    grid_res: float
        Resolution (in degrees) of the regular grid used when
        'resampled=False'. Default is 0.25°.
    resampled: bool
        'True': to use high-resolution resampled ERA5 data combined with a DEM.
        'False': ERA5 data are taken directly from a coarse grid.
    dem_dir: str
        Directory containing DEM tiles used for the high-resolution resampling.
    dem_res: int
        DEM resolution (in meters) used when 'resampled=True'.

    Returns
    -------
    xr.Dataset
        A dataset containing Ta and Td (in °C) at the point corresponding to
        the ICOS station and the given date.

    """
    # Get Ta and Td from the resampled ERA5 data
    if resampled:
        # Get the ROI bbox (UTM) around the ICOS station (10km x 10km)
        roi_bbox_utm, roi_crs_utm = station.work_area_from_coord_station(
            10000, 10000
        )["utm"]
        # Get the DEM from ROI
        dem = get_dem_from_roi(roi_bbox_utm, roi_crs_utm, dem_dir, dem_res)
        # Add attributes
        add_time_attrs(dem, date)
        updated = add(
            dem,
            dataset=ERA5Dataset.ERA5,
            variables=[
                ERA5Var.TEMPERATURE,
                ERA5Var.DEWPOINT_TEMPERATURE,
            ],
            path=output,
        )

        # get the coords (x,y) of the station in UTM
        x, y = (
            station.work_area_from_coord_station(0, 0)["utm"][0].left,
            station.work_area_from_coord_station(0, 0)["utm"][0].bottom,
        )
        z = (
            dem["height"]
            .interp({"x": x, "y": y}, method="linear")
            .values.item()
        )
        diff = station.elevation - z

        logger.info(f"Elevation DEM : {z:.2f} m")
        logger.info(
            f"Elevation station : {station.name} et {station.elevation:.2f} m"
        )
        logger.info(f"Difference : {diff:.2f} m")
        # assert (
        #    np.abs(diff) <= 20.0
        # ), f"""Difference of elevation between the DEM and the metadata of the
        # station is too important: {np.abs(diff):.2f} m > 20 m"""

        # get Ta and Td by bilinear interpolation and convert them from kelvin
        # to celsius in the resampled ERA5 data
        ta, td = (
            kelvin_to_celsius(
                updated["ta"].interp({"x": x, "y": y}, method="linear").values
            ),
            kelvin_to_celsius(
                updated["tdp"].interp({"x": x, "y": y}, method="linear").values
            ),
        )

        # print(f"ta={ta} et tdp={td}")
    # Get Ta and Td from the ERA5 data
    else:
        # Get the bounds (latitude/longitude) around the ICOS station
        # (50km x 50km)
        bounds = station.work_area_from_coord_station(50000, 50000)["lat/lon"]
        grid = create_grid_dataset(bounds[0], bounds[1], grid_res)
        grid["height"] = grid["grid"].copy()
        # Add attributes
        add_time_attrs(grid, date)
        updated = add(
            data=grid,
            dataset=ERA5Dataset.ERA5,
            variables=[
                ERA5Var.TEMPERATURE,
                ERA5Var.DEWPOINT_TEMPERATURE,
            ],
            path=output,
        )

        # get Ta and Td by bilinear interpolation and convert them from kelvin
        # to celsius in the ERA5 data
        ta, td = (
            kelvin_to_celsius(
                updated["ta"]
                .interp({"x": lon, "y": lat}, method="linear")
                .values
            ),
            kelvin_to_celsius(
                updated["tdp"]
                .interp({"x": lon, "y": lat}, method="linear")
                .values
            ),
        )
    return create_xr_point_dataset(date, ta, td)


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


def generate_timeseries_era5(
    station: ICOSStation,
    output: str,
    start_date: dt.date,
    end_date: dt.date,
    hour_start=0,
    hour_end=22,
    hour_step=2,
    day_step=1,
):
    """
    Description
    -----------
    Generate a time series of temperature (Ta) and dewpoint temperature (Td)
    from ERA5 and resampled ERA5 data for a given ICOS station over a specified
    date range. For each day and for each selected hour, the function calls
    'process_era5_point()' to extract point-level ERA5 data.

    Parameters
    -----------
    station : ICOSStation
        The ICOS station
    output : str
        Path where ERA5 input files are stored
    start_date : dt.date
        First date of the time series
    end_date : dt.date
        Last date of the time series
    hour_start : int
        First hour of the day to include in the time series
    hour_end : int
        Last hour of the day to include
    hour_step : int
        Time step between two extracted points
    day_step : int, optional
        Step between two processed days

    Returns
    -------
    - 'era5' : xr.Dataset
        Coarse grid ERA5 temperature and dewpoint temperature in °C for all
        selected timestamps.

    - 'era5_resampled' : xr.Dataset
        High-resolution DEM-based resampled ERA5 temperature and dewpoint
        temperature in °C for the same timestamps.


    """
    total_ds = []
    total_ds_proj = []

    cur_date = start_date
    while cur_date <= end_date:
        filename = f"download_era5_{cur_date.isoformat()}.zip"
        filepath = os.path.join(output, "ERA5_data", filename)

        read(product=filepath)
        times = [
            dt.datetime.combine(cur_date, dt.time(hour=h))
            for h in range(hour_start, hour_end + 1, hour_step)
        ]

        ds_list = []
        ds_proj_list = []
        for t in times:
            #  ERA5 data are extracted directly from a coarse
            # grid defined around the station coordinates.

            ds_list.append(
                process_era5_point(
                    t,
                    station,
                    station.latitude,
                    station.longitude,
                    output,
                    resampled=False,
                    dem_dir=f"rasters_COP30/{station.name}",
                )
            )
            # ERA5 data are extracted after resampling ERA5 variables at the
            # resolution DEM around the station
            ds_proj_list.append(
                process_era5_point(
                    t,
                    station,
                    station.latitude,
                    station.longitude,
                    output,
                    resampled=True,
                    dem_dir=f"rasters_COP30/{station.name}",
                )
            )
        day_ds = xr.concat(ds_list, dim="time")
        day_ds_proj = xr.concat(ds_proj_list, dim="time")
        total_ds.append(day_ds)
        total_ds_proj.append(day_ds_proj)

        cur_date += dt.timedelta(days=day_step)

    era5_ = xr.concat(total_ds, dim="time")
    era5_proj = xr.concat(total_ds_proj, dim="time")
    return era5_, era5_proj


def generate_timeseries_with_csv(
    station: ICOSStation,
    output: str,
    start_date: dt.date,
    end_date: dt.date,
    out_csv: str,
    hour_start=0,
    hour_end=22,
    hour_step=2,
    day_step=1,
    *,
    skip_existing: bool = True,
):
    """
    Description
    ----------
    Generate ICOS, ERA5, and projected ERA5 temperature/dewpoint time series and
    store them in a CSV file.

    This function builds a date-time grid, extracts:
      - ICOS in-situ air temperature and dew point,
      - ERA5 reanalysis variables (air temperature, dew point),
      - Projected ERA5 variables after elevation correction,
    and appends them to a CSV file for the station.

    Parameters
    ----------
    station : ICOSStation
        ICOS station
    output : str
        Path to the ERA5 output directory
    start_date : dt.date
        Start date (included).
    end_date : dt.date
        End date (included).
    out_csv : str
        Directory where the timeseries CSV will be written.
    hour_start : int
        First hour of each day to include.
    hour_end : int
        Last hour of each day to include
    hour_step : int,
        Hour interval between processed timestamps
    day_step : int
        Interval between processed days
    skip_existing : bool
        If True (default), timestamps already present in the CSV are not
        recomputed.

    Returns
    -------
    str
        Full path to the generated CSV file.
    """
    os.makedirs(out_csv, exist_ok=True)
    csv_path = os.path.join(out_csv, f"{station.name}_timeseries.csv")
    # Load or initialize the CSV
    if skip_existing and os.path.exists(csv_path):
        df = pd.read_csv(csv_path, parse_dates=["time"])

    else:
        df = pd.DataFrame(
            columns=[
                "time",
                "Ta_icos",
                "Tdp_icos",
                "Ta_era5",
                "Tdp_era5",
                "Ta_era5_proj",
                "Tdp_era5_proj",
            ]
        )
        df.to_csv(csv_path, index=False)
    # Load ERA5 datasets and resampled ERA5
    era5_ds, era5_proj_ds = generate_timeseries_era5(
        station,
        output,
        start_date,
        end_date,
        hour_start,
        hour_end,
        hour_step,
        day_step,
    )
    # Filter ICOS dataset to the required time grid
    filtered_df = station.date_hour_filter(
        start_date, end_date, hour_start, hour_end, hour_step
    )

    for t in pd.date_range(start=start_date, end=end_date, freq=f"{day_step}D"):
        for h in range(hour_start, hour_end + 1, hour_step):
            dt_obj = dt.datetime.combine(t.date(), dt.time(hour=h))
            if skip_existing and (df["time"] == dt_obj).any():
                continue
            # ICOS extraction
            ta_icos, _, td_icos = station.get_ta_rh_td(
                filtered_df[filtered_df["TIMESTAMP_START"] == dt_obj]
            )
            ta_icos = float(ta_icos.values[0]) if not ta_icos.empty else np.nan
            td_icos = float(td_icos.values[0]) if not td_icos.empty else np.nan
            # ERA5 extraction
            try:
                ta_era5 = era5_ds.sel(time=dt_obj)["ta"].values.item()
                td_era5 = era5_ds.sel(time=dt_obj)["td"].values.item()
                ta_era5_proj = era5_proj_ds.sel(time=dt_obj)["ta"].values.item()
                td_era5_proj = era5_proj_ds.sel(time=dt_obj)["td"].values.item()
            except KeyError:
                ta_era5 = td_era5 = ta_era5_proj = td_era5_proj = np.nan

            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        {
                            "time": [dt_obj],
                            "Ta_icos": [ta_icos],
                            "Tdp_icos": [td_icos],
                            "Ta_era5": [ta_era5],
                            "Tdp_era5": [td_era5],
                            "Ta_era5_proj": [ta_era5_proj],
                            "Tdp_era5_proj": [td_era5_proj],
                        }
                    ),
                ],
                ignore_index=True,
            )
        df = df.dropna(subset=["Ta_icos", "Tdp_icos"])
        df.to_csv(csv_path, index=False)

    return csv_path


def run_station_process(
    station_id: str,
    cfg: dict[str, StationConfig],
    output: str,
    start_date: dt.date,
    end_date: dt.date,
    out_csv: str,
    skip_existing=True,
    hour_start=0,
    hour_end=22,
    hour_step=2,
    day_step=1,
):
    if station_id not in cfg:
        logger.warning(
            f" Station {station_id} not found in stations_dict. skipping."
        )
    # Initialize station object
    station = ICOSStation(station_id, cfg)
    logger.info(f"\n Processing station: {station.name} ({station_id})")
    try:
        generate_timeseries_with_csv(
            station=station,
            output=output,
            start_date=start_date,
            end_date=end_date,
            out_csv=out_csv,
            skip_existing=skip_existing,
            hour_start=hour_start,
            hour_end=hour_end,
            hour_step=hour_step,
            day_step=day_step,
        )
        logger.info(f"{station_id} finished")
    except Exception:
        logger.exception("Error")


def generate_timeseries_for_stations_multiprocess(
    station_ids,
    cfg,
    output,
    start_date,
    end_date,
    out_csv,
    skip_existing=True,
    hour_start=0,
    hour_end=22,
    hour_step=2,
    day_step=1,
):
    """
    Description
    ----------
    Launch the generation of time series (ICOS, ERA5, projected ERA5)
    for stations in parallel using multiprocessing.

    Each station is processed in a separate subprocess, calling the function
    'run_station_process'

    Parameters
    ----------
    station_ids : list
        List of station identifiers
    cfg : dict
        Configuration station
    output : str
        Path to the ERA5 directory
    start_date : dt.date
        Start date
    end_date : dt.date
        End date
    out_csv : str
        Directory where the station time series CSV files will be written.
    skip_existing : bool
        If True (default), timestamps already written in CSV are not
        recomputed.
    hour_start : int
        First hour
    hour_end : int
        Last hour
    hour_step : int
        Hour interval between processed timestamps
    day_step : int
        Interval between processed days
    """
    procs = []
    # Spawn one process per station
    for sid in station_ids:
        p = Process(
            target=run_station_process,
            args=(
                sid,
                cfg,
                output,
                start_date,
                end_date,
                out_csv,
                skip_existing,
                hour_start,
                hour_end,
                hour_step,
                day_step,
            ),
        )
        procs.append(p)
        p.start()
    # Block until all station processes complete
    for p in procs:
        p.join()


def generate_timeseries_for_stations(
    station_ids: list[str],
    cfg: dict[str, StationConfig],
    output: str,
    start_date: dt.date,
    end_date: dt.date,
    out_csv: str,
    *,
    skip_existing: bool = True,
    hour_start=0,
    hour_end=22,
    hour_step=2,
    day_step=1,
):
    """
    Description
    ----------
    Loop on the list of stations to generate for each ICOS, ERA5, and projected
    ERA5 temperature/dewpoint time series and CSV file.

    Parameters
    ----------
    station : list[str]
        ICOS stations list
    cfg : dict[str, StationConfig]
        dictionnary with metadata of the stations
    output : str
        Path to the ERA5 directory
    start_date : dt.date
        Start date
    end_date : dt.date
        End date
    out_csv : str
        Directory where the timeseries CSV will be written.
    hour_start : int
        First hour of each day to include.
    hour_end : int, optional
        Last hour of each day to include
    hour_step : int,
        Hour interval between processed timestamps
    day_step : int
        Interval between processed days
    skip_existing : bool
        If True (default), timestamps already present in the CSV are not
        recomputed.

    """
    results = {}

    for sid in station_ids:
        if sid not in cfg:
            logger.warning(
                f" Station {sid} not found in stations_dict. skipping."
            )
            continue
        # Initialize station object
        station = ICOSStation(sid, cfg)
        logger.info(f"\n Processing station: {station.name} ({sid})")

        csv_out = generate_timeseries_with_csv(
            station=station,
            output=output,
            start_date=start_date,
            end_date=end_date,
            out_csv=out_csv,
            hour_start=hour_start,
            hour_end=hour_end,
            hour_step=hour_step,
            day_step=day_step,
            skip_existing=skip_existing,
        )
        results[sid] = csv_out

    return results


#########################################################
##                                                     ##
##                                                     ##
##                    PLOTS                            ##
##                                                     ##
##                                                     ##
#########################################################


def _plot_variable_subplots(stations: dict, var: str, color: dict):
    n = len(stations)
    figsize = (20 * n, 8)

    fig, axes = plt.subplots(1, n, figsize=figsize, sharex=False)
    if n == 1:
        axes = [axes]

    for ax, (station_name, df_it) in zip(axes, stations.items(), strict=True):
        df = df_it.sort_values("time")

        ax.plot(
            df["time"],
            df[f"{var}_icos"],
            label=f"{var} ICOS",
            color=color["icos"],
            linestyle="-",
            marker="o",
        )
        ax.plot(
            df["time"],
            df[f"{var}_era5"],
            label=f"{var} ERA5",
            color=color["era5"],
            linestyle="--",
            marker="s",
        )
        ax.plot(
            df["time"],
            df[f"{var}_era5_proj"],
            label=f"{var} ERA5 proj",
            color=color["era5_sampled"],
            linestyle=":",
            marker="^",
        )

        ax.set_title(f"Station {station_name} — {var}", fontsize=13)
        ax.set_ylabel(f"{var} (°C)")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(fontsize=10)

    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def plot_ta_tdp_csv(stations: dict, start: str | None, end: str | None):
    start_date = pd.to_datetime(start) if start else None
    end_date = pd.to_datetime(end) if end else None

    filtered_stations = {}

    for name, station_df in stations.items():
        if start:
            mask_start = station_df["time"] >= start_date

        if end:
            mask_end = station_df["time"] <= end_date

        filtered_stations[name] = station_df[mask_start & mask_end]

    color_ta = {"icos": "#1f77b4", "era5": "#32b332", "era5_sampled": "#9543bb"}
    color_tdp = {
        "icos": "#ff9137",
        "era5": "#d62d10",
        "era5_sampled": "#a76223",
    }

    _plot_variable_subplots(filtered_stations, "Ta", color_ta)
    _plot_variable_subplots(filtered_stations, "Tdp", color_tdp)

    plt.show()


def _plot_lr_subplots(
    stations: dict,
    var: str,
    color: dict,
    mean_theoretical: float,
):
    n = len(stations)
    figsize = (20 * n, 8)

    fig, axes = plt.subplots(1, n, figsize=figsize, sharex=False)
    if n == 1:
        axes = [axes]

    for ax, (station_name, df_it) in zip(axes, stations.items(), strict=True):
        df = df_it.sort_values("time")

        ax.plot(
            df["time"],
            df[f"{var}"],
            label=f"{var} ERA5",
            color=color["era5"],
            linestyle="-",
            marker="o",
        )

        ax.axhline(
            y=mean_theoretical,
            color=color.get("mean", "black"),
            linestyle="--",
            linewidth=2,
            label="Theoretical value",
        )

        ax.set_title(f"Station {station_name} — {var}", fontsize=13)
        ax.set_ylabel(f"{var} (K/km)")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(fontsize=10)

    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def plot_lr_csv(stations: dict, start: str | None, end: str | None):
    start_date = pd.to_datetime(start) if start else None
    end_date = pd.to_datetime(end) if end else None

    filtered_stations = {}

    for name, station_df in stations.items():
        if start:
            mask_start = station_df["time"] >= start_date

        if end:
            mask_end = station_df["time"] <= end_date

        filtered_stations[name] = station_df[mask_start & mask_end]

    color_ta = {"era5": "#32b332"}
    color_tdp = {"era5": "#9543bb"}

    _plot_lr_subplots(filtered_stations, "lapse_rate_ta", color_ta, -0.0065)
    _plot_lr_subplots(filtered_stations, "lapse_rate_tdp", color_tdp, -0.0065)

    plt.show()


#########################################################
##                                                     ##
##                                                     ##
##                    METRICS                          ##
##                                                     ##
##                                                     ##
#########################################################


def slope_forced_origin(x, y):
    """
    Description
    -----------
    Compute the slope of a linear regression forced through the origin.

    This regression assumes a model of the form:
        y = a * x
    with no intercept term. The slope is estimated using a least-squares
    approach:

        a = Σ(x_i * y_i) / Σ(x_i^2)

    Any pair (x_i, y_i) containing NaN values is removed before computation.
    If Σ(x_i²) = 0, NaN is returned.

    Parameters
    ----------
    x : array-like
        Predictor values (ERA5 values).
    y : array-like
        observed values (ICOS values).

    Returns
    -------
    float
        Estimated slope 'a' of the regression forced through zero.
    """
    mask = ~np.isnan(x) & ~np.isnan(y)
    x, y = x[mask], y[mask]
    return np.sum(x * y) / np.sum(x**2) if np.sum(x**2) != 0 else np.nan


def plot_forced_origin(x, y, slope, title):
    plt.scatter(x, y, alpha=0.6)
    plt.plot([0, max(x)], [0, slope * max(x)], "r", lw=2)
    plt.xlabel("ICOS temperature (°C)")
    plt.ylabel("ERA5 temperature (°C)")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def compute_mean_bias_error(x, y):
    """
    Description
    -----------
    Compute the Mean Bias Error (MBE) between two datasets.

    The Mean Bias Error quantifies the average difference between
    estimated ERA5 values (x) and observed ICOS values (y). A positive MBE
    indicates that x underestimates y on average (bias toward lower values),
    whereas a negative MBE indicates overestimation.

    The MBE is computed as:

        MBE = (1 / n) * Σ (y_i - x_i)

    where:
    - x_i are estimated values,
    - y_i are observed values,
    - n is the number of valid (non-NaN) observations.

    Parameters
    ----------
    x : array-like
        estimated values.
    y : array-like
        observed values.

    Returns
    -------
    float
        Mean Bias Error (MBE)
    """
    return np.nanmean(y - x)


def compute_monthly(csv_path: str, metric: str, *, plot: bool = False):
    """
    Description
    -----------
    Compute monthly performance metrics between reference data (ICOS) and ERA5
    reanalysis for temperature (Ta) and dew point (Tdp).

    Parameters
    ----------
    csv_path : str
        Path to a CSV file containing the dataset
    metric : str
        Performance metric to compute (the same as used in OpenET:https://etdata.org/accuracy/#metrics):
         - sklearn.metrics.root_mean_squared_error : https://scikit-learn.org/stable/modules/generated/sklearn.metrics.root_mean_squared_error.html#sklearn.metrics.root_mean_squared_error
         - sklearn.metrics.r2_score : https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html#sklearn.metrics.r2_score
         - sklearn.metrics.mean_absolute_error : https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html
    Returns
    -------
    pd.DataFrame
        A dataframe with one row per month including:
        - month : YYYY-MM format
        - Ta_{metric} : metric between Ta_icos and Ta_era5
        - Ta_{metric}_proj : metric between Ta_icos and Ta_era5_proj
        - Tdp_{metric} : metric between Tdp_icos and Tdp_era5
        - Tdp_{metric}_proj : metric between Tdp_icos and Tdp_era5_proj
        - n_points : number of valid samples used for that month
    """
    df = pd.read_csv(csv_path, parse_dates=["time"])

    metrics = {
        "rmse": root_mean_squared_error,
        "r2": r2_score,
        "mae": mean_absolute_error,
        "mbe": compute_mean_bias_error,
        "slope": slope_forced_origin,
    }

    if metric not in metrics:
        raise ValueError(f"Metric must be one of: {list(metrics.keys())}")

    f = metrics[metric]

    df["month"] = df["time"].dt.to_period("M")
    per_month = []
    for month, group in df.groupby("month"):
        group_valid = group.dropna(subset=["Ta_icos", "Tdp_icos"])
        if group_valid.empty:
            per_month.append(
                {
                    "month": str(month),
                    f"Ta_{metric}": np.nan,
                    f"Ta_{metric}_proj": np.nan,
                    f"Tdp_{metric}": np.nan,
                    f"Tdp_{metric}_proj": np.nan,
                }
            )
            continue
        per_month.append(
            {
                "month": str(month),
                f"Ta_{metric}": f(
                    group_valid["Ta_icos"], group_valid["Ta_era5"]
                ),
                f"Ta_{metric}_proj": f(
                    group_valid["Ta_icos"], group_valid["Ta_era5_proj"]
                ),
                f"Tdp_{metric}": f(
                    group_valid["Tdp_icos"], group_valid["Tdp_era5"]
                ),
                f"Tdp_{metric}_proj": f(
                    group_valid["Tdp_icos"], group_valid["Tdp_era5_proj"]
                ),
                "n_points": len(group_valid),
            }
        )
        if plot and metric == "slope":
            plot_forced_origin(
                group_valid["Ta_icos"],
                group_valid["Ta_era5"],
                per_month[-1][f"Ta_{metric}"],
                f"Ta - ERA5 - {month}",
            )
            plot_forced_origin(
                group_valid["Ta_icos"],
                group_valid["Ta_era5_proj"],
                per_month[-1][f"Ta_{metric}_proj"],
                f"Ta - ERA5_resampled - {month}",
            )
            plot_forced_origin(
                group_valid["Tdp_icos"],
                group_valid["Tdp_era5"],
                per_month[-1][f"Tdp_{metric}"],
                f"Tdp - ERA5 - {month}",
            )
            plot_forced_origin(
                group_valid["Tdp_icos"],
                group_valid["Tdp_era5_proj"],
                per_month[-1][f"Tdp_{metric}_proj"],
                f"Tdp - ERA5_resampled - {month}",
            )
    return pd.DataFrame(per_month)

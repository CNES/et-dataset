import datetime as dt
import os
import zipfile
from dataclasses import dataclass
from enum import Enum
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
                    group_valid["Ta_era5"], group_valid["Ta_icos"]
                ),
                f"Ta_{metric}_proj": f(
                    group_valid["Ta_era5_proj"], group_valid["Ta_icos"]
                ),
                f"Tdp_{metric}": f(
                    group_valid["Tdp_era5"], group_valid["Tdp_icos"]
                ),
                f"Tdp_{metric}_proj": f(
                    group_valid["Tdp_era5_proj"], group_valid["Tdp_icos"]
                ),
                "n_points": len(group_valid),
            }
        )
        if plot and metric == "slope":
            plot_forced_origin(
                group_valid["Ta_era5"],
                group_valid["Ta_icos"],
                per_month[-1][f"Ta_{metric}"],
                f"Ta - ERA5 - {month}",
            )
            plot_forced_origin(
                group_valid["Ta_era5_proj"],
                group_valid["Ta_icos"],
                per_month[-1][f"Ta_{metric}_proj"],
                f"Ta - ERA5_resampled - {month}",
            )
            plot_forced_origin(
                group_valid["Tdp_era5"],
                group_valid["Tdp_icos"],
                per_month[-1][f"Tdp_{metric}"],
                f"Tdp - ERA5 - {month}",
            )
            plot_forced_origin(
                group_valid["Tdp_era5_proj"],
                group_valid["Tdp_icos"],
                per_month[-1][f"Tdp_{metric}_proj"],
                f"Tdp - ERA5_resampled - {month}",
            )
    return pd.DataFrame(per_month)


def compute_monthly_lr(csv_path: str, metric: str):
    """
    Description
    -----------
    Compute the monthly performance metrics between the mean theoretical value
    of the lapse rate and the lapse rates of TA and TDP
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
        group_valid = group.dropna(subset=["lapse_rate_ta", "lapse_rate_tdp"])
        if group_valid.empty:
            per_month.append(
                {
                    "month": str(month),
                    f"lr_ta_{metric}": np.nan,
                    f"lr_tdp_{metric}": np.nan,
                }
            )
            continue
        size = group_valid["lapse_rate_ta"].shape[0]
        array_th = np.full(size, -0.0065)
        per_month.append(
            {
                "month": str(month),
                f"lr_ta_{metric}": f(array_th, group_valid["lapse_rate_ta"]),
                f"lr_tdp_{metric}": f(array_th, group_valid["lapse_rate_tdp"]),
                "n_points": len(group_valid),
            }
        )
    return pd.DataFrame(per_month)

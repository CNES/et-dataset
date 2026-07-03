# Copyright: (c) 2023 CESBIO / Centre National d'Etudes Spatiales /
#            Université Paul Sabatier (UT3)
#
"""
Common function
"""

import datetime as dt
import re
from collections.abc import Generator
from datetime import datetime

import geopandas as gpd
import numpy as np
import numpy.typing as npt
import pandas as pd
import pyproj
import rasterio as rio
import skimage.morphology as skm
import xarray as xr
from fiona.errors import DriverError
from pyproj import CRS
from rasterio.warp import transform_bounds
from sensorsio import mgrs
from sensorsio.sentinel2 import get_theia_tiles
from shapely import covers
from shapely.geometry import Point, Polygon, box

from etdataset.logging import LoggerManager

logger = LoggerManager.get_logger(__name__)
MGRS_FORMAT = re.compile(r"[0-6][0-9][C-X][A-Z]{2}")


def check_mgrs_format(tile: str) -> None:
    """
    Check MGRS tile format
    """
    if MGRS_FORMAT.match(tile) is None:
        raise ValueError(f"Wrong format for MGRS tile ID (got: {tile})")


def BBoxException(Exception):  # noqa
    """
    Exception for bounding box
    """


def create_polygon(
    ul_lat: float,
    ul_long: float,
    ur_lat: float,
    ur_long: float,
    ll_lat: float,
    ll_long: float,
    lr_lat: float,
    lr_long: float,
) -> Polygon:
    ul = Point(ul_long, ul_lat)
    ur = Point(ur_long, ur_lat)
    ll = Point(ll_long, ll_lat)
    lr = Point(lr_long, lr_lat)
    points = [ul, ur, lr, ll]
    return Polygon(list(points))


def bbox_to_polygon(bounds: list[float] | rio.coords.BoundingBox) -> Polygon:
    # Convert bounds to polygon
    return Polygon(
        [
            [bounds[0], bounds[1]],
            [bounds[0], bounds[3]],
            [bounds[2], bounds[3]],
            [bounds[2], bounds[1]],
        ]
    )


def get_bbox_from_roi(roi_path: str) -> tuple[rio.coords.BoundingBox, CRS]:
    """
    Get bounding box information (bbox and CRS)
    from a ROI shapefile
    """
    # Roi shapefile
    try:
        roi_gdf = gpd.read_file(roi_path)
        return rio.coords.BoundingBox(
            *roi_gdf.bounds.iloc[0].values
        ), roi_gdf.crs
    except DriverError as e:
        raise BBoxException(f"Unable to read shapefile {roi_path}: {e}")


def get_utm_bbox_from_roi(roi_path: str) -> tuple[rio.coords.BoundingBox, CRS]:
    """
    Get bounding box information (bbox and CRS) in UTM coordinates
    from a ROI shapefile
    """
    # Roi shapefile
    try:
        roi_gdf = gpd.read_file(roi_path)
        epsg_code = roi_gdf.crs
        if epsg_code == 4326:
            # Auto-detect UTM zone based on centroid
            centroid = roi_gdf.geometry[0].centroid
            utm_zone = int((centroid.x + 180) / 6) + 1
            epsg_code = 32600 + utm_zone  # 32600 for northern hemisphere
            if centroid.y < 0:
                epsg_code = 32700 + utm_zone  # use 32700 for southern
            roi_gdf = roi_gdf.to_crs(epsg=epsg_code)
        return rio.coords.BoundingBox(
            *roi_gdf.bounds.iloc[0].values
        ), roi_gdf.crs
    except DriverError as e:
        raise BBoxException(f"Unable to read shapefile {roi_path}: {e}")


def get_bbox_from_mgrs_tile(tile: str) -> tuple[rio.coords.BoundingBox, CRS]:
    """
    Get bounding box information (bbox and CRS)
    from a MGRS_tile
    """
    # Tile
    try:
        return mgrs.get_bbox_mgrs_tile(tile), CRS("epsg:4326")
    except IndexError:
        raise BBoxException("Unkown tile {tile}")


def get_mgrs_tile_names_overlapping_roi(
    roi_bbox: rio.coords.BoundingBox, roi_crs: CRS, overlap: float = 10.0
) -> list[str]:
    """
    Get list of MGRS tiles that overlap a ROI with a minimum overlap area
    """
    # Get MGRS tiles
    mgrs_tiles = mgrs.get_mgrs_tiles_from_roi(roi_bbox, roi_crs)
    # Filter
    mgrs_tiles = mgrs_tiles[mgrs_tiles["overlap_percentage"] > overlap]
    return list(mgrs_tiles.Name.values)


def get_mgrs_tile_names_from_roi(roi_bbox: rio.coords.BoundingBox, roi_crs):
    """
    List of MGRS tiles corresponding to a ROI.
    The objective is to get a list of tiles that covers or
    intersects a given ROI while avoiding redundant overlapping
    """
    # Convert bounds to 4326
    wgs84_bounds = transform_bounds(roi_crs, 4326, *roi_bbox)
    # Convert bounds to polygon
    roi_poly = Polygon(
        [
            [wgs84_bounds[0], wgs84_bounds[1]],
            [wgs84_bounds[0], wgs84_bounds[3]],
            [wgs84_bounds[2], wgs84_bounds[3]],
            [wgs84_bounds[2], wgs84_bounds[1]],
        ]
    )
    # Get MGRS tiles
    mgrs_tiles = mgrs.get_mgrs_tiles_from_roi(roi_bbox, roi_crs)
    # Sort by overlap percentage descending
    mgrs_tiles = mgrs_tiles.sort_values("overlap_percentage", ascending=False)
    # Selection
    selected = []
    covered = None

    for idx, row in mgrs_tiles.iterrows():
        geom = box(*row["geometry"].bounds)
        if covered is None:
            covered = geom
            selected.append(idx)
        else:
            new_union = covered.union(geom)
            if not covers(covered, new_union):
                covered = new_union
                selected.append(idx)

        # Break if fully covered
        if covers(covered, roi_poly):
            break

    # Return minimal set of tiles
    return mgrs_tiles.loc[selected, "Name"].values


def check_theia_tiles(tile_ids: list[str]) -> list[str]:
    """
    Given a list of MGRS tile IDs, return the tiles available on THEIA platform
    """
    theia_tiles = get_theia_tiles()
    return list(set(tile_ids) & set(theia_tiles.index))


def read_product_list(
    path: str, crs: CRS | int, geometry: str = "geometry"
) -> gpd.GeoDataFrame:
    """
    Read a product list in CSV format and convert it to
    GeoDataFrame.
    """
    # Read to DataFrame
    df = pd.read_csv(path)
    # Parse date
    df["Date"] = df["Date"].apply(
        lambda date: datetime.strptime(date, "%Y-%m-%d").date()
    )
    # Remove unnecessary column
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns="Unnamed: 0")

    # Rename geometry columns if it already exists
    if geometry != "geometry" and geometry in df.columns:
        df = df.rename(columns={"geometry": "old_geometry"})

    # Convert geometry
    def convert_polygon(poly: str) -> Polygon:
        pattern = re.compile("POLYGON (((.*?)))")
        matching = pattern.search(poly)
        if matching is None:
            raise ValueError("Error during polygon conversion")
        coords = matching.group(1)
        coords = (
            tuple(float(item) for item in coord.split(" ") if item.strip())
            for coord in coords.split(",")
        )
        return Polygon(coords)

    try:
        # Convert the WKT column to geometry
        geometry = gpd.GeoSeries.from_wkt(df[geometry])
    except Exception:  # noqa
        geometry = df["geometry"].apply(lambda row: convert_polygon(row))

    return gpd.GeoDataFrame(
        data=df,
        geometry=geometry,
        crs=crs,
    )


def mask_bits(arr: npt.ArrayLike, pos: int, mask: str = "1") -> npt.NDArray:
    """
    Search for a binary code at a specific position in bytes array and
    return a mask array
    """
    if pos < 0:
        raise ValueError("Position fo bits extraction must be positive")
    array = np.array(arr)
    res = np.zeros_like(array)
    res[~np.isnan(array)] = np.isin(
        (array[~np.isnan(array)].astype(int) >> pos) & int("1" * len(mask), 2),
        int(mask, 2),
    )
    return res.astype(bool)


def dilate_mask(data: xr.DataArray, dilation: int = 1) -> xr.DataArray:
    """
    Dilate a binary mask
    """
    # Ensure the data is boolean (binary image)
    binary_mask = data.values.astype(bool)
    # Apply binary dilation
    footprint = skm.footprint_rectangle((2 * dilation + 1, 2 * dilation + 1))
    binary_mask = skm.binary_dilation(binary_mask, footprint=footprint)

    return xr.DataArray(binary_mask, dims=data.dims, coords=data.coords)


def close_mask(data: xr.DataArray, dilation: int = 1) -> xr.DataArray:
    """
    Perform a binary closing mask
    """
    # Ensure the data is boolean (binary image)
    binary_mask = data.values.astype(bool)
    # Apply binary dilation
    footprint = skm.footprint_rectangle((2 * dilation + 1, 2 * dilation + 1))
    binary_mask = skm.binary_closing(binary_mask, footprint=footprint)

    return xr.DataArray(binary_mask, dims=data.dims, coords=data.coords)


def get_utm_crs_from_roi(roi: rio.coords.BoundingBox):
    """
    Get UTM zone from a ROI
    """
    geom = box(*roi)
    lon, lat = geom.centroid.x, geom.centroid.y
    utm_zone = int((lon + 180) / 6) + 1
    is_northern = lat >= 0
    epsg = 32600 + utm_zone if is_northern else 32700 + utm_zone
    return CRS.from_epsg(epsg)


def get_utm_crs_from_lat_lon(lat: float, lon: float) -> pyproj.CRS:
    """
    Description
    -----------
    Determines the appropriate UTM CRS based on the station's latitude and
    longitude.

    It creates an Area of Interest (AOI) around the point and
    retrieves the corresponding UTM CRS metadata.

    Parameters
    ----------
    lat : float
        latitude
    longitude : float
        longitude

    Returns
    -------
    pyproj.CRS
        The UTM crs
    """
    # Query the PROJ database to obtain UTM CRS info that matches the point
    info_utm = pyproj.database.query_utm_crs_info(
        datum_name="WGS 84",
        area_of_interest=pyproj.aoi.AreaOfInterest(
            west_lon_degree=lon,
            south_lat_degree=lat,
            east_lon_degree=lon,
            north_lat_degree=lat,
        ),  # Thearea of interest reduces the CRS selection to exactly the
        # UTM zone covering this location.
    )[0]
    return pyproj.CRS.from_epsg(info_utm.code)


def create_bbox_from_lat_lon(
    lat: float, lon: float, width_m: float, height_m: float, crs: CRS
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
    """
    utm_crs = get_utm_crs_from_lat_lon(lat, lon)

    # For the UTM bounding box
    transformer_to_utm = pyproj.Transformer.from_crs(
        crs, utm_crs, always_xy=True
    )
    x, y = transformer_to_utm.transform(lon, lat)

    half_w, half_h = width_m / 2, height_m / 2
    bbox_utm = rio.coords.BoundingBox(
        left=x - half_w, bottom=y - half_h, right=x + half_w, top=y + half_h
    )
    # For the lat/lon bounding box
    transformer_to_lat_lon = pyproj.Transformer.from_crs(
        utm_crs, crs, always_xy=True
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


def work_area_from_coord_point(
    lat: float, lon: float, w: float, h: float, crs: CRS
):
    """
    Description
    ----------
    Get the working area around the given lat/lon point with UTM coordinates and
    geographic coordinates.

    Parameters
    ----------
    lat : float
        latitude
    lon : float
        longitude
    w : float
        width of the bounding box in meters.
    h : float
        height of the bounding box in meters.
    crs : CRS
        crs of the point in lat/lon

    Returns
    -------
    dict:
        - "utm" : UTM coordinates and CRS
        - "lat/lon" : geographic coordinates and CRS
    """
    bbox_utm, utm_crs, bbox_lat_lon = create_bbox_from_lat_lon(
        lat, lon, w, h, crs
    )
    return {
        "utm": (bbox_utm, utm_crs),
        "lat/lon": (bbox_lat_lon, crs),
    }


def filter_dataset_by_roi(
    ds: xr.Dataset | xr.DataArray,
    roi_bbox: rio.coords.BoundingBox,
    roi_crs: CRS,
    target_crs: str = "EPSG:4326",
) -> xr.Dataset | xr.DataArray:
    """
    Description
    -----------
    Filter Dataset to keep only data inside ROI.

    Handles ROI in:
    - lat/lon (EPSG:4326)
    - UTM or any other CRS
    Parameters
    ----------
    ds : xr.Dataset
        Input dataset.
    roi_bbox : rio.BoundingBox
        Bounding box of the ROI
    roi_crs : CRS
        CRS of the bounding box

    Returns
    -------
    xr.Dataset
        Dataset filtered to include only the ROI
    """

    if ds.rio.crs is None:
        ds = ds.rio.write_crs(target_crs)
    dataset_crs = ds.rio.crs
    roi_crs = CRS.from_user_input(roi_crs)

    if dataset_crs is None:
        raise ValueError("Dataset does not have defined CRS")

    if roi_crs != dataset_crs:
        min_lon, min_lat, max_lon, max_lat = transform_bounds(
            roi_crs,
            dataset_crs,
            roi_bbox.left,
            roi_bbox.bottom,
            roi_bbox.right,
            roi_bbox.top,
        )
    else:
        min_lon = roi_bbox.left
        max_lon = roi_bbox.right
        min_lat = roi_bbox.bottom
        max_lat = roi_bbox.top

    if ds.longitude.max() > 180:
        min_lon, max_lon = np.mod([min_lon, max_lon], 360)

    latitudes = ds.latitude.values

    if latitudes[0] > latitudes[-1]:
        lat_slice = slice(max_lat, min_lat)
    else:
        lat_slice = slice(min_lat, max_lat)
    ds = ds.rio.write_crs(target_crs, inplace=False)
    return ds.sel(
        latitude=lat_slice,
        longitude=slice(min_lon, max_lon),
    )


def filter_dataset_by_hours(
    ds: xr.Dataset,
    date: dt.date,
    times: list[dt.time],
    name_column: str = "time",
) -> xr.Dataset:
    """
    Description
    -----------
    Filter a dataset by a given list of hours

    Parameters
    -----------
    ds : xr.Dataset
        Dataset to filter
    list_dt : list[dt.datetime]
        List of datetimes
    name_column : str
        Name of the datetime column of the dataset

    Returns
    -------
    xr.Dataset
        Filtered dataset
    """
    list_dt = [dt.datetime.combine(date, t) for t in times]
    if name_column not in ds.coords:
        raise ValueError(f"'{name_column}' is not a coordinate in the dataset")
    ds = ds.copy()
    ds[name_column] = pd.to_datetime(ds[name_column].values)
    return ds.sel({name_column: list_dt})


def filter_dataset_by_pressure_levels(
    ds: xr.Dataset,
    pressure_levels: str | list[str],
    pressure_dim: str = "pressure_level",
) -> xr.Dataset:
    """
    Description
    -----------
    Filter an xarray Dataset to retain only the specified pressure level(s).

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset.
    pressure_levels : str or list of str
        Pressure level(s) to keep. Values must match the dataset's
        pressure coordinate labels (e.g., "500" or ["1000", "850", "500"]).
    pressure_dim : str, default "pressure_level"
        Name of the pressure dimension or coordinate in the dataset.
    Returns
    -------
    xr.Dataset
        Dataset filtered to include only the requested pressure level(s).

    Raises
    ------
    ValueError
        If the specified pressure dimension or coordinate does not exist.
    """

    if pressure_dim not in ds.dims and pressure_dim not in ds.coords:
        raise ValueError(
            f"The dimension/coordinate '{pressure_dim}' does not exist in the\
              dataset."
        )

    if isinstance(pressure_levels, str):
        pressure_levels = [pressure_levels]

    pressure_levels_sorted = sorted(
        pressure_levels,
        key=lambda x: float(x),
        reverse=True,  # highest pressure first
    )

    ds_sel = ds.sel({pressure_dim: pressure_levels_sorted}, method="nearest")

    ds_sel = ds_sel.sortby(ds_sel[pressure_dim], ascending=False)
    ds_sel = ds_sel.isel(
        {pressure_dim: ~ds_sel[pressure_dim].to_index().duplicated()}
    )
    return ds_sel


def kelvin_to_celsius(
    kelvin: xr.DataArray | npt.NDArray,
) -> xr.DataArray | npt.NDArray:
    """
    Description
    -----------
    Compute the temperature in celsius from a temperature in kelvin
    """
    if isinstance(kelvin, xr.DataArray):
        return kelvin - 273.15
    return np.array(kelvin) - 273.15


def celsius_to_kelvin(
    celsius: xr.DataArray | npt.NDArray,
) -> xr.DataArray | npt.NDArray:
    """
    Description
    -----------
    Compute the temperature in kelvin from a temperature in celsius
    """
    if isinstance(celsius, xr.DataArray):
        return celsius + 273.15
    return np.array(celsius) + 273.15


def normalize_longitude_latitude(
    ds: xr.Dataset,
    lon_name: str = "longitude",
    lat_name: str = "latitude",
    target: str = "-180_180",
) -> xr.Dataset:
    """
    Description
    -----------
    Normalize longitude coordinates of an xarray ERA5 Dataset to a desired range

    Parameters
    ----------
    ds : xr.Dataset
        Input ERA5 dataset containing longitude and latitude coordinates.
    lon_name : str
        Name of the longitude coordinate
    lat_name : str
        Name of the latitude coordinate
    target : str
        Target longitude convention:
        - "-180_180": longitudes in [-180, 180]
        - "0_360": longitudes in [0, 360]
    Returns
    -------
    xr.Dataset
        Dataset with normalized longitude coordinates and sorted by longitude.

    """
    lon = ds[lon_name]
    lat = ds[lat_name]

    lon_min = float(lon.min())
    lon_max = float(lon.max())
    lat_min = float(lat.min())
    lat_max = float(lat.max())

    if target == "-180_180":
        # If longitudes exceed 180, we assume they are in [0, 360]
        if lon_max > 180:
            lon = ((lon + 180) % 360) - 180
            logger.warning(
                f"Longitudes detected in 0-360 range ({lon_min} to {lon_max})"
            )
    elif target == "0_360":
        # If longitudes contain negative values, assume [-180, 180]
        if lon_min < 0:
            logger.warning(
                f"Converting longitude from -180-180 to 0-360 "
                f"({lon_min:.2f} to {lon_max:.2f})"
            )
            lon = lon % 360
    else:
        raise ValueError("The target longitude must be [-180_180] or [0_360]")

    if lat_min < -90 or lat_max > 90:
        raise ValueError("Unvalid latitude")

    ds = ds.assign_coords({lon_name: lon})
    ds = ds.sortby(lon_name)
    return ds


def generate_dates(
    start_date: dt.date,
    end_date: dt.date,
    day_step: int = 1,
) -> Generator[dt.date, None, None]:
    """
    Description
    -----------
    Generate dates between two dates with a given step

    This function yields one date at a time, starting from 'start_date'
    up to and including 'end_date', incremented by 'day_step' days

    Parameters
    ----------
    start_date : dt.date
        First date to generate
    end_date : dt.date
        Last date to generate
    day_step : int, optional
        Number of days between generated dates

    Yields
    ------
    dt.date
        A date in the specified range.
    """
    cur_date = start_date
    while cur_date <= end_date:
        yield cur_date
        cur_date += dt.timedelta(days=day_step)


def generate_hours(
    hour_start: int = 0,
    hour_end: int = 22,
    hour_step: int = 2,
) -> list[dt.time]:
    """
    Description
    -----------
    Generate hours between two hours with a given step.

    Parameters
    -----------
    hour_start : int
        Start hour
    hour_end : int
        End hour
    hour_step : int
        Hour step

    Returns
    -------
    list[dt.time]
    """
    if hour_step <= 0:
        raise ValueError("Hour step must be positive")

    hours = range(hour_start, hour_end + 1, hour_step)

    return [dt.time(hour=h) for h in hours]


def filter_dataset_by_location(
    ds: xr.Dataset,
    latitude: float,
    longitude: float,
) -> xr.Dataset:
    """
    Description
    -----------
    Filter Dataset to keep only the specified latitude and longitude

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset.
    latitude : float
        Latitude's location to keep
    longitude : float
        Longitude's location to keep

    Returns
    -------
    xr.Dataset
        Dataset filtered to include only the requested location

    """
    return ds.sel(latitude=latitude, longitude=longitude, method="nearest")


def get_xy_dims(ds: xr.Dataset) -> dict[str, str]:
    """
    Description
    -----------
    Determine the names of the spatial dimensions for interpolation in an
    xarray Dataset

    This function looks the dataset's dimensions
    It supports datasets with dimensions
    named either ("x", "y") or ("latitude", "longitude")

    Parameters
    ----------
    ds : xr.Dataset

    Returns
    -------
    dict[str, str]
        A dictionary mapping "x" and "y" to the corresponding dimension names
        in the dataset.
        e.g. {"x": "x", "y": "y"} or `{"x": "latitude", "y": "longitude"}

    """
    if "x" in ds.dims and "y" in ds.dims:
        return {"x": "x", "y": "y"}
    if "latitude" in ds.dims and "longitude" in ds.dims:
        return {"x": "longitude", "y": "latitude"}
    raise ValueError("Unable to detect spatial dimensions")


def get_ta_td_celsius_at_location(
    data: xr.Dataset,
    lat: float,
    lon: float,
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Description
    -----------
    Get Air temperature and dewpoint temperature at a specified location
    from data

    Parameters
    ----------
     data: xr.Dataset
        Data
     x : float
        Coord x (longitude or utm)
     y : float
        Coord y  (latitude or utm)

    Return
    -----------
    ta, td : tuple[xr.DataArray,xr.DataArray]
        Air temperature and dewpoint temperature at station location
    """
    # Determine the names of the spatial dimensions in the dataset
    dims = get_xy_dims(data)

    # If the dataset uses projected coordinates (x/y)
    if dims["x"] == "x" and dims["y"] == "y":
        x, y = (
            work_area_from_coord_point(lat, lon, 0, 0, CRS.from_epsg(4326))[
                "utm"
            ][0].left,
            work_area_from_coord_point(lat, lon, 0, 0, CRS.from_epsg(4326))[
                "utm"
            ][0].bottom,
        )

    # If the dataset uses geographic coordinates (longitude/latitude)
    if dims["x"] == "longitude" and dims["y"] == "latitude":
        x = lon
        y = lat

        if data[dims["x"]].max() > 180:
            # x = (x + 360) % 360
            x = ((x + 180) % 360) - 180
    # Interpolate air temperature and dewpoint temperature at the specified
    # location

    ta, td = (
        xr.DataArray(
            kelvin_to_celsius(
                data["ta"].interp(
                    {dims["x"]: x, dims["y"]: y}, method="nearest"
                )
            )
        ),
        xr.DataArray(
            kelvin_to_celsius(
                data["td"].interp(
                    {dims["x"]: x, dims["y"]: y}, method="nearest"
                )
            )
        ),
    )
    return ta, td


def fix_hour_format(s: pd.Series):
    """
    Description
    -----------
    Fix timestamps using the "24:00:00" hour convention (end of day) by
    converting them to "00:00:00" of the next day

    Parameters
    ----------
    s : pd.Series
        Series of datetime, some of which contain " 24:"
        (e.g. "2021-04-11 24:00:00").

    Returns
    -------
    pd.Series
        Series of parsed datetimes, with " 24:" timestamps shifted to
        "00:00:00" of the following day.
    """
    mask = s.str.contains(" 24:", na=False)
    if not mask.any():
        return s

    fixed_str = s.str.replace(" 24:", " 00:", regex=False)
    result = pd.to_datetime(fixed_str, format="%Y-%m-%d %H:%M:%S")

    result[mask] += pd.Timedelta(days=1)

    return result

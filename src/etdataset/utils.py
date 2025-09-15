# Copyright: (c) 2023 CESBIO / Centre National d'Etudes Spatiales /
#            Université Paul Sabatier (UT3)
#
"""
Common function
"""

import re
from datetime import datetime

import geopandas as gpd
import numpy as np
import numpy.typing as npt
import pandas as pd
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

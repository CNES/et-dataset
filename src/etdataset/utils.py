# Copyright: (c) 2023 CESBIO / Centre National d'Etudes Spatiales /
#            Université Paul Sabatier (UT3)
#
"""
Common function
"""

import re
from datetime import datetime

import geopandas as gpd
import pandas as pd
import rasterio as rio
from fiona.errors import DriverError
from pyproj import CRS
from sensorsio import mgrs
from sensorsio.sentinel2 import get_theia_tiles
from shapely.geometry import Point, Polygon

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


def get_mgrs_tile_names_from_roi(
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


def check_theia_tiles(tile_ids: list[str]) -> list[str]:
    """
    Given a list of MGRS tile IDs, return the tiles available on THEIA platform
    """
    theia_tiles = get_theia_tiles()
    return list(set(tile_ids) & set(theia_tiles.index))


def read_product_list(path: str, crs: CRS | int) -> gpd.GeoDataFrame:
    """
    Read a product list in CSV format and convert it to
    GeoDataFrame
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

    geometry = df["geometry"].apply(lambda row: convert_polygon(row))
    return gpd.GeoDataFrame(
        data=df[df.columns.difference(["b"], sort=False)],
        geometry=geometry,
        crs=crs,
    )

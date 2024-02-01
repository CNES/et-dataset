#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright: (c) 2023 CESBIO / Centre National d'Etudes Spatiales / Université Paul Sabatier (UT3)
#
"""
Common function
"""

import os
import re
import pandas as pd
import geopandas as gpd
import rasterio as rio
from fiona.errors import DriverError
from sensorsio import mgrs
from sensorsio.sentinel2 import get_theia_tiles, find_tile_orbit_pairs
from pyproj import CRS
from shapely.geometry import Point, Polygon
from etdataset.logging import LoggerManager

logger = LoggerManager.get_logger(__name__)
MGRS_FORMAT = re.compile(r'[0-6][0-9][C-X][A-Z]{2}')


def check_mgrs_format(tile: str) -> None:
    """
    Check MGRS tile format
    """
    if MGRS_FORMAT.match(tile) is None:
        raise ROIException(f"Wrong format for MGRS tile ID (got: {tile})")

def BBoxException(Exception):
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
    return Polygon([i for i in points])

def bbox_to_polygon(bounds: list[float]|rio.coords.BoundingBox)-> Polygon:
    # Convert bounds to polygon
    return Polygon([[bounds[0], bounds[1]], [bounds[0], bounds[3]],
                    [bounds[2], bounds[3]], [bounds[2], bounds[1]]])


def get_bbox_from_roi(roi_path: str) -> (rio.coords.BoundingBox, CRS):
    """
    Get bounding box information (bbox and CRS)
    from a ROI shapefile 
    """
    # Roi shapefile
    try:
        roi_gdf = gpd.read_file(roi_path)
        return rio.coords.BoundingBox(*roi_gdf.bounds.iloc[0].values), roi_gdf.crs
    except DriverError as e:
        raise BBoxException(f"Unable to read shapefile {roi}: {e}")


def get_bbox_from_mgrs_tile(tile: str) -> (rio.coords.BoundingBox, CRS):
    """
    Get bounding box information (bbox and CRS)
    from a MGRS_tile
    """
    # Tile
    try:
        return mgrs.get_bbox_mgrs_tile(tile), CRS("epsg:4326")
    except IndexError:
        raise BBoxException("Unkown tile {tile}")


def get_mgrs_tile_names_from_roi(roi_bbox: rio.coords.BoundingBox, roi_crs: CRS, overlap:float = 10.0) -> list[str]:
    """
    Get list of MGRS tiles that overlap a ROI with a minimum overlap area
    """
    # Get MGRS tiles
    mgrs_tiles = mgrs.get_mgrs_tiles_from_roi(roi_bbox, roi_crs)
    # Filter
    mgrs_tiles = mgrs_tiles[mgrs_tiles["overlap_percentage"] > overlap]
    return list(mgrs_tiles.Name.values)


def get_optimal_relative_orbit_for_mgrs_tile(tile_id: str) -> int:
    """
    Given a MGRS tile return the best relative orbit
    """
    tile_bbox = mgrs.get_bbox_mgrs_tile(tile_id)
    # Convert bounds to polygon
    aoi = Polygon([[tile_bbox[0], tile_bbox[1]], [tile_bbox[0], tile_bbox[3]],
                   [tile_bbox[2], tile_bbox[3]], [tile_bbox[2], tile_bbox[1]]])

    orbits_df = gpd.read_file(
        os.path.join(os.path.dirname(os.path.abspath(mgrs.__file__)), 'data/sentinel2/orbits.gpkg'))
    intersections = []
    orbits = []
    for _, orbit_row in orbits_df.iterrows():
        # Last test is to exclude weird duplicates (malformed gpkg ?)
        if orbit_row.geometry.intersects(aoi)and orbit_row.orbit_number not in orbits:
            orbits.append(orbit_row.orbit_number)
            inter_aoi_orbit = aoi.intersection(orbit_row.geometry)
            mgrs_orbit_coverage = inter_aoi_orbit.area / aoi.area
            intersections.append((orbit_row.orbit_number, mgrs_orbit_coverage))
    labels = ['relative_orbit_number', 'tile_and_orbit_coverage']
    return  int(pd.DataFrame.from_records(intersections, columns=labels).sort_values(by="tile_and_orbit_coverage", ascending=False).iloc[0].relative_orbit_number)

def check_theia_tiles(tile_ids: list[str]) -> list[str]:
    """
    Given a list of MGRS tile IDs, return the tiles available on THEIA platform
    """
    theia_tiles = get_theia_tiles()
    return list( set(tile_ids) & set(theia_tiles.index) )

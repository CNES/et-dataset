#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright: (c) 2023 CESBIO / Centre National d'Etudes Spatiales / Université Paul Sabatier (UT3)
#
"""
Common function
"""

import geopandas as gpd
import rasterio as rio
from fiona.errors import DriverError
from sensorsio import mgrs
from pyproj import CRS
from shapely.geometry import Point, Polygon


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

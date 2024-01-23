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


def get_bbox(roi: str = None, tile: str = None):
    """
    Get bounding box information (bbox and CRS)
    from a ROI shapefile or a MGRS tile
    """
    if roi is not None:
        # Roi shapefile
        try:
            roi_gdf = gpd.read_file(roi)
            return rio.coords.BoundingBox(*roi_gdf.bounds.iloc[0].values), roi_gdf.crs
        except DriverError as e:
            raise BBoxException(f"Unable to read shapefile {roi}: {e}")
    elif tile is not None:
        # Tile
        try:
            return mgrs.get_bbox_mgrs_tile(tile), 4326
        except IndexError:
            raise BBoxException("Unkown tile {tile}")
    raise BBoxException("Roi format unkown: must be a shapefile or a tile")

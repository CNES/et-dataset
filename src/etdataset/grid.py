#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright: (c) 2023 CESBIO / Centre National d'Etudes Spatiales / Université Paul Sabatier (UT3)
#
"""
Functions for S2 grid
"""
import geopandas as gpd
import rasterio as rio

from shapely.geometry.polygon import Polygon

from sensorsio import utils 


def get_s2_polygon(tile_id: str, 
                s2_grid: str | gpd.GeoDataFrame) -> Polygon:
    """
    Get S2 polygon corresponding to a tile ID
    """
    if isinstance(s2_grid,str):
        # Read S2 grid, if necessary
        s2_grid = gpd.read_file(s2_grid)
    tile = s2_grid[s2_grid.Name == tile_id]
    if len(tile) ==0:
        raise Exception(f"Tile {tile_id} not found")
    return tile.geometry.values[0]


def get_s2_tiles_from_roi(roi_path: str,
                 s2_grid: str | gpd.GeoDataFrame) -> list[str]:
    """
    Get S2 tile ID list which cover a ROI
    """
    if isinstance(s2_grid,str):
        # Read S2 grid, if necessary
        s2_grid = gpd.read_file(s2_grid)
    # Read ROI
    roi = gpd.read_file(roi_path)
    # Get tile IDs corresponding to the ROI
    roi_tiles = gpd.overlay(s2_grid,roi,how="intersection")
    tile_ids = list(roi_tiles.Name.values)
    return tile_ids


def get_bb_from_s2_tile(tile_id: str,
                        s2_grid: str | gpd.GeoDataFrame,
                        epsg: int | None = None) -> Polygon:
    """
    Get bounding box from S2 tile ID
    """
    if isinstance(s2_grid,str):
        # Read S2 grid, if necessary
        s2_grid = gpd.read_file(s2_grid)
    # Get bounding box for S2 tile
    bounds = s2_grid[s2_grid.Name == tile_id].geometry.bounds.values[0]
    bb = rio.coords.BoundingBox(*bounds)
    if epsg is not None:
       bb = utils.bb_transform(s2_grid.crs.to_epsg(),epsg,bb)
    return bb


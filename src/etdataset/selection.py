#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright: (c) 2023 CESBIO / Centre National d'Etudes Spatiales / Université Paul Sabatier (UT3)
#
"""
Select product
"""

from datetime import timedelta

import geopandas as gpd
import pandas as pd
import rasterio as rio
from pyproj import CRS
from rasterio.warp import transform_bounds
from shapely.geometry import Polygon
from tqdm import tqdm

from etdataset.logging import LoggerManager

logger = LoggerManager.get_logger(__name__)


def filter_with_roi(
    gdf: gpd.GeoDataFrame,
    roi_bbox: rio.coords.BoundingBox,
    roi_crs: CRS | None,
    min_overlap: float = 30,
) -> gpd.GeoDataFrame:
    """
    Filter a GeoDataFrame with a ROI
    """
    logger.debug(f"Number of products: {len(gdf)}")
    # Convert ROI
    bounds = transform_bounds(roi_crs, gdf.crs, *roi_bbox)
    # Convert bounds to polygon
    aoi_poly = Polygon(
        [
            [bounds[0], bounds[1]],
            [bounds[0], bounds[3]],
            [bounds[2], bounds[3]],
            [bounds[2], bounds[1]],
        ]
    )
    aoi = gpd.GeoDataFrame(data={"id": [1], "geometry": [aoi_poly]}, crs=gdf.crs)
    aoi_area = aoi_poly.area
    try:
        filtered = gpd.GeoDataFrame(
            gpd.overlay(gdf, aoi, how="intersection")
            .drop(["id"], axis=1)
            .merge(
                gdf[["Product_name", "geometry"]],
                how="inner",
                on="Product_name",
                suffixes=("_roi", "_orig"),
            )
            .rename(
                columns={
                    "geometry_roi": "overlap_geometry",
                    "geometry_orig": "geometry",
                }
            )
        )
        filtered["overlap_percentage"] = filtered.apply(
            lambda x: 100 * x.overlap_geometry.area / min(aoi_area, x.geometry.area),
            axis=1,
        )
        if min_overlap is not None:
            filtered = filtered[filtered["overlap_percentage"] > min_overlap]
    except AttributeError:
        logger.warning("Not products found")
        # No matches found
        return None
    logger.debug(f"Products found after filtering: {len(filtered)}")
    return filtered


def select_products(
    gdf1: gpd.GeoDataFrame,
    gdf2: gpd.GeoDataFrame,
    delta: timedelta,
    min_overlap: float,
    best_match=False,
) -> pd.DataFrame:
    """
    For each product in the first list gdf1,
    search for a product in the second list gdf2,
    whose acquisition date is within delta days of the date of the first product.
    If best_match is set to True, only the best match is returned.
    The best match has the closest day and then the best overlap.
    """
    gotchas = []

    dates = gdf1["Date"].unique()
    for d in tqdm(dates, total=len(dates), desc="Matching datasets ..."):
        # Select products in gdf1 corresponding to date d
        gdf1_selection = gdf1[gdf1["Date"] == d]
        # Select products in gdf2 corresponding to date +/- delta
        gdf2_selection = gdf2[(gdf2["Date"] >= d - delta) & (gdf2["Date"] <= d + delta)]

        # If a combination exists
        if len(gdf2_selection) > 0:
            try:
                res_inter = gpd.overlay(
                    gdf1_selection[["Product_name", "Date", "geometry"]],
                    gdf2_selection[["Product_name", "Date", "geometry"]],
                    how="intersection",
                )
                if len(res_inter):
                    results = res_inter.copy()
                    overlaps = []
                    for r in results.itertuples():
                        overlap = (
                            100
                            * r.geometry.area
                            / gdf2[gdf2["Product_name"] == r.Product_name_2]
                            .iloc[0]
                            .geometry.area
                        )
                        overlaps.append(overlap)
                    results["overlap"] = overlaps
                    gotchas.append(results)
            except AttributeError as e:
                logger.warning(e)

    if len(gotchas) == 0:
        logger.warning("No matching found")
        return pd.DataFrame(
            columns=["Product_name1", "Date_1", "Product_name_2", "Date_2"]
        )

    matches = pd.concat(gotchas).drop("geometry", axis=1)
    matches = matches[matches.overlap > min_overlap]

    if len(matches) == 0:
        logger.warning("No matching found")
    else:
        if best_match:
            best_matches = []
            for (product_name_1, date_1), group in matches.groupby(
                ["Product_name_1", "Date_1"]
            ):
                best_date = date_1 + delta + pd.Timedelta("1 day")
                best_overlap = 0
                best_product = None
                for _, row in group.iterrows():
                    date_2 = row.Date_2
                    product_name_2 = row.Product_name_2
                    overlap = row.overlap
                    if abs((date_1 - best_date).days) > abs((date_2 - date_1).days):
                        best_date = date_2
                        best_overlap = overlap
                        best_product = product_name_2
                    elif (
                        abs((date_1 - best_date).days) == abs((date_2 - date_1).days)
                        and overlap > best_overlap
                    ):
                        best_date = date_2
                        best_overlap = overlap
                        best_product = product_name_2
                best_matches.append(
                    [product_name_1, date_1, best_product, best_date, best_overlap]
                )
            matches = pd.DataFrame(
                data=best_matches,
                columns=[
                    "Product_name_1",
                    "Date_1",
                    "Product_name_2",
                    "Date_2",
                    "overlap",
                ],
            )
        logger.info(f"Found {len(matches)} matches")
    return matches.reset_index(drop=True)

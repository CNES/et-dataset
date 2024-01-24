#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright: (c) 2023 CESBIO / Centre National d'Etudes Spatiales / Université Paul Sabatier (UT3)
#
"""
Select product
"""
import os
from datetime import timedelta

import geopandas as gpd
import pandas as pd
import rasterio as rio
from tqdm import tqdm

from etdataset.logging import LoggerManager

logger = LoggerManager.get_logger(__name__)


def select_products(
    gdf1: gpd.GeoDataFrame, gdf2: gpd.GeoDataFrame, delta: timedelta, min_overlap: float
) -> pd.DataFrame:
    """
    For each product in the first list gdf1,
    search for a product in the second list gdf2,
    whose acquisition date is within delta days of the date of the first product.
    """
    gotchas = []

    dates = gdf1["date"].unique()
    for d in tqdm(dates, total=len(dates), desc="Matching datasets ..."):
        # Select products in gdf1 corresponding to date d
        gdf1_selection = gdf1[gdf1["date"] == d]
        # Select products in gdf2 corresponding to date +/- delta
        gdf2_selection = gdf2[(gdf2["date"] >= d - delta) & (gdf2["date"] <= d + delta)]

        # If a combination exists
        if len(gdf2_selection) > 0:
            try:
                res_inter = gpd.overlay(
                    gdf1_selection[["product_name", "date", "geometry"]],
                    gdf2_selection[["product_name", "date", "geometry"]],
                    how="intersection",
                )
                if len(res_inter):
                    results = res_inter.copy()
                    overlaps = []
                    for r in results.itertuples():
                        overlap = (
                            100
                            * r.geometry.area
                            / gdf2[gdf2["product_name"] == r.product_name_2]
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
            columns=["product_name1", "date_1", "product_name_2", "date_2"]
        )

    gotcha = pd.concat(gotchas).drop("geometry", axis=1)
    gotcha = gotcha[gotcha.overlap > min_overlap]
    if len(gotcha) == 0:
        logger.warning("No matching found")
    else:
        logger.info(f"Found {len(gotcha)} matches")
    return gotcha.reset_index(drop=True)

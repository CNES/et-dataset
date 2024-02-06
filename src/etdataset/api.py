#!/usr/bin/env python
# coding: utf8

# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales
from datetime import datetime, timedelta

import os
import numpy as np
import geopandas as gpd
import pandas as pd
import rasterio as rio
import xarray as xr
from pyproj import CRS
from sensorsio import utils

from etdataset.database import (
    create_eco_db,
    create_ls8_db,
    create_s2_db,
    to_collectionV2,
)
from etdataset.logging import LoggerManager
from etdataset.provider import Collection, get_provider
from etdataset.reader import get_product_reader
from etdataset.selection import select_products
from etdataset.utils import check_mgrs_format

logger = LoggerManager.get_logger(__name__)


class DatasetException(Exception):
    """
    Exception for dataset creation
    """


class FileParserException(Exception):
    """
    Exception for file parser
    """


class SearchException(Exception):
    """
    Exception for dataset creation
    """


class ROIException(Exception):
    """
    Exception for dataset creation
    """


def parse_date(date_str: str) -> datetime:
    """
    Parse date expected format YYYY-MM-DD
    """
    try:
        date = datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        raise ValueError(
            f"Error: The expected format for date must be Year-Month-Day (got: {date_str})"
        )
    return date


def create_dataset(
    vis_path: str,
    tir_path: str | None = None,
    tile_id: str | None = None,
    use_mask: bool = False,
) -> xr.Dataset:
    """
    Create a dataset
    """
    if tir_path is None:
        tir_path = vis_path
    # Get reader
    vis_reader = get_product_reader(vis_path)
    tir_reader = get_product_reader(tir_path)

    # Check tile
    if vis_reader.tile is None and tir_reader.tile is None:
        if tile_id is None:
            logger.error("None of the products have MGRS tile information")
            raise DatasetException("You must provide the MGRS tile ID")
        vis_reader.tile = tile_id
        tir_reader.tile = tile_id
    elif vis_reader.tile is None and tir_reader.tile is not None:
        logger.info("Use MGRS tile from TIR product")
        vis_reader.tile = tir_reader.tile
    elif vis_reader.tile is not None and tir_reader.tile is None:
        logger.info("Use MGRS tile from VIS product")
        tir_reader.tile = vis_reader.tile
    elif vis_reader.tile != tir_reader.tile:
        logger.error(
            f"The products are not on the same MGRS tile : "
            f"VIS tile = {vis_reader.tile} and TIR tile = {tir_reader.tile}"
        )
        raise DatasetException("The products are not on the same MGRS tile")

    # Force the same bounding box
    common_bbox, common_crs = utils.bb_common(
        bounds=[vis_reader.bb, tir_reader.bb],
        src_crs=[vis_reader.crs, tir_reader.crs],
        snap=60,
        target_crs=vis_reader.crs,
    )
    vis_reader.crs = common_crs
    vis_reader.bb = common_bbox
    tir_reader.crs = common_crs
    tir_reader.bb = common_bbox

    # Read VIS
    vis_xr = vis_reader.read_vis_bands(use_mask)
    logger.debug(f"Read VIS: {type(vis_xr)}")

    # Read TIR
    tir_xr = tir_reader.read_tir_bands(use_mask)
    logger.debug(f"Read TIR: {type(tir_xr)}")

    # Merge
    merged_xr = xr.merge((vis_xr, tir_xr), combine_attrs="no_conflicts")
    logger.debug(f"Merged: {merged_xr.attrs}")

    return merged_xr


def find_products(
    ls8_db_path: str | None = None,
    eco_db_path: str | None = None,
    s2_db_path: str | None = None,
    min_date: datetime | None = None,
    max_date: datetime | None = None,
    max_cloud_cover: float = 25,
    delta: timedelta = pd.Timedelta("3 day"),
    roi_bbox: rio.coords.BoundingBox = None,
    roi_crs: CRS | int = None,
    min_roi_overlap: float = 50,
    min_product_overlap: float = 40,
) -> pd.DataFrame:
    """
    Find products
    """
    databases = {}
    # Get landsat products
    if ls8_db_path is not None:
        ls8_db = create_ls8_db(
            ls8_db_path,
            min_date=min_date,
            max_date=max_date,
            max_cloud_cover=max_cloud_cover,
            roi_bbox=roi_bbox,
            roi_crs=roi_crs,
            min_roi_overlap=min_roi_overlap,
        )
        databases["landsat"] = ls8_db

    # Get ECOSTRESS products
    if eco_db_path is not None:
        eco_db = create_eco_db(
            eco_db_path,
            min_date=min_date,
            max_date=max_date,
            roi_bbox=roi_bbox,
            roi_crs=roi_crs,
            min_roi_overlap=min_roi_overlap,
        )
        eco_db_v2 = to_collectionV2(eco_db, roi_bbox, roi_crs, min_roi_overlap)
        databases["ecostress"] = eco_db_v2

    # Get Sentinel2 products
    if s2_db_path is not None:
        s2_db = create_s2_db(
            s2_db_path,
            min_date=min_date,
            max_date=max_date,
            roi_bbox=roi_bbox,
            roi_crs=roi_crs,
            min_roi_overlap=min_roi_overlap,
        )
        databases["sentinel2"] = s2_db

    # Get ECOSTRESS products
    # Select matches
    res = pd.DataFrame()
    if len(databases) == 1:
        key = list(databases.keys())[0]
        res = databases[key]
        res.rename(
            columns={
                "product_name": f"product_name_{key}",
                "date": f"date_{key}",
            }
        )
    elif len(databases) > 1:
        keys = list(databases.keys())
        if len(databases) > 2:
            logger.warning(
                f"Matching only performed between the first two product lists : {keys[0]} and {keys[1]}"
            )
        res = select_products(
            databases[keys[0]], databases[keys[1]], delta, min_product_overlap
        )
        res.rename(
            columns={
                "product_name_1": f"product_name_{keys[0]}",
                "date_1": f"date_{keys[0]}",
                "product_name_2": f"product_name_{keys[1]}",
                "date_2": f"date_{keys[1]}",
            }
        )

    return res


def search(
    collection: Collection,
    min_date: str,
    max_date: str,
    tile_id: str | None = None,
    roi_bbox: rio.coords.BoundingBox | None = None,
    roi_crs: CRS | int | None = None,
    max_cloud_cover: float = 20,
) -> gpd.GeoDataFrame:
    """
    Search products in a collection
    """
    # Checks
    _ = parse_date(min_date)
    _ = parse_date(max_date)
    if tile_id is None and roi_bbox is None:
        raise SearchException(
            "You must provide either a ROI bounding box or MGRS tile ID"
        )
    if roi_crs is None and roi_bbox is not None:
        raise SearchException("You must provide a ROI bounding box with a CRS")
    if tile_id is not None:
        check_mgrs_format(tile_id)
    if max_cloud_cover < 0 or max_cloud_cover > 100:
        raise SearchException(
            f"Cloud cover criteria must be between 0 and 100 (got : {max_cloud_cover}"
        )

    latlon_bbox = None
    if roi_bbox is not None:
        # Convert to latlon
        latlon_bbox = utils.bb_transform(roi_crs, 4326, roi_bbox)

    # Get provider
    provider = get_provider(collection)
    logger.debug(f"Provider: {provider}")

    # Search
    results = provider.search(min_date, max_date, tile_id, latlon_bbox, max_cloud_cover)
    logger.info(f"Number of products found: {len(results)}")

    return results


def download(
        products: pd.DataFrame,
        output_dir: str = os.path.join(os.getcwd(),"download"),
     ) -> None:
    """
    Download products from catalog
    The products to dowload are listed in a DataFrame.
    The required column are "Product_name" and "URL" 
    which must contain the URLs to download a product. 
    The URLs column can be a str or a list of str.
    """
    # Checks
    os.makedirs(output_dir,exist_ok=True)
    if not "Product_name" in products.columns:
        logger.exception("You must provide the product names.")
    if not "URL" in products.columns:
        logger.exception("You must provide the URL to download.")
    # Download
    for collection, group in products.groupby("Collection"):
        logger.debug(f"Collection: {collection}")
        provider = get_provider(Collection[collection])
        logger.debug(f"Provider: {provider}")
        urls = group[["Product_name","URL"]].copy()
        if "Checksum" in group.columns:
            urls["Checksum"] = group["Checksum"].values
        else:
            urls["Checksum"] = np.nan
        provider.download(urls, output_dir)



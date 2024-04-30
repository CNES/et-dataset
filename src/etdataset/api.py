#!/usr/bin/env python
# coding: utf8
# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales
import os
from datetime import datetime

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio as rio
import xarray as xr
from pyproj import CRS
from sensorsio import utils

from etdataset.logging import LoggerManager
from etdataset.provider import Collection, get_provider
from etdataset.reader import get_product_reader
from etdataset.selection import filter_with_roi, select_products
from etdataset.utils import check_mgrs_format, get_bbox_from_mgrs_tile

logger = LoggerManager.get_logger(__name__)


class APIException(Exception):
    """
    Exception related to arguments
    """


def parse_date(date_str: str) -> datetime:
    """
    Parse date expected format YYYY-MM-DD
    """
    try:
        date = datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        raise APIException(
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
            raise APIException("You must provide the MGRS tile ID")
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
        raise APIException("The products are not on the same MGRS tile")

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


def search(
    collection: Collection,
    min_date: str,
    max_date: str,
    tile_id: str | None = None,
    roi_bbox: rio.coords.BoundingBox | None = None,
    roi_crs: CRS | int | None = None,
    max_cloud_cover: float = 20,
    min_roi_overlap: float = 0,
) -> gpd.GeoDataFrame:
    """
    Search products in a collection
    """
    # Checks
    _ = parse_date(min_date)
    _ = parse_date(max_date)
    if tile_id is None and roi_bbox is None:
        raise APIException("You must provide either a ROI bounding box or MGRS tile ID")
    if roi_crs is None and roi_bbox is not None:
        raise APIException("You must provide a ROI bounding box with a CRS")
    if tile_id is not None:
        check_mgrs_format(tile_id)
    if max_cloud_cover < 0 or max_cloud_cover > 100:
        raise APIException(
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
    logger.info(f"Products found for {collection.name} in catalog: {len(results)}")

    # Get bounding box from tile
    if tile_id is not None:
        roi_bbox, roi_crs = get_bbox_from_mgrs_tile(tile_id)

    # Filter with additional criteria for collection 1
    results = filter_with_roi(results, roi_bbox, roi_crs, min_overlap=min_roi_overlap)
    logger.info(
        f"Products found for {collection.name} " f"after ROI filtering: {len(results)}"
    )

    return results


def download(
    products: pd.DataFrame,
    output_dir: str = os.path.join(os.getcwd(), "download"),
) -> None:
    """
    Download products from catalog
    The products to dowload are listed in a DataFrame.
    The required column are "Product_name" and "URL"
    which must contain the URLs to download a product.
    The URLs column can be a str or a list of str.
    """
    # Checks
    os.makedirs(output_dir, exist_ok=True)
    if "Product_name" not in products.columns:
        logger.exception("You must provide the product names.")
    if "URL" not in products.columns:
        logger.exception("You must provide the URL to download.")
    # Download
    for collection, group in products.groupby("Collection"):
        logger.debug(f"Collection: {collection}")
        provider = get_provider(Collection[collection])
        logger.debug(f"Provider: {provider}")
        urls = group[["Product_name", "URL"]].copy()
        if "Checksum" in group.columns:
            urls["Checksum"] = group["Checksum"].values
        else:
            urls["Checksum"] = np.nan
        provider.download(urls, output_dir)


def select(
    collection1: Collection,
    collection2: Collection,
    min_date: str,
    max_date: str,
    delta: str | None = "3 day",
    tile_id: str | None = None,
    roi_bbox: rio.coords.BoundingBox = None,
    roi_crs: CRS | int = None,
    max_cloud_cover: float = 20,
    min_roi_overlap: float = 40,
    min_product_overlap: float = 40,
    only_best_match: bool = False,
) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """
    Select products
    """
    # Checks
    min_date_str = min_date
    min_date = parse_date(min_date_str)
    max_date_str = max_date
    max_date = parse_date(max_date_str)
    if max_date < min_date:
        raise APIException(
            "Maximum acquisition date must be more recent than minimum date"
        )
    try:
        delta = pd.Timedelta(delta)
    except ValueError:
        raise APIException(
            "Error: The format for delta acquisition time is not recognized (ex: 1 day)"
        )
    if tile_id is None and roi_bbox is None:
        raise APIException("You must provide either a ROI bounding box or MGRS tile ID")
    if roi_crs is None and roi_bbox is not None:
        raise APIException("You must provide a ROI bounding box with a CRS")
    if tile_id is not None:
        check_mgrs_format(tile_id)
    if max_cloud_cover < 0 or max_cloud_cover > 100:
        raise APIException(
            f"Cloud cover criteria must be between 0 and 100 (got : {max_cloud_cover}"
        )
    if min_roi_overlap < 0 or min_roi_overlap > 100:
        raise APIException(
            f"Cloud cover criteria must be between 0 and 100 (got : {max_cloud_cover}"
        )
    if min_product_overlap < 0 or min_product_overlap > 100:
        raise APIException(
            f"Cloud cover criteria must be between 0 and 100 (got : {max_cloud_cover}"
        )

    # Search into collection 1
    selection1 = search(
        collection1,
        min_date_str,
        max_date_str,
        tile_id=tile_id,
        roi_bbox=roi_bbox,
        roi_crs=roi_crs,
        max_cloud_cover=max_cloud_cover,
        min_roi_overlap=min_roi_overlap,
    )

    # Search into collection 2
    selection2 = search(
        collection2,
        min_date_str,
        max_date_str,
        tile_id=tile_id,
        roi_bbox=roi_bbox,
        roi_crs=roi_crs,
        max_cloud_cover=max_cloud_cover,
        min_roi_overlap=min_roi_overlap,
    )

    if len(selection1) == 0 and len(selection2) == 0:
        logger.warning("No product in one of the collection")
        return pd.DataFrame()

    # Select matches
    matches = select_products(
        selection1, selection2, delta, min_product_overlap, only_best_match
    )
    matches = matches.rename(
        columns={
            "Product_name_1": f"Product_name_{collection1.name}",
            "Date_1": f"Date_{collection1.name}",
            "Product_name_2": f"Product_name_{collection2.name}",
            "Date_2": f"Date_{collection2.name}",
        }
    )
    if len(matches) > 0:
        selection1 = (
            selection1.merge(
                matches[[f"Product_name_{collection1.name}"]],
                left_on="Product_name",
                right_on=f"Product_name_{collection1.name}",
            )
            .drop(columns=[f"Product_name_{collection1.name}"])
            .drop_duplicates(subset=["Product_name"])
            .reset_index(drop=True)
        )
        selection2 = (
            selection2.merge(
                matches[[f"Product_name_{collection2.name}"]],
                left_on="Product_name",
                right_on=f"Product_name_{collection2.name}",
            )
            .drop(columns=[f"Product_name_{collection2.name}"])
            .drop_duplicates(subset=["Product_name"])
            .reset_index(drop=True)
        )

    else:
        selection1 = pd.Dataframe()
        selection2 = pd.Dataframe()

    # Return results
    return selection1, selection2, matches

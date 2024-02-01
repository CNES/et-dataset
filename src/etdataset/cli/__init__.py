#!/usr/bin/env python
# coding: utf8

# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales
import argparse
import logging
import os
import sys
from datetime import datetime

import pandas as pd

from etdataset.api import create_dataset, find_products, search
from etdataset.provider import Collection
from etdataset.logging import LoggerManager
from etdataset.utils import get_bbox_from_roi, get_bbox_from_mgrs_tile, check_mgrs_format
from etdataset.writer import export_matlab, write_dataset, write_matches, write_results

LS8_PATH = "landsat_ot_c2_l2_659ed86540bcec65.csv"
S2_PATH = "SENTINEL2.csv"
ECO_PATH = "ecostress_eco2lste_655f877cf0476b91.csv"

logger = LoggerManager.get_logger(__name__)


# sub-command functions
def cli_create(args: argparse.ArgumentParser) -> None:
    """
    Create dataset
    """
    logger.debug(f"Create arguments: {args}")
    # Check VIS product path
    if not os.path.isdir(args.vis):
        raise FileNotFoundError(f"VIS product not found ({args.vis})")
    logger.debug(f"VIS path: {args.vis}")

    # Check TIR path
    if args.tir is None:
        args.tir = args.vis
    if not os.path.isdir(args.tir):
        raise FileNotFoundError(f"TIR product not found ({args.tir})")
    logger.debug(f"TIR path: {args.tir}")

    # Check output path
    if not os.path.isdir(args.output):
        logger.debug(f"Create output path: {args.output}")
        os.makedirs(args.output, exist_ok=True)

    # Create dataset
    data = create_dataset(
        vis_path=args.vis, tir_path=args.tir, tile_id=args.tile, use_mask=args.use_mask
    )

    # Write dataset
    if data is not None:
        if args.matlab:
            export_matlab(data, directory=args.output)
        else:
            write_dataset(data, directory=args.output)


def cli_find(args: argparse.ArgumentParser) -> None:
    """
    Find products
    """
    logger.debug(f"Find arguments: {args}")

    # Landsat8 DB
    ls8_db_path = None
    if args.landsat:
        ls8_db_path = os.path.join(os.environ["METADATA_PATH"], LS8_PATH)
        if not os.path.isfile(ls8_db_path):
            raise FileNotFoundError(f"Landsat8 database not found ({ls8_db_path})")
    logger.debug(f"Landsat8 DB path: {ls8_db_path}")

    # ECOSTRESS DB
    eco_db_path = None
    if args.ecostress:
        eco_db_path = os.path.join(os.environ["METADATA_PATH"], ECO_PATH)
        if not os.path.isfile(eco_db_path):
            raise FileNotFoundError(f"ECOSTRESS database not found ({eco_db_path})")
    logger.debug(f"ECOSTRESS DB path: {eco_db_path}")

    # Sentinel2 DB
    s2_db_path = None
    if args.sentinel2:
        s2_db_path = os.path.join(os.environ["METADATA_PATH"], S2_PATH)
        if not os.path.isfile(s2_db_path):
            raise FileNotFoundError(f"Sentinel2 database not found ({s2_db_path})")
    logger.debug(f"Sentinel2 DB path: {s2_db_path}")

    if args.landsat and args.ecostress and args.sentinel2:
        logger.warning("Matches can be found only between two product list")
        logger.warning("Skip Sentinel2")
        s2_db_path = None

    # Process date
    try:
        min_date = datetime.strptime(args.min_date, "%Y-%m-%d")
    except ValueError:
        raise ValueError(
            "Error: The format for minimum acqsuisition date must be Year-Month-Day"
        )
    try:
        max_date = datetime.strptime(args.max_date, "%Y-%m-%d")
    except ValueError:
        raise ValueError(
            "Error: The format for maximum acquisition date must be Year-Month-Day"
        )
    logging.debug(f"Minimum acquisition date: {min_date}")
    logging.debug(f"Maximum acquisition date: {max_date}")
    if max_date < min_date:
        raise ValueError("Maximum acquisition date must be more recent than minimum date")
    try:
        delta = pd.Timedelta(args.delta)
    except ValueError:
        raise ValueError(
            "Error: The format for delta acquisition time is not recognized (ex: 1 day)"
        )
    logger.debug(f"Delta acquisition date: {delta}")

    # Get ROI bounding box and CRS from tile or shapefile
    roi_bbox = None
    roi_crs = None
    if args.tile is not None:
        roi_bbox, roi_crs = get_bbox_from_mgrs_tile(args.tile)
    else:
        roi_bbox, roi_crs = get_bbox_from_roi(args.roi)

    matches = find_products(
        ls8_db_path=ls8_db_path,
        eco_db_path=eco_db_path,
        s2_db_path=s2_db_path,
        min_date=min_date,
        max_date=max_date,
        max_cloud_cover=args.max_cloud_cover,
        delta=delta,
        roi_bbox=roi_bbox,
        roi_crs=roi_crs,
        min_roi_overlap=args.min_roi_overlap,
    )

    # Write results
    product_columns = [column for column in matches.columns if "product_name" in column]
    for product_column in product_columns:
        logger.info(f"{product_column} = {list(matches[product_column].unique())}")
    if len(product_columns) > 1:
        for product, group in matches.groupby(product_columns[0]):
            logger.debug(
                f"Image: {product} - List of images:  {list(group[product_columns[1]].unique())}"
            )
    write_matches(matches, args.output)

def cli_search(args: argparse.ArgumentParser) -> None:
    """
    Search products
    """
    logger.debug(f"Search arguments: {args}")

    # Check arguments
    try:
        min_date = datetime.strptime(args.min_date, "%Y-%m-%d")
    except ValueError:
        raise ValueError(
            "Error: The format for minimum acqsuisition date must be Year-Month-Day"
        )
    try:
        max_date = datetime.strptime(args.max_date, "%Y-%m-%d")
    except ValueError:
        raise ValueError(
            "Error: The format for maximum acquisition date must be Year-Month-Day"
        )
    if max_date < min_date:
        raise ValueError("Maximum acquisition date must be more recent than minimum date")
    if args.tile is not None:
        check_mgrs_format(args.tile)
    roi_bbox, roi_crs = None, None
    if args.roi is not None:
        if not os.path.isfile(args.roi):
            raise FileNotFoundError(f"ROI file not found ({args.roi})")
        roi_bbox, roi_crs = get_bbox_from_roi(args.roi)

    # Search
    results = search(args.collection,
           args.min_date,
           args.max_date,
           args.tile,
           roi_bbox,
           roi_crs, 
           args.max_cloud_cover,
           ) 

    # Write results
    write_results(results, args.output)

def get_parser() -> argparse.ArgumentParser:
    """
    Generate argument parser for cli
    """
    # create the top-level parser
    parser = argparse.ArgumentParser(description="Manage ET dataset")
    subparsers = parser.add_subparsers(help="Command list", required=True)

    # create the parser for the "create" command
    parser_create = subparsers.add_parser(
        "create",
        help="Create dataset from Landsat/Ecostress/Sentinel2 products. The dataset is resampling at 60m resolution and corresponds to a MGRS tile.",
    )
    parser_create.add_argument(
        "-v", "--verbose", dest="verbose", action="store_true", help="Verbose mode"
    )
    parser_create.add_argument(
        "--vis", type=str, required=True, help="Path to VIS product (optical bands)"
    )
    parser_create.add_argument("--tir", type=str, help="Path to TIR product (thermal bands), if not provided the VIS product is used for thermal bands.")
    parser_create.add_argument(
        "-t", "--tile", type=str, help="Tile ID (Only for Landsat)"
    )
    parser_create.add_argument(
        "--use_mask", dest="use_mask", action="store_true", help="Filter data with masks (Quality, Cloud, Water)"
    )
    parser_create.add_argument(
            "--output", type=str, help="Output dataset directory path (default: current directory)", default=os.getcwd()
    )
    parser_create.add_argument(
        "-m",
        "--matlab_export",
        dest="matlab",
        action="store_true",
        help="Write the dataset in Matlab format",
    )
    parser_create.set_defaults(func=cli_create)

    # create the parser for the "find" command
    parser_find = subparsers.add_parser(
        "find", help="Find matches between products that satisfy required criteria"
    )
    parser_find.add_argument(
        "-v", "--verbose", dest="verbose", action="store_true", help="Verbose mode"
    )
    parser_find.add_argument(
        "-l",
        "--landsat",
        action="store_true",
        help="Find among Landsat products",
    )
    parser_find.add_argument(
        "-s",
        "--sentinel2",
        action="store_true",
        help="Find among Sentinel2 products",
    )
    parser_find.add_argument(
        "-e",
        "--ecostress",
        action="store_true",
        help="Find among Ecostress products",
    )
    parser_find.add_argument(
        "--min_date", help="Minimum date for acquisition in YYYY-MM-DD format", type=str, required=True, default=None
    )
    parser_find.add_argument(
        "--max_date", help="Maximum date for acquisition in YYYY-MM-DD format", type=str, required=True, default=None
    )
    parser_find.add_argument(
        "--max_cloud_cover",
        type=int,
        default=25,
        help="Maximum cloud cover (only for Landsat8 products) (default: 25)",
    )
    parser_find.add_argument(
        "--delta",
        type=str,
        default="3 days",
        help="Maximum time delta allowed between acquisitions (default: 3 days)",
    )
    group = parser_find.add_mutually_exclusive_group(required=True)
    group.add_argument("-t", "--tile", type=str, help="Tile ID")
    group.add_argument(
        "-r", "--roi", type=str, help="Path of the region of interest in Shapefile format"
    )
    parser_find.add_argument(
        "--min_roi_overlap",
        type=int,
        default=50,
        help="Minimum overlap between ROI and a product (default: 50)",
    )
    parser_find.add_argument(
        "--min_product_overlap",
        type=int,
        default=40,
        help="Minimum overlap between two products (default: 40)",
    )
    parser_find.add_argument(
        "--metadata",
        type=str,
        default=None,
        help="Path to the directory containing metadata csv files",
    )
    parser_find.add_argument(
            "--output", type=str, help="CSV output file (default: matches.csv)", default="matches.csv"
    )
    parser_find.set_defaults(func=cli_find)
    
    # create the parser for the "search" command
    parser_search = subparsers.add_parser(
        "search", help="Search products in a collection that satisfy required criteria"
    )
    parser_search.add_argument(
        "-v", "--verbose", dest="verbose", action="store_true", help="Verbose mode"
    )
    parser_search.add_argument(
        "-c",
        "--collection",
        type=lambda arg: Collection[arg], 
        choices=Collection,
        help="Collection of products",
    )
    parser_search.add_argument(
        "--min_date", help="Minimum date for acquisition in YYYY-MM-DD format", type=str, required=True, default=None
    )
    parser_search.add_argument(
        "--max_date", help="Maximum date for acquisition in YYYY-MM-DD format", type=str, required=True, default=None
    )
    parser_search.add_argument(
        "--max_cloud_cover",
        type=int,
        default=25,
        help="Maximum cloud cover (only for Landsat8 products) (default: 25)",
    )
    group = parser_search.add_mutually_exclusive_group(required=True)
    group.add_argument("-t", "--tile", type=str, help="Tile ID")
    group.add_argument(
        "-r", "--roi", type=str, help="Path of the region of interest in Shapefile format"
        )
    parser_search.add_argument(
            "--output", type=str, help="CSV output file (default: results.csv)", default="results.csv"
    )
    parser_search.set_defaults(func=cli_search)

    return parser


def etdataset() -> None:
    """
    Entry point
    """
    # Parser arguments
    parser = get_parser()
    args = parser.parse_args()

    # Check environment variables
    if not os.environ.get("METADATA_PATH") and args.metatada is None:
        raise Exception("You must provide the path to the directory containing the metadata csv files. You can use either the environment variable METADATA_PATH or the option --metadata in the command line")

    # Configure logging
    log_level = logging.INFO
    if args.verbose:
        log_level = logging.DEBUG

    LoggerManager.set_level(log_level)

    # Execute command
    args.func(args)

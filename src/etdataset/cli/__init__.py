# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales
import argparse
import logging
import os
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from etdataset.api import (
    add_aux,
    create_dataset,
    download,
    download_aux,
    search,
    select,
)
from etdataset.logging import LoggerManager
from etdataset.provider import Collection
from etdataset.utils import (
    check_mgrs_format,
    get_bbox_from_mgrs_tile,
    get_bbox_from_roi,
    read_product_list,
)
from etdataset.writer import (
    export_matlab,
    write_dataset,
    write_results,
)

logger = LoggerManager.get_logger(__name__)


class CLIException(Exception):
    """
    Exception related to CLI arguments
    """


@dataclass
class CreateArgs(argparse.Namespace):
    vis: str
    tir: str | None
    roi: str | None
    tile: str | None
    resolution: float
    output: str
    aux: bool
    matlab: bool


@dataclass
class SelectArgs(argparse.Namespace):
    coll1: Collection
    coll2: Collection
    min_date: str
    max_date: str
    max_cloud_cover: int
    delta: str
    roi: str | None
    tile: str | None
    min_roi_overlap: int
    min_product_overlap: int
    output: str


@dataclass
class SearchArgs(argparse.Namespace):
    collection: Collection
    min_date: str
    max_date: str
    max_cloud_cover: int
    roi: str | None
    tile: str | None
    output: str


@dataclass
class DownloadArgs(argparse.Namespace):
    list: str
    output: str


class DownloadAuxArgs(argparse.Namespace):
    list: str
    output: str


# sub-command functions
def cli_create(args: CreateArgs) -> None:
    """
    Create dataset
    """
    logger.debug(f"Create arguments: {args}")
    # Check VIS product path
    if not os.path.isdir(args.vis):
        raise CLIException(f"VIS product not found ({args.vis})")
    logger.debug(f"VIS path: {args.vis}")

    # Check TIR path
    if args.tir is None:
        args.tir = args.vis
    if not os.path.isdir(args.tir):
        raise CLIException(f"TIR product not found ({args.tir})")
    logger.debug(f"TIR path: {args.tir}")

    # Check output path
    if not os.path.isdir(args.output):
        logger.debug(f"Create output path: {args.output}")
        os.makedirs(args.output, exist_ok=True)

    # Get ROI bounding box and CRS from tile or shapefile
    roi_bbox = None
    roi_crs = None
    if args.tile is not None:
        roi_bbox, roi_crs = get_bbox_from_mgrs_tile(args.tile)
    elif args.roi is not None:
        roi_bbox, roi_crs = get_bbox_from_roi(args.roi)

    # Create dataset
    data = create_dataset(
        vis_path=args.vis,
        tir_path=args.tir,
        roi_bbox=roi_bbox,
        roi_crs=roi_crs,
        resolution=args.resolution,
    )
    logger.debug("Create dataset: OK")

    # Add auxiliary data
    if args.aux:
        data = add_aux(data=data)
        logger.debug("Add auxiliary data: OK")

    # Write dataset
    if data is not None:
        if args.matlab:
            export_matlab(data, directory=args.output)
        else:
            write_dataset(data, directory=args.output)


def cli_select(args: SelectArgs) -> None:
    """
    Select products from 2 collections
    """
    logger.debug(f"Select arguments: {args}")

    # Process date
    try:
        min_date = datetime.strptime(args.min_date, "%Y-%m-%d")
    except ValueError:
        raise CLIException(
            "Error: The format for minimum acqsuisition "
            "date must be Year-Month-Day"
        )
    try:
        max_date = datetime.strptime(args.max_date, "%Y-%m-%d")
    except ValueError:
        raise CLIException(
            "Error: The format for maximum acquisition "
            "date must be Year-Month-Day"
        )
    if max_date < min_date:
        raise CLIException(
            "Maximum acquisition date must be more recent than minimum date"
        )
    try:
        _ = pd.Timedelta(args.delta)
    except ValueError:
        raise CLIException(
            "Error: The format for delta acquisition time is "
            "not recognized (ex: 1 day)"
        )
    if args.max_cloud_cover < 0 or args.max_cloud_cover > 100:
        raise CLIException(
            "Cloud cover criteria must be between 0 and 100 "
            f"(got : {args.max_cloud_cover})"
        )
    if args.min_roi_overlap < 0 or args.min_roi_overlap > 100:
        raise CLIException(
            "Min ROI overlap criteria must be between 0 and 100 "
            f"(got : {args.min_roi_overlap})"
        )
    if args.min_product_overlap < 0 or args.min_product_overlap > 100:
        raise CLIException(
            "Min product overlap criteria must be between 0 and 100 "
            f"(got : {args.min_product_overlap})"
        )
    # Check output path
    if not os.path.isdir(args.output):
        logger.debug(f"Create output path: {args.output}")
        os.makedirs(args.output, exist_ok=True)

    # Get ROI bounding box and CRS from tile or shapefile
    roi_bbox = None
    roi_crs = None
    if args.tile is not None:
        roi_bbox, roi_crs = get_bbox_from_mgrs_tile(args.tile)
    elif args.roi is not None:
        roi_bbox, roi_crs = get_bbox_from_roi(args.roi)

    products1, products2, matches = select(
        args.coll1,
        args.coll2,
        args.min_date,
        args.max_date,
        delta=args.delta,
        roi_bbox=roi_bbox,
        roi_crs=roi_crs,
        max_cloud_cover=args.max_cloud_cover,
        min_roi_overlap=args.min_roi_overlap,
        min_product_overlap=args.min_product_overlap,
    )

    # Write results
    logger.info(f"Number of products in {args.coll1} = {len(products1)}")
    logger.info(f"Number of products in {args.coll2} = {len(products2)}")
    write_results(matches, os.path.join(args.output, "matches.csv"))
    write_results(
        products1, os.path.join(args.output, f"products_{args.coll1}.csv")
    )
    write_results(
        products2, os.path.join(args.output, f"products_{args.coll2}.csv")
    )


def cli_search(args: SearchArgs) -> None:
    """
    Search products
    """
    logger.debug(f"Search arguments: {args}")

    # Check arguments
    try:
        min_date = datetime.strptime(args.min_date, "%Y-%m-%d")
    except ValueError:
        raise CLIException(
            "Error: The format for minimum acqsuisition "
            "date must be Year-Month-Day"
        )
    try:
        max_date = datetime.strptime(args.max_date, "%Y-%m-%d")
    except ValueError:
        raise CLIException(
            "Error: The format for maximum acquisition "
            "date must be Year-Month-Day"
        )
    if max_date < min_date:
        raise CLIException(
            "Maximum acquisition date must be more recent than minimum date"
        )
    if args.tile is not None:
        check_mgrs_format(args.tile)
    roi_bbox, roi_crs = None, None
    if args.roi is not None:
        if not os.path.isfile(args.roi):
            raise CLIException(f"ROI file not found ({args.roi})")
        roi_bbox, roi_crs = get_bbox_from_roi(args.roi)
    if args.max_cloud_cover < 0 or args.max_cloud_cover > 100:
        raise CLIException(
            "Cloud cover criteria must be "
            f"between 0 and 100 (got : {args.max_cloud_cover})"
        )

    # Search
    results = search(
        args.collection,
        args.min_date,
        args.max_date,
        args.tile,
        roi_bbox,
        roi_crs,
        args.max_cloud_cover,
    )

    # Write results
    write_results(results, args.output)


def cli_download(args: DownloadArgs) -> None:
    """
    Download products
    """
    logger.debug(f"Download arguments: {args}")

    if not os.path.isfile(args.list):
        raise CLIException(f"File with products lit not found ({args.roi})")

    try:
        products = read_product_list(args.list, 4326)
    # TODO: Correct catch blind exception
    except Exception as e:  # noqa
        raise CLIException(f"Error while reading product list: {e}")

    # Download
    download(products, args.output)


def cli_download_aux(args: DownloadAuxArgs) -> None:
    """
    Download auxiliary products
    """
    logger.debug(f"Download aux arguments: {args}")

    if not os.path.isfile(args.list):
        raise CLIException(f"File with product list not found ({args.list})")

    try:
        products = read_product_list(args.list, 4326)
    # TODO: Correct catch blind exception
    except Exception as e:  # noqa
        raise CLIException(f"Error while reading product list: {e}")

    # Download
    download_aux(
        products=products,
        output_dir=args.output,
    )


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
        help=(
            "Create dataset from Landsat/Ecostress/Sentinel2/HLS products. "
            "The dataset is resampling at 60m resolution "
            "and corresponds to a MGRS tile."
        ),
    )
    parser_create.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        action="store_true",
        help="Verbose mode",
    )
    parser_create.add_argument(
        "--vis",
        type=str,
        required=True,
        help="Path to VIS product (optical bands)",
    )
    parser_create.add_argument(
        "--tir",
        type=str,
        help=(
            "Path to TIR product (thermal bands), "
            "if not provided the VIS product is used for thermal bands."
        ),
    )
    parser_create.add_argument(
        "--radiation", type=str, required=True, help="Path to radiation product"
    )
    parser_create.add_argument(
        "-t", "--tile", type=str, help="Tile ID (Only for Landsat)"
    )
    parser_create.add_argument(
        "--output",
        type=str,
        help="Output dataset directory path (default: current directory)",
        default=os.getcwd(),
    )
    parser_create.add_argument(
        "--aux",
        dest="aux",
        action="store_true",
        help="Add auxiliary data",
    )
    parser_create.add_argument(
        "-m",
        "--matlab_export",
        dest="matlab",
        action="store_true",
        help="Write the dataset in Matlab format",
    )
    parser_create.set_defaults(func=cli_create)

    # create the parser for the "select" command
    parser_select = subparsers.add_parser(
        "select",
        help=(
            "Select product matches between 2 collections "
            "that satisfy required criteria"
        ),
    )
    parser_select.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        action="store_true",
        help="Verbose mode",
    )
    parser_select.add_argument(
        "--coll1",
        type=lambda arg: Collection[arg],
        choices=Collection,
        help="Fisrt collection of products",
    )
    parser_select.add_argument(
        "--coll2",
        type=lambda arg: Collection[arg],
        choices=Collection,
        help="Second collection of products",
    )
    parser_select.add_argument(
        "--min_date",
        help="Minimum date for acquisition in YYYY-MM-DD format",
        type=str,
        required=True,
        default=None,
    )
    parser_select.add_argument(
        "--max_date",
        help="Maximum date for acquisition in YYYY-MM-DD format",
        type=str,
        required=True,
        default=None,
    )
    parser_select.add_argument(
        "--max_cloud_cover",
        type=int,
        default=25,
        help="Maximum cloud cover (only for Landsat8 products) (default: 25)",
    )
    parser_select.add_argument(
        "--delta",
        type=str,
        default="3 days",
        help="Maximum time delta allowed between acquisitions (default: 3d)",
    )
    group = parser_select.add_mutually_exclusive_group(required=True)
    group.add_argument("-t", "--tile", type=str, help="Tile ID")
    group.add_argument(
        "-r",
        "--roi",
        type=str,
        help="Path of the region of interest in Shapefile format",
    )
    parser_select.add_argument(
        "--min_roi_overlap",
        type=int,
        default=50,
        help="Minimum overlap between ROI and a product (default: 50)",
    )
    parser_select.add_argument(
        "--min_product_overlap",
        type=int,
        default=40,
        help="Minimum overlap between two products (default: 40)",
    )
    parser_select.add_argument(
        "--output",
        type=str,
        help="Output dir (current directory)",
        default=os.getcwd(),
    )
    parser_select.set_defaults(func=cli_select)

    # create the parser for the "search" command
    parser_search = subparsers.add_parser(
        "search",
        help="Search products in a collection that satisfy required criteria",
    )
    parser_search.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        action="store_true",
        help="Verbose mode",
    )
    parser_search.add_argument(
        "-c",
        "--collection",
        type=lambda arg: Collection[arg],
        choices=Collection,
        help="Collection of products",
    )
    parser_search.add_argument(
        "--min_date",
        help="Minimum date for acquisition in YYYY-MM-DD format",
        type=str,
        required=True,
        default=None,
    )
    parser_search.add_argument(
        "--max_date",
        help="Maximum date for acquisition in YYYY-MM-DD format",
        type=str,
        required=True,
        default=None,
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
        "-r",
        "--roi",
        type=str,
        help="Path of the region of interest in Shapefile format",
    )
    parser_search.add_argument(
        "--output",
        type=str,
        help="CSV output file (default: results.csv)",
        default="results.csv",
    )
    parser_search.set_defaults(func=cli_search)

    # create the parser for the "download" command
    parser_download = subparsers.add_parser(
        "download", help="Download products from a list in CSV format"
    )
    parser_download.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        action="store_true",
        help="Verbose mode",
    )
    parser_download.add_argument(
        "-l",
        "--list",
        help="List of products (in CSV format)",
        type=str,
        required=True,
        default=None,
    )
    parser_download.add_argument(
        "--output",
        type=str,
        help="Download directory path (default: download)",
        default="download",
    )
    parser_download.set_defaults(func=cli_download)

    # create the parser for the "download_aux" command
    parser_download_aux = subparsers.add_parser(
        "download_aux",
        help=(
            "Download auxiliary products for a date "
            "and tile ID or from a list in CSV format"
        ),
    )
    parser_download_aux.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        action="store_true",
        help="Verbose mode",
    )
    parser_download_aux.add_argument(
        "-l",
        "--list",
        help="List of products (in CSV format)",
        required=True,
        type=str,
    )
    parser_download_aux.add_argument(
        "--output",
        type=str,
        help="Download directory path (default: current directory)",
        default=os.getcwd(),
    )
    parser_download_aux.set_defaults(func=cli_download_aux)

    return parser


def etdataset() -> None:
    """
    Entry point
    """
    # Parser arguments
    parser = get_parser()
    args = parser.parse_args()

    # Configure logging
    log_level = logging.INFO
    if args.verbose:
        log_level = logging.DEBUG

    LoggerManager.set_level(log_level)

    # Execute command
    args.func(args)

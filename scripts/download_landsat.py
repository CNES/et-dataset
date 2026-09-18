#!/usr/bin/env python
#
# SPDX-License-Identifier: AGPL-3.0-only
#
# Copyright (C) 2024 CESBIO / Centre National d'Etudes Spatiales
#
import argparse
import datetime as dt
import logging
import os
import sys

import rioxarray  # noqa # Use to activate rio attributes
from sensorsio.utils import bb_transform

from etdataset.api import download, download_aux, search
from etdataset.cli import CLIException
from etdataset.logging import LoggerManager
from etdataset.provider import Collection
from etdataset.utils import get_utm_bbox_from_roi

logger = LoggerManager.get_logger(__name__)


def download_landsat(
    roi: str,
    start_date: str,
    end_date: str,
    output: str,
    max_cloud_cover: float = 20,
    min_roi_coverage: float = 33,
):
    """
    Prepare landsat data
    """

    # Inputs

    # ROI
    roi_bbox, roi_crs = get_utm_bbox_from_roi(roi)
    logger.debug(f"ROI bbox: {roi_bbox}")
    logger.debug(f"ROI crs: {roi_crs}")
    latlon_bbox = bb_transform(str(roi_crs), "EPSG:4326", roi_bbox)
    logger.debug(f"ROI bbox (lat/lon): {latlon_bbox}")

    output_path = output
    os.makedirs(output_path, exist_ok=True)

    # Search

    res = search(
        Collection.LANDSAT,
        start_date,
        end_date,
        roi_bbox=roi_bbox,
        roi_crs=roi_crs,
        max_cloud_cover=max_cloud_cover,
        min_roi_overlap=min_roi_coverage,
    )
    logger.info(f"Number of products found: {len(res)}")
    logger.info("Search: OK")
    if len(res) == 0:
        logger.warning("No product found")
        sys.exit(1)

    # Download products
    download(products=res, output_dir=output_path)
    logger.info("Download Landsat products: OK")

    # Download auxiliary data
    download_aux(products=res, output_dir=output_path)
    logger.info("Download auxiliary data: OK")


def get_parser() -> argparse.ArgumentParser:
    """
    Generate argument parser for cli
    """
    # create the top-level parser
    parser = argparse.ArgumentParser(
        description="Download Landsat product and auxiliary data"
    )

    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        action="store_true",
        help="Verbose mode",
    )
    parser.add_argument(
        "-r",
        "--roi",
        type=str,
        help="Path of the region of interest in Shapefile format",
        required=True,
    )
    parser.add_argument(
        "-s",
        "--start-date",
        type=str,
        help="Start date (YYYY-MM-DD)",
        required=True,
    )
    parser.add_argument(
        "-e",
        "--end-date",
        type=str,
        help="End date (YYYY-MM-DD)",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Output dataset directory path (default: current directory)",
        default=os.getcwd(),
    )
    parser.add_argument(
        "--max_cloud_cover",
        type=int,
        default=25,
        help="Maximum cloud cover (default: 25)",
    )
    parser.add_argument(
        "--min_roi_overlap",
        type=int,
        default=33,
        help="Minimum overlap between ROI and a product (default: 33)",
    )

    return parser


if __name__ == "__main__":
    # Parser arguments
    parser = get_parser()
    args = parser.parse_args()

    # Configure logging
    log_level = logging.INFO
    if args.verbose:
        log_level = logging.DEBUG

    LoggerManager.set_level(log_level)

    # Check arguments
    logger.debug(f"Arguments: {args}")
    if not os.path.isfile(args.roi):
        raise FileNotFoundError(f"File not found {args.roi}")
    try:
        min_date = dt.datetime.strptime(args.start_date, "%Y-%m-%d")
    except ValueError:
        raise CLIException(
            "Error: The format for minimum acqsuisition "
            "date must be Year-Month-Day"
        )
    try:
        max_date = dt.datetime.strptime(args.end_date, "%Y-%m-%d")
    except ValueError:
        raise CLIException(
            "Error: The format for maximum acquisition "
            "date must be Year-Month-Day"
        )
    if max_date < min_date:
        raise CLIException(
            "Maximum acquisition date must be more recent than minimum date"
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
    if not os.path.isdir(args.output):
        logger.debug(f"Create output path: {args.output}")
        os.makedirs(args.output, exist_ok=True)

    # Run
    download_landsat(
        args.roi,
        args.start_date,
        args.end_date,
        args.output,
        args.max_cloud_cover,
        args.min_roi_overlap,
    )

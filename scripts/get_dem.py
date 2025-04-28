#!/usr/bin/env python
# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales
import argparse
import logging

from etdataset.dem import get_dem_from_roi
from etdataset.logging import LoggerManager
from etdataset.utils import get_utm_bbox_from_roi
from etdataset.writer import write_to_tif

logger = LoggerManager.get_logger(__name__)


def get_parser() -> argparse.ArgumentParser:
    """
    Generate argument parser for cli
    """
    # create the top-level parser
    parser = argparse.ArgumentParser(description="Get DEM")

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

    return parser


def get_dem() -> None:
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

    # Get bbox and crs
    roi_bbox, roi_crs = get_utm_bbox_from_roi(args.roi)

    # Get DEM
    dem = get_dem_from_roi(roi_bbox=roi_bbox, roi_crs=roi_crs)

    # Write
    write_to_tif(dem, "dem.tif")


if __name__ == "__main__":
    get_dem()

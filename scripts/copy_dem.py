#!/usr/bin/env python
# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales
import argparse
import logging
import os

from etdataset.dem import copy_tiles
from etdataset.logging import LoggerManager
from etdataset.utils import get_bbox_from_roi

logger = LoggerManager.get_logger(__name__)


def get_parser() -> argparse.ArgumentParser:
    """
    Generate argument parser for cli
    """
    # create the top-level parser
    parser = argparse.ArgumentParser(description="Copy DEM tiles")

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
        "--mnt_path",
        type=str,
        help="Path to the DEM directory",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output",
        default=os.getcwd(),
        type=str,
        help="Output directory",
        required=False,
    )

    return parser


def copy_dem() -> None:
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

    # Check
    if not os.path.isfile(args.roi):
        raise FileNotFoundError(f"File not found {args.roi}")
    if not os.path.isdir(args.mnt_path):
        raise FileNotFoundError(f"MNT directory not found {args.mnt_path}")
    os.makedirs(args.output, exist_ok=True)

    # Get bbox and crs
    roi_bbox, roi_crs = get_bbox_from_roi(args.roi)

    # Get DEM tiles
    copy_tiles(
        roi_bbox=roi_bbox,
        roi_crs=roi_crs,
        base_dir=args.mnt_path,
        output_dir=args.output,
    )


if __name__ == "__main__":
    copy_dem()

#!/usr/bin/env python
# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales

import argparse
import datetime as dt
import os

from etdataset.api import (
    prepare_daily_et_timeseries,
    prepare_daily_radiation_timeseries,
)
from etdataset.cli import CLIException
from etdataset.logging import LoggerManager
from etdataset.utils import get_utm_bbox_from_roi

logger = LoggerManager.get_logger(__name__)


def get_parser() -> argparse.ArgumentParser:
    """
    Generate argument parser for cli
    """
    # create the top-level parser
    parser = argparse.ArgumentParser("Prepare timeseries")

    parser.add_argument(
        "-s",
        "--start_date",
        type=str,
        help="start date (YY-MM-DD)",
        required=True,
    )

    parser.add_argument(
        "-e",
        "--end_date",
        type=str,
        help="end date (YY-MM-DD)",
        required=True,
    )

    parser.add_argument(
        "-r",
        "--roi",
        type=str,
        help="Path of the region of interest in Shapefile format",
        required=True,
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Output directory",
        default=os.getcwd(),
    )

    return parser


def prepare_timeseries() -> None:
    # Parser arguments
    parser = get_parser()
    args = parser.parse_args()

    # Check
    try:
        min_date = dt.datetime.strptime(args.start_date, "%Y-%m-%d")
    except ValueError:
        raise CLIException(
            "Error: The format for start date must be Year-Month-Day"
        )
    try:
        max_date = dt.datetime.strptime(args.end_date, "%Y-%m-%d")
    except ValueError:
        raise CLIException(
            "Error: The format for end date must be Year-Month-Day"
        )
    if max_date < min_date:
        raise CLIException("End date must be more recent than start date")
    if not os.path.isfile(args.roi):
        raise FileNotFoundError(f"File not found {args.roi}")
    if not os.path.isdir(args.output):
        logger.debug(f"Create output path: {args.output}")
        os.makedirs(args.output, exist_ok=True)

    # Run
    resolution = 3000
    roi_bbox, roi_crs = get_utm_bbox_from_roi(args.roi)
    # Prepare daily radiation files
    prepare_daily_radiation_timeseries(
        start_date=args.start_date,
        end_date=args.end_date,
        roi_bbox=roi_bbox,
        roi_crs=roi_crs,
        resolution=resolution,
        output=args.output,
    )

    # Prepare et single dat files
    prepare_daily_et_timeseries(
        start_date=args.start_date,
        end_date=args.end_date,
        roi_bbox=roi_bbox,
        roi_crs=roi_crs,
        resolution=resolution,
        output=args.output,
    )


if __name__ == "__main__":
    prepare_timeseries()

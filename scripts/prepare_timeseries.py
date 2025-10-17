#!/usr/bin/env python
import argparse
import os

from etdataset.api import prepare_daily_radiation, prepare_et_single_date


def get_parser() -> argparse.ArgumentParser:
    """
    Generate argument parser for cli
    """
    # create the top-level parser
    parser = argparse.ArgumentParser("Prepare timeseries")

    parser.add_argument(
        "--t1",
        type=str,
        help="start date in str type",
        required=True,
    )

    parser.add_argument(
        "--t2",
        type=str,
        help="end date in str type",
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
        default=os.getcwd(),
        type=str,
        help="Output directory",
        required=False,
    )

    return parser


def prepare_timeseries() -> None:
    # Parser arguments
    parser = get_parser()
    args = parser.parse_args()

    # Prepare daily radiation files
    prepare_daily_radiation(args.t1, args.t2, args.roi, args.output)

    # Prepare et single dat files
    prepare_et_single_date(args.t1, args.t2, args.roi, args.output)


if __name__ == "__main__":
    prepare_timeseries()

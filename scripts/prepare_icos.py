#!/usr/bin/env python
# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales

#######################
# Prepare ICOS data
#######################

# Imports
import argparse
import logging
import os

import pandas as pd

from etdataset.icos import (
    compute_dewpoint_temp,
    download_icos_station,
    filter_valid_data,
    get_csv_with_valid_icos_stations,
    get_stations_config,
    read_csv_data,
    save_station_data,
)
from etdataset.logging import LoggerManager

logger = LoggerManager.get_logger(__name__)


def prepare_icos_stations(
    stations: list[str] | str = "all",
    data_dir: str | None = None,
    out_dir: str = "icos_data",
):
    """
    Prepare ICOS data
    """

    # download files
    download_icos_station(stations, data_dir)

    # Valid stations only
    csv_path = get_csv_with_valid_icos_stations()
    df = pd.read_csv(csv_path)
    valid_stations_list = df["id"].tolist()

    if stations == "all":
        ids = valid_stations_list
    elif isinstance(stations, str) and stations != "all":
        ids = [stations]
    elif isinstance(stations, list) and stations != "all":
        ids = stations

    invalid_stations = [s for s in ids if s not in valid_stations_list]
    if invalid_stations:
        raise ValueError(f"Invalid station ID given: {invalid_stations}.")
    csv_paths = []

    # Iterates over a list of ICOS station IDs
    for station_id in ids:
        # for each stations, get it configuration (latitude, longitude,
        # elevation)
        cfg = get_stations_config(station_id)
        # From the station's CSV, get its data
        data = read_csv_data(cfg)
        # Keep only valid data
        data = filter_valid_data(data)

        # Compute dewpoint temperature with Air temperature and Relative
        # humidity
        td = compute_dewpoint_temp(data["TA"], data["RH"])
        # add dewpoint temperature
        data["TD"] = td
        # Save datas of station in csv
        csv_saved = save_station_data(cfg, data, out_dir)
        csv_paths.append(csv_saved)

    logger.info(f"All stations are completed {csv_paths}")
    return csv_paths


def get_parser() -> argparse.ArgumentParser:
    """
    Generate argument parser for cli
    """
    parser = argparse.ArgumentParser(description="Prepare ICOS data")

    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        action="store_true",
        help="Verbose mode",
    )
    parser.add_argument(
        "-s",
        "--stations",
        nargs="+",
        type=str,
        help="ICOS ID Stations to prepare (ex: CH-Dav, DE-Geb)",
        required=True,
    )
    parser.add_argument(
        "-d",
        "--data_dir",
        type=str,
        help="Downloaded ICOS data directory path",
        default="ICOS",
        required=False,
    )
    parser.add_argument(
        "-o",
        "--out_dir",
        type=str,
        help="Output directory path where prepared ICOS data will be stocked",
        default="icos_data",
        required=False,
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

    if not os.path.isdir(args.out_dir):
        logger.debug(f"Create output directory: {args.out_dir}")
        os.makedirs(args.out_dir, exist_ok=True)
    # Run
    prepare_icos_stations(
        args.stations,
        args.data_dir,
        args.out_dir,
    )

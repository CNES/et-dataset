#!/usr/bin/env python
# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales

#######################
# Prepare Fluxnet data
#######################

# Imports
import argparse
import logging
import os

from etdataset.fluxnet import (
    download_fluxnet_data,
    get_fluxnet_archive,
    get_fluxnet_stations_config,
    get_fluxnet_stations_list,
    read_fluxnet_data,
    save_fluxnet_station,
)
from etdataset.logging import LoggerManager

logger = LoggerManager.get_logger(__name__)


def prepare_fluxnet_stations(
    stations: list[str] | str = "all",
    data_dir: str | None = None,
    out_dir: str = "fluxnet_processed",
):
    """
    Prepare ICOS data
    """
    # download files

    download_fluxnet_data(stations, output_dir=data_dir)
    # Valid stations only
    valid_stations_list = get_fluxnet_stations_list()
    if stations == "all":
        ids = valid_stations_list
    elif isinstance(stations, str):
        ids = [stations]
    else:  # list
        ids = stations

    invalid_stations = [s for s in ids if s not in valid_stations_list]
    if invalid_stations:
        raise ValueError(f"Invalid station ID given: {invalid_stations}.")

    csv_paths = []

    # Iterates over a list of Fluxnet station IDs
    for station_id in ids:
        # for each stations, get it configuration (latitude, longitude,
        # elevation)
        cfg = get_fluxnet_stations_config(station_id)
        logger.info(f"cfg = {cfg}")
        if data_dir is not None:
            path = get_fluxnet_archive(cfg.id, data_dir=data_dir)
        logger.info(f"path = {path}")

        if path is None:
            logger.warning(f"No archive for station {station_id}")
            continue

        data = read_fluxnet_data(path)
        csv_saved = save_fluxnet_station(cfg, data, out_dir)
        csv_paths.append(csv_saved)

    logger.info(f"All stations are completed {csv_paths}")
    return csv_paths


def get_parser() -> argparse.ArgumentParser:
    """
    Generate argument parser for cli
    """
    parser = argparse.ArgumentParser(description="Prepare Fluxnet data")

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
        help="Fluxnet ID Stations to prepare",
        required=True,
    )
    parser.add_argument(
        "-d",
        "--data_dir",
        type=str,
        help="Downloaded Fluxnet data directory path",
        default=os.getcwd(),
        required=False,
    )
    parser.add_argument(
        "-o",
        "--out_dir",
        type=str,
        help="Output directory path where prepared Fluxnet data will be stocked",  # noqa: E501
        default="fluxnet_processed",
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
    if args.stations == ["all"]:
        stations_to_download = "all"
    else:
        stations_to_download = args.stations
    # Run
    prepare_fluxnet_stations(
        stations_to_download,
        args.data_dir,
        args.out_dir,
    )

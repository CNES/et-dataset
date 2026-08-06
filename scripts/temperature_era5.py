#!/usr/bin/env python

# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales

#######################
# Air Temperature and Dew point Temperature rescaling
#######################

# Imports
import argparse
import datetime as dt
import logging
import os
from multiprocessing import Process

import pandas as pd
import rioxarray  # noqa # Use to activate rio attributes
import xarray as xr

from etdataset.cli import CLIException
from etdataset.era5 import read
from etdataset.icos import get_csv_with_valid_icos_stations, get_stations_config
from etdataset.logging import LoggerManager
from etdataset.temperature import (
    save_ta_td_csv,
)
from etdataset.utils import (
    filter_dataset_by_hours,
    generate_dates,
    generate_hours,
    get_ta_td_celsius_at_location,
    normalize_longitude_latitude,
)

logger = LoggerManager.get_logger(__name__)


def create_era5_sub_dataset(era5_xrds: xr.Dataset) -> xr.Dataset:
    """
    Create a subset of an ERA5 xarray Dataset containing only selected variables
    and rename them to standardized names

    It renames 't2m' and 'd2m' to 'ta' and 'tdp'.

    Parameters
    ----------
    era5_xrds : xr.Dataset
        The original ERA5 dataset containing multiple data variables.

    Returns
    -------
    era5_sub : xr.Dataset
        A new xarray Dataset
    """
    era5_sub = era5_xrds[["t2m", "d2m"]]
    era5_sub = era5_sub.rename(
        {
            "t2m": "ta",
            "d2m": "td",
        }
    )
    return era5_sub


def run_stations_process(
    start_date: dt.date,
    end_date: dt.date,
    step_date: int,
    station: str,
    list_hours: list[dt.time],
    data_path: str,
    output: str,
):
    """
    Process one station

    Parameters
    ----------
    start_date: dt.date
        Start date
    end_date: dt.date
        End date
    step_date: int
        Step for date
    station: str
        Station name
    list_hours: list[dt.time]
        List of hours
    data_path: str
        Data path
    output: str
        Output directory
    """
    # Get metadata of the station
    logger.info(f"Current station : {station}")
    cfg = get_stations_config(station)
    for d in generate_dates(start_date, end_date, step_date):
        logger.info(f"Current date : {d}")
        product_path = os.path.join(
            data_path,
            "ERA5_data",
            f"download_era5_{d.strftime('%Y-%m-%d')}.zip",
        )
        era5_xrds = read(product=product_path)
        if era5_xrds is None:
            logger.warning(
                "Skipping date %s because ERA5 file is unavailable", d
            )
            continue
        era5_xrds = normalize_longitude_latitude(era5_xrds)

        era5_filtered = filter_dataset_by_hours(era5_xrds, d, list_hours)

        # FOR ERA5 data only ###################################################
        era5_sub = create_era5_sub_dataset(era5_filtered)
        # get ta and td in celsius at station location
        ta, td = get_ta_td_celsius_at_location(era5_sub, cfg.lat, cfg.lon)
        # logger.info(f"TA:{ta} and TD :{td}")
        ds_era5_grid = xr.Dataset(
            {
                "ta": ta,
                "td": td,
            }
        )
        # save data as csv
        save_ta_td_csv(ds_era5_grid, cfg, output, name_dir="csv_era5_grid")


def generate_timeseries_for_stations_multiprocess(
    start_date: dt.date,
    end_date: dt.date,
    data_path: str,
    output: str,
    step_day: int = 1,
    hour_start: int = 0,
    hour_end: int = 22,
    hour_step: int = 2,
    stations: list[str] | str = "all",
):
    """
    Process stations

    Parameters
    ----------
    start_date: dt.date
        Start date
    end_date: dt.date
        End date
    data_path: str
        Data path
    output: str
        Output directory
    step_day: int
        Step for date
    hour_start: int
        Start hour
    hour_end: int
        End hour
    hour_step: int
        Step for hour
    stations: list[str] | str
        Station names
    """
    # download_date_by_date(start_date, end_date, ERA5Dataset.ERA5, output="out")  # noqa: E501
    list_hours = generate_hours(hour_start, hour_end, hour_step)

    # Station
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

    procs = []
    for station_id in ids:
        p = Process(
            target=run_stations_process,
            args=(
                start_date,
                end_date,
                step_day,
                station_id,
                list_hours,
                data_path,
                output,
            ),
        )
        procs.append(p)
        p.start()
    # Block until all station processes complete
    for p in procs:
        p.join()


def get_parser() -> argparse.ArgumentParser:
    """
    Generate argument parser for cli
    """
    # create the top-level parser
    parser = argparse.ArgumentParser(
        description="Temperature rescaling comparison"
    )

    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        action="store_true",
        help="Verbose mode",
    )
    parser.add_argument(
        "-s",
        "--start_date",
        type=str,
        help="Start date (dt.date(YYYY,MM,DD))",
        required=True,
    )
    parser.add_argument(
        "-e",
        "--end-date",
        type=str,
        help="End date (dt.date(YYYY,MM,DD))",
        required=True,
    )

    parser.add_argument(
        "-d",
        "--data_path",
        type=str,
        help="Directory of ERA5 data",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Output directory",
        default=os.getcwd(),
    )

    parser.add_argument(
        "-st_d",
        "--step-day",
        type=int,
        help="Day step",
        default=1,
    )
    parser.add_argument(
        "-hs",
        "--hour-start",
        type=int,
        help="Hour start",
        default=8,
    )
    parser.add_argument(
        "-he",
        "--hour-end",
        type=int,
        help="Hour end",
        default=22,
    )
    parser.add_argument(
        "-h_st",
        "--hour-step",
        type=int,
        help="Hour step",
        default=2,
    )

    parser.add_argument(
        "-ids",
        "--id-stations",
        nargs="+",
        type=str,
        help="List of stations to process",
        required=True,
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

    # Dates (YYYY-MM-DD)
    try:
        start_date = dt.date.fromisoformat(args.start_date)
    except ValueError:
        raise CLIException(
            "Error: The format for minimum acquisition date must be YYYY-MM-DD"
        )

    try:
        end_date = dt.date.fromisoformat(args.end_date)
    except ValueError:
        raise CLIException(
            "Error: The format for maximum acquisition date must be YYYY-MM-DD"
        )

    if end_date < start_date:
        raise CLIException(
            "Maximum acquisition date must be more recent than minimum date"
        )

    # Output directory
    if not os.path.isdir(args.output):
        logger.debug(f"Create output path: {args.output}")
        os.makedirs(args.output, exist_ok=True)

    if args.id_stations == ["all"]:
        stations = "all"
    else:
        stations = args.id_stations
    # Run
    generate_timeseries_for_stations_multiprocess(
        start_date=start_date,
        end_date=end_date,
        data_path=args.data_path,
        output=args.output,
        step_day=args.step_day,
        hour_start=args.hour_start,
        hour_end=args.hour_end,
        hour_step=args.hour_step,
        stations=stations,
    )

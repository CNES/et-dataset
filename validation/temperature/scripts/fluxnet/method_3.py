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

import rioxarray  # noqa # Use to activate rio attributes
import xarray as xr
from pyproj import CRS

from etdataset.cli import CLIException
from etdataset.dem import get_dem_from_roi
from etdataset.era5 import ERA5Dataset
from etdataset.fluxnet import (
    get_fluxnet_stations_config,
    get_fluxnet_stations_list,
)
from etdataset.logging import LoggerManager
from etdataset.temperature import (
    RescalTempMethod,
    TempVariable,
    add_temp,
    save_ta_td_csv,
)
from etdataset.utils import (
    generate_dates,
    generate_hours,
    get_ta_td_celsius_at_location,
    work_area_from_coord_point,
)

logger = LoggerManager.get_logger(__name__)


def run_stations_process_method_3(
    start_date: dt.date,
    end_date: dt.date,
    step_date: int,
    station: str,
    list_hours: list[dt.time],
    mnt_path: str,
    data_path: str,
    output: str,
):  # Get metadats of the station
    logger.info(f"Current station : {station}")

    cfg = get_fluxnet_stations_config(station)
    for d in generate_dates(start_date, end_date, step_date):
        for h in list_hours:
            date = dt.datetime.combine(d, h)
            logger.info(f"Processing {date}")

            # ROI
            roi_bbox_utm, roi_crs_utm = work_area_from_coord_point(
                cfg.lat, cfg.lon, 25000, 25000, CRS.from_epsg(4326)
            )["utm"]

            # DEM
            data = get_dem_from_roi(
                roi_bbox=roi_bbox_utm,
                roi_crs=roi_crs_utm,
                base_dir=mnt_path,
                resolution=60,
            )

            # attributes nécessaires
            data.attrs["vis_date"] = date.date()
            data.attrs["vis_time"] = date.time()
            data.attrs["tir_date"] = date.date()
            data.attrs["tir_time"] = date.time()

            updated = add_temp(
                data=data,
                path=data_path,
                dataset=ERA5Dataset.ERA5,
                variables=[TempVariable.TD, TempVariable.TA],
                method=RescalTempMethod.LR_PROFILE_ALT_DEP,
            )
            T_station, Td_station = get_ta_td_celsius_at_location(
                updated, cfg.lat, cfg.lon
            )

            ds_out = xr.Dataset(
                {
                    "ta": ("time", [float(T_station.item())]),
                    "tdp": ("time", [float(Td_station.item())]),
                },
                coords={"time": [date]},
            )

            save_ta_td_csv(
                ds_out,
                cfg,
                output,
                name_dir="csv_era5_rescaled",
            )


def generate_timeseries_for_stations_multiprocess(
    start_date: dt.date,
    end_date: dt.date,
    mnt_path: str,
    data_path: str,
    output: str,
    step_day: int = 1,
    hour_start: int = 0,
    hour_end: int = 22,
    hour_step: int = 2,
    stations: list[str] | str = "all",
):
    # download_date_by_date(start_date, end_date, ERA5Dataset.ERA5, output="out")  # noqa: E501
    list_hours = generate_hours(hour_start, hour_end, hour_step)

    # Station
    valid_stations_list = get_fluxnet_stations_list()

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
            target=run_stations_process_method_3,
            args=(
                start_date,
                end_date,
                step_day,
                station_id,
                list_hours,
                mnt_path,
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
        "-p",
        "--mnt_path",
        type=str,
        help="Directory of DEM tiles",
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
        default="method_3",
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

    # DEM directory
    if args.mnt_path is not None and not os.path.isdir(args.mnt_path):
        raise FileNotFoundError(f"DEM directory not found {args.mnt_path}")

    if args.id_stations == ["all"]:
        stations = "all"
    else:
        stations = args.id_stations
    # Run
    generate_timeseries_for_stations_multiprocess(
        start_date=start_date,
        end_date=end_date,
        mnt_path=args.mnt_path,
        data_path=args.data_path,
        output=args.output,
        step_day=args.step_day,
        hour_start=args.hour_start,
        hour_end=args.hour_end,
        hour_step=args.hour_step,
        stations=stations,
    )

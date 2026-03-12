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

import numpy as np
import pandas as pd
import rioxarray  # noqa # Use to activate rio attributes
import xarray as xr
from pyproj import CRS

from etdataset.cli import CLIException
from etdataset.dem import get_dem_from_roi
from etdataset.era5 import ERA5Dataset
from etdataset.icos import (
    get_csv_with_valid_icos_stations,
    get_stations_config,
)
from etdataset.logging import LoggerManager
from etdataset.utils import (
    work_area_from_coord_point,
)
from etdataset.validation_temp.temperature_rescaling import (
    add_dewpoint_to_ds,
    compute_lapse_rate_from_2_levels,
    filter_dataset_by_hours,
    filter_dataset_by_location,
    filter_dataset_by_pressure_levels,
    generate_dates,
    generate_hours,
    get_ta_td_celsius_at_location,
    prepare_temperature_inputs,
    read_era5_file,
    save_ta_td_csv,
    temperature_rescaling_variable_lapse_rate,
)

logger = LoggerManager.get_logger(__name__)

G_CST = 9.80665


def run_stations_process_method_5(
    start_date: dt.date,
    end_date: dt.date,
    step_date: int,
    station: str,
    list_hours: list[dt.time],
    mnt_path: str,
    data_path: str,
    output: str,
):
    logger.info(f"Current station : {station}")
    cfg = get_stations_config(station)

    for d in generate_dates(start_date, end_date, step_date):
        logger.info(f"Date: {d}")
        logger.info(f"Station elevation: {cfg.elev}")

        # READ ERA5 PRESSURE
        era5_pressure = read_era5_file(d, ERA5Dataset.ERA5PRESSURE, data_path)
        if era5_pressure is None:
            logger.warning("Skipping date %s (ERA5PRESSURE unavailable)", d)
            continue
        # FILTER ERA5 PRESSURE BY HOURS
        era5_pressure = filter_dataset_by_hours(era5_pressure, d, list_hours)
        era5_pressure = filter_dataset_by_pressure_levels(
            era5_pressure,
            [
                "700",
                "725",
                "750",
                "775",
                "800",
                "825",
                "850",
                "875",
                "900",
                "925",
                "950",
                "975",
            ],
        )

        # FILTER ERA5 PRESSURE ON LOCATION
        era5_filt_location = filter_dataset_by_location(
            era5_pressure, cfg.lat, cfg.lon
        )

        # READ ERA5 SURFACE
        era5_data = read_era5_file(d, ERA5Dataset.ERA5, "out")
        if era5_data is None:
            logger.warning("Skipping date %s (ERA5 unavailable)", d)
            continue
        # FILTER ERA5 BY HOURS
        era5_surface = filter_dataset_by_hours(era5_data, d, list_hours)

        # GET DEM
        roi_bbox_utm, roi_crs_utm = work_area_from_coord_point(
            cfg.lat, cfg.lon, 10000, 10000, CRS.from_epsg(4326)
        )["utm"]
        # get dem from the roi
        dem = get_dem_from_roi(
            roi_bbox=roi_bbox_utm,
            roi_crs=roi_crs_utm,
            base_dir=mnt_path,
            resolution=60,
        )

        # GET HEIGHT OF THE STATION
        x, y = (
            work_area_from_coord_point(
                cfg.lat, cfg.lon, 0, 0, CRS.from_epsg(4326)
            )["utm"][0].left,
            work_area_from_coord_point(
                cfg.lat, cfg.lon, 0, 0, CRS.from_epsg(4326)
            )["utm"][0].bottom,
        )
        z_station_dem = dem["height"].sel(x=x, y=y, method="nearest").values

        ta_out = []
        td_out = []

        # LOOP OVER TIME
        for t in era5_pressure.time.values:
            hourly_p = era5_pressure.sel(time=t)
            hourly_p_filt_location = era5_filt_location.sel(time=t)
            hourly_surface = era5_surface.sel(time=[t])

            z_levels = hourly_p_filt_location["z"].values / G_CST

            ind = np.searchsorted(z_levels, z_station_dem)
            if ind == 0:
                # si station inférieure au premier niveau, on utilise Tref=Tera5
                # et lapse_rate entre 925 Pa et 850 Pa
                era5_filt_pressure = filter_dataset_by_pressure_levels(
                    hourly_p_filt_location,
                    ["925", "850"],
                )
                logger.info(f"era5 filtré par pression: {era5_filt_pressure}")
                era5_add_dewpoint = add_dewpoint_to_ds(era5_filt_pressure)
                lr_t = compute_lapse_rate_from_2_levels(era5_add_dewpoint)
                lr_td = compute_lapse_rate_from_2_levels(
                    era5_add_dewpoint, "td"
                )
                era5_data = hourly_surface
                dataset = ERA5Dataset.ERA5
            else:
                ind_bot, ind_top = ind - 1, ind

                # LAPSE RATE
                # récupérer les niveaux de pression correspondants
                p_levels = hourly_p_filt_location["pressure_level"].values
                p_bot = str(p_levels[ind_bot])
                p_top = str(p_levels[ind_top])

                # filtrer ERA5 sur ces deux niveaux
                era5_filt_pressure = filter_dataset_by_pressure_levels(
                    hourly_p_filt_location,
                    [p_bot, p_top],
                )
                # ajouter Td
                era5_add_dewpoint = add_dewpoint_to_ds(era5_filt_pressure)
                ####### Get lapse rates
                lr_t = compute_lapse_rate_from_2_levels(era5_add_dewpoint)
                lr_td = compute_lapse_rate_from_2_levels(
                    era5_add_dewpoint, "td"
                )

                # dataset au niveau bottom uniquement
                era5_data = filter_dataset_by_pressure_levels(
                    hourly_p,
                    [p_bot],
                ).expand_dims(time=[t])
                era5_data = add_dewpoint_to_ds(era5_data)

                dataset = ERA5Dataset.ERA5PRESSURE

            # prepare inputs for rescaling
            new_dem, era5_dem, updated_data = prepare_temperature_inputs(
                data=dem,
                era5_data=era5_data,
                dataset=dataset,
            )
            # rescaling
            updated = temperature_rescaling_variable_lapse_rate(
                updated_data=updated_data,
                dataset=dataset,
                dem=new_dem,
                era5_data=era5_data,
                era5_dem=era5_dem,
                lr_ta=lr_t,
                lr_tdp=lr_td,
            )
            T_station, Td_station = get_ta_td_celsius_at_location(updated, cfg)

            ta_out.append(float(T_station.item()))
            td_out.append(float(Td_station.item()))

        ta_da = xr.DataArray(
            ta_out,
            coords={"time": era5_pressure.time},
            dims=["time"],
        )

        td_da = xr.DataArray(
            td_out,
            coords={"time": era5_pressure.time},
            dims=["time"],
        )
        # BUILD OUTPUT DATASET
        ta_da = xr.DataArray(
            ta_out,
            coords={"time": era5_pressure.time},
            dims=["time"],
        )

        td_da = xr.DataArray(
            td_out,
            coords={"time": era5_pressure.time},
            dims=["time"],
        )

        ds_out = xr.Dataset(
            {
                "ta": ta_da,
                "tdp": td_da,
            }
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
            target=run_stations_process_method_5,
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
        default="method_5",
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
        mnt_path=args.mnt_path,
        data_path=args.data_path,
        output=args.output,
        step_day=args.step_day,
        hour_start=args.hour_start,
        hour_end=args.hour_end,
        hour_step=args.hour_step,
        stations=stations,
    )

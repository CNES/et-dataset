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
from scipy.interpolate import interp1d

from etdataset.cli import CLIException
from etdataset.dem import get_dem_from_roi
from etdataset.era5 import (
    ERA5Dataset,
    get_era5_dem,
)
from etdataset.icos import (
    get_csv_with_valid_icos_stations,
    get_stations_config,
    kelvin_to_celsius,
)
from etdataset.logging import LoggerManager
from etdataset.utils import (
    work_area_from_coord_point,
)
from etdataset.validation_temp.temperature_rescaling import (
    compute_dewpoint_temp_from_e,
    compute_vapor_pressure,
    filter_dataset_by_hours,
    filter_dataset_by_location,
    filter_dataset_by_pressure_levels,
    generate_dates,
    generate_hours,
    get_saturation_vapor_pressure,
    normalize_longitude_latitude,
    read_era5_file,
    save_ta_td_csv,
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

        era5_pressure = normalize_longitude_latitude(era5_pressure)

        era5_pressure = filter_dataset_by_hours(era5_pressure, d, list_hours)
        era5_pressure = filter_dataset_by_location(
            era5_pressure, cfg.lat, cfg.lon
        )
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

        # READ ERA5 SURFACE
        era5_data = read_era5_file(d, ERA5Dataset.ERA5, data_path)
        if era5_data is None:
            logger.warning("Skipping date %s (ERA5PRESSURE unavailable)", d)
            continue
        era5_data = normalize_longitude_latitude(era5_data)

        era5_surface = filter_dataset_by_hours(era5_data, d, list_hours)
        era5_surface = filter_dataset_by_location(
            era5_surface, cfg.lat, cfg.lon
        )

        # GET ELEVATION AT THE LOCATION FROM DEM
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
        x, y = (
            work_area_from_coord_point(
                cfg.lat, cfg.lon, 0, 0, CRS.from_epsg(4326)
            )["utm"][0].left,
            work_area_from_coord_point(
                cfg.lat, cfg.lon, 0, 0, CRS.from_epsg(4326)
            )["utm"][0].bottom,
        )
        z_station = dem["height"].sel(x=x, y=y, method="nearest")
        # GET ERA5 ELEVATION
        z_surface = get_era5_dem()
        z_surface = xr.DataArray(
            z_surface.data,
            dims=("latitude", "longitude"),
            coords={
                "latitude": era5_data.latitude,
                "longitude": era5_data.longitude,
            },
        ).rio.write_crs(CRS(4326))
        z_surface_ds = z_surface.to_dataset(name="elevation")
        z_surface_ds = normalize_longitude_latitude(z_surface_ds)
        z_surface = z_surface_ds["elevation"]
        z_surface = z_surface.sel(
            latitude=cfg.lat, longitude=cfg.lon, method="nearest"
        )

        # DIFFERENCE
        delta_z = z_station - z_surface
        logger.info(
            f"Altitude difference station - ERA5 surface: {delta_z.values}"
        )

        ta_out = []
        td_out = []

        # LOOP OVER TIME
        for t in era5_pressure.time.values:
            hourly_pressure = era5_pressure.sel(time=t)
            hourly_surface = era5_surface.sel(time=t)

            # SURFACE HEIGHT
            z_surface_t = z_surface

            # RELATIVE HEIGHTS
            z_station_rel = z_station
            z_levels_rel = hourly_pressure["z"].values / G_CST

            z_2m_rel = 2.0 + z_surface_t

            # VARIABLES
            T_levels = hourly_pressure["t"].values

            # recalcul pression de vapeur pour CE pas de temps
            T_levels_c = kelvin_to_celsius(hourly_pressure["t"].values)
            es_levels = get_saturation_vapor_pressure(T_levels_c)
            e_levels = compute_vapor_pressure(
                hourly_pressure["r"].values,
                es_levels,
            )

            T_2m = hourly_surface["t2m"].values

            Td_2m = kelvin_to_celsius(hourly_surface["d2m"].values)

            z_clean = z_levels_rel
            T_clean = T_levels
            Td_clean = compute_dewpoint_temp_from_e(e_levels)
            if len(z_clean) < 2:
                ta_out.append(np.nan)
                td_out.append(np.nan)
                continue

            # ADD 2m LEVEL
            z_add = np.insert(z_clean, 0, z_2m_rel)
            T_add = np.insert(T_clean, 0, T_2m)
            Td_add = np.insert(Td_clean, 0, Td_2m)
            logger.info(f"z_add:{z_add}")
            # SORT VERTICALLY
            sort_idx = np.argsort(z_add)
            z_full = z_add[sort_idx]
            T_full = T_add[sort_idx]
            Td_full = Td_add[sort_idx]
            logger.info(f"Td_full = {Td_full}")

            f_t = interp1d(
                z_full,
                T_full,
                kind="linear",
                fill_value="extrapolate",
            )
            f_td = interp1d(
                z_full,
                Td_full,
                kind="linear",
                fill_value="extrapolate",
            )
            T_station = f_t(z_station_rel)
            Td_station = f_td(z_station_rel)

            ta_out.append(T_station)
            td_out.append(Td_station)

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
                "ta": kelvin_to_celsius(ta_da),
                "tdp": td_da,
            }
        )

        # SAVE
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

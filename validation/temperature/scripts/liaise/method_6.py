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
from pathlib import Path

import geopandas as gpd
import rioxarray  # noqa # Use to activate rio attributes
from rasterio.coords import BoundingBox

from etdataset.cli import CLIException
from etdataset.dem import get_dem_from_roi
from etdataset.era5 import ERA5Dataset
from etdataset.logging import LoggerManager
from etdataset.temperature import RescalTempMethod, TempVariable, add_temp
from etdataset.utils import (
    generate_dates,
    generate_hours,
)

logger = LoggerManager.get_logger(__name__)

G_CST = 9.80665


def run_stations_process_method_6(
    start_date: dt.date,
    end_date: dt.date,
    step_date: int,
    list_hours: list[dt.time],
    mnt_path: str,
    data_path: str,
    output: str,
):
    gdf = gpd.read_file(
        "/home/mliateni/Bureau/meriem/et-dataset/notebooks/Data_Liaise/Zone_Clip/Zone_etude.shp"
    )
    xmin, ymin, xmax, ymax = gdf.total_bounds
    roi_bbox_utm = BoundingBox(
        left=xmin, bottom=ymin, right=xmax, top=ymax
    )  # 20000
    roi_crs_utm = gdf.crs
    for d in generate_dates(start_date, end_date, step_date):
        for h in list_hours:
            date = dt.datetime.combine(d, h)
            logger.info(f"Processing {date}")

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
                method=RescalTempMethod.INTERP_LR_HYBRID,
            )
            # Sauvegarde du dataset updated pour cette date
            date_str = date.strftime("%Y%m%d_%H%M%S")
            out_path = Path(output) / date.strftime("%Y/%m")
            out_path.mkdir(parents=True, exist_ok=True)
            updated_to_save = updated.copy()

            for var in [
                updated_to_save,
                *list(updated_to_save.data_vars.values()),
            ]:
                for k, v in var.attrs.items():
                    if not isinstance(v, (str, int, float, list, tuple, bytes)):
                        var.attrs[k] = str(v)

            updated_to_save.to_netcdf(out_path / f"updated_{date_str}.nc")
            logger.info(f"Saved updated for {date} -> {out_path}")


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
):
    # download_date_by_date(start_date, end_date, ERA5Dataset.ERA5, output="out")  # noqa: E501
    list_hours = generate_hours(hour_start, hour_end, hour_step)

    procs = []
    p = Process(
        target=run_stations_process_method_6,
        args=(
            start_date,
            end_date,
            step_day,
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
        default="method_6",
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
    )

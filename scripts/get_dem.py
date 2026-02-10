# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales
import argparse
import logging
import os

from etdataset.dem import get_dem_from_roi
from etdataset.logging import LoggerManager
from etdataset.utils import get_utm_bbox_from_roi
from etdataset.writer import write_to_netcdf, write_to_tif

logger = LoggerManager.get_logger(__name__)


def get_parser() -> argparse.ArgumentParser:
    """
    Generate argument parser for cli
    """
    # create the top-level parser
    parser = argparse.ArgumentParser(description="Get DEM")

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
        "-f",
        "--format",
        default="tif",
        choices=["tif", "netcdf"],
        type=str,
        help="Output format (tif, netcdf)",
        required=False,
    )

    return parser


def get_dem() -> None:
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

    # Get bbox and crs
    roi_bbox, roi_crs = get_utm_bbox_from_roi(args.roi)

    # Get DEM
    dem = get_dem_from_roi(
        roi_bbox=roi_bbox, roi_crs=roi_crs, base_dir=args.mnt_path
    )

    # Write
    if args.format == "tif":
        write_to_tif(dem, "dem.tif")
    else:
        write_to_netcdf(dem, "dem.nc")


if __name__ == "__main__":
    get_dem()

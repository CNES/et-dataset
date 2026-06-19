#!/usr/bin/env python
# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales

#######################
# Prepare Landsat data
#######################

# Imports

import argparse
import logging
import os
import tarfile
from pathlib import Path

import rasterio as rio
import rioxarray  # noqa # Use to activate rio attributes
from sensorsio.utils import bb_transform

from etdataset.api import add_aux_data, create_dataset
from etdataset.dem import add_dem
from etdataset.logging import LoggerManager
from etdataset.utils import dilate_mask, get_utm_bbox_from_roi
from etdataset.writer import write_dataset

logger = LoggerManager.get_logger(__name__)


def prepare_landsat(
    roi: str,
    output: str,
    landsat_path: str | None = None,
    era5_path: str | None = None,
    radiation_path: str | None = None,
    mnt_path: str | None = None,
):
    """
    Prepare landsat data
    """

    # Inputs

    # ROI
    roi_bbox, roi_crs = get_utm_bbox_from_roi(roi)
    logger.debug(f"ROI bbox: {roi_bbox}")
    logger.debug(f"ROI crs: {roi_crs}")
    latlon_bbox = bb_transform(str(roi_crs), "EPSG:4326", roi_bbox)
    logger.debug(f"ROI bbox (lat/lon): {latlon_bbox}")

    output_path = output
    os.makedirs(output_path, exist_ok=True)

    # Download directory
    if radiation_path is None:
        radiation_dir = Path(output_path) / "MSG_data"
    else:
        radiation_dir = Path(radiation_path)
    if era5_path is None:
        era5_dir = Path(output_path) / "ERA5_data"
    else:
        era5_dir = Path(era5_path)

    # Extract archives

    # List archives
    if landsat_path is None:
        landsat_dir = Path(output_path) / "LANDSAT"
    else:
        landsat_dir = Path(landsat_path)
    archives = list(landsat_dir.rglob("*.tar"))

    # Untar archives
    for archive in archives:
        # Open and extract
        with tarfile.open(archive) as tar:
            extract_path = landsat_dir / archive.stem
            if not extract_path.exists():
                tar.extractall(path=extract_path)

    # List products
    products = [f for f in landsat_dir.iterdir() if f.is_dir()]

    # Preprocess products
    etdataset_path = Path(output_path) / "et_data"
    os.makedirs(etdataset_path, exist_ok=True)

    for product in products:
        logger.info(f"Process product {product}...")

        # Create dataset
        data = create_dataset(
            vis_path=str(product),
            roi_bbox=roi_bbox,
            roi_crs=roi_crs,
            resolution=90,
            resampling=rio.enums.Resampling.average,
        )
        logger.debug("Init dataset: OK")
        if mnt_path is not None:
            data = add_dem(data, mnt_dir=mnt_path)
            logger.debug("Add auxilary data: OK")

        # Apply a dilation of cloud mask
        data["cloud"] = dilate_mask(data["cloud"], dilation=20)
        logger.debug("Apply dilation of cloud on mask: OK")
        logger.info("Create dataset: OK")

        # Add auxiliary data
        updated_data = add_aux_data(
            data=data,
            radiation_path=str(radiation_dir),
            era5_path=str(era5_dir),
        )
        logger.info("Add auxilary data: OK")

        # Write data
        write_dataset(updated_data, directory=str(etdataset_path))
        logger.info(f"Process product {product}:OK")


def get_parser() -> argparse.ArgumentParser:
    """
    Generate argument parser for cli
    """
    # create the top-level parser
    parser = argparse.ArgumentParser(description="Prepare Landsat data")

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
        "-o",
        "--output",
        type=str,
        help="Output dataset directory path (default: current directory)",
        default=os.getcwd(),
    )
    parser.add_argument(
        "--landsat_path",
        type=str,
        help="Directory of Landsat products",
    )
    parser.add_argument(
        "--era5_path",
        type=str,
        help="Directory of ERA5 data",
    )
    parser.add_argument(
        "--radiation_path",
        type=str,
        help="Directory of radiation data",
    )
    parser.add_argument(
        "--mnt_path",
        type=str,
        help="Directory of DEM tiles",
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
    if not os.path.isfile(args.roi):
        raise FileNotFoundError(f"File not found {args.roi}")
    if not os.path.isdir(args.output):
        logger.debug(f"Create output path: {args.output}")
        os.makedirs(args.output, exist_ok=True)
    if args.landsat_path is not None and not os.path.isdir(args.landsat_path):
        raise FileNotFoundError(
            f"Landsat directory not found {args.landsat_path}"
        )
    if args.radiation_path is not None and not os.path.isdir(
        args.radiation_path
    ):
        raise FileNotFoundError(
            f"Radiation directory not found {args.radiation_path}"
        )
    if args.era5_path is not None and not os.path.isdir(args.era5_path):
        raise FileNotFoundError(f"ERA5 directory not found {args.era5_path}")
    if args.mnt_path is not None and not os.path.isdir(args.mnt_path):
        raise FileNotFoundError(f"DEM directory not found {args.mnt_path}")

    # Run
    prepare_landsat(
        args.roi,
        args.output,
        args.landsat_path,
        args.era5_path,
        args.radiation_path,
        args.mnt_path,
    )

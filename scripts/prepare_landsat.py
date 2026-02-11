#!/usr/bin/env python
# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales

#######################
# Prepare Landsat data
#######################

# Imports

import argparse
import datetime as dt
import logging
import os
import sys
import tarfile
from pathlib import Path

import rasterio as rio
import rioxarray  # noqa # Use to activate rio attributes
from sensorsio.utils import bb_transform

from etdataset.api import add_aux, create_dataset, download, search
from etdataset.cli import CLIException
from etdataset.dem import add_dem
from etdataset.logging import LoggerManager
from etdataset.provider import Collection
from etdataset.utils import dilate_mask, get_utm_bbox_from_roi
from etdataset.writer import write_dataset

logger = LoggerManager.get_logger(__name__)


def prepare_landsat(
    roi: str,
    start_date: str,
    end_date: str,
    output: str,
    max_cloud_cover: float = 20,
    min_roi_coverage: float = 33,
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

    # Search

    res = search(
        Collection.LANDSAT,
        start_date,
        end_date,
        roi_bbox=roi_bbox,
        roi_crs=roi_crs,
        max_cloud_cover=max_cloud_cover,
        min_roi_overlap=min_roi_coverage,
    )
    # Keep only landsat 8
    res = res[res["Product_name"].str.contains("LC08_", case=False, na=False)]
    logger.info(f"Number of products found: {len(res)}")
    logger.info("Search: OK")
    if len(res) == 0:
        logger.warning("No product found")
        sys.exit(1)
    # Download results
    download(products=res, output_dir=output_path)
    logger.info("Download: OK")

    # Extract archives

    # List archives
    folder = Path(output_path) / "LANDSAT"
    archives = list(folder.rglob("*.tar"))

    # Untar archives
    landsat_path = Path(output_path) / "LANDSAT"
    for archive in archives:
        # Open and extract
        with tarfile.open(archive) as tar:
            extract_path = landsat_path / archive.stem
            if not extract_path.exists():
                tar.extractall(path=extract_path)

    # List products
    landsat_path = Path(output_path) / "LANDSAT"
    products = [f for f in landsat_path.iterdir() if f.is_dir()]

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

        # Add auxilary data
        updated_data = add_aux(data=data, path=output_path)
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
        "-s",
        "--start-date",
        type=str,
        help="Start date (YYYY-MM-DD)",
        required=True,
    )
    parser.add_argument(
        "-e",
        "--end-date",
        type=str,
        help="End date (YYYY-MM-DD)",
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
        "--max_cloud_cover",
        type=int,
        default=25,
        help="Maximum cloud cover (default: 25)",
    )
    parser.add_argument(
        "--min_roi_overlap",
        type=int,
        default=33,
        help="Minimum overlap between ROI and a product (default: 33)",
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
    try:
        min_date = dt.datetime.strptime(args.start_date, "%Y-%m-%d")
    except ValueError:
        raise CLIException(
            "Error: The format for minimum acqsuisition "
            "date must be Year-Month-Day"
        )
    try:
        max_date = dt.datetime.strptime(args.end_date, "%Y-%m-%d")
    except ValueError:
        raise CLIException(
            "Error: The format for maximum acquisition "
            "date must be Year-Month-Day"
        )
    if max_date < min_date:
        raise CLIException(
            "Maximum acquisition date must be more recent than minimum date"
        )
    if args.max_cloud_cover < 0 or args.max_cloud_cover > 100:
        raise CLIException(
            "Cloud cover criteria must be between 0 and 100 "
            f"(got : {args.max_cloud_cover})"
        )
    if args.min_roi_overlap < 0 or args.min_roi_overlap > 100:
        raise CLIException(
            "Min ROI overlap criteria must be between 0 and 100 "
            f"(got : {args.min_roi_overlap})"
        )
    if not os.path.isdir(args.output):
        logger.debug(f"Create output path: {args.output}")
        os.makedirs(args.output, exist_ok=True)
    if args.mnt_path is not None and not os.path.isdir(args.mnt_path):
        raise FileNotFoundError(f"DEM directory not found {args.mnt_path}")

    # Run
    prepare_landsat(
        args.roi,
        args.start_date,
        args.end_date,
        args.output,
        args.max_cloud_cover,
        args.min_roi_overlap,
        args.mnt_path,
    )

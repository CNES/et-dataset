#!/usr/bin/env python

#######################
# Prepare Landsat data
#######################

# Imports

import argparse
import datetime as dt
import json
import logging
import os
import sys
import tarfile
from pathlib import Path

import rasterio as rio
import rioxarray  # noqa # Use to activate rio attributes
from sensorsio.utils import bb_transform

from etdataset import msg
from etdataset.api import create_dataset, download, search
from etdataset.cli import CLIException
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
    download(products=res, output_dir=os.path.join(output_path, "Landsat"))
    logger.info("Download: OK")

    # Extract archives

    # List archives
    folder = Path(output_path) / "Landsat"
    archives = list(folder.rglob("*.tar"))

    # Untar archives
    landsat_path = Path(output_path) / "Landsat"
    for archive in archives:
        # Open and extract
        with tarfile.open(archive) as tar:
            extract_path = landsat_path / archive.stem
            if not extract_path.exists():
                tar.extractall(path=extract_path)

    # List products
    landsat_path = Path(output_path) / "Landsat"
    products = [f for f in landsat_path.iterdir() if f.is_dir()]

    # Preprocess products
    etdataset_path = Path(output_path) / "et_data"
    os.makedirs(etdataset_path, exist_ok=True)

    for product in products:
        logger.info(f"Process product {product}...")

        # Create dataset
        data = create_dataset(
            vis_path=product,
            roi_bbox=roi_bbox,
            roi_crs=roi_crs,
            resolution=100,
            resampling=rio.enums.Resampling.average,
        )
        logger.debug("Create dataset: OK")

        # Apply masks
        data["lst"] = data["lst"].where(~data["water"])
        data["lst"] = data["lst"].where(~dilate_mask(data["cloud"], dilation=2))
        data["lst"] = data["lst"].where(data["qa"])
        logger.debug("Create dataset: OK")

        # Get datetime for a product
        product_path = Path(product)
        filename = product_path.name + "_MTL.json"
        with open(product_path / filename) as mtl_file:
            mtl_cfg = json.load(mtl_file)
        d = mtl_cfg["LANDSAT_METADATA_FILE"]["IMAGE_ATTRIBUTES"][
            "DATE_ACQUIRED"
        ]
        h = mtl_cfg["LANDSAT_METADATA_FILE"]["IMAGE_ATTRIBUTES"][
            "SCENE_CENTER_TIME"
        ]
        d = dt.datetime.strptime(d, "%Y-%m-%d").date()
        # Trim to microseconds (6 digits)
        h = h[:15] + "Z"  # "11:27:30.842007Z"
        h = dt.datetime.strptime(h, "%H:%M:%S.%fZ").time()
        acquisition_date = dt.datetime.combine(d, h)

        logger.debug(f"Acquisition datetime: {acquisition_date}")

        # Download MSG data
        msg.download(
            date=acquisition_date, latlon_bbox=latlon_bbox, path=output_path
        )
        logger.debug("Download MSG data: OK")

        # Activate rioxarray accessor
        if not hasattr(data, "rio"):
            crs = data.attrs["crs"]
            data = data.rio.write_crs(crs)

        # Add MSG data
        updated_data = msg.add(data=data, path=output_path)

        # Write data
        write_dataset(updated_data, directory=etdataset_path)


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

    # Run
    prepare_landsat(
        args.roi,
        args.start_date,
        args.end_date,
        args.output,
        args.max_cloud_cover,
        args.min_roi_overlap,
    )

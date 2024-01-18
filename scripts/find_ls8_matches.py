#!/usr/bin/env python
# coding: utf8

# Copyright: (c) 2022 CESBIO / Centre National d'Etudes Spatiales
import os
import sys
import argparse
import logging
import geopandas as gpd
import pandas as pd

from datetime import datetime
from sensorsio.mgrs import get_bbox_mgrs_tile

from etdataset.database import create_ls8_db
from etdataset.logging import LoggerManager

"""
List Landsat products
"""

LS8_PATH="landsat_ot_c2_l2_655f86cd74953c8b.csv"


def get_parser() -> argparse.ArgumentParser:
    """
    Generate argument parser for cli
    """
    parser = argparse.ArgumentParser(
        os.path.basename(__file__),
        description="List Landsat products")

    parser.add_argument('-v','--verbose', 
                        dest='verbose', 
                        action='store_true',
                        help="Verbose mode")

    parser.add_argument(
        '-l', '--landsat8_csv',
        type=str,
        help='Path to csv file with landsat C2L2 earth explorer archive export')

    parser.add_argument('--min_date',
                        help='Minimum date for acquisition',
                        type=str,
                        default=None)

    parser.add_argument('--max_date',
                        help='Maximum date for acquisition',
                        type=str,
                        default=None)

    parser.add_argument('--max_cloud_cover',
                        type=int,
                        default=25,
                        help='Maximum cloud cover for Landsat8 products')
   
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-t', '--tile',
                        type=str,
                        help='Tile ID')

    group.add_argument('-r', '--roi',
                        type=str,
                        help='Region of interest in Shapefile format')

    parser.add_argument('--min_roi_overlap',
                        type=int,
                        default=50,
                        help='Minimum ROI overlap with Landsat8 products')

    parser.add_argument(
        '--output',
        type=str,
        help='CSV output file',
        default='ls8_matches.csv')

    return parser


if __name__ == '__main__':

    # Parser arguments
    parser = get_parser()
    args = parser.parse_args()

    # Configure logging
    log_level = logging.INFO
    if args.verbose:
        log_level = logging.DEBUG

    logger = LoggerManager.get_logger(__name__)
    LoggerManager.set_level(log_level)
    
    # Landsat8 DB
    ls8_db_path = args.landsat8_csv
    if ls8_db_path is None:
        if not os.environ.get("METADATA_PATH"):
            raise Exception("METADATA_PATH not defined")
        ls8_db_path = os.path.join(os.environ["METADATA_PATH"],LS8_PATH)
    if not os.path.isfile(ls8_db_path):
        raise Exception(f"Landsat8 database not found ({ls8_db_path})")
    logger.debug(f"Landsat8 DB path: {ls8_db_path}")

    # Process date
    try: 
        min_date = datetime.strptime(args.min_date,'%Y-%m-%d')
    except:
        raise Exception("Error: The date format must be Year-Month-Day")
    try: 
        max_date = datetime.strptime(args.max_date,'%Y-%m-%d')
    except:
        raise Exception("Error: The date format must be Year-Month-Day")
    logger.debug(f"Minimum acquisition date: {min_date}")
    logger.debug(f"Maximum acquisition date: {max_date}")
    if max_date < min_date:
        logger.error("Maximum acquisition date must be more recent than minimum date")
        sys.exit(1)

    # Get ROI bounding box and CRS from tile or shapefile
    roi_bbox = None
    roi_crs = None
    if args.tile is not None:
        roi_bbox = get_bbox_mgrs_tile(args.tile)
        roi_crs = 4326
    else:
        roi = gpd.read_file(args.roi)
        roi_crs = roi.crs
        roi_bbox = rio.coords.BoundingBox(*roi.bounds.iloc[0].values)
    logger.debug(f"ROI bounding box: {roi_bbox}")
    logger.debug(f"ROI CRS: {roi_crs}")

    # Get matches
    ls8_db = create_ls8_db(ls8_db_path, 
                           min_date = min_date, 
                           max_date = max_date, 
                           max_cloud_cover=args.max_cloud_cover,                    
                           roi_bbox = roi_bbox,
                           roi_crs = roi_crs,
                           min_roi_overlap=args.min_roi_overlap)

    # Write resuts
    ls8_db.to_csv(args.output)


    

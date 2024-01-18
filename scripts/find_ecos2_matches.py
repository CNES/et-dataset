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

from etdataset.database import create_eco_db, create_s2_db, select_products, to_collectionV2
from etdataset.logging import LoggerManager

"""
Find ECOSTRESS/Sentinel2 matches
"""

S2_PATH="SENTINEL2.csv"
ECO_PATH="ecostress_eco2lste_655f877cf0476b91.csv"


def get_parser() -> argparse.ArgumentParser:
    """
    Generate argument parser for cli
    """
    parser = argparse.ArgumentParser(
        os.path.basename(__file__),
        description="Find ECOSTRESS/Sentinel2 matches")

    parser.add_argument('-v','--verbose', 
                        dest='verbose', 
                        action='store_true',
                        help="Verbose mode")

    parser.add_argument(
        '-s', '--sentinel2_csv',
        type=str,
        help='Path to csv file with Sentinel2 archive export')

    parser.add_argument(
        '-e', '--ecostress_csv',
        type=str,
        help='Path to csv file with ECOSTRESS earth explorer archive export')

    parser.add_argument('--min_date',
                        help='Minimum date for acquisition',
                        type=str,
                        default=None)

    parser.add_argument('--max_date',
                        help='Maximum date for acquisition',
                        type=str,
                        default=None)

    parser.add_argument('--delta',
                        type=str,
                        default='3 day',
                        help='Maximum time delta allowed between ecostress and landsat8 acquisitions'
                        )
    
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
        default='ecos2_matches.csv')

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
    
    # Sentinel2 DB
    s2_db_path = args.sentinel2_csv
    if s2_db_path is None:
        if not os.environ.get("METADATA_PATH"):
            raise Exception("METADATA_PATH not defined")
        s2_db_path = os.path.join(os.environ["METADATA_PATH"],S2_PATH)
    if not os.path.isfile(s2_db_path):
        raise Exception(f"Sentinel2 database not found ({s2_db_path})")
    logging.debug(f"(Sentinel2 DB path: {s2_db_path}")

    # ECOSTRESS DB
    eco_db_path = args.ecostress_csv
    if eco_db_path is None:
        if not os.environ.get("METADATA_PATH"):
            raise Exception("METADATA_PATH not defined")
        eco_db_path = os.path.join(os.environ["METADATA_PATH"],ECO_PATH)
    if not os.path.isfile(eco_db_path):
        raise Exception(f"ECOSTRESS database not found ({eco_db_path})")
    logging.debug(f"(ECOSTRESS DB path: {eco_db_path}")

    # Process date
    try: 
        min_date = datetime.strptime(args.min_date,'%Y-%m-%d')
    except:
        raise Exception("Error: The date format must be Year-Month-Day")
    try: 
        max_date = datetime.strptime(args.max_date,'%Y-%m-%d')
    except:
        raise Exception("Error: The date format must be Year-Month-Day")
    logging.debug(f"Minimum acquisition date: {min_date}")
    logging.debug(f"Maximum acquisition date: {max_date}")
    if max_date < min_date:
        logging.error("Maximum acquisition date must be more recent than minimum date")
        sys.exit(1)
    try:
        delta = pd.Timedelta(args.delta)
    except:
        raise Exception("Error: The date format must be Year-Month-Day")
    logging.debug(f"Minimum acquisition date: {min_date}")


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

    # Get Sentinel2 products
    s2_db = create_s2_db(s2_db_path, 
                           min_date = min_date, 
                           max_date = max_date, 
                           roi_bbox = roi_bbox,
                           roi_crs = roi_crs,
                           min_roi_overlap=args.min_roi_overlap)

    # Get ECOSTRESS products
    eco_db = create_eco_db(eco_db_path, 
                           min_date = min_date, 
                           max_date = max_date, 
                           roi_bbox = roi_bbox,
                           roi_crs = roi_crs,
                           min_roi_overlap=args.min_roi_overlap)
    eco_db_v2 = to_collectionV2(eco_db,roi_bbox, roi_crs,30)
    
    # Select matches
    res = select_products(s2_db,eco_db_v2,delta, 40)

    # Write results
    res.to_csv(args.output)

    

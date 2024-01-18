#!/usr/bin/env python
# coding: utf8

# Copyright: (c) 2022 CESBIO / Centre National d'Etudes Spatiales
import os
import sys
import argparse
import logging
import geopandas as gpd

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from sensorsio import mgrs

from etdataset.ls8 import create_dataset
from etdataset.common import write_dataset

"""
Create dataset from landsat8 dataset
"""


def get_parser() -> argparse.ArgumentParser:
    """
    Generate argument parser for cli
    """
    parser = argparse.ArgumentParser(
        os.path.basename(__file__),
        description="Create dataset from landsat8 product")

    parser.add_argument('-v','--verbose', 
                        dest='verbose', 
                        action='store_true',
                        help="Verbose mode")

    parser.add_argument(
        '-l', '--landsat8',
        type=str,
        required=True,
        help='Path to landsat8 product')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-t', '--tile',
                        type=str,
                        help='Tile ID')

    group.add_argument('-r', '--roi',
                        type=str,
                        help='Region of interest in Shapefile format')

    parser.add_argument(
        '--output',
        type=str,
        help='Output dataset dir path')

    return parser


if __name__ == '__main__':

    # Parser arguments
    parser = get_parser()
    args = parser.parse_args()

    # Configure logging
    log_level = logging.INFO
    if args.verbose:
        log_level = logging.DEBUG

    logging.basicConfig(datefmt='%y-%m-%d %H:%M:%S',
                        format='%(asctime)s :: %(levelname)s :: %(message)s')
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)
    
    # Check Landsat8 path
    ls8_path = args.landsat8
    if not os.path.isdir(ls8_path):
        raise Exception(f"Landsat8 product not found ({ls8_path})")
    logger.debug(f"Landsat8 path: {ls8_path}")

    # Check output path
    output_path = args.output
    if output_path is None:
        output_path = os.getcwd()
    if not os.path.isdir(output_path):
        logger.debug(f"Create output path: {output_path}")
        os.makedirs(output_path, exist_ok=True)

    # Get tiles ID
    tile_ids = [] 
    if args.tile is not None:
        tile_ids.append(args.tile)
    if args.roi is not None:
        roi = gpd.read_file(args.roi)
        roi_poly = roi.geometry[0]
        crs_poly = roi.crs
        tile_ids = mgrs.get_mgrs_tiles_from_roi(roi_poly,crs_poly)
    logger.info(f"Tiles ID:{tile_ids}")

    # Create datasets
    logger.info(f"Create datasets...")
    datasets = []
    with logging_redirect_tqdm():
        for tile_id in tqdm(tile_ids, desc="Processing tiles..."):
            logger.debug(f"Process tile: {tile_id}")
            ls8 = create_dataset(ls8_path, tile_id)
            datasets.append(ls8)

    # Write 
    logger.info(f"Write datasets...")
    for ds in tqdm(datasets):
        write_dataset(ds,dir=output_path, desc="Writing dataset...")

    # End

#!/usr/bin/env python
# coding: utf8

# Copyright: (c) 2022 CESBIO / Centre National d'Etudes Spatiales
import os
import sys
import argparse
import logging
import geopandas as gpd

from etdataset.ecos2 import create_dataset
from etdataset.common import write_dataset

"""
Create dataset from sentinel2 and ecostress products
"""


def get_parser() -> argparse.ArgumentParser:
    """
    Generate argument parser for cli
    """
    parser = argparse.ArgumentParser(
        os.path.basename(__file__),
        description="Create dataset from sentinel2 and ecostress products")

    parser.add_argument('-v','--verbose', 
                        dest='verbose', 
                        action='store_true',
                        help="Verbose mode")

    parser.add_argument(
        '-s', '--sentinel2',
        type=str,
        required=True,
        help='Path to sentinel2 product')

    parser.add_argument(
        '-e', '--ecostress',
        type=str,
        required=True,
        help='Path to ecostress product')

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
    
    # Check Seninel2 path
    s2_path = args.sentinel2
    if not os.path.isdir(s2_path):
        raise Exception(f"Sentinel2 product not found ({s2_path})")
    logger.debug(f"Sentinel2 path: {s2_path}")

    # Check Ecostress path
    eco_path = args.ecostress
    if not os.path.isdir(eco_path):
        raise Exception(f"Ecostress product not found ({eco_path})")
    logger.debug(f"Ecotress path: {eco_path}")

    # Check output path
    output_path = args.output
    if output_path is None:
        output_path = os.getcwd()
    if not os.path.isdir(output_path):
        logger.debug(f"Create output path: {output_path}")
        os.makedirs(output_path, exist_ok=True)

    # Create datasets
    logger.info(f"Create dataset...")
    ds = create_dataset(s2_path, eco_path)
    
    # Write 
    logger.info(f"Write dataset...")
    write_dataset(ds,dir=output_path)

    # End

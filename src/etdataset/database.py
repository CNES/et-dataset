#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright: (c) 2023 CESBIO / Centre National d'Etudes Spatiales / Université Paul Sabatier (UT3)
#
"""
Landsat8
"""
import geopandas as gpd
import pandas as pd
import os
import rasterio as rio
import warnings

from datetime import datetime, timedelta
from functools import lru_cache
from pyproj import CRS
from rasterio.warp import transform_bounds
from sensorsio import mgrs
from shapely.geometry import Polygon
from tqdm import tqdm

from etdataset.logging import LoggerManager
from etdataset.common import create_polygon

warnings.filterwarnings("ignore", category=FutureWarning, module='rasterio')
logger = LoggerManager.get_logger(__name__)

def convert_date(date: str) -> datetime:
    try:
        date = datetime.strptime(date,"%Y-%m-%d %H:%M:%S.%f")
    except:
        try: 
            date = datetime.strptime(date,"%Y-%m-%d %H:%M:%S")
        except:
            raise Exception(f"Unregonized format for date ({date})")
    return date


def filter_with_roi(gdf: gpd.GeoDataFrame,
                    roi_bbox: rio.coords.BoundingBox,
                    roi_crs: [CRS|int],
                    min_overlap: float = None) -> gpd.GeoDataFrame:
    """
    Filter a GeoDataFrame with a ROI
    """
    # Convert ROI 
    bounds = transform_bounds(roi_crs, gdf.crs, *roi_bbox)
    # Convert bounds to polygon
    aoi_poly = Polygon([[bounds[0], bounds[1]], [bounds[0], bounds[3]],
                              [bounds[2], bounds[3]], [bounds[2], bounds[1]]])
    aoi = gpd.GeoDataFrame(data={'id':[1],'geometry':[aoi_poly]},crs=gdf.crs)
    aoi_area = aoi_poly.area
    try: 
        filtered = gpd.GeoDataFrame(
                   gpd.overlay(gdf,aoi,how="intersection")
                   .drop(["id"],axis=1)
                   .merge(gdf[["product_name","geometry"]], how='inner', on='product_name',suffixes=('_roi','_orig'))
                   .rename(columns={"geometry_roi":"overlap_geometry",
                                    "geometry_orig":"geometry",
                                    })
                   )
        filtered['overlap_percentage'] = filtered.apply(lambda x: 100 * x.overlap_geometry.area / aoi_area,axis = 1)
        if min_overlap is not None:
            filtered = filtered[filtered['overlap_percentage'] > min_overlap]
    except AttributeError:
        # No matches found
        return None
    return filtered

def convert_name_to_collectionV2(product_name:str, mgrs_tile_name:str) -> str:
    """
    Convert ECOSTRESS (collection v1) product name to a product name for collection 2
    """
    # remove .h5 extension
    
    elts_v1 = product_name.split(".h5")[0].split("_")
    elts_v2 = ["ECOv002","L2T"] + elts_v1[2:5] + [mgrs_tile_name] + elts_v1[-3:]
    return "_".join(elts_v2)

def convert_product_to_collectionV2(product: pd.Series, mgrs_tiles: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Convert a ECOSTRESS collection v1 product to products colllection V2
    """
    products_v2 = []
    for _,mgrs_tile in mgrs_tiles.iterrows():
        product_v2 = pd.Series(mgrs_tile)
        product_v2["product_name"] = convert_name_to_collectionV2(product["product_name"],mgrs_tile["Name"])
        product_v2["date"] = product["date"]
        products_v2.append(product_v2[["product_name","date","geometry","overlap_geometry","overlap_percentage"]])
    return gpd.GeoDataFrame(products_v2)


def to_collectionV2(eco_gdf: gpd.GeoDataFrame, roi_bbox: rio.coords.BoundingBox, roi_crs: [CRS|int], min_overlap: float = 30):
    mgrs_tiles = mgrs.get_mgrs_tiles_from_roi(roi_bbox, roi_crs)
    mgrs_tiles = mgrs_tiles[mgrs_tiles["overlap_percentage"] > min_overlap]
    df = eco_gdf.apply(lambda x: convert_product_to_collectionV2(x,mgrs_tiles),axis=1)
    df = pd.concat(df.values).reset_index(drop=True)
    return gpd.GeoDataFrame(df[df.columns[~df.columns.isin(['geometry'])]],geometry=df['geometry'],crs=eco_gdf.crs)


@lru_cache
def read_raw_ls8_db(ls8_csv_path:str) -> pd.DataFrame:
    """
    Read raw Landsat database in CSV file format
    """
    return pd.read_csv(ls8_csv_path,
                         parse_dates=['Date Acquired','Date Product Generated L2','Date Product Generated L1'],
                         encoding='ISO-8859-1')

def create_ls8_db(ls8_csv_path:str,
              min_date: datetime = None,
              max_date: datetime = None,
              roi_bbox: rio.coords.BoundingBox = None,
              roi_crs: str = None,
              max_cloud_cover:float = 25,
              min_roi_overlap:float = 33) -> gpd.GeoDataFrame:
    """
    Create database for Landsat8 product from CSV metadata file

    """
    ls8_df = read_raw_ls8_db(ls8_csv_path)
    logger.debug(f'Found {len(ls8_df)} Landsat8 products')
    try:
        # Filter day acquisitions
        ls8_df = ls8_df[ls8_df['Day/Night Indicator'] == 'DAY']
        logger.debug(f'Found {len(ls8_df)} Landsat8 products after filtering day acquisitions')
        # Filter on clouds
        if max_cloud_cover is not None: 
            ls8_df = ls8_df[ls8_df['Land Cloud Cover'] < max_cloud_cover]
            logger.debug(f'Found {len(ls8_df)} Landsat8 products after filtering cloud coverage')

        # Filter dates
        ls8_df['min_date'] = ls8_df['Start Time'].apply(lambda x: convert_date(x))
        ls8_df['max_date'] = ls8_df['Stop Time'].apply(lambda x: convert_date(x))
        ls8_df['date'] = ls8_df['min_date'].dt.date
        if min_date is not None and max_date is not None:
            ls8_df = ls8_df[ ls8_df['date'] >= min_date.date() ]
            ls8_df = ls8_df[ ls8_df['max_date'].dt.date <= max_date.date() ]
            logger.debug(f'Found {len(ls8_df)} Landsat8 products after filtering dates')

        # Convert filtered dataframe to a geodataframe
        geometry = ls8_df.apply(lambda row: create_polygon(row['Corner Upper Left Latitude'],
                                                         row['Corner Upper Left Longitude'],
                                                         row['Corner Upper Right Latitude'],
                                                         row['Corner Upper Right Longitude'],
                                                         row['Corner Lower Left Latitude'],
                                                         row['Corner Lower Left Longitude'],
                                                         row['Corner Lower Right Latitude'],
                                                         row['Corner Lower Right Longitude'],),axis=1)
        ls8_df = gpd.GeoDataFrame(ls8_df[['Display ID','date', 'Land Cloud Cover']],
                           crs=4326,
                           geometry=geometry).rename(columns={'Display ID':'product_name','Land Cloud Cover':'cloud_cover'})

        # Filter with ROI
        if roi_bbox is not None:
            ls8_df = filter_with_roi(ls8_df, roi_bbox, roi_crs, min_roi_overlap)
            logger.debug(f'Found {len(ls8_df)} Landsat products after filtering ROI')
    except AttributeError as e:
        if len(ls8_df) == 0:
            logger.warning("No product found")
            return gpd.GeoDataFrame(columns=['product_name','date', 'cloud_cover','geometry'],crs=4326)
        raise e

    if len(ls8_df) == 0:
        logger.warning("No product found")
    else:
        nb_ls8_filtered = len(ls8_df)
        logger.info(
            f'Found {nb_ls8_filtered} Landsat products fullfilling the requirements'
        )

    return ls8_df.reset_index(drop=True)

@lru_cache
def read_raw_eco_db(eco_csv_path: str) -> pd.DataFrame:
    """
    Read raw ECOSTRESS database in CSV file format
    """
    cols = ["Local Granule ID","Entity ID","Acquisition Start Date","Acquisition End Date","Orbit Number",
        "Scene Number","DOI Authority","DOI Name","Center Latitude","Center Longitude","NW Corner Lat",
        "NW Corner Long","NE Corner Lat","NE Corner Long","SE Corner Lat","SE Corner Long","SW Corner Lat","SW Corner Long",
        "Center Latitude dec","Center Longitude dec","NW Corner Lat dec","NW Corner Long dec","NE Corner Lat dec","NE Corner Long dec","SE Corner Lat dec",
        "SE Corner Long dec","SW Corner Lat dec","SW Corner Long dec","Display ID","index"]
    return pd.read_csv(eco_csv_path, 
                         encoding='ISO-8859-1',
                         names=cols,
                         skiprows=1)


def create_eco_db(eco_csv_path:str,
              min_date: datetime = None,
              max_date: datetime = None,
              roi_bbox: rio.coords.BoundingBox = None,
              roi_crs: str = None,
              min_roi_overlap:float = 33) -> gpd.GeoDataFrame:
    """
    Create database for ECOSTRESS product from CSV metadata file

    """
    eco_df = read_raw_eco_db(eco_csv_path)
    logger.debug(f'Found {len(eco_df)} ECOSTRESS products')
    try:
        # Filter dates
        eco_df['min_date'] = pd.to_datetime(eco_df['Acquisition Start Date'])
        eco_df['max_date'] = pd.to_datetime(eco_df['Acquisition End Date'])
        eco_df['date'] = eco_df['min_date'].dt.date 
        if min_date is not None and max_date is not None:
            eco_df = eco_df[ eco_df['date'] >= min_date.date() ]
            eco_df = eco_df[ eco_df['max_date'].dt.date <= max_date.date() ]
            logger.debug(f'Found {len(eco_df)} ECOSTRESS products after filtering dates')

        # Convert filtered dataframe to a geodataframe
        geometry = eco_df.apply(lambda row: create_polygon(row['NW Corner Lat dec'],
                                                     row['NW Corner Long dec'],
                                                     row['NE Corner Lat dec'],
                                                     row['NE Corner Long dec'],
                                                     row['SW Corner Lat dec'],
                                                     row['SW Corner Long dec'],
                                                     row['SE Corner Lat dec'],
                                                     row['SE Corner Long dec'],),axis=1)
        eco_df = gpd.GeoDataFrame(eco_df[['Local Granule ID','date']],
                           crs=4326,
                           geometry=geometry).rename(columns={'Local Granule ID':'product_name'})

        # Filter with ROI and compute area coverage
        if roi_bbox is not None:
            eco_df = filter_with_roi(eco_df, roi_bbox, roi_crs, min_roi_overlap)
            logger.debug(f'Found {len(eco_df)} ECOSTRESS products after filtering ROI')
    except AttributeError as e:
        if len(eco_df) == 0:
            logger.warning("No product found")
            return gpd.GeoDataFrame(columns=['product_name','date', 'geometry'],crs=4326)
        raise e

    if len(eco_df) == 0:
        logger.warning("No product found")
    else: 
        nb_eco_filtered = len(eco_df)
        logger.info(
            f'Found {nb_eco_filtered} ECOSTRESS products fullfilling the requirements'
        )

    return eco_df.reset_index(drop=True)


@lru_cache
def read_raw_s2_db(s2_csv_path:str) -> pd.DataFrame:
    """
    Read raw S2 database in CSV file format
    """
    s2_df = pd.read_csv(s2_csv_path,delimiter="\t", parse_dates=["acquisition_date"])
    return s2_df[ s2_df['level'] == "L2A" ]


def create_s2_db(s2_csv_path:str,
              min_date: datetime = None,
              max_date: datetime = None,
              roi_bbox: rio.coords.BoundingBox = None,
              roi_crs: str = None,
              min_roi_overlap: float = 33) -> gpd.GeoDataFrame:
    """
    Create database for S2 product from CSV metadata file

    """
    s2_df = read_raw_s2_db(s2_csv_path)
    logger.debug(f'Found {len(s2_df)} Sentinel2 products')
    try:
        # Filter dates
        s2_df['min_date'] = s2_df['acquisition_date']
        s2_df['date'] = s2_df['acquisition_date'].dt.date
        if min_date is not None and max_date is not None:
            s2_df = s2_df[ s2_df['date'] >= min_date.date() ]
            s2_df = s2_df[ s2_df['date'] <= max_date.date() ]
            logger.debug(f'Found {len(s2_df)} Sentinel2 products after filtering dates')

        # Convert filtered dataframe to a geodataframe
        mgrs_grid = gpd.read_file('/vsizip/' + os.path.join(os.path.dirname(os.path.abspath(mgrs.__file__)), 'data/sentinel2/mgrs_tiles.gpkg.zip', 'mgrs_tiles.gpkg'))
        s2_df = s2_df.merge(mgrs_grid,left_on="mgrs_tile",right_on="Name")
        s2_df = gpd.GeoDataFrame(s2_df[['product_id','date']],
                           crs=4326,
                           geometry=s2_df["geometry"]).rename(columns={'product_id':'product_name'})

        # Filter with ROI and compute area coverage
        if roi_bbox is not None:
            s2_df = filter_with_roi(s2_df, roi_bbox, roi_crs, min_roi_overlap)
            logger.debug(f'Found {len(s2_df)} Sentinel2 products after filtering ROI')
        # Filter with ROI and compute area coverage
    except AttributeError as e:
        if len(s2_df) == 0:
            logger.warning("No product found")
            return gpd.GeoDataFrame(columns=['product_name','date','geometry'],crs=4326)
        raise e

    if len(s2_df) == 0:
        logger.warning("No product found")
    else: 
        nb_s2_filtered = len(s2_df)
        logger.info(
            f'Found {nb_s2_filtered} Sentinel2 products fullfilling the requirements'
        )

    return s2_df.reset_index(drop=True)


def select_products(gdf1: gpd.GeoDataFrame,
                    gdf2: gpd.GeoDataFrame, 
                    delta: timedelta,
                    min_overlap: float):
    """
    For each product in the first list gdf1, 
    search for a product in the second list gdf2,
    whose acquisition date is within delta days of the date of the first product.
    """
    gotchas = []
    
    dates = gdf1['date'].unique()
    for d in tqdm(dates,
                  total=len(dates),
                  desc='Matching datasets ...'):
        # Select products in gdf1 corresponding to date d
        gdf1_selection = gdf1[gdf1['date'] == d]
        # Select products in gdf2 corresponding to date +/- delta
        gdf2_selection = gdf2[(gdf2['date'] >= d - delta) & (gdf2['date'] <= d + delta)]

        # If a combination exists
        if len(gdf2_selection) > 0:
            try:
                res_inter = gpd.overlay(gdf1_selection,
                                        gdf2_selection,
                                        how='intersection')
                if len(res_inter):
                    results = res_inter.copy()
                    overlaps = []
                    for r in results.itertuples():
                        overlap = 100 * r.geometry.area / gdf2[
                            gdf2['product_name'] == r.product_name_2].iloc[0].geometry.area
                        overlaps.append(overlap)
                    results['overlap'] = overlaps
                    gotchas.append(results)
            except AttributeError as e:
                logger.warning(e)

    if len(gotchas) == 0:
        logger.warning("No matching found")
        return pd.DataFrame()
        
    gotcha = pd.concat(gotchas)
    gotcha = gotcha[gotcha.overlap > min_overlap]
    if len(gotcha) == 0:
        logger.warning("No matching found")
    else:
        logger.info(
            f'Found {len(gotcha)} matches'
        )
    return gotcha.reset_index(drop=True)

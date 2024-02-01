#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright: (c) 2023 CESBIO / Centre National d'Etudes Spatiales / Université Paul Sabatier (UT3)
#
"""
Manage provider for THEIA, EarthData and earthExplorer
"""
from abc import abstractmethod
from dataclasses import dataclass, field

import os
import re
import pandas as pd
import geopandas as gpd
import rasterio as rio
import earthaccess
from enum import Enum
from datetime import datetime
from landsatxplore.api import API
from earthaccess.auth import Auth as EarthDataAuth
from earthaccess.results import DataGranule
from theia_picker.download import TheiaCatalog
from theia_picker.download import RequestsManager as TheiaAuth
from shapely.geometry import Polygon, Point
from sensorsio.sentinel2 import get_theia_tiles, find_tile_orbit_pairs
from sensorsio.mgrs import get_bbox_mgrs_tile
from etdataset.logging import LoggerManager
from etdataset.utils import get_optimal_relative_orbit_for_mgrs_tile, bbox_to_polygon, create_polygon, MGRS_FORMAT

logger = LoggerManager.get_logger(__name__)



class ProviderException(Exception):
    """
    Exception for Provider
    """

    pass


def AuthenticationException(Exception):
    """
    Exception for authentication
    """


@dataclass
class Provider:
    """
    Abstract class for provider
    """
    
    @abstractmethod
    def login(self) -> None:
        """
        Login to provider
        """
        pass

    @abstractmethod
    def search(self, 
               min_date: str, 
               max_date: str,
               tile_ids: list[str] | None = None,
               bbox_latlon: rio.coords.BoundingBox | None = None,
               max_cloud_cover: float = 20) -> gpd.GeoDataFrame:
        """
        Search on catalog
        """
        pass


@dataclass
class TheiaProvider(Provider):
    """
    Provider for Theia
    """

    name: str = field(default="THEIA")
    catalog: TheiaCatalog = field(init=False)
    auth: TheiaAuth = field(init=False)

    def __post_init__(self):
        """
        Initialize dataset
        """
        self.login()


    def login(self) -> None:
        """
        Login to Theia catalog
        """
        # Authentication
        try:
            username = os.environ["THEIA_IDENT"]
            password = os.environ["THEIA_PASS"]
        except KeyError:
            raise AuthenticationException("Variables THEIA_IDENT and THEIA_PASS must be set")

        self.catalog = TheiaCatalog(
                            credentials={"ident": username, "pass": password}
                            )
        self.auth = self.catalog._requests_mgr


    def search(self, 
               min_date: str, 
               max_date: str,
               tile_id: str | None = None,
               latlon_bbox: rio.coords.BoundingBox | None = None,
               max_cloud_cover: float = 20) -> gpd.GeoDataFrame:
        """
        Search on catalog
        """
        logger.debug(f"Search on THEIA catalog: min_date={min_date}, max_date = {max_date}, tile_id = {tile_id}, bbox = {latlon_bbox}, max_cloud_cover = {max_cloud_cover}")
        tile_name = None
        relative_orbit = None
        bbox = None
        if tile_id is not None:
            tile_name = "T"+tile_id
            relative_orbit = get_optimal_relative_orbit_for_mgrs_tile(tile_id)
        if latlon_bbox is not None:
            bbox = tuple(latlon_bbox)
        # Request
        results = self.catalog.search(
            start_date=min_date,
            end_date=max_date,
            tile_name=tile_name,
            level="LEVEL2A",
            bbox=bbox,
            relative_orbit_number=relative_orbit
            )
        logger.debug(f"Number of products found on Theia: {len(results)}")
        # Filter with cloud cover
        if len(results) > 0:
            results = [ result for result in results if result.properties.cloud_cover < max_cloud_cover ]
        logger.debug(f"Number of products found on Theia after cloud cover filtering: {len(results)}")
        # Convert to GeoDataFrame
        data = []
        geometry = []
        if len(results) > 0:
            data = [(result.properties.product_identifier,
                 result.properties.acquisition_date.date(), 
                 self.name,
                 result.properties.collection,
                 result.properties.tile[1:],
                 result.properties.cloud_cover,
                 result.properties.services.download.url,
                 result.properties.services.download.checksum) for result in results ]
            geometry = [Polygon(tuple(tuple(coord) for coord in result.geometry["coordinates"][0])) for result in results ]
        return gpd.GeoDataFrame(data=data, columns=["Product_name","Date","Provider","Collection","Tile_ID","Cloud_cover","URL","Checksum"],geometry=geometry, crs=4326)

@dataclass
class EarthDataProvider(Provider):
    """
    Provider for EarthData
    """

    short_name: str 
    doi: str
    collection: str
    name: str = "EarthData"
    auth: EarthDataAuth = field(init=False)

    def __post_init__(self):
        """
        Initialize dataset
        """
        self.login()


    def login(self) -> None:
        """
        Login to EarthData catalog
        """
        # Authentication
        try:
            username = os.environ["EARTHDATA_USERNAME"]
            password = os.environ["EARTHDATA_PASSWORD"]
        except KeyError:
            raise AuthenticationException("Variables EARTHDATA_USERNAME and EARTHDATA_PASSWORD must be set")
        self.auth = earthaccess.login()


    def _get_geometry(self,result: DataGranule) -> None:
        """
        Extract geometry from result
        """
        return None


    def _get_tile_id(self,result: DataGranule) -> None:
        """
        Extract tile ID from result
        """
        return None


    def _get_cloud_cover(self,result: DataGranule) -> None:
        """
        Extract tile ID from result
        """
        return None


    def _get_date(self, result: DataGranule) -> None:
        """
        Extract acquisition date
        """
        return datetime.strptime(result["umm"]["TemporalExtent"]["RangeDateTime"]["BeginningDateTime"].split("T")[0], "%Y-%m-%d").date()

    def search(self, 
               min_date: str, 
               max_date: str,
               tile_id: str | None = None,
               latlon_bbox: rio.coords.BoundingBox | None = None,
               max_cloud_cover: float = 20) -> gpd.GeoDataFrame:
        """
        Search on catalog
        """
        logger.debug(f"Search on EarthData catalog: min_date={min_date}, max_date = {max_date}, tile_id = {tile_id}, bbox = {latlon_bbox}, max_cloud_cover = {max_cloud_cover}")
        bbox = None
        if tile_id is not None:
            # Extract ROI
            bbox = tuple(get_bbox_mgrs_tile(tile_id))
        if latlon_bbox is not None:
            bbox = tuple(latlon_bbox)
        # Request
        results = earthaccess.search_data(
                    short_name=self.short_name,
                    doi=self.doi,
                    temporal = (min_date,max_date),
                    bounding_box = bbox,
                                )
        logger.debug(f"Number of products found on EarthData: {len(results)}")
        # Convert to GeoDataFrame
        data = []
        geometry = []
        if len(results) > 0:
            data = [[result["umm"]["GranuleUR"],
                     self._get_date(result),
                     self.name,
                     self.collection, 
                     self._get_tile_id(result),
                     self._get_cloud_cover(result),
                     result.data_links(),
                     None] 
                     for result in results ]
            geometry = [ self._get_geometry(result) for result in results ]
        gdf = gpd.GeoDataFrame(data=data, columns=["Product_name","Date","Provider","Collection","Tile_ID","Cloud_cover","URL","Checksum"],geometry=geometry, crs=4326)
        # Filter one specific tile if tile_id is provided
        if tile_id is not None:
            gdf = gdf[gdf["Tile_ID"] == tile_id]
            logger.debug(f"Number of products found on EarthData after tile filtering: {len(gdf)}")
        # Filter on cloud cover if the information exists
        if gdf[["Cloud_cover"]].notna().any().any():
            gdf = gdf[gdf["Cloud_cover"] < max_cloud_cover]
            logger.debug(f"Number of products found on EarthData after cloud cover filtering: {len(gdf)}")
        return gdf


@dataclass
class EcostressProvider(EarthDataProvider):
    """
    Provider for Ecostress via EarthData
    """
    short_name: str = "ECO_L2T_LSTE"
    doi: str = "10.5067/ECOSTRESS/ECO_L2T_LSTE.002"
    collection: str = "Ecostress"
    name: str = "EarthData"

    def _get_geometry(self,result) -> Polygon:
        """
        Extract geometry from result
        """
        return bbox_to_polygon(list(result["umm"]["SpatialExtent"]['HorizontalSpatialDomain']['Geometry']['BoundingRectangles'][0].values())) 


    def _get_tile_id(self,result) -> str:
        """
        Extract tile ID from result
        """
        return MGRS_FORMAT.search(result["umm"]["GranuleUR"]).group(0)


@dataclass
class HLSSProvider(EarthDataProvider):
    """
    Provider for HLS Sentinel2 via EarthData
    """
    short_name: str = "HLSS30"
    doi: str = "10.5067/HLS/HLSS30.002"
    collection: str = "HLSSentinel2"
    name: str = "EarthData"

    def _get_geometry(self,result) -> None:
        """
        Extract geometry from result
        """
        coords = result["umm"]["SpatialExtent"]["HorizontalSpatialDomain"]["Geometry"]["GPolygons"][0]["Boundary"]["Points"]
        points = [ Point(coord["Longitude"],coord["Latitude"]) for coord in coords]
        return Polygon(points)


    def _get_tile_id(self,result) -> str:
        """
        Extract tile ID from result
        """
        elt = next(filter(lambda e: e['Name'] == 'MGRS_TILE_ID',result["umm"]["AdditionalAttributes"]))
        if elt is not None :
            elt = elt["Values"][0]
        return elt


    def _get_cloud_cover(self,result) -> float:
        """
        Extract tile ID from result
        """
        elt = next(filter(lambda e: e['Name'] == 'CLOUD_COVERAGE',result["umm"]["AdditionalAttributes"]))
        if elt is not None :
            elt = float(elt["Values"][0])
        return elt


@dataclass
class HLSLProvider(EarthDataProvider):
    """
    Provider for HLS Landsat via EarthData
    """
    short_name: str = "HLSL30"
    doi: str = "10.5067/HLS/HLSL30.002"
    collection: str = "HLSLandsat"
    name: str = "EarthData"

    def _get_geometry(self,result) -> None:
        """
        Extract geometry from result
        """
        coords = result["umm"]["SpatialExtent"]["HorizontalSpatialDomain"]["Geometry"]["GPolygons"][0]["Boundary"]["Points"]
        points = [ Point(coord["Longitude"],coord["Latitude"]) for coord in coords]
        return Polygon(points)


    def _get_tile_id(self,result) -> None:
        """
        Extract tile ID from result
        """
        elt = next(filter(lambda e: e['Name'] == 'MGRS_TILE_ID',result["umm"]["AdditionalAttributes"]))
        if elt is not None :
            elt = elt["Values"][0]
        return elt


    def _get_cloud_cover(self,result) -> None:
        """
        Extract tile ID from result
        """
        elt = next(filter(lambda e: e['Name'] == 'CLOUD_COVERAGE',result["umm"]["AdditionalAttributes"]))
        if elt is not None :
            elt = float(elt["Values"][0])
        return elt


@dataclass
class LandsatProvider(Provider):
    """
    Provider for Landsat via EarthExplorer
    """

    name: str = "EarthExplorer"
    catalog: API = field(init=False)
    dataset: str = "landsat_ot_c2_l2"
    collection: str = "Landsat"

    def __post_init__(self):
        """
        Initialize dataset
        """
        self.login()


    def login(self) -> None:
        """
        Login to EarthExplorer catalog
        """
        # Authentication
        try:
            username = os.environ["LANDSATXPLORE_USERNAME"]
            password = os.environ["LANDSATXPLORE_PASSWORD"]
        except KeyError:
            raise AuthenticationException("Variables LANDSATXPLORE_USERNAME and LANDSATXPLORE_PASSWORD must be set")
        
        # Initialize a new API instance and get an access key
        self.catalog = API(username, password)


    def _get_geometry(self,result) -> None:
        """
        Extract geometry from result
        """
        return create_polygon(
              result['corner_upper_left_latitude'],
              result['corner_upper_left_longitude'],
              result['corner_upper_right_latitude'] ,
              result['corner_upper_right_longitude'],
              result['corner_lower_left_latitude'],
              result['corner_lower_left_longitude'],
              result['corner_lower_right_latitude'],
              result['corner_lower_right_longitude'],
               )


    def search(self, 
               min_date: str, 
               max_date: str,
               tile_id: str | None = None,
               latlon_bbox: rio.coords.BoundingBox | None = None,
               max_cloud_cover: float = 20) -> gpd.GeoDataFrame:
        """
        Search on catalog
        """
        logger.debug(f"Search on EarthExplorer catalog: min_date={min_date}, max_date = {max_date}, tile_id = {tile_id}, bbox = {latlon_bbox}, max_cloud_cover = {max_cloud_cover}")
        bbox = None
        if tile_id is not None:
            # Extract ROI
            bbox = tuple(get_bbox_mgrs_tile(tile_id))
        if latlon_bbox is not None:
            bbox = tuple(latlon_bbox)
        # Request
        results = self.catalog.search(
            dataset=self.dataset,
            bbox=bbox,
            start_date=min_date,
            end_date=max_date,
            max_cloud_cover=max_cloud_cover
        )
        logger.debug(f"Number of products found on EarthExplorer: {len(results)}")
        # Convert to GeoDataFrame
        data = []
        geometry = []
        if len(results) > 0:
            data = [[
                   result['display_id'],
                   result['acquisition_date'].date(),
                   self.name,
                   self.collection,
                   None,
                   result['cloud_cover'],
                   result['landsat_product_id'],
                   None]
                   for result in results]
            geometry = [self._get_geometry(result) for result in results]
        gdf = gpd.GeoDataFrame(data=data, columns=["Product_name","Date","Provider","Collection","Tile_ID","Cloud_cover","URL","Checksum"],geometry=geometry, crs=4326)
        # Filter on cloud cover if the information exists
        gdf = gdf[gdf["Cloud_cover"] < max_cloud_cover]
        logger.debug(f"Number of products found on EarthData after cloud cover filtering: {len(gdf)}")
        return gdf

class Collection(Enum):
    ECOSTRESS = EcostressProvider
    LANDSAT = LandsatProvider
    SENTINEL2 = TheiaProvider
    HLSS = HLSSProvider
    HLSL = HLSLProvider

def get_provider(collection: Collection) -> Provider:
    """
    Get the provider from a collection
    """
    logger.debug(f"Collection: {collection}")
    logger.debug(f"Provider: {collection.value}")
    return collection.value()

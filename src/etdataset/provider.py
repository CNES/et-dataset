#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright: (c) 2023 CESBIO / Centre National d'Etudes Spatiales / Université Paul Sabatier (UT3)
#
"""
Manage provider for THEIA, EarthData and earthExplorer
"""
import os
import re
from abc import abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

import earthaccess
import geopandas as gpd
import pandas as pd
import rasterio as rio
from earthaccess.auth import Auth as EarthDataAuth
from earthaccess.results import DataGranule
from landsatxplore.api import API
from landsatxplore.earthexplorer import EarthExplorer
from sensorsio import mgrs
from sensorsio.sentinel2 import find_tile_orbit_pairs
from shapely.geometry import Point, Polygon
from theia_picker.download import Feature
from theia_picker.download import RequestsManager as TheiaAuth
from theia_picker.download import TheiaCatalog

from etdataset.logging import LoggerManager
from etdataset.utils import MGRS_FORMAT, bbox_to_polygon, create_polygon

logger = LoggerManager.get_logger(__name__)


class ProviderException(Exception):
    """
    Exception for Provider
    """


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
    def search(
        self,
        min_date: str,
        max_date: str,
        tile_ids: list[str] | None = None,
        bbox_latlon: rio.coords.BoundingBox | None = None,
        max_cloud_cover: float = 20,
    ) -> gpd.GeoDataFrame:
        """
        Search on catalog
        """
        pass

    @abstractmethod
    def download(self, products: pd.DataFrame, local_path: str = os.getcwd()) -> None:
        """
        Download produtcs from catalog
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
            raise AuthenticationException(
                "Variables THEIA_IDENT and THEIA_PASS must be set"
            )

        self.catalog = TheiaCatalog(credentials={"ident": username, "pass": password})
        self.auth = self.catalog.requests_mgr

    def _get_optimal_relative_orbit_for_mgrs_tile(self, tile_id: str) -> int:
        """
        Given a MGRS tile return the best relative orbit
        """
        tile_bbox = mgrs.get_bbox_mgrs_tile(tile_id)
        # Convert bounds to polygon
        aoi = Polygon(
            [
                [tile_bbox[0], tile_bbox[1]],
                [tile_bbox[0], tile_bbox[3]],
                [tile_bbox[2], tile_bbox[3]],
                [tile_bbox[2], tile_bbox[1]],
            ]
        )

        orbits_df = gpd.read_file(
            os.path.join(
                os.path.dirname(os.path.abspath(mgrs.__file__)),
                "data/sentinel2/orbits.gpkg",
            )
        )
        intersections = []
        orbits = []
        for _, orbit_row in orbits_df.iterrows():
            # Last test is to exclude weird duplicates (malformed gpkg ?)
            if (
                orbit_row.geometry.intersects(aoi)
                and orbit_row.orbit_number not in orbits
            ):
                orbits.append(orbit_row.orbit_number)
                inter_aoi_orbit = aoi.intersection(orbit_row.geometry)
                mgrs_orbit_coverage = inter_aoi_orbit.area / aoi.area
                intersections.append((orbit_row.orbit_number, mgrs_orbit_coverage))
        labels = ["relative_orbit_number", "tile_and_orbit_coverage"]
        return int(
            pd.DataFrame.from_records(intersections, columns=labels)
            .sort_values(by="tile_and_orbit_coverage", ascending=False)
            .iloc[0]
            .relative_orbit_number
        )

    def _filter(
        self,
        results: gpd.GeoDataFrame,
        latlon_bbox: rio.coords.BoundingBox,
    ) -> gpd.GeoDataFrame:
        """
        Given a ROI, filter with tile_id and relative_orbit_number
        """
        to_keep = find_tile_orbit_pairs(latlon_bbox, 4326)
        to_supp = to_keep[to_keep["tile_and_orbit_coverage"] <= 0.1][
            ["tile_id", "relative_orbit_number"]
        ].values
        logger.debug(f"Tiles to remove: {to_supp}")
        to_keep = to_keep[to_keep["tile_and_orbit_coverage"] > 0.1]
        idx = (
            to_keep.groupby(["tile_id"])["tile_and_orbit_coverage"].transform("max")
            == to_keep["tile_and_orbit_coverage"]
        )
        to_keep = to_keep[idx][["tile_id", "relative_orbit_number"]].values
        logger.debug(f"Tiles to keep: {to_keep}")
        for id, orbit in to_supp:
            results = results.drop(
                results[
                    (results.Tile_ID == str(id))
                    & (results.Relative_orbit == int(orbit))
                ].index
            )
        for id, orbit in to_keep:
            results = results.drop(
                results[
                    (results.Tile_ID == str(id))
                    & (results.Relative_orbit != int(orbit))
                ].index
            )
        return results

    def search(
        self,
        min_date: str,
        max_date: str,
        tile_id: str | None = None,
        latlon_bbox: rio.coords.BoundingBox | None = None,
        max_cloud_cover: float = 20,
    ) -> gpd.GeoDataFrame:
        """
        Search on catalog
        """
        logger.debug(
            f"Search on THEIA catalog: min_date={min_date}, max_date = {max_date}, "
            f"tile_id = {tile_id}, bbox = {latlon_bbox}, max_cloud_cover = {max_cloud_cover}"
        )
        tile_name = None
        relative_orbit = None
        bbox = None
        if tile_id is not None:
            tile_name = "T" + tile_id
            relative_orbit = self._get_optimal_relative_orbit_for_mgrs_tile(tile_id)
        if latlon_bbox is not None:
            bbox = tuple(latlon_bbox)
        # Request
        results = self.catalog.search(
            start_date=min_date,
            end_date=(
                datetime.strptime(max_date, "%Y-%m-%d") + timedelta(days=1)
            ).strftime("%Y-%m-%d"),
            tile_name=tile_name,
            level="LEVEL2A",
            bbox=bbox,
            relative_orbit_number=relative_orbit,
        )
        logger.debug(f"Number of products found on Theia: {len(results)}")
        # Filter with cloud cover
        if len(results) > 0:
            results = [
                result
                for result in results
                if result.properties.cloud_cover < max_cloud_cover
            ]
        logger.debug(
            f"Number of products found on Theia after cloud cover filtering: {len(results)}"
        )
        # Convert to GeoDataFrame
        url_pattern = re.compile("(.*?/download/)")
        data = []
        geometry = []
        if len(results) > 0:
            data = [
                (
                    result.properties.product_identifier,
                    result.properties.acquisition_date.date(),
                    self.name,
                    result.properties.collection,
                    result.properties.tile[1:],
                    result.properties.cloud_cover,
                    result.properties.relative_orbit_number,
                    url_pattern.search(result.properties.services.download.url).group(
                        0
                    ),
                    result.properties.services.download.checksum,
                )
                for result in results
            ]
            geometry = [Polygon(result.geometry.polygon[0]) for result in results]
        gdf = gpd.GeoDataFrame(
            data=data,
            columns=[
                "Product_name",
                "Date",
                "Provider",
                "Collection",
                "Tile_ID",
                "Cloud_cover",
                "Relative_orbit",
                "URL",
                "Checksum",
            ],
            geometry=geometry,
            crs=4326,
        )
        if latlon_bbox is not None:
            gdf = self._filter(gdf, latlon_bbox)
            logger.debug(
                f"Number of products found on Theia after tile filtering: {len(gdf)}"
            )
        return gdf

    def download(self, products: pd.DataFrame, local_path: str = os.getcwd()) -> None:
        """
        Download produtcs from catalog
        """
        logger.debug(f"List of products to download: {products['Product_name'].values}")
        url_id = re.compile("SENTINEL2/(.*?)/download/")
        for _, product in products.iterrows():
            # Create feature instance
            data = {
                "type": "Feature",
                "id": url_id.search(product.URL).group(1),
                "properties": {
                    "collection": "",
                    "productIdentifier": product.Product_name,
                    "title": "",
                    "productType": "",
                    "startDate": datetime.now(),
                    "processingLevel": "",
                    "waterCover": 0,
                    "snowCover": 0,
                    "cloudCover": 0,
                    "relativeOrbitNumber": 0,
                    "location": "",
                    "services": {
                        "download": {
                            "url": product.URL,
                            "mimeType": "application/zip",
                            "checksum": product.Checksum,
                        }
                    },
                },
                "geometry": {"coordinates": list()},
            }
            feature = Feature(requests_mgr=self.auth, **data)
            # Download
            feature.download_archive(download_dir=local_path, renew_token=True)


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
        if self.__class__ == EarthDataProvider:
            raise TypeError("Cannot instantiate EarthDataProvider class.")

    def login(self) -> None:
        """
        Login to EarthData catalog
        """
        # Authentication
        try:
            _ = os.environ["EARTHDATA_USERNAME"]
            _ = os.environ["EARTHDATA_PASSWORD"]
        except KeyError:
            raise AuthenticationException(
                "Variables EARTHDATA_USERNAME and EARTHDATA_PASSWORD must be set"
            )
        self.auth = earthaccess.login()

    def _get_geometry(self, result: DataGranule) -> None:
        """
        Extract geometry from result
        """
        return None

    def _get_tile_id(self, result: DataGranule) -> None:
        """
        Extract tile ID from result
        """
        return None

    def _get_cloud_cover(self, result: DataGranule) -> None:
        """
        Extract tile ID from result
        """
        return None

    def _get_date(self, result: DataGranule) -> None:
        """
        Extract acquisition date
        """
        return datetime.strptime(
            result["umm"]["TemporalExtent"]["RangeDateTime"]["BeginningDateTime"].split(
                "T"
            )[0],
            "%Y-%m-%d",
        ).date()

    def search(
        self,
        min_date: str,
        max_date: str,
        tile_id: str | None = None,
        latlon_bbox: rio.coords.BoundingBox | None = None,
        max_cloud_cover: float = 20,
    ) -> gpd.GeoDataFrame:
        """
        Search on catalog
        """
        logger.debug(
            f"Search on EarthData catalog: min_date={min_date}, max_date = {max_date}, "
            f"tile_id = {tile_id}, bbox = {latlon_bbox}, max_cloud_cover = {max_cloud_cover}"
        )
        bbox = None
        if tile_id is not None:
            # Extract ROI
            bbox = tuple(mgrs.get_bbox_mgrs_tile(tile_id))
        if latlon_bbox is not None:
            bbox = tuple(latlon_bbox)
        # Request
        results = earthaccess.search_data(
            short_name=self.short_name,
            doi=self.doi,
            temporal=(
                min_date,
                (datetime.strptime(max_date, "%Y-%m-%d") + timedelta(days=1)).strftime(
                    "%Y-%m-%d"
                ),
            ),
            bounding_box=bbox,
        )
        logger.debug(f"Number of products found on EarthData: {len(results)}")
        # Convert to GeoDataFrame
        data = []
        geometry = []
        if len(results) > 0:
            data = [
                [
                    result["umm"]["GranuleUR"],
                    self._get_date(result),
                    self.name,
                    self.collection,
                    self._get_tile_id(result),
                    self._get_cloud_cover(result),
                    None,
                    ",".join(result.data_links()),
                    None,
                ]
                for result in results
            ]
            geometry = [self._get_geometry(result) for result in results]
        gdf = gpd.GeoDataFrame(
            data=data,
            columns=[
                "Product_name",
                "Date",
                "Provider",
                "Collection",
                "Tile_ID",
                "Cloud_cover",
                "Relative_orbit",
                "URL",
                "Checksum",
            ],
            geometry=geometry,
            crs=4326,
        )
        # Filter one specific tile if tile_id is provided
        if tile_id is not None:
            gdf = gdf[gdf["Tile_ID"] == tile_id]
            logger.debug(
                f"Number of products found on EarthData after tile filtering: {len(gdf)}"
            )
        # Filter on cloud cover if the information exists
        if gdf[["Cloud_cover"]].notna().any().any():
            gdf = gdf[gdf["Cloud_cover"] < max_cloud_cover]
            logger.debug(
                f"Number of products found on EarthData after cloud cover filtering: {len(gdf)}"
            )
        return gdf

    def download(self, products: pd.DataFrame, local_path: str = os.getcwd()) -> None:
        """
        Download produtcs from catalog
        """
        logger.debug(f"List of products to download: {products['Product_name'].values}")
        results = []
        for _, product in products.iterrows():
            product_name = product.Product_name
            product_dir = os.path.join(local_path, product_name)
            os.makedirs(os.path.join(local_path, product_name), exist_ok=True)
            # Parse URL
            urls = [url for url in product.URL.split(",")]
            results.append(earthaccess.download(urls, product_dir))
        logger.info(f"Download products: {results}")


@dataclass
class EcostressProvider(EarthDataProvider):
    """
    Provider for Ecostress via EarthData
    """

    short_name: str = "ECO_L2T_LSTE"
    doi: str = "10.5067/ECOSTRESS/ECO_L2T_LSTE.002"
    collection: str = "ECOSTRESS"
    name: str = "EarthData"

    def _get_geometry(self, result) -> Polygon:
        """
        Extract geometry from result
        """
        geom = result["umm"]["SpatialExtent"]["HorizontalSpatialDomain"]["Geometry"][
            "BoundingRectangles"
        ][0]
        return bbox_to_polygon(
            [
                geom["WestBoundingCoordinate"],
                geom["SouthBoundingCoordinate"],
                geom["EastBoundingCoordinate"],
                geom["NorthBoundingCoordinate"],
            ]
        )

    def _get_tile_id(self, result) -> str:
        """
        Extract tile ID from result
        """
        return MGRS_FORMAT.search(result["umm"]["GranuleUR"]).group(0)


@dataclass
class HLSProvider(EarthDataProvider):
    """
    Provider for HLS
    """

    def __post_init__(self):
        """
        Initialize dataset
        """
        self.login()
        if self.__class__ == EarthDataProvider:
            raise TypeError("Cannot instantiate EarthDataProvider class.")

    def _get_geometry(self, result) -> None:
        """
        Extract geometry from result
        """
        coords = result["umm"]["SpatialExtent"]["HorizontalSpatialDomain"]["Geometry"][
            "GPolygons"
        ][0]["Boundary"]["Points"]
        points = [Point(coord["Longitude"], coord["Latitude"]) for coord in coords]
        return Polygon(points)

    def _get_tile_id(self, result) -> str:
        """
        Extract tile ID from result
        """
        elt = next(
            filter(
                lambda e: e["Name"] == "MGRS_TILE_ID",
                result["umm"]["AdditionalAttributes"],
            )
        )
        if elt is not None:
            elt = elt["Values"][0]
        return elt

    def _get_cloud_cover(self, result) -> float:
        """
        Extract tile ID from result
        """
        elt = next(
            filter(
                lambda e: e["Name"] == "CLOUD_COVERAGE",
                result["umm"]["AdditionalAttributes"],
            )
        )
        if elt is not None:
            elt = float(elt["Values"][0])
        return elt


@dataclass
class HLSSProvider(HLSProvider):
    """
    Provider for HLS Sentinel2 via EarthData
    """

    short_name: str = "HLSS30"
    doi: str = "10.5067/HLS/HLSS30.002"
    collection: str = "HLSSENTINEL2"


@dataclass
class HLSLProvider(HLSProvider):
    """
    Provider for HLS Landsat via EarthData
    """

    short_name: str = "HLSL30"
    doi: str = "10.5067/HLS/HLSL30.002"
    collection: str = "HLSLANDSAT"


@dataclass
class LandsatProvider(Provider):
    """
    Provider for Landsat via EarthExplorer
    """

    name: str = "EarthExplorer"
    catalog: API = field(init=False)
    dataset: str = "landsat_ot_c2_l2"
    collection: str = "LANDSAT"
    auth: EarthExplorer = field(init=False)

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
            raise AuthenticationException(
                "Variables LANDSATXPLORE_USERNAME and LANDSATXPLORE_PASSWORD must be set"
            )

        # Initialize a new API instance and get an access key
        self.catalog = API(username, password)
        self.auth = EarthExplorer(username, password)

    def _get_geometry(self, result) -> None:
        """
        Extract geometry from result
        """
        return create_polygon(
            result["corner_upper_left_latitude"],
            result["corner_upper_left_longitude"],
            result["corner_upper_right_latitude"],
            result["corner_upper_right_longitude"],
            result["corner_lower_left_latitude"],
            result["corner_lower_left_longitude"],
            result["corner_lower_right_latitude"],
            result["corner_lower_right_longitude"],
        )

    def search(
        self,
        min_date: str,
        max_date: str,
        tile_id: str | None = None,
        latlon_bbox: rio.coords.BoundingBox | None = None,
        max_cloud_cover: float = 20,
    ) -> gpd.GeoDataFrame:
        """
        Search on catalog
        """
        logger.debug(
            f"Search on EarthExplorer catalog: min_date={min_date}, max_date = {max_date}, "
            f"tile_id = {tile_id}, bbox = {latlon_bbox}, max_cloud_cover = {max_cloud_cover}"
        )
        bbox = None
        if tile_id is not None:
            # Extract ROI
            bbox = tuple(mgrs.get_bbox_mgrs_tile(tile_id))
        if latlon_bbox is not None:
            bbox = tuple(latlon_bbox)
        # Request
        results = self.catalog.search(
            dataset=self.dataset,
            bbox=bbox,
            start_date=min_date,
            end_date=max_date,
            max_cloud_cover=max_cloud_cover,
        )
        logger.debug(f"Number of products found on EarthExplorer: {len(results)}")
        # Convert to GeoDataFrame
        data = []
        geometry = []
        if len(results) > 0:
            data = [
                [
                    result["display_id"],
                    result["acquisition_date"].date(),
                    self.name,
                    self.collection,
                    None,
                    result["cloud_cover"],
                    None,
                    result["landsat_product_id"],
                    None,
                ]
                for result in results
            ]
            geometry = [self._get_geometry(result) for result in results]
        gdf = gpd.GeoDataFrame(
            data=data,
            columns=[
                "Product_name",
                "Date",
                "Provider",
                "Collection",
                "Tile_ID",
                "Cloud_cover",
                "Relative_orbit",
                "URL",
                "Checksum",
            ],
            geometry=geometry,
            crs=4326,
        )
        # Filter on cloud cover if the information exists
        gdf = gdf[gdf["Cloud_cover"] < max_cloud_cover]
        logger.debug(
            f"Number of products found on EarthData after cloud cover filtering: {len(gdf)}"
        )
        return gdf

    def download(self, products: pd.DataFrame, local_path: str = os.getcwd()) -> None:
        """
        Download produtcs from catalog
        """
        logger.debug(f"List of products to download: {products['Product_name'].values}")
        for _, product in products.iterrows():
            if os.path.isfile(os.path.join(local_path, product.URL + ".tar")):
                logger.info(
                    f"Product directory for {product.Product_name} exists already. Skip product..."
                )
                continue
            self.auth.download(product.URL, output_dir=local_path)


class Collection(Enum):
    ECOSTRESS = EcostressProvider
    LANDSAT = LandsatProvider
    SENTINEL2 = TheiaProvider
    HLSSENTINEL2 = HLSSProvider
    HLSLANDSAT = HLSLProvider


def get_provider(collection: Collection) -> Provider:
    """
    Get the provider from a collection
    """
    logger.debug(f"Collection: {collection}")
    logger.debug(f"Provider: {collection.value}")
    return collection.value()

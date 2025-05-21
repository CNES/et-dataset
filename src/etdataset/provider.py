#
# Copyright: (c) 2023 CESBIO / Centre National d'Etudes Spatiales /
#             Université Paul Sabatier (UT3)
#
"""
Manage provider for THEIA, EarthData and earthExplorer
"""

from __future__ import annotations

import json  # to open and read json files
import os
import re
from abc import abstractmethod
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Any

import earthaccess
import geopandas as gpd
import pandas as pd
import rasterio as rio
import requests  # to send url requests
from earthaccess.auth import Auth as EarthDataAuth
from earthaccess.results import DataGranule
from sensorsio import mgrs
from sensorsio.sentinel2 import find_tile_orbit_pairs
from shapely.geometry import Point, Polygon
from theia_picker.download import Feature, TheiaCatalog
from theia_picker.download import RequestsManager as TheiaAuth
from tqdm import tqdm  # to print progress bars

from etdataset.logging import LoggerManager
from etdataset.utils import MGRS_FORMAT, bbox_to_polygon

logger = LoggerManager.get_logger(__name__)


class ProviderException(Exception):
    """
    Exception for Provider
    """


class AuthenticationException(Exception):
    """
    Exception for authentication
    """


class RequestException(Exception):
    """
    Exception for requests
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

    @abstractmethod
    def search(
        self,
        min_date: str,
        max_date: str,
        tile_ids: str | None = None,
        bbox_latlon: rio.coords.BoundingBox | None = None,
        max_cloud_cover: float = 20,
    ) -> gpd.GeoDataFrame:
        """
        Search on catalog
        """

    @abstractmethod
    def download(
        self, products: pd.DataFrame, local_path: str = os.getcwd()
    ) -> None:
        """
        Download produtcs from catalog
        """


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

        self.catalog = TheiaCatalog(
            credentials={"ident": username, "pass": password}
        )
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
                intersections.append(
                    (orbit_row.orbit_number, mgrs_orbit_coverage)
                )
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
            to_keep.groupby(["tile_id"])["tile_and_orbit_coverage"].transform(
                "max"
            )
            == to_keep["tile_and_orbit_coverage"]
        )
        to_keep = to_keep[idx][["tile_id", "relative_orbit_number"]].values
        logger.debug(f"Tiles to keep: {to_keep}")
        for i, orbit in to_supp:
            results = results.drop(
                results[
                    (results.Tile_ID == str(i))
                    & (results.Relative_orbit == int(orbit))
                ].index
            )
        for i, orbit in to_keep:
            results = results.drop(
                results[
                    (results.Tile_ID == str(i))
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
            f"Search on THEIA catalog: min_date={min_date}, "
            f"max_date = {max_date}, "
            f"tile_id = {tile_id}, bbox = {latlon_bbox}, "
            f"max_cloud_cover = {max_cloud_cover}"
        )
        tile_name = None
        relative_orbit = None
        bbox = None
        if tile_id is not None:
            tile_name = "T" + tile_id
            relative_orbit = self._get_optimal_relative_orbit_for_mgrs_tile(
                tile_id
            )
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
            "Number of products found on Theia after cloud "
            f"cover filtering: {len(results)}"
        )
        # Convert to GeoDataFrame
        url_pattern = re.compile("(.*?/download/)")
        data: list[tuple[str, date, str, str, str, str, str, str, str]] = []
        geometry = []
        if len(results) > 0:
            for result in results:
                url_matching = url_pattern.search(
                    result.properties.services.download.url
                )
                if url_matching is None:
                    continue
                url = url_matching.group(0)
                data.append(
                    (
                        result.properties.product_identifier,
                        result.properties.acquisition_date.date(),
                        self.name,
                        result.properties.collection,
                        result.properties.tile[1:],
                        result.properties.cloud_cover,
                        result.properties.relative_orbit_number,
                        url,
                        result.properties.services.download.checksum,
                    )
                )
            geometry = [
                Polygon(result.geometry.polygon[0]) for result in results
            ]
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
                "Number of products found on Theia after "
                f"tile filtering: {len(gdf)}"
            )
        return gdf

    def download(
        self, products: pd.DataFrame, local_path: str = os.getcwd()
    ) -> None:
        """
        Download produtcs from catalog
        """
        logger.debug(
            f"List of products to download: {products['Product_name'].values}"
        )
        url_id = re.compile("SENTINEL2/(.*?)/download/")
        for _, product in products.iterrows():
            # Create feature instance
            url_id_matching = url_id.search(product.URL)
            if url_id_matching is None:
                continue
            data_id = url_id_matching.group(1)
            data: dict[str, Any] = {
                "type": "Feature",
                "id": data_id,
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
                "geometry": {"coordinates": []},
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
                "Variables EARTHDATA_USERNAME and "
                "EARTHDATA_PASSWORD must be set"
            )
        self.auth = earthaccess.login()

    @abstractmethod
    def _get_geometry(self, result: DataGranule) -> Polygon:
        """
        Extract geometry from result
        """

    @abstractmethod
    def _get_tile_id(self, result: DataGranule) -> str:
        """
        Extract tile ID from result
        """

    @abstractmethod
    def _get_cloud_cover(self, result: DataGranule) -> float:
        """
        Extract tile ID from result
        """

    def _get_date(self, result: DataGranule) -> date:
        """
        Extract acquisition date
        """
        return datetime.strptime(
            result["umm"]["TemporalExtent"]["RangeDateTime"][
                "BeginningDateTime"
            ].split("T")[0],
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
            f"Search on EarthData catalog: min_date={min_date}, "
            f"max_date = {max_date}, "
            f"tile_id = {tile_id}, bbox = {latlon_bbox}, "
            f"max_cloud_cover = {max_cloud_cover}"
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
                (
                    datetime.strptime(max_date, "%Y-%m-%d") + timedelta(days=1)
                ).strftime("%Y-%m-%d"),
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
                "Number of products found on EarthData after "
                f"tile filtering: {len(gdf)}"
            )
        # Filter on cloud cover if the information exists
        if gdf[["Cloud_cover"]].notna().any().any():
            gdf = gdf[gdf["Cloud_cover"] < max_cloud_cover]
            logger.debug(
                "Number of products found on EarthData after cloud "
                f"cover filtering: {len(gdf)}"
            )
        return gdf

    def download(
        self, products: pd.DataFrame, local_path: str = os.getcwd()
    ) -> None:
        """
        Download produtcs from catalog
        """
        logger.debug(
            f"List of products to download: {products['Product_name'].values}"
        )
        results = []
        for _, product in products.iterrows():
            product_name = product.Product_name
            product_dir = os.path.join(local_path, product_name)
            os.makedirs(os.path.join(local_path, product_name), exist_ok=True)
            # Parse URL
            urls = list(product.URL.split(","))
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
        geom = result["umm"]["SpatialExtent"]["HorizontalSpatialDomain"][
            "Geometry"
        ]["BoundingRectangles"][0]
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
        matching = MGRS_FORMAT.search(result["umm"]["GranuleUR"])
        if matching is None:
            raise ValueError("Tile not found in result")
        return matching.group(0)


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

    def _get_geometry(self, result) -> Polygon:
        """
        Extract geometry from result
        """
        coords = result["umm"]["SpatialExtent"]["HorizontalSpatialDomain"][
            "Geometry"
        ]["GPolygons"][0]["Boundary"]["Points"]
        points = [
            Point(coord["Longitude"], coord["Latitude"]) for coord in coords
        ]
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
    Provider for Landsat via USGS' m2m api
    """

    name: str = "USGS"
    dataset: str = "landsat_ot_c2_l2"
    collection: str = "LANDSAT"
    url: str = "https://m2m.cr.usgs.gov/api/api/json/stable/"
    auth: str | None = None

    def __post_init__(self):
        """
        Initialize dataset
        """
        self.login()

    def sendRequest(self, service: str, data: dict) -> str:
        """
        Send http request

        Arguments
        =========

        1. service: ``str``
            usgs service (login-token, scene-search, download-options,
            download-request, download-retrieve)
        2. data: ``dict``
            payload for request

        Returns
        =======

        1. output['data']: ``str``
            result of the http request
        """

        # Format data with json
        json_data = json.dumps(data)

        if self.auth is None:
            response = requests.post(self.url + service, json_data)
        else:
            headers = {"X-Auth-Token": self.auth}
            response = requests.post(
                self.url + service, json_data, headers=headers
            )

        # Try request
        try:
            httpStatusCode = response.status_code
            if response is None:
                logger.warning("No output from service")
            output = json.loads(response.text)
            if output["errorCode"] is not None:
                raise RequestException(
                    output["errorCode"] + " - " + output["errorMessage"]
                )
            if httpStatusCode == 404:
                raise RequestException("404 Not Found")
            if httpStatusCode == 401:
                raise RequestException("401 Unauthorized")
            if httpStatusCode == 400:
                raise RequestException("Error Code " + httpStatusCode)
        # TODO: Catc blin exception
        except Exception as e:  # noqa
            raise RequestException(e)
        response.close()

        return output["data"]

    def login(self) -> None:
        """
        Login to USGS catalog
        """
        # Authentication
        try:
            username = os.environ["USGS_USERNAME"]
            password = os.environ["USGS_PASSWORD"]
        except KeyError:
            raise AuthenticationException(
                "Variables USGS_USERNAME and USGS_PASSWORD must be set"
            )

        # Initialize a new API instance and get an access key
        self.auth = self.sendRequest(
            "login-token", {"username": username, "token": password}
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
            f"Search on EarthExplorer catalog: min_date={min_date}, "
            f"max_date = {max_date}, "
            f"tile_id = {tile_id}, bbox = {latlon_bbox}, "
            f"max_cloud_cover = {max_cloud_cover}"
        )

        # Authentication
        try:
            username = os.environ["USGS_USERNAME"]
            password = os.environ["USGS_PASSWORD"]
        except KeyError:
            raise AuthenticationException(
                "Variables USGS_USERNAME and USGS_PASSWORD must be set"
            )

        bbox = None
        if tile_id is not None:
            # Extract ROI
            bbox = tuple(mgrs.get_bbox_mgrs_tile(tile_id))
        elif latlon_bbox is not None:
            bbox = tuple(latlon_bbox)
        else:
            logger.error("Tile ID or ROI must be provided")
            raise ValueError("Tile ID or ROI must be provided")

        # Build payload
        spatialFilter = {
            "filterType": "mbr",
            "lowerLeft": {"latitude": bbox[1], "longitude": bbox[0]},
            "upperRight": {"latitude": bbox[3], "longitude": bbox[2]},
        }
        temporalFilter = {"start": min_date, "end": max_date}

        payload = {
            "datasetName": self.dataset,
            "maxResults": 200,
            "startingNumber": 1,
            "sceneFilter": {
                "spatialFilter": spatialFilter,
                "acquisitionFilter": temporalFilter,
            },
            "username": username,
            "password": password,
        }

        # Search scenes
        results = self.sendRequest("scene-search", payload)

        # Aggregate a list of scene ids
        # Add this scene to the list I would like to download
        data = [
            {
                "entityId": result["entityId"],  # type: ignore
                "Product_name": result["displayId"],  # type: ignore
                "geometry": Polygon(
                    result["spatialCoverage"]["coordinates"][0]  # type: ignore
                ),
                "Cloud_cover": result["cloudCover"],  # type: ignore
                "Date": datetime.strptime(
                    result["temporalCoverage"]["startDate"],  # type: ignore
                    "%Y-%m-%d %H:%M:%S",
                ).date(),
            }
            for result in results["results"]  # type: ignore
        ]
        df = pd.DataFrame(data)
        payload = {
            "datasetName": self.dataset,
            "entityIds": list(df["entityId"].values),
        }
        downloadOptions = self.sendRequest("download-options", payload)

        # Aggregate a list of available products
        downloads = []
        # TODO: list comprehension
        for product in downloadOptions:
            # Make sure the product is available for this scene
            if product["available"]:  # type: ignore
                downloads.append(  # noqa
                    {
                        "entityId": product["entityId"],  # type: ignore
                        "productId": product["id"],  # type: ignore
                    }
                )
        logger.debug(
            "Number of products found on EarthData after cloud cover "
            f"filtering: {len(downloads)}"
        )

        # set a label for the download request
        label = "download-sample"
        payload = {"downloads": downloads, "label": label}
        # Call the download to get the direct download urls
        requestResults = self.sendRequest("download-request", payload)

        # Get download urls
        requestedDownloadsCount = len(requestResults["availableDownloads"])  # type: ignore
        urls = [
            {"entityId": download["entityId"], "URL": download["url"]}  # type: ignore
            for download in requestResults["availableDownloads"]  # type: ignore
        ]
        urls_df = pd.DataFrame(urls)

        # Convert to GeoDataFrame
        if requestedDownloadsCount == 0:
            return gpd.GeoDataFrame(
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
                    "geometry",
                ]
            ).set_crs(epsg=4326)
        # Merge dataframes
        df = pd.merge(df, urls_df, on="entityId").drop("entityId", axis=1)
        df["Provider"] = "USGS"
        df["Collection"] = "LANDSAT"
        df["Tile_ID"] = None
        df["Relative_orbit"] = None
        df["Checksum"] = None
        gdf = gpd.GeoDataFrame(df, geometry="geometry")
        gdf = gdf.set_crs(epsg=4326)  # or whatever CRS your data uses
        gdf = gdf[
            [
                "Product_name",
                "Date",
                "Provider",
                "Collection",
                "Tile_ID",
                "Cloud_cover",
                "Relative_orbit",
                "URL",
                "Checksum",
                "geometry",
            ]
        ]
        # Filter on cloud cover if the information exists
        gdf = gdf[gdf["Cloud_cover"] < max_cloud_cover]
        logger.debug(
            "Number of products found on EarthData after cloud "
            f"cover filtering: {len(gdf)}"
        )
        return gdf

    def download_archive(
        self, product: pd.Series, local_path: str = os.getcwd()
    ) -> None:
        """
        Download product from url

        Arguments
        =========

        product: ``str``
            product dataframe
        local_path: ``str`` ``default = os.getcwd()``
            path to write files
        """

        # Get file name and path
        if product.URL.count("gen-bundle?"):
            file_name = product.URL.split("=")[1].split("&")[0] + ".tar"
        else:
            file_name = product.URL.split("/")[-1]
        file_path = os.path.join(local_path, file_name)

        # Check if file exists
        if os.path.exists(file_path):
            logger.info(
                f"Product directory for {product.Product_name} exists already."
                "Skip product..."
            )
            return None

        # Prepare download
        resp = requests.get(product.URL, stream=True)
        total = int(resp.headers.get("content-length", 0))

        # Download file
        with (
            open(file_path, "wb") as file,
            tqdm(
                desc=file_name,
                total=total,
                unit="iB",
                unit_scale=True,
                unit_divisor=1024,
            ) as bar,
        ):
            for data in resp.iter_content(chunk_size=20480):
                size = file.write(data)
                bar.update(size)

        return None

    def download(
        self, products: pd.DataFrame, local_path: str = os.getcwd()
    ) -> None:
        """
        Download produtcs from catalog
        """
        logger.debug(
            f"List of products to download: {products['Product_name'].values}"
        )
        for _, product in products.iterrows():
            self.download_archive(product, local_path)


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

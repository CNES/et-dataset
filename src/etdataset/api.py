# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales
import datetime as dt
import os

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio as rio
import xarray as xr
from pyproj import CRS
from sensorsio import utils

from etdataset import era5, msg
from etdataset.era5 import create_daily_et_dataset
from etdataset.era5 import download_date_by_date as download_et
from etdataset.interpolation import create_grid_dataset
from etdataset.logging import LoggerManager
from etdataset.msg import create_daily_radiation_dataset
from etdataset.msg import download_date_by_date as download_radiation
from etdataset.provider import Collection, get_provider
from etdataset.reader import get_product_reader
from etdataset.selection import filter_with_roi, select_products
from etdataset.utils import (
    check_mgrs_format,
    get_bbox_from_mgrs_tile,
    get_utm_bbox_from_roi,
)
from etdataset.writer import write_daily_radiation, write_et_single_date

logger = LoggerManager.get_logger(__name__)

RESOLUTION = 60


class APIException(Exception):
    """
    Exception related to arguments
    """


def parse_date(date_str: str) -> dt.datetime:
    """
    Parse date expected format YYYY-MM-DD
    """
    try:
        date = dt.datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError as exc:
        raise APIException(
            "Error: The expected format for date must "
            f"be Year-Month-Day (got: {date_str})"
        ) from exc
    return date


def create_dataset(
    vis_path: str,
    tir_path: str | None = None,
    roi_bbox: rio.coords.BoundingBox | None = None,
    roi_crs: CRS | None = None,
    resolution: float = RESOLUTION,
    resampling: rio.enums.Resampling = rio.enums.Resampling.average,
) -> xr.Dataset:
    """
    Create a dataset
    """
    if tir_path is None:
        tir_path = vis_path
    # Get reader
    vis_reader = get_product_reader(
        vis_path, roi_bbox=roi_bbox, roi_crs=roi_crs, resolution=resolution
    )
    tir_reader = get_product_reader(
        tir_path, roi_bbox=roi_bbox, roi_crs=roi_crs, resolution=resolution
    )

    # Force the same bounding box
    common_bbox, common_crs = utils.bb_common(
        bounds=[vis_reader.bb, tir_reader.bb],
        src_crs=[str(vis_reader.crs), str(tir_reader.crs)],
        snap=RESOLUTION,
        target_crs=str(vis_reader.crs),
    )
    vis_reader.crs = CRS(common_crs)
    vis_reader.bb = common_bbox
    tir_reader.crs = CRS(common_crs)
    tir_reader.bb = common_bbox

    # Read VIS
    vis_xr = vis_reader.read_vis_bands(resampling=resampling)
    logger.debug(f"Read VIS: {type(vis_xr)}")

    # Read TIR
    tir_xr = tir_reader.read_tir_bands(resampling=resampling)
    logger.debug(f"Read TIR: {type(tir_xr)}")

    # Merge
    vis_bands = list(vis_xr.data_vars)
    tir_bands = list(tir_xr.data_vars)
    common_bands = list(set(vis_bands).intersection(tir_bands))
    selected_vis_bands = list(set(vis_bands).difference(tir_bands))
    selected_tir_bands = list(set(tir_bands).difference(vis_bands))
    merged_xr = xr.merge(
        (vis_xr[selected_vis_bands], tir_xr[selected_tir_bands]),
        combine_attrs="no_conflicts",
    )
    for band in common_bands:
        merged_xr = merged_xr.assign(
            {
                str(band): (
                    merged_xr.dims,
                    np.logical_or(vis_xr[band].data, tir_xr[band].data),
                )
            }
        )
    logger.debug(f"Merged: {merged_xr.attrs}")

    return merged_xr


def add_aux(
    data: xr.Dataset,
    path: str | None = None,
) -> xr.Dataset:
    """
    Description
    -----------
    Add auxiliary data to the dataset

    Parameters
    ----------
    data: xr.Dataset
        Data
    path: str
        Directory where auxiliary data have been downloaded data
    """
    # Check inputs
    if data.attrs.get("vis_date", None) is None:
        raise ValueError("Vis date attribute is missing in dataset")
    if data.attrs.get("vis_time", None) is None:
        raise ValueError("Vis time attribute is missing in dataset")
    if len(data.data_vars) == 0:
        raise ValueError("Dataset is empty")
    if data.attrs.get("crs", None) is not None:
        crs = data.attrs["crs"]
        data = data.rio.write_crs(crs)
    elif hasattr(data, "rio"):
        crs = data.rio.crs
    else:
        raise AttributeError("No CRS is defined")
    if path is not None:
        date = dt.datetime.combine(
            data.attrs["vis_date"], data.attrs["vis_time"]
        )
        bounds = rio.coords.BoundingBox(*data.rio.bounds())
        latlon_bounds = utils.bb_transform(
            source_crs=str(crs), target_crs="EPSG:4326", bounding_box=bounds
        )
        msg.download(date=date, latlon_bbox=latlon_bounds, path=path)
        era5.download(date=date, dataset=era5.ERA5Dataset.ERA5LAND, path=path)
        era5.download(date=date, dataset=era5.ERA5Dataset.ERA5, path=path)
    updated_data = msg.add(data=data, path=path)
    updated_data = era5.add(
        data=updated_data,
        dataset=era5.ERA5Dataset.ERA5LAND,
        variables=[era5.ERA5Var.TEMPERATURE, era5.ERA5Var.DEWPOINT_TEMPERATURE],
        path=path,
    )
    updated_data = era5.add(
        data=updated_data,
        dataset=era5.ERA5Dataset.ERA5,
        variables=[
            era5.ERA5Var.SURFACE_SOLAR_RADIATION_DOWNWARD_CLEAR_SKY,
            era5.ERA5Var.SURFACE_THERMAL_RADIATION_DOWNWARD_CLEAR_SKY,
        ],
        path=path,
    )
    return updated_data


def search(
    collection: Collection,
    min_date: str,
    max_date: str,
    tile_id: str | None = None,
    roi_bbox: rio.coords.BoundingBox | None = None,
    roi_crs: CRS | None = None,
    max_cloud_cover: float = 20,
    min_roi_overlap: float = 0,
) -> gpd.GeoDataFrame:
    """
    Search products in a collection
    """
    # Checks
    _ = parse_date(min_date)
    _ = parse_date(max_date)
    if tile_id is None and roi_bbox is None:
        raise APIException(
            "You must provide either a ROI bounding box or MGRS tile ID"
        )
    if roi_crs is None and roi_bbox is not None:
        raise APIException("You must provide a ROI bounding box with a CRS")
    if tile_id is not None:
        check_mgrs_format(tile_id)
    if max_cloud_cover < 0 or max_cloud_cover > 100:
        raise APIException(
            "Cloud cover criteria must be "
            f"between 0 and 100 (got : {max_cloud_cover}"
        )

    latlon_bbox = None
    if roi_bbox is not None and roi_crs is not None:
        # Convert to latlon
        latlon_bbox = utils.bb_transform(
            roi_crs.to_string(), CRS.from_epsg(4326).to_string(), roi_bbox
        )

    # Get provider
    provider = get_provider(collection)
    logger.debug(f"Provider: {provider}")

    # Search
    results = provider.search(
        min_date, max_date, tile_id, latlon_bbox, max_cloud_cover
    )
    logger.info(
        f"Products found for {collection.name} in catalog: {len(results)}"
    )

    # Get bounding box from tile
    if tile_id is not None:
        roi_bbox, roi_crs = get_bbox_from_mgrs_tile(tile_id)

    # Filter with additional criteria for collection 1
    results = filter_with_roi(
        results, roi_bbox, roi_crs, min_overlap=min_roi_overlap
    )
    logger.info(
        f"Products found for {collection.name} "
        f"after ROI filtering: {len(results)}"
    )

    return results


def download(
    products: pd.DataFrame,
    output_dir: str = os.path.join(os.getcwd(), "download"),
) -> None:
    """
    Download products from catalog
    The products to dowload are listed in a DataFrame.
    The required column are "Product_name" and "URL"
    which must contain the URLs to download a product.
    The URLs column can be a str or a list of str.
    """
    # Checks
    os.makedirs(output_dir, exist_ok=True)
    if "Product_name" not in products.columns:
        logger.exception("You must provide the product names.")
    if "URL" not in products.columns:
        logger.exception("You must provide the URL to download.")
    # Download
    for collection, group in products.groupby("Collection"):
        collection_dir = os.path.join(output_dir, str(collection))
        os.makedirs(collection_dir, exist_ok=True)
        logger.debug(f"Collection: {collection}")
        provider = get_provider(Collection[str(collection)])
        logger.debug(f"Provider: {provider}")
        urls = group[["Product_name", "URL"]].copy()
        if "Checksum" in group.columns:
            urls["Checksum"] = group["Checksum"].values
        else:
            urls["Checksum"] = np.nan
        provider.download(urls, collection_dir)


def select(
    collection1: Collection,
    collection2: Collection,
    min_date: str,
    max_date: str,
    delta: str = "3 day",
    tile_id: str | None = None,
    roi_bbox: rio.coords.BoundingBox | None = None,
    roi_crs: CRS | None = None,
    max_cloud_cover: float = 20,
    min_roi_overlap: float = 40,
    min_product_overlap: float = 40,
    only_best_match: bool = False,  # noqa
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Select products
    """
    # Checks
    min_datetime = parse_date(min_date)
    max_datetime = parse_date(max_date)
    if max_datetime < min_datetime:
        raise APIException(
            "Maximum acquisition date must be more recent than minimum date"
        )
    try:
        delta_time = pd.Timedelta(delta)
    except ValueError:
        raise APIException(
            "Error: The format for delta acquisition "
            "time is not recognized (ex: 1 day)"
        )
    if tile_id is None and roi_bbox is None:
        raise APIException(
            "You must provide either a ROI bounding box or MGRS tile ID"
        )
    if roi_crs is None and roi_bbox is not None:
        raise APIException("You must provide a ROI bounding box with a CRS")
    if tile_id is not None:
        check_mgrs_format(tile_id)
    if max_cloud_cover < 0 or max_cloud_cover > 100:
        raise APIException(
            "Cloud cover criteria must be "
            f"between 0 and 100 (got : {max_cloud_cover}"
        )
    if min_roi_overlap < 0 or min_roi_overlap > 100:
        raise APIException(
            "Cloud cover criteria must be "
            f"between 0 and 100 (got : {max_cloud_cover}"
        )
    if min_product_overlap < 0 or min_product_overlap > 100:
        raise APIException(
            "Cloud cover criteria must be "
            f"between 0 and 100 (got : {max_cloud_cover}"
        )

    # Search into collection 1
    selection1 = search(
        collection1,
        min_date,
        max_date,
        tile_id=tile_id,
        roi_bbox=roi_bbox,
        roi_crs=roi_crs,
        max_cloud_cover=max_cloud_cover,
        min_roi_overlap=min_roi_overlap,
    )

    # Search into collection 2
    selection2 = search(
        collection2,
        min_date,
        max_date,
        tile_id=tile_id,
        roi_bbox=roi_bbox,
        roi_crs=roi_crs,
        max_cloud_cover=max_cloud_cover,
        min_roi_overlap=min_roi_overlap,
    )

    if len(selection1) == 0 and len(selection2) == 0:
        logger.warning("No product in one of the collection")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Select matches
    matches = select_products(
        selection1, selection2, delta_time, min_product_overlap, only_best_match
    )
    matches = matches.rename(
        columns={
            "Product_name_1": f"Product_name_{collection1.name}",
            "Date_1": f"Date_{collection1.name}",
            "Product_name_2": f"Product_name_{collection2.name}",
            "Date_2": f"Date_{collection2.name}",
        }
    )
    if len(matches) > 0:
        selection1 = (
            selection1.merge(
                matches[[f"Product_name_{collection1.name}"]],
                left_on="Product_name",
                right_on=f"Product_name_{collection1.name}",
            )
            .drop(columns=[f"Product_name_{collection1.name}"])
            .drop_duplicates(subset=["Product_name"])
            .reset_index(drop=True)
        )
        selection2 = (
            selection2.merge(
                matches[[f"Product_name_{collection2.name}"]],
                left_on="Product_name",
                right_on=f"Product_name_{collection2.name}",
            )
            .drop(columns=[f"Product_name_{collection2.name}"])
            .drop_duplicates(subset=["Product_name"])
            .reset_index(drop=True)
        )

    else:
        selection1 = pd.DataFrame()
        selection2 = pd.DataFrame()

    # Return results
    return selection1, selection2, matches


def download_aux(
    products: pd.DataFrame,
    output_dir: str | None,
) -> None:
    """
    Download auxiliary product related to a list of product paths.
    """
    if output_dir is None:
        output_dir = os.getcwd()
    # Create output directory if necessary
    os.makedirs(output_dir, exist_ok=True)
    # Download
    for _, product in products.iterrows():
        date = product.Date
        bounds = rio.coords.BoundingBox(*product.geometry.bounds)
        era5.download(
            date=date,
            dataset=era5.ERA5Dataset.ERA5LAND,
            path=output_dir,
        )
        msg.download(
            date=date,
            latlon_bbox=bounds,
            path=output_dir,
        )


def prepare_daily_radiation(
    start_date: str, end_date: str, roi_path: str, output: str | None = None
) -> None:
    """
    Description
    -----------
    For each day from a start to a end date:
    - Download MSG data,
    - Read it a dataset projected on a given ROI
    with a resolution of 3km per pixel
    - Create a corresponding daily radiation .tif file.

    Parameters
    ----------
    start_date: str
        Start date (YYYY-MM-DD)
    end_date: str
        End date (YYYY-MM-DD)
    roi_path: str
        Path of the region of interest in Shapefile format
    path: str
        Directory path to store .tif files
        default: current directory)
    """
    # Check
    min_date = parse_date(start_date)
    max_date = parse_date(end_date)
    if output is None:
        output = os.getcwd()
    if max_date < min_date:
        raise APIException("End date must be more recent than start date")
    if not os.path.isfile(roi_path):
        raise FileNotFoundError(f"File not found {roi_path}")
    if not os.path.isdir(output):
        logger.debug(f"Create output path: {output}")
        os.makedirs(output, exist_ok=True)
    # Run
    roi_bbox, roi_crs = get_utm_bbox_from_roi(roi_path)
    grid = create_grid_dataset(roi_bbox, roi_crs, 3000)
    date_list = download_radiation(
        min_date, max_date, roi_bbox, roi_crs, output
    )
    for date in date_list:
        dst = create_daily_radiation_dataset(date.to_pydatetime(), grid, output)
        write_daily_radiation(dst, output)


def prepare_et_single_date(
    start_date: str, end_date: str, roi_path: str, output: str | None = None
) -> None:
    """
    Description
    -----------
    For each day from a start to a end date:
    - Download ERA5-Land,
    - Read it as a dataset:
        - projecting on a given ROI with a resolution of 3km per pixel,
        - keeping only the evapotranspiration variable,
        - adding a "flags" (0-1) variable.
    - Create a corresponding et single date .tif file.

    Parameters
    ----------
    start_date: str
        Start date (YYYY-MM-DD)
    end_date: str
        End date (YYYY-MM-DD)
    roi_path: str
        Path of the region of interest in Shapefile format
    output: str
        Directory path to store .tif files (default: current directory)
    """
    # Check
    min_date = parse_date(start_date)
    max_date = parse_date(end_date)
    if output is None:
        output = os.getcwd()
    if max_date < min_date:
        raise APIException("End date must be more recent than start date")
    if not os.path.isfile(roi_path):
        raise FileNotFoundError(f"File not found {roi_path}")
    if not os.path.isdir(output):
        logger.debug(f"Create output path: {output}")
        os.makedirs(output, exist_ok=True)
    # Run
    roi_bbox, roi_crs = get_utm_bbox_from_roi(roi_path)
    grid = create_grid_dataset(roi_bbox, roi_crs, 3000)
    date_list = download_et(min_date, max_date, ["total_evaporation"], output)
    for date in date_list:
        dst = create_daily_et_dataset(date.to_pydatetime(), grid, output)
        write_et_single_date(dst, output)

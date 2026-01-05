# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales
import datetime as dt
import os
from multiprocessing import Process

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio as rio
import xarray as xr
from pyproj import CRS
from sensorsio import utils

from etdataset import era5, msg
from etdataset.era5 import (
    ERA5Dataset,
    ERA5Var,
    add,
    create_daily_et_dataset,
    read,
)
from etdataset.era5 import download_date_by_date as download_et
from etdataset.icos import (
    ICOSStation,
    ICOSVar,
    StationConfig,
    add_time_attrs,
    create_xr_point_dataset,
    get_dem_from_roi,
    kelvin_to_celsius,
)
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
G_CST = 9.80665


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
    date_list = download_et(
        min_date, max_date, ERA5Dataset.ERA5LAND, ["total_evaporation"], output
    )
    for date in date_list:
        dst = create_daily_et_dataset(date.to_pydatetime(), grid, output)
        write_et_single_date(dst, output)


#########################################################
##                                                     ##
##                                                     ##
##                  VALIDATION (ERA5)                  ##
##                      lapse_rate                     ##
##                                                     ##
##                                                     ##
#########################################################


def _compute_lapse_rate(records: list) -> float:
    """
    Description
    -----------
    Computes the vertical temperature lapse rate by selecting the two
    atmospheric levels closest to the station's altitude.

    Parameters
    ----------
    records : list
        List of atmospheric records, where each element is a dictionary
        containing at least:
        - 'z'  : altitude of the level (m)
        - 'T'  : temperature at the level (K)
        - 'dz' : absolute difference between the level altitude and the
                station altitude (m)
    Returns
    -------
    lr: float
         Computed vertical temperature lapse rate (K/m).
    """
    df = pd.DataFrame(records).sort_values("z")

    # Sort by altitude difference (dz) to identify
    # the levels closest to the reference altitude
    df_sorted = df.sort_values("dz")

    # Select the level closest to the reference altitude
    df_ref = df_sorted.iloc[0]
    z_ref = df_ref["z"]  # reference altitude
    t_ref = df_ref["T"]  # temperature at reference altitude

    # Select the second closest level
    df_second = df_sorted.iloc[1]
    z_second = df_second["z"]  # second level altitude
    t_second = df_second["T"]

    # Compute the vertical temperature gradient (lapse rate)
    lr = (t_ref - t_second) / (z_ref - z_second)
    return lr


def compute_lapse_rate(
    station: ICOSStation,
    out: str,
    output_csv: str,
    start_date: dt.date,
    end_date: dt.date,
):
    """
    Description
    -----------
    Compute daily vertical lapse rates for air temperature and dew point
    temperature at a given ICOS station using ERA5 pressure-level data.
    e.g : "Elevation Correction of ERA5 Reanalysis Temperature over the
    Qilian Mountains of China", Peng Zhao and Lihui Qian.

    For each day in the specified period, temperature, relative humidity and
    geopotential are extracted at multiple pressure levels, interpolated to the
    station location. Geopotential are converted to altitude.

    The lapse rate is then computed using the two atmospheric levels closest
    to the station elevation.

    Parameters
    -----------
    station : ICOSStation
        ICOS station object containing latitude, longitude, elevation,
        and station name.
    out : str
        Path to the directory containing downloaded ERA5 pressure-level files.
    output_csv : str
        Path to the directory where the output CSV file will be written.
    start_date : dt.datetime
        Start date (inclusive).
    end_date : dt.datetime
        End date (inclusive).

    Returns
    --------
    Results are written to a CSV file containing daily lapse rates for
    air temperature and dew point temperature.
    """

    # Get the station's data
    lat_sta = station.latitude
    lon_sta = station.longitude
    z_sta = station.elevation

    # Loop over the requested time period
    cur_date = start_date
    lapse_records = []
    while cur_date <= end_date:
        filename = f"download_era5_pressure_{cur_date.isoformat()}.zip"
        logger.info(
            f"Current file : download_era5_pressure_{cur_date.isoformat()}.zip "
        )
        logger.info(f"elevation station : {z_sta}")
        filepath = os.path.join(out, "ERA5_data", filename)
        era5_xrds = read(product=filepath)
        records_ta = []
        records_tdp = []
        # Pressure levels used to estimate the vertical gradient
        pressure_levels = [800, 825, 850, 875, 900, 925, 950, 975, 1000]
        for p in pressure_levels:
            logger.info(f"pressure level :{p}")
            # Interpolate geopotential height at station location
            height = (
                era5_xrds["z"]
                .sel(
                    time=f"{cur_date.isoformat()}T12:00",
                    pressure_level=p,
                )
                .interp(
                    latitude=lat_sta,
                    longitude=lon_sta,
                    method="linear",
                )
                .item()
            )
            # Interpolate air temperature at station location
            air_temp = (
                era5_xrds["t"]
                .sel(time=f"{cur_date.isoformat()}T12:00", pressure_level=p)
                .interp(
                    latitude=lat_sta,
                    longitude=lon_sta,
                    method="linear",
                )
                .item()
            )
            # Interpolate relative humidity at station location
            rel_hum = (
                era5_xrds["r"]
                .sel(time=f"{cur_date.isoformat()}T12:00", pressure_level=p)
                .interp(latitude=lat_sta, longitude=lon_sta)
                .item()
            )
            # Compute dew point temperature
            dp_temp = ICOSVar.compute_dewpoint_temp(
                ta=pd.Series([air_temp]),
                rh=pd.Series([rel_hum]),
            )[0]

            z = height / G_CST  # Convert geopotential to geometric height
            records_ta.append({"z": z, "T": air_temp, "dz": abs(z - z_sta)})
            records_tdp.append({"z": z, "T": dp_temp, "dz": abs(z - z_sta)})

        # Compute lapse rates using the two levels closest to station height
        lapse_rate_ta = _compute_lapse_rate(records_ta)
        lapse_rate_tdp = _compute_lapse_rate(records_tdp)

        lapse_records.append(
            {
                "time": cur_date,
                "station": station.name,
                "lapse_rate_ta": lapse_rate_ta,
                "lapse_rate_tdp": lapse_rate_tdp,
            }
        )
        cur_date += dt.timedelta(days=1)

    os.makedirs(output_csv, exist_ok=True)
    out_path = os.path.join(output_csv, f"{station.name}_lapse_rate.csv")
    df = pd.DataFrame(lapse_records)
    df.to_csv(out_path)


def run_station_process_lr(
    station_id: str,
    cfg: dict[str, StationConfig],
    out: str,
    start_date: dt.date,
    end_date: dt.date,
    out_csv: str,
):
    if station_id not in cfg:
        logger.warning(
            f" Station {station_id} not found in stations_dict. skipping."
        )
    # Initialize station object
    station = ICOSStation(station_id, cfg)
    logger.info(f"\n Processing station: {station.name} ({station_id})")
    try:
        compute_lapse_rate(
            station=station,
            out=out,
            output_csv=out_csv,
            start_date=start_date,
            end_date=end_date,
        )
        logger.info(f"{station_id} finished")
    except Exception:
        logger.exception("Error")


def generate_lapse_rate_for_stations_multiprocess(
    station_ids,
    cfg,
    out,
    start_date,
    end_date,
    out_csv,
):
    """
    Description
    ----------
    Launch the generation of time series (ICOS, ERA5, downscaled ERA5)
    for stations in parallel using multiprocessing.

    Each station is processed in a separate subprocess, calling the function
    'run_station_process_lr'

    Parameters
    ----------
    station_ids: list
        List of station identifiers
    cfg: dict
        Configuration station
    output: str
        Path to the ERA5 directory
    start_date: dt.date
        Start date
    end_date: dt.date
        End date
    out_csv: str
        Directory where the station time series CSV files will be written.
    """
    procs = []
    # Spawn one process per station
    for sid in station_ids:
        p = Process(
            target=run_station_process_lr,
            args=(
                sid,
                cfg,
                out,
                start_date,
                end_date,
                out_csv,
            ),
        )
        procs.append(p)
        p.start()
    # Block until all station processes complete
    for p in procs:
        p.join()


#########################################################
##                                                     ##
##                                                     ##
##      VALIDATION (ERA5, ERA5 DOWNSCALED, ICOS)       ##
##                      Ta and Tdp                     ##
##                                                     ##
##                                                     ##
#########################################################


def process_era5_point(
    date: dt.datetime,
    station: ICOSStation,
    lat: float,
    lon: float,
    out,
    grid_res=0.25,
    resampled=False,
    dem_dir="rasters_COP30/DE-Geb",
    dem_res=60,
) -> xr.Dataset:
    """
    Description
    -----------
    Extract temperature (Ta) and dewpoint temperature (Td) from ERA5 or
    resampled ERA5 data at the location of an ICOS station for a given date.


    -> When 'resampled=True', the function builds a high-resolution DEM around
    the station, adds ERA5 variables at this resolution, and extracts the
    corresponding point values.

    -> When 'resampled=False', ERA5 data are extracted directly from a coarse
    grid defined around the station coordinates.

    Parameters
    -----------
    date: dt.datetime
        The date for which ERA5 data are extracted.
    station : ICOSStation
        ICOS station object containing metadata and coordinate utilities.
    lat: float
        Latitude of the point where ERA5 data will be extracted
        (used when 'resampled=False').
    lon: float
        Longitude of the point where ERA5 data will be extracted
        (used when 'resampled=False').
    output:
        Path where ERA5 data are.
    grid_res: float
        Resolution (in degrees) of the regular grid used when
        'resampled=False'. Default is 0.25°.
    resampled: bool
        'True': to use high-resolution resampled ERA5 data combined with a DEM.
        'False': ERA5 data are taken directly from a coarse grid.
    dem_dir: str
        Directory containing DEM tiles used for the high-resolution resampling.
    dem_res: int
        DEM resolution (in meters) used when 'resampled=True'.

    Returns
    -------
    xr.Dataset
        A dataset containing Ta and Td (in °C) at the point corresponding to
        the ICOS station and the given date.

    """
    # Get Ta and Td from the resampled ERA5 data
    if resampled:
        # Get the ROI bbox (UTM) around the ICOS station (10km x 10km)
        roi_bbox_utm, roi_crs_utm = station.work_area_from_coord_station(
            10000, 10000
        )["utm"]
        # Get the DEM from ROI
        dem = get_dem_from_roi(roi_bbox_utm, roi_crs_utm, dem_dir, dem_res)
        # Add attributes
        add_time_attrs(dem, date)
        updated = add(
            dem,
            dataset=ERA5Dataset.ERA5,
            variables=[
                ERA5Var.TEMPERATURE,
                ERA5Var.DEWPOINT_TEMPERATURE,
            ],
            path=out,
        )

        # get the coords (x,y) of the station in UTM
        x, y = (
            station.work_area_from_coord_station(0, 0)["utm"][0].left,
            station.work_area_from_coord_station(0, 0)["utm"][0].bottom,
        )
        z = (
            dem["height"]
            .interp({"x": x, "y": y}, method="linear")
            .values.item()
        )
        diff = station.elevation - z

        logger.info(f"Elevation DEM : {z:.2f} m")
        logger.info(
            f"Elevation station : {station.name} et {station.elevation:.2f} m"
        )
        logger.info(f"Difference : {diff:.2f} m")
        # assert (
        #    np.abs(diff) <= 20.0
        # ), f"""Difference of elevation between the DEM and the metadata of the
        # station is too important: {np.abs(diff):.2f} m > 20 m"""

        # get Ta and Td by bilinear interpolation and convert them from kelvin
        # to celsius in the resampled ERA5 data
        ta, td = (
            kelvin_to_celsius(
                updated["ta"].interp({"x": x, "y": y}, method="linear").values
            ),
            kelvin_to_celsius(
                updated["tdp"].interp({"x": x, "y": y}, method="linear").values
            ),
        )

        # print(f"ta={ta} et tdp={td}")
    # Get Ta and Td from the ERA5 data
    else:
        # Get the bounds (latitude/longitude) around the ICOS station
        # (50km x 50km)
        bounds = station.work_area_from_coord_station(50000, 50000)["lat/lon"]
        grid = create_grid_dataset(bounds[0], bounds[1], grid_res)
        grid["height"] = grid["grid"].copy()
        # Add attributes
        add_time_attrs(grid, date)
        updated = add(
            data=grid,
            dataset=ERA5Dataset.ERA5,
            variables=[
                ERA5Var.TEMPERATURE,
                ERA5Var.DEWPOINT_TEMPERATURE,
            ],
            path=out,
        )

        # get Ta and Td by bilinear interpolation and convert them from kelvin
        # to celsius in the ERA5 data
        ta, td = (
            kelvin_to_celsius(
                updated["ta"]
                .interp({"x": lon, "y": lat}, method="linear")
                .values
            ),
            kelvin_to_celsius(
                updated["tdp"]
                .interp({"x": lon, "y": lat}, method="linear")
                .values
            ),
        )
    return create_xr_point_dataset(date, ta, td)


def remove_dates_from_csv(csv_path: str, dates_to_remove):
    """
    Description
    -----------
    Removes the data (Ta and Tdp of ERA5 and ERA5 resampled) of the given
    dates in the csv of a station.

    Parameters
    ----------
    csv_path: str
        path to the csv of the station
    dates_to_remove : list[str]
        list of dates to removed in the csv
    """
    df = pd.read_csv(csv_path, parse_dates=["time"])

    dates_to_remove = pd.to_datetime(dates_to_remove)

    df = df[~df["time"].isin(dates_to_remove)]

    df.to_csv(csv_path, index=False)
    # print(f"Removed dates: {len(dates_to_remove)}")


def generate_timeseries_era5(
    station: ICOSStation,
    out: str,
    start_date: dt.date,
    end_date: dt.date,
    hour_start=0,
    hour_end=22,
    hour_step=2,
    day_step=1,
):
    """
    Description
    -----------
    Generate a time series of temperature (Ta) and dewpoint temperature (Td)
    from ERA5 and resampled ERA5 data for a given ICOS station over a specified
    date range. For each day and for each selected hour, the function calls
    'process_era5_point()' to extract point-level ERA5 data.

    Parameters
    -----------
    station : ICOSStation
        The ICOS station
    out : str
        Path where ERA5 input files are stored
    start_date : dt.date
        First date of the time series
    end_date : dt.date
        Last date of the time series
    hour_start : int
        First hour of the day to include in the time series
    hour_end : int
        Last hour of the day to include
    hour_step : int
        Time step between two extracted points
    day_step : int, optional
        Step between two processed days

    Returns
    -------
    - 'era5' : xr.Dataset
        Coarse grid ERA5 temperature and dewpoint temperature in °C for all
        selected timestamps.

    - 'era5_resampled' : xr.Dataset
        High-resolution DEM-based resampled ERA5 temperature and dewpoint
        temperature in °C for the same timestamps.


    """
    total_ds = []
    total_ds_proj = []

    cur_date = start_date
    while cur_date <= end_date:
        filename = f"download_era5_{cur_date.isoformat()}.zip"
        filepath = os.path.join(out, "ERA5_data", filename)

        read(product=filepath)
        times = [
            dt.datetime.combine(cur_date, dt.time(hour=h))
            for h in range(hour_start, hour_end + 1, hour_step)
        ]

        ds_list = []
        ds_proj_list = []
        for t in times:
            #  ERA5 data are extracted directly from a coarse
            # grid defined around the station coordinates.

            ds_list.append(
                process_era5_point(
                    t,
                    station,
                    station.latitude,
                    station.longitude,
                    out,
                    resampled=False,
                    dem_dir=f"rasters_COP30/{station.name}",
                )
            )
            # ERA5 data are extracted after resampling ERA5 variables at the
            # resolution DEM around the station
            ds_proj_list.append(
                process_era5_point(
                    t,
                    station,
                    station.latitude,
                    station.longitude,
                    out,
                    resampled=True,
                    dem_dir=f"rasters_COP30/{station.name}",
                )
            )
        day_ds = xr.concat(ds_list, dim="time")
        day_ds_proj = xr.concat(ds_proj_list, dim="time")
        total_ds.append(day_ds)
        total_ds_proj.append(day_ds_proj)

        cur_date += dt.timedelta(days=day_step)

    era5_ = xr.concat(total_ds, dim="time")
    era5_proj = xr.concat(total_ds_proj, dim="time")
    return era5_, era5_proj


def generate_timeseries_with_csv(
    station: ICOSStation,
    out: str,
    start_date: dt.date,
    end_date: dt.date,
    out_csv: str,
    hour_start=0,
    hour_end=22,
    hour_step=2,
    day_step=1,
    *,
    skip_existing: bool = True,
):
    """
    Description
    ----------
    Generate ICOS, ERA5, and projected ERA5 temperature/dewpoint time series and
    store them in a CSV file.

    This function builds a date-time grid, extracts:
      - ICOS in-situ air temperature and dew point,
      - ERA5 reanalysis variables (air temperature, dew point),
      - Projected ERA5 variables after elevation correction,
    and appends them to a CSV file for the station.

    Parameters
    ----------
    station : ICOSStation
        ICOS station
    out : str
        Path to the ERA5 output directory
    start_date : dt.date
        Start date (included).
    end_date : dt.date
        End date (included).
    out_csv : str
        Directory where the timeseries CSV will be written.
    hour_start : int
        First hour of each day
    hour_end : int
        Last hour of each day
    hour_step : int,
        Hour interval between processed timestamps
    day_step : int
        Interval between processed days
    skip_existing : bool
        If True (default), timestamps already present in the CSV are not
        recomputed.

    Returns
    -------
    str
        Full path to the generated CSV file.
    """
    os.makedirs(out_csv, exist_ok=True)
    csv_path = os.path.join(out_csv, f"{station.name}_timeseries.csv")
    # Load or initialize the CSV
    if skip_existing and os.path.exists(csv_path):
        df = pd.read_csv(csv_path, parse_dates=["time"])

    else:
        df = pd.DataFrame(
            columns=[
                "time",
                "Ta_icos",
                "Tdp_icos",
                "Ta_era5",
                "Tdp_era5",
                "Ta_era5_proj",
                "Tdp_era5_proj",
            ]
        )
        df.to_csv(csv_path, index=False)
    # Load ERA5 datasets and resampled ERA5
    era5_ds, era5_proj_ds = generate_timeseries_era5(
        station,
        out,
        start_date,
        end_date,
        hour_start,
        hour_end,
        hour_step,
        day_step,
    )
    # Filter ICOS dataset to the required time grid
    filtered_df = station.date_hour_filter(
        start_date, end_date, hour_start, hour_end, hour_step
    )

    for t in pd.date_range(start=start_date, end=end_date, freq=f"{day_step}D"):
        for h in range(hour_start, hour_end + 1, hour_step):
            dt_obj = dt.datetime.combine(t.date(), dt.time(hour=h))
            if skip_existing and (df["time"] == dt_obj).any():
                continue
            # ICOS extraction
            ta_icos, _, td_icos = station.get_ta_rh_td(
                filtered_df[filtered_df["TIMESTAMP_START"] == dt_obj]
            )
            ta_icos = float(ta_icos.values[0]) if not ta_icos.empty else np.nan
            td_icos = float(td_icos.values[0]) if not td_icos.empty else np.nan
            # ERA5 extraction
            try:
                ta_era5 = era5_ds.sel(time=dt_obj)["ta"].values.item()
                td_era5 = era5_ds.sel(time=dt_obj)["td"].values.item()
                ta_era5_proj = era5_proj_ds.sel(time=dt_obj)["ta"].values.item()
                td_era5_proj = era5_proj_ds.sel(time=dt_obj)["td"].values.item()
            except KeyError:
                ta_era5 = td_era5 = ta_era5_proj = td_era5_proj = np.nan

            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        {
                            "time": [dt_obj],
                            "Ta_icos": [ta_icos],
                            "Tdp_icos": [td_icos],
                            "Ta_era5": [ta_era5],
                            "Tdp_era5": [td_era5],
                            "Ta_era5_proj": [ta_era5_proj],
                            "Tdp_era5_proj": [td_era5_proj],
                        }
                    ),
                ],
                ignore_index=True,
            )
        df = df.dropna(subset=["Ta_icos", "Tdp_icos"])
        df.to_csv(csv_path, index=False)

    return csv_path


def run_station_process(
    station_id: str,
    cfg: dict[str, StationConfig],
    out: str,
    start_date: dt.date,
    end_date: dt.date,
    out_csv: str,
    skip_existing=True,
    hour_start=0,
    hour_end=22,
    hour_step=2,
    day_step=1,
):
    if station_id not in cfg:
        logger.warning(
            f" Station {station_id} not found in stations_dict. skipping."
        )
    # Initialize station object
    station = ICOSStation(station_id, cfg)
    logger.info(f"\n Processing station: {station.name} ({station_id})")
    try:
        generate_timeseries_with_csv(
            station=station,
            out=out,
            start_date=start_date,
            end_date=end_date,
            out_csv=out_csv,
            skip_existing=skip_existing,
            hour_start=hour_start,
            hour_end=hour_end,
            hour_step=hour_step,
            day_step=day_step,
        )
        logger.info(f"{station_id} finished")
    except Exception:
        logger.exception("Error")


def generate_timeseries_for_stations_multiprocess(
    station_ids,
    cfg,
    out,
    start_date,
    end_date,
    out_csv,
    skip_existing=True,
    hour_start=0,
    hour_end=22,
    hour_step=2,
    day_step=1,
):
    """
    Description
    ----------
    Launch the generation of time series (ICOS, ERA5, projected ERA5)
    for stations in parallel using multiprocessing.

    Each station is processed in a separate subprocess, calling the function
    'run_station_process'

    Parameters
    ----------
    station_ids : list
        List of station identifiers
    cfg : dict
        Configuration station
    out : str
        Path to the ERA5 directory
    start_date : dt.date
        Start date
    end_date : dt.date
        End date
    out_csv : str
        Directory where the station time series CSV files will be written.
    skip_existing : bool
        If True (default), timestamps already written in CSV are not
        recomputed.
    hour_start : int
        First hour
    hour_end : int
        Last hour
    hour_step : int
        Hour interval between processed timestamps
    day_step : int
        Interval between processed days
    """
    procs = []
    # Spawn one process per station
    for sid in station_ids:
        p = Process(
            target=run_station_process,
            args=(
                sid,
                cfg,
                out,
                start_date,
                end_date,
                out_csv,
                skip_existing,
                hour_start,
                hour_end,
                hour_step,
                day_step,
            ),
        )
        procs.append(p)
        p.start()
    # Block until all station processes complete
    for p in procs:
        p.join()


def generate_timeseries_for_stations(
    station_ids: list[str],
    cfg: dict[str, StationConfig],
    out: str,
    start_date: dt.date,
    end_date: dt.date,
    out_csv: str,
    *,
    skip_existing: bool = True,
    hour_start=0,
    hour_end=22,
    hour_step=2,
    day_step=1,
):
    """
    Description
    ----------
    Loop on the list of stations to generate for each ICOS, ERA5, and projected
    ERA5 temperature/dewpoint time series and CSV file.

    Parameters
    ----------
    station : list[str]
        ICOS stations list
    cfg : dict[str, StationConfig]
        dictionnary with metadata of the stations
    out : str
        Path to the ERA5 directory
    start_date : dt.date
        Start date
    end_date : dt.date
        End date
    out_csv : str
        Directory where the timeseries CSV will be written.
    hour_start : int
        First hour of each day
    hour_end : int, optional
        Last hour of each day
    hour_step : int,
        Hour interval between processed timestamps
    day_step : int
        Interval between processed days
    skip_existing : bool
        If True (default), timestamps already present in the CSV are not
        recomputed.

    """
    results = {}

    for sid in station_ids:
        if sid not in cfg:
            logger.warning(
                f" Station {sid} not found in stations_dict. skipping."
            )
            continue
        # Initialize station object
        station = ICOSStation(sid, cfg)
        logger.info(f"\n Processing station: {station.name} ({sid})")

        csv_out = generate_timeseries_with_csv(
            station=station,
            out=out,
            start_date=start_date,
            end_date=end_date,
            out_csv=out_csv,
            hour_start=hour_start,
            hour_end=hour_end,
            hour_step=hour_step,
            day_step=day_step,
            skip_existing=skip_existing,
        )
        results[sid] = csv_out

    return results

"""
Module for temperature rescaling
"""

import datetime as dt

import numpy as np
import rioxarray  # noqa # Use to activate rio attributes
import xarray as xr
from pyproj import CRS
from scipy.interpolate import interp1d

from etdataset.dem import get_dem_from_roi
from etdataset.era5 import (
    ERA5Dataset,
    get_era5_dem,
)
from etdataset.icos import (
    get_stations_config,
    kelvin_to_celsius,
)
from etdataset.logging import LoggerManager
from etdataset.utils import work_area_from_coord_point
from etdataset.validation_temp.temperature_rescaling import (
    add_dewpoint_to_ds,
    compute_dewpoint_lr,
    compute_dewpoint_temp_from_e,
    compute_lapse_rate_from_2_levels,
    compute_vapor_pressure,
    filter_dataset_by_hours,
    filter_dataset_by_location,
    filter_dataset_by_pressure_levels,
    generate_dates,
    get_lapse_rate_monthly,
    get_saturation_vapor_pressure,
    get_ta_td_celsius_at_location,
    get_vapor_pressure_monthly,
    normalize_longitude_latitude,
    prepare_temperature_inputs,
    read_era5_file,
    save_ta_td_csv,
    temperature_rescaling_constant_lapse_rate,
    temperature_rescaling_variable_lapse_rate,
)

G_CST = 9.80665
logger = LoggerManager.get_logger(__name__)


# METHOD 1


def run_stations_process_method_1(
    start_date: dt.date,
    end_date: dt.date,
    step_date: int,
    station: str,
    list_hours: list[dt.time],
    mnt_path: str,
    data_path: str,
    output: str,
):  # Get metadats of the station
    logger.info(f"Current station : {station}")
    cfg = get_stations_config(station)
    for d in generate_dates(start_date, end_date, step_date):
        logger.info(f"Current date : {d} for method_1")
        era5_xrds = read_era5_file(
            d,
            ERA5Dataset.ERA5,
            data_path,
        )
        if era5_xrds is None:
            logger.warning(
                "Skipping date %s because ERA5 file is unavailable", d
            )
            continue
        era5_xrds = normalize_longitude_latitude(era5_xrds)

        era5_filtered = filter_dataset_by_hours(era5_xrds, d, list_hours)

        # FOR ERA5 RESCALED ####################################################
        # get a roi around the station
        roi_bbox_utm, roi_crs_utm = work_area_from_coord_point(
            cfg.lat, cfg.lon, 10000, 10000, CRS.from_epsg(4326)
        )["utm"]
        # get dem from the roi
        dem = get_dem_from_roi(
            roi_bbox=roi_bbox_utm,
            roi_crs=roi_crs_utm,
            base_dir=mnt_path,
            resolution=60,
        )
        new_dem, era5_dem, updated_data = prepare_temperature_inputs(
            data=dem,
            era5_data=normalize_longitude_latitude(
                era5_filtered, target="0_360"
            ),
            dataset=ERA5Dataset.ERA5,
        )
        lr_monthly = get_lapse_rate_monthly(d)
        coeff = get_vapor_pressure_monthly(d)
        dp_lr_monthly = compute_dewpoint_lr(coeff)
        updated = temperature_rescaling_constant_lapse_rate(
            updated_data,
            new_dem,
            era5_filtered,
            era5_dem,
            lr_monthly,
            dp_lr_monthly,
        )
        ta, td = get_ta_td_celsius_at_location(updated, cfg)
        ds_era5_grid = xr.Dataset(
            {
                "ta": ta,
                "tdp": td,
            }
        )
        save_ta_td_csv(ds_era5_grid, cfg, output, name_dir="csv_era5_rescaled")
    return ds_era5_grid


# METHOD 2


def run_stations_process_method_2(
    start_date: dt.date,
    end_date: dt.date,
    step_date: int,
    station: str,
    list_hours: list[dt.time],
    mnt_path: str,
    data_path: str,
    output: str,
):  # Get metadats of the station
    logger.info(f"Current station : {station}")
    cfg = get_stations_config(station)
    for d in generate_dates(start_date, end_date, step_date):
        logger.info(f"Current date : {d} for method 2")
        # Get ERA5 pressure file
        era5_xrds = read_era5_file(
            d,
            ERA5Dataset.ERA5PRESSURE,
            data_path,
        )
        if era5_xrds is None:
            logger.warning("Skipping date %s (ERA5PRESSURE unavailable)", d)
            continue
        era5_xrds = normalize_longitude_latitude(era5_xrds)

        era5_filtered = filter_dataset_by_hours(era5_xrds, d, list_hours)

        ####### COMPUTE CURRENT LAPSE RATE #####################################
        # Filter on pressure levels
        era5_filt_pressure = filter_dataset_by_pressure_levels(
            era5_filtered, ["700", "925"]
        )
        # Filter on location
        era5_filt_location = filter_dataset_by_location(
            era5_filt_pressure, cfg.lat, cfg.lon
        )
        # Compute dewpoint temperature
        era5_add_dewpoint = add_dewpoint_to_ds(era5_filt_location)
        # Get lapse rates
        lr_t = compute_lapse_rate_from_2_levels(era5_add_dewpoint)
        lr_td = compute_lapse_rate_from_2_levels(era5_add_dewpoint, "td")

        ############ RESCALING #################################################
        # get a roi around the station
        roi_bbox_utm, roi_crs_utm = work_area_from_coord_point(
            cfg.lat, cfg.lon, 10000, 10000, CRS.from_epsg(4326)
        )["utm"]
        # get dem from the roi
        dem = get_dem_from_roi(
            roi_bbox=roi_bbox_utm,
            roi_crs=roi_crs_utm,
            base_dir=mnt_path,
            resolution=60,
        )
        # Comparison
        logger.info(f"ICOS's elevation: {cfg.elev}")
        if cfg.elev > 1500:
            # Get Tera_850
            era5_data = filter_dataset_by_pressure_levels(era5_filtered, "850")
            era5_data = add_dewpoint_to_ds(era5_data)
            dataset = ERA5Dataset.ERA5PRESSURE
        elif cfg.elev < 1500:
            # Get Tera_2m
            era5_xrds = read_era5_file(
                d,
                ERA5Dataset.ERA5,
                data_path,
            )
            if era5_xrds is None:
                logger.warning("Skipping date %s (ERA5 unavailable)", d)
                continue
            era5_xrds = normalize_longitude_latitude(era5_xrds)
            era5_filtered = filter_dataset_by_hours(era5_xrds, d, list_hours)
            era5_data = era5_filtered
            dataset = ERA5Dataset.ERA5

        # prepare inputs for rescaling
        new_dem, era5_dem, updated_data = prepare_temperature_inputs(
            data=dem,
            era5_data=normalize_longitude_latitude(era5_data, target="0_360"),
            dataset=dataset,
        )
        # rescaling
        updated = temperature_rescaling_variable_lapse_rate(
            updated_data=updated_data,
            dataset=dataset,
            dem=new_dem,
            era5_data=era5_data,
            era5_dem=era5_dem,
            lr_ta=lr_t,
            lr_tdp=lr_td,
        )
        # Get temperature and dew point temperature
        ta, td = get_ta_td_celsius_at_location(updated, cfg)
        ds_era5_grid = xr.Dataset(
            {
                "ta": ta,
                "tdp": td,
            }
        )
        save_ta_td_csv(ds_era5_grid, cfg, output, name_dir="csv_era5_rescaled")
    return ds_era5_grid


# METHOD 3


def run_stations_process_method_3(
    start_date: dt.date,
    end_date: dt.date,
    step_date: int,
    station: str,
    list_hours: list[dt.time],
    mnt_path: str,
    data_path: str,
    output: str,
):  # Get metadats of the station
    logger.info(f"Current station : {station}")
    cfg = get_stations_config(station)
    for d in generate_dates(start_date, end_date, step_date):
        logger.info(f"Current date : {d} for method 3")
        # Get ERA5 pressure file
        era5_xrds = read_era5_file(
            d,
            ERA5Dataset.ERA5PRESSURE,
            data_path,
        )
        if era5_xrds is None:
            logger.warning("Skipping date %s (ERA5PRESSURE unavailable)", d)
            continue
        era5_xrds = normalize_longitude_latitude(era5_xrds)

        era5_filtered = filter_dataset_by_hours(era5_xrds, d, list_hours)

        # Filter on location
        era5_filt_location = filter_dataset_by_location(
            era5_filtered, cfg.lat, cfg.lon
        )

        ############ RESCALING #################################################
        # get a roi around the station
        roi_bbox_utm, roi_crs_utm = work_area_from_coord_point(
            cfg.lat, cfg.lon, 10000, 10000, CRS.from_epsg(4326)
        )["utm"]
        # get dem from the roi
        dem = get_dem_from_roi(
            roi_bbox=roi_bbox_utm,
            roi_crs=roi_crs_utm,
            base_dir=mnt_path,
            resolution=60,
        )
        new_dem, era5_dem, updated_data = prepare_temperature_inputs(
            data=dem,
            era5_data=era5_filtered,
            dataset=ERA5Dataset.ERA5,
        )
        # Comparison
        logger.info(f"ICOS's elevation: {cfg.elev}")
        if cfg.elev > 1500:
            # Get Lapse rate 700-850
            ####### Filter on pressure levels
            era5_filt_pressure = filter_dataset_by_pressure_levels(
                era5_filt_location, ["700", "850"]
            )
            ####### Compute dewpoint temperature
            era5_add_dewpoint = add_dewpoint_to_ds(era5_filt_pressure)
            ####### Get lapse rates
            lr_t = compute_lapse_rate_from_2_levels(era5_add_dewpoint)
            lr_td = compute_lapse_rate_from_2_levels(era5_add_dewpoint, "td")

            # Get Tera_850
            era5_data = filter_dataset_by_pressure_levels(era5_filtered, "850")
            era5_data = add_dewpoint_to_ds(era5_data)
            dataset = ERA5Dataset.ERA5PRESSURE
        elif cfg.elev < 1500:
            # Get Lapse rate 850-925
            ####### Filter on pressure levels
            era5_filt_pressure = filter_dataset_by_pressure_levels(
                era5_filt_location, ["925", "850"]
            )
            ####### Compute dewpoint temperature
            era5_add_dewpoint = add_dewpoint_to_ds(era5_filt_pressure)
            ####### Get lapse rates
            lr_t = compute_lapse_rate_from_2_levels(era5_add_dewpoint)
            lr_td = compute_lapse_rate_from_2_levels(era5_add_dewpoint, "td")

            # Get Tera_2m
            era5_xrds = read_era5_file(
                d,
                ERA5Dataset.ERA5,
                data_path,
            )
            if era5_xrds is None:
                logger.warning("Skipping date %s (ERA5 unavailable)", d)
                continue
            era5_xrds = normalize_longitude_latitude(era5_xrds)
            era5_filtered = filter_dataset_by_hours(era5_xrds, d, list_hours)
            era5_data = era5_filtered
            dataset = ERA5Dataset.ERA5
        # prepare inputs for rescaling
        new_dem, era5_dem, updated_data = prepare_temperature_inputs(
            data=dem,
            era5_data=normalize_longitude_latitude(era5_data, target="0_360"),
            dataset=dataset,
        )
        # rescaling
        updated = temperature_rescaling_variable_lapse_rate(
            updated_data=updated_data,
            dataset=dataset,
            dem=new_dem,
            era5_data=era5_data,
            era5_dem=era5_dem,
            lr_ta=lr_t,
            lr_tdp=lr_td,
        )

        ta, td = get_ta_td_celsius_at_location(updated, cfg)
        ds_era5_grid = xr.Dataset(
            {
                "ta": ta,
                "tdp": td,
            }
        )
        save_ta_td_csv(ds_era5_grid, cfg, output, name_dir="csv_era5_rescaled")
    return ds_era5_grid


# METHOD 4


def run_stations_process_method_4(
    start_date: dt.date,
    end_date: dt.date,
    step_date: int,
    station: str,
    list_hours: list[dt.time],
    mnt_path: str,
    data_path: str,
    output: str,
):  # Get metadats of the station
    logger.info(f"Current station : {station}")
    cfg = get_stations_config(station)
    ds = []
    for d in generate_dates(start_date, end_date, step_date):
        logger.info(f"Date: {d}")
        logger.info(f"Station elevation: {cfg.elev}")

        # READ ERA5 PRESSURE
        era5_pressure = read_era5_file(d, ERA5Dataset.ERA5PRESSURE, data_path)
        if era5_pressure is None:
            logger.warning("Skipping date %s (ERA5PRESSURE unavailable)", d)
            continue
        era5_pressure = normalize_longitude_latitude(era5_pressure)

        era5_pressure = filter_dataset_by_hours(era5_pressure, d, list_hours)
        era5_pressure = filter_dataset_by_location(
            era5_pressure, cfg.lat, cfg.lon
        )
        era5_pressure = filter_dataset_by_pressure_levels(
            era5_pressure,
            [
                "700",
                "725",
                "750",
                "775",
                "800",
                "825",
                "850",
                "875",
                "900",
                "925",
                "950",
                "975",
                "1000",
            ],
        )

        # READ ERA5 SURFACE
        era5_data = read_era5_file(d, ERA5Dataset.ERA5, data_path)
        if era5_data is None:
            logger.warning("Skipping date %s (ERA5PRESSURE unavailable)", d)
            continue
        era5_data = normalize_longitude_latitude(era5_data)

        # GET ELEVATION AT THE LOCATION FROM DEM
        roi_bbox_utm, roi_crs_utm = work_area_from_coord_point(
            cfg.lat, cfg.lon, 10000, 10000, CRS.from_epsg(4326)
        )["utm"]
        # get dem from the roi
        dem = get_dem_from_roi(
            roi_bbox=roi_bbox_utm,
            roi_crs=roi_crs_utm,
            base_dir=mnt_path,
            resolution=60,
        )
        x, y = (
            work_area_from_coord_point(
                cfg.lat, cfg.lon, 0, 0, CRS.from_epsg(4326)
            )["utm"][0].left,
            work_area_from_coord_point(
                cfg.lat, cfg.lon, 0, 0, CRS.from_epsg(4326)
            )["utm"][0].bottom,
        )
        z_station = dem["height"].sel(x=x, y=y, method="nearest")
        # GET ERA5 ELEVATION
        z_surface = get_era5_dem()
        z_surface = xr.DataArray(
            z_surface.data,
            dims=("latitude", "longitude"),
            coords={
                "latitude": era5_data.latitude,
                "longitude": era5_data.longitude,
            },
        ).rio.write_crs(CRS(4326))
        z_surface_ds = z_surface.to_dataset(name="elevation")
        z_surface_ds = normalize_longitude_latitude(z_surface_ds)
        z_surface = z_surface_ds["elevation"]
        z_surface = z_surface.sel(
            latitude=cfg.lat, longitude=cfg.lon, method="nearest"
        )

        ta_out = []
        td_out = []

        # LOOP OVER TIME
        for t in era5_pressure.time.values:
            hourly_pressure = era5_pressure.sel(time=t)
            # RELATIVE HEIGHTS
            z_levels_rel = hourly_pressure["z"].values / G_CST

            # VARIABLES
            T_levels = hourly_pressure["t"].values

            # recalcul pression de vapeur pour CE pas de temps
            T_levels_c = kelvin_to_celsius(hourly_pressure["t"].values)
            es_levels = get_saturation_vapor_pressure(T_levels_c)
            e_levels = compute_vapor_pressure(
                hourly_pressure["r"].values,
                es_levels,
            )

            z_clean = z_levels_rel
            T_clean = T_levels
            Td_clean = compute_dewpoint_temp_from_e(e_levels)
            if len(z_clean) < 2:
                ta_out.append(np.nan)
                td_out.append(np.nan)
                continue

            z_add = z_clean
            T_add = T_clean
            Td_add = Td_clean
            # SORT VERTICALLY
            sort_idx = np.argsort(z_add)
            z_full = z_add[sort_idx]
            T_full = T_add[sort_idx]
            Td_full = Td_add[sort_idx]

            # INTERPOLATION
            f_t = interp1d(
                z_full,
                T_full,
                kind="linear",
                fill_value="extrapolate",
            )
            f_td = interp1d(
                z_full,
                Td_full,
                kind="linear",
                fill_value="extrapolate",
            )

            T_station = f_t(z_station)
            Td_station = f_td(z_station)

            ta_out.append(T_station)
            td_out.append(Td_station)

        # BUILD OUTPUT DATASET
        ta_da = xr.DataArray(
            ta_out,
            coords={"time": era5_pressure.time},
            dims=["time"],
        )

        td_da = xr.DataArray(
            td_out,
            coords={"time": era5_pressure.time},
            dims=["time"],
        )

        ds_out = xr.Dataset(
            {
                "ta": kelvin_to_celsius(ta_da),
                "tdp": td_da,
            }
        )

        # SAVE
        save_ta_td_csv(
            ds_out,
            cfg,
            output,
            name_dir="csv_era5_rescaled",
        )
        ds.append(ds_out)
    return ds_out


# METHOD 5
def run_stations_process_method_5(
    start_date: dt.date,
    end_date: dt.date,
    step_date: int,
    station: str,
    list_hours: list[dt.time],
    mnt_path: str,
    data_path: str,
    output: str,
):
    logger.info(f"Current station : {station}")
    cfg = get_stations_config(station)
    ds = []
    for d in generate_dates(start_date, end_date, step_date):
        logger.info(f"Date: {d}")
        logger.info(f"Station elevation: {cfg.elev}")

        # READ ERA5 PRESSURE
        era5_pressure = read_era5_file(d, ERA5Dataset.ERA5PRESSURE, data_path)
        if era5_pressure is None:
            logger.warning("Skipping date %s (ERA5PRESSURE unavailable)", d)
            continue

        era5_pressure = normalize_longitude_latitude(era5_pressure)

        era5_pressure = filter_dataset_by_hours(era5_pressure, d, list_hours)
        era5_pressure = filter_dataset_by_location(
            era5_pressure, cfg.lat, cfg.lon
        )
        era5_pressure = filter_dataset_by_pressure_levels(
            era5_pressure,
            [
                "700",
                "725",
                "750",
                "775",
                "800",
                "825",
                "850",
                "875",
                "900",
                "925",
                "950",
                "975",
            ],
        )

        # READ ERA5 SURFACE
        era5_data = read_era5_file(d, ERA5Dataset.ERA5, data_path)
        if era5_data is None:
            logger.warning("Skipping date %s (ERA5PRESSURE unavailable)", d)
            continue
        era5_data = normalize_longitude_latitude(era5_data)

        era5_surface = filter_dataset_by_hours(era5_data, d, list_hours)
        era5_surface = filter_dataset_by_location(
            era5_surface, cfg.lat, cfg.lon
        )

        # GET ELEVATION AT THE LOCATION FROM DEM
        roi_bbox_utm, roi_crs_utm = work_area_from_coord_point(
            cfg.lat, cfg.lon, 10000, 10000, CRS.from_epsg(4326)
        )["utm"]
        # get dem from the roi
        dem = get_dem_from_roi(
            roi_bbox=roi_bbox_utm,
            roi_crs=roi_crs_utm,
            base_dir=mnt_path,
            resolution=60,
        )
        x, y = (
            work_area_from_coord_point(
                cfg.lat, cfg.lon, 0, 0, CRS.from_epsg(4326)
            )["utm"][0].left,
            work_area_from_coord_point(
                cfg.lat, cfg.lon, 0, 0, CRS.from_epsg(4326)
            )["utm"][0].bottom,
        )
        z_station = dem["height"].sel(x=x, y=y, method="nearest")
        # GET ERA5 ELEVATION
        z_surface = get_era5_dem()
        z_surface = xr.DataArray(
            z_surface.data,
            dims=("latitude", "longitude"),
            coords={
                "latitude": era5_data.latitude,
                "longitude": era5_data.longitude,
            },
        ).rio.write_crs(CRS(4326))
        z_surface_ds = z_surface.to_dataset(name="elevation")
        z_surface_ds = normalize_longitude_latitude(z_surface_ds)
        z_surface = z_surface_ds["elevation"]
        z_surface = z_surface.sel(
            latitude=cfg.lat, longitude=cfg.lon, method="nearest"
        )

        # DIFFERENCE
        delta_z = z_station - z_surface
        logger.info(
            f"Altitude difference station - ERA5 surface: {delta_z.values}"
        )

        ta_out = []
        td_out = []

        # LOOP OVER TIME
        for t in era5_pressure.time.values:
            hourly_pressure = era5_pressure.sel(time=t)
            hourly_surface = era5_surface.sel(time=t)

            # SURFACE HEIGHT
            z_surface_t = z_surface

            # RELATIVE HEIGHTS
            z_station_rel = z_station
            z_levels_rel = hourly_pressure["z"].values / G_CST

            z_2m_rel = 2.0 + z_surface_t

            # VARIABLES
            T_levels = hourly_pressure["t"].values

            # recalcul pression de vapeur pour CE pas de temps
            T_levels_c = kelvin_to_celsius(hourly_pressure["t"].values)
            es_levels = get_saturation_vapor_pressure(T_levels_c)
            e_levels = compute_vapor_pressure(
                hourly_pressure["r"].values,
                es_levels,
            )

            T_2m = hourly_surface["t2m"].values

            Td_2m = kelvin_to_celsius(hourly_surface["d2m"].values)

            z_clean = z_levels_rel
            T_clean = T_levels
            Td_clean = compute_dewpoint_temp_from_e(e_levels)
            if len(z_clean) < 2:
                ta_out.append(np.nan)
                td_out.append(np.nan)
                continue

            # ADD 2m LEVEL
            z_add = np.insert(z_clean, 0, z_2m_rel)
            T_add = np.insert(T_clean, 0, T_2m)
            Td_add = np.insert(Td_clean, 0, Td_2m)
            logger.info(f"z_add:{z_add}")
            # SORT VERTICALLY
            sort_idx = np.argsort(z_add)
            z_full = z_add[sort_idx]
            T_full = T_add[sort_idx]
            Td_full = Td_add[sort_idx]
            logger.info(f"Td_full = {Td_full}")

            f_t = interp1d(
                z_full,
                T_full,
                kind="linear",
                fill_value="extrapolate",
            )
            f_td = interp1d(
                z_full,
                Td_full,
                kind="linear",
                fill_value="extrapolate",
            )
            T_station = f_t(z_station_rel)
            Td_station = f_td(z_station_rel)

            ta_out.append(T_station)
            td_out.append(Td_station)

        # BUILD OUTPUT DATASET
        ta_da = xr.DataArray(
            ta_out,
            coords={"time": era5_pressure.time},
            dims=["time"],
        )

        td_da = xr.DataArray(
            td_out,
            coords={"time": era5_pressure.time},
            dims=["time"],
        )

        ds_out = xr.Dataset(
            {
                "ta": kelvin_to_celsius(ta_da),
                "tdp": td_da,
            }
        )

        # SAVE
        save_ta_td_csv(
            ds_out,
            cfg,
            output,
            name_dir="csv_era5_rescaled",
        )
        ds.append(ds_out)
    return ds_out


# METHOD 6


def run_stations_process_method_6(
    start_date: dt.date,
    end_date: dt.date,
    step_date: int,
    station: str,
    list_hours: list[dt.time],
    mnt_path: str,
    data_path: str,
    output: str,
):
    logger.info(f"Current station : {station}")

    cfg = get_stations_config(station)
    # gdf_s = get_station_location(station)
    # create_geopckg_from_gdf(gdf_s)

    for d in generate_dates(start_date, end_date, step_date):
        logger.info(f"Date: {d}")
        logger.info(f"Station elevation: {cfg.elev}")

        # READ ERA5 PRESSURE
        era5_pressure = read_era5_file(d, ERA5Dataset.ERA5PRESSURE, data_path)
        if era5_pressure is None:
            logger.warning("Skipping date %s (ERA5PRESSURE unavailable)", d)
            continue
        era5_pressure = normalize_longitude_latitude(era5_pressure)
        # FILTER ERA5 PRESSURE BY HOURS
        era5_pressure = filter_dataset_by_hours(era5_pressure, d, list_hours)
        era5_pressure = filter_dataset_by_pressure_levels(
            era5_pressure,
            [
                "700",
                "725",
                "750",
                "775",
                "800",
                "825",
                "850",
                "875",
                "900",
                "925",
                "950",
                "975",
            ],
        )
        # t = era5_pressure["t"]
        # t = t.squeeze()
        # # print(t)
        # output_file = "t_after.nc"
        # t.to_netcdf(output_file)
        # t.rio.write_crs("EPSG:4326", inplace=True)
        # t.rio.to_raster("t_after.tif")
        # FILTER ERA5 PRESSURE ON LOCATION
        era5_filt_location = filter_dataset_by_location(
            era5_pressure, cfg.lat, cfg.lon
        )

        # READ ERA5 SURFACE
        era5_data = read_era5_file(d, ERA5Dataset.ERA5, data_path)
        if era5_data is None:
            logger.warning("Skipping date %s (ERA5 unavailable)", d)
            continue
        logger.info(f"CRS DE ERA5 :{era5_data.rio.crs}")

        era5_data = normalize_longitude_latitude(era5_data)
        # logger.info(f"CRS DE ERA5 APRÈS :{era5_data.rio.crs}")

        # FILTER ERA5 BY HOURS
        era5_surface = filter_dataset_by_hours(era5_data, d, list_hours)
        # t2m = era5_surface["t2m"]
        # print(t2m)
        # output_file = "t2m_after.nc"
        # t2m.to_netcdf(output_file)
        # t2m.rio.write_crs("EPSG:4326", inplace=True)
        # t2m.rio.to_raster("t2m_after.tif")

        era5_surface_loc = filter_dataset_by_location(
            era5_surface, cfg.lat, cfg.lon
        )

        # t2m_filter = era5_surface_loc["t2m"]
        # t2m_filter = t2m_filter.squeeze()
        # lat = float(t2m_filter.latitude.values)
        # lon = float(t2m_filter.longitude.values)
        # value = float(t2m_filter.values)
        # time = str(t2m_filter.time.values)
        # geometry = Point(lon, lat)

        # gdf = gpd.GeoDataFrame(
        #     {"t2m_after": [value], "time": [time]},
        #     geometry=[geometry],
        #     crs="EPSG:4326",
        # )
        # gdf.to_file("t2m_point_after.gpkg", layer="t2m", driver="GPKG")

        logger.info(
            f"ERA5 SURFACE AU POINT STATION T = "
            f"{era5_surface_loc['t2m'].values}"
        )
        # GET DEM
        roi_bbox_utm, roi_crs_utm = work_area_from_coord_point(
            cfg.lat, cfg.lon, 10000, 10000, CRS.from_epsg(4326)
        )["utm"]
        # roi_geom = box(*roi_bbox_utm)
        # roi_gdf = gpd.GeoDataFrame(
        #     {"id": [1]},
        #     geometry=[roi_geom],
        #     crs=roi_crs_utm,
        # )
        # output_path = "roi.gpkg"

        # roi_gdf.to_file(output_path, layer="ROI DEM", driver="GPKG")

        # get dem from the roi
        dem = get_dem_from_roi(
            roi_bbox=roi_bbox_utm,
            roi_crs=roi_crs_utm,
            base_dir=mnt_path,
            resolution=60,
        )
        logger.info(f"CRS DE DEM :{dem.rio.crs}")

        # dem = create_grid_dataset(roi_bbox_utm, roi_crs_utm, 5000)
        # dem["height"] = xr.ones_like(dem["grid"]) * 165

        # roi_bbox_utm, roi_crs_utm = work_area_from_coord_point(
        #     cfg.lat, cfg.lon, 30000, 30000, CRS.from_epsg(4326)
        # )["utm"]
        # new_era5 = create_grid_dataset(roi_bbox_utm, roi_crs_utm, 10000)
        # new_era5["height"] = xr.ones_like(new_era5["grid"]) * 123
        # new_era5["t2m"] = xr.ones_like(new_era5["grid"]) * 300
        # GRILLE
        # GET HEIGHT OF THE STATION
        x, y = (
            work_area_from_coord_point(
                cfg.lat, cfg.lon, 0, 0, CRS.from_epsg(4326)
            )["utm"][0].left,
            work_area_from_coord_point(
                cfg.lat, cfg.lon, 0, 0, CRS.from_epsg(4326)
            )["utm"][0].bottom,
        )
        z_station_dem = dem["height"].sel(x=x, y=y, method="nearest").values
        logger.info(f"ALTITUDE CIBLE = {z_station_dem}")
        # z_station_dem = 165
        ta_out = []
        td_out = []

        # LOOP OVER TIME
        for t in era5_pressure.time.values:
            hourly_p = era5_pressure.sel(time=t)
            hourly_p_filt_location = era5_filt_location.sel(time=t)
            hourly_surface = era5_surface.sel(time=[t])

            z_levels = hourly_p_filt_location["z"].values / G_CST

            ind = np.searchsorted(z_levels, z_station_dem)
            if ind == 0:
                ind_bot, ind_top = ind, ind + 1
                # si station inférieure au premier niveau, on utilise Tref=Tera5
                # et lapse_rate entre 925 Pa et 850 Pa
                era5_data = hourly_surface
                dataset = ERA5Dataset.ERA5
            else:
                ind_bot, ind_top = ind - 1, ind
                p_levels = hourly_p_filt_location["pressure_level"].values
                p_bot = str(p_levels[ind_bot])
                # dataset au niveau bottom uniquement
                era5_data = filter_dataset_by_pressure_levels(
                    hourly_p,
                    [p_bot],
                ).expand_dims(time=[t])
                era5_data = add_dewpoint_to_ds(era5_data)

                dataset = ERA5Dataset.ERA5PRESSURE

            # LAPSE RATE
            # récupérer les niveaux de pression correspondants
            p_levels = hourly_p_filt_location["pressure_level"].values
            p_bot = str(p_levels[ind_bot])
            p_top = str(p_levels[ind_top])

            # filtrer ERA5 sur ces deux niveaux
            era5_filt_pressure = filter_dataset_by_pressure_levels(
                hourly_p_filt_location,
                [p_bot, p_top],
            )
            # ajouter Td
            era5_add_dewpoint = add_dewpoint_to_ds(era5_filt_pressure)
            ####### Get lapse rates
            lr_t = compute_lapse_rate_from_2_levels(era5_add_dewpoint)
            lr_td = compute_lapse_rate_from_2_levels(era5_add_dewpoint, "td")
            # logger.info(f"new_era5 = {new_era5}")
            # prepare inputs for rescaling
            new_dem, era5_dem, updated_data = prepare_temperature_inputs(
                data=dem,
                era5_data=normalize_longitude_latitude(
                    era5_data, target="0_360"
                ),
                dataset=dataset,
            )
            # dem_value = era5_dem.sel(
            #     latitude=cfg.lat,
            #     longitude=cfg.lon,
            #     method="nearest",
            # ).values
            # logger.info(f"ERA5 DEM : {dem_value}")
            # output_file = "era5_dem_after.nc"
            # era5_dem.to_netcdf(output_file)
            # era5_dem.rio.write_crs("EPSG:4326", inplace=True)
            # era5_dem.rio.to_raster("era5_dem_after.tif")

            logger.info(f"LAPSE RATE = {lr_t}")

            # rescaling
            updated = temperature_rescaling_variable_lapse_rate(
                updated_data=updated_data,
                dataset=dataset,
                dem=new_dem,
                era5_data=era5_data,  # new_era5,
                era5_dem=era5_dem,  # new_era5["height"],
                lr_ta=lr_t,
                lr_tdp=lr_td,
            )
            logger.info(f"updated = {updated['ta'].values}")
            T_station, Td_station = get_ta_td_celsius_at_location(updated, cfg)

            ta_out.append(float(T_station.item()))
            td_out.append(float(Td_station.item()))

        ta_da = xr.DataArray(
            ta_out,
            coords={"time": era5_pressure.time},
            dims=["time"],
        )

        td_da = xr.DataArray(
            td_out,
            coords={"time": era5_pressure.time},
            dims=["time"],
        )

        ds_out = xr.Dataset(
            {
                "ta": ta_da,
                "tdp": td_da,
            }
        )

        save_ta_td_csv(
            ds_out,
            cfg,
            output,
            name_dir="csv_era5_rescaled",
        )
    return ds_out

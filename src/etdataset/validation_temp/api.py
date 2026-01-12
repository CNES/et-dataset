# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales
import datetime as dt
import os
from multiprocessing import Process

import numpy as np
import pandas as pd
import xarray as xr

from etdataset.era5 import (
    ERA5Dataset,
    ERA5Var,
    add,
    read,
)
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

logger = LoggerManager.get_logger(__name__)
G_CST = 9.80665

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
                "station": station.id,
                "lapse_rate_ta": lapse_rate_ta,
                "lapse_rate_tdp": lapse_rate_tdp,
            }
        )
        cur_date += dt.timedelta(days=1)

    os.makedirs(output_csv, exist_ok=True)
    out_path = os.path.join(output_csv, f"{station.id}_lapse_rate.csv")
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
    Launch the generation of time series of lapse rate
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
        # WITH THE CLUSTER
        # mnt_path = "/work/datalake/static_aux/MNT/COP-DEM_GLO-30-\
        # DGED_S2_tiles/"

        # Get the ROI bbox (UTM) around the ICOS station (10km x 10km)
        roi_bbox_utm, roi_crs_utm = station.work_area_from_coord_station(
            10000, 10000
        )["utm"]
        # Get the DEM from ROI
        dem = get_dem_from_roi(roi_bbox_utm, roi_crs_utm, dem_dir, dem_res)

        # WITH TH CLUSTER
        # Get the DEM from ROI
        # dem = get_dem_from_roi(
        #    roi_bbox=roi_bbox_utm,
        #    roi_crs=roi_crs_utm,
        #    base_dir=mnt_path,
        #    resolution=dem_res,
        # )
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
            f"Elevation station : {station.id} et {station.elevation:.2f} m"
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
                    dem_dir=f"rasters_COP30/{station.id}",
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
                    dem_dir=f"rasters_COP30/{station.id}",
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
    csv_path = os.path.join(out_csv, f"{station.id}_timeseries.csv")
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


def generate_timeseries_for_stations_multiprocess(  # Genrate timeseries with
    # multiprocessing
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


def generate_timeseries_for_stations(  # Generate timeseries without
    # multiprocessing
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
    ERA5 temperature/dewpoint time series and CSV file. (without
    multiprocessing)

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
        logger.info(f"\n Processing station: {station.id} ({sid})")

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

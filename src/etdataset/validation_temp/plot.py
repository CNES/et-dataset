# type: ignore
# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales

"""
Functions for plotting
"""
# Skip this file with mypy

import os
from pathlib import Path

import geopandas as gpd
import pandas as pd
import plotly.graph_objects as go
import xarray as xr

from etdataset.logging import LoggerManager

logger = LoggerManager.get_logger(__name__)


#########################################
##                                     ##
##                                     ##
##        Prepare Inputs csv           ##
##                                     ##
##                                     ##
#########################################


def list_csv_files(folder: str) -> list[str]:
    """
    List all CSV files in a given folder

    Parameters
    ----------
    folder : str
        Path to the directory

    Returns
    -------
    list[str]
        List of full paths to CSV files found in the folder
    """
    return [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(".csv")
    ]


#########################################
##                                     ##
##                                     ##
##           Filter datas              ##
##                                     ##
##                                     ##
#########################################


def filter_icos_on_era5_hours(
    df_icos: pd.DataFrame,
    df_era5: pd.DataFrame,
) -> pd.DataFrame:
    """
    Filter ICOS data to keep only:
    - observations at full hours (minute == 0)
    - timestamps that are also present in ERA5 data

    Parameters
    ----------
    df_icos : pd.DataFrame
        ICOS observations containing a 'time' column
    df_era5 : pd.DataFrame
        ERA5 data containing a 'time' column

    Returns
    -------
    pd.DataFrame
        Filtered ICOS DataFrame with ERA5 hourly timestamps.
    """
    df_i = df_icos.copy()
    df_i["time"] = pd.to_datetime(df_i["time"])

    df_e = df_era5.copy()
    df_e["time"] = pd.to_datetime(df_e["time"])

    era5_hours = set(df_e["time"].unique())

    mask = (df_i["time"].dt.minute == 0) & (df_i["time"].isin(era5_hours))
    return df_i[mask]


def filter_files_by_station(
    files: list[str],
    stations: list[str] | str = "all",
) -> list[str]:
    """
    Filter a list of file paths by station names.

    Parameters
    ----------
    files : list[str]
        List of file paths.
    stations : list[str] | str, default "all"
        Station names to keep.
        - "all": keep all files
        - list of station names: keep only files starting with these names

    Returns
    -------
    list[str]
        Filtered list of file paths.
    """
    if stations == "all":
        return files

    if isinstance(stations, str):
        stations = [stations]

    stations = [s.lower() for s in stations]

    selected = []
    for f in files:
        filename = os.path.basename(f).lower()
        if any(filename.startswith(st + "_") for st in stations):
            selected.append(f)

    return selected


#########################################
##                                     ##
##                                     ##
##                 warnings            ##
##                                     ##
##                                     ##
#########################################


def warn_missing_stations(
    stations: list[str] | str,
    era5_files: list[str],
    era5r_files: list[str],
    icos_files: list[str],
):
    """
    Log warnings for stations missing in one or more data sources

    Parameters
    ----------
    stations : list[str] | str
        Stations to check, or "all" to disable warnings
    era5_files : list[str]
        List of ERA5 CSV files.
    era5r_files : list[str]
        List of ERA5 rescaled CSV files
    icos_files : list[str]
        List of ICOS CSV files

    Returns
    -------
    None
    """
    if stations == "all":
        return

    if isinstance(stations, str):
        stations = [stations]

    def extract(files):
        return {os.path.basename(f).split("_")[0].lower() for f in files}

    s_era5 = extract(era5_files)
    s_era5r = extract(era5r_files)
    s_icos = extract(icos_files)

    for s in stations:
        s_l = s.lower()
        missing = []
        if s_l not in s_era5:
            missing.append("ERA5")
        if s_l not in s_era5r:
            missing.append("ERA5_rescaled")
        if s_l not in s_icos:
            missing.append("ICOS")

        if missing:
            logger.warning(
                f" Warning: station '{s}'missing in {', '.join(missing)}"
            )


#########################################
##                                     ##
##                                     ##
##           Load Data                 ##
##                                     ##
##                                     ##
#########################################
def load_csv_data_icos(csv_dir: Path):
    """
    Load ICOS CSV files from a directory into a dictionary.

    Parameters
    ----------
    csv_dir : Path
        Directory containing ICOS CSV files

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary mapping station name to DataFrame
    """
    stations = {}

    for csv_file in csv_dir.glob("*_data.csv"):
        station_name = csv_file.stem.split("_")[0].split("-")[1]
        df = pd.read_csv(csv_file, parse_dates=["time"])
        stations[station_name] = df

    return stations


def load_csv_files(files: list[str]) -> dict[str, pd.DataFrame]:
    """
    Load multiple CSV files into a dictionary of DataFrame

    Parameters
    ----------
    files : list[str]
        List of CSV file paths

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary mapping filename to DataFrame
    """
    data = {}
    for f in files:
        key = os.path.splitext(os.path.basename(f))[0]
        data[key] = pd.read_csv(f)
    return data


def load_all_data(
    era5_folder: str,
    era5_rescaled_folder: str,
    icos_folder: str,
    stations: list[str] | str = "all",
) -> dict[str, dict[str, pd.DataFrame]]:
    """
    Load all ERA5, ERA5 rescaled and ICOS data, optionally filtered by station

    Parameters
    ----------
    era5_folder : str
        Path to ERA5 CSV files
    era5_rescaled_folder : str
        Path to ERA5 rescaled CSV files
    icos_folder : str
        Path to ICOS CSV files
    stations : list[str] | str, default "all"
        Stations to load

    Returns
    -------
    dict[str, dict[str, pd.DataFrame]]

    """
    era5_files = list_csv_files(era5_folder)
    era5r_files = list_csv_files(era5_rescaled_folder)
    icos_files = list_csv_files(icos_folder)

    warn_missing_stations(
        stations,
        era5_files,
        era5r_files,
        icos_files,
    )

    era5_files = filter_files_by_station(era5_files, stations)
    era5r_files = filter_files_by_station(era5r_files, stations)
    icos_files = filter_files_by_station(icos_files, stations)

    return {
        "era5": load_csv_files(era5_files),
        "era5_rescaled": load_csv_files(era5r_files),
        "icos": load_csv_files(icos_files),
    }


#########################################
##                                     ##
##                                     ##
##          Build timeseries           ##
##                                     ##
##                                     ##
#########################################


def build_station_timeseries(
    data: dict,
    start: str | None = None,
    end: str | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Build merged time series per station from ERA5, ERA5 rescaled and ICOS data

    Parameters
    ----------
    data : dict
        Dictionary returned by 'load_all_data'
    start : str | None, optional
        Start date for filtering
    end : str | None, optional
        End date for filtering

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary of station name to merged DataFrame
    """
    start_date = pd.to_datetime(start) if start else None
    end_date = pd.to_datetime(end) if end else None

    stations = {}

    all_keys = (
        list(data["icos"].keys())
        + list(data["era5"].keys())
        + list(data["era5_rescaled"].keys())
    )
    station_names = {k.split("_")[0] for k in all_keys}

    for station in station_names:
        #  ERA5  #################################################
        df_era5 = None
        for k, df in data["era5"].items():
            if k.startswith(station):
                df_era5 = df.copy()
                df_era5["time"] = pd.to_datetime(df_era5["time"])
                df_era5 = df_era5.add_suffix("_era5")
                df_era5 = df_era5.rename(columns={"time_era5": "time"})
                break

        if df_era5 is None or df_era5.empty:
            continue

        # Filtered ICOS ############################################
        df_icos = None
        for k, df in data["icos"].items():
            if k.startswith(station):
                df_icos = filter_icos_on_era5_hours(df, df_era5)
                df_icos = df_icos.add_suffix("_icos")
                df_icos = df_icos.rename(columns={"time_icos": "time"})
                break

        if df_icos is None or df_icos.empty:
            continue

        dfs = [df_era5, df_icos]

        #  ERA5 rescaled ##############################################
        for k, df in data["era5_rescaled"].items():
            if k.startswith(station):
                df_r = df.copy()
                df_r["time"] = pd.to_datetime(df_r["time"])
                df_r = df_r.add_suffix("_era5_rescaled")
                df_r = df_r.rename(columns={"time_era5_rescaled": "time"})
                dfs.append(df_r)
                break

        # merge ######################################################
        df_station = dfs[0]
        for df_next in dfs[1:]:
            df_station = pd.merge(
                df_station,
                df_next,
                on="time",
                how="inner",
            )

        if start_date is not None:
            df_station = df_station[df_station["time"] >= start_date]
        if end_date is not None:
            df_station = df_station[df_station["time"] <= end_date]

        if not df_station.empty:
            stations[station] = df_station.sort_values("time")

    return stations


#########################################
##                                     ##
##                                     ##
##           Plots                     ##
##                                     ##
##                                     ##
#########################################


def plot_variable(
    stations: dict[str, pd.DataFrame],
    var: str,
    start: str | None = None,
    end: str | None = None,
):
    """
    Plot a variable from ICOS, ERA5 and ERA5 rescaled for all stations

    Parameters
    ----------
    stations : dict[str, pd.DataFrame]
        Station time series data from 'build_station_timeseries'
    var : str
        Variable name to plot
    start : str | None, optional
        Start date
    end : str | None, optional
        End date

    Returns
    -------
    """
    start_date = pd.to_datetime(start) if start else None
    end_date = pd.to_datetime(end) if end else None

    fig = go.Figure()

    for station_name, df in stations.items():
        if df.empty:
            continue

        df_t = df.copy()

        if start_date is not None:
            df_t = df_t[df_t["time"] >= start_date]
        if end_date is not None:
            df_t = df_t[df_t["time"] <= end_date]

        if df_t.empty:
            continue

        fig.add_trace(
            go.Scatter(
                x=df_t["time"],
                y=df_t[f"{var}_icos"],
                name=" ICOS",
                legendgroup=station_name,
                legendgrouptitle_text=f"Station {station_name}",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=df_t["time"],
                y=df_t[f"{var}_era5"],
                name="ERA5",
                legendgroup=station_name,
                line={"dash": "dash"},
            )
        )

        fig.add_trace(
            go.Scatter(
                x=df_t["time"],
                y=df_t[f"{var}_era5_rescaled"],
                name="ERA5 rescaled",
                legendgroup=station_name,
            )
        )
    titles = {
        "ta": "Air temperature (Ta) of ICOS station",
        "td": "Dew point temperature ICOS stations",
    }
    fig.update_layout(
        title=titles.get(var, f"{var} ICOS stations"),
        xaxis_title="Time",
        yaxis_title=f"{var} (°C)",
        hovermode="x unified",
        template="plotly",
    )

    return fig


def plot_station_map_with_gpkg(csv_path: str = "stations_package.gpkg"):
    """
    Plot station locations on an interactive map using a GeoPackage file.

    Parameters
    ----------
    csv_path : str, default "stations_package.gpkg"
        Path to the GeoPackage file

    Returns
    -------
    """
    stations_gdf = gpd.read_file(csv_path, layer="stations")
    m = stations_gdf.explore(
        column="elevation",
        tooltip="name",
        popup=True,
        legend=True,
        marker_kwds={"radius": 8},
        style_kwds={"color": "black"},
        figsize=(8, 8),
    )
    return m


def plot_station_map(gdf: gpd.GeoDataFrame):
    """
    Plot station locations on an interactive map from a GeoDataFrame.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame containing station geometries and attributes.

    Returns
    -------

    """
    m = gdf.explore(
        column="elev",
        tooltip="name",
        popup=True,
        legend=True,
        marker_kwds={"radius": 8},
        style_kwds={"color": "black"},
        figsize=(8, 8),
    )
    return m


def plot_ta_tdp_icos(
    stations: dict,
    var: str,
    start: str | None = None,
    end: str | None = None,
):
    """
    Plot ICOS air temperature or dew point temperature for all stations.

    Parameters
    ----------
    stations : dict[str, pd.DataFrame]
        ICOS station data
    var : str
        Variable to plot ('ta' or 'tdp')
    start : str | None, optional
        Start date
    end : str | None, optional
        End date

    Returns
    -------
    """
    start_date = pd.to_datetime(start) if start else None
    end_date = pd.to_datetime(end) if end else None

    fig = go.Figure()

    for station_name, df in stations.items():
        if df.empty:
            continue

        df_v = df.sort_values("time")
        df_t = df_v.copy()

        if start_date is not None:
            df_t = df_t[df_t["time"] >= start_date]

        if end_date is not None:
            df_t = df_t[df_t["time"] <= end_date]

        if df_t.empty:
            continue

        fig.add_trace(
            go.Scatter(
                x=df_t["time"],
                y=df_t[f"{var}"],
                name=f"{station_name} ICOS",
                legendgroup=station_name,
                legendgrouptitle_text=f"Station {station_name}",
                line={"dash": "solid"},
                mode="lines+markers",
            )
        )
    if var == "ta":
        title = "Air temperature of ICOS stations"
    elif var == "tdp":
        title = "Dew point temperature of ICOS stations"
    else:
        title = f"{var} ICOS stations"

    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title=f"{var} (°C)",
        hovermode="x unified",
        template="plotly",
        showlegend=True,
    )

    return fig


def plot_temp_icos(
    csv_dir,
    start: str | None = None,
    end: str | None = None,
):
    """
    Plot ICOS air temperature and dew point temperature time series.

    Parameters
    ----------
    csv_dir : str or Path
        Directory containing ICOS CSV files
    start : str | None, optional
        Start date.
    end : str | None, optional
        End date

    Returns
    -------
    """
    stations = load_csv_data_icos(Path(csv_dir))

    fig_ta = plot_ta_tdp_icos(stations, "ta", start, end)
    fig_tdp = plot_ta_tdp_icos(stations, "tdp", start, end)

    fig_ta.show()
    fig_tdp.show()


def plot_temp_vs_height_with_lvpr(ds: xr.Dataset):
    """
    Plot vertical temperature profiles as a function of altitude from ERA5
    pressure-level data.

    Each curve represents a time step. Altitude is derived from geopotential
    (z / g), and pressure levels are associated with each point

    Parameters
    ----------
    ds : xr.Dataset

    Returns
    -------
    None
        Display an interactive Plotly figure.
    """

    fig = go.Figure()

    pressure = ds.pressure_level.values

    for t_idx, t_val in enumerate(ds.time.values):
        fig.add_trace(
            go.Scatter(
                x=ds.t.isel(time=t_idx).values,
                y=ds.height.isel(time=t_idx).values,
                mode="lines+markers",
                name=str(t_val),
                marker={"size": 7},
                line={"width": 2},
                customdata=pressure,
                hovertemplate=(
                    "T = %{x:.2f} K<br>"
                    "Height = %{y:.0f} m<br>"
                    "Pressure = %{customdata:.0f} hPa"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title="ERA5 temperature-elevation vertical profiles",
        xaxis_title="Temperature (K)",
        yaxis_title="Height (m)",
        template="plotly_white",
        hovermode="closest",
        legend_title="Time",
    )

    fig.show()


#######################################################################
def get_valid_method_folders(methods: dict[str, str | None]) -> dict[str, str]:
    """
    Keep only methods with a valid folder path.
    """
    return {
        name: path
        for name, path in methods.items()
        if path is not None and os.path.isdir(path)
    }


def load_all_data_multi_method(
    era5_folder: str,
    icos_folder: str,
    methods: dict[str, str | None],
    stations: list[str] | str = "all",
):
    """
    Load ERA5, ICOS and multiple rescaling methods.
    Ignores methods with folder=None.
    """
    # Filtrer méthodes valides
    valid_methods = {k: v for k, v in methods.items() if v is not None}

    # ERA5 + ICOS
    era5_files = filter_files_by_station(list_csv_files(era5_folder), stations)
    icos_files = filter_files_by_station(list_csv_files(icos_folder), stations)

    era5_data = load_csv_files(era5_files)
    icos_data = load_csv_files(icos_files)

    # Méthodes
    methods_data = {}
    for method_name, folder in valid_methods.items():
        method_files = filter_files_by_station(list_csv_files(folder), stations)
        methods_data[method_name] = load_csv_files(method_files)

    return {
        "era5": era5_data,
        "icos": icos_data,
        "methods": methods_data,
    }


def build_station_timeseries_multi(
    data: dict,
    start: str | None = None,
    end: str | None = None,
    min_months: int = 6,
):
    """
    Build merged timeseries for all stations present in ERA5 + ICOS.
    Adds all available methods. Excludes stations with too few months of data.
    """
    start_date = pd.to_datetime(start) if start else None
    end_date = pd.to_datetime(end) if end else None

    stations_out = {}

    # liste toutes les stations possibles
    all_keys = list(data["era5"].keys()) + list(data["icos"].keys())
    for method_data in data.get("methods", {}).values():
        all_keys += list(method_data.keys())
    station_names = {k.split("_")[0] for k in all_keys}
    logger.info(f"station_names:{station_names}")

    for station in station_names:
        logger.info(f"Station:{station}")
        # --- ERA5 ---
        df_era5 = None
        for k, df in data["era5"].items():
            if k.startswith(station):
                df_era5 = df.copy()
                df_era5["time"] = pd.to_datetime(df_era5["time"])
                df_era5 = df_era5.add_suffix("_era5")
                df_era5 = df_era5.rename(columns={"time_era5": "time"})
                logger.info(f"df_era5 : {df_era5}")
                break
        if df_era5 is None or df_era5.empty:
            continue

        # --- ICOS filtré sur ERA5 ---
        df_icos = None
        for k, df in data["icos"].items():
            if k.startswith(station):
                df_icos = filter_icos_on_era5_hours(df, df_era5)
                df_icos = df_icos.add_suffix("_icos")
                df_icos = df_icos.rename(columns={"time_icos": "time"})
                logger.info(f"df_era5 : {df_icos}")
                break
        if df_icos is None or df_icos.empty:
            continue

        dfs = [df_era5, df_icos]

        # --- Toutes les méthodes disponibles ---
        for method_name, method_data in data.get("methods", {}).items():
            df_m = None
            for k, df in method_data.items():
                if k.startswith(station):
                    df_m = df.copy()
                    df_m["time"] = pd.to_datetime(df_m["time"])
                    df_m = df_m.add_suffix(f"_{method_name}")
                    df_m = df_m.rename(columns={f"time_{method_name}": "time"})
                    break
            if df_m is not None and not df_m.empty:
                dfs.append(df_m)

        # --- Merge ---
        df_station = dfs[0]
        logger.info(f"df_station : {df_station}")
        for df_next in dfs[1:]:
            df_station = pd.merge(df_station, df_next, on="time", how="left")

        # --- Filtrer période ---
        if start_date is not None:
            df_station = df_station[df_station["time"] >= start_date]
        if end_date is not None:
            df_station = df_station[df_station["time"] <= end_date]

        # --- Filtrer nombre de mois minimum ---
        n_months = df_station["time"].dt.to_period("M").nunique()
        if n_months >= min_months:
            stations_out[station] = df_station.sort_values("time")

    return stations_out


def plot_all_stations_timeseries(
    stations_ts: dict[str, pd.DataFrame],
    var: str,
    hour: str | list[str] = "all",
    start: str | None = None,
    end: str | None = None,
):
    """
    Create one Plotly figure per station including ICOS, ERA5
    and all available rescaling methods.

    Returns
    -------
    dict[str, go.Figure]
        Dictionary {station_name: figure}
    """

    figures = {}

    for station, df_stations in stations_ts.items():
        df = df_stations.copy()
        df["time"] = pd.to_datetime(df["time"])

        # --- Filter hours ---
        if hour != "all":
            if isinstance(hour, str):
                hour = [hour]
            df = df[df["time"].dt.strftime("%H").isin(hour)]

        # --- Filter period ---
        if start:
            df = df[df["time"] >= pd.to_datetime(start)]
        if end:
            df = df[df["time"] <= pd.to_datetime(end)]

        if df.empty:
            continue

        fig = go.Figure()

        # ICOS
        if f"{var}_icos" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df["time"],
                    y=df[f"{var}_icos"],
                    name="ICOS",
                    line={"width": 2},
                )
            )

        # ERA5
        if f"{var}_era5" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df["time"],
                    y=df[f"{var}_era5"],
                    name="ERA5",
                    line={"dash": "dash"},
                )
            )

        # All methods dynamically
        method_cols = [
            col
            for col in df.columns
            if col.startswith(f"{var}_")
            and col not in [f"{var}_icos", f"{var}_era5"]
        ]

        for col in method_cols:
            method_name = col.replace(f"{var}_", "")
            fig.add_trace(
                go.Scatter(
                    x=df["time"],
                    y=df[col],
                    name=method_name,
                    line={"dash": "dot"},
                )
            )

        fig.update_layout(
            title=f"{station} — {var.upper()}",
            xaxis_title="Time",
            yaxis_title=var.upper(),
            template="plotly_white",
            hovermode="x unified",
        )

        figures[station] = fig

    return figures

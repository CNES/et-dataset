# type: ignore
# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales

"""
Functions for plotting
"""
# Skip this file with mypy

import math
import os
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import xarray as xr
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

from etdataset.icos import get_stations_config
from etdataset.logging import LoggerManager
from etdataset.validation_temp.metrics import altitude_class

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
    has_elev = "elev" in gdf.columns and gdf["elev"].notna().any()
    m = gdf.explore(
        column="elev" if has_elev else None,
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

        df_station = dfs[0]
        logger.info(f"df_station : {df_station}")

        for df_next in dfs[1:]:
            df_station = pd.merge(
                df_station,
                df_next,
                on="time",
                how="left",
            )

        if start_date is not None:
            df_station = df_station[df_station["time"] >= start_date]

        if end_date is not None:
            df_station = df_station[df_station["time"] <= end_date]

        n_months = df_station["time"].dt.to_period("M").nunique()

        valid_idx = (
            df_station["ta_icos"].notna() & df_station["tdp_icos"].notna()
        )

        valid_months = (
            df_station.loc[valid_idx, "time"].dt.to_period("M").nunique()
        )

        logger.info(
            f"{station} -> total months={n_months}, valid months={valid_months}"
        )

        if valid_months >= min_months:
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

        if hour != "all":
            if isinstance(hour, str):
                hour = [hour]
            df = df[df["time"].dt.strftime("%H").isin(hour)]

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


def plot_rmse(df):
    df = df.copy()

    df["month"] = pd.to_datetime(df["month"])
    df = df.sort_values("month")

    rmse_cols = [c for c in df.columns if c.startswith("ta_rmse_")]

    df_long = df.melt(
        id_vars=["category", "month"],
        value_vars=rmse_cols,
        var_name="metric",
        value_name="rmse",
    )

    df_long["method"] = df_long["metric"].str.replace("ta_rmse_", "")

    for cat in df_long["category"].unique():
        df_cat = df_long[df_long["category"] == cat]

        fig = px.line(
            df_cat,
            x="month",
            y="rmse",
            color="method",
            markers=True,
            title=f"RMSE par méthode - {cat} altitude",
        )

        fig.update_layout(
            xaxis_title="Temps (mois)",
            yaxis_title="RMSE (°C)",
        )

        fig.show()


def plot_r2(df, var="ta"):
    df = df.copy()
    df["month"] = pd.to_datetime(df["month"])

    prefix = f"{var}_r2_"
    cols = [c for c in df.columns if c.startswith(prefix)]

    methods = [c.replace(prefix, "") for c in cols]

    for cat in df["category"].unique():
        df_cat = df[df["category"] == cat].sort_values("month")

        fig = go.Figure()

        for col, method in zip(cols, methods, strict=False):
            fig.add_trace(
                go.Scatter(
                    x=df_cat["month"],
                    y=df_cat[col],
                    mode="lines+markers",
                    name=method,
                )
            )

        fig.update_layout(
            title=f"R² ({var.upper()}) - {cat}",
            xaxis_title="Mois",
            yaxis_title="R²",
        )

        fig.show()


def plot_metric_by_method(
    df,
    metric="rmse",
    var="ta",  # "ta" ou "tdp"
):
    df = df.copy()

    df["month"] = pd.to_datetime(df["month"])
    df = df.sort_values("month")

    prefix = f"{var}_{metric}_"

    cols = [c for c in df.columns if c.startswith(prefix)]

    if not cols:
        raise ValueError(f"Aucune colonne trouvée pour {prefix}")

    df_long = df.melt(
        id_vars=["category", "month"],
        value_vars=cols,
        var_name="metric_full",
        value_name=metric,
    )

    df_long["method"] = df_long["metric_full"].str.replace(prefix, "")

    for cat in df_long["category"].unique():
        df_cat = df_long[df_long["category"] == cat]

        fig = px.line(
            df_cat,
            x="month",
            y=metric,
            color="method",
            markers=True,
            title=f"{metric.upper()} ({var.upper()}) - {cat} altitude",
        )

        ylabel = {
            "rmse": "RMSE (°C)",
            "mae": "MAE (°C)",
            "mbe": "MBE (°C)",
            "r2": "R²",
            "slope": "Slope",
        }.get(metric, metric)

        fig.update_layout(
            xaxis_title="Temps (mois)",
            yaxis_title=ylabel,
        )

        fig.show()


def compute_metrics(
    measured: npt.ArrayLike, estimated: npt.ArrayLike
) -> tuple[float, float, float, float, float]:
    """
    Compute slope, mbe, mae, rmse, r2
    """
    idx = np.isfinite(measured) & np.isfinite(estimated)
    slope, _ = np.polyfit(measured[idx], estimated[idx], 1)
    mbe = np.nanmean(estimated - measured)
    mae = mean_absolute_error(measured[idx], estimated[idx])
    rmse = np.sqrt(mean_squared_error(measured[idx], estimated[idx]))
    r2 = r2_score(measured[idx], estimated[idx])
    return (slope, mbe, mae, rmse, r2)


def plot_metrics(df: pd.DataFrame, variable: str = "ta"):
    """
    Interactive plot with Plotly (hover shows date)
    """

    groups = list(df.name.unique())
    groups.append("all")
    n_groups = len(groups)

    base_groups = df.name.unique()
    colors = px.colors.qualitative.Plotly

    color_map = {g: colors[i % len(colors)] for i, g in enumerate(base_groups)}

    ncols = 3
    nrows = int(np.ceil(n_groups / ncols))

    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        subplot_titles=groups,
        horizontal_spacing=0.06,
        vertical_spacing=0.14,
    )

    for i, group in enumerate(groups):
        row = i // ncols + 1
        col = i % ncols + 1

        if group == "all":
            measured = df[f"ec_{variable}"]
            estimated = df[variable]
            for g in df.name.unique():
                subset = df[df.name == g]

                fig.add_trace(
                    go.Scatter(
                        x=subset[f"ec_{variable}"],
                        y=subset[variable],
                        mode="markers",
                        marker={
                            "size": 6,
                            "opacity": 0.5,
                            "color": color_map[g],
                        },
                        name=g,
                        legendgroup=g,
                        showlegend=(group == "all"),
                        customdata=subset["date"],
                        hovertemplate=(
                            "Group: " + g + "<br>"
                            "Measured: %{x:.3f}<br>"
                            "Estimated: %{y:.3f}<br>"
                            "Date: %{customdata|%Y-%m-%d}<extra></extra>"
                        ),
                    ),
                    row=row,
                    col=col,
                )
        else:
            subset = df[df.name == group]
            measured = subset[f"ec_{variable}"]
            estimated = subset[variable]

            fig.add_trace(
                go.Scatter(
                    x=measured,
                    y=estimated,
                    mode="markers",
                    marker={
                        "size": 6,
                        "opacity": 0.6,
                        "color": color_map.get(group, "gray"),
                    },
                    name=group,
                    showlegend=False,
                    customdata=subset["date"],
                    hovertemplate=(
                        "Measured: %{x:.2f}<br>"
                        "Estimated: %{y:.2f}<br>"
                        "Date: %{customdata|%Y-%m-%d}<extra></extra>"
                    ),
                ),
                row=row,
                col=col,
            )

        lims = [
            min(measured.min(), estimated.min()),
            max(measured.max(), estimated.max()),
        ]

        fig.add_trace(
            go.Scatter(
                x=lims,
                y=lims,
                mode="lines",
                line={"dash": "dash", "color": "black"},
                showlegend=False,
            ),
            row=row,
            col=col,
        )

        idx = np.isfinite(measured) & np.isfinite(estimated)
        slope, intercept = np.polyfit(measured[idx], estimated[idx], 1)
        fig.add_trace(
            go.Scatter(
                x=lims,
                y=[lims[0] * slope + intercept, lims[1] * slope + intercept],
                mode="lines",
                line={"color": "red"},
                showlegend=False,
            ),
            row=row,
            col=col,
        )

        slope_m, mbe, mae, rmse, r2 = compute_metrics(measured, estimated)

        fig.add_annotation(
            x=0.97,
            y=0.03,
            xref="x domain",
            yref="y domain",
            text=(
                f"slope = {slope_m:.2f}<br>"
                f"MBE = {mbe:.2f}<br>"
                f"MAE = {mae:.2f}<br>"
                f"RMSE = {rmse:.2f}<br>"
                f"R2 = {r2:.2f}"
            ),
            showarrow=False,
            align="right",
            row=row,
            col=col,
        )

    fig.update_xaxes(title_text=f"Measured {variable.upper()}")
    fig.update_yaxes(title_text=f"Estimated {variable.upper()}")
    fig.update_layout(
        height=400 * nrows,
        width=500 * ncols,
        title=f"Measured vs Estimated {variable.upper()}",
    )

    fig.show()


cluster_results = [
    {"station": "BE-Bra", "altitude": np.float64(16.0), "cluster": 2},
    {"station": "BE-Vie", "altitude": np.float64(490.0), "cluster": 0},
    {"station": "BE-Lcr", "altitude": np.float64(6.25), "cluster": 2},
    {"station": "BE-Dor", "altitude": np.float64(253.0), "cluster": 2},
    {"station": "CH-Dav", "altitude": np.float64(1637.0), "cluster": 1},
    {"station": "CH-BaK", "altitude": np.float64(273.0), "cluster": 2},
    {"station": "CZ-wet", "altitude": np.float64(426.0), "cluster": 0},
    {"station": "CZ-BK1", "altitude": np.float64(881.0), "cluster": 0},
    {"station": "DE-RuW", "altitude": np.float64(610.0), "cluster": 0},
    {"station": "DE-Tha", "altitude": np.float64(380.0), "cluster": 2},
    {"station": "DE-RuS", "altitude": np.float64(106.0), "cluster": 2},
    {"station": "DE-Har", "altitude": np.float64(201.0), "cluster": 2},
    {"station": "DE-Brs", "altitude": np.float64(78.0), "cluster": 2},
    {"station": "DE-Geb", "altitude": np.float64(163.0), "cluster": 2},
    {"station": "DK-Vng", "altitude": np.float64(67.7), "cluster": 2},
    {"station": "DK-Skj", "altitude": np.float64(2.0), "cluster": 2},
    {"station": "DK-Gds", "altitude": np.float64(86.0), "cluster": 2},
    {"station": "ES-LMa", "altitude": np.float64(265.0), "cluster": 2},
    {"station": "FI-Tvm", "altitude": np.float64(1.0), "cluster": 2},
    {"station": "FI-Sii", "altitude": np.float64(164.0), "cluster": 2},
    {"station": "FI-Kmp", "altitude": np.float64(26.0), "cluster": 2},
    {"station": "FI-Ken", "altitude": np.float64(347.0), "cluster": 2},
    {"station": "FR-Tou", "altitude": np.float64(158.0), "cluster": 2},
    {"station": "FR-Mej", "altitude": np.float64(40.0), "cluster": 2},
    {"station": "FR-Lus", "altitude": np.float64(154.0), "cluster": 2},
    {"station": "FR-Lqu", "altitude": np.float64(1040.0), "cluster": 0},
    {"station": "FR-Hes", "altitude": np.float64(310.0), "cluster": 2},
    {"station": "FR-Gri", "altitude": np.float64(125.0), "cluster": 2},
    {"station": "FR-EM2", "altitude": np.float64(85.0), "cluster": 2},
    {"station": "FR-CLt", "altitude": np.float64(2050.6), "cluster": 1},
    {"station": "FR-Bil", "altitude": np.float64(39.18), "cluster": 2},
    {"station": "UK-AMo", "altitude": np.float64(268.0), "cluster": 2},
    {"station": "GF-Guy", "altitude": np.float64(40.0), "cluster": 2},
    {"station": "GL-ZaH", "altitude": np.float64(41.0), "cluster": 2},
    {"station": "GL-ZaF", "altitude": np.float64(42.0), "cluster": 2},
    {"station": "GR-HeM", "altitude": np.float64(69.0), "cluster": 2},
    {"station": "GR-HeK", "altitude": np.float64(30.0), "cluster": 2},
    {"station": "IT-TrF", "altitude": np.float64(2100.0), "cluster": 1},
    {"station": "IT-Tor", "altitude": np.float64(2168.0), "cluster": 1},
    {"station": "IT-SR2", "altitude": np.float64(4.0), "cluster": 2},
    {"station": "IT-Ren", "altitude": np.float64(1744.0), "cluster": 1},
    {"station": "IT-OXm", "altitude": np.float64(66.0), "cluster": 2},
    {"station": "IT-Niv", "altitude": np.float64(2750.0), "cluster": 1},
    {"station": "IT-MBo", "altitude": np.float64(1550.0), "cluster": 1},
    {"station": "IT-BCi", "altitude": np.float64(10.0), "cluster": 2},
    {"station": "NL-Loo", "altitude": np.float64(33.0), "cluster": 2},
    {"station": "NO-Hur", "altitude": np.float64(275.1308), "cluster": 2},
    {"station": "SE-Htm", "altitude": np.float64(115.0), "cluster": 2},
]

# mapping station -> cluster
station_to_cluster = {d["station"]: d["cluster"] for d in cluster_results}


def plot_metrics_by_method_and_category(stations_ts, var="ta"):

    enriched = []

    for station, df in stations_ts.items():
        cfg = get_stations_config(station)
        alt = cfg.elev

        if alt is None:
            continue
        logger.info(
            f"station = {station}, altitude = {cfg.elev}, category = {
                altitude_class(station, station_to_cluster)
            }"
        )

        df_ = df.copy()
        df_["category"] = altitude_class(
            station, station_to_cluster
        )  # altitude_class(alt)
        df_["time"] = pd.to_datetime(df_["time"])
        df_["station"] = station

        enriched.append(df_)

    if not enriched:
        return

    df_all = pd.concat(enriched, ignore_index=True)

    methods = [
        c
        for c in df_all.columns
        if c.startswith(f"{var}_") and c != f"{var}_icos"
    ]

    categories = sorted(df_all["category"].dropna().unique())

    colors = px.colors.qualitative.Plotly
    color_map = {
        cat: colors[i % len(colors)] for i, cat in enumerate(categories)
    }

    for col in methods:
        method = col.replace(f"{var}_", "")

        ncols = 3
        nrows = int(np.ceil(len(categories) / ncols))

        fig = make_subplots(
            rows=nrows,
            cols=ncols,
            subplot_titles=categories,
            horizontal_spacing=0.06,
            vertical_spacing=0.12,
        )

        for i, cat in enumerate(categories):
            row = i // ncols + 1
            col_pos = i % ncols + 1

            df_cat = df_all[df_all["category"] == cat]

            mask = df_cat[f"{var}_icos"].notna() & df_cat[col].notna()
            if mask.sum() == 0:
                continue

            x = df_cat.loc[mask, f"{var}_icos"]
            y = df_cat.loc[mask, col]
            dates = df_cat.loc[mask, "time"]

            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="markers",
                    marker={
                        "size": 6,
                        "opacity": 0.6,
                        "color": color_map[cat],
                    },
                    showlegend=False,
                    customdata=dates,
                    hovertemplate=(
                        "Measured: %{x:.2f}<br>"
                        "Estimated: %{y:.2f}<br>"
                        "Date: %{customdata|%Y-%m-%d}<extra></extra>"
                    ),
                ),
                row=row,
                col=col_pos,
            )

            lims = [
                min(x.min(), y.min()),
                max(x.max(), y.max()),
            ]

            fig.add_trace(
                go.Scatter(
                    x=lims,
                    y=lims,
                    mode="lines",
                    line={"dash": "dash", "color": "black"},
                    showlegend=False,
                ),
                row=row,
                col=col_pos,
            )

            idx = np.isfinite(x) & np.isfinite(y)
            slope, intercept = np.polyfit(x[idx], y[idx], 1)

            fig.add_trace(
                go.Scatter(
                    x=lims,
                    y=[
                        lims[0] * slope + intercept,
                        lims[1] * slope + intercept,
                    ],
                    mode="lines",
                    line={"color": "red"},
                    showlegend=False,
                ),
                row=row,
                col=col_pos,
            )

            slope_m, mbe, mae, rmse, r2 = compute_metrics(x, y)

            fig.add_annotation(
                x=0.97,
                y=0.03,
                xref="x domain",
                yref="y domain",
                text=(
                    f"slope = {slope_m:.2f}<br>"
                    f"MBE = {mbe:.2f}<br>"
                    f"MAE = {mae:.2f}<br>"
                    f"RMSE = {rmse:.2f}<br>"
                    f"R² = {r2:.2f}"
                ),
                showarrow=False,
                align="right",
                row=row,
                col=col_pos,
            )

        fig.update_xaxes(title_text="ICOS")
        fig.update_yaxes(title_text="Method")

        fig.update_layout(
            height=400 * nrows,
            width=500 * ncols,
            title=f"ICOS vs {method} ({var.upper()})",
        )

        fig.show()


def plot_metrics_total(stations_ts, var="ta"):

    enriched = []

    for station, df in stations_ts.items():
        cfg = get_stations_config(station)
        alt = cfg.elev

        if alt is None:
            continue

        df_ = df.copy()
        df_["time"] = pd.to_datetime(df_["time"])
        df_["station"] = station

        enriched.append(df_)

    if not enriched:
        return

    df_all = pd.concat(enriched, ignore_index=True)

    methods = [
        c
        for c in df_all.columns
        if c.startswith(f"{var}_") and c != f"{var}_icos"
    ]

    for col in methods:
        method = col.replace(f"{var}_", "")

        x = df_all[f"{var}_icos"]
        y = df_all[col]
        dates = df_all["time"]

        mask = x.notna() & y.notna()
        x = x[mask]
        y = y[mask]
        dates = dates[mask]

        if len(x) == 0:
            continue

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                marker={"size": 6, "opacity": 0.6},
                customdata=dates,
                hovertemplate=(
                    "Measured: %{x:.2f}<br>"
                    "Estimated: %{y:.2f}<br>"
                    "Date: %{customdata|%Y-%m-%d}<extra></extra>"
                ),
            )
        )

        lims = [min(x.min(), y.min()), max(x.max(), y.max())]

        fig.add_trace(
            go.Scatter(
                x=lims,
                y=lims,
                mode="lines",
                line={"dash": "dash", "color": "black"},
                showlegend=False,
            )
        )

        idx = np.isfinite(x) & np.isfinite(y)
        slope, intercept = np.polyfit(x[idx], y[idx], 1)

        fig.add_trace(
            go.Scatter(
                x=lims,
                y=[lims[0] * slope + intercept, lims[1] * slope + intercept],
                mode="lines",
                line={"color": "red"},
                showlegend=False,
            )
        )

        slope_m, mbe, mae, rmse, r2 = compute_metrics(x, y)

        fig.add_annotation(
            x=0.97,
            y=0.03,
            xref="paper",
            yref="paper",
            text=(
                f"slope = {slope_m:.2f}<br>"
                f"MBE = {mbe:.2f}<br>"
                f"MAE = {mae:.2f}<br>"
                f"RMSE = {rmse:.2f}<br>"
                f"R² = {r2:.2f}"
            ),
            showarrow=False,
            align="right",
        )

        fig.update_layout(
            title=f"ICOS vs {method} ({var.upper()}) - all stations",
            xaxis_title="ICOS",
            yaxis_title="Method",
            width=700,
            height=600,
        )

        fig.show()


def plot_metric_by_month(
    df,
    variable,
    metric="rmse",
    figsize=(20, 6),
    rotate_xticks=45,
):
    """
    Graphe clean par mois et par méthode.
    """

    data = df[df["variable"] == variable].copy()

    data["month"] = pd.to_datetime(data["month"])
    data = data.sort_values("month")

    data["month_str"] = data["month"].dt.strftime("%Y-%m")

    data["method_clean"] = data["method"].str.replace(
        f"{variable}_", "", regex=False
    )

    palette = [
        "#172E8BE6",
        "#de1832",
        "#0f9f1d",
        "#f77f00",
        "#6a4c93",
        "#1982c4",
        "#8ac926",
        "#BB53B2",
    ]

    plt.figure(figsize=figsize)

    ax = sns.barplot(
        data=data,
        x="month_str",
        y=metric,
        hue="method_clean",
        palette=palette,
    )

    ax.set_xlabel("Months", fontsize=15)
    ax.set_ylabel(metric.upper(), fontsize=15)

    ax.set_title(f"{variable} {metric.upper()} by month", fontsize=18, pad=20)

    plt.xticks(rotation=rotate_xticks)

    ax.grid(axis="y", linestyle="--", alpha=1)
    # ax.grid(axis="x", visible=False)

    sns.despine()

    plt.legend(
        title="Method",
        bbox_to_anchor=(1, 1),
        loc="upper left",
        frameon=False,
        fontsize=14,
        title_fontsize=16,
        markerscale=1.5,
    )
    plt.tight_layout()
    plt.show()


def plot_rmse_heatmap(
    df,
    variable,
    metric="rmse",
    figsize=(12, 8),
    cmap="viridis_r",
    annot=True,
):
    """
    Heatmap des métriques par mois et méthode.
    """

    data = df[df["variable"] == variable].copy()

    data["month"] = pd.to_datetime(data["month"])
    data = data.sort_values("month")

    data["month_str"] = data["month"].dt.strftime("%Y-%m")

    data["method_clean"] = data["method"].str.replace(
        f"{variable}_", "", regex=False
    )

    pivot = data.pivot(index="month_str", columns="method_clean", values=metric)

    desired_order = [
        "era5",
        "base",
        "method_1",
        "method_2",
        "method_3",
        "method_4",
        "method_5",
        "method_6",
    ]

    existing_cols = [c for c in desired_order if c in pivot.columns]
    pivot = pivot[existing_cols]

    plt.figure(figsize=figsize)

    sns.heatmap(
        pivot,
        annot=annot,
        fmt=".2f",
        cmap=cmap,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": metric.upper()},
    )

    plt.title(
        f"{metric.upper()} heatmap — {variable}",
        fontsize=18,
        weight="bold",
        pad=20,
    )

    plt.xlabel("Method", fontsize=14)
    plt.ylabel("Month", fontsize=14)

    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.show()


def plot_metric_lineplot(
    df,
    variable,
    metric="rmse",
    figsize=(16, 7),
):
    """
    Courbes temporelles des métriques par méthode.
    """

    data = df[df["variable"] == variable].copy()

    data["month"] = pd.to_datetime(data["month"])
    data = data.sort_values("month")

    data["month_str"] = data["month"].dt.strftime("%Y-%m")

    data["method_clean"] = data["method"].str.replace(
        f"{variable}_", "", regex=False
    )

    method_order = [
        "era5",
        "base",
        "method_1",
        "method_2",
        "method_3",
        "method_4",
        "method_5",
        "method_6",
    ]

    sns.set_theme(style="whitegrid")

    palette = [
        "#00429d",
        "#d1495b",
        "#2a9d8f",
        "#f77f00",
        "#6a4c93",
        "#1982c4",
        "#8ac926",
        "#222222",
    ]

    plt.figure(figsize=figsize)

    sns.lineplot(
        data=data,
        x="month_str",
        y=metric,
        hue="method_clean",
        hue_order=method_order,
        palette=palette,
        marker="o",
        linewidth=2.5,
    )

    plt.title(
        f"{metric.upper()} by month — {variable}", fontsize=20, weight="bold"
    )

    plt.xlabel("Month", fontsize=14)
    plt.ylabel(metric.upper(), fontsize=14)

    plt.xticks(rotation=45)

    plt.legend(
        title="Method",
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
        frameon=False,
    )

    plt.tight_layout()
    plt.show()


def plot_metric_boxplot(
    df,
    variable,
    metric="rmse",
    figsize=(12, 7),
):
    """
    Distribution des métriques par méthode.
    """

    data = df[df["variable"] == variable].copy()

    data["method_clean"] = data["method"].str.replace(
        f"{variable}_", "", regex=False
    )

    method_order = [
        "era5",
        "base",
        "method_1",
        "method_2",
        "method_3",
        "method_4",
        "method_5",
        "method_6",
    ]

    sns.set_theme(style="whitegrid")

    palette = [
        "#00429d",
        "#d1495b",
        "#2a9d8f",
        "#f77f00",
        "#6a4c93",
        "#1982c4",
        "#8ac926",
        "#E6C347",
    ]

    plt.figure(figsize=figsize)

    sns.boxplot(
        data=data,
        x="method_clean",
        y=metric,
        order=method_order,
        palette=palette,
        width=0.7,
    )

    sns.stripplot(
        data=data,
        x="method_clean",
        y=metric,
        order=method_order,
        color="black",
        alpha=0.4,
        size=4,
    )

    plt.title(
        f"{metric.upper()} distribution — {variable}",
        fontsize=20,
        weight="bold",
    )

    plt.xlabel("Method", fontsize=14)
    plt.ylabel(metric.upper(), fontsize=14)

    plt.xticks(rotation=20)

    plt.tight_layout()
    plt.show()


plt.rcParams.update(
    {
        "font.size": 16,
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
    }
)


def compute_metrics_table(
    dfs,
    variable="ta",
    methods=None,
):

    truth_col = f"{variable}_icos"

    if methods is None:
        sample_df = next(iter(dfs.values()))

        methods = [
            c
            for c in sample_df.columns
            if c.startswith(variable) and c != truth_col
        ]

    rows = []

    for method in methods:
        y_true = []
        y_pred = []

        for df in dfs.values():
            tmp = df[[truth_col, method]].dropna()

            y_true.append(tmp[truth_col])
            y_pred.append(tmp[method])

        y_true = pd.concat(y_true)
        y_pred = pd.concat(y_pred)

        errors = y_pred - y_true

        rows.append(
            {
                "method": method,
                "bias": errors.mean(),
                "mae": mean_absolute_error(y_true, y_pred),
                "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
                "std": errors.std(),
                "corr": np.corrcoef(y_true, y_pred)[0, 1],
                "r2": r2_score(y_true, y_pred),
                "median_error": np.median(errors),
                "p95_abs_error": np.percentile(np.abs(errors), 95),
                "n": len(errors),
            }
        )

    metrics = pd.DataFrame(rows)

    return metrics.sort_values("rmse")


def plot_error_distributions(
    dfs,
    variable="ta",
    methods=None,
    bins=40,
    kde=True,
    fontsize=18,
):

    truth_col = f"{variable}_icos"

    if methods is None:
        sample_df = next(iter(dfs.values()))

        methods = [
            c
            for c in sample_df.columns
            if c.startswith(variable) and c != truth_col
        ]

    n_methods = len(methods)

    nrows = 2
    ncols = math.ceil(n_methods / nrows)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(7 * ncols, 5 * nrows),
        sharex=True,
        sharey=True,
    )

    axes = np.array(axes).flatten()

    fig.suptitle(
        "Global Error Distributions",
        fontsize=fontsize + 6,
        fontweight="bold",
    )

    global_errors = []

    for method in methods:
        for df in dfs.values():
            errors = (df[method] - df[truth_col]).dropna()

            global_errors.extend(errors.values)

    global_errors = np.array(global_errors)

    xlim = (
        np.percentile(global_errors, 0.1),
        np.percentile(global_errors, 99.9),
    )

    max_count = 0

    for ax, method in zip(axes, methods, strict=False):
        all_errors = []

        for df in dfs.values():
            errors = (df[method] - df[truth_col]).dropna()

            all_errors.append(errors)

        all_errors = pd.concat(all_errors)

        hist = sns.histplot(
            all_errors,
            bins=bins,
            kde=kde,
            ax=ax,
        )

        max_count = max(
            max_count, *([p.get_height() for p in hist.patches] + [0])
        )

        ax.axvline(
            0,
            color="black",
            linestyle="--",
            linewidth=2,
        )

        mean_err = all_errors.mean()
        std_err = all_errors.std()
        rmse = np.sqrt((all_errors**2).mean())

        ax.set_title(
            f"{method}\n"
            f"Bias = {mean_err:.2f} °C | "
            f"Std = {std_err:.2f} °C | "
            f"RMSE = {rmse:.2f} °C",
            fontsize=fontsize + 3,
        )

        ax.set_xlabel(
            "Error (°C)",
            fontsize=fontsize,
        )

        ax.set_ylabel(
            "Count",
            fontsize=fontsize,
        )

        ax.tick_params(
            axis="both",
            labelsize=fontsize - 2,
        )

        ax.set_xlim(xlim)

    # fixed y-axis
    for ax in axes[:n_methods]:
        ax.set_ylim(0, max_count * 1.1)

    # hide unused
    for ax in axes[n_methods:]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.show()


def cluster_stations_by_altitude(
    stations,
    n_clusters=3,
    random_state=42,
    plot=True,
):
    """
    Clustering KMeans des stations selon leur altitude.
    """

    # -----------------------------
    # Récupération des altitudes
    # -----------------------------
    station_names = []
    altitudes = []

    for station in stations:
        cfg = get_stations_config(station)
        alt = cfg.elev

        station_names.append(station)
        altitudes.append(alt)

    quantiles = pd.qcut(altitudes, q=3)
    altitudes = np.array(altitudes).reshape(-1, 1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(altitudes)

    # KMeans
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=1000,  # "auto",
    )

    labels = kmeans.fit_predict(X_scaled)

    results = []

    for station, alt, label in zip(
        station_names, altitudes.flatten(), labels, strict=False
    ):
        results.append(
            {
                "station": station,
                "altitude": alt,
                "cluster": int(label),
            }
        )

    if n_clusters > 1:
        score = silhouette_score(X_scaled, labels)
        logger.info(f"Silhouette score: {score:.3f}")

    if plot:
        plt.figure(figsize=(8, 7))

        rng = np.random.default_rng(seed=random_state)
        x_random = rng.uniform(0, 1, size=len(altitudes))

        plt.scatter(
            x_random,
            altitudes.flatten(),
            c=labels,
            s=80,
            alpha=0.8,
        )

        centers = scaler.inverse_transform(kmeans.cluster_centers_).flatten()

        plt.scatter(
            np.full(len(centers), 0.5),
            centers,
            marker="x",
            s=30,
            linewidths=1,
            label="Kmeans centers",
            color="red",
        )

    for x, station, alt in zip(
        x_random,
        station_names,
        altitudes.flatten(),
        strict=False,
    ):
        plt.text(
            x - 0.02,
            alt + 65,
            str(station),
            fontsize=8,
            alpha=0.8,
        )

    plt.ylabel("Elevation (m)")
    plt.xlabel("Randomized station position")
    plt.title(f"KMeans clustering of stations by elevation (k={n_clusters})")

    plt.xticks([])

    plt.grid(True, axis="y", alpha=0.3)

    plt.legend()

    plt.show()

    return results, kmeans, quantiles

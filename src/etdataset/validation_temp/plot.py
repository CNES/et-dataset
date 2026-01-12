# type: ignore
# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales

"""
Functions for plotting
"""
# Skip this file with mypy

from pathlib import Path

import geopandas as gpd
import pandas as pd
import plotly.graph_objects as go

from etdataset.logging import LoggerManager

logger = LoggerManager.get_logger(__name__)


def plot_variable(
    stations: dict,
    var: str,
    colors: dict,
    start: str | None = None,
    end: str | None = None,
):
    start_date = pd.to_datetime(start) if start else None
    end_date = pd.to_datetime(end) if end else None

    fig = go.Figure()

    for station_name, df in stations.items():
        df_v = df.sort_values("time")

        if start_date is not None:
            df_t = df_v[df_v["time"] >= start_date]
        if end_date is not None:
            df_t = df_v[df_v["time"] <= end_date]

        if df.empty:
            continue

        # ---------- ICOS ----------
        fig.add_trace(
            go.Scatter(
                x=df_t["time"],
                y=df_t[f"{var}_icos"],
                name=f"{station_name} ICOS",
                legendgroup=station_name,
                legendgrouptitle_text=f"Station {station_name}",
                line={"color": colors["icos"], "dash": "solid"},
                mode="lines+markers",
            )
        )

        # ---------- ERA5 ----------
        fig.add_trace(
            go.Scatter(
                x=df_t["time"],
                y=df_t[f"{var}_era5"],
                name=f"{station_name} ERA5",
                legendgroup=station_name,
                line={"color": colors["era5"], "dash": "dash"},
                mode="lines+markers",
            )
        )

        # ---------- ERA5 resampled ----------
        fig.add_trace(
            go.Scatter(
                x=df_t["time"],
                y=df_t[f"{var}_era5_proj"],
                name=f"{station_name} ERA5 resampled",
                legendgroup=station_name,
                line={"color": colors["era5_sampled"], "dash": "dot"},
                mode="lines+markers",
            )
        )

    if var == "Ta":
        title = "Air temperature of ICOS stations"
    elif var == "Tdp":
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


def load_csv_data_timeseries(csv_dir: Path):
    stations = {}

    for csv_file in csv_dir.glob("*_timeseries.csv"):
        station_name = csv_file.stem.split("_")[0].split("-")[1]
        df = pd.read_csv(csv_file, parse_dates=["time"])
        stations[station_name] = df

    return stations


def plot_ta_tdp_csv(
    csv_dir,
    start: str | None = None,
    end: str | None = None,
):
    stations = load_csv_data_timeseries(csv_dir)

    color_ta = {
        "icos": "#1f77b4",
        "era5": "#32b332",
        "era5_sampled": "#9543bb",
    }

    color_tdp = {
        "icos": "#ff9137",
        "era5": "#d62d10",
        "era5_sampled": "#a76223",
    }

    fig_ta = plot_variable(stations, "Ta", color_ta, start, end)
    fig_tdp = plot_variable(stations, "Tdp", color_tdp, start, end)

    fig_ta.show()
    fig_tdp.show()


def load_csv_data_lr(csv_dir: Path):
    stations = {}

    for csv_file in csv_dir.glob("*_lapse_rate.csv"):
        station_name = csv_file.stem.split("_")[0].split("-")[1]
        df = pd.read_csv(csv_file, parse_dates=["time"])
        stations[station_name] = df

    return stations


def plot_lr_csv(csv_dir: Path, start: str, end: str):
    stations = load_csv_data_lr(csv_dir)

    fig_ta = go.Figure()
    fig_tdp = go.Figure()

    for station_name, df in stations.items():
        if start:
            df_f = df[df["time"] >= pd.to_datetime(start)]
        if end:
            df_f = df[df["time"] <= pd.to_datetime(end)]

        fig_ta.add_trace(
            go.Scatter(
                x=df_f["time"],
                y=df_f["lapse_rate_ta"],
                mode="lines+markers",
                name=f"Station {station_name}",
            )
        )
        fig_tdp.add_trace(
            go.Scatter(
                x=df_f["time"],
                y=df_f["lapse_rate_tdp"],
                mode="lines+markers",
                name=f"Station {station_name}",
            )
        )

    fig_ta.update_layout(
        title={"text": "Standart atmospheric Lapse Rate of ICOS Stations"},
        xaxis_title="Time",
        yaxis_title="Lapse Rate (K.m-1)",
        plot_bgcolor="rgb(230, 230, 230)",
        showlegend=True,
        template="plotly",
    )
    fig_tdp.update_layout(
        title={"text": "Dew point Lapse Rate of ICOS Stations"},
        xaxis_title="Time",
        yaxis_title="Lapse Rate (K.m-1)",
        plot_bgcolor="rgb(230, 230, 230)",
        showlegend=True,
        template="plotly",
    )
    fig_ta.add_hline(
        y=-0.0065,
        line_dash="dash",
        line_color="black",
        annotation_text="Theoretical value",
    )
    fig_tdp.add_hline(
        y=-0.0065,
        line_dash="dash",
        line_color="black",
        annotation_text="Theoretical value",
    )
    fig_ta.show()
    fig_tdp.show()


def plot_station_map(csv_path: str = "stations_package.gpkg"):
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

# type: ignore
# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales

"""
Functions for plotting
"""
# Skip this file with mypy

import math
import os
import re
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import xarray as xr
from plotly.subplots import make_subplots
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from etdataset.icos import get_stations_config
from etdataset.interpolation import reproject_era5_grid
from etdataset.logging import LoggerManager
from temperature.analyze import StationClusterResult, load_csv_data_icos
from temperature.liaise import METHOD, get_method_paths
from temperature.metrics import altitude_class, compute_metrics

logger = LoggerManager.get_logger(__name__)


def plot_variable(
    stations: dict[str, pd.DataFrame],
    var: str,
    start: str | None = None,
    end: str | None = None,
):
    """
    Description
    -----------
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
    Description
    -----------
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
    Description
    -----------
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
    Description
    -----------
    Plot ICOS air temperature or dew point temperature for all stations

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
    Description
    -----------
    Plot ICOS air temperature and dew point temperature time series

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
    Description
    -----------
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


def plot_all_stations_timeseries(
    stations_ts: dict[str, pd.DataFrame],
    var: str,
    hour: str | list[str] = "all",
    start: str | None = None,
    end: str | None = None,
):
    """
     Description
    -----------
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
    var="ta",  # "ta" or "tdp"
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


def plot_metrics(df: pd.DataFrame, variable: str = "ta"):
    """
    Interactive plot with Plotly
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


# cluster_results = [
#     {"station": "BE-Vie", "altitude": float(2), "cluster": 2},
# ]

# mapping station -> cluster
# station_to_cluster = {d["station"]: d["cluster"] for d in cluster_results}


def plot_metrics_by_method_and_category(
    stations_ts: dict[str, pd.DataFrame],
    station_to_cluster: list[StationClusterResult],
    var="ta",
):

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
    Graph per months and per method
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
    Heatmap of metrics per months and per methods
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
    Temporal lineplot of metrics per methods
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
    Distribution of metrics per methods
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


def plot_rmse_all_stations(csv_dir: str):

    csv_files = [f for f in os.listdir(csv_dir) if f.endswith(".csv")]

    for file in csv_files:
        station_name = file.replace("_monthly_metrics.csv", "")
        df = pd.read_csv(os.path.join(csv_dir, file))
        df["month"] = pd.to_datetime(df["month"])

        for var in ["ta", "tdp"]:
            plt.figure(figsize=(10, 6))

            rmse_cols = [
                c
                for c in df.columns
                if c.startswith(f"{var}_rmse_")
                and not c.endswith(("min", "max"))
            ]

            for col in rmse_cols:
                method = col.replace(f"{var}_rmse_", "")

                plt.plot(df["month"], df[col], marker="o", label=method)

                min_col = f"{var}_rmse_{method}_min"
                max_col = f"{var}_rmse_{method}_max"

                if min_col in df.columns and max_col in df.columns:
                    plt.fill_between(
                        df["month"], df[min_col], df[max_col], alpha=0.2
                    )

            plt.title(f"{station_name.upper()} - RMSE mensuel ({var.upper()})")
            plt.xlabel("Mois")
            plt.ylabel("RMSE")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()


def plot_r2_all_stations(csv_dir: str):

    csv_files = [f for f in os.listdir(csv_dir) if f.endswith(".csv")]

    for file in csv_files:
        station_name = file.replace("_monthly_metrics.csv", "")
        df = pd.read_csv(os.path.join(csv_dir, file))
        df["month"] = pd.to_datetime(df["month"])

        for var in ["ta", "tdp"]:
            plt.figure(figsize=(10, 6))

            rmse_cols = [
                c
                for c in df.columns
                if c.startswith(f"{var}_r2_") and not c.endswith(("min", "max"))
            ]

            for col in rmse_cols:
                method = col.replace(f"{var}_r2_", "")

                plt.plot(df["month"], df[col], marker="o", label=method)

                min_col = f"{var}_r2_{method}_min"
                max_col = f"{var}_r2_{method}_max"

                if min_col in df.columns and max_col in df.columns:
                    plt.fill_between(
                        df["month"], df[min_col], df[max_col], alpha=0.2
                    )

            plt.title(f"{station_name.upper()} - R2 mensuel ({var.upper()})")
            plt.xlabel("Mois")
            plt.ylabel("R2")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()


def plot_slope_scatter_all_stations(
    stations_ts: dict[str, pd.DataFrame],
):

    for station_name, df_ in stations_ts.items():
        df = df_.copy()
        df["month"] = df["time"].dt.to_period("M")

        for var in ["ta", "tdp"]:
            icos_col = f"{var}_icos"
            methods = [
                c
                for c in df.columns
                if c.startswith(f"{var}_") and c != icos_col
            ]

            plt.figure(figsize=(7, 7))

            for m in methods:
                xs = []
                ys = []

                for _month, group in df.groupby("month"):
                    valid = group[[icos_col, m]].dropna()

                    if len(valid) > 5:
                        xs.append(valid[icos_col].mean())
                        ys.append(valid[m].mean())

                if xs:
                    plt.scatter(xs, ys, label=m.replace(f"{var}_", ""))

            min_val = df[icos_col].min()
            max_val = df[icos_col].max()
            plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")

            plt.title(f"{station_name.upper()} - Slope scatter ({var.upper()})")
            plt.xlabel("ICOS (réel)")
            plt.ylabel("Calculé")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()


def plot_liaise_data(
    path: str,
    year: int,
    month: int,
    day: int,
    variable: str = "ta",
    xmin: float | None = None,
    ymin: float | None = None,
    xmax: float | None = None,
    ymax: float | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    path_list = get_method_paths(path=path, year=year, month=month, day=day)
    era5_path = next((p for p in path_list if "era5" in str(p).lower()), None)
    other_paths = [p for p in path_list if "era5" not in str(p).lower()]

    if era5_path is not None and other_paths:
        ds_era5 = xr.open_dataset(era5_path)
        ds_ref = xr.open_dataset(other_paths[0])
        era5_reproj = reproject_era5_grid(ds_era5[variable], ds_ref[variable])
        datasets = []
        for p in path_list:
            if "era5" in str(p).lower():
                datasets.append(("ERA5 (reprojected)", era5_reproj))
            else:
                ds = xr.open_dataset(p)
                match = re.search(r"method_(\d+)", str(p))
                method_name = (
                    f"Method {match.group(1)}"
                    if match
                    else next((m for m in METHOD if m in str(p)), Path(p).stem)
                )
                datasets.append((method_name, ds[variable]))
    else:
        datasets = []
        for p in path_list:
            ds = xr.open_dataset(p)
            match = re.search(r"method_(\d+)", str(p))
            method_name = (
                f"Method {match.group(1)}"
                if match
                else next((m for m in METHOD if m in str(p)), Path(p).stem)
            )
            datasets.append((method_name, ds[variable]))

    if vmin is None or vmax is None:
        all_value = []
        for _, data in datasets:
            if None not in (xmin, ymin, xmax, ymax):
                y_dim, x_dim = data.dims[:2]
                data_plot = data.sel(
                    {x_dim: slice(xmin, xmax), y_dim: slice(ymax, ymin)}
                )
            else:
                data_plot = data
            vals = data_plot.values.ravel()
            all_value.append(vals[~np.isnan(vals)])

        all_values = np.concatenate(all_value)
        vmin = vmin if vmin is not None else float(np.nanmin(all_values))
        vmax = vmax if vmax is not None else float(np.nanmax(all_values))

    logger.info(f"colorbar vmin={vmin}, vmax={vmax}")

    n = len(datasets)
    cols = 4
    rows = math.ceil(n / cols)
    fig, axs = plt.subplots(
        rows,
        cols,
        figsize=(5 * cols, 4 * rows),
        squeeze=False,
        constrained_layout=True,
    )

    for i, (method_name, data) in enumerate(datasets):
        row, col = i // cols, i % cols

        if None not in (xmin, ymin, xmax, ymax):
            y_dim, x_dim = data.dims[:2]
            data_plot = data.sel(
                {x_dim: slice(xmin, xmax), y_dim: slice(ymax, ymin)}
            )
        else:
            data_plot = data

        dims = data_plot.dims
        im = axs[row, col].imshow(
            data_plot.values,
            aspect="auto",
            cmap="RdYlBu_r",
            vmin=vmin,
            vmax=vmax,
            extent=[
                data_plot[dims[1]].values.min(),
                data_plot[dims[1]].values.max(),
                data_plot[dims[0]].values.min(),
                data_plot[dims[0]].values.max(),
            ],
            origin="upper",
        )
        axs[row, col].set_title(method_name)
        axs[row, col].set_xlabel(dims[1])
        axs[row, col].set_ylabel(dims[0])

    for j in range(n, rows * cols):
        fig.delaxes(axs[j // cols, j % cols])

    fig.colorbar(im, ax=axs, location="right", fraction=0.03, pad=0.02)
    fig.suptitle(
        f"LIAISE - {variable} ({year}-{month:02d}-{day:02d})", fontsize=16
    )
    plt.show()

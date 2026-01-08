# type: ignore
# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales

"""
Functions for plotting
"""
# Skip this file with mypy

import matplotlib.pyplot as plt
import pandas as pd


def _plot_variable_subplots(stations: dict, var: str, color: dict):
    n = len(stations)
    figsize = (20 * n, 8)

    fig, axes = plt.subplots(1, n, figsize=figsize, sharex=False)
    if n == 1:
        axes = [axes]

    for ax, (station_name, df_it) in zip(axes, stations.items(), strict=True):
        df = df_it.sort_values("time")

        ax.plot(
            df["time"],
            df[f"{var}_icos"],
            label=f"{var} ICOS",
            color=color["icos"],
            linestyle="-",
            marker="o",
        )
        ax.plot(
            df["time"],
            df[f"{var}_era5"],
            label=f"{var} ERA5",
            color=color["era5"],
            linestyle="--",
            marker="s",
        )
        ax.plot(
            df["time"],
            df[f"{var}_era5_proj"],
            label=f"{var} ERA5 proj",
            color=color["era5_sampled"],
            linestyle=":",
            marker="^",
        )

        ax.set_title(f"Station {station_name} — {var}", fontsize=13)
        ax.set_ylabel(f"{var} (°C)")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(fontsize=10)

    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def plot_ta_tdp_csv(stations: dict, start: str | None, end: str | None):
    start_date = pd.to_datetime(start) if start else None
    end_date = pd.to_datetime(end) if end else None

    filtered_stations = {}

    for name, station_df in stations.items():
        if start:
            mask_start = station_df["time"] >= start_date

        if end:
            mask_end = station_df["time"] <= end_date

        filtered_stations[name] = station_df[mask_start & mask_end]

    color_ta = {"icos": "#1f77b4", "era5": "#32b332", "era5_sampled": "#9543bb"}
    color_tdp = {
        "icos": "#ff9137",
        "era5": "#d62d10",
        "era5_sampled": "#a76223",
    }

    _plot_variable_subplots(filtered_stations, "Ta", color_ta)
    _plot_variable_subplots(filtered_stations, "Tdp", color_tdp)

    plt.show()


def _plot_lr_subplots(
    stations: dict,
    var: str,
    color: dict,
    mean_theoretical: float,
):
    n = len(stations)
    figsize = (20 * n, 8)

    fig, axes = plt.subplots(1, n, figsize=figsize, sharex=False)
    if n == 1:
        axes = [axes]

    for ax, (station_name, df_it) in zip(axes, stations.items(), strict=True):
        df = df_it.sort_values("time")

        ax.plot(
            df["time"],
            df[f"{var}"],
            label=f"{var} ERA5",
            color=color["era5"],
            linestyle="-",
            marker="o",
        )

        ax.axhline(
            y=mean_theoretical,
            color=color.get("mean", "black"),
            linestyle="--",
            linewidth=2,
            label="Theoretical value",
        )

        ax.set_title(f"Station {station_name} — {var}", fontsize=13)
        ax.set_ylabel(f"{var} (K/km)")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(fontsize=10)

    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def plot_lr_csv(stations: dict, start: str | None, end: str | None):
    start_date = pd.to_datetime(start) if start else None
    end_date = pd.to_datetime(end) if end else None

    filtered_stations = {}

    for name, station_df in stations.items():
        if start:
            mask_start = station_df["time"] >= start_date

        if end:
            mask_end = station_df["time"] <= end_date

        filtered_stations[name] = station_df[mask_start & mask_end]

    color_ta = {"era5": "#32b332"}
    color_tdp = {"era5": "#9543bb"}

    _plot_lr_subplots(filtered_stations, "lapse_rate_ta", color_ta, -0.0065)
    _plot_lr_subplots(filtered_stations, "lapse_rate_tdp", color_tdp, -0.0065)

    plt.show()

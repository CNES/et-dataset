# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales

"""
Module for analyzing
"""

import os
from pathlib import Path
from typing import TypedDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

from etdataset.icos import get_stations_config
from etdataset.logging import LoggerManager

logger = LoggerManager.get_logger(__name__)


def list_csv_files(folder: str) -> list[str]:
    """
    Description
    -----------
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
    Description
    -----------
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
    Description
    -----------
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
##             warnings                ##
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
    Description
    -----------
    Log warnings for stations missing in one or more data sources

    Parameters
    ----------
    stations : list[str] | str
        Stations to check
    era5_files : list[str]
        list of ERA5 csv files
    era5r_files : list[str]
        list of ERA5 rescaled CSV files
    icos_files : list[str]
        list of ICOS CSV files

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
    Description
    -----------
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
    Description
    -----------
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
    Description
    -----------
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
    Description
    -----------
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
    """
    # Filtrer méthodes valides
    valid_methods = {k: v for k, v in methods.items() if v is not None}
    logger.info(f"icos_files : {icos_folder}")
    # ERA5 + ICOS
    logger.info(f"stations : {stations}")
    logger.info(f"stations list : {list_csv_files(icos_folder)}")
    era5_files = filter_files_by_station(list_csv_files(era5_folder), stations)
    icos_files = filter_files_by_station(list_csv_files(icos_folder), stations)
    logger.info(f"icos_files : {icos_files}")
    era5_data = load_csv_files(era5_files)
    logger.info(f"era5 : {era5_data}")
    icos_data = load_csv_files(icos_files)
    logger.info(f"icos_data : {icos_data}")
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


class StationClusterResult(TypedDict):
    station: str
    altitude: float
    cluster: int


def cluster_stations_by_altitude(
    stations: list[str],
    n_clusters: int = 3,
    random_state: int = 42,
    plot: bool = True,
) -> tuple[list[StationClusterResult], KMeans, pd.Categorical]:
    """
    Description
    -----------
    Cluster stations by altitude using KMeans.

    Parameters
    ----------
    stations : list[str]
        List of station identifiers to cluster
    n_clusters : int, default=3
        Number of clusters for KMeans
    random_state : int, default=42
        Random seed for reproducibility
    plot : bool, default=True
        Whether to display a scatter plot of the clustering result

    Returns
    -------
    tuple[list[StationClusterResult], KMeans, pd.Categorical]
    """

    station_names = []
    altitudes = []

    for station in stations:
        cfg = get_stations_config(station)
        alt = cfg.elev

        station_names.append(station)
        altitudes.append(alt)

    quantiles = pd.qcut(altitudes, q=3)
    altitudes_arr = np.array(altitudes).reshape(-1, 1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(altitudes_arr)

    # KMeans
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=1000,  # "auto",
    )

    labels = kmeans.fit_predict(X_scaled)

    results = []

    for station, alt, label in zip(
        station_names, altitudes_arr.flatten(), labels, strict=False
    ):
        results.append(
            StationClusterResult(
                station=station,
                altitude=float(alt),
                cluster=int(label),
            )
        )
    # Compute silhouette score
    if n_clusters > 1:
        score = silhouette_score(X_scaled, labels)
        logger.info(f"Silhouette score: {score:.3f}")

    if plot:
        plt.figure(figsize=(8, 7))
        # Randomize X position for visibility
        rng = np.random.default_rng(seed=random_state)
        x_random = rng.uniform(0, 1, size=len(altitudes))

        plt.scatter(
            x_random,
            altitudes_arr.flatten(),
            c=labels,
            s=80,
            alpha=0.8,
        )
        # Plot cluster centers
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
        altitudes_arr.flatten(),
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

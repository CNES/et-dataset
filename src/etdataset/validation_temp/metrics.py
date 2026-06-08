# type: ignore

#
# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales /
#            Université Paul Sabatier (UT3)
#


"""
Functions for metrics
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)

from etdataset.icos import get_stations_config
from etdataset.logging import LoggerManager

logger = LoggerManager.get_logger(__name__)

#########################################
##                                     ##
##                                     ##
##              Metrics                ##
##                                     ##
##                                     ##
#########################################


def slope_forced_origin(x, y):
    """
    Description
    -----------
    Compute the slope of a linear regression forced through the origin.

    This regression assumes a model of the form:
        y = a * x
    with no intercept term. The slope is estimated using a least-squares
    approach:

        a = Σ(x_i * y_i) / Σ(x_i^2)

    Any pair (x_i, y_i) containing NaN values is removed before computation.
    If Σ(x_i²) = 0, NaN is returned.

    Parameters
    ----------
    x : array-like
        Predictor values (ERA5 values).
    y : array-like
        observed values (ICOS values).

    Returns
    -------
    float
        Estimated slope 'a' of the regression forced through zero.
    """
    mask = ~np.isnan(x) & ~np.isnan(y)
    x, y = x[mask], y[mask]
    return np.sum(x * y) / np.sum(x**2) if np.sum(x**2) != 0 else np.nan


def plot_forced_origin(x, y, slope, title):
    plt.scatter(x, y, alpha=0.6)
    plt.plot([0, max(x)], [0, slope * max(x)], "r", lw=2)
    plt.xlabel("ICOS temperature (°C)")
    plt.ylabel("ERA5 temperature (°C)")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def compute_mean_bias_error(x, y):
    """
    Description
    -----------
    Compute the Mean Bias Error (MBE) between two datasets.

    The Mean Bias Error quantifies the average difference between
    estimated ERA5 values (x) and observed ICOS values (y). A positive MBE
    indicates that x underestimates y on average (bias toward lower values),
    whereas a negative MBE indicates overestimation.

    The MBE is computed as:

        MBE = (1 / n) * Σ (y_i - x_i)

    where:
    - x_i are estimated values,
    - y_i are observed values,
    - n is the number of valid (non-NaN) observations.

    Parameters
    ----------
    x : array-like
        estimated values.
    y : array-like
        observed values.

    Returns
    -------
    float
        Mean Bias Error (MBE)
    """
    return np.nanmean(y - x)


#########################################
##                                     ##
##                                     ##
##     Scripts to compute metrics      ##
##                                     ##
##                                     ##
#########################################


def compute_monthly(
    df: pd.DataFrame,
    metric: str,
    hour: list[str] | str = "all",
    *,
    plot: bool = False,
):
    """
    Description
    -----------
    Compute monthly performance metrics between reference data (ICOS) and ERA5
    reanalysis for temperature (Ta) and dew point (Tdp).

    Parameters
    ----------
    csv_path : str
        Path to a CSV file containing the dataset
    metric : str
        Performance metric to compute (the same as used in OpenET:https://etdata.org/accuracy/#metrics):
         - sklearn.metrics.root_mean_squared_error : https://scikit-learn.org/stable/modules/generated/sklearn.metrics.root_mean_squared_error.html#sklearn.metrics.root_mean_squared_error
         - sklearn.metrics.r2_score : https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html#sklearn.metrics.r2_score
         - sklearn.metrics.mean_absolute_error : https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html
    hour : str | list[str]
        Hours used to compute monthly metrics
    Returns
    -------
    pd.DataFrame
        A dataframe with one row per month including:
        - month : YYYY-MM format
        - Ta_{metric} : metric between Ta_icos and Ta_era5
        - Ta_{metric}_rescaled : metric between Ta_icos and Ta_era5_rescaled
        - Tdp_{metric} : metric between Tdp_icos and Tdp_era5
        - Tdp_{metric}_rescaled : metric between Tdp_icos and Tdp_era5_rescaled
        - n_points : number of valid samples used for that month
    """
    if hour != "all":
        if isinstance(hour, str):
            hour = [hour]
        df = df[df["time"].dt.strftime("%H").isin(hour)]

    metrics = {
        "rmse": root_mean_squared_error,
        # "r2": r2_score,
        "mae": mean_absolute_error,
        "mbe": compute_mean_bias_error,
        "slope": slope_forced_origin,
    }

    if metric not in metrics:
        raise ValueError(f"Metric must be one of: {list(metrics.keys())}")

    f = metrics[metric]

    df["month"] = df["time"].dt.to_period("M")
    per_month = []

    for month, group in df.groupby("month"):
        group_valid = group.dropna(subset=["ta_icos", "tdp_icos"])
        if group_valid.empty:
            per_month.append(
                {
                    "month": str(month),
                    f"ta_{metric}": np.nan,
                    f"ta_{metric}_rescaled": np.nan,
                    f"tdp_{metric}": np.nan,
                    f"tdp_{metric}_rescaled": np.nan,
                    "n_points": 0,
                }
            )
            continue

        per_month.append(
            {
                "month": str(month),
                f"ta_{metric}": f(
                    group_valid["ta_era5"], group_valid["ta_icos"]
                ),
                f"ta_{metric}_rescaled": f(
                    group_valid["ta_era5_rescaled"], group_valid["ta_icos"]
                ),
                f"tdp_{metric}": f(
                    group_valid["tdp_era5"], group_valid["tdp_icos"]
                ),
                f"tdp_{metric}_rescaled": f(
                    group_valid["tdp_era5_rescaled"], group_valid["tdp_icos"]
                ),
                "n_points": len(group_valid),
            }
        )

        if plot and metric == "slope":
            # Ta
            plot_forced_origin(
                group_valid["ta_era5"],
                group_valid["ta_icos"],
                per_month[-1][f"ta_{metric}"],
                f"Ta - ERA5 - {month}",
            )
            plot_forced_origin(
                group_valid["ta_era5_rescaled"],
                group_valid["ta_icos"],
                per_month[-1][f"ta_{metric}_rescaled"],
                f"Ta - ERA5_rescaled - {month}",
            )
            # Tdp
            plot_forced_origin(
                group_valid["tdp_era5"],
                group_valid["tdp_icos"],
                per_month[-1][f"tdp_{metric}"],
                f"Tdp - ERA5 - {month}",
            )
            plot_forced_origin(
                group_valid["tdp_era5_rescaled"],
                group_valid["tdp_icos"],
                per_month[-1][f"tdp_{metric}_rescaled"],
                f"Tdp - ERA5_rescaled - {month}",
            )

    return pd.DataFrame(per_month)


def compute_monthly_metrics_per_station(
    stations_ts: dict[str, pd.DataFrame],
    metrics_list: str | list[str] | None = None,
    out_dir: str = "metrics_csvs",
    hour: list[str] | str = " ",
    *,
    plot: bool = False,
) -> dict[str, str]:
    """
    Compute multiple monthly metrics per station and save a CSV per station.

    Parameters
    ----------
    stations_ts : dict[str, pd.DataFrame]
        Dict of DataFrames from
        'etdataset.validation_temp.plot.build_station_timeseries()', keys
        are station names.
    metrics_list : str | list[str] | None
        Metrics to compute: "rmse", "r2", "mae", "mbe", "slope".
    out_dir : str
        Folder to save CSVs.
    hour : list[str] | str
        Optional hours to filter on (format 'HH').
    plot : bool
        If True and 'slope' is in metrics_list, plots regression forced through
        origin.

    Returns
    -------
    dict[str, str]
        Keys = station names, values = path to the CSV file.
    """

    os.makedirs(out_dir, exist_ok=True)

    # normalize metrics_list
    if metrics_list is None:
        metrics_list = ["rmse", "r2", "mae", "mbe", "slope"]
    elif isinstance(metrics_list, str):
        metrics_list = [metrics_list]

    if not metrics_list:
        raise ValueError("metrics_list cannot be empty")

    csv_paths: dict[str, str] = {}

    for station_name, df_station in stations_ts.items():
        # Initialize df_all_metrics with the first metric
        first_metric = metrics_list[0].lower().strip()
        df_all_metrics = compute_monthly(
            df_station, metric=first_metric, hour=hour, plot=plot
        )
        df_all_metrics = df_all_metrics.rename(
            columns={
                col: col if col not in {"month", "n_points"} else col
                for col in df_all_metrics.columns
            }
        )

        # Merge the rest of the metrics
        for metric in metrics_list[1:]:
            metric_clean = metric.lower().strip()
            df_metric = compute_monthly(
                df_station, metric=metric_clean, hour=hour, plot=plot
            )
            df_metric = df_metric.rename(
                columns={
                    col: col if col not in {"month", "n_points"} else col
                    for col in df_metric.columns
                }
            )

            df_all_metrics = pd.merge(
                df_all_metrics,
                df_metric,
                on=["month", "n_points"],
                how="outer",
            )

        # Save CSV
        csv_path = os.path.join(out_dir, f"{station_name}_monthly_metrics.csv")
        df_all_metrics.to_csv(csv_path, index=False)
        csv_paths[station_name] = csv_path

    return csv_paths


def compute_monthly_multi(
    df: pd.DataFrame,
    metrics: list[str] | str | None = None,
    hour: list[str] | str = "all",
):
    import pandas as pd

    if metrics is None:
        metrics = ["rmse", "mae", "r2", "slope", "mbe"]
    df = df.copy()
    if isinstance(metrics, str):
        metrics = [metrics]

    # filtrer les heures si besoin
    if hour != "all":
        if isinstance(hour, str):
            hour = [hour]
        df = df[df["time"].dt.strftime("%H").isin(hour)]

    # fonctions disponibles
    metric_funcs = {
        "rmse": root_mean_squared_error,
        # "r2": r2_score,
        "mae": mean_absolute_error,
        "mbe": compute_mean_bias_error,
        "slope": slope_forced_origin,
    }

    # vérifier que toutes les métriques sont valides
    for m in metrics:
        if m not in metric_funcs:
            raise ValueError(
                f"Metric {m} not supported. Choose from {list(metric_funcs.keys())}"  # noqa: E501
            )

    df["month"] = df["time"].dt.to_period("M")

    ta_methods = [
        c for c in df.columns if c.startswith("ta_") and c != "ta_icos"
    ]
    tdp_methods = [
        c for c in df.columns if c.startswith("tdp_") and c != "tdp_icos"
    ]

    results = []

    for month, group in df.groupby("month"):
        row: dict[str, float | int | str] = {"month": str(month)}

        # -------- TA --------
        ta_valid = group.dropna(subset=["ta_icos"])
        row["n_points_ta"] = len(ta_valid)

        for c in ta_methods:
            method = c.replace("ta_", "")
            daily_rmse = []
            daily_r2 = []
            y_true_all = []
            y_pred_all = []

            # calcul RMSE journalier pour min/max
            for _day, gday in ta_valid.groupby(ta_valid["time"].dt.date):
                y_true = gday["ta_icos"]
                y_pred = gday[c]

                # garder uniquement les indices valides dans les deux séries
                valid = y_true.notna() & y_pred.notna()
                y_true = y_true[valid]
                y_pred = y_pred[valid]

                if len(y_true) > 0:
                    rmse_day = root_mean_squared_error(y_true, y_pred)
                    r2_day = r2_score(y_true, y_pred)
                    daily_rmse.append(rmse_day)
                    daily_r2.append(r2_day)
                    y_true_all.append(y_true)
                    y_pred_all.append(y_pred)

            # concaténation pour métriques globales du mois
            if y_true_all:
                y_true_month = pd.concat(y_true_all)
                y_pred_month = pd.concat(y_pred_all)

                for m in metrics:
                    f = metric_funcs[m]
                    row[f"ta_{m}_{method}"] = f(y_true_month, y_pred_month)

                if "rmse" in metrics:
                    row[f"ta_rmse_{method}_min"] = min(daily_rmse)
                    row[f"ta_rmse_{method}_max"] = max(daily_rmse)
                if "r2" in metrics:
                    row[f"ta_r2_{method}_min"] = min(daily_r2)
                    row[f"ta_r2_{method}_max"] = max(daily_r2)

        # -------- TDP --------
        tdp_valid = group.dropna(subset=["tdp_icos"])
        row["n_points_tdp"] = len(tdp_valid)

        for c in tdp_methods:
            method = c.replace("tdp_", "")
            daily_rmse = []
            daily_r2 = []
            y_true_all = []
            y_pred_all = []

            for _day, gday in tdp_valid.groupby(tdp_valid["time"].dt.date):
                y_true = gday["tdp_icos"]
                y_pred = gday[c]

                valid = y_true.notna() & y_pred.notna()
                y_true = y_true[valid]
                y_pred = y_pred[valid]

                if len(y_true) > 0:
                    rmse_day = root_mean_squared_error(y_true, y_pred)
                    r2_day = r2_score(y_true, y_pred)
                    daily_rmse.append(rmse_day)
                    daily_r2.append(r2_day)
                    y_true_all.append(y_true)
                    y_pred_all.append(y_pred)

            if y_true_all:
                y_true_month = pd.concat(y_true_all)
                y_pred_month = pd.concat(y_pred_all)

                for m in metrics:
                    f = metric_funcs[m]
                    row[f"tdp_{m}_{method}"] = f(y_true_month, y_pred_month)

                if "rmse" in metrics:
                    row[f"tdp_rmse_{method}_min"] = min(daily_rmse)
                    row[f"tdp_rmse_{method}_max"] = max(daily_rmse)
                if "r2" in metrics:
                    row[f"tdp_r2_{method}_min"] = min(daily_r2)
                    row[f"tdp_r2_{method}_max"] = max(daily_r2)

        results.append(row)

    return pd.DataFrame(results)


def compute_monthly_metrics_per_station_multi(
    stations_ts: dict[str, pd.DataFrame],
    metrics_list: str | list[str] | None = None,
    out_dir: str = "metrics_csvs",
    hour: list[str] | str = "all",
    *,
    plot: bool = False,  # noqa: ARG001
) -> dict[str, str]:

    os.makedirs(out_dir, exist_ok=True)

    if metrics_list is None:
        metrics_list = ["rmse", "r2", "mae", "mbe", "slope"]
    elif isinstance(metrics_list, str):
        metrics_list = [metrics_list]

    csv_paths: dict[str, str] = {}

    for station_name, df_station in stations_ts.items():
        if df_station.empty:
            continue

        df_all_metrics: pd.DataFrame | None = None

        for metric in metrics_list:
            df_metric = compute_monthly_multi(
                df_station,
                metrics=metric,
                hour=hour,
            )

            if df_metric.empty:
                continue

            if df_all_metrics is None:
                df_all_metrics = df_metric

            else:
                # éviter duplication des n_points
                df_metric = df_metric.drop(
                    columns=["n_points_ta", "n_points_tdp"],
                    errors="ignore",
                )

                df_all_metrics = pd.merge(
                    df_all_metrics,
                    df_metric,
                    on="month",
                    how="outer",
                )

        if df_all_metrics is not None and not df_all_metrics.empty:
            csv_path = os.path.join(
                out_dir, f"{station_name}_monthly_metrics.csv"
            )

            df_all_metrics.to_csv(csv_path, index=False)

            csv_paths[station_name] = csv_path

    return csv_paths


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


def altitude_class(station, station_to_cluster):
    """
    Retourne la classe d'altitude basée sur le clustering KMeans.
    """

    cluster = station_to_cluster.get(station)

    if cluster is None:
        raise ValueError(f"Station {station} not found")

    # mapping cluster -> label
    mapping = {
        2: "low",
        0: "mid",
        1: "high",
    }

    return mapping[cluster]


# def altitude_class(alt):
#     if alt < 1000:
#         return "low"
#     if alt > 1000 and alt < 2000:
#         return "mid"
#     return "high"


def filter_hours(df: pd.DataFrame, hour):
    if hour == "all" or hour is None:
        return df

    if isinstance(hour, str):
        hour = [hour]

    df_filtered = df[df["time"].dt.strftime("%H").isin(hour)]

    # fallback si vide
    if df_filtered.empty:
        return df

    return df_filtered


METRICS = {
    "rmse": root_mean_squared_error,
    "r2": r2_score,
    "mae": mean_absolute_error,
    "mbe": compute_mean_bias_error,
    "slope": slope_forced_origin,
}


def compute_by_altitude(
    stations_ts: dict[str, pd.DataFrame],
    metrics=("rmse", "mae", "mbe"),
    hour="all",
):

    if isinstance(metrics, str):
        metrics = [metrics]

    rows = []
    enriched = []

    # ==========================
    # 1. ENRICHMENT
    # ==========================
    for station, df in stations_ts.items():
        cfg = get_stations_config(station)
        alt = cfg.elev

        if alt is None:
            continue

        df_ = df.copy()
        df_["station"] = station
        df_["category"] = altitude_class(
            station, station_to_cluster
        )  # altitude_class(alt)
        df_["time"] = pd.to_datetime(df_["time"])
        df_["month"] = df_["time"].dt.to_period("M")

        enriched.append(df_)

    if not enriched:
        return pd.DataFrame()

    df_all = pd.concat(enriched, ignore_index=True)
    df_all = filter_hours(df_all, hour)

    if df_all.empty:
        return pd.DataFrame()

    # ==========================
    # 2. DETECT METHODS
    # ==========================
    ta_methods = [
        c for c in df_all.columns if c.startswith("ta_") and c != "ta_icos"
    ]
    tdp_methods = [
        c for c in df_all.columns if c.startswith("tdp_") and c != "tdp_icos"
    ]

    # ==========================
    # 3. GROUPING
    # ==========================
    for (category, month), g in df_all.groupby(["category", "month"]):
        row = {
            "category": category,
            "month": str(month),
            "n_points_ta": g["ta_icos"].notna().sum(),
            "n_points_tdp": g["tdp_icos"].notna().sum(),
        }

        # ======================
        # TA (pondéré)
        # ======================
        for col in ta_methods:
            method = col.replace("ta_", "")

            mask = g["ta_icos"].notna() & g[col].notna()
            if mask.sum() == 0:
                continue

            y_true = g.loc[mask, "ta_icos"]
            y_pred = g.loc[mask, col]

            errors = y_pred - y_true

            weights = np.ones_like(errors)  # pondération uniforme (modifiable)

            for m in metrics:
                if m == "mae":
                    val = np.average(np.abs(errors), weights=weights)
                elif m == "mbe":
                    val = np.average(errors, weights=weights)
                elif m == "rmse":
                    val = np.sqrt(np.average(errors**2, weights=weights))
                else:
                    val = METRICS[m](y_true, y_pred)

                row[f"ta_{m}_{method}"] = val

        # ======================
        # TDP (pondéré)
        # ======================
        for col in tdp_methods:
            method = col.replace("tdp_", "")

            mask = g["tdp_icos"].notna() & g[col].notna()
            if mask.sum() == 0:
                continue

            y_true = g.loc[mask, "tdp_icos"]
            y_pred = g.loc[mask, col]

            errors = y_pred - y_true
            weights = np.ones_like(errors)

            for m in metrics:
                if m == "mae":
                    val = np.average(np.abs(errors), weights=weights)
                elif m == "mbe":
                    val = np.average(errors, weights=weights)
                elif m == "rmse":
                    val = np.sqrt(np.average(errors**2, weights=weights))
                else:
                    val = METRICS[m](y_true, y_pred)

                row[f"tdp_{m}_{method}"] = val

        rows.append(row)

    return pd.DataFrame(rows)


def compute_metrics(
    measured: npt.ArrayLike, estimated: npt.ArrayLike
) -> tuple[float, float, float, float, float]:
    """
    Compute slope, mbe, mae, rmse, r2
    """

    idx = np.isfinite(measured) & np.isfinite(estimated)
    if idx.sum() < 2:
        return (np.nan, np.nan, np.nan, np.nan, np.nan)

    slope, _ = np.polyfit(measured[idx], estimated[idx], 1)
    mbe = np.nanmean(estimated - measured)
    mae = mean_absolute_error(measured[idx], estimated[idx])
    rmse = np.sqrt(mean_squared_error(measured[idx], estimated[idx]))
    r2 = r2_score(measured[idx], estimated[idx])
    return (slope, mbe, mae, rmse, r2)


def compute_monthly_metrics_stations(
    stations_data: dict[str, pd.DataFrame],
    output_dir: str = "monthly_metrics",
) -> dict[str, pd.DataFrame]:
    """
    Compute monthly metrics for all methods and all stations.

    Parameters
    ----------
    stations_data : dict[str, pd.DataFrame]
        Dictionary of station DataFrames.

    output_dir : str
        Directory where CSV files are saved.

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary containing metrics DataFrames for each station.
    """

    ta_methods = [
        "ta_era5",
        "ta_base",
        "ta_method_1",
        "ta_method_2",
        "ta_method_3",
        "ta_method_4",
        "ta_method_5",
        "ta_method_6",
    ]

    tdp_methods = [
        "tdp_era5",
        "tdp_base",
        "tdp_method_1",
        "tdp_method_2",
        "tdp_method_3",
        "tdp_method_4",
        "tdp_method_5",
        "tdp_method_6",
    ]

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    all_results = {}

    for station, df in stations_data.items():
        df_ = df.copy()

        # datetime
        df_["time"] = pd.to_datetime(df_["time"])

        # monthly grouping
        df_["month"] = df_["time"].dt.to_period("M")

        results = []

        for month, group in df_.groupby("month"):
            # -----------------------------
            # Air temperature
            # -----------------------------
            for method in ta_methods:
                if method not in group.columns:
                    continue

                idx = np.isfinite(group["ta_icos"]) & np.isfinite(group[method])

                slope, mbe, mae, rmse, r2 = compute_metrics(
                    measured=group["ta_icos"].values,
                    estimated=group[method].values,
                )

                results.append(
                    {
                        "month": str(month),
                        "variable": "ta",
                        "method": method,
                        "slope": slope,
                        "mbe": mbe,
                        "mae": mae,
                        "rmse": rmse,
                        "r2": r2,
                        "n": idx.sum(),
                    }
                )

            # -----------------------------
            # Dew point temperature
            # -----------------------------
            for method in tdp_methods:
                if method not in group.columns:
                    continue

                idx = np.isfinite(group["tdp_icos"]) & np.isfinite(
                    group[method]
                )

                slope, mbe, mae, rmse, r2 = compute_metrics(
                    measured=group["tdp_icos"].values,
                    estimated=group[method].values,
                )

                results.append(
                    {
                        "month": str(month),
                        "variable": "tdp",
                        "method": method,
                        "slope": slope,
                        "mbe": mbe,
                        "mae": mae,
                        "rmse": rmse,
                        "r2": r2,
                        "n": idx.sum(),
                    }
                )

        metrics_df = pd.DataFrame(results)

        # save csv
        csv_file = output_path / f"{station}_monthly_metrics.csv"
        metrics_df.to_csv(csv_file, index=False)

        all_results[station] = metrics_df

        logger.info(f"Saved: {csv_file}")

    return all_results


def compute_global_monthly_metrics(
    stations_data: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Compute monthly metrics across ALL stations
    for all available methods.

    Returns
    -------
    pd.DataFrame
        Global monthly metrics.
    """

    all_dfs = []

    # -----------------------------------------
    # concat toutes les stations
    # -----------------------------------------
    for station, df in stations_data.items():
        tmp = df.copy()

        tmp["station"] = station
        tmp["time"] = pd.to_datetime(tmp["time"])
        tmp["month"] = tmp["time"].dt.to_period("M")

        all_dfs.append(tmp)

    df_all = pd.concat(all_dfs, ignore_index=True)

    # -----------------------------------------
    # méthodes disponibles
    # -----------------------------------------
    ta_methods = [
        c for c in df_all.columns if c.startswith("ta_") and c != "ta_icos"
    ]

    tdp_methods = [
        c for c in df_all.columns if c.startswith("tdp_") and c != "tdp_icos"
    ]

    rows = []

    # -----------------------------------------
    # boucle sur les mois
    # -----------------------------------------
    for month, group in df_all.groupby("month"):
        # -----------------------------
        # TA metrics
        # -----------------------------
        for method in ta_methods:
            idx = group["ta_icos"].notna() & group[method].notna()

            if idx.sum() < 2:
                continue

            slope, mbe, mae, rmse, r2 = compute_metrics(
                measured=group.loc[idx, "ta_icos"],
                estimated=group.loc[idx, method],
            )

            rows.append(
                {
                    "month": str(month),
                    "variable": "ta",
                    "method": method,
                    "n": idx.sum(),
                    "slope": slope,
                    "mbe": mbe,
                    "mae": mae,
                    "rmse": rmse,
                    "r2": r2,
                }
            )

        # -----------------------------
        # TDP metrics
        # -----------------------------
        for method in tdp_methods:
            idx = group["tdp_icos"].notna() & group[method].notna()

            if idx.sum() < 2:
                continue

            slope, mbe, mae, rmse, r2 = compute_metrics(
                measured=group.loc[idx, "tdp_icos"],
                estimated=group.loc[idx, method],
            )

            rows.append(
                {
                    "month": str(month),
                    "variable": "tdp",
                    "method": method,
                    "n": idx.sum(),
                    "slope": slope,
                    "mbe": mbe,
                    "mae": mae,
                    "rmse": rmse,
                    "r2": r2,
                }
            )

    return pd.DataFrame(rows)

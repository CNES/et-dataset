#
# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales /
#            Université Paul Sabatier (UT3)
#


"""
Functions for metrics
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)

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
        "r2": r2_score,
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


"""
def compute_monthly_multi(
    df: pd.DataFrame,
    metric: str,
    hour: list[str] | str = "all",
    *,
    plot: bool = False,
):

    df = df.copy()

    if hour != "all":
        if isinstance(hour, str):
            hour = [hour]
        df = df[df["time"].dt.strftime("%H").isin(hour)]

    metrics = {
        "rmse": root_mean_squared_error,
        "r2": r2_score,
        "mae": mean_absolute_error,
        "mbe": compute_mean_bias_error,
        "slope": slope_forced_origin,
    }

    if metric not in metrics:
        raise ValueError(f"Metric must be one of: {list(metrics.keys())}")

    f = metrics[metric]

    df["month"] = df["time"].dt.to_period("M")

    per_month = []

    ta_methods = [
        c for c in df.columns if c.startswith("ta_") and c not in ["ta_icos"]
    ]

    tdp_methods = [
        c for c in df.columns if c.startswith("tdp_") and c not in ["tdp_icos"]
    ]

    for month, group in df.groupby("month"):
        row: dict[str, float | int | str] = {"month": str(month)}

        # ---------------- TA ----------------

        ta_valid = group.dropna(subset=["ta_icos"])
        row["n_points_ta"] = len(ta_valid)

        for c in ta_methods:
            valid = ta_valid[[c, "ta_icos"]].dropna()

            if valid.empty:
                continue

            val = f(valid["ta_icos"], valid[c])

            method = c.replace("ta_", "")

            row[f"ta_{metric}_{method}"] = val
            row[f"ta_{metric}_{method}_min"] = valid[c].min()
            row[f"ta_{metric}_{method}_max"] = valid[c].max()

        # ---------------- TDP ----------------

        tdp_valid = group.dropna(subset=["tdp_icos"])
        row["n_points_tdp"] = len(tdp_valid)

        for c in tdp_methods:
            valid = tdp_valid[[c, "tdp_icos"]].dropna()

            if valid.empty:
                continue

            val = f(valid["tdp_icos"], valid[c])

            method = c.replace("tdp_", "")

            row[f"tdp_{metric}_{method}"] = val
            row[f"tdp_{metric}_{method}_min"] = valid[c].min()
            row[f"tdp_{metric}_{method}_max"] = valid[c].max()

        per_month.append(row)

    return pd.DataFrame(per_month)
"""


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
        "r2": r2_score,
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

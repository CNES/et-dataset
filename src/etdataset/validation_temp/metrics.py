#
# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales /
#            Université Paul Sabatier (UT3)
#


#########################################################
##                                                     ##
##                                                     ##
##                    METRICS                          ##
##                                                     ##
##                                                     ##
#########################################################

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)


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


def compute_monthly(csv_path: str, metric: str, *, plot: bool = False):
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
    Returns
    -------
    pd.DataFrame
        A dataframe with one row per month including:
        - month : YYYY-MM format
        - Ta_{metric} : metric between Ta_icos and Ta_era5
        - Ta_{metric}_proj : metric between Ta_icos and Ta_era5_proj
        - Tdp_{metric} : metric between Tdp_icos and Tdp_era5
        - Tdp_{metric}_proj : metric between Tdp_icos and Tdp_era5_proj
        - n_points : number of valid samples used for that month
    """
    df = pd.read_csv(csv_path, parse_dates=["time"])

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
        group_valid = group.dropna(subset=["Ta_icos", "Tdp_icos"])
        if group_valid.empty:
            per_month.append(
                {
                    "month": str(month),
                    f"Ta_{metric}": np.nan,
                    f"Ta_{metric}_proj": np.nan,
                    f"Tdp_{metric}": np.nan,
                    f"Tdp_{metric}_proj": np.nan,
                }
            )
            continue
        per_month.append(
            {
                "month": str(month),
                f"Ta_{metric}": f(
                    group_valid["Ta_era5"], group_valid["Ta_icos"]
                ),
                f"Ta_{metric}_proj": f(
                    group_valid["Ta_era5_proj"], group_valid["Ta_icos"]
                ),
                f"Tdp_{metric}": f(
                    group_valid["Tdp_era5"], group_valid["Tdp_icos"]
                ),
                f"Tdp_{metric}_proj": f(
                    group_valid["Tdp_era5_proj"], group_valid["Tdp_icos"]
                ),
                "n_points": len(group_valid),
            }
        )
        if plot and metric == "slope":
            plot_forced_origin(
                group_valid["Ta_era5"],
                group_valid["Ta_icos"],
                per_month[-1][f"Ta_{metric}"],
                f"Ta - ERA5 - {month}",
            )
            plot_forced_origin(
                group_valid["Ta_era5_proj"],
                group_valid["Ta_icos"],
                per_month[-1][f"Ta_{metric}_proj"],
                f"Ta - ERA5_resampled - {month}",
            )
            plot_forced_origin(
                group_valid["Tdp_era5"],
                group_valid["Tdp_icos"],
                per_month[-1][f"Tdp_{metric}"],
                f"Tdp - ERA5 - {month}",
            )
            plot_forced_origin(
                group_valid["Tdp_era5_proj"],
                group_valid["Tdp_icos"],
                per_month[-1][f"Tdp_{metric}_proj"],
                f"Tdp - ERA5_resampled - {month}",
            )
    return pd.DataFrame(per_month)


def compute_monthly_lr(csv_path: str, metric: str):
    """
    Description
    -----------
    Compute the monthly performance metrics between the mean theoretical value
    of the lapse rate and the lapse rates of TA and TDP
    Parameters
    ----------
    csv_path : str
        Path to a CSV file containing the dataset
    metric : str
        Performance metric to compute (the same as used in OpenET:https://etdata.org/accuracy/#metrics):
         - sklearn.metrics.root_mean_squared_error : https://scikit-learn.org/stable/modules/generated/sklearn.metrics.root_mean_squared_error.html#sklearn.metrics.root_mean_squared_error
         - sklearn.metrics.r2_score : https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html#sklearn.metrics.r2_score
         - sklearn.metrics.mean_absolute_error : https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html
    Returns
    -------
    pd.DataFrame
        A dataframe with one row per month including:
        - month : YYYY-MM format
        - Ta_{metric} : metric between Ta_icos and Ta_era5
        - Ta_{metric}_proj : metric between Ta_icos and Ta_era5_proj
        - Tdp_{metric} : metric between Tdp_icos and Tdp_era5
        - Tdp_{metric}_proj : metric between Tdp_icos and Tdp_era5_proj
        - n_points : number of valid samples used for that month
    """
    df = pd.read_csv(csv_path, parse_dates=["time"])

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
        group_valid = group.dropna(subset=["lapse_rate_ta", "lapse_rate_tdp"])
        if group_valid.empty:
            per_month.append(
                {
                    "month": str(month),
                    f"lr_ta_{metric}": np.nan,
                    f"lr_tdp_{metric}": np.nan,
                }
            )
            continue
        size = group_valid["lapse_rate_ta"].shape[0]
        array_th = np.full(size, -0.0065)
        per_month.append(
            {
                "month": str(month),
                f"lr_ta_{metric}": f(array_th, group_valid["lapse_rate_ta"]),
                f"lr_tdp_{metric}": f(array_th, group_valid["lapse_rate_tdp"]),
                "n_points": len(group_valid),
            }
        )
    return pd.DataFrame(per_month)

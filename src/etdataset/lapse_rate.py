# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales

"""
Module to compute temperature lapse rate
"""

import datetime as dt

import numpy as np
import numpy.typing as npt
import xarray as xr

from etdataset.logging import LoggerManager

logger = LoggerManager.get_logger(__name__)
G_CST = 9.80665

# MONTHLY LAPSE RATE

# Lapse rate values from "A Meteorological Distribution System for
#    High-Resolution Terrestrial Modeling (MicroMet)", Journal of
#    Hydrometeorology, by G.E. Liston, K. Elder
LAPSE_RATE_BY_MONTH = {
    1: -0.0044,
    2: -0.0059,
    3: -0.0071,
    4: -0.0078,
    5: -0.0081,
    6: -0.0082,
    7: -0.0081,
    8: -0.0081,
    9: -0.0077,
    10: -0.0068,
    11: -0.0065,
    12: -0.0047,
}
# Vapor pressure values from "A Meteorological Distribution System for
#    High-Resolution Terrestrial Modeling (MicroMet)", Journal of
#    Hydrometeorology, by G.E. Liston, K. Elder
VAPOR_PRESSURE_BY_MONTH = {
    1: 0.00041,
    2: 0.00042,
    3: 0.0004,
    4: 0.00039,
    5: 0.00038,
    6: 0.00036,
    7: 0.00033,
    8: 0.00033,
    9: 0.00036,
    10: 0.00037,
    11: 0.00040,
    12: 0.00040,
}


# Lapse rate Monthly
def get_lapse_rate_monthly(dt: dt.date) -> float:
    return LAPSE_RATE_BY_MONTH[dt.month]


# Calculation of dewpoint lapse rate by using Vapor pressure Monthly
def get_vapor_pressure_monthly(dt: dt.date) -> float:
    return VAPOR_PRESSURE_BY_MONTH[dt.month]


def compute_dewpoint_lr(
    coeff: float | npt.ArrayLike, b: float = 17.502, c: float = 240.97
) -> float | npt.NDArray[np.float64]:
    """
    Compute lapse rate for dewpoint temperature

    Parameters
    ----------
    coeff : float | np.arraylike
        Coefficielt
    b : float
        Parameter
    c : float
        Parameter

    Returns
    -------
    lr: float | npt.NDArray[np.float64]
        Dewpoint lapse rate
    """
    arr = np.asarray(coeff, dtype=float)
    result = -arr * c / b

    if arr.ndim == 0:
        return float(result)
    return result


def compute_lapse_rate_from_2_levels_roi(
    ds: xr.Dataset,
    temperature_var: str = "ta",
    height_var: str = "z",
    level_dim: str = "pressure_level",
) -> xr.DataArray:
    """
    Compute time-dependent lapse rate between two pressure levels

    The lapse rate is computed as:
        (T_high - T_low) / (Z_high - Z_low)

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing exactly two pressure levels
    temperature_var : str
        Name of temperature variable
    height_var : str
        Name of height variable
    level_dim : str
        Name of pressure level dimension

    Returns
    -------
    xr.DataArray
        Lapse rate as a function of time.
    """

    if ds.sizes[level_dim] != 2:
        raise ValueError("Dataset must contain exactly two pressure levels.")

    t = ds[temperature_var]
    z = ds[height_var] / G_CST

    # sort by height
    order = z.mean(("latitude", "longitude")).argsort()

    t = t.isel({level_dim: order})
    z = z.isel({level_dim: order})

    # level index 0 = low, 1 = high
    t_low = t.isel({level_dim: 0})
    t_high = t.isel({level_dim: 1})

    z_low = z.isel({level_dim: 0})
    z_high = z.isel({level_dim: 1})

    lapse_rate = (t_high - t_low) / (z_high - z_low)

    lapse_rate.name = "lapse_rate"
    lapse_rate.attrs["units"] = "°C m-1"
    lapse_rate.attrs["description"] = (
        "Temperature lapse rate between two pressure levels"
    )

    return lapse_rate

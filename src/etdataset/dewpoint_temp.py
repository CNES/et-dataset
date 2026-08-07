# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales

"""
Module for dewpoint temperature
"""

import numpy as np
import numpy.typing as npt
import xarray as xr

from etdataset.logging import LoggerManager
from etdataset.utils import (
    celsius_to_kelvin,
    kelvin_to_celsius,
)

logger = LoggerManager.get_logger(__name__)
G_CST = 9.80665


def compute_dewpoint_temp(
    ta: npt.ArrayLike, rh: npt.ArrayLike, f: float = 243.04, d: float = 17.625
) -> npt.NDArray:
    """
    Description
    -----------
    Compute dew point temperature Tp from air temperature Ta (°C) and
    relative humidity RH (%):

            RH = 100 * exp[d*Td/(Td+f)-d*Ta/(Ta+f)]

            it gives:

            Td = f*(I + d*Ta/(Ta+f))/(d-I-d*Ta/(Ta+f))

            with I = ln(RH/100)

    from "The Relationship between Relative Humidity and the Dewpoint
    Temperature in Moist Air: A Simple Conversion and Applications"
    by Mark G. Lawrence

    Parameters
    -----------
    ta : ntp.ArrayLike
        Air temperature from ICOS
    rh : ntp.ArrayLike
        Relative humidity from ICOS

    Return
    -----------
    tp : ntp.NDArray
        Dew point temperature
    """
    ta = np.array(ta)
    rh = np.array(rh)
    L = np.log(rh / 100)
    gamma = L + d * ta / (ta + f)

    num = f * gamma
    den = d - gamma
    tp = num / den

    return tp


def add_dewpoint(
    ds: xr.Dataset,
    temp_var: str = "t",
    rh_var: str = "r",
    output_var: str = "td",
) -> xr.Dataset:
    """
    Description
    -----------
    Compute the dew point temperature from temperature and
    relative humidity and add it to an xarray Dataset

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset containing temperature and relative humidity
    temp_var : str, default "t"
        Name of the air temperature variable (expected in Kelvin)
    rh_var : str, default "r"
        Name of the relative humidity variable (in %)
    output_var : str, default "tdp"
        Name of the dew point variable to be added to the dataset

    Returns
    -------
    xr.Dataset
        A new dataset with the dew point temperature added
        (in Kelvin).
    """

    ds = ds.copy()
    t = ds[temp_var]
    rh = ds[rh_var]
    rh = xr.where(rh <= 0, np.nan, rh)
    # Convert temperature to Celsius
    t_c = kelvin_to_celsius(t)

    # Compute dew point in Celsius
    td_c = compute_dewpoint_temp(t_c, rh)

    # Convert back to Kelvin
    td_k = celsius_to_kelvin(td_c)

    ds[output_var] = (t.dims, td_k)
    ds[output_var].attrs.update(
        {"units": "K", "long_name": "Dew point temperature"}
    )
    return ds


def get_saturation_vapor_pressure(
    t: npt.ArrayLike, a: float = 611.21, b: float = 17.502, c: float = 240.97
) -> npt.NDArray:
    """
    Description
    -----------
    Compute saturation vapor pressure (Pa) at a temperature t (°C):

                es = a * exp(b*t/(c+t))

    from "A Meteorological Distribution System for High-Resolution
    Terrestrial Modeling (MicroMet)", Journal of Hydrometeorology,
    by G.E. Liston, K. Elder

    - For water : a = 611.21 Pa, b = 17.502, c = 240.97°C
    - For ice : a = 611.15 Pa, b = 22.452, c = 272.55°C

    Parameters
    ----------
    t : npt.ArrayLike
        Temperature (in °C)
    a : float
    b : float
    c : float

    Return
    -----------
    npt.NDArray
    Saturation vapor pressure at temperature t
    """
    return a * np.exp(b * np.array(t) / (c + np.array(t)))


def compute_vapor_pressure(rh: npt.ArrayLike, es: npt.ArrayLike) -> npt.NDArray:
    """
    Description
    -----------
    Compute actual vapor pressure (Pa):

                RH = 100 * e / es
    it gives:
                e = RH * es / 100

    from "A Meteorological Distribution System for High-Resolution
    Terrestrial Modeling (MicroMet)", Journal of Hydrometeorology,
    by G.E. Liston, K. Elder

    Parameters
    ----------
    rh : npt.ArrayLike
        Relative humidity (in %)
    es : npt.ArrayLike
        Saturation vapor pressure (Pa)

    Return
    -----------
    npt.NDArray
    Vapor pressure (Pa)
    """
    return np.array(rh) * np.array(es) / 100


def compute_dewpoint_temp_from_e(
    e: npt.ArrayLike, a: float = 611.21, b: float = 17.502, c: float = 240.97
):
    """
    Description
    -----------
    Compute dewpoint temperature from vapor pressure (e):

                td = c * ln(e/a)/(b - ln(e/a))

    from "A Meteorological Distribution System for High-Resolution
    Terrestrial Modeling (MicroMet)", Journal of Hydrometeorology,
    by G.E. Liston, K. Elder

    - For water : a = 611.21 Pa, b = 17.502, c = 240.97°C
    - For ice : a = 611.15 Pa, b = 22.452, c = 272.55°C

    Parameters
    ----------
    e: npt.ArrayLike
        Vapor pressure (Pa)
    a : float
    b : float
    c : float

    Return
    -----------
    npt.NDArray
    Dewpoint temperature (in °C)
    """
    num = c * np.log(np.array(e) / a)
    den = b - np.log(np.array(e) / a)
    return num / den

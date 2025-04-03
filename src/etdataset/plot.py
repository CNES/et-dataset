#
# Copyright: (c) 2023 CESBIO / Centre National d'Etudes Spatiales /
#             Université Paul Sabatier (UT3)
#
"""
Functions for plotting
"""

import math

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import ListedColormap


def rescale(data: np.ndarray, qmin: float, qmax: float) -> np.ndarray:
    arr = np.clip(data, qmin, qmax)
    min_value = np.nanmin(arr)
    max_value = np.nanmax(arr)
    arr = (arr - min_value) / (max_value - min_value)
    return arr


def plot_images(arr: xr.Dataset, title: str = "Dataset", outfname=None):
    """
    Plot dataset
    """
    # Compute number of plot
    bands = list(arr.data_vars)
    nb_plots = 0
    if set(["red", "green", "blue"]).intersection(set(bands)) == set(  # noqa
        ["red", "green", "blue"]
    ):
        nb_plots += 1
    for band in bands:
        if band in [
            "lst",
            "emis",
            "cloud",
            "water",
            "qa",
            "albedo",
            "ndvi",
            "lai",
            "rsd",
            "rld",
            "fdiff",
        ]:
            nb_plots += 1
    ncol = 2
    nrow = math.ceil(nb_plots / 2)

    fig, axes = plt.subplots(
        ncols=ncol,
        nrows=nrow,
        figsize=(7 * ncol, 5 * nrow),
        constrained_layout=True,
    )

    # RGB
    icol = 0
    irow = 0
    if set(["red", "green", "blue"]).intersection(set(bands)) == set(  # noqa
        ["red", "green", "blue"]
    ):
        rgb = xr.DataArray(
            np.dstack((arr.red.data, arr.green.data, arr.blue.data)),
            arr.coords.assign(band=["r", "g", "b"]),
            {"y": arr.sizes["y"], "x": arr.sizes["x"], "band": 3},
        )
        xr.plot.imshow(  # type: ignore
            darray=rgb,
            ax=axes[irow, icol],
            rgb="band",
            vmin=rgb.quantile(0.01),
            vmax=rgb.quantile(0.99),
        )
        axes[irow, icol].set_title("RGB")
        icol += 1
        if icol == 2:
            icol = 0
            irow += 1

    # LST
    if "lst" in bands:
        arr.lst.plot(
            ax=axes[irow, icol],
            vmin=arr.lst.quantile(0.01),
            vmax=arr.lst.quantile(0.99),
        )
        axes[irow, icol].set_title("LST")
        axes[irow, icol].grid(True)
        icol += 1
        if icol == 2:
            icol = 0
            irow += 1

    # Emissivity
    if "emis" in bands:
        arr.emis.plot(
            ax=axes[irow, icol],
            vmin=arr.emis.quantile(0.01),
            vmax=arr.emis.quantile(0.99),
        )
        axes[irow, icol].set_title("Emissivity")
        axes[irow, icol].grid(True)
        icol += 1
        if icol == 2:
            icol = 0
            irow += 1

    # Albedo
    if "albedo" in bands:
        arr.albedo.plot(
            ax=axes[irow, icol],
            vmin=arr.albedo.quantile(0.01),
            vmax=arr.albedo.quantile(0.99),
        )
        axes[irow, icol].set_title("Albedo")
        axes[irow, icol].grid(True)
        icol += 1
        if icol == 2:
            icol = 0
            irow += 1

    # Water
    if "water" in bands:
        valid_cMap = ListedColormap(
            [(0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 1.0, 1.0)]
        )
        arr.water.plot(ax=axes[irow, icol], cmap=valid_cMap, add_colorbar=False)
        axes[irow, icol].set_title("Water")
        axes[irow, icol].grid(True)
        icol += 1
        if icol == 2:
            icol = 0
            irow += 1

    # Cloud
    if "cloud" in bands:
        valid_cMap = ListedColormap(
            [(0.0, 0.0, 0.0, 0.0), (0.5, 0.5, 0.5, 1.0)]
        )
        arr.cloud.plot(ax=axes[irow, icol], cmap=valid_cMap, add_colorbar=False)
        axes[irow, icol].set_title("Cloud")
        axes[irow, icol].grid(True)
        icol += 1
        if icol == 2:
            icol = 0
            irow += 1
    # QA
    if "qa" in bands:
        valid_cMap = ListedColormap([(0.0, 0.0, 0.0, 0.0)])
        if arr.qa.sum() == 0:
            valid_cMap = ListedColormap(
                [(1.0, 0.0, 0.0, 1.0), (0.0, 0.0, 0.0, 0.0)]
            )
        arr.qa.plot(ax=axes[irow, icol], cmap=valid_cMap, add_colorbar=False)
        axes[irow, icol].set_title("QA")
        axes[irow, icol].grid(True)
        icol += 1
        if icol == 2:
            icol = 0
            irow += 1

    # NDVI
    if "ndvi" in bands:
        arr.ndvi.plot(
            ax=axes[irow, icol],
            vmin=arr.ndvi.quantile(0.01),
            vmax=arr.ndvi.quantile(0.99),
        )
        axes[irow, icol].set_title("NDVI")
        axes[irow, icol].grid(True)
        icol += 1
        if icol == 2:
            icol = 0
            irow += 1

    # LAI
    if "lai" in bands:
        arr.lai.plot(
            ax=axes[irow, icol],
            vmin=arr.lai.quantile(0.01),
            vmax=arr.lai.quantile(0.99),
        )
        axes[irow, icol].set_title("LAI")
        axes[irow, icol].grid(True)
        icol += 1
        if icol == 2:
            icol = 0
            irow += 1

    # Downwelling shortwave radiation
    if "rsd" in bands:
        arr.rsd.plot(
            ax=axes[irow, icol],
            vmin=arr.rsd.quantile(0.01),
            vmax=arr.rsd.quantile(0.99),
        )
        axes[irow, icol].set_title("Downwelling shortwave radiation")
        axes[irow, icol].grid(True)
        icol += 1
        if icol == 2:
            icol = 0
            irow += 1

    # Diffuse fraction
    if "fdiff" in bands:
        arr.fdiff.plot(
            ax=axes[irow, icol],
            vmin=arr.fdiff.quantile(0.01),
            vmax=arr.fdiff.quantile(0.99),
        )
        axes[irow, icol].set_title("Diffuse fraction")
        axes[irow, icol].grid(True)
        icol += 1
        if icol == 2:
            icol = 0
            irow += 1

    # Downwelling longwave radiation
    if "rld" in bands:
        arr.rld.plot(
            ax=axes[irow, icol],
            vmin=arr.rld.quantile(0.01),
            vmax=arr.rld.quantile(0.99),
        )
        axes[irow, icol].set_title("Downwelling longwave radiation")
        axes[irow, icol].grid(True)
        icol += 1
        if icol == 2:
            icol = 0
            irow += 1

    fig.suptitle(f"{title}")
    if outfname is not None:
        fig.savefig(outfname, format="png", bbox_inches="tight")

    return fig, axes

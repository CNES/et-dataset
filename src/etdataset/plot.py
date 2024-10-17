#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright: (c) 2023 CESBIO / Centre National d'Etudes Spatiales / Université Paul Sabatier (UT3)
#
"""
Functions for plotting
"""

import math

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def rescale(data: np.array, qmin: float, qmax: float) -> np.array:
    arr = np.clip(data, qmin, qmax)
    min = np.nanmin(arr)
    max = np.nanmax(arr)
    arr = (arr - min) / (max - min)
    return arr


def plot_images(arr: xr.Dataset, title: str = "Dataset", outfname=None):
    """
    Plot dataset
    """
    # Compute number of plot
    bands = [i for i in arr.data_vars]
    nb_plots = 0
    if set(["red", "green", "blue"]).intersection(set(bands)) == set(
        ["red", "green", "blue"]
    ):
        nb_plots += 1
    for band in bands:
        if band in ["lst", "emis", "albedo", "ndvi", "lai", "rg", "ra"]:
            nb_plots += 1
    ncol = 2
    nrow = math.ceil(nb_plots / 2)

    fig, axes = plt.subplots(
        ncols=ncol, nrows=nrow, figsize=(7 * ncol, 5 * nrow), constrained_layout=True
    )

    # RGB
    icol = 0
    irow = 0
    if set(["red", "green", "blue"]).intersection(set(bands)) == set(
        ["red", "green", "blue"]
    ):
        rgb = xr.DataArray(
            np.dstack((arr.red.data, arr.green.data, arr.blue.data)),
            arr.coords.assign(band=["r", "g", "b"]),
            {"y": arr.sizes["y"], "x": arr.sizes["x"], "band": 3},
        )
        rgb.plot.imshow(
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

    # Rg
    if "rg" in bands:
        arr.rg.plot(
            ax=axes[irow, icol],
            vmin=arr.rg.quantile(0.01),
            vmax=arr.rg.quantile(0.99),
        )
        axes[irow, icol].set_title("Rg")
        axes[irow, icol].grid(True)
        icol += 1
        if icol == 2:
            icol = 0
            irow += 1

    # Ra
    if "ra" in bands:
        arr.ra.plot(
            ax=axes[irow, icol],
            vmin=arr.ra.quantile(0.01),
            vmax=arr.ra.quantile(0.99),
        )
        axes[irow, icol].set_title("Ra")
        axes[irow, icol].grid(True)
        icol += 1
        if icol == 2:
            icol = 0
            irow += 1

    fig.suptitle(f"{title}")
    if outfname is not None:
        fig.savefig(outfname, format="png", bbox_inches="tight")

    return fig, axes

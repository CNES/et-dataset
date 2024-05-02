#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright: (c) 2023 CESBIO / Centre National d'Etudes Spatiales / Université Paul Sabatier (UT3)
#
"""
Functions for plotting
"""

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
    fig, axes = plt.subplots(
        ncols=2, nrows=3, figsize=(15, 15), constrained_layout=True
    )

    # RGB
    # Create RGB dataset
    rgb = xr.DataArray(
        np.dstack((arr.red.data, arr.green.data, arr.blue.data)),
        arr.coords.assign(band=["r", "g", "b"]),
        {"y": arr.sizes["y"], "x": arr.sizes["x"], "band": 3},
    )
    rgb.plot.imshow(
        ax=axes[0, 0], rgb="band", vmin=rgb.quantile(0.01), vmax=rgb.quantile(0.99)
    )
    axes[0, 0].set_title("RGB")

    # LST
    arr.lst.plot(
        ax=axes[0, 1], vmin=arr.lst.quantile(0.01), vmax=arr.lst.quantile(0.99)
    )
    axes[0, 1].set_title("LST")
    axes[0, 1].grid(True)

    # Emissivity
    arr.emis.plot(
        ax=axes[1, 0], vmin=arr.emis.quantile(0.01), vmax=arr.emis.quantile(0.99)
    )
    axes[1, 0].set_title("Emissivity")
    axes[1, 0].grid(True)

    # Albedo
    arr.albedo.plot(
        ax=axes[1, 1], vmin=arr.albedo.quantile(0.01), vmax=arr.albedo.quantile(0.99)
    )
    axes[1, 1].set_title("Albedo")
    axes[1, 1].grid(True)

    # NDVI
    arr.ndvi.plot(
        ax=axes[2, 0], vmin=arr.ndvi.quantile(0.01), vmax=arr.ndvi.quantile(0.99)
    )
    axes[2, 0].set_title("NDVI")
    axes[2, 0].grid(True)

    # LAI
    arr.lai.plot(
        ax=axes[2, 1], vmin=arr.lai.quantile(0.01), vmax=arr.lai.quantile(0.99)
    )
    axes[2, 1].set_title("LAI")
    axes[2, 1].grid(True)

    fig.suptitle(f"{title}")
    if outfname is not None:
        fig.savefig(outfname, format="png", bbox_inches="tight")

    return fig, axes

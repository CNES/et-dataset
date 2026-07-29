# type: ignore
#
# Copyright: (c) 2023 CESBIO / Centre National d'Etudes Spatiales /
#             Université Paul Sabatier (UT3)
#
"""
Functions for plotting
"""

# Skip this file with mypy
import math

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
            "albedo",
            "ndvi",
            "lai",
            "fcover",
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
            # vmin=arr.lai.quantile(0.01),
            # vmax=arr.lai.quantile(0.99),
        )
        axes[irow, icol].set_title("LAI")
        axes[irow, icol].grid(True)
        icol += 1
        if icol == 2:
            icol = 0
            irow += 1

    # LAI
    if "fcover" in bands:
        arr.fcover.plot(
            ax=axes[irow, icol],
            # vmin=arr.lai.quantile(0.01),
            # vmax=arr.lai.quantile(0.99),
        )
        axes[irow, icol].set_title("FCOVER")
        axes[irow, icol].grid(True)
        icol += 1
        if icol == 2:
            icol = 0
            irow += 1

    fig.suptitle(f"{title}")
    if outfname is not None:
        fig.savefig(outfname, format="png", bbox_inches="tight")

    return fig, axes


def plot_dataset(
    data: xr.Dataset,
):
    """
    Plot EF models
    """
    # Set figure subplots
    nb = len(data.data_vars)
    row = int(np.ceil(nb / 2))
    col = 2
    fig = plt.figure(figsize=(4 * col, 3 * row), constrained_layout=True)

    i = 0
    j = 0
    # Loop over variables
    for var in sorted(data.data_vars):
        ax = plt.subplot2grid((row, col), (i, j))
        if var in ["water", "cloud", "qa"]:
            valid_cmap = ListedColormap(
                [(1.0, 0.0, 0.0, 1.0), (0.0, 1.0, 0.0, 1.0)]
            )
            data[var].plot(ax=ax, cmap=valid_cmap, add_colorbar=False)  # type: ignore
        elif var in ["red", "blue", "green", "nir", "swir1", "swir2", "albedo"]:
            data[var].plot(  # type: ignore
                ax=ax,
                vmin=data[var].quantile(0.01),
                vmax=data[var].quantile(0.99),
                cmap="viridis",
            )
        else:
            data[var].plot(ax=ax, cmap="viridis")  # type: ignore
        ax.set_title(str(var))
        j += 1
        if j == 2:
            j = 0
            i += 1
    # title
    fig.suptitle("Dataset", fontsize=12)

    plt.show()


def plot_dem(xrds_dem: xr.Dataset):
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 4))
    xrds_dem["height"].plot(  # type: ignore
        ax=axes[0],
        vmin=xrds_dem["height"].min(),
        vmax=xrds_dem["height"].max(),
        cmap="RdYlGn_r",
    )
    xrds_dem["slope"].plot(  # type: ignore
        ax=axes[1],
        vmin=xrds_dem["slope"].min(),
        vmax=xrds_dem["slope"].max(),
        cmap="Reds",
    )
    xrds_dem["aspect"].plot(  # type: ignore
        ax=axes[2],
        vmin=xrds_dem["aspect"].min(),
        vmax=xrds_dem["aspect"].max(),
        cmap="twilight_shifted",
    )


def density_plot(
    data: xr.Dataset,
):
    """
    Plot
    """
    # dimensions
    _, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 4))

    for i, var in enumerate(["ndvi", "fcover", "albedo"]):
        ax = axes[i]
        df = pd.DataFrame(
            data={
                "lst": data["lst"].values.reshape(-1),
                "var": data[var].values.reshape(-1),
            }
        ).dropna(axis=0, how="any")
        y = df["lst"].values
        # Flatten and mask values
        x = df["var"].values
        mask = np.isfinite(x) & np.isfinite(y)  # type: ignore
        x = x[mask]
        y = y[mask]
        ax.hist2d(x, y, (150, 150), cmap="viridis", cmin=1)

        # Fix limits
        xmin = np.nanmin(x) - 0.05
        xmax = np.nanmax(x) + 0.05
        ymin = np.nanmin(y) - 5
        ymax = np.nanmax(y) + 5
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])

        # Labels
        ax.set_xlabel(var)
        ax.set_ylabel("Land Surface Temperature K")

    plt.show()


def plot_station_map(gdf: gpd.GeoDataFrame):
    """
    Description
    -----------
    Plot station locations on an interactive map from a GeoDataFrame.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame containing station geometries and attributes.

    Returns
    -------

    """
    has_elev = "elev" in gdf.columns and gdf["elev"].notna().any()
    m = gdf.explore(
        column="elev" if has_elev else None,
        tooltip="name",
        popup=True,
        legend=True,
        marker_kwds={"radius": 8},
        style_kwds={"color": "black"},
        figsize=(8, 8),
    )
    return m

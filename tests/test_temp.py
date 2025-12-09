# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales
#
"""
This module contains tests for air temperature and dewpoint temperature
"""

import numpy as np
import xarray as xr
from pyproj import CRS

from etdataset import era5


def test_interpolate_temparture():
    crs = CRS(4326)
    xs = np.array([0, 1])
    ys = np.array([0, 1])

    src_dem = xr.DataArray(
        10, coords={"y": ys, "x": xs}, dims=["y", "x"], name="height"
    ).rio.write_crs(crs)

    src_temp = xr.DataArray(
        [[10, 20], [10, 20]],
        coords={"y": ys, "x": xs},
        dims=["y", "x"],
        name="temp_ref",
    ).rio.write_crs(crs)

    # xdst = np.linspace(0, 3, 4)
    # ydst = np.linspace(0, 3, 4)
    xdst = np.array([0, 1])
    ydst = np.array([0, 1])
    Xdst, Ydst = np.meshgrid(xdst, ydst)

    dst_dem = xr.DataArray(
        300 + 2 * Xdst,
        coords={"y": ydst, "x": xdst},
        dims=["y", "x"],
        name="height",
    ).rio.write_crs(crs)

    lapse_rate = -2

    result = era5.interpolate_temperature(
        src_temp, src_dem, dst_dem, lapse_rate
    )

    expected = xr.DataArray(
        [[-570, -564], [-570, -564]],
        coords={"y": ydst, "x": xdst},
        dims=["y", "x"],
        name="height",
    )
    assert isinstance(result, xr.DataArray)
    assert result.shape == dst_dem.shape
    assert result.max() < src_temp.max(), (
        f"Les températures à l'altitude du DEM doivent être inférieures \
        à celles initiales "
        f"(max obtenu = {result.max()}, max attendu < {src_temp.max()})."
    )
    assert (result.values == expected.values).all()

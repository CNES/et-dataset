# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales
#
"""
This module contains tests for air temperature and dewpoint temperature
"""

import datetime as dt

import numpy as np
import pytest
import xarray as xr
from pyproj import CRS

from etdataset import era5
from etdataset.validation_temp import temperature_rescaling


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


# TEST GENERATE DATES
def test_generate_dates():
    start = dt.date(2024, 1, 1)
    end = dt.date(2024, 1, 5)

    result = list(temperature_rescaling.generate_dates(start, end))

    assert result == [
        dt.date(2024, 1, 1),
        dt.date(2024, 1, 2),
        dt.date(2024, 1, 3),
        dt.date(2024, 1, 4),
        dt.date(2024, 1, 5),
    ]


def test_generate_dates_with_step():
    start = dt.date(2024, 1, 1)
    end = dt.date(2024, 1, 10)

    result = list(temperature_rescaling.generate_dates(start, end, day_step=3))

    assert result == [
        dt.date(2024, 1, 1),
        dt.date(2024, 1, 4),
        dt.date(2024, 1, 7),
        dt.date(2024, 1, 10),
    ]


def test_generate_dates_start_after_end():
    start = dt.date(2024, 1, 10)
    end = dt.date(2024, 1, 1)

    result = list(temperature_rescaling.generate_dates(start, end))

    assert result == []


# TEST GENERATE HOURS


def test_generate_hours_default():
    hours = temperature_rescaling.generate_hours()
    assert hours == [
        dt.time(0),
        dt.time(2),
        dt.time(4),
        dt.time(6),
        dt.time(8),
        dt.time(10),
        dt.time(12),
        dt.time(14),
        dt.time(16),
        dt.time(18),
        dt.time(20),
        dt.time(22),
    ]


def test_generate_hours_custom():
    hours = temperature_rescaling.generate_hours(6, 12, 3)
    assert hours == [dt.time(6), dt.time(9), dt.time(12)]


def test_generate_hours_invalid_step():
    with pytest.raises(ValueError):
        temperature_rescaling.generate_hours(hour_step=0)


def test_filter_dataset_by_hours():
    date = dt.date(2024, 1, 1)
    times = [dt.time(0), dt.time(6)]

    ds = xr.Dataset(
        data_vars={"ta": ("time", [1, 2, 3])},
        coords={
            "time": [
                dt.datetime(2024, 1, 1, 0),
                dt.datetime(2024, 1, 1, 6),
                dt.datetime(2024, 1, 1, 12),
            ]
        },
    )

    filtered = temperature_rescaling.filter_dataset_by_hours(ds, date, times)

    assert len(filtered.time) == 2
    assert filtered.time.values[0].hour == 0
    assert filtered.time.values[1].hour == 6


def test_filter_dataset_missing_time_coord():
    ds = xr.Dataset(data_vars={"ta": ("x", [1, 2])})

    with pytest.raises(ValueError):
        temperature_rescaling.filter_dataset_by_hours(
            ds, dt.date(2026, 2, 9), []
        )


def test_create_era5_sub_dataset():
    ds = xr.Dataset(
        data_vars={
            "t2m": (("x", "y"), np.ones((2, 2))),
            "d2m": (("x", "y"), np.ones((2, 2)) * 2),
            "u10": (("x", "y"), np.ones((2, 2)) * 3),
        }
    )

    sub = temperature_rescaling.create_era5_sub_dataset(ds)

    assert set(sub.data_vars) == {"ta", "tdp"}


# tests for get xy dimensions function
def test_get_xy_dims_projected():
    ds = xr.Dataset(coords={"x": [1], "y": [2]})
    dims = temperature_rescaling.get_xy_dims(ds)

    assert dims == {"x": "x", "y": "y"}


def test_get_xy_dims_geographic():
    ds = xr.Dataset(coords={"latitude": [4], "longitude": [3]})
    dims = temperature_rescaling.get_xy_dims(ds)

    assert dims == {"x": "longitude", "y": "latitude"}


def test_get_xy_dims_invalid():
    ds = xr.Dataset(coords={"time": [0]})

    with pytest.raises(ValueError):
        temperature_rescaling.get_xy_dims(ds)

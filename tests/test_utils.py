#
# SPDX-License-Identifier: AGPL-3.0-only
#
# Copyright (C) 2024 CESBIO / Centre National d'Etudes Spatiales
#
"""
This module contains tests for utils
"""

import datetime as dt

import pytest
import xarray as xr

from etdataset.utils import (
    filter_dataset_by_hours,
    generate_dates,
    generate_hours,
    get_xy_dims,
    kelvin_to_celsius,
)


@pytest.mark.unit
def test_generate_dates():
    start = dt.date(2024, 1, 1)
    end = dt.date(2024, 1, 5)

    result = list(generate_dates(start, end))

    assert result == [
        dt.date(2024, 1, 1),
        dt.date(2024, 1, 2),
        dt.date(2024, 1, 3),
        dt.date(2024, 1, 4),
        dt.date(2024, 1, 5),
    ]


@pytest.mark.unit
def test_generate_dates_with_step():
    start = dt.date(2024, 1, 1)
    end = dt.date(2024, 1, 10)

    result = list(generate_dates(start, end, day_step=3))

    assert result == [
        dt.date(2024, 1, 1),
        dt.date(2024, 1, 4),
        dt.date(2024, 1, 7),
        dt.date(2024, 1, 10),
    ]


@pytest.mark.unit
def test_generate_dates_start_after_end():
    start = dt.date(2024, 1, 10)
    end = dt.date(2024, 1, 1)

    result = list(generate_dates(start, end))

    assert result == []


@pytest.mark.unit
def test_generate_hours_default():
    hours = generate_hours()
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


@pytest.mark.unit
def test_generate_hours_custom():
    hours = generate_hours(6, 12, 3)
    assert hours == [dt.time(6), dt.time(9), dt.time(12)]


@pytest.mark.unit
def test_generate_hours_invalid_step():
    with pytest.raises(ValueError):
        generate_hours(hour_step=0)


@pytest.mark.unit
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

    filtered = filter_dataset_by_hours(ds, date, times)

    assert len(filtered.time) == 2
    hours = filtered.time.dt.hour.values
    assert hours[0] == 0
    assert hours[1] == 6


@pytest.mark.unit
def test_filter_dataset_missing_time_coord():
    ds = xr.Dataset(data_vars={"ta": ("x", [1, 2])})

    with pytest.raises(ValueError):
        filter_dataset_by_hours(ds, dt.date(2026, 2, 9), [])


@pytest.mark.unit
def test_get_xy_dims_projected():
    ds = xr.Dataset(coords={"x": [1], "y": [2]})
    dims = get_xy_dims(ds)

    assert dims == {"x": "x", "y": "y"}


@pytest.mark.unit
def test_get_xy_dims_geographic():
    ds = xr.Dataset(coords={"latitude": [4], "longitude": [3]})
    dims = get_xy_dims(ds)

    assert dims == {"x": "longitude", "y": "latitude"}


@pytest.mark.unit
def test_get_xy_dims_invalid():
    ds = xr.Dataset(coords={"time": [0]})

    with pytest.raises(ValueError):
        get_xy_dims(ds)


@pytest.mark.unit
def test_kelvin_to_celsius():
    data = xr.DataArray([273.15, 274.15])

    celsius = kelvin_to_celsius(data)

    assert float(celsius[0]) == 0.0
    assert float(celsius[1]) == 1.0

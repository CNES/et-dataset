#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales
#
"""
This module contains tests for ERA5 and ERA5-land data driver
"""
import os
from dataclasses import dataclass
from typing import List

import affine  # type: ignore
import numpy as np
import pytest
import rasterio as rio
from pyproj import CRS

from etdataset import era5


def get_era5_product() -> str:
    """
    Retrieve ERA5 product path
    """
    return os.path.join(
        os.environ["ETDATASET_TEST_DATA_PATH"],
        "era5",
        "download_era5_2023-10-10_44.35_0.40_43.14_1.99.nc",
    )


def get_era5land_product() -> str:
    """
    Retrieve ERA5-Land product path
    """
    return os.path.join(
        os.environ["ETDATASET_TEST_DATA_PATH"],
        "era5",
        "download_era5land_2023-10-10_44.35_0.40_43.14_1.99.nc",
    )


def test_era5var_instantiate():
    """
    Test ERA5Var
    """
    var = era5.ERA5Var.from_key("ssrd")
    assert var.key == "ssrd"
    var = era5.ERA5Var("ssrd")
    assert var.key == "ssrd"
    with pytest.raises(ValueError):
        var = era5.ERA5Var("toto")


@pytest.mark.requires_test_data
def test_era5land_instantiate():
    """
    Test ERA5Data class instantiation
    """
    era5_ds = era5.ERA5Data(get_era5land_product())

    assert era5_ds.filename == get_era5land_product()
    assert era5_ds.get_available_dates() == [
        "2023-10-10 00:00:00",
        "2023-10-10 01:00:00",
        "2023-10-10 02:00:00",
        "2023-10-10 03:00:00",
        "2023-10-10 04:00:00",
        "2023-10-10 05:00:00",
        "2023-10-10 06:00:00",
        "2023-10-10 07:00:00",
        "2023-10-10 08:00:00",
        "2023-10-10 09:00:00",
        "2023-10-10 10:00:00",
        "2023-10-10 11:00:00",
        "2023-10-10 12:00:00",
        "2023-10-10 13:00:00",
        "2023-10-10 14:00:00",
        "2023-10-10 15:00:00",
        "2023-10-10 16:00:00",
        "2023-10-10 17:00:00",
        "2023-10-10 18:00:00",
        "2023-10-10 19:00:00",
        "2023-10-10 20:00:00",
        "2023-10-10 21:00:00",
        "2023-10-10 22:00:00",
        "2023-10-10 23:00:00",
    ]
    assert era5_ds.get_available_variables() == [
        "u10",
        "v10",
        "d2m",
        "t2m",
        "ssrd",
        "strd",
        "tp",
    ]
    assert era5_ds.bounds == rio.coords.BoundingBox(
        left=0.34496750000000004,
        bottom=43.0879675,
        right=1.9460324999999998,
        top=44.3890325,
    )
    assert era5_ds.transform == affine.Affine(
        0.100065, 0.0, 0.34496750000000004, 0.0, -0.100065, 44.3890325
    )
    assert era5_ds.crs == CRS.from_epsg(4326).to_string()
    np.testing.assert_almost_equal(era5_ds.resolution, 0.1, decimal=2)


@pytest.mark.requires_test_data
def test_era5_instantiate():
    """
    Test ERA5Data class instantiation
    """
    era5_ds = era5.ERA5Data(get_era5_product())

    assert era5_ds.filename == get_era5_product()
    assert era5_ds.get_available_dates() == [
        "2023-10-10 00:00:00",
        "2023-10-10 01:00:00",
        "2023-10-10 02:00:00",
        "2023-10-10 03:00:00",
        "2023-10-10 04:00:00",
        "2023-10-10 05:00:00",
        "2023-10-10 06:00:00",
        "2023-10-10 07:00:00",
        "2023-10-10 08:00:00",
        "2023-10-10 09:00:00",
        "2023-10-10 10:00:00",
        "2023-10-10 11:00:00",
        "2023-10-10 12:00:00",
        "2023-10-10 13:00:00",
        "2023-10-10 14:00:00",
        "2023-10-10 15:00:00",
        "2023-10-10 16:00:00",
        "2023-10-10 17:00:00",
        "2023-10-10 18:00:00",
        "2023-10-10 19:00:00",
        "2023-10-10 20:00:00",
        "2023-10-10 21:00:00",
        "2023-10-10 22:00:00",
        "2023-10-10 23:00:00",
    ]
    assert sorted(era5_ds.get_available_variables()) == sorted(
        [
            "u10",
            "v10",
            "d2m",
            "t2m",
            "ssrdc",
            "ssrd",
            "strdc",
            "strd",
            "tco3",
            "tcw",
            "tp",
            "tcwv",
            "z",
        ]
    )
    assert era5_ds.bounds == rio.coords.BoundingBox(
        left=0.269917, bottom=43.012917, right=2.021083, top=44.264083
    )
    assert era5_ds.transform == affine.Affine(
        0.250166, 0.0, 0.269917, 0.0, -0.250166, 44.264083
    )
    assert era5_ds.crs == CRS.from_epsg(4326).to_string()
    np.testing.assert_almost_equal(era5_ds.resolution, 0.25, decimal=2)


@dataclass(frozen=True)
class ERA5ReadParams:
    """
    Class to store read parameters
    """

    variables: List[era5.ERA5Var] | None = None
    dates: List[str] | None = None


@pytest.mark.requires_test_data
@pytest.mark.parametrize(
    "parameters",
    [
        # Default
        ERA5ReadParams(),
        # Other options
        ERA5ReadParams(variables=[era5.ERA5Var.TEMPERATURE]),
        ERA5ReadParams(
            dates=[
                "2023-10-10 17:00:00",
                "2023-10-10 18:00:00",
            ]
        ),
        ERA5ReadParams(
            variables=[era5.ERA5Var.TEMPERATURE],
            dates=[
                "2023-10-10 17:00:00",
                "2023-10-10 18:00:00",
            ],
        ),
    ],
)
def test_era5_read(parameters: ERA5ReadParams):
    """
    Test ERA5Data class instantiation
    """
    era5_ds = era5.ERA5Data(get_era5_product())
    xarr = era5_ds.read(**parameters.__dict__)
    assert xarr

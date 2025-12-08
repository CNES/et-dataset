# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales
#
"""
This module contains tests for ERA5 and ERA5-land data driver
"""

import os

import pytest

from etdataset import era5


def get_era5_product() -> str:
    """
    Retrieve ERA5 product path
    """
    return os.path.join(
        os.environ["ETDATASET_TEST_DATA_PATH"],
        "era5",
        "download_era5_2023-03-03.zip",
    )


def get_era5land_product() -> str:
    """
    Retrieve ERA5-Land product path
    """
    return os.path.join(
        os.environ["ETDATASET_TEST_DATA_PATH"],
        "era5",
        "download_era5land_2023-03-03.zip",
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
    era5_ds = era5.read(get_era5land_product())

    assert era5_ds
    assert sorted(era5_ds.data_vars) == sorted(
        [
            "u10",
            "v10",
            "d2m",
            "t2m",
            "ssrd",
            "strd",
            "tp",
            "e",
        ]
    )


@pytest.mark.requires_test_data
def test_era5_instantiate():
    """
    Test ERA5Data class instantiation
    """
    era5_ds = era5.read(get_era5_product())

    assert era5_ds
    assert sorted(era5_ds.data_vars) == sorted(
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

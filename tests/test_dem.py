# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales

import os
from unittest import TestCase

import numpy as np
import pytest

from etdataset.dem import (
    get_dem_from_tile,
    get_dem_from_tiles,
    get_elevation_from_tile,
)


def get_test_data_path() -> str:
    """
    Get test data path for DEM tiles
    """
    return os.path.join(
        os.environ["ETDATASET_TEST_DATA_PATH"], "DEM_Copercinus_30m"
    )


@pytest.mark.functional
@pytest.mark.require_test_data
def test_get_dem_from_tile() -> None:
    """
    Test get_dem_from_tile() method
    """
    dem = get_dem_from_tile("32TML", base_dir=get_test_data_path())
    assert dem
    assert dem.sizes["y"] == 1830
    assert dem.sizes["x"] == 1830
    assert dem.crs == "EPSG:32632"
    TestCase().assertDictEqual({"x": 60, "y": 60}, dem.resolution)
    np.testing.assert_array_equal(
        sorted([str(v) for v in dem.data_vars]), ["aspect", "height", "slope"]
    )
    for v in dem.data_vars:
        assert not np.isnan(np.sum(dem[v].data))


@pytest.mark.functional
@pytest.mark.require_test_data
def test_get_elevation_from_tile() -> None:
    """
    Test get_dem_from_tile() method
    """
    dem = get_elevation_from_tile("32TML", base_dir=get_test_data_path())
    assert dem.sizes["y"] == 1830
    assert dem.sizes["x"] == 1830
    assert dem.crs == "EPSG:32632"
    TestCase().assertDictEqual({"x": 60, "y": 60}, dem.resolution)
    assert not np.isnan(np.sum(dem.data))


@pytest.mark.functional
@pytest.mark.require_test_data
def test_get_dem_from_tiles() -> None:
    """
    Test get_dem_from_tiles() method
    """
    dem = get_dem_from_tiles(["32TML"], base_dir=get_test_data_path())
    assert dem
    assert dem.sizes["y"] == 1830
    assert dem.sizes["x"] == 1830
    assert dem.crs == "EPSG:32632"
    TestCase().assertDictEqual({"x": 60, "y": 60}, dem.resolution)
    np.testing.assert_array_equal(
        sorted([str(v) for v in dem.data_vars]), ["aspect", "height", "slope"]
    )
    for v in dem.data_vars:
        assert not np.isnan(np.sum(dem[v].data))

    dem = get_dem_from_tiles(["32TML", "32TNL"], base_dir=get_test_data_path())
    assert dem
    assert dem.sizes["y"] == 1830
    assert dem.sizes["x"] == 3497
    assert dem.crs == "EPSG:32632"
    TestCase().assertDictEqual({"x": 60, "y": 60}, dem.resolution)
    np.testing.assert_array_equal(
        sorted([str(v) for v in dem.data_vars]), ["aspect", "height", "slope"]
    )
    for v in dem.data_vars:
        assert not np.isnan(np.sum(dem[v].data))

# Copyright: (c) 2026 CESBIO / Centre National d'Etudes Spatiales

"""
Test API
"""

import datetime as dt
import os
from pathlib import Path

import pytest

from etdataset.et import (
    create_daily_et_dataset,
    create_daily_explanatory_dataset,
)
from etdataset.interpolation import create_grid_dataset
from etdataset.utils import get_utm_bbox_from_roi


def get_data_path() -> str:
    """
    Get data path
    """
    data = Path(os.environ["ETDATASET_TEST_DATA_PATH"]) / "test_api_prepare"
    return str(data)


@pytest.mark.functional
def test_create_daily_et_dataset(tmp_path):
    """
    Test the function for creating daily ET dataset
    """
    d = tmp_path / "out"
    d.mkdir()
    # ROI
    roi_path = os.path.join("tests", "data", "Zone_Senegal_Centre.shp")
    roi_bbox, roi_crs = get_utm_bbox_from_roi(roi_path)
    # Grid
    grid = create_grid_dataset(bounds=roi_bbox, crs=roi_crs, resolution=3000)
    # acquisition dates
    date = dt.datetime(2023, 3, 1)
    res = create_daily_et_dataset(date=date, grid=grid, path=get_data_path())
    assert res


@pytest.mark.functional
def test_create_daily_explanatory_dataset(tmp_path):
    """
    Test the function for creating daily explanatory dataset
    """
    d = tmp_path / "out"
    d.mkdir()
    # ROI
    roi_path = os.path.join("tests", "data", "Zone_Senegal_Centre.shp")
    roi_bbox, roi_crs = get_utm_bbox_from_roi(roi_path)
    # Grid
    grid = create_grid_dataset(bounds=roi_bbox, crs=roi_crs, resolution=3000)
    # acquisition dates
    date = dt.datetime(2023, 3, 1)
    res = create_daily_explanatory_dataset(
        date=date, grid=grid, path=get_data_path()
    )
    assert res

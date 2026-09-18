#
# SPDX-License-Identifier: AGPL-3.0-only
#
# Copyright (C) 2024 CESBIO / Centre National d'Etudes Spatiales
#
"""
Test prepare API
"""

import os
import shutil
from pathlib import Path

import pytest

from etdataset.api import (
    prepare_daily_et_timeseries,
    prepare_daily_radiation_timeseries,
)
from etdataset.utils import get_utm_bbox_from_roi


@pytest.fixture
def prepare_tmpdir(tmp_path):
    """
    Copy data to avoid download
    """
    # Get shared tmp path
    base = tmp_path / "out"
    base.mkdir()

    def copy_data(required: list[str] | None = None):
        if required is None:
            required = ["era5", "msg"]
        # Test if data already exists
        data = Path(os.environ["ETDATASET_TEST_DATA_PATH"]) / "test_api_prepare"
        if "era5" in required:
            shutil.copytree(
                data / "ERA5_data", base / "ERA5_data", dirs_exist_ok=True
            )
        if "msg" in required:
            shutil.copytree(
                data / "MSG_data", base / "MSG_data", dirs_exist_ok=True
            )
        return base

    return copy_data


def assert_tree(actual: Path, expected: dict):
    """
    Compare_two file trees

    Parameters
    ----------
    actual: str
        Path of the root tree to be compared
    expected: dict
        Description of the expected structure
    """
    actual_items = {p.name for p in actual.iterdir()}
    expected_items = set(expected.keys())

    # No missing or extra files
    assert sorted(actual_items) == sorted(expected_items)

    # Check
    for name, spec in expected.items():
        path = actual / name

        if isinstance(spec, dict):
            assert path.is_dir()
            assert_tree(path, spec)
        else:
            assert path.is_file()
            assert path.stat().st_size > 0


@pytest.mark.functional
def test_prepare_daily_radiation_timeseries(tmp_path):
    """
    Test prepare daily radiation timeseries over a period
    """
    d = tmp_path / "out"
    d.mkdir()
    # ROI
    roi_path = os.path.join("tests", "data", "Zone_Senegal_Centre.shp")
    roi_bbox, roi_crs = get_utm_bbox_from_roi(roi_path)
    # acquisition dates
    min_date = "2023-03-01"  # Format YYYY-MM-DD
    max_date = "2023-03-02"  # Format YYYY-MM-DD
    prepare_daily_radiation_timeseries(
        start_date=min_date,
        end_date=max_date,
        roi_bbox=roi_bbox,
        roi_crs=roi_crs,
        resolution=3000,
        output=str(d),
    )
    expected_tree = {
        "MSG_data": {
            "MSG_2023-03-01": {
                (
                    "NETCDF4_LSASAF_MSG_DIDSSF_MSG-Disk_202303010000.nc"
                ): "non-empty"
            },
            "MSG_2023-03-02": {
                (
                    "NETCDF4_LSASAF_MSG_DIDSSF_MSG-Disk_202303020000.nc"
                ): "non-empty"
            },
        },
        "timeseries": {
            "daily_radiation": {
                "radiation_20230301.tif": "non-empty",
                "radiation_20230302.tif": "non-empty",
            }
        },
    }
    assert_tree(d, expected_tree)


@pytest.mark.functional
@pytest.mark.require_test_data
@pytest.mark.parametrize(
    ("extra_variables", "expected"),
    [
        pytest.param(
            False,
            {
                "ERA5_data": {
                    "download_era5land_2023-03-01.zip": "non-empty",
                    "download_era5land_2023-03-02.zip": "non-empty",
                },
                "timeseries": {
                    "et": {
                        "et_single_date_20230301.tif": "non-empty",
                        "et_single_date_20230302.tif": "non-empty",
                    },
                },
            },
        ),
        pytest.param(
            True,
            {
                "ERA5_data": {
                    "download_era5land_2023-03-01.zip": "non-empty",
                    "download_era5land_2023-03-02.zip": "non-empty",
                },
                "timeseries": {
                    "et": {
                        "et_single_date_20230301.tif": "non-empty",
                        "et_single_date_20230302.tif": "non-empty",
                    },
                    "extra": {
                        "extra_20230301.tif": "non-empty",
                        "extra_20230302.tif": "non-empty",
                    },
                },
            },
        ),
    ],
)
def test_prepare_daily_et_timeseries(extra_variables, expected, prepare_tmpdir):
    """
    Test prepare daily ET timeseries with data already downloaded
    """
    d = prepare_tmpdir(["era5"])
    # ROi
    roi_path = os.path.join("tests", "data", "Zone_Senegal_Centre.shp")
    roi_bbox, roi_crs = get_utm_bbox_from_roi(roi_path)
    # acquisition dates
    min_date = "2023-03-01"  # Format YYYY-MM-DD
    max_date = "2023-03-02"  # Format YYYY-MM-DD
    # run
    prepare_daily_et_timeseries(
        start_date=min_date,
        end_date=max_date,
        roi_bbox=roi_bbox,
        roi_crs=roi_crs,
        resolution=3000,
        output=str(d),
        extra_variables=extra_variables,
    )
    assert_tree(d, expected)

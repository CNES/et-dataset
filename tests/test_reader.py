#!/usr/bin/env python
# coding: utf8

# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales

"""
Test API
"""
import os
import pytest
from dataclasses import dataclass

from etdataset.reader import (
    LandsatReader,
    EcostressReader,
    HLSReader,
    Sentinel2Reader,
    get_product_reader,
)


@pytest.mark.requires_test_data
def test_landsat_reader():
    """
    Test Landsat reader
    """
    landsat_path = os.path.join(
        os.environ["ETDATASET_TEST_DATA_PATH"],
        "Landsat",
        "LC09_L2SP_205050_20230327_20230329_02_T1/",
    )
    reader = LandsatReader(landsat_path)
    assert reader


@pytest.mark.requires_test_data
def test_ecostress_reader():
    """
    Test Ecostress reader
    """
    eco_path = os.path.join(
        os.environ["ETDATASET_TEST_DATA_PATH"],
        "Ecostress",
        "ECOv002_L2T_LSTE_26870_001_28PCA_20230402T115902_0710_01",
    )
    reader = EcostressReader(eco_path)
    assert reader


@pytest.mark.requires_test_data
def test_sentinel2_reader():
    """
    Test Sentinel2 reader
    """
    s2_path = os.path.join(
        os.environ["ETDATASET_TEST_DATA_PATH"],
        "Sentinel2",
        "SENTINEL2B_20230402-114740-682_L2A_T28PCA_C_V3-1",
    )
    reader = Sentinel2Reader(s2_path)
    assert reader


@pytest.mark.requires_test_data
def test_hlsl_reader():
    """
    Test Ecostress reader
    """
    hls_path = os.path.join(
        os.environ["ETDATASET_TEST_DATA_PATH"],
        "HLS",
        "HLS.L30.T28PCA.2023086T112728.v2.0",
    )
    reader = HLSReader(hls_path)
    assert reader


@pytest.mark.requires_test_data
def test_hlss_reader():
    """
    Test Ecostress reader
    """
    hls_path = os.path.join(
        os.environ["ETDATASET_TEST_DATA_PATH"],
        "HLS",
        "HLS.S30.T28PCA.2023092T113319.v2.0",
    )
    reader = HLSReader(hls_path)
    assert reader


@dataclass(frozen=True)
class ReaderParams:
    """
    Class to store read parameters
    """

    product_path: str


@pytest.mark.requires_test_data
@pytest.mark.parametrize(
    "parameters",
    [
        ReaderParams(
            os.path.join(
                os.environ["ETDATASET_TEST_DATA_PATH"],
                "Landsat",
                "LC09_L2SP_205050_20230327_20230329_02_T1/",
            )
        ),
        ReaderParams(
            os.path.join(
                os.environ["ETDATASET_TEST_DATA_PATH"],
                "Ecostress",
                "ECOv002_L2T_LSTE_26870_001_28PCA_20230402T115902_0710_01",
            )
        ),
        ReaderParams(
            os.path.join(
                os.environ["ETDATASET_TEST_DATA_PATH"],
                "Sentinel2",
                "SENTINEL2B_20230402-114740-682_L2A_T28PCA_C_V3-1",
            )
        ),
        ReaderParams(
            os.path.join(
                os.environ["ETDATASET_TEST_DATA_PATH"],
                "HLS",
                "HLS.L30.T28PCA.2023086T112728.v2.0",
            )
        ),
        ReaderParams(
            os.path.join(
                os.environ["ETDATASET_TEST_DATA_PATH"],
                "HLS",
                "HLS.S30.T28PCA.2023092T113319.v2.0",
            )
        ),
    ],
)
def test_get_product_reader(parameters: ReaderParams):
    """
    Test get_product_reader method
    """
    reader = get_product_reader(**parameters.__dict__)
    assert reader


def test_get_product_reader_error():
    """
    Test get_product_reader method
    """
    with pytest.raises(Exception):
        get_product_reader("not found")

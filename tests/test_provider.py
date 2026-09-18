#
# SPDX-License-Identifier: AGPL-3.0-only
#
# Copyright (C) 2024 CESBIO / Centre National d'Etudes Spatiales
#
"""
Test provider module
"""

import os

import pytest
import rasterio as rio

from etdataset.provider import (
    Collection,
    EcostressProvider,
    HLSLProvider,
    HLSSProvider,
    LandsatProvider,
    get_provider,
)
from etdataset.utils import read_product_list


@pytest.mark.functional
@pytest.mark.parametrize(
    ("collection", "expected"),
    [
        pytest.param(Collection.ECOSTRESS, EcostressProvider),
        pytest.param(Collection.HLSLANDSAT, HLSLProvider),
        pytest.param(Collection.HLSSENTINEL2, HLSSProvider),
    ],
)
def test_get_provider(collection, expected):
    """
    Test get_provider function
    """
    provider = get_provider(collection)
    assert isinstance(provider, expected)


@pytest.mark.functional
@pytest.mark.slow
def test_get_provider_usgs():
    """
    Test get_provider function (login)
    """
    provider = get_provider(Collection.LANDSAT)
    assert isinstance(provider, LandsatProvider)


@pytest.mark.functional
@pytest.mark.parametrize(
    "collection",
    [
        pytest.param("foo"),
    ],
)
def test_get_provider_error(collection):
    """
    Test get_provider function with exception raising
    """
    with pytest.raises(TypeError):
        get_provider(collection)


@pytest.mark.functional
@pytest.mark.parametrize(
    ("collection", "expected"),
    [
        pytest.param(Collection.ECOSTRESS, 10),
        pytest.param(Collection.HLSLANDSAT, 2),
        pytest.param(Collection.HLSSENTINEL2, 4),
    ],
)
def test_search_with_roi(collection, expected):
    """
    Test search function for each provider (ROI)
    """
    # ROI
    tile_id = "28PCA"
    # Acquisition dates
    min_date = "2023-03-01"  # Format YYYY-MM-DD
    max_date = "2023-03-10"  # Format YYYY-MM-DD
    # Other criteria to filter products
    max_cloud_cover = 20
    # Instantiate the provider
    provider = get_provider(collection)
    res = provider.search(
        min_date=min_date,
        max_date=max_date,
        tile_id=tile_id,
        max_cloud_cover=max_cloud_cover,
    )
    assert len(res) == expected


@pytest.mark.functional
@pytest.mark.parametrize(
    ("collection", "expected"),
    [
        pytest.param(Collection.ECOSTRESS, 12),
        pytest.param(Collection.HLSLANDSAT, 4),
        pytest.param(Collection.HLSSENTINEL2, 4),
    ],
)
def test_search_with_bbox(collection, expected):
    """
    Test search function for each provider (bounding box)
    """
    # Bounding box
    bbox = rio.coords.BoundingBox(
        left=-17.1, bottom=13.8, right=-15.9, top=15.0
    )
    # Acquisition dates
    min_date = "2023-03-02"  # Format YYYY-MM-DD
    max_date = "2023-03-02"  # Format YYYY-MM-DD
    # Other criteria to filter products
    max_cloud_cover = 20
    # Instantiate the provider
    provider = get_provider(collection)
    res = provider.search(
        min_date=min_date,
        max_date=max_date,
        latlon_bbox=bbox,
        max_cloud_cover=max_cloud_cover,
    )
    assert len(res) == expected


@pytest.mark.functional
@pytest.mark.slow
def test_search_usgs_with_roi():
    """
    Test search function for USGS provider (ROI)
    """
    # ROI
    tile_id = "28PCA"
    # Acquisition dates
    min_date = "2023-03-01"  # Format YYYY-MM-DD
    max_date = "2023-03-10"  # Format YYYY-MM-DD
    # Other criteria to filter products
    max_cloud_cover = 20
    # Instantiate the provider
    provider = get_provider(Collection.LANDSAT)
    res = provider.search(
        min_date=min_date,
        max_date=max_date,
        tile_id=tile_id,
        max_cloud_cover=max_cloud_cover,
    )
    assert len(res) == 4


@pytest.mark.functional
@pytest.mark.slow
def test_search_usgs_with_bbox():
    """
    Test search function for Landsat provider (bounding box)
    """
    # Bounding box
    bbox = rio.coords.BoundingBox(
        left=-17.1, bottom=13.8, right=-15.9, top=15.0
    )
    # Acquisition dates
    min_date = "2023-03-01"  # Format YYYY-MM-DD
    max_date = "2023-03-05"  # Format YYYY-MM-DD
    # Other criteria to filter products
    max_cloud_cover = 20
    # Instantiate the provider
    provider = get_provider(Collection.LANDSAT)
    res = provider.search(
        min_date=min_date,
        max_date=max_date,
        latlon_bbox=bbox,
        max_cloud_cover=max_cloud_cover,
    )
    assert len(res) == 1


@pytest.mark.functional
@pytest.mark.slow
@pytest.mark.parametrize(
    ("collection", "search"),
    [
        pytest.param(Collection.ECOSTRESS, "ecostress_search.csv"),
        pytest.param(Collection.HLSLANDSAT, "hlslandsat_search.csv"),
        pytest.param(Collection.HLSSENTINEL2, "hlssentinel2_search.csv"),
        pytest.param(Collection.LANDSAT, "landsat_search.csv"),
    ],
)
def test_download(collection, search, tmp_path):
    """
    Test download function for each provider (ROI)
    """
    # Download path
    download_path = tmp_path / "download"
    download_path.mkdir()
    # Instantiate the provider
    provider = get_provider(collection)
    # Get product list
    products = read_product_list(os.path.join("tests", "data", search), 4326)
    # Download
    provider.download(products=products, local_path=download_path)
    # Check
    assert sum(1 for _ in download_path.iterdir()) == len(products)

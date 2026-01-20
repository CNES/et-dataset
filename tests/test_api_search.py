# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales

"""
Test API
"""

import pytest
import rasterio as rio
from pyproj import CRS

from etdataset.api import APIException, search
from etdataset.provider import Collection


@pytest.mark.functional
@pytest.mark.parametrize(
    ("collection", "expected"),
    [
        pytest.param(Collection.ECOSTRESS, 10),
    ],
)
def test_search_by_tile(collection, expected):
    """
    Test search api
    """
    # ROi
    tile_id = "28PCA"
    # acquisition dates
    min_date = "2023-03-01"  # Format YYYY-MM-DD
    max_date = "2023-03-10"  # Format YYYY-MM-DD
    # Other criteria to filter products
    max_cloud_cover = 20
    res = search(
        collection,
        min_date,
        max_date,
        tile_id=tile_id,
        max_cloud_cover=max_cloud_cover,
    )
    assert len(res) == expected


@pytest.mark.functional
@pytest.mark.parametrize(
    ("collection", "cloud", "overlap", "expected"),
    [
        pytest.param(Collection.HLSSENTINEL2, 100, 50, 5),
        pytest.param(Collection.HLSSENTINEL2, 10, 30, 7),
        pytest.param(Collection.HLSSENTINEL2, 0, 30, 0),
        pytest.param(Collection.HLSSENTINEL2, 100, 10, 14),
    ],
)
def test_search_by_roi(collection, cloud, overlap, expected):
    """
    Test search api
    """
    # Bounding box
    bbox = rio.coords.BoundingBox(
        left=-17.1, bottom=13.8, right=-15.9, top=15.0
    )
    # acquisition dates
    min_date = "2023-03-01"  # Format YYYY-MM-DD
    max_date = "2023-03-10"  # Format YYYY-MM-DD
    res = search(
        collection,
        min_date,
        max_date,
        roi_bbox=bbox,
        roi_crs=CRS(4326),
        max_cloud_cover=cloud,
        min_roi_overlap=overlap,
    )
    assert len(res) == expected


@pytest.mark.functional
@pytest.mark.parametrize(
    (
        "collection",
        "min_date",
        "max_date",
        "tile_id",
        "roi_bbox",
        "roi_crs",
        "cloud",
        "overlap",
        "expected_exception",
        "expected_msg",
    ),
    [
        pytest.param(
            Collection.HLSSENTINEL2,
            "20230301",
            "2023-03-10",
            "28PCA",
            None,
            None,
            None,
            None,
            APIException,
            "expected format for date",
        ),
        pytest.param(
            Collection.HLSSENTINEL2,
            "2023-03-01",
            "2023-03-10",
            None,
            None,
            None,
            None,
            None,
            APIException,
            "You must provide either a ROI bounding box or MGRS tile ID",
        ),
        pytest.param(
            Collection.HLSSENTINEL2,
            "2023-03-01",
            "2023-03-10",
            None,
            rio.coords.BoundingBox(
                left=-17.1, bottom=13.8, right=-15.9, top=15.0
            ),
            None,
            None,
            None,
            APIException,
            "You must provide a ROI bounding box with a CRS",
        ),
        pytest.param(
            Collection.HLSSENTINEL2,
            "2023-03-01",
            "2023-03-10",
            "toto",
            None,
            None,
            None,
            None,
            ValueError,
            "Wrong format for MGRS tile",
        ),
        pytest.param(
            Collection.HLSSENTINEL2,
            "2023-03-01",
            "2023-03-10",
            "28PCA",
            None,
            None,
            110,
            None,
            APIException,
            "Cloud cover criteria",
        ),
    ],
)
def test_search_with_error(
    collection,
    min_date,
    max_date,
    tile_id,
    roi_bbox,
    roi_crs,
    cloud,
    overlap,
    expected_exception,
    expected_msg,
):
    """
    Test search api
    """
    with pytest.raises(expected_exception, match=expected_msg):
        search(
            collection=collection,
            min_date=min_date,
            max_date=max_date,
            tile_id=tile_id,
            roi_bbox=roi_bbox,
            roi_crs=roi_crs,
            max_cloud_cover=cloud,
            min_roi_overlap=overlap,
        )

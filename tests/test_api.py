# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales

"""
Test API
"""

from etdataset.api import search
from etdataset.provider import Collection


def test_search():
    """
    Test search API
    """
    # ROI
    s2_tile_id = "28PCA"
    # Acquisition dates
    min_date = "2023-02-01"  # Format YYYY-MM-DD
    max_date = "2023-03-31"  # Format YYYY-MM-DD
    # Other criteria to filter products
    max_cloud_cover = 20
    res = search(
        Collection.SENTINEL2,
        min_date,
        max_date,
        tile_id=s2_tile_id,
        max_cloud_cover=max_cloud_cover,
    )
    assert len(res) == 10

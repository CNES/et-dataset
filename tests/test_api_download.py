# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales

"""
Test API
"""

import os

import pytest

from etdataset.api import download
from etdataset.utils import read_product_list


@pytest.mark.functional
def test_download(tmp_path):
    """
    Test api for download products
    """
    # Download path
    download_path = tmp_path / "download"
    download_path.mkdir()
    # Get product list
    products = read_product_list(
        os.path.join("tests", "data", "products.csv"), 4326
    )
    # Download
    download(products=products, output_dir=download_path)
    # Check
    assert sum(
        len(list(subdir.iterdir()))
        for subdir in download_path.iterdir()
        if subdir.is_dir()
    ) == len(products)

# Copyright: (c) 2026 CESBIO / Centre National d'Etudes Spatiales

"""
Test writer module
"""

import numpy as np
import xarray as xr

from etdataset import writer

NB_ROW = 30
NB_COL = 40


def setup_data() -> xr.Dataset:
    """
    Create a xarray dataset for tests
    """
    # Set the dimensions for the dataset
    nbx = NB_COL
    nby = NB_ROW
    # Create random data
    data = np.random.rand(nby, nbx)
    # Create the Dataset
    return xr.Dataset(
        data_vars={
            "band1": (("y", "x"), data),
            "band2": (("y", "x"), data),
            "band3": (("y", "x"), data),
        },
        coords={
            "x": np.arange(nbx),
            "y": np.arange(nby),
        },
        attrs={"meta": "Test data"},
    )


def test_get_row_col():
    """
    Test get_row_col function
    """
    # Setup data
    data = setup_data()
    assert writer.get_row_col(data) == (NB_ROW, NB_COL)

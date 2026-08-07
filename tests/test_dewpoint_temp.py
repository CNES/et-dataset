# Copyright: (c) 2026 CESBIO / Centre National d'Etudes Spatiales
#
"""
This module contains tests for dewpoint temperature
"""

import numpy as np
import pytest

from etdataset import dewpoint_temp as dt


@pytest.mark.unit
def test_compute_dewpoint_temp():
    ta = np.array([20.0])  # °C
    rh = np.array([90.0])  # %
    tdp = dt.compute_dewpoint_temp(ta, rh)

    calc = np.array([18.309116])
    assert np.allclose(tdp, calc, atol=0.1)

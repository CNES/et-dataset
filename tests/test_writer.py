# Copyright: (c) 2026 CESBIO / Centre National d'Etudes Spatiales

"""
Test writer module
"""

import datetime as dt

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from shapely.geometry import Point

from etdataset import writer

NB_ROW = 30
NB_COL = 40


def setup_data(
    with_vis: bool = False, with_tir: bool = False, with_geo: bool = False
) -> xr.Dataset:
    """
    Create a xarray dataset for tests
    """
    # Set the dimensions for the dataset
    nbx = NB_COL
    nby = NB_ROW
    # Define coordinates for the raster
    lon_min, lon_max = 0, NB_COL
    lat_min, lat_max = 0, NB_ROW
    # Create random data
    data = np.random.rand(nby, nbx)
    # Create the Dataset
    xrds = xr.Dataset(
        data_vars={
            "band1": (("y", "x"), data),
            "band2": (("y", "x"), data),
            "band3": (("y", "x"), data),
        },
        coords={
            "y": np.arange(lat_max, lat_min, -((lat_max - lat_min) / NB_ROW)),
            "x": np.arange(lon_min, lon_max, (lon_max - lon_min) / NB_COL),
        },
        attrs={"meta": "Test data"},
    )
    if with_vis:
        xrds = xrds.assign_attrs(
            vis="VIS",
            vis_date=dt.date(2026, 1, 30),
            vis_time=dt.time(14, 0, 0, tzinfo=dt.UTC),
        )
    if with_tir:
        xrds = xrds.assign_attrs(
            tir="TIR",
            tir_date=dt.date(2026, 1, 30),
            tir_time=dt.time(15, 0, 0, tzinfo=dt.UTC),
        )
    if with_geo:
        # Set the spatial attributes using rioxarray
        xrds = xrds.rio.write_crs("EPSG:4326")
        xrds = xrds.rio.set_spatial_dims(x_dim="x", y_dim="y")
        xrds.rio.transform(recalc=True)
    return xrds


@pytest.mark.unit
def test_get_row_col():
    """
    Test get_row_col function
    """
    # Setup data
    data = setup_data()
    assert writer.get_row_col(data) == (NB_ROW, NB_COL)


@pytest.mark.functional
@pytest.mark.parametrize(
    ("with_vis", "with_tir", "with_geo", "bands", "separated", "expected"),
    [
        pytest.param(
            True,
            True,
            True,
            None,
            False,
            "VIS_20260130T140000+0000_TIR_20260130T150000+0000.tif",
        ),
        pytest.param(
            True,
            True,
            True,
            ["band1", "band2", "band3"],
            True,
            "VIS_20260130T140000+0000_TIR_20260130T150000+0000",
        ),
        pytest.param(
            True, False, True, None, False, "VIS_20260130T140000+0000.tif"
        ),
        pytest.param(False, False, False, ["band1"], False, "data.tif"),
    ],
)
def test_write_dataset(
    with_vis, with_tir, with_geo, bands, separated, expected, tmp_path
):
    """
    Test write_dataset
    """
    d = tmp_path / "out"
    d.mkdir()
    data = setup_data(
        with_vis=with_vis,
        with_tir=with_tir,
        with_geo=with_geo,
    )
    writer.write_dataset(
        xrds=data, bands=bands, directory=str(d), separate=separated
    )
    files = list(d.iterdir())
    assert len(files) == 1
    assert files[0].name == expected


@pytest.mark.functional
@pytest.mark.parametrize(
    ("with_vis", "with_tir", "with_geo", "band", "expected"),
    [
        pytest.param(
            True,
            True,
            True,
            "band1",
            "VIS_20260130T140000+0000_TIR_20260130T150000+0000.tif",
        ),
        pytest.param(
            True, False, True, "band2", "VIS_20260130T140000+0000.tif"
        ),
        pytest.param(False, False, False, "band1", "data.tif"),
    ],
)
def test_write_band(with_vis, with_tir, with_geo, band, expected, tmp_path):
    """
    Test write_dataset
    """
    d = tmp_path / "out"
    d.mkdir()
    data = setup_data(
        with_vis=with_vis,
        with_tir=with_tir,
        with_geo=with_geo,
    )
    writer.write_band(xrds=data, band=band, directory=str(d))
    files = list(d.iterdir())
    assert len(files) == 1
    assert files[0].name == expected


@pytest.mark.functional
def test_write_results(tmp_path):
    """
    Test write_dataset
    """
    d = tmp_path / "out"
    d.mkdir()
    res = d / "results.csv"
    data = pd.DataFrame(data={"col1": [1, 2], "col2": [3, 4]})
    writer.write_results(res=data, output=res)
    files = list(d.iterdir())
    assert len(files) == 1
    assert str(files[0]) == str(res)


@pytest.mark.functional
@pytest.mark.parametrize(
    ("with_vis", "with_tir", "with_geo", "bands", "expected"),
    [
        pytest.param(
            True,
            True,
            True,
            None,
            3,
        ),
        pytest.param(
            True,
            True,
            True,
            ["band1", "band2", "band3"],
            3,
        ),
        pytest.param(True, False, True, None, 3),
        pytest.param(False, False, False, ["band1"], 1),
    ],
)
def test_export_matlab(with_vis, with_tir, with_geo, bands, expected, tmp_path):
    """
    Test write_dataset
    """
    d = tmp_path / "out"
    d.mkdir()
    data = setup_data(
        with_vis=with_vis,
        with_tir=with_tir,
        with_geo=with_geo,
    )
    writer.export_matlab(xrds=data, bands=bands, directory=str(d))
    files = list(d.iterdir())
    assert len(files) == expected


@pytest.mark.functional
@pytest.mark.parametrize(
    ("with_vis", "with_tir", "with_geo", "bands"),
    [
        pytest.param(
            True,
            True,
            True,
            None,
        ),
        pytest.param(
            True,
            True,
            True,
            ["band1", "band2", "band3"],
        ),
        pytest.param(True, False, True, None),
        pytest.param(False, False, False, ["band1"]),
    ],
)
def test_write_to_tif(with_vis, with_tir, with_geo, bands, tmp_path):
    """
    Test write_dataset
    """
    d = tmp_path / "out"
    d.mkdir()
    filename = d / "img.tif"
    data = setup_data(
        with_vis=with_vis,
        with_tir=with_tir,
        with_geo=with_geo,
    )
    if bands is not None:
        data = data[bands]
    writer.write_to_tif(xrds=data, filename=filename)
    files = list(d.iterdir())
    assert len(files) == 1


@pytest.mark.functional
@pytest.mark.parametrize(
    ("with_vis", "with_tir", "with_geo", "bands"),
    [
        pytest.param(
            True,
            True,
            True,
            None,
        ),
        pytest.param(
            True,
            True,
            True,
            ["band1", "band2", "band3"],
        ),
        pytest.param(True, False, True, None),
        pytest.param(False, False, False, ["band1"]),
    ],
)
def test_write_to_netcdf(with_vis, with_tir, with_geo, bands, tmp_path):
    """
    Test write_dataset
    """
    d = tmp_path / "out"
    d.mkdir()
    filename = d / "img.nc"
    data = setup_data(
        with_vis=with_vis,
        with_tir=with_tir,
        with_geo=with_geo,
    )
    if bands is not None:
        data = data[bands]
    writer.write_to_netcdf(xrds=data, filename=filename)
    files = list(d.iterdir())
    assert len(files) == 1


@pytest.mark.functional
@pytest.mark.parametrize(
    ("with_vis", "with_tir", "with_geo", "bands", "expected"),
    [
        pytest.param(
            True,
            True,
            True,
            None,
            "VIS_20260130T140000+0000_TIR_20260130T150000+0000.nc",
        ),
        pytest.param(True, False, True, None, "VIS_20260130T140000+0000.nc"),
        pytest.param(False, False, False, ["band1"], "data.nc"),
    ],
)
def test_write_dataset_to_netcdf(
    with_vis, with_tir, with_geo, bands, expected, tmp_path
):
    """
    Test write_dataset
    """
    d = tmp_path / "out"
    d.mkdir()
    data = setup_data(
        with_vis=with_vis,
        with_tir=with_tir,
        with_geo=with_geo,
    )
    if bands is not None:
        data = data[bands]
    writer.write_dataset_to_netcdf(
        xrds=data,
        directory=str(d),
    )
    files = list(d.iterdir())
    assert len(files) == 1
    assert files[0].name == expected


@pytest.mark.unit
def test_create_geopckg_from_gdf(tmp_path):
    gdf = gpd.GeoDataFrame(
        {"id": ["toto"]},
        geometry=[Point(1, 2)],
        crs="EPSG:4326",
    )

    writer.create_geopckg_from_gdf(
        gdf, output="stations_package.gpkg", path=tmp_path
    )

    assert (tmp_path / "pckg" / "stations_package.gpkg").exists()

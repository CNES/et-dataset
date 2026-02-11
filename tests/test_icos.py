# type: ignore

# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales
#
"""
This module contains tests for icos
"""

import os

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from shapely.geometry import Point

from etdataset import icos
from etdataset.icos import StationConfig


def test_get_csv_with_valid_icos_stations(monkeypatch, tmp_path):
    # create your own list stations
    monkeypatch.setattr(
        icos.meta,
        "list_stations",
        lambda: [
            type(
                "s",
                (),
                {
                    "type_uri": "/ES",
                    "uri": "ES",
                    "id": "toto",
                    "name": "TOTO",
                    "country_code": "FR",
                    "lat": 0,
                    "lon": 0,
                    "elevation": 0,
                },
            )(),
            type(
                "s",
                (),
                {
                    "type_uri": "/IS",
                    "uri": "IS",
                    "id": "momo",
                    "name": "MOMO",
                    "country_code": "FR",
                    "lat": 0,
                    "lon": 0,
                    "elevation": 0,
                },
            )(),
        ],
    )
    # Create your own list_datatypes
    monkeypatch.setattr(
        icos.meta,
        "list_datatypes",
        lambda: [
            type(
                "d",
                (),
                {
                    "uri": "Meteo",
                    "label": "L2",
                },
            )(),
            type(
                "d",
                (),
                {
                    "uri": "Meteosens",
                    "label": "L2",
                },
            )(),
        ],
    )
    # create your list_data_objects
    monkeypatch.setattr(
        icos.meta,
        "list_data_objects",
        lambda **kw: ["data"] if kw["station"] == "ES" else [],
    )

    monkeypatch.chdir(tmp_path)
    # get the csv file created
    df = pd.read_csv(icos.get_csv_with_valid_icos_stations())

    # test if you only have the correct station
    assert (df["id"] == "toto").all()


def test_get_station_list(monkeypatch, tmp_path):
    df = pd.DataFrame(
        {
            "id": ["toto", "momo"],
            "name": ["TOTO", "MOMO"],
            "country": ["FR", "DE"],
        }
    )
    csv = tmp_path / "stations.csv"
    df.to_csv(csv, index=False)

    monkeypatch.setattr(
        icos,
        "get_csv_with_valid_icos_stations",
        lambda: str(csv),
    )

    stations = icos.get_station_list()

    assert stations == ["toto", "momo"]


def test_filter_stations_by_country_code(monkeypatch, tmp_path):
    df = pd.DataFrame(
        {
            "id": ["toto", "momo"],
            "country": ["FR", "DE"],
        }
    )
    csv = tmp_path / "stations.csv"
    df.to_csv(csv, index=False)

    monkeypatch.setattr(
        icos,
        "get_csv_with_valid_icos_stations",
        lambda: str(csv),
    )

    filtered = icos.filter_stations_by_country_code("FR")

    assert len(filtered) == 1
    assert filtered.iloc[0]["id"] == "toto"


def test_filter_valid_data():
    df = pd.DataFrame(
        {
            "TIMESTAMP_START": ["202301010000", "202301010030"],
            "TA": [10, -9999],
            "RH": [90, -9999],
        }
    )

    filtered = icos.filter_valid_data(df)

    assert len(filtered) == 1
    assert filtered.iloc[0]["TA"] == 10
    assert filtered.iloc[0]["RH"] == 90


def test_get_stations_config(monkeypatch, tmp_path):
    df = pd.DataFrame(
        {
            "id": ["toto"],
            "name": ["TOTO"],
            "country": ["FR"],
            "lat": [1.0],
            "lon": [2.0],
            "elev": [100],
        }
    ).set_index("id")

    csv = tmp_path / "stations.csv"
    df.to_csv(csv)

    monkeypatch.setattr(
        icos,
        "get_csv_with_valid_icos_stations",
        lambda: str(csv),
    )

    cfg = icos.get_stations_config("toto")

    assert cfg.id == "toto"
    assert cfg.country == "FR"
    assert cfg.name == "TOTO"
    assert cfg.lat == 1.0
    assert cfg.lon == 2.0
    assert cfg.elev == 100


def test_get_stations_config_unknown(monkeypatch, tmp_path):
    df = pd.DataFrame(columns=["id", "name", "country", "lat", "lon", "elev"])
    csv = tmp_path / "stations.csv"
    df.to_csv(csv, index=False)

    monkeypatch.setattr(
        icos,
        "get_csv_with_valid_icos_stations",
        lambda: str(csv),
    )

    with pytest.raises(ValueError):
        icos.get_stations_config("UNKNOWN")


def test_get_station_location(monkeypatch, tmp_path):
    df = pd.DataFrame(
        {
            "id": ["toto"],
            "name": ["TOTO"],
            "lat": [48.0],
            "lon": [2.0],
            "elev": [100],
        }
    )
    csv = tmp_path / "stations.csv"
    df.to_csv(csv, index=False)

    monkeypatch.setattr(
        icos,
        "get_csv_with_valid_icos_stations",
        lambda: str(csv),
    )

    gdf = icos.get_station_location("toto")

    assert isinstance(gdf, gpd.GeoDataFrame)
    assert isinstance(gdf.geometry.iloc[0], Point)


def test_download_icos_station_no_token(monkeypatch):
    monkeypatch.delenv("ICOS_API_TOKEN", raising=False)

    with pytest.raises(ValueError, match="ICOS_API_TOKEN is not provided"):
        icos.download_icos_station("toto")


def test_download_icos_station(monkeypatch, tmp_path):
    # fake env token
    monkeypatch.setenv("ICOS_API_TOKEN", "TOKEN")

    # fake CSV of valid stations
    df = pd.DataFrame({"id": ["toto", "momo"]})
    csv = tmp_path / "stations.csv"
    df.to_csv(csv, index=False)

    monkeypatch.setattr(
        icos,
        "get_csv_with_valid_icos_stations",
        lambda: str(csv),
    )

    # fake meta.list_datatypes
    monkeypatch.setattr(
        icos.meta,
        "list_datatypes",
        lambda: [
            type(
                "d",
                (),
                {
                    "uri": "Meteo",
                    "label": "L2",
                },
            )(),
        ],
    )

    # fake meta.list_data_objects
    monkeypatch.setattr(
        icos.meta,
        "list_data_objects",
        lambda **_: [
            type("obj", (), {"uri": "uri"})(),
        ],
    )

    calls = []

    def fake_download_file(data_objects, cookies, station_id, output):
        calls.append(
            {
                "data_objects": data_objects,
                "cookies": cookies,
                "station_id": station_id,
                "output": output,
            }
        )

    monkeypatch.setattr(icos, "download_file", fake_download_file)

    # call function
    station_ids = ["toto", "momo"]
    icos.download_icos_station(station_ids, output=str(tmp_path))

    assert len(calls) == 2

    for call, station_id in zip(calls, station_ids, strict=False):
        assert call["cookies"] == {"cpauthToken": "TOKEN"}
        assert call["station_id"] == station_id
        assert len(call["data_objects"]) == 1
        assert call["data_objects"][0].uri == "uri"


def test_read_csv_data(tmp_path):
    cfg = StationConfig(
        id="toto", name="TOTO", country="FR", lat=1, lon=2, elev=10
    )

    folder = tmp_path / "ICOS" / "ICOSETC_toto_METEO_L2"
    folder.mkdir(parents=True)

    csv = folder / "ICOSETC_toto_METEO_L2.csv"
    pd.DataFrame(
        {
            "TIMESTAMP_START": ["202301010000"],
            "TA": [10],
            "RH": [50],
        }
    ).to_csv(csv, index=False)

    data = icos.read_csv_data(cfg, path=tmp_path)

    assert list(data.columns) == ["TIMESTAMP_START", "TA", "RH"]
    assert len(data) == 1
    assert str(data.iloc[0]["TIMESTAMP_START"]) == "202301010000"
    assert data.iloc[0]["TA"] == 10
    assert data.iloc[0]["RH"] == 50


def test_create_geopckg_from_gdf(tmp_path):
    gdf = gpd.GeoDataFrame(
        {"id": ["toto"]},
        geometry=[Point(1, 2)],
        crs="EPSG:4326",
    )

    icos.create_geopckg_from_gdf(gdf, path=tmp_path)

    assert (tmp_path / "pckg" / "stations_package.gpkg").exists()


def test_save_station_data(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    cfg = StationConfig(
        id="toto", name="TOTO", country="FR", lat=0, lon=0, elev=0
    )

    df = pd.DataFrame(
        {
            "TIMESTAMP_START": ["2023"],
            "TA": [10],
            "RH": [50],
        }
    )

    csv_path = icos.save_station_data(cfg, df)

    assert os.path.exists(csv_path)


def test_compute_dewpoint_temp():
    ta = pd.Series([20.0])  # °C
    rh = pd.Series([90.0])  # %
    tdp = icos.compute_dewpoint_temp(ta, rh)

    calc = pd.Series([18.309116])
    assert isinstance(tdp, pd.Series)
    assert np.allclose(tdp, calc, atol=0.1)


def test_kelvin_to_celsius():
    data = xr.DataArray([273.15, 274.15])

    celsius = icos.kelvin_to_celsius(data)

    assert float(celsius[0]) == 0.0
    assert float(celsius[1]) == 1.0

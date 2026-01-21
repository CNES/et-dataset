# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales
#
"""
This module contains tests for icos
"""

import datetime as dt
from io import StringIO

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import pytest
import rasterio as rio
from pyproj import CRS
from shapely.geometry import Point

from etdataset import icos
from etdataset.icos import ICOSStation, ICOSVar


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


def test_load_stations_config():
    csv_data = """id,name,lat,lon,elev,country
toto,TOTO,2.3,4.5,6,FR
momo,MOMO,7.8,9.10,11,UK
"""
    csv_file = StringIO(csv_data)

    # create a test csv
    df = pd.read_csv(csv_file)
    test_csv_path = "test_stations.csv"
    df.to_csv(test_csv_path, index=False)

    result = icos.load_stations_config(test_csv_path)

    # check
    assert len(result) == 2
    assert "toto" in result
    assert "momo" in result

    # test for station toto
    toto = result["toto"]
    assert toto["name"] == "TOTO"
    assert toto["lat"] == 2.3
    assert toto["lon"] == 4.5
    assert toto["elevation"] == 6
    assert toto["crs"] == CRS.from_epsg(4326)
    assert toto["country_code"] == "FR"

    # test for station momo
    momo = result["momo"]
    assert momo["name"] == "MOMO"
    assert momo["lat"] == 7.8
    assert momo["lon"] == 9.10
    assert momo["elevation"] == 11
    assert momo["crs"] == CRS.from_epsg(4326)
    assert momo["country_code"] == "UK"


def test_download_icos_station_no_token():
    with pytest.raises(
        ValueError, match="Authentification token is not provided"
    ):
        icos.download_icos_station(None, ["toto"])


def test_download_icos_station(monkeypatch):
    # Fake meta.list_datatypes
    monkeypatch.setattr(
        icos.meta,
        "list_datatypes",
        lambda: [
            type(
                "d",
                (),
                {
                    "uri": "Meteo L2",
                    "label": "level2",
                },
            )(),
        ],
    )
    # Fake meta.list_data_objects
    monkeypatch.setattr(
        icos.meta,
        "list_data_objects",
        lambda station, datatype, order_by: [  # noqa: ARG005
            type(
                "d",
                (),
                {
                    "uri": "uri",
                },
            )(),
        ],
    )
    # Fake download_file
    calls = []

    def download_file(data_objects, cookies, station_id):
        calls.append(
            {
                "data_objects": data_objects,
                "cookies": cookies,
                "station_id": station_id,
            }
        )

    monkeypatch.setattr(icos, "download_file", download_file)

    # call of the function
    auth_token = "TOKEN"
    station_ids = ["toto", "momo"]
    icos.download_icos_station(auth_token, station_ids)

    assert len(calls) == len(station_ids)

    for i, call in enumerate(calls):
        assert call["cookies"] == {"cpauthToken": auth_token}
        assert call["station_id"] == station_ids[i]
        assert len(call["data_objects"]) == 1
        assert call["data_objects"][0].uri == "uri"


def test_get_gpkg_file(monkeypatch):
    # Create a fake cfg
    cfg = {
        "toto": {
            "id": "toto",
            "name": "TOTO",
            "lat": 2.3,
            "lon": 4.5,
            "elevation": 6,
            "crs": CRS.from_epsg(4326),
            "country_code": "FR",
        },
        "momo": {
            "id": "momo",
            "name": "MOMO",
            "lat": 7.8,
            "lon": 9.10,
            "elevation": 11,
            "crs": CRS.from_epsg(4326),
            "country_code": "UK",
        },
    }

    # fake args for the function to_file()
    args = {}

    def to_file(
        self,  # noqa: ARG001
        output,
        layer,
        driver,
    ):  # add 'self' because \
        # gdf.to_file(...) is an instance method and receives gdf as self
        args.update({"output": output, "layer": layer, "driver": driver})

    monkeypatch.setattr(gpd.GeoDataFrame, "to_file", to_file)

    station = ["toto"]
    gdf = icos.get_gpkg_file(station, cfg, output="output.gpkg")

    assert isinstance(gdf, gpd.GeoDataFrame)
    assert len(gdf) == 1
    row = gdf.iloc[0]
    assert row["id"] == "toto"
    assert row["name"] == "TOTO"
    assert row["lat"] == 2.3
    assert row["lon"] == 4.5
    assert row["elevation"] == 6
    assert isinstance(row.geometry, Point)
    assert row.geometry.x == 4.5  # longitude
    assert row.geometry.y == 2.3  # latitude
    assert row.geometry.z == 6

    assert args["output"] == "output.gpkg"
    assert args["layer"] == "stations"
    assert args["driver"] == "GPKG"


# Test for the ICOSStation class
def test_station_id_not_found():
    with pytest.raises(KeyError, match="Station 'momo' not find."):
        ICOSStation("momo", stations_cfg={})


def test_icosstation_init(monkeypatch):
    # create a dataframe with the usecols (time cols, ta and tdp cols)
    df = pd.DataFrame(
        {
            "TIMESTAMP_START": ["202303030800", "202303030830"],
            "TIMESTAMP_END": ["202303030830", "202303030900"],
            ICOSVar.AIR_TEMPERATURE.key: [10.0, -9999],  # one invalid data
            ICOSVar.RELATIVE_HUMIDITY.key: [20.0, 21.0],
        }
    )
    monkeypatch.setattr(pd, "read_csv", lambda *args, usecols: df.copy())  # noqa: ARG005
    # create your station configuration
    stations_cfg = {
        "toto": {
            "name": "TOTO",
            "lat": 1,
            "lon": 2,
            "elevation": 3,
            "crs": "EPSG:4326",
            "country_code": "FR",
        }
    }
    station = ICOSStation("toto", stations_cfg)
    assert station.id == "toto"
    assert station.name == "TOTO"
    assert (
        station.csv_path
        == "/home/mliateni/Bureau/meriem/et-dataset/notebooks/ICOS/ICOSETC_toto_METEO_L2/ICOSETC_toto_METEO_L2.csv"  # noqa: E501
    )
    assert station.latitude == 1
    assert station.longitude == 2
    assert station.elevation == 3
    assert len(station.data) == 1  # only one line left (because of the invalid\
    # data)


# test of the date_hour_filter function
def test_date_hour_filter_with_station(monkeypatch):
    df = pd.DataFrame(
        {
            "TIMESTAMP_START": [
                "202303030800",
                "202303030830",
                "202303031000",
                "202303031200",
            ],
            "TIMESTAMP_END": [
                "202303030830",
                "202303030900",
                "202303031030",
                "202303031230",
            ],
            ICOSVar.AIR_TEMPERATURE.key: [10.0, 11.0, 12.0, 13.0],
            ICOSVar.RELATIVE_HUMIDITY.key: [20.0, 21.0, 22.0, 23.0],
        }
    )

    monkeypatch.setattr(pd, "read_csv", lambda *args, usecols: df.copy())  # noqa: ARG005

    stations_cfg = {
        "toto": {
            "name": "TOTO",
            "lat": 1,
            "lon": 2,
            "elevation": 3,
            "crs": "EPSG:4326",
            "country_code": "FR",
        }
    }

    station = ICOSStation("toto", stations_cfg)

    result = station.date_hour_filter(
        start_date=dt.date(2023, 3, 3),
        end_date=dt.date(2023, 3, 3),
        hour_start=8,
        hour_end=10,
        hour_step=2,
    )

    # the only valid hours : 08:00 and 10:00
    assert list(result["TIMESTAMP_START"].dt.hour) == [8, 10]
    assert all(result["TIMESTAMP_START"].dt.minute == 0)


def test_compute_dewpoint_temp():
    ta = pd.Series([20.0])  # °C
    rh = pd.Series([90.0])  # %
    tdp = ICOSVar.compute_dewpoint_temp(ta, rh)

    calc = pd.Series([18.309116])
    assert isinstance(tdp, pd.Series)
    assert np.allclose(tdp, calc, atol=0.1)


# test of the functions for the ROI


def test_create_bbox():
    # create a new ICOS Station
    station = ICOSStation.__new__(ICOSStation)
    station.latitude = 43.604464  # lat of Toulouse
    station.longitude = 1.444243  # lon of Toulouse
    station.crs = pyproj.CRS("EPSG:4326")

    # inputs
    w = 1000
    h = 500

    # call of the function
    bbox_utm, utm_crs, bbox_lat_lon = station.create_bbox(w, h)

    assert isinstance(bbox_utm, rio.coords.BoundingBox)
    assert isinstance(bbox_lat_lon, rio.coords.BoundingBox)
    # TEST FOR THE UTM BBOX
    ### test of the width and the height of the bbox
    assert bbox_utm.right - bbox_utm.left == w
    assert bbox_utm.top - bbox_utm.bottom == h
    ### test of the localisation of the station
    x_center = (bbox_utm.left + bbox_utm.right) / 2
    y_center = (bbox_utm.bottom + bbox_utm.top) / 2

    transformer = pyproj.Transformer.from_crs(
        station.crs, utm_crs, always_xy=True
    )
    x_station, y_station = transformer.transform(
        station.longitude, station.latitude
    )
    assert x_center == pytest.approx(x_station, abs=1)
    assert y_center == pytest.approx(y_station, abs=1)

    # TEST FOR THE LAT/LON BBOX

    assert bbox_lat_lon.left <= station.longitude <= bbox_lat_lon.right
    assert bbox_lat_lon.bottom <= station.latitude <= bbox_lat_lon.top

    ### test of the localisation of the station
    lon_center = (bbox_lat_lon.left + bbox_lat_lon.right) / 2
    lat_center = (bbox_lat_lon.bottom + bbox_lat_lon.top) / 2

    assert lon_center == pytest.approx(station.longitude, abs=1e-4)
    assert lat_center == pytest.approx(station.latitude, abs=1e-4)

    transformer = pyproj.Transformer.from_crs(
        station.crs, utm_crs, always_xy=True
    )

    x_min, y_min = transformer.transform(bbox_lat_lon.left, bbox_lat_lon.bottom)
    x_max, y_max = transformer.transform(bbox_lat_lon.right, bbox_lat_lon.top)
    ### test of the width and the height of the bbox
    assert (x_max - x_min) == pytest.approx(1000, abs=1)
    assert (y_max - y_min) == pytest.approx(500, abs=1)

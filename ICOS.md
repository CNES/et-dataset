# ICOS


**Table of Contents**

- [Introduction](#introduction)
- [ICOS stations](#icos_stations)
- [Digital Elevation Models (DEM)](#digital-elevation-models-dem)


## Introduction
This part compares ICOS meteorological data (air temperature and relative humidty) with ERA5 reanalysis data, using two approaches:

* Direct ERA5 extraction (coarse resolution : 25 km x 25 km)
* Downscaling ERA5 extraction with high-resolution DEM-based spatial interpolation at 60 m resolution

The aim is to evaluate whether downscaling improves the match with ICOS measurements, based on:

* Time series comparison (seasonal/monthly performance)
* Error metrics : RMSE, R2, slope, MAE, MBE.

The goal of the script is to:

* Download and load ICOS weather data
* Extract ERA5 weather fields around each ICOS station
* Compare Ta and Td (air and dewpoint temperature) of ICOS vs ERA5 vs ERA5 downscaled



## ICOS stations

The Integrated Carbon Observation System (ICOS) is a European research infrastructure that provides long-term, high-precision, and standardized measurements of ecosystem–atmosphere exchanges. More information is available at:  https://www.icos-cp.eu/.

### Access to ICOS data

The ICOS data used in this study were accessed through the ICOS Carbon Portal, using the icoscp_core Python package. This library provides programmatic access to ICOS metadata and observational datasets, facilitating reproducible workflows and transparent data retrieval.

Access to ICOS data requires user authentication via the ICOS Carbon Portal. Users must create an account on the ICOS website and generate a personal access token, which is used for programmatic access to the data. This token is required by the icoscp_core Python package to enable remote queries and data downloads.

The following ICOS stations were selected to represent contrasting climatic and topographic conditions and were used to evaluate the impact of ERA5 downscaling:


* Gebesee station :
https://meta.icos-cp.eu/objects/uFKBeFR9-DR0v5FB97EfMyAK
    crs: 4326
    lat: 51.099735°
    lon: 10.914622°
    elevation: 163.0 m


* Davos station :
https://meta.icos-cp.eu/objects/xJMiP_jEL-xSF8c4mEIyNTdB
    crs: 4326
    lat: 46.81533°
    lon: 9.85591°
    elevation: 1637.0 m

* Borgo Cioffi station :
https://meta.icos-cp.eu/objects/HrsikDdZY3BlFe8PJ8g11wgC
    crs: 4326
    lat: 40.52375°
    lon: 14.957444°
    elevation: 10.0 m


* Col du Lautaret station :
https://meta.icos-cp.eu/objects/-fyKVR2bhCNaofuXT4hlL5I4
    crs: 4326
    lat: 45.0414°
    lon: 6.41053°
    elevation: 2050.6°


### Digital Elevation Models (DEM)

At this stage, Copernicus GLO-30 Digital Elevation Models (DEMs) are extracted from **OpenTopography** : https://portal.opentopography.org/raster?opentopoID=OTSDEM.032021.4326.3 .
The DEM files are stored locally following the directory structure below:

```text
notebooks/
├── rasters_COP30/
│   ├── CH-Dav/
│   │   └── output_hh.tif
│   ├── DE-Geb/
│   │   └── output_hh.tif
│   ├── IT-BCi/
│   │   └── output_hh.tif
│   └── FR-CLt/
│       └── output_hh.tif
```

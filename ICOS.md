# ICOS


**Table of Contents**

- [Introduction](#introduction)
- [ICOS stations](#icos_stations)
- [ICOS data preparation](#icos-data-preparation)


## Introduction
This part compares ICOS meteorological data (air temperature and relative humidty).

The goal of the script is to:

- Download and load ICOS weather data (air temperature and relative humidity)
- Filter and format valid observations
- Compute ICOS dew point temperature

## ICOS stations

The Integrated Carbon Observation System (ICOS) is a European research infrastructure that provides long-term, high-precision, and standardized measurements of ecosystem–atmosphere exchanges. More information is available at: https://www.icos-cp.eu/.

### Access to ICOS data

The ICOS data used in this study were accessed through the ICOS Carbon Portal using the 'icoscp_core' Python package. This library provides programmatic access to ICOS metadata and observational datasets.

Access to ICOS data requires user authentication via the ICOS Carbon Portal. Users must create an account on the ICOS website and generate a 24h personal access token, which is used for programmatic access to the data. This token is required by the 'icoscp_core' Python package to enable remote queries and data downloads.

## ICOS data preparation

ICOS valid stations are selected using the function `get_csv_with_valid_icos_stations()` located in `src/etdataset/icos.py`.

For each selected station:

- meteorological data are downloaded using
  `download_icos_station()` (`src/etdataset/icos.py`)
- only the following variables are retained by
  `read_csv_data()` (`src/etdataset/icos.py`):
  - air temperature
  - relative humidity
- only valid observations are kept using
  `filter_valid_data()` (`src/etdataset/icos.py`)
- dew point temperature is computed from air temperature and relative humidity using
  `compute_dewpoint_temp()` (`src/etdataset/icos.py`)
- the processed data are saved to a station-specific CSV file using
  `save_station_data()` (`src/etdataset/icos.py`)

All steps related to downloading, filtering, and formatting ICOS data are implemented in
`scripts/prepare_icos.py`.

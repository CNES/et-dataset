# ET Dataset


**Table of Contents**

- [Introduction](#introduction)
- [Pre-requistes](#pre-requistes)
- [Installation](#installation)
- [Usage](#usage)


## Introduction

Python package to prepare datasets for evapotranspiration processing:
* Search on catalogs that satisfy criteria
* Download products from a product list
* Download auxiliary data from a product list
* Select products between 2 collections that satisfy criteria
* Create ET dataset from products

## Pre-requistes

To search and download products and auxiliary data, you need accounts and set environment variables corresponding to login and password for each catalog:
* [USGS machine to machine](https://m2m.cr.usgs.gov/): required to define `USGS_USERNAME` and `USGS_PASSWORD` (the api key is used as a password here)
* [THEIA](https://theia.cnes.fr/atdistrib/rocket/#/search?collection=SENTINEL2): required to define `THEIA_IDENT` and `THEIA_PASS`
* [Earth Data](https://search.earthdata.nasa.gov/search): required to define `EARTHDATA_USERNAME` and `EARTHDATA_PASSWORD`
* [LSA SAF Data](https://datalsasaf.lsasvcs.ipma.pt/): required to define `LSASAF_USER` and `LSASAF_PASSWORD`
* [Climate Data Store](https://cds.climate.copernicus.eu/): required to configure CDS API ([see instructions](https://cds.climate.copernicus.eu/how-to-api))

## Installation

The installation requires to have:
* **git**
* **pixi**. The instructions to install pixi can be found in [https://pixi.sh/](https://pixi.sh/).

### Clone the repository

```console
git clone https://src.koda.cnrs.fr/trishna/et-dataset.git
```

### Install

The dependencies are installed by pixi.
```console
cd et-dataset
pixi install
```

To activate the environment
```console
pixi shell
```

To use notebook
```console
pip install .[notebook]
```

To install in a developper mode
```console
pixi install -e dev
```

### Update

For a classic installation
```console
cd et-dataset
git pull
pixi shell
pip install .[notebook]
```

For a developper installtion
```console
cd et-dataset
git pull
pixi shell -e dev
```

## Usage

### Search products

The command enables to search products in catalogs:
* For Landsat [USGS machine to machine](https://m2m.cr.usgs.gov/)
* For Sentinel2 [THEIA](https://theia.cnes.fr/atdistrib/rocket/#/search?collection=SENTINEL2)
* For ECOSTRESS (collection v2) and HLS [Earth Data](https://search.earthdata.nasa.gov/search)

#### Command line
```bash
-> et-dataset search -h
usage: et-dataset search [-h] [-v] [-c {Collection.ECOSTRESS,Collection.LANDSAT,Collection.SENTINEL2,Collection.HLSS,Collection.HLSL}] --min_date MIN_DATE --max_date MAX_DATE [--max_cloud_cover MAX_CLOUD_COVER] (-t TILE | -r ROI) [--output OUTPUT]

options:
  -h, --help            show this help message and exit
  -v, --verbose         Verbose mode
  -c {Collection.ECOSTRESS,Collection.LANDSAT,Collection.SENTINEL2,Collection.HLSS,Collection.HLSL}, --collection {Collection.ECOSTRESS,Collection.LANDSAT,Collection.SENTINEL2,Collection.HLSS,Collection.HLSL}
                        Collection of products
  --min_date MIN_DATE   Minimum date for acquisition in YYYY-MM-DD format
  --max_date MAX_DATE   Maximum date for acquisition in YYYY-MM-DD format
  --max_cloud_cover MAX_CLOUD_COVER
                        Maximum cloud cover (only for Landsat8 products) (default: 25)
  -t TILE, --tile TILE  Tile ID
  -r ROI, --roi ROI     Path of the region of interest in Shapefile format
  --output OUTPUT       CSV output file (default: results.csv)
```

#### Notebooks

* [search.ipynb](notebooks/search.ipynb): Examples of searches in various catalog
* [search_landsat.ipynb](notebooks/search_landsat.ipynb): Examples of searches for Landsat


### Download products

The command enables to download products from a product list from catalogs:
* For Landsat [USGS machine to machine](https://m2m.cr.usgs.gov/)
* For Sentinel2 [THEIA](https://theia.cnes.fr/atdistrib/rocket/#/search?collection=SENTINEL2)
* For ECOSTRESS (collection v2) and HLS [Earth Data](https://search.earthdata.nasa.gov/search)

The product list must be provided in a CSV file format.
The required column are *Product_name* and *URL* which must contain the URLs to download a product.
The *URL* column can be a str or a list of str.

#### Command line
```bash
usage: et-dataset download [-h] [-v] -l LIST [--output OUTPUT]

options:
  -h, --help            show this help message and exit
  -v, --verbose         Verbose mode
  -l LIST, --list LIST  List of products (in CSV format)
  --output OUTPUT       Download directory path (default: download)
```

#### Notebooks

* [download.ipynb](notebooks/download.ipynb): Examples of download products

### Download auxiliary data

The command enables to download auxiliary data from a product list coming from:
* [LSA SAF Data](https://datalsasaf.lsasvcs.ipma.pt/)
* [Climate Data Store](https://cds.climate.copernicus.eu/)

The product list must be provided in a CSV file format.
The required column is *Date* which must contain the date of the products.

#### Command line
```bash
usage: et-dataset download [-h] [-v] -l LIST [--output OUTPUT]

options:
  -h, --help            show this help message and exit
  -v, --verbose         Verbose mode
  -l LIST, --list LIST  List of products (in CSV format)
  --output OUTPUT       Download directory path (default: download)
```

#### Notebooks

* [download.ipynb](notebooks/download.ipynb): Examples of download products and auxiliary data


### Select products

The command give the list of products in two collections that satisfy criteria.
The products can then be downloaded via the download command.

#### Command line

``` bash
usage: et-dataset select [-h] [-v]
                         [--coll1 {Collection.ECOSTRESS,Collection.LANDSAT,Collection.SENTINEL2,Collection.HLSSENTINEL2,Collection.HLSLANDSAT}]
                         [--coll2 {Collection.ECOSTRESS,Collection.LANDSAT,Collection.SENTINEL2,Collection.HLSSENTINEL2,Collection.HLSLANDSAT}]
                         --min_date MIN_DATE --max_date MAX_DATE
                         [--max_cloud_cover MAX_CLOUD_COVER] [--delta DELTA]
                         (-t TILE | -r ROI)
                         [--min_roi_overlap MIN_ROI_OVERLAP]
                         [--min_product_overlap MIN_PRODUCT_OVERLAP]
                         [--output OUTPUT]

options:
  -h, --help            show this help message and exit
  -v, --verbose         Verbose mode
  --coll1 {Collection.ECOSTRESS,Collection.LANDSAT,Collection.SENTINEL2,Collection.HLSSENTINEL2,Collection.HLSLANDSAT}
                        Fisrt collection of products
  --coll2 {Collection.ECOSTRESS,Collection.LANDSAT,Collection.SENTINEL2,Collection.HLSSENTINEL2,Collection.HLSLANDSAT}
                        Second collection of products
  --min_date MIN_DATE   Minimum date for acquisition in YYYY-MM-DD format
  --max_date MAX_DATE   Maximum date for acquisition in YYYY-MM-DD format
  --max_cloud_cover MAX_CLOUD_COVER
                        Maximum cloud cover (only for Landsat8 products)
                        (default: 25)
  --delta DELTA         Maximum time delta allowed between acquisitions
                        (default: 3 days)
  -t TILE, --tile TILE  Tile ID
  -r ROI, --roi ROI     Path of the region of interest in Shapefile format
  --min_roi_overlap MIN_ROI_OVERLAP
                        Minimum overlap between ROI and a product (default:
                        50)
  --min_product_overlap MIN_PRODUCT_OVERLAP
                        Minimum overlap between two products (default: 40)
  --output OUTPUT       Output dir (current directory)
```

#### Notebooks

* [select_ecols8_matches.ipynb](notebooks/select_ecols8_matches.ipynb): Find matches between ECOSTRESS/Sentinel2
* [select_ecos2_matches.ipynb](notebooks/select_ecos2_matches.ipynb): Find matches between ECOSTRESS/Landsat

### Create command

This command create a dataset by resampling on the same grid a VIS product and a TIR product.

Compatible VIS products are:
* Landsat L2A
* Sentinel2 L2A

Compatible TIR products are:
* Landsat L2A
* ECOSTRESS collectin V2 only

LAI is calculated from the spectral reflectances with the [pyBVNET](https://forge.ird.fr/cesbio/modelisation/pybvnet) package, using the BVNET neural networks.

Albedo is computed with these following formula:

* For Landsat (*Liang, S. Narrowband to Broadband Conversions of Land Surface Albedo I: Algorithms. Remote Sens. Environ. 2001, 76, 213–238.*)
```math
Albedo = 0.356 * band_B2 + 0.130 * band_B4 + 0.373 * band_5 + 0.085 * band_B6 + 0.072 * band_B7 - 0.0018
```
* For Sentinel2 (*Bonafoni and al., Albedo Retrieval From Sentinel-2 by New Narrow-to-Broadband Conversion Coefficients, IEEE Geoscience and Remote Sensing Letters, 2020*)
```math
Albedo = 0.2266 * band_B2 + 0.1236 * band_B3 + 0.1573 * band_B4 + 0.3417 * band_B8 + 0.1170 * band_B11 +  0.0338 * band_B12
```

#### Command line

```bash
-> et-dataset create -h
usage: et-dataset create [-h] [-v] --vis VIS [--tir TIR] [-t TILE] [--output OUTPUT] [-m]

Create dataset from Landsat/Ecostress/Sentinel2 products. The dataset is resampling at 60m resolution and corresponds to a MGRS tile.

options:
  -h, --help            show this help message and exit
  -v, --verbose         Verbose mode
  --vis VIS             Path to VIS product (optical bands)
  --tir TIR             Path to TIR product (thermal bands), if not provided the VIS product is used for thermal bands.
  -t TILE, --tile TILE  Tile ID (Only for Landsat)
  --output OUTPUT       Output dataset directory path (default: current directory)
  --aux                 Add auxiliary data
  -m, --matlab_export   Write the dataset in Matlab format

```

#### Notebooks

**Warning: Notebooks have to be updated**
* [create_ecols8_dataset.ipynb](notebooks/create_ecols8_dataset.ipynb): Create dataset with ECOSTRESS and Landsat
* [create_ecos2_dataset.ipynb](notebooks/create_ecos2_dataset.ipynb): Create dataset with ECOSTRESS and Sentinel2
* [create_ls8_dataset.ipynb](notebooks/create_ls8_dataset.ipynb): Create dataset with Landsat

## Data preparation

A complete notebook is available to describe landsat data preparation: [prepare_landsat.ipynb](notebooks/prepare_landsat.ipynb).

## Scripts

Several scripts are available:
* [prepare_landsat.py](scripts/prepare_landsat.py): Prepare landsat data
* [get_dem.py](scripts/get_dem.py): Extract DEM based on a ROI from the copernicus DEM.
* [copy_dem.py](scripts/copy_dem.py): Copy DEM tiles from the tiled copernicus DEM to a directory.

### Prepare landsat

```bash
python scripts/prepare_landsat.py -h
usage: prepare_landsat.py [-h] [-v] -r ROI -s START_DATE -e END_DATE [-o OUTPUT] [--max_cloud_cover MAX_CLOUD_COVER] [--min_roi_overlap MIN_ROI_OVERLAP] [--mnt_path MNT_PATH]

Prepare Landsat data

options:
  -h, --help            show this help message and exit
  -v, --verbose         Verbose mode
  -r ROI, --roi ROI     Path of the region of interest in Shapefile format
  -s START_DATE, --start-date START_DATE
                        Start date (YYYY-MM-DD)
  -e END_DATE, --end-date END_DATE
                        End date (YYYY-MM-DD)
  -o OUTPUT, --output OUTPUT
                        Output dataset directory path (default: current directory)
  --max_cloud_cover MAX_CLOUD_COVER
                        Maximum cloud cover (default: 25)
  --min_roi_overlap MIN_ROI_OVERLAP
                        Minimum overlap between ROI and a product (default: 33)
  --mnt_path MNT_PATH   Directory of DEM tiles
```

Example
```bash
python scripts/prepare_landsat.py -r notebooks/data/Zone_Senegal_Centre.shp -s "2023-03-01" -e "2023-03-12" -o notebooks/out --mnt_path $HOME/data/MNT/DEM_Copercinus_30m/
```

### Get DEM

```bash
python scripts/get_dem.py -h
usage: get_dem.py [-h] [-v] -r ROI --mnt_path MNT_PATH [-f {tif,netcdf}]

Get DEM

options:
  -h, --help            show this help message and exit
  -v, --verbose         Verbose mode
  -r ROI, --roi ROI     Path of the region of interest in Shapefile format
  --mnt_path MNT_PATH   Path to the DEM directory
  -f {tif,netcdf}, --format {tif,netcdf}
                        Output format (tif, netcdf)
```

### Copy DEM

```bash
python scripts/copy_dem.py -h
usage: copy_dem.py [-h] [-v] -r ROI --mnt_path MNT_PATH [-o OUTPUT]

Copy DEM tiles

options:
  -h, --help            show this help message and exit
  -v, --verbose         Verbose mode
  -r ROI, --roi ROI     Path of the region of interest in Shapefile format
  --mnt_path MNT_PATH   Path to the DEM directory
  -o OUTPUT, --output OUTPUT
                        Output directory
```

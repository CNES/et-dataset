# ET Dataset


**Table of Contents**

- [Introduction](#introduction)
- [Pre-requistes](#pre-requistes)
- [Installation](#installation)
- [Usage](#usage)


## Introduction

Python package to prepare datasets for evapotranspiration processing:
* Find products
* Create ET dataset

## Pre-requistes

* Export metadata for Landsat and ECOSTRESS collections on Earth Explorer 
* Export Sentinel2 metadata with datalakeutils

## Installation

### Create virtual env

```bash
python -m venv venv-etdataset
source venv-etdataset/bin/activate
```
### Install sensorsio

```bash
git clone https://framagit.org/jmichel-otb/sensorsio.git
cd sensorsio
pip install .
cd ..
```

### Install et-dataset

```bash
git clone https://gitlab.cnes.fr/cesbio/et-dataset.git
cd et-dataset
pip install .
```

With notebooks :
```bash
pip install .[notebook]
```

## Usage

### Find products

The command give the list of products that satisfy criteria.
For now, the products must be manually download :
* For Landsat [Earth Explorer](https://earthexplorer.usgs.gov/)
* For Sentinel2 [THEIA](https://theia.cnes.fr/atdistrib/rocket/#/search?collection=SENTINEL2)
* For ECOSTRESS (collection v2) [Earth Data](https://search.earthdata.nasa.gov/search)

#### Command line

``` bash
et-dataset find -h
usage: et-dataset find [-h] [-v] [-l] [-s] [-e] --min_date MIN_DATE --max_date MAX_DATE [--max_cloud_cover MAX_CLOUD_COVER] [--delta DELTA] (-t TILE | -r ROI) [--min_roi_overlap MIN_ROI_OVERLAP]
                       [--min_product_overlap MIN_PRODUCT_OVERLAP] [--metadata METADATA] [--output OUTPUT]

Find matches between products that satisfy required criteria.

options:
  -h, --help            show this help message and exit
  -v, --verbose         Verbose mode
  -l, --landsat         Find among Landsat products
  -s, --sentinel2       Find among Sentinel2 products
  -e, --ecostress       Find among Ecostress products
  --min_date MIN_DATE   Minimum date for acquisition in YYYY-MM-DD format
  --max_date MAX_DATE   Maximum date for acquisition in YYYY-MM-DD format
  --max_cloud_cover MAX_CLOUD_COVER
                        Maximum cloud cover (only for Landsat8 products) (default: 25)
  --delta DELTA         Maximum time delta allowed between acquisitions (default: 3 days)
  -t TILE, --tile TILE  Tile ID
  -r ROI, --roi ROI     Path of the region of interest in Shapefile format
  --min_roi_overlap MIN_ROI_OVERLAP
                        Minimum overlap between ROI and a product (default: 50)
  --min_product_overlap MIN_PRODUCT_OVERLAP
                        Minimum overlap between two products (default: 40)
  --metadata METADATA   Path to the directory containing metadata csv files
  --output OUTPUT       CSV output file (default: matches.csv)
```
  
Example: Find ECOSTRESS/Sentinel2 product matches between 2023/02/15 and 2023/04/14 for tile 28PCA
```bash
et-dataset find  --ecostress --sentinel2 --min_date 2023-02-15 --max_date 2023-04-14 -t 28PCA
```
  
Example: Find Landsat products between 2023/01/31 and 2023/04/30 over a ROI with cloud cover percentage below 10%
```bash
et-dataset find --landsat --min_date 2023-01-31 --max_date 2023-04-30 --max_cloud_cover 10 -r $DATA_PATH/zones/Zone_Senegal_Centre.shp
```

#### Notebooks

* find_ecols8_matches.ipynb: Find matches between ECOSTRESS/Sentinel2
* find_ecos2_matches.ipynb: Find matches between ECOSTRESS/Landsat 
* find_ls8_matches.ipynb: Find Landsat products

### Create command

This command create a dataset by resampling on the same grid a VIS product and a TIR product.

Compatible VIS products are:
* Landsat L2A
* Sentinel2 L2A

Compatible TIR products are:
* Landsat L2A
* ECOSTRESS collectin V2 only 

LAI is calculated from NDVI with the following formula:
```math 
LAI = 0.119 * exp(3.457 * NDVI) - 0.062
```

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
et-dataset create -h
usage: et-dataset create [-h] [-v] --vis VIS [--tir TIR] [-t TILE] [--output OUTPUT] [-m]

Create dataset from Landsat/Ecostress/Sentinel2 products. The dataset is resampling at 60m resolution and corresponds to a MGRS tile.

options:
  -h, --help            show this help message and exit
  -v, --verbose         Verbose mode
  --vis VIS             Path to VIS product (optical bands)
  --tir TIR             Path to TIR product (thermal bands), if not provided the VIS product is used for thermal bands.
  -t TILE, --tile TILE  Tile ID (Only for Landsat)
  --output OUTPUT       Output dataset directory path (default: current directory)
  -m, --matlab_export   Write the dataset in Matlab format

```

#### Notebooks

* create_ecols8_dataset.ipynb: Create dataset with ECOSTRESS and Landsat 
* create_ecos2_dataset.ipynb: Create dataset with ECOSTRESS and Sentinel2 
* create_ls8_dataset.ipynb: Create dataset with Landsat 
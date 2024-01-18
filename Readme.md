# Create dataset for evapotranspiration

## Generate database

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
pip install -e .
cd ..
```

### Install et-dataset

```bash
git clone https://gitlab.cnes.fr/cesbio/et-dataset.git
cd et-dataset
export PYTHON=$PWD/src/:$PYTHONPATH
pip install -r requirements.txt
```

## Find matches

Do not forget to set `METADATA_PATH` to indicate the directory containing CSV exports of Landsat/ECOSTRESS/Sentinel2.


```bash
% python find_ls8_matches.py -h
usage: find_ls8_matches.py [-h] [-v] [-l LANDSAT8_CSV] [--min_date MIN_DATE] [--max_date MAX_DATE] [--max_cloud_cover MAX_CLOUD_COVER] (-t TILE | -r ROI) [--min_roi_overlap MIN_ROI_OVERLAP] [--output OUTPUT]

List Landsat products

options:
  -h, --help            show this help message and exit
  -v, --verbose         Verbose mode
  -l LANDSAT8_CSV, --landsat8_csv LANDSAT8_CSV
                        Path to csv file with landsat C2L2 earth explorer archive export
  --min_date MIN_DATE   Minimum date for acquisition
  --max_date MAX_DATE   Maximum date for acquisition
  --max_cloud_cover MAX_CLOUD_COVER
                        Maximum cloud cover for Landsat8 products
  -t TILE, --tile TILE  Tile ID
  -r ROI, --roi ROI     Region of interest in Shapefile format
  --min_roi_overlap MIN_ROI_OVERLAP
                        Minimum ROI overlap with Landsat8 products
  --output OUTPUT       CSV output file
```

```bash
-> python find_ecols8_matches.py -h
usage: find_ecols8_matches.py [-h] [-v] [-l LANDSAT8_CSV] [-e ECOSTRESS_CSV] [--min_date MIN_DATE] [--max_date MAX_DATE] [--max_cloud_cover MAX_CLOUD_COVER] [--delta DELTA] (-t TILE | -r ROI) [--min_roi_overlap MIN_ROI_OVERLAP] [--output OUTPUT]

Find ECOSTRESS/Landsat matches

options:
  -h, --help            show this help message and exit
  -v, --verbose         Verbose mode
  -l LANDSAT8_CSV, --landsat8_csv LANDSAT8_CSV
                        Path to csv file with landsat C2L2 earth explorer archive export
  -e ECOSTRESS_CSV, --ecostress_csv ECOSTRESS_CSV
                        Path to csv file with ECOSTRESS earth explorer archive export
  --min_date MIN_DATE   Minimum date for acquisition
  --max_date MAX_DATE   Maximum date for acquisition
  --max_cloud_cover MAX_CLOUD_COVER
                        Maximum cloud cover for Landsat8 products
  --delta DELTA         Maximum time delta allowed between ecostress and landsat8 acquisitions
  -t TILE, --tile TILE  Tile ID
  -r ROI, --roi ROI     Region of interest in Shapefile format
  --min_roi_overlap MIN_ROI_OVERLAP
                        Minimum ROI overlap with Landsat8 products
  --output OUTPUT       CSV output file
```

```bash
python find_ecos2_matches.py -h
usage: find_ecos2_matches.py [-h] [-v] [-s SENTINEL2_CSV] [-e ECOSTRESS_CSV] [--min_date MIN_DATE] [--max_date MAX_DATE] [--delta DELTA] (-t TILE | -r ROI) [--min_roi_overlap MIN_ROI_OVERLAP] [--output OUTPUT]

Find ECOSTRESS/Sentinel2 matches

options:
  -h, --help            show this help message and exit
  -v, --verbose         Verbose mode
  -s SENTINEL2_CSV, --sentinel2_csv SENTINEL2_CSV
                        Path to csv file with Sentinel2 archive export
  -e ECOSTRESS_CSV, --ecostress_csv ECOSTRESS_CSV
                        Path to csv file with ECOSTRESS earth explorer archive export
  --min_date MIN_DATE   Minimum date for acquisition
  --max_date MAX_DATE   Maximum date for acquisition
  --delta DELTA         Maximum time delta allowed between ecostress and landsat8 acquisitions
  -t TILE, --tile TILE  Tile ID
  -r ROI, --roi ROI     Region of interest in Shapefile format
  --min_roi_overlap MIN_ROI_OVERLAP
                        Minimum ROI overlap with Landsat8 products
  --output OUTPUT       CSV output file
```

## Create dataset

```bash
python create_dataset_from_landsat8.py -h
usage: create_dataset_from_landsat8.py [-h] [-v] -l LANDSAT8 (-t TILE | -r ROI) [--output OUTPUT]

Create dataset from landsat8 product

options:
  -h, --help            show this help message and exit
  -v, --verbose         Verbose mode
  -l LANDSAT8, --landsat8 LANDSAT8
                        Path to landsat8 product
  -t TILE, --tile TILE  Tile ID
  -r ROI, --roi ROI     Region of interest in Shapefile format
  --output OUTPUT       Output dataset dir path
```

```bash
python create_dataset_from_eco_and_ls8.py -h
usage: create_dataset_from_eco_and_ls8.py [-h] [-v] -l LANDSAT8 -e ECOSTRESS [--output OUTPUT]

Create dataset from landsat8 and ecostress products

options:
  -h, --help            show this help message and exit
  -v, --verbose         Verbose mode
  -l LANDSAT8, --landsat8 LANDSAT8
                        Path to landsat8 product
  -e ECOSTRESS, --ecostress ECOSTRESS
                        Path to ecostress product
  --output OUTPUT       Output dataset dir path
```

```bash
python create_dataset_from_eco_and_s2.py -h
usage: create_dataset_from_eco_and_s2.py [-h] [-v] -s SENTINEL2 -e ECOSTRESS [--output OUTPUT]

Create dataset from sentinel2 and ecostress products

options:
  -h, --help            show this help message and exit
  -v, --verbose         Verbose mode
  -s SENTINEL2, --sentinel2 SENTINEL2
                        Path to sentinel2 product
  -e ECOSTRESS, --ecostress ECOSTRESS
                        Path to ecostress product
  --output OUTPUT       Output dataset dir path
```

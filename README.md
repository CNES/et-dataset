# ET Dataset


**Table of Contents**

- [Introduction](#introduction)
- [Pre-requistes](#pre-requistes)
- [Installation](#installation)


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
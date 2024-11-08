#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright: (c) 2023 CESBIO / Centre National d'Etudes Spatiales / Université Paul Sabatier (UT3)
#
"""
Remote sensing products
"""

import datetime
from abc import abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Type

import affine
import numpy as np
import rasterio as rio
import xarray as xr
from sensorsio import ecostress_v2, hls, landsat, mgrs, sentinel2, utils

from etdataset.logging import LoggerManager
from etdataset.vegetation_indices import compute_lai_from_ndvi, compute_ndvi

logger = LoggerManager.get_logger(__name__)


class ProductReaderException(Exception):
    """
    Exception for ReaderProduct
    """


@dataclass
class ProductReader:
    """
    Abstract class for product reader
    """

    path: str
    bb: rio.coords.BoundingBox = field(init=False)
    crs: str = field(init=False)
    _tile: str = field(init=False)
    date: datetime.date = field(init=False)
    time: datetime.time = field(init=False)
    resolution: int = field(default=60)

    @property
    def tile(self):
        return self._tile

    @tile.setter
    def tile(self, tile_id: str):
        """
        Set MGRS tile
        """
        self._tile = tile_id

    def rename_bands(self, data: xr.Dataset, name_mapping: dict) -> xr.Dataset:
        """
        Rename bands
        """
        # Apply name mapping
        renamed = data.rename_vars(name_mapping)
        # Remove any variable not listeed in tir_band_mapping
        for var in renamed.data_vars.keys():
            if var not in name_mapping.values():
                renamed = renamed.drop(var)
        return renamed

    @abstractmethod
    def read_vis_bands(self) -> xr.Dataset:
        """
        Read VIS bands
        """
        pass

    @abstractmethod
    def read_tir_bands(self) -> xr.Dataset:
        """
        Read TIR bands
        """
        pass


@dataclass
class LandsatReader(ProductReader):
    """
    Reader for Landsat product
    """

    ds: landsat.Landsat = field(init=False)

    vis_band_mapping = {
        "SR_B2": "blue",
        "SR_B3": "green",
        "SR_B4": "red",
        "SR_B5": "nir",
        "SR_B6": "swir1",
        "SR_B7": "swir2",
        "cloud": "cloud",
        "water": "water",
        "qa": "qa",
    }
    tir_band_mapping = {
        "ST_B10": "lst",
        "ST_EMIS": "emis",
        "cloud": "cloud",
        "water": "water",
        "qa": "qa",
    }

    def __post_init__(self):
        """
        Initialize dataset
        """
        # Create an instance of Landsat8 from the product path
        self.ds = landsat.Landsat(self.path)
        self._tile = None
        self.date = self.ds.date
        self.time = self.ds.time

    @property
    def tile(self):
        return self._tile

    @tile.setter
    def tile(self, tile_id: str):
        """
        Set MGRS tile
        """
        self._tile = tile_id
        # Get bounding box for MRGS tile
        bb = mgrs.get_bbox_mgrs_tile(tile_id, False)
        self.bb = utils.bb_transform(
            mgrs.get_crs_mgrs_tile(tile_id).to_string(), self.ds.crs, bb
        )
        # Snap bbox
        self.bb = utils.bb_snap(bb, align=self.resolution)
        self.crs = self.ds.crs

    def compute_albedo(self, data: xr.Dataset) -> xr.DataArray:
        """
        Compute albedo
        Liang, S. Narrowband to Broadband Conversions of
        Land Surface Albedo I: Algorithms. Remote Sens. Environ. 2001, 76, 213–238.
        """
        return (
            0.356 * data.blue
            + 0.130 * data.red
            + 0.373 * data.nir
            + 0.085 * data.swir1
            + 0.072 * data.swir2
            - 0.0018
        )

    def read_vis_bands(self) -> xr.Dataset:
        """
        # Read landsat data
        # Every bands in the product is sampled at 30m
        # RGB: B4, B3, B2
        # NIR: B5
        # SWIR: B6,B7
        """
        ls_xr = self.ds.read_as_xarray(
            [
                landsat.Landsat.B1,
                landsat.Landsat.B2,
                landsat.Landsat.B3,
                landsat.Landsat.B4,
                landsat.Landsat.B5,
                landsat.Landsat.B6,
                landsat.Landsat.B7,
            ],
            resolution=self.resolution,
            crs=self.crs,
            bounds=self.bb,
            algorithm=rio.enums.Resampling.average,
        )
        if ls_xr is None:
            raise ValueError(f"No data found ({self.path})")

        # Add transform
        ls_xr.attrs["transform"] = affine.Affine(
            self.resolution, 0.0, self.bb.left, 0.0, -self.resolution, self.bb.top
        )
        # Add capteur name
        ls_xr.attrs["vis"] = "Landsat"
        # Add acquisition date
        ls_xr.attrs["vis_date"] = self.ds.date
        ls_xr.attrs["vis_time"] = self.ds.time
        # Add tile id
        ls_xr.attrs["tile"] = self.tile

        # Retrieve masks
        # Convention 1 for masked pixels
        ls_xr = ls_xr.assign(
            dict(
                water=(
                    ls_xr.dims,
                    utils.extract_bitmask(ls_xr.QA_PIXEL.values, 7).astype(bool),
                ),
                cloud=(
                    ls_xr.dims,
                    ~utils.extract_bitmask(ls_xr.QA_PIXEL.values, 6).astype(bool),
                ),
                qa=(
                    ls_xr.dims,
                    ~utils.extract_bitmask(ls_xr.QA_PIXEL.values, 0).astype(bool),
                ),
            ),
        )

        # Apply name mapping
        ls_xr = self.rename_bands(ls_xr, LandsatReader.vis_band_mapping)

        # Drop time dimension
        ls_xr = ls_xr.isel(t=0, drop=True)

        # Compute NDVI
        ls_xr["ndvi"] = compute_ndvi(ls_xr)

        # Compute LAI with exponential relation between NDVI and LAI
        # Cf. https://src.koda.cnrs.fr/activites-ia-cesbio/ds-cb/blob/master/Jordi_PPL/bmci_slides.pdf
        ls_xr["lai"] = compute_lai_from_ndvi(ls_xr, 0.119, 3.457, -0.062)

        # Compute albedo
        ls_xr["albedo"] = self.compute_albedo(ls_xr)

        return ls_xr

    def read_tir_bands(self) -> xr.Dataset:
        """
        # Read Landsat TIR bands
        # Every bands in the product is sampled at 30m
        # TIR: B10
        """
        ls_xr = self.ds.read_as_xarray(
            [
                landsat.Landsat.B10,
                landsat.Landsat.ST_EMIS,
            ],
            resolution=self.resolution,
            crs=self.crs,
            bounds=self.bb,
            algorithm=rio.enums.Resampling.average,
        )
        if ls_xr is None:
            raise ValueError(f"No data found ({self.path})")

        # Add transform
        ls_xr.attrs["transform"] = affine.Affine(
            self.resolution, 0.0, self.bb.left, 0.0, -self.resolution, self.bb.top
        )
        # Add capteur name
        ls_xr.attrs["tir"] = "Landsat"
        # Add acquisition date
        ls_xr.attrs["tir_date"] = self.ds.date
        ls_xr.attrs["tir_time"] = self.ds.time
        # Add tile id
        ls_xr.attrs["tile"] = self.tile

        # Retrieve masks
        # Convention 1 for masked pixels
        ls_xr = ls_xr.assign(
            dict(
                water=(
                    ls_xr.dims,
                    utils.extract_bitmask(ls_xr.QA_PIXEL.values, 7).astype(bool),
                ),
                cloud=(
                    ls_xr.dims,
                    ~utils.extract_bitmask(ls_xr.QA_PIXEL.values, 6).astype(bool),
                ),
                qa=(
                    ls_xr.dims,
                    ~utils.extract_bitmask(ls_xr.QA_PIXEL.values, 0).astype(bool),
                ),
            ),
        )

        # Apply name mapping
        ls_xr = self.rename_bands(ls_xr, LandsatReader.tir_band_mapping)

        # Drop time dimension
        ls_xr = ls_xr.isel(t=0, drop=True)

        return ls_xr


@dataclass
class HLSReader(ProductReader):
    """
    Reader for HLS product
    Only VIS data is used
    This product can not be used for TIR (no emissivity).
    """

    ds: hls.HLS = field(init=False)

    class HLSParams(Enum):
        HLSLandsat = {
            "bands": [
                hls.HLS.Band.B2,
                hls.HLS.Band.B3,
                hls.HLS.Band.B4,
                hls.HLS.Band.B5,
                hls.HLS.Band.B6,
                hls.HLS.Band.B7,
            ],
            "mapping": {
                "B02": "blue",
                "B03": "green",
                "B04": "red",
                "B05": "nir",
                "B06": "swir1",
                "B07": "swir2",
                "cloud": "cloud",
                "water": "water",
                "qa": "qa",
            },
        }
        HLSSentinel2 = {
            "bands": [
                hls.HLS.Band.B2,
                hls.HLS.Band.B3,
                hls.HLS.Band.B4,
                hls.HLS.Band.B8A,
                hls.HLS.Band.B11,
                hls.HLS.Band.B12,
            ],
            "mapping": {
                "B02": "blue",
                "B03": "green",
                "B04": "red",
                "B8A": "nir",
                "B11": "swir1",
                "B12": "swir2",
                "cloud": "cloud",
                "water": "water",
                "qa": "qa",
            },
        }

    def __post_init__(self):
        """
        Initialize dataset
        """
        # Test if band 8A is available to know if
        # it is a HLSLandsat or HLSSentinel2
        self.params = self.HLSParams.HLSLandsat
        try:
            _ = hls.HLSSentinel2(self.path).build_band_path(hls.HLS.Band.B8A)
            self.params = self.HLSParams.HLSSentinel2
        except FileNotFoundError:
            pass
        # Create an instance of HLS from the product path
        if self.params == self.HLSParams.HLSLandsat:
            self.ds = hls.HLSLandsat(self.path)
        else:
            self.ds = hls.HLSSentinel2(self.path)
        # Metadata
        self.tile = self.ds.tile
        self.bb = utils.bb_snap(self.ds.bounds, align=self.resolution)
        self.crs = self.ds.crs
        self.date = self.ds.date
        self.time = self.ds.time

    def compute_albedo(self, data: xr.Dataset) -> xr.DataArray:
        """
        Compute albedo
        Liang, S. Narrowband to Broadband Conversions of
        Land Surface Albedo I: Algorithms. Remote Sens. Environ. 2001, 76, 213–238.
        """
        return (
            0.356 * data.blue
            + 0.130 * data.red
            + 0.373 * data.nir
            + 0.085 * data.swir1
            + 0.072 * data.swir2
            - 0.0018
        )

    def read_vis_bands(self) -> xr.Dataset:
        """
        Read VIS bands
        """
        # Read HLS data (optical bands)
        hls_xr = self.ds.read_as_xarray(
            self.params.value["bands"],
            resolution=self.resolution,
            crs=self.crs,
            bounds=self.bb,
            algorithm=rio.enums.Resampling.average,
        )
        if hls_xr is None:
            raise ValueError(f"No data found ({self.path})")

        # Drop time dimension
        hls_xr = hls_xr.isel(t=0, drop=True)

        # Retrieve mask
        hls_xr = hls_xr.assign(
            dict(
                water=(
                    hls_xr.dims,
                    utils.extract_bitmask(hls_xr[hls.HLS.QA.value].values, 5).astype(
                        bool
                    ),
                ),
                cloud=(
                    hls_xr.dims,
                    utils.extract_bitmask(hls_xr[hls.HLS.QA.value].values, 1).astype(
                        bool
                    ),
                ),
                qa=(hls_xr.dims, np.ones_like(hls_xr[hls.HLS.QA.value])),
            )
        )

        # Rename bands
        hls_xr = self.rename_bands(hls_xr, self.params.value["mapping"])
        # Compute NDVI
        hls_xr["ndvi"] = compute_ndvi(hls_xr)
        # Compute LAI with exponential relation between NDVI and LAI
        # Cf. https://src.koda.cnrs.fr/activites-ia-cesbio/ds-cb/blob/master/Jordi_PPL/bmci_slides.pdf
        hls_xr["lai"] = compute_lai_from_ndvi(hls_xr, 0.119, 3.457, -0.062)

        # Compute albedo
        hls_xr["albedo"] = self.compute_albedo(hls_xr)

        # Add transform
        hls_xr.attrs["transform"] = affine.Affine(
            self.resolution, 0.0, self.bb.left, 0.0, -self.resolution, self.bb.top
        )
        # Add capteur name
        hls_xr.attrs["vis"] = self.params.name
        # Add acquisition date
        hls_xr.attrs["vis_date"] = self.ds.date
        hls_xr.attrs["vis_time"] = self.ds.time
        # Add tile id
        hls_xr.attrs["tile"] = self.tile
        return hls_xr

    def read_tir_bands(self) -> xr.Dataset:
        """
        Read TIR bands
        """
        raise ProductReaderException("No TIR bands for HLS product")


@dataclass
class Sentinel2Reader(ProductReader):
    """
    Reader for Sentinel2 product
    """

    ds: sentinel2.Sentinel2 = field(init=False)

    vis_band_mapping = {
        "B2": "blue",
        "B3": "green",
        "B4": "red",
        "B6": "red_edge",
        "B8": "nir",
        "B8A": "nir2",
        "B11": "swir1",
        "B12": "swir2",
        "cloud": "cloud",
        "water": "water",
        "qa": "qa",
    }

    def __post_init__(self):
        """
        Initiliazation
        """
        # Create an instance of Sentinel2 from the product path
        self.ds = sentinel2.Sentinel2(self.path)
        self.tile = self.ds.tile
        # Snap bbox
        self.bb = utils.bb_snap(self.ds.bounds, align=self.resolution)
        self.crs = self.ds.crs
        self.date = self.ds.date
        self.time = self.ds.time

    def compute_albedo(self, data: xr.Dataset) -> xr.DataArray:
        """
        Compute albedo
        Bonafoni and al., Albedo Retrieval From Sentinel-2 by New Narrow-to-Broadband Conversion Coefficients,
        IEEE Geoscience and Remote Sensing Letters, 2020
        """
        return (
            0.2266 * data.blue
            + 0.1236 * data.green
            + 0.1573 * data.red
            + 0.3417 * data.nir
            + 0.1170 * data.swir1
            + 0.0338 * data.swir2
        )

    def read_vis_bands(self) -> xr.Dataset:
        """
        Read VIS bands
        """
        # Read sentinel2 data (optical bands)
        s2_xr = self.ds.read_as_xarray(
            [
                sentinel2.Sentinel2.B2,
                sentinel2.Sentinel2.B3,
                sentinel2.Sentinel2.B4,
                sentinel2.Sentinel2.B6,
                sentinel2.Sentinel2.B8,
                sentinel2.Sentinel2.B8A,
                sentinel2.Sentinel2.B11,
                sentinel2.Sentinel2.B12,
            ],
            resolution=self.resolution,
            crs=self.crs,
            bounds=self.bb,
            algorithm=rio.enums.Resampling.average,
        )

        # Filter pixels
        # https://labo.obs-mip.fr/multitemp/sentinel-2/theias-sentinel-2-l2a-product-format/#English
        # Retrieve mask
        s2_xr = s2_xr.assign(
            dict(
                water=(
                    s2_xr.dims,
                    utils.extract_bitmask(
                        s2_xr[sentinel2.Sentinel2.MG2.value].values, 0
                    ).astype(
                        bool
                    ),  # Bit 0 water
                ),
                cloud=(
                    s2_xr.dims,
                    np.where(
                        s2_xr[sentinel2.Sentinel2.CLM.value].values == 0, 0, 1
                    ),  # Cloud pixels
                ),
                qa=(s2_xr.dims, np.ones_like(s2_xr[sentinel2.Sentinel2.CLM.value])),
            )
        )

        # Drop time dimension
        s2_xr = s2_xr.isel(t=0, drop=True)

        # Rename bands
        s2_xr = self.rename_bands(s2_xr, Sentinel2Reader.vis_band_mapping)

        # Compute NDVI
        s2_xr["ndvi"] = compute_ndvi(s2_xr)
        # Compute LAI with exponential relation between NDVI and LAI
        # Cf. https://src.koda.cnrs.fr/activites-ia-cesbio/ds-cb/blob/master/Jordi_PPL/bmci_slides.pdf
        s2_xr["lai"] = compute_lai_from_ndvi(s2_xr, 0.119, 3.457, -0.062)

        # Compute albedo
        s2_xr["albedo"] = self.compute_albedo(s2_xr)

        # Add attributes
        del s2_xr.attrs["type"]
        # Add transform
        s2_xr.attrs["transform"] = affine.Affine(
            self.resolution, 0.0, self.bb.left, 0.0, -self.resolution, self.bb.top
        )
        # Add capteur name
        s2_xr.attrs["vis"] = "Sentinel2"
        # Add acquisition date
        s2_xr.attrs["vis_date"] = self.date
        s2_xr.attrs["vis_time"] = self.time
        # Add tile id
        s2_xr.attrs["tile"] = self.tile
        return s2_xr

    def read_tir_bands(self) -> xr.Dataset:
        """
        Read TIR bands
        """
        raise ProductReaderException("No TIR bands for Sentinel2 product")


@dataclass
class EcostressReader(ProductReader):
    """
    Reader for Ecostress (collection V2) product
    """

    ds: ecostress_v2.EcostressV2 = field(init=False)

    tir_band_mapping = {
        "LST": "lst",
        "EmisWB": "emis",
        "cloud": "cloud",
        "water": "water",
        "qa": "qa",
    }

    def __post_init__(self):
        """
        Initiliazation
        """
        # Create an instance of Ecostress from the product path
        self.ds = ecostress_v2.EcostressV2(self.path)
        self.tile = self.ds.tile
        # Snap bbox
        self.bb = utils.bb_snap(self.ds.bounds, align=self.resolution)
        self.crs = self.ds.crs
        self.date = self.ds.date
        self.time = self.ds.time

    def read_vis_bands(self) -> xr.Dataset:
        """
        Read VIS bands
        """
        raise ProductReaderException("No VIS bands for Ecostress product")

    def read_tir_bands(self) -> xr.Dataset:
        """
        Read TIR bands
        """
        # Read ecostress product
        eco_xr = self.ds.read_as_xarray(
            [ecostress_v2.EcostressV2.LST, ecostress_v2.EcostressV2.EMIS],
            resolution=self.resolution,
            crs=self.crs,
            bounds=self.bb,
            algorithm=rio.enums.Resampling.cubic,
        )
        if eco_xr is None:
            raise ValueError(f"No data found ({self.path})")

        # Masks from ecostress
        # https://ecostress.jpl.nasa.gov/downloads/userguides/2_ECOSTRESS_L2_UserGuide_06182019.pdf
        b0_mask = ~utils.extract_bitmask(
            eco_xr[ecostress_v2.EcostressV2.QUALITY.value].values, 0
        )
        b1_mask = ~utils.extract_bitmask(
            eco_xr[ecostress_v2.EcostressV2.QUALITY.value].values, 1
        )
        eco_xr = eco_xr.assign(
            dict(
                water=(
                    eco_xr.dims,
                    np.where(
                        eco_xr[ecostress_v2.EcostressV2.WATER.value].values, 1, 0
                    ).astype(bool),
                ),
                cloud=(
                    eco_xr.dims,
                    np.where(
                        eco_xr[ecostress_v2.EcostressV2.CLOUDS.value].values, 1, 0
                    ).astype(bool),
                ),
                qa=(eco_xr.dims, np.logical_and(b0_mask, b1_mask)),
            ),
        )

        # Drop time dimension
        eco_xr = eco_xr.isel(t=0, drop=True)

        # Rename bands
        eco_xr = self.rename_bands(eco_xr, EcostressReader.tir_band_mapping)

        # Add attributes
        # Add transform
        eco_xr.attrs["transform"] = affine.Affine(
            self.resolution, 0.0, self.bb.left, 0.0, -self.resolution, self.bb.top
        )
        # Add capteur name
        eco_xr.attrs["tir"] = "Ecostress"
        # Add acquisition date
        eco_xr.attrs["tir_date"] = self.date
        eco_xr.attrs["tir_time"] = self.time
        # Add tile id
        eco_xr.attrs["tile"] = self.tile

        return eco_xr


def get_product_reader(product_path: str, resolution=60) -> ProductReader:
    """
    Get the product reader
    """
    reader_list: List[Type[ProductReader]] = [
        LandsatReader,
        Sentinel2Reader,
        EcostressReader,
        HLSReader,
    ]
    for product_reader in reader_list:
        try:
            return product_reader(product_path, resolution=resolution)
        except Exception:
            continue
    raise ProductReaderException("No reader compatible")

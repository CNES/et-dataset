#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright: (c) 2023 CESBIO / Centre National d'Etudes Spatiales / Université Paul Sabatier (UT3)
#
"""
Remote sensing products
"""
from abc import abstractmethod
from dataclasses import dataclass, field

import affine
import numpy as np
import rasterio as rio
import xarray as xr
from sensorsio import ecostress_v2, landsat, mgrs, sentinel2, utils

from etdataset.logging import LoggerManager
from etdataset.vegetation_indices import compute_lai_from_ndvi, compute_ndvi

logger = LoggerManager.get_logger(__name__)


class ProductReaderException(Exception):
    """
    Exception for ReaderProduct
    """

    pass


@dataclass
class ProductReader:
    """
    Abstract class for product reader
    """

    path: str
    tile: str = field(init=False)
    resolution: int = field(default=60)

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
    def read_vis_bands(self, use_mask: bool = False) -> xr.Dataset:
        """
        Read VIS bands
        """
        pass

    @abstractmethod
    def read_tir_bands(self, use_mask: bool = False) -> xr.Dataset:
        """
        Read TIR bands
        """
        pass


@dataclass
class LandsatReader(ProductReader):
    """
    Reader for Landsat product
    """

    bb: str = field(init=False)
    ds: landsat.Landsat = field(init=False)

    vis_band_mapping = {
        "SR_B2": "blue",
        "SR_B3": "green",
        "SR_B4": "red",
        "SR_B5": "nir",
        "SR_B6": "swir1",
        "SR_B7": "swir2",
    }
    tir_band_mapping = {
        "ST_B10": "lst",
        "ST_EMIS": "emis",
    }

    def __post_init__(self):
        """
        Initialize dataset
        """
        # Create an instance of Landsat8 from the product path
        self.ds = landsat.Landsat(self.path)
        self._tile = None
        self.date = self.ds.date

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
        self.bb = utils.bb_transform(mgrs.get_crs_mgrs_tile(tile_id), self.ds.crs, bb)
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

    def read_vis_bands(self,use_mask: bool = False) -> xr.Dataset:
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
        # Add transform
        ls_xr.attrs["transform"] = affine.Affine(
            self.resolution, 0.0, self.bb.left, 0.0, -self.resolution, self.bb.top
        )
        # Add capteur name
        ls_xr.attrs["vis"] = "Landsat"
        # Add acquisition date
        ls_xr.attrs["vis_date"] = self.ds.date
        # Add tile id
        ls_xr.attrs["tile"] = self.tile

        # Filter QA from ls8
        if use_mask:
            clear_pixels_mask = utils.extract_bitmask(ls_xr.QA_PIXEL.values, 6)
            not_filled_mask = ~utils.extract_bitmask(ls_xr.QA_PIXEL.values, 0)
            not_water_mask = ~utils.extract_bitmask(ls_xr.QA_PIXEL.values, 7)
            ls_xr = ls_xr.where(
                np.logical_and(
                    np.logical_and(clear_pixels_mask, not_filled_mask), not_water_mask
                ),
                np.nan,
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

    def read_tir_bands(self, use_mask: bool = False) -> xr.Dataset:
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
        # Add transform
        ls_xr.attrs["transform"] = affine.Affine(
            self.resolution, 0.0, self.bb.left, 0.0, -self.resolution, self.bb.top
        )
        # Add capteur name
        ls_xr.attrs["tir"] = "Landsat"
        # Add acquisition date
        ls_xr.attrs["tir_date"] = self.ds.date
        # Add tile id
        ls_xr.attrs["tile"] = self.tile

        # Filter QA from ls8
        if use_mask:
            clear_pixels_mask = utils.extract_bitmask(ls_xr.QA_PIXEL.values, 6)
            not_filled_mask = ~utils.extract_bitmask(ls_xr.QA_PIXEL.values, 0)
            not_water_mask = ~utils.extract_bitmask(ls_xr.QA_PIXEL.values, 7)
            ls_xr = ls_xr.where(
                np.logical_and(
                    np.logical_and(clear_pixels_mask, not_filled_mask), not_water_mask
                ),
                np.nan,
            )

        # Apply name mapping
        ls_xr = self.rename_bands(ls_xr, LandsatReader.tir_band_mapping)

        # Drop time dimension
        ls_xr = ls_xr.isel(t=0, drop=True)

        return ls_xr


@dataclass
class Sentinel2Reader(ProductReader):
    """
    Reader for Sentinel2 product
    """

    bb: str = field(init=False)
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

    def read_vis_bands(self, use_mask: bool = False) -> xr.Dataset:
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
        if use_mask:
            not_water_mask = ~utils.extract_bitmask(s2_xr[sentinel2.Sentinel2.MG2.value].values, 0).astype(bool) # Bit 0 water
            clear_pixels_mask = np.where(s2_xr[sentinel2.Sentinel2.CLM.value].values == 0,1,0) # Clear pixels
            mask = np.logical_and(not_water_mask,clear_pixels_mask)
            s2_xr = s2_xr.where(mask, np.nan)

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
        s2_xr.attrs["vis_date"] = self.ds.date
        # Add tile id
        s2_xr.attrs["tile"] = self.tile
        return s2_xr

    def read_tir_bands(self, use_mask: bool = False) -> xr.Dataset:
        """
        Read TIR bands
        """
        raise ProductReaderException("No TIR bands for Sentinel2 product")


@dataclass
class EcostressReader(ProductReader):
    """
    Reader for Ecostress (collection V2) product
    """

    bb: str = field(init=False)
    ds: ecostress_v2.EcostressV2 = field(init=False)

    tir_band_mapping = {
        "LST": "lst",
        "EmisWB": "emis",
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

    def read_vis_bands(self, use_mask: bool = False) -> xr.Dataset:
        """
        Read VIS bands
        """
        raise ProductReaderException("No VIS bands for Ecostress product")

    def read_tir_bands(self, use_mask: bool = False) -> xr.Dataset:
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

        # Filter QA from ecostress
        #https://ecostress.jpl.nasa.gov/downloads/userguides/2_ECOSTRESS_L2_UserGuide_06182019.pdf
        if use_mask:
            b0_mask = ~utils.extract_bitmask(eco_xr[ecostress_v2.EcostressV2.QUALITY.value].values, 0).astype(bool)
            b1_mask = ~utils.extract_bitmask(eco_xr[ecostress_v2.EcostressV2.QUALITY.value].values, 1).astype(bool)
            qa_mask = np.logical_and(b0_mask, b1_mask) 
            not_cloud_mask = np.where(eco_xr[ecostress_v2.EcostressV2.CLOUDS.value].values,0,1) 
            not_water_mask = np.where(eco_xr[ecostress_v2.EcostressV2.WATER.value].values,0,1) 
            mask = np.logical_and(np.logical_and(qa_mask,not_cloud_mask),not_water_mask)
            eco_xr = eco_xr.where(mask, np.nan)

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
        eco_xr.attrs["tir_date"] = self.ds.date
        # Add tile id
        eco_xr.attrs["tile"] = self.tile

        return eco_xr


def get_product_reader(product_path: str) -> ProductReader:
    """
    Get the product reader
    """
    reader = None
    for product_reader in [LandsatReader, Sentinel2Reader, EcostressReader]:
        try:
            reader = product_reader(product_path)
            if reader is not None:
                return reader
        except Exception:
            continue
    raise ProductReaderException("No reader compatible")

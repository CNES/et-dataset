#
# SPDX-License-Identifier: AGPL-3.0-only
#
# Copyright (C) 2023 CESBIO / Centre National d'Etudes Spatiales
#
"""
Module for reading remote sensing products
"""

import datetime
import os
from abc import abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from fnmatch import fnmatch
from json import load
from types import MappingProxyType

import affine
import numpy as np
import rasterio as rio
import xarray as xr
from pyproj import CRS
from sensorsio import ecostress_v2, hls, landsat, sentinel2, utils

from etdataset.logging import LoggerManager
from etdataset.vegetation_indices import (
    compute_bvnet,
    compute_lai_from_ndvi,
    compute_ndvi,
)

logger = LoggerManager.get_logger(__name__)

RESOLUTION = 60.0


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
    bb: rio.coords.BoundingBox | None = None
    crs: CRS | None = None
    _bb: rio.coords.BoundingBox = field(init=False, repr=False)
    _crs: CRS = field(init=False, repr=False)
    date: datetime.date = field(init=False)
    time: datetime.time = field(init=False)
    resolution: float = field(default=RESOLUTION)

    def rename_bands(self, data: xr.Dataset, name_mapping: dict) -> xr.Dataset:
        """
        Rename bands
        """
        # Apply name mapping
        renamed = data.rename_vars(name_mapping)
        # Remove any variable not listed in tir_band_mapping
        for var in renamed.data_vars:
            if var not in name_mapping.values():
                renamed = renamed.drop(var)
        return renamed

    @abstractmethod
    def read_vis_bands(
        self,
        resampling: rio.enums.Resampling = rio.enums.Resampling.average,
    ) -> xr.Dataset:
        """
        Read VIS bands
        """

    @abstractmethod
    def read_tir_bands(
        self,
        resampling: rio.enums.Resampling = rio.enums.Resampling.average,
    ) -> xr.Dataset:
        """
        Read TIR bands
        """

    def add_band_attributes(self, band: xr.DataArray, name: str, unit: str):
        """
        Add band attributes (for netcdf)
        """
        band.attrs.clear()
        band.attrs["standard_name"] = name
        band.attrs["long_name"] = name
        band.attrs["name"] = name
        band.attrs["unit"] = unit
        band.attrs["description"] = name
        return band


@dataclass
class LandsatReader(ProductReader):
    """
    Reader for Landsat product
    """

    ds: landsat.Landsat = field(init=False)
    sun_elevation: float = field(init=False)
    sun_azimuth: float = field(init=False)
    satellite: str = field(init=False)

    vis_band_mapping = MappingProxyType(
        {
            "SR_B2": "blue",
            "SR_B3": "green",
            "SR_B4": "red",
            "SR_B5": "nir",
            "SR_B6": "swir",
            "SR_B7": "swir2",
            "cloud": "cloud",
            "water": "water",
            "qa": "qa",
        }
    )
    tir_band_mapping = MappingProxyType(
        {
            "ST_B10": "lst",
            "ST_EMIS": "emis",
            "cloud": "cloud",
            "water": "water",
            "qa": "qa",
        }
    )

    def __post_init__(self):
        """
        Initialize dataset
        """
        # Create an instance of Landsat8 from the product path
        self.ds = landsat.Landsat(self.path)
        if self.bb is None:
            self._bb = self.ds.bounds
        else:
            self._bb = self.bb
        if self.crs is None:
            self._crs = self.ds.crs
        else:
            self._crs = self.crs
        self.date = self.ds.date
        self.time = self.ds.time
        # Snap bbox
        self._bb = utils.bb_snap(self._bb, align=self.resolution)
        # Satellite
        if "LC08" in self.ds.product_name:
            self.satellite = "landsat8"
        elif "LC09" in self.ds.product_name:
            self.satellite = "landsat9"
        else:
            msg = "Only Landsat8/9 can be used"
            raise ValueError(msg)
        # Angles
        # Open metadata
        meta_data_file = None
        for root, _, files in os.walk(self.path):
            for name in files:
                if fnmatch(name, "*_MTL.json"):
                    meta_data_file = os.path.join(root, name)
        if meta_data_file is None:
            msg = "Metadata file is missing"
            raise OSError(msg)
        with open(meta_data_file) as data_file:
            metadata = load(data_file)["LANDSAT_METADATA_FILE"][
                "IMAGE_ATTRIBUTES"
            ]
            self.sun_elevation = float(metadata["SUN_ELEVATION"])
            self.sun_azimuth = float(metadata["SUN_AZIMUTH"])

    def compute_angles(self):
        """
        Compute angles using sun position at the center of the image
        """

    def compute_albedo(self, data: xr.Dataset) -> xr.DataArray:
        """
        Compute albedo
        Liang, S. Narrowband to Broadband Conversions of
        Land Surface Albedo I: Algorithms.
        Remote Sens. Environ. 2001, 76, 213-238
        """
        albedo = (
            0.356 * data.blue
            + 0.130 * data.red
            + 0.373 * data.nir
            + 0.085 * data.swir
            + 0.072 * data.swir2
            - 0.0018
        )
        return albedo.clip(0.01, 0.99)

    def read_vis_bands(
        self,
        resampling: rio.enums.Resampling = rio.enums.Resampling.average,
    ) -> xr.Dataset:
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
            crs=str(self._crs),
            bounds=self._bb,
            algorithm=resampling,
        )
        if ls_xr is None:
            raise ValueError(f"No data found ({self.path})")

        # Add transform
        ls_xr.attrs["transform"] = affine.Affine(
            self.resolution,
            0.0,
            self._bb.left,
            0.0,
            -self.resolution,
            self._bb.top,
        )
        # Add sensor name
        ls_xr.attrs["vis"] = "Landsat"
        # Add acquisition date
        ls_xr.attrs["vis_date"] = self.ds.date
        ls_xr.attrs["vis_time"] = self.ds.time

        # Retrieve masks
        # Convention 1 for masked pixels
        ls_xr = ls_xr.assign(
            {
                "water": (
                    ls_xr.dims,
                    utils.extract_bitmask(ls_xr.QA_PIXEL.values, 7).astype(
                        bool
                    ),
                ),
                "cloud": (
                    ls_xr.dims,
                    ~utils.extract_bitmask(ls_xr.QA_PIXEL.values, 6).astype(
                        bool
                    ),
                ),
                "qa": (
                    ls_xr.dims,
                    ~utils.extract_bitmask(ls_xr.QA_PIXEL.values, 0).astype(
                        bool
                    ),
                ),
            },
        )

        # Apply name mapping
        ls_xr = self.rename_bands(ls_xr, LandsatReader.vis_band_mapping)  # type: ignore

        # Add attributes
        for band in ls_xr.data_vars:
            self.add_band_attributes(ls_xr[band], str(band), "-")

        # Drop time dimension
        ls_xr = ls_xr.isel(t=0, drop=True)

        # Compute NDVI
        ls_xr["ndvi"] = compute_ndvi(ls_xr)

        # Add angles
        ls_xr["cos(View_Zenith)"] = xr.ones_like(ls_xr["red"])
        ls_xr["cos(Sun_Zenith)"] = xr.full_like(
            ls_xr["red"],
            fill_value=np.cos(
                np.deg2rad(90 - self.sun_elevation), dtype=np.float32
            ),
        )
        ls_xr["cos(Rel_Azimuth)"] = xr.full_like(
            ls_xr["red"],
            fill_value=np.cos(
                np.deg2rad(90 - self.sun_azimuth), dtype=np.float32
            ),
        )

        # Compute LAI with BVnet
        # Cf. https://forge.ird.fr/cesbio/modelisation/pybvnet/-/tree/main?ref_type=heads
        ls_xr["lai"], ls_xr["fcover"] = compute_bvnet(
            ls_xr,
            band_list=[
                "green",
                "red",
                "nir",
                "swir",
                "swir2",
                "cos(View_Zenith)",
                "cos(Sun_Zenith)",
                "cos(Rel_Azimuth)",
            ],
            satellite=self.satellite,
            version="V3.1",
        )

        # Delete angles
        ls_xr = ls_xr.drop_vars(
            ["cos(View_Zenith)", "cos(Sun_Zenith)", "cos(Rel_Azimuth)"]
        )

        # Compute albedo
        ls_xr["albedo"] = self.compute_albedo(ls_xr)

        return ls_xr

    def read_tir_bands(
        self,
        resampling: rio.enums.Resampling = rio.enums.Resampling.average,
    ) -> xr.Dataset:
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
            crs=str(self._crs),
            bounds=self._bb,
            algorithm=resampling,
        )
        if ls_xr is None:
            raise ValueError(f"No data found ({self.path})")

        # Add transform
        ls_xr.attrs["transform"] = affine.Affine(
            self.resolution,
            0.0,
            self._bb.left,
            0.0,
            -self.resolution,
            self._bb.top,
        )
        # Add sensor name
        ls_xr.attrs["tir"] = "Landsat"
        # Add acquisition date
        ls_xr.attrs["tir_date"] = self.ds.date
        ls_xr.attrs["tir_time"] = self.ds.time

        # Retrieve masks
        # Convention 1 for masked pixels
        ls_xr = ls_xr.assign(
            {
                "water": (
                    ls_xr.dims,
                    utils.extract_bitmask(ls_xr.QA_PIXEL.values, 7).astype(
                        bool
                    ),
                ),
                "cloud": (
                    ls_xr.dims,
                    ~utils.extract_bitmask(ls_xr.QA_PIXEL.values, 6).astype(
                        bool
                    ),
                ),
                "qa": (
                    ls_xr.dims,
                    ~utils.extract_bitmask(ls_xr.QA_PIXEL.values, 0).astype(
                        bool
                    ),
                ),
            },
        )

        # Apply name mapping
        ls_xr = self.rename_bands(ls_xr, LandsatReader.tir_band_mapping)  # type: ignore

        # Add attributes
        for band in ls_xr.data_vars:
            self.add_band_attributes(ls_xr[band], str(band), "-")
        self.add_band_attributes(ls_xr["lst"], "lst", "K")

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
        HLSLandsat = MappingProxyType(
            {
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
                    "B06": "swir",
                    "B07": "swir2",
                    "cloud": "cloud",
                    "water": "water",
                    "qa": "qa",
                },
            }
        )
        HLSSentinel2 = MappingProxyType(
            {
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
                    "B11": "swir",
                    "B12": "swir2",
                    "cloud": "cloud",
                    "water": "water",
                    "qa": "qa",
                },
            }
        )

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
        if self.bb is None:
            self._bb = self.ds.bounds
        else:
            self._bb = self.bb
        if self.crs is None:
            self._crs = self.ds.crs
        else:
            self._crs = self.crs
        self._bb = utils.bb_snap(self._bb, align=self.resolution)
        self.date = self.ds.date
        self.time = self.ds.time

    def compute_albedo(self, data: xr.Dataset) -> xr.DataArray:
        """
        Compute albedo
        Liang, S. Narrowband to Broadband Conversions of
        Land Surface Albedo I: Algorithms.
        Remote Sens. Environ. 2001, 76, 213-238.
        """
        albedo = (
            0.356 * data.blue
            + 0.130 * data.red
            + 0.373 * data.nir
            + 0.085 * data.swir
            + 0.072 * data.swir2
            - 0.0018
        )
        return albedo.clip(0.01, 0.99)

    def read_vis_bands(
        self,
        resampling: rio.enums.Resampling = rio.enums.Resampling.average,
    ) -> xr.Dataset:
        """
        Read VIS bands
        """
        # Read HLS data (optical bands)
        hls_xr = self.ds.read_as_xarray(
            self.params.value["bands"],
            resolution=self.resolution,
            crs=str(self._crs),
            bounds=self._bb,
            algorithm=resampling,
        )
        if hls_xr is None:
            raise ValueError(f"No data found ({self.path})")

        # Drop time dimension
        hls_xr = hls_xr.isel(t=0, drop=True)

        # Retrieve mask
        hls_xr = hls_xr.assign(
            {
                "water": (
                    hls_xr.dims,
                    utils.extract_bitmask(
                        hls_xr[hls.HLS.QA.value].values, 5
                    ).astype(bool),
                ),
                "cloud": (
                    hls_xr.dims,
                    utils.extract_bitmask(
                        hls_xr[hls.HLS.QA.value].values, 1
                    ).astype(bool),
                ),
                "qa": (hls_xr.dims, np.ones_like(hls_xr[hls.HLS.QA.value])),
            }
        )

        # Rename bands
        hls_xr = self.rename_bands(hls_xr, self.params.value["mapping"])

        # Add attributes
        for band in hls_xr.data_vars:
            self.add_band_attributes(hls_xr[band], str(band), "-")

        # Compute NDVI
        hls_xr["ndvi"] = compute_ndvi(hls_xr)
        # Compute LAI with exponential relation between NDVI and LAI
        # Cf. https://src.koda.cnrs.fr/activites-ia-cesbio/ds-cb/blob/master/Jordi_PPL/bmci_slides.pdf
        hls_xr["lai"] = compute_lai_from_ndvi(hls_xr, 0.119, 3.457, -0.062)

        # Compute albedo
        hls_xr["albedo"] = self.compute_albedo(hls_xr)
        self.add_band_attributes(hls_xr["albedo"], "albedo", "-")

        # Add transform
        hls_xr.attrs["transform"] = affine.Affine(
            self.resolution,
            0.0,
            self._bb.left,
            0.0,
            -self.resolution,
            self._bb.top,
        )
        # Add sensor name
        hls_xr.attrs["vis"] = self.params.name
        # Add acquisition date
        hls_xr.attrs["vis_date"] = self.ds.date
        hls_xr.attrs["vis_time"] = self.ds.time
        return hls_xr

    def read_tir_bands(
        self,
        resampling: rio.enums.Resampling = rio.enums.Resampling.average,  # noqa ARG002
    ) -> xr.Dataset:
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

    vis_band_mapping = MappingProxyType(
        {
            "B2": "blue",
            "B3": "green",
            "B4": "red",
            "B6": "red_edge",
            "B8": "nir",
            "B8A": "nir2",
            "B11": "swir",
            "B12": "swir2",
            "cloud": "cloud",
            "water": "water",
            "qa": "qa",
        }
    )

    def __post_init__(self):
        """
        Initialization
        """
        # Create an instance of Sentinel2 from the product path
        self.ds = sentinel2.Sentinel2(self.path)
        if self.bb is None:
            self._bb = self.ds.bounds
        else:
            self._bb = self.bb
        if self.crs is None:
            self.crs = self.ds.crs
        else:
            self._crs = self.crs
        # Snap bbox
        self._bb = utils.bb_snap(self._bb, align=self.resolution)
        self.date = self.ds.date
        self.time = self.ds.time

    def compute_albedo(self, data: xr.Dataset) -> xr.DataArray:
        """
        Compute albedo
        Bonafoni and al., Albedo Retrieval From Sentinel-2 by
        New Narrow-to-Broadband Conversion Coefficients,
        IEEE Geoscience and Remote Sensing Letters, 2020
        """
        albedo = (
            0.2266 * data.blue
            + 0.1236 * data.green
            + 0.1573 * data.red
            + 0.3417 * data.nir
            + 0.1170 * data.swir
            + 0.0338 * data.swir2
        )
        return albedo.clip(0.01, 0.99)

    def read_vis_bands(
        self,
        resampling: rio.enums.Resampling = rio.enums.Resampling.average,
    ) -> xr.Dataset:
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
            crs=str(self._crs),
            bounds=self._bb,
            algorithm=resampling,
        )

        # Filter pixels
        # https://labo.obs-mip.fr/multitemp/sentinel-2/theias-
        # sentinel-2-l2a-product-format/#English
        # Retrieve mask
        s2_xr = s2_xr.assign(
            {
                "water": (
                    s2_xr.dims,
                    utils.extract_bitmask(
                        s2_xr[sentinel2.Sentinel2.MG2.value].values, 0
                    ).astype(bool),  # Bit 0 water
                ),
                "cloud": (
                    s2_xr.dims,
                    np.where(
                        s2_xr[sentinel2.Sentinel2.CLM.value].values == 0, 0, 1
                    ),  # Cloud pixels
                ),
                "qa": (
                    s2_xr.dims,
                    np.ones_like(s2_xr[sentinel2.Sentinel2.CLM.value]),
                ),
            }
        )

        # Drop time dimension
        s2_xr = s2_xr.isel(t=0, drop=True)

        # Rename bands
        s2_xr = self.rename_bands(s2_xr, Sentinel2Reader.vis_band_mapping)  # type: ignore

        # Add attributes
        for band in s2_xr.data_vars:
            self.add_band_attributes(s2_xr[band], str(band), "-")

        # Compute NDVI
        s2_xr["ndvi"] = compute_ndvi(s2_xr)
        # Compute LAI with exponential relation between NDVI and LAI
        # Cf. https://src.koda.cnrs.fr/activites-ia-cesbio/ds-cb/
        # blob/master/Jordi_PPL/bmci_slides.pdf
        s2_xr["lai"] = compute_lai_from_ndvi(s2_xr, 0.119, 3.457, -0.062)

        # Compute albedo
        s2_xr["albedo"] = self.compute_albedo(s2_xr)
        self.add_band_attributes(s2_xr["albedo"], "albedo", "-")

        # Add attributes
        del s2_xr.attrs["type"]
        # Add transform
        s2_xr.attrs["transform"] = affine.Affine(
            self.resolution,
            0.0,
            self._bb.left,
            0.0,
            -self.resolution,
            self._bb.top,
        )
        # Add sensor name
        s2_xr.attrs["vis"] = "Sentinel2"
        # Add acquisition date
        s2_xr.attrs["vis_date"] = self.date
        s2_xr.attrs["vis_time"] = self.time
        return s2_xr

    def read_tir_bands(
        self,
        resampling: rio.enums.Resampling = rio.enums.Resampling.average,  # noqa ARG002
    ) -> xr.Dataset:
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

    tir_band_mapping = MappingProxyType(
        {
            "LST": "lst",
            "EmisWB": "emis",
            "cloud": "cloud",
            "water": "water",
            "qa": "qa",
        }
    )

    def __post_init__(self):
        """
        Initialization
        """
        # Create an instance of Ecostress from the product path
        self.ds = ecostress_v2.EcostressV2(self.path)
        if self.bb is None:
            self._bb = self.ds.bounds
        else:
            self._bb = self.bb
        if self.crs is None:
            self._crs = self.ds.crs
        else:
            self._crs = self.crs
        # Snap bbox
        self._bb = utils.bb_snap(self._bb, align=self.resolution)
        self.date = self.ds.date
        self.time = self.ds.time

    def read_vis_bands(
        self,
        resampling: rio.enums.Resampling = rio.enums.Resampling.average,  # noqa ARG002
    ) -> xr.Dataset:
        """
        Read VIS bands
        """
        raise ProductReaderException("No VIS bands for Ecostress product")

    def read_tir_bands(
        self,
        resampling: rio.enums.Resampling = rio.enums.Resampling.average,
    ) -> xr.Dataset:
        """
        Read TIR bands
        """
        # Read ecostress product
        eco_xr = self.ds.read_as_xarray(
            [ecostress_v2.EcostressV2.LST, ecostress_v2.EcostressV2.EMIS],
            resolution=self.resolution,
            crs=str(self._crs),
            bounds=self._bb,
            algorithm=resampling,
        )
        if eco_xr is None:
            raise ValueError(f"No data found ({self.path})")

        # Masks from ecostress
        # https://ecostress.jpl.nasa.gov/downloads/userguides/ \
        # 2_ECOSTRESS_L2_UserGuide_06182019.pdf
        b0_mask = ~utils.extract_bitmask(
            eco_xr[ecostress_v2.EcostressV2.QUALITY.value].values, 0
        )
        b1_mask = ~utils.extract_bitmask(
            eco_xr[ecostress_v2.EcostressV2.QUALITY.value].values, 1
        )
        eco_xr = eco_xr.assign(
            {
                "water": (
                    eco_xr.dims,
                    np.where(
                        eco_xr[ecostress_v2.EcostressV2.WATER.value].values,
                        1,
                        0,
                    ).astype(bool),
                ),
                "cloud": (
                    eco_xr.dims,
                    np.where(
                        eco_xr[ecostress_v2.EcostressV2.CLOUDS.value].values,
                        1,
                        0,
                    ).astype(bool),
                ),
                "qa": (eco_xr.dims, np.logical_and(b0_mask, b1_mask)),
            },
        )

        # Drop time dimension
        eco_xr = eco_xr.isel(t=0, drop=True)

        # Rename bands
        eco_xr = self.rename_bands(eco_xr, EcostressReader.tir_band_mapping)  # type: ignore

        # Add attributes
        for band in eco_xr.data_vars:
            self.add_band_attributes(eco_xr[band], str(band), "-")
        self.add_band_attributes(eco_xr["lst"], "lst", "K")

        # Add transform
        eco_xr.attrs["transform"] = affine.Affine(
            self.resolution,
            0.0,
            self._bb.left,
            0.0,
            -self.resolution,
            self._bb.top,
        )
        # Add sensor name
        eco_xr.attrs["tir"] = "Ecostress"
        # Add acquisition date
        eco_xr.attrs["tir_date"] = self.date
        eco_xr.attrs["tir_time"] = self.time

        return eco_xr


def get_product_reader(
    product_path: str,
    roi_bbox: rio.coords.BoundingBox | None = None,
    roi_crs: CRS | None = None,
    resolution: float = RESOLUTION,
) -> ProductReader:
    """
    Get the product reader
    """
    reader_list: list[type[ProductReader]] = [
        LandsatReader,
        Sentinel2Reader,
        EcostressReader,
        HLSReader,
    ]
    for product_reader in reader_list:
        try:
            return product_reader(
                path=product_path,
                bb=roi_bbox,  # type: ignore
                crs=roi_crs,  # type: ignore
                resolution=resolution,
            )
        # TODO: Improve exception catching
        except Exception:  # noqa
            continue
    raise ProductReaderException("No reader compatible")

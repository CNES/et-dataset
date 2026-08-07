#
# Copyright: (c) 2023 CESBIO / Centre National d'Etudes Spatiales /
#            Université Paul Sabatier (UT3)
#
"""
Vegetation indices
Albedo
"""

from typing import Any

import numpy as np
import xarray as xr
from pyBVNET.pyBVNET import apply_NNT


def compute_ndvi(data: xr.Dataset) -> xr.DataArray:
    """
    Compute NDVI
    Rouse, J., Jr.; Haas, R.; Deering, D.; Schell, J.; Harlan, J.
    Monitoring the Vernal Advancement and Retrogradation
    (Green Wave Effect) of Natural Vegetation; Great Plains Corridor;
    Texas A&M University Remote Sensing Center: College Station, TX, USA, 1974.
    """
    xarr = np.clip((data.nir - data.red) / (data.nir + data.red + 1e-9), -1, 1)
    xarr.attrs["standard_name"] = "ndvi"
    xarr.attrs["long_name"] = "ndvi"
    xarr.attrs["name"] = "ndvi"
    xarr.attrs["unit"] = "-"
    xarr.attrs["description"] = "NDVI"
    return xarr


def ndvi_to_lai(ndvi: float, a: float, b: float, c: float = 0.0) -> float:
    """
    Compute LAI from ndvi using an exponential mathematical form
    LAI = a x exp(b x ndvi) + c
    """
    lai = np.nan
    if not np.isnan(ndvi) and ndvi >= 0.0 and ndvi <= 1:
        lai = a * np.exp(b * ndvi) + c
        if lai < 0:
            lai = 0
    return lai


def compute_lai_from_ndvi(
    data: xr.Dataset, a: float, b: float, c: float = 0.0
) -> xr.DataArray:
    """
    Compute LAI from ndvi using an exponential mathematical form
    LAI = a x exp(b x ndvi) + c
    """
    if "ndvi" not in data.variables:
        ndvi = compute_ndvi(data)
    else:
        ndvi = data["ndvi"]
    xarr = xr.apply_ufunc(ndvi_to_lai, ndvi, a, b, c, vectorize=True)
    xarr.attrs["standard_name"] = "lai"
    xarr.attrs["long_name"] = "lai"
    xarr.attrs["name"] = "lai"
    xarr.attrs["unit"] = "-"
    xarr.attrs["description"] = "LAI"
    return xarr


def compute_bvnet(
    data: xr.Dataset,
    band_list: list[str],
    satellite: str,
    version: str | None = None,
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Compute LAI and Fcover with BVNet

    Parameters
    ----------
    data: xr.Dataset
        Data
    band_list: list[str]
        List of bands to use for BVNET
    satellite: str
        Model to use for BVNET
    version: optional(str)
        Model version to use for BVNET

    Parameters
    ----------
    lai: xr.DataArray
        LAI estimated
    fcover: sxr.DataArray
        FCOVER estimated
    """
    stacked_inputs = xr.concat([data[var] for var in band_list], dim="band")

    # Assign band names or indices as a coordinate for the new dimension
    stacked_inputs = stacked_inputs.assign_coords(band=band_list)

    # Convert to a dataset with one variable
    stacked_inputs = stacked_inputs.to_dataset(name="band_data")  # type: ignore

    # Define dimensions and arrays for output data
    dims = data[band_list[0]].dims
    output_data = np.empty(shape=data[band_list[0]].shape, dtype=np.float32)
    output_flag_data = np.zeros(shape=data[band_list[0]].shape, dtype=np.int8)
    output_uncertainty_data = np.empty(
        shape=data[band_list[0]].shape, dtype=np.float32
    )

    # Create output dataset
    datavars: dict[str, Any] = {}
    datavars["LAI"] = (dims, output_data)
    datavars["FCOVER"] = (dims, output_data)
    datavars["LAI_flag"] = (dims, output_flag_data)
    datavars["FCOVER_flag"] = (dims, output_flag_data)
    datavars["LAI_uncertainty"] = (dims, output_uncertainty_data)
    datavars["FCOVER_uncertainty"] = (dims, output_uncertainty_data)
    output = xr.Dataset(
        data_vars=datavars,
        coords={"y": data.coords["y"], "x": data.coords["x"]},
    )

    bvnet_xr = apply_NNT(
        stacked_inputs,
        output,
        ["LAI", "FCOVER"],
        satellite=satellite,
        version=version,
    )[["LAI", "FCOVER"]]
    # Clip Fcover
    bvnet_xr["FCOVER"] = bvnet_xr["FCOVER"].clip(0.0, 1.0)
    bvnet_xr["LAI"].attrs["standard_name"] = "lai"
    bvnet_xr["LAI"].attrs["long_name"] = "lai"
    bvnet_xr["LAI"].attrs["name"] = "lai"
    bvnet_xr["LAI"].attrs["unit"] = "-"
    bvnet_xr["LAI"].attrs["description"] = "LAI"
    bvnet_xr["FCOVER"].attrs["standard_name"] = "fcover"
    bvnet_xr["FCOVER"].attrs["long_name"] = "fcover"
    bvnet_xr["FCOVER"].attrs["name"] = "fcover"
    bvnet_xr["FCOVER"].attrs["unit"] = "-"
    bvnet_xr["FCOVER"].attrs["description"] = "Fraction cover"
    return bvnet_xr["LAI"], bvnet_xr["FCOVER"]

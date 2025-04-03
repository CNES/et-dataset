#
# Copyright: (c) 2023 CESBIO / Centre National d'Etudes Spatiales /
#            Université Paul Sabatier (UT3)
#
"""
Vegetation indices
Albedo
"""

import os
from fnmatch import fnmatch
from json import load

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
    return np.clip(
        (data.nir - data.red) / (data.nir + data.red + 1e-9), -1, 1
    )  # .transpose('y','x')


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
    return xr.apply_ufunc(ndvi_to_lai, ndvi, a, b, c, vectorize=True)


def compute_lai(
    data: xr.Dataset, image_path: str, satellite: str
) -> xr.DataArray:
    # Create a new "band" dimension from the variables
    # representing spectral bands
    band_list = [
        "green",
        "red",
        "nir",
        "swir1",
        "swir2",
        "swir2",
        "swir2",
        "swir2",
    ]
    stacked_inputs = xr.concat([data[var] for var in band_list], dim="band")

    # Assign band names or indices as a coordinate for the new dimension
    stacked_inputs = stacked_inputs.assign_coords(band=band_list)

    # Convert to a dataset with one variable
    # TODO: check mypy error
    stacked_inputs = stacked_inputs.to_dataset(name="band_data")  # type: ignore

    # Name band coordinates
    stacked_inputs["band"] = [
        "green",
        "red",
        "nir",
        "swir1",
        "swir2",
        "cos(View_Zenith)",
        "cos(Sun_Zenith)",
        "cos(Rel_Azimuth)",
    ]

    # Open metadata
    for root, _, files in os.walk(image_path):
        for name in files:
            if fnmatch(name, "*_MTL.json"):
                meta_data_file = os.path.join(root, name)
    with open(meta_data_file) as data_file:
        metadata = load(data_file)["LANDSAT_METADATA_FILE"]["IMAGE_ATTRIBUTES"]

    # Add angles
    stacked_inputs.loc[{"band": "cos(View_Zenith)"}] = np.float32(1)
    stacked_inputs.loc[{"band": "cos(Sun_Zenith)"}] = np.cos(
        np.deg2rad(90 - float(metadata["SUN_ELEVATION"])), dtype=np.float32
    )
    stacked_inputs.loc[{"band": "cos(Rel_Azimuth)"}] = np.cos(
        np.deg2rad(90 - float(metadata["SUN_AZIMUTH"])), dtype=np.float32
    )

    # Define dimensions and arrays for output data
    dims = data[band_list[0]].dims
    output_data = np.empty(shape=data[band_list[0]].shape, dtype=np.float32)
    output_flag_data = np.zeros(shape=data[band_list[0]].shape, dtype=np.int8)

    # Create output dataset
    datavars = {}
    datavars["LAI"] = (dims, output_data)
    # TODO: Check mypy error
    datavars["LAI_flag"] = (dims, output_flag_data)  # type: ignore
    output = xr.Dataset(
        data_vars=datavars,
        coords={"y": data.coords["y"], "x": data.coords["x"]},
    )

    return apply_NNT(stacked_inputs, output, ["LAI"], satellite)["LAI"]

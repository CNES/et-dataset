#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright: (c) 2023 CESBIO / Centre National d'Etudes Spatiales / Université Paul Sabatier (UT3)
#
"""
Vegetation indices
Albedo
"""
import numpy as np
import xarray as xr

def compute_ndvi(data: xr.Dataset) -> None:  
    """
    Compute NDVI
    Rouse, J., Jr.; Haas, R.; Deering, D.; Schell, J.; Harlan, J. Monitoring the Vernal Advancement and Retrogradation 
    (Green Wave Effect) of Natural Vegetation; Great Plains Corridor; 
    Texas A&M University Remote Sensing Center: College Station, TX, USA, 1974.
    """
    data['ndvi'] = np.clip((data.nir - data.red) / 
                           (data.nir + data.red + 1e-9),
                           -1,1) #.transpose('y','x')


def ndvi_to_lai(ndvi: float,
                a: float,
                b: float,
                c: float = 0.0) -> float:
    """
    Compute LAI from ndvi using an exponential mathematical form
    LAI = a x exp(b x ndvi) + c
    """
    lai = np.nan
    if not np.isnan(ndvi) and ndvi >= 0.0 and ndvi <= 1:
        lai = a * np.exp(b * ndvi) +c
        if lai < 0:
            lai = 0
        else:
            lai = np.nan
    return lai


def compute_lai_from_ndvi(data: xr.Dataset,
                          a: float,
                          b: float,
                          c: float = 0.0) -> None:  
    """
    Compute LAI from ndvi using an exponential mathematical form
    LAI = a x exp(b x ndvi) + c
    """
    if not 'ndvi' in data.variables:
        compute_ndvi(data)
    data['lai'] = xr.apply_ufunc(ndvi_to_lai, data['ndvi'],a,b,c,vectorize=True)



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


def compute_ndwi(data: xr.Dataset) -> None:  
    """
    Compute NDWI
    """
    data['ndwi'] = np.clip((data.green - data.nir) / 
                           (data.green + data.nir + 1e-9),
                           -1,1) #.transpose('y','x')


def compute_savi(data:xr.Dataset) -> None:
    """
    Compute SAVI
    Huete, A.R (August 1988). "A soil-adjusted vegetation index (SAVI)". Remote Sensing of Environment. 25 (3): 295–309.
    """
    L = 0.5
    data['savi'] = np.clip((1.0 + L) * (data.nir - data.red) / (data.nir + data.red + L), -(1.0+L), (1.0+L))


def compute_lai_from_savi(data: xr.Dataset) -> None:  
    """
    Compute LAI
    Pôças, I.; Paço, T.; Cunha, M.; Andrade, J.A.; Silvestre, J.; Sousa, A.; Santos, F.L.; Pereira, L.S.; Allen, R.G. Satellite-based evapotranspiration of a super-intensive olive orchard: Application of METRIC algorithms. Biosyst. Eng. 2014, 128, 69–81
    """
    data['lai'] = -np.log((0.69-data["savi"])/0.59)/0.91


def compute_evi2(data:xr.Dataset) -> None:
    """
    Compute EVI2
    Jiang, Z.; Huete, A.; Didan, K.; Miura, T. Development of a two-band enhanced vegetation index without a blue band. Remote Sens. Environ. 2008, 112, 3833–3845.
    """
    data['evi2'] = np.clip(2.5 * ((data.nir - data.red)/((data.nir+2.4*data.red)+1)),0.0,np.inf)


def compute_lai_from_evi2(data: xr.Dataset) -> None:  
    """
    Compute LAI
    Kang, Y.; Özdo  ̆ gan, M.; Zipper, S.C.; Román, M.O.; Walker, J.P.; Hong, S.Y.; Marshall, M.; Magliulo, V.; Moreno, J.; Alonso, L.; et al. How Universal is the Relationship between Remotely Sensed Vegetation Indices and Crop Leaf Area Index? A Global Assessment. Remote. Sens. 2016, 8, 597
    """
    data['lai'] = (2.92*np.sqrt(data["evi2"])-0.43)**2 


def compute_lai_from_ndvi(data: xr.Dataset) -> None:  
    """
    Compute LAI
    Jordi Inglada, Bio/geo-physical model calibration and inversion with probabilistic programming
    """
    data['lai'] = 0.11905544 * np.exp(data["ndvi"] * 3.45660706) - 0.06174835


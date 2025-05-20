# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales
"""
Module for DEM management
"""

import os

import numpy as np
import numpy.typing as npt
import pyproj
import rasterio as rio
import requests
import xarray as xr
from pyproj import CRS, Transformer
from rasterio.merge import merge as rio_merge
from rasterio.transform import array_bounds
from sensorsio import mgrs
from sensorsio.regulargrid import read_as_numpy

from etdataset.logging import LoggerManager
from etdataset.utils import get_mgrs_tile_names_from_roi

logger = LoggerManager.get_logger(__name__)


def get_dem_from_tile(
    tile_id: str,
    base_dir: str,
    resolution: float = 60,
) -> xr.Dataset:
    """
    Read one tile for DEM Copernicus
    Then, resample it at a specific resolution and
    compute slope et aspect

    Parameters
    ----------
    tile_id: str
        Tile ID
    resolution: str, deflaut=60
        DEM spatial resolution
    base_dir: str
        Path to the DEM directory
        Required to set MNT_PATH environment variable

    Returns
    -------
    xarr: xarray.Dataset
    """
    file_name = os.path.join(base_dir, f"COP-DEM_GLO-30-DGED_{tile_id}.tif")
    # TODO: Remove assert
    assert os.path.isfile(file_name)  # noqa
    elevation, xcoords, ycoords, crs = read_as_numpy(
        [file_name],
        resolution=resolution,
        algorithm=rio.enums.Resampling.cubic,
        dtype=np.int16,
    )
    elevation = elevation[0, 0, :, :]
    x, y = np.gradient(elevation.astype(np.float32))
    slope = np.degrees(np.arctan(np.sqrt(x * x + y * y) / resolution))
    # Aspect unfolding rules from
    # https://github.com/r-barnes/richdem/blob/ \
    # 603cd9d16164393e49ba8e37322fe82653ed5046/include/ \
    # richdem/methods/terrain_attributes.hpp#L236
    aspect = np.rad2deg(np.arctan2(x, -y))
    lt_0 = aspect < 0
    gt_90 = aspect > 90
    remaining = np.logical_and(aspect >= 0, aspect <= 90)
    aspect[lt_0] = 90 - aspect[lt_0]
    aspect[gt_90] = 360 - aspect[gt_90] + 90
    aspect[remaining] = 90 - aspect[remaining]
    left = np.min(xcoords) - resolution / 2
    top = np.max(ycoords) + resolution / 2
    transform = rio.Affine(resolution, 0.0, left, 0.0, -resolution, top)
    bounds = tuple(
        [float(x) for x in array_bounds(len(ycoords), len(xcoords), transform)]
    )
    var: dict[str, tuple[list[str], np.ndarray]] = {}
    var["height"] = (["y", "x"], elevation)
    var["slope"] = (["y", "x"], slope)
    var["aspect"] = (["y", "x"], aspect)
    xarr = xr.Dataset(
        var,
        coords={"x": xcoords, "y": ycoords},
        attrs={
            "crs": crs,
            "resolution": {"x": resolution, "y": resolution},
            "transform": transform,
            "bounds": bounds,
        },
    )
    return xarr


def get_dem_from_tiles(
    tile_ids: list[str],
    base_dir: str,
    resolution: float = 60,
) -> xr.Dataset:
    """
    Read several tiles for DEM Copernicus
    Then, resample them at a specific resolution and
    compute slope et aspect
    All tiles are merged in the same dataset.

    Parameters
    ----------
    tile_ids: List[str]
        List of tile IDs
    resolution: str, deflaut=60
        DEM spatial resolution
    base_dir: str
        Path to the DEM directory
        Required to set MNT_PATH environment variable

    Returns
    -------
    xarr: xarray.Dataset
    """
    # TODO: remove assert
    assert len(tile_ids) > 0  # noqa
    # Get CRS from first tile
    crs = f"EPSG:{mgrs.get_crs_mgrs_tile(tile_ids[0]).to_epsg()}"
    # Get file paths
    file_names = [
        os.path.join(base_dir, f"COP-DEM_GLO-30-DGED_{tile_id}.tif")
        for tile_id in tile_ids
    ]
    # Use rasterio merge to read DEM files
    elevation, transform = rio_merge(
        file_names,
        res=resolution,
        nodata=np.nan,
        resampling=rio.enums.Resampling.cubic,
        dtype=np.float32,
    )
    elevation = elevation[0, :, :]
    # Get bounds
    left, bottom, right, top = rio.transform.array_bounds(
        elevation.shape[0], elevation.shape[1], transform
    )
    xcoords: np.ndarray = np.linspace(
        left + 0.5 * resolution, right - 0.5 * resolution, elevation.shape[1]
    )

    ycoords: np.ndarray = np.linspace(
        top - 0.5 * resolution, bottom + 0.5 * resolution, elevation.shape[0]
    )
    x, y = np.gradient(elevation.astype(np.float32))
    slope = np.degrees(np.arctan(np.sqrt(x * x + y * y) / resolution))
    # Aspect unfolding rules from
    # https://github.com/r-barnes/richdem/blob/ \
    # 603cd9d16164393e49ba8e37322fe82653ed5046/include/ \
    # richdem/methods/terrain_attributes.hpp#L236
    aspect = np.rad2deg(np.arctan2(x, -y))
    lt_0 = aspect < 0
    gt_90 = aspect > 90
    remaining = np.logical_and(aspect >= 0, aspect <= 90)
    aspect[lt_0] = 90 - aspect[lt_0]
    aspect[gt_90] = 360 - aspect[gt_90] + 90
    aspect[remaining] = 90 - aspect[remaining]
    left = np.min(xcoords) - resolution / 2
    top = np.max(ycoords) + resolution / 2
    transform = rio.Affine(resolution, 0.0, left, 0.0, -resolution, top)
    bounds = tuple(
        [float(x) for x in array_bounds(len(ycoords), len(xcoords), transform)]
    )
    var: dict[str, tuple[list[str], np.ndarray]] = {}
    var["height"] = (["y", "x"], elevation)
    var["slope"] = (["y", "x"], slope)
    var["aspect"] = (["y", "x"], aspect)
    xarr = xr.Dataset(
        var,
        coords={"x": xcoords, "y": ycoords},
        attrs={
            "crs": crs,
            "resolution": {"x": resolution, "y": resolution},
            "transform": transform,
            "bounds": bounds,
        },
    )
    return xarr


def get_elevation_from_tile(
    tile_id: str,
    base_dir: str,
    resolution: float = 60,
) -> xr.DataArray:
    """
    Read one tile for DEM Copernicus

    Parameters
    ----------
    tile_id: str
        Tile ID
    resolution: str, deflaut=60
        DEM spatial resolution
    base_dir: str
        Path to the DEM directory
        Required to set MNT_PATH environment variable

    Returns
    -------
    xarr: xarray.Dataset
    """
    file_name = os.path.join(base_dir, f"COP-DEM_GLO-30-DGED_{tile_id}.tif")
    # TODO: Remove assert
    assert os.path.isfile(file_name)  # noqa
    elevation, xcoords, ycoords, crs = read_as_numpy(
        [file_name],
        resolution=resolution,
        algorithm=rio.enums.Resampling.cubic,
        dtype=np.int16,
    )
    elevation = elevation[0, 0, :, :]
    left = np.min(xcoords) - resolution / 2
    top = np.max(ycoords) + resolution / 2
    transform = rio.Affine(resolution, 0.0, left, 0.0, -resolution, top)
    bounds = tuple(
        [float(x) for x in array_bounds(len(ycoords), len(xcoords), transform)]
    )
    return xr.DataArray(
        elevation,
        coords=[ycoords, xcoords],
        dims=["y", "x"],
        attrs={
            "crs": crs,
            "resolution": {"x": resolution, "y": resolution},
            "transform": transform,
            "bounds": bounds,
        },
    )


def get_dem_from_roi(
    roi_bbox: rio.coords.BoundingBox,
    roi_crs: CRS,
    base_dir: str,
    resolution: float = 60,
) -> xr.Dataset:
    """
    Description
    -----------
    Read several tiles for DEM Copernicus based on a ROI.
    Then, resample them at a specific resolution and
    compute slope et aspect

    Parameters
    ----------
    roi_bbox: roi.coords.BoundingBox
       ROI bounding box
    roi_crs: pyproj.CRS
       ROI CRS
    resolution: str, deflaut=60
        DEM spatial resolution
    base_dir: str
        Path to the DEM directory
        Required to set MNT_PATH environment variable

    Returns
    -------
    xarr: xarray.Dataset
    """
    logger.debug(f"ROI bbox: {roi_bbox}")
    logger.debug(f"ROI crs: {roi_crs}")
    # Get DEM tiles
    tile_ids = get_mgrs_tile_names_from_roi(
        roi_bbox=roi_bbox, roi_crs=roi_crs, overlap=5
    )
    logger.debug(f"Tiles ids form DEM: {tile_ids}")
    # Get DEM from tiles
    dem = get_dem_from_tiles(
        tile_ids=tile_ids, resolution=resolution, base_dir=base_dir
    )
    # Transform to rioxarray
    dem = dem.rio.write_crs(roi_crs)
    # Crop
    dem = dem.rio.clip_box(*roi_bbox)
    crs = dem.rio.crs
    transform = dem.rio.transform(recalc=True)
    bounds = rio.coords.BoundingBox(*dem.rio.bounds())
    # Clean rio attributes
    dem = dem.drop_vars("spatial_ref", errors="ignore")
    dem.attrs["crs"] = crs
    dem.attrs["transform"] = transform
    dem.attrs["bounds"] = bounds
    return dem


def download_egm96_height() -> None:
    """
    Description
    -----------
    Check if the egm96 height exists.
    If not, download if from github

    https://github.com/OSGeo/PROJ-data/raw/refs/heads/master/us_nga/us_nga_egm08_25.tif
    """
    # Check
    data_dir = pyproj.datadir.get_data_dir()
    egm96_file = os.path.join(data_dir, "us_nga_egm96_15.tif")
    if os.path.isfile(egm96_file):
        logger.debug("EGM96 already downloaded")
        return
    egm96_url = (
        "https://github.com/OSGeo/PROJ-data/raw/refs/"
        "heads/master/us_nga/us_nga_egm96_15.tif"
    )
    with requests.get(egm96_url, stream=True) as r:
        r.raise_for_status()
        with open(egm96_file, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    logger.debug("EGM96 downloaded")


def download_egm08_height() -> None:
    """
    Description
    -----------
    Check if the egm96 height exists.
    If not, download if from github

    """
    # Check
    data_dir = pyproj.datadir.get_data_dir()
    egm08_file = os.path.join(data_dir, "us_nga_egm08_25.tif")
    if os.path.isfile(egm08_file):
        return
    egm08_url = (
        "https://github.com/OSGeo/PROJ-data/raw/refs/"
        "heads/master/us_nga/us_nga_egm08_25.tif"
    )
    with requests.get(egm08_url, stream=True) as r:
        r.raise_for_status()
        with open(egm08_file, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)


def get_egm96_height(lat: npt.ArrayLike, lon: npt.ArrayLike) -> npt.NDArray:
    """
    Description
    -----------
    Get the height above the geoid (EGM96) if the point
    were exactly on the WGS84 ellipsoid surface at that lat/lon.
    This allows you to compute the geoid undulation NN,
    which is the vertical distance between the WGS84
    ellipsoid and the EGM96 geoid at that location.
    Considering: ellipsoidal_height = 0
    The output orthometric_height will be:
    orthometric_height = 0 - N = -N
    So the orthometric height will be negative, and:
    N = -orthometric_height
    This gives the geoid height N at that latitude/longitude
    relative to the ellipsoid.

    Parameters
    ----------
    lat: float
       Latitude
    lon: float
       Longitude

    Returns
    -------

    """
    # EGM96 geoid model
    transformer = Transformer.from_crs(
        "epsg:4979",  # WGS84 3D (lat/lon/ellipsoidal height)
        "epsg:9707",  # WGS84 lat/lon + EGM96 geoid (4326+5773)
        always_xy=True,
    )
    _, _, egm96_height = transformer.transform(
        np.array(lon), np.array(lat), np.ones_like(lon)
    )
    return -egm96_height


def compute_egm96_height(data: xr.DataArray | xr.Dataset) -> xr.DataArray:
    """
    Description
    -----------
    Compute elevation using EGM96

    Parameters
    ----------
    data: xr.DataArray
        Data

    Return
    ------
    dem: xr.DataArray
        EGM96 height
    """
    # Extract coordiates
    x = data["x"].values
    y = data["y"].values
    # Create a 2D meshgrid of coordinates
    x2d, y2d = np.meshgrid(x, y)
    # Convert coordinates to lat/lon
    if hasattr(data, "rio"):
        if data.rio.crs is not None:
            crs = data.rio.crs
        else:
            raise ValueError("No CRS provided")
    elif data.attrs.get("crs", None) is not None:
        crs = data.attrs["crs"]
    else:
        raise ValueError("No CRS provided")
    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    # Convert to lat/lon using pyproj (easting, northing → lon, lat)
    lon2d, lat2d = transformer.transform(x2d, y2d)
    # Vectorize the get_elevation function
    vectorized_elevation = np.vectorize(get_egm96_height)
    # Apply it to the meshgrid
    elevation_data = vectorized_elevation(lat2d, lon2d)

    # Wrap it back into a DataArray
    return xr.DataArray(
        data=elevation_data,
        coords={"x": x, "y": y},
        dims=("y", "x"),
        name="elevation",
    )

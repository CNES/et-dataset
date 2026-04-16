import datetime as dt
import os
import tempfile
from collections.abc import Iterable
from enum import Enum

import numpy as np
import numpy.typing as npt
import rasterio as rio
import xarray as xr
from pyproj import CRS, Transformer

from etdataset.era5 import (
    ERA5Dataset,
    ERA5pressureVar,
    download,
    get_era5_dem,
    get_era5land_dem,
    read,
    rescale_temperature_with_lapserate,
)
from etdataset.interpolation import interpolate_time
from etdataset.logging import LoggerManager
from etdataset.validation_temp.temperature_rescaling import (
    add_dewpoint_to_ds,
    compute_dewpoint_lr,
    compute_lapse_rate_from_2_levels_roi,
    filter_dataset_by_pressure_levels,
    get_lapse_rate_monthly,
    get_vapor_pressure_monthly,
    normalize_longitude_latitude,
    rescale_temperature_with_variable_lapserate,
)

logger = LoggerManager.get_logger(__name__)
G_CST = 9.80665


class ERA5name(Enum):
    T2M = "t2m"
    D2M = "d2m"
    T = "t"
    TA = "ta"
    TD = "td"


NAME_MAP = {
    ERA5name.T2M.value: ERA5name.TA.value,
    ERA5name.D2M.value: ERA5name.TD.value,
    ERA5name.T.value: ERA5name.TA.value,
}


class RescalTempMethod(str, Enum):
    CONST_LR = "const_lr"
    MONTHLY_LR = "lr_per_month"
    LR_PROFILE_FIXED_LEVELS = "lr_between_925_700_levels"
    LR_PROFILE_ALT_DEP = "lr_depends_on_altitude"
    VERTICAL_INTERP = "interpolation_extrapolation_pressure_lvl"
    INTERP_PROFILE_SURF = "interpolation_pressure_and_surface_lvl"
    INTERP_LR_HYBRID = "interpolation_pressure_lvl_and_lr_surface"


METHOD_REQUIREMENTS = {
    RescalTempMethod.CONST_LR: {
        "surface": True,
        "pressure": False,
    },
    RescalTempMethod.MONTHLY_LR: {
        "surface": True,
        "pressure": False,
    },
    RescalTempMethod.LR_PROFILE_FIXED_LEVELS: {
        "surface": True,
        "pressure": True,
    },
    RescalTempMethod.LR_PROFILE_ALT_DEP: {
        "surface": True,
        "pressure": True,
    },
    RescalTempMethod.VERTICAL_INTERP: {
        "surface": False,
        "pressure": True,
    },
    RescalTempMethod.INTERP_PROFILE_SURF: {
        "surface": True,
        "pressure": True,
    },
    RescalTempMethod.INTERP_LR_HYBRID: {
        "surface": True,
        "pressure": True,
    },
}


class TempVariable(str, Enum):
    TA = "ta"
    TD = "td"


def rename_var_ds(ds: xr.Dataset, mapping: dict) -> xr.Dataset:
    """
    Rename variables in an xarray Dataset based on a mapping dictionary.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset
    mapping : dict
        Dictionary mapping original variable names to new names

    Returns
    -------
    xr.Dataset
        Dataset with renamed variables.
    """
    available = {k: v for k, v in mapping.items() if k in ds.data_vars}
    return ds.rename(available)


def normalize_variables(
    variables: Iterable[str] | None = None,
) -> set[TempVariable]:
    """
    Normalize provided variables into a set of TempVariable enums.

    Parameters
    ----------
    variables : iterable or None
        Iterable of variable names (e.g., ["ta", "td"]) or None

    Returns
    -------
    set[TempVariable]
        Set of validated TempVariable enums

    """
    if variables is None:
        return {TempVariable.TA, TempVariable.TD}
    try:
        return {TempVariable(v) for v in variables}
    except ValueError as e:
        raise ValueError(
            f"Invalid variable in {variables}. Allowed: {[v.value for v in TempVariable]}"  # noqa: E501
        ) from e


def build_output_ds(
    res_ta: xr.DataArray | None,
    res_td: xr.DataArray | None,
    variables: set[TempVariable],
) -> xr.Dataset:
    """
    Build an output xarray Dataset from interpolated variables

    Parameters
    ----------
    res_ta : xr.DataArray | None
        Interpolated air temperature.
    res_td : xr.DataArray | None
        Interpolated dew point temperature.
    variables : set[TempVariable]
        Variables to include in the output dataset.

    Returns
    -------
    xr.Dataset
        Dataset containing only requested and available variables.
    """
    data_vars = {}

    if TempVariable.TA in variables and res_ta is not None:
        data_vars["ta"] = res_ta

    if TempVariable.TD in variables and res_td is not None:
        data_vars["td"] = res_td

    return xr.Dataset(data_vars)


def crop_ds(
    ds: xr.Dataset | None,
    dem: xr.DataArray,
) -> xr.Dataset:
    """
    Crop a dataset to the area of a DEM.

    The DEM bounding box is transformed to WGS84 (EPSG:4326) before cropping
    the dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset to crop
    dem : xr.DataArray
        DEM used to define the area

    Returns
    -------
    xr.Dataset
        Cropped dataset
    """
    if ds is None:
        raise ValueError("Dataset/ DataArray to crop is None")
    minx, miny, maxx, maxy = dem.rio.bounds()

    transformer = Transformer.from_crs(dem.rio.crs, "EPSG:4326", always_xy=True)
    minlon, minlat = transformer.transform(minx, miny)
    maxlon, maxlat = transformer.transform(maxx, maxy)

    return ds.rio.clip_box(
        minx=minlon,
        miny=minlat,
        maxx=maxlon,
        maxy=maxlat,
    )


def interp_column(
    z_src: npt.ArrayLike,
    t_src: npt.ArrayLike,
    z_target: npt.ArrayLike,
) -> np.ndarray:
    """
    Perform 1D vertical interpolation of a variable along a column.

    Parameters
    ----------
    z_src : array-like
        Source vertical coordinates
    t_src : array-like
        Source variable values
    z_target : array-like
        Target vertical coordinates

    Returns
    -------
    np.ndarray
        Interpolated values at target heights
    """
    z_src = np.asarray(z_src)
    t_src = np.asarray(t_src)
    z_target = np.asarray(z_target)
    # Sort data by vertical coordinate
    idx = np.argsort(z_src)
    z_sorted = z_src[idx]
    t_sorted = t_src[idx]
    # linear interpolation
    return np.interp(z_target, z_sorted, t_sorted)


# TO DO : TYPE
def interp_column_surface(z_profile, t_profile, z_surface, t_surface, z_target):
    """
    Perform vertical interpolation

    This function adds a surface point to a vertical profile
    interpolates the variable onto a target vertical grid.

    Parameters
    ----------
    z_profile : array-like
        Vertical coordinates of the atmospheric profile.
    t_profile : array-like
        Profile variable values
    z_surface : float
        Surface elevation
    t_surface : float
        Surface variable value.
    z_target : array-like
        Target vertical coordinates

    Returns
    -------
    np.ndarray
        Interpolated values at target heights
    """
    mask = np.isfinite(z_profile) & np.isfinite(t_profile)
    z_profile = z_profile[mask]
    t_profile = t_profile[mask]

    if len(z_profile) == 0:
        return np.full(len(z_target), np.nan)

    threshold = 100
    mask = np.abs(z_profile - z_surface) > threshold

    z_profile = z_profile[mask]
    t_profile = t_profile[mask]

    z_profile = np.insert(z_profile, 0, z_surface)
    t_profile = np.insert(t_profile, 0, t_surface)

    idx = np.argsort(z_profile)
    z_profile = z_profile[idx]
    t_profile = t_profile[idx]

    return np.interp(z_target, z_profile, t_profile)


def method_const_lr(
    data: xr.Dataset,
    variables_set: set[TempVariable],
    dem: xr.DataArray,
    era5_surface: xr.Dataset,
    era5_dem_surface: xr.DataArray,
) -> xr.Dataset:
    """
    Apply a constant lapse rate correction to surface temperature variables.

    This method rescales ERA5 temperature to the target DEM elevation using
    fixed, predefined lapse rates. It assumes a constant vertical gradient for
    temperature.

    Parameters
    ----------
    data : xr.Dataset
        Input dataset to update
    variables_set : set[TempVariable]
        Set of temperature variables to process (e.g., TA, TD).
    dem : xr.DataArray
        Target DEM
    era5_surface : xr.Dataset
        ERA5 surface dataset containing variables
    era5_dem_surface : xr.DataArray
        Surface geopotential height (in meters).

    Returns
    -------
    xr.Dataset
        Updated dataset with rescaled temperature variables.

    Notes
    -----
    - Uses constant lapse rates:
        -> Air temperature: -6.5 K/km
        -> Dew point: -5.2 K/km
    """
    updated_data = data.copy()

    # AIR TEMPERATURE
    if TempVariable.TA in variables_set:
        # Apply constant lapse rate correction
        rescaled = rescale_temperature_with_lapserate(
            dem=dem,
            era5_data=era5_surface.get(TempVariable.TA, None),
            era5_dem=era5_dem_surface,
            lapse_rate=-0.0065,
            key="ta",
            description="2m air temperature",
        )
        if rescaled is not None:
            updated_data["ta"] = rescaled
            logger.debug("Add temperature:OK")

    # DEW POINT TEMPERATURE
    if TempVariable.TD in variables_set:
        rescaled = rescale_temperature_with_lapserate(
            dem=dem,
            era5_data=era5_surface.get(TempVariable.TA, None),
            era5_dem=era5_dem_surface,
            lapse_rate=-0.0052,
            key="td",
            description="dewpoint temperature",
        )
        if rescaled is not None:
            updated_data["td"] = rescaled
            logger.debug("Add temperature:OK")
    return updated_data


def method_monthly_lr(
    data: xr.Dataset,
    date: dt.datetime,
    variables_set: set[TempVariable],
    dem: xr.DataArray,
    era5_surface: xr.Dataset,
    era5_dem_surface: xr.DataArray,
) -> xr.Dataset:
    """
    Description
    -----------
    Apply a monthly varying lapse rate correction to temperature variables.

    This method rescales ERA5 temperature using lapse rates that vary depending
    on the month. This allows capturing seasonal variability in vertical
    temperature gradients.

    Lapse rate values from "A Meteorological Distribution System for
    High-Resolution Terrestrial Modeling (MicroMet)", Journal of
    Hydrometeorology, by G.E. Liston, K. Elder

    Parameters
    ----------
    data : xr.Dataset
        Input dataset to update.
    date : datetime
        Date used to determine the monthly lapse rate.
    variables_set : set[TempVariable]
        Set of temperature variables to process (e.g., TA, TD).
    dem : xr.DataArray
        Target Digital Elevation Model (DEM).
    era5_surface : xr.Dataset
        ERA5 surface dataset.
    era5_dem_surface : xr.DataArray
        Surface geopotential height (in meters).

    Returns
    -------
    xr.Dataset
        Updated dataset with rescaled temperature variables.

    """
    updated_data = data.copy()

    # AIR TEMPERATURE
    if TempVariable.TA in variables_set:
        # Get monthly lapse rate
        lr_ta_monthly = get_lapse_rate_monthly(date)

        # Apply rescaling
        rescaled = rescale_temperature_with_lapserate(
            dem=dem,
            era5_data=era5_surface.get(TempVariable.TA, None),
            era5_dem=era5_dem_surface,
            lapse_rate=lr_ta_monthly,
            key="ta",
            description="2m air temperature",
        )
        if rescaled is not None:
            updated_data["ta"] = rescaled
            logger.debug("Add temperature:OK")

    # DEW POINT TEMPERATURE
    if TempVariable.TD in variables_set:
        # Compute dew point lapse rate from monthly coefficients
        coeff = get_vapor_pressure_monthly(date)
        lr_td_monthly = compute_dewpoint_lr(coeff)

        # Apply rescaling
        rescaled = rescale_temperature_with_lapserate(
            dem=dem,
            era5_data=era5_surface.get(TempVariable.TD, None),
            era5_dem=era5_dem_surface,
            lapse_rate=lr_td_monthly,
            key="td",
            description="dewpoint temperature",
        )
        if rescaled is not None:
            updated_data["td"] = rescaled
            logger.debug("Add temperature:OK")
    return updated_data


def method_lr_profile_fixed_levels(
    data: xr.Dataset,
    variables_set: set[TempVariable],
    dem: xr.DataArray,
    era5_surface: xr.Dataset,
    era5_dem_surface: xr.DataArray,
    era5_pressure: xr.Dataset,
    era5_dem_pressure: xr.DataArray,
) -> xr.Dataset:
    """
    Description
    -----------
    Apply lapse rate correction using fixed pressure levels and altitude
    threshold.

    This method computes a lapse rate from two fixed pressure levels
    (700-925 hPa) and applies different rescaling strategies depending
    on elevation:

    - < 1500 m: use surface data
    - > 1500 m: use 850 hPa level as reference

    from "Elevation correction of ERA5-Interim temperature data in complex
    terrain", L. Gao, M. Bernhardt, and K. Schulz

    Parameters
    ----------
    data : xr.Dataset
        Input dataset to update
    variables_set : set[TempVariable]
        Variables to process
    dem : xr.DataArray
        Target DEM
    era5_surface : xr.Dataset
        ERA5 surface dataset
    era5_dem_surface : xr.DataArray
        Surface elevation (m)
    era5_pressure : xr.Dataset
        ERA5 pressure-level dataset
    era5_dem_pressure : xr.DataArray
        Pressure-level heights (m)

    Returns
    -------
    xr.Dataset
        Dataset with rescaled temperature

    Notes
    -----
    - Uses fixed pressure levels for lapse rate estimation
    - Switches reference level depending on altitude threshold (1500 m)
    """
    # Create two datasets for low and high elevation cases
    updated_data_high = data.copy()
    updated_data_low = data.copy()

    # Compute lapse rate from 700-925 hPa
    era5_700_925 = filter_dataset_by_pressure_levels(
        era5_pressure, ["700", "925"]
    )

    # Define high elevation mask
    high_alt = dem > 1500

    # Extract 850 hPa level as reference for high altitudes
    era5_850 = filter_dataset_by_pressure_levels(era5_pressure, ["850"])
    era5_850 = era5_850.rio.write_crs(CRS(4326))

    # AIR TEMPERATURE
    if TempVariable.TA in variables_set:
        lr_ta = compute_lapse_rate_from_2_levels_roi(era5_700_925)
        # WHEN DEM > 1500
        rescaled_high = rescale_temperature_with_variable_lapserate(
            dem=dem,
            era5_data=era5_850.get(TempVariable.TA, None),
            era5_dem=era5_dem_pressure.sel(pressure_level="850"),
            lapse_rate=lr_ta,
            key="ta",
            description="2m air temperature",
        )

        if rescaled_high is not None:
            updated_data_high["ta"] = rescaled_high
            logger.debug("Add temperature:OK")

        # WHEN DEM < 1500
        rescaled_low = rescale_temperature_with_variable_lapserate(
            dem=dem,
            era5_data=era5_surface.get(TempVariable.TA, None),
            era5_dem=era5_dem_surface,
            lapse_rate=lr_ta,
            key="ta",
            description="2m air temperature",
        )
        if rescaled_low is not None:
            updated_data_low["ta"] = rescaled_low
            logger.debug("Add temperature:OK")

    # DEWPOINT TEMPERATURE
    if TempVariable.TD in variables_set:
        lr_td = compute_lapse_rate_from_2_levels_roi(era5_700_925, "td")
        # WHEN DEM > 1500
        rescaled_high = rescale_temperature_with_variable_lapserate(
            dem=dem,
            era5_data=era5_850.get(TempVariable.TD, None),
            era5_dem=era5_dem_pressure.sel(pressure_level="850"),
            lapse_rate=lr_td,
            key="td",
            description="dewpoint temperature",
        )
        if rescaled_high is not None:
            updated_data_high["td"] = rescaled_high
            logger.debug("Add temperature:OK")

        # WHEN DEM < 1500
        rescaled_low = rescale_temperature_with_variable_lapserate(
            dem=dem,
            era5_data=era5_surface.get(TempVariable.TD, None),
            era5_dem=era5_dem_surface,
            lapse_rate=lr_td,
            key="td",
            description="dewpoint temperature",
        )
        if rescaled_low is not None:
            updated_data_low["td"] = rescaled_low
            logger.debug("Add temperature:OK")

    # Merge results depending on elevation
    return xr.where(high_alt, updated_data_high, updated_data_low)


def method_lr_profile_alt_dep(
    data: xr.Dataset,
    variables_set: set[TempVariable],
    dem: xr.DataArray,
    era5_surface: xr.Dataset,
    era5_dem_surface: xr.DataArray,
    era5_pressure: xr.Dataset,
    era5_dem_pressure: xr.DataArray,
) -> xr.Dataset:
    """
    Description
    -----------
    Apply altitude-dependent lapse rate correction using different pressure
    levels.

    This method computes different lapse rates depending on elevation:
    - > 1500 m: lapse rate from 700-850 hPa
    - < 1500 m: lapse rate from 925-850 hPa

    Different reference levels are used for rescaling.

    from "Elevation correction of ERA5-Interim temperature data in complex
    terrain", L. Gao, M. Bernhardt, and K. Schulz

    Parameters
    ----------
    data : xr.Dataset
        Input dataset.
    variables_set : set[TempVariable]
        Variables to process
    dem : xr.DataArray
        Target DEM
    era5_surface : xr.Dataset
        ERA5 surface dataset
    era5_dem_surface : xr.DataArray
        Surface elevation
    era5_pressure : xr.Dataset
        ERA5 pressure-level dataset
    era5_dem_pressure : xr.DataArray
        Pressure-level heights

    Returns
    -------
    xr.Dataset
        Dataset with altitude-dependent lapse rate corrections applied

    """
    # Separate datasets for high and low elevation
    updated_data_high = data.copy()
    updated_data_low = data.copy()

    # Define altitude mask
    high_alt = dem > 1500

    # Extract relevant pressure levels
    era5_850 = filter_dataset_by_pressure_levels(era5_pressure, ["850"])
    era5_700_850 = filter_dataset_by_pressure_levels(
        era5_pressure, ["700", "850"]
    )
    era5_925_850 = filter_dataset_by_pressure_levels(
        era5_pressure, ["925", "850"]
    )

    # AIR TEMPERATURE
    if TempVariable.TA in variables_set:
        # WHEN DEM > 1500
        ## High altitude lapse rate
        lr_ta_high = compute_lapse_rate_from_2_levels_roi(era5_700_850)

        rescaled_high = rescale_temperature_with_variable_lapserate(
            dem=dem,
            era5_data=era5_850.get(TempVariable.TA, None),
            era5_dem=era5_dem_pressure.sel(pressure_level="850"),
            lapse_rate=lr_ta_high,
            key="ta",
            description="2m air temperature",
        )

        if rescaled_high is not None:
            updated_data_high["ta"] = rescaled_high
            logger.debug("Add temperature:OK")

        # WHEN DEM < 1500
        lr_ta_low = compute_lapse_rate_from_2_levels_roi(era5_925_850)
        rescaled_low = rescale_temperature_with_variable_lapserate(
            dem=dem,
            era5_data=era5_surface.get(TempVariable.TA, None),
            era5_dem=era5_dem_surface,
            lapse_rate=lr_ta_low,
            key="ta",
            description="2m air temperature",
        )

        if rescaled_low is not None:
            updated_data_low["ta"] = rescaled_low
            logger.debug("Add temperature:OK")

    # DEWPOINT TEMPERATURE
    if TempVariable.TD in variables_set:
        # WHEN DEM > 1500
        lr_td_high = compute_lapse_rate_from_2_levels_roi(era5_700_850, "td")

        rescaled_high = rescale_temperature_with_variable_lapserate(
            dem=dem,
            era5_data=era5_850.get(TempVariable.TD, None),
            era5_dem=era5_dem_pressure.sel(pressure_level="850"),
            lapse_rate=lr_td_high,
            key="td",
            description="dewpoint temperature",
        )

        if rescaled_high is not None:
            updated_data_high["td"] = rescaled_high
            logger.debug("Add temperature:OK")

        # WHEN DEM < 1500
        lr_td_low = compute_lapse_rate_from_2_levels_roi(era5_925_850, "td")
        rescaled_low = rescale_temperature_with_variable_lapserate(
            dem=dem,
            era5_data=era5_surface.get(TempVariable.TD, None),
            era5_dem=era5_dem_surface,
            lapse_rate=lr_td_low,
            key="td",
            description="dewpoint temperature",
        )

        if rescaled_low is not None:
            updated_data_low["td"] = rescaled_low
            logger.debug("Add temperature:OK")
    # Merge results depending on elevation
    return xr.where(high_alt, updated_data_high, updated_data_low)


def method_vertical_interp(
    data: xr.Dataset,
    variables_set: set[TempVariable],
    dem: xr.DataArray,
    era5_pressure: xr.Dataset,
    era5_dem_pressure: xr.DataArray,
    dz: int = 225,
) -> xr.Dataset:
    """
    Description
    -----------
    Vertical interpolation from ERA5 pressure levels to DEM elevation.

    This method interpolates ERA5 temperature variables along vertical columns
    defined by pressure levels. The variables are first interpolated onto a
    regular vertical grid, then horizontally reprojected to match the DEM, and
    finally interpolated to the DEM elevation.

    Parameters
    ----------
    variables_set : set[TempVariable]
        Set of temperature variables to process (e.g., TA, TD)
    dem : xr.DataArray
        Target DEM
    era5_pressure : xr.Dataset
        ERA5 pressure-level dataset
    era5_dem_pressure : xr.DataArray
        Geopotential height at pressure levels (in meters)
    dz : int
        Vertical resolution of the interpolation grid (in meters)

    Returns
    -------
    xr.Dataset
        Dataset containing interpolated temperature variables on the DEM grid.

    Notes
    -----
    - Only pressure-level data is used (no surface constraint).
    - A regular vertical grid is constructed before interpolation.
    - Vertical extrapolation is enabled when needed.
    """
    updated_data = data.copy()

    z_min = float(era5_dem_pressure.min())
    z_max = float(era5_dem_pressure.max())

    # Create regular vertical grid
    z = np.arange(z_min, z_max, dz)

    # AIR TEMPERATURE
    if TempVariable.TA in variables_set:
        ta_interp = xr.apply_ufunc(
            interp_column,
            era5_dem_pressure,
            era5_pressure["ta"],
            input_core_dims=[["pressure_level"], ["pressure_level"]],
            output_core_dims=[["z"]],
            output_sizes={"z": len(z)},
            vectorize=True,
            kwargs={"z_target": z},
            dask="parallelized",
            output_dtypes=[float],
        )
        ta_interp = ta_interp.transpose("z", "latitude", "longitude")

        ta_interp = ta_interp.assign_coords(z=z)

    # DEWPOINT TEMPERATURE
    if TempVariable.TD in variables_set:
        td_interp = xr.apply_ufunc(
            interp_column,
            era5_dem_pressure,
            era5_pressure["td"],
            input_core_dims=[["pressure_level"], ["pressure_level"]],
            output_core_dims=[["z"]],
            output_sizes={"z": len(z)},
            vectorize=True,
            kwargs={"z_target": z},
            dask="parallelized",
            output_dtypes=[float],
        )
        td_interp = td_interp.transpose("z", "latitude", "longitude")

        td_interp = td_interp.assign_coords(z=z)

    res_ta = None
    res_td = None

    # REPROJECTION AND INTERPOLATION TO DEM
    if TempVariable.TA in variables_set:
        temp_a = xr.DataArray(
            ta_interp,
            dims=("z", "latitude", "longitude"),
            coords={
                "z": z,
                "latitude": era5_pressure.latitude,
                "longitude": era5_pressure.longitude,
            },
        ).rio.write_crs(CRS(4326))
        temp_a = temp_a.rio.reproject_match(
            dem, resampling=rio.enums.Resampling.bilinear
        )

        res_ta = temp_a.interp(z=dem, kwargs={"fill_value": "extrapolate"})
        if res_ta is not None:
            updated_data["ta"] = res_ta

    if TempVariable.TD in variables_set:
        temp_d = xr.DataArray(
            td_interp,
            dims=("z", "latitude", "longitude"),
            coords={
                "z": z,
                "latitude": era5_pressure.latitude,
                "longitude": era5_pressure.longitude,
            },
        ).rio.write_crs(CRS(4326))
        temp_d = temp_d.rio.reproject_match(
            dem, resampling=rio.enums.Resampling.bilinear
        )
        res_td = temp_d.interp(z=dem, kwargs={"fill_value": "extrapolate"})
        if res_td is not None:
            updated_data["td"] = res_td
    return updated_data


def method_interp_profile_surf(
    data: xr.Dataset,
    variables_set: set[TempVariable],
    dem: xr.DataArray,
    era5_surface: xr.Dataset,
    era5_dem_surface: xr.DataArray,
    era5_pressure: xr.Dataset,
    era5_dem_pressure: xr.DataArray,
    dz: int = 225,
) -> xr.Dataset:
    """
    Vertical profile interpolation (pressure levels + surface level).

    This method interpolates ERA5 temperature variables along vertical columns
    using both pressure-level data and surface data.

    The interpolation is performed on a regular vertical grid, then reprojected
    to match the DEM and interpolated to the DEM elevation.

    Parameters
    ----------
    variables_set : set[TempVariable]
        Set of variables (e.g., TA, TD)
    dem : xr.DataArray
        Target DEM
    era5_surface : xr.Dataset
        ERA5 surface dataset
    era5_dem_surface : xr.DataArray
        Surface geopotential height converted to meters
    era5_pressure : xr.Dataset
        ERA5 pressure-level dataset
    era5_dem_pressure : xr.DataArray
        Geopotential height at pressure levels
    dz : int, optional
        Vertical resolution of the interpolation grid (in meters)

    Returns
    -------
    xr.Dataset
        Dataset containing rescaled temperature variables on the DEM grid.

    Notes
    -----
    - Surface data is explicitly included in the vertical interpolation
    - A continuous vertical profile is constructed from surface to upper levels
    - Extrapolation is enabled when DEM elevations fall outside the vertical
    grid
    """
    updated_data = data.copy()
    era5_pressure = filter_dataset_by_pressure_levels(
        era5_pressure,
        [
            "700",
            "725",
            "750",
            "775",
            "800",
            "825",
            "850",
            "875",
            "900",
            "925",
            "950",
            "975",
        ],
    )
    era5_dem_pressure = era5_pressure[ERA5pressureVar.GEOPOTENTIAL.key] / G_CST
    # Define vertical grid bounds including both pressure levels and surface
    z_min = float(
        min(era5_dem_pressure.min().item(), era5_dem_surface.min().item())
    )
    z_max = float(
        max(era5_dem_pressure.max().item(), era5_dem_surface.max().item())
    )
    # Create regular vertical grid
    z = np.arange(z_min, z_max, dz)

    # Air temperature interpolation
    if TempVariable.TA in variables_set:
        ta_interp = xr.apply_ufunc(
            interp_column_surface,
            era5_dem_pressure,
            era5_pressure["ta"],
            era5_dem_surface,
            era5_surface["ta"],
            input_core_dims=[["pressure_level"], ["pressure_level"], [], []],
            output_core_dims=[["z"]],
            output_sizes={"z": len(z)},
            vectorize=True,
            kwargs={"z_target": z},
            dask="parallelized",
            output_dtypes=[float],
        )
        ta_interp = ta_interp.transpose("z", "latitude", "longitude")

        ta_interp = ta_interp.assign_coords(z=z)

    # Dewpoint temperature
    if TempVariable.TD in variables_set:
        td_interp = xr.apply_ufunc(
            interp_column_surface,
            era5_dem_pressure,
            era5_pressure["td"],
            era5_dem_surface,
            era5_surface["td"],
            input_core_dims=[["pressure_level"], ["pressure_level"], [], []],
            output_core_dims=[["z"]],
            output_sizes={"z": len(z)},
            vectorize=True,
            kwargs={"z_target": z},
            dask="parallelized",
            output_dtypes=[float],
        )
        td_interp = td_interp.transpose("z", "latitude", "longitude")

        td_interp = td_interp.assign_coords(z=z)

    res_ta = None
    res_td = None

    ##### Reprojection and interpolation to DEM #####
    if TempVariable.TA in variables_set:
        temp_a = xr.DataArray(
            ta_interp,
            dims=("z", "latitude", "longitude"),
            coords={
                "z": z,
                "latitude": era5_pressure.latitude,
                "longitude": era5_pressure.longitude,
            },
        ).rio.write_crs(CRS(4326))

        # Reproject to match DEM grid
        temp_a = temp_a.rio.reproject_match(
            dem, resampling=rio.enums.Resampling.bilinear
        )
        res_ta = temp_a.interp(z=dem, kwargs={"fill_value": "extrapolate"})
        if res_ta is not None:
            updated_data["ta"] = res_ta

    if TempVariable.TD in variables_set:
        temp_d = xr.DataArray(
            td_interp,
            dims=("z", "latitude", "longitude"),
            coords={
                "z": z,
                "latitude": era5_pressure.latitude,
                "longitude": era5_pressure.longitude,
            },
        ).rio.write_crs(CRS(4326))
        temp_d = temp_d.rio.reproject_match(
            dem, resampling=rio.enums.Resampling.bilinear
        )
        res_td = temp_d.interp(z=dem, kwargs={"fill_value": "extrapolate"})
        if res_td is not None:
            updated_data["td"] = res_td
        logger.info(f"updated_data = {updated_data}")
    return updated_data


def method_interp_lr_hybrid(
    data: xr.Dataset,
    variables_set: set[TempVariable],
    dem: xr.DataArray,
    era5_surface: xr.Dataset,
    era5_dem_surface: xr.DataArray,
    era5_pressure: xr.Dataset,
    era5_dem_pressure: xr.DataArray,
    dz: int = 225,
) -> xr.Dataset:
    """
    Hybrid interpolation method combining vertical interpolation and lapse rate
    correction.

    This function interpolates ERA5 temperature variables from pressure levels
    to a regular vertical grid, reprojects them to match the target DEM,
    and then interpolates them onto the DEM elevation.

    For elevations below the lowest available ERA5 pressure level, a lapse rate
    correction based on surface level is applied.

    Parameters
    ----------
    data : xr.Dataset
        Input dataset to update
    variables_set : set[TempVariable]
        Set of variables (e.g., TA, TD)
    dem : xr.DataArray
        Target DEM used
    era5_surface : xr.Dataset
        ERA5 surface dataset
    era5_dem_surface : xr.DataArray
        Surface geopotential height
    era5_pressure : xr.Dataset
        ERA5 pressure-level dataset
    era5_dem_pressure : xr.DataArray
        Geopotential height at pressure levels

    Returns
    -------
    xr.Dataset
        Updated dataset with rescaled temperature variables.

    -----
    - Vertical interpolation is performed on a regular height grid ('z').
    - Horizontal reprojection is applied to match the DEM resolution.
    - Below the lowest pressure level, a lapse rate-based extrapolation is used.
    """

    updated_data = data.copy()
    era5_pressure = filter_dataset_by_pressure_levels(
        era5_pressure,
        [
            "700",
            "725",
            "750",
            "775",
            "800",
            "825",
            "850",
            "875",
            "900",
            "925",
            "950",
            "975",
        ],
    )
    era5_dem_pressure = era5_pressure[ERA5pressureVar.GEOPOTENTIAL.key] / G_CST

    # Regular vertical grid
    z_min = float(era5_dem_pressure.min())
    z_max = float(era5_dem_pressure.max())

    z = np.arange(z_min, z_max, dz)

    # AIR TEMPERATURE
    if TempVariable.TA in variables_set:
        # Interpolate temperature along vertical columns
        ta_interp = xr.apply_ufunc(
            interp_column,
            era5_dem_pressure,
            era5_pressure["ta"],
            input_core_dims=[["pressure_level"], ["pressure_level"]],
            output_core_dims=[["z"]],
            output_sizes={"z": len(z)},
            vectorize=True,
            kwargs={"z_target": z},
            dask="parallelized",
            output_dtypes=[float],
        )
        ta_interp = ta_interp.transpose("z", "latitude", "longitude")

        ta_interp = ta_interp.assign_coords(z=z)
    # DEWPOINT TEMPERATURE
    if TempVariable.TD in variables_set:
        td_interp = xr.apply_ufunc(
            interp_column,
            era5_dem_pressure,
            era5_pressure["td"],
            input_core_dims=[["pressure_level"], ["pressure_level"]],
            output_core_dims=[["z"]],
            output_sizes={"z": len(z)},
            vectorize=True,
            kwargs={"z_target": z},
            dask="parallelized",
            output_dtypes=[float],
        )
        td_interp = td_interp.transpose("z", "latitude", "longitude")

        td_interp = td_interp.assign_coords(z=z)

    res_ta = None
    res_td = None

    if TempVariable.TA in variables_set:
        # reproject to DEM grid
        temp_a = xr.DataArray(
            ta_interp,
            dims=("z", "latitude", "longitude"),
            coords={
                "z": z,
                "latitude": era5_pressure.latitude,
                "longitude": era5_pressure.longitude,
            },
        ).rio.write_crs(CRS(4326))

        temp_a = temp_a.rio.reproject_match(
            dem, resampling=rio.enums.Resampling.bilinear
        )
        # Interpolate vertically to DEM elevation
        res_ta = temp_a.interp(z=dem)

    if TempVariable.TD in variables_set:
        temp_d = xr.DataArray(
            td_interp,
            dims=("z", "latitude", "longitude"),
            coords={
                "z": z,
                "latitude": era5_pressure.latitude,
                "longitude": era5_pressure.longitude,
            },
        ).rio.write_crs(CRS(4326))
        temp_d = temp_d.rio.reproject_match(
            dem, resampling=rio.enums.Resampling.bilinear
        )
        res_td = temp_d.interp(z=dem)

    if TempVariable.TA in variables_set:
        mask_below_ta = dem < temp_a.z.min()
        mask_above_ta = dem > temp_a.z.max()

    if TempVariable.TD in variables_set:
        mask_below_td = dem < temp_d.z.min()
        mask_above_td = dem > temp_d.z.max()

    era5_975_950 = filter_dataset_by_pressure_levels(
        era5_pressure, ["975", "950"]
    )
    # Lapse rate correction (below era5 levels)
    if TempVariable.TA in variables_set and mask_below_ta.any():
        lr_ta = compute_lapse_rate_from_2_levels_roi(era5_975_950)
        rescaled = rescale_temperature_with_variable_lapserate(
            dem=dem,
            era5_data=era5_surface.get(TempVariable.TA, None),
            era5_dem=era5_dem_surface,
            lapse_rate=lr_ta,
            key="ta",
            description="2m air temperature",
        )

        if rescaled is not None:
            updated_data["ta"] = xr.where(mask_below_ta, rescaled, res_ta)
            logger.debug("Add temperature:OK")

    if TempVariable.TD in variables_set and mask_below_td.any():
        lr_td = compute_lapse_rate_from_2_levels_roi(era5_975_950, "td")
        rescaled = rescale_temperature_with_variable_lapserate(
            dem=dem,
            era5_data=era5_surface.get(TempVariable.TD, None),
            era5_dem=era5_dem_surface,
            lapse_rate=lr_td,
            key="td",
            description="dewpoint temperature",
        )

        if rescaled is not None:
            updated_data["td"] = xr.where(mask_below_td, rescaled, res_td)
            logger.debug("Add temperature:OK")

    if (TempVariable.TA in variables_set and mask_above_ta.any()) or (
        TempVariable.TD in variables_set and mask_above_td.any()
    ):
        logger.warning("DEM above ERA5 levels: add pressure levels")
    return updated_data


def add_temp(
    data: xr.Dataset,
    path: str | None = None,
    dataset: ERA5Dataset = ERA5Dataset.ERA5,
    variables: TempVariable | Iterable[TempVariable] | None = None,
    method: RescalTempMethod = RescalTempMethod.CONST_LR,
):
    """
    Description
    -----------
    Add temperature variables to a dataset using ERA5/ERA5-Land data.

    This function retrieves the required ERA5 datasets (surface and/or pressure)
    depending on the selected rescaling method, processes them, and applies
    a temperature rescaling algorithm.

    Parameters
    ----------
    data : xr.Dataset
        Input dataset containing at least:
        - 'height' variable (DEM)
        - 'vis_date' and 'vis_time' attributes
        - CRS information

    path : str | None
        Path to a directory containing ERA5 data files

    dataset : ERA5Dataset
        Surface dataset to use (ERA5 or ERA5-Land)

    variables : TempVariable | Iterable[TempVariable] | None,
        Temperature variables to compute (e.g., TA, TD)

    method : RescalTempMethod
        Temperature rescaling method to apply

    Returns
    -------
    xr.Dataset
        Dataset with added/rescaled temperature variables.

    """

    logger.info(f"ADD TEMP | method={method}")

    ###################### METHOD REQUIREMENTS ################################
    req = METHOD_REQUIREMENTS.get(method)
    if req is None:
        raise ValueError(f"Unsupported method: {method}")

    ###################### CHECK INPUTS #######################################
    if data.attrs.get("vis_date", None) is None:
        raise ValueError("Vis date attribute is missing in dataset")

    if data.attrs.get("vis_time", None) is None:
        raise ValueError("Vis time attribute is missing in dataset")

    if len(data.data_vars) == 0:
        raise ValueError("Dataset is empty")

    date = dt.datetime.combine(data.attrs["vis_date"], data.attrs["vis_time"])

    ###################### GET ERA5 DATAS ####################################
    # Download data
    if path is None:
        logger.debug("Download only required ERA5 data")
        temp_dir = tempfile.TemporaryDirectory()
        path = temp_dir.name
        logger.debug(f"Temp dir: {path}")

        if req["surface"]:
            logger.debug(f"Downloading surface dataset: {dataset.key}")
            download(date=date, dataset=dataset, path=path)

        if req["pressure"]:
            logger.debug("Downloading ERA5 pressure dataset")
            download(date=date, dataset=ERA5Dataset.ERA5PRESSURE, path=path)

    # Product path data
    surface_product_path = os.path.join(
        path,
        "ERA5_data",
        f"download_{dataset.key}_{date.strftime('%Y-%m-%d')}.zip",
    )
    pressure_product_path = os.path.join(
        path,
        "ERA5_data",
        f"download_era5_pressure_{date.strftime('%Y-%m-%d')}.zip",
    )

    logger.debug(f"Surface path: {surface_product_path}")
    logger.debug(f"Pressure path: {pressure_product_path}")

    # Check file
    use_surface = req["surface"] and os.path.isfile(surface_product_path)
    use_pressure = req["pressure"] and os.path.isfile(pressure_product_path)

    if req["surface"] and not use_surface:
        raise ValueError(
            f"Missing surface file ({dataset.key}) for method {method}"
        )

    if not use_surface:
        logger.warning(f"ERA5 surface not used for method {method}")

    if req["pressure"] and not use_pressure:
        raise ValueError(f"Missing ERA5 pressure file for method {method}")

    if not use_pressure:
        logger.warning(f"ERA5 pressure not used for method {method}")

    ######################## EXTRACT DEM ######################################
    # DEM
    if data.rio.crs is None:
        if data.attrs.get("crs") is not None:
            crs = data.attrs["crs"]
            data = data.rio.write_crs(crs)
        raise ValueError("crs attribute is missing in dataset")

    crs = data.rio.crs

    height = data.get("height", None)

    if height is None:
        raise ValueError("DEM (height) is missing in dataset")

    dem = height.rio.write_crs(crs)

    ###################### VARIABLES ##########################################
    variables_set = normalize_variables(variables)

    ###################### READ ERA5 FILES ####################################
    era5_surface = None
    era5_pressure = None

    # SURFACE DATA
    if use_surface and surface_product_path is not None:
        era5_surface_data = read(product=surface_product_path)

        if era5_surface_data is not None:
            era5_surface_data = interpolate_time(
                data=era5_surface_data, date=date
            )
            era5_surface = normalize_longitude_latitude(era5_surface_data)
            era5_surface = rename_var_ds(era5_surface, NAME_MAP)

            if era5_surface is not None:
                era5_surface = era5_surface.rio.write_crs("EPSG:4326")
                era5_surface = crop_ds(era5_surface, dem)
        else:
            logger.warning("ERA5 surface read failed")

    # PRESSURE DATA
    if use_pressure and pressure_product_path is not None:
        era5_pressure_data = read(product=pressure_product_path)

        if era5_pressure_data is not None:
            era5_pressure_data = interpolate_time(
                data=era5_pressure_data, date=date
            )
            era5_pressure = normalize_longitude_latitude(era5_pressure_data)

            if TempVariable.TD in variables_set:
                era5_pressure = add_dewpoint_to_ds(era5_pressure)

            era5_pressure = rename_var_ds(era5_pressure, NAME_MAP)

            if era5_pressure is not None:
                era5_pressure = era5_pressure.rio.write_crs("EPSG:4326")
                era5_pressure = crop_ds(era5_pressure, dem)
        else:
            logger.warning("ERA5 pressure read failed")

    ###################### BUILD ERA5 DEM ######################################
    ## HEIGHT ERA5 SURFACE
    era5_dem_surface = None
    if era5_surface is not None:
        if dataset == ERA5Dataset.ERA5:
            era5_dem_surface_init = get_era5_dem()
        elif dataset == ERA5Dataset.ERA5LAND:
            era5_dem_surface_init = get_era5land_dem()
        era5_dem_surface = xr.DataArray(
            era5_dem_surface_init.data,
            dims=("latitude", "longitude"),
            coords={
                "latitude": era5_surface_data.latitude,
                "longitude": era5_surface_data.longitude,
            },
        ).rio.write_crs(CRS(4326))
        era5_dem_surface_ = era5_dem_surface.to_dataset(name="dem")
        era5_dem_surface_ = normalize_longitude_latitude(era5_dem_surface_)
        era5_dem_surface = era5_dem_surface_["dem"]
        era5_dem_surface = era5_dem_surface.rio.write_crs("EPSG:4326")
        era5_dem_surface = crop_ds(era5_dem_surface, dem)

    ## HEIGHT ERA5 PRESSURE
    era5_dem_pressure = None
    if era5_pressure is not None:
        era5_dem_pressure = (
            era5_pressure[ERA5pressureVar.GEOPOTENTIAL.key] / G_CST
        )
        era5_dem_pressure = era5_dem_pressure.rio.write_crs("EPSG:4326")
        if era5_dem_pressure is not None:
            era5_dem_pressure_ = era5_dem_pressure.to_dataset(name="dem")
            era5_dem_pressure_ = crop_ds(era5_dem_pressure_, dem)
            era5_dem_pressure = era5_dem_pressure_["dem"]

    ##################### PROCESSING METHODS ###################################
    if method == RescalTempMethod.CONST_LR:
        if era5_surface is None or era5_dem_surface is None:
            raise ValueError(
                "Missing ERA5 surface or DEM surface for CONST_LR method"
            )

        rescaled_data = method_const_lr(
            data=data,
            variables_set=variables_set,
            dem=dem,
            era5_surface=era5_surface,
            era5_dem_surface=era5_dem_surface,
        )

    if method == RescalTempMethod.MONTHLY_LR:
        if era5_surface is None or era5_dem_surface is None:
            raise ValueError(
                "Missing ERA5 surface or DEM surface for MONTHLY_LR method"
            )
        rescaled_data = method_monthly_lr(
            data=data,
            date=date,
            variables_set=variables_set,
            dem=dem,
            era5_surface=era5_surface,
            era5_dem_surface=era5_dem_surface,
        )

    if method == RescalTempMethod.LR_PROFILE_FIXED_LEVELS:
        if (
            era5_surface is None
            or era5_dem_surface is None
            or era5_pressure is None
            or era5_dem_pressure is None
        ):
            raise ValueError(
                "Missing ERA5 datas for LR_PROFILE_FIXED_LEVELS method"
            )
        rescaled_data = method_lr_profile_fixed_levels(
            data=data,
            variables_set=variables_set,
            dem=dem,
            era5_surface=era5_surface,
            era5_dem_surface=era5_dem_surface,
            era5_pressure=era5_pressure,
            era5_dem_pressure=era5_dem_pressure,
        )
    if method == RescalTempMethod.LR_PROFILE_ALT_DEP:
        if (
            era5_surface is None
            or era5_dem_surface is None
            or era5_pressure is None
            or era5_dem_pressure is None
        ):
            raise ValueError("Missing ERA5 datas for LR_PROFILE_ALT_DEP method")
        rescaled_data = method_lr_profile_alt_dep(
            data=data,
            variables_set=variables_set,
            dem=dem,
            era5_surface=era5_surface,
            era5_dem_surface=era5_dem_surface,
            era5_pressure=era5_pressure,
            era5_dem_pressure=era5_dem_pressure,
        )

    if method == RescalTempMethod.VERTICAL_INTERP:
        if era5_pressure is None or era5_dem_pressure is None:
            raise ValueError(
                "Missing ERA5 pressure or DEM pressure for VERTICAL_INTERP "
                r"\method"
            )
        rescaled_data = method_vertical_interp(
            data=data,
            variables_set=variables_set,
            dem=dem,
            era5_pressure=era5_pressure,
            era5_dem_pressure=era5_dem_pressure,
            dz=100,
        )

    if method == RescalTempMethod.INTERP_PROFILE_SURF:
        if (
            era5_surface is None
            or era5_dem_surface is None
            or era5_pressure is None
            or era5_dem_pressure is None
        ):
            raise ValueError("Missing ERA5 datas for INTERP_PROFILE_SURF")
        rescaled_data = method_interp_profile_surf(
            data=data,
            variables_set=variables_set,
            dem=dem,
            era5_surface=era5_surface,
            era5_dem_surface=era5_dem_surface,
            era5_pressure=era5_pressure,
            era5_dem_pressure=era5_dem_pressure,
            dz=100,
        )

    if method == RescalTempMethod.INTERP_LR_HYBRID:
        if (
            era5_surface is None
            or era5_dem_surface is None
            or era5_pressure is None
            or era5_dem_pressure is None
        ):
            raise ValueError("Missing ERA5 datas for INTERP_LR_HYBRID method")
        rescaled_data = method_interp_lr_hybrid(
            data=data,
            variables_set=variables_set,
            dem=dem,
            era5_surface=era5_surface,
            era5_dem_surface=era5_dem_surface,
            era5_pressure=era5_pressure,
            era5_dem_pressure=era5_dem_pressure,
            dz=100,
        )
    return rescaled_data

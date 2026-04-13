import datetime as dt
import os
from collections.abc import Iterable
from enum import Enum

import numpy as np
import rasterio as rio
import xarray as xr
from pyproj import CRS

from etdataset.era5 import (
    ERA5pressureVar,
    get_era5_dem,
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


def rename_var_era5(ds: xr.Dataset, mapping: dict):
    available = {k: v for k, v in mapping.items() if k in ds.data_vars}
    return ds.rename(available)


class TempVariable(str, Enum):
    TA = "ta"
    TD = "td"


def normalize_variables(variables):
    if variables is None:
        return {TempVariable.TA, TempVariable.TD}
    try:
        return {TempVariable(v) for v in variables}
    except ValueError as e:
        raise ValueError(
            f"Invalid variable in {variables}. Allowed: {[v.value for v in TempVariable]}"  # noqa: E501
        ) from e


def build_output_dataset(res_ta, res_td, variables: set[TempVariable]):
    data_vars = {}

    if TempVariable.TA in variables and res_ta is not None:
        data_vars["ta"] = res_ta

    if TempVariable.TD in variables and res_td is not None:
        data_vars["td"] = res_td

    return xr.Dataset(data_vars)


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


def method_const_lr(
    data: xr.Dataset,
    variables_set: set[TempVariable],
    dem: xr.DataArray,
    era5_surface: xr.Dataset,
    era5_dem_surface: xr.DataArray,
) -> xr.Dataset:
    updated_data = data.copy()
    if TempVariable.TA in variables_set:
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
    updated_data = data.copy()

    if TempVariable.TA in variables_set:
        lr_ta_monthly = get_lapse_rate_monthly(date)
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

    if TempVariable.TD in variables_set:
        coeff = get_vapor_pressure_monthly(date)
        lr_td_monthly = compute_dewpoint_lr(coeff)
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
    updated_data_high = data.copy()
    updated_data_low = data.copy()
    era5_700_925 = filter_dataset_by_pressure_levels(
        era5_pressure, ["700", "925"]
    )
    high_alt = dem > 1500
    era5_850 = filter_dataset_by_pressure_levels(era5_pressure, ["850"])
    era5_850 = era5_850.rio.write_crs(CRS(4326))

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
    updated_data_high = data.copy()
    updated_data_low = data.copy()

    high_alt = dem > 1500
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

    return xr.where(high_alt, updated_data_high, updated_data_low)


def method_vertical_interp(
    variables_set: set[TempVariable],
    dem: xr.DataArray,
    era5_pressure: xr.Dataset,
    era5_dem_pressure: xr.DataArray,
) -> xr.Dataset:
    z_min = float(era5_dem_pressure.min())
    z_max = float(era5_dem_pressure.max())
    z = np.arange(z_min, z_max, 100)

    # AIR TEMPERATURE
    if TempVariable.TA in variables_set:
        ta_interp = np.full(
            (
                len(z),
                len(era5_pressure.latitude),
                len(era5_pressure.longitude),
            ),
            np.nan,
        )

    # DEWPOINT TEMPERATURE
    if TempVariable.TD in variables_set:
        td_interp = np.full(
            (
                len(z),
                len(era5_pressure.latitude),
                len(era5_pressure.longitude),
            ),
            np.nan,
        )

    for i in range(len(era5_pressure.latitude.values)):
        for j in range(len(era5_pressure.longitude.values)):
            z_profile = era5_dem_pressure[0, :, i, j].values

            # Sorting profile
            idx = np.argsort(z_profile)
            z_profile = z_profile[idx]

            # AIR TEMPERATURE
            if TempVariable.TA in variables_set:
                ta_profile = era5_pressure["ta"][0, :, i, j].values
                ta_profile = ta_profile[idx]

                # vertical interpolation
                ta_interp[:, i, j] = np.interp(
                    z,
                    z_profile,
                    ta_profile,
                )

            # DEWPOINT TEMPERATURE
            if TempVariable.TD in variables_set:
                td_profile = era5_pressure["td"][0, :, i, j].values
                td_profile = td_profile[idx]

                # vertical interpolation
                td_interp[:, i, j] = np.interp(
                    z,
                    z_profile,
                    td_profile,
                )
    res_ta = None
    res_td = None
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
    return build_output_dataset(res_ta, res_td, variables_set)


def method_interp_profile_surf(
    variables_set: set[TempVariable],
    dem: xr.DataArray,
    era5_surface: xr.Dataset,
    era5_dem_surface: xr.DataArray,
    era5_pressure: xr.Dataset,
    era5_dem_pressure: xr.DataArray,
) -> xr.Dataset:
    z_min = float(
        min(era5_dem_pressure.min().item(), era5_dem_surface.min().item())
    )
    z_max = float(
        max(era5_dem_pressure.max().item(), era5_dem_surface.max().item())
    )
    z = np.arange(z_min, z_max, 100)
    if TempVariable.TA in variables_set:
        ta_interp = np.full(
            (
                len(z),
                len(era5_pressure.latitude),
                len(era5_pressure.longitude),
            ),
            np.nan,
        )
    if TempVariable.TD in variables_set:
        td_interp = np.full(
            (
                len(z),
                len(era5_pressure.latitude),
                len(era5_pressure.longitude),
            ),
            np.nan,
        )
    for i in range(len(era5_pressure.latitude.values)):
        for j in range(len(era5_pressure.longitude.values)):
            z_profile = era5_dem_pressure[0, :, i, j].values
            z_surface = era5_dem_surface[i, j].values

            threshold = 100
            mask = np.abs(z_profile - z_surface) > threshold
            z_profile = z_profile[mask]

            z_profile = np.insert(z_profile, 0, z_surface)

            # Sorting profile
            idx = np.argsort(z_profile)
            z_profile = z_profile[idx]
            if TempVariable.TA in variables_set:
                ta_profile = era5_pressure["ta"][0, :, i, j].values
                ta_surface = era5_surface["ta"][0, i, j].values

                ta_profile = ta_profile[mask]
                ta_profile = np.insert(ta_profile, 0, ta_surface)
                ta_profile = ta_profile[idx]
                # vertical interpolation
                ta_interp[:, i, j] = np.interp(
                    z,
                    z_profile,
                    ta_profile,
                )
            if TempVariable.TD in variables_set:
                td_profile = era5_pressure["td"][0, :, i, j].values
                td_surface = era5_surface["td"][0, i, j].values

                td_profile = td_profile[mask]
                td_profile = np.insert(td_profile, 0, td_surface)
                td_profile = td_profile[idx]
                # vertical interpolation
                td_interp[:, i, j] = np.interp(
                    z,
                    z_profile,
                    td_profile,
                )
    res_ta = None
    res_td = None
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

    return build_output_dataset(res_ta, res_td, variables_set)


def method_interp_lr_hybrid(
    data: xr.Dataset,
    variables_set: set[TempVariable],
    dem: xr.DataArray,
    era5_surface: xr.Dataset,
    era5_dem_surface: xr.DataArray,
    era5_pressure: xr.Dataset,
    era5_dem_pressure: xr.DataArray,
) -> xr.Dataset:
    updated_data = data.copy()
    z_min = float(
        min(era5_dem_pressure.min().item(), era5_dem_surface.min().item())
    )
    z_max = float(
        max(era5_dem_pressure.max().item(), era5_dem_surface.max().item())
    )
    z = np.arange(z_min, z_max, 100)

    # AIR TEMPERATURE
    if TempVariable.TA in variables_set:
        ta_interp = np.full(
            (
                len(z),
                len(era5_pressure.latitude),
                len(era5_pressure.longitude),
            ),
            np.nan,
        )
    # DEWPOINT TEMPERATURE
    if TempVariable.TD in variables_set:
        td_interp = np.full(
            (
                len(z),
                len(era5_pressure.latitude),
                len(era5_pressure.longitude),
            ),
            np.nan,
        )
    for i in range(len(era5_pressure.latitude.values)):
        for j in range(len(era5_pressure.longitude.values)):
            z_profile = era5_dem_pressure[0, :, i, j].values

            # Sorting profile
            idx = np.argsort(z_profile)
            z_profile = z_profile[idx]

            # AIR TEMPERATURE
            if TempVariable.TA in variables_set:
                ta_profile = era5_pressure["ta"][0, :, i, j].values
                ta_profile = ta_profile[idx]

                # vertical interpolation
                ta_interp[:, i, j] = np.interp(
                    z,
                    z_profile,
                    ta_profile,
                    left=np.nan,
                    right=np.nan,
                )

            # DEWPOINT TEMPERATURE
            if TempVariable.TD in variables_set:
                td_profile = era5_pressure["td"][0, :, i, j].values
                td_profile = td_profile[idx]

                # vertical interpolation
                td_interp[:, i, j] = np.interp(
                    z,
                    z_profile,
                    td_profile,
                    left=np.nan,
                    right=np.nan,
                )
    res_ta = None
    res_td = None
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
    mnt_path: str,
    era5_surface_path: str | None = None,
    era5_pressure_path: str | None = None,
    variables: TempVariable | Iterable[TempVariable] | None = None,
    method: RescalTempMethod = RescalTempMethod.CONST_LR,
):
    """ """

    logger.info(f"ADD TEMP | method={method}")

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
    if not mnt_path:
        logger.warning("DEM (mnt_path) is missing")
        return None

    if not os.path.exists(mnt_path):
        logger.warning(f"DEM path does not exist: {mnt_path}")
        return None

    use_surface = (
        req["surface"]
        and era5_surface_path
        and os.path.exists(era5_surface_path)
    )
    use_pressure = (
        req["pressure"]
        and era5_pressure_path
        and os.path.exists(era5_pressure_path)
    )

    if req["surface"] and not use_surface:
        raise ValueError(f"Missing ERA5 surface file for method {method}")
    if not use_surface:
        logger.warning(f"ERA5 surface not used for method {method}")

    if req["pressure"] and not use_pressure:
        raise ValueError(f"Missing ERA5 pressure file for method {method}")
    if not use_pressure:
        logger.warning(f"ERA5 pressure not used for method {method}")

    ######################## EXTRACT DATA ######################################
    date = dt.datetime.combine(data.attrs["vis_date"], data.attrs["vis_time"])
    ## DEM
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

    ## VARIABLES (TA OR TD OR TA AND TD)
    variables_set = normalize_variables(variables)

    ## READ ERA5 FILES
    era5_surface = None
    era5_pressure = None

    if use_surface and era5_surface_path is not None:
        era5_surface_data = read(product=era5_surface_path)
        if era5_surface_data is not None:
            era5_surface_data = interpolate_time(
                data=era5_surface_data, date=date
            )
            era5_surface = normalize_longitude_latitude(era5_surface_data)
            era5_surface = rename_var_era5(era5_surface, NAME_MAP)
        else:
            logger.warning("ERA5 surface read failed")

    if use_pressure and era5_pressure_path is not None:
        era5_pressure_data = read(product=era5_pressure_path)
        if era5_pressure_data is not None:
            era5_pressure_data = interpolate_time(
                data=era5_pressure_data, date=date
            )
            era5_pressure = normalize_longitude_latitude(era5_pressure_data)
            if TempVariable.TD in variables_set:
                era5_pressure = add_dewpoint_to_ds(era5_pressure)
            era5_pressure = rename_var_era5(era5_pressure, NAME_MAP)
        else:
            logger.warning("ERA5 pressure read failed")

    ###################### HEIGHT DATAS ######################################
    ## HEIGHT ERA5 SURFACE
    era5_dem_surface = None
    if era5_surface is not None:
        era5_dem_surface_init = get_era5_dem()
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

    ## HEIGHT ERA5 PRESSURE
    era5_dem_pressure = None
    if era5_pressure is not None:
        era5_dem_pressure = (
            era5_pressure[ERA5pressureVar.GEOPOTENTIAL.key] / G_CST
        )

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
            variables_set=variables_set,
            dem=dem,
            era5_pressure=era5_pressure,
            era5_dem_pressure=era5_dem_pressure,
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
            variables_set=variables_set,
            dem=dem,
            era5_surface=era5_surface,
            era5_dem_surface=era5_dem_surface,
            era5_pressure=era5_pressure,
            era5_dem_pressure=era5_dem_pressure,
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
        )
    return rescaled_data

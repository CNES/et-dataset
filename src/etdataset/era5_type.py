from dataclasses import dataclass
from enum import Enum

from etdataset.logging import LoggerManager

logger = LoggerManager.get_logger(__name__)

# Gravitational constant
G_CST = 9.80665


class ERA5Exception(Exception):
    """
    Exception for ERA5
    """


@dataclass
class DatasetInfo:
    """Class for dataset info"""

    key: str
    label: str
    variables: list[str]


class ERA5Dataset(DatasetInfo, Enum):
    """
    ERA5 dataset
    """

    ERA5 = (
        "era5",
        "reanalysis-era5-single-levels",
        [
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "2m_dewpoint_temperature",
            "2m_temperature",
            "surface_solar_radiation_downward_clear_sky",
            "surface_solar_radiation_downwards",
            "surface_thermal_radiation_downward_clear_sky",
            "surface_thermal_radiation_downwards",
            "total_column_ozone",
            "total_column_water",
            "total_precipitation",
            "total_column_water_vapour",
        ],
    )
    ERA5LAND = (
        "era5land",
        "reanalysis-era5-land",
        [
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "2m_dewpoint_temperature",
            "2m_temperature",
            "surface_solar_radiation_downwards",
            "surface_thermal_radiation_downwards",
            "total_precipitation",
            "total_evaporation",
            "surface_runoff",
            "skin_reservoir_content",
            "volumetric_soil_water_layer_1",
        ],
    )

    ERA5PRESSURE = (
        "era5_pressure",
        "reanalysis-era5-pressure-levels",
        ["Temperature", "Geopotential", "Relative humidity"],
    )


@dataclass
class ERA5DataInfo:
    """Class for describing ERA5 data"""

    key: str
    label: str
    unit: str


class ERA5Var(ERA5DataInfo, Enum):
    """
    ERA5 variables
    """

    DEWPOINT_TEMPERATURE = ("d2m", "2m dewpoint temperature", "K")
    TEMPERATURE = ("t2m", "2m temperature", "K")
    GEOPOTENTIAL = ("z", "Geopotential", "m2 s-2")
    HEIGHT = ("h", "Geopotential height", "m")
    SURFACE_PRESSURE = ("sp", "Surface pressure", "Pa")
    SURFACE_SOLAR_RADIATION_DOWNWARD_CLEAR_SKY = (
        "ssrdc",
        "Surface solar radiation downward, clear sky",
        "J m-2",
    )
    SURFACE_SOLAR_RADIATION_DOWNWARD = (
        "ssrd",
        "Surface solar radiation downwards",
        "J m-2",
    )
    SURFACE_THERMAL_RADIATION_DOWNWARD_CLEAR_SKY = (
        "strdc",
        "Surface thermal radiation downward, clear sky",
        "J m-2",
    )
    SURFACE_THERMAL_RADIATION_DOWNWARD = (
        "strd",
        "Surface thermal radiation downwards",
        "J m-2",
    )
    TOTAL_COLUMN_OZONE = ("tco3", "Total column ozone", "kg m-2")
    TOTAL_COLUMN_WATER = ("tcw", "Total column water", "kg m-2")
    TOTAL_COLUMN_WATER_VAPOR = ("tcwv", "Total column water vapour", "kg m-2")
    TOTAL_PRECIPITATION = ("tp", "Total precipitation", "m")
    U_WIND = ("u10", "10m u-component of wind", "m s-1")
    V_WIND = ("v10", "10m v-component of wind", "m s-1")
    TOTAL_EVAPORATION = ("e", "Total evaporation", "m")
    SURFACE_RUNOFF = ("sro", "Surface runoff", "m")
    SKIN_RESERVOIR_CONTENT = ("src", "Skin reservoir content", "m")
    SOIL_WATER_LEVEL1 = ("swvl1", "Volumetric soil water level 1", "m-3 m3")

    @classmethod
    def from_key(cls, key):
        """
        Create enum from a key value
        """
        for value in cls:
            if value.key == key:
                return value
        raise ValueError(f"No variable found with key {key}")

    @classmethod
    def _missing_(cls, value):
        """
        Overload the missing method to call from_key method
        if enum is instanciated with a string
        """
        if isinstance(value, str):
            return cls.from_key(value)
        return super()._missing_(value)


class ERA5pressureVar(ERA5DataInfo, Enum):
    """
    ERA5 pressure variables
    """

    TEMPERATURE = ("t", "Temperature", "K")
    GEOPOTENTIAL = ("z", "Geopotential", "m2 s-2")
    RELATIVE_HUMIDITY = ("r", "Relative humidity", "%")

    @classmethod
    def from_key(cls, key):
        """
        Create enum from a key value
        """
        for value in cls:
            if value.key == key:
                return value
        raise ValueError(f"No variable found with key {key}")

    @classmethod
    def _missing_(cls, value):
        """
        Overload the missing method to call from_key method
        if enum is instanciated with a string
        """
        if isinstance(value, str):
            return cls.from_key(value)
        return super()._missing_(value)

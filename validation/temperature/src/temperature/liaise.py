# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales

"""
Module for liaise data
"""

import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from etdataset.dewpoint_temp import compute_dewpoint_temp
from etdataset.logging import LoggerManager
from etdataset.utils import fix_hour_format

logger = LoggerManager.get_logger(__name__)
G_CST = 9.80665


METHOD = ("method_", "base", "era5")
NAME_MAP = {
    # ELS-PLANS
    "2mB_RHUM": "rh",  # 2mB:     RHUM : 1.2m mean Rotronics Relative humidity [%]  # noqa: E501
    "2mB_RTEMP": "ta",  # 2mB:    RTEMP : 1.2m mean Rotronics Temperature [deg C]  # noqa: E501
    # IRTA-CORN
    "TIMESTAMP": "timestamp",
    "TA_3_1_1": "ta",  # "TA_3_1_1", Average temperature (C) measured using temperature/humidity probe (HC2S3).  # noqa: E501
    "RH_3_1_1": "rh",  # "RH_3_1_1", Average relative humidity (%) measured using temperature/humidity probe (HC2S3).  # noqa: E501
    "T_DP_3_1_1": "td_mesured",  # "T_DP_3_1_1", Average dew point temperaure (C) measured using temperature/humidity probe (HC2S3).  # noqa: E501
    # PREIXANA & LA-CENDROSA
    "ta_2": "ta",  # ta_2 : air temperature measured at 2m (celsius)
    "hur_2": "rh",  # hur_2 : relative humidity at 2m (%)
}
MISSING_VALUE = 1e11
MISSING_THRESHOLD = 1e10

VALUE_RE = re.compile(r"^[+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?$")
FLAG_RE = re.compile(r"^[A-Za-z]$")


KNOWN_FLAGS = {"m", "D", "X"}


def get_method_paths(path: str, year: int, month: int, day: int) -> list[str]:
    """
    Description
    -----------
    Return a list of filepaths .nc of methods for a date (year, month, day)

    Parameters
    ----------
    path : str
        Path to stations data
    year : int
        Year  (e.g. 2021)
    month : int
        Month   (e.g. 4)
    day : int
        Day   (e.g. 11)

    Returns
    -------
    List of absolute path to found files
    """
    station_dir = os.path.abspath(path)
    # station_dir = os.path.join(path, "stations")
    if not os.path.isdir(station_dir):
        raise FileNotFoundError(f"Directory not found : {station_dir}")

    # updated_YYYYMMDD_120000.nc
    filename = f"updated_{year:04d}{month:02d}{day:02d}_120000.nc"

    month_folder = f"{month:02d}"

    found_paths = []

    for entry in sorted(os.scandir(station_dir), key=lambda e: e.name):
        if not entry.is_dir():
            continue
        if not any(entry.name.startswith(p) for p in METHOD):
            continue

        method_dir = entry.path

        year_dir = os.path.join(method_dir, str(year))
        if not os.path.isdir(year_dir):
            raise FileNotFoundError(
                f"[{entry.name}] Directory not found for year : {year_dir}"
            )

        available_months = [d.name for d in os.scandir(year_dir) if d.is_dir()]
        matched_month_dir = None
        for m in available_months:
            if m == month_folder:
                matched_month_dir = os.path.join(year_dir, m)
                break

        if matched_month_dir is None:
            raise ValueError(
                f"[{entry.name}] Month '{month_folder}' not found in {year_dir}"
                f"Months founded : {sorted(available_months)}"
            )

        file_path = os.path.join(matched_month_dir, filename)
        if os.path.isfile(file_path):
            found_paths.append(file_path)
        else:
            logger.info(f"Absent file : {file_path}", file=sys.stderr)

    return found_paths


def read_netcdf_liaise_data(file_path: str) -> pd.DataFrame:
    """
    Description
    -----------
    Read NetCDF Liaise data files from a directory and return a DataFrame
    containing air temperature, relative humidity, and dewpoint temperature

    Parameters
    ----------
    file_path : str
        Path to the directory containing the .nc files

    Returns
    -------
    pd.DataFrame
    """
    input_path = os.path.abspath(file_path)

    files = sorted(Path(input_path).glob("*.nc"))

    if not files:
        logger.warning(f"No Netcdf file found at {input_path}")
        return pd.DataFrame()

    ds = xr.open_mfdataset(files, combine="by_coords")
    rename_map = {
        old: new
        for old, new in NAME_MAP.items()
        if old in ds.variables or old in ds.dims
    }
    ds = ds.rename(rename_map)

    # logger.info(f"ds : {ds}")
    # logger.info(f"Dimensions: {dict(ds.dims)}")
    # logger.info(f"Coordonnées: {list(ds.coords)}")
    df = ds[["ta", "rh"]].to_dataframe().reset_index()
    df["td"] = compute_dewpoint_temp(df["ta"], df["rh"])
    df = (
        df[["time", "ta", "rh", "td"]]
        .sort_values("time")
        .reset_index(drop=True)
    )
    return df


def read_irta_corn(
    filepath: str | Path, name_map: dict = NAME_MAP
) -> pd.DataFrame:
    """
    Description
    -----------
    Read an IRTA-Corn station CSV file and return a DataFrame.

    Parameters
    ----------
    filepath : str | Path
        Path to the IRTA-Corn CSV file
    name_map : dict,
        Mapping column names

    Returns
    -------
    pd.DataFrame
    """
    filepath = Path(filepath)

    # 4 lines header:
    #   0 : stations info
    #   1 : columns name
    #   2 : unit
    #   3 : methods of calcul
    df = pd.read_csv(
        filepath,
        skiprows=[0, 2, 3],
        na_values=["NAN", "nan", "NaN", ""],
        low_memory=False,
    )
    # Keep only certain columns
    cols_available = [c for c in name_map if c in df.columns]
    df = df[cols_available].rename(columns=name_map)

    if "timestamp" in df.columns:
        df["timestamp"] = fix_hour_format(df["timestamp"])
        # logger.info(f"df = {df}")

    # Compute dewpoint temperature
    df["td"] = compute_dewpoint_temp(df["ta"], df["rh"])

    return df


def station_irta_corn_to_csv(
    filepath: str | Path, output_path: str | Path, name_map: dict = NAME_MAP
) -> None:
    """
    Description
    -----------
    Read an IRTA-Corn station file and export the data to CSV.

    Parameters
    ----------
    filepath : str | Path
        Path to the input IRTA-Corn CSV file
    output_path : str | Path
        Path where the processed CSV file will be saved
    name_map : dict,
        Mapping column names

    Returns
    -------
    """
    df = read_irta_corn(filepath, name_map)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Irta-Corn {len(df)} : {output_path}")


def save_liaise_station(
    df: pd.DataFrame, name_station: str, output_dir: str = "liaise_processed"
) -> str:
    """
    Description
    -----------
    Save a processed LIAISE station DataFrame to a CSV file.

    Parameters
    ----------
    df : pd.DataFrame
        Processed station data to save
    name_station : str
        Name of the station
    output_dir : str
        Directory where the CSV file will be saved

    Returns
    -------
    str
        Full path to the saved CSV file.
    """
    if os.path.isabs(output_dir):
        folder = output_dir
    else:
        folder = os.path.join(os.getcwd(), output_dir)

    os.makedirs(folder, exist_ok=True)
    csv_path = os.path.join(folder, f"{name_station}_processed.csv")
    df.to_csv(csv_path, index=False)
    return csv_path


def _parse_els_plans_header(filepath: Path) -> list[str]:
    """
    Description
    -----------
    Extract column names


    The structure is:
      - first '!' level: description metadata
      - second '!' line: levels (e.g. 50m, 50mB, 2mB, time, …)
      - last '!' line: variable names (e.g. HOUR, UTOT, …)

     Parameters
    ----------
    filepath : Path
        Path to the file

    Returns
    -------
    list[str]
        Returns a list of names as "level_VARIABLE"
        (e.g. ["HOUR", "50m_UTOT", "50m_USCL", …]).
    """
    comment_lines = []
    with open(filepath) as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("!"):
                comment_lines.append(stripped)
            else:
                break

    level_line = comment_lines[-2]
    var_line = comment_lines[-1]

    def extract_tokens(line: str) -> list[str]:
        # Delete the '!'
        content = line.lstrip("!").strip()
        tokens = re.split(r"\s{2,}", content)
        return [t.strip() for t in tokens if t.strip()]

    levels = extract_tokens(level_line)
    variables = extract_tokens(var_line)

    # Build the columns names : "time_HOUR"-> keep "HOUR"
    col_names = []
    for level, var in zip(levels, variables, strict=False):
        if level.lower() in ("time", ""):
            col_names.append(var)  # e.g. "HOUR"
        else:
            col_names.append(f"{level}_{var}")

    return col_names


def _parse_els_plans_line(line: str, col_names: list[str]):
    """
    Description
    -----------
    Parse a single data line from an ELS-PLANS .dat file into values

    Parameters
    ----------
    line : str
        Raw data line to parse
    col_names : list[str]
        Column names

    Returns
    -------
    list[float]
    """

    parts = line.strip().split("|")

    values = []

    for part in parts:
        tokens = part.split()

        for t in tokens:
            # Skip flag
            if t in {"m", "D", "X"}:
                continue

            try:
                val = float(t)

                if val > MISSING_THRESHOLD:
                    val = np.nan

                values.append(val)

            except ValueError:
                continue

    n = len(col_names)
    values = (values + [np.nan] * n)[:n]

    return values


def read_station_els_plans_file(filepath: Path) -> pd.DataFrame:
    """
    Description
    -----------
    Read a single ELS-PLANS station .dat file.

    Parameters
    ----------
    filepath : Path
        Path to the .dat file.

    Returns
    -------
    pd.DataFrame
        DataFrame with a "timestamp" column (file date + HOUR) and

    """
    col_names = _parse_els_plans_header(filepath)
    # logger.info(col_names[:200])
    # logger.info(len(col_names))

    # Date in the filename
    date_match = re.search(r"(\d{8})", filepath.stem)
    if date_match:
        file_date = pd.to_datetime(date_match.group(1), format="%Y%m%d")
    else:
        file_date = None
        logger.info(f"No date found is the name file: {filepath.name}")

    rows_values = []

    with open(filepath) as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("!") or not stripped:
                continue
            vals = _parse_els_plans_line(stripped, col_names)
            rows_values.append(vals)

    df_vals = pd.DataFrame(rows_values, columns=col_names)

    df = pd.concat([df_vals], axis=1)

    # timestamp = file date + HOUR
    if "HOUR" in df.columns and file_date is not None:
        df["timestamp"] = file_date + pd.to_timedelta(df["HOUR"], unit="h")

    df["file_date"] = file_date
    return df


def read_station_els_plans(
    folder: str | Path, name_map: dict = NAME_MAP, pattern: str = "*.dat"
) -> pd.DataFrame:
    """
    Description
    -----------
    Read all .dat files in a folder, concatenate them,
    and apply the NAME_MAP

    name_map selects and renames the data columns

    Parameters
    ----------
    folder : str | Path
        Path to the folder containing the .dat files
    name_map : dict, default=NAME_MAP
        Mapping from column names to standardized column names.
    pattern : str, default="*.dat"

    Returns
    -------
    pd.DataFrame
    """
    folder = Path(folder)
    files = sorted(folder.glob(pattern))

    if not files:
        raise FileNotFoundError(f"No '{pattern}' file found in {folder}")

    dfs = []
    for f in files:
        logger.info(f"Reading: {f.name}")
        dfs.append(read_station_els_plans_file(f))

    df = pd.concat(dfs, ignore_index=True)

    logger.info(df.columns.tolist())

    logger.info("NAME_MAP keys:")
    logger.info(list(name_map.keys()))

    logger.info("Matched columns:")
    logger.info([c for c in name_map if c in df.columns])

    cols_data = [c for c in name_map if c in df.columns]
    cols_flags = [f"{c}_flag" for c in cols_data if f"{c}_flag" in df.columns]

    keep = []
    if "timestamp" in df.columns:
        keep.append("timestamp")
    keep += cols_data + cols_flags

    df = df[keep].copy()

    df = df.rename(columns=name_map)

    flag_rename = {
        f"{old}_flag": f"{new}_flag"
        for old, new in name_map.items()
        if f"{old}_flag" in df.columns
    }
    df = df.rename(columns=flag_rename)

    if "timestamp" in df.columns:
        df = df.sort_values("timestamp").reset_index(drop=True)
    # Compute dewpoint temp
    df["td"] = compute_dewpoint_temp(df["ta"], df["rh"])

    return df


def station_els_plans_to_csv(
    folder: str | Path,
    output_path: str | Path,
    name_map: dict = NAME_MAP,
    pattern: str = "*.dat",
) -> None:
    """
    Description
    -----------
    Read all ELS-PLANS station files in a folder and export the data
    to a CSV file

    Parameters
    ----------
    folder : str | Path
        Path to the folder containing the .dat files
    output_path : str | Path
        Path where the processed CSV file will be saved
    name_map : dict, default=NAME_MAP
    pattern : str, default="*.dat"

    Returns
    -------
    None
    """
    logger.info(f"ELS PLANS : {folder}")
    df = read_station_els_plans(folder, name_map, pattern)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"ELS-PLANS {len(df)} : {output_path}")

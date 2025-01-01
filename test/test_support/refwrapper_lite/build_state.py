from datetime import datetime
import numpy as np
import pandas as pd
from pathlib import Path

from py_retrieve.app.tropomi_setup import read_tropomi_surface_altitude

from . import utils


def build_state_from_sounding(
    sounding_dir,
    atm_state_file,
    tropomi_state_file,
    src_rad_file=None,
    src_irr_file=None,
):
    file_id = read_measid_file(Path(sounding_dir) / "Measurement_ID.asc")
    initial_state = _make_atmospheric_state(file_id, sounding_dir, atm_state_file)

    atm_species = {
        "TATM": "temperature",
        "CO": "co",
        "CH4": "ch4",
        "H2O": "h2o",
        "HDO": "hdo",
    }
    atm = {"pressure": 100 * initial_state["pressure"]}  # convert hPa -> Pa
    for muses_key, state_key in atm_species.items():
        atm[state_key] = initial_state[muses_key]

    # change order to space-to-surface
    for k, v in atm.items():
        atm[k] = np.flip(v)

    # now ancillary state
    tropomi_state = read_measid_file(tropomi_state_file)

    ancillary = {
        "time": datetime.strptime(file_id["TROPOMI_utcTime"], "%Y-%m-%dT%H:%M:%S.%fZ"),
        "lon": float(file_id["TROPOMI_Longitude_BAND7"]),
        "lat": float(file_id["TROPOMI_Latitude_BAND7"]),
        "albedo": float(tropomi_state["surface_albedo_BAND7"]),
        "albedo_slope": 0.0,
        "albedo_curvature": 0.0,
        "surf_gph": float(initial_state["surface_altitude"]),
        "cloud_frac": float(tropomi_state["cloud_fraction"]),
        "cloud_pres": float(tropomi_state["cloud_pressure"]),
        "cloud_albedo": float(tropomi_state["cloud_Surface_Albedo"]),
        "sza": float(tropomi_state["sza_BAND7"]),
        "vza": float(tropomi_state["vza_BAND7"]),
        "raa": float(tropomi_state["raz_BAND7"]),
    }

    if ancillary["cloud_frac"] == 0.0:
        # This is a kludge done in script_retrieval_setup_ms
        ancillary["cloud_frac"] = 0.01

    return {"atm": atm, "ancillary": ancillary, "file_id": file_id}


def _make_atmospheric_state(
    file_id,
    sounding_dir,
    atm_state_file,
    latlon_key_pattern="TROPOMI_{xy}",
    alt_latlon_key_pattern="TROPOMI_{xy}_BAND7",
):
    # The MUSES code is a little inconsistent - it always seems to use Band 3 lat/lon for the initial state,
    # but uses the actual Band 7 lat/lon for surface altitude... Realistically, that won't matter very often:
    # the initial state is pretty coarse in space.
    lat_for_alt = float(file_id[alt_latlon_key_pattern.format(xy="Latitude")])
    lon_for_alt = float(file_id[alt_latlon_key_pattern.format(xy="Longitude")])
    with utils.in_dir(sounding_dir):
        surface_altitude = read_tropomi_surface_altitude(lat_for_alt, lon_for_alt)
    state = _load_atm_state_file(atm_state_file)

    # This definitely isn't included if we read from a diagnostic file, and may
    # not be if get the state from the OSPs (haven't checked)
    state["surface_altitude"] = surface_altitude
    return state


def _load_atm_state_file(atm_state_file):
    with open(atm_state_file) as f:
        for line in f:
            if "End_of_Header" in line:
                break

        colnames = f.readline().strip().split()
        f.readline()  # skip the units
        df = pd.read_csv(f, header=None, delim_whitespace=True)
        df.columns = colnames

    # Convert into a dictionary of arrays to avoid some potential weirdness with
    # trying to align Pandas series.
    # Also need to rename "Pressure" to match what the caller expects.
    state = {k: v.to_numpy() for k, v in df.items()}
    state["pressure"] = state.pop("Pressure")
    return state


def read_measid_file(meas_id_file: Path):
    meas_info = dict()
    with open(meas_id_file) as f:
        for line in f:
            if "End_of_Header" in line:
                break
            else:
                key, value = line.split("=", maxsplit=1)
                meas_info[key.strip()] = value.strip()

    return meas_info

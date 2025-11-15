from __future__ import annotations
from loguru import logger
from .mpy import (
    have_muses_py,
    mpy_cdf_var_attributes,
    mpy_cdf_var_names,
    mpy_cdf_var_map,
)
from .identifier import (
    RetrievalType,
    ProcessLocation,
    StateElementIdentifier,
    InstrumentIdentifier,
)
from .refractor_uip import AttrDictAdapter
from .retrieval_lite_output import CdfWriteLiteTes
from netCDF4 import Dataset
from pathlib import Path
import importlib
import json
import os
import copy
import numpy as np
import datetime
import pytz
import re
import typing
from typing import Any

if typing.TYPE_CHECKING:
    from .retrieval_strategy import RetrievalStrategy
    from .retrieval_strategy_step import RetrievalStrategyStep
    from .retrieval_configuration import RetrievalConfiguration
    from .retrieval_result import RetrievalResult
    from .current_state import CurrentState
    from .sounding_metadata import SoundingMetadata
    from .muses_strategy import CurrentStrategyStep
    from .muses_observation import MusesObservation


class ExtraL2Output:
    """All of the muses-py elements are hard wired in the L2 output. We possibly want
    to have new elements added.

    This is a sort of half baked placeholder for this, I think once we have some actual
    examples we will want to rethink this. But for now, the idea is that a central place
    will be used to store this information. The idea is that StateElements will populate this
    somehow, possibly as they are created or find initial values.

    This contains the information write_list needs for writing a variable.
    """

    def __init__(self) -> None:
        pass

    def should_add_to_l2(
        self, sid: StateElementIdentifier, instruments: list[InstrumentIdentifier]
    ) -> bool:
        """True if this should get added to the L2 file."""
        # Right now, have everything false here.
        return False

    def net_cdf_variable_name(self, sid: StateElementIdentifier) -> str:
        """Variable name in netcdf file."""
        return str(sid)

    def net_cdf_struct_units(
        self, sid: StateElementIdentifier
    ) -> dict[str, str | float]:
        """Returns the attributes attached to a netCDF write out of this
        StateElement."""
        return {
            "Longname": str(sid),
            "Units": "",
            "FillValue": "",
            "MisingValue": "",
        }

    def net_cdf_group_name(self, sid: StateElementIdentifier) -> str:
        """Group that variable goes into in a netCDF file. Use the empty string
        if this doesn't go into a group, but rather is a top level variable."""
        return ""


extra_l2_output = ExtraL2Output()


class RetrievalOutput:
    """Observer of RetrievalStrategy, common behavior for Products files."""

    def notify_add(self, retrieval_strategy: RetrievalStrategy) -> None:
        self.retrieval_strategy = retrieval_strategy

    def notify_update(
        self,
        retrieval_strategy: RetrievalStrategy,
        location: ProcessLocation,
        retrieval_strategy_step: RetrievalStrategyStep | None = None,
        **kwargs: Any,
    ) -> None:
        logger.debug(f"Call to {self.__class__.__name__}::notify_update")
        self.retrieval_strategy_step = retrieval_strategy_step

    @property
    def retrieval_config(self) -> RetrievalConfiguration:
        return self.retrieval_strategy.retrieval_config

    @property
    def step_directory(self) -> Path:
        return self.output_directory / f"Step{self.step_number:02d}_{self.step_name}"

    @property
    def input_directory(self) -> Path:
        return self.step_directory / "ELANORInput"

    @property
    def analysis_directory(self) -> Path:
        return self.step_directory / "StepAnalysis"

    @property
    def elanor_directory(self) -> Path:
        return self.step_directory / "Diagnostics"

    @property
    def output_directory(self) -> Path:
        # This is usually the same as the self.retrieval_strategy.run_dir, but
        # could in principle be different. So we calculate this the same way
        # muses-py does
        return (
            Path(self.retrieval_config["outputDirectory"])
            / self.retrieval_config["sessionID"]
        )

    @property
    def lite_directory(self) -> Path:
        return Path(self.retrieval_config["liteDirectory"])

    @property
    def special_tag(self) -> str:
        if self.retrieval_strategy.retrieval_type != RetrievalType("default"):
            return f"-{self.retrieval_strategy.retrieval_type.lower()}"
        return ""

    @property
    def species_tag(self) -> str:
        res = self.step_name
        res = res.rstrip(", ")
        if "EMIS" in res and res.index("EMIS") > 0:
            res = res.replace("EMIS", "")
        if res.endswith(",_OMI"):
            res = res.replace(",_OMI", "_OMI")  #  Change "H2O,O3,_OMI" to "H2O,O3_OMI"
        res = res.rstrip(", ")
        return res

    @property
    def step_number(self) -> int:
        return self.retrieval_strategy.strategy_step.step_number

    @property
    def step_name(self) -> str:
        return self.retrieval_strategy.strategy_step.step_name

    @property
    def results(self) -> RetrievalResult:
        if (
            self.retrieval_strategy_step is None
            or self.retrieval_strategy_step.results is None
        ):
            raise RuntimeError("retrieval_strategy_step.results needs to not be None")
        return self.retrieval_strategy_step.results

    def state_value(self, state_name: str) -> float:
        """Get the state value for the given state name"""
        return self.current_state.state_value(StateElementIdentifier(state_name))[0]

    def state_value_vec(self, state_name: str) -> np.ndarray:
        """Get the state value for the given state name"""
        return self.current_state.state_value(StateElementIdentifier(state_name))

    def state_constraint_vec(self, state_name: str) -> np.ndarray:
        """Get the state value for the given state name"""
        # TODO Should this actually be fmprime?
        return self.current_state.state_constraint_vector_fmprime(
            StateElementIdentifier(state_name)
        )

    def state_constraint_vec_fm(self, state_name: str) -> np.ndarray:
        """Get the state value for the given state name"""
        # TODO Not really clear why some things are fmprime, and some fm.
        # But duplicate what muses-py currently does here
        return self.current_state.state_constraint_vector(
            StateElementIdentifier(state_name)
        )

    def state_constraint(self, state_name: str) -> float:
        """Get the state value for the given state name"""
        return self.current_state.state_constraint_vector(
            StateElementIdentifier(state_name)
        )[0]

    def state_step_initial_value_vec(self, state_name: str) -> np.ndarray:
        """Get the state value for the given state name"""
        return self.current_state.state_step_initial_value(
            StateElementIdentifier(state_name)
        )

    def state_retrieval_initial_value_vec(self, state_name: str) -> np.ndarray:
        """Get the state value for the given state name"""
        return self.current_state.state_retrieval_initial_value(
            StateElementIdentifier(state_name)
        )

    def state_step_initial_value(self, state_name: str) -> float:
        """Get the state value for the given state name"""
        return self.current_state.state_step_initial_value(
            StateElementIdentifier(state_name)
        )[0]

    def state_sd_wavelength(self, state_name: str) -> np.ndarray:
        """Get the spectral domain wavelength in nm for state element"""
        t = self.current_state.state_spectral_domain_wavelength(
            StateElementIdentifier(state_name)
        )
        if t is None:
            raise RuntimeError(
                f"{state_name} doesn't have state_spectral_domain_wavelength"
            )
        return t

    @property
    def current_state(self) -> CurrentState:
        return self.retrieval_strategy.current_state

    @property
    def current_strategy_step(self) -> CurrentStrategyStep:
        return self.retrieval_strategy.current_strategy_step

    def observation(self, instrument_name: str) -> MusesObservation:
        return self.retrieval_strategy.observation_handle_set.observation(
            InstrumentIdentifier(instrument_name), None, None, None
        )

    @property
    def radiance_full(self) -> dict:
        return self.results.radiance_full

    @property
    def obs_list(self) -> list[MusesObservation]:
        return self.results.obs_list

    @property
    def radiance_step(self) -> AttrDictAdapter:
        return self.results.rstep

    @property
    def instruments(self) -> list[InstrumentIdentifier]:
        return self.results.instruments

    @property
    def species_list_fm(self) -> list[str]:
        return self.results.species_list_fm

    @property
    def pressure_list_fm(self) -> np.ndarray:
        return self.results.pressure_list_fm

    @property
    def sounding_metadata(self) -> SoundingMetadata:
        return self.results.sounding_metadata

    def cdf_write(
        self,
        struct_in: dict[str, Any],
        filename: str | os.PathLike[str],
        struct_units: list[dict[str, str | float]],
    ) -> None:
        t = CdfWriteTes()
        t.cdf_write(struct_in, filename, struct_units)


class CdfWriteTes:
    """We want to be able to extend the variables written out, so we have our own
    lightly modified version of cdf_write_tes. This basically just makes some of
    the fixed sized arrays things that can be modified.

    Since we want to do this by run we wrap this in a class, which is little more than
    cdf_write_tes function plus some arrays that can be extended.
    """

    def __init__(self) -> None:
        # muses-py has a lot of hard coded things related to the
        # species names and netcdf output.  It would be good a some
        # point to just replace this all with a better thought out
        # output format. But for now, we need to support the existing
        # output format.  TODO - Replace with better thought out
        # output format

        # So we don't depend on muses_py, we save the variable to a json file.
        # Only need muses_py to generate this or update it. We just create this file
        # if not available, so you can manually delete this to force it to be recreated.
        if not importlib.resources.is_resource(
            "refractor.muses", "retrieval_output.json"
        ):
            if not have_muses_py:
                raise RuntimeError(
                    "Require muses-py to create the file retrieval_output.json"
                )
            d = {
                "cdf_var_attributes": mpy_cdf_var_attributes,
                "groupvarnames": mpy_cdf_var_names(),
                "exact_cased_variable_names": mpy_cdf_var_map(),
            }
            with importlib.resources.path(
                "refractor.muses", "retrieval_output.json"
            ) as fspath:
                logger.info(f"Creating the file {fspath}")
                with open(fspath, "w") as fh:
                    json.dump(d, fh, indent=4)
        d = json.loads(
            importlib.resources.read_text("refractor.muses", "retrieval_output.json")
        )
        self.cdf_var_attributes: dict[str, dict[str, str | float]] = d[
            "cdf_var_attributes"
        ]
        self.groupvarnames: list[list[str]] = d["groupvarnames"]
        self.exact_cased_variable_names: dict[str, str] = d[
            "exact_cased_variable_names"
        ]

    def write(
        self,
        dataOut: dict,
        filenameOut: str,
        current_state: CurrentState,
        tracer_species: str = "species",
        retrieval_pressures: list[float] | None = None,
        write_met: bool = False,
        version: str | None = None,
        liteVersion: str | None = None,
        runtimeAttributes: dict | None = None,
        state_element_out: list[StateElementIdentifier] | None = None,
    ) -> None:
        """We pass in state_element_out for StateElementIdentifier not otherwise handled. This
        separates out the species that were in muses-py vs stuff we may have added.
        Perhaps we'll get all the StateElements handled the same way at some point,
        muses-py is really overly complicated. On the other hand, this is just output
        code which tends to get convoluted to create specific output, so perhaps
        this isn't so much an issue."""
        if runtimeAttributes is None:
            runtimeAttributes = {}
        if state_element_out is None:
            state_element_out = []
        dims = {}

        if "FILENAME" in dataOut:
            del dataOut["FILENAME"]

        if "NCEP_TEMPERATURESURFACE" in dataOut:
            del dataOut["NCEP_TEMPERATURESURFACE"]

        if "NCEP_TEMPERATURE" in dataOut:
            del dataOut["NCEP_TEMPERATURE"]

        # correct naming mistakes
        # Loop through all keys and look for '_TROPOSPHERI'
        for key_name, v in dataOut.items():
            if "_TROPOSPHERI" in key_name:
                new_key = key_name.split("_")[0]
                del dataOut[key_name]
                dataOut[new_key] = v
            elif (
                "OZONETROPOSPHERI" in key_name
                or "OZONEUPPER" in key_name
                or "OZONELOWER" in key_name
            ):
                new_key = key_name[5:]  # Extract from 'UPPER" on from 'OZONE'
                del dataOut[key_name]
                dataOut[new_key] = v
            elif "OZONEDOFS" in key_name:
                new_key = key_name[5:]  # Extract from 'DOFS' of 'OZONEDOFS'
                del dataOut[key_name]
                dataOut[new_key] = v

        tracer_species = tracer_species.lstrip().rstrip()

        if "STEPNAME" in dataOut:
            del dataOut["STEPNAME"]

        # TODO Probably bad that this is hardcoded. Should perhaps pass this
        # in instead
        grid_pressure_FM: list[float] | np.ndarray = [
            -999,
            1.21153e03,
            1.10070e03,
            1.0e03,
            908.514,
            825.402,
            749.893,
            681.291,
            618.966,
            562.342,
            510.898,
            464.160,
            421.698,
            383.117,
            348.069,
            316.227,
            287.298,
            261.016,
            237.137,
            215.444,
            195.735,
            177.829,
            161.561,
            146.779,
            133.352,
            121.152,
            110.069,
            100.000,
            90.8518000,
            82.5406000,
            74.9896000,
            68.1295000,
            61.8963000,
            56.2339000,
            51.0896000,
            46.4158000,
            42.1696000,
            38.3119000,
            34.8071000,
            31.6229000,
            28.7299000,
            26.1017000,
            23.7136000,
            21.5443000,
            19.5734000,
            17.7828000,
            16.156,
            14.678,
            13.3352000,
            12.1153000,
            11.007,
            10.000,
            9.0851400,
            8.25402,
            6.8129100,
            5.10898,
            4.64160,
            3.16227,
            2.6101600,
            2.15443,
            1.6156000,
            1.33352,
            1.0,
            0.681292,
            0.3831180,
            0.215443,
            0.1,
        ]
        if len(dataOut["PRESSURE"]) == 20:
            grid_pressure_FM = np.array(
                [
                    1.00000,
                    0.947364,
                    0.894734,
                    0.842105,
                    0.789472,
                    0.736844,
                    0.684210,
                    0.631580,
                    0.578948,
                    0.526316,
                    0.473684,
                    0.421053,
                    0.368420,
                    0.315789,
                    0.263158,
                    0.210526,
                    0.157895,
                    0.105263,
                    0.0526316,
                    0.000100000,
                ]
            )

        # AT_LINE 161 TOOLS/cdf_write_tes.pro
        if len(dataOut["PRESSURE"]) != len(grid_pressure_FM):
            if retrieval_pressures is None:
                raise RuntimeError(
                    "You must include retrieval_pressures, which is the grid_pressure"
                )
            grid_pressure = retrieval_pressures
            dims["dim_pressure"] = len(grid_pressure)
            dims["dim_pressure_fm"] = len(grid_pressure_FM)
        else:
            dims["dim_pressure"] = len(grid_pressure_FM)

        grid_rtvmr_levels = []
        grid_rtvmr_map = []
        if "RTVMR" in dataOut:
            # my_lists.extend(['GRID_RTVMR_MAP', 'GRID_RTVMR_LEVELS'])
            grid_rtvmr_levels = [0, 1]
            dims["dim_rtvmr_levels"] = len(grid_rtvmr_levels)

            grid_rtvmr_map = [0, 1, 2, 3, 4]
            dims["dim_rtvmr_map"] = len(grid_rtvmr_map)

        if "NIR_AEROD" in dataOut:
            dims["dim_nir_aerosol"] = len(dataOut["NIR_AEROD"])

        # AT_LINE 145 TOOLS/cdf_write_tes.pro
        grid_iters: list[int] | np.ndarray = []
        grid_iterlist: list[int] | np.ndarray = []
        if "LMRESULTS_RESNORM" in dataOut:
            grid_iters = np.arange(0, len(dataOut["lmresults_resnorm".upper()]))
            grid_iterlist = np.arange(
                0, len(dataOut["lmresults_iterlist".upper()][0, :])
            )

        if len(grid_iters) > 0:
            dims["dim_iters"] = len(grid_iters)
            dims["dim_iterlist"] = len(grid_iterlist)

        # AT_LINE 157 TOOLS/cdf_write_tes.pro

        # AT_LINE 170 TOOLS/cdf_write_tes.pro
        if "CT_CO2" in dataOut:
            raise RuntimeError("Not implemented yet")

        # AT_LINE 181 TOOLS/cdf_write_tes.pro
        if "NCEP_TEMPERATURE" in dataOut:
            grid_ncep = [1000, 900, 800, 700, 600, 500, 400, 300, 200, 100, 10]
            dims["dim_ncep"] = len(grid_ncep)
        # end if 'NCEP_TEMPERATURE' in dataOut:

        # AT_LINE 186 TOOLS/cdf_write_tes.pro
        if "EMISSIVITY" in dataOut:
            GRID_EMISSIVITY = np.asarray(
                [
                    600,
                    650,
                    675,
                    700,
                    725,
                    750,
                    770,
                    780,
                    790,
                    800,
                    810,
                    830,
                    850,
                    900,
                    910,
                    920,
                    930,
                    940,
                    950,
                    960,
                    970,
                    980,
                    990,
                    1000,
                    1020,
                    1030,
                    1040,
                    1050,
                    1060,
                    1070,
                    1080,
                    1090,
                    1100,
                    1105,
                    1110,
                    1115,
                    1120,
                    1125,
                    1130,
                    1135,
                    1140,
                    1145,
                    1150,
                    1155,
                    1160,
                    1165,
                    1170,
                    1175,
                    1180,
                    1185,
                    1190,
                    1195,
                    1200,
                    1205,
                    1210,
                    1215,
                    1220,
                    1225,
                    1230,
                    1235,
                    1240,
                    1245,
                    1250,
                    1255,
                    1260,
                    1265,
                    1270,
                    1275,
                    1280,
                    1285,
                    1290,
                    1295,
                    1300,
                    1350,
                    1400,
                    1450,
                    1500,
                    1550,
                    1600,
                    1650,
                    1700,
                    1750,
                    1800,
                    1830,
                    1850,
                    1870,
                    1900,
                    1930,
                    1950,
                    1960,
                    1980,
                    2000,
                    2030,
                    2050,
                    2060,
                    2100,
                    2120,
                    2140,
                    2160,
                    2180,
                    2200,
                    2220,
                    2240,
                    2260,
                    2280,
                    2300,
                    2320,
                    2340,
                    2360,
                    2380,
                    2400,
                    2460,
                    2500,
                    2540,
                    2560,
                    2600,
                    2700,
                    2800,
                    2900,
                    3000,
                    3100,
                ]
            )
            dims["dim_emissivity"] = len(GRID_EMISSIVITY)
        # end if 'EMISSIVITY' in dataOut:

        # AT_LINE 191 TOOLS/cdf_write_tes.pro
        if len(dataOut["CLOUDEFFECTIVEOPTICALDEPTH"]) == 25:
            GRID_CLOUD = np.asarray(
                [
                    600,
                    650,
                    700,
                    750,
                    800,
                    850,
                    900,
                    950,
                    975,
                    1000,
                    1025,
                    1050,
                    1075,
                    1100,
                    1150,
                    1200,
                    1250,
                    1300,
                    1350,
                    1400,
                    1900,
                    2000,
                    2100,
                    2200,
                    2250,
                ]
            )
        else:
            GRID_CLOUD = np.asarray(
                [
                    600,
                    650,
                    700,
                    750,
                    800,
                    850,
                    900,
                    950,
                    975,
                    1000,
                    1025,
                    1050,
                    1075,
                    1100,
                    1140,
                    1200,
                    1250,
                    1300,
                    1350,
                    1400,
                    1900,
                    2000,
                    2040,
                    2060,
                    2080,
                    2100,
                    2200,
                    2250,
                ]
            )
        # end if len(dataOut['CLOUDEFFECTIVEOPTICALDEPTH']) == 25:
        dims["dim_cloud"] = len(GRID_CLOUD)

        #### AT_LINE 199 TOOLS/cdf_write_tes.pro
        if "MAP" in list(dataOut.keys()):
            if retrieval_pressures is None:
                raise RuntimeError("retrieval_pressures is None")
            if len(dataOut["pressure".upper()]) != len(dataOut["map".upper()][:, 0]):
                # for composite files (e.g. HDO-H2O) also stack of pressures
                grid_pressure_composite: np.ndarray = np.concatenate(
                    (retrieval_pressures, retrieval_pressures), axis=0
                )
                dims["dim_pressure_composite"] = len(grid_pressure_composite)

        # AT_LINE 208 TOOLS/cdf_write_tes.pro
        # add and define RTVMR field names
        if "RTVMR" in dataOut:
            grid_rtvmr_levels = [0, 1]
            grid_rtvmr_map = [0, 1, 2, 3, 4]
        # end if 'RTVMR' in dataOut:

        if "LMRESULTS_RESNORM" in dataOut:
            # INDGEN(N_ELEMENTS(data[0].lmresults_resnorm))
            grid_iters = np.arange(0, len(dataOut["lmresults_resnorm".upper()]))
            # INDGEN(N_ELEMENTS(data[0].lmresults_iterlist[0,*]))
            grid_iterlist = np.arange(
                0, len(dataOut["lmresults_iterlist".upper()][0, :])
            )
        # end if 'LMRESULTS_RESNORM' in dataOut:

        # my_lists.extend(['GRID_COLUMN', 'GRID_FILTER'])
        GRID_COLUMN = [0, 1, 2, 3, 4]
        dims["dim_column"] = len(GRID_COLUMN)

        GRID_FILTER = np.arange(len(dataOut["FILTER_INDEX"]))
        dims["dim_filter"] = len(GRID_FILTER)

        dataOut["GRID_PRESSURE_FM"] = grid_pressure_FM

        # form new time called YYYYMMDD.fractionofday
        # TAI seconds from 1993
        ymd = datetime.datetime(1993, 1, 1) + datetime.timedelta(
            seconds=int(dataOut["TIME"])
        )
        # Account for leapseconds. Note that this is *wrong*, it corresponds to the
        # wrong_tai in SoundingMetadata. This only accounts for leapseconds up to 2008.
        # But do the same wrong calculation here so we match the expected output.
        if ymd.year <= 2005:
            ymd -= datetime.timedelta(seconds=5)
        if ymd.year >= 2006:
            ymd -= datetime.timedelta(seconds=6)
        yyear = ymd.year
        ymonth = ymd.month
        yday = ymd.day
        yyyymmdd = np.float64(yyear * 10000.0 + ymonth * 100.0 + yday * 1.0)

        dataOut["YYYYMMDD"] = yyyymmdd

        ut_hour = round(ymd.hour + (ymd.minute + ymd.second / 60.0) / 60.0, 4)

        dataOut["UT_HOUR"] = ut_hour

        if "OMI_NRADWAV" in dataOut:
            dims["size2"] = 2
            dims["size3"] = 3

        dataOut = self.cdf_var_add_strings(dataOut, dims)

        # ===============================
        # netcdf file global variable
        # ===============================

        # It is convenient in testing to have a fixed creation_date, just so we can
        # compare files created at different times with h5diff and have them compare as
        # identical
        if "MUSES_FAKE_CREATION_DATE" in os.environ:
            history_entry = "created " + os.environ["MUSES_FAKE_CREATION_DATE"]
        else:
            history_entry = "created " + datetime.datetime.now(tz=pytz.utc).strftime(
                "%Y%m%dT%H%M%SZ"
            )

        # Create .met file and retrieve the output
        if write_met:
            raise RuntimeError(
                "Function Met_Write() has not been implemented.  Must exit for now."
            )
        else:
            global_attr: dict[str, Any] = {}

        global_attr["fileversion"] = 2
        global_attr["history"] = history_entry

        # PYTHON_NOTE: Below are an exhaustive definition of attributes of every possible variable.  Because this is Python,
        # we make sure the part before the _attr is uppercase because later on, we will use it to access the attributes defined
        # here.
        SPECIES_attr: dict[str, str | float] = {
            "Longname": tracer_species + " volume mixing ratio",
            "Units": "volume mixing ratio relative to dry air",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        }

        self.cdf_var_attributes["SPECIES_attr"] = SPECIES_attr

        SPECIES_FM_attr: dict[str, str | float] = {
            "Longname": tracer_species + " volume mixing ratio",
            "Units": "volume mixing ratio relative to dry air on fm grid",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        }

        self.cdf_var_attributes["SPECIES_FM_attr"] = SPECIES_FM_attr

        # ===============================
        # Link attributes to their respective variables
        # ===============================
        # use ordering in the array names
        # if that variable exists, then add its attribute to the list

        # AT_LINE 758 TOOLS/cdf_write_tes.pro
        dict_of_variables_and_their_attributes = {}

        names = [t[1] for t in self.groupvarnames]
        # dict preserves order
        names = list(dict.fromkeys(names))
        names = [x.upper() for x in names]

        for ii in range(0, len(names)):
            tag_name = names[ii]
            if tag_name in dataOut:
                attr_name = names[ii] + "_attr"
                if attr_name in self.cdf_var_attributes:
                    dict_of_variables_and_their_attributes[tag_name] = (
                        self.cdf_var_attributes[attr_name]
                    )
                else:
                    logger.info("Using generic attribute spec for " + attr_name + ".")
                    # Create a generic_attr depend on name of the variable.
                    # A string variable has 'GRID_STRING' or 'Grid_String' in it.
                    if "STRING" in attr_name or "String" in attr_name:
                        generic_attr: dict[str, str | float] = {
                            "Longname": "string",
                            "Units": "",
                            "FillValue": -999,
                            "MissingValue": -999,
                        }
                    else:
                        generic_attr = {
                            "Longname": "",
                            "Units": "",
                            "FillValue": -999,
                            "MissingValue": -999,
                        }

                    dict_of_variables_and_their_attributes[tag_name] = generic_attr

                # If there are extra attributes whose values are defined at runtime, add them in now
                dict_of_variables_and_their_attributes[tag_name].update(
                    runtimeAttributes.get(tag_name, dict())
                )
        # for ii in range(0,len(names)):

        # ===============================
        # Link data to their respective variables
        # ===============================
        # use tagnames to put data into newdata
        dataNew_keys = list(dataOut.keys())
        newdata = {}
        for ii in range(0, len(names)):
            tag_name = names[ii]
            if tag_name in dataOut:
                if tag_name not in dataNew_keys:
                    raise RuntimeError("tag_name not in dataNew_keys")
                newdata[tag_name] = dataOut[tag_name]

        structIn = newdata

        # To ensure we have matching values in both list: structIn and dict_of_variables_and_their_attributes, we extract
        # from dict_of_variables_and_their_attributes based on key.
        structUnits = []
        generic_attr = {
            "Longname": "",
            "Units": "",
            "Fillvalue": -999.0,
            "Missingvalue": -999.0,
        }
        structKeys = list(structIn.keys())
        for ii in range(len(structKeys)):
            tag_name = structKeys[ii]
            if tag_name in dict_of_variables_and_their_attributes:
                attributes = dict_of_variables_and_their_attributes[tag_name]
                structUnits.append(attributes)
            else:
                structUnits.append(copy.deepcopy(generic_attr))

        for ii in range(len(structKeys)):
            tag_name = structKeys[ii]
            variable_data = structIn[tag_name]
            if isinstance(variable_data, list):
                structIn[tag_name] = np.asarray(variable_data)

        # StateElements we haven't already gotten. These are
        # StateElement that weren't originally in muses-py.

        for sid in state_element_out:
            v = current_state.state_value(sid)
            # For simplicity, value is always a numpy array. If it is size 1,
            # we want to pull this out so the data is written in netcdf as
            # a scalar rather than a array of size 1
            if len(v.shape) == 1 and v.shape[0] == 1:
                structIn[str(sid)] = v[0]
            else:
                structIn[str(sid)] = v
            structUnits.append(extra_l2_output.net_cdf_struct_units(sid))
            self.exact_cased_variable_names[str(sid)] = (
                extra_l2_output.net_cdf_variable_name(sid)
            )
            self.groupvarnames.append(
                [
                    extra_l2_output.net_cdf_group_name(sid),
                    extra_l2_output.net_cdf_variable_name(sid),
                ]
            )

        # ===============================
        # write data (Use the service of cdf_write() function.)
        # ===============================

        # Note that for this call to cdf_write:
        #   1.  We pass in dims which is a list containing all dimension names and their sizes.
        #   2.  We also pass in a dictionary containing the exact variable names.
        self.cdf_write(
            structIn,
            filenameOut,
            structUnits,
            dims=dims,
            lowercase=False,
            exact_cased_variable_names=self.exact_cased_variable_names,
            groupvarnames=self.groupvarnames,
            global_attr=global_attr,
        )

    def write_lite(
        self,
        stepNumber: int,
        filenameIn: str,
        current_state: CurrentState,
        instrument: list[InstrumentIdentifier],
        liteDirectory: str,
        data1In: dict,
        data2: dict | None = None,
        species_name: str = "",
        runtimeAttributes: dict | None = None,
        state_element_out: list[StateElementIdentifier] | None = None,
    ) -> None:
        """This is a lightly edited version of make_lite_casper_script_retrieval,
        mainly we want this to call our cdf_write_tes so we can add new species in.

        We pass in state_element_out for StateElementIdentifier not otherwise handled. This
        separates out the species that were in muses-py vs stuff we may have added.
        Perhaps we'll get all the StateElements handled the same way at some point,
        muses-py is really overly complicated. On the other hand, this is just output
        code which tends to get convoluted to create specific output, so perhaps
        this isn't so much an issue.
        """
        data1 = copy.deepcopy(data1In)

        version = "v006"
        version = "CASPER"

        versionLite = "v08"
        pressuresMax = [
            1040.0,
        ]

        # These are hardcode values. So we don't depend on muses-py, just set the
        # value here that will get returned
        # starttai = mpy_tai(
        #    {"year": 2003, "month": 1, "day": 1, "hour": 0, "minute": 0, "second": 0},
        #    True,
        # )
        # endtai = mpy_tai(
        #    {"year": 2103, "month": 1, "day": 1, "hour": 0, "minute": 0, "second": 0},
        #    True,
        # )
        starttai = 315532805
        endtai = 3471206406

        if not filenameIn.endswith(".nc"):
            filename = os.path.basename(filenameIn)
            nc_pos = filename.index(".")

            optional_separator = ""
            if not (os.path.dirname(filenameIn)).endswith(os.path.sep):
                optional_separator = os.path.sep

            filenameOut = (
                os.path.dirname(filenameIn)
                + optional_separator
                + "Lite_"
                + filename[0 : nc_pos + 1]
                + "nc"
            )
        else:
            filenameOut = filenameIn

        runs = filenameIn
        dataAnc = copy.deepcopy(data1)
        data, pressuresMax = CdfWriteLiteTes().make_one_lite(
            species_name,
            current_state,
            runs,
            starttai,
            endtai,
            [str(i) for i in instrument],
            pressuresMax,
            liteDirectory,
            version,
            versionLite,
            data1,
            data2,
            dataAnc,
        )
        tracer = species_name
        retrieval_pressures = pressuresMax
        version = version
        liteVersion = versionLite
        write_met = False
        self.write(
            data,
            filenameOut,
            current_state,
            tracer,
            retrieval_pressures,
            write_met,
            version,
            liteVersion,
            runtimeAttributes=runtimeAttributes,
            state_element_out=state_element_out,
        )

    def cdf_write(
        self,
        struct_in: dict[str, Any],
        filename: str | os.PathLike[str],
        struct_units: list[dict[str, str | float]],
        dims: dict[str, int] | None = None,
        lowercase: bool = False,
        exact_cased_variable_names: dict[str, str] | None = None,
        groupvarnames: list[Any] | None = None,
        global_attr: dict[str, Any] | None = None,
    ) -> None:
        nco = Dataset(filename, "w")
        nco.set_auto_maskandscale(False)
        if dims is not None:
            for dim_key, dim_value in dims.items():
                nco.createDimension(dim_key, dim_value)

        # Loop over struct and write variable.
        # AT_LINE 483 TOOLS/cdf_write.pro cdf_write
        self.cdf_write_struct(
            nco,
            struct_in,
            filename,
            struct_units,
            dims,
            lowercase,
            exact_cased_variable_names,
            groupvarnames,
        )

        # update creation date
        # It is convenient in testing to have a fixed creation_date, just so we can
        # compare files created at different times with h5diff and have them compare as
        # identical
        if "MUSES_FAKE_CREATION_DATE" in os.environ:
            # In test environment, used the supplied date
            nco.setncattr("creation_date", os.environ["MUSES_FAKE_CREATION_DATE"])
        else:
            # Otherwise use the real date
            nco.setncattr(
                "creation_date",
                datetime.datetime.now(tz=pytz.utc).strftime("%Y%m%dT%H%M%SZ"),
            )

        if global_attr is not None:
            nco.setncatts(global_attr)
        nco.close()

    def cdf_var_add_strings(
        self, dataOut: dict[str, Any], dims: dict[str, int]
    ) -> dict[str, Any]:
        """add in string lengths and modify strings to be the same length by
        adding spaces at the end"""
        count = 0
        tt = list(dataOut.keys())
        dataNew = copy.deepcopy(dataOut)
        for jj in range(0, len(tt)):
            key_name = tt[jj]
            ss = self.idl_size(dataOut, key_name)

            # Handle byte data type.
            # AT_LINE 783 TOOLS/cdf_write_tes.pro
            if ss[len(ss) - 2] == 1 and ss[0] == 2:
                strlength = ss[1]
                dims["string" + str(strlength)] = strlength

            # AT_LINE 806 TOOLS/cdf_write_tes.pro
            # Handle string data type.
            if ss[len(ss) - 2] == 7:  # string data type
                # get string array
                # Check to see if the type of dataOut[key_name] is just a plain old string.
                if ss[0] == 0:
                    nn = 1  # There is only 1 string.
                    strarr = []
                    strarr.append(
                        dataOut[key_name]
                    )  # An array of element, which is the string.
                    ll = np.asarray(
                        [len(dataOut[key_name])]
                    )  # Save the length of the one string.
                else:
                    # get maximum string length
                    strarr = dataOut[key_name]
                    nn = len(strarr)
                    ll = np.zeros(shape=(nn), dtype=np.int32)
                    for kk in range(0, nn):
                        ll[kk] = len(strarr[kk])
                    # end for kk in range(0,nn):

                strlength = np.amax(ll)  # maximum string length.
                if strlength < 2:
                    strlength = 2

                # make all strings the same length
                if nn > 1:  # If there are more than 1 string.
                    space = "                                                                        "
                    for kk in range(0, nn):
                        strarr[kk] = strarr[kk] + space[0 : strlength - len(strarr[kk])]

                dims["string" + str(strlength)] = strlength

                # AT_LINE 806 TOOLS/cdf_write_tes.pro
                # now add to structure in format of a bytearr
                if strlength == 0:
                    strlength = 1

                if nn > 1:  # If there are more than 1 string.
                    result = np.zeros(shape=(strlength, nn), dtype=np.dtype("b"))
                else:
                    result = np.zeros(shape=(strlength,), dtype=np.dtype("b"))

                # transfer information
                if nn > 1:  # If there are more than 1 string.
                    for kk in range(0, nn):
                        # convert to byte array
                        result[:, kk] = bytearray(strarr[kk], "utf8")

                        # remove trailing spaces
                        il = strlength - 1
                        test = result[il, kk]
                        while test == 32 and il > -1:
                            result[il, kk] = 0
                            il = il - 1
                            if il >= 0:
                                test = result[il, kk]
                        # end while test EQ 32 and il > -1:
                    # for kk in range(0,nn):
                else:
                    result[...] = bytearray(strarr[0], "utf8")

                dataNew[key_name] = result
                count = count + 1
            # end if ss[len(ss)-2] == 7:  # string data type
        # end FOR jj = 0, N_ELEMENTS(tt)-1 DO BEGIN
        # AT_LINE 871 TOOLS/cdf_write_tes.pro

        return dataNew

    def idl_size(self, dataOut: dict[str, Any], key_name: str) -> list[int]:
        o_ss = None  # The returned array from SIZE() function.

        # Getting the dimension is tricky.  We have to check to see if the type of the variable
        # is scalar or otherwise.
        if np.ndim(dataOut[key_name]) == 0:  # CHECK_FOR SCALAR
            if not np.isscalar(dataOut[key_name]):
                # 0D arrays can have their sole value converted to a non-array scalar like so
                value = dataOut[key_name].item()
            else:
                value = dataOut[key_name]

            o_ss = [
                0
            ]  # Scalar does not have dimension so we set the value of 0 as first element.
            if isinstance(value, float) or isinstance(value, np.float32):
                o_ss.append(4)  # Using IDL convention of assigning 4 to Float
                o_ss.append(
                    1
                )  # Add the last element which is the number of element for scalar.
            elif isinstance(value, int) or isinstance(value, np.int32):
                o_ss.append(2)  # Using IDL convention of assigning 2 to Int
                o_ss.append(
                    1
                )  # Add the last element which is the number of element for scalar.
            elif isinstance(value, np.int16):
                o_ss.append(2)  # Using IDL convention of assigning 2 to Int
                o_ss.append(
                    1
                )  # Add the last element which is the number of element for scalar.
            elif isinstance(value, np.int32):
                o_ss.append(2)  # Using IDL convention of assigning 2 to Int
                o_ss.append(
                    1
                )  # Add the last element which is the number of element for scalar.
            elif isinstance(value, np.int64):
                o_ss.append(2)  # Using IDL convention of assigning 2 to Int
                o_ss.append(
                    1
                )  # Add the last element which is the number of element for scalar.
            elif isinstance(value, str):
                o_ss.append(7)  # Using IDL convention of assigning 7 to String
                o_ss.append(
                    1
                )  # Add the last element which is the number of element for scalar.
            else:
                raise RuntimeError("Do not know your type yet")
        elif isinstance(dataOut[key_name], list):  # CHECK_FOR LIST
            o_ss = [1]  # The dimension of list is 1.
            o_ss.append(len(dataOut[key_name]))
            if isinstance(dataOut[key_name][0], str):
                o_ss.append(7)  # Using IDL convention of assigning 7 to String
                o_ss.append(len(dataOut[key_name]))
            elif isinstance(dataOut[key_name][0], int):
                o_ss.append(2)  # Using IDL convention of assigning 2 to Int
                o_ss.append(len(dataOut[key_name]))
            elif isinstance(dataOut[key_name][0], list) and isinstance(
                dataOut[key_name][0][0], float
            ):
                # Not sure if this correct.
                o_ss.append(4)  # Using IDL convention of assigning 4 to Float
                o_ss.append(len(dataOut[key_name][0]))  # Add the length of the list.
            elif isinstance(dataOut[key_name][0], list) and isinstance(
                dataOut[key_name][0][0], int
            ):
                # Not sure if this correct.
                o_ss.append(2)  # Using IDL convention of assigning 2 to Int
                o_ss.append(len(dataOut[key_name][0]))  # Add the length of the list.
            else:
                raise RuntimeError("Do not know your type yet")
        else:  # CHECK_FOR OTHER
            o_ss = [len(dataOut[key_name].shape)]
            if len(dataOut[key_name].shape) == 1:
                o_ss.append(dataOut[key_name].shape[0])  # One dimension is first.
            elif len(dataOut[key_name].shape) == 2:
                o_ss.append(
                    dataOut[key_name].shape[0]
                )  # First dimension is [0] element.
                o_ss.append(
                    dataOut[key_name].shape[1]
                )  # Second dimension is [1] element.
            else:
                raise RuntimeError("Do not know your type yet")

            if (dataOut[key_name].dtype is np.int32) or str(
                dataOut[key_name].dtype
            ) == "int32":
                o_ss.append(4)  # Using IDL convention of assigning 2 to Int
                o_ss.append(dataOut[key_name].size)

            if (dataOut[key_name].dtype is np.int64) or str(
                dataOut[key_name].dtype
            ) == "int64":
                o_ss.append(14)  # Using IDL convention of assigning 14 to Long64
                o_ss.append(dataOut[key_name].size)

            if (dataOut[key_name].dtype is np.float32) or str(
                dataOut[key_name].dtype
            ) == "float32":
                o_ss.append(4)  # Using IDL convention of assigning 4 to Float
                o_ss.append(dataOut[key_name].size)

            if (dataOut[key_name].dtype is np.float64) or str(
                dataOut[key_name].dtype
            ) == "float64":
                o_ss.append(5)  # Using IDL convention of assigning 5 to Double
                o_ss.append(dataOut[key_name].size)

        return o_ss

    def cdf_write_struct(
        self,
        nco: Dataset,
        struct: dict[str, Any],
        filename: str | os.PathLike[str],
        structUnits: list[dict[str, str | float]],
        dims: dict[str, int] | None = None,
        lowercase: bool = False,
        exact_cased_variable_names: dict[str, str] | None = None,
        groupvarnames: list[Any] | None = None,
    ) -> None:
        tagnamesStruct = list(struct.keys())

        # Special handling not found in IDL: Check to see if the variable_name belongs to a group.  If it is, we create the group first
        # then save each id of the group in dict_of_group_ids_for_writing so we can retrieve it later.
        # Note that in Python, when you call createGroup(), if the group exist, you get the group id back.
        size_to_dim: dict[int, str] = {}
        if dims is not None:
            for k, v in dims.items():
                if v not in size_to_dim:
                    size_to_dim[v] = k
        group_struct: dict[str, list[str]] = {}
        dict_of_group_ids_for_writing = {}
        if groupvarnames is not None:
            for gname, vname in groupvarnames:
                if gname not in group_struct:
                    group_struct[gname] = []
                group_struct[gname].append(vname)
            for jj in range(0, len(struct)):
                variable_name = tagnamesStruct[jj]
                if lowercase:
                    variable_name = tagnamesStruct[jj].lower()
                else:
                    variable_name = variable_name.upper()  # To be consistent with the output from IDL, we change the variable to upper case.
                    if (
                        exact_cased_variable_names is not None
                        and variable_name in exact_cased_variable_names
                    ):
                        variable_name = exact_cased_variable_names[
                            variable_name
                        ]  # Get the exact case of the variable name.

                t = re.sub(r"(_DUPLICATE_KEY_|_duplicate_key_).*", "", variable_name)
                belonged_to_group = ""
                for k, v2 in group_struct.items():
                    if t in v2:
                        belonged_to_group = k
                        break
                if belonged_to_group != "":
                    # Only create the group if we had not created it before.
                    if belonged_to_group not in dict_of_group_ids_for_writing:
                        fileID2 = nco.createGroup(belonged_to_group)
                        dict_of_group_ids_for_writing[belonged_to_group] = (
                            fileID2  # Save the group id associated with the variable.
                        )

        # AT_LINE 152 TOOLS/cdf_write.pro cdf_write_struct program
        # loop over all elements in the struct
        tagnamesStruct = list(struct.keys())

        list_of_dimension_names = []  # Will hold the list of dimension names used in createDimension().
        list_of_dimension_sizes = []  # Will hold the list of dimension sizes used in createDimension().
        tuples_of_dimension_names: list[
            tuple
        ] = []  # Will hold the list of tuple of dimension names for every variable.
        list_of_inner_units = []

        # write dimensions
        for jj in range(0, len(struct)):
            variable_name = tagnamesStruct[jj]

            # AT_LINE 157 TOOLS/cdf_write.pro cdf_write_struct program
            # figure out which dimension(s) it matches
            shape_array = [1]  # Scalar have a shape of [1]
            val = struct[
                variable_name
            ]  # Note that we use the exact value of tagnamesStruct[jj] as key to access the dictionary.
            if isinstance(val, list):
                shape_array = [
                    len(val)
                ]  # If a list, create a list of one element containing the length of the list.
            elif isinstance(val, str):
                shape_array = [
                    len(val)
                ]  # If a string, create a list of one element containing the length of the string.
            elif not isinstance(val, dict):
                if isinstance(val, bytes):
                    shape_array = [len(val)]  # If an array of bytes
                elif isinstance(val, np.ndarray):
                    shape_array = val.shape
                else:
                    shape_array = [1]

            # AT_LINE 166 TOOLS/cdf_write.pro
            if isinstance(val, dict):
                # Eventhough this entry in the dictionary is another dictionary, we must also append a dummy name to tuples_of_dimension_names.
                # So when a tuple of dimensions is retrieved, it is retrieved for the correct key in the dictionary.
                tuples_of_dimension_names.append(("DUMMY_DIMENSION_NAME_FOR_DICT",))
            else:
                # AT_LINE 174 TOOLS/cdf_write.pro
                # Take care of strings
                # convert then continue
                if isinstance(val, str):
                    shape_array = [len(val)]
                # AT_LINE 182 TOOLS/cdf_write.pro

                # AT_LINE 200 TOOLS/cdf_write.pro
                units: dict[str, str | float] = {}
                if structUnits is not None:
                    if len(structUnits) > 0:
                        # In order to get to the jj element in structUnits dictionary, we make a loop first.
                        unit_index = 0
                        for ii in range(0, len(structUnits)):
                            # If structUnits is a list, we know that it would contain a list of dictionaries.
                            if isinstance(structUnits, dict):
                                inner_keys = list(
                                    structUnits.keys()
                                )  # We are only expecting one key.
                                one_dict_element = structUnits[
                                    inner_keys[0]
                                ]  # Use the exact key to get to the first element
                            else:
                                one_dict_element = structUnits[ii]

                            if isinstance(one_dict_element, list):
                                # Since one_dict_element is a list, we know that it contains a list of dictionaries, we can loop through it.
                                inner_keys = []
                                for mm in range(0, len(one_dict_element)):
                                    inner_keys.append(
                                        list(one_dict_element[mm].keys())[0]
                                    )  # Add just one key since each element in the list is one key.

                                unit_index = 0
                                for mm in range(0, len(inner_keys)):
                                    key = inner_keys[mm]
                                    elem = one_dict_element[mm][key]
                                    if jj == mm:
                                        units[key] = (
                                            elem  # If we found a matching index, we have found our unit.
                                        )
                                        list_of_inner_units.append(units)
                                    unit_index += 1
                            else:
                                unit_index = 0
                                if jj == ii:
                                    # If we found a matching index, we have found our unit.
                                    list_of_inner_units.append(units)
                                unit_index += 1
                        # end for ii in range(0,len(structUnits)):
                    # end if len(structUnits) > 0:
                # end if structUnits is not None:

                # Don't forget to handle scalar.
                if len(shape_array) == 0:
                    dimension_name_1 = "one"
                    tuples_of_dimension_names.append((dimension_name_1,))

                # vector
                # AT_LINE 209 TOOLS/cdf_write.pro
                # Here, we collect all possible names of dimensions so we can create them after the file is created.
                if len(shape_array) == 1:
                    if shape_array[0] == 1:
                        dimension_name = "one"
                    else:
                        dimension_name = "grid_" + str(shape_array[0])

                    if dimension_name not in list_of_dimension_names:
                        list_of_dimension_names.append(dimension_name)
                        list_of_dimension_sizes.append(shape_array[0])

                    tuples_of_dimension_names.append(
                        (dimension_name,)
                    )  # Note that for a tuple of 1, we have to make sure to add a comma.

                if len(shape_array) == 2:
                    # The name of the dimensions are 'grid_' + str(s[0]
                    # Special case: if the dimension is 1, we use one
                    if shape_array[0] == 1:
                        dimension_name = "one"
                        dimension_name_1 = "one"
                    else:
                        dimension_name = "grid_" + str(shape_array[0])
                        dimension_name_1 = "grid_" + str(shape_array[0])

                    if (
                        dimension_name not in list_of_dimension_names
                        and shape_array[0] != 1
                    ):
                        list_of_dimension_names.append(dimension_name)
                        list_of_dimension_sizes.append(shape_array[0])

                    if shape_array[1] == 1:
                        dimension_name = "one"
                        dimension_name_2 = "one"
                    else:
                        dimension_name = "grid_" + str(shape_array[1])
                        dimension_name_2 = "grid_" + str(shape_array[1])

                    if (
                        dimension_name not in list_of_dimension_names
                        and shape_array[1] != 1
                    ):
                        list_of_dimension_names.append(dimension_name)
                        list_of_dimension_sizes.append(shape_array[1])

                    tuples_of_dimension_names.append(
                        (dimension_name_1, dimension_name_2)
                    )

                if len(shape_array) == 3:
                    if shape_array[0] == 1:
                        dimension_name = "one"
                        dimension_name_1 = "one"
                    else:
                        dimension_name = "grid_" + str(shape_array[0])
                        dimension_name_1 = "grid_" + str(shape_array[0])

                    if (
                        dimension_name not in list_of_dimension_names
                        and shape_array[0] != 1
                    ):
                        list_of_dimension_names.append(dimension_name)
                        list_of_dimension_sizes.append(shape_array[0])

                    if shape_array[1] == 1:
                        dimension_name = "one"
                        dimension_name_2 = "one"
                    else:
                        dimension_name = "grid_" + str(shape_array[1])
                        dimension_name_2 = "grid_" + str(shape_array[1])

                    if (
                        dimension_name not in list_of_dimension_names
                        and shape_array[1] != 1
                    ):
                        list_of_dimension_names.append(dimension_name)
                        list_of_dimension_sizes.append(shape_array[1])

                    if shape_array[2] == 1:
                        dimension_name = "one"
                        dimension_name_3 = "one"
                    else:
                        dimension_name = "grid_" + str(shape_array[2])
                        dimension_name_3 = "grid_" + str(shape_array[2])

                    if (
                        dimension_name not in list_of_dimension_names
                        and shape_array[2] != 1
                    ):
                        list_of_dimension_names.append(dimension_name)
                        list_of_dimension_sizes.append(shape_array[2])
                    tuples_of_dimension_names.append(
                        (dimension_name_1, dimension_name_2, dimension_name_3)
                    )
                # AT_LINE 277 TOOLS/cdf_write.pro
        # end for jj in range(0,len(struct)):

        # Now that we have the dimension names and their sizes, we can
        # write them to the NetCDF file.  Note that inorder to write a
        # variable to a netCDF file, the dimension must have been
        # created already so we are doing this before writing any
        # variables.  Also note that we only create the dimension here
        # if dims is not passed in, which means we had to create the
        # dimension names ourselves with 'grid_' plus the size of the
        # array.

        if dims is None:
            for ii in range(0, len(list_of_dimension_names)):
                nco.createDimension(
                    list_of_dimension_names[ii], list_of_dimension_sizes[ii]
                )
            nco.createDimension("str_dim", 1)

        # Now that all the dimensions of the root has been created, we get a list of them.
        list_of_root_dimensions = []
        for key, _ in nco.dimensions.items():
            list_of_root_dimensions.append(key)

        # AT_LINE 323 TOOLS/cdf_write.pro

        # write data
        for jj in range(0, len(struct)):
            # AT_LINE 157 TOOLS/cdf_write.pro
            # figure out which dimension(s) it matches
            val = struct[tagnamesStruct[jj]]
            variable_name = tagnamesStruct[jj]
            if lowercase:
                variable_name = tagnamesStruct[jj].lower()
            else:
                variable_name = variable_name.upper()  # To be consistent with the output from IDL, we change the variable to upper case.
                if (
                    exact_cased_variable_names is not None
                    and variable_name in exact_cased_variable_names
                ):
                    variable_name = exact_cased_variable_names[
                        variable_name
                    ]  # Get the exact case of the variable name.

            shape_array = [1]  # Scalar have a shape of [1]
            if isinstance(val, list):
                shape_array = [
                    len(val)
                ]  # If a list, create a list of one element containing the length of the list.
            elif isinstance(val, str):
                shape_array = [
                    len(val)
                ]  # If a string, create a list of one element containing the length of the string.
            elif not isinstance(val, dict):
                if isinstance(val, bytes):
                    shape_array = [len(val)]  # If an array of bytes
                elif isinstance(val, np.ndarray):
                    shape_array = val.shape
                else:
                    shape_array = [1]

            if isinstance(val, dict):
                fileID2 = nco.createGroup(variable_name)
                structUnits2 = None
                if structUnits is not None:
                    # For group variable, we use the array list_of_inner_units which should now contain the list of units for the group variable.
                    structUnits2 = list_of_inner_units

                # Make a recursive call with structUnits2 as the list of units for the group variable.
                self.cdf_write_struct(
                    fileID2,
                    val,
                    str(filename),
                    structUnits2,
                    dims,
                    lowercase,
                    exact_cased_variable_names,
                )
            else:
                variable_index = 0  # Since there is only 1 time index in NetCDF file, the index starts at 0.
                variable_data = val  # This is the variable we wish to write.

                variable_attributes_dict = {}
                if structUnits is not None:
                    variable_attributes_dict = structUnits[jj]

                actual_file_id_to_write = nco  # This is the default root id
                variable_belong_to_group_flag = False

                # Special handling: Check to see if the variable_name belongs to a group.  If it is, we get the group id from above.
                if groupvarnames is not None:
                    t = re.sub(
                        r"(_DUPLICATE_KEY_|_duplicate_key_).*", "", variable_name
                    )
                    belonged_to_group = ""
                    for k, v2 in group_struct.items():
                        if t in v2:
                            belonged_to_group = k
                            break
                    if belonged_to_group != "":
                        # Retrieve the group id from dict_of_group_ids_for_writing when we created the groups above.
                        # Then use group id in call to cdf_write_variable().  Note that we do not overwrite fileID because
                        # it is for writing root variables.
                        groupID = dict_of_group_ids_for_writing[belonged_to_group]
                        actual_file_id_to_write = (
                            groupID  # Update the file id to write to the group.
                        )
                        variable_belong_to_group_flag = True

                # If dims is passed in, we will attempt to get the dimension names from the variable_data.
                if dims is not None:
                    if np.isscalar(variable_data):
                        variable_dimension_tuple: tuple[str, ...] = ()
                    else:
                        try:
                            variable_dimension_tuple = tuple(
                                [size_to_dim[i] for i in variable_data.shape]
                            )
                        except KeyError:
                            variable_dimension_tuple = ()

                    # For Lite products, if there are more than 1 dimensions, we also transpose the dimensions to match the IDL output.
                    if len(variable_dimension_tuple) >= 2:
                        # This is really just noise in the log file, so comment this out
                        # logger.debug(f'Transposing variable: {variable_name}, {filename}')
                        variable_dimension_tuple = tuple(
                            reversed(variable_dimension_tuple)
                        )

                    # For Lite products, we merely transpose the variable to match the IDL output.
                    if not np.isscalar(variable_data) and len(variable_data.shape) > 1:
                        variable_data = variable_data.T
                else:
                    variable_dimension_tuple = tuples_of_dimension_names[jj]
                    # It is possible to receive a scalar for the type of variable_data.
                    # To make life easier, we convert it to an array of 1 element from this point so we don't have to constantly check for scalar.
                    if np.isscalar(variable_data):
                        variable_data = np.asarray(variable_data)

                # Do a sanity check on the variable name if it contains extranous tokens such as "_DUPLICATE_KEY_" or "_duplicate_key_".
                variable_name = re.sub(
                    r"(_DUPLICATE_KEY_|_duplicate_key_).*", "", variable_name
                )

                self.cdf_write_variable(
                    actual_file_id_to_write,
                    variable_name,
                    variable_index,
                    variable_data,
                    variable_attributes_dict,
                    variable_dimension_tuple,
                    variable_belong_to_group_flag,
                    list_of_root_dimensions,
                )

    def cdf_write_variable(
        self,
        i_output_file_handle: Dataset,
        i_variable_name: str,
        i_variable_index: int,
        i_variable_data: np.ndarray,
        i_variable_attributes_dict: dict[str, str | float],
        i_tuples_of_dimension_names: tuple[str, ...],
        i_variable_belong_to_group_flag: bool,
        i_list_of_root_dimensions: list[str],
    ) -> None:
        if len(i_tuples_of_dimension_names) == 0:
            if np.isscalar(i_variable_data) or len(i_variable_data.shape) == 0:
                pass
            elif len(i_variable_data.shape) == 1:
                i_tuples_of_dimension_names = ("grid_" + str(i_variable_data.shape[0]),)
            elif len(i_variable_data.shape) == 2:
                i_tuples_of_dimension_names = (
                    "grid_" + str(i_variable_data.shape[0]),
                    "grid_" + str(i_variable_data.shape[1]),
                )
            else:
                raise RuntimeError(
                    f"Not yet supporting these dimensions: {i_variable_data.shape}"
                )

        # PYTHON_NOTE: It is also possible that the number of tokens in i_tuples_of_dimension_names
        # is different than the shape of the variable, we need to correct the dimension.
        # Ignore scalar variables which has no shape.
        # Also note that for scalar, the name of the only dimension is 'one' so we check that ei_tuples_of_dimension_names[0] is not 'one'
        if (
            len(i_tuples_of_dimension_names) > 0 and not np.isscalar(i_variable_data)
        ) and i_tuples_of_dimension_names[0] != "one":
            if len(i_tuples_of_dimension_names) != len(i_variable_data.shape):
                if len(i_variable_data.shape) == 0:
                    pass
                    # i_variable_name utctime, i_tuples_of_dimension_names ('GRID29',)
                    # Some variables has no shape (for example utctime), so we keep the i_tuples_of_dimension_names as is.
                elif len(i_variable_data.shape) == 1:
                    i_tuples_of_dimension_names = (
                        "grid_" + str(i_variable_data.shape[0]),
                    )
                elif len(i_variable_data.shape) == 2:
                    i_tuples_of_dimension_names = (
                        "grid_" + str(i_variable_data.shape[0]),
                        "grid_" + str(i_variable_data.shape[1]),
                    )
                else:
                    raise RuntimeError(
                        f"Not yet supporting these dimensions: {i_variable_data.shape}"
                    )

        encountered_object_flag = False

        # Special note: When a string type variable is passed in, it has no shape.  We need to make a special note
        # so when the variable is created with createVariable() function, we can either pass in a dimension array or not.
        variable_has_no_shape_flag = False

        # The shape function cannot be used with scalar so we must check to see that it is not a scalar before checking for shape
        # to avoid error such as: AttributeError: 'int' object has no attribute 'shape'
        if not np.isscalar(i_variable_data) and len(i_variable_data.shape) == 0:
            variable_has_no_shape_flag = True

        # If the dimension does not yet, exist, we create it.
        # Becareful if creating dimension for square matrix where both dimensions are the same.
        if (
            i_output_file_handle.dimensions is not None
        ) and i_tuples_of_dimension_names is not None:
            exist_nc_dims = [
                dim for dim in i_output_file_handle.dimensions
            ]  # List of existing nc dimensions.

            # Becareful if creating dimension for square matrices where both dimensions are the same.
            # To prevent from creating dimensions of a square matrix, we keep a list of already created dimensions.
            already_created_dimensions = []
            for _, one_dimension in enumerate(i_tuples_of_dimension_names):
                if (
                    one_dimension not in exist_nc_dims
                    and one_dimension not in already_created_dimensions
                ):
                    if "grid" in one_dimension:
                        str_split = one_dimension.split("_")
                    else:
                        # Split 'GRID36' to get to '36'
                        str_split = one_dimension.split("GRID")
                    if len(str_split) > 1:
                        new_dimension_size = int(str_split[1])
                        if (
                            i_list_of_root_dimensions is not None
                            and one_dimension not in i_list_of_root_dimensions
                        ):
                            i_output_file_handle.createDimension(
                                one_dimension, new_dimension_size
                            )
                already_created_dimensions.append(
                    one_dimension
                )  # Keep track of what had already been created.

        if i_variable_name == "lat":
            dimensions_list: tuple[str, ...] = ("lat",)
        elif i_variable_name == "lon":
            dimensions_list = ("lon",)
        elif i_variable_name == "start_time":
            dimensions_list = ("time", "time_string_len")
        else:
            dimensions_list = i_tuples_of_dimension_names

        ncdf4_data_type: str | type = ""

        if type(i_variable_data) is str:
            ncdf4_data_type = "S1"  # The start_time variable is a single-character string type: char start_time(time, time_string_len)
        elif np.isscalar(i_variable_data):
            # This is probably not correct.
            if isinstance(i_variable_data, np.float64) or isinstance(
                i_variable_data, float
            ):
                ncdf4_data_type = "f8"
            elif isinstance(i_variable_data, np.float32) or isinstance(
                i_variable_data, float
            ):
                ncdf4_data_type = "f4"
            elif isinstance(i_variable_data, np.int64) or isinstance(
                i_variable_data, int
            ):  # i_variable_data.dtype == "int64":
                ncdf4_data_type = "i8"
            elif isinstance(i_variable_data, np.int32) or isinstance(
                i_variable_data, int
            ):  # i_variable_data.dtype == "int32":
                ncdf4_data_type = "i4"
            elif isinstance(i_variable_data, np.int16) or isinstance(
                i_variable_data, int
            ):  # i_variable_data.dtype == "int16":
                ncdf4_data_type = "i2"
            elif isinstance(i_variable_data, np.int8) or isinstance(
                i_variable_data, int
            ):  # i_variable_data.dtype == "int8":
                ncdf4_data_type = "i1"
            elif isinstance(i_variable_data, np.uint8) or isinstance(
                i_variable_data, int
            ):  # i_variable_data.dtype == "uint8":
                ncdf4_data_type = "uint8"
        else:
            if i_variable_data.dtype == "float64":
                ncdf4_data_type = "f8"
            elif i_variable_data.dtype == "float32":
                ncdf4_data_type = "f4"
            elif i_variable_data.dtype == "int64":
                ncdf4_data_type = "i8"
            elif i_variable_data.dtype == "int32":
                ncdf4_data_type = "i4"
            elif i_variable_data.dtype == "int16":
                ncdf4_data_type = "i2"
            elif i_variable_data.dtype == "int8":
                ncdf4_data_type = "i1"
            elif i_variable_data.dtype == "uint8":
                ncdf4_data_type = "uint8"
            elif "S" in str(i_variable_data.dtype) or "U" in str(i_variable_data.dtype):
                # For 'S' or 'U', we assume it is of type 'S' for writing.
                if "S" in str(i_variable_data.dtype):
                    ncdf4_data_type = "S"
                if "U" in str(i_variable_data.dtype):
                    ncdf4_data_type = "U"
            elif str(i_variable_data.dtype) == "object":
                if "microwindow" in i_variable_name.lower():
                    ncdf4_data_type = str
                else:
                    # If we don't know the type, we treat it as byte with 'i1' type.
                    # We also need to change the dimension of the variable.
                    max_length = 0
                    for ii in range(0, len(i_variable_data)):
                        # EM Insertion - Added for CRIS TROPOMI combination, got a strange object which wouldn't be read, so inserted some protection
                        try:
                            max_length = max(max_length, len(i_variable_data[ii]))
                        except TypeError:
                            max_length = 0
                            return

                    dimensions_list = (dimensions_list[0], "grid_" + str(max_length))
                    ncdf4_data_type = "i1"
                    encountered_object_flag = True

                    # Get a list of dimensions so we can check.
                    existed_dimensions = []
                    for key, _ in i_output_file_handle.dimensions.items():
                        existed_dimensions.append(key)

                    # If the 2nd dimension probably not has been created yet, we do it here.
                    if dimensions_list[1] not in existed_dimensions:
                        i_output_file_handle.createDimension(
                            dimensions_list[1], max_length
                        )
                # end: if 'microwindow' in i_variable_name:
            else:
                ncdf4_data_type = "BAD_DATA_TYPE"
                raise RuntimeError("BAD_DATA_TYPE")

            # end: if i_variable_data.dtype == "float64":
        # end: if type(i_variable_data) == str:

        # Create the variable is it does not already exist, else, just retrieve it from the file.
        # This is often the case for the multi dimension variable with the (time,) as the first dimension.
        # Note: NetCDF does not allow the creation of a variable if it already exist.
        # By checking to see if the variable is in the list of variables
        # we can avoid the error of attempting to create a variable that exist.

        if i_variable_name in i_output_file_handle.variables:
            variable_to_write = i_output_file_handle.variables[i_variable_name]
        else:
            if np.isscalar(i_variable_data) or variable_has_no_shape_flag:
                # A scalar has no dimension.  We do not send the dimension_list at all to createVariable() function.
                variable_to_write = i_output_file_handle.createVariable(
                    i_variable_name, ncdf4_data_type, fill_value=False
                )
            else:
                variable_to_write = i_output_file_handle.createVariable(
                    i_variable_name, ncdf4_data_type, dimensions_list, fill_value=False
                )

            # Turn off auto application of the scale and offset otherwise the variable written will be wrong.
            variable_to_write.set_auto_maskandscale(False)
        # end: if i_variable_name in i_output_file_handle.variables:

        # For every attribute name, get the value and write it back out.  It is possible for a variable to have no attribute.
        attributes_written = 0

        # If a varible has multiple dimension, we need to check to see if the attribute has been written in previous write.
        for (
            variable_attribute_name,
            variable_attribute_value,
        ) in i_variable_attributes_dict.items():
            attribute_type = type(variable_attribute_value)

            if variable_attribute_name in variable_to_write.ncattrs():
                continue

            # Depend on if the type is of type number or text, we write it accordingly.
            # Note that we also include the normal Python data types that are numbers.
            if (
                attribute_type is np.ndarray
                or attribute_type is np.int64
                or attribute_type is np.int32
                or attribute_type is np.int16
                or attribute_type is np.int8
                or attribute_type is int
                or attribute_type is int
                or attribute_type is np.float64
                or attribute_type is np.float32
                or attribute_type is float
                or attribute_type is float
            ):
                variable_to_write.setncattr(
                    variable_attribute_name, variable_attribute_value
                )
            elif attribute_type is str:
                variable_to_write.setncattr(
                    variable_attribute_name, str(variable_attribute_value)
                )
            else:
                # This shouldn't happen.  Perhaps should exit program so the program can inspect.
                variable_to_write.setncattr(
                    variable_attribute_name, str(variable_attribute_value)
                )

            attributes_written += 1
        # end: for variable_attribute_name, variable_attribute_value in i_variable_attributes_dict.items():

        if (i_variable_name == "start_time") or (isinstance(i_variable_data, str)):
            # For every character in i_variable_data, set it to i_output_file_handle.variables[i_variable_name][0,character_index:character_index+1]
            # Basically, set one character at a time: ["2" "0" "1" "5" "0" "3" "0" "3" "T" "0" "0" "0" "0" "0" "0" "Z" ""]
            character_index = 0
            for one_character in i_variable_data:
                i_output_file_handle.variables[i_variable_name][
                    i_variable_index, character_index : character_index + 1
                ] = one_character
                character_index += 1
        else:
            num_dimensions = len(i_tuples_of_dimension_names)
            if num_dimensions == 1 or num_dimensions == 0:
                if encountered_object_flag:
                    for ii in range(0, len(i_variable_data)):
                        i_output_file_handle.variables[i_variable_name][ii, :] = ord(
                            " "
                        )  # Fill the entire row with blank spaces first.
                        i_output_file_handle.variables[i_variable_name][
                            ii, 0 : len(i_variable_data[ii])
                        ] = i_variable_data[ii]
                else:
                    i_output_file_handle.variables[i_variable_name][:] = i_variable_data
            else:
                i_output_file_handle.variables[i_variable_name][:] = i_variable_data

        if i_variable_name == "start_time":
            if i_output_file_handle.dimensions["time"].size == 1:
                i_output_file_handle.setncattr("first_start_time", i_variable_data)
                i_output_file_handle.setncattr("last_start_time", i_variable_data)
            else:
                i_output_file_handle.setncattr("last_start_time", i_variable_data)


__all__ = ["RetrievalOutput", "CdfWriteTes", "extra_l2_output"]

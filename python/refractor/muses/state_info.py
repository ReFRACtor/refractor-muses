from __future__ import annotations  # We can remove this when we upgrade to python 3.9
import abc
from .priority_handle_set import PriorityHandleSet
from .observation_handle import mpy_radiance_from_observation_list
from .tes_file import TesFile
from .order_species import order_species
import refractor.muses.muses_py as mpy  # type: ignore
from loguru import logger
import copy
from pathlib import Path
import refractor.framework as rf  # type: ignore
import numpy as np
import os
import pickle
from typing import Any
import typing

if typing.TYPE_CHECKING:
    from .retrieval_info import RetrievalInfo
    from .retrieval_configuration import RetrievalConfiguration
    from .observation_handle import ObservationHandleSet
    from .strategy_table import StrategyTable
    from .retrieval_strategy import RetrievalStrategy
    from .muses_strategy_executor import CurrentStrategyStep


class PropagatedQA:
    """There are a few parameters that get propagated from one step to
    the next. Not sure exactly what this gets looked for, it look just
    like flags copied from one step to the next. But pull this
    together into one place so we can track this.

    TODO Note that we might just make this appear like any other StateElement,
    but at least for now leave this as separate because that is how

    """

    def __init__(self):
        self.propagated_qa = {"TATM": 1, "H2O": 1, "O3": 1}

    @property
    def tatm_qa(self):
        return self.propagated_qa["TATM"]

    @property
    def h2o_qa(self):
        return self.propagated_qa["H2O"]

    @property
    def o3_qa(self):
        return self.propagated_qa["O3"]

    def update(self, retrieval_state_element: list[str], qa_flag: int):
        """Update the QA flags for items that we retrieved."""
        for state_element_name in retrieval_state_element:
            if state_element_name in self.propagated_qa:
                self.propagated_qa[state_element_name] = qa_flag


class StateElement(object, metaclass=abc.ABCMeta):
    """Muses-py tends to call everything in its state "species",
    although in a few places things things are called
    "parameters". These should really be thought of as "things that go
    into a StateVector". So we refer to these as StateElements, which also
    parallels the species we retrieve on referred to as RetrievalElements.

    We try to treat all these things as identical at some level, but
    there is some behavior that is species dependent. We'll sort that
    out, and try to figure out the right design here.

    These get referenced by a "name", usually called a "species_name"
    in the muses-py code. The StateInfo can be used to look these
    up."""

    def __init__(self, state_info: StateInfo, name: str):
        self._name = name
        self.state_info = state_info

    @property
    def name(self):
        return self._name

    @property
    def retrieval_config(self):
        return self.state_info.retrieval_config

    def sa_covariance(self):
        """Return sa covariance matrix, and also pressure. This is what
        ErrorAnalysis needs."""
        raise NotImplementedError()

    def sa_cross_covariance(self, selem2: StateElement):
        """Return the cross covariance matrix with selem 2. This returns None
        if there is no cross covariance."""
        return None

    def should_write_to_l2_product(self, instruments):
        """Give a list of instruments that a retrieval step operates on, return
        True if this should get written to a netCDF L2 Product and Lite file
        (in RetrievalL2Output).

        StateElements that are already in muses-py should return False
        here, since they get otherwise handled. We may change this behavior and move
        the muses-py StateElements to operate the same way, but for now this is
        how this gets handled (see the discussion on RetrievalOutput for the
        state_element_out keyword)."""
        return False

    def net_cdf_struct_units(self):
        """Returns the attributes attached to a netCDF write out of this
        StateElement."""
        return {
            "Longname": self.name.lower(),
            "Units": "degrees",
            "FillValue": "",
            "MisingValue": "",
        }

    def net_cdf_variable_name(self):
        """Variable name to use when writing to a netCDF file."""
        return self.name

    def net_cdf_group_name(self):
        """Group that variable goes into in a netCDF file. Use the empty string
        if this doesn't go into a group, but rather is a top level variable."""
        return ""

    def clone_for_other_state(self):
        """StateInfo has copy_current_initialInitial and copy_current_initial.
        The simplest thing would be to just copy the current dict. However,
        the muses-py StateElement maintain their state outside of the classes in
        various dicts in StateInfo (probably left over from IDL). So we have
        this function. For ReFRACtor StateElement, this should just be a copy of
        StateElement, but for muses-py we return None. The copy_current_initialInitial
        and copy_current_initial then handle these two cases."""
        # Default is to just copy all the data, other than the reference
        # to state_info we have. Derived classes should override this to make
        # sure we aren't copying things that shouldn't be copied (e.g., a
        # open file handle or something like that).
        sinfo_save = self.state_info
        try:
            self.state_info = None
            res = copy.deepcopy(self)
            res.state_info = sinfo_save
        finally:
            # Restore this even if an error occurs
            self.state_info = sinfo_save
        return res

    @property
    @abc.abstractmethod
    def value(self):
        raise NotImplementedError


class RetrievableStateElement(StateElement):
    """This has additional functionality to have a StateElement be retrievable,
    so things like having a priori and initial guess needed in a retrieval. Most
    StateElements are retrievable, but not all - so we separate out the
    functionality."""

    @abc.abstractmethod
    def update_state_element(
        self,
        retrieval_info: RetrievalInfo,
        results_list: np.ndarray,
        update_next: bool,
        retrieval_config: RetrievalConfiguration,
        step: int,
        do_update_fm: np.ndarray,
    ):
        """Update the state element based on retrieval results.
        The current state is always updated, but in some cases we don't want this
        propogated to the next retrieval step. So we have both a "current" and a
        "next_state". If update_next is False, we update current only. Otherwise
        both get updated."""
        raise NotImplementedError

    @abc.abstractmethod
    def update_initial_guess(self, current_strategy_step: CurrentStrategyStep):
        """Create/update a initial guess. This currently fills in a number
        of member variables. I'm not sure that all of this is actually needed,
        we may clean up this list. But right now RetrievalInfo needs all these
        values. We'll perhaps clean up RetrievalInfo, and then in turn clean this
        up.

        The list of variables filled in are:

        self.mapType
        self.pressureList
        self.altitudeList
        self.constraintVector
        self.initialGuessList
        self.trueParameterList
        self.pressureListFM
        self.altitudeListFM
        self.constraintVectorFM
        self.initialGuessListFM
        self.trueParameterListFM
        self.minimum
        self.maximum
        self.maximum_change
        self.mapToState
        self.mapToParameters
        self.constraintMatrix
        """
        raise NotImplementedError


class StateElementHandle(object, metaclass=abc.ABCMeta):
    """Return 3 StateElement objects, for initialInitial, initial and
    current state. For many classes, this will just be a deepcopy of the same class,
    but at least for now the older muses-py code stores these in different places.
    We can perhaps change this interface in the future as we move out the old muses-py
    stuff, it is sort of odd to create these 3 things at once. But we'll match
    what muses-py does for now.

    Note StateElementHandle can assume that they are called for the same target, until
    notify_update_target is called. So if it makes sense, these objects can do internal
    caching for things that don't change when the target being retrieved is the same from
    one call to the next."""

    def notify_update_target(self, rs: RetrievalStrategy):
        """Clear any caching associated with assuming the target being retrieved is fixed"""
        # Default is to do nothing
        pass

    @abc.abstractmethod
    def state_element_object(
        self, state_info: StateInfo, name: str
    ) -> tuple[bool, tuple[StateElement, StateElement, StateElement] | None]:
        raise NotImplementedError


class StateElementHandleSet(PriorityHandleSet):
    """This maps a species name to the SpeciesOrParametersState object that handles
    it.

    Note StatElementHandle can assume that they are called for the same target, until
    notify_update_target is called. So if it makes sense, these objects can do internal
    caching for things that don't change when the target being retrieved is the same from
    one call to the next."""

    def state_element_object(
        self, state_info: StateInfo, name: str
    ) -> tuple[StateElement, StateElement, StateElement]:
        return self.handle(state_info, name)

    def notify_update_target(self, rs: RetrievalStrategy):
        """Clear any caching associated with assuming the target being retrieved is fixed"""
        for p in sorted(self.handle_set.keys(), reverse=True):
            for h in self.handle_set[p]:
                h.notify_update_target(rs)

    def handle_h(
        self, h: StateElementHandle, state_info: StateInfo, name: str
    ) -> tuple[bool, tuple[StateElement, StateElement, StateElement] | None]:
        return h.state_element_object(state_info, name)


class SoundingMetadata:
    """Not really clear that this belongs in the StateInfo, but the muses-py seems
    to at least allow the possibility of this changing from one step to the next.
    I'm not sure if that actually can happen, but there isn't another obvious place
    to put this metadata so we'll go ahead and keep this here."""

    def __init__(self, state_info, step="current"):
        if step not in ("current", "initial", "initialInitial"):
            raise RuntimeError(
                "Don't support anything other than the current, initial, or initialInitial step"
            )
        self._latitude = rf.DoubleWithUnit(
            state_info.state_info_dict[step]["latitude"], "deg"
        )
        self._longitude = rf.DoubleWithUnit(
            state_info.state_info_dict[step]["longitude"], "deg"
        )
        self._surface_altitude = rf.DoubleWithUnit(
            state_info.state_info_dict[step]["tsa"]["surfaceAltitudeKm"], "km"
        )
        self._height = rf.ArrayWithUnit_double_1(
            state_info.state_info_dict[step]["heightKm"], "km"
        )
        self._surface_type = state_info.state_info_dict[step]["surfaceType"].upper()
        self._tai_time = state_info._tai_time
        self._sounding_id = state_info._sounding_id
        self._utc_time = state_info._utc_time

    @property
    def latitude(self):
        return self._latitude

    @property
    def longitude(self):
        return self._longitude

    @property
    def surface_altitude(self):
        return self._surface_altitude

    @property
    def height(self):
        return self._height

    @property
    def tai_time(self) -> float:
        return self._tai_time

    @property
    def utc_time(self) -> str:
        return self._utc_time

    @property
    def local_hour(self):
        timestruct = mpy.utc(self.utc_time)
        hour = timestruct["hour"] + self.longitude.convert("deg").value / 180.0 * 12
        if hour < 0:
            hour += 24
        if hour > 24:
            hour -= 24
        return hour

    @property
    def wrong_tai_time(self):
        """The muses-py function mpy.tai uses the wrong number of leapseconds, it
        doesn't include anything since 2006. To match old data, return the incorrect
        value so we can match the file. This should get fixed actually."""
        timestruct = mpy.utc(self.utc_time)
        if timestruct["yearfloat"] >= 2017.0:
            extraleapscond = 4
        elif timestruct["yearfloat"] >= 2015.5:
            extraleapscond = 3
        elif timestruct["yearfloat"] >= 2012.5:
            extraleapscond = 2
        elif timestruct["yearfloat"] >= 2009.0:
            extraleapscond = 1
        else:
            extraleapscond = 0
        return self._tai_time - extraleapscond

    @property
    def sounding_id(self):
        return self._sounding_id

    @property
    def surface_type(self):
        return self._surface_type

    @property
    def is_ocean(self):
        return self.surface_type == "OCEAN"

    @property
    def is_land(self):
        return self.surface_type == "LAND"


class StateInfo:
    """State Info during a retrieval - so what gets passed between RetrievalStrategyStep.

    Note that StateInfo can assume that they are called for the same target, until
    notify_update_target is called. So if it makes sense, these objects can do internal
    caching for things that don't change when the target being retrieved is the same from
    one call to the next.

    This is also true for all the StateElementHandle we have.

    """

    def __init__(self):
        self.state_element_handle_set = copy.deepcopy(
            StateElementHandleSet.default_handle_set()
        )
        self.notify_update_target(None)

    def notify_update_target(self, rs: RetrievalStrategy):
        logger.debug(f"Call to {self.__class__.__name__}::notify_update")
        self.state_info_dict: dict[str, Any] = {}
        self.initialInitial: dict[str, StateElement] = {}
        self.initial: dict[str, StateElement] = {}
        self.current: dict[str, StateElement] = {}
        self.next_state: dict[str, StateElement] = {}
        self.next_state_dict: dict[str, StateElement] | None = {}

        self.propagated_qa = PropagatedQA()
        self.brightness_temperature_data: dict[int, dict[str, float | None]] = {}

        # Odds and ends that are currently in the StateInfo. Doesn't exactly have
        # to do with the state, but we don't have another place for these.
        # Perhaps SoundingMetadata can migrate into its own thing, and these can
        # get moved over to there.
        self._tai_time = -999.0
        self._utc_time = ""
        self._sounding_id = ""
        self.retrieval_config = rs.retrieval_config if rs is not None else None
        self.state_element_handle_set.notify_update_target(rs)

    def init_state(
        self,
        strategy_table: StrategyTable,
        observation_handle_set: ObservationHandleSet,
        retrieval_elements_all: list[str],
        error_analysis_interferents_all: list[str],
        instrument_name_all: list[str],
        run_dir: Path,
    ):
        (_, _, _, _, _, _, self.state_info_dict) = self.script_retrieval_setup_ms(
            strategy_table.strategy_table_dict, False
        )
        # state_initial_update needs radiance for some of the instruments. It used this
        # in the function calls like supplier_nh3_type_cris. We only need this for CRIS,
        # AIRS, and TES, but we just collect it for everything
        olist = [
            observation_handle_set.observation(iname, None, None, None)
            for iname in instrument_name_all
        ]
        rad = mpy_radiance_from_observation_list(olist, full_band=True)

        fake_table = {
            "errorSpecies": order_species(
                list(set(error_analysis_interferents_all) | set(retrieval_elements_all))
            )
        }
        self.state_info_dict = mpy.states_initial_update(
            self.state_info_dict,
            fake_table,
            rad,
            instrument_name_all,
        )

        # Read some metadata that isn't already available
        f = TesFile(run_dir / "DateTime.asc")
        self._tai_time = float(f["TAI_Time_of_ZPD"])
        self._utc_time = f["UTC_Time"]
        self._sounding_id = TesFile(run_dir / "Measurement_ID.asc")["key"]
        self.next_state_dict = None

    def script_retrieval_setup_ms(self, i_table_struct, i_writeOutput):
        # IDL_LEGACY_NOTE: This function script_retrieval_setup_ms is the same as script_retrieval_setup_ms in script_retrieval_setup_ms.pro file.

        utilDir = mpy.UtilDir()

        o_airs = None
        o_cris = None
        o_omi = None
        o_tropomi = None
        o_tes = None
        o_oco2 = None
        o_stateInfo = None

        # Open and read measurement ID file.
        # AT_LINE 60 script_retrieval_setup_ms.pro script_retrieval_setup_ms
        instrument_file_name = "Measurement_ID.asc"

        (_, o_file_content) = mpy.read_all_tes(instrument_file_name)
        file_id = mpy.tes_file_get_struct(o_file_content)

        # AT_LINE 61 script_retrieval_setup_ms.pro script_retrieval_setup_ms
        if "oceanFlag" in file_id["preferences"]:
            oceanFlag = int(file_id["preferences"]["oceanFlag"])
        elif "OCEANFLAG" in file_id["preferences"]:
            oceanFlag = int(file_id["preferences"]["OCEANFLAG"])
        else:
            logger.info(
                "ERROR: Could not find 'oceanflag' or 'OCEANFLAG' from preferences:",
                file_id["preferences"],
            )
            assert False

        # GMAO to be used in get_state_intial
        gmao_path = "../GMAO/"
        gmao_type = ""
        if "GMAO" in file_id["preferences"] and "GMAO_TYPE" in file_id["preferences"]:
            gmao_path = str(file_id["preferences"]["GMAO"])
            gmao_type = str(file_id["preferences"]["GMAO_TYPE"])

        # AT_LINE 62 script_retrieval_setup_ms.pro script_retrieval_setup_ms
        my_key = file_id[
            "preferences"
        ][
            "key"
        ]  # The token 'key' is a reserved word in Python.  We change the variable (left hand side) from key to my_key.

        # AT_LINE 65 script_retrieval_setup_ms.pro script_retrieval_setup_ms
        # Open and read Retrieval Strategy Table.
        # PYTHON_NOTE: There is a good chance that the strategy table would have already been read and parsed.  We don't need to do that
        #              again.

        directoryIG = mpy.table_get_pref(i_table_struct, "initialGuessDirectory")
        directoryConstraint = mpy.table_get_pref(
            i_table_struct, "constraintVectorDirectory"
        )  # where state goes is specified by the Table

        if i_writeOutput:
            outdir = "." + os.path.sep + "Input"
            directoryIG = "." + os.path.sep + directoryIG
            directoryConstraint = "." + os.path.sep + directoryConstraint

            # Make sure directories exist.
            utilDir.make_dir(outdir)
            utilDir.make_dir(directoryIG)
            utilDir.make_dir(directoryConstraint)

        # AT_LINE 84 script_retrieval_setup_ms.pro script_retrieval_setup_ms

        # First see what instruments are used in the retrieval. Read through all windows files and get instrument list

        # Get micro windows from strategy table for all retrieval steps.
        windows = mpy.table_new_mw_from_all_steps(i_table_struct)

        # There may be more than one instruments in windows list.
        instruments = []
        for one_window in windows:
            if one_window["instrument"] not in instruments:
                instruments.append(one_window["instrument"])

        instrument_name = "DUMMY_INSTRUMENT_NAME"

        # AT_LINE 88 src_ms-2019-05-29/script_retrieval_setup_ms.pro script_retrieval_setup_ms

        # Get lat/lon/time from appropriate instrument

        if "TES" in instruments:
            instrument_name = "TES"
        elif "AIRS" in instruments:
            instrument_name = "AIRS"
        elif "CRIS" in instruments:
            instrument_name = "CRIS"
        elif "OMI" in instruments:
            instrument_name = "OMI"
        elif "TROPOMI" in instruments:
            instrument_name = "TROPOMI"
        elif "OCO2" in instruments:
            instrument_name = "OCO2"
        else:
            logger.info(
                "ERROR: Unknown instrument.  Must have TES, AIRS, CRIS, OMI, TROPOMI, OCO2 specified as instrument in windows files"
            )
            assert False

        if instrument_name not in [
            "TES",
            "AIRS",
            "CRIS",
            "OMI",
            "OMPS-NM",
            "TROPOMI",
            "OCO2",
        ]:
            logger.info(
                "ERROR: Unknown instrument. Must have TES, AIRS, CRIS, OMI, OMPS-NM, TROPOMI or OCO2 specified as instrument in Measurement ID file"
            )
            assert False

        # NOTE in this file, latitude and longitude are the ONLY things capitalized of all variables in this file...
        # change this but accept capitalized also because of multiple users of this code.
        # replaces code that tediously spells out everything instrument by instrument.
        # add tofile_id['preferences'] as {instrument_name}_utcTime, {instrument_name}_longitude, {instrument_name}_latitude

        o_latitude = 0
        o_longitude = 0
        o_dateStruct = {}

        # allow utcTime or time (tai)
        try:
            dateStruct = mpy.utc_from_string(
                file_id["preferences"][f"{instrument_name}_utcTime"]
            )
            o_dateStruct["dateStruct"] = dateStruct
            o_dateStruct["year"] = dateStruct["utctime"].year
            o_dateStruct["month"] = dateStruct["utctime"].month
            o_dateStruct["day"] = dateStruct["utctime"].day
            o_dateStruct["hour"] = dateStruct["utctime"].hour
            o_dateStruct["minute"] = dateStruct["utctime"].minute
            o_dateStruct["second"] = dateStruct["utctime"].second
        except KeyError:
            o_dateStruct = mpy.tai(
                np.float64(file_id["preferences"][f"{instrument_name}_time"])
            )
            file_id["preferences"][f"{instrument_name}_utcTime"] = mpy.utc(
                o_dateStruct, True
            )

        # allow capitalized or lower case latitude, longitude
        try:
            o_latitude = float(file_id["preferences"][f"{instrument_name}_latitude"])
            o_longitude = float(file_id["preferences"][f"{instrument_name}_longitude"])
        except KeyError:
            o_latitude = float(file_id["preferences"][f"{instrument_name}_Latitude"])
            o_longitude = float(file_id["preferences"][f"{instrument_name}_Longitude"])
            file_id["preferences"][f"{instrument_name}_latitude"] = o_latitude
            file_id["preferences"][f"{instrument_name}_longitude"] = o_longitude

        # PYTHON_NOTE:  To keep this function from getting too big, parts specific to an instrument will be farmed out to specific class
        #               to handle reading in calculated radiance from previous run or to calculate the radiance for the first time.

        # Need to add capability to handle synthetic radiances
        # See script_retrieval_setup_ms.pro line 150 in ssund branch

        # if radianceSource is set to "Synthetic" then:
        # get instrument info from Measurement_ID.asc, OCO_filename
        # and use the initial guess and constraint from this location also.

        # could add a new mode where generates radiances
        # current synthetic assumes radiances are premade
        if mpy.table_get_pref(i_table_struct, "radianceSource") == "Synthetic":
            if "TES" in instruments:
                logger.info(
                    "ERROR: This section of the code for synthetic input is not implemented. instrument_name: ",
                    instrument_name,
                )
                assert False

            if "AIRS" in instruments:
                logger.info(
                    "ERROR: This section of the code for synthetic input is not implemented. instrument_name: ",
                    instrument_name,
                )
                assert False
            # end instrument_name == 'AIRS':

            # AT_LINE 285 src_ms-2019-05-29/script_retrieval_setup_ms.pro script_retrieval_setup_ms
            if "CRIS" in instruments:
                logger.info(
                    "ERROR: This section of the code for synthetic input is not implemented. instrument_name: ",
                    instrument_name,
                )
                assert False
            # end instrument_name == 'CRIS':

            if "OMI" in instruments:
                logger.info(
                    "ERROR: This section of the code for synthetic input is not implemented. instrument_name: ",
                    instrument_name,
                )
                assert False
            # end instrument_name == 'OMI':

            if "OMPS-NM" in instruments:
                logger.info(
                    "ERROR: This section of the code for synthetic input is not implemented. instrument_name: ",
                    instrument_name,
                )
                assert False
            # end instrument_name == 'OMPS-NM':

            if "TROPOMI" in instruments:
                logger.info(
                    "ERROR: This section of the code for synthetic input is not implemented. instrument_name: ",
                    instrument_name,
                )
                assert False
            # end instrument_name == 'TROPOMI':

            if "OCO2" in instruments:
                # get info from pre-generated synthetic radiance and state
                filename = file_id["preferences"]["OCO2_filename"]
                file, dunno, dunno = mpy.cdf_read_dict(filename)

                # convert from bytearray to strings
                mpy.dict_convert_string(file)

                o_stateInfo = file["state"]

                # writer has issue that it writes single numbers as arrays, convert all single # arrays back to single #s
                mpy.dict_convert_single(o_stateInfo)

                o_oco2 = file["oco2"]
                # set table.pressurefm to stateConstraint.pressure because OCO-2 is on sigma levels
                i_table_struct["pressureFM"] = o_stateInfo["constraint"]["pressure"]

                o_oco2["observation_id"] = file_id["preferences"]["OCO2_sounding_id"]

                # check if radiance is 0 or NESR, if so, run
                if np.mean(o_oco2["radianceStruct"]["radiance"]) < np.mean(
                    o_oco2["radianceStruct"]["NESR"]
                ):
                    # uip, dunno, dunno = cdf_read_dict(Path(Path(filename).parent,Path('uip.nc')))
                    uip = file["uip_true"]
                    radiance, xx, xy, xz, xa = mpy.fm_wrapper_one(i_uip=uip)
                    # add radiance + possibly noise
                    o_oco2["radianceStruct"]["radiance"] = o_oco2["radiance_error"] + (
                        radiance["radiance"]
                    ).reshape(radiance["radiance"].size)
                    # fm_wrapper_one(write_output=True)
                    # os.chdir(cwd)

                outputFilenameOCO2 = "./Input/Radiance_OCO2_" + my_key + ".nc"

            if i_writeOutput:
                # Because the write_state function modify the 'current' fields of stateInitial structure, we give it a copy.
                stateConstraintCopy = copy.deepcopy(o_stateInfo["constraint"])
                mpy.write_state(
                    directoryConstraint,
                    mpy.ObjectView(o_stateInfo),
                    mpy.ObjectView(stateConstraintCopy),
                    my_key="_" + my_key,
                    writeAltitudes=0,
                )
                del stateConstraintCopy  # Delete the temporary object.

                # Because the write_state function modify the 'current' fields of stateInitial structure, we give it a copy.
                stateInitialCopy = copy.deepcopy(o_stateInfo["initial"])
                mpy.write_state(
                    directoryIG,
                    mpy.ObjectView(o_stateInfo),
                    mpy.ObjectView(stateInitialCopy),
                    my_key="_" + my_key,
                    writeAltitudes=0,
                )
                del stateInitialCopy  # Delete the temporary object.

                utilDir.make_dir(outdir + "/True")
                # Because the write_state function modify the 'current' fields of stateInitial structure, we give it a copy.
                stateTrueCopy = copy.deepcopy(o_stateInfo["true"])
                mpy.write_state(
                    outdir + "/True/",
                    mpy.ObjectView(o_stateInfo),
                    mpy.ObjectView(stateTrueCopy),
                    my_key="_" + my_key,
                    writeAltitudes=0,
                )
                del stateTrueCopy  # Delete the temporary object.

            # At this point, we have 5 separate fields in o_stateInfo with 5 separate memory locations.
            return (
                o_airs,
                o_cris,
                o_omi,
                o_tropomi,
                o_tes,
                o_oco2,
                o_stateInfo,
            )  # More instrument data later.
        else:
            # get info from L1B

            # PYTHON_NOTE:  To keep this function from getting too big, parts specific to an instrument will be farmed out to specific class
            #               to handle reading in calculated radiance from previous run or to calculate the radiance for the first time.

            ## FUTURE IMPROVEMENT NOTE - CONDENSE INTO FUNCTION CALLS, CURRENTLY UNCESSARILY OVERCOMPLEX IN DETAIL
            if "TES" in instruments:
                # outputFilenameTES = './Input/Radiance_TES_ ' + my_key + '.nc'
                o_tes = mpy.read_tes_l1b(file_id, windows)

                # apodize here if specified
                mpy.table_get_pref(i_table_struct, "apodizationMethodObs")
                mpy.table_get_pref(i_table_struct, "apodizationMethodFit")
                mpy.table_get_pref(i_table_struct, "apodizationWindowCombineThreshold")
                apodStrength = mpy.table_get_pref(
                    i_table_struct, "NortonBeerApodizationStrength"
                )
                apodizationFunction = mpy.table_get_pref(
                    i_table_struct, "apodizationFunction"
                )

                if apodizationFunction == "NORTON_BEER":
                    status, file = mpy.read_all_tes(
                        mpy.table_get_pref(
                            i_table_struct, "defaultSpectralWindowsDefinitionFilename"
                        ),
                        "asc",
                    )
                    maxOPD = np.array(mpy.tes_file_get_column(file, "MAXOPD"))
                    filter = np.array(mpy.tes_file_get_column(file, "FILTER"))
                    spacing = np.array(mpy.tes_file_get_column(file, "RET_FRQ_SPC"))

                    # apodization radiance and NESR
                    # if this is synthetic data, need to modify additive noise also.
                    radianceStruct = mpy.radiance_apodize(
                        o_tes["radianceStruct"], apodStrength, filter, maxOPD, spacing
                    )

                    o_tes["radianceStruct"] = radianceStruct
            # end: if 'TES' in instruments:

            # AT_LINE 199 src_ms-2019-05-29/script_retrieval_setup_ms.pro script_retrieval_setup_ms
            if "AIRS" in instruments:
                # Get the radiance either from instrument data or from previous run.
                # AT_LINE 199 script_retrieval_setup_ms.pro script_retrieval_setup_ms

                outputFilenameAIRS = "./Input/Radiance_AIRS_ " + my_key + ".nc"

                if not os.path.isfile(outputFilenameAIRS):
                    o_airs = mpy.read_airs(file_id, windows)

                    if i_writeOutput:
                        # TODO: Pickle radiance data
                        # write_radiance_airs(o_airs)
                        pass
                else:
                    # TODO: Unpickle radiance data
                    # read_radiance_airs(o_airs)

                    logger.info("ERROR: Not implemented", outputFilenameAIRS)
                    assert False

            # AT_LINE 285 src_ms-2019-05-29/script_retrieval_setup_ms.pro script_retrieval_setup_ms
            if "CRIS" in instruments:
                # AT_LINE 281-350 src_ms-2018-12-10/script_retrieval_setup_ms.pro
                # AT_LINE 281-367 src_ms-2019-05-29/script_retrieval_setup_ms.pro
                outputFilenameCRIS = "./Input/Radiance_CRIS_" + my_key + ".nc"

                if not os.path.isfile(outputFilenameCRIS):
                    filename = file_id["preferences"]["CRIS_filename"]

                    logger.info("CRIS_filename", filename)

                    # AT_LINE 291 src_ms-2019-05-29/script_retrieval_setup_ms.pro

                    # check filename to see source
                    # check filename to see source
                    if "nasa_nsr" in filename:
                        cris_type = "suomi_nasa_nsr"  # 0
                        # uses fsr reader, same content format as fsr
                        o_cris = mpy.read_nasa_cris_fsr(file_id["preferences"])
                    elif "nasa_fsr" in filename:
                        cris_type = "suomi_nasa_fsr"  # 1
                        # add in type 2 with nomw based on date.
                        # CrIS Suomi NPP (Full Spectral Resolution), L1B data is generated by NASA
                        # /project/muses/input/cris/nasa_fsr
                        o_cris = mpy.read_nasa_cris_fsr(file_id["preferences"])
                    elif "jpss_1_fsr" in filename:
                        cris_type = "jpss1_nasa_fsr"  # 3
                        # CrIS JPSS-1 / NOAA-20 (Full Spectral Resolution), L1B data is generated by NASA
                        # /project/muses/input/cris/jpss_1_fsr
                        o_cris = mpy.read_nasa_cris_fsr(file_id["preferences"])
                    elif "snpp_fsr" in filename:
                        cris_type = "suomi_cspp_fsr"  # 4
                        # CrIS Suomi NPP, L1B data is generated by CSPP (Community Satellite Processing Package)
                        # /project/muses/input/cris_cspp/snpp_fsr
                        o_cris = mpy.read_noaa_cris_fsr(file_id["preferences"])
                    elif "noaa_fsr" in filename:
                        cris_type = "suomi_noaa_fsr"  # 5
                        # CrIS JPSS-1 / NOAA-20 (Full Spectral Resolution), L1B data is generated by CSPP (Community Satellite Processing Package)
                        # /project/muses/input/cris_cspp/noaa_fsr
                        o_cris = mpy.read_noaa_cris_fsr(file_id["preferences"])
                    else:
                        logger.info("ERROR: Add case for CRIS type", filename)
                        assert False

                    # AT_LINE 334 src_ms-2019-05-29/script_retrieval_setup_ms.pro
                    radiance = o_cris["radiance".upper()]
                    frequency = o_cris["frequency".upper()]
                    nesr = o_cris["nesr".upper()]

                    filters = np.array(["CrIS-fsr-lw" for ii in range(len(nesr))])
                    ind_arr = np.where(frequency > 1200)[0]
                    if len(ind_arr) > 0:
                        filters[ind_arr] = "CrIS-fsr-mw"
                    ind_arr = np.where(frequency > 2145)[0]
                    if len(ind_arr) > 0:
                        filters[ind_arr] = "CrIS-fsr-sw"

                    old = 0
                    if old:
                        filters = np.array(["2B1" for ii in range(len(nesr))])

                        ind_arr = np.where(frequency > 950)[0]
                        if len(ind_arr) > 0:
                            filters[ind_arr] = (
                                "1B2"  # All frequencies above 950 are '1B2'
                            )

                        ind_arr = np.where(frequency > 1119.8)[0]
                        if len(ind_arr) > 0:
                            filters[ind_arr] = (
                                "2A1"  # All frequencies above 1119.8 are '2A1'
                            )

                        ind_arr = np.where(frequency > 1444)[0]
                        if len(ind_arr) > 0:
                            filters[ind_arr] = (
                                "2A3"  # All frequencies above 1444 are '2A3'
                            )

                        ind_arr = np.where(frequency > 1890.8)[0]
                        if len(ind_arr) > 0:
                            filters[ind_arr] = (
                                "1A1"  # All frequencies above 1890.8 are '1A1'
                            )

                    o_cris["radianceStruct".upper()] = mpy.radiance_data(
                        radiance, nesr, [0], frequency, filters, "CRIS"
                    )

                    if i_writeOutput:
                        # TODO: Pickle radiance data
                        # write_radiance_cris(o_cris)
                        pass
                else:
                    # TODO: Unpickle radiance data
                    # read_radiance_cris(o_cris)

                    logger.info("ERROR: Not implemented", outputFilenameCRIS)
                    assert False
            # end instrument_name == 'CRIS':

            # AT_LINE 378 src_ms-2019-05-29/script_retrieval_setup_ms.pro script_retrieval_setup_ms
            if "OMI" in instruments:
                # AT_LINE 360-389 src_ms-2018-12-10/script_retrieval_setup_ms.pro
                # AT_LINE 378-406 src_ms-2019-05-29/script_retrieval_setup_ms.pro

                outputFilenameOMI = "./Input/Radiance_OMI_" + my_key + ".pkl"

                if not os.path.isfile(outputFilenameOMI):
                    OMI_RAD_CALRUN_FLAG = 0
                    if "OMI_Rad_calRun_flag" in file_id["preferences"]:
                        OMI_RAD_CALRUN_FLAG = int(
                            float(file_id["preferences"]["OMI_Rad_calRun_flag"])
                        )

                    filename = file_id["preferences"]["OMI_filename"]

                    if "OMI_Cloud_filename" in file_id["preferences"]:
                        cldFilename = file_id["preferences"]["OMI_Cloud_filename"]
                    else:
                        cldFilename = None

                    if OMI_RAD_CALRUN_FLAG == 0:
                        calibrationFilename = mpy.table_get_pref(
                            i_table_struct, "omi_calibrationFilename"
                        )
                    else:
                        calibrationFilename = None
                    o_omi = mpy.read_omi(
                        filename,
                        int(file_id["preferences"]["OMI_XTrack_UV2_Index"]),
                        int(file_id["preferences"]["OMI_ATrack_Index"]),
                        file_id["preferences"]["OMI_utcTime"],
                        calibrationFilename,
                        cldFilename=cldFilename,
                    )

                    # If something goes wrong with read_omi() function, we return immediately.
                    if o_omi is None:
                        return (
                            o_airs,
                            o_cris,
                            o_omi,
                            o_tropomi,
                            o_tes,
                            o_oco2,
                            o_stateInfo,
                        )  # Return due to error.

                    if i_writeOutput:
                        # TODO:
                        # Create directory if it doesn't already exist then write PICKLE file.
                        # output_dir = os.path.dirname(outputFilenameOMI)
                        # utilDir.make_dir(output_dir)

                        # logger.info("PICKLE_ME", outputFilenameOMI)
                        # with open(outputFilenameOMI, 'wb') as pickle_handle:
                        #     pickle.dump(o_omi, pickle_handle, protocol=pickle.HIGHEST_PROTOCOL)
                        pass
                else:
                    logger.info("ERROR: Not implemented", outputFilenameOMI)
                    assert False

                    # TODO:
                    # logger.info("UNPICKLE_ME", outputFilenameOMI)

                    # with open(outputFilenameOMI, 'rb') as pickle_handle:
                    #     o_omi = pickle.load(pickle_handle)
                # end else portion of if not (os.path.isfile(outputFilenameOMI)):

                # modify the OMI NESR if year >= 2010
                if o_dateStruct["year"] >= 2010:
                    ind = np.where(o_omi["Earth_Radiance"]["EarthRadianceNESR"] > 0)[0]
                    o_omi["Earth_Radiance"]["EarthRadianceNESR"][ind] = (
                        o_omi["Earth_Radiance"]["EarthRadianceNESR"][ind] * 2
                    )
            # end: if 'OMI' in instruments:

            if "OMPS-NM" in instruments:
                ######### OMPS-NM
                # EM NOTE - modlling after TROPOMI, keeping seperate for the time being as calibration should not be necessary, plus different
                # L1b structures
                # Only one OMPS band
                outputFilenameTROPOMI = "./Input/Radiance_TROPOMI_" + my_key + ".pkl"
                if (os.getenv("MUSES_DEFAULT_RUN_DIR", "") != "") and (
                    not os.path.isabs(outputFilenameTROPOMI)
                ):
                    outputFilenameTROPOMI = (
                        os.getenv("MUSES_DEFAULT_RUN_DIR", "")
                        + os.path.sep
                        + outputFilenameTROPOMI
                    )

                # EM - Depending on the type of retrieval for TROPOMI, the number of filenames will vary, therefore
                # all filenames are fed in as a list, comma seperated. This is split in 'read_tropomi'.
                filename = []
                XTrack = []
                for ii in (
                    windows
                ):  # There are 8 TROPOMI bands, check which windows are invoked
                    if (
                        ii["instrument"] == "TROPOMI"
                    ):  # EM Need this check for duel band purposes
                        if (
                            file_id["preferences"][f"TROPOMI_filename_{ii['filter']}"]
                            is None
                        ):
                            logger.info(
                                "ERROR: TROPOMI L1B file for BAND not found",
                                ii["filter"],
                            )
                            assert False
                        else:
                            filename.append(
                                file_id["preferences"][
                                    f"TROPOMI_filename_{ii['filter']}"
                                ]
                            )
                            XTrack.append(
                                file_id["preferences"][
                                    f"TROPOMI_XTrack_Index_{ii['filter']}"
                                ]
                            )

                irrFilename = file_id["preferences"]["TROPOMI_IRR_filename"]
                cldFilename = file_id["preferences"]["TROPOMI_Cloud_filename"]

                # if (os.getenv('MUSES_DEFAULT_RUN_DIR', '') != '') and (not os.path.isabs(filename)):
                #    filename = os.getenv('MUSES_DEFAULT_RUN_DIR', '') + os.path.sep + filename

                if not os.path.isfile(
                    outputFilenameTROPOMI
                ):  # THIS SHOULD BE IF NOT, REMOVED NOT SO PICKLE FILE ISN'T READ FOR TIME BEING
                    TROPOMI_RAD_CALRUN_FLAG = 0
                    if "TROPOMI_Rad_calRun_flag" in file_id["preferences"]:
                        TROPOMI_RAD_CALRUN_FLAG = int(
                            file_id["preferences"]["TROPOMI_Rad_calRun_flag"]
                        )

                    if TROPOMI_RAD_CALRUN_FLAG == 0:
                        o_tropomi = mpy.read_tropomi(
                            filename,
                            irrFilename,
                            cldFilename,
                            XTrack,  # EM - Like filename, xtrack will be variable, so fed in as list
                            int(file_id["preferences"]["TROPOMI_ATrack_Index"]),
                            file_id["preferences"]["TROPOMI_utcTime"],
                            windows,
                            calibrationFilename,  # EM - Calibration to be implemented
                        )
                    else:
                        o_tropomi = mpy.read_tropomi(
                            filename,
                            irrFilename,
                            cldFilename,
                            XTrack,  # EM - Like filename, xtrack will be variable, so fed in as list
                            int(file_id["preferences"]["TROPOMI_ATrack_Index"]),
                            file_id["preferences"]["TROPOMI_utcTime"],
                            windows,
                        )
                    # end if OMI_RAD_CALRUN_FLAG == 0:

                    # If something goes wrong with read_omi() function, we return immediately.
                    if o_tropomi is None:
                        return (
                            o_airs,
                            o_cris,
                            o_omi,
                            o_tropomi,
                            o_tes,
                            o_oco2,
                            o_stateInfo,
                        )  # Return due to error.

                    if i_writeOutput:
                        # Create directory if it doesn't already exist then write PICKLE file.
                        output_dir = os.path.dirname(outputFilenameTROPOMI)
                        utilDir.make_dir(output_dir)

                        logger.info("PICKLE_ME", outputFilenameTROPOMI)

                        with open(outputFilenameTROPOMI, "wb") as pickle_handle:
                            pickle.dump(
                                o_tropomi,
                                pickle_handle,
                                protocol=pickle.HIGHEST_PROTOCOL,
                            )
                else:
                    logger.info("UNPICKLE_ME", outputFilenameTROPOMI)

                    with open(outputFilenameTROPOMI, "rb") as pickle_handle:
                        o_tropomi = pickle.load(pickle_handle)
                # end else portion of if not (os.path.isfile(outputFilenameTROPOMI)):
            # end: if 'OMPS-NM' in instruments:

            ######### OMPS-NM

            ################   TROPOMI
            if "TROPOMI" in instruments:
                # EM NOTE Retrievals must be set as 'fullfilter' if retrieving O3 or any other gas from multiple bands seperately
                # This invokes the defaults spectral windows, which are specified in the OSP/Strategy_Tables/Defaults,
                # with the TROPOMI bands defined in the *_CrIS_TROPOMI.asc and the *_TROPOMI.asc files.

                # TODO: Here we need soft calibration for UV1 (BAND1), UV2 (BAND2) and UVIS (BAND3) for O3 retrieval, unclear if necessary for other bands.
                # calibrationFilename = table_get_pref(i_table_struct,  'tropomi_calibrationFilename')

                outputFilenameTROPOMI = "./Input/Radiance_TROPOMI_" + my_key + ".pkl"

                # EM - Depending on the type of retrieval for TROPOMI, the number of filenames will vary, therefore
                # all filenames are fed in as a list, comma seperated. This is split in 'read_tropomi'.
                filename, XTrack, ATrack = mpy.get_tropomi_measurement_id_info(
                    file_id, windows
                )
                tropomi_albedo_from_dler_db = (
                    file_id["preferences"].get("TROPOMI_Initial_Albedo_DLER", "0")
                    == "1"
                )

                if not os.path.isfile(
                    outputFilenameTROPOMI
                ):  # THIS SHOULD BE IF NOT, REMOVED NOT SO PICKLE FILE ISN'T READ FOR TIME BEING
                    TROPOMI_RAD_CALRUN_FLAG = 0
                    if "TROPOMI_Rad_calRun_flag" in file_id["preferences"]:
                        TROPOMI_RAD_CALRUN_FLAG = int(
                            file_id["preferences"]["TROPOMI_Rad_calRun_flag"]
                        )

                    if TROPOMI_RAD_CALRUN_FLAG == 0:
                        o_tropomi = mpy.read_tropomi(
                            filename,
                            XTrack,
                            ATrack,
                            file_id["preferences"]["TROPOMI_utcTime"],
                            windows,
                            albedo_from_dler=tropomi_albedo_from_dler_db,
                        )

                        # EM - Calibration to be implemented
                        # o_tropomi = read_tropomi(
                        #     filename,
                        #     irrFilename,
                        #     cldFilename,
                        #     XTrack,  # EM - Like filename, xtrack will be variable, so fed in as list
                        #     int(file_id['preferences']['TROPOMI_ATrack_Index']),
                        #     file_id['preferences']['TROPOMI_utcTime'],
                        #     windows,
                        #     calibrationFilename
                        # )
                    else:
                        o_tropomi = mpy.read_tropomi(
                            filename,
                            XTrack,
                            ATrack,
                            file_id["preferences"]["TROPOMI_utcTime"],
                            windows,
                            albedo_from_dler=tropomi_albedo_from_dler_db,
                        )
                    # end if OMI_RAD_CALRUN_FLAG == 0:

                    # If something goes wrong with read_omi() function, we return immediately.
                    if o_tropomi is None:
                        return (
                            o_airs,
                            o_cris,
                            o_omi,
                            o_tropomi,
                            o_tes,
                            o_oco2,
                            o_stateInfo,
                        )  # Return due to error.

                    if i_writeOutput:
                        # Create directory if it doesn't already exist then write PICKLE file.
                        output_dir = os.path.dirname(outputFilenameTROPOMI)
                        utilDir.make_dir(output_dir)

                        logger.info("PICKLE_ME", outputFilenameTROPOMI)

                        with open(outputFilenameTROPOMI, "wb") as pickle_handle:
                            pickle.dump(
                                o_tropomi,
                                pickle_handle,
                                protocol=pickle.HIGHEST_PROTOCOL,
                            )
                else:
                    logger.info("UNPICKLE_ME", outputFilenameTROPOMI)

                    with open(outputFilenameTROPOMI, "rb") as pickle_handle:
                        o_tropomi = pickle.load(pickle_handle)
                # end else portion of if not (os.path.isfile(outputFilenameTROPOMI)):
            # end: if 'TROPOMI' in instruments:

            if "OCO2" in instruments:
                outputFilenameOCO2 = "./Input/Radiance_OCO2_" + my_key + ".nc"

                if not os.path.isfile(outputFilenameOCO2):
                    filename = file_id["preferences"]["OCO2_filename"]
                    radianceSource = mpy.table_get_pref(
                        i_table_struct, "radianceSource"
                    )
                    o_oco2 = mpy.oco2_read_radiance(
                        file_id["preferences"], radianceSource
                    )

                    if i_writeOutput:
                        # TODO: Pickle radiance data
                        # write_radiance_cris(o_cris)
                        pass
                else:
                    # TODO: Unpickle radiance data
                    # read_radiance_cris(o_cris)

                    logger.info("ERROR: Not implemented", outputFilenameCRIS)
                    assert False
            # end: if 'OCO2' in instruments:

            # AT_LINE 411 src_ms-2019-05-29/script_retrieval_setup_ms.pro
            ##### end of radiance section ####

            # AT_LINE 441 src_ms-2019-05-29/script_retrieval_setup_ms.pro

            # get surface altitude IN METERS
            if "TES" in instruments:
                surfaceAltitude = o_tes["surfaceElevation"]
            elif "AIRS" in instruments:
                surfaceAltitude = o_airs["surfaceAltitude"]
            elif "CRIS" in instruments:
                # A note about case of keys in o_cris.  All cases are upper to be less confusing.
                surfaceAltitude = o_cris["surfaceAltitude".upper()]
                if surfaceAltitude < 0.0:
                    surfaceAltitude = 0
            elif "OMI" in instruments:
                surfaceAltitude = o_omi["Earth_Radiance"]["ObservationTable"][
                    "TerrainHeight"
                ][0]
            elif "TROPOMI" in instruments:
                for i in range(
                    0, len(o_tropomi["Earth_Radiance"]["ObservationTable"]["ATRACK"])
                ):
                    surfaceAltitude = mpy.read_tropomi_surface_altitude(
                        o_tropomi["Earth_Radiance"]["ObservationTable"]["Latitude"][i],
                        o_tropomi["Earth_Radiance"]["ObservationTable"]["Longitude"][i],
                    )
                    o_tropomi["Earth_Radiance"]["ObservationTable"]["TerrainHeight"][
                        i
                    ] = surfaceAltitude
            elif "OCO2" in instruments:
                surfaceAltitude = np.mean(o_oco2["surface_altitude_m"])
            else:
                logger.info("instruments ~= TES, AIRS, OMI, CRIS, TROPOMI, OCO2")
                logger.info("Need surface altitude.")
                assert False
            # end else part of if 'TES' in instruments:

            # AT_LINE 460 src_ms-2019-05-29/script_retrieval_setup_ms.pro
            # initial guess
            ospDirectory = "../OSP"
            setupFilename = (
                mpy.table_get_pref(i_table_struct, "initialGuessSetupDirectory")
                + "/L2_Setup_Control_Initial.asc"
            )

            constraintFlag = False
            stateInitial = mpy.get_state_initial(
                setupFilename,
                ospDirectory,
                o_latitude,
                o_longitude,
                o_dateStruct,
                file_id["preferences"],
                constraintFlag,
                surfaceAltitude,
                oceanFlag,
                gmao_path,
                gmao_type,
            )

            # AT_LINE 479 src_ms-2019-05-29/script_retrieval_setup_ms.pro
            # constraint vector
            ospDirectory = "../OSP"
            setupFilename = (
                mpy.table_get_pref(i_table_struct, "initialGuessSetupDirectory")
                + "/L2_Setup_Control_Constraint.asc"
            )

            # state for constraint
            constraintFlag = True
            stateConstraint = mpy.get_state_initial(
                setupFilename,
                ospDirectory,
                o_latitude,
                o_longitude,
                o_dateStruct,
                file_id["preferences"],
                constraintFlag,
                surfaceAltitude,
                oceanFlag,
                gmao_path,
                gmao_type,
            )

            # AT_LINE 489 src_ms-2019-05-29/script_retrieval_setup_ms.pro
            # get tes, omi, airs pars...
            # tes: boresight angle.  OMI ring, cloud, albedo.
            # airs: angle

            # AT_LINE 492 src_ms-2019-05-29/script_retrieval_setup_ms.pro
            if "TES" in instruments:
                # note:  o_tes should contain all the fields in stateInitial['current']['tes']
                # boresightNadirRadians, orbitInclinationAngle, viewMode, instrumentAzimuth, instrumentLatitude, geoPointing, targetRadius, instrumentRadius, orbitAscending
                (stateInitial, stateConstraint) = self._clean_up_dictionaries(
                    instrument_name.lower(), stateInitial, stateConstraint, o_tes
                )

            # AT_LINE 518 src_ms-2019-05-29/script_retrieval_setup_ms.pro
            if "CRIS" in instruments:
                o_cris["l1bType"] = cris_type
                (stateInitial, stateConstraint) = self._clean_up_dictionaries(
                    instrument_name.lower(), stateInitial, stateConstraint, o_cris
                )

            # AT_LINE 533 src_ms-2019-05-29/script_retrieval_setup_ms.pro
            if "AIRS" in instruments:
                (stateInitial, stateConstraint) = self._clean_up_dictionaries(
                    instrument_name.lower(), stateInitial, stateConstraint, o_airs
                )

            # AT_LINE 548 script_retrieval_setup_ms.pro script_retrieval_setup_ms
            if "OMI" in instruments:
                # start parameters
                temp_end = 2  # Dejian revised Nov 22, 2016 for adding the functionality of radiance calibration run
                if int(file_id["preferences"]["OMI_Rad_calRun_flag"]) == 1:
                    temp_end = 1

                # PYTHON_NOTE: The keys in omi_pars should closely match the keys of omi dictionary in new_state_structures.py
                omi_pars = {
                    "surface_albedo_uv1": o_omi["SurfaceAlbedo"][
                        "MonthlyMinimumSurfaceReflectance"
                    ],
                    "surface_albedo_uv2": o_omi["SurfaceAlbedo"][
                        "MonthlyMinimumSurfaceReflectance"
                    ],
                    "surface_albedo_slope_uv2": np.float64(0.0),
                    "nradwav_uv1": np.float64(0.0),
                    "nradwav_uv2": np.float64(0.0),
                    "odwav_uv1": np.float64(0.0),
                    "odwav_uv2": np.float64(0.0),
                    "odwav_slope_uv1": np.float64(0.0),
                    "odwav_slope_uv2": np.float64(0.0),
                    "ring_sf_uv1": np.float64(1.9),
                    "ring_sf_uv2": np.float64(1.9),
                    "cloud_fraction": o_omi["Cloud"]["CloudFraction"],
                    "cloud_pressure": o_omi["Cloud"]["CloudPressure"],
                    "cloud_Surface_Albedo": 0.8,  # Same key in 'omi' dict as in new_state_structures.py
                    "xsecscaling": np.float64(1.0),
                    "resscale_uv1": np.float64(0.0) - 999,
                    "resscale_uv2": np.float64(0.0) - 999,
                    "SPACECRAFTALTITUDE": np.mean(
                        o_omi["Earth_Radiance"]["ObservationTable"][
                            "SpacecraftAltitude"
                        ]
                    ),  # Same key in 'omi' dict as in new_state_structures.py
                    "sza_uv1": o_omi["Earth_Radiance"]["ObservationTable"][
                        "SolarZenithAngle"
                    ][0],
                    "raz_uv1": o_omi["Earth_Radiance"]["ObservationTable"][
                        "RelativeAzimuthAngle"
                    ][0],
                    "vza_uv1": o_omi["Earth_Radiance"]["ObservationTable"][
                        "ViewingZenithAngle"
                    ][0],
                    "sca_uv1": o_omi["Earth_Radiance"]["ObservationTable"][
                        "ScatteringAngle"
                    ][0],
                    "sza_uv2": np.mean(
                        o_omi["Earth_Radiance"]["ObservationTable"]["SolarZenithAngle"][
                            1 : temp_end + 1
                        ]
                    ),
                    "raz_uv2": np.mean(
                        o_omi["Earth_Radiance"]["ObservationTable"][
                            "RelativeAzimuthAngle"
                        ][1 : temp_end + 1]
                    ),
                    "vza_uv2": np.mean(
                        o_omi["Earth_Radiance"]["ObservationTable"][
                            "ViewingZenithAngle"
                        ][1 : temp_end + 1]
                    ),
                    "sca_uv2": np.mean(
                        o_omi["Earth_Radiance"]["ObservationTable"]["ScatteringAngle"][
                            1 : temp_end + 1
                        ]
                    ),
                }

                ttState = []
                if stateConstraint is not None:
                    ttState = list(stateConstraint["current"]["omi"].keys())

                if stateInitial is not None:
                    ttState = list(stateInitial["current"]["omi"].keys())

                for ii in range(0, len(ttState)):
                    if stateConstraint is not None:
                        stateConstraint["current"]["omi"][ttState[ii]] = omi_pars[
                            ttState[ii]
                        ]

                    if stateInitial is not None:
                        stateInitial["current"]["omi"][ttState[ii]] = omi_pars[
                            ttState[ii]
                        ]
                # end for ii in range(0, len(ttState)):
            # end if 'OMI' in instruments:

            if "TROPOMI" in instruments:
                # start parameters, for TROPOMI, declaring them, and then fill them depending on how many
                # bands are used. This is based on the OMI setup, but assuming variable number of bands
                tropomi_pars = {
                    "surface_albedo_BAND1": np.float64(0.0),
                    "surface_albedo_BAND2": np.float64(0.0),
                    "surface_albedo_BAND3": np.float64(0.0),
                    "surface_albedo_BAND7": np.float64(0.0),
                    "surface_albedo_slope_BAND1": np.float64(0.0),
                    "surface_albedo_slope_BAND2": np.float64(0.0),
                    "surface_albedo_slope_BAND3": np.float64(0.0),
                    "surface_albedo_slope_BAND7": np.float64(0.0),
                    "surface_albedo_slope_order2_BAND2": np.float64(0.0),
                    "surface_albedo_slope_order2_BAND3": np.float64(0.0),
                    "surface_albedo_slope_order2_BAND7": np.float64(0.0),
                    "solarshift_BAND1": np.float64(0.0),
                    "solarshift_BAND2": np.float64(0.0),
                    "solarshift_BAND3": np.float64(0.0),
                    "solarshift_BAND7": np.float64(0.0),
                    "radianceshift_BAND1": np.float64(0.0),
                    "radianceshift_BAND2": np.float64(0.0),
                    "radianceshift_BAND3": np.float64(0.0),
                    "radianceshift_BAND7": np.float64(0.0),
                    "radsqueeze_BAND1": np.float64(0.0),
                    "radsqueeze_BAND2": np.float64(0.0),
                    "radsqueeze_BAND3": np.float64(0.0),
                    "radsqueeze_BAND7": np.float64(0.0),
                    "ring_sf_BAND1": np.float64(0.0),
                    "ring_sf_BAND2": np.float64(0.0),
                    "ring_sf_BAND3": np.float64(0.0),
                    "ring_sf_BAND7": np.float64(0.0),
                    "temp_shift_BAND3": np.float64(0.0),
                    "temp_shift_BAND7": np.float64(0.0),
                    "cloud_fraction": np.float64(0.0),
                    "cloud_pressure": np.float64(0.0),
                    "cloud_Surface_Albedo": np.float64(0.0),
                    "xsecscaling": np.float64(0.0),
                    "resscale_O0_BAND1": np.float64(0.0),
                    "resscale_O1_BAND1": np.float64(0.0),
                    "resscale_O2_BAND1": np.float64(0.0),
                    "resscale_O0_BAND2": np.float64(0.0),
                    "resscale_O1_BAND2": np.float64(0.0),
                    "resscale_O2_BAND2": np.float64(0.0),
                    "resscale_O0_BAND3": np.float64(0.0),
                    "resscale_O1_BAND3": np.float64(0.0),
                    "resscale_O2_BAND3": np.float64(0.0),
                    "resscale_O0_BAND7": np.float64(0.0),
                    "resscale_O1_BAND7": np.float64(0.0),
                    "resscale_O2_BAND7": np.float64(0.0),
                    "sza_BAND1": np.float64(0.0),
                    "raz_BAND1": np.float64(0.0),
                    "vza_BAND1": np.float64(0.0),
                    "sca_BAND1": np.float64(0.0),
                    "sza_BAND2": np.float64(0.0),
                    "raz_BAND2": np.float64(0.0),
                    "vza_BAND2": np.float64(0.0),
                    "sca_BAND2": np.float64(0.0),
                    "sza_BAND3": np.float64(0.0),
                    "raz_BAND3": np.float64(0.0),
                    "vza_BAND3": np.float64(0.0),
                    "sca_BAND3": np.float64(0.0),
                    "sza_BAND7": np.float64(0.0),
                    "raz_BAND7": np.float64(0.0),
                    "vza_BAND7": np.float64(0.0),
                    "sca_BAND7": np.float64(0.0),
                    "SPACECRAFTALTITUDE": np.float64(0.0),
                }

                # PYTHON_NOTE: The keys in tropomi_pars should closely match the keys of tropomi dictionary in new_state_structures.py
                # These parameters are not band specific
                if o_tropomi["Cloud"]["CloudFraction"] == 0.0:
                    tropomi_pars["cloud_fraction"] = (
                        0.01  # EM NOTE - So we can fit Cloud albedo, due to calibration errors.
                    )
                else:
                    tropomi_pars["cloud_fraction"] = o_tropomi["Cloud"]["CloudFraction"]
                tropomi_pars["cloud_pressure"] = o_tropomi["Cloud"]["CloudPressure"]
                tropomi_pars["cloud_Surface_Albedo"] = (
                    0.8  # (o_tropomi['Cloud']['CloudAlbedo']) # Same key in 'tropomi' dict as in new_state_structures.py
                )
                tropomi_pars["SPACECRAFTALTITUDE"] = np.mean(
                    o_tropomi["Earth_Radiance"]["ObservationTable"][
                        "SpacecraftAltitude"
                    ]
                )
                tropomi_pars["xsecscaling"] = np.float64(1.0)

                current_band = []
                ii_tropomi = -1
                for ii, band in enumerate(windows):  # 8 bands in TROPOMI
                    # Assuming that values are appended to o_tropomi from UV to higher wavelengths
                    if (
                        band["instrument"] == "TROPOMI"
                    ):  # EM - Necessary for dual band retrievals
                        ii_tropomi += 1
                        if current_band != band["filter"]:
                            current_band = band["filter"]
                            tropomi_pars[f"surface_albedo_{band['filter']}"] = (
                                o_tropomi["SurfaceAlbedo"][
                                    f"{band['filter']}_MonthlyMinimumSurfaceReflectance"
                                ]
                            )
                            tropomi_pars[f"surface_albedo_slope_{band['filter']}"] = (
                                np.float64(0.0)
                            )
                            tropomi_pars[
                                f"surface_albedo_slope_order2_{band['filter']}"
                            ] = np.float64(0.0)
                            tropomi_pars[f"solarshift_{band['filter']}"] = np.float64(
                                0.0
                            )
                            tropomi_pars[f"radianceshift_{band['filter']}"] = (
                                np.float64(0.0)
                            )
                            tropomi_pars[f"radsqueeze_{band['filter']}"] = np.float64(
                                0.0
                            )
                            tropomi_pars[f"temp_shift_{band['filter']}"] = np.float64(
                                1.0
                            )
                            tropomi_pars[f"ring_sf_{band['filter']}"] = np.float64(1.9)
                            tropomi_pars[f"resscale_O0_{band['filter']}"] = np.float64(
                                1.0
                            )
                            tropomi_pars[f"resscale_O1_{band['filter']}"] = np.float64(
                                0.0
                            )
                            tropomi_pars[f"resscale_O2_{band['filter']}"] = np.float64(
                                0.0
                            )
                            tropomi_pars[f"sza_{band['filter']}"] = o_tropomi[
                                "Earth_Radiance"
                            ]["ObservationTable"]["SolarZenithAngle"][ii_tropomi]
                            tropomi_pars[f"raz_{band['filter']}"] = o_tropomi[
                                "Earth_Radiance"
                            ]["ObservationTable"]["RelativeAzimuthAngle"][ii_tropomi]
                            tropomi_pars[f"vza_{band['filter']}"] = o_tropomi[
                                "Earth_Radiance"
                            ]["ObservationTable"]["ViewingZenithAngle"][ii_tropomi]
                            tropomi_pars[f"sca_{band['filter']}"] = o_tropomi[
                                "Earth_Radiance"
                            ]["ObservationTable"]["ScatteringAngle"][ii_tropomi]
                        else:
                            current_band = band["filter"]
                            continue
                    else:
                        continue

                ttState = []
                if stateConstraint is not None:
                    ttState = list(stateConstraint["current"]["tropomi"].keys())

                if stateInitial is not None:
                    ttState = list(stateInitial["current"]["tropomi"].keys())

                for ii in range(0, len(ttState)):
                    if stateConstraint is not None:
                        stateConstraint["current"]["tropomi"][ttState[ii]] = (
                            tropomi_pars[ttState[ii]]
                        )

                    if stateInitial is not None:
                        stateInitial["current"]["tropomi"][ttState[ii]] = tropomi_pars[
                            ttState[ii]
                        ]
                # end for ii in range(0, len(ttState)):
            # end if 'TROPOMI' in instruments:

            # OCO-2
            if "OCO2" in instruments:
                ttState = []
                if stateConstraint is not None:
                    ttState = list(stateConstraint["current"]["oco2"].keys())

                if stateInitial is not None:
                    ttState = list(stateInitial["current"]["oco2"].keys())

                for ii in range(0, len(ttState)):
                    if stateConstraint is not None:
                        stateConstraint["current"]["oco2"][ttState[ii]] = o_oco2[
                            ttState[ii]
                        ]

                    if stateInitial is not None:
                        stateInitial["current"]["oco2"][ttState[ii]] = o_oco2[
                            ttState[ii]
                        ]

                # PRINT, 'Get OCO pars'
                # ; get parameters specific to nir/3/...
                # nir_pars = {footprint:fileid.oco2_footprint} # not sure what need yet

                # copy angles/geolocation info to structures
                # IF N_ELEMENTS(stateConstraint) GT 0 THEN ttState = tag_names(stateConstraint.current.nir)
                # IF N_ELEMENTS(stateInitial) GT 0 THEN ttState = tag_names(stateInitial.current.nir)
                # ttnir = tag_names(nir_pars)
                # FOR ii = 0, N_ELEMENTS(ttState)-1 DO BEGIN
                #     indnir = where(ttnir EQ ttState[ii])
                #     IF N_ELEMENTS(stateConstraint) GT 0 and indnir[0] GE 0 THEN stateConstraint.current.nir.(ii) = nir_pars.(indnir)
                #     IF N_ELEMENTS(stateInitial) GT 0 and indnir[0] GE 0 THEN stateInitial.current.nir.(ii) = nir_pars.(indnir)
                # ENDFOR

                # set table.pressurefm to stateConstraint.pressure because OCO-2 is on sigma levels
                i_table_struct["pressureFM"] = stateInitial["current"]["pressure"]

                # stateConstraint['current']['cloudPars']['use'] = 'no'
                # stateInitial['current']['cloudPars']['use'] = 'no'
            # ENDIF

            # AT_LINE 593 src_ms-2019-05-29/script_retrieval_setup_ms.pro script_retrieval_setup_ms
            if len(stateInitial) > 0:
                # set surface type for products
                oceanString = ["LAND", "OCEAN"]
                stateInitial["surfaceType"] = oceanString[
                    oceanFlag
                ]  # The type of oceanFlag should be int here so we can use it as an index.
                stateInitial["current"]["surfaceType"] = oceanString[oceanFlag]

                if i_writeOutput:
                    # Because the write_state function modify the 'current' fields of stateInitial structure, we give it a copy.
                    stateCurrentCopy = copy.deepcopy(stateInitial["current"])
                    mpy.write_state(
                        directoryIG,
                        mpy.ObjectView(stateInitial),
                        mpy.ObjectView(stateCurrentCopy),
                        my_key="_" + my_key,
                        writeAltitudes=0,
                    )
                    del stateCurrentCopy  # Delete the temporary object.

            # AT_LINE 577 script_retrieval_setup_ms.pro script_retrieval_setup_ms
            # AT_LINE 610 src_ms-2019-05-29/script_retrieval_setup_ms.pro script_retrieval_setup_ms
            if len(stateConstraint) > 0:
                #  set surface type for products
                oceanString = ["LAND", "OCEAN"]
                stateConstraint["surfaceType"] = oceanString[oceanFlag]
                stateConstraint["current"]["surfaceType"] = oceanString[oceanFlag]

                if i_writeOutput:
                    # Because the write_state function modify the 'current' fields of stateConstraint structure, we give it a copy.
                    stateCurrentCopy = copy.deepcopy(stateConstraint["current"])
                    mpy.write_state(
                        directoryConstraint,
                        mpy.ObjectView(stateConstraint),
                        mpy.ObjectView(stateCurrentCopy),
                        my_key="_" + my_key,
                        writeAltitudes=0,
                    )
                    del stateCurrentCopy

            # AT_LINE 595 script_retrieval_setup_ms.pro script_retrieval_setup_ms
            # AT_LINE 628 src_ms-2019-05-29/script_retrieval_setup_ms.pro script_retrieval_setup_ms
            # set up state

            o_stateInfo = stateInitial

            # types are from a priori not initial.  ch3ohtype is needed for constraint selection
            o_stateInfo["ch3ohtype"] = stateConstraint["ch3ohtype"]

            # get type from

            #  set surface type for products
            oceanString = ["LAND", "OCEAN"]
            o_stateInfo["surfaceType"] = oceanString[oceanFlag]

            # Make a deepcopy of stateInitial['current'] and stateConstraint['current'] to o_stateInfo so each will have its own memory.

            o_stateInfo["initialInitial"] = copy.deepcopy(stateInitial["current"])
            o_stateInfo["initial"] = copy.deepcopy(stateInitial["current"])
            o_stateInfo["current"] = copy.deepcopy(stateInitial["current"])
            o_stateInfo["constraint"] = copy.deepcopy(stateConstraint["current"])

        # o_airs['solazi'] = 52

        # At this point, we have 5 separate fields in o_stateInfo with 5 separate memory locations.
        return (
            o_airs,
            o_cris,
            o_omi,
            o_tropomi,
            o_tes,
            o_oco2,
            o_stateInfo,
        )  # More instrument data later.

    def _clean_up_dictionaries(
        self, i_instrument_name, stateInitial, stateConstraint, instrumentInfo
    ):
        ttState = []
        if len(stateConstraint) > 0:
            ttState = stateConstraint["current"][i_instrument_name].keys()

        if len(stateInitial) > 0:
            ttState = stateInitial["current"][i_instrument_name].keys()

        tt_instrument = instrumentInfo.keys()

        # Keep track of various dictionaries and list so we can try to remedy if a key was missed.
        list_of_valid_state_constraint_keys = list(
            stateConstraint["current"][i_instrument_name].keys()
        )
        list_of_valid_state_initial_keys = list(
            stateInitial["current"][i_instrument_name].keys()
        )
        dict_of_state_set_with_normal_key = {}
        dict_of_state_set_with_upper_key = {}
        list_of_dict_keys_uppercased = []

        if len(ttState) > 0:
            for ii, dict_key in enumerate(tt_instrument):
                indState = dict_key  # PYTHON_NOTE: The dict_key is the name of the key.  We don't need to know the index since dictionary just need the key.
                indInstrument = dict_key  # PYTHON_NOTE: The dict_key is the name of the key.  We don't need to know the index since dictionary just need the key.
                if len(stateConstraint) > 0:
                    stateConstraint["current"][i_instrument_name][indState] = (
                        instrumentInfo[indInstrument]
                    )

                if len(stateInitial) > 0:
                    stateInitial["current"][i_instrument_name][indState] = (
                        instrumentInfo[indInstrument]
                    )
                # Keep track of what we had set so we can inspect it later.
                dict_of_state_set_with_normal_key[dict_key] = instrumentInfo[dict_key]
                dict_of_state_set_with_upper_key[dict_key.upper()] = instrumentInfo[
                    dict_key
                ]
                list_of_dict_keys_uppercased.append(dict_key.upper())

        # Do a sanity check on the numbers of attributes set above.
        # Due the way a dictionary key is set, it is not required to be in any case so sometimes the case is different which can cause a problem.

        # Clean up stateInitial dictionary by reconciling any attributes that were not set.  Perhaps it was set with a different case.
        # This is the list of optional valid keys.  If a key was set but it does not belong to this list, we can remove it.
        optional_valid_keys = [
            "latitude",
            "longitude",
            "time",
            "radiance",
            "DaytimeFlag",
            "CalChanSummary",
            "ExcludedChans",
            "NESR",
            "frequency",
            "surfaceAltitude",
            "state",
            "valid",
        ]

        #
        # Clean up stateInitial dictionary.
        #
        keys_to_delete = []
        for ii, dict_key in enumerate(stateInitial["current"][i_instrument_name]):
            if dict_key not in dict_of_state_set_with_normal_key:
                stateInitial["current"][i_instrument_name][dict_key] = (
                    dict_of_state_set_with_upper_key[dict_key.upper()]
                )
            else:
                pass
                if (
                    dict_key not in list_of_valid_state_initial_keys
                    and dict_key not in optional_valid_keys
                ):
                    keys_to_delete.append(dict_key)

        # Delete any keys if there are keys to delete.
        if len(keys_to_delete) > 0:
            for ii, key_to_delete in enumerate(keys_to_delete):
                stateInitial["current"][i_instrument_name].pop(key_to_delete, None)

        #
        # Clean up stateConstraint dictionary.
        #
        keys_to_delete = []
        for ii, dict_key in enumerate(stateConstraint["current"][i_instrument_name]):
            if dict_key not in dict_of_state_set_with_normal_key:
                stateConstraint["current"][i_instrument_name][dict_key] = (
                    dict_of_state_set_with_upper_key[dict_key.upper()]
                )
            else:
                pass
                if (
                    dict_key not in list_of_valid_state_constraint_keys
                    and dict_key not in optional_valid_keys
                ):
                    keys_to_delete.append(dict_key)

        # Delete any keys if there are keys to delete.
        if len(keys_to_delete) > 0:
            for ii, key_to_delete in enumerate(keys_to_delete):
                stateConstraint["current"][i_instrument_name].pop(key_to_delete, None)

        return (stateInitial, stateConstraint)

    @property
    def state_info_obj(self):
        return mpy.ObjectView(self.state_info_dict)

    def copy_current_initialInitial(self):
        self.state_info_dict["initialInitial"] = copy.deepcopy(
            self.state_info_dict["current"]
        )
        # The simplest thing would be to just copy the current dict. However,
        # the muses-py StateElement maintain their state outside of the classes in
        # various dicts in StateInfo (probably left over from IDL). So we have
        # this function. For ReFRACtor StateElement, this should just be a copy of
        # StateElement, but for muses-py we return None

        for k, selem in self.current.items():
            scopy = selem.clone_for_other_state()
            if scopy is not None:
                self.initialInitial[k] = scopy

    def copy_current_initial(self):
        self.state_info_dict["initial"] = copy.deepcopy(self.state_info_dict["current"])
        # The simplest thing would be to just copy the current dict. However,
        # the muses-py StateElement maintain their state outside of the classes in
        # various dicts in StateInfo (probably left over from IDL). So we have
        # this function. For ReFRACtor StateElement, this should just be a copy of
        # StateElement, but for muses-py we return None

        for k, selem in self.current.items():
            scopy = selem.clone_for_other_state()
            if scopy is not None:
                self.initial[k] = scopy

    def next_state_to_current(self):
        # muses-py StateElement maintains state outside of the classes, so
        # copy this dictionary
        if self.next_state_dict is not None:
            self.state_info_dict["current"] = self.next_state_dict
        self.next_state_dict = None
        # For ReFRACtor StateElement, we have already updated current. But
        # if the request was not to pass this on to the next step, we set this
        # aside in self.next_state. Go ahead and put that into place
        self.current.update(self.next_state)
        self.next_state = {}

    def sounding_metadata(self, step="current"):
        return SoundingMetadata(self, step=step)

    def has_true_values(self):
        """Indicate if we have true values in our state info."""
        return np.max(self.state_info_dict["true"]["values"]) > 0

    def gmao_tropopause_pressure(self):
        # Not clear how to handle incidental things like this. For now,
        # just make a clear function so we know we need some way of handling this.
        return self.state_info_dict["gmaoTropopausePressure"]

    @property
    def state_element_on_levels(self):
        return self.state_info_dict["species"]

    @property
    def nh3type(self):
        return self.state_info_dict["nh3type"]

    @property
    def hcoohtype(self):
        return self.state_info_dict["hcoohtype"]

    @property
    def ch3ohtype(self):
        return self.state_info_dict["ch3ohtype"]

    @property
    def pressure(self):
        # The pressure is kind of like a StateElementOnLevels, but it is a bit of a
        # special case. This is needed to interpret the rest of the data.
        return self.state_info_dict["current"]["pressure"]

    def omi_params(self, o_omi):
        """omi_params is used in creating the uip. This is a mix of parameters from the o_omi object
        returned by mpy.read_omi and various state elements. Put this together, so we have this
        for the UIP. Note we are trying to move away from the UIP, it is just a shuffling around
        of data found in the StateInfo. But for now go ahead and support this."""
        return {
            "surface_albedo_uv1": self.state_element("OMISURFACEALBEDOUV1").value[0],
            "surface_albedo_uv2": self.state_element("OMISURFACEALBEDOUV2").value[0],
            "surface_albedo_slope_uv2": self.state_element(
                "OMISURFACEALBEDOSLOPEUV2"
            ).value[0],
            "nradwav_uv1": self.state_element("OMINRADWAVUV1").value[0],
            "nradwav_uv2": self.state_element("OMINRADWAVUV2").value[0],
            "odwav_uv1": self.state_element("OMIODWAVUV1").value[0],
            "odwav_uv2": self.state_element("OMIODWAVUV2").value[0],
            "odwav_slope_uv1": self.state_element("OMIODWAVSLOPEUV1").value[0],
            "odwav_slope_uv2": self.state_element("OMIODWAVSLOPEUV2").value[0],
            "ring_sf_uv1": self.state_element("OMIRINGSFUV1").value[0],
            "ring_sf_uv2": self.state_element("OMIRINGSFUV2").value[0],
            "cloud_fraction": self.state_element("OMICLOUDFRACTION").value[0],
            "cloud_pressure": o_omi["Cloud"]["CloudPressure"],
            "cloud_Surface_Albedo": 0.8,
            "xsecscaling": 1.0,
            "resscale_uv1": -999.0,
            "resscale_uv2": -999.0,
            "SPACECRAFTALTITUDE": np.mean(
                o_omi["Earth_Radiance"]["ObservationTable"]["SpacecraftAltitude"]
            ),
            "sza_uv1": o_omi["Earth_Radiance"]["ObservationTable"]["SolarZenithAngle"][
                0
            ],
            "raz_uv1": o_omi["Earth_Radiance"]["ObservationTable"][
                "RelativeAzimuthAngle"
            ][0],
            "vza_uv1": o_omi["Earth_Radiance"]["ObservationTable"][
                "ViewingZenithAngle"
            ][0],
            "sca_uv1": o_omi["Earth_Radiance"]["ObservationTable"]["ScatteringAngle"][
                0
            ],
            "sza_uv2": np.mean(
                o_omi["Earth_Radiance"]["ObservationTable"]["SolarZenithAngle"][1:3]
            ),
            "raz_uv2": np.mean(
                o_omi["Earth_Radiance"]["ObservationTable"]["RelativeAzimuthAngle"][1:3]
            ),
            "vza_uv2": np.mean(
                o_omi["Earth_Radiance"]["ObservationTable"]["ViewingZenithAngle"][1:3]
            ),
            "sca_uv2": np.mean(
                o_omi["Earth_Radiance"]["ObservationTable"]["ScatteringAngle"][1:3]
            ),
        }

    def update_state(
        self,
        retrieval_info: RetrievalInfo,
        results_list: np.ndarray,
        do_not_update: np.ndarray,
        retrieval_config: RetrievalConfiguration,
        step: int,
    ):
        """Note this updates the current state, and also creates a "next_state".
        The difference is that current gets all the changes found in the
        results_list, but next_state only gets the elements updated that aren't
        listed in do_not_update. This allows things to be done in a particular
        retrieval step, but not actually propagated to the next retrieval step.
        Call next_state_to_current() to update the current state with the
        next_state (i.e., remove the changes for things listed in do_not_update)."""
        self.next_state_dict = copy.deepcopy(self.state_info_dict["current"])

        do_update_fm = np.zeros(retrieval_info.n_totalParametersFM)

        for state_element_name in retrieval_info.species_names:
            update_next = False if state_element_name in do_not_update else True
            self.state_element(state_element_name).update_state_element(
                retrieval_info,
                results_list,
                update_next,
                retrieval_config,
                step,
                do_update_fm,
            )

        # Update altitude and air density
        indt = self.state_element_on_levels.index("TATM")
        indh = self.state_element_on_levels.index("H2O")
        smeta = self.sounding_metadata()
        (results, _) = mpy.compute_altitude_pge(
            self.pressure,
            self.state_info_dict["current"]["values"][indt, :],
            self.state_info_dict["current"]["values"][indh, :],
            smeta.surface_altitude.convert("m").value,
            smeta.latitude.value,
            None,
            True,
        )
        self.state_info_dict["current"]["heightKm"] = results["altitude"] / 1000.0
        self.state_info_dict["current"]["airDensity"] = results["airDensity"]
        # Update doUpdateFM in i_retrievalInfo. Note it might be good to move this
        # out of this function, it isn't good to have "side effects". But leave for
        # now
        retrieval_info.retrieval_dict["doUpdateFM"] = do_update_fm

    def state_element_list(self, step="current"):
        """Return the list of state elements that we already have in StateInfo.
        Note that state_element creates this on first use, the list returned is
        those state elements that have already been created."""
        if step == "current":
            return list(self.current.values())
        elif step == "initialInitial":
            return list(self.initialInitial.values())
        elif step == "initial":
            return list(self.initial.values())
        else:
            raise RuntimeError("step must be initialInitial, initial, or current")

    def state_element(self, name, step="current"):
        """Return the state element with the given name."""
        # We create the StateElement objects on first use
        if name not in self.current:
            (self.initialInitial[name], self.initial[name], self.current[name]) = (
                self.state_element_handle_set.state_element_object(self, name)
            )
        if step == "initialInitial":
            return self.initialInitial[name]
        elif step == "initial":
            return self.initial[name]
        elif step == "current":
            return self.current[name]
        else:
            raise RuntimeError("step must be initialInitial, initial, or current")


__all__ = [
    "StateElement",
    "StateElementHandle",
    "RetrievableStateElement",
    "StateElementHandleSet",
    "PropagatedQA",
    "SoundingMetadata",
    "StateInfo",
]

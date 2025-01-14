from __future__ import annotations  # We can remove this when we upgrade to python 3.9
import abc
from .priority_handle_set import PriorityHandleSet
from .observation_handle import mpy_radiance_from_observation_list
from .tes_file import TesFile
import refractor.muses.muses_py as mpy  # type: ignore
from loguru import logger
import copy
from pathlib import Path
import refractor.framework as rf  # type: ignore
import numpy as np
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
        instrument_name_all: str,
        run_dir: Path,
    ):
        (_, _, _, _, _, _, self.state_info_dict) = mpy.script_retrieval_setup_ms(
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

        self.state_info_dict = mpy.states_initial_update(
            self.state_info_dict,
            strategy_table.strategy_table_dict,
            rad,
            instrument_name_all,
        )

        # Read some metadata that isn't already available
        f = TesFile(run_dir / "DateTime.asc")
        self._tai_time = float(f["TAI_Time_of_ZPD"])
        self._utc_time = f["UTC_Time"]
        self._sounding_id = TesFile(run_dir / "Measurement_ID.asc")["key"]
        self.next_state_dict = None

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

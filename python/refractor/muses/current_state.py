from __future__ import annotations
from . import muses_py as mpy  # type: ignore
import refractor.framework as rf  # type: ignore
import numpy as np
import abc
from pathlib import Path
from copy import copy
import os
import typing
from typing import Self
from .identifier import StateElementIdentifier, InstrumentIdentifier
from .tes_file import TesFile
from scipy.linalg import block_diag  # type: ignore

if typing.TYPE_CHECKING:
    from refractor.old_py_retrieve_wrapper import (  # type: ignore
        StateInfoOld,
        StateElementOld,
        StateElementHandleSetOld,
    )
    from .refractor_uip import RefractorUip
    from .retrieval_configuration import RetrievalConfiguration
    from .muses_observation import MeasurementId
    from .error_analysis import ErrorAnalysis
    from .muses_strategy import CurrentStrategyStep
    from .retrieval_info import RetrievalInfo
    from .muses_strategy import MusesStrategy
    from .observation_handle import ObservationHandleSet
    from .state_info import StateElement, StateElementHandleSet


class PropagatedQA:
    """There are a few parameters that get propagated from one step to
    the next. Not sure exactly what this gets looked for, it look just
    like flags copied from one step to the next. But pull this
    together into one place so we can track this.

    TODO Note that we might just make this appear like any other StateElementOld,
    but at least for now leave this as separate because that is how

    """

    def __init__(self) -> None:
        self.propagated_qa: dict[str, int] = {"TATM": 1, "H2O": 1, "O3": 1}

    @property
    def tatm_qa(self) -> int:
        return self.propagated_qa["TATM"]

    @property
    def h2o_qa(self) -> int:
        return self.propagated_qa["H2O"]

    @property
    def o3_qa(self) -> int:
        return self.propagated_qa["O3"]

    def update(
        self, retrieval_state_element_id: list[StateElementIdentifier], qa_flag: int
    ) -> None:
        """Update the QA flags for items that we retrieved."""
        for state_element_id in retrieval_state_element_id:
            if str(state_element_id) in self.propagated_qa:
                self.propagated_qa[str(state_element_id)] = qa_flag


class SoundingMetadata:
    """Not really clear that this belongs in the StateInfoOld, but the muses-py seems
    to at least allow the possibility of this changing from one step to the next.
    I'm not sure if that actually can happen, but there isn't another obvious place
    to put this metadata so we'll go ahead and keep this here."""

    def __init__(self) -> None:
        """Note you normally call one of the creator functions rather than __init__"""
        self._latitude = rf.DoubleWithUnit(0, "deg")
        self._longitude = rf.DoubleWithUnit(0, "deg")
        self._surface_altitude = rf.DoubleWithUnit(0, "km")
        self._day_flag = False
        self._height = rf.ArrayWithUnit_double_1(
            np.array(
                [
                    0,
                ]
            ),
            "km",
        )
        self._surface_type = ""
        self._tai_time = 0.0
        self._sounding_id = ""
        self._utc_time = ""

    @classmethod
    def create_from_measurement_id(
        cls, measurement_id: MeasurementId, instrument: InstrumentIdentifier
    ) -> Self:
        res = cls()
        instrument_name = str(instrument)
        if f"{instrument_name}_latitude" in measurement_id:
            res._latitude = rf.DoubleWithUnit(
                float(measurement_id[f"{instrument_name}_latitude"]), "deg"
            )
        else:
            res._latitude = rf.DoubleWithUnit(
                float(measurement_id[f"{instrument_name}_Latitude"]), "deg"
            )
        if f"{instrument_name}_longitude" in measurement_id:
            res._longitude = rf.DoubleWithUnit(
                float(measurement_id[f"{instrument_name}_longitude"]), "deg"
            )
        else:
            res._longitude = rf.DoubleWithUnit(
                float(measurement_id[f"{instrument_name}_Longitude"]), "deg"
            )
        if "oceanFlag" in measurement_id:
            oceanflag = int(measurement_id["oceanflag"])
        else:
            oceanflag = int(measurement_id["OCEANFLAG"])
        res._surface_type = "OCEAN" if oceanflag == 1 else "LAND"
        res._sounding_id = measurement_id["key"]
        # Couple of things in the DateTime file
        f = TesFile(measurement_id["run_dir"] / "DateTime.asc")
        res._tai_time = float(f["TAI_Time_of_ZPD"])
        res._utc_time = f["UTC_Time"]

        date_struct = mpy.utc_from_string(res._utc_time)
        i_date_struct = {}
        i_date_struct["dateStruct"] = date_struct
        i_date_struct["year"] = date_struct["utctime"].year
        i_date_struct["month"] = date_struct["utctime"].month
        i_date_struct["day"] = date_struct["utctime"].day
        i_date_struct["hour"] = date_struct["utctime"].hour
        i_date_struct["minute"] = date_struct["utctime"].minute
        i_date_struct["second"] = date_struct["utctime"].second
        res._day_flag = bool(mpy.daytime(i_date_struct, res._longitude.value))

        # Need to fill these in
        res._surface_altitude = rf.DoubleWithUnit(0, "km")
        res._height = rf.ArrayWithUnit_double_1(
            np.array(
                [
                    0,
                ]
            ),
            "km",
        )
        return res

    @classmethod
    def create_from_old_state_info(
        cls, state_info: StateInfoOld, step: str = "current"
    ) -> Self:
        if step not in ("current", "initial", "initialInitial"):
            raise RuntimeError(
                "Don't support anything other than the current, initial, or initialInitial step"
            )
        res = cls()
        res._latitude = rf.DoubleWithUnit(
            state_info.state_info_dict[step]["latitude"], "deg"
        )
        res._longitude = rf.DoubleWithUnit(
            state_info.state_info_dict[step]["longitude"], "deg"
        )
        res._surface_altitude = rf.DoubleWithUnit(
            state_info.state_info_dict[step]["tsa"]["surfaceAltitudeKm"], "km"
        )
        res._day_flag = bool(state_info.state_info_dict[step]["tsa"]["dayFlag"])
        res._height = rf.ArrayWithUnit_double_1(
            state_info.state_info_dict[step]["heightKm"], "km"
        )
        res._surface_type = state_info.state_info_dict[step]["surfaceType"].upper()
        res._tai_time = state_info._tai_time
        res._sounding_id = state_info._sounding_id
        res._utc_time = state_info._utc_time
        return res

    @property
    def latitude(self) -> rf.DoubleWithUnit:
        return self._latitude

    @property
    def longitude(self) -> rf.DoubleWithUnit:
        return self._longitude

    @property
    def surface_altitude(self) -> rf.DoubleWithUnit:
        return self._surface_altitude

    @property
    def height(self) -> rf.DoubleWithUnit:
        return self._height

    @property
    def tai_time(self) -> float:
        return self._tai_time

    @property
    def utc_time(self) -> str:
        return self._utc_time

    @property
    def local_hour(self) -> int:
        timestruct = mpy.utc(self.utc_time)
        hour = timestruct["hour"] + self.longitude.convert("deg").value / 180.0 * 12
        if hour < 0:
            hour += 24
        if hour > 24:
            hour -= 24
        return hour

    @property
    def wrong_tai_time(self) -> float:
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
    def sounding_id(self) -> str:
        return self._sounding_id

    @property
    def surface_type(self) -> str:
        return self._surface_type

    @property
    def is_day(self) -> bool:
        return self._day_flag

    @property
    def is_ocean(self) -> bool:
        return self.surface_type == "OCEAN"

    @property
    def is_land(self) -> bool:
        return self.surface_type == "LAND"


# A couple of aliases, just so we can clearly mark what grid data is on
RetrievalGridArray = np.ndarray
ForwardModelGridArray = np.ndarray
RetrievalGrid2dArray = np.ndarray
ForwardModelGrid2dArray = np.ndarray


class CurrentState(object, metaclass=abc.ABCMeta):
    """There are a number of "states" floating around
    py-retrieve/ReFRACtor, and it can be a little confusing if you
    don't know what you are looking at.

    A good reference is section III.A.1 of "Tropospheric Emission
    Spectrometer: Retrieval Method and Error Analysis" (IEEE
    TRANSACTIONS ON GEOSCIENCE AND REMOTE SENSING, VOL. 44, NO. 5, MAY
    2006). This describes the "retrieval state vector" and the "full
    state vector"

    In addition there is an intermediate set that isn't named in the
    paper.  In the py-retrieve code, this gets referred to as the
    "forward model state vector". And internally objects may map
    things to a "object state".

    A brief description of these:

    1. The "retrieval state vector" is what we use in our Solver for a
       retrieval step. This is the parameters passed to our
       CostFunction.

    2. The "forward model state vector" has the same content as the
       "retrieval state vector", but it is specified on a finer number
       of levels used by the forward model.

    3. The "full state vector" is all the parameters the various
       objects making up our ForwardModel and Observation needs. This
       includes a number of things held fixed in a particular
       retrieval step.

    4. The "object state" is the subset of the "forward model state
       vector" needed by a particular object in our ForwardModel or
       Observation, mapped to what the object needs. E.g., the forward
       model state vector is in log(vmr), but for the actual object we
       translate this to vmr.

    Each of these vectors are made up of a number of StateElement,
    each with a fixed string used as a name to identify it.  We look
    up the StateElement by these fixed names. Note that the
    StateElement name value may have different values depending on the
    context - so it might have fewer levels in a retrieval state
    vector vs forward model state vector. Note that muses-py often but
    not always refers to these as "species". We use the more general
    name "StateElement" because these aren't always gas species.

    A few examples, might illustrate this:

    One of the things that might be retrieved is log(vmr) for O3. In
    the "retrieval state vector" this has 25 log(vmr) values (for the
    25 levels). In the "forward model state vector" this as 64
    log(vmr) values (for the 64 levels the FM is run on).  In the
    "object state" for the rf.AbsorberVmr part of the FowardModel is
    64 vmr values (so log(vmr) converted to vmr needed for
    calculation).

    For the tropomi ForwardModel, a component object is a
    rf.GroundLambertian which has a polynomial with values
    "TROPOMISURFACEALBEDOBAND3", "TROPOMISURFACEALBEDOSLOPEBAND3",
    "TROPOMISURFACEALBEDOSLOPEORDER2BAND3". We might be retrieving
    only a subset of these values, holding the other fixed. So in this
    case the "retrieval state vector" might have only 2 entries for
    "TROPOMISURFACEALBEDOBAND3" and "TROPOMISURFACEALBEDOSLOPEBAND3",
    the "forward model state vector" would also only have 2 entries
    (since there aren't any levels involved, there is no difference
    between the retrieval state and the forward model state).  The
    "full state vector" would have 3 values, since we add in the part
    that is being held fixed.

    In all cases, we handle converting from one type of state vector
    to the other with a rf.StateMapping. The
    rf.MaxAPosterioriSqrtConstraint object in our CostFunction has a
    mapping going to and from the "retrieval state vector" to the
    "forward model state vector". The various pieces of the
    ForwardModel and Observation have mapping from "forward model
    state vector" to "full state vector", and the various objects
    handle mappings from "full state vector" to "object state".

    For a normal retrieval, we get all the information needed from our
    StateInfo. But for testing it can be useful to have other
    implementations, include a hard coded set of values (for small
    unit tests) or a RefractorUip (to compare against old py-retrieve
    runs where we captured the UIP).  This class gives the interface
    needed by the other classes, as well as implementing some stuff
    that doesn't really depend on where we are getting the
    information.

    """

    def __init__(self) -> None:
        # Cache these values, they don't normally change.
        self._fm_sv_loc: dict[StateElementIdentifier, tuple[int, int]] | None = None
        self._fm_state_vector_size = -1
        self._sys_sv_loc: dict[StateElementIdentifier, tuple[int, int]] | None = None
        self._sys_state_vector_size = -1
        self._retrieval_sv_loc: dict[StateElementIdentifier, tuple[int, int]] | None = (
            None
        )
        self._retrieval_state_vector_size = -1

    def current_state_override(
        self,
        do_systematic: bool,
        retrieval_state_element_override: None | list[StateElementIdentifier],
    ) -> CurrentState:
        """Create a variation of the current state that either does a systematic
        jacobian and/or overrides the retrieval_state_element_override. This is actually
        a bit awkward, but it is how muses-py was set up. This is only used in a few places,
        so we'll go ahead and use the same logic here."""
        raise NotImplementedError()

    @property
    def propagated_qa(self) -> PropagatedQA:
        raise NotImplementedError()

    @property
    def brightness_temperature_data(self) -> dict:
        raise NotImplementedError()

    def clear_cache(self) -> None:
        """Clear cache, if an update has occurred"""
        self._fm_sv_loc = None
        self._fm_state_vector_size = -1
        self._sys_sv_loc = None
        self._sys_state_vector_size = -1
        self._retrieval_sv_loc = None
        self._retrieval_state_vector_size = -1

    @property
    def initial_guess(self) -> RetrievalGridArray:
        """Initial guess, on the retrieval grid."""
        raise NotImplementedError()

    @property
    def initial_guess_fm(self) -> ForwardModelGridArray:
        """Return the initial guess for the forward model grid.  This
        isn't independent, it is directly calculated from the
        initial_guess and basis_matrix. But convenient to supply this
        (mostly as a help in unit testing).

        """
        return self.state_mapping_retrieval_to_fm.mapped_state(
            rf.ArrayAd_double_1(self.initial_guess)
        ).value

    def update_full_state_element(
        self,
        state_element_id: StateElementIdentifier,
        current: np.ndarray | None = None,
        apriori: np.ndarray | None = None,
        step_initial: np.ndarray | None = None,
        retrieval_initial: np.ndarray | None = None,
        true_value: np.ndarray | None = None,
    ) -> None:
        """We have a few places where we want to update a state element other than
        update_initial_guess. This function updates each of the various values passed in.
        A value of 'None' (the default) means skip updating that part of the state."""
        raise NotImplementedError()

    @property
    def updated_fm_flag(self) -> ForwardModelGridArray:
        """This is array of boolean flag indicating which parts of the forward
        model state vector got updated when we called notify_solution. A 1 means
        it was updated, a 0 means it wasn't. This is used in the ErrorAnalysis."""
        raise NotImplementedError()

    @property
    def constraint_matrix(self) -> RetrievalGrid2dArray:
        """Constraint matrix"""
        raise NotImplementedError()

    @property
    def sqrt_constraint(self) -> RetrievalGridArray:
        """Sqrt matrix from covariance"""
        return (mpy.sqrt_matrix(self.constraint_matrix)).transpose()

    @property
    def apriori(self) -> RetrievalGridArray:
        """Apriori value"""
        raise NotImplementedError()

    @property
    def apriori_fm(self) -> ForwardModelGrid2dArray:
        """Apriori value"""
        raise NotImplementedError()

    @property
    def true_value(self) -> RetrievalGridArray:
        """True value"""
        raise NotImplementedError()

    @property
    def true_value_fm(self) -> ForwardModelGridArray:
        """True value"""
        raise NotImplementedError()

    @property
    def basis_matrix(self) -> np.ndarray | None:
        """Basis matrix going from retrieval vector to full model vector.
        We don't always have this, so we return None if there isn't a basis matrix.
        """
        raise NotImplementedError()

    @property
    def map_to_parameter_matrix(self) -> np.ndarray | None:
        """Go the other direction from basis matrix"""
        raise NotImplementedError()

    @property
    def state_mapping_retrieval_to_fm(self) -> rf.StateMapping:
        """Return StateMapping going from RetrievalGridArray to ForwardModelGridArray.
        This is done by a basis_matrix in muses-py, but we are trying to move to a more
        general StateMapping."""
        bmatrix = self.basis_matrix
        if bmatrix is None:
            return rf.StateMappingLinear()
        return rf.StateMappingBasisMatrix(bmatrix.transpose())

    def fm_sv_slice(self, sid: StateElementIdentifier) -> slice | None:
        """Slice object needed  to subset the forward model state vector to
        the values for the StateElement. As a convention, this can be called
        with StateElement that aren't being retrieved and we return None."""
        if sid in self.fm_sv_loc:
            p, plen = self.fm_sv_loc[sid]
            return slice(p, p + plen)
        return None

    @property
    def fm_sv_loc(self) -> dict[StateElementIdentifier, tuple[int, int]]:
        """Dict that gives the starting location in the forward model
        state vector and length for a particular state element name
        (state elements not being retrieved don't get listed here)
        """
        if self._fm_sv_loc is None:
            self._fm_sv_loc = {}
            self._fm_state_vector_size = 0
            for state_element_id in self.retrieval_state_element_id:
                plen = len(self.full_state_value(state_element_id))
                self._fm_sv_loc[state_element_id] = (self._fm_state_vector_size, plen)
                self._fm_state_vector_size += plen
        return self._fm_sv_loc

    def sys_sv_slice(self, sid: StateElementIdentifier) -> slice | None:
        """Slice object needed to subset the systematic forward model
        state vector to the values for the StateElement. As a
        convention, this can be called with StateElement that aren't
        being retrieved and we return None.

        """
        if sid in self.sys_sv_loc:
            p, plen = self.sys_sv_loc[sid]
            return slice(p, p + plen)
        return None

    @property
    def sys_sv_loc(self) -> dict[StateElementIdentifier, tuple[int, int]]:
        """Dict that gives the starting location in the systematic forward model
        state vector and length for a particular state element name
        (state elements not being retrieved don't get listed here)
        """
        if self._sys_sv_loc is None:
            self._sys_sv_loc = {}
            self._sys_state_vector_size = 0
            for state_element_id in self.systematic_state_element_id:
                plen = len(self.full_state_value(state_element_id))
                self._sys_sv_loc[state_element_id] = (self._sys_state_vector_size, plen)
                self._sys_state_vector_size += plen
        return self._sys_sv_loc

    def retrieval_sv_slice(self, sid: StateElementIdentifier) -> slice | None:
        """Slice object needed  to subset the retrieval state vector to
        the values for the StateElement. As a convention, this can be called
        with StateElement that aren't being retrieved and we return None."""
        if sid in self.retrieval_sv_loc:
            p, plen = self.retrieval_sv_loc[sid]
            return slice(p, p + plen)
        return None

    @property
    def retrieval_sv_loc(self) -> dict[StateElementIdentifier, tuple[int, int]]:
        """Like fm_sv_loc, but for the retrieval state vactor (rather than the
        forward model state vector). If we don't have a basis_matrix, these are the
        same. With a basis_matrix, the total length of the fm_sv_loc is the
        basis_matrix column size, and retrieval_vector_loc is the smaller basis_matrix
        row size."""
        raise NotImplementedError()

    def state_mapping(
        self, state_element_id: StateElementIdentifier
    ) -> rf.StateMapping:
        """StateMapping used by the forward model (so taking the ForwardModelGridArray
        and mapping to the internal object state)"""
        return self.full_state_element(state_element_id).state_mapping

    def pressure_list(
        self, state_element_id: StateElementIdentifier
    ) -> RetrievalGridArray | None:
        """For state elements that are on pressure level, this returns
        the pressure levels.  This is for the retrieval state vector
        levels (generally smaller than the pressure_list_fm).
        """
        raise NotImplementedError()

    def pressure_list_fm(
        self, state_element_id: StateElementIdentifier
    ) -> ForwardModelGridArray | None:
        """For state elements that are on pressure level, this returns
        the pressure levels.  This is for the forward model state
        vector levels (generally larger than the pressure_list).
        """
        raise NotImplementedError()

    def altitude_list(
        self, state_element_id: StateElementIdentifier
    ) -> RetrievalGridArray | None:
        """For state elements that are on pressure level, this returns
        the altitude.  This is for the retrieval state vector
        levels (generally smaller than the altitude_list_fm).
        """
        raise NotImplementedError()

    def altitude_list_fm(
        self, state_element_id: StateElementIdentifier
    ) -> ForwardModelGridArray | None:
        """For state elements that are on pressure level, this returns
        the altitude.  This is for the forward model state
        vector levels (generally larger than the altitude_list).
        """
        raise NotImplementedError()

    @property
    def forward_model_state_vector_element_list(self) -> list[StateElementIdentifier]:
        """List of StateElementIdentifier for each entry in the
        forward model state vector. This is the full size of the
        forward model state vector, so in general a
        StateElementIdentifier may be listed multiple times.

        """
        res = [
            StateElementIdentifier("None"),
        ] * self.fm_state_vector_size
        for sid, (pstart, plen) in self.fm_sv_loc.items():
            res[pstart : (pstart + plen)] = [
                sid,
            ] * plen
        return res

    @property
    def systematic_model_state_vector_element_list(
        self,
    ) -> list[StateElementIdentifier]:
        """List of StateElementIdentifier for each entry in the
        systematic forward model state vector. This is the full size of the
        systematic forward model state vector, so in general a
        StateElementIdentifier may be listed multiple times.

        """
        res = [
            StateElementIdentifier("None"),
        ] * self.sys_state_vector_size
        for sid, (pstart, plen) in self.sys_sv_loc.items():
            res[pstart : (pstart + plen)] = [
                sid,
            ] * plen
        return res

    @property
    def retrieval_state_vector_element_list(self) -> list[StateElementIdentifier]:
        """List of StateElementIdentifier for each entry in the
        retrieval state vector. This is the full size of the retrieval
        state vector, so in general a StateElementIdentifier may be
        listed multiple times.

        """
        res = [
            StateElementIdentifier("None"),
        ] * self.retrieval_state_vector_size
        for sid, (pstart, plen) in self.retrieval_sv_loc.items():
            res[pstart : (pstart + plen)] = [
                sid,
            ] * plen
        return res

    @property
    def fm_state_vector_size(self) -> int:
        """Full size of the forward model state vector."""
        if self._fm_state_vector_size < 0:
            # Side effect of fm_sv_loc is filling in fm_state_vector_size
            _ = self.fm_sv_loc
        return self._fm_state_vector_size

    @property
    def sys_state_vector_size(self) -> int:
        """Full size of the systematic forward model state vector."""
        if self._sys_state_vector_size < 0:
            # Side effect of fm_sv_loc is filling in fm_state_vector_size
            _ = self.sys_sv_loc
        return self._sys_state_vector_size

    @property
    def retrieval_state_vector_size(self) -> int:
        """Full size of the retrieval state vector."""
        if self._retrieval_state_vector_size < 0:
            # Side effect of retrieval_sv_loc is filling in fm_state_vector_size
            _ = self.retrieval_sv_loc
        return self._retrieval_state_vector_size

    def object_state(
        self, state_element_id_list: list[StateElementIdentifier]
    ) -> tuple[np.ndarray, rf.StateMapping]:
        """Return a set of coefficients and a rf.StateMapping to get
        the full state values used by an object. The object passes in
        the list of state element names it uses.  In general only a
        (possibly empty) subset of the state elements are actually
        retrieved.  This gets handled by the StateMapping, which might
        have a component like rf.StateMappingAtIndexes to handle the
        subset that is in the fm_state_vector.

        """
        # TODO put in handling of log/linear
        coeff = np.concatenate(
            [self.full_state_value(nm) for nm in state_element_id_list]
        )
        rlist = self.retrieval_state_element_id
        rflag = np.concatenate(
            [
                np.full((len(self.full_state_value(nm)),), nm in rlist, dtype=bool)
                for nm in state_element_id_list
            ]
        )
        mp = rf.StateMappingAtIndexes(rflag)
        return (coeff, mp)

    def add_fm_state_vector_if_needed(
        self,
        fm_sv: rf.StateVector,
        state_element_id_list: list[StateElementIdentifier],
        obj_list: list[rf.SubStateVectorObserver],
    ) -> None:
        """This takes an object and a list of the state element names
        that object uses. This then adds the object to the forward
        model state vector if some of the elements are being
        retrieved.  This is a noop if none of the state elements are
        being retrieved. So objects don't need to try to figure out if
        they are in the retrieved set or not, then can just call this
        function to try adding themselves.

        """
        pstart = None
        for sname in state_element_id_list:
            if sname in self.fm_sv_loc:
                ps, _ = self.fm_sv_loc[sname]
                if pstart is None or ps < pstart:
                    pstart = ps
        if pstart is not None:
            fm_sv.observer_claimed_size = pstart
            for obj in obj_list:
                fm_sv.add_observer(obj)

    @abc.abstractproperty
    def retrieval_state_element_id(self) -> list[StateElementIdentifier]:
        """Return list of state elements we are retrieving."""
        raise NotImplementedError()

    @abc.abstractproperty
    def systematic_state_element_id(self) -> list[StateElementIdentifier]:
        """Return list of state elements that are in the systematic list (used by the
        ErrorAnalysis)."""
        raise NotImplementedError()

    @abc.abstractproperty
    def full_state_element_id(self) -> list[StateElementIdentifier]:
        """Return list of state elements that make up the full state, generally a
        larger list than retrieval_state_element_id"""
        raise NotImplementedError()

    def full_state_desc(self) -> str:
        """Return a description of the full state."""
        res = ""
        for selem in self.full_state_element_id:
            if self.full_state_value_str(selem) is not None:
                res += f"{str(selem)}:\n{self.full_state_value_str(selem)}\n"
            else:
                res += f"{str(selem)}:\n{self.full_state_value(selem)}\n"
        return res

    @abc.abstractproperty
    def sounding_metadata(self) -> SoundingMetadata:
        """Return the sounding metadata. It isn't clear if this really
        belongs in CurrentState or not, but there isn't another
        obvious place for this so for now we'll have this here.

        Perhaps this can migrate to the MuseObservation or MeasurementId if we decide
        that makes more sense.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def full_state_element(
        self, state_element_id: StateElementIdentifier
    ) -> StateElement:
        """Return the StateElement for the state_element_id. I'm not sure if we want to
        have this exposed or not, but there is a bit of useful information we have in
        each StateElement (such as the sa_cross_covariance). We can have this exposed for
        now, and revisit it if we end up deciding this is too much coupling. There are
        only a few spots that use full_state_element vs something like full_state_value,
        so we will just need to revisit those few spots if this becomes an issue."""
        raise NotImplementedError()

    @abc.abstractmethod
    def full_state_spectral_domain_wavelength(
        self, state_element_id: StateElementIdentifier
    ) -> np.ndarray | None:
        """Return the spectral domain (as nm) for the given state_element_id, or None if
        there isn't an associated frequency for the given state_element_id"""
        raise NotImplementedError()

    @abc.abstractmethod
    def full_state_value(
        self, state_element_id: StateElementIdentifier
    ) -> ForwardModelGridArray:
        """Return the full state value for the given state element
        name.  Just as a convention we always return a np.array, so if
        there is only one value put that in a length 1 np.array.

        """
        raise NotImplementedError()

    @abc.abstractmethod
    def full_state_value_str(
        self, state_element_id: StateElementIdentifier
    ) -> str | None:
        """A small number of values in the full state are actually str (e.g.,
        StateElementIdentifier("nh3type"). This is like full_state_value, but we
        return a str instead. Return None if this doesn't apply.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def full_state_step_initial_value(
        self, state_element_id: StateElementIdentifier, use_map: bool = False
    ) -> ForwardModelGridArray:
        """Return the initial value of the given state element identification.
        Just as a convention we always return a np.array, so if
        there is only one value put that in a length 1 np.array.

        Where this is used in the muses-py code it sometimes assumes this has been
        mapped (so a log initial guess gets exp applied). This is a bit confusing,
        it means full_state_step_initial_value and initial_guess_value_fm aren't the same.
        We handle this just by requiring a use_map=True to be passed in, meaning we apply
        the state_mapping in reverse.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def full_state_true_value(
        self, state_element_id: StateElementIdentifier
    ) -> ForwardModelGridArray | None:
        """Return the true value of the given state element identification.
        Just as a convention we always return a np.array, so if
        there is only one value put that in a length 1 np.array.

        If we don't have a true value, return None
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def full_state_retrieval_initial_value(
        self, state_element_id: StateElementIdentifier
    ) -> ForwardModelGridArray:
        """Return the initialInitial value of the given state element identification.
        Just as a convention we always return a np.array, so if
        there is only one value put that in a length 1 np.array.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def full_state_apriori_value(
        self, state_element_id: StateElementIdentifier, use_map: bool = False
    ) -> ForwardModelGridArray:
        """Return the apriori value of the given state element identification.
        Just as a convention we always return a np.array, so if
        there is only one value put that in a length 1 np.array.

        Where this is used in the muses-py code it sometimes assumes this has been
        mapped (so a log apriori gets exp applied). This is a bit confusing,
        it means full_state_step_aprior_value and apriori_value_fm aren't the same.
        We handle this just by requiring a use_map=True to be passed in, meaning we apply
        the state_mapping in reverse.
        """
        raise NotImplementedError()

    @property
    def Sa(self) -> np.ndarray:
        '''This combines the retrieval state element apriori_cov_fm and cross terms into
        a apriori_cov_fm of those elements. This S_a in the paper
        
        Tropospheric Emission Spectrometer: Retrieval Method and Error Analysis
        V. ERROR CHARACTERIZATION
        (https://ieeexplore.ieee.org/document/1624609)
        '''
        return self.apriori_cov_fm(self.retrieval_state_element_id)

    def apriori_cov_fm(self, list_state_element_id: list[StateElementIdentifier]) -> np.ndarray:
        '''Return apriori covariance for the given list of state elements, including cross
        terms. When the list is the retrieval_state_element_id this is S_a in the paper

        Tropospheric Emission Spectrometer: Retrieval Method and Error Analysis
        V. ERROR CHARACTERIZATION
        (https://ieeexplore.ieee.org/document/1624609)
        '''
        selem_list = [self.full_state_element(sname) for sname in list_state_element_id]

        # Make block diagonal covariance.
        species_list = []
        matrix_list = []
        for selem in selem_list:
            matrix = selem.apriori_cov_fm
            species_list.extend([str(selem.state_element_id)] * matrix.shape[0])
            matrix_list.append(matrix)

        res = block_diag(*matrix_list)
        # TODO Replace this with cross term state elements when we put those
        # into place
        # Off diagonal blocks for covariance.
        for i, selem1 in enumerate(selem_list):
            for selem2 in selem_list[i + 1 :]:
                matrix2 = selem1.apriori_cross_covariance_fm(selem2)
                if matrix2 is not None:
                    res[np.array(species_list) == str(selem1.state_element_id), :][
                        :, np.array(species_list) == str(selem2.state_element_id)
                    ] = matrix2
                    res[np.array(species_list) == str(selem2.state_element_id), :][
                        :, np.array(species_list) == str(selem1.state_element_id)
                    ] = np.transpose(matrix2)
        return res


    @abc.abstractmethod
    def full_state_apriori_covariance(
        self, state_element_id: StateElementIdentifier
    ) -> ForwardModelGrid2dArray:
        """Return the covariance of the apriori value of the given state element identification."""
        raise NotImplementedError()

    @property
    def step_directory(self) -> Path:
        """Return the step directory. This is a bit odd, but it is
        needed by MusesOpticalDepthFile. Since the current state
        depends on the step we are using, it isn't ridiculous to have
        this here. However if we find a better home for or better
        still remove the need for this that would be good.

        """
        raise NotImplementedError()

    @property
    def state_element_handle_set(self) -> StateElementHandleSet:
        raise NotImplementedError()

    def notify_start_retrieval(
        self,
        current_strategy_step: CurrentStrategyStep | None,
        retrieval_config: RetrievalConfiguration,
    ) -> None:
        pass

    def notify_update_target(
        self,
        measurement_id: MeasurementId,
        retrieval_config: RetrievalConfiguration,
        strategy: MusesStrategy,
        observation_handle_set: ObservationHandleSet,
    ) -> None:
        """Have updated the target we are processing."""
        pass

    def notify_start_step(
        self,
        current_strategy_step: CurrentStrategyStep | None,
        error_analysis: ErrorAnalysis,
        retrieval_config: RetrievalConfiguration,
        skip_initial_guess_update: bool = False,
    ) -> None:
        """Called when MusesStrategy has gone to the next step."""
        pass

    def notify_step_solution(self, xsol: RetrievalGridArray) -> None:
        """Called with MusesStrategy has found a solution for the current step."""
        pass


class CurrentStateUip(CurrentState):
    """Implementation of CurrentState that uses a RefractorUip"""

    def __init__(self, rf_uip: RefractorUip, ret_info: dict | None = None):
        """Get the CurrentState from a RefractorUip and ret_info. Note
        that this is just for backwards testing, we don't use the UIP
        in our current processing but rather something like
        CurrentStateStateInfo.

        The RefractorUip doesn't have everything we need, specifically
        we don't have the apriori and sqrt_constraint. We can get this
        from a ret_info, if available.  For testing we don't always
        have ret_info. This is fine if we don't actually need the
        apriori and sqrt_constraint. We still need a value for this,
        so if ret_info is None we return arrays of all zeros of the
        right size.

        """
        super().__init__()
        self.rf_uip = rf_uip
        self._initial_guess = rf_uip.current_state_x
        self._basis_matrix = rf_uip.basis_matrix
        self.ret_info = ret_info

    @property
    def initial_guess(self) -> RetrievalGridArray:
        return copy(self._initial_guess)

    @property
    def constraint_matrix(self) -> RetrievalGrid2dArray:
        # Don't think we need this. We can calculate something frm
        # sqrt_constraint if needed, but for now just leave
        # unimplemented
        raise NotImplementedError()

    @property
    def sqrt_constraint(self) -> RetrievalGridArray:
        if self.ret_info:
            return self.ret_info["sqrt_constraint"]
        else:
            # Dummy value, of the right size. Useful when we need
            # this, but don't actually care about the value (e.g., we
            # are running the forward model only in the CostFunction).
            #
            # This is entirely a matter of convenience, we could
            # instead just duplicate the stitching together part of
            # our CostFunction and skip this. But for now this seems
            # like the easiest thing thing to do. We can revisit this
            # decision in the future if needed - it is never great to
            # have fake data but in this case seemed the easiest path
            # forward. Since this function is only used for backwards
            # testing, the slightly klunky design doesn't seem like
            # much of a problem.
            return np.eye(len(self.initial_guess))

    @property
    def apriori(self) -> RetrievalGridArray:
        if self.ret_info:
            return self.ret_info["const_vec"]
        else:
            # Dummy value, of the right size. Useful when we need
            # this, but don't actually care about the value (e.g., we
            # are running the forward model only in the CostFunction).
            #
            # This is entirely a matter of convenience, we could
            # instead just duplicate the stitching together part of
            # our CostFunction and skip this. But for now this seems
            # like the easiest thing thing to do. We can revisit this
            # decision in the future if needed - it is never great to
            # have fake data but in this case seemed the easiest path
            # forward. Since this function is only used for backwards
            # testing, the slightly klunky design doesn't seem like
            # much of a problem.
            return np.zeros((len(self.initial_guess),))

    @property
    def basis_matrix(self) -> np.ndarray | None:
        return self._basis_matrix

    @property
    def map_to_parameter_matrix(self) -> np.ndarray | None:
        raise NotImplementedError()

    # We don't have the other gas species working yet. Short term,
    # just have a different implementation of fm_sv_loc. We should
    # sort this out at some point.
    @property
    def fm_sv_loc(self) -> dict[StateElementIdentifier, tuple[int, int]]:
        if self._fm_sv_loc is None:
            self._fm_sv_loc = {}
            self._fm_state_vector_size = 0
            for species_name in self.retrieval_state_element_id:
                pstart, plen = self.rf_uip.state_vector_species_index(str(species_name))
                self._fm_sv_loc[species_name] = (pstart, plen)
                self._fm_state_vector_size += plen
        return self._fm_sv_loc

    @property
    def retrieval_sv_loc(self) -> dict[StateElementIdentifier, tuple[int, int]]:
        if self._retrieval_sv_loc is None:
            self._retrieval_sv_loc = {}
            self._retrieval_state_vector_size = 0
            for species_name in self.retrieval_state_element_id:
                pstart, plen = self.rf_uip.state_vector_species_index(
                    str(species_name), use_full_state_vector=False
                )
                self._retrieval_sv_loc[species_name] = (pstart, plen)
                self._retrieval_state_vector_size += plen
        return self._retrieval_sv_loc

    @property
    def retrieval_state_element_id(self) -> list[StateElementIdentifier]:
        return [StateElementIdentifier(i) for i in self.rf_uip.jacobian_all]

    @property
    def systematic_state_element_id(self) -> list[StateElementIdentifier]:
        raise NotImplementedError()

    @property
    def full_state_element_id(self) -> list[StateElementIdentifier]:
        # I think we could come up with something here if needed, but for now
        # just punt on this
        raise NotImplementedError()

    @property
    def step_directory(self) -> Path:
        return self.rf_uip.step_directory

    @property
    def sounding_metadata(self) -> SoundingMetadata:
        raise NotImplementedError()

    def full_state_element(
        self, state_element_id: StateElementIdentifier
    ) -> StateElement:
        raise NotImplementedError()

    def full_state_spectral_domain_wavelength(
        self, state_element_id: StateElementIdentifier
    ) -> np.ndarray | None:
        raise NotImplementedError()

    def full_state_value(
        self, state_element_id: StateElementIdentifier
    ) -> ForwardModelGridArray:
        # We've extracted this logic out from update_uip
        o_uip = mpy.ObjectView(self.rf_uip.uip)
        if str(state_element_id) == "TSUR":
            return np.array(
                [
                    o_uip.surface_temperature,
                ]
            )
        elif str(state_element_id) == "EMIS":
            return np.array(o_uip.emissivity["value"])
        elif str(state_element_id) == "PTGANG":
            return np.array([o_uip.obs_table["pointing_angle"]])
        elif str(state_element_id) == "RESSCALE":
            return np.array([o_uip.res_scale])
        elif str(state_element_id) == "CLOUDEXT":
            return np.array(o_uip.cloud["extinction"])
        elif str(state_element_id) == "PCLOUD":
            return np.array([o_uip.cloud["pressure"]])
        elif str(state_element_id) == "OMICLOUDFRACTION":
            return np.array([o_uip.omiPars["cloud_fraction"]])
        elif str(state_element_id) == "OMISURFACEALBEDOUV1":
            return np.array([o_uip.omiPars["surface_albedo_uv1"]])
        elif str(state_element_id) == "OMISURFACEALBEDOUV2":
            return np.array([o_uip.omiPars["surface_albedo_uv2"]])
        elif str(state_element_id) == "OMISURFACEALBEDOSLOPEUV2":
            return np.array([o_uip.omiPars["surface_albedo_slope_uv2"]])
        elif str(state_element_id) == "OMINRADWAVUV1":
            return np.array([o_uip.omiPars["nradwav_uv1"]])
        elif str(state_element_id) == "OMINRADWAVUV2":
            return np.array([o_uip.omiPars["nradwav_uv2"]])
        elif str(state_element_id) == "OMIODWAVUV1":
            return np.array([o_uip.omiPars["odwav_uv1"]])
        elif str(state_element_id) == "OMIODWAVUV2":
            return np.array([o_uip.omiPars["odwav_uv2"]])
        elif str(state_element_id) == "OMIODWAVSLOPEUV1":
            return np.array([o_uip.omiPars["odwav_slope_uv1"]])
        elif str(state_element_id) == "OMIODWAVSLOPEUV2":
            return np.array([o_uip.omiPars["odwav_slope_uv2"]])
        elif str(state_element_id) == "OMIRINGSFUV1":
            return np.array([o_uip.omiPars["ring_sf_uv1"]])
        elif str(state_element_id) == "OMIRINGSFUV2":
            return np.array([o_uip.omiPars["ring_sf_uv2"]])
        elif str(state_element_id) == "TROPOMICLOUDFRACTION":
            return np.array([o_uip.tropomiPars["cloud_fraction"]])
        elif str(state_element_id) == "TROPOMISURFACEALBEDOBAND1":
            return np.array([o_uip.tropomiPars["surface_albedo_BAND1"]])
        elif str(state_element_id) == "TROPOMISURFACEALBEDOBAND2":
            return np.array([o_uip.tropomiPars["surface_albedo_BAND2"]])
        elif str(state_element_id) == "TROPOMISURFACEALBEDOBAND3":
            return np.array([o_uip.tropomiPars["surface_albedo_BAND3"]])
        elif str(state_element_id) == "TROPOMISURFACEALBEDOBAND7":
            return np.array([o_uip.tropomiPars["surface_albedo_BAND7"]])
        elif str(state_element_id) == "TROPOMISURFACEALBEDOBAND3TIGHT":
            return np.array([o_uip.tropomiPars["surface_albedo_BAND3"]])
        elif str(state_element_id) == "TROPOMISURFACEALBEDOSLOPEBAND2":
            return np.array([o_uip.tropomiPars["surface_albedo_slope_BAND2"]])
        elif str(state_element_id) == "TROPOMISURFACEALBEDOSLOPEBAND3":
            return np.array([o_uip.tropomiPars["surface_albedo_slope_BAND3"]])
        elif str(state_element_id) == "TROPOMISURFACEALBEDOSLOPEBAND7":
            return np.array([o_uip.tropomiPars["surface_albedo_slope_BAND7"]])
        elif str(state_element_id) == "TROPOMISURFACEALBEDOSLOPEBAND3TIGHT":
            return np.array([o_uip.tropomiPars["surface_albedo_slope_BAND3"]])
        elif str(state_element_id) == "TROPOMISURFACEALBEDOSLOPEORDER2BAND2":
            return np.array([o_uip.tropomiPars["surface_albedo_slope_order2_BAND2"]])
        elif str(state_element_id) == "TROPOMISURFACEALBEDOSLOPEORDER2BAND3":
            return np.array([o_uip.tropomiPars["surface_albedo_slope_order2_BAND3"]])
        elif str(state_element_id) == "TROPOMISURFACEALBEDOSLOPEORDER2BAND7":
            return np.array([o_uip.tropomiPars["surface_albedo_slope_order2_BAND7"]])
        elif str(state_element_id) == "TROPOMISURFACEALBEDOSLOPEORDER2BAND3TIGHT":
            return np.array([o_uip.tropomiPars["surface_albedo_slope_order2_BAND3"]])
        elif str(state_element_id) == "TROPOMISOLARSHIFTBAND1":
            return np.array([o_uip.tropomiPars["solarshift_BAND1"]])
        elif str(state_element_id) == "TROPOMISOLARSHIFTBAND2":
            return np.array([o_uip.tropomiPars["solarshift_BAND2"]])
        elif str(state_element_id) == "TROPOMISOLARSHIFTBAND3":
            return np.array([o_uip.tropomiPars["solarshift_BAND3"]])
        elif str(state_element_id) == "TROPOMISOLARSHIFTBAND7":
            return np.array([o_uip.tropomiPars["solarshift_BAND7"]])
        elif str(state_element_id) == "TROPOMIRADIANCESHIFTBAND1":
            return np.array([o_uip.tropomiPars["radianceshift_BAND1"]])
        elif str(state_element_id) == "TROPOMIRADIANCESHIFTBAND2":
            return np.array([o_uip.tropomiPars["radianceshift_BAND2"]])
        elif str(state_element_id) == "TROPOMIRADIANCESHIFTBAND3":
            return np.array([o_uip.tropomiPars["radianceshift_BAND3"]])
        elif str(state_element_id) == "TROPOMIRADIANCESHIFTBAND7":
            return np.array([o_uip.tropomiPars["radianceshift_BAND7"]])
        elif str(state_element_id) == "TROPOMIRADSQUEEZEBAND1":
            return np.array([o_uip.tropomiPars["radsqueeze_BAND1"]])
        elif str(state_element_id) == "TROPOMIRADSQUEEZEBAND2":
            return np.array([o_uip.tropomiPars["radsqueeze_BAND2"]])
        elif str(state_element_id) == "TROPOMIRADSQUEEZEBAND3":
            return np.array([o_uip.tropomiPars["radsqueeze_BAND3"]])
        elif str(state_element_id) == "TROPOMIRADSQUEEZEBAND7":
            return np.array([o_uip.tropomiPars["radsqueeze_BAND7"]])
        elif str(state_element_id) == "TROPOMIRINGSFBAND1":
            return np.array([o_uip.tropomiPars["ring_sf_BAND1"]])
        elif str(state_element_id) == "TROPOMIRINGSFBAND2":
            return np.array([o_uip.tropomiPars["ring_sf_BAND2"]])
        elif str(state_element_id) == "TROPOMIRINGSFBAND3":
            return np.array([o_uip.tropomiPars["ring_sf_BAND3"]])
        elif str(state_element_id) == "TROPOMIRINGSFBAND7":
            return np.array([o_uip.tropomiPars["ring_sf_BAND7"]])
        elif str(state_element_id) == "TROPOMIRESSCALEO0BAND2":
            return np.array([o_uip.tropomiPars["resscale_O0_BAND2"]])
        elif str(state_element_id) == "TROPOMIRESSCALEO1BAND2":
            return np.array([o_uip.tropomiPars["resscale_O1_BAND2"]])
        elif str(state_element_id) == "TROPOMIRESSCALEO2BAND2":
            return np.array([o_uip.tropomiPars["resscale_O2_BAND2"]])
        elif str(state_element_id) == "TROPOMIRESSCALEO0BAND3":
            return np.array([o_uip.tropomiPars["resscale_O0_BAND3"]])
        elif str(state_element_id) == "TROPOMIRESSCALEO1BAND3":
            return np.array([o_uip.tropomiPars["resscale_O1_BAND3"]])
        elif str(state_element_id) == "TROPOMIRESSCALEO2BAND3":
            return np.array([o_uip.tropomiPars["resscale_O2_BAND3"]])
        elif str(state_element_id) == "TROPOMITEMPSHIFTBAND3":
            return np.array([o_uip.tropomiPars["temp_shift_BAND3"]])
        elif str(state_element_id) == "TROPOMIRESSCALEO0BAND7":
            return np.array([o_uip.tropomiPars["resscale_O0_BAND7"]])
        elif str(state_element_id) == "TROPOMIRESSCALEO1BAND7":
            return np.array([o_uip.tropomiPars["resscale_O1_BAND7"]])
        elif str(state_element_id) == "TROPOMIRESSCALEO2BAND7":
            return np.array([o_uip.tropomiPars["resscale_O2_BAND7"]])
        elif str(state_element_id) == "TROPOMITEMPSHIFTBAND7":
            return np.array([o_uip.tropomiPars["temp_shift_BAND7"]])
        elif str(state_element_id) == "TROPOMITEMPSHIFTBAND3TIGHT":
            return np.array([o_uip.tropomiPars["temp_shift_BAND3"]])
        elif str(state_element_id) == "TROPOMICLOUDSURFACEALBEDO":
            return np.array([o_uip.tropomiPars["cloud_Surface_Albedo"]])
        # Check if it is a column
        try:
            return self.rf_uip.atmosphere_column(str(state_element_id))
        except ValueError:
            pass
        raise RuntimeError(f"Don't recognize {state_element_id}")

    def full_state_value_str(
        self, state_element_id: StateElementIdentifier
    ) -> str | None:
        raise NotImplementedError()

    def full_state_step_initial_value(
        self, state_element_id: StateElementIdentifier, use_map: bool = False
    ) -> ForwardModelGridArray:
        raise NotImplementedError()

    def full_state_true_value(
        self, state_element_id: StateElementIdentifier
    ) -> ForwardModelGridArray | None:
        raise NotImplementedError()

    def full_state_retrieval_initial_value(
        self, state_element_id: StateElementIdentifier
    ) -> ForwardModelGridArray:
        raise NotImplementedError()

    def full_state_apriori_value(
        self, state_element_id: StateElementIdentifier, use_map: bool = False
    ) -> ForwardModelGridArray:
        raise NotImplementedError()

    def full_state_apriori_covariance(
        self, state_element_id: StateElementIdentifier
    ) -> ForwardModelGrid2dArray:
        raise NotImplementedError()


class CurrentStateDict(CurrentState):
    """Implementation of CurrentState that just takes a dictionary of
    state elements and list of retrieval elements.

    """

    def __init__(
        self,
        state_element_dict: dict[
            StateElementIdentifier | str, np.ndarray | list[float] | float
        ],
        retrieval_element: list[StateElementIdentifier | str],
    ):
        """This takes a dictionary from state element name to value,
        and a list of retrieval elements. This is useful for creating
        unit tests that don't depend on other objects.

        Note both self.state_element_dict and self.retrieval_element
        can be updated if desired, if for whatever reason we want to
        add/tweak the data.

        """
        super().__init__()
        self._state_element_dict = {
            (
                k
                if isinstance(k, StateElementIdentifier)
                else StateElementIdentifier(k)
            ): v
            for (k, v) in state_element_dict.items()
        }
        self._retrieval_element = [
            i if isinstance(i, StateElementIdentifier) else StateElementIdentifier(i)
            for i in retrieval_element
        ]

    @property
    def state_element_dict(self) -> dict:
        return self._state_element_dict

    @state_element_dict.setter
    def state_element_dict(self, val: dict) -> None:
        self._state_element_dict = val
        # Clear cache, we need to regenerate these after update
        self.clear_cache()

    @property
    def retrieval_state_element_id(self) -> list[StateElementIdentifier]:
        return self._retrieval_element

    @retrieval_state_element_id.setter
    def retrieval_state_element_id(self, val: list[StateElementIdentifier]) -> None:
        self._retrieval_element = val
        # Clear cache, we need to regenerate these after update
        self.clear_cache()

    @property
    def systematic_state_element_id(self) -> list[StateElementIdentifier]:
        raise NotImplementedError()

    @property
    def full_state_element_id(self) -> list[StateElementIdentifier]:
        return list(self.state_element_dict.keys())

    @property
    def sounding_metadata(self) -> SoundingMetadata:
        raise NotImplementedError()

    def full_state_element(
        self, state_element_id: StateElementIdentifier
    ) -> StateElement:
        raise NotImplementedError()

    def full_state_spectral_domain_wavelength(
        self, state_element_id: StateElementIdentifier
    ) -> np.ndarray | None:
        raise NotImplementedError()

    def full_state_value(
        self, state_element_id: StateElementIdentifier
    ) -> ForwardModelGridArray:
        v = self.state_element_dict[state_element_id]
        if isinstance(v, np.ndarray):
            return v
        elif isinstance(v, list):
            return np.array(v)
        return np.array(
            [
                v,
            ]
        )

    def full_state_step_initial_value(
        self, state_element_id: StateElementIdentifier, use_map: bool = False
    ) -> ForwardModelGridArray:
        raise NotImplementedError()

    def full_state_value_str(
        self, state_element_id: StateElementIdentifier
    ) -> str | None:
        raise NotImplementedError()

    def full_state_true_value(
        self, state_element_id: StateElementIdentifier
    ) -> ForwardModelGridArray | None:
        raise NotImplementedError()

    def full_state_retrieval_initial_value(
        self, state_element_id: StateElementIdentifier
    ) -> ForwardModelGridArray:
        raise NotImplementedError()

    def full_state_apriori_value(
        self, state_element_id: StateElementIdentifier, use_map: bool = False
    ) -> ForwardModelGridArray:
        raise NotImplementedError()

    def full_state_apriori_covariance(
        self, state_element_id: StateElementIdentifier
    ) -> ForwardModelGrid2dArray:
        raise NotImplementedError()


class CurrentStateStateInfoOld(CurrentState):
    """Implementation of CurrentState that uses our StateInfoOld. This is
    the way the actual full retrieval works.

    This class uses the old py-retrieve code we wrapped, and was used for
    initial development, and testing our implementation of CurrentStateStateInfo.
    Unless you are testing old code, you don't actually want to use this class,
    but instead use CurrentStateStateInfo.
    """

    def __init__(
        self,
        state_info: StateInfoOld | None,
        retrieval_info: RetrievalInfo | None = None,
        step_directory: str | os.PathLike[str] | None = None,
    ) -> None:
        """I think we'll want to get some of the logic in
        RetrievalInfo into this class, I'm not sure that we want this
        as separate. But for now, include this as an argument.

        The retrieval_state_element_override is an odd argument, it
        overrides the retrieval_state_element_id in RetrievalInfo with a
        different set. It isn't clear why this is handled this way -
        why doesn't RetrievalInfo just figure out the right
        retrieval_state_element_id list?  But for now, do it the same way
        as py-retrieve. This seems to only be used in the
        RetrievalStrategyStepBT - I'm guessing this was a kludge put
        in to support this retrieval step.

        In addition, the CurrentState can also be used when we are
        calculating the "systematic" jacobian. This create a
        StateVector with a different set of state elements.  This
        isn't used to do a retrieval, but rather to just calculate a
        jacobian.  If do_systematic is set to True, we use this values
        instead.
        """

        super().__init__()
        self._state_info = state_info
        self.retrieval_state_element_override: None | list[StateElementIdentifier] = (
            None
        )
        self.do_systematic = False
        self._step_directory = (
            Path(step_directory) if step_directory is not None else None
        )

    @property
    def state_info(self) -> StateInfoOld:
        if self._state_info is None:
            from refractor.old_py_retrieve_wrapper import StateInfoOld  # type: ignore

            self._state_info = StateInfoOld()
        return self._state_info

    def current_state_override(
        self,
        do_systematic: bool,
        retrieval_state_element_override: None | list[StateElementIdentifier],
    ) -> CurrentState:
        res = copy(self)
        res.retrieval_state_element_override = retrieval_state_element_override
        res.do_systematic = do_systematic
        res.clear_cache()
        return res

    @property
    def initial_guess(self) -> RetrievalGridArray:
        # Not sure about systematic handling here. I think this is all
        # zeros, not sure if that is right or not.
        if self._retrieval_info is None:
            raise RuntimeError("_retrieval_info is None")
        if self.do_systematic:
            return copy(
                self._retrieval_info.retrieval_info_systematic().initialGuessList
            )
        else:
            return copy(self._retrieval_info.initial_guess_list)

    @property
    def initial_guess_fm(self) -> ForwardModelGridArray:
        # TODO
        # Not clear why this isn't directly calculated from initial_guess, but the
        # values are different. For now, just have this and we can try to sort this out
        if self._retrieval_info is None:
            raise RuntimeError("_retrieval_info is None")
        if self.do_systematic:
            return copy(
                self._retrieval_info.retrieval_info_systematic().initialGuessListFM
            )
        else:
            return copy(self._retrieval_info.initial_guess_list_fm)

    @property
    def constraint_matrix(self) -> RetrievalGrid2dArray:
        # Not sure about systematic handling here.
        if self.do_systematic:
            return np.zeros((1, 1))
        else:
            if self._retrieval_info is None:
                raise RuntimeError("_retrieval_info is None")
            return copy(self._retrieval_info.constraint_matrix)

    @property
    def sqrt_constraint(self) -> RetrievalGridArray:
        # Not sure about systematic handling here.
        if self.do_systematic:
            return np.eye(len(self.initial_guess))
        else:
            return (mpy.sqrt_matrix(self.constraint_matrix)).transpose()

    @property
    def apriori(self) -> RetrievalGridArray:
        # Not sure about systematic handling here.
        if self.do_systematic:
            return np.zeros((len(self.initial_guess),))
        else:
            if self.retrieval_info is None:
                raise RuntimeError("retrieval_info is None")
            return copy(self.retrieval_info.apriori)

    @property
    def apriori_fm(self) -> ForwardModelGridArray:
        # Not sure about systematic handling here.
        if self.do_systematic:
            return np.zeros((len(self.initial_guess_fm),))
        else:
            if self.retrieval_info is None:
                raise RuntimeError("retrieval_info is None")
            return copy(self.retrieval_info.apriori_fm)

    @property
    def true_value(self) -> RetrievalGridArray:
        if self.retrieval_info is None:
            raise RuntimeError("retrieval_info is None")
        return copy(self.retrieval_info.true_value)

    @property
    def true_value_fm(self) -> ForwardModelGridArray:
        if self.retrieval_info is None:
            raise RuntimeError("retrieval_info is None")
        return copy(self.retrieval_info.true_value_fm)

    @property
    def basis_matrix(self) -> np.ndarray | None:
        if self.do_systematic:
            return None
        else:
            if self.retrieval_info is None:
                raise RuntimeError("retrieval_info is None")
            return self.retrieval_info.basis_matrix

    @property
    def map_to_parameter_matrix(self) -> np.ndarray | None:
        if self.do_systematic:
            return None
        else:
            if self.retrieval_info is None:
                raise RuntimeError("retrieval_info is None")
            return self.retrieval_info.map_to_parameter_matrix

    @property
    def step_directory(self) -> Path:
        if self._step_directory is None:
            raise RuntimeError("Set step directory first")
        return self._step_directory

    @step_directory.setter
    def step_directory(self, val: Path) -> None:
        self._step_directory = val

    @property
    def propagated_qa(self) -> PropagatedQA:
        return self.state_info.propagated_qa

    @property
    def brightness_temperature_data(self) -> dict:
        return self.state_info.brightness_temperature_data

    @property
    def updated_fm_flag(self) -> ForwardModelGridArray:
        return self.retrieval_info.retrieval_info_obj.doUpdateFM

    def update_state(
        self,
        results_list: np.ndarray,
        do_not_update: list[StateElementIdentifier],
        retrieval_config: RetrievalConfiguration | MeasurementId,
        step: int,
    ) -> None:
        self.state_info.update_state(
            self.retrieval_info, results_list, do_not_update, retrieval_config, step
        )

    def update_full_state_element(
        self,
        state_element_id: StateElementIdentifier,
        current: np.ndarray | None = None,
        apriori: np.ndarray | None = None,
        step_initial: np.ndarray | None = None,
        retrieval_initial: np.ndarray | None = None,
        true_value: np.ndarray | None = None,
    ) -> None:
        selem = self.state_info.state_element(state_element_id)
        selem.update_state(
            current, apriori, step_initial, retrieval_initial, true_value
        )

    @property
    def retrieval_info(self) -> RetrievalInfo:
        if self._retrieval_info is None:
            raise RuntimeError("Need to set self._retrieval_info")
        return self._retrieval_info

    @retrieval_info.setter
    def retrieval_info(self, val: RetrievalInfo) -> None:
        self._retrieval_info = val
        # Clear cache, we need to regenerate these after update
        self.clear_cache()

    @property
    def retrieval_state_element_id(self) -> list[StateElementIdentifier]:
        if self.retrieval_state_element_override is not None:
            return self.retrieval_state_element_override
        if self.retrieval_info is None:
            raise RuntimeError("retrieval_info is None")
        if self.do_systematic:
            return [
                StateElementIdentifier(i) for i in self.retrieval_info.species_names_sys
            ]
        return [StateElementIdentifier(i) for i in self.retrieval_info.species_names]

    @property
    def systematic_state_element_id(self) -> list[StateElementIdentifier]:
        if self.retrieval_state_element_override is not None:
            return self.retrieval_state_element_override
        if self.retrieval_info is None:
            raise RuntimeError("retrieval_info is None")
        return [
            StateElementIdentifier(i) for i in self.retrieval_info.species_names_sys
        ]

    @property
    def full_state_element_id(self) -> list[StateElementIdentifier]:
        return [i.name for i in self.state_info.state_element_list()]

    @property
    def fm_sv_loc(self) -> dict[StateElementIdentifier, tuple[int, int]]:
        if self.retrieval_info is None:
            raise RuntimeError("retrieval_info is None")
        if self._fm_sv_loc is None:
            self._fm_sv_loc = {}
            self._fm_state_vector_size = 0
            for state_element_id in self.retrieval_state_element_id:
                if self.do_systematic:
                    plen = self.retrieval_info.species_list_sys.count(
                        str(state_element_id)
                    )
                else:
                    plen = self.retrieval_info.species_list_fm.count(
                        str(state_element_id)
                    )

                # As a convention, if plen is 0 py-retrieve pads this
                # to 1, although the state vector isn't actually used
                # - it does get set. I think this is to avoid having a
                # 0 size state vector. We should perhaps clean this up
                # as some point, there isn't anything wrong with a
                # zero size state vector (although this might have
                # been a problem with IDL). But for now, use the
                # py-retrieve convention. This can generally only
                # happen if we have retrieval_state_element_override
                # set, i.e., we are doing RetrievalStrategyStepBT.
                if plen == 0:
                    plen = 1
                self._fm_sv_loc[state_element_id] = (self._fm_state_vector_size, plen)
                self._fm_state_vector_size += plen
        return self._fm_sv_loc

    @property
    def sounding_metadata(self) -> SoundingMetadata:
        return self.state_info.sounding_metadata()

    def full_state_element(
        self, state_element_id: StateElementIdentifier
    ) -> StateElement:
        raise NotImplementedError()

    def full_state_element_old(
        self, state_element_id: StateElementIdentifier
    ) -> StateElementOld:
        return self.state_info.state_element(state_element_id)

    def full_state_spectral_domain_wavelength(
        self, state_element_id: StateElementIdentifier
    ) -> np.ndarray | None:
        selem = self.state_info.state_element(state_element_id)
        return selem.spectral_domain_wavelength

    def full_state_value(
        self, state_element_id: StateElementIdentifier
    ) -> ForwardModelGridArray:
        selem = self.state_info.state_element(state_element_id)
        return copy(selem.value)

    def full_state_step_initial_value(
        self, state_element_id: StateElementIdentifier, use_map: bool = False
    ) -> ForwardModelGridArray:
        selem = self.state_info.state_element(state_element_id, step="initial")
        return copy(selem.value)

    def full_state_value_str(
        self, state_element_id: StateElementIdentifier
    ) -> str | None:
        selem = self.state_info.state_element(state_element_id)
        if not hasattr(selem, "value_str"):
            return None
        return str(selem.value_str)

    def full_state_true_value(
        self, state_element_id: StateElementIdentifier
    ) -> ForwardModelGridArray | None:
        selem = self.state_info.state_element(state_element_id, step="true")
        return copy(selem.value)

    def full_state_retrieval_initial_value(
        self, state_element_id: StateElementIdentifier
    ) -> ForwardModelGridArray:
        selem = self.state_info.state_element(state_element_id, step="initialInitial")
        return copy(selem.value)

    def full_state_apriori_value(
        self, state_element_id: StateElementIdentifier, use_map: bool = False
    ) -> ForwardModelGridArray:
        selem = self.state_info.state_element(state_element_id)
        return copy(selem.apriori_value)

    def full_state_apriori_covariance(
        self, state_element_id: StateElementIdentifier
    ) -> ForwardModelGrid2dArray:
        selem = self.state_info.state_element(state_element_id)
        return copy(selem.sa_covariance)

    @property
    def retrieval_sv_loc(self) -> dict[StateElementIdentifier, tuple[int, int]]:
        if self.retrieval_info is None:
            raise RuntimeError("retrieval_info is None")
        if self._retrieval_sv_loc is None:
            self._retrieval_sv_loc = {}
            self._retrieval_state_vector_size = 0
            for state_element_id in self.retrieval_state_element_id:
                if self.do_systematic:
                    plen = self.retrieval_info.species_list_sys.count(
                        str(state_element_id)
                    )
                else:
                    plen = self.retrieval_info.species_list.count(str(state_element_id))

                # As a convention, if plen is 0 py-retrieve pads this
                # to 1, although the state vector isn't actually used
                # - it does get set. I think this is to avoid having a
                # 0 size state vector. We should perhaps clean this up
                # as some point, there isn't anything wrong with a
                # zero size state vector (although this might have
                # been a problem with IDL). But for now, use the
                # py-retrieve convention. This can generally only
                # happen if we have retrieval_state_element_override
                # set, i.e., we are doing RetrievalStrategyStepBT.
                if plen == 0:
                    plen = 1
                self._retrieval_sv_loc[state_element_id] = (
                    self._retrieval_state_vector_size,
                    plen,
                )
                self._retrieval_state_vector_size += plen
        return self._retrieval_sv_loc

    def state_mapping(
        self, state_element_id: StateElementIdentifier
    ) -> rf.StateMapping:
        selem = self.full_state_element_old(state_element_id)
        mtype = selem.map_type
        if mtype == "linear":
            return rf.StateMappingLinear()
        elif mtype == "log":
            return rf.StateMappingLog()
        else:
            raise RuntimeError(f"Don't recognize mtype {mtype}")

    def pressure_list(
        self, state_element_id: StateElementIdentifier
    ) -> RetrievalGridArray | None:
        selem = self.full_state_element_old(state_element_id)
        if hasattr(selem, "pressureList"):
            return selem.pressureList
        return None

    def pressure_list_fm(
        self, state_element_id: StateElementIdentifier
    ) -> ForwardModelGridArray | None:
        selem = self.full_state_element_old(state_element_id)
        if hasattr(selem, "pressureListFM"):
            return selem.pressureListFM
        return None

    def altitude_list(
        self, state_element_id: StateElementIdentifier
    ) -> RetrievalGridArray | None:
        selem = self.full_state_element_old(state_element_id)
        if hasattr(selem, "altitudeList"):
            return selem.altitudeList
        return None

    def altitude_list_fm(
        self, state_element_id: StateElementIdentifier
    ) -> ForwardModelGridArray | None:
        selem = self.full_state_element_old(state_element_id)
        if hasattr(selem, "altitudeListFM"):
            return selem.altitudeListFM
        return None

    # Some of these arguments may get put into the class, but for now have explicit
    # passing of these
    def get_initial_guess(
        self,
        current_strategy_step: CurrentStrategyStep,
        error_analysis: ErrorAnalysis,
        retrieval_config: RetrievalConfiguration,
    ) -> None:
        """Set retrieval_info, errorInitial and errorCurrent for the current step."""
        # Chicken and egg problem with circular dependency, so we just import this
        # when we need it here
        from .retrieval_info import RetrievalInfo

        for selem_id in current_strategy_step.retrieval_elements:
            selem = self.full_state_element_old(selem_id)
            selem.update_initial_guess(current_strategy_step)

        if False:
            self.state_info.snapshot_to_file(
                f"state_step{current_strategy_step.strategy_step.step_number}_1.txt"
            )
            error_analysis.snapshot_to_file(
                f"error_analysis_step{current_strategy_step.strategy_step.step_number}_1.txt"
            )

        # TODO, we'd like to get rid of RetrievalInfo
        self._retrieval_info = RetrievalInfo(
            error_analysis,
            Path(retrieval_config["speciesDirectory"]),
            current_strategy_step,
            self,
        )

        # Isn't really clear why RetrievalInfo is different, but for
        # now this update is needed. Without this we get different results.
        # We should sort trough this, we don't want to go through RetrievalInfo
        # for this.

        # Update state with initial guess so that the initial guess is
        # mapped properly, if doing a retrieval, for each retrieval step.
        nparm = self._retrieval_info.n_totalParameters
        if nparm > 0:
            xig = self._retrieval_info.initial_guess_list[0:nparm]
            self.update_state(
                xig,
                [],
                retrieval_config,
                current_strategy_step.strategy_step.step_number,
            )
        self.clear_cache()

    def notify_update_target(
        self,
        measurement_id: MeasurementId,
        retrieval_config: RetrievalConfiguration,
        strategy: MusesStrategy,
        observation_handle_set: ObservationHandleSet,
    ) -> None:
        self.state_info.notify_update_target(
            measurement_id, retrieval_config, strategy, observation_handle_set
        )

    @property
    def state_element_handle_set(self) -> StateElementHandleSet:
        raise NotImplementedError()

    @property
    def state_element_handle_set_old(self) -> StateElementHandleSetOld:
        return self.state_info.state_element_handle_set

    def notify_start_step(
        self,
        current_strategy_step: CurrentStrategyStep | None,
        error_analysis: ErrorAnalysis,
        retrieval_config: RetrievalConfiguration,
        skip_initial_guess_update: bool = False,
    ) -> None:
        # current_strategy_step being None means we are past the last step in our
        # MusesStrategy, so we just skip doing anything
        if current_strategy_step is not None:
            self.step_directory = (
                retrieval_config["run_dir"]
                / f"Step{current_strategy_step.strategy_step.step_number:02d}_{current_strategy_step.strategy_step.step_name}"
            )
            if not skip_initial_guess_update:
                self.state_info.next_state_to_current()
                self.state_info.copy_current_initial()
            # Doesn't seem right that we update initial *before* doing get_initial_guess,
            # but that seems to be what happens
            self.get_initial_guess(
                current_strategy_step, error_analysis, retrieval_config
            )
            # Save some data needed by notify_step_solution
            self._retrieval_config = retrieval_config
            self._do_not_update = current_strategy_step.retrieval_elements_not_updated
            self._step = current_strategy_step.strategy_step.step_number

    def notify_step_solution(self, xsol: RetrievalGridArray) -> None:
        """Called with MusesStrategy has found a solution for the current step."""
        self.state_info.update_state(
            self.retrieval_info,
            xsol,
            self._do_not_update,
            self._retrieval_config,
            self._step,
        )

    def notify_start_retrieval(
        self,
        current_strategy_step: CurrentStrategyStep | None,
        retrieval_config: RetrievalConfiguration,
    ) -> None:
        if current_strategy_step is not None:
            self.step_directory = (
                retrieval_config["run_dir"]
                / f"Step{current_strategy_step.strategy_step.step_number:02d}_{current_strategy_step.strategy_step.step_name}"
            )
            self.state_info.restart()
            self.state_info.copy_current_initialInitial()
            self.state_info.copy_current_initial()


__all__ = [
    "CurrentState",
    "CurrentStateUip",
    "CurrentStateDict",
    "CurrentStateStateInfoOld",
    "SoundingMetadata",
    "PropagatedQA",
]

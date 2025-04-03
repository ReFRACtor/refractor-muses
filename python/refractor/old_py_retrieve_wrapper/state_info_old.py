# Don't both typechecking the file. It is long and complicated, and we will replace most
# of it in a bit. Silence mypy, just so we don't get a lot of noise in the output
# type: ignore

from __future__ import annotations  # We can remove this when we upgrade to python 3.9
import abc
from refractor.muses import (
    PriorityHandleSet,
    StateElementIdentifier,
    SoundingMetadata,
    order_species,
    mpy_radiance_from_observation_list,
    TesFile,
    PropagatedQA,
)
import refractor.muses.muses_py as mpy  # type: ignore
from loguru import logger
import copy
from pathlib import Path
import refractor.framework as rf  # type: ignore
import numpy as np
from pprint import pprint
import sys
import os
from typing import Any
import typing

if typing.TYPE_CHECKING:
    from .muses import (
        RetrievalInfo,
        RetrievalConfiguration,
        ObservationHandleSet,
        CurrentStrategyStep,
        MeasurementId,
        MusesObservation,
        InstrumentIdentifier,
        MusesStrategy,
    )


class StateElementOld(object, metaclass=abc.ABCMeta):
    """Muses-py tends to call everything in its state "species",
    although in a few places things things are called
    "parameters". These should really be thought of as "things that go
    into a StateVector". So we refer to these as StateElements, which also
    parallels the species we retrieve on referred to as RetrievalElements.

    We try to treat all these things as identical at some level, but
    there is some behavior that is species dependent. We'll sort that
    out, and try to figure out the right design here.

    These get referenced by a "name", usually called a "species_name"
    in the muses-py code. The StateInfoOld can be used to look these
    up."""

    def __init__(self, state_info: StateInfoOld, name: StateElementIdentifier):
        self._name = name
        self.state_info = state_info
        self.mapType = "linear"

    @property
    def name(self) -> StateElementIdentifier:
        return self._name

    @property
    def map_type(self) -> str:
        """Mapping type. This can hopefully go away and we just use rf.StateMapping, but
        this is needed for now."""
        return self.mapType

    @property
    def retrieval_config(self) -> RetrievalConfiguration:
        if self.state_info.retrieval_config is None:
            raise RuntimeError("Need to call notify_update_target first")
        return self.state_info.retrieval_config

    def sa_covariance(self):
        """Return sa covariance matrix, and also pressure. This is what
        ErrorAnalysis needs."""
        raise NotImplementedError()

    def sa_cross_covariance(self, selem2: StateElementOld):
        """Return the cross covariance matrix with selem 2. This returns None
        if there is no cross covariance."""
        return None

    def clone_for_other_state(self):
        """StateInfoOld has copy_current_initialInitial and copy_current_initial.
        The simplest thing would be to just copy the current dict. However,
        the muses-py StateElementOld maintain their state outside of the classes in
        various dicts in StateInfoOld (probably left over from IDL). So we have
        this function. For ReFRACtor StateElementOld, this should just be a copy of
        StateElementOld, but for muses-py we return None. The copy_current_initialInitial
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
    def spectral_domain(self) -> rf.SpectralDomain | None:
        """For StateElementWithFrequency, this returns the frequency associated
        with it. For all other StateElement, just return None."""
        return None

    @property
    def spectral_domain_wavelength(self) -> np.ndarray | None:
        """Short cut to return the spectral domain in units of nm."""
        sd = self.spectral_domain
        if sd is None:
            return None
        return sd.convert_wave(rf.Unit("nm"))

    @abc.abstractproperty
    def value(self) -> np.ndarray:
        raise NotImplementedError

    @property
    @abc.abstractproperty
    def apriori_value(self) -> np.ndarray:
        raise NotImplementedError


class RetrievableStateElementOld(StateElementOld):
    """This has additional functionality to have a StateElement be retrievable,
    so things like having a priori and initial guess needed in a retrieval. Most
    StateElements are retrievable, but not all - so we separate out the
    functionality."""

    @abc.abstractmethod
    def update_state_element(
        self,
        retrieval_info: RetrievalInfo,
        results_list: np.ndarray,
        retrieval_config: RetrievalConfiguration | MeasurementId,
        step: int,
        do_update_fm: np.ndarray,
    ):
        """Update the state element based on retrieval results."""
        raise NotImplementedError

    @abc.abstractmethod
    def update_state(
        self,
        current: np.ndarray | None = None,
        apriori: np.ndarray | None = None,
        initial: np.ndarray | None = None,
        initial_initial: np.ndarray | None = None,
        true: np.ndarray | None = None,
    ):
        """We have a few places where we want to update a state element other than
        update_initial_guess. This function updates each of the various values passed in.
        A value of 'None' (the default) means skip updating that part of the state."""
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


class StateElementHandleOld(object, metaclass=abc.ABCMeta):
    """Return 3 StateElement objects, for initialInitial, initial and
    current state. For many classes, this will just be a deepcopy of the same class,
    but at least for now the older muses-py code stores these in different places.
    We can perhaps change this interface in the future as we move out the old muses-py
    stuff, it is sort of odd to create these 3 things at once. But we'll match
    what muses-py does for now.

    Note StateElementHandleOld can assume that they are called for the same target, until
    notify_update_target is called. So if it makes sense, these objects can do internal
    caching for things that don't change when the target being retrieved is the same from
    one call to the next."""

    def notify_update_target(self, state_info: StateInfoOld):
        """Clear any caching associated with assuming the target being retrieved is fixed"""
        # Default is to do nothing
        pass

    @abc.abstractmethod
    def state_element_object(
        self, state_info: StateInfoOld, name: StateElementIdentifier
    ) -> tuple[
        bool,
        tuple[StateElementOld, StateElementOld, StateElementOld, StateElementOld]
        | None,
    ]:
        raise NotImplementedError


class StateElementHandleSetOld(PriorityHandleSet):
    """This maps a species name to the SpeciesOrParametersState object that handles
    it.

    Note StatElementHandle can assume that they are called for the same target, until
    notify_update_target is called. So if it makes sense, these objects can do internal
    caching for things that don't change when the target being retrieved is the same from
    one call to the next."""

    def state_element_object(
        self, state_info: StateInfoOld, name: StateElementIdentifier
    ) -> tuple[StateElementOld, StateElementOld, StateElementOld, StateElementOld]:
        return self.handle(state_info, name)

    def notify_update_target(self, state_info: StateInfoOld):
        """Clear any caching associated with assuming the target being retrieved is fixed"""
        for p in sorted(self.handle_set.keys(), reverse=True):
            for h in self.handle_set[p]:
                h.notify_update_target(state_info)

    def handle_h(
        self,
        h: StateElementHandleOld,
        state_info: StateInfoOld,
        name: StateElementIdentifier,
    ) -> tuple[
        bool,
        tuple[StateElementOld, StateElementOld, StateElementOld, StateElementOld]
        | None,
    ]:
        return h.state_element_object(state_info, name)


class StateInfoOld:
    """State Info during a retrieval - so what gets passed between RetrievalStrategyStep.

    Note that StateInfoOld can assume that they are called for the same target, until
    notify_update_target is called. So if it makes sense, these objects can do internal
    caching for things that don't change when the target being retrieved is the same from
    one call to the next.

    This is also true for all the StateElementHandleOld we have.

    """

    def __init__(self) -> None:
        self.state_element_handle_set = copy.deepcopy(
            StateElementHandleSetOld.default_handle_set()
        )
        self.state_info_dict: dict[str, Any] = {}
        self.initialInitial: dict[str, StateElementOld] = {}
        self.initial: dict[str, StateElementOld] = {}
        self.current: dict[str, StateElementOld] = {}
        self.true: dict[str, StateElementOld] = {}

        self.reset_state_value: dict[StateElementIdentifier, np.ndarray] = {}
        self.propagated_qa = PropagatedQA()
        self.brightness_temperature_data: dict[int, dict[str, float | None]] = {}

        # Odds and ends that are currently in the StateInfoOld. Doesn't exactly have
        # to do with the state, but we don't have another place for these.
        # Perhaps SoundingMetadata can migrate into its own thing, and these can
        # get moved over to there.
        self._tai_time = -999.0
        self._utc_time = ""
        self._sounding_id = ""
        self.retrieval_config: RetrievalConfiguration | None = None
        self.state_element_handle_set.notify_update_target(self)

    def notify_update_target(
        self,
        measurement_id: MeasurementId,
        retrieval_config: RetrievalConfiguration,
        strategy: MusesStrategy,
        observation_handle_set: ObservationHandleSet,
    ) -> None:
        logger.debug(f"Call to {self.__class__.__name__}::notify_update")
        self.state_info_dict = {}
        self.initialInitial = {}
        self.initial = {}
        self.current = {}
        self.true = {}

        self.reset_state_value = {}
        self.propagated_qa = PropagatedQA()
        self.brightness_temperature_data = {}

        # Odds and ends that are currently in the StateInfoOld. Doesn't exactly have
        # to do with the state, but we don't have another place for these.
        # Perhaps SoundingMetadata can migrate into its own thing, and these can
        # get moved over to there.
        self._tai_time = -999.0
        self._utc_time = ""
        self._sounding_id = ""
        self.retrieval_config = retrieval_config
        self.state_element_handle_set.notify_update_target(self)
        curdir = os.getcwd()
        try:
            os.chdir(retrieval_config["run_dir"])
            self.init_state(
                measurement_id,
                observation_handle_set,
                strategy.retrieval_elements,
                strategy.error_analysis_interferents,
                strategy.instrument_name,
                retrieval_config["run_dir"],
                osp_dir=retrieval_config.osp_dir,
            )
        finally:
            os.chdir(curdir)

    def init_state(
        self,
        measurement_id: MeasurementId,
        observation_handle_set: ObservationHandleSet,
        retrieval_elements_all: list[StateElementIdentifier],
        error_analysis_interferents_all: list[StateElementIdentifier],
        instrument_name_all: list[InstrumentIdentifier],
        run_dir: Path,
        osp_dir: Path | None = None,
    ):
        odict = {}
        for iname in instrument_name_all:
            odict[str(iname)] = observation_handle_set.observation(
                iname, None, None, None
            )
        if osp_dir is None:
            osp_dir = run_dir.parent / "OSP"
        self.state_info_dict = self.script_retrieval_setup_ms(
            measurement_id, odict, osp_dir
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
                list(
                    set([str(i) for i in error_analysis_interferents_all])
                    | set([str(i) for i in retrieval_elements_all])
                )
            )
        }
        self.state_info_dict = mpy.states_initial_update(
            self.state_info_dict,
            fake_table,
            rad,
            [str(i) for i in instrument_name_all],
        )

        # Read some metadata that isn't already available
        f = TesFile(run_dir / "DateTime.asc")
        self._tai_time = float(f["TAI_Time_of_ZPD"])
        self._utc_time = f["UTC_Time"]
        self._sounding_id = measurement_id["key"]

    # Ignore typing for now. This is a long complicated function that we will
    # rewrite in a bit
    @typing.no_type_check
    def script_retrieval_setup_ms(
        self,
        measurement_id: MeasurementId,
        odict: dict[str, MusesObservation],
        osp_dir: Path,
    ):
        instruments = odict.keys()
        o_airs = None
        o_cris = None
        o_omi = None
        o_tropomi = None
        o_tes = None
        o_oco2 = None
        o_stateInfo = None
        if "AIRS" in odict:
            o_airs = odict["AIRS"].muses_py_dict
        if "CRIS" in odict:
            o_cris = odict["CRIS"].muses_py_dict
        if "OMI" in odict:
            o_omi = odict["OMI"].muses_py_dict
        if "TROPOMI" in odict:
            o_tropomi = odict["TROPOMI"].muses_py_dict
        if "TES" in odict:
            o_tes = odict["TES"].muses_py_dict
        if "OCO2" in odict:
            o_oco2 = odict["OCO2"].muses_py_dict

        if "oceanFlag" in measurement_id:
            oceanFlag = int(measurement_id["oceanFlag"])
        elif "OCEANFLAG" in measurement_id:
            oceanFlag = int(measurement_id["OCEANFLAG"])
        else:
            raise RuntimeError(
                "Could not find 'oceanflag' or 'OCEANFLAG' in measurement_id"
            )

        gmao_path = measurement_id.get("GMAO", "../GMAO/")
        gmao_type = measurement_id.get("GMAO_TYPE", "")

        instrument_name = "DUMMY_INSTRUMENT_NAME"
        for ins in ("TES", "AIRS", "CRIS", "OMI", "TROPOMI", "OCO2"):
            if ins in instruments:
                instrument_name = ins
                break
        if instrument_name not in [
            "TES",
            "AIRS",
            "CRIS",
            "OMI",
            "OMPS-NM",
            "TROPOMI",
            "OCO2",
        ]:
            raise RuntimeError(
                "Unknown instrument. Must have TES, AIRS, CRIS, OMI, OMPS-NM, TROPOMI or OCO2 specified as instrument in Measurement ID file"
            )

        o_latitude = 0
        o_longitude = 0
        o_dateStruct = {}

        # allow utcTime or time (tai)
        try:
            dateStruct = mpy.utc_from_string(
                measurement_id[f"{instrument_name}_utcTime"]
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
                np.float64(measurement_id[f"{instrument_name}_time"])
            )

        try:
            o_latitude = float(measurement_id[f"{instrument_name}_latitude"])
            o_longitude = float(measurement_id[f"{instrument_name}_longitude"])
        except KeyError:
            o_latitude = float(measurement_id[f"{instrument_name}_Latitude"])
            o_longitude = float(measurement_id[f"{instrument_name}_Longitude"])

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
                o_tropomi["Earth_Radiance"]["ObservationTable"]["TerrainHeight"][i] = (
                    surfaceAltitude
                )
        elif "OCO2" in instruments:
            surfaceAltitude = np.mean(o_oco2["surface_altitude_m"])
        else:
            raise RuntimeError(
                "instruments ~= TES, AIRS, OMI, CRIS, TROPOMI, OCO2, Need surface altitude."
            )

        setupFilename = (
            Path(measurement_id["initialGuessSetupDirectory"])
            / "L2_Setup_Control_Initial.asc"
        )

        constraintFlag = False
        stateInitial = mpy.get_state_initial(
            str(setupFilename),
            str(osp_dir),
            o_latitude,
            o_longitude,
            o_dateStruct,
            measurement_id,
            constraintFlag,
            surfaceAltitude,
            oceanFlag,
            gmao_path,
            gmao_type,
        )

        setupFilename = (
            Path(measurement_id["initialGuessSetupDirectory"])
            / "L2_Setup_Control_Constraint.asc"
        )

        # state for constraint
        constraintFlag = True
        stateConstraint = mpy.get_state_initial(
            str(setupFilename),
            str(osp_dir),
            o_latitude,
            o_longitude,
            o_dateStruct,
            measurement_id,
            constraintFlag,
            surfaceAltitude,
            oceanFlag,
            gmao_path,
            gmao_type,
        )

        if "TES" in instruments:
            (stateInitial, stateConstraint) = self._clean_up_dictionaries(
                instrument_name.lower(), stateInitial, stateConstraint, o_tes
            )

        if "CRIS" in instruments:
            (stateInitial, stateConstraint) = self._clean_up_dictionaries(
                instrument_name.lower(), stateInitial, stateConstraint, o_cris
            )

        if "AIRS" in instruments:
            (stateInitial, stateConstraint) = self._clean_up_dictionaries(
                instrument_name.lower(), stateInitial, stateConstraint, o_airs
            )

        if "OMI" in instruments:
            temp_end = 2  # Dejian revised Nov 22, 2016 for adding the functionality of radiance calibration run
            if int(measurement_id["OMI_Rad_calRun_flag"]) == 1:
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
                    o_omi["Earth_Radiance"]["ObservationTable"]["SpacecraftAltitude"]
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
                    o_omi["Earth_Radiance"]["ObservationTable"]["RelativeAzimuthAngle"][
                        1 : temp_end + 1
                    ]
                ),
                "vza_uv2": np.mean(
                    o_omi["Earth_Radiance"]["ObservationTable"]["ViewingZenithAngle"][
                        1 : temp_end + 1
                    ]
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
                    stateInitial["current"]["omi"][ttState[ii]] = omi_pars[ttState[ii]]
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
                o_tropomi["Earth_Radiance"]["ObservationTable"]["SpacecraftAltitude"]
            )
            tropomi_pars["xsecscaling"] = np.float64(1.0)

            ii_tropomi = -1
            for i in range(8):
                if (
                    f"BAND{i + 1}_MonthlyMinimumSurfaceReflectance"
                    in o_tropomi["SurfaceAlbedo"]
                ):
                    ii_tropomi += 1
                    tropomi_pars[f"surface_albedo_BAND{i + 1}"] = o_tropomi[
                        "SurfaceAlbedo"
                    ][f"BAND{i + 1}_MonthlyMinimumSurfaceReflectance"]
                    tropomi_pars[f"surface_albedo_slope_BAND{i + 1}"] = np.float64(0.0)
                    tropomi_pars[f"surface_albedo_slope_order2_BAND{i + 1}"] = (
                        np.float64(0.0)
                    )
                    tropomi_pars[f"solarshift_BAND{i + 1}"] = np.float64(0.0)
                    tropomi_pars[f"radianceshift_BAND{i + 1}"] = np.float64(0.0)
                    tropomi_pars[f"radsqueeze_BAND{i + 1}"] = np.float64(0.0)
                    tropomi_pars[f"temp_shift_BAND{i + 1}"] = np.float64(1.0)
                    tropomi_pars[f"ring_sf_BAND{i + 1}"] = np.float64(1.9)
                    tropomi_pars[f"resscale_O0_BAND{i + 1}"] = np.float64(1.0)
                    tropomi_pars[f"resscale_O1_BAND{i + 1}"] = np.float64(0.0)
                    tropomi_pars[f"resscale_O2_BAND{i + 1}"] = np.float64(0.0)
                    tropomi_pars[f"sza_BAND{i + 1}"] = o_tropomi["Earth_Radiance"][
                        "ObservationTable"
                    ]["SolarZenithAngle"][ii_tropomi]
                    tropomi_pars[f"raz_BAND{i + 1}"] = o_tropomi["Earth_Radiance"][
                        "ObservationTable"
                    ]["RelativeAzimuthAngle"][ii_tropomi]
                    tropomi_pars[f"vza_BAND{i + 1}"] = o_tropomi["Earth_Radiance"][
                        "ObservationTable"
                    ]["ViewingZenithAngle"][ii_tropomi]
                    tropomi_pars[f"sca_BAND{i + 1}"] = o_tropomi["Earth_Radiance"][
                        "ObservationTable"
                    ]["ScatteringAngle"][ii_tropomi]

            ttState = []
            if stateConstraint is not None:
                ttState = list(stateConstraint["current"]["tropomi"].keys())

            if stateInitial is not None:
                ttState = list(stateInitial["current"]["tropomi"].keys())

            for ii in range(0, len(ttState)):
                if stateConstraint is not None:
                    stateConstraint["current"]["tropomi"][ttState[ii]] = tropomi_pars[
                        ttState[ii]
                    ]

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
                    stateInitial["current"]["oco2"][ttState[ii]] = o_oco2[ttState[ii]]
        if len(stateInitial) > 0:
            # set surface type for products
            oceanString = ["LAND", "OCEAN"]
            stateInitial["surfaceType"] = oceanString[
                oceanFlag
            ]  # The type of oceanFlag should be int here so we can use it as an index.
            stateInitial["current"]["surfaceType"] = oceanString[oceanFlag]

        if len(stateConstraint) > 0:
            #  set surface type for products
            oceanString = ["LAND", "OCEAN"]
            stateConstraint["surfaceType"] = oceanString[oceanFlag]
            stateConstraint["current"]["surfaceType"] = oceanString[oceanFlag]

        o_stateInfo = stateInitial
        o_stateInfo["ch3ohtype"] = stateConstraint["ch3ohtype"]
        oceanString = ["LAND", "OCEAN"]
        o_stateInfo["surfaceType"] = oceanString[oceanFlag]

        o_stateInfo["initialInitial"] = copy.deepcopy(stateInitial["current"])
        o_stateInfo["initial"] = copy.deepcopy(stateInitial["current"])
        o_stateInfo["current"] = copy.deepcopy(stateInitial["current"])
        o_stateInfo["constraint"] = copy.deepcopy(stateConstraint["current"])

        return o_stateInfo

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
        # various dicts in StateInfoOld (probably left over from IDL). So we have
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
        # various dicts in StateInfoOld (probably left over from IDL). So we have
        # this function. For ReFRACtor StateElement, this should just be a copy of
        # StateElement, but for muses-py we return None

        for k, selem in self.current.items():
            scopy = selem.clone_for_other_state()
            if scopy is not None:
                self.initial[k] = scopy

    def restart(self):
        """Called when we restart a retrieval."""
        self.reset_state_value = {}

    def next_state_to_current(self):
        # Not clear at all why this copy is necessary. Sort of accidentally discovered
        # this, because we use to copy to create the "next_state". Instead we now
        # have the simpler reset_state_value which just changes the items on "do_not_update".
        # I'm guessing something in the code somewhere is holding on a reference to something
        # in current that is getting updated. Since we are moving to do away with the whole
        # complicated state_info_dict we won't spend too much time tracking this down.
        # But without this copy, we get a different result from our various sounding tests.
        self.state_info_dict["current"] = copy.deepcopy(self.state_info_dict["current"])
        for selem_id, v in self.reset_state_value.items():
            selem = self.state_element(selem_id)
            selem.update_state(current=v)
        self.reset_state_value = {}

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
        of data found in the StateInfoOld. But for now go ahead and support this."""
        return {
            "surface_albedo_uv1": self.state_element(
                StateElementIdentifier("OMISURFACEALBEDOUV1")
            ).value[0],
            "surface_albedo_uv2": self.state_element(
                StateElementIdentifier("OMISURFACEALBEDOUV2")
            ).value[0],
            "surface_albedo_slope_uv2": self.state_element(
                StateElementIdentifier("OMISURFACEALBEDOSLOPEUV2")
            ).value[0],
            "nradwav_uv1": self.state_element(
                StateElementIdentifier("OMINRADWAVUV1")
            ).value[0],
            "nradwav_uv2": self.state_element(
                StateElementIdentifier("OMINRADWAVUV2")
            ).value[0],
            "odwav_uv1": self.state_element(
                StateElementIdentifier("OMIODWAVUV1")
            ).value[0],
            "odwav_uv2": self.state_element(
                StateElementIdentifier("OMIODWAVUV2")
            ).value[0],
            "odwav_slope_uv1": self.state_element(
                StateElementIdentifier("OMIODWAVSLOPEUV1")
            ).value[0],
            "odwav_slope_uv2": self.state_element(
                StateElementIdentifier("OMIODWAVSLOPEUV2")
            ).value[0],
            "ring_sf_uv1": self.state_element(
                StateElementIdentifier("OMIRINGSFUV1")
            ).value[0],
            "ring_sf_uv2": self.state_element(
                StateElementIdentifier("OMIRINGSFUV2")
            ).value[0],
            "cloud_fraction": self.state_element(
                StateElementIdentifier("OMICLOUDFRACTION")
            ).value[0],
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
        do_not_update: list[StateElementIdentifier],
        retrieval_config: RetrievalConfiguration | MeasurementId,
        step: int,
    ):
        """Note this updates the current state, and also creates a "next_state".
        The difference is that current gets all the changes found in the
        results_list, but next_state only gets the elements updated that aren't
        listed in do_not_update. This allows things to be done in a particular
        retrieval step, but not actually propagated to the next retrieval step.
        Call next_state_to_current() to update the current state with the
        next_state (i.e., remove the changes for things listed in do_not_update)."""
        self.reset_state_value = {}

        do_update_fm = np.zeros(retrieval_info.n_totalParametersFM)

        # Save values before updating for things we want to reset
        for selm in do_not_update:
            if str(selm) in retrieval_info.species_names:
                self.reset_state_value[selm] = self.state_element(selm).value.copy()
        for state_element_name in retrieval_info.species_names:
            self.state_element(
                StateElementIdentifier(state_element_name)
            ).update_state_element(
                retrieval_info,
                results_list,
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
        # Tracked this down, it is *only* used in error_analysis.py in muses-py.
        # Not clear if this is ever anything other than all ones, we should perhaps
        # check this and see if we need this or not
        retrieval_info.retrieval_dict["doUpdateFM"] = do_update_fm

    def state_element_list(self, step="current"):
        """Return the list of state elements that we already have in StateInfoOld.
        Note that state_element creates this on first use, the list returned is
        those state elements that have already been created."""
        if step == "current":
            return list(self.current.values())
        elif step == "initialInitial":
            return list(self.initialInitial.values())
        elif step == "initial":
            return list(self.initial.values())
        elif step == "true":
            return list(self.true.values())
        else:
            raise RuntimeError("step must be initialInitial, initial, true, or current")

    def snapshot_to_file(self, fname):
        '''py-retrieve is big on having functions with unintended side effects. It can
        be hard to determine what changes when. This writes a complete text dump of
        this object, which we can then diff against other snapshots to see what has
        changed.'''
        with np.printoptions(precision=None, threshold=sys.maxsize):
            with open(fname, "w") as fh:
                pprint(self.state_info_dict, stream=fh)
            

    def state_element(self, name: StateElementIdentifier | str, step="current"):
        """Return the state element with the given name."""
        # We create the StateElement objects on first use
        if str(name) not in self.current:
            (
                self.initialInitial[str(name)],
                self.initial[str(name)],
                self.current[str(name)],
                self.true[str(name)],
            ) = self.state_element_handle_set.state_element_object(
                self,
                name
                if isinstance(name, StateElementIdentifier)
                else StateElementIdentifier(name),
            )
        if step == "initialInitial":
            return self.initialInitial[str(name)]
        elif step == "initial":
            return self.initial[str(name)]
        elif step == "current":
            return self.current[str(name)]
        elif step == "true":
            return self.true[str(name)]
        else:
            raise RuntimeError("step must be initialInitial, initial, true, or current")


__all__ = [
    "StateElementOld",
    "StateElementHandleOld",
    "RetrievableStateElementOld",
    "StateElementHandleSetOld",
    "StateInfoOld",
]

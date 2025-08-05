# Not really sure about the design or these, or how much to pull out.
# For now, just create these as separate classes.
from __future__ import annotations
import refractor.framework as rf  # type: ignore
import refractor.muses.muses_py as mpy  # type: ignore
from .state_element import (
    StateElementHandle,
    StateElement,
    StateElementHandleSet,
)
from .current_state import FullGridMappedArray, RetrievalGridArray, RetrievalGrid2dArray
from .state_element_osp import StateElementOspFile
from .identifier import StateElementIdentifier, RetrievalType
from .current_state_state_info import h_old
import numpy as np
from pathlib import Path
from loguru import logger
import typing
from typing import Any

if typing.TYPE_CHECKING:
    from .muses_observation import ObservationHandleSet, MeasurementId
    from .muses_strategy import MusesStrategy, CurrentStrategyStep
    from .retrieval_configuration import RetrievalConfiguration
    from .current_state import SoundingMetadata


class StateElementFreqShared(StateElementOspFile):
    """Clumsy class, hopefully will go away. It isn't clear what is in
    common between EMIS and CLOUDEXT, we'll try putting that stuff here."""

    def __init__(
        self,
        state_element_id: StateElementIdentifier,
        pressure_list_fm: FullGridMappedArray | None,
        value_fm: FullGridMappedArray,
        constraint_vector_fm: FullGridMappedArray,
        latitude: float,
        surface_type: str,
        species_directory: Path,
        covariance_directory: Path,
        spectral_domain: rf.SpectralDomain | None = None,
        selem_wrapper: Any | None = None,
        cov_is_constraint: bool = False,
        poltype: str | None = None,
        poltype_used_constraint: bool = True,
        diag_cov: bool = False,
        diag_directory: Path | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        super().__init__(
            state_element_id,
            pressure_list_fm,
            value_fm,
            constraint_vector_fm,
            latitude,
            surface_type,
            species_directory,
            covariance_directory,
            spectral_domain=spectral_domain,
            selem_wrapper=selem_wrapper,
            cov_is_constraint=cov_is_constraint,
            poltype=poltype,
            poltype_used_constraint=poltype_used_constraint,
            diag_cov=diag_cov,
            diag_directory=diag_directory,
            metadata=metadata,
        )
        self.microwindows: list[dict] = []

    def _fill_in_constraint(self) -> None:
        if self._constraint_matrix is not None:
            return
        self._constraint_matrix = self.osp_species_reader.read_constraint_matrix(
            self.state_element_id,
            self.retrieval_type,
            self.basis_matrix.shape[0],
            poltype=self.poltype if self.poltype_used_constraint else None,
            pressure_list=self.pressure_list,
            pressure_list_fm=self.pressure_list_fm,
            map_to_parameter_matrix=self.map_to_parameter_matrix,
        ).view(RetrievalGrid2dArray)

    def _fill_in_apriori(self) -> None:
        if self._apriori_cov_fm is not None:
            return
        super()._fill_in_apriori()
        # TODO - This is done to match the existing data. We'll want to remove this
        # at some point, but it will update the expected results. Go ahead and
        # hold this fixed for now, so we can make just this one change and know it
        # is why the output changes. We round this to 32 bit, and then back to 64 bit float
        self._apriori_cov_fm = self._apriori_cov_fm.astype(np.float32).astype(np.float64)
        
    def _fill_in_state_mapping(self) -> None:
        super()._fill_in_state_mapping()
        # TODO Fix this
        # Note very confusingly this is actually the spectral domain wavelength
        # instead of pressure. We should fix this naming at some point, but for
        # now match what py-retrieve expects.
        if self._retrieved_this_step:
            assert self.spectral_domain is not None
            self._pressure_list_fm = self.spectral_domain.data.view(FullGridMappedArray)
        else:
            self._pressure_list_fm = None

class StateElementEmis(StateElementFreqShared):
    def _fill_in_state_mapping_retrieval_to_fm(self) -> None:
        if self._state_mapping_retrieval_to_fm is not None:
            return
        self._fill_in_state_mapping()
        assert self.spectral_domain is not None
        # TODO See about doing this directly
        wflag = mpy.mw_frequency_needed(
            self.microwindows,
            self.spectral_domain.data,
        )
        ind = np.array([i for i in np.nonzero(wflag)[0]])
        self._state_mapping_retrieval_to_fm = rf.StateMappingBasisMatrix.from_x_subset(
            self.spectral_domain.data.view(FullGridMappedArray), ind, log_interp=False
        )
        if self._sold is None:
            raise RuntimeError("Need sold")
        t = self._sold.state_mapping_retrieval_to_fm
        # if(np.abs(self._state_mapping_retrieval_to_fm.basis_matrix - t.basis_matrix).max() > 1e-12):
        #    breakpoint()
        # Temp, until we sort this out
        self._state_mapping_retrieval_to_fm = t
        # Line 1240 state_element_old for emis
        # Line 1366 state_element_old for cloud

    def notify_step_solution(
        self, xsol: RetrievalGridArray, retrieval_slice: slice | None
    ) -> None:
        # We've already called notify_parameter_update, so no need to update
        # self._value here
        if self._sold is not None:
            self._sold.notify_step_solution(xsol, retrieval_slice)
        # Default is that the next initial value is whatever the solution was from
        # this step. But skip if we are on the not updated list
        self._next_step_initial_fm = None

        if retrieval_slice is not None:
            # Note we really are replacing value_fm as a FullGridMappedArray with
            # the results from the RetrievalGridArray solution. This is what we want
            # to do after a retrieval step. We have the "fprime" here just to make
            # that explicit that this is what we are intending on doing here. So
            # to_fmprime returns a FullGridMappedArrayFromRetGrid, but we then use
            # that as a FullGridMappedArray.

            res = (
                xsol[retrieval_slice]
                .view(RetrievalGridArray)
                .to_fmprime(self.state_mapping_retrieval_to_fm, self.state_mapping)
                .view(FullGridMappedArray)
            )
            # Not sure of the logic here. I think this is a way to calculate
            # self.updated_fm_flag
            updfl = np.abs(np.sum(self.map_to_parameter_matrix, axis=1)) >= 1e-10
            if self._value_fm is None:
                raise RuntimeError("self._value_fm can't be none")
            self._value_fm.view(np.ndarray)[updfl] = res[updfl]
            # Not sure what exactly is going on here, but somehow the
            # value is being changed before we start this step. Just
            # steal from old element for now, we'll need to sort this out.
            # Note that this is out of sync with self._step_initial_fm, which
            # actually seems to be the case. Probably a mistake, but duplicate for now
            if self._sold is None:
                raise RuntimeError("Need sold")
            self._value_fm = self._sold.value_fm
            if not self._initial_guess_not_updated:
                self._next_step_initial_fm = self._value_fm.copy()

    def notify_start_step(
        self,
        current_strategy_step: CurrentStrategyStep,
        retrieval_config: RetrievalConfiguration,
        skip_initial_guess_update: bool = False,
    ) -> None:
        super().notify_start_step(
            current_strategy_step,
            retrieval_config,
            skip_initial_guess_update,
        )
        # Grab microwindow information, needed in later calculations
        self.microwindows = []
        for swin in current_strategy_step.spectral_window_dict.values():
            self.microwindows.extend(swin.muses_microwindows())
        # Filter out the UV windows, this just isn't wanted in
        # mw_frequency_needed
        self.microwindows = [mw for mw in self.microwindows if "UV" not in mw["filter"]]
        t = self._sold.value_fm.copy()
        if np.abs(self._value_fm - t).max() > 1e-12:
            #breakpoint()
            self._value_fm = t

    @property
    def updated_fm_flag(self) -> FullGridMappedArray:
        # This one is different, we'll need to track down why.
        if self._sold is None:
            raise RuntimeError("Need sold")
        res = self._sold.updated_fm_flag
        self._check_result(res, "updated_fm_flag")
        return res

class StateElementCloudExt(StateElementFreqShared):
    def _fill_in_state_mapping_retrieval_to_fm(self) -> None:
        if self._state_mapping_retrieval_to_fm is not None:
            return
        self._fill_in_state_mapping()
        assert self.spectral_domain is not None
        wflag = mpy.mw_frequency_needed(
            self.microwindows,
            self.spectral_domain.data,
            self.retrieval_type,
            self.freq_mode,
        )
        ind = np.array([i for i in np.nonzero(wflag)[0]])
        self._state_mapping_retrieval_to_fm = rf.StateMappingBasisMatrix.from_x_subset(
            self.spectral_domain.data.view(FullGridMappedArray), ind, log_interp=False
        )
        # Line 1240 state_element_old for emis
        # Line 1366 state_element_old for cloud

    def notify_start_step(
        self,
        current_strategy_step: CurrentStrategyStep,
        retrieval_config: RetrievalConfiguration,
        skip_initial_guess_update: bool = False,
    ) -> None:
        super().notify_start_step(
            current_strategy_step,
            retrieval_config,
            skip_initial_guess_update,
        )
        # Grab microwindow information, needed in later calculations
        self.microwindows = []
        for swin in current_strategy_step.spectral_window_dict.values():
            self.microwindows.extend(swin.muses_microwindows())
        # Filter out the UV windows, this just isn't wanted in
        # mw_frequency_needed
        self.microwindows = [mw for mw in self.microwindows if "UV" not in mw["filter"]]
        self.freq_mode = retrieval_config["CLOUDEXT_IGR_Min_Freq_Spacing"]
        self.is_bt_ig_refine = current_strategy_step.retrieval_type == RetrievalType(
            "bt_ig_refine"
        )
        if(self.is_bt_ig_refine):
            self._value_fm = self._sold.value_fm[0, :].copy()

    def notify_step_solution(
        self, xsol: RetrievalGridArray, retrieval_slice: slice | None
    ) -> None:
        # We've already called notify_parameter_update, so no need to update
        # self._value here
        if self._sold is not None:
            self._sold.notify_step_solution(xsol, retrieval_slice)
        # Default is that the next initial value is whatever the solution was from
        # this step. But skip if we are on the not updated list
        self._next_step_initial_fm = None

        # We have some odd logic in StateElementOld for cloudEffExt for the
        # bt_ig_refine step. We will need to duplicate this, but short term
        # just punt and use the old value. Note that we end up at about
        # line 264, there cloudEffExt is set to an average value, but we will
        # need to work through this. Probably special handling on
        # notify_step_solution for cloudEffExt, for bt_ig_refine step.
        # Also CLOUDEXT seems to be a sort of alias for cloudEffExt, but we don't
        # have that fully supported yet - should probably add that
        if self.is_bt_ig_refine and self._sold is not None:
            self._value_fm = self._sold.value_fm[0, :].copy()
            assert self._value_fm is not None
            self._next_step_initial_fm = self._value_fm.copy()
        elif retrieval_slice is not None:
            # Note we really are replacing value_fm as a FullGridMappedArray with
            # the results from the RetrievalGridArray solution. This is what we want
            # to do after a retrieval step. We have the "fprime" here just to make
            # that explicit that this is what we are intending on doing here. Sp
            # to_fmprime returns a FullGridMappedArrayFromRetGrid, but we then use
            # that as a FullGridMappedArray.

            res = (
                xsol[retrieval_slice]
                .view(RetrievalGridArray)
                .to_fmprime(self.state_mapping_retrieval_to_fm, self.state_mapping)
                .view(FullGridMappedArray)
            )
            # Not sure of the logic here. I think this is a way to calculate
            # self.updated_fm_flag
            if self._value_fm is None:
                raise RuntimeError("value_fm can't be none")
            updfl = np.abs(np.sum(self.map_to_parameter_matrix, axis=1)) >= 1e-10
            self._value_fm.view(np.ndarray)[updfl] = res[updfl]
            if not self._initial_guess_not_updated:
                self._next_step_initial_fm = self._value_fm.copy()

    @property
    def updated_fm_flag(self) -> FullGridMappedArray:
        # Might be only bt_ig_refine that is different?
        if(not self.is_bt_ig_refine):
            return super().updated_fm_flag
        if self._sold is None:
            raise RuntimeError("Need sold")
        res = self._sold.updated_fm_flag
        self._check_result(res, "updated_fm_flag")
        return res


                
class StateElementEmisHandle(StateElementHandle):
    def __init__(
        self,
        hold: Any | None = None,
    ) -> None:
        self.obj_cls = StateElementEmis
        self.sid = StateElementIdentifier("EMIS")
        self.hold = hold
        self.measurement_id: MeasurementId | None = None
        self.retrieval_config: RetrievalConfiguration | None = None
        self.cov_is_constraint = False

    def notify_update_target(
        self,
        measurement_id: MeasurementId,
        retrieval_config: RetrievalConfiguration,
        strategy: MusesStrategy,
        observation_handle_set: ObservationHandleSet,
        sounding_metadata: SoundingMetadata,
    ) -> None:
        self.measurement_id = measurement_id
        self.retrieval_config = retrieval_config
        self.strategy = strategy
        self.observation_handle_set = observation_handle_set
        self.sounding_metadata = sounding_metadata

    def state_element(
        self, state_element_id: StateElementIdentifier
    ) -> StateElement | None:
        if state_element_id != self.sid:
            return None

        if self.measurement_id is None or self.retrieval_config is None:
            raise RuntimeError("Need to call notify_update_target first")
        if self.hold is not None:
            sold = self.hold.state_element(state_element_id)
        else:
            # Currently require sold
            return None
        pressure_level = None
        value_fm = sold.value_fm
        spectral_domain = sold.spectral_domain
        try:
            constraint_vector_fm = sold.constraint_vector_fm
        except NotImplementedError:
            constraint_vector_fm = value_fm.copy()
        res = self.obj_cls.create_from_handle(
            state_element_id,
            pressure_level,
            value_fm,
            constraint_vector_fm,
            self.measurement_id,
            self.retrieval_config,
            self.strategy,
            self.observation_handle_set,
            self.sounding_metadata,
            spectral_domain=spectral_domain,
            selem_wrapper=sold,
            cov_is_constraint=self.cov_is_constraint,
            metadata={
                "camel_distance": sold.metadata["camel_distance"],
                "prior_source": sold.metadata["prior_source"],
            },
        )
        if res is not None:
            logger.debug(f"New Creating {self.obj_cls.__name__} for {state_element_id}")
        return res


class StateElementCloudExtHandle(StateElementHandle):
    def __init__(
        self,
        hold: Any | None = None,
    ) -> None:
        self.obj_cls = StateElementCloudExt
        self.sid = StateElementIdentifier("CLOUDEXT")
        self.hold = hold
        self.measurement_id: MeasurementId | None = None
        self.retrieval_config: RetrievalConfiguration | None = None
        self.cov_is_constraint = False

    def notify_update_target(
        self,
        measurement_id: MeasurementId,
        retrieval_config: RetrievalConfiguration,
        strategy: MusesStrategy,
        observation_handle_set: ObservationHandleSet,
        sounding_metadata: SoundingMetadata,
    ) -> None:
        self.measurement_id = measurement_id
        self.retrieval_config = retrieval_config
        self.strategy = strategy
        self.observation_handle_set = observation_handle_set
        self.sounding_metadata = sounding_metadata

    def state_element(
        self, state_element_id: StateElementIdentifier
    ) -> StateElement | None:
        if state_element_id != self.sid:
            return None

        if self.measurement_id is None or self.retrieval_config is None:
            raise RuntimeError("Need to call notify_update_target first")
        if self.hold is not None:
            sold = self.hold.state_element(state_element_id)
        else:
            # currently required
            return None
        pressure_level = None
        value_fm = sold.value_fm
        spectral_domain = sold.spectral_domain
        try:
            constraint_vector_fm = sold.constraint_vector_fm
        except NotImplementedError:
            constraint_vector_fm = value_fm.copy()
        # For some reason these are 2d. I'm pretty sure this is just some left
        # over thing or other, anything other than row 0 isn't used. For nowm
        # make 1 d so we don't need some special handling. We can revisit if
        # we actually determine this should be 2d
        if len(value_fm.shape) == 2:
            value_fm = value_fm[0, :]
        if len(constraint_vector_fm.shape) == 2:
            constraint_vector_fm = constraint_vector_fm[0, :]

        res = self.obj_cls.create_from_handle(
            state_element_id,
            pressure_level,
            value_fm,
            constraint_vector_fm,
            self.measurement_id,
            self.retrieval_config,
            self.strategy,
            self.observation_handle_set,
            self.sounding_metadata,
            spectral_domain=spectral_domain,
            selem_wrapper=sold,
            cov_is_constraint=self.cov_is_constraint,
        )
        if res is not None:
            logger.debug(f"New Creating {self.obj_cls.__name__} for {state_element_id}")
        return res


StateElementHandleSet.add_default_handle(
    StateElementEmisHandle(h_old), priority_order=1
)
StateElementHandleSet.add_default_handle(
    StateElementCloudExtHandle(h_old), priority_order=1
)

__all__ = [
    "StateElementEmis",
    "StateElementCloudExt",
]

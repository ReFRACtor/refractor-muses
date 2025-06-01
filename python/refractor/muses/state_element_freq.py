# Not really sure about the design or these, or how much to pull out.
# For now, just create these as separate classes.
from __future__ import annotations
from .state_element import (
    StateElementHandle,
    StateElement,
    StateElementHandleSet,
)
from .current_state import FullGridMappedArray, RetrievalGridArray
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

    def _fill_in_constraint(self) -> None:
        if self._constraint_matrix is not None:
            return
        if self._sold is None:
            raise RuntimeError("Need sold")
        # TODO Short term work around this, until we are ready to support this.
        self._constraint_matrix = self._sold.constraint_matrix

    def _fill_in_state_mapping(self) -> None:
        super()._fill_in_state_mapping()
        if self._sold is None:
            raise RuntimeError("Need sold")
        # This actually looks like the frequency instead of pressure. But it is
        # what the muses-py code expects
        self._pressure_list_fm = self._sold.pressure_list_fm


class StateElementEmis(StateElementFreqShared):
    def __init__(
        self,
        state_element_id: StateElementIdentifier,
        pressure_list_fm: FullGridMappedArray | None,
        value_fm: FullGridMappedArray | None,
        constraint_vector_fm: FullGridMappedArray | None,
        latitude: float,
        surface_type: str,
        species_directory: Path,
        covariance_directory: Path,
        selem_wrapper: Any | None = None,
        cov_is_constraint: bool = False,
        poltype: str | None = None,
        poltype_used_constraint: bool = True,
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
            selem_wrapper,
            cov_is_constraint,
            poltype,
            poltype_used_constraint,
        )
        if self._sold is None:
            raise RuntimeError("Need sold")
        # TODO Need better way to handle this
        self._metadata["camel_distance"] = self._sold.metadata["camel_distance"]
        self._metadata["prior_source"] = self._sold.metadata["prior_source"]

    @property
    def value_fm(self) -> FullGridMappedArray:
        if self._sold is None:
            raise RuntimeError("Need sold")
        # Keep running into issues here, just punt for now
        self._value_fm = self._sold.value_fm.copy()
        res = self._value_fm
        if res is None:
            raise RuntimeError("_value_fm shouldn't be None")
        self._check_result(res, "value_fm")
        return res

    @property
    def step_initial_fm(self) -> FullGridMappedArray:
        if self._sold is None:
            raise RuntimeError("Need sold")
        # Punt
        self._step_initial_fm = self._sold.step_initial_fm.copy()
        if self._step_initial_fm is None:
            raise RuntimeError("_step_initial_fm shouldn't be None")
        res = self._step_initial_fm
        self._check_result(res, "step_initial_fm")
        return res

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
            updfl = np.abs(np.sum(self.map_to_parameter_matrix, axis=1)) >= 1e-10
            if self._value_fm is None:
                raise RuntimeError("self._value_fm can't be none")
            self._value_fm.view(np.ndarray)[updfl] = res[updfl]
            if not self._initial_guess_not_updated:
                self._next_step_initial_fm = self._value_fm.copy()
            # Not sure what exactly is going on here, but somehow the
            # value is being changed before we start this step. Just
            # steal from old element for now, we'll need to sort this out.
            # Note that this is out of sync with self._step_initial_fm, which
            # actually seems to be the case. Probably a mistake, but duplicate for now
            if self._sold is None:
                raise RuntimeError("Need sold")
            self._value_fm = self._sold.value_fm


class StateElementCloudExt(StateElementFreqShared):
    @property
    def value_fm(self) -> FullGridMappedArray:
        if self.is_bt_ig_refine and self._sold is not None:
            self._value_fm = self._sold._current_state_old.state_value("CLOUDEXT")[0, :]
            assert self._value_fm is not None
            self._next_step_initial_fm = self._value_fm.copy()
        # Keep running into issues here, just punt for now
        if self._sold is None:
            raise RuntimeError("Need sold")
        self._value_fm = self._sold.value_fm[0, :].copy().view(FullGridMappedArray)
        res = self._value_fm
        if res is None:
            raise RuntimeError("_value_fm shouldn't be None")
        self._check_result(res, "value_fm")
        return res

    def notify_start_step(
        self,
        current_strategy_step: CurrentStrategyStep,
        retrieval_config: RetrievalConfiguration,
        skip_initial_guess_update: bool = False,
    ) -> None:
        super().notify_start_step(current_strategy_step, retrieval_config, skip_initial_guess_update)
        self.is_bt_ig_refine = current_strategy_step.retrieval_type == RetrievalType(
            "bt_ig_refine"
        )

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
            self._value_fm = self._sold.value_fm[0, :]
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
                raise RuntimeError("vlaue_fm can't be none")
            updfl = np.abs(np.sum(self.map_to_parameter_matrix, axis=1)) >= 1e-10
            self._value_fm.view(np.ndarray)[updfl] = res[updfl]
            if not self._initial_guess_not_updated:
                self._next_step_initial_fm = self._value_fm.copy()


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
            selem_wrapper=sold,
            cov_is_constraint=self.cov_is_constraint,
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
        try:
            constraint_vector_fm = sold.constraint_vector_fm
        except NotImplementedError:
            constraint_vector_fm = value_fm.copy()
        # For some reason these are 2d. I'm pretty sure this is just some left
        # over thing or other, anything other than row 0 isn't used. For nowm
        # make 1 d so we don't need some special handling. We can revisit if
        # we actually determine this should be 2d
        value_fm = value_fm[0, :]
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

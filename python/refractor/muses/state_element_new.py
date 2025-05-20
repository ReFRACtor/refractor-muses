# Figure out what to call this, or possibly just move stuff to another file. But for
# now have it here to keep stuff together
from __future__ import annotations
from .state_element import (
    StateElementHandle,
    StateElement,
    StateElementHandleSet,
)
from .state_element_osp import StateElementOspFile
from .current_state_state_info import h_old
from .identifier import StateElementIdentifier
from loguru import logger
import typing
from typing import cast

if typing.TYPE_CHECKING:
    from .muses_observation import ObservationHandleSet, MeasurementId
    from .muses_strategy import MusesStrategy
    from .retrieval_configuration import RetrievalConfiguration
    from .state_element_old_wrapper import StateElementOldWrapperHandle
    from .current_state import SoundingMetadata


class StateElementOspFileHandleNew(StateElementHandle):
    def __init__(
        self,
        sid: StateElementIdentifier | None,
        hold: StateElementOldWrapperHandle | None = None,
        cls: type[StateElementOspFile] = StateElementOspFile,
        # cls: type[StateElementImplementation] = StateElementImplementation,
        cov_is_constraint: bool = False,
    ) -> None:
        self.obj_cls = cls
        # Not actually used
        self.sid = sid
        self.hold = hold
        self.measurement_id: MeasurementId | None = None
        self.retrieval_config: RetrievalConfiguration | None = None
        self.cov_is_constraint = cov_is_constraint

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
        from .state_element_old_wrapper import StateElementOldWrapper
        # Issue with a few of the StateElements, punt short term so we can get the
        # rest of stuff working.
        if(str(state_element_id) in ("CLOUDEXT", "cloudEffExt", "HDO","emissivity", "EMIS")):
           return None
           
        # if state_element_id != self.sid:
        #    return None
        if self.measurement_id is None or self.retrieval_config is None:
            raise RuntimeError("Need to call notify_update_target first")
        if self.hold is not None:
            sold = cast(
                StateElementOldWrapper, self.hold.state_element(state_element_id)
            )
        else:
            sold = None
        res = self.obj_cls.create_from_handle(
            state_element_id,
            None,
            self.measurement_id,
            self.retrieval_config,
            self.strategy,
            self.observation_handle_set,
            self.sounding_metadata,
            selem_wrapper=sold,
            cov_is_constraint=self.cov_is_constraint,
            copy_on_first_use=True,
        )
        if res is not None:
            logger.debug(f"New Creating {self.obj_cls.__name__} for {state_element_id}")
        return res


StateElementHandleSet.add_default_handle(
    StateElementOspFileHandleNew(None, h_old), priority_order=0
)

__all__ = ["StateElementOspFileHandleNew",]

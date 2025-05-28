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
from typing import Any

if typing.TYPE_CHECKING:
    from .muses_observation import ObservationHandleSet, MeasurementId
    from .muses_strategy import MusesStrategy
    from .retrieval_configuration import RetrievalConfiguration
    from .current_state import SoundingMetadata


class StateElementOspFileHandleNew(StateElementHandle):
    def __init__(
        self,
        sid: StateElementIdentifier | None,
        hold: Any | None = None,
        cls: type[StateElementOspFile] = StateElementOspFile,
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
        # Issue with a few of the StateElements, punt short term so we can get the
        # rest of stuff working.
        if str(state_element_id) in (
            "CLOUDEXT",
            "EMIS",
        ):
            return None

        # if state_element_id != self.sid:
        #    return None
        if self.measurement_id is None or self.retrieval_config is None:
            raise RuntimeError("Need to call notify_update_target first")
        if self.hold is not None:
            sold = self.hold.state_element(state_element_id)
        else:
            sold = None
        # Determining pressure is spread across a number of muses-py functions. We'll need
        # to track all this down, but short term just get this from the old data
        if self.hold is not None:
            p = self.hold.state_element(StateElementIdentifier("pressure"))
            assert p is not None
            pressure_level = p.value_fm.copy()
        else:
            # For now, just a dummy value
            pressure_level = None
        if sold is not None and sold.value_str is None:
            value_fm = sold.value_fm
            try:
                constraint_vector_fm = sold.constraint_vector_fm
            except NotImplementedError:
                constraint_vector_fm = value_fm.copy()
        else:
            value_fm = None
            constraint_vector_fm = None
        # We can create a subtype if needed, but we just have a handful of these
        # so we'll do this in line for now.
        poltype = None
        poltype_used_constraint = True
        if state_element_id == StateElementIdentifier("NH3"):
            assert sold is not None
            poltype = sold._current_state_old.state_value_str("nh3type")
            if poltype is None:
                poltype = "mod"
            poltype_used_constraint = True
        elif state_element_id == StateElementIdentifier("CH3OH"):
            assert sold is not None
            poltype = sold._current_state_old.state_value_str("ch3ohtype")
            if poltype is None:
                poltype = "mod"
            poltype_used_constraint = True
        elif state_element_id == StateElementIdentifier("HCOOH"):
            assert sold is not None
            poltype = sold._current_state_old.state_value_str("hcoohtype")
            if poltype is None:
                poltype = "mod"
            # Not used in the constraint name
            poltype_used_constraint = False
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
            poltype=poltype,
            poltype_used_constraint=poltype_used_constraint,
        )
        if res is not None:
            logger.debug(f"New Creating {self.obj_cls.__name__} for {state_element_id}")
        return res


StateElementHandleSet.add_default_handle(
    StateElementOspFileHandleNew(None, h_old), priority_order=0
)

__all__ = [
    "StateElementOspFileHandleNew",
]

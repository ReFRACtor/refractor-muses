# Figure out what to call this, or possibly just move stuff to another file. But for
# now have it here to keep stuff together
from __future__ import annotations
from .state_element import (
    StateElementHandle,
    StateElement,
    StateElementHandleSet,
)
from .state_element_osp import StateElementOspFile
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
    ) -> None:
        self.obj_cls = StateElementOspFile
        self._hold: Any | None = None
        self.measurement_id: MeasurementId | None = None
        self.retrieval_config: RetrievalConfiguration | None = None
        self.cov_is_constraint = False

    @property
    def hold(self) -> Any | None:
        # Extra level of indirection to handle cycle in including old_py_retrieve_wrapper
        if self._hold is None:
            from refractor.old_py_retrieve_wrapper import (
                state_element_old_wrapper_handle,
            )

            self._hold = state_element_old_wrapper_handle
        return self._hold

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
        spectral_domain = None
        if sold is not None and sold.value_str is None:
            spectral_domain = sold.spectral_domain
            value_fm = sold.value_fm
            try:
                constraint_vector_fm = sold.constraint_vector_fm
            except NotImplementedError:
                constraint_vector_fm = value_fm.copy()
            if str(state_element_id) in ("CLOUDEXT",):
                # For some reason these are 2d. I'm pretty sure this is just some left
                # over thing or other, anything other than row 0 isn't used. For nowm
                # make 1 d so we don't need some special handling. We can revisit if
                # we actually determine this should be 2d
                value_fm = value_fm[0, :]
                constraint_vector_fm = constraint_vector_fm[0, :]
        else:
            raise RuntimeError("Not currently supported")
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
        diag_cov = False
        # There are a handful of state element that muses-py just "knows" get
        # the apriori covariance from a different diagonal uncertainty file
        # (see get_prior_covariance.py in muses-py, about line 100)
        if str(state_element_id) in ("PCLOUD", "PSUR", "RESSCALE", "TSUR"):
            diag_cov = True
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
            # We are at the point where this isn't needed. We can still pass this
            # in if we want to track down some issue that arises, but don't normally
            # depend on StateElementOld
            # selem_wrapper=sold,
            spectral_domain=spectral_domain,
            cov_is_constraint=self.cov_is_constraint,
            poltype=poltype,
            poltype_used_constraint=poltype_used_constraint,
            diag_cov=diag_cov,
        )
        if res is not None:
            logger.debug(f"New Creating {self.obj_cls.__name__} for {state_element_id}")
        return res


# We want to replace this, but fall back to this for now until everything is in
# place
StateElementHandleSet.add_default_handle(
    StateElementOspFileHandleNew(), priority_order=-1
)

__all__ = [
    "StateElementOspFileHandleNew",
]

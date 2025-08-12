# Figure out what to call this, or possibly just move stuff to another file. But for
# now have it here to keep stuff together
from __future__ import annotations
from .state_element import (
    StateElementHandleSet,
    StateElementWithCreateHandle,
)
from .state_element_osp import StateElementOspFile
from .identifier import StateElementIdentifier
import typing
from typing import Any

if typing.TYPE_CHECKING:
    from .muses_observation import ObservationHandleSet, MeasurementId
    from .muses_strategy import MusesStrategy
    from .retrieval_configuration import RetrievalConfiguration
    from .current_state import SoundingMetadata, FullGridMappedArray


class StateElementOldInitialValue(StateElementOspFile):
    """This class uses the old muses-py StateElementOldWrapper to get the initial
    guess for the StateElement. We will replace this with our own code,
    but this is a nice class for being able to compare against while we are
    developing our new StateElement."""

    @classmethod
    def _setup_create(
        cls,
        sid: StateElementIdentifier | None,
        retrieval_config: RetrievalConfiguration,
        sounding_metadata: SoundingMetadata,
        measurement_id: MeasurementId | None = None,
        strategy: MusesStrategy | None = None,
        observation_handle_set: ObservationHandleSet | None = None,
        selem_wrapper: Any | None = None,
    ) -> tuple[
        StateElementIdentifier,
        FullGridMappedArray | None,
        FullGridMappedArray | None,
        dict[str, Any],
    ]:
        from refractor.old_py_retrieve_wrapper import state_element_old_wrapper_handle

        if sid is None:
            return StateElementIdentifier("Dummy"), None, None, {}
        sold = state_element_old_wrapper_handle.state_element(sid)
        if sold is None:
            return sid, None, None, {}
        spectral_domain = None
        spectral_domain = sold.spectral_domain
        value_fm = sold.value_fm
        try:
            constraint_vector_fm = sold.constraint_vector_fm
        except NotImplementedError:
            constraint_vector_fm = None
        if str(sid) in ("CLOUDEXT",):
            # For some reason these are 2d. I'm pretty sure this is just some left
            # over thing or other, anything other than row 0 isn't used. For nowm
            # make 1 d so we don't need some special handling. We can revisit if
            # we actually determine this should be 2d
            value_fm = value_fm[0, :]
            constraint_vector_fm = constraint_vector_fm[0, :]
        # We can create a subtype if needed, but we just have a handful of these
        # so we'll do this in line for now.
        poltype = None
        poltype_used_constraint = True
        if sid == StateElementIdentifier("NH3"):
            poltype = sold._current_state_old.state_value_str("nh3type")
            if poltype is None:
                poltype = "mod"
            poltype_used_constraint = True
        elif sid == StateElementIdentifier("CH3OH"):
            poltype = sold._current_state_old.state_value_str("ch3ohtype")
            if poltype is None:
                poltype = "mod"
            poltype_used_constraint = True
        elif sid == StateElementIdentifier("HCOOH"):
            poltype = sold._current_state_old.state_value_str("hcoohtype")
            if poltype is None:
                poltype = "mod"
            # Not used in the constraint name
            poltype_used_constraint = False
        diag_cov = False
        # There are a handful of state element that muses-py just "knows" get
        # the apriori covariance from a different diagonal uncertainty file
        # (see get_prior_covariance.py in muses-py, about line 100)
        if str(sid) in ("PCLOUD", "PSUR", "RESSCALE", "TSUR"):
            diag_cov = True
        kwarg = {
            "spectral_domain": spectral_domain,
            "poltype": poltype,
            "poltype_used_constraint": poltype_used_constraint,
            "diag_cov": diag_cov,
        }
        return sid, value_fm, constraint_vector_fm, kwarg


# We want to replace this, but fall back to this for now until everything is in
# place
StateElementHandleSet.add_default_handle(
    StateElementWithCreateHandle(None, StateElementOldInitialValue),
    priority_order=-1,
)

__all__ = [
    "StateElementOldInitialValue",
]

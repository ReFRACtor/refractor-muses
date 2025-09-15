# Figure out what to call this, or possibly just move stuff to another file. But for
# now have it here to keep stuff together
from __future__ import annotations
from .state_element import (
    StateElementHandleSet,
    StateElementWithCreateHandle,
)
from .state_element_osp import StateElementOspFile, OspSetupReturn
from .identifier import StateElementIdentifier
import typing
from typing import Any

if typing.TYPE_CHECKING:
    pass


class StateElementOldInitialValue(StateElementOspFile):
    """This class uses the old muses-py StateElementOldWrapper to get the initial
    guess for the StateElement. We will replace this with our own code,
    but this is a nice class for being able to compare against while we are
    developing our new StateElement."""

    @classmethod
    # type: ignore[override]
    def _setup_create(
        cls,
        sid: StateElementIdentifier,
        **kwarg: Any,
    ) -> OspSetupReturn | None:
        from refractor.old_py_retrieve_wrapper import state_element_old_wrapper_handle

        sold = state_element_old_wrapper_handle.state_element(sid)
        if sold is None:
            return None
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
        create_kwarg = {
            "spectral_domain": spectral_domain,
            "poltype": poltype,
            "poltype_used_constraint": poltype_used_constraint,
            "diag_cov": diag_cov,
        }
        return OspSetupReturn(
            value_fm=value_fm,
            constraint_vector_fm=constraint_vector_fm,
            create_kwargs=create_kwarg,
        )


# This was used when developing the various StateElement. We don't normally use this,
# but leave around since it might be useful if we run into some kind of an issue where
# we need to go back to the old muses-py initial guess stuff to diagnose
#StateElementHandleSet.add_default_handle(
#    StateElementWithCreateHandle(None, StateElementOldInitialValue),
#    priority_order=-1,
#)

__all__ = [
    "StateElementOldInitialValue",
]

from __future__ import annotations
from .state_element_osp import StateElementOspFile
from .tes_file import TesFile
from .identifier import StateElementIdentifier
from .state_element import (
    StateElementInitHandle,
    StateElementHandleSet,
)
from .current_state_state_info import h_old
from .current_state import FullGridMappedArray
from pathlib import Path
import numpy as np
from typing import Any
import typing

if typing.TYPE_CHECKING:
    from .retrieval_configuration import RetrievalConfiguration
    from .current_state import SoundingMetadata


class StateElementPcloud(StateElementOspFile):
    """State element for PCLOUD."""

    def __init__(
        self,
        rconfig: RetrievalConfiguration,
        smeta: SoundingMetadata,
        selem_wrapper: Any | None = None,
        **kwargs: Any,
    ) -> None:
        f = TesFile(Path(rconfig["Single_State_Directory"]) / "State_Cloud_IR.asc")
        value_fm = np.array(
            [
                float(f["CloudPressure"]),
            ]
        ).view(FullGridMappedArray)
        # There are a handful of state element that muses-py just "knows" get
        # the apriori covariance from a different diagonal uncertainty file
        # (see get_prior_covariance.py in muses-py, about line 100)
        diag_cov = True
        super().__init__(
            StateElementIdentifier("PCLOUD"),
            None,
            value_fm,
            value_fm,
            smeta.latitude.value,
            smeta.surface_type,
            Path(rconfig["speciesDirectory"]),
            Path(rconfig["covarianceDirectory"]),
            selem_wrapper=selem_wrapper,
            diag_cov=diag_cov,
        )


StateElementHandleSet.add_default_handle(
    StateElementInitHandle(
        StateElementIdentifier("PCLOUD"), StateElementPcloud, hold=h_old
    ),
    priority_order=0,
)

__all__ = [
    "StateElementPcloud",
]

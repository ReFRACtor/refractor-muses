from __future__ import annotations
from .state_element_osp import StateElementOspFile
from .tes_file import TesFile
from .identifier import StateElementIdentifier
from .state_element import (
    StateElementWithCreateHandle,
    StateElementHandleSet,
)
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

    @classmethod
    def _setup_create(
        cls,
        sid: StateElementIdentifier,
        retrieval_config: RetrievalConfiguration,
        sounding_metadata: SoundingMetadata,
        measurement_id: MeasurementId | None = None,
        strategy: MusesStrategy | None = None,
        observation_handle_set: ObservationHandleSet | None = None,
        selem_wrapper: Any | None = None,
    ) -> tuple[StateElementIdentifier, np.ndarray, np.ndarray | None, dict[str, Any]]:
        f = TesFile(
            Path(retrieval_config["Single_State_Directory"]) / "State_Cloud_IR.asc"
        )
        value_fm = np.array(
            [
                float(f["CloudPressure"]),
            ]
        ).view(FullGridMappedArray)
        # There are a handful of state element that muses-py just "knows" get
        # the apriori covariance from a different diagonal uncertainty file
        # (see get_prior_covariance.py in muses-py, about line 100)
        kwargs = {"diag_cov": True, "selem_wrapper": selem_wrapper}
        return StateElementIdentifier("PCLOUD"), value_fm, None, kwargs


StateElementHandleSet.add_default_handle(
    StateElementWithCreateHandle(
        StateElementIdentifier("PCLOUD"),
        StateElementPcloud,
        include_old_state_info=True,
    ),
    priority_order=0,
)

__all__ = [
    "StateElementPcloud",
]

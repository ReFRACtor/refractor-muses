from __future__ import annotations
from .state_element_osp import StateElementOspFile, OspSetupReturn
from .tes_file import TesFile
from .identifier import StateElementIdentifier
from .state_element import (
    StateElementWithCreateHandle,
    StateElementHandleSet,
)
from .retrieval_array import FullGridMappedArray
from pathlib import Path
import numpy as np
from typing import Any, Self
import typing

if typing.TYPE_CHECKING:
    from .muses_observation import ObservationHandleSet, MeasurementId
    from .muses_strategy import MusesStrategy
    from .retrieval_configuration import RetrievalConfiguration
    from .sounding_metadata import SoundingMetadata
    from .state_info import StateInfo


class StateElementFromSingle(StateElementOspFile):
    """State element listed in Species_List_From_Single in L2_Setup_Control_Initial.asc
    control file"""

    @classmethod
    def create(
        cls,
        sid: StateElementIdentifier | None = None,
        measurement_id: MeasurementId | None = None,
        retrieval_config: RetrievalConfiguration | None = None,
        strategy: MusesStrategy | None = None,
        observation_handle_set: ObservationHandleSet | None = None,
        sounding_metadata: SoundingMetadata | None = None,
        state_info: StateInfo | None = None,
        selem_wrapper: Any | None = None,
        **extra_kwargs: Any,
    ) -> Self | None:
        if retrieval_config is None:
            raise RuntimeError("Need retrieval_config")
        # Check if the state element is in the list of Species_List_From_Single, if not
        # we can't process it
        if sid is not None and sid not in [
            StateElementIdentifier(i)
            for i in retrieval_config["Species_List_From_Single"].split(",")
        ]:
            return None
        fcloud = TesFile(
            Path(retrieval_config["Single_State_Directory"]) / "State_Cloud_IR.asc"
        )
        fatm = TesFile(
            Path(retrieval_config["Single_State_Directory"]) / "State_AtmProfiles.asc"
        )
        fcal = TesFile(
            Path(retrieval_config["Single_State_Directory"])
            / "State_CalibrationData.asc"
        )
        return super(StateElementFromSingle, cls).create(
            sid,
            measurement_id,
            retrieval_config,
            strategy,
            observation_handle_set,
            sounding_metadata,
            state_info,
            selem_wrapper,
            fcloud=fcloud,
            fatm=fatm,
            fcal=fcal,
            **extra_kwargs,
        )

    @classmethod
    # type: ignore[override]
    def _setup_create(
        cls,
        sid: StateElementIdentifier,
        pressure_list_fm: FullGridMappedArray,
        fatm: TesFile,
        **kwargs: Any,
    ) -> OspSetupReturn | None:
        if fatm.table is None:
            return None
        value_fm = np.array(fatm.table[str(sid)]).view(FullGridMappedArray)
        value_fm = value_fm[(value_fm.shape[0] - pressure_list_fm.shape[0]) :].view(
            FullGridMappedArray
        )
        return OspSetupReturn(value_fm)


class StateElementPcloud(StateElementFromSingle):
    """State element for PCLOUD."""

    @classmethod
    # type: ignore[override]
    def _setup_create(
        cls,
        fcloud: TesFile,
        **kwargs: Any,
    ) -> OspSetupReturn | None:
        value_fm = np.array(
            [
                float(fcloud["CloudPressure"]),
            ]
        ).view(FullGridMappedArray)
        # There are a handful of state element that muses-py just "knows" get
        # the apriori covariance from a different diagonal uncertainty file
        # (see get_prior_covariance.py in muses-py, about line 100)
        kwargs = {"diag_cov": True}
        return OspSetupReturn(
            value_fm=value_fm,
            sid=StateElementIdentifier("PCLOUD"),
            create_kwargs=kwargs,
        )


StateElementHandleSet.add_default_handle(
    StateElementWithCreateHandle(
        StateElementIdentifier("PCLOUD"),
        StateElementPcloud,
        include_old_state_info=True,
    ),
    priority_order=0,
)

# Note, although NH3 and HCOOH are listed in Species_List_From_Single, there is
# separate logic in states_initial_update.py that overrides these in some cases.
# So we have two handles for these species

StateElementHandleSet.add_default_handle(
    StateElementWithCreateHandle(
        StateElementIdentifier("SO2"),
        StateElementFromSingle,
        include_old_state_info=True,
    ),
    priority_order=0,
)

StateElementHandleSet.add_default_handle(
    StateElementWithCreateHandle(
        StateElementIdentifier("NH3"),
        StateElementFromSingle,
        include_old_state_info=True,
    ),
    # Temp, until we get second handle in place
    # priority_order=0,
    priority_order=-100,
)

StateElementHandleSet.add_default_handle(
    StateElementWithCreateHandle(
        StateElementIdentifier("OCS"),
        StateElementFromSingle,
        include_old_state_info=True,
    ),
    priority_order=0,
)

StateElementHandleSet.add_default_handle(
    StateElementWithCreateHandle(
        StateElementIdentifier("HCOOH"),
        StateElementFromSingle,
        include_old_state_info=True,
    ),
    # Temp, until we get second handle in place
    # priority_order=0,
    priority_order=-100,
)

StateElementHandleSet.add_default_handle(
    StateElementWithCreateHandle(
        StateElementIdentifier("N2"),
        StateElementFromSingle,
        include_old_state_info=True,
    ),
    priority_order=0,
)


__all__ = ["StateElementPcloud", "StateElementFromSingle"]

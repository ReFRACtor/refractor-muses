from __future__ import annotations
import refractor.muses.muses_py as mpy  # type: ignore
from .state_element_osp import StateElementOspFile, OspSetupReturn
from .identifier import StateElementIdentifier
from .gmao_reader import GmaoReader
from .retrieval_array import FullGridMappedArray
from .state_element import (
    StateElementWithCreateHandle,
    StateElementHandleSet,
)
from .priority_handle_set import NoHandleFound
from .tes_file import TesFile
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


class StateElementFromGmao(StateElementOspFile):
    """State element listed in Species_List_From_Gmao in L2_Setup_Control_Initial.asc
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
        # Check if the state element is in the list of Species_List_From_GMAO, if not
        # we don't process it
        #
        # Note this is really just a sanity check, we only create handles down below
        # for the state elements in this list. The L2_Setup_Control_Initial.asc doesn't
        # really control this, and it doesn't look like it really controlled muses-py old
        # initial guess stuff. But we should at least notice if there is an inconsistency
        # here. Perhaps this can go away, it isn't really clear why we can't just do
        # this in our python configuration vs. a separate control file.
        #
        # Note there are additional elements that muses-py just "knows" are part of the
        # GMAO, so we check those also
        if (
            sid is not None
            and sid
            not in [
                StateElementIdentifier(i)
                for i in retrieval_config["Species_List_From_GMAO"].split(",")
            ]
            and str(sid) not in ["pressure", "gmaoTropopausePressure"]
        ):
            return None
        # Grab gmao_data from psur. This is just a convenience, to avoid
        # recalculating this multiple times. We can rework this if needed.
        if state_info is None:
            return None
        try:
            psur = state_info[StateElementIdentifier("PSUR")]
        except NoHandleFound:
            return None
        if "gmao_data" not in psur.metadata:
            return None
        gmao_data = psur.metadata["gmao_data"]
        return super(StateElementFromGmao, cls).create(
            sid,
            measurement_id,
            retrieval_config,
            strategy,
            observation_handle_set,
            sounding_metadata,
            state_info,
            selem_wrapper,
            gmao_data=gmao_data,
            **extra_kwargs,
        )

    @classmethod
    def _setup_create(  # type: ignore[override]
        cls,
        gmao_data: GmaoReader,
        **kwargs: Any,
    ) -> OspSetupReturn | None:
        return None


class StateElementFromGmaoTropopausePressure(StateElementFromGmao):
    @classmethod
    def _setup_create(  # type: ignore[override]
        cls,
        gmao_data: GmaoReader,
        **kwargs: Any,
    ) -> OspSetupReturn | None:
        return OspSetupReturn(
            value_fm=np.array(
                [
                    gmao_data.tropopause_pressure,
                ]
            ).view(FullGridMappedArray)
        )


class StateElementFromGmaoTatm(StateElementFromGmao):
    @classmethod
    def _setup_create(  # type: ignore[override]
        cls,
        gmao_data: GmaoReader,
        **kwargs: Any,
    ) -> OspSetupReturn | None:
        return OspSetupReturn(value_fm=gmao_data.tatm.view(FullGridMappedArray))


class StateElementFromGmaoH2O(StateElementFromGmao):
    @classmethod
    def _setup_create(  # type: ignore[override]
        cls,
        gmao_data: GmaoReader,
        **kwargs: Any,
    ) -> OspSetupReturn | None:
        return OspSetupReturn(value_fm=gmao_data.h2o.view(FullGridMappedArray))


class StateElementFromGmaoTsur(StateElementFromGmao):
    @classmethod
    def _setup_create(  # type: ignore[override]
        cls,
        gmao_data: GmaoReader,
        **kwargs: Any,
    ) -> OspSetupReturn | None:
        return OspSetupReturn(
            value_fm=np.array(
                [
                    gmao_data.surface_temperature,
                ]
            ).view(FullGridMappedArray)
        )


class StateElementFromGmaoPressure(StateElementOspFile):
    """Because pressure is needed as input to create a lot of other StateElement,
    we need to treat this as a special case."""

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
        # Pressure starts from the atmospheric profile, augmented with the
        # surface pressure from gmao
        fatm = TesFile(
            Path(retrieval_config["Single_State_Directory"]) / "State_AtmProfiles.asc"
        )
        if fatm.table is None:
            return None
        pressure0 = np.array(fatm.table["Pressure"])
        gmao_dir = (
            retrieval_config.gmao_dir
            if retrieval_config.gmao_dir is not None
            else retrieval_config["GMAO_Directory"]
        )
        if sounding_metadata is None:
            return None
        gmao = GmaoReader(sounding_metadata, gmao_dir)
        surface_pressure = mpy.supplier_surface_pressure(
            sounding_metadata.surface_altitude.value,
            gmao.tatm,
            gmao.h2o,
            gmao.pressure,
            sounding_metadata.latitude.value,
        )
        pressure = mpy.supplier_fm_pressures(pressure0, surface_pressure)
        return cls(
            StateElementIdentifier("pressure"),
            pressure,
            pressure,
            pressure,
            sounding_metadata.latitude.value,
            sounding_metadata.surface_type,
            Path(retrieval_config["speciesDirectory"]),
            Path(retrieval_config["covarianceDirectory"]),
            selem_wrapper=selem_wrapper,
        )


class StateElementFromGmaoPsur(StateElementOspFile):
    """Because psurf is needed as input to create a lot of other StateElement,
    we need to treat this as a special case."""

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
        # Pressure starts from the atmospheric profile, augmented with the
        # surface pressure from gmao
        if sid is not None and sid == StateElementIdentifier("pressure"):
            # Avoid infinite recursion, if we are called for "pressure" we
            # can't handle that in this creator.
            return None
        if state_info is None:
            return None
        p = state_info[StateElementIdentifier("pressure")]
        gmao_dir = (
            retrieval_config.gmao_dir
            if retrieval_config.gmao_dir is not None
            else retrieval_config["GMAO_Directory"]
        )
        if sounding_metadata is None:
            return None
        gmao = GmaoReader(sounding_metadata, gmao_dir, pressure_in=p.value_fm)

        return cls(
            StateElementIdentifier("PSUR"),
            None,
            np.array(
                [
                    gmao.surface_pressure,
                ]
            ).view(FullGridMappedArray),
            np.array(
                [
                    gmao.surface_pressure,
                ]
            ).view(FullGridMappedArray),
            sounding_metadata.latitude.value,
            sounding_metadata.surface_type,
            Path(retrieval_config["speciesDirectory"]),
            Path(retrieval_config["covarianceDirectory"]),
            selem_wrapper=selem_wrapper,
            metadata={"gmao_data": gmao},
        )


StateElementHandleSet.add_default_handle(
    StateElementWithCreateHandle(
        StateElementIdentifier("pressure"),
        StateElementFromGmaoPressure,
        include_old_state_info=False,
    ),
    priority_order=0,
)

StateElementHandleSet.add_default_handle(
    StateElementWithCreateHandle(
        StateElementIdentifier("PSUR"), StateElementFromGmaoPsur
    ),
    priority_order=0,
)

StateElementHandleSet.add_default_handle(
    StateElementWithCreateHandle(
        StateElementIdentifier("gmaoTropopausePressure"),
        StateElementFromGmaoTropopausePressure,
    ),
    priority_order=0,
)

StateElementHandleSet.add_default_handle(
    StateElementWithCreateHandle(
        StateElementIdentifier("TATM"), StateElementFromGmaoTatm
    ),
    priority_order=0,
)

StateElementHandleSet.add_default_handle(
    StateElementWithCreateHandle(
        StateElementIdentifier("H2O"), StateElementFromGmaoH2O
    ),
    priority_order=0,
)

StateElementHandleSet.add_default_handle(
    StateElementWithCreateHandle(
        StateElementIdentifier("TSUR"), StateElementFromGmaoTsur
    ),
    priority_order=0,
)


__all__ = [
    "StateElementFromGmao",
    "StateElementFromGmaoPressure",
    "StateElementFromGmaoPsur",
    "StateElementFromGmaoTropopausePressure",
    "StateElementFromGmaoTatm",
    "StateElementFromGmaoH2O",
    "StateElementFromGmaoTsur",
]

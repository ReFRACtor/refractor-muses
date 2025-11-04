from __future__ import annotations
from refractor.muses import (
    StateElementHandleSet,
    StateElementWithCreateHandle,
    StateElementOspFileFixedValue,
    StateElementFillValueHandle,
    StateElementOspFile,
    StateElementWithCreate,
    StateElementIdentifier,
    OspSetupReturn,
    InstrumentIdentifier,
    MusesOmiObservation,
    FullGridMappedArray,
)
import numpy as np
from typing import cast, Any
import typing

if typing.TYPE_CHECKING:
    from refractor.muses import (
        MusesStrategy,
        ObservationHandleSet,
        RetrievalConfiguration,
    )


def add_class_handle(
    sname: str,
    cls: type[StateElementWithCreate],
) -> None:
    StateElementHandleSet.add_default_handle(
        StateElementWithCreateHandle(StateElementIdentifier(sname), cls),
        priority_order=2,
    )


def add_fixed_handle(sname: str, initial_value: float) -> None:
    StateElementHandleSet.add_default_handle(
        StateElementWithCreateHandle(
            StateElementIdentifier(sname),
            StateElementOspFileFixedValue,
            initial_value=np.array(
                [
                    initial_value,
                ]
            ).view(FullGridMappedArray),
        ),
        priority_order=2,
    )


def add_fill_handle(
    sname: str,
) -> None:
    # Lower priority, so we only create fill data if we couldn't create the actual
    # StateElement.
    StateElementHandleSet.add_default_handle(
        StateElementFillValueHandle(StateElementIdentifier(sname)), priority_order=1
    )


class StateElementOmiCloudFraction(StateElementOspFile):
    """Variation that gets the apriori/initial guess from the observation file"""

    @classmethod
    def _setup_create(  # type: ignore[override]
        cls,
        retrieval_config: RetrievalConfiguration,
        strategy: MusesStrategy,
        observation_handle_set: ObservationHandleSet,
        **kwargs: Any,
    ) -> OspSetupReturn | None:
        if InstrumentIdentifier("OMI") not in strategy.instrument_name:
            return None
        obs = observation_handle_set.observation(
            InstrumentIdentifier("OMI"),
            None,
            None,
            None,
            osp_dir=retrieval_config.osp_dir,
        )
        value_fm = np.array([obs.cloud_fraction]).view(FullGridMappedArray)
        return OspSetupReturn(
            value_fm=value_fm, sid=StateElementIdentifier("OMICLOUDFRACTION")
        )


class StateElementOmiSurfaceAlbedo(StateElementOspFile):
    """Variation that gets the apriori/initial guess from the observation file"""

    @classmethod
    def _setup_create(  # type: ignore[override]
        cls,
        sid: StateElementIdentifier,
        retrieval_config: RetrievalConfiguration,
        strategy: MusesStrategy,
        observation_handle_set: ObservationHandleSet,
        **kwargs: Any,
    ) -> OspSetupReturn | None:
        if InstrumentIdentifier("OMI") not in strategy.instrument_name:
            return None
        obs = cast(
            MusesOmiObservation,
            observation_handle_set.observation(
                InstrumentIdentifier("OMI"),
                None,
                None,
                None,
                osp_dir=retrieval_config.osp_dir,
            ),
        )
        value_fm = np.array([obs.monthly_minimum_surface_reflectance]).view(
            FullGridMappedArray
        )
        return OspSetupReturn(value_fm)


add_class_handle("OMICLOUDFRACTION", StateElementOmiCloudFraction)
# This gets created even if we don't have OMI data
add_fill_handle("OMICLOUDFRACTION")
add_fixed_handle("OMINRADWAVUV1", 0.0)
add_fixed_handle("OMINRADWAVUV2", 0.0)
add_fixed_handle("OMIODWAVSLOPEUV1", 0.0)
add_fixed_handle("OMIODWAVSLOPEUV2", 0.0)
add_fixed_handle("OMIODWAVUV1", 0.0)
add_fixed_handle("OMIODWAVUV2", 0.0)
# Doesn't actually seem to be in muses-py, although there is a species file for this
# add_fixes_handle("OMIRESSCALE", 0.0)
add_fixed_handle("OMIRINGSFUV1", 1.9)
add_fixed_handle("OMIRINGSFUV2", 1.9)
add_fixed_handle("OMISURFACEALBEDOSLOPEUV2", 0.0)
add_class_handle("OMISURFACEALBEDOUV1", StateElementOmiSurfaceAlbedo)
add_class_handle("OMISURFACEALBEDOUV2", StateElementOmiSurfaceAlbedo)
add_fill_handle("OMISURFACEALBEDOUV1")
add_fill_handle("OMISURFACEALBEDOUV2")

__all__ = ["StateElementOmiCloudFraction", "StateElementOmiSurfaceAlbedo"]

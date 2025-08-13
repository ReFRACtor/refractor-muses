from __future__ import annotations
from refractor.muses import (
    StateElementHandleSet,
    StateElementWithCreateHandle,
    StateElementOspFileFixedValue,
    StateElementFillValueHandle,
    StateElementOspFile,
    StateElementWithCreate,
    StateElementIdentifier,
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
        SoundingMetadata,
        RetrievalConfiguration,
        MeasurementId,
        StateInfo,
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
    def _setup_create(
        cls,
        pressure_list_fm: FullGridMappedArray,
        sid: StateElementIdentifier | None,
        retrieval_config: RetrievalConfiguration,
        sounding_metadata: SoundingMetadata,
        measurement_id: MeasurementId | None = None,
        strategy: MusesStrategy | None = None,
        observation_handle_set: ObservationHandleSet | None = None,
        state_info: StateInfo | None = None,
        selem_wrapper: Any | None = None,
        **kwargs: Any,
    ) -> tuple[
        StateElementIdentifier,
        FullGridMappedArray | None,
        FullGridMappedArray | None,
        dict[str, Any],
    ]:
        if strategy is None or observation_handle_set is None:
            raise RuntimeError("Need strategy and observation_handle_set supplied")
        if InstrumentIdentifier("OMI") not in strategy.instrument_name:
            return StateElementIdentifier("Dummy"), None, None, {}
        obs = observation_handle_set.observation(
            InstrumentIdentifier("OMI"),
            None,
            None,
            None,
            osp_dir=retrieval_config.osp_dir,
        )
        value_fm = np.array([obs.cloud_fraction]).view(FullGridMappedArray)
        kwargs = {"selem_wrapper": selem_wrapper}
        return StateElementIdentifier("OMICLOUDFRACTION"), value_fm, None, kwargs


class StateElementOmiSurfaceAlbedo(StateElementOspFile):
    """Variation that gets the apriori/initial guess from the observation file"""

    @classmethod
    def _setup_create(
        cls,
        pressure_list_fm: FullGridMappedArray,
        sid: StateElementIdentifier | None,
        retrieval_config: RetrievalConfiguration,
        sounding_metadata: SoundingMetadata,
        measurement_id: MeasurementId | None = None,
        strategy: MusesStrategy | None = None,
        observation_handle_set: ObservationHandleSet | None = None,
        state_info: StateInfo | None = None,
        selem_wrapper: Any | None = None,
        **kwargs: Any,
    ) -> tuple[
        StateElementIdentifier,
        FullGridMappedArray | None,
        FullGridMappedArray | None,
        dict[str, Any],
    ]:
        if sid is None or strategy is None or observation_handle_set is None:
            raise RuntimeError("Need sid, strategy and observation_handle_set supplied")
        if InstrumentIdentifier("OMI") not in strategy.instrument_name:
            return StateElementIdentifier("Dummy"), None, None, {}
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
        kwargs = {"selem_wrapper": selem_wrapper}
        return sid, value_fm, None, kwargs


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

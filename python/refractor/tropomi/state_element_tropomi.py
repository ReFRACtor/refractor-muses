from __future__ import annotations
from refractor.muses import (
    StateElementHandleSet,
    StateElementOspFileFixedValue,
    StateElementOspFile,
    StateElementFillValueHandle,
    StateElementIdentifier,
    StateElementWithCreate,
    StateElementWithCreateHandle,
    InstrumentIdentifier,
    FullGridMappedArray,
    FullGrid2dArray,
    RetrievalGrid2dArray,
)
import numpy as np
from typing import Any, Self
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
    **kwargs: Any,
) -> None:
    StateElementHandleSet.add_default_handle(
        StateElementWithCreateHandle(StateElementIdentifier(sname), cls, **kwargs),
        priority_order=2,
    )


def add_fixed_handle(sname: str, initial_value: float, **kwargs: Any) -> None:
    StateElementHandleSet.add_default_handle(
        StateElementWithCreateHandle(
            StateElementIdentifier(sname),
            StateElementOspFileFixedValue,
            initial_value=np.array(
                [
                    initial_value,
                ]
            ).view(FullGridMappedArray),
            **kwargs,
        ),
        priority_order=2,
    )


def add_fill_handle(
    sname: str,
) -> None:
    StateElementHandleSet.add_default_handle(
        StateElementFillValueHandle(StateElementIdentifier(sname)), priority_order=1
    )


class StateElementTropomiCloudFraction(StateElementOspFile):
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
        if InstrumentIdentifier("TROPOMI") not in strategy.instrument_name:
            return StateElementIdentifier("Dummy"), None, None, {}
        obs = observation_handle_set.observation(
            InstrumentIdentifier("TROPOMI"),
            None,
            None,
            None,
            osp_dir=retrieval_config.osp_dir,
        )
        value_fm = np.array([obs.cloud_fraction]).view(FullGridMappedArray)
        kwargs = {"selem_wrapper": selem_wrapper}
        return StateElementIdentifier("TROPOMICLOUDFRACTION"), value_fm, None, kwargs


class StateElementTropomiCloudPressure(StateElementWithCreate):
    """Variation that gets the apriori/initial guess from the observation file"""

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
        **kwargs: Any,
    ) -> Self | None:
        if (
            retrieval_config is None
            or strategy is None
            or observation_handle_set is None
        ):
            raise RuntimeError("Need strategy and observation_handle_set supplied")
        if InstrumentIdentifier("TROPOMI") not in strategy.instrument_name:
            return None
        obs = observation_handle_set.observation(
            InstrumentIdentifier("TROPOMI"),
            None,
            None,
            None,
            osp_dir=retrieval_config.osp_dir,
        )
        value_fm = np.array([obs.cloud_pressure.value]).view(FullGridMappedArray)
        return cls(
            StateElementIdentifier("TROPOMICLOUDPRESSURE"),
            value_fm,
            value_fm,
            np.array([[-999.0]]).view(FullGrid2dArray),
            np.array([[-999.0]]).view(RetrievalGrid2dArray),
            selem_wrapper=selem_wrapper,
        )


class StateElementTropomiSurfaceAlbedo(StateElementOspFile):
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
        band: int = -1,
        **kwargs: Any,
    ) -> tuple[
        StateElementIdentifier,
        FullGridMappedArray | None,
        FullGridMappedArray | None,
        dict[str, Any],
    ]:
        if sid is None or strategy is None or observation_handle_set is None:
            raise RuntimeError("Need strategy and observation_handle_set supplied")
        if InstrumentIdentifier("TROPOMI") not in strategy.instrument_name:
            return StateElementIdentifier("Dummy"), None, None, {}
        obs = observation_handle_set.observation(
            InstrumentIdentifier("TROPOMI"),
            None,
            None,
            None,
            osp_dir=retrieval_config.osp_dir,
        )
        value_fm = np.array(
            [
                float(
                    obs.muses_py_dict["SurfaceAlbedo"][
                        f"BAND{band}_MonthlyMinimumSurfaceReflectance"
                    ]
                )
            ]
        ).view(FullGridMappedArray)
        kwargs = {"selem_wrapper": selem_wrapper}
        return sid, value_fm, None, kwargs


add_class_handle("TROPOMICLOUDFRACTION", StateElementTropomiCloudFraction)
# This gets created even if we don't have TROPOMI data
add_fill_handle("TROPOMICLOUDFRACTION")
add_class_handle("TROPOMICLOUDPRESSURE", StateElementTropomiCloudPressure)
add_fixed_handle("TROPOMICLOUDSURFACEALBEDO", 0.8)
add_fixed_handle("TROPOMIRADIANCESHIFTBAND1", 0.0)
add_fixed_handle("TROPOMIRADIANCESHIFTBAND2", 0.0)
add_fixed_handle("TROPOMIRADIANCESHIFTBAND3", 0.0)
add_fixed_handle("TROPOMIRADIANCESHIFTBAND7", 0.0, cov_is_constraint=True)
add_fixed_handle("TROPOMIRADSQUEEZEBAND1", 0.0)
add_fixed_handle("TROPOMIRADSQUEEZEBAND2", 0.0)
add_fixed_handle("TROPOMIRADSQUEEZEBAND3", 0.0)
add_fixed_handle("TROPOMIRADSQUEEZEBAND7", 0.0, cov_is_constraint=True)
add_fixed_handle("TROPOMIRESSCALE", 1.0)
add_fixed_handle("TROPOMIRESSCALEO0BAND1", 1.0)
add_fixed_handle("TROPOMIRESSCALEO1BAND1", 0.0)
add_fixed_handle("TROPOMIRESSCALEO2BAND1", 0.0)
add_fixed_handle("TROPOMIRESSCALEO0BAND2", 1.0)
add_fixed_handle("TROPOMIRESSCALEO1BAND2", 0.0)
add_fixed_handle("TROPOMIRESSCALEO2BAND2", 0.0)
add_fixed_handle("TROPOMIRESSCALEO0BAND3", 1.0)
add_fixed_handle("TROPOMIRESSCALEO1BAND3", 0.0)
add_fixed_handle("TROPOMIRESSCALEO2BAND3", 0.0)
add_fixed_handle("TROPOMIRESSCALEO0BAND7", 1.0, cov_is_constraint=True)
add_fixed_handle("TROPOMIRESSCALEO1BAND7", 0.0, cov_is_constraint=True)
add_fixed_handle("TROPOMIRESSCALEO2BAND7", 0.0, cov_is_constraint=True)
add_fixed_handle("TROPOMIRINGSFBAND1", 1.9)
add_fixed_handle("TROPOMIRINGSFBAND2", 1.9)
add_fixed_handle("TROPOMIRINGSFBAND3", 1.9)
add_fixed_handle("TROPOMIRINGSFBAND7", 1.9, cov_is_constraint=True)
add_fixed_handle("TROPOMISOLARSHIFTBAND1", 0.0)
add_fixed_handle("TROPOMISOLARSHIFTBAND2", 0.0)
add_fixed_handle("TROPOMISOLARSHIFTBAND3", 0.0)
add_fixed_handle("TROPOMISOLARSHIFTBAND7", 0.0, cov_is_constraint=True)
add_class_handle("TROPOMISURFACEALBEDOBAND1", StateElementTropomiSurfaceAlbedo, band=1)
add_class_handle("TROPOMISURFACEALBEDOBAND2", StateElementTropomiSurfaceAlbedo, band=2)
add_class_handle("TROPOMISURFACEALBEDOBAND3", StateElementTropomiSurfaceAlbedo, band=3)
add_class_handle(
    "TROPOMISURFACEALBEDOBAND7",
    StateElementTropomiSurfaceAlbedo,
    band=7,
    cov_is_constraint=True,
)
add_fixed_handle("TROPOMISURFACEALBEDOSLOPEBAND1", 0.0)
add_fixed_handle("TROPOMISURFACEALBEDOSLOPEBAND2", 0.0)
add_fixed_handle("TROPOMISURFACEALBEDOSLOPEBAND3", 0.0)
add_fixed_handle("TROPOMISURFACEALBEDOSLOPEBAND7", 0.0, cov_is_constraint=True)
add_fixed_handle("TROPOMISURFACEALBEDOSLOPEORDER2BAND2", 0.0)
add_fixed_handle("TROPOMISURFACEALBEDOSLOPEORDER2BAND3", 0.0)
add_fixed_handle("TROPOMISURFACEALBEDOSLOPEORDER2BAND7", 0.0, cov_is_constraint=True)
add_fixed_handle("TROPOMITEMPSHIFTBAND1", 1.0)
add_fixed_handle("TROPOMITEMPSHIFTBAND2", 1.0)
add_fixed_handle("TROPOMITEMPSHIFTBAND3", 1.0)
add_fixed_handle("TROPOMITEMPSHIFTBAND7", 1.0, cov_is_constraint=True)

__all__ = [
    "StateElementTropomiCloudFraction",
    "StateElementTropomiCloudPressure",
    "StateElementTropomiSurfaceAlbedo",
]

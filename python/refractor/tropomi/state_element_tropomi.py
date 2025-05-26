from __future__ import annotations
from refractor.muses import (
    StateElementHandleSet,
    StateElement,
    StateElementHandle,
    StateElementImplementation,
    StateElementOldWrapper,
    StateElementOldWrapperHandle,
    StateElementOspFileHandle,
    StateElementOspFile,
    StateElementFillValueHandle,
    StateElementIdentifier,
    InstrumentIdentifier,
    MusesTropomiObservation,
    MusesObservation,
    FullGridMappedArray,
    FullGrid2dArray,
    RetrievalGrid2dArray,
)
import numpy as np
from pathlib import Path
from typing import cast, Any
import typing

if typing.TYPE_CHECKING:
    from refractor.muses import (
        MusesStrategy,
        ObservationHandleSet,
        SoundingMetadata,
        StateElementOldWrapper,
        RetrievalConfiguration,
        MeasurementId,
    )


def add_handle(
    sname: str,
    constraint_value: float,
    cls: type[StateElementOspFile] = StateElementOspFile,
    cov_is_constraint: bool = False,
) -> None:
    StateElementHandleSet.add_default_handle(
        StateElementOspFileHandle(
            StateElementIdentifier(sname),
            np.array([constraint_value]).view(FullGridMappedArray),
            np.array([constraint_value]).view(FullGridMappedArray),
            cls=cls,
            cov_is_constraint=cov_is_constraint,
        ),
        priority_order=2,
    )


def add_fill_handle(
    sname: str,
) -> None:
    StateElementHandleSet.add_default_handle(
        StateElementFillValueHandle(StateElementIdentifier(sname)), priority_order=1
    )


class StateElementTropomiFileHandle(StateElementHandle):
    """Add tropomi file input and band to creation"""

    def __init__(
        self,
        sid: StateElementIdentifier,
        cls: Any,
        band: int = -1,
        h_old: StateElementOldWrapperHandle | None = None,
        cov_is_constraint: bool = False,
    ) -> None:
        self.obj_cls = cls
        self.sid = sid
        self.h_old = h_old
        self.band = band
        self.measurement_id: MeasurementId | None = None
        self.retrieval_config: RetrievalConfiguration | None = None
        self.cov_is_constraint = cov_is_constraint

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
        if state_element_id != self.sid:
            return None
        if self.measurement_id is None or self.retrieval_config is None:
            raise RuntimeError("Need to call notify_update_target first")
        if InstrumentIdentifier("TROPOMI") not in self.strategy.instrument_name:
            return None
        obs = cast(
            MusesTropomiObservation,
            self.observation_handle_set.observation(
                InstrumentIdentifier("TROPOMI"),
                None,
                None,
                None,
                osp_dir=self.retrieval_config.osp_dir,
            ),
        )
        if self.h_old is not None:
            sold = cast(
                StateElementOldWrapper, self.h_old.state_element(state_element_id)
            )
        else:
            sold = None
        res = self.obj_cls(
            state_element_id,
            obs,
            self.sounding_metadata.latitude.value,
            self.sounding_metadata.surface_type,
            Path(self.retrieval_config["speciesDirectory"]),
            Path(self.retrieval_config["covarianceDirectory"]),
            selem_wrapper=sold,
            band=self.band,
            cov_is_constraint=self.cov_is_constraint,
        )
        return res


class StateElementTropomiFileFixedHandle(StateElementHandle):
    """Add tropomi file input and band to creation"""

    def __init__(
        self,
        sid: StateElementIdentifier,
        cls: Any,
        band: int = -1,
        h_old: StateElementOldWrapperHandle | None = None,
        cov_is_constraint: bool = False,
    ) -> None:
        self.obj_cls = cls
        self.sid = sid
        self.h_old = h_old
        self.band = band
        self.measurement_id: MeasurementId | None = None
        self.retrieval_config: RetrievalConfiguration | None = None
        self.cov_is_constraint = cov_is_constraint

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
        if state_element_id != self.sid:
            return None
        if self.measurement_id is None or self.retrieval_config is None:
            raise RuntimeError("Need to call notify_update_target first")
        if InstrumentIdentifier("TROPOMI") not in self.strategy.instrument_name:
            return None
        obs = cast(
            MusesTropomiObservation,
            self.observation_handle_set.observation(
                InstrumentIdentifier("TROPOMI"),
                None,
                None,
                None,
                osp_dir=self.retrieval_config.osp_dir,
            ),
        )
        if self.h_old is not None:
            sold = cast(
                StateElementOldWrapper, self.h_old.state_element(state_element_id)
            )
        else:
            sold = None
        res = self.obj_cls(
            state_element_id,
            obs,
            selem_wrapper=sold,
            band=self.band,
            cov_is_constraint=self.cov_is_constraint,
        )
        return res


def add_tropomi_file_handle(
    sname: str, cls: Any, band: int = -1, cov_is_constraint: bool = False
) -> None:
    StateElementHandleSet.add_default_handle(
        StateElementTropomiFileHandle(
            StateElementIdentifier(sname),
            cls,
            band=band,
            cov_is_constraint=cov_is_constraint,
        ),
        priority_order=2,
    )


def add_tropomi_file_fixed_handle(
    sname: str, cls: Any, band: int = -1, cov_is_constraint: bool = False
) -> None:
    StateElementHandleSet.add_default_handle(
        StateElementTropomiFileFixedHandle(
            StateElementIdentifier(sname),
            cls,
            band=band,
            cov_is_constraint=cov_is_constraint,
        ),
        priority_order=2,
    )


class StateElementTropomiCloudFraction(StateElementOspFile):
    """Variation that gets the apriori/initial guess from the observation file"""

    def __init__(
        self,
        state_element_id: StateElementIdentifier,
        obs: MusesObservation,
        latitude: float,
        surface_type: str,
        species_directory: Path,
        covariance_directory: Path,
        selem_wrapper: StateElementOldWrapper | None = None,
        band: int = -1,
        cov_is_constraint: bool = False,
    ) -> None:
        constraint_vector_fm = np.array([obs.cloud_fraction]).view(FullGridMappedArray)
        super().__init__(
            state_element_id,
            None,
            constraint_vector_fm,
            constraint_vector_fm,
            latitude,
            surface_type,
            species_directory,
            covariance_directory,
            selem_wrapper=selem_wrapper,
            cov_is_constraint=cov_is_constraint,
        )


class StateElementTropomiCloudPressure(StateElementImplementation):
    """Variation that gets the apriori/initial guess from the observation file"""

    def __init__(
        self,
        state_element_id: StateElementIdentifier,
        obs: MusesObservation,
        selem_wrapper: StateElementOldWrapper | None = None,
        band: int = -1,
        cov_is_constraint: bool = False,
    ) -> None:
        constraint_vector_fm = np.array([obs.cloud_pressure.value]).view(
            FullGridMappedArray
        )
        super().__init__(
            state_element_id,
            constraint_vector_fm,
            constraint_vector_fm,
            np.array([[-999.0]]).view(FullGrid2dArray),
            np.array([[-999.0]]).view(RetrievalGrid2dArray),
            selem_wrapper=selem_wrapper,
        )


class StateElementTropomiSurfaceAlbedo(StateElementOspFile):
    """Variation that gets the apriori/initial guess from the observation file"""

    def __init__(
        self,
        state_element_id: StateElementIdentifier,
        obs: MusesObservation,
        latitude: float,
        surface_type: str,
        species_directory: Path,
        covariance_directory: Path,
        selem_wrapper: StateElementOldWrapper | None = None,
        band: int = -1,
        cov_is_constraint: bool = False,
    ) -> None:
        constraint_vector_fm = np.array(
            [
                float(
                    obs.muses_py_dict["SurfaceAlbedo"][
                        f"BAND{band}_MonthlyMinimumSurfaceReflectance"
                    ]
                )
            ]
        ).view(FullGridMappedArray)
        super().__init__(
            state_element_id,
            None,
            constraint_vector_fm,
            constraint_vector_fm,
            latitude,
            surface_type,
            species_directory,
            covariance_directory,
            selem_wrapper=selem_wrapper,
            cov_is_constraint=cov_is_constraint,
        )


add_tropomi_file_handle("TROPOMICLOUDFRACTION", StateElementTropomiCloudFraction)
# This gets created even if we don't have TROPOMI data
add_fill_handle("TROPOMICLOUDFRACTION")
add_tropomi_file_fixed_handle("TROPOMICLOUDPRESSURE", StateElementTropomiCloudPressure)
add_handle("TROPOMICLOUDSURFACEALBEDO", 0.8)
add_handle("TROPOMIRADIANCESHIFTBAND1", 0.0)
add_handle("TROPOMIRADIANCESHIFTBAND2", 0.0)
add_handle("TROPOMIRADIANCESHIFTBAND3", 0.0)
add_handle("TROPOMIRADIANCESHIFTBAND7", 0.0, cov_is_constraint=True)
add_handle("TROPOMIRADSQUEEZEBAND1", 0.0)
add_handle("TROPOMIRADSQUEEZEBAND2", 0.0)
add_handle("TROPOMIRADSQUEEZEBAND3", 0.0)
add_handle("TROPOMIRADSQUEEZEBAND7", 0.0, cov_is_constraint=True)
add_handle("TROPOMIRESSCALE", 1.0)
add_handle("TROPOMIRESSCALEO0BAND1", 1.0)
add_handle("TROPOMIRESSCALEO1BAND1", 0.0)
add_handle("TROPOMIRESSCALEO2BAND1", 0.0)
add_handle("TROPOMIRESSCALEO0BAND2", 1.0)
add_handle("TROPOMIRESSCALEO1BAND2", 0.0)
add_handle("TROPOMIRESSCALEO2BAND2", 0.0)
add_handle("TROPOMIRESSCALEO0BAND3", 1.0)
add_handle("TROPOMIRESSCALEO1BAND3", 0.0)
add_handle("TROPOMIRESSCALEO2BAND3", 0.0)
add_handle("TROPOMIRESSCALEO0BAND7", 1.0, cov_is_constraint=True)
add_handle("TROPOMIRESSCALEO1BAND7", 0.0, cov_is_constraint=True)
add_handle("TROPOMIRESSCALEO2BAND7", 0.0, cov_is_constraint=True)
add_handle("TROPOMIRINGSFBAND1", 1.9)
add_handle("TROPOMIRINGSFBAND2", 1.9)
add_handle("TROPOMIRINGSFBAND3", 1.9)
add_handle("TROPOMIRINGSFBAND7", 1.9, cov_is_constraint=True)
add_handle("TROPOMISOLARSHIFTBAND1", 0.0)
add_handle("TROPOMISOLARSHIFTBAND2", 0.0)
add_handle("TROPOMISOLARSHIFTBAND3", 0.0)
add_handle("TROPOMISOLARSHIFTBAND7", 0.0, cov_is_constraint=True)
add_tropomi_file_handle(
    "TROPOMISURFACEALBEDOBAND1", StateElementTropomiSurfaceAlbedo, band=1
)
add_tropomi_file_handle(
    "TROPOMISURFACEALBEDOBAND2", StateElementTropomiSurfaceAlbedo, band=2
)
add_tropomi_file_handle(
    "TROPOMISURFACEALBEDOBAND3", StateElementTropomiSurfaceAlbedo, band=3
)
add_tropomi_file_handle(
    "TROPOMISURFACEALBEDOBAND7",
    StateElementTropomiSurfaceAlbedo,
    band=7,
    cov_is_constraint=True,
)
add_handle("TROPOMISURFACEALBEDOSLOPEBAND1", 0.0)
add_handle("TROPOMISURFACEALBEDOSLOPEBAND2", 0.0)
add_handle("TROPOMISURFACEALBEDOSLOPEBAND3", 0.0)
add_handle("TROPOMISURFACEALBEDOSLOPEBAND7", 0.0, cov_is_constraint=True)
add_handle("TROPOMISURFACEALBEDOSLOPEORDER2BAND2", 0.0)
add_handle("TROPOMISURFACEALBEDOSLOPEORDER2BAND3", 0.0)
add_handle("TROPOMISURFACEALBEDOSLOPEORDER2BAND7", 0.0, cov_is_constraint=True)
add_handle("TROPOMITEMPSHIFTBAND1", 1.0)
add_handle("TROPOMITEMPSHIFTBAND2", 1.0)
add_handle("TROPOMITEMPSHIFTBAND3", 1.0)
add_handle("TROPOMITEMPSHIFTBAND7", 1.0, cov_is_constraint=True)

__all__ = [
    "StateElementTropomiCloudFraction",
    "StateElementTropomiCloudPressure",
    "StateElementTropomiSurfaceAlbedo",
]

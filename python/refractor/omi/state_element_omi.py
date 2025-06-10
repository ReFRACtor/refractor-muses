from __future__ import annotations
import refractor.framework as rf  # type: ignore
from refractor.muses import (
    StateElementHandleSet,
    StateElementOspFileHandle,
    StateElementFillValueHandle,
    StateElementOspFile,
    StateElementIdentifier,
    InstrumentIdentifier,
    MusesOmiObservation,
    MusesObservation,
    FullGridMappedArray,
)
import numpy as np
from pathlib import Path
from typing import cast, Self
import typing

if typing.TYPE_CHECKING:
    from refractor.muses import (
        MusesStrategy,
        ObservationHandleSet,
        SoundingMetadata,
        RetrievalConfiguration,
        MeasurementId,
    )
    from refractor.old_py_retrieve_wrapper import StateElementOldWrapper


def add_handle(
    sname: str,
    constraint_value: float,
    cls: type[StateElementOspFile] = StateElementOspFile,
) -> None:
    StateElementHandleSet.add_default_handle(
        StateElementOspFileHandle(
            StateElementIdentifier(sname),
            np.array([constraint_value]).view(FullGridMappedArray),
            np.array([constraint_value]).view(FullGridMappedArray),
            cls=cls,
        ),
        priority_order=2,
    )


def add_fill_handle(
    sname: str,
) -> None:
    StateElementHandleSet.add_default_handle(
        StateElementFillValueHandle(StateElementIdentifier(sname)), priority_order=1
    )


class StateElementOmiCloudFraction(StateElementOspFile):
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

    @classmethod
    def create_from_handle(
        cls,
        state_element_id: StateElementIdentifier,
        pressure_list_fm: FullGridMappedArray | None,
        value_fm: FullGridMappedArray | None,
        constraint_vector_fm: FullGridMappedArray | None,
        measurement_id: MeasurementId,
        retrieval_config: RetrievalConfiguration,
        strategy: MusesStrategy,
        observation_handle_set: ObservationHandleSet,
        sounding_metadata: SoundingMetadata,
        spectral_domain: rf.SpectralDomain | None = None,
        selem_wrapper: StateElementOldWrapper | None = None,
        cov_is_constraint: bool = False,
        poltype: str | None = None,
        poltype_used_constraint: bool = True,
        diag_cov: bool = False,
        diag_directory: Path | None = None,
    ) -> Self | None:
        """Create object from the set of parameter the StateElementOspFileHandle supplies.

        We don't actually use all the arguments, but they are there for other classes
        """
        if InstrumentIdentifier("OMI") not in strategy.instrument_name:
            return None
        obs = observation_handle_set.observation(
            InstrumentIdentifier("OMI"),
            None,
            None,
            None,
            osp_dir=retrieval_config.osp_dir,
        )
        res = cls(
            state_element_id,
            obs,
            sounding_metadata.latitude.value,
            sounding_metadata.surface_type,
            Path(retrieval_config["speciesDirectory"]),
            Path(retrieval_config["covarianceDirectory"]),
            selem_wrapper=selem_wrapper,
            cov_is_constraint=cov_is_constraint,
        )
        return res


class StateElementOmiSurfaceAlbedo(StateElementOspFile):
    """Variation that gets the apriori/initial guess from the observation file"""

    def __init__(
        self,
        state_element_id: StateElementIdentifier,
        obs: MusesOmiObservation,
        latitude: float,
        surface_type: str,
        species_directory: Path,
        covariance_directory: Path,
        selem_wrapper: StateElementOldWrapper | None = None,
        cov_is_constraint: bool = False,
    ) -> None:
        constraint_vector_fm = np.array([obs.monthly_minimum_surface_reflectance]).view(
            FullGridMappedArray
        )
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

    @classmethod
    def create_from_handle(
        cls,
        state_element_id: StateElementIdentifier,
        pressure_list_fm: FullGridMappedArray | None,
        value_fm: FullGridMappedArray | None,
        constraint_vector_fm: FullGridMappedArray | None,
        measurement_id: MeasurementId,
        retrieval_config: RetrievalConfiguration,
        strategy: MusesStrategy,
        observation_handle_set: ObservationHandleSet,
        sounding_metadata: SoundingMetadata,
        spectral_domain: rf.SpectralDomain | None = None,
        selem_wrapper: StateElementOldWrapper | None = None,
        cov_is_constraint: bool = False,
        poltype: str | None = None,
        poltype_used_constraint: bool = True,
        diag_cov: bool = False,
        diag_directory: Path | None = None,
    ) -> Self | None:
        """Create object from the set of parameter the StateElementOspFileHandle supplies.

        We don't actually use all the arguments, but they are there for other classes
        """
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
        res = cls(
            state_element_id,
            obs,
            sounding_metadata.latitude.value,
            sounding_metadata.surface_type,
            Path(retrieval_config["speciesDirectory"]),
            Path(retrieval_config["covarianceDirectory"]),
            selem_wrapper=selem_wrapper,
            cov_is_constraint=cov_is_constraint,
        )
        return res


add_handle("OMICLOUDFRACTION", -999.0, StateElementOmiCloudFraction)
# This gets created even if we don't have OMI data
add_fill_handle("OMICLOUDFRACTION")
add_handle("OMINRADWAVUV1", 0.0)
add_handle("OMINRADWAVUV2", 0.0)
add_handle("OMIODWAVSLOPEUV1", 0.0)
add_handle("OMIODWAVSLOPEUV2", 0.0)
add_handle("OMIODWAVUV1", 0.0)
add_handle("OMIODWAVUV2", 0.0)
# Doesn't actually seem to be in muses-py, although there is a species file for this
# add_handle("OMIRESSCALE", 0.0)
add_handle("OMIRINGSFUV1", 1.9)
add_handle("OMIRINGSFUV2", 1.9)
add_handle("OMISURFACEALBEDOSLOPEUV2", 0.0)
add_handle("OMISURFACEALBEDOUV1", -999.0, StateElementOmiSurfaceAlbedo)
add_handle("OMISURFACEALBEDOUV2", -999.0, StateElementOmiSurfaceAlbedo)
add_fill_handle("OMISURFACEALBEDOUV1")
add_fill_handle("OMISURFACEALBEDOUV2")

__all__ = ["StateElementOmiCloudFraction", "StateElementOmiSurfaceAlbedo"]

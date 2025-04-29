from __future__ import annotations

# Note h_old will go away, right now this
# is just to compare against StateInfoOld, until we have everything tested out.
from refractor.muses.current_state_state_info import h_old
from refractor.muses import (
    StateElementHandleSet,
    StateElementOspFileHandle,
    StateElementOspFile,
    StateElementIdentifier,
    InstrumentIdentifier
)
import numpy as np


# Register all the OMI specific state elements. Note h_old will go away, right now this
# is just to compare against StateInfoOld, until we have everything tested out.
def add_handle(sname: str, apriori_value: float, cls: type[StateElementOspFile] = StateElementOspFile) -> None:
    StateElementHandleSet.add_default_handle(
        StateElementOspFileHandle(
            StateElementIdentifier(sname), np.array([apriori_value]), h_old, cls=cls
        )
    )


class StateElementOmiCloudFraction(StateElementOspFile):
    '''Variation that gets the apriori/initial guess from the observation file'''
    def __init__(
        self,
        state_element_id: StateElementIdentifier,
        apriori_value: np.ndarray,
        measurement_id: MeasurementId,
        retrieval_config: RetrievalConfiguration,
        strategy: MusesStrategy,
        observation_handle_set: ObservationHandleSet,
        sounding_metadata: SoundingMetadata,
        selem_wrapper: StateElementOldWrapper | None = None,
    ):
        obs = observation_handle_set.observation(InstrumentIdentifier("OMI"), None,None,None,
                                                 osp_dir=retrieval_config.osp_dir)
        apriori_value = np.array([obs.cloud_fraction,])
        super().__init__(state_element_id, apriori_value, measurement_id, retrieval_config,
                         strategy, observation_handle_set, sounding_metadata, selem_wrapper)

class StateElementOmiSurfaceAlbedo(StateElementOspFile):
    '''Variation that gets the apriori/initial guess from the observation file'''
    def __init__(
        self,
        state_element_id: StateElementIdentifier,
        apriori_value: np.ndarray,
        measurement_id: MeasurementId,
        retrieval_config: RetrievalConfiguration,
        strategy: MusesStrategy,
        observation_handle_set: ObservationHandleSet,
        sounding_metadata: SoundingMetadata,
        selem_wrapper: StateElementOldWrapper | None = None,
    ):
        obs = observation_handle_set.observation(InstrumentIdentifier("OMI"), None,None,None,
                                                 osp_dir=retrieval_config.osp_dir)
        apriori_value = np.array([obs.monthly_minimum_surface_reflectance,])
        super().__init__(state_element_id, apriori_value, measurement_id, retrieval_config,
                         strategy, observation_handle_set, sounding_metadata, selem_wrapper)
        
#add_handle("OMISURFACEALBEDOUV1", -999.0, StateElementOmiSurfaceAlbedo)
#add_handle("OMISURFACEALBEDOUV2", -999.0, StateElementOmiSurfaceAlbedo)
add_handle("OMISURFACEALBEDOSLOPEUV2", 0.0)
add_handle("OMIRADWAVUV1", 0.0)
add_handle("OMIRADWAVUV2", 0.0)
add_handle("OMIODWAVUV1", 0.0)
add_handle("OMIODWAVUV2", 0.0)
add_handle("OMIODWAVSLOPEUV1", 0.0)
add_handle("OMIODWAVSLOPEUV2", 0.0)
add_handle("OMIRINGSFUV1", 1.9)
add_handle("OMIRINGSFUV2", 1.9)
#add_handle("OMICLOUDFRACTION", -999.0, StateElementOmiCloudFraction)

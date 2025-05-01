from __future__ import annotations
# Temp
from refractor.muses.current_state_state_info import h_old
from refractor.muses import (
    StateElementHandleSet,
    StateElementOspFileHandle,
    StateElementFillValueHandle,
    StateElementOspFile,
    StateElementIdentifier,
    InstrumentIdentifier,
    MusesOmiObservation,
    MusesObservation,
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
        StateElementOldWrapper,
        RetrievalConfiguration,
        MeasurementId,
    )


def add_handle(
    sname: str,
    apriori_value: float,
    cls: type[StateElementOspFile] = StateElementOspFile,
) -> None:
    StateElementHandleSet.add_default_handle(
        StateElementOspFileHandle(
            StateElementIdentifier(sname), np.array([apriori_value]), h_old, cls=cls
        ), priority_order = 2
    )

#add_handle("TROPOMICLOUDFRACTION", 0.0)
#add_handle("TROPOMICLOUDPRESSURE", 0.0)
add_handle("TROPOMICLOUDSURFACEALBEDO", 0.8)
add_handle("TROPOMIRADIANCESHIFTBAND1", 0.0)
add_handle("TROPOMIRADIANCESHIFTBAND2", 0.0)
add_handle("TROPOMIRADIANCESHIFTBAND3", 0.0)
add_handle("TROPOMIRADIANCESHIFTBAND7", 0.0)
add_handle("TROPOMIRADSQUEEZEBAND1", 0.0)
add_handle("TROPOMIRADSQUEEZEBAND2", 0.0)
add_handle("TROPOMIRADSQUEEZEBAND3", 0.0)
add_handle("TROPOMIRADSQUEEZEBAND7", 0.0)
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
add_handle("TROPOMIRESSCALEO0BAND7", 1.0)
add_handle("TROPOMIRESSCALEO1BAND7", 0.0)
add_handle("TROPOMIRESSCALEO2BAND7", 0.0)
add_handle("TROPOMIRINGSFBAND1", 1.9)
add_handle("TROPOMIRINGSFBAND2", 1.9)
add_handle("TROPOMIRINGSFBAND3", 1.9)
add_handle("TROPOMIRINGSFBAND7", 1.9)
add_handle("TROPOMISOLARSHIFTBAND1", 0.0)
add_handle("TROPOMISOLARSHIFTBAND2", 0.0)
add_handle("TROPOMISOLARSHIFTBAND3", 0.0)
add_handle("TROPOMISOLARSHIFTBAND7", 0.0)
#add_handle("TROPOMISURFACEALBEDOBAND1", 0.0)
#add_handle("TROPOMISURFACEALBEDOBAND2", 0.0)
#add_handle("TROPOMISURFACEALBEDOBAND3", 0.0)
#add_handle("TROPOMISURFACEALBEDOBAND7", 0.0)
add_handle("TROPOMISURFACEALBEDOSLOPEBAND1", 0.0)
add_handle("TROPOMISURFACEALBEDOSLOPEBAND2", 0.0)
add_handle("TROPOMISURFACEALBEDOSLOPEBAND3", 0.0)
add_handle("TROPOMISURFACEALBEDOSLOPEBAND7", 0.0)
add_handle("TROPOMISURFACEALBEDOSLOPEORDER2BAND2", 0.0)
add_handle("TROPOMISURFACEALBEDOSLOPEORDER2BAND3", 0.0)
add_handle("TROPOMISURFACEALBEDOSLOPEORDER2BAND7", 0.0)
add_handle("TROPOMITEMPSHIFTBAND1", 1.0)
add_handle("TROPOMITEMPSHIFTBAND2", 1.0)
add_handle("TROPOMITEMPSHIFTBAND3", 1.0)
add_handle("TROPOMITEMPSHIFTBAND7", 1.0)


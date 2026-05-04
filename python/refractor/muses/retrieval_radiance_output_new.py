from __future__ import annotations
import numpy as np
from .declarative_output import register_dataset, DeclarativeOutput
from .identifier import ProcessLocation
from .retrieval_output_file import RetrievalOutputFile
from loguru import logger
from pathlib import Path
import os
from typing import Any
import typing

if typing.TYPE_CHECKING:
    from .retrieval_strategy_step import RetrievalStrategyStep
    from .current_state import CurrentState


class RetrievalRadianceOutputNew(DeclarativeOutput):
    """New version of RetrievalRadianceOutput, that uses the DeclarativeOutput interface.
    We will likely rename the old RetrievalRadianceOutput to RetrievalRadianceOutputOld,
    and rename this to RetrievalRadianceOutput when we have this all in place. But for
    now leave the old one in place and have this as the "new" version."""

    def __init__(self, output_filename: str | os.PathLike[str]) -> None:
        self.output_filename = Path(output_filename)
        self.output = RetrievalOutputFile(output_filename)
        self.output.register_instances((self,))

    def notify_process_location(
        self,
        location: ProcessLocation,
        current_state: CurrentState,
        retrieval_strategy_step: RetrievalStrategyStep,
        **kwargs: Any,
    ) -> None:
        logger.debug(f"Call to {self.__class__.__name__}::notify_update")
        self.current_state = current_state
        self.retrieval_strategy_step = retrieval_strategy_step
        self.write()

    @register_dataset("/RADIANCEFULLBAND")
    def radiance_full_band(self) -> np.ndarray:
        # Placeholder
        return np.zeros((2223,))

    @register_dataset("/FREQUENCYFULLBAND")
    def frequency_full_band(self) -> np.ndarray:
        # Placeholder
        return np.zeros((2223,))

    @register_dataset("/NESRFULLBAND")
    def nesr_full_band(self) -> np.ndarray:
        # Placeholder
        return np.zeros((2223,))

    @register_dataset("/RADIANCEFIT")
    def radiance_fit(self) -> np.ndarray:
        return np.zeros((37,), dtype=np.float32)

    @register_dataset("/RADIANCEFITINITIAL")
    def radiance_fit_initial(self) -> np.ndarray:
        return np.zeros((37,), dtype=np.float32)

    @register_dataset("/RADIANCEOBSERVED")
    def radiance_observed(self) -> np.ndarray:
        return np.zeros((37,), dtype=np.float32)

    @register_dataset("/NESR")
    def nesr(self) -> np.ndarray:
        return np.zeros((37,), dtype=np.float32)

    @register_dataset("/FREQUENCY")
    def frequency(self) -> np.ndarray:
        return np.zeros((37,), dtype=np.float32)

    @register_dataset("/EMIS")
    def emis(self) -> np.ndarray:
        return np.zeros((121,), dtype=np.float32)

    @register_dataset("/EMISFREQ")
    def emis_freq(self) -> np.ndarray:
        return np.zeros((121,), dtype=np.float32)

    @register_dataset("/CLOUD")
    def cloud(self) -> np.ndarray:
        return np.zeros((28,), dtype=np.float32)

    @register_dataset("/CLOUDFREQ")
    def cloud_freq(self) -> np.ndarray:
        return np.zeros((28,), dtype=np.float32)

    @register_dataset("/LATITUDE")
    def latitude(self) -> np.float32:
        return np.float32(1.0)

    @register_dataset("/LONGITUDE")
    def longitude(self) -> np.float32:
        return np.float32(1.0)

    @register_dataset("/SURFACEALTITUDEMETERS")
    def surface_altitude(self) -> np.float32:
        return np.float32(1.0)

    @register_dataset("/RADIANCERESIDUALMEAN")
    def radiance_residual_mean(self) -> np.float32:
        return np.float32(1.0)

    @register_dataset("/RADIANCERESIDUALRMS")
    def radiance_residual_rms(self) -> np.float32:
        return np.float32(1.0)

    @register_dataset("/CLOUDOPTICALDEPTH")
    def cloud_optical_depth(self) -> np.float32:
        return np.float32(1.0)

    @register_dataset("/CLOUDTOPPRESSURE")
    def cloud_top_pressure(self) -> np.float32:
        return np.float32(1.0)

    @register_dataset("/SURFACETEMPERATURE")
    def surface_temperature(self) -> np.float32:
        return np.float32(1.0)

    @register_dataset("/LAND")
    def land(self) -> np.int16:
        return np.int16(1.0)

    @register_dataset("/QUALITY")
    def quality(self) -> np.int16:
        return np.int16(1.0)

    @register_dataset("/SCANDIRECTION")
    def scan_direction(self) -> np.int16:
        return np.int16(1.0)

    @register_dataset("/TIME")
    def time(self) -> float:
        return 1.0

    @register_dataset("/SOUNDINGID")
    def sounding_id(self) -> np.ndarray:
        # This really seems kind of hokey, but rather than just setting
        # a string this gets written as an array of bytes. Match this old
        # behavior, since things down stream probably depend on this
        return np.array(
            [
                [np.int8(i) for i in bytearray(b"20190807_065_04_08_5")],
            ],
            dtype=np.int8,
        )

    def write(self) -> None:
        self.output.write()


__all__ = [
    "RetrievalRadianceOutputNew",
]

from __future__ import annotations
from refractor.muses import (
    SpectralWindowHandle,
    MusesSpectralWindow,
    InstrumentIdentifier,
    RetrievalType,
    FilterIdentifier,
    CurrentStrategyStep,
    RetrievalConfiguration,
    MeasurementId,
)
from loguru import logger


class DummySpectralWindowHandle(SpectralWindowHandle):
    """The machine learning code doesn't actually use a spectral window. Our
    existing code is very much OE centered, and expects there to be a spectral
    window dict. We should perhaps relax that, but I think we need to get more
    experience with the ML retrievals before we know how to do that.

    For now, if the retrieval step is type "ML", we just look in the MeasurementId
    to see what instruments are listed. We then create a dummy dict that goes from
    those instruments to a placeholder MusesSpectralWindow.

    This is slightly kludgey, but until we know what the updated interface should be
    this seems like a reasonable workaround.
    """

    def __init__(self) -> None:
        self.retrieval_config: None | RetrievalConfiguration = None
        self.measurement_id: None | MeasurementId = None

    def notify_update_target(
        self, measurement_id: MeasurementId, retrieval_config: RetrievalConfiguration
    ) -> None:
        logger.debug(f"Call to {self.__class__.__name__}::notify_update")
        self.measurement_id = measurement_id
        self.retrieval_config = retrieval_config

    def spectral_window_dict(
        self,
        current_strategy_step: CurrentStrategyStep,
        filter_list_all_dict: dict[InstrumentIdentifier, list[FilterIdentifier]] | None,
    ) -> dict[InstrumentIdentifier, MusesSpectralWindow] | None:
        if self.retrieval_config is None or self.measurement_id is None:
            raise RuntimeError("Call notify_update_target before this function")
        if current_strategy_step.retrieval_type != RetrievalType("ML"):
            return None
        iset: set[InstrumentIdentifier] = set()
        for vname, iname in (
            ("AIRS_filename", "AIRS"),
            ("OMI_filename", "OMI"),
            ("CRIS_filename", "CRIS"),
            ("TES_filename_L2", "TES"),
            ("TES_filename_L1B", "TES"),
            ("TROPOMI_filename_BAND3", "TROPOMI"),
            ("TROPOMI_filename_BAND7", "TROPOMI"),
            ("TROPOMI_filename_BAND8", "TROPOMI"),
        ):
            if vname in self.measurement_id:
                iset.add(InstrumentIdentifier(iname))
        # If we don't recognize any of the instruments, we can't create a dict, so
        # return None
        if len(iset) == 0:
            return None
        logger.debug("Creating spectral_window_dict using DummySpectralWindowHandle")
        res = {}
        for ins in iset:
            res[ins] = MusesSpectralWindow(None, None)
        return res


__all__ = [
    "DummySpectralWindowHandle",
]

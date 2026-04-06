from __future__ import annotations
from refractor.muses import (
    SpectralWindowHandle,
    MusesSpectralWindow,
    InstrumentIdentifier,
    RetrievalType,
    FilterIdentifier,
    CurrentStrategyStep,
)
from loguru import logger

# TODO Hopefully this can go away when we rework strategy step


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

    def spectral_window_dict(
        self,
        current_strategy_step: CurrentStrategyStep,
        filter_list_all_dict: dict[InstrumentIdentifier, list[FilterIdentifier]] | None,
    ) -> dict[InstrumentIdentifier, MusesSpectralWindow] | None:
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
            if vname in ("CRIS_filename"):
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

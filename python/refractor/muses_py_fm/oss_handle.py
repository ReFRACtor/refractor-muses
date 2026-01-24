from __future__ import annotations
from contextlib import contextmanager

# TODO, We will replace this functionality
from .osswrapper import osswrapper
from .mpy import mpy_fm_oss_stack
import numpy as np
from typing import Self, Iterator
import typing

if typing.TYPE_CHECKING:
    from .refractor_uip import RefractorUip
    from refractor.muses import InputFileHelper, InstrumentIdentifier


class OssHandle:
    """We use to do mpy_fm_oss_init and mpy_fm_oss_delete each time we did a OSS
    retrieval. It turns out this is relatively expense, there is a lot of work done
    in the initialization. Most of the time, we are calling the same exact initialization.
    So we have pulled this out into a handler class. On initialization, we skip and
    use our previous initialization unless something has changed."""

    def __init__(self) -> None:
        pass

    @contextmanager
    def handle(
        self, rf_uip: RefractorUip, ifile_hlp: InputFileHelper | None
    ) -> Iterator[Self]:
        if ifile_hlp is None:
            from refractor.muses import InputFileHelper

            self.ifile_hlp = InputFileHelper()
        else:
            self.ifile_hlp = ifile_hlp
        self.rf_uip = rf_uip
        assert ifile_hlp is not None
        with osswrapper(self.rf_uip.uip, ifile_hlp):
            yield self

    def radiance_and_jacobian(
        self, instrument_name: InstrumentIdentifier
    ) -> tuple[np.ndarray, np.ndarray]:
        """Call OSS to get the radiance and jacobian"""
        return mpy_fm_oss_stack(self.rf_uip.uip_all(instrument_name))


oss_handle = OssHandle()

__all__ = [
    "oss_handle",
]

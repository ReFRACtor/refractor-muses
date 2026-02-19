from __future__ import annotations
import refractor.framework as rf  # type: ignore
from .muses_oss_handle import muses_oss_handle
import numpy as np
import os
import typing
from typing import Self

if typing.TYPE_CHECKING:
    from .identifier import StateElementIdentifier
    from .input_file_helper import InputFilePath, InputFileHelper

class MusesRadiativeTransferOss(rf.RadiativeTransferImpBase):
    """This uses the muses OSS code (package muses-oss). This gives a forward
    model that is the same as the py-retrieve airs/cris/tes forward model (with minor
    differences in calculation - the normal sort of round off differences).
    """

    def __init__(
        self,
        ifile_hlp: InputFileHelper,
        retrieval_state_element_id: list[StateElementIdentifier],
        species_list: list[StateElementIdentifier],
        nlevels: int,
        nfreq: int, # This seems to be the size of the emissivity. Perhaps verify,
                    # And if so change it name. This has nothing to do with the
                    # size of freq_oss that gets filled in
        sel_file : str | os.PathLike[str] | InputFilePath,
        od_file : str | os.PathLike[str] | InputFilePath,
        sol_file : str | os.PathLike[str] | InputFilePath,
        fix_file : str | os.PathLike[str] | InputFilePath,
    ) -> None:
        super().__init__()
        self.ifile_hlp = ifile_hlp
        self.retrieval_state_element_id = retrieval_state_element_id
        self.species_list = species_list
        self.nlevels = nlevels
        self.nfreq = nfreq
        self.sel_file = sel_file
        self.od_file = od_file
        self.sol_file = sol_file
        self.fix_file = fix_file


    def clone(self) -> Self:
        return MusesRadiativeTransferOss(
            self.ifile_hlp,
            self.retrieval_state_element_id,
            self.species_list,
            self.nlevels,
            self.nfreq,
            self.sel_file,
            self.od_file,
            self.sol_file,
            self.fix_file,
        )

    def reflectance(
        self, sd: rf.SpectralDomain, sensor_index: int, skip_jacobian: bool
    ) -> rf.Spectrum:
        muses_oss_handle.oss_init(self.ifile_hlp, self.retrieval_state_element_id,
                                  self.species_list, self.nlevels,
                                  self.nfreq,
                                  self.sel_file,
                                  self.od_file,
                                  self.sol_file,
                                  self.fix_file)
        a = rf.ArrayAd_double_1(np.zeros(sd.data.shape))
        sr = rf.SpectralRange(a, rf.Unit("sr^-1"))
        return rf.Spectrum(sd, sr)
        
    def stokes(self, sd: rf.SpectralDomain, sensor_index: int) -> np.ndarray:
        raise NotImplementedError(
            """Muses-oss doesn't work for  the full
            stoke vector."""
        )

    def stokes_and_jacobian(
        self, sd: rf.SpectralDomain, sensor_index: int
    ) -> rf.ArrayAd_double_2:
        raise NotImplementedError(
            """Muses-oss doesn't work for  the full
            stoke vector."""
        )

    def desc(self) -> str:
        return "MusesRadiativeTransferOss"


__all__ = [
    "MusesRadiativeTransferOss",
]

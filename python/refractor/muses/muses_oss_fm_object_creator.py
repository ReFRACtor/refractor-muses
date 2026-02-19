from __future__ import annotations
from .refractor_fm_object_creator import RefractorFmObjectCreator
import refractor.framework as rf  # type: ignore
from .identifier import InstrumentIdentifier, StateElementIdentifier
from .muses_radiative_transfer_oss import MusesRadiativeTransferOss
import os
from pathlib import Path
from functools import cached_property
import typing

if typing.TYPE_CHECKING:
    from .input_file_helper import InputFilePath
    from .current_state import CurrentState
    from .muses_observation import MusesObservation
    from .retrieval_configuration import RetrievalConfiguration

# Leverage off RefractorFmObjectCreator. We probably want to
# rework this, either make MusesOssFmObjectCreator stand alone
# or extract out a common base class. Right now RefractorFmObjectCreator
# has a lot of stuff in it that we don't need for OSS.
class MusesOssFmObjectCreator(RefractorFmObjectCreator):
    def __init__(
        self,
        current_state: CurrentState,
        retrieval_config: RetrievalConfiguration,
        observation: MusesObservation,
        fm_sv: rf.StateVector | None = None,
        dir_lut: Path | InputFilePath | None = None,
        ):
        super().__init__(current_state, retrieval_config, observation, fm_sv)
        self.dir_lut=dir_lut
        # Filled in by derived classes
        self.retrieval_state_element_id: list[StateElementIdentifier] = []
        self.species_list: list[StateElementIdentifier] = []
        self.nlevels = -1
        self.nfreq = -1
        self.sel_file : str | os.PathLike[str] | InputFilePath = ""
        self.od_file : str | os.PathLike[str] | InputFilePath = ""
        self.sol_file : str | os.PathLike[str] | InputFilePath = ""
        self.fix_file : str | os.PathLike[str] | InputFilePath = ""
        

    def ils_method(self, sensor_index: int) -> str:
        """Return the ILS method to use. This is APPLY, POSTCONV, or FASTCONV."""
        # We don't do ILS with OSS. I suppose it is possible we might at some
        # point, but for now just have this always be APPLY
        return "APPLY"

    @cached_property
    def spectrum_effect(self) -> list[rf.SpectrumEffect]:
        # No spectrum effects currently, although it is possible something
        # like radiance scaling might be a useful option.
        res = []
        for i in range(self.num_channels):
            per_channel_eff = []
            res.append(per_channel_eff)
        return res

    @cached_property
    def radiative_transfer(self) -> rf.RadiativeTransfer:
        return MusesRadiativeTransferOss(self.ifile_hlp,
                                         self.retrieval_state_element_id,
                                         self.species_list, self.nlevels, self.nfreq,
                                         self.sel_file,
                                         self.od_file, self.sol_file, self.fix_file)
    
    
    @cached_property
    def forward_model(self) -> rf.ForwardModel:
        res = rf.StandardForwardModel(
            self.instrument,
            self.spec_win,
            self.radiative_transfer,
            self.spectrum_sampling,
            self.spectrum_effect,
        )
        res.setup_grid()
        return res

    # These should probably be pushed down to a lidort FmObjectCreator
    @cached_property
    def cloud_fraction(self) -> rf.CloudFraction:
        raise NotImplementedError
    
    @cached_property
    def ground_clear(self) -> rf.Ground:
        raise NotImplementedError

    @cached_property
    def ground_cloud(self) -> rf.Ground:
        raise NotImplementedError

    def instrument_hwhm(self, sensor_index: int) -> rf.DoubleWithUnit:
        """Grating spectrometers like OMI and TROPOMI require a fixed
        half width at half max for the IlsGrating object. This can
        vary from band to band. This function must return the HWHM in
        wavenumbers for the band indicated by `sensor_index`<`, which
        will be the index from `self.channel_list()` for the current
        band.
        """
        raise NotImplementedError
    
        

class CrisFmObjectCreator(MusesOssFmObjectCreator):
    def __init__(
        self,
        current_state: CurrentState,
        retrieval_config: RetrievalConfiguration,
        observation: MusesObservation,
        fm_sv: rf.StateVector | None = None,
        dir_lut: Path | InputFilePath | None = None,
        ):
        super().__init__(current_state, retrieval_config, observation, fm_sv=fm_sv, dir_lut=dir_lut)
        # Different files depends on l1b_type
        if self.observation.instrument_name  == InstrumentIdentifier("CRIS", "suomi_nasa_nsr"):
            if self.dir_lut is None:
                self.dir_lut = self.ifile_hlp.osp_dir / "OSS_FM" / "CRIS" / "2023-01-nsr"
            self.sel_file = (
                    self.dir_lut
                    / "suomi-cris-B1B2B3-unapod-loc-clear-19V-M12.4-v1.0.train.sel"
            )
            self.od_file = (
                self.dir_lut
                / "suomi-cris-B1B2B3-unapod-loc-clear-19V-M12.4-v1.0.train.lut"
            )
            self.sol_file = self.dir_lut / "newkur.dat"
            self.fix_file = self.dir_lut / "default.dat"
        else:
            if self.dir_lut is None:
                self.dir_lut = self.ifile_hlp.osp_dir / "OSS_FM" / "CRIS" / "2017-08"
            self.sel_file = (
                self.dir_lut
                / "suomi-cris-fsr-B1B2B3-unapod-loc-cloudy-23V-M12.4-v1.0.train.sel"
            )

            self.od_file = (
                self.dir_lut
                / "suomi-cris-fsr-B1B2B3-unapod-loc-cloudy-23V-M12.4-v1.0.train.lut"
            )

            self.sol_file = self.dir_lut / "newkur.dat"
            self.fix_file = self.dir_lut / "default.dat"
            # The retrieval and species list seem to be hardcoded. I think this
            # corresponds to what is available in the various input files
            self.retrieval_state_element_id = [
                StateElementIdentifier(i)
                for i in ["H2O", "O3", "TSUR", "EMIS", "CLOUDEXT", "PCLOUD"]
            ]
            self.species_list = [
                StateElementIdentifier(i)
                for i in [
                    "PRESSURE",
                    "TATM",
                    "H2O",
                    "CO2",
                    "O3",
                    "N2O",
                    "CO",
                    "CH4",
                    "SO2",
                    "NH3",
                    "HNO3",
                    "OCS",
                    "N2",
                    "HCN",
                    "SF6",
                    "HCOOH",
                    "CCL4",
                    "CFC11",
                    "CFC12",
                    "CFC22",
                    "HDO",
                    "CH3OH",
                    "C2H4",
                    "PAN",
                ]
            ]
            # We need to come up with a way to get these values
            self.nlevels = 64
            self.nfreq = 121
    
        
    

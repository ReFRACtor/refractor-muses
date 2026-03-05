from __future__ import annotations
from .refractor_fm_object_creator import RefractorFmObjectCreator
import refractor.framework as rf  # type: ignore
from .identifier import InstrumentIdentifier, StateElementIdentifier
from .muses_radiative_transfer_oss import MusesRadiativeTransferOss
from .forward_model_handle import ForwardModelHandle, ForwardModelHandleSet
from .compare_forward_model import CompareForwardModel
from .irk_forward_model import IrkForwardModel
from .muses_tes_observation import MusesTesObservation
from .emis_state import EmisState
from .cloud_ext_state import CloudExtState
from .pointing_angle_surface import pointing_angle_surface
import os
from pathlib import Path
from loguru import logger
from functools import cached_property
import typing
from typing import Any

if typing.TYPE_CHECKING:
    from .input_file_helper import InputFilePath
    from .current_state import CurrentState
    from .muses_observation import MusesObservation, MeasurementId
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
        self.dir_lut = dir_lut
        # Filled in by derived classes
        self.species_list: list[StateElementIdentifier] = []
        self.nlevels = -1
        self.nfreq = -1
        self.sel_file: str | os.PathLike[str] | InputFilePath = ""
        self.od_file: str | os.PathLike[str] | InputFilePath = ""
        self.sol_file: str | os.PathLike[str] | InputFilePath = ""
        self.fix_file: str | os.PathLike[str] | InputFilePath = ""

    def ils_method(self, sensor_index: int) -> str:
        """Return the ILS method to use. This is APPLY, POSTCONV, or FASTCONV."""
        # We don't do ILS with OSS. I suppose it is possible we might at some
        # point, but for now just have this always be APPLY
        return "APPLY"

    @cached_property
    def surface_temperature(self) -> rf.SurfaceTemperature:
        selem = [
            StateElementIdentifier("TSUR"),
        ]
        stemp, mp = self.current_state.object_state(selem)
        tsur = rf.SurfaceTemperatureDirect(
            rf.ArrayWithUnit_double_1(stemp, rf.Unit("K")), mp
        )
        self.current_state.add_fm_state_vector_if_needed(
            self.fm_sv,
            selem,
            [tsur],
        )
        return tsur

    @cached_property
    def emissivity(self) -> EmisState:
        selem = [
            StateElementIdentifier("EMIS"),
        ]
        semis, mp = self.current_state.object_state(selem)
        semis_sd = self.current_state.state_element(selem[0]).spectral_domain
        emis = EmisState(semis, semis_sd, mp)
        self.current_state.add_fm_state_vector_if_needed(
            self.fm_sv,
            selem,
            [emis],
        )
        return emis

    @cached_property
    def cloud_ext(self) -> CloudExtState:
        selem = [
            StateElementIdentifier("CLOUDEXT"),
        ]
        scloudext, mp = self.current_state.object_state(selem)
        scloudext_sd = self.current_state.state_element(selem[0]).spectral_domain
        cext = CloudExtState(scloudext, scloudext_sd, mp)
        self.current_state.add_fm_state_vector_if_needed(
            self.fm_sv,
            selem,
            [cext],
        )
        return cext

    @cached_property
    def pcloud(self) -> rf.Pcloud:
        selem = [
            StateElementIdentifier("PCLOUD"),
        ]
        spcloud, mp = self.current_state.object_state(selem)
        pcloud = rf.PcloudDirect(rf.ArrayWithUnit_double_1(spcloud, rf.Unit("hPa")), mp)
        self.current_state.add_fm_state_vector_if_needed(
            self.fm_sv,
            selem,
            [pcloud],
        )
        return pcloud

    @cached_property
    def scale_cloud(self) -> rf.ScalePressure:
        # Not sure why this is called scalePressure, that is just the name used
        # in py-retrieve. But this is used by the OSS code to give the
        # scale of cloud log thickness (see oss_ir_module.f90 in muses-oss).
        # So we cal this scale_cloud, since that is what this is.
        selem = [
            StateElementIdentifier("scalePressure"),
        ]
        ssclp, mp = self.current_state.object_state(selem)
        sclp = rf.ScaleCloudDirect(ssclp, mp)
        self.current_state.add_fm_state_vector_if_needed(
            self.fm_sv,
            selem,
            [sclp],
        )
        return sclp

    @cached_property
    def spectrum_effect(self) -> list[list[rf.SpectrumEffect]]:
        # No spectrum effects currently, although it is possible something
        # like radiance scaling might be a useful option.
        res = []
        for i in range(self.num_channels):
            per_channel_eff: rf.SpectrumEffect = []
            res.append(per_channel_eff)
        return res

    @cached_property
    def radiative_transfer(self) -> rf.RadiativeTransfer:
        pointing_angle = self.observation.pointing_angle
        # This is a bit involved to get, so leverage off uip until we are
        # ready to work through this
        pangle = pointing_angle_surface(self._rf_uip, self.observation.instrument_name,
                                        pointing_angle)
        print(pointing_angle.convert("deg"))
        print(pangle.convert("deg"))
        breakpoint()
        return MusesRadiativeTransferOss(
            self._rf_uip,
            self.pressure_fm,
            self.temperature,
            self.surface_temperature,
            self.pcloud,
            self.scale_cloud,
            self.emissivity,
            self.cloud_ext,
            self.observation.surface_altitude,
            self.current_state.sounding_metadata.latitude,
            pangle,
            self.observation.instrument_name,
            self.ifile_hlp,
            self.current_state.systematic_state_element_id
            if self.current_state.use_systematic
            else self.current_state.retrieval_state_element_id,
            self.species_list,
            self.nlevels,
            self.nfreq,
            self.sel_file,
            self.od_file,
            self.sol_file,
            self.fix_file,
        )

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


class MusesCrisForwardModelOss(IrkForwardModel):
    """IRK specialization for CRIS"""

    def irk_angle(self) -> list[float]:
        """List of angles in degrees that run forward model for the IRK."""
        return [0.0, 14.2906, 31.8588, 46.9590, 57.3154, 61.5613]


class MusesAirsForwardModelOss(IrkForwardModel):
    """IRK specialization for AIRS"""

    def irk_angle(self) -> list[float]:
        """List of angles in degrees that run forward model for the IRK."""
        return [0.0, 14.5752, 32.5555, 48.1689, 59.0983, 63.6765]

    @cached_property
    def irk_obs(self) -> MusesObservation:
        """Observation to use in IRK calculation."""
        # Replace with a fake TES observation. This is done to get the
        # full TES frequency range.
        tes_frequency_fname = (
            self.rconf["spectralWindowDirectory"].parent.parent / "tes_frequency.nc"
        )
        return MusesTesObservation.create_fake_for_irk(
            tes_frequency_fname, self.obs.spectral_window, self.rconf.input_file_helper
        )


class MusesTesForwardModelOss(IrkForwardModel):
    """IRK specialization for TES"""

    def irk_angle(self) -> list[float]:
        """List of angles in degrees that run forward model for the IRK."""
        return [0.0, 14.5752, 32.5555, 48.1689, 59.0983, 63.6765]


class CrisFmObjectCreator(MusesOssFmObjectCreator):
    def __init__(
        self,
        current_state: CurrentState,
        retrieval_config: RetrievalConfiguration,
        observation: MusesObservation,
        fm_sv: rf.StateVector | None = None,
        dir_lut: Path | InputFilePath | None = None,
    ):
        super().__init__(
            current_state, retrieval_config, observation, fm_sv=fm_sv, dir_lut=dir_lut
        )
        # Different files depends on l1b_type
        if self.observation.instrument_name == InstrumentIdentifier(
            "CRIS", "suomi_nasa_nsr"
        ):
            if self.dir_lut is None:
                self.dir_lut = (
                    self.ifile_hlp.osp_dir / "OSS_FM" / "CRIS" / "2023-01-nsr"
                )
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
        # The species list seem to be hardcoded. I think this
        # corresponds to what is available in the various input files
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
        self.nlevels = self._rf_uip.uip["atmosphere"].shape[1]
        self.nfreq = self._rf_uip.uip["emissivity"]["frequency"].shape[0]

    @cached_property
    def forward_model(self) -> rf.ForwardModel:
        res = MusesCrisForwardModelOss(
            self.instrument,
            self.spec_win,
            self.radiative_transfer,
            self.spectrum_sampling,
            self.spectrum_effect,
            self.observation,
            self.retrieval_config,
        )
        self._add_rf_uip_update_to_fm(res)
        res.setup_grid()
        return res


class AirsFmObjectCreator(MusesOssFmObjectCreator):
    def __init__(
        self,
        current_state: CurrentState,
        retrieval_config: RetrievalConfiguration,
        observation: MusesObservation,
        fm_sv: rf.StateVector | None = None,
        dir_lut: Path | InputFilePath | None = None,
    ):
        super().__init__(
            current_state, retrieval_config, observation, fm_sv=fm_sv, dir_lut=dir_lut
        )
        if self.dir_lut is None:
            self.dir_lut = self.ifile_hlp.osp_dir / "OSS_FM" / "AIRS" / "2017-07"
        self.sel_file = (
            self.dir_lut / "aqua-airs-B1B2B3-unapod-loc-clear-23V-M12.4-v1.0.train.sel"
        )
        self.od_file = (
            self.dir_lut / "aqua-airs-B1B2B3-unapod-loc-clear-23V-M12.4-v1.0.train.lut"
        )
        self.sol_file = self.dir_lut / "newkur.dat"
        self.fix_file = self.dir_lut / "default.dat"

        # The species list seem to be hardcoded. I think this
        # corresponds to what is available in the various input files
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
        self.nlevels = self._rf_uip.uip["atmosphere"].shape[1]
        self.nfreq = self._rf_uip.uip["emissivity"]["frequency"].shape[0]

    @cached_property
    def tes_radiative_transfer(self) -> rf.RadiativeTransfer:
        # For the IRK, we want to use the more full tes frequencies. So create
        # a RT set up for this.
        tes_dir_lut = self.ifile_hlp.osp_dir / "OSS_FM" / "TES" / "2018-03-14"
        tes_sel_file = (
            tes_dir_lut
            / "aqua-tes-B2B11B22A11A1-unapod-loc-clear-23V-M12.4-v1.2.train.sel"
        )
        tes_od_file = (
            tes_dir_lut
            / "aqua-tes-B2B11B22A11A1-unapod-loc-clear-23V-M12.4-v1.2.train.lut"
        )
        tes_sol_file = tes_dir_lut / "newkur.dat"
        tes_fix_file = tes_dir_lut / "default.dat"

        pointing_angle = self.observation.pointing_angle
        # This is a bit involved to get, so leverage off uip until we are
        # ready to work through this
        pointing_angle_surface = rf.DoubleWithUnit(
            self._rf_uip.ray_info(
                self.observation.instrument_name,
                pointing_angle=pointing_angle.convert("rad").value,
            )["ray_angle_surface"],
            "rad",
        )
        return MusesRadiativeTransferOss(
            self._rf_uip,
            self.pressure_fm,
            self.temperature,
            self.surface_temperature,
            self.pcloud,
            self.scale_cloud,
            self.emissivity,
            self.cloud_ext,
            self.observation.surface_altitude,
            self.current_state.sounding_metadata.latitude,
            pointing_angle_surface,
            InstrumentIdentifier("TES"),
            self.ifile_hlp,
            self.current_state.systematic_state_element_id
            if self.current_state.use_systematic
            else self.current_state.retrieval_state_element_id,
            self.species_list,
            self.nlevels,
            self.nfreq,
            tes_sel_file,
            tes_od_file,
            tes_sol_file,
            tes_fix_file,
        )

    @cached_property
    def forward_model(self) -> rf.ForwardModel:
        fm1 = MusesAirsForwardModelOss(
            self.instrument,
            self.spec_win,
            self.radiative_transfer,
            self.spectrum_sampling,
            self.spectrum_effect,
            self.observation,
            self.retrieval_config,
            irk_radiative_transfer=self.tes_radiative_transfer,
        )
        self._add_rf_uip_update_to_fm(fm1)
        if False:
            from refractor.muses_py_fm import MusesAirsForwardModel

            fm2 = MusesAirsForwardModel(
                self.current_state, self.observation, self.retrieval_config
            )
            res = CompareForwardModel(fm2, fm1)
        else:
            res = fm1
        res.setup_grid()
        return res


class TesFmObjectCreator(MusesOssFmObjectCreator):
    def __init__(
        self,
        current_state: CurrentState,
        retrieval_config: RetrievalConfiguration,
        observation: MusesObservation,
        fm_sv: rf.StateVector | None = None,
        dir_lut: Path | InputFilePath | None = None,
    ):
        super().__init__(
            current_state, retrieval_config, observation, fm_sv=fm_sv, dir_lut=dir_lut
        )
        if self.dir_lut is None:
            self.dir_lut = self.ifile_hlp.osp_dir / "OSS_FM" / "TES" / "2018-03-14"
        self.sel_file = (
            self.dir_lut
            / "aqua-tes-B2B11B22A11A1-unapod-loc-clear-23V-M12.4-v1.2.train.sel"
        )
        self.od_file = (
            self.dir_lut
            / "aqua-tes-B2B11B22A11A1-unapod-loc-clear-23V-M12.4-v1.2.train.lut"
        )
        self.sol_file = self.dir_lut / "newkur.dat"
        self.fix_file = self.dir_lut / "default.dat"
        # The species list seem to be hardcoded. I think this
        # corresponds to what is available in the various input files
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
        self.nlevels = self._rf_uip.uip["atmosphere"].shape[1]
        self.nfreq = self._rf_uip.uip["emissivity"]["frequency"].shape[0]

    @cached_property
    def forward_model(self) -> rf.ForwardModel:
        fm1 = MusesTesForwardModelOss(
            self.instrument,
            self.spec_win,
            self.radiative_transfer,
            self.spectrum_sampling,
            self.spectrum_effect,
            self.observation,
            self.retrieval_config,
        )
        self._add_rf_uip_update_to_fm(fm1)
        if False:
            from refractor.muses_py_fm import MusesTesForwardModel

            # If we need to diagnose an issue
            fm2 = MusesTesForwardModel(
                self.current_state, self.observation, self.retrieval_config
            )
            res = CompareForwardModel(fm2, fm1)
        else:
            res = fm1
        res.setup_grid()
        return res


class CrisForwardModelHandle(ForwardModelHandle):
    def __init__(self, **creator_kwargs: Any) -> None:
        self.creator_kwargs = creator_kwargs
        self.retrieval_config: None | RetrievalConfiguration = None

    def notify_update_target(
        self, measurement_id: MeasurementId, retrieval_config: RetrievalConfiguration
    ) -> None:
        """Clear any caching associated with assuming the target being retrieved is fixed"""
        logger.debug(f"Call to {self.__class__.__name__}::notify_update")
        self.retrieval_config = retrieval_config

    def forward_model(
        self,
        instrument_name: InstrumentIdentifier,
        current_state: CurrentState,
        obs: MusesObservation,
        fm_sv: rf.StateVector,
        **kwargs: Any,
    ) -> rf.ForwardModel:
        if instrument_name != InstrumentIdentifier("CRIS"):
            return None
        if self.retrieval_config is None:
            raise RuntimeError("Call notify_update_target first")
        logger.debug("Creating OSS forward model using using CrisFmObjectCreator")
        obj_creator = CrisFmObjectCreator(
            current_state,
            self.retrieval_config,
            obs,
            fm_sv=fm_sv,
            **self.creator_kwargs,
        )
        fm = obj_creator.forward_model
        logger.info(f"Cris Forward model\n{fm}")
        return fm


class AirsForwardModelHandle(ForwardModelHandle):
    def __init__(self, **creator_kwargs: Any) -> None:
        self.creator_kwargs = creator_kwargs
        self.retrieval_config: None | RetrievalConfiguration = None

    def notify_update_target(
        self, measurement_id: MeasurementId, retrieval_config: RetrievalConfiguration
    ) -> None:
        """Clear any caching associated with assuming the target being retrieved is fixed"""
        logger.debug(f"Call to {self.__class__.__name__}::notify_update")
        self.retrieval_config = retrieval_config

    def forward_model(
        self,
        instrument_name: InstrumentIdentifier,
        current_state: CurrentState,
        obs: MusesObservation,
        fm_sv: rf.StateVector,
        **kwargs: Any,
    ) -> rf.ForwardModel:
        if instrument_name != InstrumentIdentifier("AIRS"):
            return None
        if self.retrieval_config is None:
            raise RuntimeError("Call notify_update_target first")
        logger.debug("Creating OSS forward model using using AirsFmObjectCreator")
        obj_creator = AirsFmObjectCreator(
            current_state,
            self.retrieval_config,
            obs,
            fm_sv=fm_sv,
            **self.creator_kwargs,
        )
        fm = obj_creator.forward_model
        logger.info(f"Airs Forward model\n{fm}")
        return fm


class TesForwardModelHandle(ForwardModelHandle):
    def __init__(self, **creator_kwargs: Any) -> None:
        self.creator_kwargs = creator_kwargs
        self.retrieval_config: None | RetrievalConfiguration = None

    def notify_update_target(
        self, measurement_id: MeasurementId, retrieval_config: RetrievalConfiguration
    ) -> None:
        """Clear any caching associated with assuming the target being retrieved is fixed"""
        logger.debug(f"Call to {self.__class__.__name__}::notify_update")
        self.retrieval_config = retrieval_config

    def forward_model(
        self,
        instrument_name: InstrumentIdentifier,
        current_state: CurrentState,
        obs: MusesObservation,
        fm_sv: rf.StateVector,
        **kwargs: Any,
    ) -> rf.ForwardModel:
        if instrument_name != InstrumentIdentifier("TES"):
            return None
        if self.retrieval_config is None:
            raise RuntimeError("Call notify_update_target first")
        logger.debug("Creating OSS forward model using using TesFmObjectCreator")
        obj_creator = TesFmObjectCreator(
            current_state,
            self.retrieval_config,
            obs,
            fm_sv=fm_sv,
            **self.creator_kwargs,
        )
        fm = obj_creator.forward_model
        logger.info(f"Tes Forward model\n{fm}")
        return fm


ForwardModelHandleSet.add_default_handle(
    CrisForwardModelHandle(),
    priority_order=-1,
)
ForwardModelHandleSet.add_default_handle(
    AirsForwardModelHandle(),
    priority_order=-1,
)
ForwardModelHandleSet.add_default_handle(
    TesForwardModelHandle(),
    priority_order=-1,
)

__all__ = [
    "MusesOssFmObjectCreator",
    "CrisFmObjectCreator",
    "AirsFmObjectCreator",
    "TesFmObjectCreator",
    "CrisForwardModelHandle",
    "AirsForwardModelHandle",
    "TesForwardModelHandle",
]

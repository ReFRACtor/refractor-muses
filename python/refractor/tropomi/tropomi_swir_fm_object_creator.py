from __future__ import annotations
from functools import cached_property
from refractor.muses import (
    RefractorUip,
    ForwardModelHandle,
    CurrentState,
    InstrumentIdentifier,
)
import refractor.framework as rf  # type: ignore
from .tropomi_fm_object_creator import TropomiFmObjectCreator
from loguru import logger
import numpy as np
from pathlib import Path
from typing import Callable
import typing

if typing.TYPE_CHECKING:
    from refractor.muses import MeasurementId, MusesObservation, RefractorUip


class TropomiSwirFmObjectCreator(TropomiFmObjectCreator):
    """This is the variation for handling the SWIR channels. Note that
    this might get merged in with TropomiFmObjectCreator with some logic for
    picking the bands, but for now leave this separate.

    Also, at this point we aren't overly worried about having a fully integrated
    set of OSP control files. We hard code stuff in this class just to get everything
    working. Once we figure out *what* we want to do, we can then worry about fully
    integrating that in with the rest of the system.
    """

    def __init__(
        self,
        current_state: CurrentState,
        measurement_id: MeasurementId,
        observation: MusesObservation,
        absorption_gases: list[str] = ["H2O", "CO", "CH4", "HDO"],
        primary_absorber: str = "CO",
        use_raman: bool = False,
        use_oss: bool = False,
        oss_training_data: str | None = None,
        **kwargs,
    ):
        super().__init__(
            current_state,
            measurement_id,
            observation,
            absorption_gases=absorption_gases,
            primary_absorber=primary_absorber,
            use_raman=use_raman,
            **kwargs,
        )
        self.use_oss = use_oss
        self.oss_training_data = oss_training_data

    @cached_property
    def absorber(self) -> rf.Absorber:
        """Absorber to use. This just gives us a simple place to switch
        between absco and cross section."""
        return self.absorber_absco

    def absco_filename(self, gas: str, version: str = "latest") -> Path:
        # allow one to pass in "latest" or a version number like either "1.0" or "v1.0"
        if version == "latest":
            vpat = "v*"
        elif version.startswith("v"):
            vpat = version
        else:
            vpat = f"v{version}"

        # Assumes that in the top level of the ABSCO directory there are
        # subdirectories such as "v1.0_SWIR_CO" which contain our ABSCO files.
        absco_subdir_pattern = f"{vpat}_SWIR_{gas.upper()}"
        absco_subdirs = sorted(self.absco_base_path.glob(absco_subdir_pattern))
        if version == "latest" and len(absco_subdirs) == 0:
            raise RuntimeError(
                f'Found no ABSCO directories for gas "{gas}" matching {self.absco_base_path / absco_subdir_pattern}'
            )
        elif version == "latest":
            # Assumes that the latest version will be the last after sorting (e.g. v1.1
            # > v1.0). Should technically use a semantic version parser to ensure e.g.
            # v1.0.1 would be selected over v1.0.
            gas_subdir = absco_subdirs[-1]
            logger.info(f"Using ABSCO files from {gas_subdir} for {gas}")
        elif len(absco_subdirs) == 1:
            gas_subdir = absco_subdirs[0]
        else:
            raise RuntimeError(
                f"{len(absco_subdirs)} were found for {gas} {version} in {self.absco_base_path}"
            )

        gas_pattern = gas_subdir / f"nc_ABSCO/{gas.upper()}_*_v0.0_init.nc"
        return self.find_absco_pattern(str(gas_pattern), join_to_absco_base_path=False)

    @cached_property
    def spectrum_sampling(self) -> rf.SpectrumSampling:
        # For now, we just know that the spacing is 0.01. I think we can
        # probably read that somewhere, but skip for now.
        return rf.SpectrumSamplingFixedSpacing(
            rf.ArrayWithUnit(np.array([0.01]), "cm^-1")
        )

    @cached_property
    def underlying_forward_model(self):
        if self.use_oss:
            res = rf.director.OSSForwardModel(
                self.instrument,
                self.spec_win,
                self.radiative_transfer,
                self.spectrum_sampling,
                self.spectrum_effect,
                self.oss_training_data,
            )
            res.setup_grid()
        else:
            res = rf.StandardForwardModel(
                self.instrument,
                self.spec_win,
                self.radiative_transfer,
                self.spectrum_sampling,
                self.spectrum_effect,
            )
            res.setup_grid()
        return res


class TropomiSwirForwardModelHandle(ForwardModelHandle):
    def __init__(self, **creator_kwargs):
        self.creator_kwargs = creator_kwargs
        self.measurement_id = None

    def notify_update_target(self, measurement_id: MeasurementId):
        """Clear any caching associated with assuming the target being retrieved is fixed"""
        logger.debug(f"Call to {self.__class__.__name__}::notify_update")
        self.measurement_id = measurement_id

    def forward_model(
        self,
        instrument_name: InstrumentIdentifier,
        current_state: CurrentState,
        obs: MusesObservation,
        fm_sv: rf.StateVector,
        rf_uip_func: Callable[[InstrumentIdentifier | None], RefractorUip] | None,
        **kwargs,
    ) -> rf.ForwardModel:
        if instrument_name != InstrumentIdentifier("TROPOMI"):
            return None
        obj_creator = TropomiSwirFmObjectCreator(
            current_state,
            self.measurement_id,
            obs,
            rf_uip_func=rf_uip_func,
            fm_sv=fm_sv,
            **self.creator_kwargs,
        )
        fm = obj_creator.forward_model
        logger.info(f"Tropomi SWIR Forward model\n{fm}")
        return fm


__all__ = ["TropomiSwirFmObjectCreator", "TropomiSwirForwardModelHandle"]

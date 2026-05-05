from __future__ import annotations
import numpy as np
from .misc import AttrDictAdapter
from .creator_dict import CreatorDict
from .muses_strategy_context import MusesStrategyContextMixin
from .process_location_observable import ProcessLocationObservable
from .declarative_output import register_dataset, DeclarativeOutput
from .identifier import (
    ProcessLocation,
    RetrievalType,
    InstrumentIdentifier,
    StateElementIdentifier,
)
from .retrieval_output_file import RetrievalOutputFile
from loguru import logger
from pathlib import Path
from typing import Any
import typing

if typing.TYPE_CHECKING:
    from .retrieval_strategy_step import RetrievalStrategyStep
    from .retrieval_result import RetrievalResult
    from .current_state import CurrentState
    from .sounding_metadata import SoundingMetadata


# The variable_order can be used to specify the order variables appear
# in the file. This doesn't matter much, expect it makes it easier to
# compare against old expected results. No harm in matching the order
# the py-retrieve code used.
variable_order = [
    "RADIANCEFIT",
    "RADIANCEFITINITIAL",
    "RADIANCEOBSERVED",
    "NESR",
    "FREQUENCY",
    "RADIANCEFULLBAND",
    "FREQUENCYFULLBAND",
    "NESRFULLBAND",
    "SOUNDINGID",
    "LATITUDE",
    "LONGITUDE",
    "SURFACEALTITUDEMETERS",
    "RADIANCERESIDUALMEAN",
    "RADIANCERESIDUALRMS",
    "LAND",
    "QUALITY",
    "CLOUDOPTICALDEPTH",
    "CLOUDTOPPRESSURE",
    "SURFACETEMPERATURE",
    "SCANDIRECTION",
    "TIME",
    "EMIS",
    "EMISFREQ",
    "CLOUD",
    "CLOUDFREQ",
]


class RetrievalRadianceOutput(DeclarativeOutput, MusesStrategyContextMixin):
    def __init__(self, creator_dict: CreatorDict, **kwargs: Any) -> None:
        MusesStrategyContextMixin.__init__(self, creator_dict.strategy_context)
        DeclarativeOutput.__init__(self)
        self.creator_dict = creator_dict
        self.output = RetrievalOutputFile(variable_order)
        self.output.register_instances((self,))

    @property
    def observing_process_location(self) -> list[ProcessLocation]:
        return [ProcessLocation("retrieval step")]

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
        self._obs_rad: dict | None = None
        self.write()

    @property
    def results(self) -> RetrievalResult:
        if (
            hasattr(self.retrieval_strategy_step, "results")
            and self.retrieval_strategy_step.results is not None
        ):
            return self.retrieval_strategy_step.results
        else:
            raise RuntimeError("retrieval_strategy_step.results needs to not be None")

    @property
    def smeta(self) -> SoundingMetadata:
        return self.results.sounding_metadata

    @property
    def radiance_step(self) -> AttrDictAdapter:
        res = self.results.rstep
        assert res is not None
        return res

    def state_value(self, state_name: str) -> float:
        """Get the state value for the given state name"""
        return self.current_state.state_value(StateElementIdentifier(state_name))[0]

    def state_value_vec(self, state_name: str) -> np.ndarray:
        """Get the state value for the given state name"""
        return self.current_state.state_value(StateElementIdentifier(state_name))

    def state_sd_wavenumber(self, state_name: str) -> np.ndarray:
        """Get the spectral domain wavenumber in cm^-1 for state element"""
        t = self.current_state.state_spectral_domain_wavenumber(
            StateElementIdentifier(state_name)
        )
        if t is None:
            raise RuntimeError(
                f"{state_name} doesn't have state_spectral_domain_wavenumber"
            )
        return t

    @property
    def obs_rad(self) -> dict:
        # Not sure what exactly this is doing, we may want to track through
        # this and figure this out. I'm guessing the code could be simplified.
        if self._obs_rad is not None:
            return self._obs_rad
        radiance_full = self.results.radiance_full
        instruments = self.results.instruments
        assert radiance_full is not None
        self._obs_rad = radiance_full
        for inst in ("OMI", "TROPOMI"):
            if InstrumentIdentifier(inst) in self.radiance_step.instrumentNames:
                i = self.radiance_step.instrumentNames.index(InstrumentIdentifier(inst))
                istart = sum(self.radiance_step.instrumentSizes[:i])
                iend = istart + self.radiance_step.instrumentSizes[i]
                r = range(istart, iend)
                self._obs_rad = {
                    "instrumentNames": [inst],
                    "frequency": self.radiance_step.frequency[r],
                    "radiance": self.radiance_step.radiance[r],
                    "NESR": self.radiance_step.NESR[r],
                }
        if (
            len(self._obs_rad["instrumentNames"]) == 1
            or self._obs_rad["instrumentNames"][0]
            == self._obs_rad["instrumentNames"][1]
        ):
            num_trueFreq = self._obs_rad["frequency"]
            fullRadiance = self._obs_rad["radiance"]
            fullNESR = self._obs_rad["NESR"]
        else:
            instruIndex = self._obs_rad["instrumentNames"].index(instruments[0])
            if instruIndex == 0:
                r = range(0, self._obs_rad["instrumentSizes"][0])
            elif instruIndex == 1:
                r = range(
                    self._obs_rad["instrumentSizes"][0],
                    self._obs_rad["frequency"].shape[0],
                )
            num_trueFreq = self._obs_rad["frequency"][r]
            fullRadiance = self._obs_rad["radiance"][r]
            fullNESR = self._obs_rad["NESR"][r]
        self._obs_rad = {
            "frequency": num_trueFreq,
            "radiance": fullRadiance,
            "NESR": fullNESR,
        }
        return self._obs_rad

    @property
    def out_fname(self) -> Path:
        return (
            self.retrieval_config["output_directory"]
            / "Products"
            / f"Products_Radiance-{self.species_tag}{self.special_tag}.nc"
        )

    @property
    def species_tag(self) -> str:
        res = self.step_name
        res = res.rstrip(", ")
        if "EMIS" in res and res.index("EMIS") > 0:
            res = res.replace("EMIS", "")
        if res.endswith(",_OMI"):
            res = res.replace(",_OMI", "_OMI")  #  Change "H2O,O3,_OMI" to "H2O,O3_OMI"
        res = res.rstrip(", ")
        return res

    @property
    def special_tag(self) -> str:
        if self.retrieval_type != RetrievalType("default"):
            return f"-{self.retrieval_type.lower()}"
        return ""

    @register_dataset("/RADIANCEFULLBAND")
    def radiance_full_band(self) -> np.ndarray:
        return self.obs_rad["radiance"]

    @register_dataset("/FREQUENCYFULLBAND")
    def frequency_full_band(self) -> np.ndarray:
        return self.obs_rad["frequency"]

    @register_dataset("/NESRFULLBAND")
    def nesr_full_band(self) -> np.ndarray:
        return self.obs_rad["NESR"]

    @register_dataset("/RADIANCEFIT")
    def radiance_fit(self) -> np.ndarray:
        return self.results.radiance[0, :].astype(np.float32)

    @register_dataset("/RADIANCEFITINITIAL")
    def radiance_fit_initial(self) -> np.ndarray:
        return self.results.radiance_initial[0, :].astype(np.float32)

    @register_dataset("/RADIANCEOBSERVED")
    def radiance_observed(self) -> np.ndarray:
        return self.radiance_step.radiance.astype(np.float32)

    @register_dataset("/NESR")
    def nesr(self) -> np.ndarray:
        return self.radiance_step.NESR.astype(np.float32)

    @register_dataset("/FREQUENCY")
    def frequency(self) -> np.ndarray:
        return self.results.frequency.astype(np.float32)

    @register_dataset("/EMIS")
    def emis(self) -> np.ndarray:
        return self.state_value_vec("EMIS").astype(np.float32)

    @register_dataset("/EMISFREQ")
    def emis_freq(self) -> np.ndarray:
        return self.state_sd_wavenumber("EMIS").astype(np.float32)

    @register_dataset("/CLOUD")
    def cloud(self) -> np.ndarray:
        return self.state_value_vec("CLOUDEXT").astype(np.float32)

    @register_dataset("/CLOUDFREQ")
    def cloud_freq(self) -> np.ndarray:
        return self.state_sd_wavenumber("CLOUDEXT").astype(np.float32)

    @register_dataset("/LATITUDE")
    def latitude(self) -> np.float32:
        return np.float32(self.smeta.latitude.convert("deg").value)

    @register_dataset("/LONGITUDE")
    def longitude(self) -> np.float32:
        return np.float32(self.smeta.longitude.convert("deg").value)

    @register_dataset("/SURFACEALTITUDEMETERS")
    def surface_altitude(self) -> np.float32:
        return np.float32(self.smeta.surface_altitude.convert("m").value)

    @register_dataset("/RADIANCERESIDUALMEAN")
    def radiance_residual_mean(self) -> np.float32:
        return np.float32(self.results.radianceResidualMean[0])

    @register_dataset("/RADIANCERESIDUALRMS")
    def radiance_residual_rms(self) -> np.float32:
        return np.float32(self.results.radianceResidualRMS[0])

    @register_dataset("/CLOUDOPTICALDEPTH")
    def cloud_optical_depth(self) -> np.float32:
        return np.float32(self.results.cloudODAve)

    @register_dataset("/CLOUDTOPPRESSURE")
    def cloud_top_pressure(self) -> np.float32:
        return np.float32(self.state_value("PCLOUD"))

    @register_dataset("/SURFACETEMPERATURE")
    def surface_temperature(self) -> np.float32:
        return np.float32(self.state_value("TSUR"))

    @register_dataset("/LAND")
    def land(self) -> np.int16:
        return np.int16(1 if self.smeta.is_land else 0)

    @register_dataset("/QUALITY")
    def quality(self) -> np.int16:
        return np.int16(self.results.masterQuality)

    @register_dataset("/SCANDIRECTION")
    def scan_direction(self) -> np.int16:
        # This really is just a hard coded value
        return np.int16(0)

    @register_dataset("/TIME")
    def time(self) -> float:
        return np.float64(self.smeta.tai_time)

    @register_dataset("/SOUNDINGID")
    def sounding_id(self) -> np.ndarray:
        # This really seems kind of hokey, but rather than just setting
        # a string this gets written as an array of bytes. Match this old
        # behavior, since things down stream probably depend on this
        return np.frombuffer(
            self.smeta.sounding_id.encode("utf8"), dtype=np.dtype("b")
        )[np.newaxis, :]

    def write(self) -> None:
        self.output.write(self.out_fname)


ProcessLocationObservable.register_default_observer(RetrievalRadianceOutput)

__all__ = [
    "RetrievalRadianceOutput",
]

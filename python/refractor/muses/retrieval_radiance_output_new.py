from __future__ import annotations
import numpy as np
from .creator_dict import CreatorDict
from .muses_strategy_context import MusesStrategyContextMixin
from .declarative_output import register_dataset, DeclarativeOutput
from .identifier import ProcessLocation, RetrievalType, InstrumentIdentifier
from .retrieval_output_file import RetrievalOutputFile
from loguru import logger
from pathlib import Path
from typing import Any
import typing

if typing.TYPE_CHECKING:
    from .retrieval_strategy_step import RetrievalStrategyStep
    from .current_state import CurrentState


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


class RetrievalRadianceOutputNew(DeclarativeOutput, MusesStrategyContextMixin):
    """New version of RetrievalRadianceOutput, that uses the DeclarativeOutput interface.
    We will likely rename the old RetrievalRadianceOutput to RetrievalRadianceOutputOld,
    and rename this to RetrievalRadianceOutput when we have this all in place. But for
    now leave the old one in place and have this as the "new" version."""

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
    def obs_rad(self) -> dict:
        # Not sure what exactly this is doing, we may want to track through
        # this and figure this out. I'm guessing the code could be simplified.
        if self._obs_rad is not None:
            return self._obs_rad
        if (
            hasattr(self.retrieval_strategy_step, "results")
            and self.retrieval_strategy_step.results is not None
        ):
            results = self.retrieval_strategy_step.results
        else:
            raise RuntimeError("retrieval_strategy_step.results needs to not be None")
        radiance_full = results.radiance_full
        radiance_step = results.rstep
        instruments = results.instruments
        assert radiance_full is not None
        assert radiance_step is not None
        self._obs_rad = radiance_full
        for inst in ("OMI", "TROPOMI"):
            if InstrumentIdentifier(inst) in radiance_step.instrumentNames:
                i = radiance_step.instrumentNames.index(InstrumentIdentifier(inst))
                istart = sum(radiance_step.instrumentSizes[:i])
                iend = istart + radiance_step.instrumentSizes[i]
                r = range(istart, iend)
                self._obs_rad = {
                    "instrumentNames": [inst],
                    "frequency": radiance_step.frequency[r],
                    "radiance": radiance_step.radiance[r],
                    "NESR": radiance_step.NESR[r],
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
        self.output.write(self.out_fname)


__all__ = [
    "RetrievalRadianceOutputNew",
]

from __future__ import annotations
from glob import glob
from loguru import logger
import os
from .retrieval_output import RetrievalOutput
from .identifier import InstrumentIdentifier, ProcessLocation
from .refractor_uip import AttrDictAdapter
from pathlib import Path
import numpy as np
import typing
from typing import Any, Callable

if typing.TYPE_CHECKING:
    from .retrieval_strategy import RetrievalStrategy
    from .retrieval_strategy_step import RetrievalStrategyStep


def _new_from_init(cls, *args):  # type: ignore
    """For use with pickle, covers common case where we just store the
    arguments needed to create an object."""
    inst = cls.__new__(cls)
    inst.__init__(*args)
    return inst


class RetrievalRadianceOutput(RetrievalOutput):
    """Observer of RetrievalStrategy, outputs the Products_Radiance files."""

    def __init__(self) -> None:
        self.myobsrad: None | dict = None

    def __reduce__(self) -> tuple[Callable, tuple[Any]]:
        return (_new_from_init, (self.__class__,))

    def notify_update(
        self,
        retrieval_strategy: RetrievalStrategy,
        location: ProcessLocation,
        retrieval_strategy_step: RetrievalStrategyStep | None = None,
        **kwargs: Any,
    ) -> None:
        self.retrieval_strategy = retrieval_strategy
        self.retrieval_strategy_step = retrieval_strategy_step
        if location != ProcessLocation("retrieval step"):
            return
        logger.debug(f"Call to {self.__class__.__name__}::notify_update")
        if len(glob(f"{self.out_fname}*")) == 0:
            # First argument isn't actually used in write_products_one_jacobian.
            # It is special_name, which doesn't actually apply to the jacobian file.
            os.makedirs(os.path.dirname(self.out_fname), exist_ok=True)
            # This is an odd interface, but it is how radiance gets the input data.
            # We should perhaps just rewrite write_products_one_radiance, but for
            # now just conform with what it wants.

            # Note, I think this logic is actually wrong. If a previous step had
            # OMI or TROPOMI, it looks like this get left in rather than using
            # CRIS or AIRS radiance. But leave this for now, so we duplicate what
            # was done previously
            if self.myobsrad is None:
                self.myobsrad = self.radiance_full
            for inst in ("OMI", "TROPOMI"):
                if InstrumentIdentifier(inst) in self.radiance_step.instrumentNames:
                    i = self.radiance_step.instrumentNames.index(
                        InstrumentIdentifier(inst)
                    )
                    istart = sum(self.radiance_step.instrumentSizes[:i])
                    iend = istart + self.radiance_step.instrumentSizes[i]
                    r = range(istart, iend)
                    self.myobsrad = {
                        "instrumentNames": [inst],
                        "frequency": self.radiance_step.frequency[r],
                        "radiance": self.radiance_step.radiance[r],
                        "NESR": self.radiance_step.NESR[r],
                    }

            self.write_radiance()
        else:
            logger.info(f"Found a radiance product file: {self.out_fname}")

    @property
    def out_fname(self) -> Path:
        return (
            self.output_directory
            / "Products"
            / f"Products_Radiance-{self.species_tag}{self.special_tag}.nc"
        )

    def write_radiance(self) -> None:
        if self.myobsrad is None:
            raise RuntimeError("self.myobsrad needs to not be None")
        if (
            len(self.myobsrad["instrumentNames"]) == 1
            or self.myobsrad["instrumentNames"][0]
            == self.myobsrad["instrumentNames"][1]
        ):
            num_trueFreq = self.myobsrad["frequency"]
            fullRadiance = self.myobsrad["radiance"]
            fullNESR = self.myobsrad["NESR"]
        else:
            instruIndex = self.myobsrad["instrumentNames"].index(self.instruments[0])
            if instruIndex == 0:
                r = range(0, self.myobsrad["instrumentSizes"][0])
            elif instruIndex == 1:
                r = range(
                    self.myobsrad["instrumentSizes"][0],
                    self.myobsrad["frequency"].shape[0],
                )
            num_trueFreq = self.myobsrad["frequency"][r]
            fullRadiance = self.myobsrad["radiance"][r]
            fullNESR = self.myobsrad["NESR"][r]

        my_datad = dict.fromkeys(
            [
                "radianceFit",
                "radianceFitInitial",
                "radianceObserved",
                "nesr",
                "frequency",
                "radianceFullBand",
                "frequencyFullBand",
                "nesrFullBand",
                "soundingID",
                "latitude",
                "longitude",
                "surfaceAltitudeMeters",
                "radianceResidualMean",
                "radianceResidualRMS",
                "land",
                "quality",
                "cloudOpticalDepth",
                "cloudTopPressure",
                "surfaceTemperature",
                "scanDirection",
                "time",
                "emis",
                "emisFreq",
                "cloud",
                "cloudFreq",
            ]
        )

        my_data = AttrDictAdapter(my_datad)

        my_data.radianceFullBand = fullRadiance
        my_data.frequencyFullBand = num_trueFreq
        my_data.nesrFullBand = fullNESR

        my_data.radianceFit = self.results.radiance[0, :].astype(np.float32)
        my_data.radianceFitInitial = self.results.radiance_initial[0, :].astype(
            np.float32
        )
        my_data.radianceObserved = self.radiance_step.radiance.astype(np.float32)
        my_data.nesr = self.radiance_step.NESR.astype(np.float32)
        my_data.frequency = self.results.frequency.astype(np.float32)

        smeta = self.sounding_metadata
        my_data.time = np.float64(smeta.tai_time)
        my_data.soundingID = np.frombuffer(
            smeta.sounding_id.encode("utf8"), dtype=np.dtype("b")
        )[np.newaxis, :]
        my_data.latitude = np.float32(smeta.latitude.convert("deg").value)
        my_data.longitude = np.float32(smeta.longitude.convert("deg").value)
        my_data.surfaceAltitudeMeters = np.float32(
            smeta.surface_altitude.convert("m").value
        )
        my_data.land = np.int16(1 if smeta.is_land else 0)
        my_data.scanDirection = np.int16(0)

        my_data.emis = self.state_value_vec("EMIS").astype(np.float32)
        my_data.emisFreq = self.state_sd_wavelength("EMIS").astype(np.float32)

        my_data.cloud = self.state_value_vec("CLOUDEXT").astype(np.float32)
        my_data.cloudFreq = self.state_sd_wavelength("CLOUDEXT").astype(np.float32)

        my_data.quality = np.int16(self.results.masterQuality)
        my_data.radianceResidualMean = np.float32(self.results.radianceResidualMean[0])
        my_data.radianceResidualRMS = np.float32(self.results.radianceResidualRMS[0])
        my_data.cloudTopPressure = np.float32(self.state_value("PCLOUD"))
        my_data.cloudOpticalDepth = np.float32(self.results.cloudODAve)
        my_data.surfaceTemperature = np.float32(self.state_value("TSUR"))
        # Write out, use units as dummy: "()"
        self.cdf_write(
            my_data.as_dict(my_data),
            str(self.out_fname),
            [
                {"UNITS": "()"},
            ]
            * len(my_data),
        )


__all__ = [
    "RetrievalRadianceOutput",
]

from __future__ import annotations
from glob import glob
from loguru import logger
from .mpy import mpy_specie_type, mpy_cdf_write
import os
from .retrieval_output import RetrievalOutput
from .identifier import ProcessLocation
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


class RetrievalJacobianOutput(RetrievalOutput):
    """Observer of RetrievalStrategy, outputs the Products_Jacobian files."""

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
            os.makedirs(os.path.dirname(self.out_fname), exist_ok=True)
            self.write_jacobian()
        else:
            logger.info(f"Found a jacobian product file: {self.out_fname}")

    @property
    def out_fname(self) -> Path:
        return (
            self.output_directory
            / "Products"
            / f"Products_Jacobian-{self.species_tag}{self.special_tag}.nc"
        )

    def write_jacobian(self) -> None:
        # this section is to make all pressure grids have a standard size,
        # like 65 levels

        # Python idiom for getting a unique list
        species = list(dict.fromkeys(self.species_list_fm))

        jacobianAll = self.results.jacobian[0, :, :]
        nf = jacobianAll.shape[1]

        mypressure = []
        myspecies = []
        myjacobian = []
        for spc in species:
            ind = [s == spc for s in self.species_list_fm]
            nn = np.count_nonzero(ind)
            species_type = mpy_specie_type(spc)

            nlevel = 65
            if species_type == "ATMOSPHERIC":
                my_list = [
                    "none",
                ] * nlevel
                my_list[nlevel - nn :] = [
                    spc,
                ] * nn
                pressure = np.zeros(shape=(nlevel), dtype=np.float32)
                jacobian = np.zeros(shape=(nlevel, nf), dtype=np.float64)
                pressure[nlevel - nn :] = self.pressure_list_fm[ind]
                jacobian[nlevel - nn :, :] = jacobianAll[ind, :]
            else:
                my_list = [
                    spc,
                ] * nn
                pressure = self.pressure_list_fm[ind]
                jacobian = jacobianAll[ind, :]
            mypressure.append(pressure)
            myspecies.append(my_list)
            myjacobian.append(jacobian)
        # end for ii in range(0, len(species)):
        my_datad = {
            "jacobian": np.transpose(
                np.concatenate(myjacobian, axis=0)
            ),  # We transpose to match IDL shape.
            "frequency": None,
            "species": ",".join(np.concatenate(myspecies)),
            "pressure": np.concatenate(mypressure),
            "soundingID": None,
            "latitude": None,
            "longitude": None,
            "surfaceAltitudeMeters": None,
            "radianceResidualMean": None,
            "radianceResidualRMS": None,
            "land": None,
            "quality": None,
            "cloudOpticalDepth": None,
            "cloudTopPressure": None,
            "surfaceTemperature": None,
        }

        my_data = AttrDictAdapter(my_datad)

        my_data.frequency = self.results.frequency.astype(np.float32)

        smeta = self.sounding_metadata
        my_data.soundingID = smeta.sounding_id
        my_data.latitude = np.float32(smeta.latitude.convert("deg").value)
        my_data.longitude = np.float32(smeta.longitude.convert("deg").value)
        my_data.surfaceAltitudeMeters = np.float32(
            smeta.surface_altitude.convert("m").value
        )
        my_data.land = np.int16(1 if smeta.is_land else 0)

        my_data.quality = np.int16(self.results.masterQuality)
        my_data.radianceResidualMean = np.float32(self.results.radianceResidualMean[0])
        my_data.radianceResidualRMS = np.float32(self.results.radianceResidualRMS[0])
        my_data.cloudTopPressure = np.float32(self.state_value("PCLOUD"))
        my_data.cloudOpticalDepth = np.float32(self.results.cloudODAve)
        my_data.surfaceTemperature = np.float32(self.state_value("TSUR"))

        # Write out, use units as dummy: "()"
        my_data2 = my_data.as_dict(my_data)
        mpy_cdf_write(
            my_data2,
            str(self.out_fname),
            [
                {"UNITS": "()"},
            ]
            * len(my_data2),
        )


__all__ = [
    "RetrievalJacobianOutput",
]

from __future__ import annotations
from loguru import logger
import os
from .retrieval_output import RetrievalOutput
from .identifier import ProcessLocation, InstrumentIdentifier, StateElementIdentifier
from .misc import AttrDictAdapter
import numpy as np
import math
import typing
from typing import Any, Callable

if typing.TYPE_CHECKING:
    from .retrieval_strategy import RetrievalStrategy
    from .retrieval_strategy_step import RetrievalStrategyStep
    from .misc import ResultIrk
    from .current_state import PropagatedQA


def _new_from_init(cls, *args):  # type: ignore
    """For use with pickle, covers common case where we just store the
    arguments needed to create an object."""
    inst = cls.__new__(cls)
    inst.__init__(*args)
    return inst


class RetrievalIrkOutput(RetrievalOutput):
    """Observer of RetrievalStrategy, outputs the Products_IRK files."""

    def __reduce__(self) -> tuple[Callable, tuple[Any]]:
        return (_new_from_init, (self.__class__,))

    @property
    def propagated_qa(self) -> PropagatedQA:
        return self.current_state.propagated_qa

    @property
    def results_irk(self) -> ResultIrk | None:
        if self.retrieval_strategy_step is not None and hasattr(
            self.retrieval_strategy_step, "results_irk"
        ):
            return self.retrieval_strategy_step.results_irk
        return None

    def notify_update(
        self,
        retrieval_strategy: RetrievalStrategy,
        location: ProcessLocation,
        retrieval_strategy_step: RetrievalStrategyStep | None = None,
        **kwargs: Any,
    ) -> None:
        self.retrieval_strategy = retrieval_strategy
        self.retrieval_strategy_step = retrieval_strategy_step
        if location != ProcessLocation("IRK step"):
            return
        logger.debug(f"Call to {self.__class__.__name__}::notify_update")
        self.out_fname = self.output_directory / "Products" / "Products_IRK.nc"
        os.makedirs(os.path.dirname(self.out_fname), exist_ok=True)
        self.write_irk()

    def write_irk(self) -> None:
        if self.results_irk is None:
            return

        # Copy of write_products_irk_one, so we can try cleaning this up a bit
        smeta = self.current_state.sounding_metadata

        nobs = 1
        num_points = 67
        nfreqBandFM = len(self.results_irk.fluxSegments)
        nfreqBand = len(self.results_irk.TSUR["irfk_segs"][0, :])
        nfreqEmis = len(self.results_irk.EMIS["irfk"])
        nfreqCloud = len(self.results_irk.CLOUDOD["irfk"])

        irk_datad = {
            "fmBandFlux": np.zeros(shape=(nobs), dtype=np.float32) - 999,
            "l1BBandFlux": np.zeros(shape=(nobs), dtype=np.float32) - 999,
            "fmFluxSegs": np.zeros(shape=(nfreqBandFM, nobs), dtype=np.float32) - 999,
            "fmFluxSegsCenterFreq": np.zeros(
                shape=(nfreqBandFM, nobs), dtype=np.float32
            )
            - 999,
            "l1BFluxSegs": np.zeros(shape=(nfreqBandFM, nobs), dtype=np.float32) - 999,
            "o3LIRK": np.zeros(shape=(num_points, nobs), dtype=np.float32) - 999,
            "o3IRKSegs": np.zeros(shape=(num_points, nfreqBand, nobs), dtype=np.float32)
            - 999,
            "IRKSegsCenterFreq": np.zeros(shape=(nfreqBand, nobs), dtype=np.float32)
            - 999,
            "h2oLIRK": np.zeros(shape=(num_points, nobs), dtype=np.float32) - 999,
            "tatmIRK": np.zeros(shape=(num_points, nobs), dtype=np.float32) - 999,
            "tsurIRK": np.zeros(shape=(nobs), dtype=np.float32) - 999,
            "CloudEffectiveOpticalDepthLIRK": np.zeros(
                shape=(nfreqCloud, nobs), dtype=np.float32
            ),
            "CloudTopPressureIRK": np.zeros(shape=(nobs), dtype=np.float32) - 999,
            "emisIRK": np.zeros(shape=(nfreqEmis, nobs), dtype=np.float32) - 999,
            "h2o": np.zeros(shape=(num_points, nobs), dtype=np.float32),
            "co2": np.zeros(shape=(num_points, nobs), dtype=np.float32),
            "n2o": np.zeros(shape=(num_points, nobs), dtype=np.float32),
            "o3": np.zeros(shape=(num_points, nobs), dtype=np.float32),
            "tatm": np.zeros(shape=(num_points, nobs), dtype=np.float32),
            "cloudod": np.zeros(shape=(nfreqCloud, nobs), dtype=np.float32),
            "emis": np.zeros(shape=(nfreqEmis, nobs), dtype=np.float32),
            "tsur": np.zeros(shape=(nobs), dtype=np.float32),
            "tatm_QA": np.zeros(shape=(nobs), dtype=np.int16),
            "o3_QA": np.zeros(shape=(nobs), dtype=np.int16),
            "utctime": ["  " for i in range(nobs)],
            "daynightFlag": np.zeros(shape=(nobs), dtype=np.int16) - 999,
            "dominantSurfaceType": ["  " for i in range(nobs)],
            "LATITUDE": np.zeros(shape=(nobs), dtype=np.float64) - 999,
            "LONGITUDE": np.zeros(shape=(nobs), dtype=np.float64) - 999,
            "TIME": np.zeros(shape=(nobs), dtype=np.int32) - 999,
            "BoresightNadirAngle": np.zeros(shape=(nobs), dtype=np.float32) - 999,
            "surfaceTypeFootprint": np.zeros(shape=(nobs), dtype=np.int32) - 999,
            "soundingID": ["  " for i in range(nobs)],
            "omi_sza_uv2": np.zeros(shape=(nobs), dtype=np.float32),
            "omi_raz_uv2": np.zeros(shape=(nobs), dtype=np.float32),
            "omi_vza_uv2": np.zeros(shape=(nobs), dtype=np.float32),
            "omi_sca_uv2": np.zeros(shape=(nobs), dtype=np.float32),
        }
        irk_data = AttrDictAdapter(irk_datad)

        nn = num_points - len(self.results_irk.H2O["vmr"])
        irk_data.fmBandFlux[:] = self.results_irk.flux
        irk_data.l1BBandFlux[:] = self.results_irk.flux_l1b
        irk_data.fmFluxSegs[:, 0] = self.results_irk.fluxSegments[:]
        irk_data.fmFluxSegsCenterFreq[:, 0] = self.results_irk.freqSegments[:]
        irk_data.l1BFluxSegs[:, 0] = self.results_irk.fluxSegments_l1b[:]
        irk_data.o3LIRK[nn:, 0] = self.results_irk.O3["lirfk"][:]
        irk_data.o3IRKSegs[nn:, : self.results_irk.O3["irfk_segs"].shape[1], 0] = (
            self.results_irk.O3["irfk_segs"]
        )
        irk_data.IRKSegsCenterFreq[:, 0] = self.results_irk.freqSegments_irk[:]
        irk_data.h2oLIRK[nn:, 0] = self.results_irk.H2O["lirfk"][:]
        irk_data.tatmIRK[nn:, 0] = self.results_irk.TATM["irfk"][:]
        irk_data.tsurIRK[:] = self.results_irk.TSUR["irfk"][:]
        irk_data.CloudEffectiveOpticalDepthLIRK[:, 0] = self.results_irk.CLOUDOD[
            "lirfk"
        ][:]
        irk_data.CloudTopPressureIRK[:] = self.results_irk.PCLOUD["irfk"][:]
        irk_data.emisIRK[:, 0] = self.results_irk.EMIS["irfk"][:]
        irk_data.h2o[nn:, 0] = self.results_irk.H2O["vmr"][:]
        irk_data.co2[nn:, 0] = self.state_value_vec("CO2") * 1e6
        irk_data.n2o[nn:, 0] = self.state_value_vec("N2O") * 1e6
        irk_data.o3[nn:, 0] = self.results_irk.O3["vmr"][:]
        irk_data.tatm[nn:, 0] = self.results_irk.TATM["vmr"][:]
        irk_data.cloudod[:, 0] = self.results_irk.CLOUDOD["vmr"][:]
        irk_data.emis[:, 0] = self.results_irk.EMIS["vmr"][:]
        irk_data.tsur[:] = self.results_irk.TSUR["vmr"][:]
        irk_data.tatm_QA[:] = self.propagated_qa.tatm_qa
        irk_data.o3_QA[:] = self.propagated_qa.o3_qa
        irk_data.utctime = smeta.utc_time
        if InstrumentIdentifier("OMI") in self.current_strategy_step.instrument_name:
            obs = self.observation("OMI")
            blist = [str(i[0]) for i in obs.filter_data]
            sza = obs.solar_zenith
            raz = obs.relative_azimuth
            vza = obs.observation_zenith
            sca = obs.scattering_angle
            if ("UV2") in blist:
                i = blist.index("UV2")
                irk_data.omi_sza_uv2 = np.float32(sza[i])
                irk_data.omi_raz_uv2 = np.float32(raz[i])
                irk_data.omi_vza_uv2 = np.float32(vza[i])
                irk_data.omi_sca_uv2 = np.float32(sca[i])
            else:
                irk_data.omi_sza_uv2 = np.float32(-999.0)
                irk_data.omi_raz_uv2 = np.float32(-999.0)
                irk_data.omi_vza_uv2 = np.float32(-999.0)
                irk_data.omi_sca_uv2 = np.float32(-999.0)
        else:
            irk_data.omi_sza_uv2 = np.float32(-999.0)
            irk_data.omi_raz_uv2 = np.float32(-999.0)
            irk_data.omi_vza_uv2 = np.float32(-999.0)
            irk_data.omi_sca_uv2 = np.float32(-999.0)

        if (
            StateElementIdentifier("OMI")
            in self.current_state.retrieval_state_element_id
        ):
            irk_data.BoresightNadirAngle[:] = np.float32(vza[blist.index("UV1")])
        else:
            # convert to degrees
            irk_data.BoresightNadirAngle[:] = self.state_value("PTGANG") * 180 / math.pi

        irk_data.soundingID = smeta.sounding_id
        mid = self.retrieval_strategy.measurement_id
        if "AIRS_ATrack_Index" in mid:
            irk_data.airs_granule = np.int16(mid["AIRS_Granule"])
            irk_data.airs_atrack_index = np.int16(mid["AIRS_ATrack_Index"])
            irk_data.airs_xtrack_index = np.int16(mid["AIRS_XTrack_Index"])

        if "tes_run" in mid:
            # Not sure if these attributes below exist in mid
            irk_data.tes_run = mid["tes_run"]
            irk_data.tes_sequence = mid["tes_sequence"]
            irk_data.tes_scan = mid["tes_scan"]

        if "omi_xtrack_index" in mid:
            # Not sure if these attributes below exist in mid
            irk_data.omi_atrack_index = mid["omi_atrack_index"]
            irk_data.omi_xtrack_index = mid["omi_xtrack_index"]

        irk_data.dominantSurfaceType = smeta.surface_type
        irk_data.surfaceTypeFootprint = np.int32(2) if smeta.is_ocean else np.int32(3)
        if (
            smeta.is_ocean
            and np.amin(np.abs(self.current_state.height().convert("km").value)) > 0.1
        ):
            irk_data.surfaceTypeFootprint = np.int32(1)

        irk_data.LATITUDE = smeta.latitude.value
        irk_data.LONGITUDE = smeta.longitude.value

        if smeta.local_hour >= 8 and smeta.local_hour <= 22:  # day
            irk_data.daynightFlag[0] = 1
        if smeta.local_hour <= 5 or smeta.local_hour >= 22:  # night
            irk_data.daynightFlag[0] = 0

        struct_units: list[dict[str, str | float]] = [
            {"UNITS": "()"},
        ] * len(irk_data.__dict__)
        logger.info(f"Writing: {self.out_fname}")
        self.cdf_write(irk_data.__dict__, self.out_fname, struct_units)


__all__ = [
    "RetrievalIrkOutput",
]

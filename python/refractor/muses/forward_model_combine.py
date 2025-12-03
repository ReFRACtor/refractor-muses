from __future__ import annotations
import refractor.framework as rf  # type: ignore
import numpy as np
import typing

if typing.TYPE_CHECKING:
    from .muses_observation import MusesObservation


class ForwardModelCombine(rf.ForwardModel):
    """This is a simple class that combines 1 or more forward models. We just
    pretend like these are separate spectral channels, the interest here is
    in being able to call radiance_all."""

    def __init__(
        self,
        fm_list: list[rf.ForwardModel],
        obs_list: list[MusesObservation],
        fm_sv: rf.StateVector | None = None,
    ) -> None:
        """Create a combination of the supplied fm_list and obs_list. You can call pass in
        the StateVector, just so it is available"""
        self.fm_list = fm_list
        self.obs_list = obs_list
        self.fm_sv = fm_sv
        super().__init__()

    def obs_radiance_all(self) -> np.ndarray:
        return np.concatenate(
            [obs.radiance_all().spectral_range.data for obs in self.obs_list]
        )

    def setup_grid(self) -> None:
        for fm in self.fm_list:
            fm.setup_grid()

    def _v_num_channels(self) -> int:
        return len(self.fm_list)

    def spectral_domain(self, sensor_index: int) -> rf.SpectralDomain:
        return self.fm_list[sensor_index].spectral_domain_all()

    def radiance(self, sensor_index: int, skip_jacobian: bool = False) -> rf.Spectrum:
        return self.fm_list[sensor_index].radiance_all(skip_jacobian)

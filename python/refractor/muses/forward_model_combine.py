from __future__ import annotations
import refractor.framework as rf  # type: ignore
import numpy as np
import typing

if typing.TYPE_CHECKING:
    from .muses_observation import MusesObservation


class ObservationCombine(rf.StackedRadianceMixin):
    def __init__(self, obs_list: list[MusesObservation]) -> None:
        self.obs_list = obs_list
        super().__init__()

    def _v_num_channels(self) -> int:
        return len(self.obs_list)

    def spectral_domain(self, sensor_index: int) -> rf.SpectralDomain:
        return self.obs_list[sensor_index].spectral_domain_all()

    def radiance(self, sensor_index: int, skip_jacobian: bool = False) -> rf.Spectrum:
        return self.obs_list[sensor_index].radiance_all()


class ForwardModelCombine(rf.ForwardModel):
    """This is a simple class that combines 1 or more forward models. We just
    pretend like these are separate spectral channels, the interest here is
    in being able to call radiance_all or model_measure_diff_jacobian which uses
    radiance_all"""

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

    def obs_radiance_all(self) -> rf.Spectrum:
        return ObservationCombine(self.obs_list).radiance_all()

    def model_measure_diff_jacobian(self) -> np.ndarray:
        t = self.radiance_all().spectral_range.data_ad.jacobian
        t2 = self.obs_radiance_all().spectral_range.data_ad.jacobian
        if not np.all(np.isfinite(t)):
            tsub1 = self.radiance(0).spectral_range.data_ad.jacobian
            tsub2 = self.radiance(1).spectral_range.data_ad.jacobian
            if not np.all(np.isfinite(tsub1)):
                raise RuntimeError("jacobian not finite")
            if not np.all(np.isfinite(tsub2)):
                raise RuntimeError("jacobian not finite")
            raise RuntimeError("jacobian not finite")
        if not np.all(np.isfinite(t2)):
            raise RuntimeError("jacobian not finite")
        if t2.shape[0] == 0:
            return t
        if t.shape[0] == 0:
            return -t2
        return t - t2

    def setup_grid(self) -> None:
        for fm in self.fm_list:
            fm.setup_grid()

    def _v_num_channels(self) -> int:
        return len(self.fm_list)

    def spectral_domain(self, sensor_index: int) -> rf.SpectralDomain:
        return self.fm_list[sensor_index].spectral_domain_all()

    def radiance(self, sensor_index: int, skip_jacobian: bool = False) -> rf.Spectrum:
        res = self.fm_list[sensor_index].radiance_all(skip_jacobian)
        if not np.all(np.isfinite(res.spectral_range.data_ad.jacobian)):
            raise RuntimeError("jacobian not finite")
        return res

from __future__ import annotations
import refractor.framework as rf  # type: ignore
import numpy as np


class CloudExtState(rf.GenericStateImpBase):
    """We don't have a CloudExt class at the C++ level. We may add that, but
    for now just add a python class to handle this."""

    def __init__(
        self,
        cloud_extv: np.ndarray,
        cloud_ext_sd: rf.SpectralDomain,
        update_arr: np.ndarray | None,
        mp: rf.StateMapping = rf.StateMappingLinear,
    ):
        super().__init__()
        self.init(cloud_extv, mp)
        # We want to get the logic used in determining update_arr into this class,
        # but for now leverage off what we get passed in.
        self.update_arr = update_arr
        self._cloud_ext_sd = cloud_ext_sd
        if cloud_extv.shape != cloud_ext_sd.data.shape:
            raise RuntimeError(
                "cloudext and cloudext spectral domain need to be the same size"
            )

    @property
    def cloud_ext(self) -> rf.ArrayAd_double_1:
        return self.mapped_state

    @property
    def cloud_ext_spectral_domain(self) -> rf.SpectralDomain:
        return self._cloud_ext_sd

    def desc(self) -> str:
        return "Cloud_ExtState"

    def clone(self) -> rf.GenericState:
        return CloudExtState(
            self.cloud_ext, self.cloud_ext_spectral_domain, self.state_mapping
        )

    def sub_state_identifier(self) -> str:
        return "cloud_ext_state"

    def state_vector_name_i(self, i: int) -> str:
        mname = self.state_mapping.name
        if mname != "linear":
            return f"{mname} CloudExt Freq {i + 1}"
        return f"CloudExt Freq {i + 1}"


__all__ = [
    "CloudExtState",
]

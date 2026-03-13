from __future__ import annotations
import refractor.framework as rf  # type: ignore
import numpy as np


class EmisState(rf.GenericStateImpBase):
    """We don't have a Emissivity class at the C++ level. We may add that, but
    for now just add a python class to handle this."""

    def __init__(
        self,
        emisv: np.ndarray,
        emis_sd: rf.SpectralDomain,
        update_arr: np.ndarray,
        mp: rf.StateMapping = rf.StateMappingLinear,
    ):
        super().__init__()
        self.init(emisv, mp)
        # We want to get the logic used in determining update_arr into this class,
        # but for now leverage off what we get passed in.
        self.update_arr = update_arr
        self._emis_sd = emis_sd
        if emisv.shape != emis_sd.data.shape:
            raise RuntimeError(
                "emisivity and emisivity spectral domain need to be the same size"
            )

    @property
    def emissivity(self) -> rf.ArrayAd_double_1:
        return self.mapped_state

    @property
    def emissivity_spectral_domain(self) -> rf.SpectralDomain:
        return self._emis_sd

    def desc(self) -> str:
        return "EmisState"

    def clone(self) -> rf.GenericState:
        return EmisState(
            self.emissivity, self.emissivity_spectral_domain, self.state_mapping
        )

    def sub_state_identifier(self) -> str:
        return "emis_state"

    def state_vector_name_i(self, i: int) -> str:
        mname = self.state_mapping.name
        if mname != "linear":
            return f"{mname} Emissivity Freq {i + 1}"
        return f"Emissivity Freq {i + 1}"


__all__ = [
    "EmisState",
]

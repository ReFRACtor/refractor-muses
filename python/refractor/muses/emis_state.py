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
        update_arr: np.ndarray | None,
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
        self.initial_value = emisv.copy()

    @property
    def emissivity(self) -> rf.ArrayAd_double_1:
        # Would like to move this into a StateMapping if we can figure out
        # the logic
        ms = self.mapped_state
        if self.update_arr is None or self.update_arr.shape[0] == 0:
            return ms
        # Logic only needed when we have update_arr, which is only if we
        # are retrieving this element
        res = rf.ArrayAd_double_1(ms.rows, ms.number_variable)
        for i in range(self.update_arr.shape[0]):
            if self.update_arr[i]:
                res[i] = ms[i]
            else:
                # TODO Look into this
                # Note it actually seems wrong that we have a nonzero jacobian here,
                # but this is what py-retrieve does.
                #
                # However, I did try removing this, and things changed a lot. It is
                # possible the error analysis etc. depends on the jacobian (e.g. it is
                # a bit like the systematic jacobians). Somebody smarter than me
                # will need to look into this
                res[i] = rf.AutoDerivativeDouble(self.initial_value[i], ms[i].gradient)
        return res

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

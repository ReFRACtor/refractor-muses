from __future__ import annotations
import refractor.framework as rf  # type: ignore
import numpy as np
import typing
import pprint
import subprocess

if typing.TYPE_CHECKING:
    from .cost_function import CostFunction


class CompareForwardModel(rf.ForwardModel):
    """Simple class to run two forward model in parallel, so we can look
    at differences between them"""

    def __init__(
        self,
        fm1: rf.ForwardModel,
        fm2: rf.ForwardModel,
    ) -> None:
        """The first fm1 "drives" the output, then second can be use to compare
        against. Edit the
        """
        super().__init__()
        self.fm1 = fm1
        self.fm2 = fm2

    def setup_grid(self) -> None:
        # Nothing that we need to do for this
        self.fm1.setup_grid()
        self.fm2.setup_grid()

    def _v_num_channels(self) -> int:
        return self.fm1.num_channels

    def spectral_domain(self, sensor_index: int) -> rf.SpectralDomain:
        res1 = self.fm1.spectral_domain(sensor_index)
        res2 = self.fm2.spectral_domain(sensor_index)  # noqa:F841
        # Add whatever comparison is wanted here
        return res1

    def irk(self, current_state: CurrentState) -> ResultIrk:
        # Not all ForwardModel support IRK. That's fine, we just supply this
        # for comparing one that do. This just fails if the forward model doesn't
        # have an IRK calculation.
        res1 = self.fm1.irk(current_state)
        res2 = self.fm2.irk(current_state)
        with open("fm1_uip.txt", "w") as fh:
            pprint.pprint(self.fm1.rf_uip.uip, fh)
        with open("fm2_uip.txt", "w") as fh:
            pprint.pprint(self.fm2.radiative_transfer.rf_uip.uip, fh)
        subprocess.run(["diff", "-u", "fm1_uip.txt", "fm2_uip.txt"])
        # Add whatever comparison is wanted here
        breakpoint()
        return res1

    def notify_cost_function(self, cfunc: CostFunction) -> None:
        if hasattr(self.fm1, "notify_cost_function"):
            self.fm1.notify_cost_function(cfunc)
        if hasattr(self.fm2, "notify_cost_function"):
            self.fm2.notify_cost_function(cfunc)

    def radiance(self, sensor_index: int, skip_jacobian: bool = False) -> rf.Spectrum:
        res1 = self.fm1.radiance(sensor_index, skip_jacobian)
        res2 = self.fm2.radiance(sensor_index, skip_jacobian)  # noqa:F841
        # Add whatever comparison is wanted here
        if False:
            if not np.allclose(res1.spectral_domain.data, res2.spectral_domain.data):
                breakpoint()
            with open("fm1_uip.txt", "w") as fh:
                pprint.pprint(self.fm1.rf_uip.uip, fh)
            with open("fm2_uip.txt", "w") as fh:
                pprint.pprint(self.fm2.radiative_transfer.rf_uip.uip, fh)
            subprocess.run(["diff", "-u", "fm1_uip.txt", "fm2_uip.txt"])
            if not np.allclose(res1.spectral_range.data, res2.spectral_range.data):
                breakpoint()
        if not np.allclose(res1.spectral_range.data, res2.spectral_range.data):
            breakpoint()
        if not np.allclose(res1.spectral_range.data_ad.jacobian, res2.spectral_range.data_ad.jacobian):
            breakpoint()
        return res1


__all__ = [
    "CompareForwardModel",
]

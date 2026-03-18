from __future__ import annotations
import refractor.framework as rf  # type: ignore
import numpy as np


# Note I looked at making this a full rf.AbsorberVmr. Turns out this was a bit
# involved, with no real advantage. The way this is used duck typing is fine, we
# don't need a full rf.AbsorberVmr here. We could revisit this if needed, but for
# now don't bother doing the work for no real gain.
class VmrModify:
    """This class handles things like negative VMRs."""

    def __init__(self, initial_absorber_vmr: rf.AbsorberVmr) -> None:
        self._underlying_absorber_vmr = initial_absorber_vmr

    @property
    def gas_name(self) -> str:
        return self.underlying_absorber_vmr.gas_name

    @property
    def underlying_absorber_vmr(self) -> rf.AbsorberVmr:
        return self._underlying_absorber_vmr

    def vmr_grid(
        self, press: rf.Pressure, pdir: rf.Pressure.PressureGridType
    ) -> rf.ArrayAd_double_1:
        raise NotImplementedError()

    def update_rt_radiance(
        self,
        rad: np.ndarray,
        drad_dvmr: np.ndarray | None,
        press: rf.Pressure,
        pdir: rf.Pressure.PressureGridType,
    ) -> np.ndarray:
        """drad_dvmr should be"""
        return rad


class VmrModifySmallToFixed(VmrModify):
    """Replace small or negative vmr values with a threshold value"""

    def __init__(self, initial_absorber_vmr: rf.AbsorberVmr, threshold: float) -> None:
        super().__init__(initial_absorber_vmr)
        self.threshold = threshold

    def vmr_grid(
        self, press: rf.Pressure, pdir: rf.Pressure.PressureGridType
    ) -> rf.ArrayAd_double_1:
        vmr_s = self.underlying_absorber_vmr.vmr_grid(press, pdir)
        # TODO Any reason why this is restricted to linear? This is the
        # logic in py-retrieve and we duplicate it, but might want
        # to just replace any negative value.
        if (
            not hasattr(self.underlying_absorber_vmr, "state_mapping")
            or not self.underlying_absorber_vmr.state_mapping.name == "linear"
        ):
            return vmr_s
        vmr = vmr_s.value.copy()
        vmr[vmr < self.threshold] = self.threshold
        return rf.ArrayAd_double_1(vmr, vmr_s.jacobian)


class VmrHandleNeg(VmrModify):
    """We handle negative PAN values by:

    1. Replacing negative VMR with 1e-11. Call the original VMR VMR0
    2. Run OSS (outside of this class)
    3. Modify the radiance by K @ (VMR0 - VMR)
    """

    def __init__(self, initial_absorber_vmr: rf.AbsorberVmr, threshold: float) -> None:
        super().__init__(initial_absorber_vmr)
        self.threshold = threshold
        self.have_negative = False

    def vmr_grid(
        self, press: rf.Pressure, pdir: rf.Pressure.PressureGridType
    ) -> rf.ArrayAd_double_1:
        vmr_s = self.underlying_absorber_vmr.vmr_grid(press, pdir)
        # Logic is a bit convoluted here, but we replace
        # any vmr < 1e-11, but *only* if at least one of
        # the vmrs is negative. I'm not sure if this
        # specific logic was intended or not (why
        # different than for other state elements)?, but
        # this is what is done in the py-retrieve code
        if np.count_nonzero(vmr_s.value < 0) == 0:
            return vmr_s

        self.have_negative = True
        # Save for use in update_rt_radiance, vmr before we have modified it
        self.vmr_muses = vmr_s.value
        vmr = vmr_s.value.copy()
        vmr[vmr < self.threshold] = self.threshold
        # Save for use in update_rt_radiance, vmr after modification
        self.vmr_oss = vmr
        return rf.ArrayAd_double_1(vmr, vmr_s.jacobian)

    # TODO Change to take pressure and direction, so we aren't saving data here. Don't
    # want an assume call order
    def update_rt_radiance(
        self,
        rad: np.ndarray,
        drad_dvmr: np.ndarray | None,
        press: rf.Pressure,
        pdir: rf.Pressure.PressureGridType,
    ) -> np.ndarray:
        """Perform any updates to rad, i.e., with negative VMR handling"""
        if not self.have_negative:
            return rad
        if drad_dvmr is None:
            raise RuntimeError("Need drad_dvmr for update_rt_radiance")
        # modify radiance to ACTUAL VMR (vmr_muses) using K.dx
        # vrm_oss used by OSS, vmr_muses is what we want
        return rad + drad_dvmr @ (self.vmr_muses - self.vmr_oss)


__all__ = ["VmrModify", "VmrModifySmallToFixed", "VmrHandleNeg"]

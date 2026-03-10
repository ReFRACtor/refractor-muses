from __future__ import annotations
import refractor.framework as rf  # type: ignore
from .muses_oss_handle import muses_oss_handle
from .identifier import StateElementIdentifier
import numpy as np


class VmrModify:
    """This class handles things like negative VMRs. We could have
    these modified values passed absorber_vmr_list, but for now we
    just handle this here. I think having the knowlege of the special
    case VMRs in one place makes some sense, but we can rethink that.
    """

    def __init__(self, initial_absorber_vmr: rf.AbsorberVmr) -> None:
        self.initial_absorber_vmr = initial_absorber_vmr

    def vmr_grid(
        self, press: rf.Pressure, pdir: rf.Pressure.PressureGridType
    ) -> rf.ArrayAd_double_1:
        raise NotImplementedError()

    def post_process(
        self, rad: np.ndarray, drad_datm_jac_spec: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        return rad, drad_datm_jac_spec


class VmrModifyNegToFixed(VmrModify):
    def __init__(self, initial_absorber_vmr: rf.AbsorberVmr, threshold: float) -> None:
        super().__init__(initial_absorber_vmr)
        self.threshold = threshold

    def vmr_grid(
        self, press: rf.Pressure, pdir: rf.Pressure.PressureGridType
    ) -> rf.ArrayAd_double_1:
        vmr_s = self.initial_absorber_vmr.vmr_grid(press, pdir)
        # TODO Any reason why this is restricted to linear? This is the
        # logic in py-retrieve and we duplicate it, but might want
        # to just replace any negative value.
        if (
            not hasattr(self.initial_absorber_vmr, "state_mapping")
            or not self.initial_absorber_vmr.state_mapping.name == "linear"
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
        vmr_s = self.initial_absorber_vmr.vmr_grid(press, pdir)
        # Logic is a bit convoluted here, but we replace
        # any vmr < 1e-11, but *only* if at least one of
        # the vmrs is negative. I'm not sure if this
        # specific logic was intended or not (why
        # different than for other state elements)?, but
        # this is what is done in the py-retrieve code
        if np.count_nonzero(vmr_s.value < 0) == 0:
            return vmr_s

        self.have_negative = True
        # Save for use in post_process, vmr before we have modified it
        self.vmr_muses = vmr_s.value
        vmr = vmr_s.value.copy()
        vmr[vmr < self.threshold] = self.threshold
        # Save for use in post_process, vmr after modification
        self.vmr_oss = vmr
        return rf.ArrayAd_double_1(vmr, vmr_s.jacobian)

    def post_process(
        self, rad: np.ndarray, drad_datm_jac_spec: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Perform any updates to rad, i.e., with negative VMR handling"""
        if not self.have_negative:
            return rad, drad_datm_jac_spec
        indjac = muses_oss_handle.atm_jac_spec.index(
            StateElementIdentifier(self.initial_absorber_vmr.gas_name)
        )
        k = drad_datm_jac_spec[:, :, indjac].copy() / self.vmr_oss[:, np.newaxis]
        # modify radiance to ACTUAL VMR (vmr_muses) using K.dx
        # vrm_oss used by OSS, vmr_muses is what we want
        dL = k.T @ (self.vmr_muses - self.vmr_oss)
        rad2 = rad + dL
        drad_datm_jac_spec2 = drad_datm_jac_spec.copy()
        drad_datm_jac_spec2[:, :, indjac] = k * self.vmr_muses[:, np.newaxis]
        return rad2, drad_datm_jac_spec2


class MusesOssAtmosphere:
    """The muses-oss code takes an "atmosphere" argument. This is the
    VMR for a set of absorbers.

    This class handles the mapping to the muses-oss code from our
    standard rf.AbsorberVmr objects.

    There are a few things floating around here to be aware of.

    The MUSES-OSS initialization takes two arguments, a list of gases
    and a list of gases to include in the jacobians. The list of gases
    is I think fixed, this seems to be information about what is in
    the OSS support files - basically like a list of possible absco
    gases in our OCO-2 forward models. I believe this is just
    information about the contents of the OD file used by OSS, this
    seems to correspond to the list "molecName" in ConvertModule.f90
    of muses-oss code.

    A subset of these gases are marked for having jacobians
    generated. This subset will vary from strategy step to strategy
    step.

    Both of theses lists get stored in the muses_oss_handle, we use
    this object to determine these.

    Note that the order of gases is important, it is the order that
    the ultimate oss_atmosphere columns are listed.

    In addition, a particular strategy step will only include a
    subset of gases to include in the RT calculation. This usually
    comes from looking at the metadata in the microwindows files to
    determine what gases are covered by the microwindows. However, this
    class makes no assumption about how this information is collected,
    we just take in a absorber_vmr_list of absorbers to include.

    So we have "full list of gases" >= "absorber vmr list" >= "gas jacobian"

    Note that oss_atmosphere *always* has the full number of columns
    to cover the full list of gases. However, we just zero out
    (actually 1e-20 since 0 VMR causes problems) the gases not in the
    absorber vmr list. So effectively the gases aren't used, although
    there is an entry passes to the OSS code.

    Finally, there is special handling in place for PAN going
    negative. This isn't physical, but during a retrieval it is useful
    to allow these to go negative. However the OSS code can't handle
    negative VMR.  We handle this by:

    1. Replacing negative VMR with 1e-11. Call the original VMR VMR0
    2. Run OSS (outside of this class)
    3. Modify the radiance by K @ (VMR0 - VMR)

    In addition, there are other gases where negative values are replaced
    with a fixes threshold.
    """

    def __init__(
        self,
        absorber_vmr_list: list[rf.AbsorberVmr],
    ) -> None:
        self.absorber_vmr = {
            StateElementIdentifier(vmr.gas_name): vmr for vmr in absorber_vmr_list
        }
        # Special handling for some species
        for selem, threshold in (
            (StateElementIdentifier("HCN"), 1e-12),
            (StateElementIdentifier("NH3"), 5e-11),
            (StateElementIdentifier("ACET"), 1e-12),
        ):
            if selem in self.absorber_vmr:
                self.absorber_vmr[selem] = VmrModifyNegToFixed(
                    self.absorber_vmr[selem], threshold
                )
        selem = StateElementIdentifier("PAN")
        threshold = 1e-11
        if selem in self.absorber_vmr:
            self.absorber_vmr[selem] = VmrHandleNeg(self.absorber_vmr[selem], threshold)

    def oss_atmosphere(self, press: rf.Pressure) -> tuple[np.ndarray, np.ndarray | None]:
        """Return np.ndarray that we should pass to OSS code for doing
        RT.

        Note we could return a rf.ArrayAd like we typically do. However, only
        a subset of the state elements actually have their jacobian calculated
        by the OSS code. So as a convenience we just return datm_jac_spec_dstate which
        is the subset of dvmr_dstate for columns that occur in the OSS jacobians.
        """
        # Sanity check that we don't have any absorbers not supported
        # by the OSS code
        if len(set(self.absorber_vmr.keys()) - set(muses_oss_handle.atm_spec)) > 0:
            raise RuntimeError(
                "absorber_vmr_list has gases not supported by the OSS code"
            )
        # Also, any jacobians we have need to have absorber_vmr
        if len(set(muses_oss_handle.atm_jac_spec) - set(self.absorber_vmr.keys())) > 0:
            raise RuntimeError("atm_jac_spec has gases not in absorber_vmr")
        # Go through all the absorbers needed by OSS, and get values for them.
        # For absorbers not in absorber_vmr, fill in as all 1e-20
        res = []
        res_jac = []
        for spc in muses_oss_handle.atm_spec:
            if spc in self.absorber_vmr:
                vmr = (
                    self.absorber_vmr[spc]
                    .vmr_grid(press, rf.Pressure.DECREASING_PRESSURE)
                )
                res.append(vmr.value)
                if spc in muses_oss_handle.atm_jac_spec:
                    res_jac.append(vmr.jacobian)
            else:
                res.append(np.full((press.number_level,), 1e-20))
        if len(res_jac) > 0:
            jac = np.vstack(res_jac)
        else:
            jac = None
        return np.vstack(res).T, jac

    def post_process(
        self, rad: np.ndarray, drad_datm_jac_spec: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Perform any updates to rad, i.e., with negative VMR handling"""
        rad2 = rad
        drad_datm_jac_spec2 = drad_datm_jac_spec
        for spc in muses_oss_handle.atm_spec:
            if spc in self.absorber_vmr:
                if hasattr(self.absorber_vmr[spc], "post_process"):
                    rad2, drad_datm_jac_spec2 = self.absorber_vmr[spc].post_process(
                        rad2, drad_datm_jac_spec2
                    )
        return rad2, drad_datm_jac_spec2


__all__ = [
    "MusesOssAtmosphere",
]

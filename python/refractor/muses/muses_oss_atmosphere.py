from __future__ import annotations
import refractor.framework as rf  # type: ignore
from .muses_oss_handle import muses_oss_handle
from .identifier import StateElementIdentifier
import numpy as np


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

    The special handling is put in place outside of this class, the
    absorber_vmr_list passed in by MusesOssFmObjectCreator has special
    versions of the VMR classes for these items.
    """

    def __init__(
        self,
        absorber_vmr_list: list[rf.AbsorberVmr],
    ) -> None:
        self.absorber_vmr = {
            StateElementIdentifier(vmr.gas_name): vmr for vmr in absorber_vmr_list
        }
        # TODO Move to fm_creator, pull out vmr and separately test
        # Special handling for some species

    def oss_atmosphere(
        self, press: rf.Pressure, pdir: rf.Pressure.PressureGridType
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        """Return np.ndarray that we should pass to OSS code for doing
        RT.

        Note we could return a rf.ArrayAd like we typically do. However, only
        a subset of the state elements actually have their jacobian calculated
        by the OSS code. So as a convenience we just return dvmr_dstate_subset which
        is the subset of dvmr_dstate for columns that occur in the OSS jacobians.
        This is indexed by gas_index, vmr_index, state_index

        In addition, the muses oss code calculate drad_dlog_vmr rather than drad_dvmr.
        We return dlog_vmr_dvmr to convert this. This is gas_index, vmr_index
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
        res_dvmr_dstate = []
        res_dlog_vmr_dvmr = []
        for spc in muses_oss_handle.atm_spec:
            if spc in self.absorber_vmr:
                vmr = self.absorber_vmr[spc].vmr_grid(press, pdir)
                res.append(vmr.value)
                if spc in muses_oss_handle.atm_jac_spec:
                    # Add a column for the gas index
                    res_dvmr_dstate.append(vmr.jacobian[np.newaxis, :])
                    res_dlog_vmr_dvmr.append((1 / vmr.value)[np.newaxis, :])
            else:
                res.append(np.full((press.number_level,), 1e-20))
        if len(res_dvmr_dstate) > 0:
            # This is ngas x nlevel x nstate
            dvmr_dstate = np.vstack(res_dvmr_dstate)
            dlog_vmr_dvmr = np.vstack(res_dlog_vmr_dvmr)
        else:
            dvmr_dstate = None
            dlog_vmr_dvmr = None
        return np.vstack(res).T, dvmr_dstate, dlog_vmr_dvmr

    def update_rt_radiance(
        self,
        rad: np.ndarray,
        drad_dvmr: np.ndarray | None,
        press: rf.Pressure,
        pdir: rf.Pressure.PressureGridType,
    ) -> np.ndarray:
        """Perform any updates to rad, i.e., with negative VMR handling

        drad_dvmr should be rad_index, gas_index, vmr_index
        """
        rad2 = rad
        for spc in muses_oss_handle.atm_spec:
            if spc in self.absorber_vmr:
                if hasattr(self.absorber_vmr[spc], "update_rt_radiance"):
                    if spc in muses_oss_handle.atm_jac_spec and drad_dvmr is not None:
                        jac = drad_dvmr[:, muses_oss_handle.atm_jac_spec.index(spc), :]
                    else:
                        jac = None
                    rad2 = self.absorber_vmr[spc].update_rt_radiance(
                        rad2, jac, press, pdir
                    )
        return rad2


__all__ = [
    "MusesOssAtmosphere",
]

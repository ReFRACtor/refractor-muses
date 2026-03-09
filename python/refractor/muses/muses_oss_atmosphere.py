from __future__ import annotations
import refractor.framework as rf  # type: ignore
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

    Finally, there is special handling in place for PAN and NH3 going
    negative. This isn't physical, but during a retrieval it is useful to
    allow these to go negative. However the OSS code can't handle negative VMR.
    We handle this by:

    1. Replacing negative VMR with 1e-11. Call the original VMR VMR0
    2. Run OSS (outside of this class)
    3. Modify the radiance by K @ (VMR0 - VMR)

    The special handling seemed a little like our various spectrum
    effects we support in a rf.StandardForwardModel. I had considered
    making some sort of "VmrEffect" class to support more general
    modifications. However it isn't clear how general this behavior
    actually needs to be. For now, we just hardcode this behavior, and
    hardcode the gases that perform this. If we get a few more
    examples of modifications, we may be able to come up with a
    generalized way of doing this.

    """

    def __init__(self, absorber_vmr_list: list[rf.AbsorberVmr]) -> None:
        self.absorber_vmr = {vmr.gas_name: vmr for vmr in absorber_vmr_list}

    @property
    def oss_atmosphere(self) -> np.ndarray:
        """Return np.ndarray that we should pass to OSS code for doing
        RT."""
        pass

    def post_process(self, rad: np.ndarray, drad_datm: np.ndarray) -> np.ndarray:
        """Perform any updates to rad, i.e., with negative VMR handling"""
        pass


__all__ = [
    "MusesOssAtmosphere",
]

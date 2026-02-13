from __future__ import annotations
import refractor.framework as rf  # type: ignore
from loguru import logger
import numpy as np
import typing

if typing.TYPE_CHECKING:
    import refractor.muses_py_fm

class FmMusesRayInfoUpdateUip(rf.ObserverStateVector):
    def __init__(self, rinfo: MusesRayInfo) -> None:
        super().__init__()
        self.rinfo = rinfo

    def notify_update(self, fm_sv:rf.StateVector) -> None:
        logger.debug(f"Call to {self.__class__.__name__}::notify_update")
        mp = rf.StateMappingBasisMatrix(self.rinfo.rf_uip.basis_matrix.transpose())
        # Take data from fm_sv to the RetrievalGridArray that update_uip
        # work with. Note that update_uip just turns around and creates
        # fm_vec from rvec - but we don't want to change the interface
        # to this.
        rvec = mp.retrieval_state(fm_sv.state_with_derivative).value
        self.rinfo.update_uip(rvec)

class RayInfoUpdateUip(rf.ObserverMaxAPosterioriSqrtConstraint):
    def __init__(self, rinfo: MusesRayInfo) -> None:
        super().__init__()
        self.rinfo = rinfo

    def notify_update(self, mstand: rf.MaxAPosterioriSqrtConstraint) -> None:
        logger.debug(f"Call to {self.__class__.__name__}::notify_update")
        # Directly work with cost function parameters. Very slightly
        # different than FmMusesRayInfoUpdateUip
        self.rinfo.update_uip(mstand.parameters)

class MusesRayInfo:
    """There are a number of places where RefractorFmObjectCreator and
    related objects need the "ray_info" information.

    Unfortunately, the code for this is reasonably complicated in
    py-retrieve and depends on the old UIP structure.

    We pull all of this out into one class - just to isolate this.

    Note that this class is only used to match how py-retrieve did
    calculations, we don't normally use this for a ReFRACtor
    retrieval. But it was initially useful to match how py-retrieve
    did the calculation, so we could compare the forward model runs
    with ReFRACtor without having the minor differences in the
    calculations enter into the differences.

    This is not something we normally use, instead ReFRACtor code is
    used. But this is class is useful if we want to compare against
    old py-retrieve results
    """

    def __init__(
        self,
        rf_uip: refractor.muses_py_fm.RefractorUip,
        instrument_name: str,
        pressure: rf.Pressure,
        set_pointing_angle_zero: bool = True,
    ) -> None:
        """Constructor.

        Note we use pressure to get the number of layers for some of
        the calculations.  This is often a PressureWithCloudHandling,
        which has different number of layers depending on if we are
        doing the cloud or clear part of the RT calculations. I think
        this is the only information we need here, so if for whatever
        reason there is an alternative way to get this the pressure
        could be replaced with something else - ultimately we just
        need "pressure.number_layer" to return the number of layers.

        tropomi_fm and omi_fm set the point angle to zero before
        calling mpyraylayer_nadir.  I'm not sure if always want to do
        this or not, so we have a flag option here. But as far as I know,
        we also want the default value of set_pointing_angle_zero to True.
        """
        self.rf_uip = rf_uip
        self.instrument_name = instrument_name
        self.pressure = pressure
        self.set_pointing_angle_zero = set_pointing_angle_zero

    def _ray_info(self) -> dict:
        """Return the ray info.

        To help make the interfaces clear, we consider this function
        to be private, and just have some public functions to pull
        pieces out.

        This is just to figure out the interfaces when we replace this
        object - we could make this function public if needed.
        """
        return self.rf_uip.ray_info(
            self.instrument_name, set_pointing_angle_zero=self.set_pointing_angle_zero
        )

    def _nlay(self) -> int:
        return self.pressure.number_layer

    def _nlev(self) -> int:
        return self.pressure.number_level

    def notify_cost_function(self, cfunc: CostFunction) -> None:
        # See also FmMusesRayInfoUpdateUip, it is a little more direct
        # to attach to the fm_sv instead of the CostFunction. But we
        # have both available here - the cost function is how the
        # py-retrieve solver updates the uip. Very slight differences
        # between the two - just the round of going from fm_sv - > ret_state
        # -> fm_vec that FmMusesRayInfoUpdateUip does (as opposed to
        # ret_state -> fm_vec that RayInfoUpdateUip does)
        cfunc.max_a_posteriori.add_observer_and_keep_reference(RayInfoUpdateUip(self))

    def update_uip(self, parameters: np.ndarray) -> None:
        if self.rf_uip.basis_matrix is not None:
            self.rf_uip.update_uip(parameters)
        
    def tbar(self) -> np.ndarray:
        """Return tbar. This gets used in MusesRaman, I don't think this is used
        anywhere else."""
        return self._ray_info()["tbar"][::-1][: self._nlay()]

    def pbar(self) -> np.ndarray:
        """Return tbar. This gets used in MusesRaman, I don't think this is used
        anywhere else."""
        return self._ray_info()["pbar"][::-1][: self._nlay()]
    
    def altitude_grid(self) -> np.ndarray:
        """Return altitude grid of each level. This gets used in MusesAltitude, I don't think
        it is used anywhere else"""

        t = self._ray_info()["level_params"]["radius"]
        hlev = t[:] - t[0]
        return hlev[::-1][: self._nlev()]

    def map_vmr(self) -> tuple[np.ndarray, np.ndarray]:
        """Return map_vmr_l and map_vmr_u. This gets used in MusesOpticalDepthFile."""
        t = self._ray_info()
        return t["map_vmr_l"][::-1], t["map_vmr_u"][::-1]

    def number_cloud_layer(self, cloud_pressure: float) -> int:
        """Return the number of cloud layers. This is used in RefractorFmObjectCreator"""
        return int(np.count_nonzero(self._ray_info()["pbar"] <= cloud_pressure))

    def dry_air_density(self) -> np.ndarray:
        """Return dry air density. This gets used in MusesOpticalDepthFile."""
        return self._ray_info()["column_air"][::-1][: self._nlay()]

    def gas_density_layer(self, species_name: str) -> np.ndarray:
        """Return gas density layer for given species name."""
        t = self._ray_info()
        ind = np.where(np.asarray(t["level_params"]["species"]) == species_name)[0]
        return t["column_species"][ind, ::-1].squeeze()[: self._nlay()]


__all__ = ["MusesRayInfo", "FmMusesRayInfoUpdateUip"]

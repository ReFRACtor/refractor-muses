from __future__ import annotations
import refractor.framework as rf  # type: ignore
from .muses_oss_handle import muses_oss_handle
from .identifier import StateElementIdentifier, InstrumentIdentifier
import numpy as np
import os
import copy
import typing
from typing import Self

if typing.TYPE_CHECKING:
    from .input_file_helper import InputFilePath, InputFileHelper
    from .emis_state import EmisState
    from .cloud_ext_state import CloudExtState
    from .muses_oss_atmosphere import MusesOssAtmosphere


class MusesRadiativeTransferOss(rf.RadiativeTransferImpBase):
    """This uses the muses OSS code (package muses-oss). This gives a forward
    model that is the same as the py-retrieve airs/cris/tes forward model (with minor
    differences in calculation - the normal sort of round off differences).
    """

    def __init__(
        self,
        press: rf.Pressure,
        temperature: rf.Temperature,
        tsur: rf.SurfaceTemperature,
        pcloud: rf.Pcloud,
        scale_cloud: rf.ScaleCloud,
        emissivity: EmisState,
        cloud_ext: CloudExtState,
        atmosphere: MusesOssAtmosphere,
        surface_altitude: rf.DoubleWithUnit,
        latitude: rf.DoubleWithUnit,
        pointing_angle_surface: rf.DoubleWithUnit,
        instrument_name: InstrumentIdentifier,
        ifile_hlp: InputFileHelper,
        retrieval_state_element_id: list[StateElementIdentifier],
        species_list: list[StateElementIdentifier],
        sel_file: str | os.PathLike[str] | InputFilePath,
        od_file: str | os.PathLike[str] | InputFilePath,
        sol_file: str | os.PathLike[str] | InputFilePath,
        fix_file: str | os.PathLike[str] | InputFilePath,
    ) -> None:
        """species_list is the total list of supported gases in the
        OSS. I believe this is just information about the contents of
        the OD file used by OSS, this seems to correspond to the list
        "molecName" in ConvertModule.f90 of muses-oss code. In any
        case, we take this in as an argument.

        Note only a subset of these gases are actually included in the
        RT, see MusesOssAtmosphere for a discussion of
        this. retrieval_state_element_id will in general contain only
        a subset of the gases included in the RT where we calculate
        the jacobians (as well as other things we retrieve unrelated
        to the gases).
        """
        super().__init__()
        self.pressure = press
        self.temperature = temperature
        self.tsur = tsur
        self.pcloud = pcloud
        self.scale_cloud = scale_cloud
        self.emissivity = emissivity
        self.cloud_ext = cloud_ext
        self.atmosphere = atmosphere
        self.surface_altitude = surface_altitude
        self.latitude = latitude
        self.pointing_angle_surface = pointing_angle_surface
        self.instrument_name = instrument_name
        self.ifile_hlp = ifile_hlp
        self.retrieval_state_element_id = retrieval_state_element_id
        self.species_list = species_list
        self.sel_file = sel_file
        self.od_file = od_file
        self.sol_file = sol_file
        self.fix_file = fix_file

    def clone(self) -> Self:
        return copy.deepcopy(self)

    def reflectance(
        self,
        sd: rf.SpectralDomain,
        sensor_index: int,
        skip_jacobian: bool,
        pointing_angle_surface: rf.DoubleWithUnit | None = None,
    ) -> rf.Spectrum:
        """Note that despite the name, this is actually radiance. We named this
        back when we just had LIDORT, which does return reflectance. The OSS
        forward model returns radiance.

        We do have units right in the rf.Spectrum we pass back. It is just the
        name of this function that is wrong.

        We probably need so generic name for "thing the forward model returns",
        similar to how we had SpectralDomain for "wavelength or wavenumber". But
        for now, we just use this incorrect name.

        Note that the function in StandardForwardModel is actually correctly
        named for this code (unlike our VLIDORT/LIDORT which doesn't include the
        solar model so it really does return reflectance).

        We allow the pointing_angle_surface to be passed in. This overrides the one
        passed in the constructor. This is used to support the IRK calculation. This
        does mean that our function isn't a rf.RadiativeTransfer.reflectance - it has
        an extra optional argument. It isn't clear the best way to handle the IRK, but
        for right now this seems to be the cleanest way to have this extra functionality.
        """
        muses_oss_handle.oss_init(
            self.ifile_hlp,
            self.retrieval_state_element_id,
            self.species_list,
            self.pressure.number_level,
            self.emissivity.emissivity_spectral_domain.rows,
            self.sel_file,
            self.od_file,
            self.sol_file,
            self.fix_file,
        )
        muses_oss_handle.oss_channel_select(sd)
        tsurv = self.tsur.surface_temperature(sensor_index).value
        scale_cloudv = self.scale_cloud.scale_cloud(sensor_index)
        pcloudv = self.pcloud.pressure_cloud(sensor_index).convert("mbar").value
        pres = (
            self.pressure.pressure_grid(rf.Pressure.DECREASING_PRESSURE)
            .convert("mbar")
            .value
        )
        tatm = (
            self.temperature.temperature_grid(
                self.pressure, rf.Pressure.DECREASING_PRESSURE
            )
            .convert("K")
            .value
        )
        oss_atmosphere, dvmr_dstate, dlog_vmr_dvmr = self.atmosphere.oss_atmosphere(
            self.pressure, rf.Pressure.DECREASING_PRESSURE
        )
        emisv = self.emissivity.emissivity
        cloudextv = self.cloud_ext.cloud_ext.convert("km^-1").value

        salt = self.surface_altitude.convert("m").value
        # TODO Not sure if the logic of this here, but this is what py-retrieve does
        if salt < 1e-5:
            salt = 1e-5
        # Use diffusion approximation, see oss_if_module.f90 in muses_oss
        lambertian_flag = 1
        # py-retrieve has sunang always 90. Not sure of the significance of that,
        # but do that for now
        sunang = 90.0
        # Units that we want
        rad_units = rf.Unit("W / (cm^2 sr cm^-1)")
        (
            rad,
            drad_dtemp,
            drad_dtsur,
            drad_dlog_vmr,
            drad_demis,
            drad_drefl,
            drad_dlog_pcloud,
            drad_dlog_cloudext,
        ) = muses_oss_handle.oss_forward_model(
            tsurv.value,
            scale_cloudv.value,
            pcloudv.value,
            pointing_angle_surface.convert("deg").value
            if pointing_angle_surface is not None
            else self.pointing_angle_surface.convert("deg").value,
            sunang,
            self.latitude.convert("deg").value,
            salt,
            lambertian_flag,
            pres.value,
            tatm.value,
            oss_atmosphere,
            self.emissivity.emissivity_spectral_domain.convert_wave("cm^-1"),
            emisv.value,
            self.cloud_ext.cloud_ext_spectral_domain.convert_wave("cm^-1"),
            cloudextv.value,
            rad_units,
        )
        # Convert to drad_dvmr, and shuffle axis so this is rad_index, gas_index, vmr_index.
        if dlog_vmr_dvmr is not None:
            drad_dvmr = (
                drad_dlog_vmr.transpose([1, 2, 0]) * dlog_vmr_dvmr[np.newaxis, :, :]
            )
        else:
            drad_dvmr = None
        dlog_pcloud_dpcloud = 1 / pcloudv.value
        drad_dpcloud = drad_dlog_pcloud * dlog_pcloud_dpcloud
        dlog_cloudext_dcloudext = 1 / cloudextv.value
        drad_dcloudext = drad_dlog_cloudext * dlog_cloudext_dcloudext[:, np.newaxis]
        rad = self.atmosphere.update_rt_radiance(
            rad, drad_dvmr, self.pressure, rf.Pressure.DECREASING_PRESSURE
        )

        # check finite
        if not np.all(np.isfinite(rad)):
            raise RuntimeError("Non-finite radiance")

        if drad_dvmr is not None and not np.all(np.isfinite(drad_dvmr)):
            raise RuntimeError("Non-finite jacobians")

        jac = None
        if not emisv.is_constant:
            jac = self._add_jac(jac, drad_demis.T @ emisv.jacobian)
        if not cloudextv.is_constant:
            jac = self._add_jac(jac, drad_dcloudext.T @ cloudextv.jacobian)
        if not tsurv.is_constant:
            jac = self._add_jac(
                jac, drad_dtsur[:, np.newaxis] @ tsurv.gradient[np.newaxis, :]
            )
        if not tatm.is_constant:
            jac = self._add_jac(jac, drad_dtemp.T @ tatm.jacobian)
        if dvmr_dstate is not None:
            # axes = 2 because we have both vmr_index and gas_index to loop over.
            jac = self._add_jac(jac, np.tensordot(drad_dvmr, dvmr_dstate, axes=2))
        if not pcloudv.is_constant:
            jac = self._add_jac(
                jac, drad_dpcloud[:, np.newaxis] @ pcloudv.gradient[np.newaxis, :]
            )
        if jac is not None:
            a = rf.ArrayAd_double_1(rad, jac)
        else:
            a = rf.ArrayAd_double_1(rad)
        sr = rf.SpectralRange(a, rad_units)
        return rf.Spectrum(sd, sr)

    def stokes(self, sd: rf.SpectralDomain, sensor_index: int) -> np.ndarray:
        raise NotImplementedError(
            """Muses-oss doesn't work for  the full
            stoke vector."""
        )

    def stokes_and_jacobian(
        self, sd: rf.SpectralDomain, sensor_index: int
    ) -> rf.ArrayAd_double_2:
        raise NotImplementedError(
            """Muses-oss doesn't work for  the full
            stoke vector."""
        )

    def desc(self) -> str:
        return "MusesRadiativeTransferOss"

    def _add_jac(self, jac: np.ndarray | None, jacadd: np.ndarray) -> np.ndarray:
        "Add a jacobian, correctly handling none"
        if jac is None:
            return jacadd
        return jac + jacadd


__all__ = [
    "MusesRadiativeTransferOss",
]

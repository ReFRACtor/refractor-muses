from __future__ import annotations
import refractor.framework as rf  # type: ignore
from .muses_oss_handle import muses_oss_handle
import numpy as np
import os
from contextlib import contextmanager
import copy
import typing
from typing import Self, Iterator, Any

if typing.TYPE_CHECKING:
    from .identifier import StateElementIdentifier, InstrumentIdentifier
    from .input_file_helper import InputFilePath, InputFileHelper
    from .emis_state import EmisState
    from .cloud_ext_state import CloudExtState


class MusesRadiativeTransferOss(rf.RadiativeTransferImpBase):
    """This uses the muses OSS code (package muses-oss). This gives a forward
    model that is the same as the py-retrieve airs/cris/tes forward model (with minor
    differences in calculation - the normal sort of round off differences).
    """

    def __init__(
        self,
        rf_uip: rf.RefractorUip,  # Temp, leverage off UIP. We'll remove this in a bit
        press: rf.Pressure,
        temperture: rf.Temperature,
        tsur: rf.SurfaceTemperature,
        pcloud: rf.Pcloud,
        scale_cloud: rf.ScaleCloud,
        emissivity: EmisState,
        cloud_ext: CloudExtState,
        surface_altitude: rf.DoubleWithUnit,
        latitude: rf.DoubleWithUnit,
        pointing_angle_surface: rf.DoubleWithUnit,
        instrument_name: InstrumentIdentifier,
        ifile_hlp: InputFileHelper,
        retrieval_state_element_id: list[StateElementIdentifier],
        species_list: list[StateElementIdentifier],
        nlevels: int,
        nfreq: int,  # This seems to be the size of the emissivity. Perhaps verify,
        # And if so change it name. This has nothing to do with the
        # size of freq_oss that gets filled in
        sel_file: str | os.PathLike[str] | InputFilePath,
        od_file: str | os.PathLike[str] | InputFilePath,
        sol_file: str | os.PathLike[str] | InputFilePath,
        fix_file: str | os.PathLike[str] | InputFilePath,
    ) -> None:
        super().__init__()
        self.rf_uip = rf_uip
        self.pressure = press
        self.temperture = temperture
        self.tsur = tsur
        self.pcloud = pcloud
        self.scale_cloud = scale_cloud
        self.emissivity = emissivity
        self.cloud_ext = cloud_ext
        self.surface_altitude = surface_altitude
        self.latitude = latitude
        self.pointing_angle_surface = pointing_angle_surface
        self.instrument_name = instrument_name
        self.ifile_hlp = ifile_hlp
        self.retrieval_state_element_id = retrieval_state_element_id
        self.species_list = species_list
        self.nlevels = nlevels
        self.nfreq = nfreq
        self.sel_file = sel_file
        self.od_file = od_file
        self.sol_file = sol_file
        self.fix_file = fix_file

    def clone(self) -> Self:
        return copy.deepcopy(self)

    def dEdOD(self) -> np.ndarray:
        # Not clear where exactly this belongs. Have here for now, it would
        # be good if we could pull this out perhaps into IrkForwardModel
        try:
            ray_info = self.rf_uip.ray_info(
                self.instrument_name,
                set_pointing_angle_zero=True,
                set_cloud_extinction_one=True,
            )
        except KeyError:
            ray_info = self.rf_uip.ray_info(
                "AIRS", set_pointing_angle_zero=True, set_cloud_extinction_one=True
            )
        return 1.0 / ray_info["cloud"]["tau_total"]

    @contextmanager
    def modify_pointing(
        self, pointing_angle_surface: rf.DoubleWithUnit
    ) -> Iterator[None]:
        """For the IRK calculation, we need to generate the reflectance for
        different pointing angles.

        The most natural interface would be to pass the pointing angle
        as a argument to the ForwardModel radiance, and then to this
        objects reflectance. However, the rf.ForwardModel doesn't have
        an argument for this. I briefly considered changing the
        interface for ForwardModel, but it seems much cleaner to
        handle this without doing this.

        Instead, we provide a context manager here to set the pointing
        angle that we want - basically as a way to just add an extra
        argument.  This is similar to how we handle modifying the
        spectral window for MusesObservation (see
        MusesObservation.modify_spectral_window).

        Although this is a little indirect, this seems the cleanest
        way to handle this. This is currently only used in
        IrkForwardModel, so the fact that this is a little obscure
        seems a reasonable price to pay to keep the existing
        rf.ForwardModel interface.  We can reevaluate this in the
        future if needed.
        """
        original_pointing = self.pointing_angle_surface
        try:
            self.pointing_angle_surface = pointing_angle_surface
            yield
        finally:
            self.pointing_angle_surface = original_pointing

    def reflectance(
        self, sd: rf.SpectralDomain, sensor_index: int, skip_jacobian: bool
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
        """
        muses_oss_handle.oss_init(
            self.ifile_hlp,
            self.retrieval_state_element_id,
            self.species_list,
            self.nlevels,
            self.nfreq,
            self.sel_file,
            self.od_file,
            self.sol_file,
            self.fix_file,
        )
        muses_oss_handle.oss_channel_select(sd)
        try:
            uip_all = self.rf_uip.uip_all(self.instrument_name)
        except KeyError:
            # Work around, since we use a fake tes for AIRS IRK. We could do something
            # cleaner, but hopefully we'll be removing the uip stuff. So just have a
            # fix that works for now.
            uip_all = self.rf_uip.uip_all("AIRS")
        rad, jac, rad_units = self.fm_oss_stack(uip_all, sensor_index)
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

    def fm_oss(
        self, i_uip: dict[str, Any], i_jacobians: list[str], sensor_index: int
    ) -> dict[str, Any]:
        pres = (
            self.pressure.pressure_grid(rf.Pressure.DECREASING_PRESSURE)
            .convert("mbar")
            .value.value
        )
        tatm = (
            self.temperture.temperature_grid(
                self.pressure, rf.Pressure.DECREASING_PRESSURE
            )
            .convert("K")
            .value.value
        )
        emisv = self.emissivity.emissivity
        cloudextv = self.cloud_ext.cloud_ext
        # These aren't working yet. Need to work through how these get updated
        emisv = rf.ArrayAd_double_1(i_uip["emissivity"]["value"])
        cloudextv = rf.ArrayAd_double_1(i_uip["cloud"]["extinction"])
        # Set values to 1e-20 if NOT in uip.species.
        for jj in range(len(i_uip["atmosphere_params"])):
            search = i_uip["atmosphere_params"][jj]
            if search not in i_uip["species"]:
                i_uip["atmosphere"][jj, :] = 1e-20

        # check for negative values in PAN VMR.  If there are negative values:
        # 1) set VMR0 to original VMR, set VMR to 1e-11
        # 2) run OSS
        # 3) modify radiance by K ## (VMR0 - VMR)
        # 4) re-set VMR to VMR0

        pan_negative = False
        indpan = np.where(i_uip["atmosphere_params"] == "PAN")[0]
        if len(indpan) > 0:
            # assume there is only one parameter for PAN
            indpan = indpan[0]

            # Force negative PAN for testing. Leave commented out if not testing
            # i_uip['atmosphere'][indpan, 0] = -2.16e-09

            indneg = np.where(i_uip["atmosphere"][indpan, :] < 0)[0]
            if len(indneg) > 0:
                pan_vmr_muses = np.copy(i_uip["atmosphere"][indpan, :])
                pan_vmr_oss = np.copy(pan_vmr_muses)
                indneg = np.where(i_uip["atmosphere"][indpan, :] < 1e-11)[0]
                pan_vmr_oss[indneg] = 1e-11
                i_uip["atmosphere"][indpan] = pan_vmr_oss[:]
                pan_negative = True
            # end if len(indneg) > 0:
        # end if len(indpan) > 0:

        nh3_negative = False
        indnh3 = np.where(i_uip["atmosphere_params"] == "NH3")[0]
        if len(indnh3) > 0:
            # assume indnh3 is only one parameter for NH3
            indnh3 = indnh3[0]

            # Force negative PAN for testing. Leave commented out if not testing
            # i_uip['atmosphere'][indpan, 0] = -2.16e-09

            indneg = np.where(i_uip["atmosphere"][indnh3, :] < 0)[0]
            if len(indneg) > 0:
                nh3_vmr_muses = np.copy(i_uip["atmosphere"][indnh3, :])
                nh3_vmr_oss = np.copy(nh3_vmr_muses)
                indneg = np.where(i_uip["atmosphere"][indnh3, :] < 1e-11)[0]
                nh3_vmr_oss[indneg] = 1e-11
                i_uip["atmosphere"][indpan] = nh3_vmr_oss[:]
                nh3_negative = True
            # end if len(indneg) > 0:
        # end if len(indnh3) > 0:

        # index 1: pressure, index 2: temperature.  OSS puts those separately elsewhere.
        atmosphere = (i_uip["atmosphere"][2:, :]).T

        salt = self.surface_altitude.convert("m").value
        # TODO Not sure if the logic of this here, but this is what py-retrieve does
        if salt < 1e-5:
            salt = 1e-5
        # Use diffusion approximation, see oss_if_module.f90 in muses_oss
        lambertian_flag = 1
        # py-retrieve has sunang always 90. Not sure of the significance of that,
        # but do that for now
        sunang = 90.0
        # Make the call to the FORTRAN code passing in addresses of anything that are pointers.
        # The units of rad are "W  m^-2 sr^-1 cm^-1"
        (rad, drad_dtemp, drad_dtsur, xkOutGas, xkEm, xkRf, xkCldlnPres, xkCldlnExt) = (
            muses_oss_handle.oss_forward_model(
                self.tsur.surface_temperature(sensor_index).value.value,
                self.scale_cloud.scale_cloud(sensor_index).value,
                self.pcloud.pressure_cloud(sensor_index).convert("mbar").value.value,
                self.pointing_angle_surface.convert("deg").value,
                sunang,
                self.latitude.convert("deg").value,
                salt,
                lambertian_flag,
                pres,
                tatm,
                np.array(atmosphere),
                self.emissivity.emissivity_spectral_domain.convert_wave("cm^-1"),
                emisv.value,
                self.cloud_ext.cloud_ext_spectral_domain.convert_wave("cm^-1"),
                cloudextv.value,
            )
        )
        # This comes from the OSS documentation in muses_oss
        rad_in_units = rf.Unit("W / (m^2 sr cm^-1)")
        # Units that we want
        rad_units = rf.Unit("W / (cm^2 sr cm^-1)")
        rad_unit_f = rf.conversion(rad_in_units, rad_units)
        o_result = {
            "rad_units": rad_units,
            "radiance": rad * rad_unit_f,
            "drad_dtemp": drad_dtemp * rad_unit_f,
            "drad_dtsur": drad_dtsur * rad_unit_f,
            "xkOutGas": xkOutGas * rad_unit_f,
            "xkEm": xkEm * rad_unit_f,
            "xkRf": xkRf * rad_unit_f,
            "xkCldlnPres": xkCldlnPres * rad_unit_f,
            "xkCldlnExt": xkCldlnExt * rad_unit_f,
            "nameJacobian": i_jacobians,
        }

        # AT_LINE 134 src_ms-2018-12-10/fm_oss.pro
        # update naming to be consistent with ELANOR
        for jj in range(0, len(o_result["nameJacobian"])):
            if o_result["nameJacobian"][jj] == "F11":
                o_result["nameJacobian"][jj] = "CFC11"

            if o_result["nameJacobian"][jj] == "F12":
                o_result["nameJacobian"][jj] = "CFC12"

            if o_result["nameJacobian"][jj] == "C5H8":
                o_result["nameJacobian"][jj] = "ISOP"

            if o_result["nameJacobian"][jj] == "CHCLF2":
                o_result["nameJacobian"][jj] = "CFC22"
        # end for jj in range(0,len(o_result['nameJacobian')):

        if pan_negative:
            name_jacobians_stripped = np.char.strip(o_result["nameJacobian"])
            indjac = np.where(np.char.strip(name_jacobians_stripped) == "PAN")[0]
            indpan = np.where(i_uip["atmosphere_params"] == "PAN")[0]
            if len(indjac) > 0:
                indjac = indjac[0]
                indpan = indpan[0]
                k = np.copy(o_result["xkOutGas"][:, :, indjac])

                # make linear Jacobian
                for kk in range(len(pan_vmr_oss)):
                    k[kk, :] = k[kk, :] / pan_vmr_oss[kk]

                # modify radiance to ACTUAL VMR using K.dx
                # vmr0 used by OSS, vmr is what we want
                dL = k.T @ (pan_vmr_muses - pan_vmr_oss)

                i_uip["atmosphere"][indpan, :] = pan_vmr_muses
                o_result["radiance"] = o_result["radiance"] + dL

                # update "log" Jacobian to multiply by the MUSES VMR
                for kk in range(len(pan_vmr_oss)):
                    k[kk, :] = k[kk, :] * pan_vmr_muses[kk]

                # remake "log" Jacobian with muses VMR
                o_result["xkOutGas"][:, :, indjac] = k
            # if len(ind) > 0:
        # end if pan_negative:

        if nh3_negative:
            name_jacobians_stripped = np.char.strip(o_result["nameJacobian"])
            indjac = np.where(np.char.strip(name_jacobians_stripped) == "NH3")[0]
            indnh3 = np.where(i_uip["atmosphere_params"] == "NH3")[0]
            if len(indjac) > 0:
                indjac = indjac[0]
                indnh3 = indnh3[0]
                k = np.copy(o_result["xkOutGas"][:, :, indjac])

                # make linear Jacobian
                for kk in range(len(nh3_vmr_oss)):
                    k[kk, :] = k[kk, :] / nh3_vmr_oss[kk]

                # modify radiance to ACTUAL VMR using K.dx
                # vmr0 used by OSS, vmr is what we want
                dL = k.T @ (nh3_vmr_muses - nh3_vmr_oss)

                i_uip["atmosphere"][indnh3, :] = nh3_vmr_muses
                o_result["radiance"] = o_result["radiance"] + dL

                # update "log" Jacobian to multiply by the MUSES VMR
                for kk in range(len(nh3_vmr_oss)):
                    k[kk, :] = k[kk, :] * nh3_vmr_muses[kk]

                # remake "log" Jacobian with muses VMR
                o_result["xkOutGas"][:, :, indjac] = k
            # if len(indjac) > 0:
        # end if nh3_negative:

        # check finite
        if not np.all(np.isfinite(rad)):
            raise RuntimeError("Non-finite radiance")

        if not np.all(np.isfinite(o_result["xkOutGas"])):
            raise RuntimeError("Non-finite jacobians")

        return o_result

    def fm_oss_stack(
        self, uipIn: dict[str, Any], sensor_index: int
    ) -> tuple[np.ndarray, np.ndarray, rf.Unit]:
        # AT_LINE 5 fm_oss_stack.pro
        uip = uipIn

        jacobianList = [str(s) for s in muses_oss_handle.jac_spec]

        results = self.fm_oss(uip, jacobianList, sensor_index)

        # Prepare Jacobians for pack_jacobian function.
        uip["num_atm_k"] = len(results["nameJacobian"])

        species_list = []
        k_species = []

        # AT_LINE 16 fm_oss_stack.pro
        if uip["num_atm_k"] > 0:
            # Create a list of uip['num_atm_k']+numTatm dictionaries with the format of k_struct.
            for x in range(uip["num_atm_k"]):
                k_struct = {
                    "species": results["nameJacobian"][0],
                    "k": np.zeros(
                        shape=results["drad_dtemp"].shape,
                        dtype=np.float32,
                    ),
                }
                k_species.append(k_struct)

            atm_jacobians_ils_total = {"k_species": k_species}

            # AT_LINE 22 fm_oss_stack.pro
            for jj in range(uip["num_atm_k"]):
                species_list.append(results["nameJacobian"][jj].lstrip().rstrip())
                atm_jacobians_ils_total["k_species"][jj]["species"] = (
                    results["nameJacobian"][jj].lstrip().rstrip()
                )

                atm_jacobians_ils_total["k_species"][jj]["k"] = results["xkOutGas"][
                    :, :, jj
                ]

                # AT_LINE 28 fm_oss_stack.pro
                # update naming to be consistent with ELANOR
                if atm_jacobians_ils_total["k_species"][jj]["species"] == "F11":
                    atm_jacobians_ils_total["k_species"][jj]["species"] = "CFC11"

                if atm_jacobians_ils_total["k_species"][jj]["species"] == "F12":
                    atm_jacobians_ils_total["k_species"][jj]["species"] = "CFC12"

                if atm_jacobians_ils_total["k_species"][jj]["species"] == "C5H8":
                    atm_jacobians_ils_total["k_species"][jj]["species"] = "ISOP"

                if atm_jacobians_ils_total["k_species"][jj]["species"] == "CHCLF2":
                    atm_jacobians_ils_total["k_species"][jj]["species"] = "CFC22"
            # end for jj in range(uip['num_atm_k']):

        # AT_LINE 36 fm_oss_stack.pro
        # Make cloud Jac structure.
        jacobian_cloud_map = {
            "k_height": results["xkCldlnPres"],
            "k_ext": results["xkCldlnExt"],
        }

        # Make emissivity Jac structure.
        jacobian_emiss_ils_map = {"k": results["xkEm"]}

        # AT_LINE 41 fm_oss_stack.pro
        # Pack jacobians together based on retrieval parameter ordering.
        # pack_jacobian, uip, rad
        o_jac = self.pack_jacobian(
            uip,
            results["radiance"].shape[0],
            jacobian_emiss_ils_map,
            atm_jacobians_ils_total,
            jacobian_cloud_map,
        )
        try:
            sub_basis_matrix = self.rf_uip.instrument_sub_basis_matrix(
                self.instrument_name
            )
        except KeyError:
            sub_basis_matrix = self.rf_uip.instrument_sub_basis_matrix("AIRS")
        if (
            o_jac is not None
            and sub_basis_matrix.shape[0] > 0
            and o_jac.ndim > 0
            and len(self.retrieval_state_element_id) > 0
        ):
            o_jac = np.matmul(sub_basis_matrix, o_jac).transpose()
        else:
            o_jac = None
        if o_jac is not None:
            tsurv = self.tsur.surface_temperature(sensor_index).value
            if not tsurv.is_constant:
                o_jac += (
                    results["drad_dtsur"][:, np.newaxis] @ tsurv.gradient[np.newaxis, :]
                )
            temp = (
                self.temperture.temperature_grid(
                    self.pressure, rf.Pressure.DECREASING_PRESSURE
                )
                .convert("K")
                .value
            )
            if not temp.is_constant:
                o_jac += results["drad_dtemp"].T @ temp.jacobian
        return (results["radiance"], o_jac, results["rad_units"])

    def pack_jacobian(
        self,
        uip,
        num_rad,
        jacobian_emiss_ils_map,
        atm_jacobians_ils_total,
        jacobian_cloud_map,
    ):
        from refractor.muses_py import UtilList

        utilList = UtilList()

        o_jacobian = None

        # convert to linear Jacobians in this function rather than within
        # ELANOR because affects OSS also

        # any linear atmospheric Jacobians, convert now
        if len(uip["jacobiansLinear"]) > 0 and uip["jacobiansLinear"][0] != "":
            all_jacobians_species = []
            for kkk in range(0, len(atm_jacobians_ils_total["k_species"])):
                all_jacobians_species.append(
                    atm_jacobians_ils_total["k_species"][kkk]["species"]
                )

            for jj in range(0, len(uip["jacobiansLinear"])):
                specie = uip["jacobiansLinear"][jj]

                if not np.all(
                    np.isfinite(atm_jacobians_ils_total["k_species"][0]["k"])
                ):
                    raise RuntimeError("Non-finite Jacobian")

                # if uip['jacobiansLinear'][jj] in all_jacobians_species:
                if (
                    uip["jacobiansLinear"][jj] in all_jacobians_species
                    and uip["jacobiansLinear"][jj] != "TATM"
                ):
                    # Note: Not sure if below lines are correct.
                    inds = utilList.WhereEqualIndices(
                        uip["atmosphere_params"], uip["jacobiansLinear"][jj]
                    )

                    val = uip["atmosphere"][inds, :]
                    if len(val.shape) == 2 and val.shape[0] == 1:
                        val = np.reshape(val, (val.shape[1]))  # Convert (1,64) to (64,)

                    # IDL:
                    # FOR kk = 0, N_ELEMENTS(val)-1 DO BEGIN
                    #     atm_jacobians_ils_total[0].k_species[ind].k(kk,*) = atm_jacobians_ils_total[0].k_species[ind].k(kk,*)/val[kk]
                    # ENDFOR

                    # fixed bug here:  should not be ['k_species'][0]['k'][kk, :]
                    # should search for correct index, found
                    # should be ['k_species'][found]['k'][kk, :] ssk 3/4/2023
                    for ii in range(len(atm_jacobians_ils_total["k_species"])):
                        if (
                            atm_jacobians_ils_total["k_species"][ii]["species"]
                            == specie
                        ):
                            found = ii

                    for kk in range(0, len(val)):
                        atm_jacobians_ils_total["k_species"][found]["k"][kk, :] = (
                            atm_jacobians_ils_total["k_species"][found]["k"][kk, :]
                            / val[kk]
                        )
                    # end for kk in range(0,len(val)):

                    if not np.all(
                        np.isfinite(atm_jacobians_ils_total["k_species"][0]["k"])
                    ):
                        raise RuntimeError("Non-finite Jacobian")
            # end for jj in range(0,len(uip['jacobiansLinear'])):

        # AT_LINE 12 ELANOR/pack_jacobian.pro pack_jacobian
        num_par = 0
        num_atm = len(uip["atmosphere"][0, :])

        num_det = 1

        # AT_LINE 24 ELANOR/pack_jacobian.pro pack_jacobian
        for ii in range(len(uip["jacobians"])):
            jacob = uip["jacobians"][ii].upper()

            if jacob == "TSUR" or jacob == "PTGANG":
                num_par = num_par + 1

            if jacob == "EMIS" or jacob == "EMIS_LOG":
                num_par = num_par + len(uip["emissivity"]["value"])

            if jacob == "CLOUDEXT":
                num_par = num_par + len(uip["cloud"]["frequency"])

            if jacob == "PCLOUD":
                num_par = num_par + 1

            if jacob == "TATM":
                num_par = num_par + num_atm

            if jacob == "RESSCALE":
                num_par = num_par + 1

            if jacob == "CALSCALE":
                num_par = num_par + len(uip["calibration"]["frequency"])

            if jacob == "CALOFFSET":
                num_par = num_par + len(uip["calibration"]["frequency"])

            # Have to add in atm jac for oss.  check
            if len(atm_jacobians_ils_total) > 0:
                # Collect all species in all atm_jacobians_ils_total
                all_species = []
                for jj in range(len(atm_jacobians_ils_total["k_species"])):
                    species = atm_jacobians_ils_total["k_species"][jj]["species"]
                    if species not in all_species:
                        all_species.append(species)

                if jacob in all_species:
                    num_par = num_par + num_atm
        # for ii in range(len(uip['jacobians'])):

        # AT_LINE 57 ELANOR/pack_jacobian.pro pack_jacobian
        # Now unpack jacobians
        if num_par > 0:
            o_jacobian = np.zeros(
                shape=(num_par, num_rad), dtype=np.float64
            )  # output jacobian

            ii_par = 0
            ii_dets = 0  # Start index for copying jacobina ils total.
            ii_dete = num_rad  # End index for copying jacobian ils total.

            for ii in range(len(uip["jacobians"])):
                ii_dets = 0
                ii_dete = num_rad  # PYTHON_NOTE: Because the slices in Python does not include num_rad, we don't have to subtract 1.
                jacob = uip["jacobians"][ii]
                if jacob == "TSUR":
                    ii_par = ii_par + 1
                if jacob == "TATM":
                    ii_par = ii_par + num_atm

                # AT_LINE 79 ELANOR/pack_jacobian.pro pack_jacobian
                ii_dets = 0
                ii_dete = num_rad  # PYTHON_NOTE: Because the slices in Python does not include num_rad, we don't have to subtract 1.

                # AT_LINE 90 ELANOR/pack_jacobian.pro pack_jacobian
                ii_dets = 0
                ii_dete = num_rad
                if jacob == "EMIS" or jacob == "EMIS_LOG":
                    num_em = len(uip["emissivity"]["value"])
                    ii_ps = ii_par
                    ii_pe = ii_par + num_em

                    for kk in range(num_det):
                        # Note: There is only one element for jacobian_emiss_ils_map so there's no need to use index.
                        o_jacobian[ii_ps:ii_pe, ii_dets:ii_dete] = (
                            jacobian_emiss_ils_map["k"][:]
                        )
                        ii_dets = ii_dets + num_rad
                        ii_dete = ii_dete + num_rad
                    ii_par = ii_par + num_em
                # end if (jacob == 'EMIS' or jacob == 'EMIS_LOG'):

                # AT_LINE 109 ELANOR/pack_jacobian.pro pack_jacobian
                ii_dets = 0
                ii_dete = num_rad

                # AT_LINE 127 ELANOR/pack_jacobian.pro pack_jacobian
                ii_dets = 0
                ii_dete = num_rad

                # AT_LINE 145 ELANOR/pack_jacobian.pro pack_jacobian
                ii_dets = 0
                ii_dete = num_rad

                # AT_LINE 160 ELANOR/pack_jacobian.pro pack_jacobian
                ii_dets = 0
                ii_dete = num_rad
                if jacob == "PCLOUD":
                    if num_det == 1:
                        # If we only processing 1 detector, there is only one element in jacobian_cloud_map.
                        o_jacobian[ii_par, ii_dets:ii_dete] = jacobian_cloud_map[
                            "k_height"
                        ][:]
                    else:
                        for jj in range(num_det):
                            o_jacobian[ii_par, ii_dets:ii_dete] = jacobian_cloud_map[
                                jj
                            ]["k_height"][:]
                            ii_dets = ii_dets + num_rad
                            ii_dete = ii_dete + num_rad
                        # end for jj in range(num_det):
                    ii_par = ii_par + 1
                # end if (jacob == 'PCLOUD'):

                # AT_LINE 175 ELANOR/pack_jacobian.pro pack_jacobian

                ii_dets = 0
                ii_dete = num_rad
                if jacob == "CLOUDEXT":
                    num_v = len(uip["cloud"]["frequency"])
                    ii_ps = ii_par
                    ii_pe = ii_par + num_v
                    ii_dets = 0
                    ii_dete = num_rad
                    if num_det == 1:
                        # If we only processing 1 detector, there is only one element in jacobian_cloud_map.
                        o_jacobian[ii_ps:ii_pe, ii_dets:ii_dete] = jacobian_cloud_map[
                            "k_ext"
                        ][:]
                    else:
                        for kk in range(num_det):
                            o_jacobian[ii_ps:ii_pe, ii_dets:ii_dete] = (
                                jacobian_cloud_map[kk]["k_ext"][:]
                            )
                            ii_dets = ii_dets + num_rad
                            ii_dete = ii_dete + num_rad
                        # end for kk in range(num_det):
                    ii_par = ii_par + num_v
                # end if (jacob == 'CLOUDEXT'):

                # AT_LINE 193 ELANOR/pack_jacobian.pro pack_jacobian
                ii_dets = 0
                ii_dete = num_rad  # PYTHON_NOTE: Because the slices in Python does not include num_rad, we don't have to subtract 1.
                uu = []
                if uip["num_atm_k"] > 0:
                    for mm in range(len(atm_jacobians_ils_total["k_species"])):
                        if atm_jacobians_ils_total["k_species"][mm]["species"] == jacob:
                            uu.append(mm)

                if len(uu) > 0:
                    uu = uu[0]  # We only need one.
                    ii_ps = ii_par
                    ii_pe = (
                        ii_par + num_atm
                    )  # PYTHON_NOTE: We don't need to subtract 1.
                    if num_det == 1:
                        # If we only processing 1 detector, there is only one element in atm_jacobians_ils_total.
                        o_jacobian[ii_ps:ii_pe, ii_dets:ii_dete] = (
                            atm_jacobians_ils_total["k_species"][uu]["k"][:]
                        )
                    else:
                        # AT_LINE 203 ELANOR/pack_jacobian.pro pack_jacobian
                        for kk in range(num_det):
                            o_jacobian[ii_ps:ii_pe, ii_dets:ii_dete] = (
                                atm_jacobians_ils_total[kk]["k_species"][uu]["k"]
                            )
                            ii_dets = ii_dets + num_rad
                            ii_dete = ii_dete + num_rad
                        # for kk in range(num_det):
                    # AT_LINE 209 ELANOR/pack_jacobian.pro pack_jacobian
                    ii_par = ii_par + num_atm
                # end if jacob in atm_jacobians_ils_total['k_species']['species']:
            # end for ii in range(len(uip['jacobians'])):
        # end if (num_par > 0):
        # AT_LINE 213 ELANOR/pack_jacobian.pro pack_jacobian

        return o_jacobian


__all__ = [
    "MusesRadiativeTransferOss",
]

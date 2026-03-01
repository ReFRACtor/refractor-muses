from __future__ import annotations
import refractor.framework as rf  # type: ignore
from .muses_oss_handle import muses_oss_handle
import numpy as np
import os
import typing
from typing import Self
from loguru import logger

if typing.TYPE_CHECKING:
    from .identifier import StateElementIdentifier, InstrumentIdentifier
    from .input_file_helper import InputFilePath, InputFileHelper


class MusesRadiativeTransferOss(rf.RadiativeTransferImpBase):
    """This uses the muses OSS code (package muses-oss). This gives a forward
    model that is the same as the py-retrieve airs/cris/tes forward model (with minor
    differences in calculation - the normal sort of round off differences).
    """

    def __init__(
        self,
        rf_uip: rf.RefractorUip,  # Temp, leverage off UIP. We'll remove this in a bit
        tsur: rf.SurfaceTemperature,
        pcloud: rf.Pcloud,
        scale_cloud: rf.ScaleCloud,
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
        self.tsur = tsur
        self.pcloud = pcloud
        self.scale_cloud = scale_cloud
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
        return MusesRadiativeTransferOss(
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

    def reflectance(
        self, sd: rf.SpectralDomain, sensor_index: int, skip_jacobian: bool
    ) -> rf.Spectrum:
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
        uip_all = self.rf_uip.uip_all(self.instrument_name)
        uip_all["oss_jacobianList"] = [str(s) for s in muses_oss_handle.jac_spec]
        uip_all["oss_frequencyList"] = list(sd.convert_wave("nm"))
        rad, jac = self.fm_oss_stack(uip_all, sensor_index)
        if jac is not None:
            a = rf.ArrayAd_double_1(rad, jac)
        else:
            a = rf.ArrayAd_double_1(rad)
        sr = rf.SpectralRange(a, rf.Unit("sr^-1"))
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

    def fm_oss(self, i_uip, i_jacobians, sensor_index: int):
        import math

        function_name = "fm_oss: "

        if i_jacobians is not None:
            njacob = len(i_jacobians)
        else:
            njacob = 0

        pressure = np.ndarray(shape=(i_uip["atmosphere"].shape[1]), dtype=np.float32)
        tatm = np.ndarray(shape=(i_uip["atmosphere"].shape[1]), dtype=np.float32)
        pressure[:] = i_uip["atmosphere"][0, :]
        tatm[:] = i_uip["atmosphere"][1, :]

        nchanOSS = len(i_uip["oss_frequencyList"])
        sunang = 90.0
        nemis = len(i_uip["emissivity"]["frequency"])
        ncloud = len(i_uip["cloud"]["frequency"])
        nlevels = len(pressure)

        if float(i_uip["obs_table"]["pointing_angle_surface"] * 180 / np.pi) < -990:
            print(
                function_name,
                "Error! Need to define uip.obs_table.pointing_angle_surface (radians)",
            )
            assert False

        # Set values to 1e-20 if NOT in uip.species.
        # Check for both b'CO2' and 'CO2'.  b'CO2' exists when running fm_oss from a netcdf uip file.
        ns = len(i_uip["atmosphere_params"])
        for jj in range(ns):
            search = i_uip["atmosphere_params"][jj]
            if search not in i_uip["species"]:
                i_uip["atmosphere"][jj, :] = 1e-20
        # end for jj in range(ns):

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

        # AT_LINE 72 fm_oss.pro
        # index 1: pressure, index 2: temperature.  OSS puts those separately elsewhere.
        indspecies = [(jj + 2) for jj in range(ns - 2)]

        # AT_LINE 74 fm_oss.pro
        natmosphere_params = len(indspecies)

        surfaceAltitude = i_uip["obs_table"]["surfaceAltitude"]
        if abs(surfaceAltitude) < 1e-5:
            surfaceAltitude = 1e-5

        # AT_LINE 69 fm_oss.pro
        atmosphere = (i_uip["atmosphere"][np.asarray(indspecies), :]).T
        atmosphere = atmosphere.astype(np.float32)

        # AT_LINE 71 fm_oss.pro
        ss_info = {
            "nlevels": nlevels,
            "natmosphere_params": natmosphere_params,
            "pressure": pressure,
            "tatm": tatm,
            "tsur": i_uip["surface_temperature"],
            "atmosphere": atmosphere,
            "nemis": nemis,
            "emis": (i_uip["emissivity"]["value"]).astype(np.float32),
            "refl": (1 - i_uip["emissivity"]["value"]).astype(np.float32),
            "scale_cloud": i_uip["cloud"]["scale_pressure"],
            "pcloud": i_uip["cloud"]["pressure"],
            "ncloud": ncloud,
            "cloudext": (i_uip["cloud"]["extinction"]).astype(np.float32),
            "emis_freq": (i_uip["emissivity"]["frequency"]).astype(np.float32),
            "cloud_freq": (i_uip["cloud"]["frequency"]).astype(np.float32),
            "ptgang": i_uip["obs_table"]["pointing_angle_surface"] * 180 / math.pi,
            "sunang": sunang,
            "latitude": i_uip["obs_table"]["target_latitude"] * 180 / math.pi,
            "surfaceAltitude": surfaceAltitude,
            "lambertian_flag": 1,
            "njacobians": njacob,
            "nchanOSS": nchanOSS,
        }

        # Make the call to the FORTRAN code passing in addresses of anything that are pointers.
        (y, xkTemp, dy_dtsur, xkOutGas, xkEm, xkRf, xkCldlnPres, xkCldlnExt) = (
            muses_oss_handle.oss_forward_model(
                self.tsur.surface_temperature(sensor_index).value.value,
                self.scale_cloud.scale_cloud(sensor_index).value,
                self.pcloud.pressure_cloud(sensor_index).convert("mbar").value.value,
                ss_info["ptgang"],
                ss_info["sunang"],
                ss_info["latitude"],
                surfaceAltitude,
                ss_info["lambertian_flag"],
                np.array(pressure),
                np.array(tatm),
                np.array(atmosphere),
                np.array(ss_info["emis_freq"]),
                np.array(ss_info["emis"]),
                np.array(ss_info["cloud_freq"]),
                np.array(ss_info["cloudext"]),
            )
        )
        o_result = {
            "radiance": y * 1e-4,
            "xkTemp": xkTemp * 1e-4,
            "drad_dtsur": dy_dtsur * 1e-4,
            "xkOutGas": xkOutGas * 1e-4,
            "xkEm": xkEm * 1e-4,
            "xkRf": xkRf * 1e-4,
            "xkCldlnPres": xkCldlnPres * 1e-4,
            "xkCldlnExt": xkCldlnExt * 1e-4,
            "pressure": pressure,
            "nameJacobian": i_jacobians,
            "frequency": i_uip[
                "oss_frequencyList"
            ],  # vp changed from frequencylist to oss_frequencylist
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
        if not np.all(np.isfinite(y)):
            print(function_name, "ERROR: Non-finite radiance!")
            assert False

        if not np.all(np.isfinite(o_result["xkOutGas"])):
            print(function_name, "ERROR: Non-finite jacobians!")
            assert False

        return o_result

    def fm_oss_stack(self, uipIn, sensor_index: int):
        # AT_LINE 5 fm_oss_stack.pro
        uip = uipIn

        jacobianList = uip["oss_jacobianList"]

        results = self.fm_oss(uip, jacobianList, sensor_index)

        # Prepare Jacobians for pack_jacobian function.
        uip["num_atm_k"] = len(results["nameJacobian"])

        numTatm = 0
        if "TATM" in uip["jacobians"]:
            numTatm = 1

        species_list = []
        k_species = []

        # AT_LINE 16 fm_oss_stack.pro
        if uip["num_atm_k"] > 0:
            # Create a list of uip['num_atm_k']+numTatm dictionaries with the format of k_struct.
            for x in range(uip["num_atm_k"] + numTatm):
                k_struct = {
                    "species": results["nameJacobian"][0],
                    "k": np.zeros(
                        shape=(len(results["pressure"]), len(results["frequency"])),
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

        # AT_LINE 30 fm_oss_stack.pro
        # Add tatm if present into atmospheric Jacobians.
        if "TATM" in uip["jacobians"]:
            # Use the last value of jj plus 1 from the for loop above.  We have to add 1 in Python to not overwrite the last set values.
            atm_jacobians_ils_total["k_species"][jj + 1]["species"] = "TATM"
            atm_jacobians_ils_total["k_species"][jj + 1]["k"] = results["xkTemp"]
            species_list.append("TATM")

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
        sub_basis_matrix = self.rf_uip.instrument_sub_basis_matrix(self.instrument_name)
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
        return (results["radiance"], o_jac)

    def pack_jacobian(
        self,
        uip,
        num_rad,
        jacobian_emiss_ils_map,
        atm_jacobians_ils_total,
        jacobian_cloud_map,
    ):
        from refractor.muses_py import UtilList

        function_name = "pack_jacobian: "

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

                if (
                    np.all(np.isfinite(atm_jacobians_ils_total["k_species"][0]["k"]))
                    == False
                ):
                    print(function_name, "Error! Non-finite Jacobian")
                    print(
                        function_name,
                        f"jj={jj}, uip['jacobiansLinear'][jj]={uip['jacobiansLinear'][jj]}",
                    )
                    assert False

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

                    if (
                        np.all(
                            np.isfinite(atm_jacobians_ils_total["k_species"][0]["k"])
                        )
                        == False
                    ):
                        print(function_name, "Error! Non-finite Jacobian")
                        print(
                            function_name,
                            f"jj={jj}, uip['jacobiansLinear'][jj]={uip['jacobiansLinear'][jj]}",
                        )
                        assert False
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

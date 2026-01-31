from __future__ import annotations
import refractor.framework as rf  # type: ignore
from .identifier import InstrumentIdentifier
from .forward_model_handle import ForwardModelHandle, ForwardModelHandleSet
from functools import cached_property
from loguru import logger
import tempfile
import numpy as np
import copy
import os
from pathlib import Path
from typing import Any, TypeVar
import typing

if typing.TYPE_CHECKING:
    from .current_state import CurrentState
    from .muses_observation import MeasurementId
    from .retrieval_configuration import RetrievalConfiguration
    from .muses_observation import MusesObservation
    from .cost_function import CostFunction
    from refractor.muses_py_fm import RefractorUip

# This is a work in progress. We would like to move over and simplify the vlidort
# forward model, and hopefully remove using the UIP etc. But for right now, we
# leverage off of muses-py
#
# Note that this has direct copied of stuff from muses_py_fm/muses_forward_model.py,
# since we want to independent update stuff. This is obviously not desirable long
# term.


class FmUpdateUip(rf.ObserverMaxAPosterioriSqrtConstraint):
    def __init__(self, fm: MusesForwardModelVlidortBase) -> None:
        super().__init__()
        self.fm = fm

    def notify_update(self, mstand: rf.MaxAPosterioriSqrtConstraint) -> None:
        logger.debug(f"Call to {self.__class__.__name__}::notify_update")
        self.fm.update_uip(mstand.parameters)


# Adapter to make muses-py forward model calls look like a ReFRACtor
# ForwardModel

# There are a number of things in common with the different forward models,
# so we capture these in these base classes.


class MusesForwardModelVlidortBase(rf.ForwardModel):
    def __init__(
        self,
        current_state: CurrentState,
        instrument_name: InstrumentIdentifier,
        obs: MusesObservation,
        rconf: RetrievalConfiguration,
        vlidort_nstokes: int = 2,
        vlidort_nstreams: int = 4,
        use_vlidort_temp_dir: bool = True,
        **kwargs: Any,
    ) -> None:
        """vlidort_tempdir can be passed in. This should be the same as what
        was used in RefractorUip when we pass in the vlidort_dir. We don't
        actually do anything with vlidort_tempdir, just maintain the lifetime so
        that as long as this MusesForwardModel exists we still have the tempdir.
        When the forward model gets deleted, the temporary directory gets removed.

        Note the directory is under 1MB usually, so you don't need to be too concerned
        about where this goes. You can just use the normal mkdtemp() logic used
        by tempfile.TemporaryDirectory.
        """
        super().__init__()
        self.instrument_name = instrument_name
        self.vlidort_nstreams = vlidort_nstreams
        self.vlidort_nstokes = vlidort_nstokes

        # We save the current_state value, since it might have changed
        # when we create the UIP. The semantics here is that we create
        # the UIP when we create the forward model, however we actually
        # delay that until we create it on first use. However we want to
        # create the UIP that we *would have* if we had created it now.
        #
        # Note for an actual retrieval, there is no reason to delay creating
        # the UIP now. Instead, we have unit tests that regularly set things
        # up but don't actually run the forward model. We don't want to pay
        # the time penalty of creating the UIP and/or require muses-py be
        # available. So to support that, we have a delayed create on first
        # use of the UIP.
        self.current_state = copy.deepcopy(current_state)
        self.obs = obs
        self.kwargs = kwargs
        self.rconf = rconf
        self.vlidort_tempdir: tempfile.TemporaryDirectory | None = None
        self.use_vlidort_temp_dir = use_vlidort_temp_dir
        self.have_fake_jac_in_oss = False
        self.have_create_uip = False
        self.uip_params: None | np.ndarray = None

    def update_uip(self, parameters: np.ndarray) -> None:
        if not self.have_create_uip:
            # Delay setting the UIP value until we actually create it. We don't
            # want to create this now just to set the value
            self.uip_params = parameters.copy()
        else:
            if self.rf_uip.basis_matrix is not None:
                self.rf_uip.update_uip(parameters)

    @cached_property
    def rf_uip(self) -> RefractorUip:
        """Create on on first use."""
        from refractor.muses_py_fm import RefractorUip

        self.vlidort_tempdir = None
        if self.use_vlidort_temp_dir:
            self.vlidort_tempdir = tempfile.TemporaryDirectory()
        res = RefractorUip.create_uip_from_refractor_objects(
            [
                self.obs,
            ],
            self.current_state,
            self.rconf,
            vlidort_dir=self.vlidort_tempdir.name
            if self.vlidort_tempdir is not None
            else None,
        )
        # There is special handling for an empty set of retrieval element (which we
        # run into in the BT step). It turns out the OSS code doesn't handle an empty
        # set of jacobians, it requires at least one. py-retrieve just adds a H2O
        # jacobian so there is something to calculate. However, we shouldn't actually
        # return that. So look for this condition and mark it, we'll then handle this
        # in the radiance call.

        uip_all = res.uip_all(str(self.instrument_name))
        if (
            uip_all["rts"] == ["OSS"]
            and "H2O" in [str(i) for i in uip_all["jacobians"]]
            and "H2O" not in uip_all["jacobians_all"]
        ):
            self.have_fake_jac_in_oss = True
        else:
            self.have_fake_jac_in_oss = False
        self.have_create_uip = True
        # Set any delayed parameters update
        if self.uip_params is not None and res.basis_matrix is not None:
            res.update_uip(self.uip_params)
        return res

    def bad_sample_mask(self, sensor_index: int) -> np.ndarray:
        bmask = self.obs.bad_sample_mask(sensor_index)
        if self.obs.spectral_window.include_bad_sample:
            bmask[:] = False
        # This is the full bad sample mask, for all the indices. But here we only
        # want the portion that fits in the spectral window
        with self.obs.modify_spectral_window(include_bad_sample=True):
            sd = self.obs.spectral_domain_full(sensor_index)
            gindex = self.obs.spectral_window.grid_indexes(sd, sensor_index)
        return bmask[list(gindex)]

    def setup_grid(self) -> None:
        # Nothing that we need to do for this
        pass

    def _v_num_channels(self) -> int:
        return 1

    def spectral_domain(self, sensor_index: int) -> rf.SpectralDomain:
        if sensor_index > 0:
            raise RuntimeError("sensor_index out of range")
        sd = np.concatenate(
            [self.obs.spectral_domain(i).data for i in range(self.obs.num_channels)]
        )
        return rf.SpectralDomain(sd, rf.Unit("nm"))

    def notify_cost_function(self, cfunc: CostFunction) -> None:
        # Attach to CostFunction, so uip gets updated when the parameter change
        #
        # Note, we can't just attach to the fm_sv when we create the MusesForwardModel.
        # The UIP takes in parameters on the RetrievalGridArray, *not*
        # FullGridMappedArray like the Refractor
        #
        # A note on the lifetime here. For the CostFunction, if we
        # have a UIP than the UIP state observer is *required*. If we pickle
        # and reload the cost function, it should have the UIP and observer.
        # So we use "add_observer_and_keep_reference".
        #
        # This is in contrast to the StateElement observers. The
        # StateElements are outside of the CostFunction. You can
        # have a CostFunction without any StateElements, if we pickle and reload
        # we don't want to pull all the StateElements along. So for this
        # we use "add_observer" which uses weak pointers - we notify if the
        # object is still there but don't carry around it lifetime and if the
        # object is deleted then we just don't notify it.
        cfunc.max_a_posteriori.add_observer_and_keep_reference(FmUpdateUip(self))

    def summarize_mw(self, i_uip: dict[str, Any]) -> dict[str, Any]:
        num_mw = len(i_uip["microwindows"])
        mws = 0
        nfreq_tot = 0
        mw_range = np.ndarray(
            shape=(3, num_mw), dtype=np.int32
        )  # PYTHON_NOTE: mw_range must be integer so we can use it later as indices.

        for ii_mw in range(0, num_mw):
            nfreq_mw = (
                i_uip["microwindows"][ii_mw]["enddmw"][ii_mw]
                - i_uip["microwindows"][ii_mw]["startmw"][ii_mw]
                + 1
            )
            nfreq_tot = nfreq_tot + nfreq_mw
            mwf = mws + nfreq_mw - 1

            mw_range[0, ii_mw] = mws
            mw_range[1, ii_mw] = mwf
            mw_range[2, ii_mw] = nfreq_mw

            mws = mws + nfreq_mw

        o_mw_account = {
            "mw_cnt": num_mw,
            "mw_species": np.asarray([0 for ii in range(0, len(i_uip["species"]))]),
            "freq": i_uip["frequencyList"],
            "mw_range": mw_range,
            "nfreq_tot": nfreq_tot,
        }
        return o_mw_account

    def radiance(self, sensor_index: int, skip_jacobian: bool = False) -> rf.Spectrum:
        if sensor_index != 0:
            raise ValueError("sensor_index must be 0")
        jac, rad = self.fm_call()
        # Haven't filled everything in yet, but mark as cache full.
        # otherwise bad_sample_mask and spectral_domain will enter an
        # infinite loop
        self.cache_valid_flag = True
        gmask = np.concatenate(
            [self.bad_sample_mask(i) != True for i in range(self.obs.num_channels)]
        )
        sd = self.spectral_domain(0)
        # jacobian is 1) on the forward model grid and
        # 2) transposed from the ReFRACtor convention of the
        # column being the state vector variables. So
        # translate the oss jac to what we want from ReFRACtor
        # The logic in pack_omi_jacobian and pack_tropomi_jacobian
        # over counts the size of atmosphere jacobians by 1 for each
        # species. This is harmless,
        # it gives an extra row of zeros that then gets trimmed before leaving
        # fm_wrapper. But because we are calling the lower level function
        # ourselves we need to trim this.
        sub_basis_matrix = self.rf_uip.instrument_sub_basis_matrix(self.instrument_name)
        if jac is not None and jac.shape[0] > 0 and sub_basis_matrix.shape[1] > 0:
            jac = np.matmul(
                sub_basis_matrix, jac[: sub_basis_matrix.shape[1], :]
            ).transpose()
            a = rf.ArrayAd_double_1(rad[gmask], jac[gmask, :])
        else:
            a = rf.ArrayAd_double_1(rad[gmask])
        sr = rf.SpectralRange(a, rf.Unit("sr^-1"))
        return rf.Spectrum(sd, sr)

    def fm_call2(self, i_uip, is_tropomi: bool,
                 i_osp_dir=None, i_obs=None, skip_raman_copy=False):
        # Temp, we'll pull some of this over and get other parts into mpy
        from refractor.muses_py import (
            pack_omi_jacobian,
            rev_and_fm_map,
            rtf_omi,
            get_omi_radiance,
            pack_tropomi_jacobian,
            rtf_tropomi,
            get_tropomi_radiance,
            raylayer_nadir,
            atmosphere_level,
            get_tropomi_o3xsec,
            get_tropomi_ils,
            tropomi_rev_and_fm_map,
        )

        from refractor.muses import AttrDictAdapter

        if is_tropomi:
            uip_tropomi = i_uip["uip_TROPOMI"]

            # VLIDORT I/O
            vlidort_input_dir = uip_tropomi["vlidort_input"]
            Path(vlidort_input_dir).mkdir(parents=True, exist_ok=True)

        # Get atmospheric parameters
        i_uip["obs_table"]["pointing_angle"] = 0.0
        atmparams = atmosphere_level(i_uip)

        # Computer layer and level quanity
        rayInfo = raylayer_nadir(AttrDictAdapter(i_uip), AttrDictAdapter(atmparams))

        mw_account = self.summarize_mw(i_uip)
        nfreq_tot = mw_account["nfreq_tot"]

        i_uip["num_atm_k"] = sum(
            jac in i_uip["species"] or jac == "TATM" for jac in i_uip["jacobians"]
        )

        # Create radiance and ring arrays
        radiance_ils = np.zeros(shape=(nfreq_tot), dtype=np.float64)
        radiance_clear_ils = np.zeros(shape=(nfreq_tot), dtype=np.float64)
        radiance_cloud_ils = np.zeros(shape=(nfreq_tot), dtype=np.float64)
        radiance_matrix_temperature_clear = np.zeros(
            shape=(nfreq_tot), dtype=np.float64
        )
        radiance_matrix_temperature_cloudy = np.zeros(
            shape=(nfreq_tot), dtype=np.float64
        )
        radiance_temperature_ils = np.zeros(shape=(nfreq_tot), dtype=np.float64)

        ring_clear_ils = np.zeros(shape=(nfreq_tot), dtype=np.float64)
        ring_cloud_ils = np.zeros(shape=(nfreq_tot), dtype=np.float64)
        ring_clear_ils_temperature = np.zeros(shape=(nfreq_tot), dtype=np.float64)
        ring_cloud_ils_temperature = np.zeros(shape=(nfreq_tot), dtype=np.float64)

        # Create atmospheric jacobians
        jacobians_atm_ils = None
        if i_uip["num_atm_k"] > 0:
            k_temp = []

            for kk in range(i_uip["num_atm_k"]):
                k_temp.append(
                    {
                        "species": "thisisadummystring",
                        "k": np.zeros(
                            shape=(atmparams["nlayers"] + 1, nfreq_tot),
                            dtype=np.float64,
                        ),
                    }
                )

            jacobians_atm_ils = {"k_species": k_temp}
        # end if (i_uip['num_atm_k'] > 0):

        # Create arrays for the following jacobians:
        # [Cloud Fraction,Ring Scaling Factor,Earth/Solar Wavelength Shift Parameter,od Wavelength Shift Parameter,od Wavelength Shift Parameter]

        # EM NOTE - Here we are creating empty jacobian arrays for
        # use, the original OMI code hardcoded a series of parameters,
        # but for tropomi we have knowledge of the parameter list, and
        # which band we are using, so we can dynamically create a
        # range of jacobian arrays based on what band we are
        # interested in.

        # First declare dictionary, and common elements
        if is_tropomi:
            jacobian_dictionary = {
                "jacobian_cloud_ils": np.zeros(shape=(nfreq_tot), dtype=np.float64),
                "cloud_Surface_Albedo": np.zeros(shape=(nfreq_tot), dtype=np.float64),
            }

            # Then populate dictionary with elements assigned to specific band
            for ii in range(0, len(i_uip["microwindows_all"])):
                for jj in i_uip["tropomiPars"]:
                    if (
                        i_uip["microwindows_all"][ii]["filter"] in jj
                        and "vza" not in jj
                        and "sza" not in jj
                        and "raz" not in jj
                    ):  # Ignoring the angles since these won't be jacobians
                        jacobian_dictionary[jj] = np.zeros(
                            shape=(nfreq_tot), dtype=np.float64
                        )
                    else:
                        continue
        else:
            jacobian_dictionary = {
                "jacobian_cloud_ils" : np.zeros(shape=(nfreq_tot), dtype=np.float64),
                "jacobian_ring_sf_ils_uv1" : np.zeros(shape=(nfreq_tot), dtype=np.float64),
                "jacobian_ring_sf_ils_uv2" : np.zeros(shape=(nfreq_tot), dtype=np.float64),
                "jacobian_nradwav_ils_uv1" : np.zeros(shape=(nfreq_tot), dtype=np.float64),
                "jacobian_nradwav_ils_uv2" : np.zeros(shape=(nfreq_tot), dtype=np.float64),
                "jacobian_odwav_ils_uv1" : np.zeros(shape=(nfreq_tot), dtype=np.float64),
                "jacobian_odwav_ils_uv2" : np.zeros(shape=(nfreq_tot), dtype=np.float64),
                "jacobian_odwav_slope_ils_uv1" : np.zeros(shape=(nfreq_tot), dtype=np.float64),
                "jacobian_odwav_slope_ils_uv2" : np.zeros(shape=(nfreq_tot), dtype=np.float64),
                "jacobian_OMISURFACEALBEDOUV1" : np.zeros(shape=(nfreq_tot), dtype=np.float64),
                "jacobian_OMISURFACEALBEDOUV2" : np.zeros(shape=(nfreq_tot), dtype=np.float64),
                "jacobian_OMISURFACEALBEDOSLOPEUV2" : np.zeros(
                    shape=(nfreq_tot), dtype=np.float64
                ),
            }

        jacob_str = []
        for ii in range(0, len(i_uip["jacobians"])):
            jacob_str.append(i_uip["jacobians"][ii].upper())
        jacob_str = np.asarray(jacob_str)

        nlayers = atmparams["nlayers"]
        atm_clear_jacobians_ils = None

        if i_uip["num_atm_k"] > 0:
            cnt = 0
            k_structure = {
                "species": "thisisadummystring",
                "k": np.zeros(shape=(nfreq_tot, nlayers), dtype=np.float64),
            }

            atm_clear_jacobians_ils = []
            for ii in range(0, i_uip["num_atm_k"]):
                atm_clear_jacobians_ils.append(
                    copy.deepcopy(k_structure)
                )  # Make a deepcopy so each element will have its own memory.

            for ii in range(0, len(jacob_str)):
                if jacob_str[ii] in ("O3", "SO2", "NO2"):
                    atm_clear_jacobians_ils[cnt]["species"] = jacob_str[ii]
                    cnt = cnt + 1
        if is_tropomi:
            nlayers_cloud = np.count_nonzero(
                rayInfo["pbar"] <= i_uip["tropomiPars"]["cloud_pressure"]
            )
        else:
            cloud_pressure = i_uip["omiPars"]["cloud_pressure"]
            if cloud_pressure < 0:
                raise RuntimeError(
                    "i_uip['omiPars']['cloud_pressure'] < 0. Check the OMI Cloud L2 product used as input for OMI cloud variables."
                )

            nlayers_cloud = np.count_nonzero(rayInfo["pbar"] <= cloud_pressure)
        atm_cloud_jacobians_ils = None

        if i_uip["num_atm_k"] > 0:
            cnt = 0
            k_structure = {
                "species": "thisisadummystring",
                "k": np.zeros(shape=(nfreq_tot, nlayers_cloud), dtype=np.float64),
            }

            atm_cloud_jacobians_ils = []  # replicate(temporary(k_structure),uip.num_atm_k)
            for ii in range(0, i_uip["num_atm_k"]):
                atm_cloud_jacobians_ils.append(copy.deepcopy(k_structure))

            for ii in range(0, len(jacob_str)):
                if jacob_str[ii] in ("O3", "SO2", "NO2"):
                    atm_cloud_jacobians_ils[cnt]["species"] = jacob_str[ii]
                    cnt = cnt + 1

        # Update Measured Radiances By Applying Wavelength Shift Parameters
        if is_tropomi:
            tropomi_radiance = get_tropomi_radiance(i_uip["tropomiPars"], tropomi0=i_obs)
        else:
            omi_radiance = get_omi_radiance(i_uip["omiPars"], omi0=i_obs)

        ############# EM-NOTE TEMP SECTION OF CODE TO UPDATE O3 XSEC FOR TEMPERATURE FIT ##########################
        # loop over all microwindows
        for ii_mw in range(0, mw_account["mw_cnt"]):
            if is_tropomi:
                ####### We have to recalculate the O3XSEC
                for i in range(0, len(i_uip["jacobians_all"])):
                    if i_uip["jacobians_all"][i] == "TROPOMITEMPSHIFTBAND3":
                        i_uip_temp = i_uip
                        STARTMW_FM = uip_tropomi["microwindows"][ii_mw]["startmw_fm"][ii_mw]
                        ENDDMW_FM = uip_tropomi["microwindows"][ii_mw]["enddmw_fm"][ii_mw]

                        if i_uip_temp["ils_tropomi_xsection"] == "APPLY":
                            mononfreq_spacing = []
                            num_fwhm_srf = np.float64(4.0)
                            for imw in range(0, i_uip_temp["mw_count"] + 1):
                                tempfreqIndex = (
                                    np.arange(0, (ENDDMW_FM - STARTMW_FM + 1)) + STARTMW_FM
                                )

                                # AT_LINE 199 Make_UIP_OMI.pro
                                ilsInfo = get_tropomi_ils(
                                    i_uip_temp["L2_OSP_PATH"],
                                    i_uip_temp["fullbandfrequency"],
                                    tempfreqIndex,
                                    i_uip_temp["uip_TROPOMI"]["tropomiInfo"][
                                        "Earth_Radiance"
                                    ]["EarthWavelength_Filter"],
                                    i_uip_temp["uip_TROPOMI"]["tropomiInfo"][
                                        "Earth_Radiance"
                                    ]["ObservationTable"],
                                    num_fwhm_srf,
                                    mononfreq_spacing,
                                )

                                # EM NOTE - Adding uip_master_temp into this function, in order to retrieve a temperature shift if required
                                o3_ind = np.where(
                                    np.asarray(rayInfo["level_params"]["species"]) == "O3"
                                )[0]
                                o3_col = rayInfo["column_species"][o3_ind, :]

                                if len(o3_col.shape) == 2 and o3_col.shape[0] == 1:
                                    o3_col = np.reshape(o3_col, (o3_col.shape[1]))

                                no2_col = None

                                do_temp_shift = True
                                o3_xsec = get_tropomi_o3xsec(
                                    i_uip_temp["L2_OSP_PATH"],
                                    ilsInfo,
                                    i_uip_temp["TATM"],
                                    i_uip_temp["fullbandfrequency"],
                                    tempfreqIndex,
                                    i_uip_temp,
                                    do_temp_shift,
                                    o3_col,
                                    no2_col,
                                    i_uip_temp["uip_TROPOMI"]["tropomiInfo"],
                                )

                                output_filename = f"O3Xsec_MW{imw + 1:03d}.asc"
                                output_filename = (
                                    vlidort_input_dir + os.path.sep + output_filename
                                )

                                with open(output_filename, "w") as file_handle:
                                    file_handle.write(
                                        "{:5d}".format(o3_xsec["num_points"])
                                        + " "
                                        + str(o3_xsec["nlayer"])
                                        + "\n"
                                    )
                                    for ip in range(0, o3_xsec["num_points"]):
                                        file_handle.write(
                                            "{:5d}".format(o3_xsec["freqIndex"][ip])
                                            + " "
                                            + "{:19.17f}".format(
                                                i_uip_temp["fullbandfrequency"][
                                                    o3_xsec["freqIndex"][ip]
                                                ]
                                            )
                                        )
                                        for ilayer in range(0, o3_xsec["nlayer"]):
                                            file_handle.write(
                                                " "
                                                + "{:17.16e}".format(
                                                    o3_xsec["o3xsec"][ip, ilayer]
                                                )
                                            )

                                        # Write a carriage return to end one line.
                                        file_handle.write("\n")
                                file_handle.close()
                            # end: for imw in range(0, i_uip_temp['mw_count']+1):
                        # end: if i_uip_temp['ils_tropomi_xsection'] == 'APPLY':
                # end: for ii_mw in range(0, mw_account['mw_cnt']):
                ############# EM-NOTE TEMP SECTION OF CODE TO UPDATE O3 XSEC FOR TEMPERATURE FIT ##########################

            # AT_LINE 201 OMI/omi_fm.pro
            mw_account["mw_cnt"] = ii_mw

            mws = mw_account["mw_range"][0, ii_mw]
            mwf = mw_account["mw_range"][1, ii_mw]

            # * measurement wavelength grid after ILS convolved
            v_ils_mw = mw_account["freq"][mws : mwf + 1]

            if ii_mw == 0:
                v_ils_total = v_ils_mw  # get ils convolved frequency grid

            if ii_mw > 0:
                v_ils_total = np.concatenate((v_ils_total, v_ils_mw), axis=0)

            # RADIATIVE TRANSFER for clear sky
            if is_tropomi:
                logger.info("Calling rtf_tropomi for clear sky")
                do_cloud = 0
                (
                    atm_clear_jacobians_ils,
                    atm_cloud_jacobians_ils,
                    radiance_clear_ils,
                    ring_clear_ils,
                    radiance_cloud_ils,
                    ring_cloud_ils,
                    jacobian_dictionary,
                    radiance_matrix_temperature_clear,
                    radiance_matrix_temperature_cloudy,
                    ring_clear_ils_temperature,
                    ring_cloud_ils_temperature,
                    o_success_flag,
                ) = rtf_tropomi(
                    rayInfo,
                    i_uip,
                    mw_account,
                    ii_mw,
                    do_cloud,
                    nlayers,
                    atm_clear_jacobians_ils,
                    atm_cloud_jacobians_ils,
                    radiance_clear_ils,
                    radiance_cloud_ils,
                    ring_clear_ils,
                    ring_cloud_ils,
                    radiance_matrix_temperature_clear,
                    radiance_matrix_temperature_cloudy,
                    ring_clear_ils_temperature,
                    ring_cloud_ils_temperature,
                    jacobian_dictionary,
                    i_osp_dir=i_osp_dir,
                    i_obs=i_obs,
                    skip_raman_copy=skip_raman_copy,
                )

                if o_success_flag == 0:
                    raise RuntimeError("Call to rtf_tropomi failed")

                # RADIATIVE TRANSFER for cloudy sky
                # Note that the function rtf_tropomi() for cloudy sky uses nlayers_cloud as the 6th parameter insead of nlayers for clear sky.
                logger.info("Calling rtf_tropomi for cloudy sky")
                do_cloud = 1
                (
                    atm_clear_jacobians_ils,
                    atm_cloud_jacobians_ils,
                    radiance_clear_ils,
                    ring_clear_ils,
                    radiance_cloud_ils,
                    ring_cloud_ils,
                    jacobian_dictionary,
                    radiance_matrix_temperature_clear,
                    radiance_matrix_temperature_cloudy,
                    ring_clear_ils_temperature,
                    ring_cloud_ils_temperature,
                    o_success_flag,
                ) = rtf_tropomi(
                    rayInfo,
                    i_uip,
                    mw_account,
                    ii_mw,
                    do_cloud,
                    nlayers_cloud,
                    atm_clear_jacobians_ils,
                    atm_cloud_jacobians_ils,
                    radiance_clear_ils,
                    radiance_cloud_ils,
                    ring_clear_ils,
                    ring_cloud_ils,
                    radiance_matrix_temperature_clear,
                    radiance_matrix_temperature_cloudy,
                    ring_clear_ils_temperature,
                    ring_cloud_ils_temperature,
                    jacobian_dictionary,
                    i_osp_dir=i_osp_dir,
                    i_obs=i_obs,
                    skip_raman_copy=skip_raman_copy,
                )

                if o_success_flag == 0:
                    raise RuntimeError("Call to rtf_tropomi failed")
            else:
                # RADIATIVE TRANSFER for clear sky
                logger.info("Calling rtf_omi for clear sky")

                do_cloud = 0
                (
                    atm_clear_jacobians_ils,
                    atm_cloud_jacobians_ils,
                    radiance_clear_ils,
                    ring_clear_ils,
                    radiance_cloud_ils,
                    ring_cloud_ils,
                    jacobian_dictionary["jacobian_OMISURFACEALBEDOUV1"],
                    jacobian_dictionary["jacobian_OMISURFACEALBEDOUV2"],
                    jacobian_dictionary["jacobian_OMISURFACEALBEDOSLOPEUV2"],
                    o_success_flag,
                ) = rtf_omi(
                    rayInfo,
                    i_uip,
                    mw_account,
                    ii_mw,
                    do_cloud,
                    nlayers,
                    atm_clear_jacobians_ils,
                    atm_cloud_jacobians_ils,
                    radiance_clear_ils,
                    radiance_cloud_ils,
                    ring_clear_ils,
                    ring_cloud_ils,
                    jacobian_dictionary["jacobian_OMISURFACEALBEDOUV1"],
                    jacobian_dictionary["jacobian_OMISURFACEALBEDOUV2"],
                    jacobian_dictionary["jacobian_OMISURFACEALBEDOSLOPEUV2"],
                    i_osp_dir=i_osp_dir,
                    i_obs=i_obs,
                    skip_raman_copy=skip_raman_copy,
                )

                if o_success_flag == 0:
                    raise RuntimeError("Call to rtf_omi failed")

                # RADIATIVE TRANSFER for cloud sky
                logger.info("Calling rtf_omi for cloudy sky")

                # Note that the function rtf_omi() for cloudy sky uses nlayers_cloud as the 6th parameter insead of nlayers for clear sky.
                do_cloud = 1
                (
                    atm_clear_jacobians_ils,
                    atm_cloud_jacobians_ils,
                    radiance_clear_ils,
                    ring_clear_ils,
                    radiance_cloud_ils,
                    ring_cloud_ils,
                    jacobian_OMISURFACEALBEDOUV1,
                    jacobian_OMISURFACEALBEDOUV2,
                    jacobian_OMISURFACEALBEDOSLOPEUV2,
                    o_success_flag,
                ) = rtf_omi(
                    rayInfo,
                    i_uip,
                    mw_account,
                    ii_mw,
                    do_cloud,
                    nlayers_cloud,
                    atm_clear_jacobians_ils,
                    atm_cloud_jacobians_ils,
                    radiance_clear_ils,
                    radiance_cloud_ils,
                    ring_clear_ils,
                    ring_cloud_ils,
                    jacobian_dictionary["jacobian_OMISURFACEALBEDOUV1"],
                    jacobian_dictionary["jacobian_OMISURFACEALBEDOUV2"],
                    jacobian_dictionary["jacobian_OMISURFACEALBEDOSLOPEUV2"],
                    i_osp_dir=i_osp_dir,
                    i_obs=i_obs,
                    skip_raman_copy=skip_raman_copy,
                )

                if o_success_flag == 0:
                    raise RuntimeError("Call to rtf_omi failed")
                

            #  Revert the Layer in Jacobians;  FM mapping (Layer to Level); and
            #  Combine Cloud/Clear Sky Radiances/Jacobians
            #  Compute jacobians of cloud fraction, ring scaling factor, and
            #  wavelength shift parameters
            #
            # jacobians that go into rev_and_fm_map for reference
            # jacobian_ring_sf_ils_uv1, jacobian_ring_sf_ils_uv2,
            # jacobian_nradwav_ils_uv1, jacobian_nradwav_ils_uv2,
            # jacobian_odwav_ils_uv1, jacobian_odwav_ils_uv2,
            # jacobian_odwav_slope_ils_uv1,
            # jacobian_odwav_slope_ils_uv2

            if is_tropomi:
                (
                    jacobians_atm_ils,
                    radiance_clear_ils,
                    radiance_cloud_ils,
                    jacobian_dictionary,
                ) = tropomi_rev_and_fm_map(
                    rayInfo,
                    i_uip,
                    mw_account,
                    nlayers,
                    nlayers_cloud,
                    radiance_ils,
                    radiance_clear_ils,
                    radiance_cloud_ils,
                    tropomi_radiance,
                    ring_clear_ils,
                    ring_cloud_ils,
                    jacobians_atm_ils,
                    atm_clear_jacobians_ils,
                    atm_cloud_jacobians_ils,
                    jacobian_dictionary,
                    radiance_matrix_temperature_clear,
                    radiance_matrix_temperature_cloudy,
                    radiance_temperature_ils,
                    ring_clear_ils_temperature,
                    ring_cloud_ils_temperature,
                    mws,
                    mwf,
                    ii_mw,
                )
            else:
                (
                    jacobians_atm_ils,
                    radiance_clear_ils,
                    radiance_cloud_ils,
                    jacobian_cloud_ils,
                    jacobian_ring_sf_ils_uv1,
                    jacobian_ring_sf_ils_uv2,
                    jacobian_nradwav_ils_uv1,
                    jacobian_nradwav_ils_uv2,
                    jacobian_odwav_ils_uv1,
                    jacobian_odwav_ils_uv2,
                    jacobian_odwav_slope_ils_uv1,
                    jacobian_odwav_slope_ils_uv2,
                ) = rev_and_fm_map(
                    rayInfo,
                    i_uip,
                    mw_account,
                    nlayers,
                    nlayers_cloud,
                    radiance_ils,
                    radiance_clear_ils,
                    radiance_cloud_ils,
                    omi_radiance,
                    ring_clear_ils,
                    ring_cloud_ils,
                    jacobians_atm_ils,
                    atm_clear_jacobians_ils,
                    atm_cloud_jacobians_ils,
                    jacobian_dictionary["jacobian_cloud_ils"],
                    jacobian_dictionary["jacobian_ring_sf_ils_uv1"],
                    jacobian_dictionary["jacobian_ring_sf_ils_uv2"],
                    jacobian_dictionary["jacobian_nradwav_ils_uv1"],
                    jacobian_dictionary["jacobian_nradwav_ils_uv2"],
                    jacobian_dictionary["jacobian_odwav_ils_uv1"],
                    jacobian_dictionary["jacobian_odwav_ils_uv2"],
                    jacobian_dictionary["jacobian_odwav_slope_ils_uv1"],
                    jacobian_dictionary["jacobian_odwav_slope_ils_uv2"],
                    mws,
                    mwf,
                    ii_mw,
                )
                
        # end for ii_mw in range(0, mw_account['mw_cnt']):

        # Pack Radiance and Jacobians Note that the returned value of
        # o_jacobian_pack from pack_omi_jacobian() can be None if the
        # value of num_par is 0, so becareful accessing
        # o_jacobian_pack.
        if is_tropomi:
            (o_radiance_pack, o_jacobian_pack) = pack_tropomi_jacobian(
                i_uip,
                radiance_ils,
                tropomi_radiance,
                jacobians_atm_ils,
                jacobian_dictionary,
                ii_mw,
            )
        else:
            (o_radiance_pack, o_jacobian_pack) = pack_omi_jacobian(
                i_uip,
                radiance_ils,
                omi_radiance,
                jacobians_atm_ils,
                jacobian_cloud_ils,
                jacobian_dictionary["jacobian_ring_sf_ils_uv1"],
                jacobian_dictionary["jacobian_ring_sf_ils_uv2"],
                jacobian_dictionary["jacobian_nradwav_ils_uv1"],
                jacobian_dictionary["jacobian_nradwav_ils_uv2"],
                jacobian_dictionary["jacobian_odwav_ils_uv1"],
                jacobian_dictionary["jacobian_odwav_ils_uv2"],
                jacobian_dictionary["jacobian_odwav_slope_ils_uv1"],
                jacobian_dictionary["jacobian_odwav_slope_ils_uv2"],
                jacobian_dictionary["jacobian_OMISURFACEALBEDOUV1"],
                jacobian_dictionary["jacobian_OMISURFACEALBEDOUV2"],
                jacobian_dictionary["jacobian_OMISURFACEALBEDOSLOPEUV2"],
            )
            

        # Sanity Check on NAN for radiance and jacobian.
        if not np.all(np.isfinite(o_radiance_pack)):
            raise RuntimeError("o_radiance_pack NOT FINITE!")

        if o_jacobian_pack is not None and not np.all(np.isfinite(o_jacobian_pack)):
            raise RuntimeError("o_jacobian_pack NOT FINITE!")

        # Add o_success_flag to report how the function did.
        return (
            o_jacobian_pack,
            o_radiance_pack,
        )
    


class MusesTropomiForwardModelVlidort(MusesForwardModelVlidortBase):
    def __init__(
        self,
        current_state: CurrentState,
        obs: MusesObservation,
        rconf: RetrievalConfiguration,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            current_state,
            InstrumentIdentifier("TROPOMI"),
            obs,
            rconf,
            **kwargs,
        )

    def fm_call(self):
        # We looked at rtf_tropomi and the ring code to determine what files are
        # read here. These are fixed, unless the code gets modified at some
        # point.
        from refractor.muses_py_fm import muses_py_call

        tpath = self.rconf.input_file_helper.osp_dir / "TROPOMI" / "RamanInputs"
        for fname in (
            "N2En.txt",
            "N2pos.txt",
            "N2PT.txt",
            "O2EnfZ.txt",
            "O2En.txt",
            "O2JfZ.txt",
            "O2J.txt",
            "O2pos.txt",
            "O2PT.txt",
        ):
            self.rconf.input_file_helper.notify_file_input(tpath / fname)
        with muses_py_call(
            self.rf_uip.run_dir,
            vlidort_nstokes=self.vlidort_nstokes,
            vlidort_nstreams=self.vlidort_nstreams,
        ):
            jac, rad = self.fm_call2(
                self.rf_uip.uip_all(self.instrument_name),
                is_tropomi=True,
                i_osp_dir=self.rconf.input_file_helper.osp_dir,
                i_obs=self.obs.radiance_for_uip,
                skip_raman_copy=True,
            )
        return jac, rad


class MusesOmiForwardModelVlidort(MusesForwardModelVlidortBase):
    def __init__(
        self,
        current_state: CurrentState,
        obs: MusesObservation,
        rconf: RetrievalConfiguration,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            current_state,
            InstrumentIdentifier("OMI"),
            obs,
            rconf,
            **kwargs,
        )

    def fm_call(self):
        # We looked at rtf_omi and the ring code to determine what files are
        # read here. These are fixed, unless the code gets modified at some
        # point.
        from refractor.muses_py_fm import muses_py_call

        tpath = self.rconf.input_file_helper.osp_dir / "OMI" / "RamanInputs"
        for fname in (
            "N2En.txt",
            "N2pos.txt",
            "N2PT.txt",
            "O2EnfZ.txt",
            "O2En.txt",
            "O2JfZ.txt",
            "O2J.txt",
            "O2pos.txt",
            "O2PT.txt",
        ):
            self.rconf.input_file_helper.notify_file_input(tpath / fname)
        with muses_py_call(
            self.rf_uip.run_dir,
            vlidort_nstokes=self.vlidort_nstokes,
            vlidort_nstreams=self.vlidort_nstreams,
        ):
            jac, rad = self.fm_call2(
                self.rf_uip.uip_all(self.instrument_name),
                is_tropomi=False,
                i_osp_dir=self.rconf.input_file_helper.osp_dir,
                i_obs=self.obs.radiance_for_uip,
                skip_raman_copy=True,
            )
        return jac, rad

C = TypeVar("C", bound=rf.ForwardModel)

class MusesForwardModelHandle(ForwardModelHandle):
    def __init__(
        self,
        instrument_name: InstrumentIdentifier,
        cls: type[C],
        use_vlidort_temp_dir: bool = False,
        **creator_kwargs: Any,
    ) -> None:
        self.creator_kwargs = creator_kwargs
        self.instrument_name = instrument_name
        self.cls = cls
        self.rconf: RetrievalConfiguration | None = None
        self.use_vlidort_temp_dir = use_vlidort_temp_dir

    def notify_update_target(
        self, measurement_id: MeasurementId, retrieval_config: RetrievalConfiguration
    ) -> None:
        """Clear any caching associated with assuming the target being retrieved is fixed"""
        logger.debug(f"Call to {self.__class__.__name__}::notify_update")
        self.rconf = retrieval_config

    def forward_model(
        self,
        instrument_name: InstrumentIdentifier,
        current_state: CurrentState,
        obs: MusesObservation,
        fm_sv: rf.StateVector,
        **kwargs: Any,
    ) -> None | rf.ForwardModel:
        # Note, we can't just attach to the fm_sv when we create the MusesForwardModel.
        # The UIP takes in parameters on the RetrievalGridArray, *not*
        # FullGridMappedArray like the ReFRACtor. This is handled in notify_cost_function
        # (see MusesForwardModelBase), which gets called when the CostFunction is created.

        if instrument_name != self.instrument_name:
            return None
        if self.rconf is None:
            raise RuntimeError("Need to call notify_update_target before forward_model")
        logger.debug(f"Creating forward model {self.cls.__name__}")
        return self.cls(
            current_state,
            obs,
            self.rconf,
            **kwargs,
        )
    
ForwardModelHandleSet.add_default_handle(
    MusesForwardModelHandle(InstrumentIdentifier("TROPOMI"), MusesTropomiForwardModelVlidort),
    priority_order=-1,
)
ForwardModelHandleSet.add_default_handle(
    MusesForwardModelHandle(InstrumentIdentifier("OMI"), MusesOmiForwardModelVlidort),
    priority_order=-1,
)
    
__all__ = [
    "MusesForwardModelVlidortBase",
    "MusesTropomiForwardModelVlidort",
    "MusesOmiForwardModelVlidort",
]

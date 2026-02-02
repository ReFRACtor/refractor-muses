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
import shutil
import subprocess
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

    def fm_call2(
        self, i_uip, is_tropomi: bool, i_osp_dir=None, i_obs=None, skip_raman_copy=False
    ):
        # Temp, we'll pull some of this over and get other parts into mpy
        from refractor.muses_py import (
            pack_omi_jacobian,
            rev_and_fm_map,
            get_omi_radiance,
            pack_tropomi_jacobian,
            get_tropomi_radiance,
            raylayer_nadir,
            atmosphere_level,
            tropomi_rev_and_fm_map,
        )

        from refractor.muses import AttrDictAdapter

        self.i_uip = i_uip
        self.is_tropomi = is_tropomi
        if self.is_tropomi:
            uip_tropomi = self.i_uip["uip_TROPOMI"]

            # VLIDORT I/O
            vlidort_input_dir = uip_tropomi["vlidort_input"]
            Path(vlidort_input_dir).mkdir(parents=True, exist_ok=True)

        # Get atmospheric parameters
        self.i_uip["obs_table"]["pointing_angle"] = 0.0
        atmparams = atmosphere_level(self.i_uip)

        # Computer layer and level quanity
        self.rayInfo = raylayer_nadir(
            AttrDictAdapter(self.i_uip), AttrDictAdapter(atmparams)
        )

        self.mw_account = self.summarize_mw(self.i_uip)
        nfreq_tot = self.mw_account["nfreq_tot"]

        self.i_uip["num_atm_k"] = sum(
            jac in self.i_uip["species"] or jac == "TATM"
            for jac in self.i_uip["jacobians"]
        )

        # Create radiance and ring arrays
        self.radiance_ils = np.zeros(shape=(nfreq_tot), dtype=np.float64)
        self.radiance_clear_ils = np.zeros(shape=(nfreq_tot), dtype=np.float64)
        self.radiance_cloud_ils = np.zeros(shape=(nfreq_tot), dtype=np.float64)
        self.radiance_matrix_temperature_clear = np.zeros(
            shape=(nfreq_tot), dtype=np.float64
        )
        self.radiance_matrix_temperature_cloudy = np.zeros(
            shape=(nfreq_tot), dtype=np.float64
        )
        self.radiance_temperature_ils = np.zeros(shape=(nfreq_tot), dtype=np.float64)

        self.ring_clear_ils = np.zeros(shape=(nfreq_tot), dtype=np.float64)
        self.ring_cloud_ils = np.zeros(shape=(nfreq_tot), dtype=np.float64)
        self.ring_clear_ils_temperature = np.zeros(shape=(nfreq_tot), dtype=np.float64)
        self.ring_cloud_ils_temperature = np.zeros(shape=(nfreq_tot), dtype=np.float64)

        # Create atmospheric jacobians
        self.jacobians_atm_ils = None
        if self.i_uip["num_atm_k"] > 0:
            k_temp = []

            for kk in range(self.i_uip["num_atm_k"]):
                k_temp.append(
                    {
                        "species": "thisisadummystring",
                        "k": np.zeros(
                            shape=(atmparams["nlayers"] + 1, nfreq_tot),
                            dtype=np.float64,
                        ),
                    }
                )

            self.jacobians_atm_ils = {"k_species": k_temp}
        # end if (self.i_uip['num_atm_k'] > 0):

        # Create arrays for the following jacobians:
        # [Cloud Fraction,Ring Scaling Factor,Earth/Solar Wavelength Shift Parameter,od Wavelength Shift Parameter,od Wavelength Shift Parameter]

        # EM NOTE - Here we are creating empty jacobian arrays for
        # use, the original OMI code hardcoded a series of parameters,
        # but for tropomi we have knowledge of the parameter list, and
        # which band we are using, so we can dynamically create a
        # range of jacobian arrays based on what band we are
        # interested in.

        # First declare dictionary, and common elements
        if self.is_tropomi:
            self.jacobian_dictionary = {
                "jacobian_cloud_ils": np.zeros(shape=(nfreq_tot), dtype=np.float64),
                "cloud_Surface_Albedo": np.zeros(shape=(nfreq_tot), dtype=np.float64),
            }

            # Then populate dictionary with elements assigned to specific band
            for ii in range(0, len(self.i_uip["microwindows_all"])):
                for jj in self.i_uip["tropomiPars"]:
                    if (
                        self.i_uip["microwindows_all"][ii]["filter"] in jj
                        and "vza" not in jj
                        and "sza" not in jj
                        and "raz" not in jj
                    ):  # Ignoring the angles since these won't be jacobians
                        self.jacobian_dictionary[jj] = np.zeros(
                            shape=(nfreq_tot), dtype=np.float64
                        )
                    else:
                        continue
        else:
            self.jacobian_dictionary = {
                "jacobian_cloud_ils": np.zeros(shape=(nfreq_tot), dtype=np.float64),
                "jacobian_ring_sf_ils_uv1": np.zeros(
                    shape=(nfreq_tot), dtype=np.float64
                ),
                "jacobian_ring_sf_ils_uv2": np.zeros(
                    shape=(nfreq_tot), dtype=np.float64
                ),
                "jacobian_nradwav_ils_uv1": np.zeros(
                    shape=(nfreq_tot), dtype=np.float64
                ),
                "jacobian_nradwav_ils_uv2": np.zeros(
                    shape=(nfreq_tot), dtype=np.float64
                ),
                "jacobian_odwav_ils_uv1": np.zeros(shape=(nfreq_tot), dtype=np.float64),
                "jacobian_odwav_ils_uv2": np.zeros(shape=(nfreq_tot), dtype=np.float64),
                "jacobian_odwav_slope_ils_uv1": np.zeros(
                    shape=(nfreq_tot), dtype=np.float64
                ),
                "jacobian_odwav_slope_ils_uv2": np.zeros(
                    shape=(nfreq_tot), dtype=np.float64
                ),
                "jacobian_OMISURFACEALBEDOUV1": np.zeros(
                    shape=(nfreq_tot), dtype=np.float64
                ),
                "jacobian_OMISURFACEALBEDOUV2": np.zeros(
                    shape=(nfreq_tot), dtype=np.float64
                ),
                "jacobian_OMISURFACEALBEDOSLOPEUV2": np.zeros(
                    shape=(nfreq_tot), dtype=np.float64
                ),
            }

        jacob_str = []
        for ii in range(0, len(self.i_uip["jacobians"])):
            jacob_str.append(self.i_uip["jacobians"][ii].upper())
        jacob_str = np.asarray(jacob_str)

        nlayers = atmparams["nlayers"]
        self.atm_clear_jacobians_ils = None

        if self.i_uip["num_atm_k"] > 0:
            cnt = 0
            k_structure = {
                "species": "thisisadummystring",
                "k": np.zeros(shape=(nfreq_tot, nlayers), dtype=np.float64),
            }

            self.atm_clear_jacobians_ils = []
            for ii in range(0, self.i_uip["num_atm_k"]):
                self.atm_clear_jacobians_ils.append(
                    copy.deepcopy(k_structure)
                )  # Make a deepcopy so each element will have its own memory.

            for ii in range(0, len(jacob_str)):
                if jacob_str[ii] in ("O3", "SO2", "NO2"):
                    self.atm_clear_jacobians_ils[cnt]["species"] = jacob_str[ii]
                    cnt = cnt + 1
        if self.is_tropomi:
            nlayers_cloud = np.count_nonzero(
                self.rayInfo["pbar"] <= self.i_uip["tropomiPars"]["cloud_pressure"]
            )
        else:
            cloud_pressure = self.i_uip["omiPars"]["cloud_pressure"]
            if cloud_pressure < 0:
                raise RuntimeError(
                    "self.i_uip['omiPars']['cloud_pressure'] < 0. Check the OMI Cloud L2 product used as input for OMI cloud variables."
                )

            nlayers_cloud = np.count_nonzero(self.rayInfo["pbar"] <= cloud_pressure)
        self.atm_cloud_jacobians_ils = None

        if self.i_uip["num_atm_k"] > 0:
            cnt = 0
            k_structure = {
                "species": "thisisadummystring",
                "k": np.zeros(shape=(nfreq_tot, nlayers_cloud), dtype=np.float64),
            }

            self.atm_cloud_jacobians_ils = []  # replicate(temporary(k_structure),uip.num_atm_k)
            for ii in range(0, self.i_uip["num_atm_k"]):
                self.atm_cloud_jacobians_ils.append(copy.deepcopy(k_structure))

            for ii in range(0, len(jacob_str)):
                if jacob_str[ii] in ("O3", "SO2", "NO2"):
                    self.atm_cloud_jacobians_ils[cnt]["species"] = jacob_str[ii]
                    cnt = cnt + 1

        # Update Measured Radiances By Applying Wavelength Shift Parameters

        # Notes:
        # This here just gets the jacobian part, used in rev_and_fm_map below. We
        # can perhaps get this information more directly from somewhere
        if self.is_tropomi:
            self.tropomi_radiance = get_tropomi_radiance(
                self.i_uip["tropomiPars"], tropomi0=i_obs
            )
        else:
            self.omi_radiance = get_omi_radiance(self.i_uip["omiPars"], omi0=i_obs)

        # loop over all microwindows
        for ii_mw in range(0, self.mw_account["mw_cnt"]):
            # AT_LINE 201 OMI/omi_fm.pro
            self.mw_account["mw_cnt"] = ii_mw

            mws = self.mw_account["mw_range"][0, ii_mw]
            mwf = self.mw_account["mw_range"][1, ii_mw]

            # * measurement wavelength grid after ILS convolved
            v_ils_mw = self.mw_account["freq"][mws : mwf + 1]

            if ii_mw == 0:
                v_ils_total = v_ils_mw  # get ils convolved frequency grid

            if ii_mw > 0:
                v_ils_total = np.concatenate((v_ils_total, v_ils_mw), axis=0)

            # RADIATIVE TRANSFER for clear sky
            if self.is_tropomi:
                logger.info("Calling rtf_tropomi for clear sky")
                do_cloud = 0
                self.rtf_tropomi(
                    ii_mw,
                    do_cloud,
                    nlayers,
                    i_osp_dir=i_osp_dir,
                    i_obs=i_obs,
                    skip_raman_copy=skip_raman_copy,
                )

                # RADIATIVE TRANSFER for cloudy sky
                # Note that the function rtf_tropomi() for cloudy sky uses nlayers_cloud as the 6th parameter insead of nlayers for clear sky.
                logger.info("Calling rtf_tropomi for cloudy sky")
                do_cloud = 1
                self.rtf_tropomi(
                    ii_mw,
                    do_cloud,
                    nlayers_cloud,
                    i_osp_dir=i_osp_dir,
                    i_obs=i_obs,
                    skip_raman_copy=skip_raman_copy,
                )
            else:
                # RADIATIVE TRANSFER for clear sky
                logger.info("Calling rtf_omi for clear sky")

                do_cloud = 0
                self.rtf_omi(
                    ii_mw,
                    do_cloud,
                    nlayers,
                    i_osp_dir=i_osp_dir,
                    i_obs=i_obs,
                    skip_raman_copy=skip_raman_copy,
                )

                # RADIATIVE TRANSFER for cloud sky
                logger.info("Calling rtf_omi for cloudy sky")

                # Note that the function rtf_omi() for cloudy sky uses nlayers_cloud as the 6th parameter insead of nlayers for clear sky.
                do_cloud = 1
                self.rtf_omi(
                    ii_mw,
                    do_cloud,
                    nlayers_cloud,
                    i_osp_dir=i_osp_dir,
                    i_obs=i_obs,
                    skip_raman_copy=skip_raman_copy,
                )

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

            if self.is_tropomi:
                (
                    self.jacobians_atm_ils,
                    self.radiance_clear_ils,
                    self.radiance_cloud_ils,
                    self.jacobian_dictionary,
                ) = tropomi_rev_and_fm_map(
                    self.rayInfo,
                    self.i_uip,
                    self.mw_account,
                    nlayers,
                    nlayers_cloud,
                    self.radiance_ils,
                    self.radiance_clear_ils,
                    self.radiance_cloud_ils,
                    self.tropomi_radiance,
                    self.ring_clear_ils,
                    self.ring_cloud_ils,
                    self.jacobians_atm_ils,
                    self.atm_clear_jacobians_ils,
                    self.atm_cloud_jacobians_ils,
                    self.jacobian_dictionary,
                    self.radiance_matrix_temperature_clear,
                    self.radiance_matrix_temperature_cloudy,
                    self.radiance_temperature_ils,
                    self.ring_clear_ils_temperature,
                    self.ring_cloud_ils_temperature,
                    mws,
                    mwf,
                    ii_mw,
                )
            else:
                (
                    self.jacobians_atm_ils,
                    self.radiance_clear_ils,
                    self.radiance_cloud_ils,
                    self.jacobian_cloud_ils,
                    self.jacobian_ring_sf_ils_uv1,
                    self.jacobian_ring_sf_ils_uv2,
                    self.jacobian_nradwav_ils_uv1,
                    self.jacobian_nradwav_ils_uv2,
                    self.jacobian_odwav_ils_uv1,
                    self.jacobian_odwav_ils_uv2,
                    self.jacobian_odwav_slope_ils_uv1,
                    self.jacobian_odwav_slope_ils_uv2,
                ) = rev_and_fm_map(
                    self.rayInfo,
                    self.i_uip,
                    self.mw_account,
                    nlayers,
                    nlayers_cloud,
                    self.radiance_ils,
                    self.radiance_clear_ils,
                    self.radiance_cloud_ils,
                    self.omi_radiance,
                    self.ring_clear_ils,
                    self.ring_cloud_ils,
                    self.jacobians_atm_ils,
                    self.atm_clear_jacobians_ils,
                    self.atm_cloud_jacobians_ils,
                    self.jacobian_dictionary["jacobian_cloud_ils"],
                    self.jacobian_dictionary["jacobian_ring_sf_ils_uv1"],
                    self.jacobian_dictionary["jacobian_ring_sf_ils_uv2"],
                    self.jacobian_dictionary["jacobian_nradwav_ils_uv1"],
                    self.jacobian_dictionary["jacobian_nradwav_ils_uv2"],
                    self.jacobian_dictionary["jacobian_odwav_ils_uv1"],
                    self.jacobian_dictionary["jacobian_odwav_ils_uv2"],
                    self.jacobian_dictionary["jacobian_odwav_slope_ils_uv1"],
                    self.jacobian_dictionary["jacobian_odwav_slope_ils_uv2"],
                    mws,
                    mwf,
                    ii_mw,
                )

        # end for ii_mw in range(0, self.mw_account['mw_cnt']):

        # Pack Radiance and Jacobians Note that the returned value of
        # o_jacobian_pack from pack_omi_jacobian() can be None if the
        # value of num_par is 0, so becareful accessing
        # o_jacobian_pack.
        if self.is_tropomi:
            (o_radiance_pack, o_jacobian_pack) = pack_tropomi_jacobian(
                self.i_uip,
                self.radiance_ils,
                self.tropomi_radiance,
                self.jacobians_atm_ils,
                self.jacobian_dictionary,
                ii_mw,
            )
        else:
            (o_radiance_pack, o_jacobian_pack) = pack_omi_jacobian(
                self.i_uip,
                self.radiance_ils,
                self.omi_radiance,
                self.jacobians_atm_ils,
                self.jacobian_cloud_ils,
                self.jacobian_dictionary["jacobian_ring_sf_ils_uv1"],
                self.jacobian_dictionary["jacobian_ring_sf_ils_uv2"],
                self.jacobian_dictionary["jacobian_nradwav_ils_uv1"],
                self.jacobian_dictionary["jacobian_nradwav_ils_uv2"],
                self.jacobian_dictionary["jacobian_odwav_ils_uv1"],
                self.jacobian_dictionary["jacobian_odwav_ils_uv2"],
                self.jacobian_dictionary["jacobian_odwav_slope_ils_uv1"],
                self.jacobian_dictionary["jacobian_odwav_slope_ils_uv2"],
                self.jacobian_dictionary["jacobian_OMISURFACEALBEDOUV1"],
                self.jacobian_dictionary["jacobian_OMISURFACEALBEDOUV2"],
                self.jacobian_dictionary["jacobian_OMISURFACEALBEDOSLOPEUV2"],
            )

        # Sanity Check on NAN for radiance and jacobian.
        if not np.all(np.isfinite(o_radiance_pack)):
            raise RuntimeError("o_radiance_pack NOT FINITE!")

        if o_jacobian_pack is not None and not np.all(np.isfinite(o_jacobian_pack)):
            raise RuntimeError("o_jacobian_pack NOT FINITE!")

        return (
            o_jacobian_pack,
            o_radiance_pack,
        )

    def rtf_omi(
        self,
        ii_mw,
        do_cloud,
        fm_nlayers,
        jacobian_OMISURFACEALBEDOUV1=None,
        jacobian_OMISURFACEALBEDOUV2=None,
        jacobian_OMISURFACEALBEDOSLOPEUV2=None,
        i_osp_dir=None,
        i_obs=None,
        skip_raman_copy=False,
    ):
        from refractor.muses_py import (
            apply_omi_isrf_fast,
            apply_omi_isrf_slow,
            apply_omi_srf,
            read_rtm_output,
            vlidort_run_omi,
            cli_options,
            print_ring_input,
        )

        osp_omi_dir = i_osp_dir / "OMI" if i_osp_dir is not None else Path("../OSP/OMI")

        # Radiative Calculation of OMI FM

        uip_omi = self.i_uip["uip_OMI"]

        # Default run directory if not specified.
        default_run_directory = "./"

        rt_res = vlidort_run_omi(
            default_run_directory, ii_mw, self.i_uip, fm_nlayers, self.rayInfo, do_cloud
        )

        print_ring_input(
            rt_res.vlidort_input_iter_dir,
            rt_res.vlidort_output_iter_dir,
            ii_mw,
            self.i_uip,
            self.rayInfo,
            fm_nlayers,
            do_cloud,
            i_obs=i_obs,
        )

        # Copy the RamanInputs directory as well.
        additional_input_dir_1 = "RamanInputs"
        raman_inputs_dir = Path(default_run_directory) / additional_input_dir_1

        if not raman_inputs_dir.exists():
            # Check if the the additional_input_dir_1 exist exist for copying.  If not, exit.
            raman_inputs_osp_dir = osp_omi_dir / additional_input_dir_1
            if not raman_inputs_osp_dir.exists():
                logger.error(
                    "Cannot find directory to copy",
                    str(osp_omi_dir) + os.path.sep + additional_input_dir_1,
                )
                assert False

            # copy RamanInputs dir
            if not skip_raman_copy:
                shutil.copytree(raman_inputs_osp_dir, raman_inputs_dir)
                os.system(f"chmod -R 777 {raman_inputs_dir.as_posix()}")
            else:
                raman_inputs_dir = raman_inputs_osp_dir.absolute().resolve()

            # symlink RamanInputs dir instead of copying it
            # raman_inputs_dir.symlink_to(raman_inputs_osp_dir, target_is_directory=True)
        # end: if not raman_inputs_dir.exists():

        ring_cli = cli_options.get("ring_cli", "")

        # RING CLI
        executable_filename = "ring_cli"
        ring_cli_exe = Path(ring_cli).expanduser().resolve() / executable_filename

        if not ring_cli_exe.exists():
            logger.error("Cannot find executable %s" % ring_cli_exe)
            assert False

        ring_command = [
            ring_cli_exe.as_posix(),
            "--raman-input",
            raman_inputs_dir.as_posix(),
            "--input",
            rt_res.vlidort_input_iter_dir,
            "--output",
            rt_res.vlidort_output_iter_dir,
        ]

        logger.debug(f"\nRunning:\n{' '.join(ring_command)} ")

        returnCodeFromSystem = subprocess.run(
            ring_command,
            cwd=default_run_directory,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        if returnCodeFromSystem.returncode != 0:
            logger.error("returnCodeFromSystem", returnCodeFromSystem)
            assert False
        logger.info(returnCodeFromSystem.stdout)

        radiance_matrix = rt_res.radiance_matrix
        jacobian_o3_matrix = rt_res.jacobian_o3_matrix
        jacobian_sf_matrix = rt_res.jacobian_sf_matrix

        if radiance_matrix is None:
            raise RuntimeError("Could not read radiance: Radiance.asc")

        if jacobian_o3_matrix is None:
            raise RuntimeError("Could not read jacobian_o3: IWF.asc")

        if jacobian_sf_matrix is None:
            raise RuntimeError("Could not read jacobian_sf: surf_WF.asc")

        ring_matrix = read_rtm_output(rt_res.vlidort_output_iter_dir, "Ring.asc")
        if ring_matrix is None:
            raise RuntimeError("Could not read ring: Ring.asc")

        my_filter = uip_omi["microwindows"][ii_mw]["filter"]

        ils_omi_xsection = uip_omi["ils_omi_xsection"]
        ils_omi_xsection.upper()

        # NOAPPLY is alias of POSTCONV
        if ils_omi_xsection == "NOAPPLY":
            ils_omi_xsection = "POSTCONV"

        if ils_omi_xsection == "POSTCONV":
            radiance_matrix = apply_omi_srf(
                self.i_uip, ii_mw, radiance_matrix, self.omi_radiance
            )
            jacobian_o3_matrix = apply_omi_srf(
                self.i_uip, ii_mw, jacobian_o3_matrix, self.omi_radiance
            )
            jacobian_sf_matrix = apply_omi_srf(
                self.i_uip, ii_mw, jacobian_sf_matrix, self.omi_radiance
            )

            # TODO: VK: Verify. Not sure about this. IDL convolves the optical depth
            # ring_matrix = apply_omi_srf(self.i_uip, ii_mw, ring_matrix, omi_info)
        # end: if ils_omi_xsection == 'POSTCONV':

        if ils_omi_xsection == "SLOWCONV":
            radiance_matrix = apply_omi_isrf_slow(self.i_uip, ii_mw, radiance_matrix)
            jacobian_o3_matrix = apply_omi_isrf_slow(
                self.i_uip, ii_mw, jacobian_o3_matrix
            )
            jacobian_sf_matrix = apply_omi_isrf_slow(
                self.i_uip, ii_mw, jacobian_sf_matrix
            )

        if ils_omi_xsection == "FASTCONV":
            radiance_matrix = apply_omi_isrf_fast(self.i_uip, ii_mw, radiance_matrix)
            jacobian_o3_matrix = apply_omi_isrf_fast(
                self.i_uip, ii_mw, jacobian_o3_matrix
            )
            jacobian_sf_matrix = apply_omi_isrf_fast(
                self.i_uip, ii_mw, jacobian_sf_matrix
            )

        nfreq = (
            uip_omi["microwindows"][ii_mw]["enddmw"][ii_mw]
            - uip_omi["microwindows"][ii_mw]["startmw"][ii_mw]
            + 1
        )  # from [ 20 194] get 20, from [126 306] get 126 for index 0.
        my_filter = uip_omi["microwindows"][ii_mw]["filter"]

        temp_freq_fm = radiance_matrix[0, :]
        temp_freq_ind = np.where(
            (temp_freq_fm >= uip_omi["microwindows"][ii_mw]["start"])
            & (temp_freq_fm <= uip_omi["microwindows"][ii_mw]["endd"])
        )[0]

        if len(temp_freq_ind) != nfreq:
            logger.error(
                "Number of Data points does not match to the expected values: len(temp_freq_ind), nfreq",
                len(temp_freq_ind),
                nfreq,
            )
            logger.error(
                "uip_omi['microwindows'][ii_mw]['startmw' ]",
                uip_omi["microwindows"][ii_mw]["startmw"],
            )
            logger.error(
                "uip_omi['microwindows'][ii_mw]['enddmw' ]",
                uip_omi["microwindows"][ii_mw]["enddmw"],
            )
            assert False

        temp_start_ind = self.mw_account["mw_range"][0, ii_mw]
        temp_endd_ind = self.mw_account["mw_range"][1, ii_mw]

        # clear sky condition
        if do_cloud == 0:
            if (
                radiance_matrix[1, temp_freq_ind].shape[0]
                < self.radiance_clear_ils[temp_start_ind : temp_endd_ind + 1].shape[0]
            ):
                # ValueError: could not broadcast input array from shape (107) into shape (113)
                # To solve the issue above, we must shrink the left hand side from 113 to 107 to match the shape of radiance_matrix[1,temp_freq_ind] vector.
                shrink_left_hand_size = (
                    temp_start_ind + radiance_matrix[1, temp_freq_ind].shape[0]
                )
                self.radiance_clear_ils[temp_start_ind:shrink_left_hand_size] = (
                    radiance_matrix[1, temp_freq_ind]
                )
            else:
                self.radiance_clear_ils[temp_start_ind : temp_endd_ind + 1] = (
                    radiance_matrix[1, temp_freq_ind]
                )

            if (
                ring_matrix[1, temp_freq_ind].shape[0]
                < self.ring_clear_ils[temp_start_ind : temp_endd_ind + 1].shape[0]
            ):
                # ValueError: could not broadcast input array from shape (107) into shape (113)
                # To solve the issue above, we must shrink the left hand side from 113 to 107 to match to the shape of ring_matrix[1,temp_freq_ind] vector.
                shrink_left_hand_size = (
                    temp_start_ind + ring_matrix[1, temp_freq_ind].shape[0]
                )
                self.ring_clear_ils[temp_start_ind:shrink_left_hand_size] = ring_matrix[
                    1, temp_freq_ind
                ][:]
            else:
                self.ring_clear_ils[temp_start_ind : temp_endd_ind + 1] = ring_matrix[
                    1, temp_freq_ind
                ][:]

            if self.i_uip["num_atm_k"] > 0:
                for ii in range(0, len(self.atm_clear_jacobians_ils)):
                    # ValueError: could not broadcast input array from shape (107,64) into shape (113,64)
                    # To solve the issue above, we must shrink the left hand side from 113 to 107 to match to the shape of self.atm_clear_jacobians_ils[ii]['k']
                    if (
                        np.transpose(jacobian_o3_matrix[1:, temp_freq_ind]).shape[0]
                        < self.atm_clear_jacobians_ils[ii]["k"][
                            temp_start_ind : temp_endd_ind + 1, :
                        ].shape[0]
                    ):
                        shrink_left_hand_size = (
                            temp_start_ind
                            + np.transpose(jacobian_o3_matrix[1:, temp_freq_ind]).shape[
                                0
                            ]
                        )
                        self.atm_clear_jacobians_ils[ii]["k"][
                            temp_start_ind:shrink_left_hand_size, :
                        ] = np.transpose(jacobian_o3_matrix[1:, temp_freq_ind])
                    else:
                        self.atm_clear_jacobians_ils[ii]["k"][
                            temp_start_ind : temp_endd_ind + 1, :
                        ] = np.transpose(jacobian_o3_matrix[1:, temp_freq_ind])[:, :]

            if my_filter == "UV1":
                self.jacobian_dictionary["jacobian_OMISURFACEALBEDOUV1"][
                    temp_start_ind : temp_endd_ind + 1
                ] = jacobian_sf_matrix[1, temp_freq_ind]

            if my_filter == "UV2":
                # ValueError: could not broadcast input array from shape (107) into shape (113)
                # To solve the issue above, we must shrink the left hand side from 113 to 107 to match to the shape of jacobian_sf_matrix[1,temp_freq_ind]
                if (
                    jacobian_sf_matrix[1, temp_freq_ind].shape[0]
                    < self.jacobian_dictionary["jacobian_OMISURFACEALBEDOUV2"][
                        temp_start_ind : temp_endd_ind + 1
                    ].shape[0]
                ):
                    shrink_left_hand_size = (
                        temp_start_ind + jacobian_sf_matrix[1, temp_freq_ind].shape[0]
                    )
                    self.jacobian_dictionary["jacobian_OMISURFACEALBEDOUV2"][
                        temp_start_ind:shrink_left_hand_size
                    ] = jacobian_sf_matrix[1, temp_freq_ind]
                else:
                    self.jacobian_dictionary["jacobian_OMISURFACEALBEDOUV2"][
                        temp_start_ind : temp_endd_ind + 1
                    ] = jacobian_sf_matrix[1, temp_freq_ind]

                ref_wav = np.float64(320.0)
                wave_arr = uip_omi["fullbandfrequency"][
                    uip_omi["microwindows"][ii_mw]["startmw"][ii_mw] : uip_omi[
                        "microwindows"
                    ][ii_mw]["enddmw"][ii_mw]
                    + 1
                ]
                delta_wav = wave_arr[:] - ref_wav

                if (
                    jacobian_sf_matrix[1, temp_freq_ind].shape[0]
                    < self.jacobian_dictionary["jacobian_OMISURFACEALBEDOSLOPEUV2"][
                        temp_start_ind : temp_endd_ind + 1
                    ].shape[0]
                ):
                    shrink_left_hand_size = (
                        temp_start_ind + jacobian_sf_matrix[1, temp_freq_ind].shape[0]
                    )
                    self.jacobian_dictionary["jacobian_OMISURFACEALBEDOSLOPEUV2"][
                        temp_start_ind:shrink_left_hand_size
                    ] = jacobian_sf_matrix[1, temp_freq_ind] * delta_wav[:]
                else:
                    self.jacobian_dictionary["jacobian_OMISURFACEALBEDOSLOPEUV2"][
                        temp_start_ind : temp_endd_ind + 1
                    ] = jacobian_sf_matrix[1, temp_freq_ind] * delta_wav[:]
            # end if my_filter == 'UV2':
        # end if do_cloud == 0:

        # cloud sky condition
        if do_cloud == 1:
            # ValueError: could not broadcast input array from shape (107) into shape (113)
            # To solve the issue above, we must shrink the left hand side from 113 to 107 to match to the shape of radiance_matrix[1,temp_freq_ind]
            if (
                radiance_matrix[1, temp_freq_ind].shape[0]
                < self.radiance_cloud_ils[temp_start_ind : temp_endd_ind + 1].shape[0]
            ):
                shrink_left_hand_size = (
                    temp_start_ind + radiance_matrix[1, temp_freq_ind].shape[0]
                )
                self.radiance_cloud_ils[temp_start_ind:shrink_left_hand_size] = (
                    radiance_matrix[1, temp_freq_ind][:]
                )
            else:
                self.radiance_cloud_ils[temp_start_ind : temp_endd_ind + 1] = (
                    radiance_matrix[1, temp_freq_ind][:]
                )

            # ValueError: could not broadcast input array from shape (107) into shape (113)
            # To solve the issue above, we must shrink the left hand side from 113 to 107 to match to the shape of ring_matrix[1,temp_freq_ind]
            if (
                ring_matrix[1, temp_freq_ind].shape[0]
                < self.ring_cloud_ils[temp_start_ind : temp_endd_ind + 1].shape[0]
            ):
                shrink_left_hand_size = (
                    temp_start_ind + ring_matrix[1, temp_freq_ind].shape[0]
                )
                self.ring_cloud_ils[temp_start_ind:shrink_left_hand_size] = ring_matrix[
                    1, temp_freq_ind
                ]
            else:
                self.ring_cloud_ils[temp_start_ind : temp_endd_ind + 1] = ring_matrix[
                    1, temp_freq_ind
                ]

            if self.i_uip["num_atm_k"] > 0:
                if len(self.atm_cloud_jacobians_ils) > 0:
                    for ii in range(0, len(self.atm_cloud_jacobians_ils)):
                        right_hand_side_second_size = np.transpose(
                            jacobian_o3_matrix[1:, temp_freq_ind]
                        ).shape[1]
                        # ValueError: could not broadcast input array from shape (107,60) into shape (113,60)
                        # To solve the issue above, we must shrink the left hand side from 113 to 107 to match to the shape of np.transpose(jacobian_o3_matrix[1:,temp_freq_ind])
                        if np.transpose(jacobian_o3_matrix[1:, temp_freq_ind]).shape[
                            0
                        ] < (temp_endd_ind + 1 - temp_start_ind):
                            shrink_left_hand_size = (
                                temp_start_ind
                                + np.transpose(
                                    jacobian_o3_matrix[1:, temp_freq_ind]
                                ).shape[0]
                            )
                            self.atm_cloud_jacobians_ils[ii]["k"][
                                temp_start_ind:shrink_left_hand_size,
                                0:right_hand_side_second_size,
                            ] = np.transpose(jacobian_o3_matrix[1:, temp_freq_ind])
                        else:
                            self.atm_cloud_jacobians_ils[ii]["k"][
                                temp_start_ind : temp_endd_ind + 1,
                                0:right_hand_side_second_size,
                            ] = np.transpose(jacobian_o3_matrix[1:, temp_freq_ind])[
                                :, :
                            ]
        # end if do_cloud:

    def rtf_tropomi(
        self,
        ii_mw,
        do_cloud,
        fm_nlayers,
        i_osp_dir=None,
        i_obs=None,
        skip_raman_copy=False,
    ):
        from refractor.muses_py import (
            apply_tropomi_isrf_fastconv,
            apply_tropomi_isrf,
            read_rtm_output,
            vlidort_run,
            cli_options,
            tropomi_print_ring_input,
            print_tropomi_atm,
            print_tropomi_surface_albedo,
            print_tropomi_vga,
            print_tropomi_config,
        )

        osp_tropomi_dir = (
            i_osp_dir / "TROPOMI" if i_osp_dir is not None else Path("../OSP/TROPOMI")
        )
        #
        # Radiative Calculation of TROPOMI FM

        # various CLI options that determine how to run things later
        debug = cli_options.get("debug", False)

        uip_tropomi = self.i_uip["uip_TROPOMI"]

        # Default run directory if not specified.
        default_run_directory = "./"

        # Setup VLIDORT CLI I/O and run it
        iteration = self.i_uip["iteration"]
        cloudy_str = "cloudy" if do_cloud == 1 else "clear"

        vlidort_input_dir = uip_tropomi["vlidort_input"]
        vlidort_input_iter_dir = vlidort_input_dir + "/IterLast/MWLast/cloudy/"
        if debug:
            # This is good for debugging but takes too much space during production
            vlidort_input_iter_dir = (
                vlidort_input_dir
                + f"/Iter{iteration:02d}/MW{ii_mw + 1:03d}/{cloudy_str}/"
            )
        Path(vlidort_input_iter_dir).mkdir(parents=True, exist_ok=True)

        vlidort_output_dir = uip_tropomi["vlidort_output"]
        vlidort_output_iter_dir = vlidort_output_dir + "/IterLast/MWLast/cloudy/"
        if debug:
            # This is good for debugging but takes too much space during production
            vlidort_output_iter_dir = (
                vlidort_output_dir
                + f"/Iter{iteration:02d}/MW{ii_mw + 1:03d}/{cloudy_str}/"
            )
        Path(vlidort_output_iter_dir).mkdir(parents=True, exist_ok=True)

        # VLIDORT options
        vlidort_nstokes = uip_tropomi["vlidort_nstokes"]
        vlidort_nstreams = uip_tropomi["vlidort_nstreams"]

        if cli_options.vlidort:
            if cli_options.vlidort.nstokes:
                vlidort_nstokes = cli_options.vlidort.nstokes

            if cli_options.vlidort.nstreams:
                vlidort_nstreams = cli_options.vlidort.nstreams

        print_tropomi_config(vlidort_input_iter_dir, ii_mw, self.i_uip, fm_nlayers)
        print_tropomi_vga(
            vlidort_input_iter_dir, ii_mw, self.i_uip, self.rayInfo, fm_nlayers
        )
        print_tropomi_surface_albedo(
            vlidort_input_iter_dir, ii_mw, self.i_uip, do_cloud
        )
        print_tropomi_atm(vlidort_input_iter_dir, self.i_uip, self.rayInfo, fm_nlayers)

        # Run VLIDORT CLI
        vlidort_run(
            default_run_directory,
            vlidort_input_iter_dir,
            vlidort_output_iter_dir,
            vlidort_nstokes,
            vlidort_nstreams,
        )

        # IWF = G * dI / dG, where I is a component of the stokes vector (I, Q, U, V) and G is the gas optical depth (O3 in our case)
        # IWF also known as the normalized weighting function
        # The denormalized IWF: IWF_denorm = IWF / G

        # read result files from the RT model
        radiance_matrix = read_rtm_output(vlidort_output_iter_dir, "Radiance.asc")

        # Use the normalized weighting function as provided by VLIDORT
        # MUSES needs the normalized weighting function for species retrieved in log(VMR)
        jacobian_o3_matrix = read_rtm_output(vlidort_output_iter_dir, "IWF.asc")

        # To experiment with the denormalized weighting function, i.e. if you retrieve O3 in VMR, uncomment the line below
        # jacobian_o3_matrix = read_rtm_output(vlidort_output_iter_dir, 'IWF_denorm.asc')

        jacobian_sf_matrix = read_rtm_output(vlidort_output_iter_dir, "surf_WF.asc")

        tropomi_print_ring_input(
            vlidort_input_iter_dir,
            vlidort_output_iter_dir,
            ii_mw,
            self.i_uip,
            self.rayInfo,
            fm_nlayers,
            do_cloud,
            i_obs=i_obs,
        )

        # Copy the RamanInputs directory as well.
        additional_input_dir_1 = "RamanInputs"
        raman_inputs_dir = Path(default_run_directory) / additional_input_dir_1

        if not raman_inputs_dir.exists():
            # Check if the the additional_input_dir_1 exist exist for copying.  If not, exit.
            raman_inputs_osp_dir = osp_tropomi_dir / additional_input_dir_1
            if not raman_inputs_osp_dir.exists():
                logger.error(
                    "Cannot find directory to copy",
                    str(osp_tropomi_dir)
                    + "../OSP/TROPOMI"
                    + os.path.sep
                    + additional_input_dir_1,
                )
                assert False

            # copy RamanInputs dir
            if not skip_raman_copy:
                shutil.copytree(raman_inputs_osp_dir, raman_inputs_dir)
                os.system(f"chmod -R 777 {raman_inputs_dir.as_posix()}")
            else:
                raman_inputs_dir = raman_inputs_osp_dir.absolute().resolve()

            # symlink RamanInputs dir instead of copying it
            # raman_inputs_dir.symlink_to(raman_inputs_osp_dir, target_is_directory=True)
        # end: if not raman_inputs_dir.exists():

        ring_cli = cli_options.get("ring_cli", "")

        # RING CLI
        executable_filename = "ring_cli"
        ring_cli_exe = Path(ring_cli).expanduser().resolve() / executable_filename

        if not ring_cli_exe.exists():
            logger.error("Cannot find executable %s" % ring_cli_exe)
            assert False

        ring_command = [
            ring_cli_exe.as_posix(),
            "--raman-input",
            raman_inputs_dir.as_posix(),
            "--input",
            vlidort_input_iter_dir,
            "--output",
            vlidort_output_iter_dir,
        ]

        logger.debug(f"\nRunning:\n{' '.join(ring_command)} ")

        returnCodeFromSystem = subprocess.run(
            ring_command,
            cwd=default_run_directory,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        if returnCodeFromSystem.returncode != 0:
            logger.error("returnCodeFromSystem", returnCodeFromSystem)
            assert False
        logger.info(returnCodeFromSystem.stdout)

        if radiance_matrix is None:
            raise RuntimeError("Could not read radiance: Radiance.asc")

        if jacobian_o3_matrix is None:
            raise RuntimeError("Could not read jacobian_o3: IWF.asc")

        if jacobian_sf_matrix is None:
            raise RuntimeError("Could not read jacobian_sf: surf_WF.asc")

        ring_matrix = read_rtm_output(vlidort_output_iter_dir, "Ring.asc")
        if ring_matrix is None:
            raise RuntimeError("Could not read ring: Ring.asc")

        my_filter = uip_tropomi["microwindows"][ii_mw]["filter"]

        ils_tropomi_xsection = self.i_uip["ils_tropomi_xsection"]
        ils_tropomi_xsection = ils_tropomi_xsection.upper()

        if ils_tropomi_xsection == "NOAPPLY":
            ils_tropomi_xsection = "POSTCONV"

        # MT: Implementing ILS application
        if self.i_uip["ils_tropomi_xsection"] == "POSTCONV":
            radiance_matrix = apply_tropomi_isrf(self.i_uip, ii_mw, radiance_matrix)
            jacobian_o3_matrix = apply_tropomi_isrf(
                self.i_uip, ii_mw, jacobian_o3_matrix
            )
            jacobian_sf_matrix = apply_tropomi_isrf(
                self.i_uip, ii_mw, jacobian_sf_matrix
            )

        if self.i_uip["ils_tropomi_xsection"] == "FASTCONV":
            radiance_matrix = apply_tropomi_isrf_fastconv(
                self.i_uip, ii_mw, radiance_matrix
            )
            jacobian_o3_matrix = apply_tropomi_isrf_fastconv(
                self.i_uip, ii_mw, jacobian_o3_matrix
            )
            jacobian_sf_matrix = apply_tropomi_isrf_fastconv(
                self.i_uip, ii_mw, jacobian_sf_matrix
            )

        nfreq = (
            uip_tropomi["microwindows"][ii_mw]["enddmw"][ii_mw]
            - uip_tropomi["microwindows"][ii_mw]["startmw"][ii_mw]
            + 1
        )  # from [ 20 194] get 20, from [126 306] get 126 for index 0.
        my_filter = uip_tropomi["microwindows"][ii_mw]["filter"]

        temp_freq_fm = radiance_matrix[0, :]
        temp_freq_ind = np.where(
            (temp_freq_fm >= uip_tropomi["microwindows"][ii_mw]["start"])
            & (temp_freq_fm <= uip_tropomi["microwindows"][ii_mw]["endd"])
        )[0]

        if len(temp_freq_ind) != nfreq:
            logger.error(
                "Number of Data points does not match to the expected values: len(temp_freq_ind), nfreq",
                len(temp_freq_ind),
                nfreq,
            )
            logger.error(
                "uip_tropomi['microwindows'][ii_mw]['startmw']",
                uip_tropomi["microwindows"][ii_mw]["startmw"],
            )
            logger.error(
                "uip_tropomi['microwindows'][ii_mw]['enddmw']",
                uip_tropomi["microwindows"][ii_mw]["enddmw"],
            )
            assert False

        temp_start_ind = self.mw_account["mw_range"][0, ii_mw]
        temp_endd_ind = self.mw_account["mw_range"][1, ii_mw]

        # clear sky condition
        # EM NOTE - The following section is to ensure the storage arrays and the data from VLIDORT are the same size....I think
        if do_cloud == 0:
            if (
                radiance_matrix[1, temp_freq_ind].shape[0]
                < self.radiance_clear_ils[temp_start_ind : temp_endd_ind + 1].shape[0]
            ):
                # ValueError: could not broadcast input array from shape (107) into shape (113)
                # To solve the issue above, we must shrink the left hand side from 113 to 107 to match the shape of radiance_matrix[1,temp_freq_ind] vector.
                shrink_left_hand_size = (
                    temp_start_ind + radiance_matrix[1, temp_freq_ind].shape[0]
                )
                self.radiance_clear_ils[temp_start_ind:shrink_left_hand_size] = (
                    radiance_matrix[1, temp_freq_ind]
                )
            else:
                self.radiance_clear_ils[temp_start_ind : temp_endd_ind + 1] = (
                    radiance_matrix[1, temp_freq_ind]
                )

            if (
                ring_matrix[1, temp_freq_ind].shape[0]
                < self.ring_clear_ils[temp_start_ind : temp_endd_ind + 1].shape[0]
            ):
                # ValueError: could not broadcast input array from shape (107) into shape (113)
                # To solve the issue above, we must shrink the left hand side from 113 to 107 to match to the shape of ring_matrix[1,temp_freq_ind] vector.
                shrink_left_hand_size = (
                    temp_start_ind + ring_matrix[1, temp_freq_ind].shape[0]
                )
                self.ring_clear_ils[temp_start_ind:shrink_left_hand_size] = ring_matrix[
                    1, temp_freq_ind
                ][:]
            else:
                self.ring_clear_ils[temp_start_ind : temp_endd_ind + 1] = ring_matrix[
                    1, temp_freq_ind
                ][:]

            if self.i_uip["num_atm_k"] > 0:
                for ii in range(0, len(self.atm_clear_jacobians_ils)):
                    # ValueError: could not broadcast input array from shape (107,64) into shape (113,64)
                    # To solve the issue above, we must shrink the left hand side from 113 to 107 to match to the shape of self.atm_clear_jacobians_ils[ii]['k']
                    if (
                        np.transpose(jacobian_o3_matrix[1:, temp_freq_ind]).shape[0]
                        < self.atm_clear_jacobians_ils[ii]["k"][
                            temp_start_ind : temp_endd_ind + 1, :
                        ].shape[0]
                    ):
                        shrink_left_hand_size = (
                            temp_start_ind
                            + np.transpose(jacobian_o3_matrix[1:, temp_freq_ind]).shape[
                                0
                            ]
                        )
                        self.atm_clear_jacobians_ils[ii]["k"][
                            temp_start_ind:shrink_left_hand_size, :
                        ] = np.transpose(jacobian_o3_matrix[1:, temp_freq_ind])
                    else:
                        self.atm_clear_jacobians_ils[ii]["k"][
                            temp_start_ind : temp_endd_ind + 1, :
                        ] = np.transpose(jacobian_o3_matrix[1:, temp_freq_ind])[:, :]

            # EM NOTE - Capturing albedo jacobians, have re-written this, so it may not work well
            if (
                jacobian_sf_matrix[1, temp_freq_ind].shape[0]
                < self.jacobian_dictionary[f"surface_albedo_{my_filter}"][
                    temp_start_ind : temp_endd_ind + 1
                ].shape[0]
            ):
                shrink_left_hand_size = (
                    temp_start_ind + jacobian_sf_matrix[1, temp_freq_ind].shape[0]
                )
                self.jacobian_dictionary[f"surface_albedo_{my_filter}"][
                    temp_start_ind:shrink_left_hand_size
                ] = jacobian_sf_matrix[1, temp_freq_ind]
            else:
                self.jacobian_dictionary[f"surface_albedo_{my_filter}"][
                    temp_start_ind : temp_endd_ind + 1
                ] = jacobian_sf_matrix[1, temp_freq_ind]

            if my_filter != "BAND1":
                wave_arr = uip_tropomi["fullbandfrequency"][
                    uip_tropomi["microwindows"][ii_mw]["startmw"][ii_mw] : uip_tropomi[
                        "microwindows"
                    ][ii_mw]["enddmw"][ii_mw]
                    + 1
                ]

                STARTMW_FM = uip_tropomi["microwindows"][ii_mw]["startmw_fm"][ii_mw]
                ENDDMW_FM = uip_tropomi["microwindows"][ii_mw]["enddmw_fm"][ii_mw]

                start_wav = uip_tropomi["fullbandfrequency"][STARTMW_FM]
                endd_wav = uip_tropomi["fullbandfrequency"][ENDDMW_FM]

                ref_wav = (
                    (np.float64(endd_wav) - np.float64(start_wav)) / np.float64(2.0)
                ) + np.float64(start_wav)

                delta_wav = wave_arr[:] - ref_wav

                if (
                    jacobian_sf_matrix[1, temp_freq_ind].shape[0]
                    < self.jacobian_dictionary[f"surface_albedo_slope_{my_filter}"][
                        temp_start_ind : temp_endd_ind + 1
                    ].shape[0]
                ):
                    shrink_left_hand_size = (
                        temp_start_ind + jacobian_sf_matrix[1, temp_freq_ind].shape[0]
                    )
                    self.jacobian_dictionary[f"surface_albedo_slope_{my_filter}"][
                        temp_start_ind:shrink_left_hand_size
                    ] = jacobian_sf_matrix[1, temp_freq_ind] * delta_wav[:]
                    self.jacobian_dictionary[
                        f"surface_albedo_slope_order2_{my_filter}"
                    ][temp_start_ind:shrink_left_hand_size] = (
                        jacobian_sf_matrix[1, temp_freq_ind] * delta_wav[:] ** 2
                    )
                else:
                    self.jacobian_dictionary[f"surface_albedo_slope_{my_filter}"][
                        temp_start_ind : temp_endd_ind + 1
                    ] = jacobian_sf_matrix[1, temp_freq_ind] * delta_wav[:]
                    self.jacobian_dictionary[
                        f"surface_albedo_slope_order2_{my_filter}"
                    ][temp_start_ind : temp_endd_ind + 1] = (
                        jacobian_sf_matrix[1, temp_freq_ind] * delta_wav[:] ** 2
                    )
        # end if do_cloud == 0:

        # cloud sky condition
        if do_cloud == 1:
            # ValueError: could not broadcast input array from shape (107) into shape (113)
            # To solve the issue above, we must shrink the left hand side from 113 to 107 to match to the shape of radiance_matrix[1,temp_freq_ind]
            if (
                radiance_matrix[1, temp_freq_ind].shape[0]
                < self.radiance_cloud_ils[temp_start_ind : temp_endd_ind + 1].shape[0]
            ):
                shrink_left_hand_size = (
                    temp_start_ind + radiance_matrix[1, temp_freq_ind].shape[0]
                )
                self.radiance_cloud_ils[temp_start_ind:shrink_left_hand_size] = (
                    radiance_matrix[1, temp_freq_ind][:]
                )
            else:
                self.radiance_cloud_ils[temp_start_ind : temp_endd_ind + 1] = (
                    radiance_matrix[1, temp_freq_ind][:]
                )

            # ValueError: could not broadcast input array from shape (107) into shape (113)
            # To solve the issue above, we must shrink the left hand side from 113 to 107 to match to the shape of ring_matrix[1,temp_freq_ind]
            if (
                ring_matrix[1, temp_freq_ind].shape[0]
                < self.ring_cloud_ils[temp_start_ind : temp_endd_ind + 1].shape[0]
            ):
                shrink_left_hand_size = (
                    temp_start_ind + ring_matrix[1, temp_freq_ind].shape[0]
                )
                self.ring_cloud_ils[temp_start_ind:shrink_left_hand_size] = ring_matrix[
                    1, temp_freq_ind
                ]
            else:
                self.ring_cloud_ils[temp_start_ind : temp_endd_ind + 1] = ring_matrix[
                    1, temp_freq_ind
                ]

            if self.i_uip["num_atm_k"] > 0:
                if len(self.atm_cloud_jacobians_ils) > 0:
                    for ii in range(0, len(self.atm_cloud_jacobians_ils)):
                        right_hand_side_second_size = np.transpose(
                            jacobian_o3_matrix[1:, temp_freq_ind]
                        ).shape[1]
                        # ValueError: could not broadcast input array from shape (107,60) into shape (113,60)
                        # To solve the issue above, we must shrink the left hand side from 113 to 107 to match to the shape of np.transpose(jacobian_o3_matrix[1:,temp_freq_ind])
                        if np.transpose(jacobian_o3_matrix[1:, temp_freq_ind]).shape[
                            0
                        ] < (temp_endd_ind + 1 - temp_start_ind):
                            shrink_left_hand_size = (
                                temp_start_ind
                                + np.transpose(
                                    jacobian_o3_matrix[1:, temp_freq_ind]
                                ).shape[0]
                            )
                            self.atm_cloud_jacobians_ils[ii]["k"][
                                temp_start_ind:shrink_left_hand_size,
                                0:right_hand_side_second_size,
                            ] = np.transpose(jacobian_o3_matrix[1:, temp_freq_ind])
                        else:
                            self.atm_cloud_jacobians_ils[ii]["k"][
                                temp_start_ind : temp_endd_ind + 1,
                                0:right_hand_side_second_size,
                            ] = np.transpose(jacobian_o3_matrix[1:, temp_freq_ind])[
                                :, :
                            ]

            # EM NOTE - Capturing albedo jacobians, for clouds, this may be only necessary for the TROPOMI implementation due to calibration errors
            if (
                jacobian_sf_matrix[1, temp_freq_ind].shape[0]
                < self.jacobian_dictionary["cloud_Surface_Albedo"][
                    temp_start_ind : temp_endd_ind + 1
                ].shape[0]
            ):
                shrink_left_hand_size = (
                    temp_start_ind + jacobian_sf_matrix[1, temp_freq_ind].shape[0]
                )
                self.jacobian_dictionary["cloud_Surface_Albedo"][
                    temp_start_ind:shrink_left_hand_size
                ] = jacobian_sf_matrix[1, temp_freq_ind]
            else:
                self.jacobian_dictionary["cloud_Surface_Albedo"][
                    temp_start_ind : temp_endd_ind + 1
                ] = jacobian_sf_matrix[1, temp_freq_ind]
        # end if do_cloud:


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
    MusesForwardModelHandle(
        InstrumentIdentifier("TROPOMI"), MusesTropomiForwardModelVlidort
    ),
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

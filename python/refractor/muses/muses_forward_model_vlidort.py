from __future__ import annotations
import refractor.framework as rf  # type: ignore
from .identifier import InstrumentIdentifier
from .forward_model_handle import ForwardModelHandle, ForwardModelHandleSet
from functools import cached_property
from loguru import logger
import tempfile
import numpy as np
import copy
import subprocess
from pathlib import Path
from typing import Any
import typing

if typing.TYPE_CHECKING:
    from .current_state import CurrentState
    from .muses_observation import MeasurementId
    from .retrieval_configuration import RetrievalConfiguration
    from .muses_observation import MusesObservation
    from .cost_function import CostFunction
    from .refractor_fm_object_creator import RefractorFmObjectCreator
    from refractor.muses_py_fm import RefractorUip

# This is a work in progress. We would like to move over and simplify the vlidort
# forward model, and hopefully remove using the UIP etc. But for right now, we
# leverage off of muses-py
#
# Note that this has direct copied of stuff from muses_py_fm/muses_forward_model.py,
# since we want to independent update stuff. This is obviously not desirable long
# term.


class FmUpdateUip(rf.ObserverMaxAPosterioriSqrtConstraint):
    def __init__(self, fm: MusesForwardModelVlidort) -> None:
        super().__init__()
        self.fm = fm

    def notify_update(self, mstand: rf.MaxAPosterioriSqrtConstraint) -> None:
        logger.debug(f"Call to {self.__class__.__name__}::notify_update")
        self.fm.update_uip(mstand.parameters)


class MusesForwardModelVlidort(rf.ForwardModel):
    """Forward model that uses VLIDORT. This matches the existing
    py-retrieve. Note that the version of VLIDORT is older than the
    LIDORT version we use. Also, the code is hardcoded to using O3
    only as an absorber. We could modify this, but it would require
    changing the package muses-vlidort. Also we use a different
    version of the Raman Scattering calculation (ring). This uses
    the stand alone executable (from the the package muses-rrs).
    """

    def __init__(
        self,
        ocreator: RefractorFmObjectCreator,
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
        # TODO We'll pull out the objects we need here, but for now just
        # grab the whole RefractorFmObjectCreator
        self.ocreator = ocreator
        # TODO I think this can probably go away after we clean
        # everything up
        self.is_tropomi = False
        if self.instrument_name == InstrumentIdentifier("TROPOMI"):
            self.is_tropomi = True

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

    def fm_call(self):
        from refractor.muses_py_fm import muses_py_call

        # TODO Should be able to move this down. This is read by the ring_cli
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
        # This should be able to get removed, and fm_call2 just put in here
        with muses_py_call(
            self.rf_uip.run_dir,
            vlidort_nstokes=self.vlidort_nstokes,
            vlidort_nstreams=self.vlidort_nstreams,
        ):
            jac, rad = self.fm_call2()
        return jac, rad

    def fm_call2(self):
        # Temp, we'll pull some of this over and get other parts into mpy
        from refractor.muses_py import (
            get_omi_radiance,
            get_tropomi_radiance,
            raylayer_nadir,
            atmosphere_level,
        )

        from refractor.muses import AttrDictAdapter

        self.i_uip = self.rf_uip.uip_all(self.instrument_name)
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

        self.nlayers = atmparams["nlayers"]
        self.atm_clear_jacobians_ils = None

        if self.i_uip["num_atm_k"] > 0:
            cnt = 0
            k_structure = {
                "species": "thisisadummystring",
                "k": np.zeros(shape=(nfreq_tot, self.nlayers), dtype=np.float64),
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
            self.nlayers_cloud = np.count_nonzero(
                self.rayInfo["pbar"] <= self.i_uip["tropomiPars"]["cloud_pressure"]
            )
        else:
            cloud_pressure = self.i_uip["omiPars"]["cloud_pressure"]
            if cloud_pressure < 0:
                raise RuntimeError(
                    "self.i_uip['omiPars']['cloud_pressure'] < 0. Check the OMI Cloud L2 product used as input for OMI cloud variables."
                )

            self.nlayers_cloud = np.count_nonzero(
                self.rayInfo["pbar"] <= cloud_pressure
            )
        self.atm_cloud_jacobians_ils = None

        if self.i_uip["num_atm_k"] > 0:
            cnt = 0
            k_structure = {
                "species": "thisisadummystring",
                "k": np.zeros(shape=(nfreq_tot, self.nlayers_cloud), dtype=np.float64),
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
                self.i_uip["tropomiPars"], tropomi0=self.obs.radiance_for_uip
            )
        else:
            self.omi_radiance = get_omi_radiance(
                self.i_uip["omiPars"], omi0=self.obs.radiance_for_uip
            )

        # loop over all microwindows
        for self.ii_mw in range(0, self.mw_account["mw_cnt"]):
            # AT_LINE 201 OMI/omi_fm.pro
            self.mw_account["mw_cnt"] = self.ii_mw

            self.mws = self.mw_account["mw_range"][0, self.ii_mw]
            self.mwf = self.mw_account["mw_range"][1, self.ii_mw]

            # * measurement wavelength grid after ILS convolved
            v_ils_mw = self.mw_account["freq"][self.mws : self.mwf + 1]

            if self.ii_mw == 0:
                v_ils_total = v_ils_mw  # get ils convolved frequency grid

            if self.ii_mw > 0:
                v_ils_total = np.concatenate((v_ils_total, v_ils_mw), axis=0)

            logger.info("Calling rtf for clear sky")
            self.rtf(do_cloud=0)

            logger.info("Calling rtf for cloudy sky")
            self.rtf(do_cloud=1)

            #  Revert the Layer in Jacobians;  FM mapping (Layer to Level); and
            #  Combine Cloud/Clear Sky Radiances/Jacobians
            #  Compute jacobians of cloud fraction, ring scaling factor, and
            #  wavelength shift parameters

            if self.is_tropomi:
                self.tropomi_rev_and_fm_map()
            else:
                self.omi_rev_and_fm_map()

        # end for ii_mw in range(0, self.mw_account['mw_cnt']):

        # Pack Radiance and Jacobians.

        (o_radiance_pack, o_jacobian_pack) = self.pack_jacobian()

        # Sanity Check on NAN for radiance and jacobian.
        if not np.all(np.isfinite(o_radiance_pack)):
            raise RuntimeError("o_radiance_pack NOT FINITE!")

        if o_jacobian_pack is not None and not np.all(np.isfinite(o_jacobian_pack)):
            raise RuntimeError("o_jacobian_pack NOT FINITE!")

        return (
            o_jacobian_pack,
            o_radiance_pack,
        )

    def rtf(
        self,
        do_cloud,
    ):
        from refractor.muses_py import (
            apply_omi_isrf_fast,
            apply_omi_isrf_slow,
            apply_omi_srf,
            cli_options,
            print_ring_input,
            vlidort_run,
            print_omi_surface_albedo,
            print_omi_o3od,
            print_omi_atm,
            print_omi_vga,
            print_omi_config,
            apply_tropomi_isrf_fastconv,
            apply_tropomi_isrf,
            read_rtm_output,
            tropomi_print_ring_input,
            print_tropomi_atm,
            print_tropomi_surface_albedo,
            print_tropomi_vga,
            print_tropomi_config,
        )

        # Default run directory if not specified.
        default_run_directory = "./"

        vlidort_input_dir = self.i_uip["vlidort_input"]
        vlidort_input_iter_dir = vlidort_input_dir + "/IterLast/MWLast/cloudy/"
        Path(vlidort_input_iter_dir).mkdir(parents=True, exist_ok=True)
        vlidort_output_dir = self.i_uip["vlidort_output"]
        vlidort_output_iter_dir = vlidort_output_dir + "/IterLast/MWLast/cloudy/"
        Path(vlidort_output_iter_dir).mkdir(parents=True, exist_ok=True)

        fm_nlayers = self.nlayers_cloud if do_cloud else self.nlayers
        if self.is_tropomi:
            print_tropomi_config(
                vlidort_input_iter_dir, self.ii_mw, self.i_uip, fm_nlayers
            )
            print_tropomi_vga(
                vlidort_input_iter_dir, self.ii_mw, self.i_uip, self.rayInfo, fm_nlayers
            )
            print_tropomi_surface_albedo(
                vlidort_input_iter_dir, self.ii_mw, self.i_uip, do_cloud
            )
            print_tropomi_atm(
                vlidort_input_iter_dir, self.i_uip, self.rayInfo, fm_nlayers
            )
        else:
            print_omi_config(vlidort_input_iter_dir, self.ii_mw, self.i_uip, fm_nlayers)
            print_omi_vga(vlidort_input_iter_dir, self.ii_mw, self.i_uip, self.rayInfo)
            print_omi_atm(vlidort_input_iter_dir, self.i_uip, self.rayInfo, fm_nlayers)
            print_omi_o3od(vlidort_input_iter_dir, self.i_uip, self.rayInfo)
            print_omi_surface_albedo(
                vlidort_input_iter_dir, self.ii_mw, self.i_uip, do_cloud
            )

        # Run VLIDORT CLI
        vlidort_run(
            default_run_directory,
            vlidort_input_iter_dir,
            vlidort_output_iter_dir,
            self.vlidort_nstokes,
            self.vlidort_nstreams,
        )
        # IWF = G * dI / dG, where I is a component of the stokes vector (I, Q, U, V) and G is the gas optical depth (O3 in our case)
        # IWF also known as the normalized weighting function
        # The denormalized IWF: IWF_denorm = IWF / G

        # read result files from the RT model
        radiance_matrix = read_rtm_output(vlidort_output_iter_dir, "Radiance.asc")

        # Use the normalized weighting function as provided by VLIDORT
        # MUSES needs the normalized weighting function for species
        # retrieved in log(VMR)
        jacobian_o3_matrix = read_rtm_output(vlidort_output_iter_dir, "IWF.asc")

        # To experiment with the denormalized weighting function,
        # i.e. if you retrieve O3 in VMR, uncomment the line below
        # jacobian_o3_matrix =
        # read_rtm_output(vlidort_output_iter_dir, 'IWF_denorm.asc')

        jacobian_sf_matrix = read_rtm_output(vlidort_output_iter_dir, "surf_WF.asc")

        if self.is_tropomi:
            tropomi_print_ring_input(
                vlidort_input_iter_dir,
                vlidort_output_iter_dir,
                self.ii_mw,
                self.i_uip,
                self.rayInfo,
                fm_nlayers,
                do_cloud,
                i_obs=self.obs.radiance_for_uip,
            )
        else:
            print_ring_input(
                vlidort_input_iter_dir,
                vlidort_output_iter_dir,
                self.ii_mw,
                self.i_uip,
                self.rayInfo,
                fm_nlayers,
                do_cloud,
                i_obs=self.obs.radiance_for_uip,
            )

        raman_inputs_dir = self.rconf.input_file_helper.osp_dir / "OMI" / "RamanInputs"

        ring_cli = cli_options.get("ring_cli", "")

        # RING CLI
        executable_filename = "ring_cli"
        ring_cli_exe = Path(ring_cli).expanduser().resolve() / executable_filename

        if not ring_cli_exe.exists():
            raise RuntimeError(f"Cannot find executable {ring_cli_exe}")

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

        subprocess.run(
            ring_command,
            cwd=default_run_directory,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True,
        )

        if radiance_matrix is None:
            raise RuntimeError("Could not read radiance: Radiance.asc")

        if jacobian_o3_matrix is None:
            raise RuntimeError("Could not read jacobian_o3: IWF.asc")

        if jacobian_sf_matrix is None:
            raise RuntimeError("Could not read jacobian_sf: surf_WF.asc")

        ring_matrix = read_rtm_output(vlidort_output_iter_dir, "Ring.asc")
        if ring_matrix is None:
            raise RuntimeError("Could not read ring: Ring.asc")

        my_filter = self.i_uip["microwindows"][self.ii_mw]["filter"]

        # Note I don't think the ILS actually works. We have this
        # copied from py-retrieve, where I don't believe it works there.
        # But copy over, if nothing else this should give a starting point
        # for fixing if needed.
        if self.is_tropomi:
            ils_tropomi_xsection = self.i_uip["ils_tropomi_xsection"]
            ils_tropomi_xsection = ils_tropomi_xsection.upper()
            if ils_tropomi_xsection == "NOAPPLY":
                ils_tropomi_xsection = "POSTCONV"

            # MT: Implementing ILS application
            if self.i_uip["ils_tropomi_xsection"] == "POSTCONV":
                radiance_matrix = apply_tropomi_isrf(
                    self.i_uip, self.ii_mw, radiance_matrix
                )
                jacobian_o3_matrix = apply_tropomi_isrf(
                    self.i_uip, self.ii_mw, jacobian_o3_matrix
                )
                jacobian_sf_matrix = apply_tropomi_isrf(
                    self.i_uip, self.ii_mw, jacobian_sf_matrix
                )

            if self.i_uip["ils_tropomi_xsection"] == "FASTCONV":
                radiance_matrix = apply_tropomi_isrf_fastconv(
                    self.i_uip, self.ii_mw, radiance_matrix
                )
                jacobian_o3_matrix = apply_tropomi_isrf_fastconv(
                    self.i_uip, self.ii_mw, jacobian_o3_matrix
                )
                jacobian_sf_matrix = apply_tropomi_isrf_fastconv(
                    self.i_uip, self.ii_mw, jacobian_sf_matrix
                )
        else:
            ils_omi_xsection = self.i_uip["ils_omi_xsection"]
            ils_omi_xsection.upper()
            if ils_omi_xsection == "NOAPPLY":
                ils_omi_xsection = "POSTCONV"

            if ils_omi_xsection == "POSTCONV":
                radiance_matrix = apply_omi_srf(
                    self.i_uip, self.ii_mw, radiance_matrix, self.omi_radiance
                )
                jacobian_o3_matrix = apply_omi_srf(
                    self.i_uip, self.ii_mw, jacobian_o3_matrix, self.omi_radiance
                )
                jacobian_sf_matrix = apply_omi_srf(
                    self.i_uip, self.ii_mw, jacobian_sf_matrix, self.omi_radiance
                )

                # TODO: VK: Verify. Not sure about this. IDL convolves the optical depth
                # ring_matrix = apply_omi_srf(self.i_uip, ii_mw, ring_matrix, omi_info)
            # end: if ils_omi_xsection == 'POSTCONV':

            if ils_omi_xsection == "SLOWCONV":
                radiance_matrix = apply_omi_isrf_slow(
                    self.i_uip, self.ii_mw, radiance_matrix
                )
                jacobian_o3_matrix = apply_omi_isrf_slow(
                    self.i_uip, self.ii_mw, jacobian_o3_matrix
                )
                jacobian_sf_matrix = apply_omi_isrf_slow(
                    self.i_uip, self.ii_mw, jacobian_sf_matrix
                )

            if ils_omi_xsection == "FASTCONV":
                radiance_matrix = apply_omi_isrf_fast(
                    self.i_uip, self.ii_mw, radiance_matrix
                )
                jacobian_o3_matrix = apply_omi_isrf_fast(
                    self.i_uip, self.ii_mw, jacobian_o3_matrix
                )
                jacobian_sf_matrix = apply_omi_isrf_fast(
                    self.i_uip, self.ii_mw, jacobian_sf_matrix
                )

        nfreq = (
            self.i_uip["microwindows"][self.ii_mw]["enddmw"][self.ii_mw]
            - self.i_uip["microwindows"][self.ii_mw]["startmw"][self.ii_mw]
            + 1
        )  # from [ 20 194] get 20, from [126 306] get 126 for index 0.
        my_filter = self.i_uip["microwindows"][self.ii_mw]["filter"]

        temp_freq_fm = radiance_matrix[0, :]
        temp_freq_ind = np.where(
            (temp_freq_fm >= self.i_uip["microwindows"][self.ii_mw]["start"])
            & (temp_freq_fm <= self.i_uip["microwindows"][self.ii_mw]["endd"])
        )[0]

        if len(temp_freq_ind) != nfreq:
            logger.error(
                "Number of Data points does not match to the expected values: len(temp_freq_ind), nfreq",
                len(temp_freq_ind),
                nfreq,
            )
            logger.error(
                "self.i_uip['microwindows'][self.ii_mw]['startmw' ]",
                self.i_uip["microwindows"][self.ii_mw]["startmw"],
            )
            logger.error(
                "self.i_uip['microwindows'][self.ii_mw]['enddmw' ]",
                self.i_uip["microwindows"][self.ii_mw]["enddmw"],
            )
            assert False

        temp_start_ind = self.mw_account["mw_range"][0, self.ii_mw]
        temp_endd_ind = self.mw_account["mw_range"][1, self.ii_mw]

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

            if self.is_tropomi:
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
                    wave_arr = self.i_uip["fullbandfrequency"][
                        self.i_uip["microwindows"][self.ii_mw]["startmw"][
                            self.ii_mw
                        ] : self.i_uip["microwindows"][self.ii_mw]["enddmw"][self.ii_mw]
                        + 1
                    ]

                    STARTMW_FM = self.i_uip["microwindows"][self.ii_mw]["startmw_fm"][
                        self.ii_mw
                    ]
                    ENDDMW_FM = self.i_uip["microwindows"][self.ii_mw]["enddmw_fm"][
                        self.ii_mw
                    ]

                    start_wav = self.i_uip["fullbandfrequency"][STARTMW_FM]
                    endd_wav = self.i_uip["fullbandfrequency"][ENDDMW_FM]

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
                            temp_start_ind
                            + jacobian_sf_matrix[1, temp_freq_ind].shape[0]
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
            else:
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
                            temp_start_ind
                            + jacobian_sf_matrix[1, temp_freq_ind].shape[0]
                        )
                        self.jacobian_dictionary["jacobian_OMISURFACEALBEDOUV2"][
                            temp_start_ind:shrink_left_hand_size
                        ] = jacobian_sf_matrix[1, temp_freq_ind]
                    else:
                        self.jacobian_dictionary["jacobian_OMISURFACEALBEDOUV2"][
                            temp_start_ind : temp_endd_ind + 1
                        ] = jacobian_sf_matrix[1, temp_freq_ind]

                    ref_wav = np.float64(320.0)
                    wave_arr = self.i_uip["fullbandfrequency"][
                        self.i_uip["microwindows"][self.ii_mw]["startmw"][
                            self.ii_mw
                        ] : self.i_uip["microwindows"][self.ii_mw]["enddmw"][self.ii_mw]
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
                            temp_start_ind
                            + jacobian_sf_matrix[1, temp_freq_ind].shape[0]
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
            if self.is_tropomi:
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

    def tropomi_rev_and_fm_map(
        self,
    ):
        # Functionality:
        # 1. Applied any FM Maping e.g., Layer-to-Level
        # 2. Combined cloud and clear sky radiances/jacobians
        uip_tropomi = self.i_uip["uip_TROPOMI"]
        uip_tropomi_pars = self.i_uip["tropomiPars"]

        STARTMW = uip_tropomi["microwindows"][self.ii_mw]["startmw"][self.ii_mw]
        ENDDMW = uip_tropomi["microwindows"][self.ii_mw]["enddmw"][self.ii_mw]

        wave_arr = uip_tropomi["fullbandfrequency"][STARTMW : ENDDMW + 1]

        # For calculating continuum shift
        STARTMW_FM = uip_tropomi["microwindows"][self.ii_mw]["startmw_fm"][self.ii_mw]
        ENDDMW_FM = uip_tropomi["microwindows"][self.ii_mw]["enddmw_fm"][self.ii_mw]

        start_wav = uip_tropomi["fullbandfrequency"][STARTMW_FM]
        endd_wav = uip_tropomi["fullbandfrequency"][ENDDMW_FM]

        ref_wav = (
            (np.float64(endd_wav) - np.float64(start_wav)) / np.float64(2.0)
        ) + np.float64(start_wav)
        delta_wav = float(1.0) - (wave_arr[:] / ref_wav)

        cloud_cf = np.float64(uip_tropomi_pars["cloud_fraction"])

        # add ring to simulated radiances
        # Ring is only relevent to bands 1, 2 and 3 of TROPOMI....maybe others, will check
        my_filter = uip_tropomi["microwindows"][self.ii_mw]["filter"]
        if my_filter in ["BAND1", "BAND2", "BAND3"]:
            # EM addtional fit for poor calibration, only applied if called
            temp_Rext = (
                np.float64(uip_tropomi_pars[f"resscale_O0_{my_filter}"])
                + np.float64(uip_tropomi_pars[f"resscale_O1_{my_filter}"]) * delta_wav
                + np.float64(uip_tropomi_pars[f"resscale_O2_{my_filter}"])
                * delta_wav**2
            )

            temp_Rext_p = temp_Rext + np.float64(0.001)

            # clear
            temp_val = np.float64(uip_tropomi_pars[f"ring_sf_{my_filter}"])
            temp_val_p = temp_val + np.float64(0.001)

            temp_ring = self.ring_clear_ils[
                self.mws : self.mwf + 1
            ] * temp_val + np.float64(1.0)
            temp_ring_p = self.ring_clear_ils[
                self.mws : self.mwf + 1
            ] * temp_val_p + np.float64(1.0)

            self.radiance_clear_ils[self.mws : self.mwf + 1] = (
                self.radiance_clear_ils[self.mws : self.mwf + 1]
                * temp_ring[:]
                * temp_Rext[:]
            )
            self.radiance_clear_ils_p = (
                self.radiance_clear_ils[self.mws : self.mwf + 1]
                * temp_ring_p[:]
                * temp_Rext_p[:]
            )

            # cloudy
            temp_val = np.float64(uip_tropomi_pars[f"ring_sf_{my_filter}"])
            temp_val_p = temp_val + np.float64(0.001)

            temp_ring = self.ring_cloud_ils[
                self.mws : self.mwf + 1
            ] * temp_val + np.float64(1.0)
            temp_ring_p = self.ring_cloud_ils[
                self.mws : self.mwf + 1
            ] * temp_val_p + np.float64(1.0)

            self.radiance_cloud_ils[self.mws : self.mwf + 1] = (
                self.radiance_cloud_ils[self.mws : self.mwf + 1]
                * temp_ring[:]
                * temp_Rext[:]
            )
            self.radiance_cloud_ils_p = (
                self.radiance_cloud_ils[self.mws : self.mwf + 1]
                * temp_ring_p[:]
                * temp_Rext_p[:]
            )

            # combined cloud/clear sky radiances
            self.radiance_ils[self.mws : self.mwf + 1] = self.radiance_cloud_ils[
                self.mws : self.mwf + 1
            ] * cloud_cf + self.radiance_clear_ils[self.mws : self.mwf + 1] * (
                np.float64(1.0) - cloud_cf
            )
            self.radiance_ils_p = (
                self.radiance_cloud_ils_p * cloud_cf
                + self.radiance_clear_ils_p * (np.float64(1.0) - cloud_cf)
            )

            dI = self.radiance_ils_p[:] - self.radiance_ils[self.mws : self.mwf + 1]
            dX = np.float64(0.001)

            # compute continuum scaling jacobians
            self.jacobian_dictionary[f"resscale_O0_{my_filter}"][
                self.mws : self.mwf + 1
            ] = (dI / dX) / delta_wav
            self.jacobian_dictionary[f"resscale_O1_{my_filter}"][
                self.mws : self.mwf + 1
            ] = dI / dX
            self.jacobian_dictionary[f"resscale_O2_{my_filter}"][
                self.mws : self.mwf + 1
            ] = (dI / dX) * delta_wav

            # compute cloud fraction jacobian
            self.jacobian_dictionary["jacobian_cloud_ils"][self.mws : self.mwf + 1] = (
                self.radiance_cloud_ils[self.mws : self.mwf + 1]
                - self.radiance_clear_ils[self.mws : self.mwf + 1]
            )

            # compute Ring Scaling Factor jacobians
            self.jacobian_dictionary[f"ring_sf_{my_filter}"][
                self.mws : self.mwf + 1
            ] = dI / dX

            # Calculate solar shift jacobians
            # EM NOTE - not sure if this is necessary for all bands, so putting into 1, 2 and 3 for the time being
            temp_jac = self.tropomi_radiance["normwav_jac"][uip_tropomi["freqIndex"]]
            self.jacobian_dictionary[f"solarshift_{my_filter}"][
                self.mws : self.mwf + 1
            ] = temp_jac[self.mws : self.mwf + 1]

            # compute radiance/irradiance wavelength shift jacobians
            # EM - NOTE This is possibly useful for other bands as well, will change if necessary
            temp_jac = self.tropomi_radiance["odwav_jac"][uip_tropomi["freqIndex"]]
            self.jacobian_dictionary[f"radianceshift_{my_filter}"][
                self.mws : self.mwf + 1
            ] = temp_jac[self.mws : self.mwf + 1]

            # compute radiance/irradiance wavelength squeeze jacobians
            # EM NOTE - Although this is present, the standard omi implementation doesn't use these.
            # They may also be useful for other bands, but will cross that bridge later.
            temp_jac = self.tropomi_radiance["odwav_slope_jac"][
                uip_tropomi["freqIndex"]
            ]
            self.jacobian_dictionary[f"radsqueeze_{my_filter}"][
                self.mws : self.mwf + 1
            ] = temp_jac[self.mws : self.mwf + 1]
        # end: if my_filter in ['BAND1', 'BAND2', 'BAND3']:

        # EM NOTE - Other bands will have other parameters to fit, e.g. BANDS 7-8 will need aerosol scattering, these
        # must be accounted for here in the future.

        # reverse layer index of jacobians for atmospheric species
        # map them to the forward model levels
        if self.i_uip["num_atm_k"] > 0:
            for ii_sp in range(0, self.i_uip["num_atm_k"]):
                k_clear = copy.deepcopy(
                    self.atm_clear_jacobians_ils[ii_sp]["k"][self.mws : self.mwf + 1, :]
                )  # Make a copy so as not to disturb original matrix.
                k_cloud = copy.deepcopy(
                    self.atm_cloud_jacobians_ils[ii_sp]["k"][self.mws : self.mwf + 1, :]
                )  # Make a copy so as not to disturb original matrix.

                temp_array_k_jac = (
                    k_clear - k_clear
                )  # This clear out a matrix of shape k_clear.shape
                for ilay in range(0, self.nlayers):
                    # layers above cloud top
                    if ilay <= self.nlayers_cloud - 1:
                        temp_array_k_jac[:, ilay] = (
                            k_clear[:, ilay] * (1.0 - cloud_cf)
                            + k_cloud[:, ilay] * cloud_cf
                        )

                    # layers below cloud top
                    if ilay > self.nlayers_cloud - 1:
                        temp_array_k_jac[:, ilay] = k_clear[:, ilay] * (1.0 - cloud_cf)
                # end for ilay in range(0,self.nlayers):

                # in order to flip the order of jacobians so that the first row is at surface
                temp_jac = (
                    temp_array_k_jac - temp_array_k_jac
                )  # This clear out a matrix of shape temp_array_k_jac.shape to zeros.
                for ilay in range(0, self.nlayers):
                    temp_jac[:, ilay] = temp_array_k_jac[:, self.nlayers - ilay - 1]
                # end for ilay in range(0,self.nlayers):

                species_k = self.atm_clear_jacobians_ils[ii_sp]["species"]
                self.jacobians_atm_ils["k_species"][ii_sp]["species"] = species_k
                temp_array = temp_jac  # Because temp_array will be used on the right hand side, we can do the assignment and not a deepcopy.

                ## Only compare the first layer
                species_as_np_array = np.asarray(self.i_uip["species"])

                # Loop through all layers to do comparison before the multiplication.
                for jj_layer in range(0, self.nlayers):
                    uu = np.where(species_as_np_array == species_k)[0]
                # end for jj_layer in range(0,self.nlayers):

                for jj_layer in range(0, self.nlayers):
                    uu = np.where(species_as_np_array == species_k)[0]
                    if len(uu) > 0:
                        self.jacobians_atm_ils["k_species"][ii_sp]["k"][
                            jj_layer, self.mws : self.mwf + 1
                        ] = (
                            self.jacobians_atm_ils["k_species"][ii_sp]["k"][
                                jj_layer, self.mws : self.mwf + 1
                            ]
                            + temp_array[:, jj_layer]
                            * self.rayInfo["map_vmr_l"][uu[0], jj_layer]
                        )

                        self.jacobians_atm_ils["k_species"][ii_sp]["k"][
                            jj_layer + 1, self.mws : self.mwf + 1
                        ] = (
                            self.jacobians_atm_ils["k_species"][ii_sp]["k"][
                                jj_layer + 1, self.mws : self.mwf + 1
                            ]
                            + temp_array[:, jj_layer]
                            * self.rayInfo["map_vmr_u"][uu[0], jj_layer]
                        )
                    # end if len(uu) > 0:
                # end for jj_layer in range(0,self.nlayers):
            # end for ii_sp in range(0,self.i_uip['num_atm_k']):
        # end if (self.i_uip['num_atm_k'] > 0):

        # AT_LINE 129 OMI/rev_and_fm_map.pro
        # EM NOTE - What is this??
        if wave_arr[0] > 335.0:
            # The array self.jacobians_atm_ils can be None so we look to see if self.i_uip['num_atm_k'] is more than 0 before accessing it.
            if self.i_uip["num_atm_k"] > 0:
                self.jacobians_atm_ils["k_species"][:]["k"][
                    :, self.mws : self.mwf + 1
                ] = np.float64(0.0)

    def omi_rev_and_fm_map(self):
        # Description: simulate omi radiances and jacobians

        uip_omi = self.i_uip["uip_OMI"]
        uip_omi_pars = self.i_uip["omiPars"]

        # Functionality:
        # 1. Applied any FM Maping e.g., Layer-to-Level
        # 2. Combined cloud and clear sky radiances/jacobians

        # Create some float64 values we may need.
        ring_sf_uv1 = np.float64(uip_omi_pars["ring_sf_uv1"])
        ring_sf_uv2 = np.float64(uip_omi_pars["ring_sf_uv2"])
        cloud_cf = np.float64(uip_omi_pars["cloud_fraction"])

        # AT_LINE 27 OMI/rev_and_fm_map.pro
        startmw = uip_omi["microwindows"][self.ii_mw]["startmw"][self.ii_mw]
        enddmw = uip_omi["microwindows"][self.ii_mw]["enddmw"][self.ii_mw]

        wave_arr = uip_omi["fullbandfrequency"][startmw : enddmw + 1]

        my_filter = uip_omi["microwindows"][self.ii_mw]["filter"]

        # AT_LINE 32 OMI/rev_and_fm_map.pro
        # add ring to simulated radiances
        if my_filter == "UV1":
            temp_ring = (
                np.float64(1.0)
                + self.ring_clear_ils[self.mws : self.mwf + 1] * ring_sf_uv1
            )
            temp_ring_p = np.float64(1.0) + self.ring_clear_ils[
                self.mws : self.mwf + 1
            ] * (ring_sf_uv1 + np.float64(0.001))

            self.radiance_clear_ils[self.mws : self.mwf + 1] = (
                self.radiance_clear_ils[self.mws : self.mwf + 1] * temp_ring[:]
            )
            self.radiance_clear_ils_p = (
                self.radiance_clear_ils[self.mws : self.mwf + 1] * temp_ring_p[:]
            )

            temp_ring = (
                np.float64(1.0)
                + self.ring_cloud_ils[self.mws : self.mwf + 1] * ring_sf_uv1
            )
            temp_ring_p = np.float64(1.0) + self.ring_cloud_ils[
                self.mws : self.mwf + 1
            ] * (ring_sf_uv1 + np.float64(0.001))

            self.radiance_cloud_ils[self.mws : self.mwf + 1] = (
                self.radiance_cloud_ils[self.mws : self.mwf + 1] * temp_ring[:]
            )
            self.radiance_cloud_ils_p = (
                self.radiance_cloud_ils[self.mws : self.mwf + 1] * temp_ring_p[:]
            )
        # end if my_filter == 'UV1':

        # AT_LINE 46 OMI/rev_and_fm_map.pro
        if my_filter == "UV2":
            temp_ring = (
                np.float64(1.0)
                + self.ring_clear_ils[self.mws : self.mwf + 1] * ring_sf_uv2
            )
            temp_ring_p = np.float64(1.0) + self.ring_clear_ils[
                self.mws : self.mwf + 1
            ] * (ring_sf_uv2 + np.float64(0.001))

            self.radiance_clear_ils[self.mws : self.mwf + 1] = (
                self.radiance_clear_ils[self.mws : self.mwf + 1] * temp_ring[:]
            )
            self.radiance_clear_ils_p = (
                self.radiance_clear_ils[self.mws : self.mwf + 1] * temp_ring_p[:]
            )

            temp_ring = (
                np.float64(1.0)
                + self.ring_cloud_ils[self.mws : self.mwf + 1] * ring_sf_uv2
            )
            temp_ring_p = np.float64(1.0) + self.ring_cloud_ils[
                self.mws : self.mwf + 1
            ] * (ring_sf_uv2 + np.float64(0.001))

            self.radiance_cloud_ils[self.mws : self.mwf + 1] = (
                self.radiance_cloud_ils[self.mws : self.mwf + 1] * temp_ring[:]
            )
            self.radiance_cloud_ils_p = (
                self.radiance_cloud_ils[self.mws : self.mwf + 1] * temp_ring_p[:]
            )
        # end if my_filter == 'UV2':

        # AT_LINE 60 OMI/rev_and_fm_map.pro
        # combined cloud/clear sky radiances

        # see:
        # https://acp.copernicus.org/articles/11/7155/2011/acp-11-7155-2011.pdf
        # 2.2 Analytical formulation
        self.radiance_ils[self.mws : self.mwf + 1] = self.radiance_cloud_ils[
            self.mws : self.mwf + 1
        ] * cloud_cf + self.radiance_clear_ils[self.mws : self.mwf + 1] * (
            1.0 - cloud_cf
        )

        self.radiance_ils_p = (
            self.radiance_cloud_ils_p * cloud_cf
            + self.radiance_clear_ils_p * (1.0 - cloud_cf)
        )

        # compute cloud fraction jacobian
        self.jacobian_dictionary["jacobian_cloud_ils"][self.mws : self.mwf + 1] = (
            self.radiance_cloud_ils[self.mws : self.mwf + 1]
            - self.radiance_clear_ils[self.mws : self.mwf + 1]
        )

        dI = self.radiance_ils_p[:] - self.radiance_ils[self.mws : self.mwf + 1]
        dX = np.float64(0.001)

        # AT_LINE 67 OMI/rev_and_fm_map.pro
        # compute Ring Scaling Factor jacobians
        if my_filter == "UV1":
            self.jacobian_dictionary["jacobian_ring_sf_ils_uv1"][
                self.mws : self.mwf + 1
            ] = dI / dX
        if my_filter == "UV2":
            self.jacobian_dictionary["jacobian_ring_sf_ils_uv2"][
                self.mws : self.mwf + 1
            ] = dI / dX

        # AT_LINE 72 OMI/rev_and_fm_map.pro
        # compute normrad wavelength shift jacobians
        temp_jac = self.omi_radiance["normwav_jac"][uip_omi["freqIndex"]]
        if my_filter == "UV1":
            self.jacobian_dictionary["jacobian_nradwav_ils_uv1"][
                self.mws : self.mwf + 1
            ] = temp_jac[self.mws : self.mwf + 1]
        if my_filter == "UV2":
            self.jacobian_dictionary["jacobian_nradwav_ils_uv2"][
                self.mws : self.mwf + 1
            ] = temp_jac[self.mws : self.mwf + 1]

        # AT_LINE 77 OMI/rev_and_fm_map.pro
        # compute OD wavelength shift jacobians
        temp_jac = self.omi_radiance["odwav_jac"][uip_omi["freqIndex"]]
        if my_filter == "UV1":
            self.jacobian_dictionary["jacobian_odwav_ils_uv1"][
                self.mws : self.mwf + 1
            ] = temp_jac[self.mws : self.mwf + 1]
        if my_filter == "UV2":
            self.jacobian_dictionary["jacobian_odwav_ils_uv2"][
                self.mws : self.mwf + 1
            ] = temp_jac[self.mws : self.mwf + 1]

        # AT_LINE 82 OMI/rev_and_fm_map.pro
        # compute OD wavelength shift slope jacobians
        temp_jac = self.omi_radiance["odwav_slope_jac"][uip_omi["freqIndex"]]
        if my_filter == "UV1":
            self.jacobian_dictionary["jacobian_odwav_slope_ils_uv1"][
                self.mws : self.mwf + 1
            ] = temp_jac[self.mws : self.mwf + 1]
        if my_filter == "UV2":
            self.jacobian_dictionary["jacobian_odwav_slope_ils_uv2"][
                self.mws : self.mwf + 1
            ] = temp_jac[self.mws : self.mwf + 1]

        # AT_LINE 87 OMI/rev_and_fm_map.pro
        # reverse layer index of jacobians for atmospheric species
        # map them to the forward model levels
        if self.i_uip["num_atm_k"] > 0:
            for ii_sp in range(0, self.i_uip["num_atm_k"]):
                # AT_LINE 93 OMI/rev_and_fm_map.pro
                k_clear = copy.deepcopy(
                    self.atm_clear_jacobians_ils[ii_sp]["k"][self.mws : self.mwf + 1, :]
                )
                k_cloud = copy.deepcopy(
                    self.atm_cloud_jacobians_ils[ii_sp]["k"][self.mws : self.mwf + 1, :]
                )

                temp_array_k_jac = (
                    k_clear - k_clear
                )  # This clear out a matrix of shape k_clear.shape
                for ilay in range(0, self.nlayers):
                    # layers above cloud top
                    if ilay <= self.nlayers_cloud - 1:
                        temp_array_k_jac[:, ilay] = (
                            cloud_cf * k_cloud[:, ilay]
                            + (1.0 - cloud_cf) * k_clear[:, ilay]
                        )

                    # layers below cloud top
                    if ilay > self.nlayers_cloud - 1:
                        temp_array_k_jac[:, ilay] = (1.0 - cloud_cf) * k_clear[:, ilay]
                # end for ilay in range(0,self.nlayers):

                # AT_LINE 104 OMI/rev_and_fm_map.pro
                # Flip the order of jacobians so that the first row is at surface
                temp_jac = (
                    temp_array_k_jac - temp_array_k_jac
                )  # This clear out a matrix of shape temp_array_k_jac.shape to zeros.
                for ilay in range(0, self.nlayers):
                    temp_jac[:, ilay] = temp_array_k_jac[:, self.nlayers - ilay - 1]
                # end for ilay in range(0,self.nlayers):

                self.jacobians_atm_ils["k_species"][ii_sp]["species"] = (
                    self.atm_clear_jacobians_ils[ii_sp]["species"]
                )

                ## Only compare the first layer
                # AT_LINE 114 OMI/rev_and_fm_map.pro
                species_as_np_array = np.asarray(self.i_uip["species"])
                species_k = self.jacobians_atm_ils["k_species"][ii_sp]["species"]
                uu = np.where(species_as_np_array == species_k)[0]

                if len(uu) > 0:
                    # see: https://www.sciencedirect.com/science/article/pii/S0022407314001277
                    # 2.4 Transformation rules for level-atmosphere Jacobians (18)
                    for jj_layer in range(0, self.nlayers):
                        jac_ii_sp = self.jacobians_atm_ils["k_species"][ii_sp]["k"]
                        jac_ii_sp[jj_layer, self.mws : self.mwf + 1] = (
                            jac_ii_sp[jj_layer, self.mws : self.mwf + 1]
                            + temp_jac[:, jj_layer]
                            * self.rayInfo["map_vmr_l"][uu[0], jj_layer]
                        )

                        jac_ii_sp[jj_layer + 1, self.mws : self.mwf + 1] = (
                            jac_ii_sp[jj_layer + 1, self.mws : self.mwf + 1]
                            + temp_jac[:, jj_layer]
                            * self.rayInfo["map_vmr_u"][uu[0], jj_layer]
                        )
                    # end for jj_layer in range(0,self.nlayers):
                # end if len(uu) > 0:
            # end for ii_sp in range(0,self.i_uip['num_atm_k']):
        # end if (self.i_uip['num_atm_k'] > 0):

        # AT_LINE 129 OMI/rev_and_fm_map.pro
        if wave_arr[0] > 335.0:
            # The array self.jacobians_atm_ils can be None so we look to see if self.i_uip['num_atm_k'] is more than 0 before accessing it.
            if self.i_uip["num_atm_k"] > 0:
                self.jacobians_atm_ils["k_species"][:]["k"][
                    :, self.mws : self.mwf + 1
                ] = np.float64(0.0)

    def pack_jacobian(self):
        o_radiance_pack = self.radiance_ils[:]

        my_filter = self.i_uip["microwindows"][self.ii_mw]["filter"]

        # Initialization of parameters
        num_rad = len(self.radiance_ils)
        num_elem = len(o_radiance_pack)
        num_par = 0
        num_atm = len(self.i_uip["atmosphere"][0, :])

        # Count number of parameters
        if self.i_uip["num_atm_k"] > 0:
            num_par = self.i_uip["num_atm_k"] * num_atm

        # Add number of jacobians
        num_par = num_par + len(self.i_uip["jacobians"])

        #  Pack Jacobians
        if num_par <= 0:
            o_jacobian_pack = None
        else:
            o_jacobian_pack = np.zeros(
                shape=(num_par, num_elem), dtype=np.float64
            )  # This is our output jacobian
            ii_par = 0

            found_jacobian_flag = False  # Use this flag to make sure each species jacobian has been looked for in the for loop.

            for ii in range(0, len(self.i_uip["jacobians"])):
                # Pack CloudFraction jac
                jacob_name = self.i_uip["jacobians"][ii]
                if jacob_name == "TROPOMICLOUDFRACTION":
                    o_jacobian_pack[ii_par, :] = self.jacobian_dictionary[
                        "jacobian_cloud_ils"
                    ][:]
                    ii_par = ii_par + 1
                # Pack Surface albedo jac
                elif jacob_name == f"TROPOMISURFACEALBEDO{my_filter}":
                    o_jacobian_pack[ii_par, :] = self.jacobian_dictionary[
                        f"surface_albedo_{my_filter}"
                    ][:]
                elif jacob_name == f"TROPOMISURFACEALBEDO{my_filter}TIGHT":
                    o_jacobian_pack[ii_par, :] = self.jacobian_dictionary[
                        f"surface_albedo_{my_filter}"
                    ][:]
                    ii_par = ii_par + 1
                # Pack Suface albedo slope jac
                elif jacob_name == f"TROPOMISURFACEALBEDOSLOPE{my_filter}":
                    o_jacobian_pack[ii_par, :] = self.jacobian_dictionary[
                        f"surface_albedo_slope_{my_filter}"
                    ][:]
                    ii_par = ii_par + 1
                elif jacob_name == f"TROPOMISURFACEALBEDOSLOPE{my_filter}TIGHT":
                    o_jacobian_pack[ii_par, :] = self.jacobian_dictionary[
                        f"surface_albedo_slope_{my_filter}"
                    ][:]
                    ii_par = ii_par + 1
                # Pack Suface albedo second order slope jac
                elif jacob_name == f"TROPOMISURFACEALBEDOSLOPEORDER2{my_filter}":
                    o_jacobian_pack[ii_par, :] = self.jacobian_dictionary[
                        f"surface_albedo_slope_order2_{my_filter}"
                    ][:]
                    ii_par = ii_par + 1
                elif jacob_name == f"TROPOMISURFACEALBEDOSLOPEORDER2{my_filter}TIGHT":
                    o_jacobian_pack[ii_par, :] = self.jacobian_dictionary[
                        f"surface_albedo_slope_order2_{my_filter}"
                    ][:]
                    ii_par = ii_par + 1
                elif jacob_name == f"TROPOMISOLARSHIFT{my_filter}":
                    o_jacobian_pack[ii_par, :] = self.jacobian_dictionary[
                        f"solarshift_{my_filter}"
                    ][:]
                    ii_par = ii_par + 1
                elif jacob_name == f"TROPOMIRADIANCESHIFT{my_filter}":
                    o_jacobian_pack[ii_par, :] = self.jacobian_dictionary[
                        f"radianceshift_{my_filter}"
                    ][:]
                    ii_par = ii_par + 1
                elif jacob_name == f"TROPOMIRADSQUEEZE{my_filter}":
                    o_jacobian_pack[ii_par, :] = self.jacobian_dictionary[
                        f"radsqueeze_{my_filter}"
                    ][:]
                    ii_par = ii_par + 1
                elif jacob_name == f"TROPOMIRINGSF{my_filter}":
                    o_jacobian_pack[ii_par, :] = self.jacobian_dictionary[
                        f"ring_sf_{my_filter}"
                    ][:]
                    ii_par = ii_par + 1
                elif jacob_name == f"TROPOMIRESSCALEO0{my_filter}":
                    o_jacobian_pack[ii_par, :] = self.jacobian_dictionary[
                        f"resscale_O0_{my_filter}"
                    ][:]
                    ii_par = ii_par + 1
                elif jacob_name == f"TROPOMIRESSCALEO1{my_filter}":
                    o_jacobian_pack[ii_par, :] = self.jacobian_dictionary[
                        f"resscale_O1_{my_filter}"
                    ][:]
                    ii_par = ii_par + 1
                elif jacob_name == f"TROPOMIRESSCALEO2{my_filter}":
                    o_jacobian_pack[ii_par, :] = self.jacobian_dictionary[
                        f"resscale_O2_{my_filter}"
                    ][:]
                    ii_par = ii_par + 1
                elif jacob_name == "TROPOMITEMPSHIFTBAND3":
                    o_jacobian_pack[ii_par, :] = self.jacobian_dictionary[
                        "temp_shift_BAND3"
                    ][:]
                    ii_par = ii_par + 1
                elif jacob_name == "TROPOMITEMPSHIFTBAND3TIGHT":
                    o_jacobian_pack[ii_par, :] = self.jacobian_dictionary[
                        "temp_shift_BAND3"
                    ][:]
                    ii_par = ii_par + 1
                elif jacob_name == "TROPOMICLOUDSURFACEALBEDO":
                    o_jacobian_pack[ii_par, :] = self.jacobian_dictionary[
                        "cloud_Surface_Albedo"
                    ][:]
                    ii_par = ii_par + 1
                elif jacob_name == "OMICLOUDFRACTION":
                    o_jacobian_pack[ii_par, :] = self.jacobian_dictionary[
                        "jacobian_cloud_ils"
                    ][:]
                    ii_par = ii_par + 1
                elif jacob_name == "OMISURFACEALBEDOUV1":
                    o_jacobian_pack[ii_par, :] = self.jacobian_dictionary[
                        "jacobian_OMISURFACEALBEDOUV1"
                    ][:]
                    ii_par = ii_par + 1
                elif jacob_name == "OMISURFACEALBEDOUV2":
                    o_jacobian_pack[ii_par, :] = self.jacobian_dictionary[
                        "jacobian_OMISURFACEALBEDOUV2"
                    ][:]
                    ii_par = ii_par + 1
                elif jacob_name == "OMISURFACEALBEDOSLOPEUV2":
                    o_jacobian_pack[ii_par, :] = self.jacobian_dictionary[
                        "jacobian_OMISURFACEALBEDOSLOPEUV2"
                    ][:]
                    ii_par = ii_par + 1
                elif jacob_name == "OMINRADWAVUV1":
                    o_jacobian_pack[ii_par, :] = self.jacobian_dictionary[
                        "jacobian_nradwav_ils_uv1"
                    ][:]
                    ii_par = ii_par + 1
                elif jacob_name == "OMINRADWAVUV2":
                    o_jacobian_pack[ii_par, :] = self.jacobian_dictionary[
                        "jacobian_nradwav_ils_uv2"
                    ][:]
                    ii_par = ii_par + 1
                elif jacob_name == "OMIODWAVUV1":
                    o_jacobian_pack[ii_par, :] = self.jacobian_dictionary[
                        "jacobian_odwav_ils_uv1"
                    ][:]
                    ii_par = ii_par + 1
                elif jacob_name == "OMIODWAVUV2":
                    o_jacobian_pack[ii_par, :] = self.jacobian_dictionary[
                        "jacobian_odwav_ils_uv2"
                    ][:]
                    ii_par = ii_par + 1
                elif jacob_name == "OMIODWAVSLOEPUV1":
                    o_jacobian_pack[ii_par, :] = self.jacobian_dictionary[
                        "jacobian_odwav_slope_ils_uv1"
                    ][:]
                    ii_par = ii_par + 1
                elif jacob_name == "OMIODWAVSLOPEUV2":
                    o_jacobian_pack[ii_par, :] = self.jacobian_dictionary[
                        "jacobian_odwav_slope_ils_uv2"
                    ][:]
                    ii_par = ii_par + 1
                elif jacob_name == "OMIRINGSFUV1":
                    o_jacobian_pack[ii_par, :] = self.jacobian_dictionary[
                        "jacobian_ring_sf_ils_uv1"
                    ][:]
                    ii_par = ii_par + 1
                elif jacob_name == "OMIRINGSFUV2":
                    o_jacobian_pack[ii_par, :] = self.jacobian_dictionary[
                        "jacobian_ring_sf_ils_uv2"
                    ][:]
                    ii_par = ii_par + 1
                elif not found_jacobian_flag:
                    logger.info("NON_OMI_SPECIES", jacob_name)

                # Pack Atmopheric Species Jac
                ii_dets = 0
                ii_dete = num_rad - 1
                uu = -1

                if self.i_uip["num_atm_k"] > 0:
                    uu = []
                    for ii in range(0, len(self.jacobians_atm_ils["k_species"])):
                        if (
                            self.jacobians_atm_ils["k_species"][ii]["species"]
                            == jacob_name
                        ):
                            uu.append(ii)

                    if len(uu) > 0:
                        ii_ps = ii_par
                        ii_pe = ii_par + num_atm - 1

                        o_jacobian_pack[ii_ps : ii_pe + 1, ii_dets : ii_dete + 1] = (
                            self.jacobians_atm_ils["k_species"][uu[0]]["k"]
                        )

                        ii_dets = ii_dets + num_rad
                        ii_dete = ii_dete + num_rad
                        ii_par = ii_par + num_atm
                    # end if len(uu) > 0):
                # end if (self.i_uip['num_atm_k'] > 0):
            # end for ii in range(0,len(self.i_uip['jacobians'])):
        # end if (num_par > 0):

        # AT_LINE 186 OMI/pack_omi_jacobian
        return (o_radiance_pack, o_jacobian_pack)


class MusesForwardModelVlidortHandle(ForwardModelHandle):
    """Handle for creating a MusesForwardModelVlidort. Note we don't
    just directly use TropomiForwardModelHandle or
    OmiForwardModelHandle because of issues with circular imports. We
    work around this by just adding one level of indirection here.
    """

    def __init__(
        self,
        instrument_name: InstrumentIdentifier,
        use_vlidort_temp_dir: bool = False,
        **creator_kwargs: Any,
    ) -> None:
        self.creator_kwargs = creator_kwargs
        self.instrument_name = instrument_name
        self.measurement_id: None | MeasurementId = None
        self.retrieval_config: None | RetrievalConfiguration = None
        self.use_vlidort_temp_dir = use_vlidort_temp_dir

    def notify_update_target(
        self, measurement_id: MeasurementId, retrieval_config: RetrievalConfiguration
    ) -> None:
        """Clear any caching associated with assuming the target being retrieved is fixed"""
        logger.debug(f"Call to {self.__class__.__name__}::notify_update")
        self.measurement_id = measurement_id
        self.retrieval_config = retrieval_config

    def forward_model(
        self,
        instrument_name: InstrumentIdentifier,
        current_state: CurrentState,
        obs: MusesObservation,
        fm_sv: rf.StateVector,
        **kwargs: Any,
    ) -> None | rf.ForwardModel:
        if instrument_name != self.instrument_name:
            return None
        if self.instrument_name == InstrumentIdentifier("TROPOMI"):
            from refractor.tropomi import TropomiFmObjectCreator

            cls = TropomiFmObjectCreator
        elif self.instrument_name == InstrumentIdentifier("OMI"):
            from refractor.omi import OmiFmObjectCreator

            cls = OmiFmObjectCreator
        else:
            return None
        if self.measurement_id is None or self.retrieval_config is None:
            raise RuntimeError("Call notify_update_target first")
        logger.debug(
            f"Creating forward model MusesForwardModelVlidort for {self.instrument_name}"
        )
        obj_creator = cls(
            current_state,
            self.measurement_id,
            self.retrieval_config,
            obs,
            fm_sv=fm_sv,
            use_vlidort=True,
            match_py_retrieve=True,
            **self.creator_kwargs,
        )
        fm = obj_creator.forward_model
        return fm


ForwardModelHandleSet.add_default_handle(
    MusesForwardModelVlidortHandle(InstrumentIdentifier("TROPOMI")),
    priority_order=-1,
)
ForwardModelHandleSet.add_default_handle(
    MusesForwardModelVlidortHandle(InstrumentIdentifier("OMI")),
    priority_order=-1,
)

__all__ = [
    "MusesForwardModelVlidort",
]

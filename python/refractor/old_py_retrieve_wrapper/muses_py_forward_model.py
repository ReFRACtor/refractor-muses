# Don't both typechecking the file. This is old code, only used for backwards testing.
# Silence mypy, just so we don't get a lot of noise in the output
# type: ignore

from __future__ import annotations
import refractor.muses.muses_py as mpy  # type: ignore
import refractor.framework as rf  # type: ignore
from refractor.muses import (
    RefractorUip,
    osswrapper,
    suppress_replacement,
    register_replacement_function_in_block,
    muses_py_call,
)
import os
import pickle
import tempfile
from loguru import logger
import copy
import numpy as np
import pandas as pd
from weakref import WeakSet

# ============================================================================
# Note the classes in this file shouldn't be used in general.
#
# Instead, MusesForwardModel or other ReFRACtor ForwardModel should be used.
#
# These classes are used by RefractorTropOrOmiFmMusesPy to
# initially to do a detailed comparison between the existing muses-py code
# and our ReFRACtor forward models.
#
# Because this is so wrapped up with the specific tropomi and omi code,
# this is pretty convoluted.
#
# I imagine this is fragile, changes to muses-py may well break this. If
# this happens, we can probably just abandon this code - it really has already
# served its function by doing the initial comparison of ReFRACtor and
# muses-py. But we'll leave this in place, it may be useful when diagnosing
# some issue.
# ============================================================================


class WatchUipUpdate(mpy.ObserveFunctionObject):
    """We  unfortunately can't just use the uip passed to tropomi_fm or
    omi_fm because we also need the basis_matrix to
    get the state vector update. So we watch calls to update_uip.

    This object just forwards the calls to the object in the notify_set.

    This class is a singleton, so it is ok if it gets created multiple times -
    it is the same underlying object each time."""

    instance = None

    def __new__(cls):
        if cls.instance is None:
            cls.instance = super().__new__(cls)
            cls.instance.notify_set = WeakSet()
        return cls.instance

    def add_notify_object(self, obj):
        self.notify_set.add(obj)

    def remove_notify_object(self, obj):
        self.notify_set.remove(obj)

    def notify_function_call(self, func_name, parms):
        fm_vec = np.matmul(
            parms["i_retrieval_vec"], parms["i_ret_info"]["basis_matrix"]
        )
        for obj in self.notify_set:
            obj.update_state(fm_vec, parms=parms)


watch_uip = WatchUipUpdate()


class MusesPyForwardModel:
    """
    NOTE - this is deprecated

    This is an adapter than makes a muse-py forward model call look
    like a ReFRACtor ForwardModel.

    Note that the muses-py returns all the channels at once. To fit this
    into ForwardModel we pretend that there is only one "channel", and
    have radiance(0) return everything. We could put the logic to split this
    up if needed.

    Also, we don't yet have the director stuff in place for a ForwardModel.
    We'll probably do that at some point, but for now we don't actually
    derive from ForwardModel. Since we most likely will just use this for
    comparing ReFRACtor ForwardModel with muses-py this is probably fine. But
    if we want use any of the ReFRACtor functions that use a ForwardModel
    (e.g., our solver framework) we'll need to get that plumbing in place.
    """

    def __init__(self, rf_uip, use_current_dir=False):
        """Constructor. As a convenience we take a RefractorUip, however
        muses-py just used the uip/dict part of this. We could change the
        interface if it proves useful, but for now this is what we have.

        By default we use the captured directory in rf_uip, but optionally
        we can just skip that and assume we are in a directory that has been
        set up for us"""
        self.rf_uip = rf_uip
        self.use_current_dir = use_current_dir

    def setup_grid(self):
        # Probably don't need this here, default for ForwardModel is to
        # do nothing
        pass

    @property
    def num_channels(self):
        # Fake 1 pseudo-channel to contain everything. Can divide up if
        # this proves useful
        return 1

    def spectral_domain(self, channel_index):
        # TODO Fill this in, should be able to extract this from uip
        pass

    @property
    def have_obj_creator(self):
        return False

    def radiance_all(self, skip_jacobian=False):
        # This is automatic if we eventually derive from rf.ForwardModel,
        # so we can remove this.
        return self.radiance(0, skip_jacobian=skip_jacobian)

    def _radiance_extracted_dir(self):
        """Run in an directory saved in rf_uip. Pulled out into
        its own function just so we don't have a deeply nested structure
        in radiance"""
        curdir = os.getcwd()
        old_run_dir = os.environ.get("MUSES_DEFAULT_RUN_DIR")
        try:
            with tempfile.TemporaryDirectory() as tmpdirname:
                self.rf_uip.extract_directory(path=tmpdirname)
                dirname = (
                    tmpdirname
                    + "/"
                    + os.path.basename(os.path.dirname(self.rf_uip.strategy_table))
                )
                os.environ["MUSES_DEFAULT_RUN_DIR"] = dirname
                os.chdir(dirname)
                # This is (o_radianceOut, o_jacobianOut, o_bad_flag, o_measured_radiance_omi, o_measured_radiance_tropomi)
                rad, jac, _, _, _ = mpy.fm_wrapper(self.rf_uip.uip, None, oco_info={})
        finally:
            if old_run_dir:
                os.environ["MUSES_DEFAULT_RUN_DIR"] = old_run_dir
            elif "MUSES_DEFAULT_RUN_DIR" in os.environ:
                del os.environ["MUSES_DEFAULT_RUN_DIR"]
            os.chdir(curdir)
        return rad, jac

    def radiance(self, channel_index, skip_jacobian=False):
        """Return spectrum for one pseudo-channel"""
        if channel_index != 0:
            raise IndexError("channel_index should be 0, was %d" % channel_index)
        if self.use_current_dir:
            # This should not have had struct_combine called on it,
            # remove duplicate if needed
            uip = copy.copy(self.rf_uip.uip)
            if "jacobians" in uip:
                if "uip_OMI" in uip:
                    for k in uip["uip_OMI"].keys():
                        del uip[k]
                if "uip_TROPOMI" in uip:
                    for k in uip["uip_TROPOMI"].keys():
                        del uip[k]
            rad, jac, _, _, _ = mpy.fm_wrapper(uip, None, oco_info={})
        else:
            rad, jac = self._radiance_extracted_dir()
        jac = jac.transpose()
        sd = rf.SpectralDomain(rad["frequency"], rf.Unit("nm"))
        d = rad["radiance"][0, :]
        if not skip_jacobian:
            d = rf.ArrayAd_double_1(d, jac)
        # TODO Check on these units
        sr = rf.SpectralRange(d, rf.Unit("ph / nm / s"))
        return rf.Spectrum(sd, sr)


# Support for capturing data

if mpy.have_muses_py:

    class _FakeUipExecption(Exception):
        def __init__(self, uip, i_windows, oco_info):
            self.uip = uip
            self.windows = i_windows
            self.oco_info = oco_info

    class _CaptureUip(mpy.ReplaceFunctionObject):
        def __init__(self, func_count=1):
            self.func_count = func_count

        def should_replace_function(self, func_name, parms):
            self.func_count -= 1
            if self.func_count <= 0:
                return True
            return False

        def replace_function(self, func_name, parms):
            raise _FakeUipExecption(
                parms["i_uip"], parms["i_windows"], parms["oco_info"]
            )

    class _CaptureRetInfo(mpy.ReplaceFunctionObject):
        def __init__(self):
            self.ret_info = None
            self.retrieval_vec = None

        def should_replace_function(self, func_name, parms):
            # Never replace the function, just grab the argument
            self.ret_info = parms["i_ret_info"]
            self.retrieval_vec = parms["i_retrieval_vec"]


class RefractorTropOrOmiFmBase(mpy.ReplaceFunctionObject):
    """
    NOTE - this is deprecated

    Base class for RefractorTropOmiFm and RefractorOmiFm (there is enough
    overlap it is worth combining them). This adapts a ReFRACtor forward model
    to replace the tropomi_fm or omi_fm call in muses-py.

    An object needs to be registered with muses-py to get called in place
    of tropomi_fm or omi_fm. This can be done with a call to the
    register_with_muses_py function.
    """

    def __init__(
        self,
        func_name="tropomi_fm",
        py_retrieve_debug=False,
        py_retrieve_vlidort_nstokes=2,
        py_retrieve_vlidort_nstreams=4,
    ):
        self.sv_extra_index = {}
        self.run_dir = "."
        self.rf_uip = None
        self.basis_matrix = None
        self.py_retrieve_debug = py_retrieve_debug
        self.py_retrieve_vlidort_nstokes = py_retrieve_vlidort_nstokes
        self.py_retrieve_vlidort_nstreams = py_retrieve_vlidort_nstreams
        self.func_name = func_name

    def __enter__(self):
        self.register_with_muses_py()
        return self

    def __exit__(self, *exc):
        self.unregister_with_muses_py()
        return False

    def register_with_muses_py(self):
        """Register this object and the helper objects with muses-py,
        to replace a call to omi_fm.

        Note for testing you can also use this object as a context manager,
        which handles registration and then cleanup - so
        rt = RefractorOmiFm()
        with rt:
           call to muses-py function that calls omi_fm
        """
        mpy.register_replacement_function(self.func_name, self)
        mpy.register_observer_function("update_uip", watch_uip)
        watch_uip.add_notify_object(self)

    def unregister_with_muses_py(self):
        mpy.unregister_replacement_function(self.func_name)
        watch_uip.remove_notify_object(self)

    def should_replace_function(self, func_name, parms):
        # Currently we only handle the OMI instrument. For other
        # instruments just continue using the normal omi_fm.
        if (
            self.func_name == "tropomi_fm"
            and "TROPOMI" in parms["i_uip"]["instruments"]
        ):
            return True
        if self.func_name == "omi_fm" and "OMI" in parms["i_uip"]["instruments"]:
            return True
        return False

    @classmethod
    def uip_from_muses_retrieval_step(
        cls,
        rstep,
        iteration,
        pickle_file,
    ):
        """Grab the UIP and directory that can be used to call
        tropomi_fm/omi_fm.
        This starts with MusesRetrievalStep, and gets the UIP passed to
        tropomi_fm in the given iteration number (1 based). Output is
        written to the pickle file, which can then be used for calling
        tropomi_fm/omi_fm."""
        cretinfo = _CaptureRetInfo()
        with register_replacement_function_in_block("update_uip", cretinfo):
            with register_replacement_function_in_block(
                "fm_wrapper", _CaptureUip(func_count=iteration)
            ):
                try:
                    rstep.run_retrieval()
                except _FakeUipExecption as e:
                    res = RefractorUip(
                        uip=e.uip, basis_matrix=cretinfo.ret_info["basis_matrix"]
                    )
        res.tar_directory(os.environ["MUSES_DEFAULT_RUN_DIR"] + "/Table.asc")
        pickle.dump(res, open(pickle_file, "wb"))

    def run_pickle_file(
        self,
        pickle_file,
        path=".",
        osp_dir=None,
        gmao_dir=None,
    ):
        """This goes with uip_from_muses_retrieval_step, it turns around
        and calls tropomi_fm/omi_fm with the saved data."""
        curdir = os.getcwd()
        try:
            rf_uip = RefractorUip.load_uip(
                pickle_file,
                path=path,
                change_to_dir=True,
                osp_dir=osp_dir,
                gmao_dir=gmao_dir,
            )
            self.basis_matrix = rf_uip.basis_matrix
            self.run_dir = rf_uip.run_dir
            with osswrapper(rf_uip.uip):
                with muses_py_call(
                    ".",
                    debug=self.py_retrieve_debug,
                    vlidort_nstokes=self.py_retrieve_vlidort_nstokes,
                    vlidort_nstreams=self.py_retrieve_vlidort_nstreams,
                ):
                    if self.func_name == "tropomi_fm":
                        return self.tropomi_fm(rf_uip.uip_all("TROPOMI"))
                    else:
                        return self.omi_fm(rf_uip.uip_all("OMI"))
        finally:
            os.chdir(curdir)

    @property
    def vlidort_input(self):
        return f"{self.run_dir}/{self.rf_uip.vlidort_input}"

    @property
    def vlidort_output(self):
        return f"{self.run_dir}/{self.rf_uip.vlidort_output}"

    @property
    def clear_in(self):
        if not self.py_retrieve_debug:
            raise RuntimeError(
                "You need to run with py_retrieve_debug=True to get the iteration output if you want to view it."
            )
        iteration = self.rf_uip.uip["iteration"]
        ii_mw = 0
        return f"{self.vlidort_input}/Iter{iteration:02d}/MW{ii_mw + 1:03d}/clear"

    @property
    def cloudy_in(self):
        if not self.py_retrieve_debug:
            raise RuntimeError(
                "You need to run with py_retrieve_debug=True to get the iteration output if you want to view it."
            )
        iteration = self.rf_uip.uip["iteration"]
        ii_mw = 0
        return f"{self.vlidort_input}/Iter{iteration:02d}/MW{ii_mw + 1:03d}/cloudy"

    @property
    def clear_out(self):
        if not self.py_retrieve_debug:
            raise RuntimeError(
                "You need to run with py_retrieve_debug=True to get the iteration output if you want to view it."
            )
        iteration = self.rf_uip.uip["iteration"]
        ii_mw = 0
        return f"{self.vlidort_output}/Iter{iteration:02d}/MW{ii_mw + 1:03d}/clear"

    @property
    def cloudy_out(self):
        if not self.py_retrieve_debug:
            raise RuntimeError(
                "You need to run with py_retrieve_debug=True to get the iteration output if you want to view it."
            )
        iteration = self.rf_uip.uip["iteration"]
        ii_mw = 0
        return f"{self.vlidort_output}/Iter{iteration:02d}/MW{ii_mw + 1:03d}/cloudy"

    def in_dir(self, do_cloud):
        """Either cloudy_in or clear_in depending on do_cloud"""
        return self.cloudy_in if do_cloud else self.clear_in

    def out_dir(self, do_cloud):
        """Either cloudy_out or clear_out depending on do_cloud"""
        return self.cloudy_out if do_cloud else self.clear_out

    def replace_function(self, func_name, parms):
        if func_name == "tropomi_fm":
            return self.tropomi_fm(**parms)
        return self.omi_fm(**parms)

    def fd_jac(self, index, delta):
        """Calculate a finite difference jacobian for one index. We do
        just one index because these take a while to run, and it can be
        useful to go index by index.

        Return the finite difference and value, and also the jacobian
        as returned by tropomi_fm/omi_fm, i.e. this can be used to evaluate
        how accurate the tropomi_fm/omi_fm jacobian is.
        """
        # Save so we can reset this value before exiting.
        retrieval_vec_0 = np.copy(self.rf_uip.current_state_x)
        if self.func_name == "tropomi_fm":
            f = self.tropomi_fm
        else:
            f = self.omi_fm
        with muses_py_call(self.run_dir):
            jac, rad0, meas_rad0, _ = f(self.rf_uip.uip)
            r = np.copy(retrieval_vec_0)
            r[index] += delta
            self.update_retrieval_vec(r)
            _, rad1, meas_rad1, _ = f(self.rf_uip.uip)
            self.update_retrieval_vec(retrieval_vec_0)
            # The jacobian is actually of the residual, not rad. Note that the
            # residual is (rad-meas_rad)/meas_err. However at the point
            # that we are calculating this, the jacobian hasn't been scaled
            # yet. So this is correct, even though later in
            # residual_fm_jacobian this gets scaled by meas_err.
            jacfd = (
                (rad1 - meas_rad1["measured_radiance_field"])
                - (rad0 - meas_rad0["measured_radiance_field"])
            ) / delta
            # The logic in pack_tropomi_jacobian over counts the size of
            # atmosphere jacobians by 1 for each species. This is harmless,
            # it gives an extra row of zeros that then gets trimmed before
            # leaving
            # fm_wrapper. But we need to trim this to do this step
            if jac.shape[0] > self.rf_uip.basis_matrix.shape[1]:
                jac = jac[: self.rf_uip.basis_matrix.shape[1], :]
            jaccalc = np.matmul(self.rf_uip.basis_matrix, jac)[index]
            return jacfd, jaccalc

    def update_retrieval_vec(self, retrieval_vec):
        """Update the retrieval vector, both saved in this class and used
        by muses-py"""
        self.rf_uip.update_uip(retrieval_vec)
        self.update_state(self.rf_uip.current_state_x_fm)

    def tropomi_fm(self, i_uip, **kwargs):
        """Substitutes for the muses-py tropomi_fm function

        This returns
        (o_jacobian, o_radiance, o_measured_radiance_tropomi, o_success_flag)

        o_success_flag is 1 if the data is good, 0 otherwise.
        """
        self.rf_uip = RefractorUip(i_uip, basis_matrix=self.basis_matrix)
        self.rf_uip.run_dir = os.getcwd()
        if hasattr(self, "obj_creator"):
            # Make sure all the objects are created and registered before updating
            # state vector
            _ = self.obj_creator.forward_model
            self.obj_creator.fm_sv.update_state(self.rf_uip.current_state_x_fm)
        mrad = self.observation.radiance(0)
        o_measured_radiance_tropomi = {
            "measured_radiance_field": mrad.spectral_range.data,
            "measured_nesr": mrad.spectral_range.uncertainty,
        }
        o_success_flag = 1

        spec = self.radiance_all()
        o_radiance = spec.spectral_range.data.copy()
        if spec.spectral_range.data_ad.is_constant:
            o_jacobian = np.array([])
        else:
            o_jacobian = spec.spectral_range.data_ad.jacobian.transpose().copy()

        if not mrad.spectral_range.data_ad.is_constant:
            # - because we are giving the jacobian of fm - rad
            o_jacobian -= mrad.spectral_range.data_ad.jacobian.transpose()
        # We've calculated the jacobian relative to the full state vector,
        # including specifies that aren't used by OMI/TROPOMI. muses-py
        # expects just the subset, so we need to subset the jacobian
        our_jac = [
            spec in self.rf_uip.state_vector_params("TROPOMI")
            for spec in i_uip["speciesListFM"]
        ]
        if len(o_jacobian) > 0:
            o_jacobian = o_jacobian[our_jac, :]
        return (o_jacobian, o_radiance, o_measured_radiance_tropomi, o_success_flag)

    def omi_fm(self, i_uip, **kwargs):
        """Substitutes for the muses-py omi_fm function

        This returns
        (o_jacobian, o_radiance, o_measured_radiance_omi, o_success_flag)

        o_success_flag is 1 if the data is good, 0 otherwise.
        """

        self.rf_uip = RefractorUip(i_uip, basis_matrix=self.basis_matrix)
        self.rf_uip.run_dir = os.getcwd()
        if hasattr(self, "obj_creator"):
            # Make sure all the objects are created and registered before updating
            # state vector
            _ = self.obj_creator.forward_model
            self.obj_creator.fm_sv.update_state(self.rf_uip.current_state_x_fm)
        o_measured_radiance_omi = self.rf_uip.measured_radiance("OMI")
        o_success_flag = 1

        spec = self.radiance_all()
        o_radiance = spec.spectral_range.data.copy()
        if spec.spectral_range.data_ad.is_constant:
            o_jacobian = np.array([])
        else:
            o_jacobian = spec.spectral_range.data_ad.jacobian.transpose().copy()

        # The ForwardModel currently doesn't have the solar model shift
        # included in it, this gets accounted for in get_omi_radiance
        # called by self.rf_uip.omi_measured_radiance. So we need to
        # attach this piece into the o_jacobian. Note that our
        # replacement RefractorResidualFmJacobian handles this, it is just
        # this old code that needs this handling. Also TROPOMI get handled
        # (we needed this to fix a problem in the jacobian) - only OMI
        # needs this code here.
        #
        # This here duplicates what pack_omi_jacobian does
        mw = [
            slice(0, self.rf_uip.nfreq_mw(0, "OMI")),
            slice(self.rf_uip.nfreq_mw(0, "OMI"), None),
        ]
        if "OMINRADWAVUV1" in self.rf_uip.state_vector_params("OMI"):
            indx = list(self.rf_uip.uip["speciesListFM"]).index("OMINRADWAVUV1")
            o_jacobian[indx, mw[0]] = o_measured_radiance_omi["normwav_jac"][mw[0]]
        if "OMINRADWAVUV2" in self.rf_uip.state_vector_params("OMI"):
            indx = list(self.rf_uip.uip["speciesListFM"]).index("OMINRADWAVUV2")
            o_jacobian[indx, mw[1]] = o_measured_radiance_omi["normwav_jac"][mw[1]]
        if "OMIODWAVUV1" in self.rf_uip.state_vector_params("OMI"):
            indx = list(self.rf_uip.uip["speciesListFM"]).index("OMIODWAVUV1")
            o_jacobian[indx, mw[0]] = o_measured_radiance_omi["odwav_jac"][mw[0]]
        if "OMIODWAVUV2" in self.rf_uip.state_vector_params("OMI"):
            indx = list(self.rf_uip.uip["speciesListFM"]).index("OMIODWAVUV2")
            o_jacobian[indx, mw[1]] = o_measured_radiance_omi["odwav_jac"][mw[1]]
        if "OMIODWAVSLOPEUV1" in self.rf_uip.state_vector_params("OMI"):
            indx = list(self.rf_uip.uip["speciesListFM"]).index("OMIODWAVSLOPEUV1")
            o_jacobian[indx, mw[0]] = o_measured_radiance_omi["odwav_slope_jac"][mw[0]]
        if "OMIODWAVSLOPEUV2" in self.rf_uip.state_vector_params("OMI"):
            indx = list(self.rf_uip.uip["speciesListFM"]).index("OMIODWAVSLOPEUV2")
            o_jacobian[indx, mw[1]] = o_measured_radiance_omi["odwav_slope_jac"][mw[1]]

        # We've calculated the jacobian relative to the full state vector,
        # including species that aren't used by OMI/TROPOMI. muses-py
        # expects just the subset, so we need to subset the jacobian
        our_jac = [
            spec in self.rf_uip.state_vector_params("OMI")
            for spec in i_uip["speciesListFM"]
        ]
        if len(o_jacobian) > 0:
            o_jacobian = o_jacobian[our_jac, :]
        return (o_jacobian, o_radiance, o_measured_radiance_omi, o_success_flag)

    def radiance_all(self, skip_jacobian=False):
        """The forward model radiance_all results"""
        raise NotImplementedError()

    def update_state(self, fm_vec, parms=None):
        """Called with muses-py updated the state vector."""
        pass

    # To do initial comparisons between muses-py and ReFRACtor it
    # can be useful to have detailed information about the run. We
    # supply functions here, which are really just for diagnostic use.
    # The default is to raise a NotImplementedError, but we can override
    # this for various derived classes to we can compare things. A derived
    # class might not implement this - this fine and it just means we can't
    # do a details comparison of things.
    def raman_ring_spectrum(self, do_cloud):
        """Return Spectrum of Raman scattering, clear or cloudy.
        Note this is the "ring" calculation, which gets translated to the
        raman correction by a scale factor + 1"""
        raise NotImplementedError()

    def surface_albedo(self, do_cloud):
        """Return Spectrum of surface albedo, clear or cloudy"""
        raise NotImplementedError()

    def geometry(self, do_cloud):
        """Return solar zenith angle, observation zenith, and
        relative azimuth, clear or cloudy"""
        raise NotImplementedError()

    def pressure_grid(self, do_cloud):
        """Return pressure grid for each level, clear or cloudy"""
        return NotImplementedError()

    def temperature_grid(self, do_cloud):
        """Return temperature  grid for each level, clear or cloudy"""
        return NotImplementedError()

    def altitude_grid(self, do_cloud):
        """Return pressure grid for each level, clear or cloudy"""
        return NotImplementedError()

    def gas_number_density(self, do_cloud):
        """Return gas numebr density, clear or cloudy."""
        return NotImplementedError()

    def taur(self, do_cloud):
        """Return optical depth from Rayleigh, clear and cloudy."""
        return NotImplementedError()

    def taug(self, do_cloud):
        """Return optical depth from Gas (e.g., O3), clear and cloudy."""
        return NotImplementedError()

    def taut(self, do_cloud):
        """Return total optical depth, clear and cloudy."""
        return NotImplementedError()

    def rt_radiance(self, do_cloud):
        """Return the radiance from the RT (e.g., VLIDORT, PCA, or LIDORT),
        which is before any spectrum effects are added (e.g., MusesRaman)."""
        return NotImplementedError()


class RefractorTropOrOmiFmPyRetrieve(RefractorTropOrOmiFmBase):
    """
    NOTE - this is deprecated

    Turn around and call tropomi_fm/omi_fm, without change. This
    gives a way to do a direct comparison with muses-py vs
    ReFRACtor. Ultimately this should give the same results as
    RefractorTropOrOmiFmMusesPy, but this skips the mucking around
    with jacobians etc. that RefractorTropOrOmiFmBase does - so this
    lets us establish that RefractorTropOmiFmMusesPy is correct."""

    def omi_fm(self, i_uip, **kwargs):
        self.rf_uip = RefractorUip(i_uip)
        self.rf_uip.run_dir = os.getcwd()
        self.rf_uip.basis_matrix = self.basis_matrix
        with suppress_replacement("omi_fm"):
            return mpy.omi_fm(i_uip)

    def tropomi_fm(self, i_uip, **kwargs):
        self.rf_uip = RefractorUip(i_uip)
        self.rf_uip.run_dir = os.getcwd()
        self.rf_uip.basis_matrix = self.basis_matrix
        with suppress_replacement("tropomi_fm"):
            return mpy.tropomi_fm(i_uip)


class RefractorTropOrOmiFmMusesPy(RefractorTropOrOmiFmBase):
    """This just turns around and calls MusesPyForwardModel. This is useful
    to test the interconnection with muses-py, since a retrieval should be
    identical to one without a replacement."""

    def radiance_all(self, skip_jacobian=False):
        """The forward model radiance_all results"""
        # In the next call for tropomi_fm, we don't actually want to
        # replace this
        with suppress_replacement("tropomi_fm"):
            fm = MusesPyForwardModel(self.rf_uip, use_current_dir=True)
            return fm.radiance_all(skip_jacobian=skip_jacobian)

    def update_state(self, rvec, parms=None):
        """Called with muses-py updated the state vector."""
        logger.info(
            f"RefractorTropOrOmiFmMusesPy updating state to: {rvec.shape} : {rvec}"
        )

    def raman_ring_spectrum(self, do_cloud):
        """Return Spectrum of Raman scattering, clear or cloudy.
        Note this is the "ring" calculation, which gets translated to the
        raman correction by a scale factor + 1.

        Need to run with py_retrieve_debug=True for the data to be available for this
        function."""
        ring = mpy.read_rtm_output(self.out_dir(do_cloud), "Ring.asc")
        mw = self.rf_uip.micro_windows(0).value[0, :]
        slc = (ring[0, :] >= mw[0]) & (ring[0, :] <= mw[1])
        sd = rf.SpectralDomain(ring[0, slc], rf.Unit("nm"))
        # Units don't matter here, but lets just assign something reasonable
        sr = rf.SpectralRange(
            ring[1, slc],
            rf.Unit("ph / s / m^2 / micron W / (cm^-1) / (ph / (s) / (micron)) sr^-1"),
        )
        return rf.Spectrum(sd, sr)

    def surface_albedo(self, do_cloud):
        """Return Spectrum of surface albedo, clear or cloudy

        Need to run with py_retrieve_debug=True for the data to be available for this
        function."""
        alb = pd.read_csv(
            f"{self.in_dir(do_cloud)}/Surfalb_MW001.asc",
            sep=r"\s+",
            skiprows=1,
            header=None,
            names=["wavelength", "albedo"],
        ).to_numpy()
        mw = self.rf_uip.micro_windows(0).value[0, :]
        slc = (alb[:, 0] >= mw[0]) & (alb[:, 0] <= mw[1])
        sd = rf.SpectralDomain(alb[slc, 0], rf.Unit("nm"))
        sr = rf.SpectralRange(alb[slc, 1], rf.Unit("dimensionless"))
        return rf.Spectrum(sd, sr)

    def geometry(self, do_cloud):
        """Return solar zenith angle, observation zenith, and
        relative azimuth, clear or cloudy.

        Need to run with py_retrieve_debug=True for the data to be available for this
        function."""

        vga = pd.read_csv(
            f"{self.in_dir(do_cloud)}/Vga_MW001.asc", sep="[ ,]+", engine="python"
        )
        return (vga["SZA"][0], vga["VZA"][0], vga["RAZ"][0])

    def pressure_grid(self, do_cloud):
        """Return pressure grid for each level, clear or cloudy

        Need to run with py_retrieve_debug=True for the data to be available for this
        function."""

        atm = pd.read_csv(
            f"{self.in_dir(do_cloud)}/Atm_level.asc",
            sep=r"\s+",
            skiprows=2,
            header=None,
            names=["Pres(mb)", "T(K)", "Altitude(m)"],
        )
        return rf.ArrayWithUnit(atm["Pres(mb)"].to_numpy(), "hPa")

    def temperature_grid(self, do_cloud):
        """Return temperature  grid for each level, clear or cloudy

        Need to run with py_retrieve_debug=True for the data to be available for this
        function."""

        atm = pd.read_csv(
            f"{self.in_dir(do_cloud)}/Atm_level.asc",
            sep=r"\s+",
            skiprows=2,
            header=None,
            names=["Pres(mb)", "T(K)", "Altitude(m)"],
        )
        # This doesn't include the temperature shift in the output file, although the
        # shifted temperature is used in the calculation, see get_tropomi_o3xsec in
        # muses-py
        if self.func_name == "tropomi_fm":
            toffset = self.rf_uip.tropomi_params["temp_shift_BAND3"]
        else:
            toffset = 0
        return rf.ArrayWithUnit(atm["T(K)"].to_numpy() + toffset, "K")

    def altitude_grid(self, do_cloud):
        """Return pressure grid for each level, clear or cloudy

        Need to run with py_retrieve_debug=True for the data to be available for this
        function."""

        atm = pd.read_csv(
            f"{self.in_dir(do_cloud)}/Atm_level.asc",
            sep=r"\s+",
            skiprows=2,
            header=None,
            names=["Pres(mb)", "T(K)", "Altitude(m)"],
        )
        return rf.ArrayWithUnit(atm["Altitude(m)"].to_numpy(), "m")

    def gas_number_density(self, do_cloud):
        """Return gas number density, clear or cloudy.

        Need to run with py_retrieve_debug=True for the data to be available for this
        function."""

        atm = pd.read_csv(
            f"{self.in_dir(do_cloud)}/Atm_layer.asc",
            sep=r"\s+",
            skiprows=2,
            header=None,
            names=["Pressure layer(mb)", "Temperature layer (K)", "Gas Density"],
        )
        return rf.ArrayWithUnit(atm["Gas Density"].to_numpy(), "cm^-2")

    def taur(self, do_cloud):
        """Return optical depth from Rayleigh, clear and cloudy.

        Need to run with py_retrieve_debug=True for the data to be available for this
        function."""

        t = pd.read_csv(
            f"{self.out_dir(do_cloud)}/taur.asc", sep=r"\s+", header=None
        ).to_numpy()
        mw = self.rf_uip.micro_windows(0).value[0, :]
        slc = (t[:, 0] >= mw[0]) & (t[:, 0] <= mw[1])
        sd = rf.SpectralDomain(t[slc, 0], rf.Unit("nm"))
        return sd, t[slc, 1:]

    def taug(self, do_cloud):
        """Return optical depth from Gas (e.g., O3), clear and cloudy.

        Need to run with py_retrieve_debug=True for the data to be available for this
        function."""

        t = pd.read_csv(
            f"{self.out_dir(do_cloud)}/taug.asc", sep=r"\s+", header=None
        ).to_numpy()
        mw = self.rf_uip.micro_windows(0).value[0, :]
        slc = (t[:, 0] >= mw[0]) & (t[:, 0] <= mw[1])
        sd = rf.SpectralDomain(t[slc, 0], rf.Unit("nm"))
        return sd, t[slc, 1:]

    def taut(self, do_cloud):
        """Return total optical depth, clear and cloudy.

        Need to run with py_retrieve_debug=True for the data to be available for this
        function."""

        t = pd.read_csv(
            f"{self.out_dir(do_cloud)}/taut.asc", sep=r"\s+", header=None
        ).to_numpy()
        mw = self.rf_uip.micro_windows(0).value[0, :]
        slc = (t[:, 0] >= mw[0]) & (t[:, 0] <= mw[1])
        sd = rf.SpectralDomain(t[slc, 0], rf.Unit("nm"))
        return sd, t[slc, 1:]

    def rt_radiance(self, do_cloud):
        """Return the radiance from the RT (e.g., VLIDORT, PCA, or LIDORT),
        which is before any spectrum effects are added (e.g., MusesRaman).


        Need to run with py_retrieve_debug=True for the data to be available for this
        function."""
        rad = mpy.read_rtm_output(self.out_dir(do_cloud), "Radiance.asc")
        mw = self.rf_uip.micro_windows(0).value[0, :]
        slc = (rad[0, :] >= mw[0]) & (rad[0, :] <= mw[1])
        sd = rf.SpectralDomain(rad[0, slc], rf.Unit("nm"))
        sr = rf.SpectralRange(rad[1, slc], rf.Unit("sr^-1"))
        return rf.Spectrum(sd, sr)

    @property
    def observation(self):
        raise NotImplementedError()


class _CaptureSpectrum(rf.ObserverPtrNamedSpectrum):
    """Helper class to capture the radiative transfer output before we apply the
    RamanSioris effect."""

    def __init__(self):
        super().__init__()
        self.spectrum = []

    def notify_update(self, named_spectrum):
        # The name we use right after the RT is high_res_rt.
        logger.debug(f"Call to {self.__class__.__name__}::notify_update")
        if named_spectrum.name == "high_res_rt":
            self.spectrum.append(named_spectrum.copy())


class RefractorTropOrOmiFm(RefractorTropOrOmiFmBase):
    """
    NOTE - this is deprecated

    Use a ReFRACtor ForwardModel as a replacement for tropomi_fm/omi_fm."""

    def __init__(self, func_name, **kwargs):
        super().__init__(func_name=func_name)
        self.xsec_table_to_notify = None
        self.obj_creator_args = kwargs

    def update_state(self, fm_vec, parms=None):
        self.ret_info = parms["i_ret_info"]
        self.rf_uip = RefractorUip(parms["i_uip"])
        if self.have_obj_creator:
            # Make sure all the objects are created and registered before updating
            # state vector
            _ = self.obj_creator.forward_model
            self.obj_creator.fm_sv.update_state(fm_vec)

    @property
    def fm(self):
        """Forward model, creating a new one if needed"""
        return self.obj_creator.forward_model

    @property
    def observation(self):
        # Creating the state vector sets up the connection between
        # observation and the state vector.
        raise NotImplementedError()

    def radiance_all(self, skip_jacobian=False):
        logger.info(f"FM state vector:\n{self.obj_creator.fm_sv}")
        spec = self.fm.radiance_all(skip_jacobian=skip_jacobian)
        return spec

    def raman_ring_spectrum(self, do_cloud):
        """Return Spectrum of Raman scattering, clear or cloudy.
        Note this is the "ring" calculation, which gets translated to the
        raman correction by a scale factor + 1"""
        self.fm.set_do_cloud(do_cloud)
        sd = self.fm.spectral_domain(0)
        # Units don't matter here, but lets just assign something reasonable
        sr = rf.SpectralRange(
            [1] * sd.rows,
            rf.Unit("ph / s / m^2 / micron W / (cm^-1) / (ph / (s) / (micron)) sr^-1"),
        )
        s = rf.Spectrum(sd, sr)
        self.obj_creator.raman_effect(0).apply_effect(
            s, self.fm.underlying_forward_model.spectral_grid
        )
        sr = rf.SpectralRange(
            (s.spectral_range.data - 1)
            / self.obj_creator.raman_effect(0).coefficient[0].value,
            s.spectral_range.units,
        )
        s = rf.Spectrum(sd, sr)
        return s

    def surface_albedo(self, do_cloud):
        """Return Spectrum of surface albedo, clear or cloudy"""
        self.fm.set_do_cloud(do_cloud)
        sd = self.fm.spectral_domain(0)
        sr = rf.SpectralRange(
            np.array(
                [
                    self.obj_creator.ground.surface_parameter(wn, 0).value[0]
                    for wn in sd.convert_wave("cm^-1")
                ]
            ),
            rf.Unit("dimensionless"),
        )
        return rf.Spectrum(sd, sr)

    def geometry(self, do_cloud):
        """Return solar zenith angle, observation zenith, and
        relative azimuth, clear or cloudy"""
        return (
            self.rf_uip.solar_zenith(self.rf_uip.filter_name(0)),
            self.rf_uip.observation_zenith(self.rf_uip.filter_name(0)),
            self.rf_uip.relative_azimuth(self.rf_uip.filter_name(0)),
        )

    def pressure_grid(self, do_cloud):
        """Return pressure grid for each level, clear or cloudy"""
        self.fm.set_do_cloud(do_cloud)
        pgrid = self.obj_creator.pressure.pressure_grid()
        return rf.ArrayWithUnit(pgrid.value.value, pgrid.units)

    def temperature_grid(self, do_cloud):
        """Return temperature  grid for each level, clear or cloudy"""
        self.fm.set_do_cloud(do_cloud)
        tgrid = self.obj_creator.temperature.temperature_grid(self.obj_creator.pressure)
        return rf.ArrayWithUnit(tgrid.value.value, tgrid.units)

    def altitude_grid(self, do_cloud):
        """Return pressure grid for each level, clear or cloudy"""
        self.fm.set_do_cloud(do_cloud)
        agrid = self.obj_creator.atmosphere.altitude(0)
        return rf.ArrayWithUnit(agrid.value.value, agrid.units)

    def gas_number_density(self, do_cloud):
        """Return gas numebr density, clear or cloudy."""
        self.fm.set_do_cloud(do_cloud)
        glay = self.obj_creator.absorber.gas_number_density_layer(0)
        return rf.ArrayWithUnit(glay.value.value[:, 0], glay.units)

    def taur(self, do_cloud):
        """Return optical depth from Rayleigh, clear and cloudy."""
        self.fm.set_do_cloud(do_cloud)
        sd = self.fm.spectral_domain(0)
        g = np.vstack(
            [
                np.array(
                    [
                        self.obj_creator.rayleigh.optical_depth_each_layer(wn, 0).value
                        for wn in sd.convert_wave("cm^-1")
                    ]
                )
            ]
        )
        return sd, g

    def taug(self, do_cloud):
        """Return optical depth from Gas (e.g., O3), clear and cloudy."""
        self.fm.set_do_cloud(do_cloud)
        sd = self.fm.spectral_domain(0)
        g = np.vstack(
            [
                np.array(
                    [
                        self.obj_creator.absorber.optical_depth_each_layer(wn, 0).value[
                            :, 0
                        ]
                        for wn in sd.convert_wave("cm^-1")
                    ]
                )
            ]
        )
        return sd, g

    def taut(self, do_cloud):
        """Return total optical depth, clear and cloudy."""
        self.fm.set_do_cloud(do_cloud)
        sd = self.fm.spectral_domain(0)
        g = np.vstack(
            [
                np.array(
                    [
                        self.obj_creator.atmosphere.optical_depth_wrt_state_vector(
                            wn, 0
                        ).value
                        for wn in sd.convert_wave("cm^-1")
                    ]
                )
            ]
        )
        return sd, g

    def rt_radiance(self, do_cloud):
        """Return the radiance from the RT (e.g., VLIDORT, PCA, or LIDORT),
        which is before any spectrum effects are added (e.g., MusesRaman)."""
        self.fm.set_do_cloud(do_cloud)
        cap = _CaptureSpectrum()
        self.fm.underlying_forward_model.add_observer(cap)
        self.fm.underlying_forward_model.radiance_all()
        self.fm.underlying_forward_model.remove_observer(cap)
        return cap.spectrum[0]

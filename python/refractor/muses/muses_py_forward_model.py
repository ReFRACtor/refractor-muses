import refractor.muses.muses_py as mpy
import refractor.framework as rf
from .refractor_uip import RefractorUip, WatchUipCreation, WatchUipUpdate
from .replace_function_helper import (suppress_replacement,
                                      register_replacement_function_in_block)
from .refractor_capture_directory import muses_py_call
import os
import pickle
import tempfile
import logging
import copy
import numpy as np
import pandas as pd

logger = logging.getLogger('py-retrieve')

#============================================================================
# Note the classes in this file shouldn't be used in general.
# 
# Instead, RefractorResidualFmJacobian should be used.
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
#============================================================================

class MusesPyForwardModel:
    '''
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
    '''
    def __init__(self, rf_uip, use_current_dir = False):
        '''Constructor. As a convenience we take a RefractorUip, however
        muses-py just used the uip/dict part of this. We could change the
        interface if it proves useful, but for now this is what we have.

        By default we use the captured directory in rf_uip, but optionally
        we can just skip that and assume we are in a directory that has been 
        set up for us'''
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

    def radiance_all(self, skip_jacobian = False):
        # This is automatic if we eventually derive from rf.ForwardModel,
        # so we can remove this.
        return self.radiance(0, skip_jacobian=skip_jacobian)

    def _radiance_extracted_dir(self):
        '''Run in an directory saved in rf_uip. Pulled out into
        its own function just so we don't have a deeply nested structure
        in radiance'''
        curdir = os.getcwd()
        old_run_dir = os.environ.get("MUSES_DEFAULT_RUN_DIR")
        try:
            with tempfile.TemporaryDirectory() as tmpdirname:
                self.rf_uip.extract_directory(path=tmpdirname)
                dirname = tmpdirname + "/" + os.path.basename(os.path.dirname(self.rf_uip.strategy_table))
                os.environ["MUSES_DEFAULT_RUN_DIR"] = dirname
                os.chdir(dirname)
                # This is (o_radianceOut, o_jacobianOut, o_bad_flag, o_measured_radiance_omi, o_measured_radiance_tropomi)
                rad, jac, _, _, _ = mpy.fm_wrapper(self.rf_uip.uip, self.rf_uip.windows, self.rf_uip.oco_info)
        finally:
            if(old_run_dir):
                os.environ["MUSES_DEFAULT_RUN_DIR"] = old_run_dir
            elif("MUSES_DEFAULT_RUN_DIR" in os.environ):
                del os.environ["MUSES_DEFAULT_RUN_DIR"]
            os.chdir(curdir)
        return rad, jac
    
    def radiance(self, channel_index, skip_jacobian = False):
        '''Return spectrum for one pseudo-channel'''
        if(channel_index != 0):
            raise IndexError("channel_index should be 0, was %d" % channel_index)
        if(self.use_current_dir):
            # This should not have had struct_combine called on it,
            # remove duplicate if needed
            uip = copy.copy(self.rf_uip.uip)
            if('jacobians' in uip):
                if('uip_OMI' in uip):
                    for k in uip['uip_OMI'].keys():
                        del uip[k]
                if('uip_TROPOMI' in uip):
                    for k in uip['uip_TROPOMI'].keys():
                        del uip[k]
            rad, jac, _, _, _ = mpy.fm_wrapper(uip, self.rf_uip.windows, self.rf_uip.oco_info)
        else:
            rad, jac = self._radiance_extracted_dir()
        jac = jac.transpose()
        sd = rf.SpectralDomain(rad['frequency'], rf.Unit("nm"))
        d = rad['radiance'][0,:]
        if(not skip_jacobian):
            d = rf.ArrayAd_double_1(d, jac)
        # TODO Check on these units
        sr = rf.SpectralRange(d, rf.Unit("ph / nm / s"))
        return rf.Spectrum(sd, sr)

# Support for capturing data

if(mpy.have_muses_py):
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
            raise _FakeUipExecption(parms['i_uip'], parms['i_windows'],
                                    parms['oco_info'])

    class _CaptureRetInfo(mpy.ReplaceFunctionObject):
        def __init__(self):
            self.ret_info = None
            self.retrieval_vec = None

        def should_replace_function(self, func_name, parms):
            # Never replace the function, just grab the argument
            self.ret_info = parms["i_ret_info"]
            self.retrieval_vec = parms["i_retrieval_vec"]

class RefractorTropOrOmiFmBase(mpy.ReplaceFunctionObject if mpy.have_muses_py else object):
    '''
    NOTE - this is deprecated

    Base class for RefractorTropOmiFm and RefractorOmiFm (there is enough
    overlap it is worth combining them). This adapts a ReFRACtor forward model
    to replace the tropomi_fm or omi_fm call in muses-py.

    An object needs to be registered with muses-py to get called in place
    of tropomi_fm or omi_fm. This can be done with a call to the 
    register_with_muses_py function.
    '''

    def __init__(self, func_name="tropomi_fm",
                 py_retrieve_debug=False, py_retrieve_vlidort_nstokes=2,
                 py_retrieve_vlidort_nstreams=4):
        self.sv_extra_index = {}
        self.rundir = "."
        self.ret_info = None
        self.retrieval_vec = None
        self.rf_uip = None
        self.py_retrieve_debug=py_retrieve_debug
        self.py_retrieve_vlidort_nstokes=py_retrieve_vlidort_nstokes
        self.py_retrieve_vlidort_nstreams=py_retrieve_vlidort_nstreams
        self.func_name = func_name

    def register_with_muses_py(self):
        '''Register this object and the helper objects with muses-py,
        to replace a call to omi_fm.
        '''
        mpy.register_replacement_function(self.func_name, self)
        WatchUipCreation().add_notify_object(self)
        WatchUipUpdate().add_notify_object(self)

    def unregister_with_muses_py(self):
        mpy.unregister_replacement_function(self.func_name)
        WatchUipCreation().remove_notify_object(self)
        WatchUipUpdate().remove_notify_object(self)

    def should_replace_function(self, func_name, parms):
        # Currently we only handle the OMI instrument. For other
        # instruments just continue using the normal omi_fm.
        if (self.func_name == "tropomi_fm" and
            'TROPOMI' in parms['i_uip']['instruments']):
            return True
        if (self.func_name == "omi_fm" and
            'OMI' in parms['i_uip']['instruments']):
            return True
        return False

    @classmethod
    def uip_from_muses_retrieval_step(cls, rstep, iteration, pickle_file):
        '''Grab the UIP and directory that can be used to call
        tropomi_fm/omi_fm.
        This starts with MusesRetrievalStep, and gets the UIP passed to
        tropomi_fm in the given iteration number (1 based). Output is
        written to the pickle file, which can then be used for calling
        tropomi_fm/omi_fm.'''
        cretinfo = _CaptureRetInfo()
        with register_replacement_function_in_block("update_uip", cretinfo):
            with register_replacement_function_in_block("fm_wrapper",
                                 _CaptureUip(func_count=iteration)):
                try:
                    rstep.run_retrieval()
                except _FakeUipExecption as e:
                    res = RefractorUip(uip=e.uip, ret_info=cretinfo.ret_info,
                        windows=e.windows, oco_info=e.oco_info,
                        retrieval_vec=cretinfo.retrieval_vec,
                        strategy_table=os.environ["MUSES_DEFAULT_RUN_DIR"] + "/Table.asc")
        res.tar_directory()
        pickle.dump(res, open(pickle_file, "wb"))

    def run_pickle_file(self, pickle_file, path=".",
                        osp_dir=None, gmao_dir=None,
         vlidort_cli="~/muses/muses-vlidort/build/release/vlidort_cli"):
        '''This goes with uip_from_muses_retrieval_step, it turns around
        and calls tropomi_fm/omi_fm with the saved data.'''
        curdir = os.getcwd()
        need_oss_delete = False
        try:
            uip = RefractorUip.load_uip(pickle_file,path=path,
                                        change_to_dir=True, osp_dir=osp_dir,
                                        gmao_dir=gmao_dir)
            self.ret_info = uip.ret_info
            self.retrieval_vec = np.copy(uip.retrieval_vec)
            self.rundir = uip.capture_directory.rundir
            # This might not be the best place for this, but we need to initialize
            # OSS code if it is going to be used
            if('CRIS' in uip.uip["instruments"]):
                os.environ["MUSES_PYOSS_LIBRARY_DIR"] = mpy.pyoss_dir
                uall = mpy.struct_combine(uip.uip, uip.uip['uip_CRIS'])
                uall,_,_ = mpy.fm_oss_init(mpy.ObjectView(uall), "CRIS")
                need_oss_delete = True
                uall = mpy.fm_oss_windows(mpy.ObjectView(uall))
            with muses_py_call(".", vlidort_cli=vlidort_cli,debug=self.py_retrieve_debug,
                               vlidort_nstokes=self.py_retrieve_vlidort_nstokes,
                               vlidort_nstreams=self.py_retrieve_vlidort_nstreams):
                if(self.func_name == "tropomi_fm"):
                    return self.tropomi_fm(uip.uip_all)
                else:
                    return self.omi_fm(uip.uip_all)
        finally:
            os.chdir(curdir)
            if(need_oss_delete):
                mpy.fm_oss_delete()

    @property
    def vlidort_input(self):
        return f"{self.rundir}/{self.rf_uip.vlidort_input}"

    @property
    def vlidort_output(self):
        return f"{self.rundir}/{self.rf_uip.vlidort_output}"

    @property
    def clear_in(self):
        if(not self.py_retrieve_debug):
            raise RuntimeError("You need to run with py_retrieve_debug=True to get the iteration output if you want to view it.")
        iteration = self.rf_uip.uip['iteration']
        ii_mw = 0
        return f"{self.vlidort_input}/Iter{iteration:02d}/MW{ii_mw+1:03d}/clear"

    @property
    def cloudy_in(self):
        if(not self.py_retrieve_debug):
            raise RuntimeError("You need to run with py_retrieve_debug=True to get the iteration output if you want to view it.")
        iteration = self.rf_uip.uip['iteration']
        ii_mw = 0
        return f"{self.vlidort_input}/Iter{iteration:02d}/MW{ii_mw+1:03d}/cloudy"
    
    @property
    def clear_out(self):
        if(not self.py_retrieve_debug):
            raise RuntimeError("You need to run with py_retrieve_debug=True to get the iteration output if you want to view it.")
        iteration = self.rf_uip.uip['iteration']
        ii_mw = 0
        return f"{self.vlidort_output}/Iter{iteration:02d}/MW{ii_mw+1:03d}/clear"

    @property
    def cloudy_out(self):
        if(not self.py_retrieve_debug):
            raise RuntimeError("You need to run with py_retrieve_debug=True to get the iteration output if you want to view it.")
        iteration = self.rf_uip.uip['iteration']
        ii_mw = 0
        return f"{self.vlidort_output}/Iter{iteration:02d}/MW{ii_mw+1:03d}/cloudy"

    def in_dir(self, do_cloud):
        '''Either cloudy_in or clear_in depending on do_cloud'''
        return self.cloudy_in if do_cloud else self.clear_in

    def out_dir(self, do_cloud):
        '''Either cloudy_out or clear_out depending on do_cloud'''
        return self.cloudy_out if do_cloud else self.clear_out

    def replace_function(self, func_name, parms):
        if(func_name == "tropomi_fm"):
            return self.tropomi_fm(**parms)
        return self.omi_fm(**parms)
        

    def fd_jac(self, index, delta):
        '''Calculate a finite difference jacobian for one index. We do
        just one index because these take a while to run, and it can be
        useful to go index by index.

        Return the finite difference and value, and also the jacobian
        as returned by tropomi_fm/omi_fm, i.e. this can be used to evaluate
        how accurate the tropomi_fm/omi_fm jacobian is.
        '''
        # Save so we can reset this value before exiting.
        retrieval_vec_0 = np.copy(self.rf_uip.retrieval_vec)
        if(self.func_name == "tropomi_fm"):
            f = self.tropomi_fm
        else:
            f = self.omi_fm
        with(muses_py_call(self.rundir)):
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
            jacfd = ((rad1-meas_rad1['measured_radiance_field']) -
                     (rad0-meas_rad0['measured_radiance_field'])) / delta
            # The logic in pack_tropomi_jacobian over counts the size of
            # atmosphere jacobians by 1 for each species. This is harmless,
            # it gives an extra row of zeros that then gets trimmed before
            # leaving
            # fm_wrapper. But we need to trim this to do this step
            if(jac.shape[0] > self.ret_info['basis_matrix'].shape[1]):
                jac = jac[:self.ret_info['basis_matrix'].shape[1], :]
            jaccalc = np.matmul(self.ret_info['basis_matrix'], jac)[index]
            return jacfd, jaccalc

    def update_retrieval_vec(self, retrieval_vec):
        '''Update the retrieval vector, both saved in this class and used
        by py-retrieve'''
        self.rf_uip.update_uip(retrieval_vec)
        self.retrieval_vec = np.copy(retrieval_vec)
        fm_vec = np.matmul(retrieval_vec,
                           self.ret_info['basis_matrix'])
        print(retrieval_vec, fm_vec)
        self.update_state(fm_vec)

    def tropomi_fm(self, i_uip, **kwargs):
        '''Substitutes for the py-retrieve tropomi_fm function

        This returns
        (o_jacobian, o_radiance, o_measured_radiance_tropomi, o_success_flag)

        o_success_flag is 1 if the data is good, 0 otherwise.
        '''
        self.rf_uip = RefractorUip(i_uip)
        self.rf_uip.rundir = os.getcwd()
        self.rf_uip.ret_info = self.ret_info
        self.rf_uip.retrieval_vec = np.copy(self.retrieval_vec)
        mrad = self.observation.radiance(0)
        o_measured_radiance_tropomi = {"measured_radiance_field" : mrad.spectral_range.data, "measured_nesr" : mrad.spectral_range.uncertainty}
        o_success_flag = 1

        spec = self.radiance_all()
        o_radiance = spec.spectral_range.data.copy()
        o_jacobian = spec.spectral_range.data_ad.jacobian.transpose().copy()

        if(not mrad.spectral_range.data_ad.is_constant):
            # - because we are giving the jacobian of fm - rad
            o_jacobian -= mrad.spectral_range.data_ad.jacobian.transpose()
        # We've calculated the jacobian relative to the full state vector,
        # including specifies that aren't used by OMI/TROPOMI. py-retrieve
        # expects just the subset, so we need to subset the jacobian
        our_jac = [spec in self.rf_uip.uip_all['jacobians'] for spec in i_uip['speciesListFM'] ]
        return (o_jacobian[our_jac,:], o_radiance, o_measured_radiance_tropomi, o_success_flag)

    def omi_fm(self, i_uip, **kwargs):
        '''Substitutes for the py-retrieve omi_fm function

        This returns
        (o_jacobian, o_radiance, o_measured_radiance_omi, o_success_flag)

        o_success_flag is 1 if the data is good, 0 otherwise.
        '''

        self.rf_uip = RefractorUip(i_uip)
        self.rf_uip.rundir = os.getcwd()
        self.rf_uip.ret_info = self.ret_info
        self.rf_uip.retrieval_vec = np.copy(self.retrieval_vec)
        o_measured_radiance_omi = self.rf_uip.measured_radiance("OMI")
        o_success_flag = 1

        spec = self.radiance_all()
        o_radiance = spec.spectral_range.data.copy()
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
        mw = [slice(0, self.rf_uip.nfreq_mw(0, "OMI")),
              slice(self.rf_uip.nfreq_mw(0, "OMI"), None)]
        if('OMINRADWAVUV1' in self.rf_uip.state_vector_params):
            indx = self.rf_uip.uip["speciesListFM"].index('OMINRADWAVUV1')
            o_jacobian[indx, mw[0]] = \
                o_measured_radiance_omi['normwav_jac'][mw[0]]
        if('OMINRADWAVUV2' in self.rf_uip.state_vector_params):
            indx = self.rf_uip.uip["speciesListFM"].index('OMINRADWAVUV2')
            o_jacobian[indx, mw[1]] = \
                o_measured_radiance_omi['normwav_jac'][mw[1]]
        if('OMIODWAVUV1' in self.rf_uip.state_vector_params):
            indx = self.rf_uip.uip["speciesListFM"].index('OMIODWAVUV1')
            o_jacobian[indx, mw[0]] = \
                o_measured_radiance_omi['odwav_jac'][mw[0]]
        if('OMIODWAVUV2' in self.rf_uip.state_vector_params):
            indx = self.rf_uip.uip["speciesListFM"].index('OMIODWAVUV2')
            o_jacobian[indx, mw[1]] = \
                o_measured_radiance_omi['odwav_jac'][mw[1]]
        if('OMIODWAVSLOPEUV1' in self.rf_uip.state_vector_params):
            indx = self.rf_uip.uip["speciesListFM"].index('OMIODWAVSLOPEUV1')
            o_jacobian[indx, mw[0]] = \
                o_measured_radiance_omi['odwav_slope_jac'][mw[0]]
        if('OMIODWAVSLOPEUV2' in self.rf_uip.state_vector_params):
            indx = self.rf_uip.uip["speciesListFM"].index('OMIODWAVSLOPEUV2')
            o_jacobian[indx, mw[1]] = \
                o_measured_radiance_omi['odwav_slope_jac'][mw[1]]

        return (o_jacobian, o_radiance, o_measured_radiance_omi, o_success_flag)
    
    def radiance_all(self, skip_jacobian=False):
        '''The forward model radiance_all results'''
        raise NotImplementedError()

    def invalidate_cache(self):
        '''Called when a new strategy step starts, should invalidate any 
        caching (e.g., a ForwardModel)'''
        pass

    def update_state(self, rvec, parms=None):
        '''Called with muses-py updated the state vector.'''
        self.update_state_do(rvec, parms)

    def update_state_do(self, rvec, parms=None):
        '''Called with muses-py updated the state vector.'''
        pass

    # To do initial comparisons between py-retrieve and ReFRACtor it
    # can be useful to have detailed information about the run. We
    # supply functions here, which are really just for diagnostic use.
    # The default is to raise a NotImplementedError, but we can override
    # this for various derived classes to we can compare things. A derived
    # class might not implement this - this fine and it just means we can't
    # do a details comparison of things.
    def raman_ring_spectrum(self, do_cloud):
        '''Return Spectrum of Raman scattering, clear or cloudy.
        Note this is the "ring" calculation, which gets translated to the
        raman correction by a scale factor + 1'''
        raise NotImplementedError()

    def surface_albedo(self, do_cloud):
        '''Return Spectrum of surface albedo, clear or cloudy'''
        raise NotImplementedError()

    def geometry(self, do_cloud):
        '''Return solar zenith angle, observation zenith, and
        relative azimuth, clear or cloudy'''
        raise NotImplementedError()

    def pressure_grid(self, do_cloud):
        '''Return pressure grid for each level, clear or cloudy'''
        return NotImplementedError()

    def temperature_grid(self, do_cloud):
        '''Return temperature  grid for each level, clear or cloudy'''
        return NotImplementedError()

    def altitude_grid(self, do_cloud):
        '''Return pressure grid for each level, clear or cloudy'''
        return NotImplementedError()

    def gas_number_density(self, do_cloud):
        '''Return gas numebr density, clear or cloudy.'''
        return NotImplementedError()

    def taur(self, do_cloud):
        '''Return optical depth from Rayleigh, clear and cloudy.'''
        return NotImplementedError()

    def taug(self, do_cloud):
        '''Return optical depth from Gas (e.g., O3), clear and cloudy.'''
        return NotImplementedError()

    def taut(self, do_cloud):
        '''Return total optical depth, clear and cloudy.'''
        return NotImplementedError()

    def rt_radiance(self, do_cloud):
        '''Return the radiance from the RT (e.g., VLIDORT, PCA, or LIDORT),
        which is before any spectrum effects are added (e.g., MusesRaman).'''
        return NotImplementedError()
    
class RefractorTropOrOmiFmPyRetrieve(RefractorTropOrOmiFmBase):
    '''
    NOTE - this is deprecated

    Turn around and call tropomi_fm/omi_fm, without change. This
    gives a way to do a direct comparison with py-retrieve vs
    ReFRACtor. Ultimately this should give the same results as
    RefractorTropOrOmiFmMusesPy, but this skips the mucking around
    with jacobians etc. that RefractorTropOrOmiFmBase does - so this
    lets us establish that RefractorTropOmiFmMusesPy is correct.'''
    def omi_fm(self, i_uip, **kwargs):
        self.rf_uip = RefractorUip(i_uip)
        self.rf_uip.rundir = os.getcwd()
        self.rf_uip.ret_info = self.ret_info
        self.rf_uip.retrieval_vec = np.copy(self.retrieval_vec)
        with suppress_replacement("omi_fm"):
            return mpy.omi_fm(i_uip)

    def tropomi_fm(self, i_uip, **kwargs):
        self.rf_uip = RefractorUip(i_uip)
        self.rf_uip.rundir = os.getcwd()
        self.rf_uip.ret_info = self.ret_info
        self.rf_uip.retrieval_vec = np.copy(self.retrieval_vec)
        with suppress_replacement("tropomi_fm"):
            return mpy.tropomi_fm(i_uip)
    
class RefractorTropOrOmiFmMusesPy(RefractorTropOrOmiFmBase):
    '''This just turns around and calls MusesPyForwardModel. This is useful
    to test the interconnection with py-retrieve, since a retrieval should be
    identical to one without a replacement.'''

    def radiance_all(self, skip_jacobian=False):
        '''The forward model radiance_all results'''
        # In the next call for tropomi_fm, we don't actually want to
        # replace this
        with suppress_replacement("tropomi_fm"):
            fm = MusesPyForwardModel(self.rf_uip, use_current_dir=True)
            return fm.radiance_all(skip_jacobian=skip_jacobian)
    
    def update_state_do(self, rvec, parms=None):
        '''Called with muses-py updated the state vector.'''
        logger.info(f"RefractorTropOrOmiFmMusesPy updating state to: {rvec.shape} : {rvec}")

    def raman_ring_spectrum(self, do_cloud):
        '''Return Spectrum of Raman scattering, clear or cloudy.
        Note this is the "ring" calculation, which gets translated to the
        raman correction by a scale factor + 1.

        Need to run with py_retrieve_debug=True for the data to be available for this
        function. '''
        ring = mpy.read_rtm_output(self.out_dir(do_cloud), "Ring.asc")
        mw = self.rf_uip.micro_windows(0).value[0,:]
        slc = (ring[0,:] >= mw[0]) & (ring[0,:] <= mw[1])
        sd = rf.SpectralDomain(ring[0,slc], rf.Unit("nm"))
        # Units don't matter here, but lets just assign something reasonable
        sr = rf.SpectralRange(ring[1,slc], rf.Unit("ph / s / m^2 / micron W / (cm^-1) / (ph / (s) / (micron)) sr^-1"))
        return rf.Spectrum(sd, sr)
    
    def surface_albedo(self, do_cloud):
        '''Return Spectrum of surface albedo, clear or cloudy

        Need to run with py_retrieve_debug=True for the data to be available for this
        function. '''
        alb = pd.read_csv(f"{self.in_dir(do_cloud)}/Surfalb_MW001.asc",
                               sep='\s+', skiprows=1,header=None,
                               names=['wavelength','albedo']).to_numpy()
        mw = self.rf_uip.micro_windows(0).value[0,:]
        slc = (alb[:,0] >= mw[0]) & (alb[:,0] <= mw[1])
        sd = rf.SpectralDomain(alb[slc,0], rf.Unit("nm"))
        sr = rf.SpectralRange(alb[slc,1], rf.Unit("dimensionless"))
        return rf.Spectrum(sd, sr)

    def geometry(self, do_cloud):
        '''Return solar zenith angle, observation zenith, and
        relative azimuth, clear or cloudy.

        Need to run with py_retrieve_debug=True for the data to be available for this
        function. '''
        
        vga = pd.read_csv(f"{self.in_dir(do_cloud)}/Vga_MW001.asc", sep='[ ,]+',engine='python')
        return (vga["SZA"][0], vga["VZA"][0], vga["RAZ"][0])

    def pressure_grid(self, do_cloud):
        '''Return pressure grid for each level, clear or cloudy

        Need to run with py_retrieve_debug=True for the data to be available for this
        function. '''
        
        atm = pd.read_csv(f"{self.in_dir(do_cloud)}/Atm_level.asc",
                          sep='\s+', skiprows=2,
                          header=None,
                          names=["Pres(mb)", "T(K)", 'Altitude(m)'])
        return rf.ArrayWithUnit(atm['Pres(mb)'].to_numpy(), "hPa")

    def temperature_grid(self, do_cloud):
        '''Return temperature  grid for each level, clear or cloudy

        Need to run with py_retrieve_debug=True for the data to be available for this
        function. '''
        
        atm = pd.read_csv(f"{self.in_dir(do_cloud)}/Atm_level.asc",
                          sep='\s+', skiprows=2,
                          header=None,
                          names=["Pres(mb)", "T(K)", 'Altitude(m)'])
        # This doesn't include the temperature shift in the output file, although the
        # shifted temperature is used in the calculation, see get_tropomi_o3xsec in
        # py-retrieve
        if(self.func_name == "tropomi_fm"):
            toffset = self.rf_uip.tropomi_params["temp_shift_BAND3"]
        else:
            toffset = 0
        return rf.ArrayWithUnit(atm['T(K)'].to_numpy() + toffset, "K")

    def altitude_grid(self, do_cloud):
        '''Return pressure grid for each level, clear or cloudy

        Need to run with py_retrieve_debug=True for the data to be available for this
        function. '''
        
        atm = pd.read_csv(f"{self.in_dir(do_cloud)}/Atm_level.asc",
                          sep='\s+', skiprows=2,
                          header=None,
                          names=["Pres(mb)", "T(K)", 'Altitude(m)'])
        return rf.ArrayWithUnit(atm['Altitude(m)'].to_numpy(), "m")

    def gas_number_density(self, do_cloud):
        '''Return gas number density, clear or cloudy.

        Need to run with py_retrieve_debug=True for the data to be available for this
        function. '''
        
        atm = pd.read_csv(f"{self.in_dir(do_cloud)}/Atm_layer.asc",
                          sep='\s+', skiprows=2,
                          header=None,
                          names=["Pressure layer(mb)", "Temperature layer (K)",
                                 "Gas Density"])
        return rf.ArrayWithUnit(atm["Gas Density"].to_numpy(), "cm^-2")

    def taur(self, do_cloud):
        '''Return optical depth from Rayleigh, clear and cloudy.

        Need to run with py_retrieve_debug=True for the data to be available for this
        function. '''
        
        t = pd.read_csv(f"{self.out_dir(do_cloud)}/taur.asc", sep='\s+',
                        header=None).to_numpy()
        mw = self.rf_uip.micro_windows(0).value[0,:]
        slc = (t[:,0] >= mw[0]) & (t[:,0] <= mw[1])
        sd = rf.SpectralDomain(t[slc,0], rf.Unit("nm"))
        return sd, t[slc,1:]

    def taug(self, do_cloud):
        '''Return optical depth from Gas (e.g., O3), clear and cloudy.

        Need to run with py_retrieve_debug=True for the data to be available for this
        function. '''
        
        t = pd.read_csv(f"{self.out_dir(do_cloud)}/taug.asc", sep='\s+',
                        header=None).to_numpy()
        mw = self.rf_uip.micro_windows(0).value[0,:]
        slc = (t[:,0] >= mw[0]) & (t[:,0] <= mw[1])
        sd = rf.SpectralDomain(t[slc,0], rf.Unit("nm"))
        return sd, t[slc,1:]

    def taut(self, do_cloud):
        '''Return total optical depth, clear and cloudy.

        Need to run with py_retrieve_debug=True for the data to be available for this
        function. '''
        
        t = pd.read_csv(f"{self.out_dir(do_cloud)}/taut.asc", sep='\s+',
                        header=None).to_numpy()
        mw = self.rf_uip.micro_windows(0).value[0,:]
        slc = (t[:,0] >= mw[0]) & (t[:,0] <= mw[1])
        sd = rf.SpectralDomain(t[slc,0], rf.Unit("nm"))
        return sd, t[slc,1:]

    def rt_radiance(self, do_cloud):
        '''Return the radiance from the RT (e.g., VLIDORT, PCA, or LIDORT),
        which is before any spectrum effects are added (e.g., MusesRaman).


        Need to run with py_retrieve_debug=True for the data to be available for this
        function. '''
        rad = mpy.read_rtm_output(self.out_dir(do_cloud), "Radiance.asc")
        mw = self.rf_uip.micro_windows(0).value[0,:]
        slc = (rad[0,:] >= mw[0]) & (rad[0,:] <= mw[1])
        sd = rf.SpectralDomain(rad[0,slc], rf.Unit("nm"))
        sr = rf.SpectralRange(rad[1,slc], rf.Unit("sr^-1"))
        return rf.Spectrum(sd, sr)

    @property
    def observation(self):
        raise NotImplementedError()

class _CaptureSpectrum(rf.ObserverPtrNamedSpectrum):
    '''Helper class to capture the radiative transfer output before we apply the
    RamanSioris effect.'''
    def __init__(self):
        super().__init__()
        self.spectrum = []

    def notify_update(self, named_spectrum):
        # The name we use right after the RT is high_res_rt.
        if named_spectrum.name == "high_res_rt":
            self.spectrum.append(named_spectrum.copy())
    
    
class RefractorTropOrOmiFm(RefractorTropOrOmiFmBase):
    '''
    NOTE - this is deprecated

    Use a ReFRACtor ForwardModel as a replacement for tropomi_fm/omi_fm.'''

    def __init__(self, func_name, **kwargs):
        super().__init__(func_name=func_name)
        self._fm_cache = None
        self._obj_creator_cache = None
        self._sv = None
        self.xsec_table_to_notify = None
        self.obj_creator_args = kwargs

    def invalidate_cache(self):
        self._fm_cache = None
        self._obj_creator_cache = None
        self._sv = None

    def update_state_do(self, rvec, parms=None):
        # Mild logic complication, we can always create a state vector
        # from self.obj_creator.state_vector, but here we only want to
        # update a state vector that has already been created and attached
        # to a forward model. So we cache a copy in self._sv when we
        # create the forward model, and only update that one.
        if(self._sv is not None):

            update_vec = rvec[self.rf_uip.state_vector_update_indexes]

            self._sv.update_state(update_vec)

            logger.info(f"RefractorTropOrOmiFm updating state to:\n{self._sv}")

    @property
    def fm(self):
        '''Forward model, creating a new one if needed'''
        if(self._fm_cache is None):
            self._fm_cache = self.obj_creator.forward_model
            self._sv = self.obj_creator.state_vector
        return self._fm_cache

    @property
    def observation(self):
        # Creating the state vector sets up the connection between
        # observation and the state vector.
        raise NotImplementedError()

    def radiance_all(self, skip_jacobian=False):
        logger.info(f"FM state vector:\n{self.obj_creator.state_vector}")
        spec = self.fm.radiance_all(skip_jacobian=skip_jacobian)
        return spec

    def raman_ring_spectrum(self, do_cloud):
        '''Return Spectrum of Raman scattering, clear or cloudy.
        Note this is the "ring" calculation, which gets translated to the
        raman correction by a scale factor + 1'''
        self.fm.set_do_cloud(do_cloud)
        sd = self.fm.spectral_domain(0)
        # Units don't matter here, but lets just assign something reasonable
        sr = rf.SpectralRange([1] * sd.rows, rf.Unit("ph / s / m^2 / micron W / (cm^-1) / (ph / (s) / (micron)) sr^-1"))
        s = rf.Spectrum(sd, sr)
        self.obj_creator.raman_effect[0].apply_effect(s, self.fm.underlying_forward_model.spectral_grid)
        sr = rf.SpectralRange((s.spectral_range.data -1) /
                  self.obj_creator.raman_effect[0].coefficient[0].value,
                  s.spectral_range.units)
        s = rf.Spectrum(sd, sr)
        return s
        
    def surface_albedo(self, do_cloud):
        '''Return Spectrum of surface albedo, clear or cloudy'''
        self.fm.set_do_cloud(do_cloud)
        sd = self.fm.spectral_domain(0)
        sr = rf.SpectralRange(np.array([self.obj_creator.ground.surface_parameter(wn,0).value[0] for wn in sd.convert_wave("cm^-1")]), rf.Unit("dimensionless"))
        return rf.Spectrum(sd, sr)

    def geometry(self, do_cloud):
        '''Return solar zenith angle, observation zenith, and
        relative azimuth, clear or cloudy'''
        return (self.rf_uip.solar_zenith(0), self.rf_uip.observation_zenith(0),
                self.rf_uip.relative_azimuth(0))

    def pressure_grid(self, do_cloud):
        '''Return pressure grid for each level, clear or cloudy'''
        self.fm.set_do_cloud(do_cloud)
        pgrid = self.obj_creator.pressure.pressure_grid()
        return rf.ArrayWithUnit(pgrid.value.value, pgrid.units)

    def temperature_grid(self, do_cloud):
        '''Return temperature  grid for each level, clear or cloudy'''
        self.fm.set_do_cloud(do_cloud)
        tgrid = self.obj_creator.temperature.temperature_grid(self.obj_creator.pressure)
        return rf.ArrayWithUnit(tgrid.value.value, tgrid.units)

    def altitude_grid(self, do_cloud):
        '''Return pressure grid for each level, clear or cloudy'''
        self.fm.set_do_cloud(do_cloud)
        agrid = self.obj_creator.atmosphere.altitude(0)
        return rf.ArrayWithUnit(agrid.value.value, agrid.units)

    def gas_number_density(self, do_cloud):
        '''Return gas numebr density, clear or cloudy.'''
        self.fm.set_do_cloud(do_cloud)
        glay = self.obj_creator.absorber.gas_number_density_layer(0)
        return rf.ArrayWithUnit(glay.value.value[:,0], glay.units)

    def taur(self, do_cloud):
        '''Return optical depth from Rayleigh, clear and cloudy.'''
        self.fm.set_do_cloud(do_cloud)
        sd = self.fm.spectral_domain(0)
        g = np.vstack([np.array([self.obj_creator.rayleigh.optical_depth_each_layer(wn,0).value for wn in sd.convert_wave("cm^-1")])])
        return sd, g

    def taug(self, do_cloud):
        '''Return optical depth from Gas (e.g., O3), clear and cloudy.'''
        self.fm.set_do_cloud(do_cloud)
        sd = self.fm.spectral_domain(0)
        g = np.vstack([np.array([self.obj_creator.absorber.optical_depth_each_layer(wn,0).value[:,0] for wn in sd.convert_wave("cm^-1")])])
        return sd, g

    def taut(self, do_cloud):
        '''Return total optical depth, clear and cloudy.'''
        self.fm.set_do_cloud(do_cloud)
        sd = self.fm.spectral_domain(0)
        g = np.vstack([np.array([self.obj_creator.atmosphere.optical_depth_wrt_state_vector(wn,0).value for wn in sd.convert_wave("cm^-1")])])
        return sd, g

    def rt_radiance(self, do_cloud):
        '''Return the radiance from the RT (e.g., VLIDORT, PCA, or LIDORT),
        which is before any spectrum effects are added (e.g., MusesRaman).'''
        self.fm.set_do_cloud(do_cloud)
        cap = _CaptureSpectrum()
        self.fm.underlying_forward_model.add_observer(cap)
        self.fm.underlying_forward_model.radiance_all()
        self.fm.underlying_forward_model.remove_observer(cap)
        return cap.spectrum[0]

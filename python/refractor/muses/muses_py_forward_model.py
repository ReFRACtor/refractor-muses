import refractor.muses.muses_py as mpy
import refractor.framework as rf
import os
import tempfile
import copy

class MusesPyForwardModel:
    '''This is an adapter than makes a muse-py forward model call look
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
        

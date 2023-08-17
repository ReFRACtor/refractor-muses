from . import muses_py as mpy
from .replace_function_helper import suppress_replacement
import os
import copy

class WatchOssInit(mpy.ObserveFunctionObject if mpy.have_muses_py else object):
    '''Helper object to update osswrapper.have_oss when py-retrieve calls
    fm_oss_init.'''
    def notify_function_call(self, func_name, parms):
        osswrapper.have_oss = True
        osswrapper.first_oss_initialize = False
        
class WatchOssDelete(mpy.ObserveFunctionObject if mpy.have_muses_py else object):
    '''Helper object to update osswrapper.have_oss when py-retrieve calls
    fm_oss_delete.'''
    def notify_function_call(self, func_name, parms):
        osswrapper.have_oss = False

class osswrapper:
    '''The OSS library needs to be initialized, have windows set up,
    and freed when done. But it is a global function, e.g., you can't
    have two window sets available (standard global variables in
    fortran code). Depending on how a function that needs the OSS is
    called this may or may not have already been set up.

    This simple class provides a context manager than ensure that we only
    do the initialization once, and clean up wherever that occurs. This
    wrapper can then be nested - so a function in an osswrapper can call
    another function that uses the osswrapper and the OSS initialization
    will only happen once.

    Note that if we do the initialization, the uip passed in is modified
    to add oss_jacobianList. This duplicates what muses-py does.

    We also interact with muse py to catch calls to fm_oss_init and
    fm_oss_delete done outside of ReFRACtor (e.g., in run_retrieval).

    We unfortunately can't do anything to ensure that we don't try
    creating two oss_wrapper with different uip. This doesn't work
    and will fail.

    Note, another probably bug with OSS is that *first* call to it
    returns difference results then future calls. Not clear what is going
    on here, but it makes it hard to have repeatable code. To work around
    this the very first time we initialize code we do it twice -
    initialize + delete followed by a second initialization. This should
    probably get sorted out at some point, but for now we just work around
    this.
    '''
    have_oss = False
    first_oss_initialize = True
    def __init__(self, uip):
        if(hasattr(uip, 'as_dict')): 
            self.uip = uip.as_dict(uip)
        else:
            self.uip = uip
        self.uip = copy.deepcopy(self.uip)
        self.need_cleanup = False

    @classmethod
    def register_with_muses_py(self):
        mpy.register_observer_function("fm_oss_init", WatchOssInit())
        mpy.register_observer_function("fm_oss_delete", WatchOssDelete())
        
    def __enter__(self):
        uip_all = None
        if(not osswrapper.have_oss):
            for inst in ('CRIS','AIRS', 'TES'):
                if(f'uip_{inst}' in self.uip):
                    os.environ["MUSES_PYOSS_LIBRARY_DIR"] = mpy.pyoss_dir
                    # Delete frequencyList if found. I don't think we
                    # run into that in actual muses-py runs, but we do
                    # with some of our test data based on where we are
                    # in the processing.
                    self.uip.pop("frequencyList", None)
                    uip_all = mpy.struct_combine(self.uip, self.uip[f"uip_{inst}"])
                    # Special handling for the first time through, working
                    # around what is a bug or "feature" of the OSS code
                    if(osswrapper.first_oss_initialize):
                        with suppress_replacement("fm_oss_init"):
                            (uip_all, frequencyListFullOSS, jacobianList) = mpy.fm_oss_init(mpy.ObjectView(uip_all), inst)
                        mpy.fm_oss_windows(mpy.ObjectView(uip_all))
                        with suppress_replacement("fm_oss_delete"):
                            mpy.fm_oss_delete()
                        osswrapper.first_oss_initialize = False
                        
                    with suppress_replacement("fm_oss_init"):
                        (uip_all, frequencyListFullOSS, jacobianList) = mpy.fm_oss_init(mpy.ObjectView(uip_all), inst)
                        self.uip['oss_jacobianList'] = jacobianList
                    mpy.fm_oss_windows(mpy.ObjectView(uip_all))
                    self.need_cleanup = True
                    osswrapper.have_oss =  True
        if(uip_all is not None):
            self.oss_dir_lut = uip_all["oss_dir_lut"]
            self.oss_jacobianList = uip_all["oss_jacobianList"]
            self.oss_frequencyList = uip_all["oss_frequencyList"]
            self.oss_frequencyListFull = uip_all["oss_frequencyListFull"]
        else:
            self.oss_dir_lut = None
            self.oss_jacobianList = None
            self.oss_frequencyList = None
            self.oss_frequencyListFull = None
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if(self.need_cleanup):
            with suppress_replacement("fm_oss_delete"):
                mpy.fm_oss_delete()
            self.need_cleanup = False
            osswrapper.have_oss = False

if(mpy.have_muses_py):
    osswrapper.register_with_muses_py()
            
__all__ = ["osswrapper"]

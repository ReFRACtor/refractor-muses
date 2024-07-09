from .creator_handle import CreatorHandleSet, CreatorHandle
import refractor.muses.muses_py as mpy
import logging
import abc

logger = logging.getLogger("py-retrieve")

class SpectralWindowHandle(CreatorHandle, metaclass=abc.ABCMeta):
    '''Base class for SpectralWindowHandle. Note we use duck typing,
    don't need to actually derive from this object. But it can be
    useful because it 1) provides the interface and 2) documents
    that a class is intended for this.
    '''
    def notify_update_target(self):
        '''Clear any caching associated with assuming the target being retrieved is fixed'''
        # Default is to do nothing
        pass
    
    @abc.abstractmethod
    def spectral_window_dict(self) -> "dict(str, MusesSpectralWindow)":
        '''Return a dictionary that goes from instrument name to the MusesSpectralWindow
        for that instrument. Note because of the extra metadata and bad sample/full band
        handing we need we currently require a MusesSpectralWindow. We could perhaps
        relax this in the future if we have another way of handling this extra functionality.'''
        raise NotImplementedError()


class SpectralWindowHandleSet(CreatorHandleSet):
    '''This takes a retrieval step name and maps that to a dict. The
    dict in turn maps a instrument name to the MusesSpectralWindow to
    use for that instrument.
    '''
    def __init__(self):
        super().__init__("spectral_window_dict")
        
    def spectral_window_dict(self) -> "dict(str, MusesSpectralWindow)":
        '''Return a dictionary that goes from instrument name to the MusesSpectralWindow
        for that instrument. Note because of the extra metadata and bad sample/full band
        handing we need we currently require a MusesSpectralWindow. We could perhaps
        relax this in the future if we have another way of handling this extra functionality.'''
        return self.handle()

class MusesPySpectralWindowHandle(SpectralWindowHandle):
    '''This wraps the old muses-py code for determining the spectral window. Note
    the logic used in this code is a bit complicated, this looks like something that
    has been extended and had special cases added over time. We should probably replace
    this with newer code, but this older wrapper is useful for doing testing if nothing
    else.'''
    def __init__(self):
        pass
    
    
    def notify_update_target(self):
        '''Clear any caching associated with assuming the target being retrieved is fixed'''
        # We'll add grabbing the stuff out of RetrievalConfiguration in a bit
        pass
    
    def spectral_window_dict(self) -> "dict(str, MusesSpectralWindow)":
        # We'll add creating a spectral window dict in a bit
        return None

    def spectral_filename(self, viewing_mode : str,
                          spectral_window_directory : str, retrieval_elements : 'list(str)',
                          step_name : str, retrieval_type : str, spec_file = None):
        '''Call table_get_spectral_filename. This isn't something we actually use,
        but is nice to call as we work out duplicating this functionality.'''
        # Wrap arguments into format expected by table_new_mw_from_step. This
        # creates a dummy strategy_table dict with the values it expects to find
        stable = { }
        stable["preferences"] = \
            {"viewingMode" : viewing_mode,
             "spectralWindowDirectory" : spectral_window_directory}
        t1 = [",".join(retrieval_elements), step_name, retrieval_type]
        t2 = ["retrievalElements", "stepName", "retrievalType"]
        if(spec_file is not None):
            t1.append(spec_file)
            t2.append("specFile")
        stable["data"] = [" ".join(t1),]
        stable["labels1"] = " ".join(t2)
        stable["numRows"] = 1
        stable["numColumns"] = len(t2)
        return mpy.table_get_spectral_filename(stable, 0)

        

from .creator_handle import CreatorHandleSet, CreatorHandle
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
    
    
    

from .creator_handle import CreatorHandleSet, CreatorHandle
from .muses_spectral_window import MusesSpectralWindow
import refractor.muses.muses_py as mpy
import logging
import abc
import os

logger = logging.getLogger("py-retrieve")

class QaDataHandle(CreatorHandle, metaclass=abc.ABCMeta):
    '''Base class for QaDatawHandle. Note we use duck typing,
    don't need to actually derive from this object. But it can be
    useful because it 1) provides the interface and 2) documents
    that a class is intended for this.
    '''
    def notify_update_target(self, measurement_id : 'MeasurementId'):
        '''Clear any caching associated with assuming the target being retrieved is fixed'''
        # Default is to do nothing
        pass

    @abc.abstractmethod
    def qa_file_name(self,
           current_strategy_step : 'CurrentStrategyStep') -> "Optional(str)":
        '''Return the qa file name to use.'''
        raise NotImplementedError()


# TODO Rework this. Can we move the whole QA into this?    
class QaDataHandleSet(CreatorHandleSet):
    '''This takes a CurrentStrategyStep and maps that to a QA file name.
    Note it isn't clear that this is really what we want - why can't we just do
    the entire QA processing based off the handle? But at least for now, we are
    wrapping older py-retrieve code that *only* reads the QA data from a file.
    '''
    def __init__(self):
        super().__init__("qa_file_name")

    def qa_file_name(self,
             current_strategy_step : 'CurrentStrategyStep') -> str:
        '''Return the qa file name to use.'''
        return self.handle(current_strategy_step)

class MusesPyQaDataHandle(QaDataHandle):
    '''This wraps the old muses-py code for determining the spectral window. Note
    the logic used in this code is a bit complicated, this looks like something that
    has been extended and had special cases added over time. We should probably replace
    this with newer code, but this older wrapper is useful for doing testing if nothing
    else.'''
    def __init__(self):
        self.viewing_mode = None
        self.spectral_window_directory = None
        self.qa_flag_directory = None
        
    def notify_update_target(self, measurement_id : 'MeasurementId'):
        '''Clear any caching associated with assuming the target being retrieved is fixed'''
        # We'll add grabbing the stuff out of RetrievalConfiguration in a bit
        self.spectral_window_directory = measurement_id.filename("spectralWindowDirectory")
        self.viewing_mode = measurement_id.value("viewingMode")
        self.qa_flag_directory = measurement_id.filename("QualityFlagDirectory")

    def qa_file_name(self,
           current_strategy_step : 'CurrentStrategyStep') -> "Optional(str)":
        '''Return the qa file name to use.'''
        # Name is derived from the microwindows file name
        mwfname = MusesSpectralWindow.muses_microwindows_fname_from_muses_py(
            self.viewing_mode, self.spectral_window_directory,
            current_strategy_step.retrieval_elements,
            current_strategy_step.step_name,
            current_strategy_step.retrieval_type,
            current_strategy_step.microwindow_file_name_override)
        res = os.path.basename(mwfname)
        res = res.replace("Microwindows_", "QualityFlag_Spec_")
        res = res.replace("Windows_", "QualityFlag_Spec_")
        res = f"{self.qa_flag_directory}/{res}"
        # if this does not exist use generic nadir / limb quality flag
        if not os.path.isfile(res):
            logger.warning(f'Could not find quality flag file: {res}')
            viewMode = self.viewing_mode.lower().capitalize()
            res = f"{os.path.dirname(res)}/QualityFlag_Spec_{viewMode}.asc"
            logger.warning(f"Using generic quality flag file: {res}")
            # One last check.
            if not os.path.isfile(res):
                raise RuntimeError(f"Quality flag filename not found: {res}")
        return os.path.abspath(res)

# For now, just fall back to the old muses-py code.    
QaDataHandleSet.add_default_handle(MusesPyQaDataHandle())

__all__ = ["QaDataHandle", "QaDataHandleSet", "MusesPyQaDataHandle"]
    
    

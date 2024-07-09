from .tes_file import TesFile
import logging
import abc
logger = logging.getLogger("py-retrieve")

class FilterMetadata(object, metaclass=abc.ABCMeta):
    '''muses-py code has additional metadata associated with each filter name
    (see MusesSpectralWindow for a description of the filter names). It isn't
    clear how much of this metadata is actually used. Right now, this isn't used
    at all by ReFRACtor, the metadata is only used in the old muses-py UIP structure
    that is used by some ForwardModels.

    In addition to the filter level metadata, there is microwindow level metadata
    (in general a particular filter name has multiple microwindows). That information
    is not maintained in this class, but rather in MusesSpectralWindow.
    '''
    @abc.abstractmethod
    def filter_metadata(self, filter_name : 'Optional(str)') -> dict:
        '''Return a dict with the extra metadata for the given filter_name. As a convenience,
        the filter_name can be passed as None, and a empty dict is returned. This allows
        microwindows with a filter name to be handled transparently, this is just microwindows
        not used by the UIP.'''
        raise NotImplementedError()

class DictFilterMetadata(FilterMetadata):
    '''This is a FilterMetadata where we just store the extra metadata as a dict from
    a filter name to a dict of metadata.  For filter names not in the dict, we return an
    empty dict of metadata.'''
    def __init__(self, metadata : 'dict(str, dict)'):
        self.metadata = metadata

    def filter_metadata(self, filter_name : 'Optional(str)') -> dict:
        if(filter_name is None):
            return {}
        return self.metadata.get(filter_name, {})

class FileFilterMetadata(FilterMetadata):
    '''This is a FilterMetadata where the data is read from a file. Note
    that the file name usually comes from the RetrievalConfiguration as
    "defaultSpectralWindowsDefinitionFilename". This name is a little confusing, the
    defaultSpectralWindowsDefinitionFilename isn't actually extra default spectral windows,
    but rather metadata for spectral windows that isn't already specified in the
    spectral_filename file.'''
    def __init__(self, filename : str):
        self.filename = filename
        f = TesFile(filename)
        self.metadata = {}
        for row in f.table.iloc:
            res = {}
            res["monoextend"] = float(row["MONO_BOUND_EXTENS"])
            res["monoSpacing"] = float(row["MONO_FRQ_SPC"])
            res["speciesList"] = row["LINE_SPECIES_LIST"]
            res["maxopd"] = float(row["MAXOPD"])
            res["spacing"] = float(row["RET_FRQ_SPC"])
            # This doesn't seem to be read anywhere, just hardcoded. Not sure this
            # is actually used, but carry at least for now.
            res["num_points"] = 0 
            self.metadata[row["FILTER"]] = res
    
    def filter_metadata(self, filter_name : 'Optional(str)') -> dict:
        if(filter_name is None):
            return {}
        return self.metadata.get(filter_name, {})

__all__ = ["FilterMetadata", "DictFilterMetadata", "FileFilterMetadata"]    

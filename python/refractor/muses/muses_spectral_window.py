from .tes_file import TesFile
from .filter_metadata import FilterMetadata, DictFilterMetadata
import refractor.framework as rf
import copy
import numpy as np
import refractor.muses.muses_py as mpy

class MusesSpectralWindow(rf.SpectralWindow):
    '''The refractor retrieval just uses normal SpectralWindow (e.g.,
    a SpectralWindowRange).  However there are places where it wants
    1) the data restricted to microwindows but including bad pixels
    (which are otherwise removed with the normal SpectralWindow) and
    2) the full data (referred to as full band).

    This class adds support for this. It wraps around and existing
    SpectralWindow and adds flags that can be set to
    "include_bad_data" or "full_band".

    In addition, the old muses-py code names each of the microwindows
    with a "filter" name. This is similar to, but not the same as the
    sensor index that refractor.framework uses. A particular sensor
    index value may have more than one filter assigned to it.

    The use of the filter name is fairly limited. It is used as metadata in
    the output files, the set of input data read by the observation
    (i.e., avoid reading the full L1B file and just read the data that
    will be used later in a retrieval), and as an index for other
    metadata.

    But we go ahead and include the filter name in the
    MusesSpectralWindow. If you have a new instrument that doesn't
    have filter names, this can be given the value of None.

    There is some additional metadata included in the
    microwindows. The metadata data has two flavors, the filter level
    metadata and microwindow level (in general a given filter has
    multiple microwindows). The filter level metadata is handled a
    separate class FilterMetadata which handles determining the
    metadata, see that class for a description of this. This separate
    class is used by function "muses_microwindows" to get the metadata
    to include.

    The microwindow metadata is filter name (already discussed), the
    RT, and the species list. Note we don't actually use the RT to
    control which radiative transfer code is used, this is just
    metadata passed to the UIP. We pass the RT and species list to the
    constructor.

    In all cases, the metadata can be specified as None. So new
    instruments don't need to make up data here, this metadata is only
    need for ForwardModels that use the old muses-py UIP structure. At
    least currently none of the metadata is used by refractor code,
    this really is only needed to generate the UIP.
    '''

    def __init__(self, spec_win : "Optional(rf.SpectralWindowRange)",
                 obs : 'Optional(MusesObservation)',
                 raman_ext=3.01,
                 instrument_name : 'Optional(str)' = None,
                 filter_metadata : 'Optional(FilterMetadata)' = None,
                 filter_name : 'Optional(np.array)' = None,
                 rt : 'Optional(np.array)' = None,
                 species_list : 'Optional(np.array)' = None):
        '''Create a MusesSpectralWindow. The passed in spec_win should *not* have
        bad samples removed. We get the bad sample for the obs passed in and
        add it to spec_win.

        Right now we only work with a SpectralWindowRange using it's
        bad_sample_mask. We could extend this if needed - we really
        just need a SpectralWindow that we can create two versions - a
        with and without bad sample. But for right now, restrict ourselves to
        SpectralWindowRange.

        As a convenience spec_win and obs can be passed in as None - this is like
        always having a full_band. This is the same as having no SpectralWindow, but
        it is convenient to just always have a SpectralWindow so code doesn't need to have
        a special case.

        In addition to the normal spectral window, there is the one used in the
        RamanSioris calculation. This is a widened range, with the raman_ext added
        to each end. Note in the py-retrieve code the widening is hard coded to 3.01 nm.

        There is additional metadata that the muses-py microwindow structure has. I'm
        not sure how much of this is actually used, but for now we'll keep all this
        metadata as auxiliary information so we can create the microwindows from a
        MusesSpectralWindow. The data is split between a FilterMetadata and information
        in the table read for the filter_name, RT listed in the file (which doesn't actually
        control the RT we use, this is just metadata in the file) and the
        Species.
        '''
        super().__init__()
        self.instrument_name = instrument_name
        if(filter_metadata is None):
            self.filter_metadata = DictFilterMetadata(metadata={})
        else:
            self.filter_metadata = filter_metadata
        # Either take the values passed in, or fill in dummy values for this
        # metadata
        self.filter_name = None
        self.rt = None
        self.species_list = None
        for v, sv in ((filter_name, "filter_name"), (rt, "rt"),
                      (species_list, "species_list")):
            if(v is None):
                if(spec_win is not None):
                    setattr(self, sv, np.full(spec_win.range_array.value.shape, None,
                                              dtype=np.dtype(object)))
            else:
                setattr(self, sv, v)
            
        self.include_bad_sample = False
        self.full_band = False
        self.do_raman_ext = False
        self._raman_ext = raman_ext
        # Let bad samples pass through
        self._spec_win_with_bad_sample = spec_win
        if(spec_win is not None):
            swin = copy.deepcopy(spec_win)
            if(obs is not None):
                # Remove bad samples
                for i in range(obs.num_channels):
                    swin.bad_sample_mask(obs.bad_sample_mask(i), i)
            self._spec_win = swin
            d = spec_win.range_array.value
            draman_ext = np.zeros_like(d)
            for i in range(d.shape[0]):
                for j in range(d.shape[1]):
                    if(d[i,j,1] > d[i,j,0]):
                        draman_ext[i,j,0] = d[i,j,0]-self._raman_ext
                        draman_ext[i,j,1] = d[i,j,1]+self._raman_ext
            draman_ext = rf.ArrayWithUnit_double_3(draman_ext, rf.Unit("nm"))
            self._spec_win_raman_ext = rf.SpectralWindowRange(draman_ext)
        else:
            self._spec_win = None
            self._spec_win_raman_ext = None

    def _v_number_spectrometer(self):
        return self._spec_win.number_spectrometer

    def _v_spectral_bound(self):
        return self._spec_win.spectral_bound

    def desc(self):
        return "MusesSpectralWindow"

    def grid_indexes(self, grid, spec_index):
        if(self._spec_win is None or self.full_band):
            return list(range(grid.data.shape[0]))
        if(self.do_raman_ext):
            return self._spec_win_raman_ext.grid_indexes(grid, spec_index)
        if(self.include_bad_sample):
            return self._spec_win_with_bad_sample.grid_indexes(grid, spec_index)
        return self._spec_win.grid_indexes(grid, spec_index)

    def muses_microwindows(self):
        '''Return the muses-py list of dict structure used as microwindows. This is
        used in a few places, e.g., for creating a UIP for forward models that depend
        on this.'''
        res = []
        d = self._spec_win.range_array.value
        for i in range(d.shape[0]):
            for j in range(d.shape[1]):
                if(d[i,j,0] < d[i,j,1]):
                    v = {'start' : d[i,j,0],
                         'endd' : d[i,j,1],
                         'instrument' : self.instrument_name,
                         'RT' : self.rt[i,j] if self.rt[i,j] is not None else "None",
                         'filter' : self.filter_name[i,j] if self.filter_name[i,j] is not None else "None",
                         }
                    v2 = self.filter_metadata.filter_metadata(self.filter_name[i,j])
                    # Make a copy so we can update v2 without changing anything it might
                    # point to in self.filter_metadata
                    v2 = copy.deepcopy(v2)
                    # Prefer our species_list if found, but otherwise use the
                    # one in self.filter_metadata
                    if(self.species_list[i,j] is None or self.species_list[i,j] == ""):
                        if("speciesList" in v2):
                            v["speciesList"] = v2["speciesList"]
                        else:
                            v["speciesList"] = ''
                    else:
                        v["speciesList"] = self.species_list[i,j]
                    # Values in both v and v2 prefer the v one based on the rules for
                    # update.
                    v2.update(v)
                    res.append(v2)
        return res

    @classmethod
    def filter_list_dict_from_file(cls, spec_fname : str) -> 'dict(str,list(str))':
        '''Return a dictionary going from instrument name to the list of filters for that
        given instrument.'''
        fspec = TesFile(spec_fname)
        res = {}
        for iname in list(dict.fromkeys(fspec.table["Instrument"].to_list())):
            res[iname] = list(dict.fromkeys(fspec.table[fspec.table["Instrument"] == iname]["Filter"].to_list()))
        return res

    @classmethod
    def create_dict_from_file(cls, spec_fname,
                              filter_metadata : 'Optional(FilterMetadata)' = None):
        '''Create a dict from instrument name to MusesSpectralWindow from the
        given microwindows file name. We also take an optional FilterMetadata which is used
        for additional metadata in the muses_microwindows function.
        '''
        res = {}
        for iname in cls.filter_list_dict_from_file(spec_fname).keys():
            res[iname] = cls.create_from_file(spec_fname, iname, filter_metadata)
        return res
    
    @classmethod
    def create_from_file(cls, spec_fname, instrument_name,
                         filter_metadata : 'Optional(FilterMetadata)' = None,
                         different_filter_different_sensor_index=True):
        '''Create a MusesSpectralWindow for the given instrument name from the given
        microwindow file name. We also take an optional FilterMetadata which is used
        for additional metadata in the muses_microwindows function.

        For some instruments we consider different filters as different sensor_index
        and for others we don't. The argument different_filter_different_sensor_index
        is used to control this.'''
        fspec = TesFile(spec_fname)
        rowlist = fspec.table[fspec.table["Instrument"] == instrument_name]
        
        flist = list(dict.fromkeys(rowlist["Filter"].to_list()))
        # I think it is ok to have this always True, but leave this knob in
        # place for now until we determine that this isn't needed.
        if(not different_filter_different_sensor_index):
            flist = [None,]
            nmw = [len(rowlist)]
        else:
            nmw = [len(rowlist[rowlist["Filter"] == flt]) for flt in flist]
        mw_range = np.zeros((len(flist), max(nmw), 2))
        filter_name = np.full((len(flist), max(nmw)), None, dtype=np.dtype(object))
        rt = np.full((len(flist), max(nmw)), None, dtype=np.dtype(object))
        species_list = np.full((len(flist), max(nmw)), None, dtype=np.dtype(object))
        for i,flt in enumerate(flist):
            if(flt is None):
                mwlist = rowlist
            else:
                mwlist = rowlist[rowlist["Filter"] == flt]
            for j, mw in enumerate(mwlist.iloc):
                mw_range[i,j,0] = mw['WindowStart']
                mw_range[i,j,1] = mw['WindowEnd']
                filter_name[i,j] = mw["Filter"]
                rt[i,j] = mw["RT"]
                species_list[i,j] = mw["Species"]
        mw_range = rf.ArrayWithUnit_double_3(mw_range, rf.Unit("nm"))
        return cls(spec_win=rf.SpectralWindowRange(mw_range), obs=None,
                   instrument_name=instrument_name,
                   filter_name = filter_name,
                   rt=rt,
                   species_list=species_list,
                   filter_metadata=filter_metadata)
    

    @classmethod
    def muses_microwindows_from_dict(cls,
                       spec_win_dict : 'dict(str, MusesSpectralWindow)') -> 'list(dict)':
        '''Create the muses-py microwindows list of dict structure from a dict going
        from instrument name to MusesSpectralWindow'''
        res = []
        for iname, swin in spec_win_dict.items():
            res.extend(swin.muses_microwindows())
        return res

    @classmethod
    def muses_microwindows_fname_from_muses_py(cls, viewing_mode : str,
                                              spectral_window_directory : str,
                                              retrieval_elements : 'list(str)',
                                              step_name : str, retrieval_type : str,
                                              spec_file = None):
        '''For testing purposes, this calls the old mpy.table_get_spectral_filename to
        determine the microwindow file name use. This can be used to verify that
        we are finding the right name. This shouldn't be used for real code,
        instead use the SpectralWindowHandleSet.'''
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
        
    @classmethod
    def muses_microwindows_from_muses_py(cls, default_spectral_window_fname : str,
                                         viewing_mode : str,
                                         spectral_window_directory : str,
                                         retrieval_elements : 'list(str)',
                                         step_name : str, retrieval_type : str,
                                         spec_file = None):
        '''For testing purposes, this calls the old mpy.table_new_mw_from_step. This can
        be used to verify that the microwindows we generate are correct. This shouldn't
        be used for real code, instead use the SpectralWindowHandleSet.'''
        # Wrap arguments into format expected by table_new_mw_from_step. This
        # creates a dummy strategy_table dict with the values it expects to find
        stable = { }
        stable["preferences"] = \
            {"defaultSpectralWindowsDefinitionFilename" : default_spectral_window_fname,
             "viewingMode" : viewing_mode,
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
        return mpy.table_new_mw_from_step(stable, 0)

__all__ = ["MusesSpectralWindow",]
           
    

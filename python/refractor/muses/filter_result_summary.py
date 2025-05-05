from __future__ import annotations
import refractor.muses.muses_py as mpy  # type: ignore
from .identifier import FilterIdentifier

class FilterResultSummary:
    '''Summarize the filter information we have in mpy_radiance. This
    is from mpy_radiance_from_observation_list, where all the radiance data
    is smushed together, and then we need to pull individual parts out. This
    is very much just something used for generating the output file - if you
    actually want the radiance for a single band of one instrument you can
    more directly get this. But we need this for generating data in our output
    files.'''
    def __init__(self, rstep : mpy.ObjectView) -> None:
        '''rstep should be the output of mpy_radiance_from_observation_list'''
        # Calculate the various summary pieces
        self._filter_index = [0,]
        self._filter_index.extend(i.filter_index for i in rstep.filterNames)
        self._filter_list = ['ALL',]
        self._filter_list.extend(i.spectral_name for i in rstep.filterNames)
        self._filter_start = [0]
        istart = 0
        for fs in rstep.filterSizes:
            self._filter_start.append(istart)
            istart += fs
        self._filter_end = [0]
        iend = -1
        for fs in rstep.filterSizes:
            iend += fs
            self._filter_end.append(iend)
        self._filter_end[0] = iend
        
        # The output depends on the filters being in a specific order. 
        # Get an array to indicate how we need to reorder things, and the
        # rorder the data
        sorder = FilterIdentifier.spectral_order([FilterIdentifier("ALL"), *rstep.filterNames])
        self._filter_index = [self._filter_index[i] for i in sorder]
        self._filter_list = [self._filter_list[i] for i in sorder]
        self._filter_start = [self._filter_start[i] for i in sorder]
        self._filter_end = [self._filter_end[i] for i in sorder]

    @property
    def filter_index(self) -> list[int]:
        return self._filter_index

    @property
    def filter_list(self) -> list[str]:
        return self._filter_list
    
    @property
    def filter_start(self) -> list[int]:
        return self._filter_start

    @property
    def filter_end(self) -> list[int]:
        return self._filter_end

    @property
    def filter_slice(self) -> list[slice]:
        '''Filter start and end presented as a slice, as a convenience'''
        return [slice(s,e+1) for (s,e) in zip(self.filter_start, self.filter_end)]
    

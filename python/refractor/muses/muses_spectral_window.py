import refractor.framework as rf
import copy
import numpy as np

class MusesSpectralWindow(rf.SpectralWindow):
    '''Muses-py retrieval just uses normal SpectralWindow (e.g., a SpectralWindowRange).
    However there are places where it wants 1) the data restricted to microwindows but
    including bad pixels (which are otherwise removed with the normal SpectralWindow) and
    2) the full data (referred to as full band).

    This class adds support for this. It wraps around and existing SpectralWindow and
    adds flags that can be set to "include_bad_data" or "full_band".
    '''
    def __init__(self, spec_win : "Optional(rf.SpectralWindowRange)",
                 obs : 'Optional(MusesObservation)',
                 raman_ext=3.01):
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
        '''
        super().__init__()
        self.include_bad_sample = False
        self.full_band = False
        self.do_raman_ext = False
        self._raman_ext = raman_ext
        # Let bad samples pass through
        self._spec_win_with_bad_sample = spec_win
        if(spec_win is not None):
            swin = copy.deepcopy(spec_win)
            for i in range(obs.num_channels):
                swin.bad_sample_mask(obs.bad_sample_mask(i), i)
            # Remove bad samples
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

__all__ = ["MusesSpectralWindow",]
           
    

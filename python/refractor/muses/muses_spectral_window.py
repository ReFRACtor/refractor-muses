import refractor.framework as rf
import copy

class MusesSpectralWindow(rf.SpectralWindow):
    '''Muses-py retrieval just uses normal SpectralWindow (e.g., a SpectralWindowRange).
    However there are places where it wants 1) the data restricted to microwindows but
    including bad pixels (which are otherwise removed with the normal SpectralWindow) and
    2) the full data (referred to as full band).

    This class adds support for this. It wraps around and existing SpectralWindow and
    adds flags that can be set to "include_bad_data" or "full_band".
    '''
    def __init__(self, spec_win : "Optional(rf.SpectralWindowRange)",
                 obs : 'Optional(MusesObservation)'):
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
        '''
        super().__init__()
        self.include_bad_sample = False
        self.full_band = False
        # Let bad samples pass through
        self._spec_win_with_bad_sample = spec_win
        if(spec_win is not None):
            swin = copy.deepcopy(spec_win)
            for i in range(obs.num_channels):
                swin.bad_sample_mask(obs.bad_sample_mask(i), i)
            # Remove bad samples
            self._spec_win = swin
        else:
            self._spec_win = None

    def _v_number_spectrometer(self):
        return self._spec_win.number_spectrometer

    def _v_spectral_bound(self):
        return self._spec_win.spectral_bound

    def desc(self):
        return "MusesSpectralWindow"

    def grid_indexes(self, grid, spec_index):
        if(self._spec_win is None or self.full_band):
            return list(range(grid.data.shape[0]))
        if(self.include_bad_sample):
            return self._spec_win_with_bad_sample.grid_indexes(grid, spec_index)
        return self._spec_win.grid_indexes(grid, spec_index)

__all__ = ["MusesSpectralWindow",]
           
    

import os
import h5py
import numpy as np
import refractor.framework as rf
import netCDF4 as ncdf

class MusesIlsInfo:
    '''This maintains information about the ILS method used by
    MUSES. This is complicated enough to have its own object.

    Note this handles a single instrument.

    This combines OMI and TROPOMI. It isn't clear if we 1) should
    actually move this into the OMI and TROPOMI directories or 2) Just
    handle this in RefractorFmObjectCreator without having a separate
    class.  But at least for now we'll organize this as this separate
    class. If nothing else, it should make testing this easier if it
    is handled as stand alone.

    Note this doesn't actually handle the ILS convolution, this just
    maintains the information needed to set up the ILS function.

    The ILS method should currently be one of the following:

    POSTCONV - A normal ILS, where we apply the ILS after creating a high
               resolution monchromatic grid (e.g., rf.IlsTableLinear)
    NOAPPLY -  Synonym for POSTCONV.
    APPLY -    Oddly enough, this doesn't mean "apply ILS". Rather this
               skips doing any kind of ILS at the Instrument level. Instead,
               the ILS is preconvolved with the absorbers (e.g., O3).
               See MusesOpticalDepthFile.
    FASTCONV - This uses Matt's quick approximation to applying the ILS. See
               rf.IlsFastApply

    '''
    def __init__(self, ils_method, instrument_name, 
                 atrack_index=None, xtrack_index=None,
                 obs_type="NORMAL",
                 osp_dir=None, omi_ils_dir=None, tropomi_ils_dir=None):
        self._instrument_name = instrument_name
        self._ils_method = ils_method
        self.atrack_index = atrack_index
        self.xtrack_index = xtrack_index
        self.obs_type = obs_type
        # NOAPPLY is alias of POSTCONV
        if(self._ils_method == "NOAPPLY"):
            self._ils_method = "POSTCONV"
        if(self._ils_method not in ('FASTCONV', 'POSTCONV', 'APPLY')):
            raise RuntimeError(f"Unsupported ILS method {self._ils_method}, we currently support 'FASTCONV', 'POSTCONV', and 'APPLY'")
        # If we don't have a osp_dir passed in, assume "../OSP"
        if(osp_dir is None):
            self.osp_dir = os.path.abspath("../OSP")
        else:
            self.osp_dir = os.path.abspath(osp_dir)
        # The OMI ILS directory is normally OMI/OMI_ILS, but allow
        # this to be overriden.
        if(omi_ils_dir is None):
            self.omi_ils_dir = os.path.abspath(f"{self.osp_dir}/OMI/OMI_ILS")
        else:
            self.omi_ils_dir = os.path.abspath(omi_ils_dir)
        # Similar for TROPOMI ILS directory
        if(tropomi_ils_dir is None):
            self.tropomi_ils_dir = os.path.abspath(f"{self.osp_dir}//TROPOMI/isrf_release/isrf")
        else:
            self.tropomi_ils_dir = os.path.abspath(tropomi_ils_dir)

    @property
    def ils_method(self):
        return self._ils_method

    @property
    def instrument_name(self):
        return self._instrument_name
    
    def omi_ils_postconv(self, channel_name):
        '''channel_name should be one of "UV1", "UV2" or
        "VIS".
        '''
        ils_filename = f"{self.omi_ils_dir}/{self.obs_type}/{channel_name}/OMI_ILS_{self.obs_type}_{channel_name}_{int(self.atrack_index):02d}.h5"
        if(not os.path.exists(ils_filename)):
            raise RuntimeError(f"File {ils_filename} is not found.")
        f = h5py.File(ils_filename, "r")
        wavenumber = []
        delta_lambda = []
        response = []

        freq_wl = f['FREQ_MONO'][:]
        center_wl = f['XCF0'][:]

        center_wn = np.flip(rf.ArrayWithUnit(center_wl, "nm").convert_wave("cm^-1").value)
        freq_wn = np.flip(rf.ArrayWithUnit(freq_wl, "nm").convert_wave("cm^-1").value).transpose()

        response = np.flip(f['ILS_MONO'][:]).transpose()

        delta_wn = np.zeros(freq_wn.shape)
        for center_idx in range(center_wn.shape[0]):
            delta_wn[center_idx, :] = freq_wn[center_idx, :] - center_wn[center_idx]

        interp_wavenumber = True
        return rf.IlsTableLinear(center_wn, delta_wn, response, channel_name,
                                 channel_name, interp_wavenumber)

    def omi_ils_quickconv(self, channel_name):
        # Should probably have somebody else fill this in, the code is
        # a bit complicated in py-retrieve and we should have this in
        # place with some kind of a test.
        raise NotImplementedError("We don't have this implemented yet")

    def tropomi_ils_quickconv(self, band_index):
        # Should probably have somebody else fill this in, the code is
        # a bit complicated in py-retrieve and we should have this in
        # place with some kind of a test.
        raise NotImplementedError("We don't have this implemented yet")
    
    def tropomi_ils_postconv(self, band_index):
        '''band_index should be 1 to 8.'''
        # Not positive about this. The notebook
        # https://github.jpl.nasa.gov/MUSES-Processing/refractor_example_notebooks/blob/main/tropomi_nir_with_ils_retieval.ipynb
        # uses different files and a bit of a different handling than
        # the py-retrieve code. I'm not sure how tested any of the non
        # SWIR stuff is, so I'll go ahead and follow the notebook rather
        # than py-retrieve
        # Josh: You should perhaps take a look at this and modify as
        # needed.
           
        if(band_index < 1 or band_index > 8):
            raise RuntimeError("band_index must be in the range 1 to 8")
        band_name = f'BAND{band_index}'

        # The SWIR data and bands 1-6 are different format. Not sure why,
        # I think these might be different versions.
        if(band_index < 7):
            fname = f"{self.tropomi_ils_dir}/raw_uvn/isrf.band{band_index}.ckd.nc" 
            fname2 = f"{self.tropomi_ils_dir}/raw_uvn/wavelength.band{band_index}.ckd.nc" 
            with ncdf.Dataset(fname) as ds:
                band_group = ds.groups[band_name]
                response = band_group['isrf'][self.xtrack_index]['value']
                delta_wl = band_group['isrf_wavelength_grid'][:].filled(np.nan)
            with ncdf.Dataset(fname2) as ds:
                band_group = ds.groups[band_name]
                central_wavelength = band_group['wavelength_map'][self.xtrack_index]['value']
        else:
            fname = f"{self.tropomi_ils_dir}/raw_swir/ckd.swir_isrf_v20160331.detector4.nc"
            with ncdf.Dataset(fname) as ds:
                band_group = ds.groups[band_name]
                delta_wl = band_group['delta_wavelength'][:].filled(np.nan)
                central_wavelength = band_group['central_wavelength'][:].filled(np.nan)
                response = band_group['isrf'][self.xtrack_index].filled(np.nan)
        
        
        # Calculate the wavelength grid first - delta's in wavelength
        # don't translate to wavenumber deltas
        response_wavelength = central_wavelength.reshape(-1,1) + delta_wl.reshape(1,-1)
    
        # Convert to frequency-ordered wavenumber arrays. The 2D
        # arrays need flipped on both axes since the order reverses in
        # both

        center_wn = np.flip(rf.ArrayWithUnit(central_wavelength, 'nm').convert_wave('cm^-1').value)
        response_wn = np.flip(np.flip(rf.ArrayWithUnit(response_wavelength, 'nm').convert_wave('cm^-1').value, axis=1), axis=0)
        response = np.flip(np.flip(response, axis=1), axis=0)
    
        # Calculate the deltas in wavenumber space
        delta_wn = response_wn - center_wn.reshape(-1,1)
    
        # Build a table of ILSs at the sampled wavelengths/frequencies
        interp_wavenumber = True
        return rf.IlsTableLinear(center_wn, delta_wn, response, band_name,
                                 band_name, interp_wavenumber)
            

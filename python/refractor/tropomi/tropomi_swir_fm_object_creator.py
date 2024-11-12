from functools import cached_property, lru_cache
from refractor.muses import (RefractorFmObjectCreator,
                             RefractorUip, 
                             ForwardModelHandle,
                             MusesRaman, MusesSpectrumSampling,
                             CurrentState, CurrentStateUip,
                             SurfaceAlbedo)
from refractor.muses import muses_py as mpy
import refractor.framework as rf
from .tropomi_fm_object_creator import TropomiFmObjectCreator
from loguru import logger
import numpy as np
import re
import glob
import copy
import os
from netCDF4 import Dataset

class TropomiSwirFmObjectCreator(TropomiFmObjectCreator):
    '''This is the variation for handling the SWIR channels. Note that
    this might get merged in with TropomiFmObjectCreator with some logic for
    picking the bands, but for now leave this separate.

    Also, at this point we aren't overly worried about having a fully integrated
    set of OSP control files. We hard code stuff in this class just to get everything
    working. Once we figure out *what* we want to do, we can then worry about fully
    integrating that in with the rest of the system.
    '''
    def __init__(self, current_state : 'CurrentState',
                 measurement_id : 'MeasurementId',
                 observation : 'MusesObservation',
                 absorption_gases = ['H2O', 'CO', 'CH4', 'HDO'],
                 primary_absorber = "CO",
                 use_raman=False,
                 **kwargs):
        super().__init__(current_state, measurement_id, observation,
                         absorption_gases=absorption_gases, primary_absorber=primary_absorber,
                         use_raman=use_raman,
                         **kwargs)
        
        # JLL: I always get an HDF error if I try to read the ABSCO netCDF file in the
        # spectrum_sampling method. I'm guessing something else is accessing it at that
        # point, in a way that confounds Python's netCDF4. To get around that for now,
        # I'll just read in the absco grid here.
        primary_absco_file = self.absco_filename(self.primary_absorber)
        with Dataset(primary_absco_file) as ds:
            absco_grid = ds['Spectral_Grid'][:].filled(np.nan)
            absco_grid_units = ds['Spectral_Grid'].units
        # Special case, since ReFRACtor expects a ^ in cm^-1 and the ABSCO files
        # just use cm-1...
        absco_grid_units = 'cm^-1' if absco_grid_units == 'cm-1' else absco_grid_units
        self.full_absco_grid = rf.ArrayWithUnit(absco_grid, absco_grid_units)

    @cached_property
    def absorber(self):
        '''Absorber to use. This just gives us a simple place to switch
        between absco and cross section.'''
        return self.absorber_absco

    def absco_filename(self, gas, version='latest'):
        # allow one to pass in "latest" or a version number like either "1.0" or "v1.0"
        if version == 'latest':
            vpat = 'v*'
        elif version.startswith('v'):
            vpat = version
        else:
            vpat = f'v{version}'

        # Assumes that in the top level of the ABSCO directory there are
        # subdirectories such as "v1.0_SWIR_CO" which contain our ABSCO files.
        absco_subdir_pattern = f'{vpat}_SWIR_{gas.upper()}'
        full_pattern = f"{self.absco_base_path}/{absco_subdir_pattern}"
        absco_subdirs = sorted(glob.glob(full_pattern))
        if version == 'latest' and len(absco_subdirs) == 0:
            raise RuntimeError(f'Found no ABSCO directories for gas "{gas}" matching {full_pattern}')
        elif version == 'latest':
            # Assumes that the latest version will be the last after sorting (e.g. v1.1
            # > v1.0). Should technically use a semantic version parser to ensure e.g.
            # v1.0.1 would be selected over v1.0.
            gas_subdir = absco_subdirs[-1]
            logger.info(f'Using ABSCO files from {gas_subdir} for {gas}')
        elif len(absco_subdirs) == 1:
            gas_subdir = absco_subdirs[0]
        else:
            raise RuntimeError(f'{len(absco_subdirs)} were found for {gas} {version} in {self.absco_base_path}')

        gas_pattern = f"{gas_subdir}/nc_ABSCO/{gas.upper()}_*_v0.0_init.nc"
        return self.find_absco_pattern(gas_pattern, join_to_absco_base_path=False)
    
    @cached_property
    def spectrum_sampling(self):
        hres_spec = []
        for i in range(self.num_channels):
            if self.filter_list[i] == 'BAND7':
                absco_grid = self.full_absco_grid.convert_wave('nm').value
                absco_grid_units = self.full_absco_grid.units.name

                # We should only need monochromatic wavelengths within the microwindow(s)
                # being used. To be safe, we'll go 3x the ILS width outside the window,
                # that should be plenty to make sure the ILS has monochromatic lines over
                # its whole span.

                mw_bounds = self.rf_uip.micro_windows(i).convert_wave('nm').value
                # If there are multiple sub windows, this will just keep the monochromatic
                # wavelengths in between the sub windows. We can optimize those out later
                # if need be.
                mw_start = np.min(mw_bounds)
                mw_end = np.max(mw_bounds)
                ils_width = np.abs(np.max(self.ils_params(i)['delta_wavelength']))
                to_keep = (absco_grid >= (mw_start - 3*ils_width)) & (absco_grid <= (mw_end + 3*ils_width))
                hres_spec.append(rf.SpectralDomain(absco_grid[to_keep], rf.Unit('nm')))
            else:
                # Not sure how this will work for multi-band retrievals yet...
                hres_spec.append(None)

        return MusesSpectrumSampling(hres_spec)
        

class TropomiSwirForwardModelHandle(ForwardModelHandle):
    def __init__(self, **creator_kwargs):
        self.creator_kwargs = creator_kwargs
        self.measurement_id = None
        
    def notify_update_target(self, measurement_id : 'MeasurementId'):
        '''Clear any caching associated with assuming the target being retrieved is fixed'''
        self.measurement_id = measurement_id
        
    def forward_model(self, instrument_name : str,
                      current_state : 'CurrentState',
                      obs : 'MusesObservation',
                      fm_sv: rf.StateVector,
                      rf_uip_func,
                      **kwargs):
        if(instrument_name != "TROPOMI"):
            return None
        obj_creator = TropomiSwirFmObjectCreator(current_state,
                                                 self.measurement_id, obs,
                                                 rf_uip=rf_uip_func(),
                                                 fm_sv=fm_sv,
                                                 **self.creator_kwargs)
        fm = obj_creator.forward_model
        logger.info(f"Tropomi SWIR Forward model\n{fm}")
        return fm
    
__all__ = ["TropomiSwirFmObjectCreator", "TropomiSwirForwardModelHandle"]
    

import refractor.muses.muses_py as mpy
from .retrieval_strategy_step import (RetrievalStrategyStep,
                                      RetrievalStrategyStepSet)
from loguru import logger
import copy
import os
import numpy as np

class RetrievalStrategyStepIRK(RetrievalStrategyStep):
    '''IRK strategy step.'''
    def retrieval_step(self, retrieval_type : str,
                       rs : 'RetrievalStrategy') -> (bool, None):
        if retrieval_type != "irk":
            return (False,  None)
        logger.debug(f"Call to {self.__class__.__name__}::retrieval_step")
        o_xxx = {"AIRS" : None, "TES" : None, "CRIS" : None, "OMI" : None,
                 "TROPOMI" : None, "OCO2" : None}
        cstep = rs.current_strategy_step
        for iname in cstep.instrument_name:
            if iname in o_xxx:
                obs = rs.observation_handle_set.observation(
                    iname, None, cstep.spectral_window_dict[iname],None)
                if hasattr(obs, "muses_py_dict"):
                    o_xxx[iname] = obs.muses_py_dict
        logger.info("Running run_irk ...")
        self.cfunc = rs.create_cost_function()
        self.results_irk = self.irk(
            rs._strategy_executor.strategy._stable.strategy_table_dict,
            rs.state_info.state_info_dict,
            rs._strategy_executor.strategy._stable.microwindows(),
            rs.retrieval_info.retrieval_info_obj,
            rs.retrieval_info.species_names, 
            rs.retrieval_info.species_list_fm, 
            self.radiance_step(),
            airs=o_xxx["AIRS"], tes=o_xxx["TES"], cris=o_xxx["CRIS"],
            omi=o_xxx["OMI"],
            oco2=o_xxx["OCO2"])
        rs.notify_update("IRK step", retrieval_strategy_step=self)
        return (True, None)

    def irk(self,
            i_table,
            stateIn,
            windows,
            retrievalInfo,
            jacobian_speciesIn,
            jacobian_specieslistIn,
            radianceStep,
            airs=None,
            cris=None,
            tes=None,
            omi=None,
    	    oco2=None):
        '''This was originally the run_irk.py code from py-retrieve. We
        have our own copy of this so we can clean this code up a bit.
    
        This is currently only used by RetrievalStrategyStepIRK. We may move this
        function into that class, but for now go ahead and keep this separate because
        if its size'''
        if isinstance(i_table, mpy.ObjectView):
            i_table = i_table.__dict__
            
        # Make a copy of stateIn so we won't affect it.
        stateInfo = copy.deepcopy(stateIn)  
    
        # in degree
        TES_angles = [0.0, 14.5752, 32.5555, 48.1689, 59.0983, 63.6765] 
        CRIS_angles = [0.0, 14.2906, 31.8588, 46.9590, 57.3154, 61.5613]
        AIRS_angles = [0.0, 14.5752, 32.5555, 48.1689, 59.0983, 63.6765]
        
        xi_list = [0.0,] * len(TES_angles)

        # If we are using AIRS, then replace with the TES forward model
        if windows[0]['instrument'] == 'AIRS':
            tes = self.replace_airs_with_tes(i_table, windows, stateInfo)
        
        for iangle in range(len(TES_angles)):
            stateInfo['current']['cris']['scanAng'] = CRIS_angles[iangle]
            stateInfo['current']['tes']['boresightNadirRadians'] = TES_angles[iangle]
            stateInfo['current']['airs']['scanAng'] = AIRS_angles[iangle]
    
    
            (uip, radianceOut, jacobianOut) = mpy.run_forward_model(
                i_table, stateInfo, windows, retrievalInfo,
                jacobian_speciesIn, jacobian_specieslistIn, 
                radianceStep,
                airs, cris, tes, omi, None, oco2,
                None)
    
            results = radianceOut   # Do this assignment so it matches the IDL code where only 1 variable is returned.
    
    
            # uip is a dict.
            if 'CRIS' in uip['instruments']:
                gi_angle = CRIS_angles[iangle]
                uip_all = {**uip,**uip['uip_CRIS']}
    
            # AT_LINE 129 src_ms-2018-12-10/run_irk.pro
            if 'TES' in uip['instruments']:
                gi_angle = TES_angles[iangle]
                # uip_all = struct_combine(uip, uip['tesPars'])
                uip_all = {**uip,**uip['uip_TES']}
    
            # AT_LINE 62 run_irk.pro
            # AT_LINE 133 src_ms-2018-12-10/run_irk.pro
            if 'AIRS' in uip['instruments']:
                gi_angle = AIRS_angles[iangle]
                uip_all = mpy.struct_combine(uip, uip['uip_AIRS'])
    
            # AT_LINE 141 src_ms-2018-12-10/run_irk.pro
            if 'OMI' in uip['instruments']:
                raise RuntimeError("Not implemented yet")
    
            # AT_LINE 84 run_irk.pro
            # compute nadir_column and slant_column
            atmparams = mpy.atmosphere_level(uip_all)
            rayInfo = mpy.raylayer_nadir(mpy.ObjectView(uip_all), mpy.ObjectView(atmparams))
            rayInfo = mpy.ObjectView(rayInfo)
    
            # AT_LINE 89 run_irk.pro
            if gi_angle == 0.0: 
                nadir_column = np.sum(rayInfo.column_air)
                
                uip_all1 = copy.deepcopy(uip_all)
                uip_all1['cloud']['extinction'][:] = 1.
                
                atmparams1 = mpy.atmosphere_level(uip_all1)
                ray1Info = mpy.raylayer_nadir(mpy.ObjectView(uip_all1), mpy.ObjectView(atmparams1))
                
                dEdOD = 1. / ray1Info['cloud']['tau_total']
            else:
                # AT_LINE 101 run_irk.pro
                slant_column = np.sum(rayInfo.column_air)
                xi_list[iangle] = nadir_column / slant_column
    
            # AT_LINE 113 run_irk.pro
            # Slow method for now.  Not sure why these are saved.
            if (iangle) == 0:
                radiance0 = copy.deepcopy(results['radiance'])
                jacobian0 = copy.deepcopy(jacobianOut['jacobian_data']) # Note that we have chosen 'jacobian_data' as the key in jacobian dict
     
            #outfile = "lkuai/tmp.txt"
            #ifreq = 0
    
            #np.savetxt(outfile, jacobian0[0, 0:64, ifreq])
           
    
    
            if (iangle) == 1:
                radiance1 = copy.deepcopy(results['radiance'])
                jacobian1 = copy.deepcopy(jacobianOut['jacobian_data']) # Note that we have chosen 'jacobian_data' as the key in jacobian dict
            
            if (iangle) == 2:
                radiance2 = copy.deepcopy(results['radiance'])
                jacobian2 = copy.deepcopy(jacobianOut['jacobian_data']) # Note that we have chosen 'jacobian_data' as the key in jacobian dict
            
            if (iangle) == 3:
                radiance3 = copy.deepcopy(results['radiance'])
                jacobian3 = copy.deepcopy(jacobianOut['jacobian_data']) # Note that we have chosen 'jacobian_data' as the key in jacobian dict
            
            if (iangle) == 4:
                radiance4 = copy.deepcopy(results['radiance'])
                jacobian4 = copy.deepcopy(jacobianOut['jacobian_data']) # Note that we have chosen 'jacobian_data' as the key in jacobian dict
            
            if (iangle) == 5:
                radiance5 = copy.deepcopy(results['radiance'])
                jacobian5 = copy.deepcopy(jacobianOut['jacobian_data']) # Note that we have chosen 'jacobian_data' as the key in jacobian dict
        # end for iangle in range(lenTES_angles):
    
        # AT_LINE 124 run_irk.pro
        freq_step = results['frequency'][1:] - results['frequency'][0:len(results['frequency'])-1]
        
        # missing one value
        temp_freq = copy.deepcopy(freq_step) # Make a copy of freq_step because we will create a new memory for it.
    
        freq_step = [temp_freq[0]]  # Fetch the first element from temp_freq.
        freq_step.extend(temp_freq) # Add all elements in temp_freq to the newly created freq_step.
        freq_step = np.asarray(freq_step)
    
        # AT_LINE 207 src_ms-2018-12-10/run_irk.pro
        frqL1b = radianceStep['frequency']
        radL1b = radianceStep['radiance']
        nL1b = len(frqL1b)
        
        # need remove missing data in L1b radiance
        ifrq_missing = np.where(radL1b == 0.0)
        valid_indices = np.where(radL1b != 0.0)[0]  # Ensure 1-D array
        interpolated_values = np.interp(ifrq_missing, valid_indices, radL1b[valid_indices])
        radL1b[ifrq_missing]= interpolated_values
    
    
        # Remember that in Python, the slice does not include the end.  
        # IDL_ CODE: (frqL1b[2:*] - frqL1b[0:nL1b-3])/2.
        freq_stepL1b_temp = (frqL1b[2:] - frqL1b[0:nL1b-2]) / 2.  
        freq_stepL1b = np.concatenate((np.asarray([frqL1b[1] - frqL1b[0]]), freq_stepL1b_temp, np.asarray([frqL1b[nL1b-1] - frqL1b[nL1b-2]])), axis=0)
    
        # Sometimes, the shape of the array is (1,zzz), we change it to 1-D (zzz)
        if radiance0.shape[0] == 1:
            radiance0 = np.reshape(radiance0, (radiance0.shape[1]))
            radiance1 = np.reshape(radiance1, (radiance1.shape[1]))
            radiance2 = np.reshape(radiance2, (radiance2.shape[1]))
            radiance3 = np.reshape(radiance3, (radiance3.shape[1]))
            radiance4 = np.reshape(radiance4, (radiance4.shape[1]))
            radiance5 = np.reshape(radiance5, (radiance5.shape[1]))
    
        if jacobian0.shape[0] == 1 or jacobian0.shape[1] == 1:
            # Sometimes, the shape of the array is (1,yyy,zzz), we change it to 2-D (yyy,zzz)
            if jacobian0.shape[0] == 1:
                jacobian0 = jacobian0[0, :, :]
                jacobian1 = jacobian1[0, :, :]
                jacobian2 = jacobian2[0, :, :]
                jacobian3 = jacobian3[0, :, :]
                jacobian4 = jacobian4[0, :, :]
                jacobian5 = jacobian5[0, :, :]
    
            # Sometimes, the shape of the array is (yyy,1,zzz), we change it to 2-D (yyy,zzz)
            if jacobian0.shape[1] == 1:
                jacobian0 = jacobian0[:, 0, :]
                jacobian1 = jacobian1[:, 0, :]
                jacobian2 = jacobian2[:, 0, :]
                jacobian3 = jacobian3[:, 0, :]
                jacobian4 = jacobian4[:, 0, :]
                jacobian5 = jacobian5[:, 0, :]
    
        # AT_LINE 132 run_irk.pro
        radianceWeighted = np.float64(2.0) * (np.float64(0.015748) * radiance5 + \
                                              np.float64(0.073909) * radiance4 + \
                                              np.float64(0.146387) * radiance3 + \
                                              np.float64(0.167175) * radiance2 + \
                                              np.float64(0.096782) * radiance1)
    
        # AT_LINE 138 run_irk.pro
        # AT_LINE 225 src_ms-2018-12-10/run_irk.pro
        radratio = radiance0 / radianceWeighted
        frequency = results['frequency'] # fm frequence
    
        # AT_LINE 140 run_irk.pro
        # compute band flux (980:1080)
    
        # AT_LINE 228 src_ms-2018-12-10/run_irk.pro
        ifrq = self._find_bin(frequency, frqL1b)
        radratio = radratio[ifrq] # same dimension as L1b frq and rad
    
        # compute FM band flux (980:1080)
        # AT_LINE 236 src_ms-2018-12-10/run_irk.pro
        ifreq = np.where((frequency >= 980.) & (frequency <= 1080.))[0]
    
        # AT_LINE 237 src_ms-2018-12-10/run_irk.pro
        flux = 1e4 * np.pi * np.sum(freq_step[ifreq] * radianceWeighted[ifreq])
    
        # AT_LINE 240 src_ms-2018-12-10/run_irk.pro
        # compute L1b band flux (980:1080)
        ifreqL1b = np.where((frqL1b >= 980.) & (frqL1b <= 1080.))[0]
        flux_l1b = 1e4 * np.pi * np.sum(freq_stepL1b[ifreqL1b] * radL1b[ifreqL1b]/radratio[ifreqL1b])
    
        # AT_LINE 152 run_irk.pro
        minn = np.amin(frequency)
        maxx = np.amax(frequency)
        minn = 970.
        maxx = 1120.
    
        # AT_LINE 159 run_irk.pro
        nf = int((maxx-minn)/3)
        freqSegments = np.ndarray(shape=(nf), dtype=np.float32)
        freqSegments.fill(0) # It is import to start with 0 because not all elements will be calculated.
        
        fluxSegments = np.ndarray(shape=(nf), dtype=np.float32)
        fluxSegments.fill(0) # It is import to start with 0 because not all elements will be calculated.
        
        fluxSegments_l1b = np.ndarray(shape=(nf), dtype=np.float32)
        fluxSegments_l1b.fill(0) # It is import to start with 0 because not all elements will be calculated.
        
        # get split into 3 cm-1 segments
        for ii in range(nf):
            ind = np.where((frequency >= minn+ii*3) & (frequency < minn+ii*3+3))[0]
            # AT_LINE 262 src_ms-2018-12-10/run_irk.pro
            indL1b = np.where((frqL1b >= minn+ii*3) & ((frqL1b < minn+ii*3+3) & (radL1b > 0.)))[0]
    
            if len(indL1b) > 0:
                fluxSegments_l1b[ii] = 1e4*np.pi*np.sum(freq_stepL1b[indL1b]*radL1b[indL1b]/radratio[indL1b])
    
            if len(ind) > 0: # We only calculate fluxSegments, fluxSegments_l1b, and freqSegments if there is at least 1 value in ind vector.
                fluxSegments[ii] = 1e4 * np.pi * np.sum(freq_step[ind]*radianceWeighted[ind])
                freqSegments[ii] = np.mean(frequency[ind])
        # end for ii in range(nf):
    
        # AT_LINE 183 run_irk.pro
        jacWeighted = np.float64(2.) * (np.float64(0.015748) * jacobian5 + \
                                        np.float64(0.073909) * jacobian4 + \
                                        np.float64(0.146387) * jacobian3 + \
                                        np.float64(0.167175) * jacobian2 + \
                                        np.float64(0.096782) * jacobian1)
    
        # weight by freq_step
        nn = retrievalInfo.n_totalParametersFM
        for jj in range(nn):
            jacWeighted[jj, :] = jacWeighted[jj, :] * freq_step[:]
    
        # AT_LINE 195 run_irk.pro
        o_results_irk = {
            'flux':flux,
            'flux_l1b': flux_l1b,                 
            'fluxSegments': fluxSegments,         
            'freqSegments': freqSegments,         
            'fluxSegments_l1b': fluxSegments_l1b 
        } 
    
        # AT_LINE 197 run_irk.pro
        # smaller range for irk average 
        indf = np.where((frequency >= 979.99) & (frequency <= 1078.999))[0]
    
        irk_array = 1e4 * np.pi * mpy.my_total(jacWeighted[:,indf], 1)
    
        minn = 980.0
        maxx = 1080.0
    
        nf = int((maxx-minn)/3)
        irk_segs = np.zeros(shape=(nn, nf),dtype=np.float32)
        freq_segs = np.zeros(shape=(nf),   dtype=np.float32)
    
        for ii in range(nf):
            ind = np.where((frequency >= minn+ii*3) & (frequency < minn+ii*3+3))[0]
            if len(ind) > 1:  # We only calculate irk_segs and freq_segs if there are more than 1 values in ind vector.
                irk_segs[:, ii] = 1e4 * np.pi * mpy.my_total(jacWeighted[:, ind], 1)
                freq_segs[ii] = np.mean(frequency[ind])
        # end for ii in range(nf):
    
        # AT_LINE 333 src_ms-2018-12-10/run_irk.pro
        radarr_fm = np.concatenate((radiance0, radiance1, radiance2, radiance3, radiance4, radiance5), axis=0)
        radInfo = {
            'gi_angle'  : gi_angle,  
            'radarr_fm' : radarr_fm, 
            'freq_fm'   : frequency, 
            'rad_L1b'   : radL1b,    
            'freq_L1b'  : frqL1b
        }
        o_results_irk['freqSegments_irk'] = freq_segs
        o_results_irk['radiances'] = radInfo
    
        # calculate irk for each type
        for ispecies in range(len(jacobian_speciesIn)):
            species_name = retrievalInfo.species[ispecies]
            ii = retrievalInfo.parameterStartFM[ispecies]
            jj = retrievalInfo.parameterEndFM[ispecies]
            vmr = retrievalInfo.initialGuessListFM[ii: jj+1]
            if retrievalInfo.mapType[ispecies] == 'log':
                vmr = np.exp(vmr)
            pressure = retrievalInfo.pressureListFM[ii: jj+1]
    
            myirfk = copy.deepcopy(irk_array[ii:jj+1]);
            myirfk_segs = copy.deepcopy(irk_segs [ii:jj+1, :])
    
            # convert cloudext to cloudod
            # dL/dod = dL/dext * dext/dod
            if species_name == 'CLOUDEXT':
                myirfk = np.multiply(myirfk, dEdOD)
                for pp in range(dEdOD.shape[0]):
                    myirfk_segs[pp, :] = myirfk_segs[pp, :] * dEdOD[pp]
    
                species_name = 'CLOUDOD'  
                vmr = np.divide(vmr, dEdOD)
    
            mm = (jj-ii+1)
            if species_name == 'TATM' or species_name == 'TSUR':
                mylirfk = np.multiply(myirfk,vmr)
                mylirfk_segs = copy.deepcopy(myirfk_segs)
                for kk in range(mm):
                    mylirfk_segs[kk, :] = mylirfk_segs[kk, :] * vmr[kk]
            else:
                mylirfk = copy.deepcopy(myirfk)
                myirfk = np.divide(myirfk, vmr)
                mylirfk_segs = copy.deepcopy(myirfk_segs)
                # myirfk_segs  = myirfk_segs;  This line doesn't do anything.  Is it a bug?
                for kk in range(mm):
                    myirfk_segs[kk, :] = myirfk_segs[kk, :] / vmr[kk]
    
            mult_factor = 1.0
            unit = ' '  # Set to one space just in case nobody below sets it.
            if species_name == 'O3':
                mult_factor = 1.0/1e9 # W/m2/ppb
                unit = 'W/m2/ppb'
            elif species_name == 'O3':
                mult_factor = 1.0
                unit = 'W/m2/ppb'
            elif species_name == 'H2O':
                mult_factor = 1.0/1e6  # W/m2/ppm
                unit = 'W/m2/ppm'
            elif species_name == 'H2O':
                mult_factor = 1.0
                unit = 'W/m2/ppm'
            elif species_name == 'TATM':
                mult_factor = 1.0
                unit = 'W/m2/K'
            elif species_name == 'TSUR':
                mult_factor = 1.0
                unit = 'W/m2/K'
            elif species_name == 'EMIS':
                mult_factor = 1.0
                unit = 'W/m2'
            elif species_name == 'CLOUDDOD':
                mult_factor = 1.0
                unit = 'W/m2'
            elif species_name == 'PCLOUD':
                mult_factor = 1.0
                unit = 'W/m2/hPa'
            else:
                # Fall back
                mult_factor = 1.0
                unit = ' '
                
            myirfk = np.multiply(myirfk, mult_factor)
            myirfk_segs = np.multiply(myirfk_segs, mult_factor)
    
            # subset only freqs in range
            if species_name == 'CLOUDOD':
                myirfk_segs = myirfk_segs[:, 0]
                myirfk_segs = np.reshape(myirfk_segs, (myirfk_segs.shape[0]))
                    
                mylirfk_segs = mylirfk_segs[:,0]
                mylirfk_segs = np.reshape(mylirfk_segs, (mylirfk_segs.shape[0]))
    
            vmr = np.divide(vmr, mult_factor)
    
            # Build a structure of result for each species_name.
            result_per_species = {
                'irfk'      : myirfk,       
                'lirfk'     : mylirfk,      
                'pressure'  : pressure,     
                'unit'      : unit,         
                'irfk_segs' : myirfk_segs,  
                'lirfk_segs': mylirfk_segs, 
                'vmr'       : vmr
            }
    
            # Add the result for each species_name to our structure to return.
            # Note that the name of the species is the key for the dictionary structure.
    
            o_results_irk[species_name] = copy.deepcopy(result_per_species)  # o_results_irk
        # end for ispecies in range(len(jacobian_speciesIn)):
    
        return o_results_irk

    def replace_airs_with_tes(self, i_table, windows, stateInfo):
        '''For AIRS, we use TES forward model as a replacement. This creates
        the structure needed by the UIP for this.'''
        my_dir = mpy.table_get_pref(i_table, 'spectralWindowDirectory')
        cdf_file_name = my_dir + '/../../tes_frequency.nc'
    
        logger.debug(f"cdf_file_name {cdf_file_name}")
        my_file = mpy.cdf_read_tes_frequency(cdf_file_name)
    
        my_file = self._make_case_right(my_file)
        my_file = self._transpose_2d_arrays(my_file)
    
        # Convert filterNames from array of bytes (ASCII)  to array of strings since the NetCDF file store the strings as bytes.
        filterNamesAsStrings = []
        for ii in range(0, len(my_file['filterNames'])):
            filterNamesAsStrings.append(''.join(chr(i) for i in my_file['filterNames'][ii]))
        my_file['filterNames'] = filterNamesAsStrings
             # Convert instrumentNames from bytes (ASCII) to string.
        instrumentNamesAsStrings = []
        instrumentNamesAsStrings.append(''.join(chr(i) for i in my_file['instrumentNames']))
        my_file['instrumentNames'] = instrumentNamesAsStrings
             # Remove first element from 'instrumentSizes' if it is 0
        if my_file['instrumentSizes'][0] == 0:
            my_file['instrumentSizes'] = np.delete(my_file['instrumentSizes'], 0)
             # Change 'instrumentSizes' from ndarray to list since some later function expects a list.
        if isinstance(my_file['instrumentSizes'],np.ndarray):
            my_file['instrumentSizes'] = my_file['instrumentSizes'].tolist() 
             # Change 'instrumentNames' from ndarray to list since some later function expects a list.
        if isinstance(my_file['instrumentNames'], np.ndarray):
            my_file['instrumentNames'] = my_file['instrumentNames'].tolist() 
             # Change 'numDetectorsOrig' from ndarray to scalar.
        if isinstance(my_file['numDetectorsOrig'], np.ndarray) or isinstance(my_file['numDetectorsOrig'], list):
            my_file['numDetectorsOrig'] = my_file['numDetectorsOrig'][0]
             # Remove first element from 'filterSizes' if it is 0 and perform calculation of actual filter sizes.
        if my_file['filterSizes'][0] == 0:
            # Because the way the file specifies the my_file['filterSizes'] as [ 0 4451  8402 12553 18554]
            # We have to subtract each number from the previous to get the exact numbers of the frequencies for each filter
            # to get [4451 3951 4151 6001]
            actual_filter_sizes = []
            for ii in range(1, len(my_file['filterSizes'])):
                # Start with the 2nd index, subtract the 2nd index from the first to get the actual filter size for each filter.
                actual_filter_sizes.append(my_file['filterSizes'][ii] - my_file['filterSizes'][ii-1])
            my_file['filterSizes'] = np.asarray(actual_filter_sizes)
 
        tes_struct = {
            'radianceStruct': my_file
        }

        # Set all 'instrument' field in all windows to 'TES'
        for ii in range(len(windows)):
            windows[ii]['instrument'] = 'TES'
        pge_flag = None
        target_radius = mpy.earth_radius(stateInfo['current']['latitude'],pge_flag)
        sat_altitude = 705000
        sat_radius = sat_altitude + target_radius
    
        stateInfo['current']['tes']['instrumentRadius'] = sat_radius
        stateInfo['current']['tes']['targetRadius'] = target_radius
        stateInfo['current']['tes']['orbitInclinationAngle'] = -99
        return tes_struct
    
    def _make_case_right(self, my_file):
        # Because all the fields in tesRadiance are uppercased from when they were read from external file, 
        # we have to make the cases right before calling radiance_set_windows(),
        # otherwise, the function will barf that it cannot find the correct key.
    
        translation_table = {
            'FILENAME' : 'filename',
            'INSTRUMENT' : 'instrument',
            'COMMENTS' : 'comments',
            'PREFERENCES' : 'preferences',
            'DETECTORS' : 'detectors',
            'MWS' : 'mws',
            'NUMDETECTORSORIG' : 'numDetectorsOrig',
            'NUMDETECTORS' : 'numDetectors',
            'NUM_FREQUENCIES' : 'num_frequencies',
            'RADIANCE' : 'radiance',
            'NESR' : 'NESR',
            'FREQUENCY' : 'frequency',
            'VALID' : 'valid',
            'FILTERSIZES' : 'filterSizes',
            'FILTERNAMES' : 'filterNames',
            'INSTRUMENTSIZES' : 'instrumentSizes',
            'INSTRUMENTNAMES' : 'instrumentNames',
            'PIXELSUSED' : 'pixelsUsed',
            'INTERPIXELVAR' : 'interpixelVar',
            'FREQSHIFT' : 'freqShift',
            'IMAGINARYMEAN' : 'imaginaryMean',
            'IMAGINARYRMS' : 'imaginaryRMS',
            'BT8' : 'bt8',
            'BT10' : 'bt10',
            'BT11' : 'bt11',
            'SCANDIRECTION' : 'scanDirection',
            'VALID' : 'valid'
        }
        return { translation_table.get(k, k) : v for k, v in my_file.items() }
    
    def _transpose_2d_arrays(self, my_file):
        # Because of how the 2D (and greater dimensions) arrays are
        # read in from external NetCDF file, we have to transpose them
        # so the shape will be correct.  otherwise, the function will
        # barf that it cannot find the correct key.
        
        res = {} # Returned dictionary with the right hand side transposed with the correctd shape.
    
        for key, value in my_file.items():
            # Keep the value on the right side as is as default.
            res[key] = value
    
            # Check to see if the array is 2 D or more so we can transpose it.
            # Also, don't transpose 'filterNames' since it is of string type.
            if (isinstance(value, np.ndarray) and len(value.shape) >= 2 and
                key != 'filterNames'):
                # Don't transpose if the shape ends with 1: (18554, 1)
                if (value.shape[-1] != 1):
                    res[key] = np.transpose(value)
    
        return res
    
    def _find_bin(self, x, y):
        # IDL_LEGACY_NOTE: This function _find_bin is the same as findbin in run_irk.pro file.
        #
        # Returns the bin numbers for nearest value of x array to values of y
        #       returns nearest bin for values outside the range of x
        #
        ny = len(y)
    
        o_bin = np.ndarray(shape=(ny), dtype=np.int32)
        for iy in range(0, ny):
            ix = np.argmin(abs(x - y[iy]))
            o_bin[iy] = ix
    
        if (ny == 1):
            o_bin = np.asarray([o_bin[0]])
    
        return o_bin
    
RetrievalStrategyStepSet.add_default_handle(RetrievalStrategyStepIRK())
    
__all__ = [ "RetrievalStrategyStepIRK",]

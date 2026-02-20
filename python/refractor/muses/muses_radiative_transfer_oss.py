from __future__ import annotations
import refractor.framework as rf  # type: ignore
from .muses_oss_handle import muses_oss_handle
import numpy as np
import os
import typing
from typing import Self

if typing.TYPE_CHECKING:
    from .identifier import StateElementIdentifier, InstrumentIdentifier
    from .input_file_helper import InputFilePath, InputFileHelper

class MusesRadiativeTransferOss(rf.RadiativeTransferImpBase):
    """This uses the muses OSS code (package muses-oss). This gives a forward
    model that is the same as the py-retrieve airs/cris/tes forward model (with minor
    differences in calculation - the normal sort of round off differences).
    """

    def __init__(
        self,
        rf_uip: rf.RefractorUip, # Temp, leverage off UIP. We'll remove this in a bit
        instrument_name: InstrumentIdentifier,
        ifile_hlp: InputFileHelper,
        retrieval_state_element_id: list[StateElementIdentifier],
        species_list: list[StateElementIdentifier],
        nlevels: int,
        nfreq: int, # This seems to be the size of the emissivity. Perhaps verify,
                    # And if so change it name. This has nothing to do with the
                    # size of freq_oss that gets filled in
        sel_file : str | os.PathLike[str] | InputFilePath,
        od_file : str | os.PathLike[str] | InputFilePath,
        sol_file : str | os.PathLike[str] | InputFilePath,
        fix_file : str | os.PathLike[str] | InputFilePath,
    ) -> None:
        super().__init__()
        self.rf_uip = rf_uip
        self.instrument_name = instrument_name
        self.ifile_hlp = ifile_hlp
        self.retrieval_state_element_id = retrieval_state_element_id
        self.species_list = species_list
        self.nlevels = nlevels
        self.nfreq = nfreq
        self.sel_file = sel_file
        self.od_file = od_file
        self.sol_file = sol_file
        self.fix_file = fix_file


    def clone(self) -> Self:
        return MusesRadiativeTransferOss(
            self.ifile_hlp,
            self.retrieval_state_element_id,
            self.species_list,
            self.nlevels,
            self.nfreq,
            self.sel_file,
            self.od_file,
            self.sol_file,
            self.fix_file,
        )

    def reflectance(
        self, sd: rf.SpectralDomain, sensor_index: int, skip_jacobian: bool
    ) -> rf.Spectrum:
        muses_oss_handle.oss_init(self.ifile_hlp, self.retrieval_state_element_id,
                                  self.species_list, self.nlevels,
                                  self.nfreq,
                                  self.sel_file,
                                  self.od_file,
                                  self.sol_file,
                                  self.fix_file)
        muses_oss_handle.oss_channel_select(sd)
        uip_all = self.rf_uip.uip_all(self.instrument_name)
        uip_all["oss_jacobianList"] = [str(s) for s in muses_oss_handle.jac_spec]
        uip_all["oss_frequencyList"] = list(sd.convert_wave("nm"))
        rad, jac = self.fm_oss_stack(uip_all)
        sub_basis_matrix = self.rf_uip.instrument_sub_basis_matrix(
            self.instrument_name
        )
        # See MusesForwardModelHandle for a discussion of have_fake_jac_in_oss
        if jac is not None and jac.ndim > 0 and len(self.retrieval_state_element_id) > 0:
            jac = np.matmul(sub_basis_matrix, jac).transpose()
            a = rf.ArrayAd_double_1(rad, jac)
        else:
            a = rf.ArrayAd_double_1(rad)
        sr = rf.SpectralRange(a, rf.Unit("sr^-1"))
        return rf.Spectrum(sd, sr)
        
    def stokes(self, sd: rf.SpectralDomain, sensor_index: int) -> np.ndarray:
        raise NotImplementedError(
            """Muses-oss doesn't work for  the full
            stoke vector."""
        )

    def stokes_and_jacobian(
        self, sd: rf.SpectralDomain, sensor_index: int
    ) -> rf.ArrayAd_double_2:
        raise NotImplementedError(
            """Muses-oss doesn't work for  the full
            stoke vector."""
        )

    def desc(self) -> str:
        return "MusesRadiativeTransferOss"


    def fm_oss(self, i_uip, i_jacobians):
        from refractor.muses_py import fm_oss_load
        from ctypes import c_float, POINTER, c_int
        import math
        function_name = "fm_oss: "

        oss_wrapper = fm_oss_load() 

        # AT_LINE 8 fm_oss.pro
        #if dirOSS is None or dirOSS == '':
        #    dirOSS = '../OSP/OSS_FM/OSS/2017-02/'

        # AT_LINE 10 fm_oss.pro
        if i_jacobians is not None:
            njacob = len(i_jacobians)
        else:
            njacob = 0

        # AT_LINE 11 fm_oss.pro
        # The value of nameJacobian_arr is not used.
        # byte_array     = [elem for elem in str.encode(i_jacobians)]
        # nameJacobian_arr = np.array(byte_array)
        # nameJacobian_arr = nameJacobian_arr.astype(np.float32)

        # AT_LINE 12 fm_oss.pro
        # PYTHON_NOTE: Because the array i_uip['atmosphere'] will be messed with later, thus corrupt our memory,
        #              we need to allocate memory for both pressure and tatm arrays.
        pressure = np.ndarray(shape=(i_uip['atmosphere'].shape[1]), dtype=np.float32)
        tatm = np.ndarray(shape=(i_uip['atmosphere'].shape[1]), dtype=np.float32)
        pressure[:] = i_uip['atmosphere'][0, :]
        tatm[:] = i_uip['atmosphere'][1, :]

        # AT_LINE 17 fm_oss.pro
        #  res = call_external(dirOSS+'lib/linux_x86_64/liboss.so', 'IDLsetChanSelect', 1, /F_VALUE)

        oss_wrapper.FMOSSSetChanSelect.argtypes = [c_int, c_int]
        oss_wrapper.FMOSSSetChanSelect(1, 0)

        # AT_LINE 19 fm_oss.pro
        nchanOSS = len(i_uip['oss_frequencyList'])
        sunang = 90. 
        nemis = len(i_uip['emissivity']['frequency'])
        ncloud = len(i_uip['cloud']['frequency'])
        nlevels = len(pressure)

        y = np.zeros(shape=(nchanOSS), dtype=np.float32)
        xkTemp = np.zeros(shape=(nlevels, nchanOSS), dtype=np.float32)
        xkTskin = np.zeros(shape=(nchanOSS), dtype=np.float32)
        if njacob > 1:
            # If njacob is greater than 1, we don't need to include the 3rd dimension as it is not needed.
            xkOutGas = np.zeros(shape=(nlevels, nchanOSS, njacob), dtype=np.float32)
        else:
            # If njacob is 1, we don't need to include the 3rd dimension of the xkOutGas array.
            xkOutGas = np.zeros(shape=(nlevels, nchanOSS), dtype=np.float32)

        xkEm = np.zeros(shape=(nemis, nchanOSS), dtype=np.float32)
        xkRf = np.zeros(shape=(nemis, nchanOSS), dtype=np.float32)
        xkCldlnPres = np.zeros(shape=(nchanOSS), dtype=np.float32)
        xkCldlnExt = np.zeros(shape=(ncloud, nchanOSS), dtype=np.float32)

        if float(i_uip['obs_table']['pointing_angle_surface'] * 180 / np.pi) < -990:
            print(function_name, 'Error! Need to define uip.obs_table.pointing_angle_surface (radians)')
            assert False

        # Set values to 1e-20 if NOT in uip.species.
        # Check for both b'CO2' and 'CO2'.  b'CO2' exists when running fm_oss from a netcdf uip file.
        ns = len(i_uip['atmosphere_params'])
        for jj in range(ns):
            search = i_uip['atmosphere_params'][jj]
            if search not in i_uip['species']:
                i_uip['atmosphere'][jj, :] = 1e-20
        # end for jj in range(ns):

        # check for negative values in PAN VMR.  If there are negative values:
        # 1) set VMR0 to original VMR, set VMR to 1e-11
        # 2) run OSS
        # 3) modify radiance by K ## (VMR0 - VMR)
        # 4) re-set VMR to VMR0

        pan_negative = False
        indpan = np.where(i_uip['atmosphere_params'] == 'PAN')[0]
        if len(indpan) > 0:
            # assume there is only one parameter for PAN
            indpan = indpan[0]

            # Force negative PAN for testing. Leave commented out if not testing
            # i_uip['atmosphere'][indpan, 0] = -2.16e-09

            indneg = np.where(i_uip['atmosphere'][indpan, :] < 0)[0]
            if len(indneg) > 0:
                pan_vmr_muses = np.copy(i_uip['atmosphere'][indpan, :])
                pan_vmr_oss = np.copy(pan_vmr_muses)
                indneg = np.where(i_uip['atmosphere'][indpan, :] < 1e-11)[0]
                pan_vmr_oss[indneg] = 1e-11
                i_uip['atmosphere'][indpan] = pan_vmr_oss[:]
                pan_negative = True
            # end if len(indneg) > 0:
        # end if len(indpan) > 0:


        nh3_negative = False
        indnh3 = np.where(i_uip['atmosphere_params'] == 'NH3')[0]
        if len(indnh3) > 0:
            # assume indnh3 is only one parameter for NH3
            indnh3 = indnh3[0]

            # Force negative PAN for testing. Leave commented out if not testing
            # i_uip['atmosphere'][indpan, 0] = -2.16e-09

            indneg = np.where(i_uip['atmosphere'][indnh3, :] < 0)[0]
            if len(indneg) > 0:
                nh3_vmr_muses = np.copy(i_uip['atmosphere'][indnh3, :])
                nh3_vmr_oss = np.copy(nh3_vmr_muses)
                indneg = np.where(i_uip['atmosphere'][indnh3, :] < 1e-11)[0]
                nh3_vmr_oss[indneg] = 1e-11
                i_uip['atmosphere'][indpan] = nh3_vmr_oss[:]
                nh3_negative = True
            # end if len(indneg) > 0:
        # end if len(indnh3) > 0:


        # AT_LINE 72 fm_oss.pro
        # index 1: pressure, index 2: temperature.  OSS puts those separately elsewhere.
        indspecies = [(jj + 2) for jj in range(ns - 2)]

        # AT_LINE 74 fm_oss.pro
        natmosphere_params = len(indspecies)

        surfaceAltitude = i_uip['obs_table']['surfaceAltitude']
        if abs(surfaceAltitude) < 1e-5:
            surfaceAltitude = 1e-5

        # AT_LINE 69 fm_oss.pro
        atmosphere = (i_uip['atmosphere'][np.asarray(indspecies), :]).T
        atmosphere = atmosphere.astype(np.float32)

        # AT_LINE 71 fm_oss.pro
        ss_info = {
            'nlevels': nlevels,                     
            'natmosphere_params': natmosphere_params,          
            'pressure': pressure,                    
            'tatm': tatm,                        
            'tsur': i_uip['surface_temperature'],
            'atmosphere': atmosphere,                      
            'nemis': nemis,                           
            'emis': (i_uip['emissivity']['value']).astype(np.float32),    
            'refl': (1 - i_uip['emissivity']['value']).astype(np.float32),  
            'scale_pressure': i_uip['cloud']['scale_pressure'],
            'pcloud': i_uip['cloud']['pressure'],          
            'ncloud': ncloud,                              
            'cloudext': (i_uip['cloud']['extinction']).astype(np.float32),        
            'emis_freq': (i_uip['emissivity']['frequency']).astype(np.float32),    
            'cloud_freq': (i_uip['cloud']['frequency']).astype(np.float32),         
            'ptgang': i_uip['obs_table']['pointing_angle_surface'] * 180 / math.pi, 
            'sunang': sunang,                                                   
            'latitude': i_uip['obs_table']['target_latitude'] * 180 / math.pi,        
            'surfaceAltitude': surfaceAltitude,     
            'something': 1,                   
            'njacobians': njacob,                   
            'nchanOSS': nchanOSS,                 
            'y': y,                        
            'xktemp': xkTemp,                   
            'xktskin': xkTskin,                  
            'xkoutgas': xkOutGas,                   
            'xkem': xkEm,                       
            'xkrf': xkRf,                       
            'xkcldlnpres': xkCldlnPres,             
            'xkcldlnext': xkCldlnExt
        }

        # write out diagnostic to see exactly what passed into OSS.
        #from py_retrieve.app.tools.dict_tools.cdf_write_dict import cdf_write_dict
        #cdf_write_dict(ss_info, 'oss_input.nc')

        # AT_LINE 91 fm_oss.pro
        # Res = call_external(dirOSS+'lib/linux_x86_64/liboss.so', 'IDLfwdWrapper', $

        # Define the function signature.  The Python intepreter will barf if the types are wrong, which is good.
        oss_wrapper.FMOSSfwdWrapper.argtypes = [
            c_int,              # ss_info['nlevels'] 
            c_int,              # ss_info['natmosphere_params']
            POINTER(c_float),   # ss_info['pressure']
            POINTER(c_float),   # ss_info['tatm']
            c_float,            # ss_info['tsur']
            POINTER(c_float),   # atmosphere
            c_int,              # ss_info['nemis']
            POINTER(c_float),   # ss_info['emis']
            POINTER(c_float),   # ss_info['refl']
            c_float,            # ss_info['scale_pressure']
            c_float,            # ss_info['pcloud']
            c_int,              # ss_info['ncloud']
            POINTER(c_float),   # ss_info['cloudext']
            POINTER(c_float),   # ss_info['emis_freq']
            POINTER(c_float),   # ss_info['cloud_freq']
            c_float,            # ss_info['ptgang']
            c_float,            # ss_info['sunang']
            c_float,            # ss_info['latitude']
            c_float,            # ss_info['surfaceAltitude']
            c_int,              # ss_info['something']
            c_int,              # ss_info['njacobians']
            c_int,              # ss_info['nchanOSS']
            POINTER(c_float),   # y
            POINTER(c_float),   # xkTemp
            POINTER(c_float),   # xkTskin
            POINTER(c_float),   # xkOutGas
            POINTER(c_float),   # xkEm
            POINTER(c_float),   # xkRf
            POINTER(c_float),   # xkCldlnPres
            POINTER(c_float)    # xkCldlnExt
        ]

        # Create pointers so we can pass them to C function.  After the call to the FORTRAN code, we will need to extract the values back.
        #see: https://cvstuff.wordpress.com/2014/11/27/wraping-c-code-with-python-ctypes-memory-and-pointers/
        # inputs
        pressure_ptr = (c_float * len(ss_info['pressure']))(*ss_info['pressure'].flatten())  
        tatm_ptr = (c_float * len(ss_info['tatm']))(*ss_info['tatm'].flatten())      

        atmosphere_ptr = (c_float * atmosphere.size)(*atmosphere.flatten(order='F'))  
        emis_ptr = (c_float * len(ss_info['emis']))(*ss_info['emis'].flatten())
        refl_ptr = (c_float * len(ss_info['refl']))(*ss_info['refl'].flatten())

        cloudext_ptr = (c_float * len(ss_info['cloudext']))(*ss_info['cloudext'].flatten())

        emis_freq_ptr = (c_float * len(ss_info['emis_freq']))(*ss_info['emis_freq'].flatten())
        cloud_freq_ptr = (c_float * len(ss_info['cloud_freq']))(*ss_info['cloud_freq'].flatten())

        # outputs
        y_ptr = (c_float*nchanOSS)(*y.flatten(order='F'))
        xkTemp_ptr = (c_float*xkTemp.size)(*xkTemp.flatten(order='F'))

        xkTskin_ptr = (c_float * nchanOSS)(*xkTskin.flatten())
        xkOutGas_ptr = (c_float * xkOutGas.size)(*xkOutGas.flatten(order='F'))
        xkEm_ptr = (c_float * xkEm.size)(*xkEm.flatten(order='F'))
        xkRf_ptr = (c_float * xkRf.size)(*xkRf.flatten(order='F'))
        xkCldlnPres_ptr = (c_float * nchanOSS)(*xkCldlnPres.flatten())
        xkCldlnExt_ptr = (c_float * xkCldlnExt.size)(*xkCldlnExt.flatten(order='F'))

        # These are the names of the parameters to FMOSSfwdWrapper function:
        # ss_info['nlevels'],   ss_info['natmosphere_params'], ss_info['pressure'], ss_info['tatm'], ss_info['tsur'],
        # ss_info['atmosphere'], ss_info['nemis'],             ss_info['emis'],    ss_info['refl'],     ss_info['scale_pressure'],
        # ss_info['pcloud'],    ss_info['ncloud'],            ss_info['cloudext'], ss_info['emis_freq'], ss_info['cloud_freq'],
        # ss_info['ptgang'],    ss_info['sunang'],            ss_info['latitude'], ss_info['surfaceAltitude'], ss_info['something'],
        # ss_info['njacobians'], ss_info['nchanOSS'],          y,                  xkTemp,               xkTskin,
        # xkOutGas,             xkEm,                         xkRf,               xkCldlnPres,          xkCldlnExt)

        # Make the call to the FORTRAN code passing in addresses of anything that are pointers.
        oss_wrapper.FMOSSfwdWrapper(
            ss_info['nlevels'], 
            ss_info['natmosphere_params'], 
            pressure_ptr,           
            tatm_ptr,            
            ss_info['tsur'],
            atmosphere_ptr,             
            ss_info['nemis'],             
            emis_ptr,           
            refl_ptr,            
            ss_info['scale_pressure'],
            ss_info['pcloud'],    
            ss_info['ncloud'],            
            cloudext_ptr,           
            emis_freq_ptr,            
            cloud_freq_ptr,
            ss_info['ptgang'],    
            ss_info['sunang'],            
            ss_info['latitude'], 
            ss_info['surfaceAltitude'], 
            ss_info['something'],
            ss_info['njacobians'], 
            ss_info['nchanOSS'],          
            y_ptr,           
            xkTemp_ptr,            
            xkTskin_ptr,
            xkOutGas_ptr,
            xkEm_ptr,
            xkRf_ptr,
            xkCldlnPres_ptr,
            xkCldlnExt_ptr
        )

        # The c function above would have filled in the values in the pointers, we now assigned them back, esssentially receiving output
        # from the c function.

        y[:] = y_ptr[:]

        xkTemp[:, :] = np.reshape(xkTemp_ptr, (nlevels, nchanOSS), order='F')

        xkTskin[:] = xkTskin_ptr[:]

        # Watching out for the 3rd dimension or not.
        if njacob > 1:
            xkOutGas[:, :, :] = np.reshape(xkOutGas_ptr, (nlevels, nchanOSS, njacob), order='F')
        else:    
            xkOutGas[:, :] = np.reshape(xkOutGas_ptr, (nlevels, nchanOSS), order='F')

        xkEm[:, :] = np.reshape(xkEm_ptr, (nemis, nchanOSS), order='F')

        xkRf[:, :] = np.reshape(xkRf_ptr, (nemis, nchanOSS), order='F')

        xkCldlnPres[:] = xkCldlnPres_ptr[:]

        xkCldlnExt[:, :] = np.reshape(xkCldlnExt_ptr, (ncloud, nchanOSS), order='F')

        # AT_LINE 124 fm_oss.pro
        o_result = {
            'radiance': y * 1e-4,
            'xkTemp': xkTemp * 1e-4,
            'xkTskin': xkTskin * 1e-4,
            'xkOutGas': xkOutGas * 1e-4,
            'xkEm': xkEm * 1e-4,
            'xkRf': xkRf * 1e-4,
            'xkCldlnPres': xkCldlnPres * 1e-4,
            'xkCldlnExt': xkCldlnExt * 1e-4,
            'pressure': pressure,
            'nameJacobian': i_jacobians,
            'frequency': i_uip['oss_frequencyList'] # vp changed from frequencylist to oss_frequencylist
        }

        # AT_LINE 134 src_ms-2018-12-10/fm_oss.pro
        # update naming to be consistent with ELANOR
        for jj in range(0, len(o_result['nameJacobian'])):
            if o_result['nameJacobian'][jj] == 'F11':
                o_result['nameJacobian'][jj] = 'CFC11'

            if o_result['nameJacobian'][jj] == 'F12':
                o_result['nameJacobian'][jj] = 'CFC12'

            if o_result['nameJacobian'][jj] == 'C5H8':
                o_result['nameJacobian'][jj] = 'ISOP'

            if o_result['nameJacobian'][jj] == 'CHCLF2':
                o_result['nameJacobian'][jj] = 'CFC22'
        # end for jj in range(0,len(o_result['nameJacobian')):

        if pan_negative:
            name_jacobians_stripped = np.char.strip(o_result['nameJacobian'])
            indjac = np.where(np.char.strip(name_jacobians_stripped) == 'PAN')[0]
            indpan = np.where(i_uip['atmosphere_params'] == 'PAN')[0]
            if len(indjac) > 0:
                indjac = indjac[0]
                indpan = indpan[0]
                if len(o_result['nameJacobian']) > 1:
                    k = np.copy(o_result['xkOutGas'][:, :, indjac])
                elif len(o_result['nameJacobian']) == 1:
                    k = np.copy(o_result['xkOutGas'])

                # make linear Jacobian
                for kk in range(len(pan_vmr_oss)):
                    k[kk, :] = k[kk, :] / pan_vmr_oss[kk]

                # modify radiance to ACTUAL VMR using K.dx
                # vmr0 used by OSS, vmr is what we want
                dL = k.T @ (pan_vmr_muses - pan_vmr_oss)

                i_uip['atmosphere'][indpan, :] = pan_vmr_muses
                o_result['radiance'] = o_result['radiance'] + dL

                # update "log" Jacobian to multiply by the MUSES VMR
                for kk in range(len(pan_vmr_oss)):
                    k[kk, :] = k[kk, :] * pan_vmr_muses[kk]

                # remake "log" Jacobian with muses VMR
                if len(o_result['nameJacobian']) > 1:
                    o_result['xkOutGas'][:, :, indjac] = k
                elif len(o_result['nameJacobian']) == 1:
                    o_result['xkOutGas'] = k
            # if len(ind) > 0:
        # end if pan_negative:




        if nh3_negative:
            name_jacobians_stripped = np.char.strip(o_result['nameJacobian'])
            indjac = np.where(np.char.strip(name_jacobians_stripped) == 'NH3')[0]
            indnh3 = np.where(i_uip['atmosphere_params'] == 'NH3')[0]
            if len(indjac) > 0:
                indjac = indjac[0]
                indnh3 = indnh3[0]
                if len(o_result['nameJacobian']) > 1:
                    k = np.copy(o_result['xkOutGas'][:, :, indjac])
                elif len(o_result['nameJacobian']) == 1:
                    k = np.copy(o_result['xkOutGas'])

                # make linear Jacobian
                for kk in range(len(nh3_vmr_oss)):
                    k[kk, :] = k[kk, :] / nh3_vmr_oss[kk]

                # modify radiance to ACTUAL VMR using K.dx
                # vmr0 used by OSS, vmr is what we want
                dL = k.T @ (nh3_vmr_muses - nh3_vmr_oss)

                i_uip['atmosphere'][indnh3, :] = nh3_vmr_muses
                o_result['radiance'] = o_result['radiance'] + dL

                # update "log" Jacobian to multiply by the MUSES VMR
                for kk in range(len(nh3_vmr_oss)):
                    k[kk, :] = k[kk, :] * nh3_vmr_muses[kk]

                # remake "log" Jacobian with muses VMR
                if len(o_result['nameJacobian']) > 1:
                    o_result['xkOutGas'][:, :, indjac] = k
                elif len(o_result['nameJacobian']) == 1:
                    o_result['xkOutGas'] = k
            # if len(indjac) > 0:
        # end if nh3_negative:


        # check finite
        if not np.all(np.isfinite(y)):
            print(function_name, "ERROR: Non-finite radiance!")
            assert False

        if not np.all(np.isfinite(o_result['xkOutGas'])):
            print(function_name, "ERROR: Non-finite jacobians!")
            assert False


        return o_result

    def fm_oss_stack(self, uipIn):
        from refractor.muses_py import pack_jacobian

        # AT_LINE 5 fm_oss_stack.pro
        uip = uipIn

        jacobianList = uip['oss_jacobianList']

        results = self.fm_oss(uip, jacobianList)

        rad = results['radiance']

        # Prepare Jacobians for pack_jacobian function.
        uip['num_atm_k'] = len(results['nameJacobian'])

        numTatm = 0
        if 'TATM' in uip['jacobians']:
            numTatm = 1

        species_list = []
        k_species = []

        # AT_LINE 16 fm_oss_stack.pro
        if uip['num_atm_k'] > 0:
            # Create a list of uip['num_atm_k']+numTatm dictionaries with the format of k_struct.
            for x in range(uip['num_atm_k']+numTatm):
                k_struct = {
                    'species': results['nameJacobian'][0], 
                    'k': np.zeros(shape=(len(results['pressure']), len(results['frequency'])), dtype=np.float32)
                }
                k_species.append(k_struct)

            atm_jacobians_ils_total = {'k_species': k_species}


            # AT_LINE 22 fm_oss_stack.pro
            for jj in range(uip['num_atm_k']):
                species_list.append(results['nameJacobian'][jj].lstrip().rstrip())
                atm_jacobians_ils_total['k_species'][jj]['species'] = results['nameJacobian'][jj].lstrip().rstrip()

                if uip['num_atm_k'] > 1:
                    atm_jacobians_ils_total['k_species'][jj]['k'] = results['xkOutGas'][:, :, jj]

                if uip['num_atm_k'] == 1:
                    atm_jacobians_ils_total['k_species'][jj]['k'] = results['xkOutGas']

                # AT_LINE 28 fm_oss_stack.pro
                # update naming to be consistent with ELANOR
                if atm_jacobians_ils_total['k_species'][jj]['species'] == 'F11':
                    atm_jacobians_ils_total['k_species'][jj]['species'] = 'CFC11'

                if atm_jacobians_ils_total['k_species'][jj]['species'] == 'F12':
                    atm_jacobians_ils_total['k_species'][jj]['species'] = 'CFC12'

                if atm_jacobians_ils_total['k_species'][jj]['species'] == 'C5H8':
                    atm_jacobians_ils_total['k_species'][jj]['species'] = 'ISOP'

                if atm_jacobians_ils_total['k_species'][jj]['species'] == 'CHCLF2':
                    atm_jacobians_ils_total['k_species'][jj]['species'] = 'CFC22'
            # end for jj in range(uip['num_atm_k']):

        # AT_LINE 30 fm_oss_stack.pro
        # Add tatm if present into atmospheric Jacobians.
        if 'TATM' in uip['jacobians']:
            # Use the last value of jj plus 1 from the for loop above.  We have to add 1 in Python to not overwrite the last set values.
            atm_jacobians_ils_total['k_species'][jj + 1]['species'] = 'TATM'
            atm_jacobians_ils_total['k_species'][jj + 1]['k'] = results['xkTemp']
            species_list.append('TATM')

        # AT_LINE 36 fm_oss_stack.pro
        # Make cloud Jac structure.
        jacobian_cloud_map = {
            'k_height': results['xkCldlnPres'],
            'k_ext': results['xkCldlnExt']
        }

        # Make emissivity Jac structure.
        jacobian_emiss_ils_map = {
            'k': results['xkEm']
        }

        # AT_LINE 41 fm_oss_stack.pro
        # Pack jacobians together based on retrieval parameter ordering.
        # pack_jacobian, uip, rad

        (o_rad, o_jac) = pack_jacobian(
            uip, rad,
            jacobian_emiss_ils_map, results['xkTskin'],
            atm_jacobians_ils_total, 0, jacobian_cloud_map,
            0, 0, 0
        )

        # AT_LINE 47 fm_oss_stack.pro
        radiance = np.copy(o_rad)

        # Because we are not really sure the type of uip['jacobians_all'], we have to inspect to see if it is a list
        # or an ndarray
        to_copy_jacobian_flag = False
        if 'list' in str(type(uip['jacobians_all'])):
            if len(uip['jacobians_all']) > 0:
                to_copy_jacobian_flag = True
        elif 'ndarray' in str(type(uip['jacobians_all'])):
            if len(uip['jacobians_all']) > 0:
                to_copy_jacobian_flag = True

        if to_copy_jacobian_flag or uip['jacobians_all'] != '':
            jacobian = np.copy(o_jac)

        return (radiance, jacobian)

    
__all__ = [
    "MusesRadiativeTransferOss",
]

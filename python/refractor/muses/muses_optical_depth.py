import numpy as np
import os
import math
import refractor.framework as rf

class MusesOpticalDepth(rf.AbsorberXSec):
    '''This is like MusesOpticalDepthFile, but we try to use as much of ReFRACtor
    as possible. The optical depth information is from py-retrieve, so this handling
    doing PRECONV. But most of the calculations are done using ReFRACtor code.'''
    def __init__(self, pressure : "rf.Pressure", temperature : "rf.Temperature",
                 altitude : "rf.Altitude", absorber_vmr : "rf:AbsorberVmr",
                 num_channel : int,
                 input_dir : str):
        '''We can hopefully get rid of using the input directory, for now we read
        from there. Note that this needs to have already been generated before we
        enter this function, so the UIP should have been created.'''
        # Dummy since we are overwriting the optical_depth function
        xsec_tables = rf.vector_xsec_table()

        spec_grid = rf.ArrayWithUnit(np.array([1, 2]), "nm")
        xsec_values = rf.ArrayWithUnit(np.zeros((2, 1)), "cm^2")
        cfac = rf.cross_section_file_conversion_factors.get("O3", 1.0)
        xsec_tables.push_back(rf.XSecTableSimple(spec_grid, xsec_values, 0.0))

        # Register base director class
        rf.AbsorberXSec.__init__(self, absorber_vmr, pressure, temperature, altitude, xsec_tables)
        self.xsect_grid, self.xsect_data = self._xsect_data(num_channel, input_dir)

    def _xsect_data(self, num_channel, input_dir):
        '''Read optical depth values from MUSES written files.
        '''
        nlay = self.pressure.number_layer
        
        # The UV1 and UV2 are separate files, but we combine them into
        # one cross section table
        # Note these files are generated once per strategy step
        file_data = np.loadtxt(f"{input_dir}/O3Xsec_MW001.asc", skiprows=1)
        for mw_num in range(2, num_channel + 1):  # 1-based indexing
            # We may have more channels than we actually are running the forward model for
            # (e.g., channels with zero width spectral domain). In that case, just skip files
            try:
                mw_file_data = np.loadtxt(f"{input_dir}/O3Xsec_MW{mw_num:03}.asc", skiprows=1)
                file_data = np.concatenate([file_data, mw_file_data])
            except FileNotFoundError:
                pass

        # Data needs to be sorted by wavelength. This is a little
        # cryptic, but this sorts all the data by column 1
        file_data = file_data[file_data[:, 1].argsort()]

        xsect_grid = file_data[:, 1]
        xsect_data = file_data[:, 2:nlay+2]
        return xsect_grid, xsect_data
        
    def optical_depth_each_layer(self, wn, spec_index):
        # Convert value to units of spectral points used in file
        spec_point = rf.DoubleWithUnit(wn, "cm^-1").convert_wave("nm").value

        # Find index of closest value
        od_index = np.searchsorted(self.xsect_grid, spec_point, side="left")
        if od_index > 0 and \
           (od_index == self.xsect_grid.shape[0] or
            math.fabs(spec_point - self.xsect_grid[od_index - 1]) < math.fabs(spec_point - self.xsect_grid[od_index])):
            od_index -= 1

        # Extra axis is the species index, not used since we only know about ozone
        wn_xsect_data = self.xsect_data[od_index, :]
        gdens = self.gas_number_density_layer(spec_index).value
        wn_od_data = wn_xsect_data[:,np.newaxis] * gdens.value
        if(gdens.is_constant):
            od_result = rf.ArrayAd_double_2(wn_od_data)
        else:
            wn_od_data_jac = wn_xsect_data[:,np.newaxis,np.newaxis] * gdens.jacobian
            od_result = rf.ArrayAd_double_2(wn_od_data, wn_od_data_jac)
        return od_result
        
    def desc(self):
        s = "MusesOpticalDepth\n"
        s += self.print_parent()
        return s

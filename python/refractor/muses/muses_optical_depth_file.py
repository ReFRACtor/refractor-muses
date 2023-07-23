import numpy as np
import os
import math
import refractor.framework as rf

class MusesOpticalDepthFile(rf.AbsorberXSec, rf.CacheInvalidatedObserver):
    """
    Returns the precomputed MUSES optical depth files in the misleadingly
    named O3Xsec_MW???.asc files. Simply opens the files then returns
    the nearest value to the wavenumber given to optical_depth_each_layer.
    """

    def __init__(self, rf_uip, pressure, temperature, altitude, absorber_vmr, num_channel):

        # Dummy since we are overwriting the optical_depth function
        xsec_tables = rf.vector_xsec_table()

        spec_grid = rf.ArrayWithUnit(np.array([1, 2]), "nm")
        xsec_values = rf.ArrayWithUnit(np.zeros((2, 1)), "cm^2")
        cfac = rf.cross_section_file_conversion_factors.get("O3", 1.0)
        xsec_tables.push_back(rf.XSecTableSimple(spec_grid, xsec_values, 0.0))

        # Register base director class
        rf.AbsorberXSec.__init__(self, absorber_vmr, pressure, temperature, altitude, xsec_tables)
        rf.CacheInvalidatedObserver.__init__(self)

        self.rf_uip = rf_uip
        self.num_channel = num_channel

        self._pressure = pressure
        self._temperature = temperature
        self._altitude = altitude
        self._absorber_vmr = absorber_vmr

        # Where MUSES stores the computations it makes for VLIDORT we are leveraging
        self.input_dir = os.path.join(os.getenv('MUSES_DEFAULT_RUN_DIR', '.'), self.rf_uip.vlidort_input)

        # Reverse profile order to OD computation order
        self.map_vmr_l = rf_uip.ray_info['map_vmr_l'][::-1]
        self.map_vmr_u = rf_uip.ray_info['map_vmr_u'][::-1]

        # Initialize caches
        self.xsect_data = None
        self.xsect_data_temp = None
        self.gas_density_lay = None

        # Invalidate cache when absorber_vmr changes. This changes
        # self.cache_valid_flag to False
        for a in self._absorber_vmr:
            a.add_cache_invalidated_observer(self)

    # Target the renamed funciton due to use of %python_attribute
    def _v_number_species(self):
        # By default number_species() in Absorber return 0
        return 1

    def gas_name(self, species_index):
        # All ozone all the time
        return "O3"

    def cache_xsect_data(self):
        """Read optical depth values from MUSES written files.
        """
        # Note, it is not a mistake that we don't check self.cache_valid_flag
        # here. The xsect data gets created once, as a side effect of
        # creating the uip. This object has a life span determined by the
        # uip, so throughout the life time of this object we only have
        # one file to read. The self.cache_valid_flag is tied to the AbsorberVmr,
        # which *does* change.
        nlay = self._pressure.number_layer

        # Recompute if value is undefined or if number of layers in pressure grid has changed
        if self.xsect_data is not None and self.xsect_data.shape[1] == nlay:
            return

        # The UV1 and UV2 are separate files, but we combine them into
        # one cross section table
        # Note these files are generated once per strategy step
        file_data = np.loadtxt(f"{self.rf_uip.rundir}/{self.rf_uip.vlidort_input}/O3Xsec_MW001.asc", skiprows=1)
        for mw_num in range(2, self.num_channel + 1):  # 1-based indexing
            mw_file_data = np.loadtxt(f"{self.rf_uip.rundir}/{self.rf_uip.vlidort_input}/O3Xsec_MW{mw_num:03}.asc", skiprows=1)
            file_data = np.concatenate([file_data, mw_file_data])

        # Data needs to be sorted by wavelength. This is a little
        # cryptic, but this sorts all the data by column 1
        file_data = file_data[file_data[:, 1].argsort()]


        # If we are doing temperature shifting, then we have a second file
        # of O3 data at a different temperature value. Grab that data. But
        # only do this if we need it to calculate a jacbian
        file_data_temp = None
        if(os.path.exists(f"{self.rf_uip.rundir}/{self.rf_uip.vlidort_input}/O3Xsec_MW001_TEMP.asc") and
           "TROPOMITEMPSHIFTBAND3" in self.rf_uip.state_vector_params):
            file_data_temp = np.loadtxt(f"{self.rf_uip.rundir}/{self.rf_uip.vlidort_input}/O3Xsec_MW001_TEMP.asc", skiprows=1)
            for mw_num in range(2, self.num_channel + 1):  # 1-based indexing
                mw_file_data = np.loadtxt(f"{self.rf_uip.rundir}/{self.rf_uip.vlidort_input}/O3Xsec_MW{mw_num:03}_TEMP.asc", skiprows=1)
                file_data_temp = np.concatenate([file_data_temp, mw_file_data])
            file_data_temp = file_data_temp[file_data_temp[:, 1].argsort()]

        self.xsect_grid = file_data[:, 1]

        self.xsect_data = file_data[:, 2:nlay + 2]
        if(file_data_temp is not None):
            self.xsect_data_temp = file_data_temp[:, 2:nlay + 2]
        else:
            self.xsect_data_temp = None
            
    def total_air_number_density_layer(self, spec_index):

        # The output of this routine is used by RamanSioris
        # Return the MUSES value for consistency

        nlay = self._pressure.number_layer
        dry_air_density = self.rf_uip.ray_info['column_air'][::-1][:nlay]

        return rf.ArrayAdWithUnit_double_1(rf.ArrayAd_double_1(dry_air_density), rf.Unit("cm^-2")) 

    def cache_gas_number_density_layer(self):
        """Computes the gas number density value per layer 
        The value is cached until recomputed due to AbsorberVmr changes.
        """

        nlay = self._pressure.number_layer

        # Recompute if value is undefined or if number of layers in pressure grid has changed
        if self.cache_valid_flag and self.gas_density_lay is not None and self.gas_density_lay.shape[0] == nlay:
            return

        o3_ind = np.where(np.asarray(self.rf_uip.ray_info['level_params']['species']) == 'O3')[0]

        # Reverse from MUSES increasing altitude to internal increasing pressure order
        self.gas_density_lay = self.rf_uip.ray_info['column_species'][o3_ind, ::-1].squeeze()[:nlay]
        self.cache_valid_flag = True

    def gas_number_density_layer(self, spec_index):
        self.cache_xsect_data()
        self.cache_gas_number_density_layer()
        return rf.ArrayAdWithUnit_double_2(rf.ArrayAd_double_2(self.gas_density_lay.reshape(-1,1)), "cm^-2")

    # We shouldn't be calling the level versions anywhere, but because of how
    # we calculate layer stuff the level isn't consistent. Throw an error just so
    # we don't mistakenly use this somewhere
    def _v_total_air_number_density_level(self):
        raise NotImplementedError("We don't support level versions of functions in MusesOpticalDepthFile")

    def _v_gas_number_density_level(self):
        raise NotImplementedError("We don't support level versions of functions in MusesOpticalDepthFile")
    
    def optical_depth_each_layer(self, wn, spec_index):
        # Note that this has a pretty clumsy interface in py-retrieve.
        # We 1) Need to have print_omi_o3xsec called (part of make_uip_tropomi
        # or make_uip_omi) - i.e., this depends on side effects of calling
        # run_retrieval. 2) We only have the information for calculating
        # d optical_depth_each_layer / d log(vmr).
        # We use the chain rule to get d optical_depth_each_layer / d state_vector
        # Also, the absorber_vmr doesn't get directly used. Instead, we
        # have duplicate
        # information in the uip, which gets used by
        # mpy.atmosphere_level -> rf_uip.atm_params -> rf_uip.ray_info ->
        # self.cache_gas_number_density_layer

        # Note INCREASING_PRESSURE is important here. We want to relate
        # this to dod_dlogvmr below, which is in INCREASING_PRESSURE. It
        # is fine if the StateVector isn't in this order, the dlogvmr_dstate
        # will automatically shuffle stuff around to get the proper StateVector
        # order.
        vgrid = self.absorber_vmr("O3").vmr_grid(self._pressure,
                                         rf.Pressure.INCREASING_PRESSURE)
        is_constant = vgrid.is_constant
        if(not is_constant):
            dvmr_dstate = vgrid.jacobian
            dlogvmr_dvmr = np.diag(1 / vgrid.value)
            dlogvmr_dstate = np.matmul(dlogvmr_dvmr, dvmr_dstate)
        
        # Ensure data is read if not already cached
        self.cache_xsect_data()
        self.cache_gas_number_density_layer()

        nlay = self._pressure.number_layer

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
        wn_od_data = wn_xsect_data * self.gas_density_lay
        wn_od_data_jac = None
        if(self.xsect_data_temp is not None):
            wn_xsect_data_temp = self.xsect_data_temp[od_index, :]
            wn_od_data_jac_temp = (wn_xsect_data - wn_xsect_data_temp) * self.gas_density_lay / self._temperature.temperature_offset
            wn_od_data_jac = wn_od_data_jac_temp[:,np.newaxis] * self._temperature.coefficient.jacobian

        # We have the pieces from py_retrieve that gives us dod_dlogvmr
        dod_dlogvmr = np.zeros((nlay, 1, nlay+1))
        dod_dlogvmr[:,0,:-1] = np.diag(self.map_vmr_l[0,:wn_od_data.shape[0]] * wn_od_data)
        dod_dlogvmr[:,0,1:] += np.diag(self.map_vmr_u[0,:wn_od_data.shape[0]] * wn_od_data)
        if(wn_od_data_jac is not None):
            # Note I don't think this is working currently, so skip
            #dod_dlogvmr[:,0,:] += wn_od_data_jac
            pass
        # Map this to relative to state vector
        if(is_constant):
            od_result = rf.ArrayAd_double_2(wn_od_data[:, np.newaxis])
        else:
            dod_dstate = np.matmul(dod_dlogvmr, dlogvmr_dstate)
            od_result = rf.ArrayAd_double_2(wn_od_data[:, np.newaxis], dod_dstate)
        return od_result

    def absorber_vmr(self, gas_name):
        # We can only handle ozone right now
        if gas_name != "O3":
            raise Exception(f"Expected to handle O3 only but was asked to provide a value for {gas_name}")

        return self._absorber_vmr[0]

    def clone(self):
        return MusesOpticalDepthFile(self.rf_uip, self._pressure, self._temperature, self._altitude,
                                     self._absorber_vmr, self.num_channel)

    def print_desc(self, ostream):
        # A bit clumsy, we should perhaps put a better interface in
        # here.
        ostream.write("MusesOpticalDepthFile", len("MusesOpticalDepthFile"))


__all__ = ["MusesOpticalDepthFile", ]

from __future__ import annotations
import refractor.framework as rf  # type: ignore
from .misc import ResultIrk
from functools import cache
import numpy as np
import copy
import math
import typing

if typing.TYPE_CHECKING:
    from .current_state import CurrentState
    from .muses_observation import MusesObservation
    from .retrieval_configuration import RetrievalConfiguration
    from .muses_radiative_transfer_oss import MusesRadiativeTransferOss
    from .pointing_angle_surface import PointingAngleSurface
    from .muses_refractive_index import MusesRefractiveIndex
    from .identifier import InstrumentIdentifier


class IrkForwardModel(rf.StandardForwardModel):
    """This is a rf.StandardForwardModel with the extra code to calculate the
    IRK added."""

    def __init__(
        self,
        rf_uip: rf.RefractorUip,  # Temp, leverage off UIP. We'll remove this in a bit
        instrument: rf.Instrument,
        spec_win: rf.SpectralWindow,
        radiative_transfer: rf.MusesRadiativeTransferOss,
        spectrum_sampling: rf.SpectrumSampling,
        spectrum_effect: list[list[rf.SpectrumEffect]],
        observation: MusesObservation,
        sat_altitude: rf.DoubleWithUnit,
        earth_radius: rf.DoubleWithUnit,
        p: rf.Pressure,
        alt: rf.Altitude,
        rindex: MusesRefractiveIndex,
        rconf: RetrievalConfiguration,
        pntsurf: PointingAngleSurface,
        irk_radiative_transfer: rf.MusesRadiativeTransferOss | None = None,
    ) -> None:
        super().__init__(
            instrument, spec_win, radiative_transfer, spectrum_sampling, spectrum_effect
        )
        self.obs = observation
        self.eradius = earth_radius.convert("m").value
        self.sat_altitude = sat_altitude.convert("m").value
        self.p = p
        self.alt = alt
        self.rindex = rindex
        self.rconf = rconf
        self.pntsurf = pntsurf
        self._irk_radiative_transfer = irk_radiative_transfer
        self.rf_uip = rf_uip

    def makemap_ll(self, pbar : np.ndarray, plevel : np.ndarray) -> np.ndarray:
        o_map = np.zeros(shape=(plevel.shape[0], pbar.shape[0]), dtype=np.float64)
        for ii in range(pbar.shape[0]):
            xdelta_p = math.log(plevel[ii+1]) - math.log(plevel[ii])
            xcoeff = 1. - (math.log(pbar[ii]) - math.log(plevel[ii])) / xdelta_p
            o_map[ii, ii] = xcoeff
            o_map[ii + 1, ii] = 1 - xcoeff

        return o_map

    def atmosphere_level(self, uip):
        from refractor.muses_py import UtilGeneral, ObjectView, earth_radius, compute_altitude_pge
        # IDL_LEGACY_NOTE: This function atmosphere_level is the same as atmosphere_level function in  ELANOR/atmosphere_level.pro file.
        function_name = "atmosphere_level: "

        utilGeneral = UtilGeneral()

        if isinstance(uip, dict):
            uip = ObjectView(uip)

        Rgas = 8.31451
        Avo = 6.0225e23
        kb = 1.380622e-23

        # This procedure reads in the atmosphere and sets up all the level related parameters

        atmosphere = uip.atmosphere
        atmosphere_params_upper = [x.upper() for x in list(uip.atmosphere_params)]

        levels = None
        uu = -1
        if 'LEVEL' in atmosphere_params_upper:
            uu = atmosphere_params_upper.index('LEVEL')

        if uu > -1:
            levels = atmosphere[uu, :]

        pressure = None
        uu = -1
        if 'PRESSURE' in atmosphere_params_upper:
            uu = atmosphere_params_upper.index('PRESSURE')

        if uu > -1:
            pressure = atmosphere[uu, :]
        else:
            print(function_name, "We need pressure!")
            assert False

        tatm = None
        uu = -1
        if 'TATM' in atmosphere_params_upper:
            uu = atmosphere_params_upper.index('TATM')

        if uu > -1:
            tatm = atmosphere[uu, :]
        else:
            print(function_name, "We need TATM!")
            assert False

        h2o = None
        uu = -1
        if 'H2O' in atmosphere_params_upper:
            uu = atmosphere_params_upper.index('H2O')

        if uu > -1:
            h2o = atmosphere[uu, :]
        else:
            print(function_name, "We need H2O!")
            assert False

        h2o = np.asarray(h2o)
        tatm = np.asarray(tatm)
        pressure = np.asarray(pressure)
        uu = np.where((h2o <= 0.0) | np.all(np.isnan(h2o), axis=0) |
                      (tatm <= 0.0) | np.all(np.isnan(tatm), axis=0) |
                      (pressure <= 0.0) | np.all(np.isnan(pressure)))[0]

        if len(uu) > 0:
            print(function_name, "Bad pressure, temperature, or water at level:", uu)
            assert False

        # to use PGE altitude grid using the following:
        # put in correct units

        latitude = uip.obs_table['target_latitude']*180/math.pi

        waterType = None
        pge_flag = True

        # AT_LINE 165 ELANOR/atmosphere_level.pro  
        (results, x) = compute_altitude_pge(pressure, tatm, h2o, 
                                            uip.obs_table['surfaceAltitude'], latitude, waterType, pge_flag)

        altitude = results['altitude']/1000.0
        pge_flag = None
        radiusEarth = earth_radius(latitude, pge_flag)
        density_air = results['airDensity'] * 1e6

        # AT_LINE 175 ELANOR/atmosphere_level.pro
        num_species = len(uip.species)
        vmr_species = np.zeros(shape=(num_species, len(pressure)), dtype=np.float64) # dblarr(num_species,n_elements(pressure))

        # Fill in the selected species and check for zero values.
        # AT_LINE 179 ELANOR/atmosphere_level.pro
        for ii in range(num_species):
            uu = -1
            if uip.species[ii] in uip.atmosphere_params:
                uu = atmosphere_params_upper.index(uip.species[ii])

            if uu > -1:
                vmr_species[ii, :] = atmosphere[uu, :]

            if uu == -1:
                print(function_name, "ERROR: Species " + uip.species[ii] + " not defined in uip.atmosphere")
                assert False

            uu = utilGeneral.WhereEqualIndices(vmr_species[ii], 0.0)
            if len(uu) > 0:
                print(function_name, 'Warning:  ', "Species ", uip.species[ii], " has a bad value: ", vmr_species[ii, uu], " at level:", uu[0])
                print(function_name, 'Allowing code to continue.')

            uu = utilGeneral.WhereGreaterEqualIndices(vmr_species[ii], 1.0)
            if len(uu) > 0:
                print(function_name, "Species ", uip.species[ii], " has a bad value: ", vmr_species[ii, uu], " at level:", uu[0])
                print(function_name, "Please make better")
                assert False
        # end for ii in range(num_species):

        # AT_LINE 206 ELANOR/atmosphere_level.pro

        density_dry = density_air - density_air * h2o

        # We will need a copy of vmr_species because we will be multiplying it by the density_dry.
        density_species = copy.deepcopy(vmr_species)
        for ii in range(0, num_species):
            density_species[ii, :] = vmr_species[ii, :] * density_dry

        if np.all(np.isfinite(density_species) == False):
            print(function_name, "ERROR: Not all values are finite in density_species.")
            assert False

        radius = radiusEarth + altitude * 1000.0

        rayparams = {
            'pressure': pressure,
            'tatm': tatm,
            'h2o': h2o,
            'species': uip.species,
            'vmr': vmr_species,
            'radiusEarth': radiusEarth,
            'density_species': density_species,
            'density_air': density_air,
            'density_air_dry': density_dry,
            'radius': radius, 
            'nlayers': len(pressure)-1}

        return rayparams
    

    def tau_total(
        self, instrument_name: InstrumentIdentifier | str, current_state: CurrentState
    ) -> np.ndarray:
        from refractor.muses_py_fm import mpy_atmosphere_level
        from .misc import AttrDictAdapter

        agrid = (
            self.alt.altitude_grid(self.p, rf.Pressure.DECREASING_PRESSURE)
            .convert("m")
            .value.value
        )
        nlayers = agrid.shape[0] - 1
        
        i_uip = self.rf_uip.uip_all(instrument_name)
        i_uip["obs_table"]["pointing_angle"] = 0.0
        i_uip["cloud"]["extinction"][:] = 1.0
        i_uip = AttrDictAdapter(i_uip)
        i_atmparams = AttrDictAdapter(mpy_atmosphere_level(i_uip))

        # These parameters are needed for the atmospheric equation of state
        pressure = current_state.state_value("pressure")
        tatm = current_state.state_value("TATM")
        density_air = self.atmosphere_level(i_uip)["density_air"]

        radius = i_atmparams.radius

        pbar = np.zeros(shape=(nlayers), dtype=np.float64)
        column = np.zeros(shape=(nlayers), dtype=np.float64)
        path_level = np.zeros(shape=(nlayers + 1), dtype=np.float64)

        ds_fix = 500.0
        x_u = radius[nlayers]
        s_tot = 0.0
        for jj in reversed(range(0, nlayers)):  # go from top to bottom
            hp = -(radius[jj + 1] - radius[jj]) / np.log(
                pressure[jj + 1] / pressure[jj]
            )
            p_u = pressure[jj + 1]
            hd = -(radius[jj + 1] - radius[jj]) / np.log(
                density_air[jj + 1] / density_air[jj]
            )
            den_u = density_air[jj + 1]
            sub_layer = 0
            r_u = radius[jj + 1]
            flag = 0
            while flag == 0:  # sub layer loop
                dr = ds_fix
                # This while loop only exit if the following condition is true.
                if (r_u - dr) < radius[jj]:
                    dr = r_u - radius[jj]
                    flag = 1
                r_l = r_u - dr
                p_l = pressure[jj] * math.exp(-(r_l - radius[jj]) / hp)
                den_l = density_air[jj] * math.exp(-(r_l - radius[jj]) / hd)
                x_l = r_l
                dx = x_u - x_l
                ds = dx
                s_tot = s_tot + ds
                column[jj] = column[jj] + (ds / dr) * (den_l - den_u)
                pbar[jj] = pbar[jj] + (ds / dr) * (den_l * p_l - den_u * p_u)
                r_u = r_l
                x_u = x_l
                p_u = p_l
                den_u = den_l
                sub_layer = sub_layer + 1

            path_level[jj] = s_tot
            pbar[jj] = pbar[jj] * (hp / (hp + hd)) / column[jj]
            column[jj] = column[jj] * hd

        # end for jj in reversed(range(0,nlayers))

        ext = np.full(current_state.state_value("CLOUDEXT").shape, 1.0)
        cloud_pressure = current_state.state_value("PCLOUD")[0]

        scale = current_state.state_value("scalePressure")[0]
        ext_levels = ext[:, np.newaxis] * np.exp(
            -((np.log(pressure[np.newaxis, :]) - np.log(cloud_pressure)) ** 2)
            / scale**2
        )
        map_cloud_ll = self.makemap_ll(pbar, pressure)
        extinction = np.matmul(ext_levels, map_cloud_ll)
        path_norm = (radius[1 : nlayers + 1] - radius[0:nlayers]) / 1000.0
        tau_total = np.sum(extinction * path_norm[np.newaxis, :], axis=1)
        return tau_total

    def dEdOD(self, current_state: CurrentState) -> np.ndarray:
        try:
            t = self.tau_total(self.obs.instrument_name, current_state)
        except KeyError:
            t = self.tau_total("AIRS", current_state)
        #print(t)
        #breakpoint()
        return 1.0 / t

    def irk_angle(self) -> list[float]:
        """List of angles in degrees that run forward model for the IRK."""
        return []

    @property
    def flux_freq_range(self) -> tuple[float, float]:
        # The original run_irk.py code had this hardcoded, but presumably
        # this would change with different instruments? In any case, pull
        # this out to make it clear we are using fixed numbers
        return (980.0, 1080.0)

    @property
    def seg_freq_range(self) -> tuple[float, float]:
        # The original run_irk.py code had this hardcoded, but presumably
        # this would change with different instruments? In any case, pull
        # this out to make it clear we are using fixed numbers
        return (970.0, 1120.0)

    @property
    def irk_average_freq_range(self) -> tuple[float, float]:
        # The original run_irk.py code had this hardcoded, but presumably
        # this would change with different instruments? In any case, pull
        # this out to make it clear we are using fixed numbers
        return (979.99, 1078.999)

    @property
    def irk_weight(self) -> list[float]:
        # The weights to use when creating radianceWeighted and jacWeighted.
        # The original run_irk.py code had this hardcoded, but presumably
        # this would change with different instruments? In any case, pull
        # this out to make it clear we are using fixed numbers.
        # Note that this needs to have the same length as the
        # irk_angle
        return [0, 0.096782, 0.167175, 0.146387, 0.073909, 0.015748]

    @cache
    def irk_spectral_domain(self, spec_index: int) -> rf.SpectralDomain:
        with self.obs.modify_spectral_window(include_bad_sample=True):
            return self.obs.spectral_domain(spec_index)

    @property
    def irk_radiative_transfer(self) -> MusesRadiativeTransferOss:
        # For Airs, we want to use a tes radiative transfer to get
        # the more full frequency range.
        if self._irk_radiative_transfer is not None:
            return self._irk_radiative_transfer
        return self.radiative_transfer

    def irk_radiance(
        self,
        pointing_angle: rf.DoubleWithUnit,
    ) -> rf.Spectrum:
        """Calculate radiance/jacobian for the IRK calculation, for the
        given angle."""
        if self.num_channels != 1:
            raise RuntimeError(
                "We are currently assuming only 1 channel when doing IRK calculation. This could get extended, but we would need to modify the code to do this."
            )
        return self.irk_radiative_transfer.reflectance(
            self.irk_spectral_domain(0),
            0,
            False,
            pointing_angle_surface=self.pntsurf.pointing_angle_surface(pointing_angle),
        )

    def irk(self, current_state: CurrentState) -> ResultIrk:
        """This was originally the run_irk.py code from py-retrieve. We
        have our own copy of this so we can clean this code up a bit.
        """
        t = self.obs.radiance_all_extended(include_bad_sample=True)
        frq_l1b = np.array(t.spectral_domain.data)
        rad_l1b = np.array(t.spectral_range.data)
        radiance = []
        jacobian = []
        frequency: None | np.ndarray = None
        for gi_angle in self.irk_angle():
            r = self.irk_radiance(rf.DoubleWithUnit(gi_angle, "deg"))
            if frequency is None:
                frequency = r.spectral_domain.data
            radiance.append(r.spectral_range.data)
            jacobian.append(r.spectral_range.data_ad.jacobian.transpose())
        assert frequency is not None
        freq_step = frequency[1:] - frequency[:-1]
        freq_step = np.array([freq_step[0], *freq_step])
        n_l1b = len(frq_l1b)

        # need remove missing data in L1b radiance
        ifrq_missing = np.where(rad_l1b == 0.0)
        valid_indices = np.where(rad_l1b != 0.0)[0]  # Ensure 1-D array
        rad_l1b[ifrq_missing] = np.interp(
            ifrq_missing, valid_indices, rad_l1b[valid_indices]
        )

        freq_step_l1b_temp = (frq_l1b[2:] - frq_l1b[0 : n_l1b - 2]) / 2.0
        freq_step_l1b = np.concatenate(
            (
                np.asarray([frq_l1b[1] - frq_l1b[0]]),
                freq_step_l1b_temp,
                np.asarray([frq_l1b[n_l1b - 1] - frq_l1b[n_l1b - 2]]),
            ),
            axis=0,
        )

        radianceWeighted = 2.0 * sum(r * w for r, w in zip(radiance, self.irk_weight))

        radratio = radiance[0] / radianceWeighted
        ifrq = self._find_bin(frequency, frq_l1b)
        radratio = radratio[ifrq]
        ifreq = np.where(
            (frequency >= self.flux_freq_range[0])
            & (frequency <= self.flux_freq_range[1])
        )[0]
        flux = 1e4 * np.pi * np.sum(freq_step[ifreq] * radianceWeighted[ifreq])
        ifreq_l1b = np.where(
            (frq_l1b >= self.flux_freq_range[0]) & (frq_l1b <= self.flux_freq_range[1])
        )[0]
        flux_l1b = (
            1e4
            * np.pi
            * np.sum(
                freq_step_l1b[ifreq_l1b] * rad_l1b[ifreq_l1b] / radratio[ifreq_l1b]
            )
        )
        minn = np.amin(frequency)
        maxx = np.amax(frequency)
        minn, maxx = self.seg_freq_range
        nf = int((maxx - minn) / 3)
        freqSegments: np.ndarray = np.ndarray(shape=(nf), dtype=np.float32)
        freqSegments.fill(
            0
        )  # It is import to start with 0 because not all elements will be calculated.
        fluxSegments: np.ndarray = np.ndarray(shape=(nf), dtype=np.float32)
        fluxSegments.fill(
            0
        )  # It is import to start with 0 because not all elements will be calculated.
        fluxSegments_l1b: np.ndarray = np.ndarray(shape=(nf), dtype=np.float32)
        fluxSegments_l1b.fill(
            0
        )  # It is import to start with 0 because not all elements will be calculated.

        # get split into 3 cm-1 segments
        for ii in range(nf):
            ind = np.where(
                (frequency >= minn + ii * 3) & (frequency < minn + ii * 3 + 3)
            )[0]
            ind_l1b = np.where(
                (frq_l1b >= minn + ii * 3)
                & ((frq_l1b < minn + ii * 3 + 3) & (rad_l1b > 0.0))
            )[0]
            if len(ind_l1b) > 0:
                fluxSegments_l1b[ii] = (
                    1e4
                    * np.pi
                    * np.sum(
                        freq_step_l1b[ind_l1b] * rad_l1b[ind_l1b] / radratio[ind_l1b]
                    )
                )
            if (
                len(ind) > 0
            ):  # We only calculate fluxSegments, fluxSegments_l1b, and freqSegments if there is at least 1 value in ind vector.
                fluxSegments[ii] = (
                    1e4 * np.pi * np.sum(freq_step[ind] * radianceWeighted[ind])
                )
                freqSegments[ii] = np.mean(frequency[ind])

        jacWeighted = 2.0 * sum(jac * w for jac, w in zip(jacobian, self.irk_weight))

        # weight by freq_step
        jacWeighted *= freq_step[np.newaxis, :]

        o_results_irk = ResultIrk(
            {
                "flux": flux,
                "flux_l1b": flux_l1b,
                "fluxSegments": fluxSegments,
                "freqSegments": freqSegments,
                "fluxSegments_l1b": fluxSegments_l1b,
            }
        )

        # smaller range for irk average
        indf = np.where(
            (frequency >= self.irk_average_freq_range[0])
            & (frequency <= self.irk_average_freq_range[1])
        )[0]

        irk_array = 1e4 * np.pi * self.my_total(jacWeighted[:, indf], True)

        minn, maxx = self.flux_freq_range

        nf = int((maxx - minn) / 3)
        irk_segs = np.zeros(shape=(jacWeighted.shape[0], nf), dtype=np.float32)
        freq_segs = np.zeros(shape=(nf), dtype=np.float32)

        for ii in range(nf):
            ind = np.where(
                (frequency >= minn + ii * 3) & (frequency < minn + ii * 3 + 3)
            )[0]
            if (
                len(ind) > 1
            ):  # We only calculate irk_segs and freq_segs if there are more than 1 values in ind vector.
                irk_segs[:, ii] = 1e4 * np.pi * self.my_total(jacWeighted[:, ind], True)
                freq_segs[ii] = np.mean(frequency[ind])
        # end for ii in range(nf):

        # AT_LINE 333 src_ms-2018-12-10/run_irk.pro
        radarr_fm = np.concatenate(radiance, axis=0)
        radInfo = {
            "gi_angle": gi_angle,
            "radarr_fm": radarr_fm,
            "freq_fm": frequency,
            "rad_L1b": rad_l1b,
            "freq_L1b": frq_l1b,
        }
        o_results_irk["freqSegments_irk"] = freq_segs
        o_results_irk["radiances"] = radInfo

        # calculate irk for each type
        for selem_id in current_state.retrieval_state_vector_element_list:
            species_name = str(selem_id)
            pstart, plen = current_state.fm_sv_loc[selem_id]
            ii = pstart
            jj = pstart + plen
            vmr = current_state.initial_guess_full[ii:jj]
            vmr = (
                current_state.state_mapping(selem_id, include_subset=False)
                .mapped_state(rf.ArrayAd_double_1(vmr))
                .value
            )
            pressure = current_state.pressure_list_fm(selem_id)

            myirfk = copy.deepcopy(irk_array[ii:jj])
            myirfk_segs = copy.deepcopy(irk_segs[ii:jj, :])

            # TODO This looks like the sort of thing that can be
            # replaced with our StateElement data, to get away from
            # having all this hard coded. But for now, leave this like
            # it was

            # convert cloudext to cloudod
            # dL/dod = dL/dext * dext/dod
            if species_name == "CLOUDEXT":
                dEdOD = self.dEdOD(current_state)
                myirfk = np.multiply(myirfk, dEdOD)
                for pp in range(dEdOD.shape[0]):
                    myirfk_segs[pp, :] = myirfk_segs[pp, :] * dEdOD[pp]

                species_name = "CLOUDOD"
                vmr = np.divide(vmr, dEdOD)

            mm = jj - ii
            if species_name == "TATM" or species_name == "TSUR":
                mylirfk = np.multiply(myirfk, vmr)
                mylirfk_segs = copy.deepcopy(myirfk_segs)
                for kk in range(mm):
                    mylirfk_segs[kk, :] = mylirfk_segs[kk, :] * vmr[kk]
            else:
                mylirfk = copy.deepcopy(myirfk)
                myirfk = np.divide(myirfk, vmr)
                mylirfk_segs = copy.deepcopy(myirfk_segs)
                for kk in range(mm):
                    myirfk_segs[kk, :] = myirfk_segs[kk, :] / vmr[kk]

            if species_name == "O3":
                mult_factor = 1.0 / 1e9  # W/m2/ppb
                unit = "W/m2/ppb"
            elif species_name == "H2O":
                mult_factor = 1.0 / 1e6  # W/m2/ppm
                unit = "W/m2/ppm"
            elif species_name == "TATM":
                mult_factor = 1.0
                unit = "W/m2/K"
            elif species_name == "TSUR":
                mult_factor = 1.0
                unit = "W/m2/K"
            elif species_name == "EMIS":
                mult_factor = 1.0
                unit = "W/m2"
            elif species_name == "CLOUDDOD":
                mult_factor = 1.0
                unit = "W/m2"
            elif species_name == "PCLOUD":
                mult_factor = 1.0
                unit = "W/m2/hPa"
            else:
                # Fall back
                mult_factor = 1.0
                unit = " "

            myirfk = np.multiply(myirfk, mult_factor)
            myirfk_segs = np.multiply(myirfk_segs, mult_factor)

            # subset only freqs in range
            if species_name == "CLOUDOD":
                myirfk_segs = myirfk_segs[:, 0]
                myirfk_segs = np.reshape(myirfk_segs, (myirfk_segs.shape[0]))

                mylirfk_segs = mylirfk_segs[:, 0]
                mylirfk_segs = np.reshape(mylirfk_segs, (mylirfk_segs.shape[0]))

            vmr = np.divide(vmr, mult_factor)

            # Build a structure of result for each species_name.
            result_per_species = {
                "irfk": myirfk,
                "lirfk": mylirfk,
                "pressure": pressure,
                "unit": unit,
                "irfk_segs": myirfk_segs,
                "lirfk_segs": mylirfk_segs,
                "vmr": vmr,
            }

            # Add the result for each species_name to our structure to return.
            # Note that the name of the species is the key for the dictionary structure.

            o_results_irk[species_name] = copy.deepcopy(
                result_per_species
            )  # o_results_irk
        # end for ispecies in range(len(jacobian_speciesIn)):
        return o_results_irk

    def my_total(self, matrix_in: np.ndarray, ave_index: bool = False) -> np.ndarray:
        size_out = matrix_in.shape[0] if ave_index else matrix_in.shape[1]
        arrayOut = np.ndarray(shape=(size_out,), dtype=np.float64)
        for ii in range(size_out):
            my_vector = matrix_in[ii, :] if ave_index else matrix_in[:, ii]
            # Filter our -999 values
            val = np.sum(my_vector[np.abs(my_vector - (-999)) > 0.1])
            arrayOut[ii] = val
        return arrayOut

    def _find_bin(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        # IDL_LEGACY_NOTE: This function _find_bin is the same as findbin in run_irk.pro file.
        #
        # Returns the bin numbers for nearest value of x array to values of y
        #       returns nearest bin for values outside the range of x
        #
        ny = len(y)

        o_bin: np.ndarray = np.ndarray(shape=(ny), dtype=np.int32)
        for iy in range(0, ny):
            ix = np.argmin(abs(x - y[iy]))
            o_bin[iy] = ix

        if ny == 1:
            o_bin = np.asarray([o_bin[0]])

        return o_bin


__all__ = [
    "IrkForwardModel",
]

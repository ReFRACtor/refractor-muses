from __future__ import annotations
import refractor.framework as rf  # type: ignore
from .misc import ResultIrk
from functools import cache
import numpy as np
import copy
import typing

if typing.TYPE_CHECKING:
    from .current_state import CurrentState
    from .muses_observation import MusesObservation
    from .retrieval_configuration import RetrievalConfiguration
    from .muses_radiative_transfer_oss import MusesRadiativeTransferOss
    from .pointing_angle_surface import PointingAngleSurface


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
        rconf: RetrievalConfiguration,
        pntsurf: PointingAngleSurface,
        irk_radiative_transfer: rf.MusesRadiativeTransferOss | None = None,
    ) -> None:
        super().__init__(
            instrument, spec_win, radiative_transfer, spectrum_sampling, spectrum_effect
        )
        self.obs = observation
        self.rconf = rconf
        self.pntsurf = pntsurf
        self._irk_radiative_transfer = irk_radiative_transfer
        self.rf_uip = rf_uip

    def tau_total(self, instrument_name: str, current_state: CurrentState) -> np.ndarray:
        from refractor.muses_py_fm import mpy_atmosphere_level
        from refractor.muses_py import idl_tag_names, makemap_ll, ref_index
        from .misc import AttrDictAdapter
        import copy
        import math
        
        i_uip = self.rf_uip.uip_all(instrument_name)
        i_uip["obs_table"]["pointing_angle"] = 0.0
        i_uip["cloud"]["extinction"][:] = 1.0
        i_uip = AttrDictAdapter(i_uip)
        i_atmparams = AttrDictAdapter(mpy_atmosphere_level(i_uip))


        # These parameters are needed for the atmospheric equation of state
        pressure = i_atmparams.pressure
        lnp = np.log(pressure)
        temperature = i_atmparams.tatm
        h2o = i_atmparams.h2o
        
        density_species = np.concatenate(
            ((i_atmparams.density_air_dry.reshape(1, len(i_atmparams.density_air_dry))), i_atmparams.density_species), 
            axis=0
        )

        density_air = i_atmparams.density_air

        num_species = len(density_species) #also accounts for dry air column from the concatenate() function above.

        # AT_LINE 28 ELANOR/raylayer_nadir.pro
        radius = i_atmparams.radius
        nlayers = i_atmparams.nlayers

        # AT_LINE 33 ELANOR/raylayer_nadir.pro

        psi_level = np.zeros(shape=(nlayers + 1), dtype=np.float64)
        tbar = np.zeros(shape=(nlayers), dtype=np.float64)
        tbar_try = np.zeros(shape=(nlayers), dtype=np.float64)
        pbar = np.zeros(shape=(nlayers), dtype=np.float64)
        column = np.zeros(shape=(nlayers), dtype=np.float64)
        path_level = np.zeros(shape=(nlayers + 1), dtype=np.float64)

        # AT_LINE 42 ELANOR/raylayer_nadir.pro
        ds_fix = 500.0

        if 'SUB_LAYER_DIST' in idl_tag_names(i_uip):
            ds_fix = i_uip.SUB_LAYER_DIST # in meteres.

        column_species = np.zeros(shape=(num_species, nlayers), dtype=np.float64)
        map_vmr_u = np.zeros(shape=(num_species, nlayers), dtype=np.float64)
        map_vmr_l = np.zeros(shape=(num_species, nlayers), dtype=np.float64)
        map_tatm_u = np.zeros(shape=(nlayers), dtype=np.float64)
        map_tatm_l = np.zeros(shape=(nlayers), dtype=np.float64)

        # AT_LINE 54 src_ms-2018-12-10/ELANOR/raylayer_nadir.pro
        density_species_l = np.zeros(shape=(num_species), dtype=np.float64)

        psi_tot = np.float64(0.0)


        # PYTHON_NOTE: There is only one obs_table so we cannot use the index.
        radiusSat = i_uip.obs_table['sat_radius']  # There is only one obs_table so we cannot use the index.
        
        sin_theta_u = radiusSat * math.sin(i_uip.obs_table['pointing_angle']) / radius[nlayers]

        snells_constant = radius[nlayers] * sin_theta_u

        cos_theta_u = math.sqrt(1.0 - sin_theta_u**2)

        x_u = radius[nlayers] * cos_theta_u

        psi_tot = 0
        s_tot = 0.0

        # get indices of linear VMR's

        indsLinear = np.zeros(shape=(len(i_uip.species) +  1), dtype=int)
        indsLog = np.ndarray(shape=(len(i_uip.species) + 1), dtype=int)
        indsLog.fill(1)    # Fill with 1's in all elements.

        # AT_LINE 99 ELANOR/raylayer_nadir.pro
        for jj in range(len(i_uip.jacobiansLinear)):
            if i_uip.jacobiansLinear[jj] != '':
                if i_uip.jacobiansLinear[jj] in i_uip.species:
                    indlinear = i_uip.species.index(i_uip.jacobiansLinear[jj])
                else:
                    # PYTHON_NOTE: Because the where statement in IDL can return -1 as the first element if no match, we will try to mimic
                    #              IDL by setting indlinear to -1 so it be used as an index.
                    indlinear = -1

                indsLinear[indlinear + 1] = 1
                indsLog[indlinear + 1] = 0

        indsLinear = np.where(indsLinear == 1)[0]
        indsLog = np.where(indsLog == 1)[0]

        for jj in reversed(range(0, nlayers)): # go from top to bottom
            nupper = ref_index(temperature[jj+1], pressure[jj+1] * 100., h2o[jj+1])

            n_u = nupper # for the sub-layers

            hp = -(radius[jj+1] - radius[jj]) / np.log(pressure[jj+1] / pressure[jj])
            p_u = pressure[jj+1]

            hd = -(radius[jj+1] - radius[jj]) / np.log(density_air[jj+1] / density_air[jj])
            den_u = density_air[jj+1]

            t_u = temperature[jj+1]

            density_species_u = density_species[:, jj+1]
            hd_species = -(radius[jj+1] - radius[jj]) / np.log(density_species[:, jj+1] / density_species[:, jj])

            sub_layer = 0
            r_u = radius[jj+1]
            deltar = radius[jj+1] - radius[jj]

            flag = 0

            while flag == 0: # sub layer loop
                dr = ds_fix * cos_theta_u

                # This while loop only exit if the following condition is true.
                if (r_u - dr) < radius[jj]:
                    dr = r_u - radius[jj]
                    flag = 1

                r_l = r_u - dr

                p_l = pressure[jj] * math.exp(-(r_l - radius[jj]) / hp)

                t_l = temperature[jj] + (r_l - radius[jj]) * (temperature[jj+1] - temperature[jj]) / (radius[jj+1] - radius[jj])
                den_l = density_air[jj] * math.exp(-(r_l - radius[jj]) / hd)
                h2o_l = h2o[jj] + (np.log(p_l) - lnp[jj]) * (h2o[jj+1] - h2o[jj]) / np.log(pressure[jj+1] / pressure[jj])

                n_l = ref_index(t_l, p_l * 100., h2o_l)

                dn_dr = (n_u - n_l) / (r_u - r_l)

                sin_theta_l = snells_constant / r_l / n_l
                cos_theta_l = math.sqrt(1 - sin_theta_l**2)
                x_l = r_l * cos_theta_l


                dx = x_u - x_l

                ds_dx_l = 1. / (1 + r_l * sin_theta_l**2 / (n_l / dn_dr))
                ds_dx_u = 1. / (1 + r_u * sin_theta_u**2 / (n_u / dn_dr))

                ds = .5 * (ds_dx_u + ds_dx_l) * dx

                dpsi = snells_constant * .5 * (ds_dx_l / n_l / r_l**2 + ds_dx_u / n_u / r_u**2) * dx

                s_tot = s_tot + ds
                psi_tot = psi_tot + dpsi

                ds_dr = ds / dr

                column[jj] = column[jj] + (ds / dr) * (den_l - den_u)

                tbar[jj] = tbar[jj] + (ds/dr) * (den_l * t_l - den_u * t_u)
                pbar[jj] = pbar[jj] + (ds/dr) * (den_l * p_l - den_u * p_u)

                dum1 = (radius[jj+1] - r_l) / deltar
                dum2 = (radius[jj+1] - r_u) / deltar
                dum3 = (r_l - radius[jj]) / deltar
                dum4 = (r_u - radius[jj]) / deltar

                map_tatm_l[jj] = map_tatm_l[jj] + ds_dr * (den_l * dum1 - den_u * dum2)

                map_tatm_u[jj] = map_tatm_u[jj] + ds_dr * (den_l * dum3 - den_u * dum4)

                if indsLog[0] >= 0:
                    density_species_l[indsLog] = density_species[indsLog, jj] * np.exp(-(r_l-radius[jj]) / hd_species[indsLog])
                    column_species[indsLog, jj] = column_species[indsLog, jj] + ds / dr * (density_species_l[indsLog] - density_species_u[indsLog])
                    map_vmr_l[indsLog, jj] = map_vmr_l[indsLog, jj] + ds_dr * (density_species_l[indsLog] * dum1 - density_species_u[indsLog] * dum2)
                    map_vmr_u[indsLog, jj] = map_vmr_u[indsLog, jj] + ds_dr * (density_species_l[indsLog] * dum3 - density_species_u[indsLog] * dum4)
                # end if indsLog[0] >= 0:

                if len(indsLinear) > 0 and indsLinear[0] >= 0:
                    density_species_l[indsLinear] = density_species[indsLinear, jj] + \
                        (density_species[indsLinear, jj+1] - density_species[indsLinear, jj]) * (r_l - radius[jj]) / (radius[jj+1]-radius[jj])

                    dens_u = density_species_u[indsLinear]
                    dens_l = density_species_l[indsLinear]

                    column_species[indsLinear, jj] = column_species[indsLinear, jj] + \
                        hd * ds / dr * (dens_l - dens_u - hd * (dens_u / den_u-dens_l / den_l) * (den_u - den_l) / dr)


                    map_vmr_l[indsLinear, jj] = map_vmr_l[indsLinear, jj] + ds_dr * (den_l * dum1 - den_u * dum2)
                    map_vmr_u[indsLinear, jj] = map_vmr_u[indsLinear, jj] + ds_dr * (den_l * dum3 - den_u * dum4)
                # end if indsLinear[0] >= 0:

                cos_theta_u = cos_theta_l
                sin_theta_u = sin_theta_l
                r_u = r_l
                x_u = x_l
                n_u = n_l

                t_u = t_l
                p_u = p_l
                den_u = den_l
                density_species_u = copy.deepcopy(density_species_l)  # PYTHON_NOTE: We must make a new memory for density_species_u, otherwise both points to the smae address space.
                sub_layer = sub_layer + 1
            # end while (flag == 0): # sub layer loop

            # AT_LINE 288 ELANOR/raylayer_nadir.pro
            psi_level[jj] = psi_tot
            path_level[jj] = s_tot
            tbar[jj] = tbar[jj] / column[jj] + hd * (temperature[jj+1] - temperature[jj]) / (radius[jj+1] - radius[jj])
            pbar[jj] = pbar[jj] * (hp / (hp + hd)) / column[jj]
            tbar_try[jj] = tbar_try[jj]/column[jj]

            column[jj] = column[jj] * hd
            column_species[indsLog, jj] = column_species[indsLog, jj] * hd_species[indsLog]

            map_vmr_l[indsLog, jj] = hd_species[indsLog] * ((-1. / deltar) + map_vmr_l[indsLog, jj] / column_species[indsLog, jj])
            map_vmr_u[indsLog, jj] = hd_species[indsLog] * ((1. / deltar) + map_vmr_u[indsLog, jj] / column_species[indsLog, jj])

            map_tatm_l[jj] = (tbar[jj] / temperature[jj]) * hd * ((-1. / deltar) + map_tatm_l[jj] / column[jj])
            map_tatm_u[jj] = (tbar[jj] / temperature[jj+1]) * hd * ((1. / deltar) + map_tatm_u[jj] / column[jj])

            if len(indsLinear) > 0 and indsLinear[0] >= 0:
                map_vmr_l[indsLinear, jj] = (column_species[indsLinear, jj] / column[jj]) / (density_species[indsLinear, jj] / density_air[jj]) \
                                            * hd * ((-1. / deltar) + map_vmr_l[indsLinear, jj] / column[jj])

                map_vmr_u[indsLinear, jj] = (column_species[indsLinear, jj] / column[jj]) / (density_species[indsLinear, jj] / density_air[jj]) \
                                            * hd * ((1. / deltar) + map_vmr_u[indsLinear, jj] / column[jj])
            # end if indsLinear[0] >= 0:
        # end for jj in reversed(range(0,nlayers)) 

        ext = np.full(current_state.state_value("CLOUDEXT").shape, 1.0)
        cloud_pressure = i_uip.cloud['pressure']

        scale = current_state.state_value("scalePressure")[0]
        ext_levels = ext[:,np.newaxis] * np.exp(-(np.log(pressure[np.newaxis,:]) - np.log(cloud_pressure))**2 / scale**2)
        pressure_clevel = pressure
        pressure_clayer = pbar

        map_cloud_ll = makemap_ll(pressure_clayer, pressure_clevel)
        extinction = np.matmul(ext_levels, map_cloud_ll)
        path_norm = (radius[1:nlayers+1] - radius[0:nlayers]) / 1000.
        tau_total = np.sum(extinction * path_norm[np.newaxis,:], axis=1)
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

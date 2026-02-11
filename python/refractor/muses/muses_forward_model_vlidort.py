from __future__ import annotations
import refractor.framework as rf  # type: ignore
from .identifier import InstrumentIdentifier
from .forward_model_handle import ForwardModelHandle, ForwardModelHandleSet
from functools import cached_property
from loguru import logger
import tempfile
import numpy as np
import copy
import subprocess
from pathlib import Path
from typing import Any
import typing

if typing.TYPE_CHECKING:
    from .current_state import CurrentState
    from .muses_observation import MeasurementId
    from .retrieval_configuration import RetrievalConfiguration
    from .muses_observation import MusesObservation
    from .cost_function import CostFunction
    from .refractor_fm_object_creator import RefractorFmObjectCreator
    from refractor.muses_py_fm import RefractorUip

# This is a work in progress. We would like to move over and simplify the vlidort
# forward model, and hopefully remove using the UIP etc. But for right now, we
# leverage off of muses-py
#
# Note that this has direct copied of stuff from muses_py_fm/muses_forward_model.py,
# since we want to independent update stuff. This is obviously not desirable long
# term.


class FmUpdateUip(rf.ObserverMaxAPosterioriSqrtConstraint):
    def __init__(self, fm: MusesForwardModelVlidort) -> None:
        super().__init__()
        self.fm = fm

    def notify_update(self, mstand: rf.MaxAPosterioriSqrtConstraint) -> None:
        logger.debug(f"Call to {self.__class__.__name__}::notify_update")
        self.fm.update_uip(mstand.parameters)


class MusesForwardModelVlidort(rf.ForwardModel):
    """Forward model that uses VLIDORT. This matches the existing
    py-retrieve. Note that the version of VLIDORT is older than the
    LIDORT version we use. Also, the code is hardcoded to using O3
    only as an absorber. We could modify this, but it would require
    changing the package muses-vlidort. Also we use a different
    version of the Raman Scattering calculation (ring). This uses
    the stand alone executable (from the the package muses-rrs).
    """

    def __init__(
        self,
        ocreator: RefractorFmObjectCreator,
        current_state: CurrentState,
        instrument_name: InstrumentIdentifier,
        obs: MusesObservation,
        rconf: RetrievalConfiguration,
        vlidort_nstokes: int = 2,
        vlidort_nstreams: int = 4,
        use_vlidort_temp_dir: bool = True,
        **kwargs: Any,
    ) -> None:
        """vlidort_tempdir can be passed in. This should be the same as what
        was used in RefractorUip when we pass in the vlidort_dir. We don't
        actually do anything with vlidort_tempdir, just maintain the lifetime so
        that as long as this MusesForwardModel exists we still have the tempdir.
        When the forward model gets deleted, the temporary directory gets removed.

        Note the directory is under 1MB usually, so you don't need to be too concerned
        about where this goes. You can just use the normal mkdtemp() logic used
        by tempfile.TemporaryDirectory.
        """
        super().__init__()
        self.instrument_name = instrument_name
        self.vlidort_nstreams = vlidort_nstreams
        self.vlidort_nstokes = vlidort_nstokes
        # TODO We'll pull out the objects we need here, but for now just
        # grab the whole RefractorFmObjectCreator
        self._vlidort_tempdir: tempfile.TemporaryDirectory | None = None
        self.use_vlidort_temp_dir = use_vlidort_temp_dir
        self.ocreator = ocreator
        # Temp, awkward interface
        self.ocreator.vlidort_tempdir=self.vlidort_tempdir
        self.ray_info = self.ocreator.ray_info
        self.ground = self.ocreator.ground
        self.absorber = self.ocreator.absorber
        self.cloud_fraction = self.ocreator.cloud_fraction
        self.pressure = self.ocreator.pressure
        self.temperature = self.ocreator.temperature
        self.altitude = self.ocreator.altitude
        self.nchan = self.ocreator.num_channels
        self._do_cloud = False

        # We save the current_state value, since it might have changed
        # when we create the UIP. The semantics here is that we create
        # the UIP when we create the forward model, however we actually
        # delay that until we create it on first use. However we want to
        # create the UIP that we *would have* if we had created it now.
        #
        # Note for an actual retrieval, there is no reason to delay creating
        # the UIP now. Instead, we have unit tests that regularly set things
        # up but don't actually run the forward model. We don't want to pay
        # the time penalty of creating the UIP and/or require muses-py be
        # available. So to support that, we have a delayed create on first
        # use of the UIP.
        self.current_state = copy.deepcopy(current_state)
        self.obs = obs
        self.kwargs = kwargs
        self.rconf = rconf
        self.have_create_uip = False
        self.uip_params: None | np.ndarray = None

    @property
    def do_cloud(self) -> bool:
        return self._do_cloud

    @do_cloud.setter
    def do_cloud(self, v:bool) -> None:
        self._do_cloud = v
        self.ground.do_cloud = v
        self.pressure.do_cloud = v

    def update_uip(self, parameters: np.ndarray) -> None:
        if not self.have_create_uip:
            # Delay setting the UIP value until we actually create it. We don't
            # want to create this now just to set the value
            self.uip_params = parameters.copy()
        else:
            if self.rf_uip.basis_matrix is not None:
                self.rf_uip.update_uip(parameters)

    @property
    def vlidort_tempdir(self) -> None | str:
        if self._vlidort_tempdir is not None:
            return self._vlidort_tempdir.name
        if self.use_vlidort_temp_dir:
            self._vlidort_tempdir = tempfile.TemporaryDirectory()
            return self._vlidort_tempdir.name
        return None

    @cached_property
    def rf_uip(self) -> RefractorUip:
        """Create on on first use."""
        from refractor.muses_py_fm import RefractorUip

        res = RefractorUip.create_uip_from_refractor_objects(
            [
                self.obs,
            ],
            self.current_state,
            self.rconf,
            vlidort_dir=self.vlidort_tempdir
        )
        self.have_create_uip = True
        # Set any delayed parameters update
        if self.uip_params is not None and res.basis_matrix is not None:
            res.update_uip(self.uip_params)
        return res

    def bad_sample_mask(self, sensor_index: int) -> np.ndarray:
        bmask = self.obs.bad_sample_mask(sensor_index)
        if self.obs.spectral_window.include_bad_sample:
            bmask[:] = False
        # This is the full bad sample mask, for all the indices. But here we only
        # want the portion that fits in the spectral window
        with self.obs.modify_spectral_window(include_bad_sample=True):
            sd = self.obs.spectral_domain_full(sensor_index)
            gindex = self.obs.spectral_window.grid_indexes(sd, sensor_index)
        return bmask[list(gindex)]

    def setup_grid(self) -> None:
        # Nothing that we need to do for this
        pass

    def _v_num_channels(self) -> int:
        return self.nchan

    def spectral_domain(self, sensor_index: int) -> rf.SpectralDomain:
        return self.obs.spectral_domain(sensor_index)

    def notify_cost_function(self, cfunc: CostFunction) -> None:
        cfunc.max_a_posteriori.add_observer_and_keep_reference(FmUpdateUip(self))

    def radiance(self, sensor_index: int, skip_jacobian: bool = False) -> rf.Spectrum:
        self.sensor_index = sensor_index
        # Special handling for empty spectrum. We may get this handled in rtf, but for
        # now handle this.
        # ii_mw only counts nonempty spectral domain, so it value isn't always sensor_index
        self.ii_mw = 0
        for i in range(self.sensor_index):
            if self.spectral_domain(i).data.shape[0] > 0:
                self.ii_mw += 1
        
        # TODO Get logic in for skipping bad pixels. Right now we generate these,
        # and then throw away in radiance()
        
        self.i_uip = self.rf_uip.uip_all(self.instrument_name)
        self.ray_info2 = self.rf_uip.ray_info(self.instrument_name)

        self.do_cloud = False
        self.nlayers = self.pressure.number_layer
        self.do_cloud = True
        self.nlayers_cloud = self.pressure.number_layer
        self.do_cloud = False

        # This is used in a few places, so grab this once here for use later
        # This is describes in "On the generation of atmospheric property
        # Jacobians form the (V)LIDORT linearized radiative transfer models"
        # Rob Spurr, Matt Christi, Journal of Quantitative Spectroscopy and
        # Radiative Transfer, July 2024 pages 109-115
        # https://doi.org/10.1016/j.jqsrt.2014.03.011
        self.layer_to_levels = np.zeros((self.nlayers, self.nlayers + 1))
        # TODO we may want to put this into a function, and/or get this from
        # somewhere other than ray_info

        map_vmr_l, map_vmr_u = self.ray_info2["map_vmr_l"][::-1], self.ray_info2["map_vmr_u"]
        # map_vmr_l and map_vmr_u is nspecies x nlayers in size. We
        # only have a single O3 species, so we just grab the first one
        self.layer_to_levels[:, :-1] = np.diag(
               map_vmr_l[0, :]
        )
        self.layer_to_levels[:, 1:] += np.diag(
                map_vmr_u[0, :]
        )
        vgrid = self.absorber.absorber_vmr("O3").vmr_grid(
                self.pressure, rf.Pressure.DECREASING_PRESSURE
        )
        if not vgrid.is_constant:
            dvmr_dstate = vgrid.jacobian
            dlogvmr_dvmr = np.diag(1 / vgrid.value)
            self.dlogvmr_dstate = dlogvmr_dvmr @ dvmr_dstate
        else:
            self.dlogvmr_dstate = None
        
        logger.info("Calling rtf for clear sky")
        radclear = self.rtf(do_cloud=0)

        logger.info("Calling rtf for cloudy sky")
        radcloud = self.rtf(do_cloud=1)

        cfrac = self.cloud_fraction.cloud_fraction
        # Can't directly multiple a ArrayAd_double_1 (this is a tradeoff at the C++
        # level between flexibility and simpler arrangement). But we can just calculate
        # this element by element
        rad = rf.ArrayAd_double_1(radclear.rows, max(radclear.number_variable, radcloud.number_variable, cfrac.number_variable))
        for i in range(radclear.rows):
            rad[i] = radclear[i] * (1-cfrac) + radcloud[i] * cfrac
        # Sanity Check on NAN for radiance and jacobian.
        if not np.all(np.isfinite(rad.value)):
            raise RuntimeError("rad_t not finite")
        if not rad.is_constant is not None and not np.all(np.isfinite(rad.jacobian)):
            raise RuntimeError("jac_t not finite")

        # TODO Get gmask applied when we run VLIDORT.
        gmask = self.bad_sample_mask(sensor_index) != True
        if not rad.is_constant:
            a = rf.ArrayAd_double_1(rad.value[gmask], rad.jacobian[gmask, :])
        else:
            a = rf.ArrayAd_double_1(rad.value[gmask])
        sr = rf.SpectralRange(a, rf.Unit("sr^-1"))
        sd = self.spectral_domain(sensor_index)
        return rf.Spectrum(sd, sr)

    def rtf(
        self,
        do_cloud,
    ):
        self.do_cloud = True if do_cloud == 1 else False
        # Default run directory if not specified.
        if self.vlidort_tempdir is not None:
            vlidort_input_dir = Path(self.vlidort_tempdir)
        else:
            vlidort_input_dir = self.rconf["run_dir"]
        vlidort_input_iter_dir = vlidort_input_dir / "IterLast" / "MWLast" / "cloudy"
        vlidort_input_iter_dir.mkdir(parents=True, exist_ok=True)
        vlidort_output_iter_dir = vlidort_input_dir / "IterLast" / "MWLast" / "cloudy"
        vlidort_output_iter_dir.mkdir(parents=True, exist_ok=True)

        # Temp, we will get nfreq more directly when we set up writing O3Xsec_MW
        od_data = np.loadtxt(
                    vlidort_input_dir / "input" / f"O3Xsec_MW{self.ii_mw+1:03}.asc", skiprows=1
                )
        freq = od_data[:,1]
        wn = rf.ArrayWithUnit_double_1(od_data[:, 1], "nm").convert_wave("cm^-1").value
        nfreq = len(freq)
        fm_nlayers = self.nlayers_cloud if do_cloud else self.nlayers
        # Write the control file for vlidort
        with open(vlidort_input_iter_dir / "config_rtm.asc", "w") as fh:
            print("'atm_lay.asc'", file=fh)
            print("'atm_lev.asc'", file=fh)
            print(f"'../../../input/O3Xsec_MW{self.ii_mw+1:03d}.asc'", file=fh)
            print("'surf_alb.asc'",file=fh)
            print("'vga.asc'",file=fh)
            print(f"{nfreq:>5d}", file=fh)
            print(f"{fm_nlayers:>5d}", file=fh)

        # Write the Vga file
        elv = self.obs.surface_height[self.sensor_index]
        lat = self.obs.latitude[self.sensor_index]
        sza = self.obs.solar_zenith[self.sensor_index]
        # Clamp zen to be in range
        if sza <= 0.0:
            sza = 0.0001
        if sza >= 90.0:
            sza = 89.98
        vza = self.obs.observation_zenith[self.sensor_index]
        raz = self.obs.relative_azimuth[self.sensor_index]
        with open(vlidort_input_iter_dir / "vga.asc", "w") as fh:
            print("ELEV,       DLAT,       SZA,        VZA,        RAZ", file=fh)
            print(f"{elv:12.4f} {lat:12.4f} {sza:12.4f} {vza:12.4f} {raz:12.4f}", file=fh)
            
        # Write surface albedo
        with open(vlidort_input_iter_dir / "surf_alb.asc", "w") as fh:
            print(nfreq,file=fh)
            for wnv,freqv in zip(wn,freq):
                alb = self.ground.surface_parameter(wnv, self.sensor_index).value[0]
                print(f"{freqv:16.7f} {alb:16.7f}", file=fh)

        # Write atmosphere levels
        pgrid = self.pressure.pressure_grid(rf.Pressure.INCREASING_PRESSURE)
        plev = pgrid.convert("hPa").value.value
        tlev = self.temperature.temperature_grid(self.pressure, rf.Pressure.INCREASING_PRESSURE).value.value
        hlev = [self.altitude[0].altitude(p).convert("m").value.value for p in pgrid]
        with open(vlidort_input_iter_dir / "atm_lev.asc", "w") as fh:
            print(len(plev), file=fh)
            print("Table Columns: Pres(mb), T(K), Altitude(m) (TOA to Surf each level)", file=fh)
            for p,t,h in zip(plev, tlev, hlev):
                print(f"{p:16.8f} {t:16.5f} {h:16.5f}", file=fh)
        # Write atmosphere layers
        # TODO, pull this out of rayinfo
        pbar = self.ray_info2["pbar"][::-1]
        tbar = self.ray_info2["tbar"][::-1]
        o3 = self.ray_info2["column_species"][np.array(self.ray_info2["level_params"]["species"]) == "O3",:][0,::-1]
        with open(vlidort_input_iter_dir / "atm_lay.asc", "w") as fh:
            print(self.pressure.number_layer, file=fh)
            print("Table Columns: Pres(mb), T(K), Column Density (molec/cm2)", file=fh)
            for i in range(self.pressure.number_layer):
                print(f"{pbar[i]:16.6f} {tbar[i]:16.7f} {o3[i]:16.7e}", file=fh)
                
        # Run VLIDORT CLI
        vlidort_command = [
            "vlidort_cli",
            '--input', f"{vlidort_input_iter_dir}/",
            '--output', f"{vlidort_output_iter_dir}/",
            '--nstokes', f'{self.vlidort_nstokes}',
            '--nstreams', f'{self.vlidort_nstreams}'
        ]
        logger.debug(f'\nRunning:\n{" ".join(vlidort_command)} ')
        subprocess.run(
            vlidort_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True,
        )
        
        # IWF = G * dI / dG, where I is a component of the stokes vector (I, Q, U, V) and G is the gas optical depth (O3 in our case)
        # IWF also known as the normalized weighting function
        # The denormalized IWF: IWF_denorm = IWF / G

        # read result files from the RT model
        radiance_matrix = np.loadtxt(vlidort_output_iter_dir / "Radiance.asc", skiprows=1)

        # Use the normalized weighting function as provided by VLIDORT
        # MUSES needs the normalized weighting function for species
        # retrieved in log(VMR)
        jacobian_o3_matrix = np.loadtxt(vlidort_output_iter_dir / "IWF.asc",skiprows=1)

        # To experiment with the denormalized weighting function,
        # i.e. if you retrieve O3 in VMR, uncomment the line below
        # jacobian_o3_matrix =
        # read_rtm_output(f"{vlidort_output_iter_dir}/", 'IWF_denorm.asc')

        jacobian_sf_matrix = np.loadtxt(vlidort_output_iter_dir / "surf_WF.asc",skiprows=1)
        
        # Note I don't think the ILS actually works for py-retrieve. We've
        # removed the code here, since we would need to test this. And in
        # any case, I think we would want to just use refractor objects for this
        if self.ocreator.ils_method(self.sensor_index) != "APPLY":
            raise RuntimeError("We don't currently support using and ILS, we need a test case to work through the logic here")

        temp_freq_fm = radiance_matrix[:, 0]
        mwin = self.ocreator.spec_win.muses_microwindows()
        temp_freq_ind = np.where(
            (temp_freq_fm >= mwin[self.ii_mw]["start"])
            & (temp_freq_fm <= mwin[self.ii_mw]["endd"])
        )[0]
        
        # Translate jacobian_sf_matrix to a jacobian relative to the state vector
        
        # TODO We should just use the spectral domain for this forward model,
        # but for now grab what vlidort has
        
        # Low level surface_parameter works with wave number only, so convert
        t = jacobian_sf_matrix[temp_freq_ind, 1:]
        if self.ground.surface_parameter(wn[0], self.sensor_index).rows != 1:
            raise RuntimeError("MusesForwardModelVlidort is hard coded to using an albedo only")
        if not self.ground.surface_parameter(wn[0], self.sensor_index).is_constant:
            jac_sf = np.concatenate([t[i,:][np.newaxis,:] @ self.ground.surface_parameter(wnv, self.sensor_index).jacobian for i, wnv in enumerate(wn[temp_freq_ind])])
        else:
            jac_sf = None
        jac_tot = jac_sf

        # Translate jacobian_o3_matrix to a jacobian relative to the state vector
        jac_o3 = jacobian_o3_matrix[temp_freq_ind, 1:]
        if do_cloud:
            # Pad jacobian to be nlayers, just adding 0 for layers below the cloud
            jac_o3 = np.pad(jac_o3, pad_width=((0,0),(0,self.nlayers-self.nlayers_cloud)))
        # jac is in increasing pressure order, flip to get
        # decreasing pressure. Also convert to drad_dstate. Note the
        # jacobian_o3_matrix is apparently relative to dlogvmr (on layers)
        if self.dlogvmr_dstate is not None:
            jac_o3 = jac_o3[:,::-1] @ self.layer_to_levels @ self.dlogvmr_dstate
        else:
            jac_o3 = None

        # Combine to an overall jacobian
        if jac_o3 is not None:
            if jac_tot is None:
                jac_tot = jac_o3
            else:
                jac_tot += jac_o3
        if jac_tot is not None:
            rad = rf.ArrayAd_double_1(radiance_matrix[temp_freq_ind, 1], jac_tot)
        else:
            rad = rf.ArrayAd_double_1(radiance_matrix[temp_freq_ind, 1])
        sd = rf.SpectralDomain(freq[temp_freq_ind],rf.Unit("nm"))
        spec = rf.Spectrum(sd, rf.SpectralRange(rad, rf.Unit("sr^-1")))
        self.ocreator.raman_effect(self.sensor_index).apply_effect(spec, None)
        return spec.spectral_range.data_ad
        

class MusesForwardModelVlidortHandle(ForwardModelHandle):
    """Handle for creating a MusesForwardModelVlidort. Note we don't
    just directly use TropomiForwardModelHandle or
    OmiForwardModelHandle because of issues with circular imports. We
    work around this by just adding one level of indirection here.
    """

    def __init__(
        self,
        instrument_name: InstrumentIdentifier,
        use_vlidort_temp_dir: bool = False,
        **creator_kwargs: Any,
    ) -> None:
        self.creator_kwargs = creator_kwargs
        self.instrument_name = instrument_name
        self.measurement_id: None | MeasurementId = None
        self.retrieval_config: None | RetrievalConfiguration = None
        self.use_vlidort_temp_dir = use_vlidort_temp_dir

    def notify_update_target(
        self, measurement_id: MeasurementId, retrieval_config: RetrievalConfiguration
    ) -> None:
        """Clear any caching associated with assuming the target being retrieved is fixed"""
        logger.debug(f"Call to {self.__class__.__name__}::notify_update")
        self.measurement_id = measurement_id
        self.retrieval_config = retrieval_config

    def forward_model(
        self,
        instrument_name: InstrumentIdentifier,
        current_state: CurrentState,
        obs: MusesObservation,
        fm_sv: rf.StateVector,
        **kwargs: Any,
    ) -> None | rf.ForwardModel:
        if instrument_name != self.instrument_name:
            return None
        if self.instrument_name == InstrumentIdentifier("TROPOMI"):
            from refractor.tropomi import TropomiFmObjectCreator

            cls = TropomiFmObjectCreator
        elif self.instrument_name == InstrumentIdentifier("OMI"):
            from refractor.omi import OmiFmObjectCreator

            cls = OmiFmObjectCreator
        else:
            return None
        if self.measurement_id is None or self.retrieval_config is None:
            raise RuntimeError("Call notify_update_target first")
        logger.debug(
            f"Creating forward model MusesForwardModelVlidort for {self.instrument_name}"
        )
        obj_creator = cls(
            current_state,
            self.measurement_id,
            self.retrieval_config,
            obs,
            fm_sv=fm_sv,
            use_vlidort=True,
            match_py_retrieve=True,
            **self.creator_kwargs,
        )
        fm = obj_creator.forward_model
        return fm


ForwardModelHandleSet.add_default_handle(
    MusesForwardModelVlidortHandle(InstrumentIdentifier("TROPOMI")),
    priority_order=-1,
)
ForwardModelHandleSet.add_default_handle(
    MusesForwardModelVlidortHandle(InstrumentIdentifier("OMI")),
    priority_order=-1,
)

__all__ = [
    "MusesForwardModelVlidort",
]

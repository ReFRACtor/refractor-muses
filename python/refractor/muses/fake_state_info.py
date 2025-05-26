from __future__ import annotations
from .identifier import StateElementIdentifier, InstrumentIdentifier
import numpy as np
import typing
from typing import Any

if typing.TYPE_CHECKING:
    from .current_state import CurrentState
    from .muses_observation import MusesObservation


class FakeStateInfo:
    """We are trying to move away from depending on the specific
    StateInfo object/dict used by muses-py. However there are a
    handful of muses-py functions that we want to call which is
    tightly coupled to that old StateInfo object/dict. The content of
    this is a subset of what we have in our CurrentState plus metadata
    for MusesObservation, however the format is substantially
    different.  This class produces a "fake" StateInfo object/dict
    which is a reformat of the CurrentState data.  The purpose of this
    is just to call the old code, an alternative might be to replace
    this old code. But for now, we are massaging our data to call the
    old code.

    Some of the content depends on the MusesObservation metadata. From
    the places we are calling we don't always have easy access to the
    MusesObservation, so we skip that portion of the FakeStateInfo. We
    just "know" that this part isn't needed in the portion that
    doesn't fill it in.
    """

    def __init__(
        self,
        current_state: CurrentState,
        obs_list: list[MusesObservation] | None = None,
    ):
        self._current: dict[str, Any] = {}
        self._initial: dict[str, Any] = {}
        self._initial_initial: dict[str, Any] = {}
        self._constraint: dict[str, Any] = {}
        self._true: dict[str, Any] = {}
        self.current_state = current_state
        # If we have an observation, fill in information into the default dict
        obs_dict = {}
        if obs_list is not None:
            for obs in obs_list:
                obs_dict[obs.instrument_name] = obs

        self._current["PCLOUD"] = current_state.state_value(
            StateElementIdentifier("PCLOUD")
        )
        self._true["PCLOUD"] = np.zeros(self._current["PCLOUD"].shape)
        self._current["PSUR"] = self.state_value("PSUR")
        self._current["TSUR"] = self.state_value("TSUR")
        self._current["latitude"] = current_state.sounding_metadata.latitude.value
        self._current["longitude"] = current_state.sounding_metadata.longitude.value
        self._current["tsa"] = {}
        self._current["tsa"]["surfaceAltitudeKm"] = (
            current_state.sounding_metadata.surface_altitude.value
        )
        self._current["heightKm"] = current_state.sounding_metadata.height.value
        self._current["scalePressure"] = self.state_value("scalePressure")

        # I think this is a fixed list
        if InstrumentIdentifier("TROPOMI") in obs_dict:
            self._species = [
                "TATM",
                "H2O",
                "CO2",
                "O3",
                "N2O",
                "CO",
                "CH4",
                "SO2",
                "NO2",
                "NH3",
                "HNO3",
                "OCS",
                "N2",
                "HCN",
                "SF6",
                "HCOOH",
                "CCL4",
                "CFC11",
                "CFC12",
                "CFC22",
                "HDO",
                "CH3OH",
                "C2H4",
                "PAN",
            ]
        else:
            self._species = [
                "TATM",
                "H2O",
                "CO2",
                "O3",
                "N2O",
                "CO",
                "CH4",
                "SO2",
                "NH3",
                "HNO3",
                "OCS",
                "N2",
                "HCN",
                "SF6",
                "HCOOH",
                "CCL4",
                "CFC11",
                "CFC12",
                "CFC22",
                "HDO",
                "CH3OH",
                "C2H4",
                "PAN",
            ]

        self._num_species = len(self._species)
        t = current_state.state_value(StateElementIdentifier("TATM"))
        varr = np.zeros((len(self._species), t.shape[0]))
        varr2 = np.zeros((len(self._species), t.shape[0]))
        varr3 = np.zeros((len(self._species), t.shape[0]))
        varr4 = np.zeros((len(self._species), t.shape[0]))
        self._true["values"] = np.full((len(self._species), t.shape[0]), -999.0)
        for s in self._species:
            varr[self.species.index(s), :] = current_state.state_value(
                StateElementIdentifier(s)
            )
            varr2[self.species.index(s), :] = current_state.state_constraint_vector(
                StateElementIdentifier(s)
            )
            varr3[self.species.index(s), :] = current_state.state_step_initial_value(
                StateElementIdentifier(s)
            )
            varr4[self.species.index(s), :] = (
                current_state.state_retrieval_initial_value(StateElementIdentifier(s))
            )

        self._current["values"] = varr
        self._constraint["values"] = varr2
        self._initial["values"] = varr3
        self._initial_initial["values"] = varr4
        self._current["pressure"] = current_state.state_value(
            StateElementIdentifier("pressure")
        )
        self._num_pressures = self._current["pressure"].shape[0]
        self._constraint["pressure"] = current_state.state_constraint_vector(
            StateElementIdentifier("pressure")
        )
        self._initial["pressure"] = current_state.state_step_initial_value(
            StateElementIdentifier("pressure")
        )
        self._initial_initial["pressure"] = current_state.state_retrieval_initial_value(
            StateElementIdentifier("pressure")
        )
        # Pretty sure this is always the case, since we don't actually have true values
        self._true["pressure"] = self._initial_initial["pressure"]

        self._constraint["TSUR"] = current_state.state_constraint_vector(
            StateElementIdentifier("TSUR")
        )[0]
        self._gmao_tropopause_pressure = current_state.state_value(
            StateElementIdentifier("gmaoTropopausePressure")
        )[0]
        self._cloud_pars: dict[str, Any] = {}
        # I think this is always 'yes', it looks like the logic in muses-py for setting
        # this to 'no' is never active.
        self._cloud_pars["use"] = "yes"
        self._cloud_pars["frequency"] = current_state.state_spectral_domain_wavelength(
            StateElementIdentifier("cloudEffExt")
        )
        self._cloud_pars["num_frequencies"] = self._cloud_pars["frequency"].shape[0]
        self._current["cloudEffExt"] = np.array(
            current_state.state_value(StateElementIdentifier("cloudEffExt"))
        )
        self._true["cloudEffExt"] = np.zeros(self._current["cloudEffExt"].shape)
        self._emis_pars: dict[str, Any] = {}
        # I think this is always 'yes', it looks like the logic in muses-py for setting
        # this to 'no' is never active.
        self._emis_pars["use"] = "yes"
        self._emis_pars["frequency"] = current_state.state_spectral_domain_wavelength(
            StateElementIdentifier("emissivity")
        )
        self._emis_pars["num_frequencies"] = self._emis_pars["frequency"].shape[0]
        self._current["emissivity"] = np.array(
            current_state.state_value(StateElementIdentifier("emissivity"))
        )
        self._constraint["emissivity"] = np.array(
            current_state.state_constraint_vector(StateElementIdentifier("emissivity"))
        )
        self._true["emissivity"] = np.zeros(self._current["emissivity"].shape)
        self._calibration_pars: dict[str, Any] = {}
        # I think this is always 'no', it looks like the logic in muses-py for setting
        # this to 'yes' is never active.
        self._calibration_pars["use"] = "no"
        self._calibration_pars["frequency"] = (
            current_state.state_spectral_domain_wavelength(
                StateElementIdentifier("calibrationScale")
            )
        )
        self._calibration_pars["num_frequencies"] = self._calibration_pars[
            "frequency"
        ].shape[0]
        self._current["calibrationScale"] = current_state.state_value(
            StateElementIdentifier("calibrationScale")
        )
        self._current["calibrationOffset"] = current_state.state_value(
            StateElementIdentifier("calibrationOffset")
        )
        self._true["calibrationScale"] = np.zeros(
            self._current["calibrationScale"].shape
        )
        self._current["residualScale"] = current_state.state_value(
            StateElementIdentifier("residualScale")
        )
        self._current["airs"] = self.default_airs()
        self._current["cris"] = self.default_cris()
        self._current["omi"] = self.default_omi()
        self._current["tropomi"] = self.default_tropomi()
        self._current["nir"] = self.default_nir()
        self._current["tes"] = self.default_tes()
        # print(current_state._state_info.state_info_dict["current"]["tes"])
        if InstrumentIdentifier("OMI") in obs_dict:
            self.fill_omi(current_state, obs_dict[InstrumentIdentifier("OMI")])
        if InstrumentIdentifier("TROPOMI") in obs_dict:
            self.fill_tropomi(current_state, obs_dict[InstrumentIdentifier("TROPOMI")])
        if InstrumentIdentifier("AIRS") in obs_dict:
            self.fill_airs(current_state, obs_dict[InstrumentIdentifier("AIRS")])
        if InstrumentIdentifier("CRIS") in obs_dict:
            self.fill_cris(current_state, obs_dict[InstrumentIdentifier("CRIS")])
        if InstrumentIdentifier("TES") in obs_dict:
            self.fill_tes(current_state, obs_dict[InstrumentIdentifier("TES")])
        # print(self._current['tes'])
        # print(current_state._state_info.state_info_dict["current"]["tes"].keys())
        # print(self._current['tes'].keys())
        # breakpoint()

        # Fields we needed from different functions we call:
        # Write quality flag
        # x - stateInfo.current['PCLOUD'][0]
        # x - stateInfo.current['TSUR']
        # x - stateInfo.constraint['TSUR']
        # x - stateInfo.current['values'][stateInfo.species.index('TATM'),0]
        # write_retrieval_summary
        # x - stateInfo.true['values']
        # x - stateInfo.species, 'TATM'
        # x - stateInfo.species, 'H2O')'
        # x - stateInfo.species, 'O3')'
        # x - stateInfo.current['pressure']
        # x - stateInfo.current['latitude']
        # x - stateInfo.gmaoTropopausePressure
        # x - stateInfo.current['scalePressure']
        # x - stateInfo.current['tsa']['surfaceAltitudeKm']
        # x - stateInfo.cloudPars['frequency']
        # x - stateInfo.current['cloudEffExt']
        # x - stateInfo.cloudPars['use']
        # x - stateInfo.emisPars['frequency']
        # x - stateInfo.current['emissivity']
        # x - stateInfo.current['heightKm']
        # x - retrievalInfo.species[ispecie] - all species that are in stateInfo.species
        # x - stateInfo.constraint['pressure']
        # x - current, constraint, initial, initialInitial in values, pressure
        # x - has true, but we should probably make sure have_true is false
        # x - stateInfo.current['omi']['cloud_fraction']
        # x - stateInfo.current['tropomi']['cloud_fraction']
        # error_analysis_wrapper
        # x - stateInfo.emisPars['num_frequencies']
        # x - stateInfo.current['emissivity']
        # x - stateInfo.true['emissivity'][0:n_count]
        # x - stateInfo.cloudPars['num_frequencies']
        # x - stateInfo.current['cloudEffExt']
        # x - stateInfo.true['cloudEffExt'][
        # x - stateInfo.calibrationPars['num_frequencies']
        # x - stateInfo.current['calibrationScale']
        # x - stateInfo.true['calibrationScale']
        # x - stateInfo.current['PCLOUD']
        # x - stateInfo.true['PCLOUD']
        # x - stateInfo.current['TSUR']
        # x - stateInfo.current['tes']['boresightNadirRadians']
        # x - stateInfo.current['PSUR']
        # x - stateInfo.num_pressures
        # x - stateInfo.current['heightKm']
        # make_uip_master
        # x - state.num_species
        # x - state.emisPars['use']
        # x - state.calibrationPars['use']
        # x - calibrationOffset
        # stateOne.airs
        # stateOne.cris
        # stateOne.omi
        # stateOne.tropomi
        # stateOne.nir
        # make_uip_tes
        # ['tes']['boresightNadirRadians']
        # ['tes']['instrumentAzimuth']
        # ['tes']['instrumentAltitude']
        # ['tes']['orbitInclinationAngle']

    @property
    def current(self) -> dict[str, Any]:
        return self._current

    @property
    def species(self) -> list[str]:
        return self._species

    @property
    def constraint(self) -> dict[str, Any]:
        return self._constraint

    @property
    def initial(self) -> dict[str, Any]:
        return self._initial

    @property
    def true(self) -> dict[str, Any]:
        return self._true

    @property
    def initialInitial(self) -> dict[str, Any]:
        return self._initial_initial

    @property
    def num_pressures(self) -> int:
        return self._num_pressures

    @property
    def num_species(self) -> int:
        return self._num_species

    @property
    def gmaoTropopausePressure(self) -> float:
        return self._gmao_tropopause_pressure

    @property
    def cloudPars(self) -> dict[str, Any]:
        return self._cloud_pars

    @property
    def emisPars(self) -> dict[str, Any]:
        return self._emis_pars

    @property
    def calibrationPars(self) -> dict[str, Any]:
        return self._calibration_pars

    def default_omi(self) -> dict[str, Any]:
        # Copied from new_state_structures.py in muses-py, this is the values if we don't
        # otherwise set
        omi = {
            "surface_albedo_uv1": 0.0 - 999,
            "surface_albedo_uv2": 0.0 - 999,
            "surface_albedo_slope_uv2": 0.0 - 999,
            "nradwav_uv1": 0.0 - 999,
            "nradwav_uv2": 0.0 - 999,
            "odwav_uv1": 0.0 - 999,
            "odwav_uv2": 0.0 - 999,
            "odwav_slope_uv1": 0.0 - 999,
            "odwav_slope_uv2": 0.0 - 999,
            "ring_sf_uv1": 0.0 - 999,
            "ring_sf_uv2": 0.0 - 999,
            "cloud_fraction": 0.0 - 999,
            "cloud_pressure": 0.0 - 999,
            "cloud_Surface_Albedo": 0.0 - 999,
            "xsecscaling": 0.0 - 999,
            "resscale_uv1": 0.0 - 999,
            "resscale_uv2": 0.0 - 999,
            "sza_uv1": 0.0 - 999,
            "raz_uv1": 0.0 - 999,
            "vza_uv1": 0.0 - 999,
            "sca_uv1": 0.0 - 999,
            "sza_uv2": 0.0 - 999,
            "raz_uv2": 0.0 - 999,
            "vza_uv2": 0.0 - 999,
            "sca_uv2": 0.0 - 999,
            "SPACECRAFTALTITUDE": 0.0 - 999,
        }
        # Need cloud fraction even if other part isn't filled in
        omi["cloud_fraction"] = self.state_value("OMICLOUDFRACTION")
        return omi

    def default_tropomi(self) -> dict[str, Any]:
        tropomi = {
            "surface_albedo_BAND1": 0.0 - 999,
            "surface_albedo_BAND2": 0.0 - 999,
            "surface_albedo_BAND3": 0.0 - 999,
            "surface_albedo_BAND7": 0.0 - 999,
            "surface_albedo_slope_BAND1": 0.0 - 999,
            "surface_albedo_slope_BAND2": 0.0 - 999,
            "surface_albedo_slope_order2_BAND2": 0.0 - 999,
            "surface_albedo_slope_BAND3": 0.0 - 999,
            "surface_albedo_slope_order2_BAND3": 0.0 - 999,
            "surface_albedo_slope_BAND7": 0.0 - 999,
            "surface_albedo_slope_order2_BAND7": 0.0 - 999,
            "solarshift_BAND1": 0.0 - 999,
            "solarshift_BAND2": 0.0 - 999,
            "solarshift_BAND3": 0.0 - 999,
            "solarshift_BAND7": 0.0 - 999,
            "radianceshift_BAND1": 0.0 - 999,
            "radianceshift_BAND2": 0.0 - 999,
            "radianceshift_BAND3": 0.0 - 999,
            "radianceshift_BAND7": 0.0 - 999,
            "radsqueeze_BAND1": 0.0 - 999,
            "radsqueeze_BAND2": 0.0 - 999,
            "radsqueeze_BAND3": 0.0 - 999,
            "temp_shift_BAND3": 0.0 - 999,
            "radsqueeze_BAND7": 0.0 - 999,
            "temp_shift_BAND7": 0.0 - 999,
            "ring_sf_BAND1": 0.0 - 999,
            "ring_sf_BAND2": 0.0 - 999,
            "ring_sf_BAND3": 0.0 - 999,
            "ring_sf_BAND7": 0.0 - 999,
            "cloud_fraction": 0.0 - 999,
            "cloud_pressure": 0.0 - 999,
            "cloud_Surface_Albedo": 0.0 - 999,
            "xsecscaling": 0.0 - 999,
            "resscale_O0_BAND1": 0.0 - 999,
            "resscale_O1_BAND1": 0.0 - 999,
            "resscale_O2_BAND1": 0.0 - 999,
            "resscale_O0_BAND2": 0.0 - 999,
            "resscale_O1_BAND2": 0.0 - 999,
            "resscale_O2_BAND2": 0.0 - 999,
            "resscale_O0_BAND3": 0.0 - 999,
            "resscale_O1_BAND3": 0.0 - 999,
            "resscale_O2_BAND3": 0.0 - 999,
            "resscale_O0_BAND7": 0.0 - 999,
            "resscale_O1_BAND7": 0.0 - 999,
            "resscale_O2_BAND7": 0.0 - 999,
            "sza_BAND1": 0.0 - 999,
            "raz_BAND1": 0.0 - 999,
            "vza_BAND1": 0.0 - 999,
            "sca_BAND1": 0.0 - 999,
            "sza_BAND2": 0.0 - 999,
            "raz_BAND2": 0.0 - 999,
            "vza_BAND2": 0.0 - 999,
            "sca_BAND2": 0.0 - 999,
            "sza_BAND3": 0.0 - 999,
            "raz_BAND3": 0.0 - 999,
            "vza_BAND3": 0.0 - 999,
            "sca_BAND3": 0.0 - 999,
            "sza_BAND7": 0.0 - 999,
            "raz_BAND7": 0.0 - 999,
            "vza_BAND7": 0.0 - 999,
            "sca_BAND7": 0.0 - 999,
            "SPACECRAFTALTITUDE": 0.0 - 999,
        }
        # Need cloud fraction even if other part isn't filled in
        tropomi["cloud_fraction"] = self.state_value("TROPOMICLOUDFRACTION")
        return tropomi

    def default_airs(self) -> dict[str, Any]:
        airs = {
            "scanAng": 0.0,
            "satZen": 0.0,
            "satAzi": 0.0,
            "sza": 0.0,
            "solazi": 0.0,
            "sunGlintDistance": 0.0,
            "landFraction": 0.0,
            "satHeight": 888888.0,
        }
        return airs

    def default_cris(self) -> dict[str, Any]:
        cris = {
            "scanAng": 0.0,
            "satZen": 0.0,
            "satAzi": 0.0,
            "sza": 0.0,
            "solazi": 0.0,
            "satAlt": 0.0,
            "l1bType": "N/A",  # track specific cris instrument and l1b type used, see types in script_retrieval_setup_ms
            "APODIZATION": "xxxx",
            "APODIZATIONFUNCTION": "xxxx",
        }
        return cris

    def default_tes(self) -> dict[str, Any]:
        tes = {}
        # Need pointing angle even if other part not filled in.
        tes["boresightNadirRadians"] = self.state_value("PTGANG")
        return tes

    def default_nir(self) -> dict[str, Any]:
        # 180 wavelengths
        minn = 0.7565
        maxx = 0.7730
        wavelengths1 = np.array(range(60)) / 59.0 * (maxx - minn) + minn

        minn = 1.590
        maxx = 1.624
        wavelengths2 = np.array(range(60)) / 59.0 * (maxx - minn) + minn

        minn = 2.298
        maxx = 2.349
        wavelengths3 = np.array(range(60)) / 59.0 * (maxx - minn) + minn

        albplwave = np.zeros((180), dtype=np.float64)
        albplwave[0:60] = wavelengths1
        albplwave[60:120] = wavelengths2
        albplwave[120:180] = wavelengths3
        naer = 5
        nalbplwave = len(albplwave)
        nir = {
            "naer": np.int32(naer),
            "aertype": np.empty(naer, dtype="S80"),
            "aerod": np.zeros(shape=(naer), dtype=np.float64)
            - 999,  # aerosol OD [previously [parameter type, aerosol type] and od was in log]
            "aerp": np.zeros(shape=(naer), dtype=np.float64) - 999,  # aerosol pressure
            "aerw": np.zeros(shape=(naer), dtype=np.float64) - 999,  # aerosol width
            "albtype": -999,  # albedo type: 1 = Lambertian; 2 = BRDF; 3 = CoxMunk
            "albpl": np.zeros(shape=(nalbplwave), dtype=np.float64) - 999,
            "albplwave": np.zeros(shape=(nalbplwave), dtype=np.float64) + albplwave,
            "disp": np.zeros(shape=(3, 6), dtype=np.float64)
            - 999,  # [band, order] called spectral_coef in L1B and Refractor
            "eof": np.zeros(shape=(3, 3), dtype=np.float64) - 999,  # [band, type]
            "cloud3doffset": np.zeros(
                shape=(3), dtype=np.float64
            ),  # set to 0.0 default
            "cloud3dslope": np.zeros(shape=(3), dtype=np.float64),  # set to 0.0 default
            "fluor": np.zeros(shape=(2), dtype=np.float64)
            - 999,  # Scaled:  To get the Refractor inputs, multiply by [1e17,1e-3].  Similarly multiply Refractor Jacobians by [1e-17,1e3]
            "wind": -999.00,
        }
        return nir

    def state_value(self, state_name: str) -> float:
        """Get the state value for the given state name"""
        return self.current_state.state_value(StateElementIdentifier(state_name))[0]

    def fill_omi(self, current_state: CurrentState, obs: MusesObservation) -> None:
        d = self._current["omi"]
        blist = [str(i[0]) for i in obs.filter_data]
        sza = obs.solar_zenith
        raz = obs.relative_azimuth
        vza = obs.observation_zenith
        sca = obs.scattering_angle
        for bout in ["UV1", "UV2"]:
            if bout in blist:
                i = blist.index(bout)
                bout_lc = bout.lower()
                d[f"sza_{bout_lc}"] = np.float32(sza[i])
                d[f"raz_{bout_lc}"] = raz[i]
                d[f"vza_{bout_lc}"] = np.float32(vza[i])
                d[f"sca_{bout_lc}"] = sca[i]
                d[f"surface_albedo_{bout_lc}"] = self.state_value(
                    f"OMISURFACEALBEDO{bout}"
                )
                if bout != "UV1":
                    # For who knows what reason this isn't present for UV1.
                    d[f"surface_albedo_slope_{bout_lc}"] = self.state_value(
                        f"OMISURFACEALBEDOSLOPE{bout}"
                    )
                d[f"nradwav_{bout_lc}"] = self.state_value(f"OMINRADWAV{bout}")
                d[f"odwav_{bout_lc}"] = self.state_value(f"OMIODWAV{bout}")
                d[f"odwav_slope_{bout_lc}"] = self.state_value(f"OMIODWAVSLOPE{bout}")
                d[f"ring_sf_{bout_lc}"] = self.state_value(f"OMIRINGSF{bout}")
        d["cloud_fraction"] = self.state_value("OMICLOUDFRACTION")
        d["cloud_pressure"] = int(obs.cloud_pressure.value)
        d["cloud_Surface_Albedo"] = 0.8
        d["xsecscaling"] = 1.0
        d["SPACECRAFTALTITUDE"] = obs.spacecraft_altitude

    def fill_tropomi(self, current_state: CurrentState, obs: MusesObservation) -> None:
        d = self._current["tropomi"]
        blist = [str(i[0]) for i in obs.filter_data]
        sza = obs.solar_zenith
        raz = obs.relative_azimuth
        vza = obs.observation_zenith
        sca = obs.scattering_angle
        for bout in ["BAND1", "BAND2", "BAND3", "BAND7"]:
            if bout in blist:
                i = blist.index(bout)
                d[f"sza_{bout}"] = np.float32(sza[i])
                d[f"raz_{bout}"] = raz[i]
                d[f"vza_{bout}"] = np.float32(vza[i])
                d[f"sca_{bout}"] = sca[i]
                d[f"resscale_O0_{bout}"] = 1.0
                d[f"resscale_O1_{bout}"] = 0.0
                d[f"resscale_O2_{bout}"] = 0.0
                d[f"surface_albedo_{bout}"] = self.state_value(
                    f"TROPOMISURFACEALBEDO{bout}"
                )
                d[f"solarshift_{bout}"] = self.state_value(f"TROPOMISOLARSHIFT{bout}")
                d[f"radianceshift_{bout}"] = self.state_value(
                    f"TROPOMIRADIANCESHIFT{bout}"
                )
                d[f"radsqueeze_{bout}"] = self.state_value(f"TROPOMIRADSQUEEZE{bout}")
                d[f"ring_sf_{bout}"] = self.state_value(f"TROPOMIRINGSF{bout}")
                d[f"surface_albedo_slope_{bout}"] = self.state_value(
                    f"TROPOMISURFACEALBEDOSLOPE{bout}"
                )
                if bout != "BAND1":
                    # For who knows what reason this isn't present for band 1.
                    d[f"surface_albedo_slope_order2_{bout}"] = self.state_value(
                        f"TROPOMISURFACEALBEDOSLOPEORDER2{bout}"
                    )
                if bout in ("BAND3", "BAND7"):
                    d[f"temp_shift_{bout}"] = self.state_value(
                        f"TROPOMITEMPSHIFT{bout}"
                    )
            else:
                d[f"sza_{bout}"] = 0.0
                d[f"raz_{bout}"] = 0.0
                d[f"vza_{bout}"] = 0.0
                d[f"sca_{bout}"] = 0.0
                d[f"resscale_O0_{bout}"] = 0.0
                d[f"resscale_O1_{bout}"] = 0.0
                d[f"resscale_O2_{bout}"] = 0.0
                d[f"surface_albedo_{bout}"] = 0.0
                d[f"solarshift_{bout}"] = 0.0
                d[f"radianceshift_{bout}"] = 0.0
                d[f"radsqueeze_{bout}"] = 0.0
                d[f"ring_sf_{bout}"] = 0.0
                d[f"surface_albedo_slope_{bout}"] = 0.0
                if bout != "BAND1":
                    # For who knows what reason this isn't present for band 1.
                    d[f"surface_albedo_slope_order2_{bout}"] = 0.0
                if bout in ("BAND3", "BAND7"):
                    d[f"temp_shift_{bout}"] = 0.0
        d["cloud_fraction"] = self.state_value("TROPOMICLOUDFRACTION")
        d["cloud_pressure"] = obs.cloud_pressure.value
        d["cloud_Surface_Albedo"] = 0.8
        d["xsecscaling"] = 1.0
        d["SPACECRAFTALTITUDE"] = obs.spacecraft_altitude

    def fill_airs(self, current_state: CurrentState, obs: MusesObservation) -> None:
        d = {}
        d2 = obs.muses_py_dict
        for k in (
            "scanAng",
            "satZen",
            "satAzi",
            "sza",
            "solazi",
            "sunGlintDistance",
            "landFraction",
            "satHeight",
            "latitude",
            "longitude",
            "time",
            "radiance",
            "DaytimeFlag",
            "CalChanSummary",
            "ExcludedChans",
            "NESR",
            "frequency",
            "surfaceAltitude",
            "state",
            "valid",
        ):
            # This one field has a different name on obs vs. state_info
            if k == "satHeight":
                d[k] = d2["satheight"]
            else:
                d[k] = d2[k]
        self._current["airs"] = d

    def fill_cris(self, current_state: CurrentState, obs: MusesObservation) -> None:
        d = {}
        d2 = obs.muses_py_dict
        for k in (
            "scanAng",
            "satZen",
            "satAzi",
            "sza",
            "solazi",
            "satAlt",
            "l1bType",
            "APODIZATION",
            "APODIZATIONFUNCTION",
            "NESR",
        ):
            # This one field has a different name on obs vs. state_info
            if k == "l1bType":
                d[k] = d2[k]
            else:
                d[k] = d2[k.upper()]
        self._current["cris"] = d

    def fill_tes(self, current_state: CurrentState, obs: MusesObservation) -> None:
        d: dict[str, Any] = {}
        d2 = obs.muses_py_dict
        for k in (
            "boresightNadirRadians",
            "orbitInclinationAngle",
            "viewMode",
            "instrumentAzimuth",
            "instrumentAltitude",
            "instrumentLatitude",
            "geoPointing",
            "targetRadius",
            "instrumentRadius",
            "orbitAscending",
        ):
            # Special case for IRK, where we only have a fake tes observation. We just
            # use various fill values for stuff not actually in d2.
            if k not in d2:
                if k == "viewMode":
                    d[k] = "Nadir"
                elif k == "orbitAscending":
                    d[k] = 0
                else:
                    d[k] = 0.0
            else:
                d[k] = d2[k]
        self._current["tes"] = d


__all__ = [
    "FakeStateInfo",
]

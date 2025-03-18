from __future__ import annotations
from .identifier import StateElementIdentifier
import numpy as np
import typing
from typing import Any

if typing.TYPE_CHECKING:
    from .current_state import CurrentState
    from .state_info import StateInfo


class FakeStateInfo:
    """We are trying to move away from depending on the specific
    StateInfo object/dict used by muses-py. However there are a
    handful of muses-py functions that we want to call which is
    tightly coupled to that old StateInfo object/dict. The content of
    this is largely what we have in our CurrentState, however the
    format is substantially different.  This class produces a "fake"
    StateInfo object/dict which is a reformat of the CurrentState
    data.  The purpose of this is just to call the old code, an
    alternative might be to replace this old code. But for now, we are
    massaging our data to call the old code.
    """

    # Note state_info is just temporary, so we can more easily figure out how to
    # duplicate things. This will go away
    def __init__(
            self, current_state: CurrentState, species_name : list[str] | None = None,
            state_info: StateInfo | None = None
    ):
        self._current : dict[str, Any] = {}
        self._initial : dict[str, Any] = {}
        self._initial_initial : dict[str, Any] = {}
        self._constraint : dict[str, Any] = {}
        self._true : dict[str, Any] = {}
        
        self._current["PCLOUD"] = current_state.full_state_value(
            StateElementIdentifier("PCLOUD")
        )
        self._true["PCLOUD"] = np.zeros(self._current["PCLOUD"].shape)
        self._current["PSUR"] = current_state.full_state_value(StateElementIdentifier("PSUR"))[0]
        self._current["TSUR"] = current_state.full_state_value(
            StateElementIdentifier("TSUR")
        )[0]
        self._current['latitude'] = current_state.sounding_metadata.latitude.value
        self._current['tsa'] = {}
        self._current['tsa']['surfaceAltitudeKm'] = current_state.sounding_metadata.surface_altitude.value
        self._current['heightKm'] = current_state.sounding_metadata.height.value
        self._current['scalePressure'] = current_state.full_state_value(StateElementIdentifier("scalePressure"))
                    
        self._species = ["TATM", "H2O", "O3"]
        if(species_name is not None):
            for s in species_name:
                if s not in self._species and StateElementIdentifier(s) in current_state.full_state_element_on_levels_id:
                    self._species.append(s)
        t = current_state.full_state_value(StateElementIdentifier("TATM"))
        varr = np.zeros((len(self._species), t.shape[0]))
        varr2 = np.zeros((len(self._species), t.shape[0]))
        varr3 = np.zeros((len(self._species), t.shape[0]))
        varr4 = np.zeros((len(self._species), t.shape[0]))
        self._true["values"] = np.full((len(self._species), t.shape[0]), -999.0)
        for s in self._species:
            varr[self.species.index(s), :] = current_state.full_state_value(StateElementIdentifier(s))
            varr2[self.species.index(s), :] = current_state.full_state_apriori_value(StateElementIdentifier(s))
            varr3[self.species.index(s), :] = current_state.full_state_initial_value(StateElementIdentifier(s))
            varr4[self.species.index(s), :] = current_state.full_state_initial_initial_value(StateElementIdentifier(s))
            
        self._current["values"] = varr
        self._constraint["values"] = varr2
        self._initial["values"] = varr3
        self._initial_initial["values"] = varr4
        self._current['pressure'] = current_state.full_state_value(StateElementIdentifier("pressure"))
        self._num_pressures = self._current['pressure'].shape[0]
        self._constraint['pressure'] = current_state.full_state_apriori_value(StateElementIdentifier("pressure"))
        self._initial['pressure'] = current_state.full_state_initial_value(StateElementIdentifier("pressure"))
        self._initial_initial['pressure'] = current_state.full_state_initial_initial_value(StateElementIdentifier("pressure"))
        
        self._constraint["TSUR"] = current_state.full_state_apriori_value(
            StateElementIdentifier("TSUR")
        )[0]
        self._current['omi'] = {}
        self._current['tropomi'] = {}
        if StateElementIdentifier("OMICLOUDFRACTION") in current_state.full_state_element_id:
            self._current['omi']['cloud_fraction'] = current_state.full_state_value(StateElementIdentifier("OMICLOUDFRACTION"))[0]
        else:
            self._current['omi']['cloud_fraction'] = -999.0
        if StateElementIdentifier("TROPOMICLOUDFRACTION") in current_state.full_state_element_id:
            self._current['tropomi']['cloud_fraction'] = current_state.full_state_value(StateElementIdentifier("TROPOMICLOUDFRACTION"))[0]
        else:
            self._current['tropomi']['cloud_fraction'] = -999.0
        self._current['tes'] = {}
        self._current['tes']['boresightNadirRadians'] = current_state.full_state_value(StateElementIdentifier("PTGANG"))[0]
        self._gmao_tropopause_pressure = current_state.full_state_value(StateElementIdentifier("gmaoTropopausePressure"))[0]
        self._cloud_pars : dict[str, Any] = {}
        # I think this is always 'yes', it looks like the logic in muses-py for setting
        # this to 'no' is never active.
        self._cloud_pars['use'] = 'yes'
        self._cloud_pars['frequency'] = current_state.full_state_spectral_domain_wavelength(StateElementIdentifier("cloudEffExt"))
        self._cloud_pars['num_frequencies'] = self._cloud_pars['frequency'].shape[0]
        self._current['cloudEffExt'] = current_state.full_state_value(StateElementIdentifier("cloudEffExt"))
        self._true['cloudEffExt'] = np.zeros(self._current['cloudEffExt'].shape)
        self._emis_pars : dict[str, Any] = {}
        self._emis_pars['frequency'] = current_state.full_state_spectral_domain_wavelength(StateElementIdentifier("emissivity"))
        self._emis_pars['num_frequencies'] = self._emis_pars['frequency'].shape[0]
        self._current['emissivity'] = current_state.full_state_value(StateElementIdentifier("emissivity"))
        self._constraint['emissivity'] = current_state.full_state_apriori_value(StateElementIdentifier("emissivity"))
        self._true['emissivity'] = np.zeros(self._current['emissivity'].shape)
        self._calibration_pars : dict[str, Any] = {}
        self._calibration_pars['frequency'] = current_state.full_state_spectral_domain_wavelength(StateElementIdentifier("calibrationScale"))
        self._calibration_pars['num_frequencies'] = self._calibration_pars['frequency'].shape[0]
        self._current['calibrationScale'] = current_state.full_state_value(StateElementIdentifier("calibrationScale"))
        self._true['calibrationScale'] = np.zeros(self._current['calibrationScale'].shape)
        
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


    @property
    def current(self):
        return self._current

    @property
    def species(self):
        return self._species

    @property
    def constraint(self):
        return self._constraint

    @property
    def initial(self):
        return self._initial

    @property
    def initialInitial(self):
        return self._initial_initial

    @property
    def true(self):
        return self._true

    @property
    def num_pressures(self):
        return self._num_pressures
    
    @property
    def gmaoTropopausePressure(self):
        return self._gmao_tropopause_pressure
    
    @property
    def cloudPars(self):
        return self._cloud_pars

    @property
    def emisPars(self):
        return self._emis_pars

    @property
    def calibrationPars(self):
        return self._calibration_pars
    

__all__ = [
    "FakeStateInfo",
]

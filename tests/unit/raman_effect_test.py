import numpy as np
import numpy.testing as npt
import refractor.framework as rf


class RingInputFile(object):
    def __init__(self, filename):
        nw, nz, albedo, self.sza, self.vza, _, self.sca = np.loadtxt(
            filename, max_rows=1
        )
        self.num_grid, self.num_layers = int(nw), int(nz)

        self.temperature_layers, self.air_density = np.loadtxt(
            filename, skiprows=1, max_rows=2
        )

        dat = np.loadtxt(filename, skiprows=3)
        self.grid = dat[:, 0]
        self.solar_irradiance = dat[:, 1]
        self.optical_depth = dat[:, 2:]


class RingOutputFile(object):
    def __init__(self, filename):
        conv = {1: lambda v: float(v.replace("D", "E"))}
        rd = np.loadtxt(filename, skiprows=1, converters=conv, encoding='utf-8')
        self.grid = rd[:, 0]
        self.spec = rd[:, 1]


def test_raman_effect(omi_config_dir, omi_test_in_dir, omi_test_expected_results_dir):
    config_filename = omi_config_dir / "muses_simulation_config.py"
    uip_filename = omi_test_in_dir.parent / "raman/uip-FM.sav"

    config_module = rf.load_config_module(config_filename)
    config_func = rf.find_config_function(config_module)
    config_def = config_func(uip_filename)
    config_inst = rf.process_config(config_def)

    fm = config_inst.forward_model
    atm = config_inst.atmosphere

    scale_factor = 1.9
    channel_index = 0

    rf_scenario = config_def["scenario"]
    observation_zenith = rf_scenario["observation_zenith"][channel_index]
    solar_zenith = rf_scenario["solar_zenith"][channel_index]
    relative_azimuth = rf_scenario["relative_azimuth"][channel_index]

    # Use grid and solar irradiance from MUSES test case to match their results
    ring_inp_filename = omi_test_in_dir.parent / "raman/Ring_input.asc"
    raman_inputs = RingInputFile(ring_inp_filename)

    grid_sd = rf.SpectralDomain(raman_inputs.grid, rf.Unit("nm"))

    # Units of solar spectrum from SAO2010 Solar Irradiance Reference Spectrum file
    sol_sr = rf.SpectralRange(
        raman_inputs.solar_irradiance, rf.Unit("Ph s^-1 cm^-2 nm^-1")
    )
    sol_spec = rf.Spectrum(grid_sd, sol_sr)
    solar_model = rf.SolarReferenceSpectrum(sol_spec, None)

    do_upwelling = True

    # Here to complete interface so we can supply padding_fraction
    mapping = rf.StateMappingLinear()

    raman_effect = rf.RamanSiorisEffect(
        grid_sd,
        scale_factor,
        channel_index,
        solar_zenith,
        observation_zenith,
        relative_azimuth,
        atm,
        solar_model,
        mapping,
        do_upwelling,
    )

    # Create a spectrum of just ones so we get just the scaling effect from the Raman class
    rad_spec = rf.Spectrum(
        grid_sd, rf.SpectralRange(np.ones(raman_inputs.grid.shape[0]), sol_sr.units)
    )

    # Output will be scaling that would have been applied to a spectrum
    raman_effect.apply_effect(rad_spec, fm.spectral_grid)

    ring_expt_filename = omi_test_expected_results_dir / "raman/Ring.asc"
    ring_expt = RingOutputFile(ring_expt_filename)

    # Convert to scale factor just as the raman effect apply to our spectrum of ones
    expt_spec = ring_expt.spec * scale_factor + 1.0

    npt.assert_allclose(expt_spec, rad_spec.spectral_range.data, rtol=2e-3)

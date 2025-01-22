import os

import numpy as np
import h5py  # type: ignore

from refractor.framework.factory import creator, param  # type: ignore
import refractor.framework as rf  # type: ignore

config_dir = os.path.dirname(__file__)

# ILS and solar reference spectra used below, are in the OSP directory
# config_dir is at omi/config
if "MUSES_OSP_PATH" in os.environ:
    osp_dir = os.environ["MUSES_OSP_PATH"]
else:
    osp_dir = os.path.expanduser("~/OSP")
ils_path = f"{osp_dir}/OMI/OMI_ILS/NORMAL/"

# Same file used by MUSES retrieval
# Listed as a 3 year mean of solar spectra
solar_ref_3yr_filename = f"{osp_dir}/OMI/OMI_Solar/omisol_v003_avg_nshi_backup.h5"
num_channels = 2
channel_names = ["UV1", "UV2"]


def omi_windows(**kwargs):
    win_ranges = np.zeros((num_channels, 2))

    # Spanning just a bit beyond what L1B files give as the full range
    win_ranges[0, :] = [262, 315]  # UV-1
    win_ranges[1, :] = [306, 386]  # UV-2

    return rf.ArrayWithUnit(win_ranges, "nm")


def omi_ref_points(**kwargs):
    # Use mid point of full spectral range
    full_spectral_ranges = omi_windows().value
    ref_point_vals = []
    for chan_idx in range(full_spectral_ranges.shape[0]):
        ref_point_vals.append(np.sum(full_spectral_ranges[chan_idx, :]) / 2)
    return rf.ArrayWithUnit(np.array(ref_point_vals), "nm")


def diagonal_cov(size, value):
    cov = np.identity(size) * value
    return cov


class OmiSolarReference3yr(creator.base.Creator):
    solar_reference_filename = param.Scalar(str)
    across_track_indexes = param.Array(1)

    num_channels = param.Scalar(int)
    doppler = param.Iterable(rf.SolarDopplerShift, required=False)

    def create(self, **kwargs):
        xtrack_indexes = self.across_track_indexes()
        num_channels = self.num_channels()

        doppler_shifts = self.doppler()

        if doppler_shifts is None:
            doppler_shifts = [None] * num_channels

        solar_file = h5py.File(self.solar_reference_filename(), "r")

        solar_objs = []
        for chan_idx, chan_name in enumerate(channel_names):
            # Get values for the current channel at the appropriate across track index
            wav_vals = solar_file[f"WAV_{chan_name}"][:, xtrack_indexes[chan_idx]]
            irad_vals = solar_file[f"SOL_{chan_name}"][:, xtrack_indexes[chan_idx]]

            # File does not have units contained within it
            # Same units as the OMI L1B files, but use irradiance units
            sol_domain = rf.SpectralDomain(wav_vals, rf.Unit("nm"))
            sol_range = rf.SpectralRange(irad_vals, rf.Unit("ph / nm / s"))
            sol_spec = rf.Spectrum(sol_domain, sol_range)

            ref_spec = rf.SolarReferenceSpectrum(sol_spec, doppler_shifts[chan_idx])

            solar_objs.append(ref_spec)

        solar_file.close()

        return solar_objs


class OmiIlsTable(creator.instrument.IlsTable):
    across_track_indexes = param.Array(1)
    ils_path = param.Scalar(str)

    def create(self, **kwargs):
        self.ils_path()
        channel_name = self.hdf_band_name()
        across_track_indexes = self.across_track_indexes()

        wavenumber = []
        delta_lambda = []
        response = []
        for chan_idx in range(self.num_channels()):
            chan_name = channel_name[chan_idx]
            ils_filename = os.path.join(
                self.ils_path(),
                chan_name,
                "OMI_ILS_NORMAL_{}_{:02d}.h5".format(
                    chan_name, across_track_indexes[chan_idx]
                ),
            )

            ils_data = h5py.File(ils_filename, "r")

            chan_freq_wl = ils_data["FREQ_MONO"][:]
            chan_center_wl = ils_data["XCF0"][:]

            chan_center_wn = np.flip(
                rf.ArrayWithUnit(chan_center_wl, "nm").convert_wave("cm^-1").value
            )
            chan_freq_wn = np.flip(
                rf.ArrayWithUnit(chan_freq_wl, "nm").convert_wave("cm^-1").value
            ).transpose()

            chan_response = np.flip(ils_data["ILS_MONO"][:]).transpose()

            delta_lambda_wn = np.zeros(chan_freq_wn.shape)
            for center_idx in range(chan_center_wn.shape[0]):
                delta_lambda_wn[center_idx, :] = (
                    chan_freq_wn[center_idx, :] - chan_center_wn[center_idx]
                )

            wavenumber.append(chan_center_wn)
            delta_lambda.append(delta_lambda_wn)
            response.append(chan_response)

        self.config_def["wavenumber"] = wavenumber
        self.config_def["delta_lambda"] = delta_lambda
        self.config_def["response"] = response

        return super().create(**kwargs)


# Common configuration definition shared amongst retrieval and simulation types of configuration
def base_config_definition(
    micro_windows=None,
    retrieval_elements=None,
    solver_parameters=None,
    covariance_storage=None,
    **kwargs,
):
    # Store covariances for updating between steps
    if covariance_storage is None:
        covariance_storage = {}

    # Default retrieval elements
    if retrieval_elements is None:
        retrieval_elements = ["O3", "ground", "raman", "dispersion"]

    if micro_windows is None:
        micro_windows = np.array(
            [
                [267, 307],  # UV-1
                [309, 332],
            ]
        )  # UV-2
        micro_windows = rf.ArrayWithUnit(micro_windows, "nm")

    config_def = {
        "creator": creator.base.SaveToCommon,
        "order": [
            "solar_model",
            "input",
            "common",
            "scenario",
            "spec_win",
            "spectrum_sampling",
            "instrument",
            "atmosphere",
            "radiative_transfer",
            "forward_model",
            "retrieval",
        ],
        # Used in input, so needs to be created before that
        "solar_model": {
            "creator": OmiSolarReference3yr,
            "solar_reference_filename": solar_ref_3yr_filename,
            "num_channels": num_channels,
            "across_track_indexes": None,
        },
        "input": {
            # Filled in by derived config, not required
        },
        "common": {
            "creator": creator.base.SaveToCommon,
            "desc_band_name": channel_names,
            "hdf_band_name": channel_names,
            "band_reference": omi_ref_points,
            "num_channels": num_channels,
            "constants": {
                "creator": creator.common.DefaultConstants,
            },
        },
        # Wrap the common temporal/spatial values into the scenario block which will
        # be exposed to other creators
        "scenario": {
            # Place holders for the value required, must be filled in by derived config
            "creator": creator.base.SaveToCommon,
            "time": None,
            "latitude": None,
            "longitude": None,
            "surface_height": None,
            "altitude": None,  # same as surface_height
            "solar_distance": None,
            "solar_zenith": None,
            "solar_azimuth": None,
            "observation_zenith": None,
            "observation_azimuth": None,
            "relative_velocity": None,
            "spectral_coefficient": None,
            "stokes_coefficient": None,
        },
        "spec_win": {
            "creator": creator.forward_model.MicroWindowRanges,
            "full_ranges": omi_windows,
            "micro_windows": micro_windows,
        },
        "spectrum_sampling": {
            "creator": creator.forward_model.FixedSpacingSpectrumSampling,
            "high_res_spacing": rf.DoubleWithUnit(0.1, "nm"),
        },
        "instrument": {
            "creator": creator.instrument.IlsGratingInstrument,
            "ils_half_width": {
                "creator": creator.value.ArrayWithUnit,
                "value": np.array([0.63 / 2.0, 0.42 / 2.0]),
                "units": "nm",
            },
            "dispersion": {
                "creator": creator.instrument.DispersionPolynomial,
                "polynomial_coeffs": {
                    "creator": creator.value.NamedCommonValue,
                    "name": "spectral_coefficient",
                },
                "number_samples": {
                    "creator": creator.l1b.ValueFromLevel1b,
                    "field": "number_sample",
                },
                "spectral_variable": {
                    "creator": creator.l1b.ValueFromLevel1b,
                    "field": "spectral_variable",
                },
                "num_parameters": 1,
            },
            # Dynamic by across track index
            "ils_function": {
                "creator": OmiIlsTable,
                "ils_path": ils_path,
            },
            "instrument_correction": {
                "creator": creator.instrument.InstrumentCorrectionList,
                "corrections": [],
            },
        },
        "atmosphere": {
            "creator": creator.modifier.singleton(creator.atmosphere.AtmosphereCreator),
            "pressure": {
                "creator": creator.atmosphere.PressureGrid,
                "surface_pressure": None,
                "pressure_levels": None,
            },
            "temperature": {
                "creator": creator.atmosphere.TemperatureLevel,
                "temperature_profile": None,
            },
            "altitude": {
                "creator": creator.atmosphere.AltitudeHydrostatic,
            },
            "rayleigh": {
                "creator": creator.rayleigh.RayleighBodhaine,
            },
            # Implement in same way as CRiS
            "absorber": {
                "creator": creator.absorber.AbsorberXSec,
                "gases": ["O3", "H2O"],
                "default_gas_definition": {
                    "creator": creator.absorber.AbsorberGasDefinition,
                    "vmr": {
                        "creator": creator.absorber.AbsorberVmrLevel,
                        "vmr_profile": None,
                    },
                    "cross_section": {
                        "creator": creator.absorber.CrossSectionTableAscii,
                        "filename": lambda gas_name=None: rf.cross_section_filenames[
                            gas_name
                        ],
                        "conversion_factor": lambda gas_name=None: rf.cross_section_file_conversion_factors.get(
                            gas_name, 1.0
                        ),
                    },
                },
            },
            "relative_humidity": {
                "creator": creator.atmosphere.RelativeHumidity,
            },
            "ground": {
                "creator": creator.base.PickChild,
                "child": "lambertian",
                "lambertian": {
                    "creator": creator.ground.GroundLambertian,
                    "polynomial_coeffs": np.array([[0.1, 0]] * 2),
                    "which_retrieved": np.array(
                        [[True, False], [True, True]], dtype=bool
                    ),
                },
            },
        },
        "radiative_transfer": {
            "creator": creator.modifier.singleton(creator.rt.LidortRt),
            "num_streams": 4,
            "num_mom": 16,
            "use_thermal_emission": False,
            "use_solar_sources": True,
        },
        "forward_model": {
            "creator": creator.forward_model.ForwardModel,
            "spectrum_effect": {
                "creator": creator.forward_model.SpectrumEffectList,
                "effects": ["raman"],
                # raman below will use the solar model above
                "raman": {
                    "creator": creator.forward_model.RamanSiorisEffect,
                    "scale_factors": np.array([1.9, 1.9, 1.9]),
                    "albedo": np.array([0.0, 0.0, 0.0]),
                },
            },
        },
        "retrieval": {
            "creator": creator.retrieval.NLLSRetrieval,
            "retrieval_components": {
                "creator": creator.retrieval.SVObserverComponents,
                "include": retrieval_elements,
                "order": ["O3", "ground", "dispersion", "raman_sioris"],
            },
            "state_vector": {
                "creator": creator.retrieval.StateVector,
            },
            "initial_guess": {
                "creator": creator.retrieval.InitialGuessFromSV,
            },
            "a_priori": {
                "creator": creator.retrieval.AprioriFromIG,
            },
            "covariance": {
                "creator": creator.retrieval.CovarianceByComponent,
                "interstep_storage": covariance_storage,
                "values": {
                    # From Liu 2010 Table 1
                    "ground/lambertian": np.identity(num_channels * 2)
                    * np.array([2.5e-3, 1.0e-4] * num_channels),
                    "raman_sioris_1": diagonal_cov(1, 1.0),
                    "raman_sioris_2": diagonal_cov(1, 1.0),
                    "dispersion/UV1": np.identity(2) * np.array([4.0e-4, 1.0e-20]),
                    "dispersion/UV2": np.identity(2) * np.array([1.6e-5, 1.0e-20]),
                },
            },
            "solver": {
                "creator": creator.retrieval.NLLSSolverLM,
                "max_iteration": 20,
                # Setting a tolerance to 0 effectively disables it
                "dx_tol_abs": 0.0,
                "dx_tol_rel": 0.0,
                "g_tol_abs": 0.0,
                "g_tol_rel": 0.0,
                "verbose": True,
            },
        },
    }

    # Update parameters for solver from dictionary
    if solver_parameters is not None:
        config_def["retrieval"]["solver"].update(solver_parameters)

    return config_def

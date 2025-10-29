from enum import Enum
from netCDF4 import Dataset
import numpy as np
from pathlib import Path
import refractor.framework as rf
from refractor.muses.refractor_fm_object_creator import RefractorFmObjectCreator

from .clouds import BasicCloudProperties
from .absco_files import AbscoStub

from typing import Optional, Sequence, Tuple


class RtKind(Enum):
    PCA = "pca"
    Lidort = "lidort"


def setup_atmosphere_from_tropomi(
    raw_tropomi_state: dict,
    absco_gases: Sequence[Tuple[str, str]],
    albedo_key: str = "albedo_sim_band1",
    n_alb_terms: int = 1,
    wl_beg_nm: float = 2390,
    wl_end_nm: float = 2300,
    wn_spacing: float = 0.01,
    cloud_props: Optional[BasicCloudProperties] = None,
):
    """Set up an atmosphere and state vector component of Refractor from SRON TROPOMI atmosphere/land/orbit information

    Parameters
    ----------
    raw_tropomi_state
        A dictionary containing atmospheric, land, and orbit data. Usually this will be a dictionary returned by :func:`tropomi.load_tropomi_state`.

    absco_gases
        A list of tuples with a gas name (lower case) and path to an ABSCO file for that gas. The gases will be added as absorbers in the order they
        are presented in this list.

    albedo_key
        The key in ``raw_tropomi_state['orbit']`` that contains the albedo value to use.

    n_alb_terms
        How many terms to include in the albedo, must be at least one. The first one will be the value pointed to be ``albedo_key``,
        any extra ones will be set to 0 initially.

    wl_beg_nm
        The lowest waveLENGTH to simulate.

    wl_end_nm
        The greatest waveLENGTH to simulate.

    wn_space
        Simulation FREQUENCY grid spacing in wavenumbers.

    Returns
    -------
    atmosphere
        A refractor ``AtmosphereStandard`` object representing the absorbers, pressure, etc. given in the ``raw_tropomi_state``.

    state_vector
        The Refractor ``StateVector`` object that corresponds to the atmosphere.

    grid_points
        The frequency grid on which the RT calculations will be done.

    grid_units
        The units of ``grid_points``.
    """
    if n_alb_terms < 1:
        raise ValueError(f"n_alb_terms must be >= 1, value given was {n_alb_terms}")

    raw_tropomi_orbit = raw_tropomi_state["ancillary"]
    raw_tropomi_atm = raw_tropomi_state["atm"]

    # Orbital and land parameters
    latitude = rf.DoubleWithUnit(float(raw_tropomi_orbit["lat"]), "deg")
    surface_height = rf.DoubleWithUnit(float(raw_tropomi_orbit["surf_gph"]), "m")

    # This can stay a plain float. For now, use the albedo from the simulation file by default because that
    # gives results that match better with the SRON sims.
    albedo = raw_tropomi_orbit[albedo_key]

    # Atmospheric parameters
    pressure = rf.PressureSigma(
        raw_tropomi_atm["pressure"], raw_tropomi_atm["pressure"][-1]
    )
    temperature = rf.TemperatureLevel(raw_tropomi_atm["temperature"], pressure)
    altitude = rf.AltitudeHydrostatic(pressure, temperature, latitude, surface_height)
    # James added the rf.StateMappingLog() inputs when he made this capable of running the retrieval.
    # I believe that tells the retrieval to treat these terms in log space.
    vmr_profs = {
        "co": rf.AbsorberVmrLevel(
            pressure, raw_tropomi_atm["co"], "CO", rf.StateMappingLog()
        ),
        "ch4": rf.AbsorberVmrLevel(
            pressure, raw_tropomi_atm["ch4"], "CH4", rf.StateMappingLog()
        ),
        "h2o": rf.AbsorberVmrLevel(
            pressure, raw_tropomi_atm["h2o"], "H2O", rf.StateMappingLog()
        ),
        "hdo": rf.AbsorberVmrLevel(
            pressure, raw_tropomi_atm["hdo"], "HDO", rf.StateMappingLog()
        ),
    }

    # ------------------- #
    # SPECTROSCOPIC SETUP #
    # ------------------- #

    grid_units = "cm^-1"
    wn_beg = rf.DoubleWithUnit(wl_beg_nm, "nm").convert_wave(grid_units).value
    wn_end = rf.DoubleWithUnit(wl_end_nm, "nm").convert_wave(grid_units).value
    grid_points = rf.ArrayWithUnit(np.arange(wn_beg, wn_end, wn_spacing), grid_units)
    band_center = grid_points[grid_points.rows // 2 : grid_points.rows // 2 + 1]

    table_scale = 1.0
    cache_size = 5000
    interp_method = rf.AbscoAer.NEAREST_NEIGHBOR_WN

    # ----------- #
    # MODEL SETUP #
    # ----------- #

    vmrs = rf.vector_absorber_vmr()
    absorptions = []
    sv = rf.StateVector()
    state_elements = []

    for gas, absco_file in absco_gases:
        absco_obj = rf.AbscoAer(absco_file, table_scale, cache_size, interp_method)
        absco_stub_obj = AbscoStub(absco_obj)
        vmrs.push_back(vmr_profs[gas])
        absorptions.append(absco_stub_obj)
        sv.add_observer(vmr_profs[gas])
        # James also changed this line - not sure why
        state_elements.append(vmr_profs[gas].coefficient.value)

    altitudes = [altitude]

    constants = rf.DefaultConstant()

    num_sub_layers = 10

    absorber = rf.AbsorberAbsco(
        vmrs, pressure, temperature, altitudes, absorptions, constants, num_sub_layers
    )

    rayleigh = rf.RayleighBodhaine(pressure, altitudes, constants)

    relative_humidity = rf.RelativeHumidity(absorber, temperature, pressure)

    albedo_array = [albedo] + (n_alb_terms - 1) * [0]
    ground_poly = np.array([albedo_array])

    band_names = rf.vector_string()
    band_names.push_back("NIR")

    ground = rf.GroundLambertian(ground_poly, band_center, band_names)

    if cloud_props is not None:
        cld_pres_lev = cloud_props.cloud_pres_level(
            pressure.pressure_grid().value.value
        )
        pressure = rf.PressureWithCloudHandling(pressure, cld_pres_lev)
        cld_surface = cloud_props.cloud_albedo()
        ground = rf.GroundWithCloudHandling(ground, cld_surface)

    atmosphere = rf.AtmosphereStandard(
        absorber,
        pressure,
        temperature,
        rayleigh,
        relative_humidity,
        ground,
        altitudes,
        constants,
    )

    # ------------ #
    # STATE VECTOR #
    # ------------ #

    x = np.concatenate(state_elements)
    sv.update_state(x)

    sv.add_observer(atmosphere)

    return atmosphere, sv, grid_points, grid_units


def setup_tropomi_ils(xtrack_index: int, l1b_file: Path, isrf_file: Path, band=7):
    with Dataset(isrf_file) as ds:
        band_group = ds.groups[f"band_{band}"]
        delta_wavelength = band_group["delta_wavelength"][:].filled(np.nan)
        central_wavelength = band_group["central_wavelength"][:].filled(np.nan)
        isrf = band_group["isrf"][xtrack_index].filled(np.nan)

    with Dataset(l1b_file) as ds:
        l1b_wavelengths = rf.ArrayWithUnit(
            ds["BAND7_RADIANCE/STANDARD_MODE/INSTRUMENT/nominal_wavelength"][
                0, xtrack_index, :
            ],
            "nm",
        )

    return RefractorFmObjectCreator._construct_postconv_ils(
        central_wavelength=central_wavelength,
        delta_wavelength=delta_wavelength,
        isrf=isrf,
        hwhm=rf.DoubleWithUnit(0.36, "cm^-1"),
        sample_grid_spectral_domain=rf.SpectralDomain(l1b_wavelengths),
        band_name="BAND7",
        obs_band_name="BAND7",
    )


def run_simulation(
    raw_tropomi_state: dict,
    atmosphere: rf.AtmosphereStandard,
    grid_points: rf.ArrayWithUnit,
    grid_units: str,
    primary_absorber: str = "CO",
    band: str = "NIR",
    rt_mode: RtKind = RtKind.PCA,
    num_streams=None,
    ils: Optional[rf.Ils] = None,
    compute_jacobians: bool = False,
    solar_model_file: Optional[str] = None,
    cloud_props: Optional[BasicCloudProperties] = None,
    no_sim: bool = False,
) -> Tuple[rf.Spectrum, dict]:
    """Run a radiative transfer simulation using Refractor.

    Parameters
    ----------
    raw_tropomi_state
        A dictionary containing ancillary TROPOMI satellite data and atmospheric profiles. This should be
        created by :func:`tropomi.make_full_state_from_tccon` or `tropomi.load_tropomi_state`.

    atmosphere
        The Refractor atmosphere object to use in the simulation.

    grid_points
        The array of wavenumbers to simulate.

    grid_units
        The units for the grid_points array.

    primary_absorber
        The primary absorber for the simulation, typically "CO", "CH4", "H2O", or "HDO".

    band
        The spectral band name for the requested grid, such as "NIR" or "UV".

    rt_mode
        The radiative transfer mode to use. It can be either RtKind.PCA or RtKind.Lidort.

    num_streams
        The number of streams to use for the RT. By default, PCA uses 4 streams and 3 moments, while LIDORT
        uses 8 streams and 17 moments. If this value is not ``None``, then the number of moments will be
        ``2*num_streams+1`` for both RT types. (Note: these values come from examples in ReFRACtor notebooks
        or MUSES code, so the logic behind different choices is not clear to me.)

    ils
        The instrument line shape (ILS) to use. If None, an identity ILS is used.

    compute_jacobians
        Whether to compute the Jacobian matrices for the simulation. Default is False.

    solar_model_file
        The file path for the solar model data. Default is the ``SOLAR_MODEL_FILE`` defined in the :mod:`paths` module.
        Set this to ``None`` to skip using a solar model.

    no_sim
        If ``True``, all the components are set up, but the radiative transfer is not run, and the returned spectrum
        will be ``None``.

    Returns
    -------
    spectrum
        The :class:`rf.Spectrum` instance returned by the radiative transfer, or ``None`` if ``no_sim = True``.

    components
        A dictionary containing the various Refractor components set up for the simulation, specifically the
        radiative transfer ("rt"), forward model ("fm"), and solar model ("solar_model").
    """
    rt_mode = RtKind(rt_mode)

    raw_tropomi_orbit = raw_tropomi_state["ancillary"]
    time = rf.Time.parse_time(str(raw_tropomi_orbit["time"]))
    latitude = rf.DoubleWithUnit(float(raw_tropomi_orbit["lat"]), "deg")
    surface_height = rf.DoubleWithUnit(float(raw_tropomi_orbit["surf_gph"]), "m")
    solar_zenith = rf.DoubleWithUnit(raw_tropomi_orbit["sza"], "deg")
    observation_zenith = rf.DoubleWithUnit(raw_tropomi_orbit["vza"], "deg")
    relative_azimuth = rf.DoubleWithUnit(raw_tropomi_orbit["raa"], "deg")

    band_names = rf.vector_string()
    band_names.push_back(band)

    constants = rf.DefaultConstant()

    # ------------------ #
    # RADIATIVE TRANSFER #
    # ------------------ #

    num_mom = 2 * 8 + 1
    use_solar_sources = True
    use_thermal_emission = False
    do_3m_correction = False

    # stokes = rf.StokesCoefficientConstant([[1.0, 0.0, 0.0, 0.0]])
    # This is what we use in MUSES I think so keeping the single element for now.
    stokes = rf.StokesCoefficientConstant([[1.0]])

    # Extract just the values in degrees for the RT
    rt_sza = np.array([solar_zenith.convert("deg").value])
    rt_oza = np.array([observation_zenith.convert("deg").value])
    rt_raz = np.array([relative_azimuth.convert("deg").value])

    if rt_mode == RtKind.PCA:
        if num_streams is None:
            num_streams = 4
            num_mom = 3
        else:
            num_mom = 2 * num_streams + 1
        print(f"Using PCA RT with {num_streams} streams and {num_mom} moments")
        bin_method = rf.PCABinning.UVVSWIR_V4
        num_bins = 11
        num_eofs = 4
        first_order_rt = None  # Use built in
        rt = rf.PCARt(
            atmosphere,
            primary_absorber,
            bin_method,
            num_bins,
            num_eofs,
            stokes,
            rt_sza,
            rt_oza,
            rt_raz,
            num_streams,
            num_mom,
            use_solar_sources,
            use_thermal_emission,
            do_3m_correction,
            first_order_rt,
        )

        # Copied these settings from MUSES code
        lid_interface = rt.lidort.rt_driver.lidort_interface
        lid_interface.lidort_modin.mbool().ts_do_focorr(False)
        lid_interface.lidort_modin.mbool().ts_do_focorr_nadir(False)
        lid_interface.lidort_modin.mbool().ts_do_focorr_outgoing(False)
        lid_interface.lidort_modin.mbool().ts_do_rayleigh_only(False)
        lid_interface.lidort_modin.mbool().ts_do_double_convtest(False)
        lid_interface.lidort_modin.mbool().ts_do_deltam_scaling(False)
        lid_interface.lidort_modin.mchapman().ts_earth_radius(6371.0)
    elif rt_mode == RtKind.Lidort:
        if num_streams is None:
            num_streams = 8
        num_mom = 2 * num_streams + 1
        print(f"Using Lidort RT with {num_streams} streams and {num_mom} moments")
        stokes = rf.StokesCoefficientConstant([[1.0, 0.0, 0.0, 0.0]])
        pure_nadir = False
        multiple_scattering_only = False
        # print(type(atmosphere), type(stokes), type(solar_zenith), type(observation_zenith), type(relative_azimuth), type(pure_nadir), type(num_streams), type(num_mom), type(multiple_scattering_only))
        rt = rf.LidortRt(
            atmosphere,
            stokes,
            rt_sza,
            rt_oza,
            rt_raz,
            pure_nadir,
            num_streams,
            num_mom,
            multiple_scattering_only,
        )
    else:
        raise NotImplementedError(f"{rt_mode=}")

    # ---------- #
    # INSTRUMENT #
    # ---------- #
    # Defines the low resolution / instrument grid
    sample_grid = rf.SampleGridSpectralDomain(
        rf.SpectralDomain(grid_points), band_names[0]
    )

    # Define the indentity ILS which doesn't apply a convolution
    if ils is None:
        ils = rf.IdentityIls(sample_grid)

        # By default this sets units in nm, set extension to correct units
        # TODO fix IdentityIls to accept a units argument in the constructor
        ils.high_res_extension(rf.DoubleWithUnit(0, grid_units))

    ils_vec = [ils]

    instrument = rf.IlsInstrument(ils_vec)

    # ----------- #
    # SOLAR MODEL #
    # ----------- #
    if solar_model_file is not None:
        solar_model_data = rf.HdfFile(solar_model_file)

        do_doppler_shift = True
        doppler = rf.SolarDopplerShiftPolynomial(
            time,
            latitude,
            solar_zenith,
            relative_azimuth,
            surface_height,
            constants,
            do_doppler_shift,
        )

        absorption = rf.SolarAbsorptionTable(
            solar_model_data, "/Solar/Absorption/Absorption_1"
        )

        convert_from_photon = False
        continuum = rf.SolarContinuumTable(
            solar_model_data, "/Solar/Continuum/Continuum_1", convert_from_photon
        )

        solar_model = rf.SolarAbsorptionAndContinuum(doppler, absorption, continuum)
    else:
        solar_model = None

    # ---------- #
    # SIMULATION #
    # ---------- #

    # Defines the extents of the high resolution grid
    win_bounds = np.array([np.min(grid_points.value), np.max(grid_points.value)])[
        np.newaxis, np.newaxis, :
    ]
    spec_win = rf.SpectralWindowRange(rf.ArrayWithUnit(win_bounds, grid_points.units))

    # Defines the high resolution grid
    spectrum_sampling = rf.SpectrumSamplingFixedSpacing(
        rf.ArrayWithUnit(np.array([0.01]), grid_units)
    )

    # Add solar model as a specrtrum effect for use by the forward model
    spectrum_effect = rf.vector_vector_spectrum_effect()

    per_channel_eff = rf.vector_spectrum_effect()
    if solar_model is not None:
        per_channel_eff.push_back(solar_model)

    spectrum_effect.push_back(per_channel_eff)

    forward_model = rf.StandardForwardModel(
        instrument, spec_win, rt, spectrum_sampling, spectrum_effect
    )
    forward_model.setup_grid()

    if cloud_props is not None:
        forward_model = rf.ForwardModelWithCloudHandling(
            forward_model, cloud_props.cloud_fraction()
        )
        forward_model.add_cloud_handling_object(atmosphere.pressure)
        forward_model.add_cloud_handling_object(atmosphere.ground)

    if no_sim:
        spectrum = None
    else:
        spectrum = forward_model.radiance_all(compute_jacobians)
    components = {"rt": rt, "fm": forward_model, "solar_model": solar_model}
    return spectrum, components

import numpy as np

from refractor.framework.factory import creator, param  # type: ignore
import refractor.framework as rf  # type: ignore

from refractor.omi.level1 import (
    OmiLevel1RadianceFile,
    OmiLevel1IrradianceFile,
    OmiLevel1Reflectance,
)

from .base_config import base_config_definition, num_channels


class BadSampleFromL1b(creator.base.Creator):
    l1b = param.InstanceOf(rf.Level1b)

    def create(self, **kwargs):
        l1b = self.l1b()
        num_channels = l1b.number_spectrometer()

        bad_sample_list = []
        for chan_idx in range(num_channels):
            bad_sample_list.append(l1b.bad_sample_mask(chan_idx))

        return bad_sample_list


class OmiLevel1Creator(creator.base.Creator):
    input_filename = param.Scalar(str)
    along_track_index = param.Scalar(int)
    across_track_indexes = param.Iterable(int)

    def create(self, **kwargs):
        return OmiLevel1RadianceFile(
            self.input_filename(), self.along_track_index(), self.across_track_indexes()
        )


class OmiReflectanceCreator(creator.base.Creator):
    input_filename = param.Scalar(str)
    along_track_index = param.Scalar(int)
    across_track_indexes = param.Iterable(int)
    solar_model = param.Iterable(rf.SolarModel)

    def create(self, **kwargs):
        return OmiLevel1Reflectance(
            self.input_filename(),
            self.along_track_index(),
            self.across_track_indexes(),
            self.solar_model(),
        )


def retrieval_base_config_definition(
    l1b_file, along_track_index, across_track_indexes, irradiance_file=None, **kwargs
):
    config_def = base_config_definition(**kwargs)

    # Override solar model to use OMI Irradiance file values
    if irradiance_file is not None:
        irr_obj = OmiLevel1IrradianceFile(irradiance_file, across_track_indexes)

        grid_values = []
        irradiance_values = []
        for chan_idx in range(irr_obj.number_channel()):
            chan_grid = irr_obj.sample_grid(chan_idx)
            chan_irr = irr_obj.irradiance(chan_idx)

            grid_values.append(rf.ArrayWithUnit(chan_grid.data, chan_grid.units))
            irradiance_values.append(rf.ArrayWithUnit(chan_irr.data, chan_irr.units))

        config_def["solar_model"] = {
            "creator": creator.solar_model.SolarReferenceSpectrum,
            "grid": grid_values,
            "irradiance": irradiance_values,
            "num_channels": num_channels,
        }
    else:
        config_def["solar_model"]["across_track_indexes"] = np.array(
            across_track_indexes
        )

    # Create doppler for solar model with a seperate L1B object since it needs
    # to be created before the L1B reflectance, but needs access to values from
    # the L1B object
    config_def["solar_model"]["doppler"] = {
        "creator": creator.solar_model.SolarDopplerShiftPolynomialFromL1b,
        "num_channels": num_channels,
        "l1b": {
            "creator": OmiLevel1Creator,
            "input_filename": l1b_file,
            "along_track_index": along_track_index,
            "across_track_indexes": across_track_indexes,
        },
    }

    # OmiReflectanceCreator depends on the solar model which needs to be created first
    config_def["input"] = {
        "creator": creator.base.SaveToCommon,
        "l1b": {
            "creator": creator.modifier.singleton(OmiReflectanceCreator),
            "input_filename": l1b_file,
            "along_track_index": along_track_index,
            "across_track_indexes": across_track_indexes,
        },
        "along_track_index": along_track_index,
        "across_track_indexes": np.array(across_track_indexes),
    }

    # Add masking of bad samples from the L1B file
    config_def["spec_win"]["bad_sample_mask"] = BadSampleFromL1b

    # Set up scenario values that have the same name as they are named in the L1B
    config_def["scenario"] = {
        "creator": creator.scenario.ScenarioFromL1b,
    }

    return config_def


def retrieval_muses_config_definition(
    l1b_file, along_track_index, across_track_indexes, osp_dir, gmao_dir=None, **kwargs
):
    config_def = retrieval_base_config_definition(
        l1b_file, along_track_index, across_track_indexes, **kwargs
    )

    # OSP information
    config_def["input"]["osp_directory"] = osp_dir
    config_def["input"]["osp_instrument_name"] = "OMI-AIRS-v8"

    # Pressure
    config_def["atmosphere"]["pressure"]["creator"] = creator.muses.PressureGridMUSES

    # Temperature
    if gmao_dir is not None:
        config_def["input"]["gmao_directory"] = gmao_dir

        config_def["atmosphere"]["temperature"]["creator"] = (
            creator.atmosphere.TemperatureLevel
        )
        config_def["atmosphere"]["temperature"]["pressure"] = creator.gmao.PressureGMAO
        config_def["atmosphere"]["temperature"]["temperature_profile"] = (
            creator.gmao.TemperatureProfileGMAO
        )

        config_def["atmosphere"]["pressure"]["surface_pressure"] = {
            "creator": creator.atmosphere.SurfacePressureFromAltitude,
            "pressure": creator.gmao.PressureGMAO,
            "temperature_profile": creator.gmao.TemperatureProfileGMAO,
        }

    else:
        config_def["atmosphere"]["temperature"]["creator"] = (
            creator.muses.TemperatureMUSES
        )

    # Process all gases VMR using MUSES values
    gas_def = config_def["atmosphere"]["absorber"]["default_gas_definition"]
    gas_def["vmr"]["creator"] = creator.muses.AbsorberVmrMUSES

    # Covariance
    config_def["retrieval"]["covariance"]["creator"] = creator.muses.CovarianceMUSES

    return config_def

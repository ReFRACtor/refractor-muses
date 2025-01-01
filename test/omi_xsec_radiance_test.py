import os
from time import perf_counter
from test_support import *

import numpy.testing as npt
import netCDF4

from refractor.framework import ComparisonExecutor, write_shelve
from refractor.framework import load_config_module, find_config_function
from refractor.framework.factory import process_config

example_xsec_config_filename = "simulation_example_config.py"
expected_xsec_results_filename = (
    f"{omi_test_expected_results_dir}/radiance_comparison/expected_radiance_xsec.nc"
)


def compare_fm(config_filename, expt_results_filename):
    expt_data = netCDF4.Dataset(expt_results_filename)

    exc = ComparisonExecutor(config_filename)
    exc.execute_simulation()

    if len(exc.captured_radiances.convolved_spectrum) == 0:
        raise Exception("No convolved radiances captured")

    if len(exc.captured_radiances.high_res_spectrum) == 0:
        raise Exception("No high resolution spectrum captured")

    for spec_idx, (conv_spec, hr_spec) in enumerate(
        zip(
            exc.captured_radiances.convolved_spectrum,
            exc.captured_radiances.high_res_spectrum,
        )
    ):
        expt_conv_rad = expt_data["Channel_{}/Convolved/radiance".format(spec_idx + 1)][
            :
        ]
        expt_hr_rad = expt_data[
            "Channel_{}/Monochromatic/radiance".format(spec_idx + 1)
        ][:]

        calc_conv_rad = conv_spec.spectral_range.data
        calc_hr_rad = hr_spec.spectral_range.data

        npt.assert_allclose(expt_conv_rad, calc_conv_rad, atol=1e-6)
        npt.assert_allclose(expt_hr_rad, calc_hr_rad, atol=1e-6)


def write_expected(config_filename, expt_results_filename):
    expt_data = netCDF4.Dataset(expt_results_filename, "w")

    exc = ComparisonExecutor(config_filename)
    exc.execute_simulation()

    if len(exc.captured_radiances.convolved_spectrum) == 0:
        raise Exception("No convolved radiances captured")

    if len(exc.captured_radiances.high_res_spectrum) == 0:
        raise Exception("No high resolution spectrum captured")

    for spec_idx, (conv_spec, hr_spec) in enumerate(
        zip(
            exc.captured_radiances.convolved_spectrum,
            exc.captured_radiances.high_res_spectrum,
        )
    ):
        mono_grid_dim = f"monochromatic_grid_{spec_idx+1}"
        conv_grid_dim = f"convolved_grid_{spec_idx+1}"

        expt_data.createDimension(mono_grid_dim, hr_spec.spectral_domain.data.shape[0])
        expt_data.createDimension(
            conv_grid_dim, conv_spec.spectral_domain.data.shape[0]
        )

        calc_conv_rad = conv_spec.spectral_range.data
        calc_hr_rad = hr_spec.spectral_range.data

        mono_grid_var = expt_data.createVariable(
            f"Channel_{spec_idx+1}/Monochromatic/grid", float, mono_grid_dim
        )
        mono_rad_var = expt_data.createVariable(
            f"Channel_{spec_idx+1}/Monochromatic/radiance", float, mono_grid_dim
        )

        mono_grid_var[:] = hr_spec.spectral_domain.data
        mono_grid_var.units = hr_spec.spectral_domain.units.name

        mono_rad_var[:] = hr_spec.spectral_range.data
        mono_rad_var.units = hr_spec.spectral_range.units.name

        conv_grid_var = expt_data.createVariable(
            f"Channel_{spec_idx+1}/Convolved/grid", float, conv_grid_dim
        )
        conv_rad_var = expt_data.createVariable(
            f"Channel_{spec_idx+1}/Convolved/radiance", float, conv_grid_dim
        )

        conv_grid_var[:] = conv_spec.spectral_domain.data
        conv_grid_var.units = conv_spec.spectral_domain.units.name

        conv_rad_var[:] = conv_spec.spectral_range.data
        conv_rad_var.units = conv_spec.spectral_range.units.name


@capture_test
def test_capture_atmosphere(omi_config_dir, osp_dir, gmao_dir):
    config_filename = os.path.join(omi_config_dir, "retrieval_example_config.py")
    strategy_filename = os.path.join(omi_config_dir, "strategy.py")
    serialized_fn = os.path.join(test_in_dir, "atmosphere.bin.gz")

    step_index = 0
    exc = ComparisonExecutor(config_filename)

    step_keywords = exc.strategy_list[step_index]
    exec_inst = exc.config_instance(**step_keywords)
    config_inst = exec_inst.file_config

    write_shelve(serialized_fn, config_inst.atmosphere)


@long_test
def test_multi_radiance(omi_config_dir):
    "Test running radiative transfer multiple times in a row and gather timing information"

    test_start = perf_counter()
    config_filename = os.path.join(omi_config_dir, example_xsec_config_filename)
    config_module = load_config_module(config_filename)
    config_func = find_config_function(config_module)
    retrieval_example_config_def = config_func()
    config_inst = process_config(retrieval_example_config_def)
    loaded = perf_counter()
    print(f"Loading {config_filename} took {loaded - test_start:.2f} seconds")
    fm = config_inst.forward_model
    channel_index = 0
    num_repeats = 10
    repeat_block_start = perf_counter()
    for i in range(num_repeats):
        before_rad = perf_counter()
        test_rad = fm.radiance(channel_index)
        after_rad = perf_counter()
        print(f"Radiance call {i + 1} took {after_rad - before_rad:.2f} seconds")
    repeat_block_end = perf_counter()
    repeat_block_diff = repeat_block_end - repeat_block_start
    print(f"Block of radiance calls took {repeat_block_diff:.2f}")
    print(f"Avg. time / call was {repeat_block_diff / num_repeats:.2f} ")


@capture_test
def test_capture_expected_xsec(omi_config_dir):
    "Overwrite expected results for example cross section radiance test"

    config_filename = os.path.join(omi_config_dir, example_xsec_config_filename)
    write_expected(config_filename, expected_xsec_results_filename)


def test_example_xsec(omi_config_dir):
    "Test running example simulation using cross section tables and compare to expected output"

    config_filename = f"{omi_config_dir}/{example_xsec_config_filename}"
    compare_fm(config_filename, expected_xsec_results_filename)

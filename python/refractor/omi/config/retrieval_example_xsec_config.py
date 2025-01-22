import os

from refractor.framework import creator, refractor_config  # type: ignore

from .retrieval_base_config import retrieval_muses_config_definition

if "REFRACTOR_TEST_DATA" in os.environ:
    test_in_dir = f"{os.environ['REFRACTOR_TEST_DATA']}/omi/in/"
else:
    test_in_dir = os.path.abspath(
        f"{os.path.dirname(__file__)}/../../../../../refractor_test_data/omi/in"
    )


@refractor_config
def config(**kwargs):
    osp_dir = os.environ["MUSES_OSP_PATH"]

    l1b_filename = f"{test_in_dir}/OMI-Aura_L1-OML1BRUG_2016m0414t2324-o62498_v003-2016m0415t050532.he4"
    irr_filename = f"{test_in_dir}/OMI-Aura_L1-OML1BIRR_2016m0414t0337-o62486_v003-2016m0621t174011.he4"

    along_track_index = 786
    across_track_indexes = [26, 52]  # UV1, UV2

    config_def = retrieval_muses_config_definition(
        l1b_filename,
        along_track_index,
        across_track_indexes,
        osp_dir,
        irradiance_file=irr_filename,
        **kwargs,
    )

    config_def["atmosphere"]["absorber"]["creator"] = creator.absorber.AbsorberXSec
    config_def["atmosphere"]["absorber"]["default_gas_definition"]["creator"] = (
        creator.absorber.CrossSectionGasDefinition
    )

    return config_def

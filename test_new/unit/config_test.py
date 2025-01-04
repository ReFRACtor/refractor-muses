from refractor.framework.test import dummy_env_var, check_config_loading


@dummy_env_var("MUSES_OSP_PATH")
@dummy_env_var("ABSCO_PATH")
@dummy_env_var("OMI_RUG_FILENAME")
@dummy_env_var("OMI_SIM_FILENAME")
@dummy_env_var("OMI_ALONG_TRACK_INDEX", "0")
@dummy_env_var("OMI_ACROSS_TRACK_INDEXES", "0,0")
def test_config_load(omi_config_dir, omi_test_in_dir):
    # Dummy values for configs that need arguments
    function_args = {
        # This file needs to exist since it is loaded in the config function itself
        "uip_config": (omi_test_in_dir.parent / "raman/uip-FM.sav",)
    }

    filenames = omi_config_dir.glob("*.py")
    check_config_loading(filenames, function_args)

import os
import re

from refractor.framework import refractor_config  # type: ignore

from .retrieval_base_config import retrieval_muses_config_definition


def get_input(name, required=True):
    if name not in os.environ:
        if required:
            raise Exception(
                f"The environment variable {name} is required to process this configuration"
            )
        else:
            return None
    else:
        return os.environ[name]


@refractor_config
def retrieval_run_config(**kwargs):
    osp_dir = get_input("MUSES_OSP_PATH")

    l1b_filename = get_input("OMI_RUG_FILENAME")
    irr_filename = get_input("OMI_IRR_FILENAME", required=False)

    gmao_dir = get_input("GMAO_PATH", required=False)

    along_track_index = int(get_input("OMI_ALONG_TRACK_INDEX"))

    across_track_string = get_input("OMI_ACROSS_TRACK_INDEXES")

    across_track_indexes = [int(v) for v in re.split("[,\s]+", across_track_string)]

    if not len(across_track_indexes) == 2:
        raise Exception(
            "Expected 2 integers for across track indexes seperated by commas or spaces"
        )

    return retrieval_muses_config_definition(
        l1b_filename,
        along_track_index,
        across_track_indexes,
        osp_dir,
        irradiance_file=irr_filename,
        gmao_dir=gmao_dir,
        **kwargs,
    )

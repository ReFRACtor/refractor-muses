from __future__ import annotations
import refractor.muses_py as mpy  # type: ignore
from loguru import logger
import json
import importlib

# This has various things that we use to have in refractor.muses. We
# pulled all this into refractor.old_py_retrieve_wrapper just so we
# have a clean separation, and also because most of this is just for
# backwards testing, i.e., it really does belong in
# old_py_retrieve_wrapper, but needs to be in refractor.muses objects

def create_retrieval_output_json():
    '''muses-py has a lot of hard coded things related to the species
    names and netcdf output.  It would be good a some point to just
    replace this all with a better thought out output format. But for
    now, we need to support the existing output format.
    '''
    # TODO - Replace with better thought out output format
    if not importlib.resources.is_resource(
            "refractor.muses", "retrieval_output.json"
    ):
        if not mpy.have_muses_py:
            raise RuntimeError(
                "Require muses-py to create the file retrieval_output.json"
            )
        d = {
            "cdf_var_attributes": mpy.cdf_var_attributes,
            "groupvarnames": mpy.cdf_var_names(),
            "exact_cased_variable_names": mpy.cdf_var_map(),
        }
        with importlib.resources.path(
                "refractor.muses", "retrieval_output.json"
        ) as fspath:
            logger.info(f"Creating the file {fspath}")
            with open(fspath, "w") as fh:
                json.dump(d, fh, indent=4)
    
__all__ = ["create_retrieval_output_json",]

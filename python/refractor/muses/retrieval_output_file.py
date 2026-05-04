from __future__ import annotations
from loguru import logger
import importlib
import json
import os
import netCDF4
from pathlib import Path
from .declarative_output import DeclarativeOutputHandle
from typing import Any, Callable, Sequence


class RetrievalOutputFile(DeclarativeOutputHandle):
    """This writes out a retrieval output file.

    Note that py-retrieve has a lot hard coded things related to the species
    names and netcdf output. We will likely replace this at some point with
    a TemplatedOutput file, where we use an existing netCDF4 template file
    that defines the structure. However, as a transition we need to match
    the old output file.

    TODO Replace this with TemplatedOutput
    """

    def __init__(
        self,
        output_filename: str | os.PathLike[str],
    ) -> None:
        self.output_filename = Path(output_filename)
        self.variable_data_functions: dict = {}
        self.attribute_value_functions: dict = {}

        # muses-py has a lot of hard coded things related to the
        # species names and netcdf output.  It would be good a some
        # point to just replace this all with a better thought out
        # output format. But for now, we need to support the existing
        # output format.

        # So we don't depend on muses_py, we save the variable to a json file.
        # Only need muses_py to generate this or update it. We just create this file
        # if not available, so you can manually delete this to force it to be recreated.
        if not importlib.resources.is_resource(
            "refractor.muses", "retrieval_output.json"
        ):
            from refractor.old_py_retrieve_wrapper import create_retrieval_output_json

            create_retrieval_output_json()
        d = json.loads(
            importlib.resources.read_text("refractor.muses", "retrieval_output.json")
        )
        self.cdf_var_attributes: dict[str, dict[str, str | float]] = d[
            "cdf_var_attributes"
        ]
        self.groupvarnames: list[list[str]] = d["groupvarnames"]
        self.exact_cased_variable_names: dict[str, str] = d[
            "exact_cased_variable_names"
        ]

    def register_instances(self, obj_list: Sequence[object]) -> None:
        "Calls each class instance to register their output into this class"

        for obj in obj_list:
            if not isinstance(obj, object):
                raise Exception(f"Object to register should be class instance: {obj}")

            if not hasattr(obj, "register_output"):
                raise Exception(
                    f"Class instance should contain the register_output function: {obj}"
                )

            obj.register_output(self)

    def register_dataset(self, name: str, function: Callable[..., Any]) -> None:
        "Alias for register variable"

        self.register_variable(name, function)

    def register_variable(self, name: str, function: Callable[..., Any]) -> None:
        """Registers a simple dataset variable"""
        self.variable_data_functions[name] = function

    def register_attribute(self, name: str, value: Any) -> None:
        """Registers an attribute"""
        if hasattr(value, "__call__"):
            self.attribute_value_functions[name] = value()
        else:
            self.attribute_value_functions[name] = value

    def write(self) -> None:
        """Calls all registered functions to write the NetCDF4 output
        file Variables must be defined in the template NetCDF4 file.
        Attributes will be written even if they are not defined in the
        template file
        """

        logger.info(f"Writing to output file: {self.output_filename}")
        fh = netCDF4.Dataset(self.output_filename, "w")

        try:
            # self._write_attributes(output_contents)
            # self._write_variables(output_contents)
            pass
        finally:
            fh.close()


__all__ = [
    "RetrievalOutputFile",
]

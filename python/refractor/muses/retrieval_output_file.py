from __future__ import annotations
from loguru import logger
import importlib
import datetime
import pytz
import json
import os
import xarray as xr
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

    def _write_variables(self) -> None:
        data_var = {}
        for var_name, var_data_func in self.variable_data_functions.items():
            # A variable can optionally define a creator function to
            # create the variable where data is written.
            #
            # We currently ignore this, but leave this in place for now in
            # case we want to add this at some point
            # creator_func = (
            #    var_data_func._creator if hasattr(var_data_func, "_creator") else None  # noqa: SLF001
            # )

            # A modifier allows direct modifications of the variable
            # object before data values are set
            modifier_func = (
                var_data_func._modifier if hasattr(var_data_func, "_modifier") else None  # noqa: SLF001
            )

            # Obtain the data from the registered function/method
            data = var_data_func()

            # Skip setting values if data is None
            if data is None:
                continue

            # Standard names for the grids, with for some reason "grid_1" being
            # called "one" in py-retrieve. Don't know if this matters, but match
            # that behavior
            if hasattr(data, "shape"):
                dim_name = [f"grid_{d}" if d != 1 else "one" for d in data.shape]
            else:
                dim_name = []

            vname = Path(var_name)
            if vname.parent != Path("/"):
                raise RuntimeError("Don't handle groups yet")

            # xarray can't handle "/" in a name
            data_var[vname.name] = xr.DataArray(
                data=data, dims=dim_name, attrs=({"Units": "()"})
            )

            # Note that xarray puts in a fillvalue for each field (defaults to
            # NaN. This isn't actually a bad thing, but to match the old files
            # we need to remove this.
            data_var[vname.name].encoding = {"_FillValue": None}

            # Allow direct modifications to the variable object itself
            # such as dynamic attribute modifications
            if modifier_func is not None:
                modifier_func(var_data_func.__self__, data_var[var_name.name], var_name)

        # Xarray Dataset can't be updated, not sure why this constraint. But we can
        # create a modified version.
        self.ds: xr.Dataset = self.ds.assign(data_var)

    def write(self) -> None:
        """Calls all registered functions to write the NetCDF4 output
        file Variables must be defined in the template NetCDF4 file.
        Attributes will be written even if they are not defined in the
        template file
        """

        logger.info(f"Writing to output file: {self.output_filename}")

        # It is convenient in testing to have a fixed creation_date, just so we can
        # compare files created at different times with h5diff and have them compare as
        # identical
        cdate = os.environ.get(
            "MUSES_FAKE_CREATION_DATE",
            datetime.datetime.now(tz=pytz.utc).strftime("%Y%m%dT%H%M%SZ"),
        )
        self.ds = xr.Dataset(attrs={"creation_date": cdate})
        try:
            # self._write_attributes(output_contents)
            self._write_variables()
        finally:
            # Note that xarray puts in a fillvalue for each field (defaults to
            # NaN. This isn't actually a bad thing, but to match the old files
            # we need to remove this
            self.ds.to_netcdf(self.output_filename)


__all__ = [
    "RetrievalOutputFile",
]

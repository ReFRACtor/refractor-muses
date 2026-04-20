import os
import re
from loguru import logger
from shutil import copyfile
import netCDF4
import shutil
import tempfile
from nco import Nco


def call_ncks(input_filename, output_filename, options, overwrite=False):
    nco = Nco()

    use_temp = False
    if os.path.realpath(input_filename) == os.path.realpath(output_filename):
        if not overwrite:
            raise IOError(f"Will not overwrite original file {input_filename}")

        temp_fd, dest_filename = tempfile.mkstemp()
        use_temp = True
    else:
        dest_filename = output_filename

    nco.ncks(input=input_filename, output=dest_filename, options=options)

    if not os.path.exists(dest_filename):
        raise IOError(f"ncks failed to create {dest_filename} from {input_filename}")

    if use_temp:
        shutil.copyfile(dest_filename, output_filename)
        os.remove(dest_filename)
        os.close(temp_fd)


def remove_unlimited_dims(input_filename, output_filename, **kwargs):
    ncks_options = ["--history --fix_rec_dmn all"]

    return call_ncks(input_filename, output_filename, options=ncks_options, **kwargs)


def netcdf_variables(item: netCDF4.Dataset | netCDF4.Variable | netCDF4.Group):
    """
    Returns all netCDF variables in a flat generator
    """

    if isinstance(item, netCDF4.Variable):
        yield item
    elif isinstance(item, netCDF4.Dataset) or isinstance(item, netCDF4.Group):
        for child_item in item.variables.values():
            yield from netcdf_variables(child_item)
        for child_item in item.groups.values():
            yield from netcdf_variables(child_item)
    else:
        raise Exception(f"Unsure how to handle data type: {type(item)}")


def netcdf_full_path(item: netCDF4.Dataset | netCDF4.Variable | netCDF4.Group) -> str:
    """A way to provide a consistent full path to a variable by going
    up the parent tree and appending strings
    """

    if isinstance(item, netCDF4.Variable):
        return os.path.join(netcdf_full_path(item.group()), item.name)
    elif isinstance(item, netCDF4.Group):
        return os.path.join(netcdf_full_path(item.parent), item.name)
    elif isinstance(item, netCDF4.Dataset):
        return item.name
    else:
        raise Exception(f"Unsure how to handle data type: {type(item)}")


class TemplatedOutput:
    """Writes data into an output product from a set of call
    functions registered to handle certain aspects of the process such
    as dimensions, attributes and variables. The variables are
    expected to already exist by the register_dataset callbacks. This
    is intended to be used with an existing netCDF4 template file that
    defines the structure. If creating a file from scratch then the
    variables need to be created before the can be registered.

    Parameters
    ----------
    template_filename : str
        Filename of the NetCDF template file describing the contents of the product

    output_filename : str
        Filename to write output product into

    """

    def __init__(self, template_filename, output_filename, grid_mapping=None) -> None:
        self.output_filename = output_filename
        self.grid_mapping = grid_mapping

        self.variable_data_functions = {}
        self.attribute_value_functions = {}

        copyfile(template_filename, self.output_filename)

    def _template_variables_list(self, output_contents):
        "Loads fill values from the template file"

        template_variables = [
            netcdf_full_path(var) for var in netcdf_variables(output_contents)
        ]

        return template_variables

    def register_instances(self, obj_list):
        "Calls each class instance to register their output into this class"

        for obj in obj_list:
            if not isinstance(obj, object):
                raise Exception(f"Object to register should be class instance: {obj}")

            if not hasattr(obj, "register_output"):
                raise Exception(
                    f"Class instance should contain the register_output function: {obj}"
                )

            obj.register_output(self)

    def register_dataset(self, name, function):
        "Alias for register variable"

        return self.register_variable(name, function)

    def register_variable(self, name, function):
        """
        Registers a simple dataset variable

        Parameters
        ----------
        name : str
            Full dataset variable name, with slashes defining the group heirarchy

        function : function
            Call back function that returns the data being registered, it takes no arguments

        """

        self.variable_data_functions[name] = function

    def register_attribute(self, name, value):
        """
        Registers an attribute

        Parameters
        ----------
        name : str
            Full attribute name, with slashes defining the group and variables  heirarchy

        value : value
            Value of the attribute. Could be a scalar or np array
        """

        if hasattr(value, "__call__"):
            self.attribute_value_functions[name] = value()
        else:
            self.attribute_value_functions[name] = value

    def get_variable(self, output_contents, var_name):
        """
        Gets NetCDF4 variable given a variable's full name (including groups)
        Returns None if not present instead of throwing a KeyError
        """

        try:
            return output_contents[var_name]
        except (IndexError, KeyError):
            # IndexError is for just the variable name, KeyError for name with a group in it
            return None

    def get_attr_parent(self, rootgrp, attr_name):
        """
        Gets attributes NetCDF4 variable given an attribute's full name (including groups)
        """
        attr_name = attr_name.strip("/")
        toks = re.split("/+", attr_name)
        parent = rootgrp
        grandparent = rootgrp
        for gname in toks[:-1]:
            parent = grandparent[gname]
            grandparent = parent

        return parent, toks[-1]

    def _write_attributes(self, output_contents):
        for attr_full_name, value in self.attribute_value_functions.items():
            try:
                parent, attr_name = self.get_attr_parent(
                    output_contents, attr_full_name
                )
                if isinstance(value, str):
                    parent.setncattr_string(attr_name, value)
                else:
                    parent.setncattr(attr_name, value)
            except AttributeError as exc:
                raise AttributeError(
                    f"Error setting attribute {attr_full_name} with value {value} : {exc}"
                ) from exc

    def _write_variables(self, output_contents):
        template_variables = self._template_variables_list(output_contents)

        variables_written = []

        for var_name, var_data_func in self.variable_data_functions.items():
            # A variable can optionally define a creator function to create the variable
            # where data is written.
            creator_func = (
                hasattr(var_data_func, "_creator") and var_data_func._creator or None  # noqa: SLF001
            )

            # A modifier allows direct modifications of the variable object before data values are set
            modifier_func = (
                hasattr(var_data_func, "_modifier") and var_data_func._modifier or None  # noqa: SLF001
            )

            # See if the variable is already defined in the output file
            var = self.get_variable(output_contents, var_name)

            # If the creator_func is None and the variable is not defined existing file, then we
            # can not set a value
            if var is None and creator_func is None:
                raise KeyError(
                    f"{var_name} not present in {self.output_filename} and no creator function provided"
                )

            # Use a templated definition of a variable instead of the creator function, even if it is defined
            # This allows transistioning from purely created with code to template plus code or pure code
            if var is None:
                var = creator_func(var_data_func.__self__, output_contents, var_name)
            elif creator_func is not None:
                # Warn that we are using the templated value despite there being a creator function in the code
                logger.warning(
                    f"Using existing variable definition {var_name} in {self.output_filename} despite definition of a creator function"
                )

            if hasattr(var, "grid_mapping") and self.grid_mapping is not None:
                var.grid_mapping = self.grid_mapping

            # Obtain the data from the registered function/method
            data = var_data_func()

            # Allow direct modifications to the variable object itself such as dynamic attribute modifications
            if modifier_func is not None:
                modifier_func(var_data_func.__self__, var, var_name)

            # Skip setitng values if data is None
            if data is None:
                continue

            try:
                var[...] = data
                variables_written.append(netcdf_full_path(var))
            except (IndexError, ValueError) as exc:
                raise exc.__class__(
                    f"Data mismatch for variable {var_name}. Source shape {data.shape} expected shape {var.shape}. {exc}"
                ) from exc
            except Exception as exc:
                raise exc.__class__(
                    f"Error assigning data for variable {var_name}. {exc}"
                ) from exc

        # Check for consistency with the template definition
        if template_variables is not None:
            # Issue warnings for any variables defined in the template but not written above
            for var_name in template_variables:
                if var_name not in variables_written:
                    logger.warning(
                        f"{var_name} defined within netCDF4 template but no value was registered or written"
                    )

            # Issue warnings for any variables written because they were registered but that were not defined by the template
            for var_name in variables_written:
                if var_name not in template_variables:
                    logger.warning(
                        f"{var_name} written but not defined within the netCDF4 template"
                    )

    def write(self):
        """
        Calls all registered functions to write the NetCDF4 output file
        Variables must be defined in the template NetCDF4 file.
        Attributes will be written even if they are not defined in the template file
        """

        logger.info(f"Writing to output file: {self.output_filename}")
        output_contents = netCDF4.Dataset(self.output_filename, "a")

        try:
            self._write_attributes(output_contents)
            self._write_variables(output_contents)

        finally:
            output_contents.close()

        logger.info("Finalizing output file to remove unlimited dimensions")
        remove_unlimited_dims(
            self.output_filename, self.output_filename, overwrite=True
        )


__all__ = [
    "TemplatedOutput",
]

"""
Title	: read_nc.py
To Run	: from read_nc import read_nc
Author	: Frank!
Date	: 20240722
Modf	: 20241205: added global_attrs to read_nc() and _ReadNcObject ()
          20250218: fixed type(dummy) is str in read_nc()

"""

# Import netCDF4
from netCDF4 import Dataset
import os


class _ReadNcObject:
    ##############
    # Generates an object for read_nc
    #
    # Parameters
    # ---------
    # None
    #
    # Returns
    # -------
    # None
    ##############
    def __init__(
        self, global_attrs: dict, datasets: list, values: dict, attributes: dict
    ) -> None:
        self.global_attrs = global_attrs
        self.datasets = datasets
        self.values = values
        self.attributes = attributes


def read_nc(
    file_name: str | os.PathLike[str] = "", data_sets: list[str] | None = None
) -> _ReadNcObject:
    # ========================================================
    # A generic NetCDF reader
    #
    # Parameters
    # ---------
    # file_name : string; name of the h5 file
    # data_sets : string array; an array of group names to be read
    #
    # Returns
    # -------
    # _ReadNcObject : arrays and 2 dictionaries containing the names of datasets, values and information
    # ========================================================
    file = Dataset(file_name, "r")

    global_attrs = {}
    for i_var in file.ncattrs():
        global_attrs[i_var] = file.getncattr(i_var)

    datasets = []
    values = {}
    attributes = {}

    for i_var in file.variables.keys():
        name = "/" + i_var
        if data_sets is None or name in data_sets:
            datasets.append(name)
            dummy = file.variables[i_var][:].data
            values[str(name)] = dummy
            attributes[str(name)] = str(file.variables[i_var])

    for i_gr in file.groups.keys():
        for i_var in file.groups[i_gr].variables.keys():
            name = "/" + i_gr + "/" + i_var
            if data_sets is None or name in data_sets:
                datasets.append(name)
                dummy = file.groups[i_gr].variables[i_var][:]
                if type(dummy) is str:
                    values[str(name)] = dummy
                else:
                    values[str(name)] = dummy.data
                attributes[str(name)] = str(file.groups[i_gr].variables[i_var])

    file.close()

    return _ReadNcObject(global_attrs, datasets, values, attributes)

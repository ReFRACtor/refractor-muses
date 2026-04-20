import inspect


class OutputType(object):
    dataset = "dataset"
    attribute = "attribute"


def _attach_name_to_func(type_name, data_name):
    def _wrapper(func):
        def _attach_creator_to_data(creator):
            func._creator = creator  # noqa:SLF001
            return creator

        def _attach_modifier_to_data(modifier):
            func._modifier = modifier  # noqa:SLF001
            return modifier

        func._output_type = type_name  # noqa:SLF001
        func._data_name = data_name  # noqa:SLF001

        # A "chained" decorator connected to the
        # data function that defines how to create
        # the variable
        func.creator = _attach_creator_to_data

        # A "chained" decorator connected to the
        # data function that allows modification
        # of the NetCDF variable object itself
        func.modifier = _attach_modifier_to_data

        return func

    return _wrapper


def register_dataset(var_name):
    return _attach_name_to_func(OutputType.dataset, var_name)


def register_attribute(attr_name):
    return _attach_name_to_func(OutputType.attribute, attr_name)


class DeclarativeOutput(object):
    def __new__(cls, *vargs, **kwargs):
        cls.output_definition = {}

        # Go through all classes and look for any functions that has an output_type
        # defined on them to register them as sources of data for datasets and attributes
        for cl_type in inspect.getmro(cls):
            members = inspect.getmembers(cl_type, predicate=inspect.isfunction)
            for func_name, func in members:
                if hasattr(func, "_output_type"):
                    output_type = getattr(func, "_output_type")
                    data_name = getattr(func, "_data_name")

                    data_defs = cls.output_definition[output_type] = (
                        cls.output_definition.get(output_type, {})
                    )
                    data_defs[func_name] = data_name

        return super().__new__(cls)

    def register_output(self, output):
        for func_name, data_name in self.output_definition.get(
            OutputType.dataset, {}
        ).items():
            output.register_dataset(data_name, getattr(self, func_name))

        for func_name, data_name in self.output_definition.get(
            OutputType.attribute, {}
        ).items():
            output.register_attribute(data_name, getattr(self, func_name))


__all__ = ["DeclarativeOutput", "register_attribute", "register_dataset"]

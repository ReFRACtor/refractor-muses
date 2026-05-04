from __future__ import annotations
import inspect
from typing import Any, Self, Callable, ParamSpec, TypeVar
import abc


class OutputType(object):
    dataset = "dataset"
    attribute = "attribute"


P = ParamSpec("P")
R = TypeVar("R")


def _attach_name_to_func(
    type_name: str, data_name: str
) -> Callable[
    [
        Callable[P, R],
    ],
    Callable[P, R],
]:
    def _wrapper(func: Callable[P, R]) -> Callable[P, R]:
        # TODO Sort out these types, probably need some examples here to
        # work this through
        def _attach_creator_to_data(creator: Any) -> Any:
            func._creator = creator  # type: ignore[attr-defined] # noqa:SLF001

            return creator

        def _attach_modifier_to_data(modifier: Any) -> Any:
            func._modifier = modifier  # type: ignore[attr-defined] # noqa:SLF001
            return modifier

        func._output_type = type_name  # type: ignore[attr-defined] # noqa:SLF001
        func._data_name = data_name  # type: ignore[attr-defined] # noqa:SLF001

        # A "chained" decorator connected to the
        # data function that defines how to create
        # the variable
        func.creator = _attach_creator_to_data  # type:ignore[attr-defined]

        # A "chained" decorator connected to the
        # data function that allows modification
        # of the NetCDF variable object itself
        func.modifier = _attach_modifier_to_data  # type:ignore[attr-defined]

        return func

    return _wrapper


def register_dataset(
    var_name: str,
) -> Callable[
    [
        Callable[P, R],
    ],
    Callable[P, R],
]:
    return _attach_name_to_func(OutputType.dataset, var_name)


def register_attribute(
    attr_name: str,
) -> Callable[
    [
        Callable[P, R],
    ],
    Callable[P, R],
]:
    return _attach_name_to_func(OutputType.attribute, attr_name)


class DeclarativeOutputHandle(object, metaclass=abc.ABCMeta):
    """Base class that handles the actual output from a DeclarativeOutput
    class, e.g., TemplatedOutput"""

    @abc.abstractmethod
    def register_dataset(self, name: str, function: Callable[..., Any]) -> None:
        raise NotImplementedError()

    @abc.abstractmethod
    def register_attribute(self, name: str, value: Any) -> None:
        raise NotImplementedError()


class DeclarativeOutput:
    def __new__(cls, *vargs: Any, **kwargs: Any) -> Self:
        cls.output_definition: dict[str, dict[str, str]] = {}

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

    def register_output(self, output: DeclarativeOutputHandle) -> None:
        for func_name, data_name in self.output_definition.get(
            OutputType.dataset, {}
        ).items():
            output.register_dataset(data_name, getattr(self, func_name))

        for func_name, data_name in self.output_definition.get(
            OutputType.attribute, {}
        ).items():
            output.register_attribute(data_name, getattr(self, func_name))


__all__ = [
    "DeclarativeOutput",
    "DeclarativeOutputHandle",
    "register_attribute",
    "register_dataset",
]

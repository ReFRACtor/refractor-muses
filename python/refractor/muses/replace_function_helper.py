from __future__ import annotations
from contextlib import contextmanager
from . import muses_py as mpy  # type: ignore

# Some helper functions for managing the mpy.replace_function. We may migrate
# some of this into muses-py, but for now we'll collect stuff here.


@contextmanager
def register_replacement_function_in_block(func_name: str, obj):
    """Register a replacement function object, execute whatever is in
    the block of the context manager, and then reset the replacement
    function to whatever existing previously (including nothing, if
    there wasn't something already defined.

    """
    old_f = mpy.register_replacement_function(func_name, obj)
    try:
        yield
    finally:
        mpy.register_replacement_function(func_name, old_f)


@contextmanager
def suppress_replacement(func_name: str):
    """Synonym for register_replacement_function(func_name, None),
    which just makes the intent clearer

    """
    old_f = mpy.register_replacement_function(func_name, None)
    try:
        yield
    finally:
        mpy.register_replacement_function(func_name, old_f)


__all__ = ["register_replacement_function_in_block", "suppress_replacement"]

from __future__ import annotations
from builtins import object
from docopt import docopt, ParsedOptions  # type: ignore
import re
from typing import Any


class DocOptSimple(object):
    """The package docopt (http://docopt.org) is a nice package,
    but it has the disadvantage that getting the options etc. from it
    uses a somewhat unnatural interface (e.g., --my-opt=n comes as
    int(d["--my-opt"]) rather than OptionParser cleaner sort of
    options.my_opt). This class tries to give a simple interface that
    is sufficient for many purposes. If you try to access the attribute
    "my_val", we first look for the argument "my_val", then "<my_val>",
    then "--my_val", then "--my-val". If the value looks like a integer,
    we return it to an integer. If it looks like a float, we return it to
    a float. If we don't otherwise recognize it, we return this as a string."""

    def __init__(
        self,
        doc: str,
        argv: list[str] | None = None,
        default_help: bool = True,
        version: None | str = None,
        options_first: bool = False,
    ) -> None:
        self.args: dict[str, Any] | ParsedOptions = docopt(
            doc,
            argv=argv,
            default_help=default_help,
            version=version,
            options_first=options_first,
        )

    def __getstate__(self) -> dict:
        return {"args": self.args}

    def __setstate__(self, d: dict) -> None:
        self.args = d["args"]

    def __contains__(self, name: str) -> bool:
        for key in (name, "<" + name + ">", "--" + name, "--" + name.replace("_", "-")):
            if key in self.args:
                return True
        return False

    def __getattr__(self, name: str) -> Any:
        # Don't normally get called with "args", but can before object is
        # fully initialized. So catch this an return an empty dict, without
        # this handling we can enter an infinite recursion
        if name == "args":
            self.args = {}
            return self.args
        for key in (name, "<" + name + ">", "--" + name, "--" + name.replace("_", "-")):
            if key in self.args:
                return self.__find_type(key)
        raise AttributeError(name)

    def __find_type(self, key: str) -> Any:
        """Find the type of the value, and return in"""
        v = self.args[key]
        if isinstance(v, str):
            if re.match(r"[+-]?\d+$", v):
                return int(v)
            if re.match(r"[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$", v):
                return float(v)
            if re.match(r"[-+]?[0-9]+\.([eE][-+]?[0-9]+)?$", v):
                return float(v)
        return v


def docopt_simple(
    doc: str,
    argv: list[str] | None = None,
    default_help: bool = True,
    version: None | str = None,
    options_first: bool = False,
) -> DocOptSimple:
    """The package docopt (http://docopt.org) is a nice package,
    but it has the disadvantage that getting the options etc. from it
    uses a somewhat unnatural interface. This gives a simpler interface
    sufficient for many programs. See DocOptSimple class for details
    on the interface.
    """
    return DocOptSimple(
        doc,
        argv=argv,
        default_help=default_help,
        version=version,
        options_first=options_first,
    )


__all__ = [
    "docopt_simple",
]

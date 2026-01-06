from __future__ import annotations
import importlib
import json

# muses-py has a lot of hard coded things related to the species names and netcdf output.
# It would be good a some point to just replace this all with a better thought out output
# format. But for now, we need to support the existing output format.
# TODO - Replace with better thought out output format

# So we don't depend on muses_py, we save the variable to a json file.
# Only need muses_py to generate this or update it. We just create this file
# if not available, so you can manually delete this to force it to be recreated.

class _helper:
    _instance : _helper | None = None
    def __init__(self) -> None:
        if not importlib.resources.is_resource("refractor.muses", "order_species.json"):
            from refractor.old_py_retrieve_wrapper import create_order_species_json
            create_order_species_json()
        d = json.loads(importlib.resources.read_text("refractor.muses", "order_species.json"))
        self.ordered_species_list = d["ordered_species_list"]
        self.atmospheric_species_list = d["atmospheric_species_list"]

    @classmethod
    def instance(cls) -> _helper:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

def is_atmospheric_species(species_name: str) -> bool:
    """Some species are marked as "atmospheric_species". This is used in the
    determination of the microwindows file name, this wants to filter out things
    like O3_EMIS, O3_TSUR, and just have O3 pass. I don't think this gets used
    anywhere else."""
    return species_name.upper() in _helper.instance().atmospheric_species_list


def species_type(species_name: str) -> str:
    """Return the type of species this is"""
    if is_atmospheric_species(species_name):
        return "ATMOSPHERIC"
    if species_name.upper() in ["CLOUDOD", "CLOUDEXT", "CALSCALE", "CALOFFSET", "EMIS"]:
        return "SPECTRAL"
    if species_name.upper() in ["PCLOUD", "TSUR"]:
        return "SINGLE"
    return "unknown"


def order_species(species_list: list[str]) -> list[str]:
    """This provides an order to the given species list. This is used to
    make sure various matrixes etc. have the state elements all ordered the
    same way.

    We use the list order the mpy.ordered_species_list have for items
    found in that list, and then we just alphabetize any species not
    found in that list. We could adapt this logic if needed (e.g.,
    extend ordered_species_list), but this seems sufficient for now.

    """
    ordered_species_list = _helper.instance().ordered_species_list
    # The use of a tuple here is a standard python "trick" to separate
    # out the values sorted by _ordered_species_list
    # vs. alphabetically. In python False < True, so all the items in
    # _ordered_species_list are put first in the sorted list. The
    # second part of the tuple is only sorted for things that are in
    # the same set for the first test, so using integers from index
    # vs. string comparison is separated nicely.
    return sorted(
        species_list,
        key=lambda v: (
            v not in ordered_species_list,
            ordered_species_list.index(v) if v in ordered_species_list else v,
        ),
    )


def compare_species(s1: str, s2: str) -> int:
    """Return -1, 0 or 1 for s1 < s2, s1 == s2, s1 > s2"""
    ordered_species_list = _helper.instance().ordered_species_list
    if s1 in ordered_species_list and s2 not in ordered_species_list:
        return -1
    if s1 not in ordered_species_list and s2 in ordered_species_list:
        return 1
    t1 = (
        s1 in ordered_species_list,
        ordered_species_list.index(s1) if s1 in ordered_species_list else s1,
    )
    t2 = (
        s2 in ordered_species_list,
        ordered_species_list.index(s2) if s2 in ordered_species_list else s2,
    )
    if t1 < t2:
        return -1
    elif t1 == t2:
        return 0
    return 1


__all__ = [
    "is_atmospheric_species",
    "order_species",
    "compare_species",
    "species_type",
]

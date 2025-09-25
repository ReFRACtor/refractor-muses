from __future__ import annotations
import refractor.muses.muses_py as mpy  # type: ignore

_ordered_species_list = mpy.ordered_species_list()


def order_species(species_list: list[str]) -> list[str]:
    """This provides an order to the given species list. This is used to
    make sure various matrixes etc. have the state elements all ordered the
    same way.

    We use the list order the mpy.ordered_species_list have for items
    found in that list, and then we just alphabetize any species not
    found in that list. We could adapt this logic if needed (e.g.,
    extend ordered_species_list), but this seems sufficient for now.

    """

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
            v not in _ordered_species_list,
            _ordered_species_list.index(v) if v in _ordered_species_list else v,
        ),
    )


def compare_species(s1: str, s2: str) -> int:
    """Return -1, 0 or 1 for s1 < s2, s1 == s2, s1 > s2"""
    if s1 in _ordered_species_list and s2 not in _ordered_species_list:
        return -1
    if s1 not in _ordered_species_list and s2 in _ordered_species_list:
        return 1
    t1 = (
        s1 in _ordered_species_list,
        _ordered_species_list.index(s1) if s1 in _ordered_species_list else s1,
    )
    t2 = (
        s2 in _ordered_species_list,
        _ordered_species_list.index(s2) if s2 in _ordered_species_list else s2,
    )
    if t1 < t2:
        return -1
    elif t1 == t2:
        return 0
    return 1


__all__ = [
    "order_species",
    "compare_species",
]

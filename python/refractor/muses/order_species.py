from __future__ import annotations
import refractor.muses.muses_py as mpy  # type: ignore

_ordered_species_list = mpy.ordered_species_list()


def order_species(species_list: list[str]) -> list[str]:
    """This provides an order to the given species list.

    It isn't really clear why this is needed, but it is in the existing muses-py.
    We use the list order they have for items found in that list, and then we
    just alphabetize any species not found in that list. We could adapt this
    logic if needed (e.g., extend ordered_species_list), but is isn't really
    clear why the ordering is done in the first place."""

    # The use of a tuple here is a standard python "trick" to separate out
    # the values sorted
    # by _ordered_species_list vs. alphabetically. In python
    # False < True, so all the items in _ordered_species_list are put first in
    # the sorted list. The second part of the tuple is only sorted for things
    # that are in the same set for the first test, so using integers from index
    # vs. string comparison is separated nicely.
    return sorted(
        species_list,
        key=lambda v: (
            v not in _ordered_species_list,
            _ordered_species_list.index(v) if v in _ordered_species_list else v,
        ),
    )


__all__ = [
    "order_species",
]

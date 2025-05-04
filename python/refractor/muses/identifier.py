from __future__ import annotations
import abc
from .order_species import order_species


class Identifier(object, metaclass=abc.ABCMeta):
    """There are various places where we indicate a list of things, such
    as the retrieval_elements or instrument_names. muses-py used strings
    to identify this, so a strategy table had a list of state element names
    to give the list of what we are retrieving.

    We abstract that. It is useful if nothing else to indicate what a
    particular str is for. Plus we may find it useful to have other
    identifiers in the future.

    """

    # Not clear what all we want here. Right now, just the minimum we need
    # to have a set that can be used in dict and converted to str that muses_py
    # wants
    @abc.abstractmethod
    def __str__(self) -> str:
        """And Identifier should be able to be printed out"""
        raise NotImplementedError

    @abc.abstractmethod
    def __eq__(self, other: object) -> bool:
        """Test for equality. Different Identifier objects that point to the same
        Identifier should be equal (so default test that objects are the same isn't
        what we want here)."""
        raise NotImplementedError

    @abc.abstractmethod
    def __hash__(self) -> int:
        """Hash to go along with __eq__. We want Identifier objects that have the
        same value with the same hash (so we can find in a dict for example, without
        the default of the objects being identical)
        """
        raise NotImplementedError

    @classmethod
    def sort_identifier(cls, lst: list) -> list:
        """Take a list of Identifier, and apply any natural sort to it."""
        # Default is just to sort by the str value.
        return sorted(lst, key=cls.__str__)


class IdentifierStr(Identifier):
    """Right now, most of our identifiers are only str. We may extend this in the
    future. But we have all the Identifiers separated out from str, so any
    future changes should have the plumbing in place to support."""

    def __init__(self, s: str) -> None:
        self.s = s

    def __eq__(self, other: object) -> bool:
        if not hasattr(other, "s"):
            raise RuntimeError("other should be a IdentifierStr")
        return self.s == other.s

    def __hash__(self) -> int:
        return hash(self.s)

    def __str__(self) -> str:
        return self.s


class InstrumentIdentifier(IdentifierStr):
    '''Identify an instrument, e.g., "AIRS"'''

    pass


class StateElementIdentifier(IdentifierStr):
    '''Identify an state element, e.g., "TROPOMIRINGSFBAND3"'''

    def __eq__(self, other: object) -> bool:
        return super().__eq__(other)

    # unhashable type: 'StateElementIdentifier' if not defined
    def __hash__(self) -> int:
        return super().__hash__()

    @classmethod
    def sort_identifier(cls, lst: list) -> list:
        """Take a list of Identifier, and apply any natural sort to it."""
        return [cls(s) for s in order_species([str(s) for s in lst])]


class RetrievalType(IdentifierStr):
    '''Identify an retrieval step type, e.g., "BG"'''

    def __eq__(self, other: object) -> bool:
        if not hasattr(other, "s"):
            raise RuntimeError("other should be a RetrievalType")
        # Do case insensitive compare
        return self.s.lower() == other.s.lower()

    def __hash__(self) -> int:
        return hash(self.s.lower())

    def lower(self) -> str:
        return self.s.lower()


class StrategyStepIdentifier(Identifier):
    """Identify an step in a strategy, e.g. "Step 0"
    Note we identify by the name and initial step number. The step_number can be updated
    without changing this identifier, just the string it prints out gets updated"""

    def __init__(self, initial_step_number: int, step_name: str) -> None:
        self.initial_step_number = initial_step_number
        self.step_name = step_name
        self.step_number = self.initial_step_number

    def __eq__(self, other: object) -> bool:
        if not hasattr(other, "initial_step_number") or not hasattr(other, "step_name"):
            raise RuntimeError("other should be a StrategyStepIdentifier")
        return (
            self.initial_step_number == other.initial_step_number
            and self.step_name == other.step_name
        )

    def __hash__(self) -> int:
        return hash((self.initial_step_number, self.step_name))

    def __str__(self) -> str:
        return f"Step: {self.step_number}, Step Name: {self.step_name}"


class ProcessLocation(IdentifierStr):
    """Identify a process location, passed to RetrievalStrategy observers"""


class FilterIdentifier(IdentifierStr):
    """Identify a filter in l1b data."""
    # The L2 output has hard coded mappings from the filter
    # identifier name (e.g., "CrIS-fsr-lw") to a band name
    # (e.g., "TIR1"). In addition to the hard coded names, it
    # has a hard coded index. This is somewhat clumsy (why two names?)
    # and also not extensible. But put this into place so we can
    # support this. For data that doesn't fit we'll return and index of
    # -999 and a name of the identifier
    _filter_map = {
            "ALL" : (0, "ALL"),
            "UV1" : (1, "UV1"),
            "UV2" : (2, "UV2"),
            "VIS" : (3, "VIS"),
            "CrIS-fsr-lw" : (11, "TIR1"),
            "CrIS-fsr-mw" : (13, "TIR3"),
            "CrIS-fsr-sw" : (14, "TIR4"),
            "2B1" : (11, "TIR1"),
            "1B2" : (12, "TIR2"),
            "2A1" : (13, "TIR3"),
            "1A1" : (14, "TIR4"),
            "BAND1" : (1, "UV1"),
            "BAND2" : (2, "UV2"),
            "BAND3" : (4, "UVIS"),
            "BAND4" : (3, "VIS"), 
            "BAND5" : (5, "NIR1"),
            "BAND6" : (6, "NIR2"),
            "BAND7" : (9, "SWIR3"),
            "BAND8" : (10, "SWIR4"),
            "O2A" : (5, "NIR1"),
            "WCO2" : (7, "SWIR1"),
            "SCO2" : (8, "SWIR2"),
            "CH4" : (9, "SWIR3"),
        }
    # From
    #filters = [
    #    "ALL", 0
    #    "UV1", 1
    #    "UV2", 2
    #    "VIS", 3
    #    "UVIS", 4
    #    "NIR1", 5
    #    "NIR2", 6
    #    "SWIR1", 7
    #    "SWIR2", 8
    #    "SWIR3", 9
    #    "SWIR4", 10
    #    "TIR1", 11
    #    "TIR2", 12
    #    "TIR3", 13
    #    "TIR4", 14
    #]
    _filters_order = [
        "ALL",
        "UV1",
        "UV2",
        "VIS",
        "CrIS-fsr-lw",
        "CrIS-fsr-mw",
        "CrIS-fsr-sw",
        "2B1",
        "1B2",
        "2A1",
        "1A1",
        "BAND1",
        "BAND2",
        "BAND3",
        "BAND4",
        "BAND5",
        "BAND6",
        "BAND7",
        "BAND8",
        "O2A",
        "WCO2",
        "SCO2",
        "CH4",
    ]

    @classmethod
    def spectral_order(cls, lst : list[FilterIdentifier]) -> list[int]:
        '''In addition to the name translation, the output has a particular order
        that is wants things. order the data by list'''
        filter_order = list(cls._filters_order)
        # Extend by anything not in hardcoded list, just in the order it is found (i.e., move
        # all this stuff to the end, in the order it came in)
        filter_order.extend([str(fid) for fid in lst if str(fid) not in filter_order])
        # Then sort the data
        res = sorted(lst, key=lambda i: filter_order.index(str(i)))
        # And translate to an index
        return [lst.index(i) for i in res]
    
    @property
    def spectral_name(self) -> str:
        '''Translate the filter identifier to the spectral name. For
        identifiers not in the hard coded list, we just return the
        filter identifiers'''
        filname = str(self)
        if filname in self._filter_map:
            return self._filter_map[filname][1]
        return filname

    @property
    def filter_index(self) -> int:
        '''Return the hard code filter index for the given filter identifier. If
        this isn't in the hard coded list, we return -999'''
        filname = str(self)
        if filname in self._filter_map:
            return self._filter_map[filname][0]
        return -999

class IdentifierSortByWaveLength:
    """muses-py assumes that InstrumentIdentifier and FilterIdentifier
    are sorted by wavelength. I believe this is just used in some
    output routines, it might be nice to relax this assumption. But
    for now, provide for sorting.
    """

    def __init__(self) -> None:
        self._d: dict[Identifier, float] = {}

    def add(self, id: Identifier, starting_wavelength: float) -> None:
        if id not in self._d:
            self._d[id] = starting_wavelength
        self._d[id] = min(starting_wavelength, self._d[id])

    def sorted_identifer(self) -> list[Identifier]:
        lst = sorted(self._d.items(), key=lambda v: v[1])
        return [v[0] for v in lst]


__all__ = [
    "Identifier",
    "IdentifierStr",
    "InstrumentIdentifier",
    "StateElementIdentifier",
    "RetrievalType",
    "StrategyStepIdentifier",
    "ProcessLocation",
    "FilterIdentifier",
    "IdentifierSortByWaveLength",
]

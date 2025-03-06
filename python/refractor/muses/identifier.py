import abc


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
    def __eq__(self, other):
        """Test for equality. Different Identifier objects that point to the same
        Identifier should be equal (so default test that objects are the same isn't
        what we want here)."""
        raise NotImplementedError

    @abc.abstractmethod
    def __hash__(self):
        """Hash to go along with __eq__. We want Identifier objects that have the
        same value with the same hash (so we can find in a dict for example, without
        the default of the objects being identical)
        """
        raise NotImplementedError


class IdentifierStr(Identifier):
    """Right now, most of our identifiers are only str. We may extend this in the
    future. But we have all the Identifiers separated out from str, so any
    future changes should have the plumbing in place to support."""

    def __init__(self, s):
        self.s = s

    def __eq__(self, other):
        return self.s == other.s

    def __hash__(self):
        return hash(self.s)

    def __str__(self) -> str:
        return self.s


class InstrumentIdentifier(IdentifierStr):
    '''Identify an instrument, e.g., "AIRS"'''

    pass


class StateElementIdentifier(IdentifierStr):
    '''Identify an state element, e.g., "TROPOMIRINGSFBAND3"'''

    def __eq__(self, other):
        # Allow for comparison to str without casting
        if isinstance(other, str):
            return self.s == other
        return super().__eq__(other)

    # unhashable type: 'StateElementIdentifier' if not defined
    def __hash__(self):
        return super().__hash__()


class RetrievalType(IdentifierStr):
    '''Identify an retrieval step type, e.g., "BG"'''

    def __eq__(self, other):
        # Do case insensitive compare
        return self.s.lower() == other.s.lower()

    def __hash__(self):
        return hash(self.s.lower())

    def lower(self) -> str:
        return self.s.lower()


class StrategyStepIdentifier(Identifier):
    """Identify an step in a strategy, e.g. "Step 0"
    Note we identify by the name and initial step number. The step_number can be updated
    without changing this identifier, just the string it prints out gets updated"""

    def __init__(self, initial_step_number: int, step_name: str):
        self.initial_step_number = initial_step_number
        self.step_name = step_name
        self.step_number = self.initial_step_number

    def __eq__(self, other):
        return (
            self.initial_step_number == other.initial_step_number
            and self.step_name == other.step_name
        )

    def __hash__(self):
        return hash((self.initial_step_number, self.step_name))

    def __str__(self) -> str:
        return f"Step: {self.step_number}, Step Name: {self.step_name}"


class ProcessLocation(IdentifierStr):
    """Identify a process location, passed to RetrievalStrategy observers"""


class FilterIdentifier(IdentifierStr):
    """Identify a filter in l1b data."""

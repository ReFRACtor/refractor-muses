import abc


class Identifier(object, metaclass=abc.ABCMeta):
    """There are various places where we indicate a list of things, such
    as the retrieval_elements or instrument_names. muses-py used strings
    to identify this, so a strategy table had a list of state element names
    to give the list of what we are retrieving.

    We abstract that. It is useful if nothing else to indicate what a
    particular str is for if nothing else. Plus we may find it useful to
    have other identifiers in the future."""

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
    """Right now, our identifiers are only str. We may extend this in the
    future, but this would prove a bit complicated and we would want to think
    through the design. But we have all the Identifiers separated out from str, so any
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

    pass


class RetrievalStepIdentifier(IdentifierStr):
    '''Identify an retrieval step, e.g., "BG"'''

    def __eq__(self, other):
        # Do case insensitive compare
        return self.s.lower() == other.s.lower()

    def __hash__(self):
        return hash(self.s.lower())

    def lower(self) -> str:
        return self.s.lower()


class StrategyStepIdentifier(IdentifierStr):
    '''Identify an step in a strategy, e.g. "Steo 0"'''

    def __init__(self, i):
        super().__init__(f"Step {i}")


class FilterIdentifier(IdentifierStr):
    """Identify a filter in l1b data."""

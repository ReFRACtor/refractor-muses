# Modify the constraint matrix for NH3. Note that the constraint matrix is
# *usually* the apriori_cov, and would be for a MaxAPosteriori type retrieval.
#
# The constraint matrix is used to regularize the solver, it is added
# as augmented terms to the cost function as a penalty for moving away
# from the constraint_vector. When the constraint matrix is apriori
# covariance this is a maximum a posteriori problem, which is a common
# step used in the retrieval strategy. However we can also just use an
# ad hoc constraint providing e.g. smoothness (see II.B of the paper
# mentioned below). For example the ig_refine step after getting the
# OMI or TropOMI cloud fraction in a brightness temperature uses a
# tighter constraint than the apriori covariance.

# This example replaces the NH3 constraint matrix with one * 10

from __future__ import annotations
from refractor.muses import (
    StateElementFromClimatologyNh3,
    RetrievalStrategy,
    RetrievalGrid2dArray,
    StateElementWithCreateHandle,
    StateElementIdentifier,
)

rs = RetrievalStrategy(None)


class Nh3LooserConstraint(StateElementFromClimatologyNh3):
    @property
    def constraint_matrix(self) -> RetrievalGrid2dArray:
        original = super().constraint_matrix
        return original * 10.0


# We then register a creator for Nh3LooserConstraint, and set it to "cut in line" with
# the priority handle set. We set the priority order to 100, so it is a higher
# priority than any other creator we have registered

rs.state_element_handle_set.add_handle(
    StateElementWithCreateHandle(
        StateElementIdentifier("NH3"),
        Nh3LooserConstraint,
    ),
    priority_order=100,
)

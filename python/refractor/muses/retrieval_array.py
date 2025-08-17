from __future__ import annotations
from . import muses_py as mpy  # type: ignore
import refractor.framework as rf  # type: ignore
import numpy as np

# To help keep things straight, we have separate types for each of state vector
# types. In all cases, these are just np.ndarray with a little extra type information.
# Where it is useful, we add a handful of member functions (e.g., for converting form
# one type to the other).


class RetrievalGridArray(np.ndarray):
    """Data in the retrieval state vector. Unmapped (e.g., might be log(vmr)).
    Generally smaller number of levels than FullGridArray.

    Note that while you can use a np.ndarray constructor, most of the
    time we get numpy arrays from other functions (e.g. np.zeros,
    np.array).  Numpy has support for this, see
    https://numpy.org/doc/stable/user/basics.subclassing.html.  We can
    use "view casting", basically just tack on
    "view(RetrievalGridArray)" to an array (see
    https://numpy.org/doc/2.2/reference/generated/numpy.ndarray.view.html).
    This adds type information to the numpy array, which is actually a
    bit closer to what we want anyways. We really just want a normal
    numpy array with additional labeling saying what kind of data it
    has.

    Numpy can "forget" the type as you do various operations, in
    particular passing this into framework C++ objects. This is
    generally fine, these labels are useful in CurrentState and
    related classes, but once we for example get to the ForwardModel
    everything goes through the StateVector which is always
    RetrievalGridArray or FullGridArray (depending on where in the
    cost function).  Labeling it isn't particularly useful at that
    point.

    See CurrentState for a description of the various state vectors.

    """

    def to_full(self, state_mapping_retrieval_to_fm: rf.StateMapping) -> FullGridArray:
        # rf.StateMapping only works with ArrayAd_double_1. We could extend this to
        # work directly with numpy, but is it easy enough for us just to route this this
        # class
        return state_mapping_retrieval_to_fm.mapped_state(
            rf.ArrayAd_double_1(self)
        ).value.view(FullGridArray)

    def to_fmprime(
        self,
        state_mapping_retrieval_to_fm: rf.StateMapping,
        state_mapping: rf.StateMapping,
    ) -> FullGridMappedArrayFromRetGrid:
        # rf.StateMapping only works with ArrayAd_double_1. We could extend this to
        # work directly with numpy, but is it easy enough for us just to route this this
        # class
        return state_mapping.mapped_state(
            state_mapping_retrieval_to_fm.mapped_state(rf.ArrayAd_double_1(self))
        ).value.view(FullGridMappedArrayFromRetGrid)


class FullGridArray(np.ndarray):
    """Data in the forward model state vector/full state vector.
    Unmapped (e.g., might be log(vmr)). Generally more levels than RetrievalGridArray.

    See discussion in RetrievalGridArray about adding this type to a numpy array.

    See CurrentState for a description of the various state vectors."""

    def to_ret(
        self, state_mapping_retrieval_to_fm: rf.StateMapping
    ) -> RetrievalGridArray:
        # rf.StateMapping only works with ArrayAd_double_1. We could extend this to
        # work directly with numpy, but is it easy enough for us just to route this this
        # class
        return state_mapping_retrieval_to_fm.retrieval_state(
            rf.ArrayAd_double_1(self)
        ).value.view(RetrievalGridArray)

    def to_fm(self, state_mapping: rf.StateMapping) -> FullGridMappedArray:
        # rf.StateMapping only works with ArrayAd_double_1. We could extend this to
        # work directly with numpy, but is it easy enough for us just to route this this
        # class
        return state_mapping.mapped_state(rf.ArrayAd_double_1(self)).value.view(
            FullGridMappedArray
        )


class FullGridMappedArray(np.ndarray):
    """Data in in the forward model state vector/full state vector.
    Mapped (e.g., log(vmr) is converted to VMR).

    See discussion in RetrievalGridArray about adding this type to a
    numpy array.

    See CurrentState for a description of the various state vectors.

    """

    def to_full(self, state_mapping: rf.StateMapping) -> FullGridArray:
        # rf.StateMapping only works with ArrayAd_double_1. We could extend this to
        # work directly with numpy, but is it easy enough for us just to route this this
        # class
        return state_mapping.retrieval_state(rf.ArrayAd_double_1(self)).value.view(
            FullGridArray
        )

    def to_ret(
        self,
        state_mapping_retrieval_to_fm: rf.StateMapping,
        state_mapping: rf.StateMapping,
    ) -> RetrievalGridArray:
        # rf.StateMapping only works with ArrayAd_double_1. We could extend this to
        # work directly with numpy, but is it easy enough for us just to route this this
        # class
        return state_mapping_retrieval_to_fm.retrieval_state(
            state_mapping.retrieval_state(rf.ArrayAd_double_1(self))
        ).value.view(RetrievalGridArray)

    def to_fmprime(
        self,
        state_mapping_retrieval_to_fm: rf.StateMapping,
        state_mapping: rf.StateMapping,
        should_fix_negative: bool = False,
    ) -> FullGridMappedArrayFromRetGrid:
        """Convert to fmprime. Note muses-py has handling for negative values if
        the state mapping is linear. If the flag "should_fix_negative" is True,
        we include that handling. One place to get this is from StateElement
        should_fix_negative property, it has logic to figure out if we should
        do this for a StateElement"""
        v = self.to_ret(state_mapping_retrieval_to_fm, state_mapping).copy()
        if should_fix_negative and v.min() < 0 and v.max() > 0:
            v[v < 0] = v[v > 0].min()
        return v.to_fmprime(state_mapping_retrieval_to_fm, state_mapping)
        return v


def _from_x_subset(
    cls: type[rf.StateMappingBasisMatrix],
    x: FullGridMappedArray,
    ind: np.ndarray,
    log_interp: bool = True,
) -> rf.StateMappingBasisMatrix:
    """We often have a full grid defined by a set of parameters (pressure, or for a
    few StateElement wavelength), and have the retrieval grid defined as a subset.
    This function calculates the state_mapping_retrieval_to_fm needed to go to
    and from this. This is the equivalent of the muses-py code mpy.make_maps.

    For something like pressure, we work in log(x). For wavelength, we work in x.
    You can select x interpolation by passing log_interp as False."""
    # Temp
    lv = ind + 1
    t = mpy.make_maps(x, lv, i_linearFlag=(not log_interp))
    return rf.StateMappingBasisMatrix(t["toState"].transpose(), t["toPars"].transpose())


rf.StateMappingBasisMatrix.from_x_subset = classmethod(_from_x_subset)


def _from_x_subset_exclude_gap(
    cls: type[rf.StateMappingBasisMatrix],
    x: FullGridMappedArray,
    ind: np.ndarray,
    log_interp: bool = True,
    gap_threshold: float = 50.0,
) -> rf.StateMappingBasisMatrix:
    """This is a variation of from_x_subset where we exclude frequencies not
    retrieved that are in a gap > threshold. The idea here is that for points
    far from where we are actually retrieving we are better just leaving the
    data as is rather than interpolating through where we are far from the points"""
    lv = ind + 1
    t = mpy.make_maps(x, lv, i_linearFlag=(not log_interp))
    x_ret = x[ind]
    m = t["toState"]
    ind = np.searchsorted(x_ret, x)
    for k in range(m.shape[1]):
        if ind[k] > 0 and ind[k] < x_ret.shape[0]:
            diff1 = x[k] - x_ret[ind[k] - 1]
            diff2 = x_ret[ind[k]] - x[k]
            if diff1 > 0 and diff2 > 0 and diff1 + diff2 > gap_threshold:
                m[:, k] = 0
    return rf.StateMappingBasisMatrix(t["toState"].transpose(), t["toPars"].transpose())


rf.StateMappingBasisMatrix.from_x_subset_exclude_gap = classmethod(
    _from_x_subset_exclude_gap
)


class FullGridMappedArrayFromRetGrid(np.ndarray):
    """Data in in the forward model state vector/full state vector.
    Mapped (e.g., log(vmr) is converted to VMR).

    This is similar to FullGridMappedArray, however this gets mapped
    from the FullGridMappedArray to the RetrievalGridArray and then back.
    Because the RetrievalGridArray has fewer levels, there is an infinite
    number of FullGridMappedArray values that map to a RetrievalGridArray. The
    inverse mapping picks one, which in general isn't the one we started from.
    So although FullGridMappedArray.to_ret == FullGridMappedArrayFromRetGrid.to_ret,
    in general FullGridMappedArray != FullGridMappedArrayFromRetGrid.

    Do distinguish this, if we have FullGridMappedArray as fm we call
    FullGridMappedArrayFromRetGrid fmprime.

    It isn't completely clear if muses-py *should* be using
    FullGridMappedArrayFromRetGrid, but it currently does. In the current code,
    it isn't clear if this was intended or not. But we will duplicate the current
    behavior, and by using a separate grid we'll at least make it explicit that this
    is what is being done. We can then take a second pass through the code and
    decide if this is actually what we *should* be doing.

    See discussion in RetrievalGridArray about adding this type to a
    numpy array.

    See CurrentState for a description of the various state vectors.
    """

    def to_full(self, state_mapping: rf.StateMapping) -> FullGridArray:
        # rf.StateMapping only works with ArrayAd_double_1. We could extend this to
        # work directly with numpy, but is it easy enough for us just to route this this
        # class
        return state_mapping.retrieval_state(rf.ArrayAd_double_1(self)).value.view(
            FullGridArray
        )

    def to_ret(
        self,
        state_mapping_retrieval_to_fm: rf.StateMapping,
        state_mapping: rf.StateMapping,
    ) -> RetrievalGridArray:
        # rf.StateMapping only works with ArrayAd_double_1. We could extend this to
        # work directly with numpy, but is it easy enough for us just to route this this
        # class
        return state_mapping_retrieval_to_fm.retrieval_state(
            state_mapping.retrieval_state(rf.ArrayAd_double_1(self))
        ).value.view(RetrievalGridArray)


class RetrievalGrid2dArray(np.ndarray):
    """2d matrix going with RetrievalGridArray (e.g., the constraint matrix). This
    is unmapped (TODO Check this, I'm pretty sure this is true)

    See discussion in RetrievalGridArray about adding this type to a
    numpy array.

    See CurrentState for a description of the various state vectors.

    """

    pass


class FullGrid2dArray(np.ndarray):
    """2d matrix going with FullGridArray (e.g. the apriori matrix).

    See discussion in RetrievalGridArray about adding this type to a
    numpy array.

    TODO - Is this mapped or unmapped? Not sure, we should track down. If mapped,
    we might want to rename this.

    See CurrentState for a description of the various state vectors."""

    pass


__all__ = [
    "RetrievalGridArray",
    "FullGridArray",
    "FullGridMappedArray",
    "FullGridMappedArrayFromRetGrid",
    "RetrievalGrid2dArray",
    "FullGrid2dArray",
]

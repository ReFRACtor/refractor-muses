class CurrentState:
    '''There are a number of "states" floating around
    py-retrieve/ReFRACtor, and it can be a little confusing if you
    don't know what you are looking at.

    A good reference is section III.A.1 of "Tropospheric Emission
    Spectrometer: Retrieval Method and Error Analysis" (IEEE
    TRANSACTIONS ON GEOSCIENCE AND REMOTE SENSING, VOL. 44, NO. 5, MAY
    2006). This describes the "retrieval state vector" and the "full
    state vector"

    In addition there is an intermediate set that isn't named in the
    paper.  In the py-retrieve code, this gets referred to as the
    "forward model state vector". And internally objects may map
    things it a "object state".

    A brief description of these:

    1. The "retrieval state vector" is what we use in our Solver for a
    retrieval step. This is the parameters passed to our CostFunction.

    2. The "forward model state vector" has the same content as the
    "retrieval state vector", but it is specified on a finer number of
    levels used by the forward model.

    3. The "full state vector" is all the parameters the various
    objects making up our ForwardModel and Observation needs. This
    includes a number of things held fixed in a particular retrieval
    step.

    4. The "object state", which is the subset of the "forward model
    state vector" needed by a particular object in our ForwardModel or
    Observation, mapped to what the object needs. E.g., the forward
    model state vector is in log(vmr), but for the actual object we
    translate this to vmr.

    Each of these vectors are made up of a number of StateElement,
    each with a fixed string used as a name to identify it.  We look
    up the StateElement by these fixed names. Note that the
    StateElement name value may have different values depending on the
    context - so it might have fewer levels in a retrieval state
    vector vs forward model state vector. Note that muses-py often but
    not always refers to these as "species". We use the more general
    name "StateElement" because these are always gas species.

    A few examples, one of the things that might be retrieved is
    log(vmr) for O3. In the "retrieval state vector" this has 25
    log(vmr) values (for the 25 levels). In the "forward model state
    vector" this as 64 log(vmr) values (for the 64 levels the FM is
    run on).  In the "object state" for the rf.AbsorberVmr part of the
    FowardModel is 64 vmr values (so log(vmr) converted to vmr needed
    for calculation).

    For the tropomi ForwardModel, and component object is a
    rf.GroundLambertian which has a polynomial with values
    "TROPOMISURFACEALBEDOBAND3", "TROPOMISURFACEALBEDOSLOPEBAND3",
    "TROPOMISURFACEALBEDOSLOPEORDER2BAND3". We might be retrieving
    only a subset of these values, holding the other fixed. So in this
    case the "retrieval state vector" might have only 2 entries for
    "TROPOMISURFACEALBEDOBAND3" and "TROPOMISURFACEALBEDOSLOPEBAND3",
    the "forward model state vector" would also only have 2 entries
    (since there aren't any levels involved, there is no difference
    between the retrieval state and the forward model state).  The
    "full state vector" would have 3 values, so we add in the part
    that is being held fixed.

    In all cases, we handle converting from one type of state vector
    to the other with a rf.StateMapping. The
    rf.MaxAPosterioriSqrtConstraint object in our CostFunction has a
    mapping going to and from the "retrieval state vector" to the
    "forward model state vector". The various pieces of the
    ForwardModel and Observation have mapping from "forward model
    state vector" to "full state vector", and the various objects
    handle mappings from "full state vector" to "object state".
    '''
    pass

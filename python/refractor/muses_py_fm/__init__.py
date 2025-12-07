# Just import any files we find in this directory, rather than listing
# everything.

import os as _os
import re as _re
import glob as _glob
import typing as _typing

for _i in _glob.glob(_os.path.dirname(__file__) + "/*.py"):
    mname = _os.path.basename(_i).split(".")[0]
    # Don't load ipython, which is ipython magic extensions, or unit tests
    if (
        not mname == "ipython"
        and not mname == "version"
        and not mname == "cython_try"
        and not _re.search("_test", mname)
    ):
        exec("from .%s import *" % mname)

if _typing.TYPE_CHECKING:
    # mypy doesn't correctly support import *. Pretty annoying, there are threads going
    # back years about why this doesn't work. We don't want to spend a whole lot of
    # time working around this, the point of mypy is to help us and reduce our work, not
    # to make a bunch of make work. But to the degree useful, we can work around this by
    # having an explicit imports for things needed by mypy. We don't want this in general, it
    # is fragile (did you remember to update __init__ here when you added that new
    # class?). So just as much as it is useful we do a whack a mole here of quieting errors
    # we get in things like refractor.omi.
    #
    # Note we guard this with the standard "if typing.TYPE_CHECKING", so this code doesn't
    # appear in real python usage of this module.
    from .refractor_uip import RefractorUip
    from .current_state_uip import CurrentStateUip
    from .uip_updater import MaxAPosterioriSqrtConstraintUpdateUip
    from .muses_py_call import muses_py_call

del _i
del _re
del _os
del _glob
del _typing

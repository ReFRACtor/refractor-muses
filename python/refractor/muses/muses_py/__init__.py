# Not sure what all we want to import here. Potentially everything in
# py-retrieve, but that seems kinds of noisy. So at least for now we just
# include things we know we want. We can extend this in the future if needed,
# or come up with a more intelligent way of doing this. Ideally, py-retrieve
# would get a serious reorganization.

try:
    import py_retrieve.app
    have_muses_py = True
except ImportError:
    have_muses_py = False

import os as _os
import re as _re
import glob as _glob

# Give a cleaner message if somebody tries to call a function that isn't
# available
if(not have_muses_py):
    def __getattr__(name):
        raise NameError("py_retrieve is not available, so can't import '%s'" % name)

# Modules in py-retrieve that don't actually load correctly, or that
# don't actually want
_broken = ["Plotters", "haversine", "map_2points", "truncate_me", "verify_me"]
_i = None
if(have_muses_py):
    import py_retrieve.app.omi
    for _i in _glob.glob(py_retrieve.app.__path__[0] + "/*.py"):
        mname = _os.path.basename(_i).split('.')[0]
        if(not mname in _broken):
            #print("from py_retrieve.app.%s import *" % mname)
            exec("from py_retrieve.app.%s import *" % mname)
    for _i in _glob.glob(py_retrieve.app.omi.__path__[0] + "/*.py"):
        mname = _os.path.basename(_i).split('.')[0]
        if(not mname in _broken):
            exec("from py_retrieve.app.omi.%s import *" % mname)
    from py_retrieve.app.optimization.optimization import *
    from py_retrieve.app.optimization.lib import *
    from py_retrieve.app.refractor.replace_function import *
    
del _broken
del _i
del _re
del _os
del _glob

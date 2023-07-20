from collections import UserDict
import numpy as np

#============================================
# Note - This we something we tried out. It turned out to be
# more work than it was worth, so this is actually dead code.
# We'll leave this here in case we ever need to return to this,
# but for now this isn't actually used for anything
#============================================

class ConstantDict(UserDict):
    '''One of the issues with py-retrieve is that things get modified
    in multiple places - so you don't always know that an argument to
    a function doesn't get modified.

    This class is a simple wrapper around a dict, e.g., the UIP. It
    treats any attempt to modify this as an error if the attribute
    "writable" is False.

    To handle nesting, we return any dict as a ConstantDict and any
    numpy array as a view that has the flags.writable attribute set to
    the same state as this class.

    This allows us to prevent "surprises". If we think that a function
    shouldn't modify an argument then we can set writable as False. If
    the code does actually change this, we will catch this as a error.
    
    We can then either modify the function to *not* update this or
    alternatively just document the fact that it does.

    We could add an observer so objects get notified when this gets
    updated, but right now I don't have an immediate use for that.
    We can add it in the future if this proves useful.
    '''
    def __init__(self, d, writable=False):
        super().__init__()
        self.data = d
        self.writable  = writable

    def __delitem__(self, key):
        # Don't all items to be deleted unless we are writable
        if(not self.writable):
            raise ValueError("ConstantDict is marked as not writable, so can't delete item")
        super().__delitem__(key)

    def __setitem__(self, key, val):
        # Don't all items to be modified unless we are writable
        if(not self.writable):
            raise ValueError("ConstantDict is marked as not writable, so can't set item")
        super().__setitem__(key, val)

    def __getitem__(self, key):
        # Make sure any objects we returned propagate the constant
        # behavior. We handle numpy arrays and dict, I'm not sure if
        # we'll run into anything else that needs to be marked as
        # constant.
        #
        # Perhaps List? for now we'll just leave that out, but we can
        # look at adding that if needed.
        v = super().__getitem__(key)
        if(isinstance(v, dict)):
            return ConstantDict(v, writable=self.writable)
        if(isinstance(v, np.ndarray)):
            r = v.view()
            # numpy uses the alternative spelling writeable, so this is correct.
            # Note this is one of the odd words with two spellings
            r.flags.writeable = self.writable
            return r
        return v

    # Some of py-retrieve likes to have an ObjectView, so go ahead and toss in that
    # functionality
    def __getattr__(self, nm):
        if(nm in self.data):
            return self[nm]
        raise AttributeError

    def __setattr__(self, nm, value):
        if(nm in ("data", "writable")):
            super().__setattr__(nm, value)
        if(nm in self.data):
            self[nm] = value
        else:
            super().__setattr__(nm, value)
    
__all__ = ["ConstantDict",]

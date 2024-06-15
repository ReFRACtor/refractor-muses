import collections.abc
import os
import re
from .tes_file import TesFile

class RetrievalConfiguration(collections.abc.MutableMapping):
    '''There are a number of configuration parameters, e.g. directory for various outputs,
    run parameters like vlidort_nstokes etc.

    This class is little more than a dict which handles these values. Note that the
    "canonical" way to gets these values is to read a muses-py strategy table file.
    However it can be useful for testing to just set these values, or for some special
    test to just override something in a strategy table file after reading it. So
    we separate this configuration from a strategy table file - while most of the time
    you'll read one of these the rest of the code doesn't make any assumption about this.

    I'm not sure how to best capture what values are expected, since the list seems to
    be a bit dynamic (e.g., if you are using ILS then things like
    "apodizationMethodObs" are needed - otherwise not). At least for now, we make no
    assumption in this class than any particular value is here, you simply get an error
    if a value is looked for and not found.

    Note that a strategy table file tends to use relative paths from wherever the file
    is located. We translate these to absolute paths so you don't need to assume that you
    are in the same directory as the strategy table file.
    '''
    def __init__(self):
        self._data = {}

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, val):
        self._data[key] = val

    def __delitem__(self, key):
        del self._data[key]

    def __iter__(self):
        return self._data.__iter__()

    def __len__(self):
        return len(self._data)

    @classmethod
    def create_from_strategy_file(cls, fname, osp_dir=None):
        res = cls()
        strategy_table_dir = os.path.abspath(os.path.dirname(fname))
        strategy_table_fname = os.path.abspath(fname)
        f = TesFile(strategy_table_fname)
        res._data = dict(f)
        res._abs_dir(strategy_table_dir, osp_dir)

        # Start with default values, and then add anything we find in the table
        f = TesFile(f"{res['defaultStrategyTableDirectory']}/{res['defaultStrategyTableFilename']}")
        d = dict(f)
        d.update(res)
        res._data = d
        # There really should be a liteDirectory included here, but for some reason
        # muses-py treats this differently as a hard coded value - probably the general
        # problem of always solving problems locally rather than the best way.
        #
        # Go ahead and put into the data if it isn't there so we can treat this the
        # same everywhere.
        if('liteDirectory' not in res._data):
            res._data['liteDirectory'] = "../OSP/Lite/"
        res._abs_dir(strategy_table_dir, osp_dir)

        # There is a table included in the strategy table file that lists the required
        # options. Note sure if this is complete, but if we are missing one of these
        # then muses-py marks this as a failure
        f = TesFile(res["tableOptionsFilename"])
        for k in f.keys():
            if k not in res:
                raise RuntimeError(f"Required option {k} is not found in the file {fname}")

        # muses-py created some derived quantities. I think we can skip this, we'll at
        # least try that for now.
        return res
    
    def _abs_dir(self, base_dir, osp_dir):
        '''Convert values like ../OSP to the osp_dir passed in. Expand user ~ and
        environment variables. Convert relative paths to absolute paths.'''
        t = os.environ.get("strategy_table_dir")
        try:
            os.environ["strategy_table_dir"] = base_dir
            klist = self.keys() # Collect at beginning, since we update the dict
            for k in klist:
                v = self[k]
                v = os.path.expandvars(os.path.expanduser(v))
                m = re.match(r'^\.\./OSP/(.*)', v)
                if(m and osp_dir):
                    v = f"{osp_dir}/{m[1]}"
                if(re.match(r'^\.\./', v) or re.match(r'^\./', v)):
                    v = os.path.normpath(f"{base_dir}/{v}")
                self[k] = v
        finally:
            if(t is not None):
                os.environ["strategy_table_dir"] = t
            else:
                del os.environ["strategy_table_dir"]
        
__all__ = ["RetrievalConfiguration",]

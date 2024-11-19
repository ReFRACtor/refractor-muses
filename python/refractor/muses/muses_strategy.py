from .strategy_table import StrategyTable

class MusesStrategy:
    '''A MusesStrategy is a list of steps to be executed by a MusesStrategyExecutor.
    Each step is represented by a StrategyStep, which give the list
    of retrieval elements, spectral windows, etc.

    The canonical MusesStrategy is to read a Table.asc file (e.g., StrategyTable)
    to get the information about each step.

    Note that later steps can be changed based off of the results of previous
    steps. This part of the code may well need to be more developed, the only
    example we have right now is choosing steps based off of brightness temperature
    of the observation. If we get for examples of this, we may want to rework
    how conditional processing is handled.
    '''
    pass

class MusesStrategyOldStrategyTable(MusesStrategy):
    '''This wraps the old py-retrieve StrategyTable code as a MusesStrategy.
    Note that this class has largely been replaced with MusesStrategyTable,
    but we leave this in place for backwards testings.'''
    def __init__(self, filename : str, osp_dir=None):
        self._stable = StrategyTable(filename, osp_dir=osp_dir)

__all__ = ["MusesStrategy", "MusesStrategyOldStrategyTable"]

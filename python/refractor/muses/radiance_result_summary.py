from __future__ import annotations
import numpy as np

class RadianceResultSummary:
    '''Summarize radiance data with various statistics.'''
    def __init__(self, robs : np.ndarray, rad_calc : np.ndarray, nesr : np.ndarray,
                 min_count: int = 5) -> None:
        '''Take the observed radiance (from rstep) and the calculated radiance (from
        the ret_res of MusesLevmarSolver) and calculate statistics'''
        gpt = nesr > 0
        if(np.count_nonzero(gpt) > min_count):
            scaled_diff = (robs[gpt] - rad_calc[gpt]) / nesr[gpt]
            self._radiance_residual_mean = np.mean(scaled_diff)
        else:
            self._radiance_residual_mean = None

    @property
    def radiance_residual_mean(self) -> float | None:
        return self._radiance_residual_mean

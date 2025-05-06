from __future__ import annotations
import numpy as np
import math

class RadianceResultSummary:
    '''Summarize radiance data with various statistics.'''
    def __init__(self, robs : np.ndarray, rad_calc : np.ndarray, rad_initial: np.ndarray,
                 nesr : np.ndarray,
                 min_count: int = 5) -> None:
        '''Take the observed radiance (from rstep) and the calculated radiance (from
        the ret_res of MusesLevmarSolver) and calculate statistics'''
        self._radiance_residual_mean : float | None = None
        self._radiance_residual_rms : float | None = None
        self._radiance_residual_mean_initial : float | None = None
        self._radiance_residual_rms_initial : float | None = None
        self._radiance_snr : float | None = None
        self._radiance_residual_rms_relative_continuum : float | None = None
        self._radiance_continuum : float | None = None
        self._residual_slope : float | None = None
        self._residual_quadratic : float | None = None
        gpt = nesr > 0
        if(np.count_nonzero(gpt) > min_count):
            scaled_diff = (robs[gpt] - rad_calc[gpt]) / nesr[gpt]
            self._radiance_residual_mean = np.mean(scaled_diff)
            self._radiance_residual_rms = math.sqrt(np.var(scaled_diff))
            scaled_diff = (robs[gpt] - rad_initial[gpt]) / nesr[gpt]
            self._radiance_residual_mean_initial = np.mean(scaled_diff)
            self._radiance_residual_rms_initial = math.sqrt(np.var(scaled_diff))
            self._radiance_snr = np.mean(rad_calc[gpt] / nesr[gpt])
            
            valsv = np.sort(rad_calc[gpt])
            if(len(valsv) > 50):
                # TODO I don't think this actually does what is intended.
                # Should this be np.mean(vals[-50:]?
                vals = np.mean(valsv[int(len(valsv)*49/50):len(valsv)])
            else:
                vals = np.max(valsv)
            diff = (robs[gpt] - rad_calc[gpt])
            uu_var = np.var(diff)
            uu_mean = np.mean(diff)
            self._radiance_residual_rms_relative_continuum = math.sqrt(uu_var + uu_mean * uu_mean) / vals
            self._radiance_continuum = vals

            myx = rad_calc[gpt] / vals
            myy = (robs[gpt] - rad_calc[gpt]) / nesr[gpt]
            # cut off the very few points "above" the continuum
            indx = (np.where((myx > 0.0)*(myx < 1.00)))[0]
            myx = myx[indx]
            myy = myy[indx]
            linear_fit = np.polyfit(myx, myy, 1)
            quadratic_fit = np.polyfit(myx, myy, 2) 
            self._residual_slope = linear_fit[0]
            self._residual_quadratic = quadratic_fit[0]
            
            
    @property
    def radiance_residual_mean(self) -> float | None:
        return self._radiance_residual_mean

    @property
    def radiance_residual_rms(self) -> float | None:
        return self._radiance_residual_rms

    @property
    def radiance_residual_mean_initial(self) -> float | None:
        return self._radiance_residual_mean_initial

    @property
    def radiance_residual_rms_initial(self) -> float | None:
        return self._radiance_residual_rms_initial

    @property
    def radiance_snr(self) -> float | None:
        return self._radiance_snr

    @property
    def radiance_residual_rms_relative_continuum(self) -> float | None:
        return self._radiance_residual_rms_relative_continuum

    @property
    def radiance_continuum(self) -> float | None:
        return self._radiance_continuum

    @property
    def residual_slope(self) -> float | None:
        return self._residual_slope
        
    @property
    def residual_quadratic(self) -> float | None:
        return self._residual_quadratic
    
__all__ = ["RadianceResultSummary"]

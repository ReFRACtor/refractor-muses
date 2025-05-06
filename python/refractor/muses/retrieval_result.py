from __future__ import annotations
import refractor.muses.muses_py as mpy  # type: ignore
from .observation_handle import mpy_radiance_from_observation_list
from .identifier import StateElementIdentifier
from .filter_result_summary import FilterResultSummary
from .radiance_result_summary import RadianceResultSummary
from .cloud_result_summary import CloudResultSummary
from .column_result_summary import ColumnResultSummary
import math
import numpy as np
from typing import Any
import typing

if typing.TYPE_CHECKING:
    from .cost_function import CostFunction
    from .muses_observation import MusesObservation
    from .current_state import CurrentState, PropagatedQA
    from .error_analysis import ErrorAnalysis


class RetrievalResult:
    """There are a few top level functions that work with a structure called
    retrieval_results. Pull all this together into an object so we can clearly
    see the interface and possibly change things.

    The coupling of this isn't great, we may want to break this up. The current
    set of steps needed to fully generate this can be found in
    RetrievalStrategyStepRetrieve

    1. Use results of the run_retrieval to create RetrievalResult using the
       constructor (__init__).
    2. Generate the systematic Jacobian using this updated StateInfo, and
       store results in RetrievalResult
    3. Run the error analysis (ErrorAnalysis.update_retrieval_result) and
       store the results in RetrievalResult
    4. Run the QA analysis (QaDataHandleSet.qa_update_retrieval_result) and
       store the results in RetrievalResult

    """

    def __init__(
        self,
        ret_res: dict,
        current_state: CurrentState,
        obs_list: list[MusesObservation],
        radiance_full: dict,
        propagated_qa: PropagatedQA,
        jacobian_sys: np.ndarray | None = None,
    ):
        """ret_res is what we get returned from MusesLevmarSolver"""
        self.rstep = mpy.ObjectView(
            mpy_radiance_from_observation_list(obs_list, include_bad_sample=True)
        )
        self._filter_result_summary = FilterResultSummary(self.rstep)
        self.radiance_full = radiance_full
        self.obs_list = obs_list
        self.instruments = [obs.instrument_name for obs in self.obs_list]
        self.current_state = current_state
        self.sounding_metadata = current_state.sounding_metadata
        self.ret_res = mpy.ObjectView(ret_res)
        self.jacobianSys = jacobian_sys
        self.propagated_qa = propagated_qa
        # Get old retrieval results structure, and merge in with this object
        d = self.set_retrieval_results()
        self.__dict__.update(d)
        # Filled in with ErrorAnalysis.write_retrieval_summary. We may move some of
        # this calculation to this class, but for now it gets done there.
        rowsSys = len(self.current_state.systematic_model_state_vector_element_list)
        rowsFM = len(self.current_state.forward_model_state_vector_element_list)
        num_species = len(self.current_state.retrieval_state_element_id)
        # This really is exactly 5. See the column calculation. This is
        # ["Column", "Trop", "UpperTrop", "LowerTrop", ""Strato
        num_col = 5 
        # TODO Would be good to calculate this somewhat
        max_num_species = 20
        if rowsSys == 0:
            rowsSys = 1
        self.cloudODVar = 0.0
        self.cloudODAveError = 0.0
        self.emisDev = 0.0
        self.emissionLayer = 0.0
        self.H2O_H2OQuality = 0.0
        self.O3_columnErrorDU = 0.0
        self.O3_tropo_consistency = 0.0
        self.ozoneCcurve = 0.0
        self.ozone_slope_QA = -999.0
        self.errorFM = np.zeros(shape=(rowsFM), dtype=np.float64)
        self.columnDOFS = np.zeros(shape=(num_col, max_num_species), dtype=np.float32)
        self.A =  np.zeros(shape=(rowsFM, rowsFM), dtype=np.float32)
        self.Sa =  np.zeros(shape=(rowsFM, rowsFM), dtype=np.float64)
        self.Sb =  np.zeros(shape=(rowsFM, rowsFM), dtype=np.float64)
        self.Sx =  np.zeros(shape=(rowsFM, rowsFM), dtype=np.float64)
        self.columnPriorError = np.full((num_col, max_num_species), -999, dtype=np.float64)
        self.columnInitial = np.full((num_col, max_num_species), -999, dtype=np.float64)
        self.columnInitialInitial = np.full((num_col, max_num_species), -999, dtype=np.float64)
        self.columnError= np.full((num_col, max_num_species), -999, dtype=np.float64)
        self.columnPrior= np.full((num_col, max_num_species), -999, dtype=np.float64)
        self.column = np.full((num_col, max_num_species), -999, dtype=np.float64)
        self.columnAir = np.full((num_col), -999, dtype=np.float64)
        self.columnTrue = np.full((num_col, max_num_species), -999, dtype=np.float64)
        self.columnPressureMax = np.zeros(shape=(num_col), dtype=np.float32)
        self.columnPressureMin = np.zeros(shape=(num_col), dtype=np.float32)
        self.columnSpecies = ["",] * max_num_species
        self.DeviationBad_QA = np.full((num_species), -999, dtype=np.int32)
        self.num_deviations_QA = np.full((num_species), -999, dtype=np.int32)
        self.deviation_QA = np.full((num_species), -999, dtype=np.float32)

        self._radiance_result_summary = [RadianceResultSummary(self.rstep.radiance[slc],
                                                               self.radiance[0,slc],
                                                               self.radianceInitial[0,slc],
                                                               self.rstep.NESR[slc])
                                         for slc in self._filter_result_summary.filter_slice]
        
    def update_jacobian_sys(self, cfunc_sys: CostFunction) -> None:
        """Run the forward model in cfunc to get the jacobian_sys set."""
        self.jacobianSys = (
            cfunc_sys.max_a_posteriori.model_measure_diff_jacobian.transpose()[
                np.newaxis, :, :
            ]
        )

    def update_error_analysis(self, error_analysis : ErrorAnalysis):
        self._cloud_result_summary = CloudResultSummary(self, error_analysis)
        self._column_result_summary = ColumnResultSummary(self, error_analysis)

    def state_value(self, state_name : str) -> float:
        return self.current_state.full_state_value(StateElementIdentifier(state_name))[0]

    def state_value_vec(self, state_name : str) -> np.ndarray:
        return self.current_state.full_state_value(StateElementIdentifier(state_name))
    
    @property
    def tropopause_pressure(self) -> float:
        res = self.state_value("gmaoTropopausePressure")
        if(res <= -990):
            raise RuntimeError("GMA tropopause pressure is not defined")
        return res

    @property
    def tropopausePressure(self) -> float:
        return self.tropopause_pressure
    
    @property
    def omi_cloudfraction(self) -> float:
        return self.state_value("OMICLOUDFRACTION")

    @property
    def tropomi_cloudfraction(self) -> float:
        return self.state_value("TROPOMICLOUDFRACTION")

    @property
    def radiance_initial(self) -> np.ndarray:
        if(self.num_iterations == 0):
            return self.ret_res.radiance["radiance"][:, :]
        return self.ret_res.radianceIterations[0, :, :]

    @property
    def radianceInitial(self) -> np.ndarray:
        return self.radiance_initial
    
    @property
    def LMResults_costThresh(self) -> np.ndarray:
        return self.ret_res.stopCriteria[:, 0]

    @property
    def LMResults_resNorm(self) -> np.ndarray:
        return self.ret_res.resdiag[:, 0]

    @property
    def LMResults_resNormNext(self) -> np.ndarray:
        return self.ret_res.resdiag[:, 1]

    @property
    def LMResults_jacresNorm(self) -> np.ndarray:
        return self.ret_res.resdiag[:, 2]

    @property
    def LMResults_jacResNormNext(self) -> np.ndarray:
        return self.ret_res.resdiag[:, 3]

    @property
    def LMResults_pnorm(self) -> np.ndarray:
        return self.ret_res.resdiag[:, 4]

    @property
    def LMResults_delta(self) -> np.ndarray:
        return self.ret_res.delta

    @property
    def LMResults_iterList(self) -> np.ndarray:
        res = [self.current_state.initial_guess]
        for i in range(1, self.ret_res.num_iterations + 1):
            res.append(self.ret_res.xretIterations[i, :])
        return np.vstack(res)

    @property
    def frequency(self) -> np.ndarray:
        return self.rstep.frequency

    @property
    def radiance(self) -> np.ndarray:
        if(len(self.ret_res.radiance["radiance"].shape) == 1):
            return self.ret_res.radiance["radiance"][np.newaxis,:]
        return self.ret_res.radiance["radiance"]

    @property
    def radianceObserved(self) -> np.ndarray:
        return self.rstep.radiance

    @property
    def NESR(self) -> np.ndarray:
        return self.rstep.NESR

    @property
    def jacobian(self) -> np.ndarray:
        return self.ret_res.jacobian["jacobian_data"][np.newaxis,:,:]
    
    @property
    def resultsList(self) -> np.ndarray:
        return self.current_state.initial_guess if self.best_iteration == 0 else self.ret_res.xretIterations[self.best_iteration, :]

    @property
    def resultsListFM(self) -> np.ndarray:
        return self.ret_res.xretFM

    @property
    def is_ocean(self) -> bool:
        return self.current_state.sounding_metadata.is_ocean

    @property
    def retIteration(self) -> int:
        return self.ret_res.xretIterations
    
    @property
    def bestIteration(self) -> int:
        return self.ret_res.bestIteration
    
    @property
    def num_iterations(self) -> int:
        return self.ret_res.num_iterations

    @property
    def stopCode(self) -> int:
        return self.ret_res.stopCode

    @property
    def Desert_Emiss_QA(self) -> float:
        wlen = self.current_state.full_state_spectral_domain_wavelength(
            StateElementIdentifier("emissivity")
        )
        if wlen is None:
            raise RuntimeError("Expected to find emissivity frequencies")
        ind = np.argmin(np.abs(wlen - 1025))
        return self.state_value_vec("emissivity")[ind]
        
    @property
    def species_list_fm(self) -> list[str]:
        """This is the length of the forward model state vector, with a
        retrieval_element name for each location."""
        return [
            str(i) for i in self.current_state.forward_model_state_vector_element_list
        ]

    @property
    def species_list_retrieval(self) -> list[str]:
        """This is the length of the retrieval state vector, with a
        retrieval_element name for each location."""
        return [str(i) for i in self.current_state.retrieval_state_element_id]

    @property
    def pressure_list_fm(self) -> np.ndarray:
        pdata = []
        # Convention of muses-py is to use [-2] for items that aren't on
        # pressure levels
        for sid in self.current_state.retrieval_state_element_id:
            d = self.current_state.pressure_list_fm(sid)
            if d is not None:
                pdata.append(d)
            else:
                pdata.append(np.array([-2.0]))
        return np.concatenate(pdata)

    @property
    def best_iteration(self) -> int:
        return self.bestIteration  # type: ignore[attr-defined]

    @property
    def results_list(self) -> np.ndarray:
        return self.resultsList  # type: ignore[attr-defined]

    @property
    def master_quality(self) -> int:
        return self.masterQuality

    @master_quality.setter
    def master_quality(self, val: int) -> None:
        self.masterQuality = val

    @property
    def jacobian_sys(self) -> np.ndarray | None:
        return self.jacobianSys

    @property
    def filter_index(self) -> list[int]:
        return self._filter_result_summary.filter_index
    
    @property
    def filter_list(self) -> list[str]:
        return self._filter_result_summary.filter_list
    
    @property
    def filterStart(self) -> list[int]:
        return self._filter_result_summary.filter_start

    @property
    def filterEnd(self) -> list[int]:
        return self._filter_result_summary.filter_end
    
    @property
    def propagatedTATMQA(self) -> int:
        return self.propagated_qa.tatm_qa
    
    @property
    def propagatedO3QA(self) -> int:
        return self.propagated_qa.o3_qa
    
    @property
    def propagatedH2OQA(self) -> int:
        return self.propagated_qa.h2o_qa

    @property
    def radianceMaximumSNR(self) -> float:
        gpt = self.rstep.frequency > 0.0
        return np.amax(self.rstep.radiance[gpt] / self.rstep.NESR[gpt])

    def _handle_fill(self, r : list[float|None], fill_zero : float = 0.0,
                     fill_rest : float = -999.0) -> list[float]:
        # Different fill value for first entry. Odd, but the output separate these
        # out and treats the "ALL" filter at the front differently
        if(r[0] is None):
            r[0] = fill_zero
        return([ fill_rest if i is None else i for i in r])
            
    @property
    def radianceResidualMean(self) -> np.ndarray:
        res = [ r.radiance_residual_mean for r in self._radiance_result_summary]
        return np.array(self._handle_fill(res))

    @property
    def radianceResidualRMS(self) -> np.ndarray:
        res = [ r.radiance_residual_rms for r in self._radiance_result_summary]
        return np.array(self._handle_fill(res))

    @property
    def radianceResidualMeanInitial(self) -> np.ndarray:
        res = [ r.radiance_residual_mean_initial for r in self._radiance_result_summary]
        return np.array(self._handle_fill(res))

    @property
    def radianceResidualRMSInitial(self) -> np.ndarray:
        res = [ r.radiance_residual_rms_initial for r in self._radiance_result_summary]
        return np.array(self._handle_fill(res))

    @property
    def radianceResidualRMSRelativeContinuum(self) -> np.ndarray:
        res = [ r.radiance_residual_rms_relative_continuum for r in self._radiance_result_summary]
        return np.array(self._handle_fill(res,0.0,0.0))

    @property
    def radianceContinuum(self) -> np.ndarray:
        res = [ r.radiance_continuum for r in self._radiance_result_summary]
        return np.array(self._handle_fill(res,0.0,0.0))

    @property
    def radianceSNR(self) -> np.ndarray:
        res = [ r.radiance_snr for r in self._radiance_result_summary]
        return np.array(self._handle_fill(res,0.0,0.0))
    

    @property
    def residualNormInitial(self) -> float:
        t1 = self.radianceResidualMeanInitial[0]
        t2 = self.radianceResidualRMSInitial[0]
        return math.sqrt(t1*t1+t2*t2)

    @property
    def residualNormFinal(self) -> float:
        t1 = self.radianceResidualMean[0]
        t2 = self.radianceResidualRMS[0]
        return math.sqrt(t1*t1+t2*t2)

    @property
    def residualSlope(self) -> np.ndarray:
        res = [ r.residual_slope for r in self._radiance_result_summary]
        return np.array(self._handle_fill(res,-999.0,-999.0))

    @property
    def residualQuadratic(self) -> np.ndarray:
        res = [ r.residual_quadratic for r in self._radiance_result_summary]
        return np.array(self._handle_fill(res,-999.0,-999.0))

    @property
    def cloud_factor(self) -> float:
        scale_pressure = self.state_value("scalePressure")
        if(scale_pressure == 0):
            scale_pressure = 0.1
        res = mpy.compute_cloud_factor(
            self.state_value_vec("pressure"),
            self.state_value_vec("TATM"),
            self.state_value_vec("H2O"),
            self.state_value("PCLOUD"),
            scale_pressure,
            self.current_state.sounding_metadata.surface_altitude.value*1000,
            self.current_state.sounding_metadata.latitude.value,
        )
        # TODO Rounding currently done. I', not sure this makes a lot of sense,
        # this was to match the old IDL code. I don't know that we actually want
        # to do that, but for now have this in place.
        res = round(res, 7)
        return res

    @property
    def cloudODAve(self) -> float:
        freq = self.current_state.full_state_spectral_domain_wavelength(
            StateElementIdentifier("cloudEffExt")
        )
        if(freq is None):
            raise RuntimeError("This shouldn't happen")
        ind = np.where(
            (freq >= 974) & 
            (freq <= 1201)
        )[0]
        ceffect = self.state_value_vec("cloudEffExt")
        if len(ind) > 0:
            res = np.sum(ceffect[0, ind]) / len(ceffect[0, ind]) * self.cloud_factor
        else:
            res = 0
        return res
    
    def set_retrieval_results(self) -> dict:
        """This is our own copy of mpy.set_retrieval_results, so we
        can start making changes to clean up the coupling of this.

        """
        # Convert any dict to ObjectView so we can have a consistent
        # way of referring to our input.
        num_species = len(self.current_state.retrieval_state_element_id)
        nfreqs = len(self.rstep.frequency)
        num_filters = len(self.filter_index)
        
        # get the total number of frequency points in all microwindows for the
        # gain matrix
        rows = len(self.current_state.retrieval_state_vector_element_list)
        rowsSys = len(self.current_state.systematic_model_state_vector_element_list)
        rowsFM = len(self.current_state.forward_model_state_vector_element_list)
        if rowsSys == 0:
            rowsSys = 1

        o_results: dict[str, Any] = {
            "radianceResidualRMSSys": 0.0,
            "error": np.zeros(shape=(rows), dtype=np.float64),
            "precision": np.zeros(shape=(rowsFM), dtype=np.float64),
            "resolution": np.zeros(shape=(rowsFM), dtype=np.float64),
            # jacobians - for last outputStep
            "GdL": np.zeros(shape=(nfreqs, rowsFM), dtype=np.float64),
            "jacobianSys": None,
            # error stuff follows - calc later
            "A_ret": np.zeros(shape=(rows, rows), dtype=np.float32),
            "KtSyK": np.zeros(shape=(rows, rows), dtype=np.float32),
            "Sa_ret": np.zeros(shape=(rows, rows), dtype=np.float64),
            "Sx_ret_smooth": np.zeros(shape=(rows, rows), dtype=np.float64),
            "Sx_ret_crossState": np.zeros(shape=(rows, rows), dtype=np.float64),
            "Sx_ret_rand": np.zeros(shape=(rows, rows), dtype=np.float64),
            "Sx_ret_sys": np.zeros(shape=(rows, rows), dtype=np.float64),
            "Sx_ret_mapping": np.zeros(shape=(rows, rows), dtype=np.float64),
            "Sx_smooth_self": np.zeros(shape=(rowsFM, rowsFM), dtype=np.float64),
            "Sx_smooth": np.zeros(shape=(rowsFM, rowsFM), dtype=np.float64),
            "Sx_crossState": np.zeros(shape=(rowsFM, rowsFM), dtype=np.float64),
            "Sx_sys": np.zeros(shape=(rowsFM, rowsFM), dtype=np.float64),
            "Sx_rand": np.zeros(shape=(rowsFM, rowsFM), dtype=np.float64),
            "Sx_mapping": np.zeros(shape=(rowsFM, rowsFM), dtype=np.float64),
            "SxActual": np.zeros(shape=(rowsFM, rowsFM), dtype=np.float64),
            "GMatrix": np.zeros(shape=(nfreqs, rows), dtype=np.float64),
            "GMatrixFM": np.zeros(shape=(nfreqs, rowsFM), dtype=np.float64),
            # by species
            "informationContentSpecies": np.zeros(
                shape=(num_species), dtype=np.float64
            ),
            "degreesOfFreedomNoise": np.zeros(shape=(num_species), dtype=np.float64),
            "degreesOfFreedomForSignal": np.zeros(
                shape=(num_species), dtype=np.float64
            ),
            "degreesOfFreedomForSignalTrop": np.zeros(
                shape=(num_species), dtype=np.float64
            ),
            "bestDegreesOfFreedomList": ["" for x in range(num_species)],
            "bestDegreesOfFreedomTotal": ["" for x in range(num_species)],
            "verticalResolution": np.zeros(shape=(num_species), dtype=np.float64),
            "deviationVsError": 0.0,
            "deviationVsRetrievalCovariance": 0.0,
            "deviationVsAprioriCovariance": 0.0,
            "deviationVsErrorSpecies": np.zeros(shape=(num_species), dtype=np.float64),
            "deviationVsRetrievalCovarianceSpecies": np.zeros(
                shape=(num_species), dtype=np.float64
            ),
            "deviationVsAprioriCovarianceSpecies": np.zeros(
                shape=(num_species), dtype=np.float64
            ),
            # quality and general
            "KDotDL": 0.0,
            "KDotDL_list": np.zeros(shape=(rows), dtype=np.float32),
            "KDotDL_byspecies": np.zeros(shape=(num_species), dtype=np.float32),
            "KDotDL_species": ["" for x in range(num_species)],
            "KDotDL_byfilter": np.zeros(shape=(num_filters), dtype=np.float32),
            "maxKDotDLSys": 0.0,
        }

        struct2 = {
            "LDotDL": 0.0,
            "LDotDL_byfilter": np.zeros(shape=(num_filters), dtype=np.float32),
            "calscaleMean": 0.0,
            "masterQuality": -999,
            # EM NOTE - Modified to increase vector size to allow for stratosphere capture
            "tsur_minus_tatm0": -999.0,
            "tsur_minus_prior": -999.0,
            "ch4_evs": np.zeros(shape=(10), dtype=np.float32),  # FLTARR(10)
        }
        o_results.update(struct2)
        return o_results

__all__ = [
    "RetrievalResult",
]

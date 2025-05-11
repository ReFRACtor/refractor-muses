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
    from .muses_levmar_solver import SolverResult


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
        ret_res: SolverResult,
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
        self.ret_res = ret_res
        self.jacobianSys = jacobian_sys
        self.propagated_qa = propagated_qa
        # Filled in with ErrorAnalysis.write_retrieval_summary. We may move some of
        # this calculation to this class, but for now it gets done there.
        rowsSys = len(self.current_state.systematic_model_state_vector_element_list)
        num_species = len(self.current_state.retrieval_state_element_id)
        # This really is exactly 5. See the column calculation. This is
        # ["Column", "Trop", "UpperTrop", "LowerTrop", ""Strato
        num_col = 5 
        # TODO Would be good to calculate this somewhat
        max_num_species = 20
        if rowsSys == 0:
            rowsSys = 1

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

    def update_error_analysis(self, error_analysis : ErrorAnalysis) -> None:
        self._error_analysis = error_analysis
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
    def retIteration(self) -> np.ndarray:
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
        return self.bestIteration

    @property
    def results_list(self) -> np.ndarray:
        return self.resultsList

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
    def Sb(self) -> np.ndarray:
        return self._error_analysis.Sb

    @property
    def Sa(self) -> np.ndarray:
        return self._error_analysis.Sa

    @property
    def Sa_ret(self) -> np.ndarray:
        return self._error_analysis.Sa_ret
    
    @property
    def KtSyK(self) -> np.ndarray:
        return self._error_analysis.KtSyK
    
    @property
    def KtSyKFM(self) -> np.ndarray:
        return self._error_analysis.KtSyKFM

    @property
    def Sx(self) -> np.ndarray:
        return self._error_analysis.Sx

    @property
    def Sx_smooth(self) -> np.ndarray:
        return self._error_analysis.Sx_smooth

    @property
    def Sx_ret_smooth(self) -> np.ndarray:
        return self._error_analysis.Sx_ret_smooth
    
    @property
    def Sx_smooth_self(self) -> np.ndarray:
        return self._error_analysis.Sx_smooth_self

    @property
    def Sx_crossState(self) -> np.ndarray:
        return self._error_analysis.Sx_crossState

    @property
    def Sx_ret_crossState(self) -> np.ndarray:
        return self._error_analysis.Sx_ret_crossState
    
    @property
    def Sx_rand(self) -> np.ndarray:
        return self._error_analysis.Sx_rand

    @property
    def Sx_ret_rand(self) -> np.ndarray:
        return self._error_analysis.Sx_ret_rand

    @property
    def Sx_ret_mapping(self) -> np.ndarray:
        return self._error_analysis.Sx_ret_mapping
    
    @property
    def Sx_sys(self) -> np.ndarray:
        return self._error_analysis.Sx_sys

    @property
    def Sx_ret_sys(self) -> np.ndarray:
        return self._error_analysis.Sx_ret_sys
    
    @property
    def radianceResidualRMSSys(self) -> float:
        return self._error_analysis.radianceResidualRMSSys

    @property
    def A(self) -> np.ndarray:
        return self._error_analysis.A

    @property
    def A_ret(self) -> np.ndarray:
        return self._error_analysis.A_ret
    
    @property
    def GMatrix(self) -> np.ndarray:
        return self._error_analysis.GMatrix

    @property
    def GMatrixFM(self) -> np.ndarray:
        return self._error_analysis.GMatrixFM
    
    @property
    def deviationVsErrorSpecies(self) -> np.ndarray:
        return self._error_analysis.deviationVsErrorSpecies

    @property
    def deviationVsRetrievalCovarianceSpecies(self) -> np.ndarray:
        return self._error_analysis.deviationVsRetrievalCovarianceSpecies
        
    @property
    def deviationVsAprioriCovarianceSpecies(self) -> np.ndarray:
        return self._error_analysis.deviationVsAprioriCovarianceSpecies
            
    @property
    def degreesOfFreedomForSignal(self) -> np.ndarray:
        return self._error_analysis.degreesOfFreedomForSignal
                
    @property
    def degreesOfFreedomNoise(self) -> np.ndarray:
        return self._error_analysis.degreesOfFreedomNoise

    @property
    def KDotDL_list(self) -> np.ndarray:
        return self._error_analysis.KDotDL_list
        
    @property
    def KDotDL(self) -> float:
        return self._error_analysis.KDotDL
            
    @property
    def KDotDL_species(self) -> list[str]:
        return self._error_analysis.KDotDL_species
                
    @property
    def KDotDL_byspecies(self) -> np.ndarray:
        return self._error_analysis.KDotDL_byspecies
                    
    @property
    def LDotDL(self) -> np.ndarray:
        return self._error_analysis.LDotDL
                        
    @property
    def KDotDL_byfilter(self) -> np.ndarray:
        return self._error_analysis.KDotDL_byfilter
                            
    @property
    def maxKDotDLSys(self) -> float:
        return self._error_analysis.maxKDotDLSys
                                
    @property
    def errorFM(self) -> np.ndarray:
        return self._error_analysis.errorFM
                                    
    @property
    def precision(self) -> np.ndarray:
        return self._error_analysis.precision
                                        
    @property
    def GdL(self) -> np.ndarray:
        return self._error_analysis.GdL
                                            
    @property
    def ch4_evs(self) -> np.ndarray:
        return self._error_analysis.ch4_evs
    
    @property
    def cloudODAve(self) -> float:
        return self._cloud_result_summary.cloudODAve

    @property
    def cloudODVar(self) -> float:
        return self._cloud_result_summary.cloudODVar

    @property
    def cloudODAveError(self) -> float:
        return self._cloud_result_summary.cloudODAveError

    @property
    def emisDev(self) -> float:
        return self._cloud_result_summary.emisDev

    @property
    def emissionLayer(self) -> float:
        return self._cloud_result_summary.emissionLayer

    @property
    def ozoneCcurve(self) -> float:
        return self._cloud_result_summary.ozoneCcurve

    @property
    def ozone_slope_QA(self) -> float:
        return self._cloud_result_summary.ozone_slope_QA

    @property
    def deviation_QA(self) -> np.ndarray:
        return self._cloud_result_summary.deviation_QA

    @property
    def num_deviations_QA(self) -> np.ndarray:
        return self._cloud_result_summary.num_deviations_QA

    @property
    def DeviationBad_QA(self) -> np.ndarray:
        return self._cloud_result_summary.DeviationBad_QA

    @property
    def H2O_H2OQuality(self) -> float:
        return self._column_result_summary.H2O_H2OQuality
    
    @property
    def O3_columnErrorDU(self) -> float:
        return self._column_result_summary.O3_columnErrorDU
    
    @property
    def O3_tropo_consistency(self) -> float:
        return self._column_result_summary.O3_tropo_consistency
    
    @property
    def columnDOFS(self) -> np.ndarray:
        return self._column_result_summary.columnDOFS
    
    @property
    def columnPriorError(self) -> np.ndarray:
        return self._column_result_summary.columnPriorError
    
    @property
    def columnInitial(self) -> np.ndarray:
        return self._column_result_summary.columnInitial
    
    @property
    def columnInitialInitial(self) -> np.ndarray:
        return self._column_result_summary.columnInitialInitial
    
    @property
    def columnError(self) -> np.ndarray:
        return self._column_result_summary.columnError
    
    @property
    def columnPrior(self) -> np.ndarray:
        return self._column_result_summary.columnPrior
    
    @property
    def column(self) -> np.ndarray:
        return self._column_result_summary.column
    
    @property
    def columnAir(self) -> np.ndarray:
        return self._column_result_summary.columnAir
    
    @property
    def columnTrue(self) -> np.ndarray:
        return self._column_result_summary.columnTrue
    
    @property
    def columnPressureMax(self) -> np.ndarray:
        return self._column_result_summary.columnPressureMax
    
    @property
    def columnPressureMin(self) -> np.ndarray:
        return self._column_result_summary.columnPressureMin
    
    @property
    def columnSpecies(self) -> list[str]:
        return self._column_result_summary.columnSpecies
        
    @property
    def calscaleMean(self) -> float:
        # Seems to be an old value, not actually calculated anymore. But still needed
        # for output
        return 0.0

    @property
    def tsur_minus_prior(self) -> float:
        # Seems to be an old value, not actually calculated anymore. But still needed
        # for output
        return -999.0

    @property
    def tsur_minus_tatm0(self) -> float:
        # Seems to be an old value, not actually calculated anymore. But still needed
        # for output
        return -999.0
    
__all__ = [
    "RetrievalResult",
]

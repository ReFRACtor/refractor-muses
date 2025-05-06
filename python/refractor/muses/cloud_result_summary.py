from __future__ import annotations
import refractor.muses.muses_py as mpy  # type: ignore
from .fake_state_info import FakeStateInfo
from .fake_retrieval_info import FakeRetrievalInfo
import numpy as np
import math
import typing

if typing.TYPE_CHECKING:
    from .retrieval_result import RetrievalResult
    from .error_analysis import ErrorAnalysis

# Needs a lot of cleanup, we are just shoving stuff into place
class CloudResultSummary:
    def __init__(self, retrieval_result: RetrievalResult, error_analysis : ErrorAnalysis) -> None:
        utilList = mpy.UtilList()
        utilGeneral = mpy.UtilGeneral()
        stateInfo = FakeStateInfo(retrieval_result.current_state)
        retrievalInfo = FakeRetrievalInfo(retrieval_result.current_state)
        errorCurrent = error_analysis.error_current

        num_species = retrievalInfo.n_species
        
        retrieval_result.cloudODVar = 0
        retrieval_result.cloudODAveError = 0
    
        # AT_LINE 57 Write_Retrieval_Summary.pro
        factor = retrieval_result.cloud_factor
    
        # step = 1
        # stepName = TATM,H2O,HDO,N2O,CH4,TSUR,CLOUDEXT,EMIS
        # product name Products_Jacobian-TATM,H2O,HDO,N2O,CH4,TSUR,CLOUDEXT-bar_land.nc
            
        ind = np.where(np.asarray(retrievalInfo.speciesList) == 'CLOUDEXT')[0]
        indFM = np.where(np.asarray(retrievalInfo.speciesListFM) == 'CLOUDEXT')[0]
    
        # NOTE: mpy.get_one_map will return maps that have columns and rows switched compared to the IDL implementation
        my_map = mpy.get_one_map(retrievalInfo, 'CLOUDEXT')
    
        # AT_LINE 77 Write_Retrieval_Summary.pro
        if len(ind) > 0:
            # map error to ret
            errlog = retrieval_result.errorFM[indFM] @ my_map['toPars']
            
            cloudod = np.exp(retrieval_result.resultsList[ind]) * factor
            err = (np.exp(np.log(cloudod) + errlog) - cloudod) * factor
            myMean = np.sum(cloudod / err / err) / np.sum(1 / err / err)
    
            if myMean == np.nan:
                myMean = np.mean(cloudod)
                err[:] = myMean
    
            if np.nan in err:
                myMean = np.mean(cloudod)
                err[:] = myMean
    
            if np.inf in err:
                myMean = np.mean(cloudod)
                err[:] = myMean
    
            x = np.var((cloudod - myMean) / err, ddof=1)
            retrieval_result.cloudODVar = math.sqrt(x)
        else:
            # cloud not retrieved... use 975-1200
            # NOTE: code has not been tested.
    
            # Look through all indices that meet the pressure criteria, and look to see if same index in errorCurrent.species matches with 'CLOUDEXT'.
            ind4 : list[int] = []
            for ii in range(0, len(errorCurrent.pressure)):
                if (errorCurrent.pressure[ii] >= 975 and errorCurrent.pressure[ii] <= 1200) and (errorCurrent.species[ii] == 'CLOUDEXT'):
                    if ii not in ind4:
                        ind4.append(ii)
    
            if len(ind4) > 0:
                # found 975-1200
                # error_current = utilGeneral.ManualArrayGetWithRHSIndices(errorCurrent.data, ind, ind)
                error_current = errorCurrent.data[ind4, ind4]
    
                retrieval_result.cloudODAveError = math.sqrt(np.sum(error_current)) / len(ind4) * retrieval_result.cloudODAve
    
                ind3 = np.where(
                    (stateInfo.cloudPars['frequency'] >= 974) & 
                    (stateInfo.cloudPars['frequency'] <= 1201)
                )[0]
    
                cloudod = stateInfo.current['cloudEffExt'][0, ind3] * factor
    
                # error_current = utilGeneral.ManualArrayGetWithRHSIndices(errorCurrent.data, ind, ind)
                error_current = errorCurrent.data[ind3, ind3]
    
                err = stateInfo.current['cloudEffExt'][0, ind3] * np.sqrt(error_current)
                myMean = np.sum(cloudod / err / err) / np.sum(1 / err / err)
    
                if myMean == np.nan:
                    myMean = np.mean(cloudod)
                    err[:] = myMean
    
                if np.nan in err:
                    myMean = np.mean(cloudod)
                    err[:] = myMean
    
                if np.inf in err:
                    myMean = np.mean(cloudod)
                    err[:] = myMean
    
                x = np.var((cloudod - myMean) / err, ddof=1)
            # end if len(ind) > 0:
        # end else part of if (len(ind) > 0):
    
        # AT_LINE 107 Write_Retrieval_Summary.pro
        ind = np.where(np.asarray(retrievalInfo.speciesList) == 'CLOUDEXT')[0]
        indFM = np.where(np.asarray(retrievalInfo.speciesListFM) == 'CLOUDEXT')[0]
    
        # NOTE: mpy.get_one_map will return maps that have columns and rows switched compared to the IDL implementation
        my_map = mpy.get_one_map(retrievalInfo, 'CLOUDEXT')
            
        if len(ind) > 0:
            errlog = retrieval_result.errorFM[indFM] @ my_map['toPars']
    
            cloudod = np.exp(retrieval_result.resultsList[ind]) * factor
            err = (np.exp(np.log(cloudod) + errlog) - cloudod) * factor
            myMean = np.sum(cloudod / err / err) / np.sum(1 / err / err)
                
            x = np.var((cloudod - myMean) / err, ddof=1)
            retrieval_result.cloudODVar = math.sqrt(x)
    
        ind = np.where(
            (stateInfo.emisPars['frequency'] >= 975) & 
            (stateInfo.emisPars['frequency'] <= 1200)
        )[0]
    
        if len(ind) > 0:
            retrieval_result.emisDev = np.mean(stateInfo.current['emissivity'][ind]) - np.mean(stateInfo.constraint['emissivity'][ind])
    
        ind10 = np.asarray([])  # Start with an empty list so we have the variable set.
        if 'O3' in retrievalInfo.speciesListFM:
            ind10 = utilList.WhereEqualIndices(retrievalInfo.speciesListFM, 'O3')
    
        # AT_LINE 162 Write_Retrieval_Summary.pro
        if len(ind10) > 0:
            indt = np.where(np.array(stateInfo.species) == 'TATM')[0][0]
            TATM = stateInfo.current['values'][indt, :]
            TSUR = stateInfo.current['TSUR']
            
            indo3 = np.where(np.array(stateInfo.species) == 'O3')[0][0]
            o3 = stateInfo.current['values'][indo3, :]
            o3ig = stateInfo.constraint['values'][indo3, :]
            
            aveTATM = 0.0
            my_sum = 0.0
            for ii in range(0, 3):
                my_sum = my_sum + o3[ii] - o3ig[ii]
                aveTATM = aveTATM + TATM[ii]
            
            aveTATM = aveTATM / 3
            if my_sum/3 >= 1.5e-9:
                retrieval_result.emissionLayer = aveTATM - TSUR
    
    
        retrieval_result.ozoneCcurve = 1
        retrieval_result.ozone_slope_QA = 1
        ind = utilList.WhereEqualIndices(retrievalInfo.speciesListFM, 'O3')
        if len(ind) > 0:
            pressure = stateInfo.current['pressure']
            o3 = stateInfo.current['values'][stateInfo.species.index('O3'),:]
            o3ig = stateInfo.initial['values'][stateInfo.species.index('O3'),:]
            indLow = np.where(pressure >= 700)[0]
            indHigh = np.where((pressure >= 200) & (pressure <= 700))[0]
            if len(indLow) > 0 and len(indHigh) > 0:
                maxlo = np.amax(o3[indLow])
                minhi = np.amin(o3[indHigh])
                meanlo = np.mean(o3[indLow])
                meanloig = np.mean(o3ig[indLow])
                surf = o3[np.amin(indLow)]
    
                # pull out mapToState and mapToPars
                # NOTE: mpy.get_one_map will return maps that have columns and rows switched compared to the IDL implementation
                my_map = mpy.get_one_map(retrievalInfo, 'O3')
    
                # This may not be correct.
                # AK = results.A[ind, ind, :]
                # Let's use the manual way.
                AK = utilGeneral.ManualArrayGetWithRHSIndices(retrieval_result.A, ind, ind)
    
                AKzz = np.matmul(np.matmul(my_map['toState'], AK), my_map['toPars'])
                meanAKlo = np.var(AKzz[indLow, indLow])
    
                if maxlo * 1e9 > 150 or \
                    ((maxlo*1e9 > 100 or meanlo / meanloig > 1.8) and (meanAKlo < 0.1)) or \
                    ((maxlo / minhi > 2.5 or maxlo / minhi >= 2) and (maxlo / surf >= 1.1)):
                    retrieval_result.ozoneCcurve = 0
            # end if len(indLow) > 0 and len(indHigh) > 0:
    
            # slope c-curve flag
            o3 = stateInfo.current['values'][stateInfo.species.index('O3'), :]
            indp = np.where(o3 > 0)[0]
            altitude = stateInfo.current['heightKm'][indp]
            o3 = o3[indp]
    
            slope = mpy.ccurve_jessica(altitude, o3)
            retrieval_result.ozone_slope_QA = slope
        # end if len(ind) > 0:
    
        # AT_LINE 255 Write_Retrieval_Summary.pro
        # Now get species dependent preferences
        for ispecie in range(0, num_species):
            my_sum = 0.0
    
            species_name = retrievalInfo.species[ispecie]
    
            loc = -1
            if species_name in stateInfo.species:
                loc = np.where(np.array(stateInfo.species) == species_name)[0][0]
    
            # AT_LINE 269 Write_Retrieval_Summary.pro
            # deviation quality flag - refined for O3 and HCN
            retrieval_result.deviation_QA[ispecie] = 1
            retrieval_result.num_deviations_QA[ispecie] = 1
            retrieval_result.DeviationBad_QA[ispecie] = 1
            pressure = stateInfo.current['pressure']
            
            if loc != -1:
                # Only use the index() function if the species_name is in the stateInfo.species array.
                profile = stateInfo.current['values'][loc, :]
                constraint = stateInfo.constraint['values'][loc, :]
            
            ind = utilList.WhereEqualIndices(retrievalInfo.speciesListFM, species_name)
    
            ak_diag = retrieval_result.A[ind, ind]
    
            # Also note that the value of profile is only set above if loc is not -1 above.
            if loc != -1:
                # AT_LINE 279 Write_Retrieval_Summary.pro
                result_quality = mpy.quality_deviation(pressure, profile, constraint, ak_diag, species_name)
                retrieval_result.deviation_QA[ispecie] = result_quality.deviation_QA
                retrieval_result.num_deviations_QA[ispecie] = result_quality.num_deviations
                retrieval_result.DeviationBad_QA[ispecie] = result_quality.deviationBad

                
__all__ = ["CloudResultSummary"]

from __future__ import annotations
from .fake_state_info import FakeStateInfo
from .fake_retrieval_info import FakeRetrievalInfo
from .identifier import StateElementIdentifier
from .muses_altitude_pge import MusesAltitudePge
from .refractor_uip import AttrDictAdapter
import numpy as np
import math
import typing

if typing.TYPE_CHECKING:
    from .error_analysis import ErrorAnalysis
    from .current_state import CurrentState


# Needs a lot of cleanup, we are just shoving stuff into place
class CloudResultSummary:
    def __init__(
        self,
        current_state: CurrentState,
        result_list: np.ndarray,
        error_analysis: ErrorAnalysis,
    ) -> None:
        self.current_state = current_state
        stateInfo = FakeStateInfo(self.current_state)
        retrievalInfo = FakeRetrievalInfo(self.current_state)

        num_species = retrievalInfo.n_species

        self._cloudODVar = 0.0
        self._cloudODAveError = 0.0

        factor = self.cloud_factor

        indw = np.where(np.asarray(retrievalInfo.speciesList) == "CLOUDEXT")[0]
        indFM = np.where(np.asarray(retrievalInfo.speciesListFM) == "CLOUDEXT")[0]

        if len(indw) > 0:
            my_map = self.get_one_map(retrievalInfo, "CLOUDEXT")
            errlog = error_analysis.errorFM[indFM] @ my_map["toPars"]

            cloudod = np.exp(result_list[indw]) * factor
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
            self._cloudODVar = math.sqrt(x)
        else:
            # cloud not retrieved... use 975-1200
            # NOTE: code has not been tested.
            cov = None
            selem = self.current_state.state_element(StateElementIdentifier("CLOUDEXT"))
            if selem is not None and selem.pressure_list_fm is not None:
                plist = selem.pressure_list_fm
                plist_ind = (plist >= 975) & (plist <= 1200)
                try:
                    cov = self.current_state.previous_aposteriori_cov_fm(
                        [StateElementIdentifier("CLOUDEXT")]
                    )
                except KeyError:
                    # If not in previous_aposteriori_cov_fm, then we just skip the next part
                    pass
            if cov is not None and np.count_nonzero(plist_ind) > 0:
                error_current = np.diag(cov[plist_ind, :][:, plist_ind])
                self._cloudODAveError = (
                    math.sqrt(np.sum(error_current))
                    / error_current.size
                    * self.cloudODAve
                )
        # end else part of if (len(ind) > 0):

        indw = np.where(np.asarray(retrievalInfo.speciesList) == "CLOUDEXT")[0]
        indFM = np.where(np.asarray(retrievalInfo.speciesListFM) == "CLOUDEXT")[0]

        if len(indw) > 0:
            my_map = self.get_one_map(retrievalInfo, "CLOUDEXT")

            errlog = error_analysis.errorFM[indFM] @ my_map["toPars"]

            cloudod = np.exp(result_list[indw]) * factor
            err = (np.exp(np.log(cloudod) + errlog) - cloudod) * factor
            myMean = np.sum(cloudod / err / err) / np.sum(1 / err / err)

            x = np.var((cloudod - myMean) / err, ddof=1)
            self._cloudODVar = math.sqrt(x)

        indw = np.where(
            (stateInfo.emisPars["frequency"] >= 975)
            & (stateInfo.emisPars["frequency"] <= 1200)
        )[0]

        self._emissionLayer = 0
        self._emisDev = -999.0
        if len(indw) > 0:
            self._emisDev = np.mean(stateInfo.current["emissivity"][indw]) - np.mean(
                stateInfo.constraint["emissivity"][indw]
            )

        if "O3" in retrievalInfo.speciesListFM:
            ind10 = [
                idx
                for idx, value in enumerate(retrievalInfo.speciesListFM)
                if value == "O3"
            ]

            if len(ind10) > 0:
                indt = np.where(np.array(stateInfo.species) == "TATM")[0][0]
                TATM = stateInfo.current["values"][indt, :]
                TSUR = stateInfo.current["TSUR"]

                indo3 = np.where(np.array(stateInfo.species) == "O3")[0][0]
                o3 = stateInfo.current["values"][indo3, :]
                o3ig = stateInfo.constraint["values"][indo3, :]

                aveTATM = 0.0
                my_sum = 0.0
                for ii in range(0, 3):
                    my_sum = my_sum + o3[ii] - o3ig[ii]
                    aveTATM = aveTATM + TATM[ii]

                aveTATM = aveTATM / 3
                if my_sum / 3 >= 1.5e-9:
                    self._emissionLayer = aveTATM - TSUR

        self._ozoneCcurve = 1
        self._ozone_slope_QA = 1.0
        ind = [
            idx
            for idx, value in enumerate(retrievalInfo.speciesListFM)
            if value == "O3"
        ]
        if len(ind) > 0:
            pressure = stateInfo.current["pressure"]
            o3 = stateInfo.current["values"][stateInfo.species.index("O3"), :]
            o3ig = stateInfo.initial["values"][stateInfo.species.index("O3"), :]
            indLow = np.where(pressure >= 700)[0]
            indHigh = np.where((pressure >= 200) & (pressure <= 700))[0]
            if len(indLow) > 0 and len(indHigh) > 0:
                maxlo = np.amax(o3[indLow])
                minhi = np.amin(o3[indHigh])
                meanlo = np.mean(o3[indLow])
                meanloig = np.mean(o3ig[indLow])
                surf = o3[np.amin(indLow)]

                my_map = self.get_one_map(retrievalInfo, "O3")

                AK = error_analysis.A[ind, :][:, ind]
                AKzz = np.matmul(np.matmul(my_map["toState"], AK), my_map["toPars"])
                meanAKlo = np.var(AKzz[indLow, indLow])

                if (
                    maxlo * 1e9 > 150
                    or (
                        (maxlo * 1e9 > 100 or meanlo / meanloig > 1.8)
                        and (meanAKlo < 0.1)
                    )
                    or (
                        (maxlo / minhi > 2.5 or maxlo / minhi >= 2)
                        and (maxlo / surf >= 1.1)
                    )
                ):
                    self._ozoneCcurve = 0
            # end if len(indLow) > 0 and len(indHigh) > 0:

            # slope c-curve flag
            o3 = stateInfo.current["values"][stateInfo.species.index("O3"), :]
            indp = np.where(o3 > 0)[0]
            altitude = stateInfo.current["heightKm"][indp]
            o3 = o3[indp]

            slope = self.ccurve_jessica(altitude, o3)
            self._ozone_slope_QA = slope
        # end if len(ind) > 0:

        # Now get species dependent preferences
        self._deviation_QA = np.zeros((num_species,))
        self._num_deviations_QA = np.zeros((num_species,), dtype=int)
        self._DeviationBad_QA = np.zeros((num_species,), dtype=int)
        for ispecie in range(0, num_species):
            my_sum = 0.0

            species_name = retrievalInfo.species[ispecie]

            loc = -1
            if species_name in stateInfo.species:
                loc = np.where(np.array(stateInfo.species) == species_name)[0][0]

            # deviation quality flag - refined for O3 and HCN
            self._deviation_QA[ispecie] = 1
            self._num_deviations_QA[ispecie] = 1
            self._DeviationBad_QA[ispecie] = 1
            pressure = stateInfo.current["pressure"]

            if loc != -1:
                profile = stateInfo.current["values"][loc, :]
                constraint = stateInfo.constraint["values"][loc, :]

            ind = [
                idx
                for idx, value in enumerate(retrievalInfo.speciesListFM)
                if value == species_name
            ]

            ak_diag = error_analysis.A[ind, ind]

            if loc != -1:
                result_quality = self.quality_deviation(
                    pressure, profile, constraint, ak_diag, species_name
                )
                self._deviation_QA[ispecie] = result_quality.deviation_QA
                self._num_deviations_QA[ispecie] = result_quality.num_deviations
                self._DeviationBad_QA[ispecie] = result_quality.deviationBad

    @property
    def cloudODAve(self) -> float:
        freq = self.current_state.state_spectral_domain_wavelength(
            StateElementIdentifier("CLOUDEXT")
        )
        if freq is None:
            raise RuntimeError("This shouldn't happen")
        ind = np.where((freq >= 974) & (freq <= 1201))[0]
        ceffect = self.state_value_vec("CLOUDEXT")
        if len(ind) > 0:
            res = np.sum(ceffect[ind]) / len(ceffect[ind]) * self.cloud_factor
        else:
            res = 0
        return res

    @property
    def cloudODVar(self) -> float:
        return self._cloudODVar

    @property
    def cloudODAveError(self) -> float:
        return self._cloudODAveError

    def ccurve_jessica(self, altitude_in: np.ndarray, ozone_in: np.ndarray) -> float:
        # check mean absolute slope value in troposphere (surface to 20 km)
        # altitude in km
        # ozone in vmr
        # units ppb/km/km
        # above 11 = 'bad'

        # find non-fill and set o3 to ppb
        ind = np.where((altitude_in > -990) & (ozone_in > -990))[0]
        altitudeAGL = altitude_in[ind] - altitude_in[ind[0]]
        ozone = ozone_in[ind] * 1e9
        if np.amax(altitudeAGL) > 200:
            altitudeAGL = altitudeAGL / 1000.0

        # calculate slope at each level
        nn = len(ozone)
        slope = np.zeros(shape=(nn), dtype=np.float32)
        for jj in range(1, nn - 1):
            slope[jj] = (ozone[jj + 1] - ozone[jj - 1]) / (
                altitudeAGL[jj + 1] - altitudeAGL[jj - 1]
            )

        # find top level below 20 km agl where O3 < 90 ppb
        ind = np.where(altitudeAGL < 20)[0]
        o3TropAltLev = int(np.amax(ind))
        for jj in range(0, np.max(ind) + 1):
            if ozone[jj] < 90:
                o3TropAltLev = jj

        # calc ave abs slope from 0 to o3TropAltLev
        slope_sum = np.sum(np.abs(slope[0 : o3TropAltLev + 1]))
        slope_avg = slope_sum / altitudeAGL[o3TropAltLev]

        return slope_avg

    @property
    def emisDev(self) -> float:
        return self._emisDev

    @property
    def emissionLayer(self) -> float:
        return self._emissionLayer

    @property
    def ozoneCcurve(self) -> float:
        return self._ozoneCcurve

    @property
    def ozone_slope_QA(self) -> float:
        return self._ozone_slope_QA

    @property
    def deviation_QA(self) -> np.ndarray:
        return self._deviation_QA

    @property
    def num_deviations_QA(self) -> np.ndarray:
        return self._num_deviations_QA

    @property
    def DeviationBad_QA(self) -> np.ndarray:
        return self._DeviationBad_QA

    def state_value(self, state_name: str) -> float:
        return self.current_state.state_value(StateElementIdentifier(state_name))[0]

    def state_value_vec(self, state_name: str) -> np.ndarray:
        return self.current_state.state_value(StateElementIdentifier(state_name))

    @property
    def cloud_factor(self) -> float:
        scale_pressure = self.state_value("scalePressure")
        if scale_pressure == 0:
            scale_pressure = 0.1
        alt = MusesAltitudePge(
            self.state_value_vec("pressure"),
            self.state_value_vec("TATM"),
            self.state_value_vec("H2O"),
            self.current_state.sounding_metadata.surface_altitude.convert("m").value,
            self.current_state.sounding_metadata.latitude.value,
            tes_pge=True,
        )
        factor = alt.cloud_factor(self.state_value("PCLOUD"), scale_pressure)
        # TODO Rounding currently done. I'm, not sure this makes a lot of sense,
        # this was to match the old IDL code. I don't know that we actually want
        # to do that, but for now have this in place.
        factor = round(factor, 7)
        return factor

    def quality_deviation(
        self,
        pressure: np.ndarray,
        species: np.ndarray,
        constraintVector: np.ndarray,
        averagingKernelDiagonal: np.ndarray,
        speciesName: str,
    ) -> AttrDictAdapter:
        # look at a new flag.  Compare deviations from the prior versus DOF's.
        # Count # of times it crosses the prior
        # Compare to DOF's for that region

        # I know that some EV for the constraint are jackknify; but don't want
        # strong jackknifing if DOFs do not warrant.

        # O3
        # deviationBad:  If 20% and DOF < 0.5
        # deviations:  # of crosses > 20%
        # maxDeviation, maxDeviationDOF:  Largest deviation from prior with
        # DOF < 1.  Record deviation and DOF for it

        # HCN
        # deviationBad:  If 20% and DOF < 0.1
        # deviations:  # of crosses > 20%
        # maxDeviation, maxDeviationDOF:  Largest deviation from prior with
        # DOF < 1.  Record deviation and DOF for it

        # CH4
        # variations more than 10 ppb

        maxDeviation = 0
        maxDeviationDOF = -999

        indp = np.where(pressure > 0)[0]
        value = (species[indp] - constraintVector[indp]) / (
            species[indp] + constraintVector[indp]
        )
        ak = averagingKernelDiagonal[indp]  # diagonal

        # where result crosses the prior
        ind = np.asarray([0])
        other_ind = np.where(
            ((value[0:-1] > 0) & (value[1:] < 0))
            | ((value[0:-1] < 0) & (value[1:] > 0))
        )[0]

        ind = np.append(ind, other_ind)

        # look for deviations more than 20%
        count = 0
        myvalsl = [0]
        mydofsl = [0]
        deviationBad = 0

        # AT_LINE 50 Write_Retrieval_Summary.pro
        for kk in range(0, len(ind) - 1):
            valueTemp = np.amax(
                np.abs(value[ind[kk] : ind[kk + 1] + 1])
            )  # PYTHON_NOTE: We have to add extrat +1 since the slice does not include it.
            dof = np.sum(ak[ind[kk] : ind[kk + 1] + 1])
            if valueTemp > 0.2:
                count = count + 1

            if speciesName != "HCN" and valueTemp > 0.2 and dof < 0.5:
                deviationBad = 1

            if (speciesName != "HCN") and (valueTemp > 0.2 and dof < 0.1):
                deviationBad = 1
            # end if (speciesName != 'HCN') and (valueTemp > 0.2 and dof < 0.1):

            myvalsl.append(valueTemp)
            mydofsl.append(dof)
        # end for kk in range(0,len(ind)-1):

        # AT_LINE 70 quality_deviation.pro
        myvals = np.asarray(
            myvalsl[1:]
        )  # Remove the 1st element since it was just a filler.
        mydofs = np.asarray(
            mydofsl[1:]
        )  # Remove the 1st element since it was just a filler.

        ind1 = np.where(mydofs < 1)[0]
        if len(ind1) > 0:
            # Get the index of the maximum of myvals, which contains the percentage differences.
            # Note that Python allows us to get the index directory with argmax() function.
            max_ind = np.argmax(myvals[ind1])
            maxDeviation = myvals[ind1[max_ind]]
            maxDeviationDOF = mydofs[ind1[max_ind]]
        # end if len(ind1) > 0:

        # AT_LINE 81 quality_deviation.pro
        num_deviations = count

        # AT_LINE 84 Write_Retrieval_Summary.pro
        if speciesName == "HCN":
            deviationQualityFlag = 1
            if maxDeviation >= 0.4 and maxDeviationDOF < 0.1:
                deviationQualityFlag = 0

            if num_deviations >= 2:
                deviationQualityFlag = 0

            deviation_QA = deviationQualityFlag
        else:
            # here are the cases where I set the quality flag to 'bad'
            # each one was tested to screen out bad preferentially
            deviationQualityFlag = 1
            if deviationBad == 1 and num_deviations >= 2:
                deviationQualityFlag = 0

            if num_deviations >= 3:
                deviationQualityFlag = 0

            if deviationBad >= 1:
                deviationQualityFlag = 0

            if maxDeviation >= 0.4 and maxDeviationDOF <= 0.75:
                deviationQualityFlag = 0

            if maxDeviation >= 0.4 and maxDeviationDOF <= 0.50:
                deviationQualityFlag = 0

            deviation_QA = deviationQualityFlag

        # AT_LINE 103 quality_deviation.pro
        results = {
            "maxDeviation": maxDeviation,
            "maxDeviationDOF": maxDeviationDOF,
            "num_deviations": num_deviations,
            "deviationBad": deviationBad,
            "deviation_QA": deviation_QA,
        }

        return AttrDictAdapter(results)

    def get_one_map(
        self, i_retrieval: FakeRetrievalInfo, specieIn: str
    ) -> dict[str, np.ndarray]:
        indSpecie = i_retrieval.species.index(specieIn)
        nn = i_retrieval.n_parameters[indSpecie]
        mm = i_retrieval.n_parametersFM[indSpecie]

        nnn = i_retrieval.parameterStart[indSpecie]
        mmm = i_retrieval.parameterStartFM[indSpecie]
        my_map = i_retrieval.mapToState[nnn : nnn + nn, mmm : mmm + mm]
        mapStar = i_retrieval.mapToParameters[mmm : mmm + mm, nnn : nnn + nn]
        return {"toState": my_map, "toPars": mapStar}


__all__ = ["CloudResultSummary"]

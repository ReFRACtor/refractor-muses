# Don't both typechecking the file. It is long and complicated, and we will replace most
# of it in a bit. Silence mypy, just so we don't get a lot of noise in the output
# type: ignore

# This is a real error, but turn off because we aren't actually running the code with
# these errors. These will get fixed hopefully when we clean this up
# mypy: disable-error-code="used-before-def"

from __future__ import annotations  # We can remove this when we upgrade to python 3.9
import refractor.muses.muses_py as mpy  # type: ignore
from .state_info_old import (
    StateElementOld,
    StateElementHandleOld,
    RetrievableStateElementOld,
    StateElementHandleSetOld,
    StateInfoOld,
)
import numpy as np
import numbers
import refractor.framework as rf  # type: ignore
import copy
import os
import glob
import math
from loguru import logger
import typing

if typing.TYPE_CHECKING:
    from refractor.muses import (
        StateElementIdentifier,
        RetrievalConfiguration,
        RetrievalInfo,
        CurrentStrategyStep,
        MeasurementId,
    )


class MusesPyStateElementOld(RetrievableStateElementOld):
    """This will need a bit of work, right now we don't exactly know what
    this interface should look like. This doesn't match the other species
    we have created, so we'll need to get this worked out."""

    def __init__(
        self, state_info: StateInfoOld, name: StateElementIdentifier, step: str
    ):
        super().__init__(state_info, name)
        self.step = step

    def clone_for_other_state(self):
        """StateInfoOld has copy_current_initialInitial and copy_current_initial.
        The simplest thing would be to just copy the current dict. However,
        the muses-py StateElement maintain their state outside of the classes in
        various dicts in StateInfoOld (probably left over from IDL). So we have
        this function. For ReFRACtor StateElement, this should just be a copy of
        StateElement, but for muses-py we return None. The copy_current_initialInitial
        and copy_current_initial then handle these two cases."""
        return None

    @property
    def value(self):
        # Temporary, define this so we can use a MusesPyStateElement, but
        # we don't actually have code for a value for this. But until we have
        # the full set of species in place, it is useful for us to just ignore that.
        raise NotImplementedError

    @property
    def apriori_value(self):
        # Temporary, define this so we can use a MusesPyStateElement, but
        # we don't actually have code for a value for this. But until we have
        # the full set of species in place, it is useful for us to just ignore that.
        raise NotImplementedError

    def sa_covariance(self):
        """Return sa covariance matrix, and also pressure. This is what
        ErrorAnalysis needs."""
        smeta = self.state_info.sounding_metadata()
        surfacetype = "OCEAN" if smeta.is_ocean else "LAND"
        # TODO Would be good to get mapType available not depending on
        # us calling update_initial_guess. But for right now, we assume
        # that this is available
        maptype = self.mapType.capitalize()

        # Kludge we had for starting to put in Band 7 stuff
        # i_directory = "../OSP/Strategy_Tables/tropomi_nir/Covariance/"
        i_directory = None
        (matrix, pressureSa) = mpy.get_prior_covariance(
            str(self.name),
            smeta.latitude.value,
            self.state_info.pressure,
            surfacetype,
            self.state_info.nh3type,
            self.state_info.ch3ohtype,
            self.state_info.hcoohtype,
            maptype,
            i_directory,
        )

        return (matrix, pressureSa)

    def sa_cross_covariance(self, selem2: StateElementOld):
        smeta = self.state_info.sounding_metadata()
        surfacetype = "OCEAN" if smeta.is_ocean else "LAND"
        matrix, _ = mpy.get_prior_cross_covariance(
            str(self.name),
            str(selem2.name),
            smeta.latitude.value,
            self.state_info.pressure,
            surfacetype,
        )
        if len(matrix.shape) > 1 and matrix[0, 0] >= -990:
            return matrix
        return None

    def update_state(
        self,
        current: np.ndarray | None = None,
        apriori: np.ndarray | None = None,
        initial: np.ndarray | None = None,
        initial_initial: np.ndarray | None = None,
        true: np.ndarray | None = None,
    ) -> None:
        """We have a few places where we want to update a state element other than
        update_initial_guess. This function updates each of the various values passed in.
        A value of 'None' (the default) means skip updating that part of the state."""
        raise NotImplementedError

    def update_state_element(
        self,
        retrieval_info: RetrievalInfo,
        results_list: np.array,
        retrieval_config: RetrievalConfiguration | MeasurementId,
        step: int,
        do_update_fm: np.array,
    ):
        ij = retrieval_info.species_names.index(str(self.name))

        FM_Flag = True
        INITIAL_Flag = True
        TRUE_Flag = False
        CONSTRAINT_Flag = False

        result = mpy.get_vector(
            results_list,
            retrieval_info.retrieval_info_obj,
            str(self.name),
            FM_Flag,
            INITIAL_Flag,
            TRUE_Flag,
            CONSTRAINT_Flag,
        )

        loc = []
        for ii in range(len(self.state_info.state_element_on_levels)):
            if str(self.name) == self.state_info.state_element_on_levels[ii]:
                loc.append(ii)

        ind1 = retrieval_info.retrieval_info_obj.parameterStartFM[ij]
        ind2 = retrieval_info.retrieval_info_obj.parameterEndFM[ij]

        # set which parameters are updated in state AND in error
        # analysis... check movement from i.g.
        myinitial = copy.deepcopy(retrieval_info.initialGuessListFM)
        mapTypeListFM = (
            np.char.asarray(retrieval_info.retrieval_info_obj.mapTypeListFM)
        ).lower()

        # For every indices where 'log' is in mapTypeListFM, we take the exponent of myinitial.
        # AT_LINE 49 Update_State.pro
        myinitial[mapTypeListFM == "log"] = np.exp(myinitial[mapTypeListFM == "log"])

        # AT_LINE 50 Update_State.pro

        abs_array = np.absolute(result - myinitial[ind1 : ind2 + 1]) / np.absolute(
            result
        )

        compare_value = 1.0e-6
        utilGeneral = mpy.UtilGeneral()
        ind = utilGeneral.WhereGreaterEqualIndices(abs_array, compare_value)

        if ind.size > 0:
            do_update_fm[ind + ind1] = 1
        else:
            # if all at i.g., then must've started at true e.g. for spectral
            # window selection.  Here we want accurate error estimates.
            do_update_fm[:] = 1

        my_map = mpy.get_one_map(retrieval_info.retrieval_info_obj, ij)

        # Get indices influenced by retrieval.
        n = retrieval_info.retrieval_info_obj.n_parametersFM[ij]

        ind = [0 for i in range(n)]
        for ii in range(0, n):
            if abs(np.sum(my_map["toState"][:, ii])) >= 1e-10:
                ind[ii] = 1

        ind = utilGeneral.WhereEqualIndices(ind, 1)

        # Code already interpolates to missing emissivity via the mapping

        # AT_LINE 70 Update_State.pro
        if str(self.name) == "EMIS":
            # only update non-zero emissivities.  Eventually move to
            # emis map
            # ind = where(result NE 0)
            if ind.size > 0:
                self.state_info.state_info_obj.current["emissivity"][ind] = result[ind]
        elif str(self.name) == "CLOUDEXT":
            # Note that the variable ind is the list of frequencies that are retrieved
            # AT_LINE 85 Update_State.pro
            if retrieval_info.retrieval_info_obj.type.lower() != "bt_ig_refine":
                if ind.size > 0:
                    # AT_LINE 87 Update_State.pro
                    self.state_info.state_info_obj.current["cloudEffExt"][0, ind] = (
                        result[ind]
                    )

                    # update all frequencies surrounded by current windows
                    # I think the PGE only updates retrieved frequencies
                    # AT_LINE 91 Update_State.pro

                    # PYTHON_NOTE: Because Python slice does not include the end point, we add 1 to np.amax(ind)
                    interpolated_array = mpy.idl_interpol_1d(
                        np.log(result[ind]),
                        self.state_info.state_info_obj.cloudPars["frequency"][ind],
                        self.state_info.state_info_obj.cloudPars["frequency"][
                            np.amin(ind) : np.amax(ind) + 1
                        ],
                    )
                    self.state_info.state_info_obj.current["cloudEffExt"][
                        0, np.amin(ind) : np.amax(ind) + 1
                    ] = np.exp(interpolated_array)[:]
                else:
                    assert False
            else:
                # IGR step
                # get update preferences
                updateAve = retrieval_config["CLOUDEXT_IGR_Average"].lower()
                maxAve = float(retrieval_config["CLOUDEXT_IGR_Max"])
                resetAve = float(retrieval_config["CLOUDEXT_Reset_Value"])

                # Python note:  During development, we see that it is possible for the value of ind.size to be 0
                #               A side effect of that is we cannot use ind array as indices into other arrays.
                #               So before we can ind, we must check for the size first.
                # average in log-space
                # AT_LINE 105 Update_State.pro
                n = ind.size
                if n < 4:
                    # Sanity check for zero size array.
                    if n > 0:
                        ave = np.exp(np.sum(np.log(result[ind])) / len(result[ind]))
                else:
                    # DONT INCLUDE ENDPOINTS!!!!
                    # AT_LINE 110 Update_State.pro
                    # IDL code has ind0 = ind[1:ind.size-2] so for Python, we subtract 1 instead because Python slices does not include the end point.
                    ind0 = ind[1 : ind.size - 1]
                    ave = np.exp(np.sum(np.log(result[ind0])) / len(ind0))

                # Set everywhere to ave but keep structure in areas retrieved
                # AT_LINE 115 Update_State.pro
                self.state_info.state_info_obj.current["cloudEffExt"][:] = ave

                if updateAve == "no":
                    if n > 0:
                        self.state_info.state_info_obj.current["cloudEffExt"][
                            0, ind
                        ] = result[ind]

                        # update areas surrounded by current windows

                        # PYTHON_NOTE: Because Python slice does not include the end point, we add 1 to np.amax(ind)
                        self.state_info.state_info_obj.current["cloudEffExt"][
                            0, np.amin(ind) : np.amax(ind) + 1
                        ] = np.exp(
                            mpy.idl_interpol_1d(
                                np.log(result[ind]),
                                self.state_info.state_info_obj.cloudPars["frequency"][
                                    ind
                                ],
                                self.state_info.state_info_obj.cloudPars["frequency"][
                                    np.amin(ind) : np.amax(ind) + 1
                                ],
                            )
                        )
                else:
                    self.state_info.state_info_obj.current["cloudEffExt"][:] = ave

                # check each value to see if > maxAve
                # don't let get "too large" in refinement step
                # AT_LINE 131 Update_State.pro
                ind = utilGeneral.WhereGreaterEqualIndices(
                    self.state_info.state_info_obj.current["cloudEffExt"][0, :], maxAve
                )

                # Sanity check for zero size array.
                if ind.size > 0:
                    self.state_info.state_info_obj.current["cloudEffExt"][0, ind] = (
                        resetAve
                    )
            # end part of: if stepType != 'bt_ig_refine':

            if self.state_info.state_info_obj.current["cloudEffExt"][0, 0] == 0.01:
                logger.warning(
                    "self.state_info.state_info_obj.current['cloudEffExt'][0, 0] == 0.01"
                )
        # end elif self.name == 'CLOUDEXT'

        elif str(self.name) == "CALSCALE":
            # Sanity check for zero size array.
            if ind.size > 0:
                self.state_info.state_info_obj.current["calibrationScale"][ind] = (
                    result[ind]
                )
        elif str(self.name) == "CALOFFSET":
            if ind.size > 0:
                self.state_info.state_info_obj.current["calibrationOffset"][ind] = (
                    result[ind]
                )
        # AT_LINE 175 Update_State.pro
        elif "TROPOMI" in str(self.name):
            # PYTHON_NOTE: Because within (self.state_info.state_info_obj.current['tropomi'] we want to replace all fields with actual value from results.
            #              Using the species_name, 'TROPOMICLOUDFRACTION', we look for 'cloud_fraction' in the keys of self.state_info.state_info_obj.current['tropomi'].
            #              So, given TROPOMICLOUDFRACTION, we return the actual_tropomi_key as 'cloud_fraction'.
            species_name = str(self.name)
            tropomiInfo = mpy.ObjectView(
                self.state_info.state_info_obj.current["tropomi"]
            )

            actual_tropomi_key = mpy.get_tropomi_key(tropomiInfo, species_name)
            self.state_info.state_info_obj.current["tropomi"][actual_tropomi_key] = (
                copy.deepcopy(result)
            )  # Use the actual key and replace the exist key.
        elif "NIR" in str(self.name)[0:3]:
            # tag_names_str = tag_names(state.current.nir)
            # ntag = n_elements(tag_names_str)
            # tag_names_str_new = strarr(ntag)
            # for tempi = 0,ntag-1 do tag_names_str_new[tempi] = 'NIR'+ replace(tag_names_str[tempi],'_','')
            # indtag = where(tag_names_str_new EQ retrieval.species[ij])
            my_species = str(self.name)[3:].lower()
            if my_species == "alblamb":
                mult = 1
                if self.state_info.state_info_obj.current["nir"]["albtype"] == 2:
                    mult = 1.0 / 0.07
                if self.state_info.state_info_obj.current["nir"]["albtype"] == 3:
                    raise RuntimeError("Mismatch in albedo type")
                self.state_info.state_info_obj.current["nir"]["albpl"] = result * mult
                my_species = "albpl"
            elif my_species == "albbrdf":
                mult = 1
                if self.state_info.state_info_obj.current["nir"]["albtype"] == 1:
                    mult = 0.07
                if self.state_info.state_info_obj.current["nir"]["albtype"] == 3:
                    raise RuntimeError("Mismatch in albedo type")
                self.state_info.state_info_obj.current["nir"]["albpl"] = result * mult
                my_species = "albpl"
            elif my_species == "albcm":
                mult = 1
                if self.state_info.state_info_obj.current["nir"]["albtype"] != 3:
                    raise RuntimeError("Mismatch in albedo type")
                self.state_info.state_info_obj.current["nir"]["albpl"] = result * mult
                my_species = "albpl"
            elif my_species == "albbrdfpl":
                mult = 1
                if self.state_info.state_info_obj.current["nir"]["albtype"] == 1:
                    mult = 0.07
                if self.state_info.state_info_obj.current["nir"]["albtype"] == 3:
                    raise RuntimeError("Mismatch in albedo type")
                self.state_info.state_info_obj.current["nir"]["albpl"] = result * mult
                my_species = "albpl"
            elif my_species == "alblambpl":
                mult = 1
                if self.state_info.state_info_obj.current["nir"]["albtype"] == 2:
                    mult = 1 / 0.07
                if self.state_info.state_info_obj.current["nir"]["albtype"] == 3:
                    raise RuntimeError("Mismatch in albedo type")
                self.state_info.state_info_obj.current["nir"]["albpl"] = result * mult
                my_species = "albpl"
            elif my_species == "disp":
                # update only part of the state
                npoly = int(len(result) / 3)
                self.state_info.state_info_obj.current["nir"]["disp"][:, 0:npoly] = (
                    np.reshape(result, (3, 2))
                )
            elif my_species == "eof":
                # reshape
                self.state_info.state_info_obj.current["nir"]["eof"][:, :] = np.reshape(
                    result, (3, 3)
                )  # checked ordering is good 12/2021
            elif my_species == "cloud3d":
                # reshape
                self.state_info.state_info_obj.current["nir"]["cloud3d"][:, :] = (
                    np.reshape(result, (3, 2))
                )  # checked ordering is good 12/2021
            else:
                self.state_info.state_info_obj.current["nir"][my_species] = result
        # AT_LINE 175 Update_State.pro
        elif str(self.name) == "PCLOUD":
            # Note: Variable result is ndarray (sequence) of size 730
            #       The variable  self.state_info.state_info_obj.current['PCLOUD'][0] is an element.  We cannot assign an array element with a sequence
            # AT_LINE 284 Update_State.pro
            if isinstance(result, np.ndarray):
                self.state_info.state_info_obj.current["PCLOUD"][0] = result[
                    0
                ]  # IDL_NOTE: With IDL, we can be sloppy, but in Python, we must use index [0] so we can just get one element.
            else:
                self.state_info.state_info_obj.current["PCLOUD"][0] = result

            # do bounds checking for refinement step
            if retrieval_info.retrieval_info_obj.type.lower() == "bt_ig_refine":
                resetValue = -1
                if "PCLOUD_IGR_Reset_Value" in retrieval_config.keys():
                    resetValue = float(retrieval_config["PCLOUD_IGR_Reset_Value"])
                if resetValue == -1:
                    resetValue = int(retrieval_config["PCLOUD_Reset_Value"])

                if (
                    self.state_info.state_info_obj.current["PCLOUD"][0]
                    > self.state_info.state_info_obj.current["pressure"][0]
                ):
                    self.state_info.state_info_obj.current["PCLOUD"][0] = (
                        self.state_info.state_info_obj.current["pressure"][1]
                    )

                if self.state_info.state_info_obj.current["PCLOUD"][0] < resetValue:
                    self.state_info.state_info_obj.current["PCLOUD"][0] = resetValue
        elif str(self.name) == "TSUR":
            self.state_info.state_info_obj.current["TSUR"] = result
        elif str(self.name) == "PSUR":
            self.state_info.state_info_obj.current["pressure"][0] = result
        elif str(self.name) == "PTGANG":
            self.state_info.state_info_obj.current["tes"]["boresightNadirRadians"] = (
                result
            )
        elif str(self.name) == "RESSCALE":
            self.state_info.state_info_obj.current.residualscale[step:] = result
        else:
            # AT_LINE 289 Update_State.pro
            max_index = (self.state_info.state_info_obj.current["values"].shape)[
                1
            ]  # Get access to the 63 in (1,63)
            self.state_info.state_info_obj.current["values"][loc, :] = result[
                0:max_index
            ]
        # end part of if (self.name == 'EMIS'):

        locHDO = utilGeneral.WhereEqualIndices(
            self.state_info.state_info_obj.species, "HDO"
        )
        locH2O = utilGeneral.WhereEqualIndices(
            self.state_info.state_info_obj.species, "H2O"
        )
        locRetHDO = utilGeneral.WhereEqualIndices(
            retrieval_info.retrieval_info_obj.species, "HDO"
        )
        if (str(self.name) == "H2O") and (locHDO.size > 0) and (locRetHDO.size == 0):
            # get initial guess ratio...
            initialRatio = (
                self.state_info.state_info_obj.initial["values"][
                    locHDO[0], 0 : len(result)
                ]
                / self.state_info.state_info_obj.initial["values"][
                    locH2O[0], 0 : len(result)
                ]
            )

            # set HDO by initial ratio multiplied by retrieved H2O
            self.state_info.state_info_obj.current["values"][
                locHDO[0], 0 : len(result)
            ] = result * initialRatio

        locH2O18 = utilGeneral.WhereEqualIndices(
            self.state_info.state_info_obj.species, "H2O18"
        )
        locH2O = utilGeneral.WhereEqualIndices(
            self.state_info.state_info_obj.species, "H2O"
        )
        locRetH2O18 = utilGeneral.WhereEqualIndices(
            retrieval_info.retrieval_info_obj.species, "H2O18"
        )
        if (
            (str(self.name) == "H2O")
            and (locH2O18.size > 0)
            and (locRetH2O18.size == 0)
        ):
            # get initial guess ratio...
            initialRatio = (
                self.state_info.state_info_obj.initial["values"][
                    locH2O18[0], 0 : len(result)
                ]
                / self.state_info.state_info_obj.initial["values"][
                    locH2O[0], 0 : len(result)
                ]
            )
            # set HDO by initial ratio multiplied by retrieved H2O
            self.state_info.state_info_obj.current["values"][
                locH2O18[0], 0 : len(result)
            ] = result * initialRatio

        locH2O17 = utilGeneral.WhereEqualIndices(
            self.state_info.state_info_obj.species, "H2O17"
        )
        locH2O = utilGeneral.WhereEqualIndices(
            self.state_info.state_info_obj.species, "H2O"
        )
        locRetH2O17 = utilGeneral.WhereEqualIndices(
            retrieval_info.retrieval_info_obj.species, "H2O17"
        )
        if (
            (str(self.name) == "H2O")
            and (locH2O17.size > 0)
            and (locRetH2O17.size == 0)
        ):
            # get initial guess ratio...
            initialRatio = (
                self.state_info.state_info_obj.initial["values"][
                    locH2O17[0], 0 : len(result)
                ]
                / self.state_info.state_info_obj.initial["values"][
                    locH2O[0], 0 : len(result)
                ]
            )
            # set HDO by initial ratio multiplied by retrieved H2O
            self.state_info.state_info_obj.current["values"][
                locH2O17[0], 0 : len(result)
            ] = result * initialRatio

    def species_information_file(self, retrieval_type: str):
        """Determine the species information file, and read the data"""
        # Open, read species file
        retrieval_type_v = "_" + retrieval_type.lower()
        if retrieval_type_v == "_default":
            retrieval_type_v = ""
        species_directory = self.retrieval_config["speciesDirectory"]
        speciesInformationFilename = (
            f"{species_directory}/{str(self.name)}{retrieval_type_v}.asc"
        )

        files = glob.glob(speciesInformationFilename)
        if len(files) == 0:
            # Look for alternate file.
            speciesInformationFilename = f"{species_directory}/{str(self.name)}.asc"
        # Can turn this one if we want to see what gets read. A bit noisy to
        # have on in general though.
        # logger.debug(f"Reading file {speciesInformationFilename}")
        # AT_LINE 156 Get_Species_Information.pro
        (_, fileID) = mpy.read_all_tes_cache(speciesInformationFilename)
        return mpy.ObjectView(fileID["preferences"])

    def update_initial_guess(self, current_strategy_step: CurrentStrategyStep):
        species_list = [str(i) for i in current_strategy_step.retrieval_elements]
        species_name = str(self.name)
        pressure = self.state_info.pressure
        # user specifies the number of forward model levels
        nfm_levels = int(self.retrieval_config["num_FMLevels"])
        if nfm_levels < len(pressure):
            pressure = pressure[:nfm_levels]

        stateInfo = mpy.ObjectView(self.state_info.state_info_dict)
        current = mpy.ObjectView(stateInfo.current)

        speciesInformationFile = self.species_information_file(
            current_strategy_step.retrieval_type
        )

        # AT_LINE 157 Get_Species_Information.pro
        mapType = speciesInformationFile.mapType.lower()
        constraintType = speciesInformationFile.constraintType.lower()
        # if(str(self.name) == "H2O"):
        #    breakpoint()
        # spectral species
        # AT_LINE 161 Get_Species_Information.pro
        # self.m_debug_mode = True
        if "TROPOMI" in species_name:
            # EM NOTE - copied from the OMI section, since tropomi build is based on the omi build.
            # OMI code above suggests 'not tested', but think this has just not been cleaned up yet.
            # This section is necessary for TROPOMI since we are using similar fitting parameters,
            # and structures to OMI.

            tropomiInfo = mpy.ObjectView(current.tropomi)

            actual_tropomi_key = mpy.get_tropomi_key(tropomiInfo, species_name)

            # AT_LINE 177 Get_Species_Information.pro
            # PYTHON_NOTE: To get access to a dictionary, we use a key instead of an index as IDL does.
            # At this point, the value of actual_omi_key is the key we want to access the 'omi' dictionary.
            initialGuessList = stateInfo.current["tropomi"][actual_tropomi_key]
            if self.state_info.has_true_values():
                trueParameterList = stateInfo.true["tropomi"][actual_tropomi_key]
            constraintVector = stateInfo.constraint["tropomi"][actual_tropomi_key]
            constraintVectorFM = stateInfo.constraint["tropomi"][actual_tropomi_key]

            # It is also possible that these values are scalar, we convert them to an array of 1.
            if np.isscalar(initialGuessList):
                initialGuessList = np.asarray([initialGuessList])

            if self.state_info.has_true_values():
                if np.isscalar(trueParameterList):
                    trueParameterList = np.asarray([trueParameterList])

            if np.isscalar(constraintVector):
                constraintVector = np.asarray([constraintVector])

            initialGuessListFM = initialGuessList[:]
            if self.state_info.has_true_values():
                trueParameterListFM = trueParameterList[:]

            # AT_LINE 184 Get_Species_Information.pro
            nn = len(initialGuessList)
            mm = len(initialGuessList)

            if mm == 1:
                mapToState = 1
                mapToParameters = 1
                num_retrievalParameters = 1

                # it's difficult to get a nx1 array.  1xn is easy.
                retrievalParameters = [0]
                pressureList = [-2]
                pressureListFM = [-2]
                altitudeList = [-2]
                altitudeListFM = [-2]

                sSubaDiagonalValues = float(
                    speciesInformationFile.sSubaDiagonalValues
                )  # Becareful that some values are of string type.
                constraintMatrix = 1 / (
                    sSubaDiagonalValues * sSubaDiagonalValues
                )  # Note the name change from .constraint to constraintMatrix
            else:
                # AT_LINE 202 Get_Species_Information.pro
                mapToState = np.identity(mm)
                mapToParameters = np.identity(mm)
                num_retrievalParameters = mm

                retrievalParameters = [ii for ii in range(mm)]  # INDGEN(mm)
                pressureList = [-2 for ii in range(mm)]  # STRARR(mm)+'-2'
                pressureListFM = [-2 for ii in range(mm)]  # STRARR(mm)+'-2'
                altitudeList = [-2 for ii in range(mm)]  # STRARR(mm)+'-2'
                altitudeListFM = [-2 for ii in range(mm)]  # STRARR(mm)+'-2'

                sSubaDiagonalValues = float(speciesInformationFile.sSubaDiagonalValues)
                constraintMatrix = np.identity(
                    mm
                )  # Note the name change from .constraint to constraintMatrix
                indx = [ii for ii in range(mm)]
                indx = np.asarray(indx)
                constraintMatrix[indx, indx] = 1 / (
                    sSubaDiagonalValues * sSubaDiagonalValues
                )  # Note the name change from .constraint to constraintMatrix

            # take log if it makes sense
            if (speciesInformationFile.mapType) == "LOG":
                initialGuessListFM = np.log(initialGuessListFM)
                constraintVector = np.log(constraintVector)
                initialGuessList = np.log(initialGuessList)
                if self.state_info.has_true_values():
                    trueParameterListFM = np.log(trueParameterListFM)
                    trueParameterList = np.log(trueParameterList)
            # end if (speciesInformationFile.mapType) == 'LOG':
        # IDL AT_LINE 240 Get_Species_Information:

        elif species_name == "NIRALBBRDFPL" or species_name == "NIRALBLAMBPL":
            if species_name == "NIRALBBRDFPL":
                mult = 1
                if stateInfo.current["nir"]["albtype"] == 1:
                    mult = 1 / 0.07
            else:
                mult = 1
                if stateInfo.current["nir"]["albtype"] == 2:
                    mult = 0.07

            if stateInfo.current["nir"]["albtype"] == 3:
                raise RuntimeError("Mismatch in albedo type")

            # get all full-state grid quantities
            initialGuessListFM = stateInfo.current["nir"]["albpl"] * mult
            nn = len(initialGuessListFM)
            initialGuessListFM = initialGuessListFM.reshape(nn)
            if self.state_info.has_true_values():
                trueParameterListFM = stateInfo.true["nir"]["albpl"].reshape(nn) * mult
            constraintVectorFM = stateInfo.constraint["nir"]["albpl"].reshape(nn) * mult
            pressureListFM = stateInfo.current["nir"]["albplwave"]
            altitudeListFM = stateInfo.current["nir"]["albplwave"]
            mm = int(speciesInformationFile.num_retrieval_levels)

            # albedo
            if (mapType == "linearpca") or (mapType == "logpca"):
                # this is state vector of the form:
                # current_fs = apriori_fs + mapToState @ current for linearpca
                # or
                # log(current_fs) = log(apriori_fs) + mapToState @ log(current) for logpca
                # Doing current_fs = mapToState @ current does not work
                # because the maps do not have a good span of the state, e.g. the stratosphere does not have sensitivity.
                # when I tried this I got Tatm = [300, ..., 62, 37, -20, -27]
                # so it must be apriori + offset

                mapsFilename = speciesInformationFile.mapsFilename

                # retrieval "levels"
                retrievalParameters = np.array(range(mm)) + 1

                # implemented for OCO-2, but if used for other satellite
                # need to change # of full state levels to read correct file
                # for oco-2:  maps_TATM_Linear_20_3.nc, where 20 is # of full state pressures
                (mapDict, _, _) = mpy.cdf_read_dict(mapsFilename)
                mapToState = np.transpose(mapDict["to_state"])
                mapToParameters = np.transpose(mapDict["to_pars"])

                pressureList = np.zeros(mm, dtype=float) - 999
                altitudeList = np.zeros(mm, dtype=float) - 999

                filename = speciesInformationFile.constraintFilename
                (constraintStruct, constraintPressure) = mpy.constraint_read(filename)
                constraintMatrix = mpy.constraint_get(constraintStruct)
                constraintPressure = mpy.constraint_get_pressures(constraintStruct)

                # since the "true" is relative to the a priori
                # the "true state" is set to e.g. 0.8 if the a priori
                # is off by -0.8K
                if mapType == "linearpca":
                    constraintVector = np.zeros(mm, dtype=np.float32) + 0
                    initialGuessList = np.transpose(mapToParameters) @ (
                        trueParameterListFM - constraintVectorFM
                    )
                    initialGuessListFM = (
                        constraintVectorFM + np.transpose(mapToState) @ initialGuessList
                    )
                    if self.state_info.has_true_values():
                        trueParameterList = np.transpose(mapToParameters) @ (
                            trueParameterListFM - constraintVectorFM
                        )
                else:
                    constraintVector = np.zeros(mm, dtype=np.float32) + 0
                    initialGuessList = np.transpose(mapToParameters) @ (
                        np.log(trueParameterListFM) - np.log(constraintVectorFM)
                    )
                    initialGuessListFM = np.exp(
                        np.log(constraintVectorFM) + mapToState @ initialGuessList
                    )
                    if self.state_info.has_true_values():
                        trueParameterList = np.transpose(mapToParameters) @ (
                            np.log(trueParameterListFM) - np.log(constraintVectorFM)
                        )
            # end part of elif (mapType == 'linearpca') or (mapType == 'logpca'):

            else:
                if speciesInformationFile.constraintType == "Diagonal":
                    values = (speciesInformationFile.sSubaDiagonalValues).split(",")
                    for kk in range(0, len(values)):
                        values[kk] = float(values[kk])
                    constraintMatrix = np.identity(mm)
                    for kk in range(0, len(values)):
                        constraintMatrix[kk, kk] = 1 / values[kk]
                elif speciesInformationFile.constraintType == "Full":
                    (covariance, _) = mpy.constraint_read(
                        speciesInformationFile.constraintFilename
                    )
                    constraintMatrix = np.invert(covariance["data"])
                    pressureList = covariance["pressure"]
                    altitudeList = pressureList
                elif speciesInformationFile.constraintType == "PREMADE":
                    filename = speciesInformationFile.constraintFilename

                    constraintMatrix, pressurex = (
                        mpy.supplier_constraint_matrix_premade(
                            species_name,
                            filename,
                            mm,
                            i_nh3type=self.state_info.nh3type,
                            i_ch3ohtype=self.state_info.ch3ohtype,
                        )
                    )

                    pressureList = pressurex
                    altitudeList = pressurex
                else:
                    raise RuntimeError(
                        f"Unknown type for {speciesInformationFile.filename} constraintType is {speciesInformationFile.constraintType}"
                    )

                # get maps from constraintMatrix
                if mm == nn:
                    mapToState = np.identity(mm)
                    mapToParameters = np.identity(mm)

                    retrievalParameters = range(mm)
                    pressureList = pressureListFM
                    altitudeList = altitudeListFM
                    constraintVector = constraintVectorFM
                    initialGuessList = initialGuessListFM
                    if self.state_info.has_true_values():
                        trueParameterList = trueParameterListFM
                else:
                    # ensure each band edge matches, match to pressureListFM, then create maps
                    ind1 = np.where(pressureList < 1.0)[0]
                    ind2 = np.where(pressureListFM < 1.0)[0]
                    if len(ind2) > 0:
                        pressureList[min(ind1)] = np.min(pressureListFM[ind2])
                        pressureList[max(ind1)] = np.max(pressureListFM[ind2])

                    ind1 = np.where((pressureList > 1.0) * (pressureList < 2.0))[0]
                    ind2 = np.where((pressureListFM > 1.0) * (pressureListFM < 2.0))[0]
                    if len(ind2) > 0:
                        pressureList[min(ind1)] = np.min(pressureListFM[ind2])
                        pressureList[max(ind1)] = np.max(pressureListFM[ind2])

                    ind1 = np.where((pressureList > 2.0) * (pressureList < 2.2))[0]
                    ind2 = np.where((pressureListFM > 2.0) * (pressureListFM < 2.2))[0]
                    if len(ind2) > 0:
                        pressureList[min(ind1)] = np.min(pressureListFM[ind2])
                        pressureList[max(ind1)] = np.max(pressureListFM[ind2])

                # change pressurelist to best matching pressurelistFM
                # this is needed for mapping
                inds = []
                for iq in range(len(pressureList)):
                    xx = np.min(abs(pressureList[iq] - pressureListFM))
                    ind = np.where(abs(pressureList[iq] - pressureListFM) == xx)[0][0]
                    pressureList[iq] = pressureListFM[ind]
                    inds.append(ind)

                inds = np.array(inds) + 1
                maps = mpy.make_maps(pressureListFM, inds, i_linearFlag=True)

                constraintVector = maps["toPars"].transpose() @ constraintVectorFM
                initialGuessList = maps["toPars"].transpose() @ initialGuessListFM
                if self.state_info.has_true_values():
                    trueParameterList = maps["toPars"].transpose() @ trueParameterListFM

                mapToParameters = maps["toPars"]
                mapToState = maps["toState"]

        elif "NIRAERX" in species_name:
            # aerosol.  Match type
            types = (speciesInformationFile.types).split(",")
            myinds = []
            for ii in range(mm):
                ind = np.where(
                    np.array(stateInfo.current["nir"]["aertype"][ii]) == types
                )[0]
                myinds.append(ind[0])

            # mykeys = ['sSubaDiagonalValues','minimum','maximum','maximumChange']
            value = (speciesInformationFile.sSubaDiagonalValues).split(",")
            if len(value) > 1:
                speciesInformationFile.sSubaDiagonalValues = np.array(value)[myinds]
            value = (speciesInformationFile.minimum).split(",")
            if len(value) > 1:
                speciesInformationFile.minimum = np.array(value)[myinds]
            value = (speciesInformationFile.maximum).split(",")
            if len(value) > 1:
                speciesInformationFile.maximum = np.array(value)[myinds]
            value = (speciesInformationFile.maximumChange).split(",")
            if len(value) > 1:
                speciesInformationFile.maximumChange = np.array(value)[myinds]

            # make key
            mykey = species_name[3:]
            mykey = mykey.lower()

            npar = len(stateInfo.current["nir"][mykey][:])
            initialGuessList = stateInfo.current["nir"][mykey]
            constraintVector = stateInfo.constraint["nir"][mykey]
            constraintVectorFM = stateInfo.constraint["nir"][mykey]
            initialGuessListFM = copy.deepcopy(initialGuessList)
            if self.state_info.has_true_values():
                trueParameterListFM = copy.deepcopy(trueParameterList)
                trueParameterList = stateInfo.true["nir"][mykey]

            nn = len(initialGuessListFM)
            mm = len(initialGuessList)
            num_retrievalParameters = mm

            # aerosol.  Match type
            types = (speciesInformationFile.types).split(",")
            myinds = []
            for ii in range(mm):
                ind = np.where(
                    np.array(stateInfo.current["nir"]["aertype"][ii]) == types
                )[0]
                myinds.append(ind[0])

            val = np.array((speciesInformationFile.sSubaDiagonalValues).split(","))
            constraintMatrix = np.identity(mm)
            for jj in range(mm):
                constraintMatrix[jj, jj] = (
                    1 / float(val[myinds[jj]]) / float(val[myinds[jj]])
                )

            mapToState = np.identity(mm)
            mapToParameters = np.identity(mm)

        elif "NIR" in species_name:
            # other NIR parameters... all diagonal

            # match up fields in stateOne.nir to parameter names
            # mylist1 = list(stateInfo.current['nir'].keys())
            # mylist2 = mylist1.copy()
            # for jj in range(len(mylist2)):
            #    mylist2[jj] = mylist2[jj].replace('_','')
            #    mylist2[jj] = 'NIR' + mylist2[jj].upper()

            # make key
            mykey = species_name[3:]
            mykey = mykey.lower()

            if "NIRAER" in species_name:
                # even if these are #'s, expressed as strings.
                types = speciesInformationFile.types.split(",")
                sSubaDiagonalValues = float(speciesInformationFile.sSubaDiagonalValues)
                minimum = speciesInformationFile.minimum.split(",")
                maximum = speciesInformationFile.maximum.split(",")
                maximumChange = speciesInformationFile.maximumChange.split(",")

                # aerosol.  Select down to exact types used.
                # each type can have a different constraint (or max/min/maxchange)
                # types = (speciesInformationFile.types).split(',')
                myinds = []
                for ik in range(stateInfo.current["nir"]["naer"]):
                    if (str(stateInfo.current["nir"]["aertype"][ik].dtype))[1] == "S":
                        ind = np.where(
                            stateInfo.current["nir"]["aertype"][ik].decode()
                            == np.array(types)
                        )[0]
                    else:
                        ind = np.where(
                            stateInfo.current["nir"]["aertype"][ik] == np.array(types)
                        )[0]

                    if len(ind) == 0:
                        pass

                    myinds.append(ind[0])

                # select values corresponding to aerosol types used and
                # place into speciesInformationFile
                myinds = np.array(myinds)
                speciesInformationFile.sSubaDiagonalValues = ",".join(
                    np.array(sSubaDiagonalValues)[myinds]
                )
                speciesInformationFile.minimum = ",".join(np.array(minimum)[myinds])
                speciesInformationFile.maximum = ",".join(np.array(maximum)[myinds])
                speciesInformationFile.maximumChange = ",".join(
                    np.array(maximumChange)[myinds]
                )
                speciesInformationFile.types = ",".join(np.array(types)[myinds])

            # for NIRALBLAMB, the polynomial order is set by the # listed in speciesfile, sSubaDiagonalValues
            if (
                species_name == "NIRALBLAMB"
                or species_name == "NIRALBBRDF"
                or species_name == "NIRALBCM"
            ):
                # check retrieval versus state type
                mult = 1
                if species_name == "NIRALBLAMB":
                    if stateInfo.current["nir"]["albtype"] == 2:
                        mult = 0.07
                    if stateInfo.current["nir"]["albtype"] == 3:
                        raise RuntimeError("Mismatch in albedo type")

                if species_name == "NIRALBBRDF":
                    if stateInfo.current["nir"]["albtype"] == 1:
                        mult = 1 / 0.07
                    if stateInfo.current["nir"]["albtype"] == 3:
                        raise RuntimeError("Mismatch in albedo type")

                npoly = int(
                    len((speciesInformationFile.sSubaDiagonalValues).split(",")) / 3
                )

                # get initial maps.  Maps will be updated when ReFRACtor is run
                nfs = len(stateInfo.current["nir"]["albpl"])
                filename = (
                    "../OSP/OCO2/map" + str(nfs) + "x" + str(int(npoly * 3)) + ".nc"
                )
                (_, mapToParameters, _) = mpy.nc_read_variable(filename, "topars")
                (_, mapToState, _) = mpy.nc_read_variable(filename, "tostate")

                initialGuessListFM = stateInfo.current["nir"]["albpl"] * mult
                initialGuessList = mapToParameters @ initialGuessListFM
                if self.state_info.has_true_values():
                    trueParameterListFM = stateInfo.true["nir"]["albpl"] * mult
                    trueParameterList = mapToParameters @ trueParameterListFM
                constraintVector = mapToParameters @ (
                    stateInfo.constraint["nir"]["albpl"] * mult
                )
                constraintVectorFM = (
                    mapToState
                    @ mapToParameters
                    @ (stateInfo.constraint["nir"]["albpl"] * mult)
                )

                mapToParameters = mapToParameters.T
                mapToState = mapToState.T

            elif species_name == "NIRDISP":
                npoly = int(
                    len((speciesInformationFile.sSubaDiagonalValues).split(",")) / 3
                )
                # get only the first npoly entries
                initialGuessList = stateInfo.current["nir"][mykey][:, 0:npoly].reshape(
                    npoly * 3
                )
                constraintVector = (
                    stateInfo.constraint["nir"][mykey][:, 0:npoly]
                ).reshape(npoly * 3)
                constraintVectorFM = (
                    stateInfo.constraint["nir"][mykey][:, 0:npoly]
                ).reshape(npoly * 3)
                initialGuessListFM = initialGuessList.copy()
                if self.state_info.has_true_values():
                    trueParameterList = (
                        stateInfo.true["nir"][mykey][:, 0:npoly]
                    ).reshape(npoly * 3)
                    trueParameterListFM = trueParameterList.copy()
            elif species_name == "NIREOF":
                npar = len(stateInfo.current["nir"][mykey][:, 0])
                nband = len(stateInfo.current["nir"][mykey][0, :])
                initialGuessList = stateInfo.current["nir"][mykey].reshape(npar * nband)
                constraintVector = stateInfo.constraint["nir"][mykey].reshape(
                    npar * nband
                )
                constraintVectorFM = stateInfo.constraint["nir"][mykey].reshape(
                    npar * nband
                )
                initialGuessListFM = copy.deepcopy(initialGuessList)
                if self.state_info.has_true_values():
                    trueParameterList = stateInfo.true["nir"][mykey].reshape(
                        npar * nband
                    )
                    trueParameterListFM = copy.deepcopy(trueParameterList)
            # elif species_name == 'NIRCLOUD3D':
            #     npar = len(stateInfo.current['nir'][mylist1[indnir]][:,0])
            #     nband = len(stateInfo.current['nir'][mylist1[indnir]][0,:])
            #     initialGuessList = stateInfo.current['nir'][mylist1[indnir]].reshape(npar*nband)
            #     trueParameterList = stateInfo.true['nir'][mylist1[indnir]].reshape(npar*nband)
            #     constraintVector = stateInfo.constraint['nir'][mylist1[indnir]].reshape(npar*nband)
            #     initialGuessListFM = deepcopy(initialGuessList)
            #     trueParameterListFM = deepcopy(trueParameterList)
            elif species_name == "NIRWIND":
                npar = 1
                initialGuessList = [stateInfo.current["nir"]["wind"]]
                constraintVector = [stateInfo.constraint["nir"]["wind"]]
                constraintVectorFM = [stateInfo.constraint["nir"]["wind"]]
                initialGuessListFM = [copy.deepcopy(initialGuessList)]
                if self.state_info.has_true_values():
                    trueParameterListFM = [stateInfo.true["nir"]["wind"]]
                    trueParameterList = [stateInfo.true["nir"]["wind"]]
            else:
                initialGuessList = stateInfo.current["nir"][mykey]
                constraintVector = stateInfo.constraint["nir"][mykey]
                constraintVectorFM = stateInfo.constraint["nir"][mykey]
                initialGuessListFM = copy.deepcopy(initialGuessList)
                if self.state_info.has_true_values():
                    trueParameterList = stateInfo.true["nir"][mykey]
                    trueParameterListFM = copy.deepcopy(trueParameterList)

            nn = len(initialGuessListFM)
            mm = len(initialGuessList)
            num_retrievalParameters = mm

            if mm == 1:
                mapToState = 1
                mapToParameters = 1

                # it's difficult to get a nx1 array.  1xn is easy.
                retrievalParameters = [0]
                pressureList = [-2]
                pressureListFM = [-2]
                altitudeList = [-2]
                altitudeListFM = [-2]

                val = (speciesInformationFile.sSubaDiagonalValues).split(",")
                for jj in range(len(val)):
                    val[jj] = np.double(val[jj])
                sSubaDiagonalValues = val
                constraintMatrix = [1 / val[0] / val[0]]

            elif mm == nn:
                mapToState = np.identity(mm)
                mapToParameters = np.identity(mm)

                retrievalParameters = range(mm)
                pressureList = np.zeros(mm) - 2
                pressureListFM = np.zeros(mm) - 2
                altitudeList = np.zeros(mm) - 2
                altitudeListFM = np.zeros(mm) - 2

                if speciesInformationFile.constraintType == "Diagonal":
                    val = (speciesInformationFile.sSubaDiagonalValues).split(",")
                    for jj in range(len(val)):
                        val[jj] = np.double(val[jj])
                    sSubaDiagonalValues = val
                    constraintMatrix = np.identity(mm)
                    for jj in range(len(val)):
                        constraintMatrix[jj, jj] = (
                            1 / sSubaDiagonalValues[jj] / sSubaDiagonalValues[jj]
                        )
                elif speciesInformationFile.constraintType == "Full":
                    (covariance, _) = mpy.constraint_read(
                        speciesInformationFile.sSubaFilename
                    )
                    constraintMatrix = np.linalg.inv(covariance["data"])

                    pressureList = covariance["pressure"]
                    altitudeList = covariance["pressure"]
                    pressureListFM = covariance["pressure"]
                    altitudeListFM = covariance["pressure"]
                elif speciesInformationFile.constraintType == "PREMADE":
                    filename = speciesInformationFile.constraintFilename

                    constraintMatrix, pressurex = (
                        mpy.supplier_constraint_matrix_premade(
                            species_name,
                            filename,
                            mm,
                            i_nh3type=self.state_info.nh3type,
                            i_ch3ohtype=self.state_info.ch3ohtype,
                        )
                    )

                    pressureList = pressurex.copy()
                    pressureListFM = pressurex.copy()
                    # pressurelistFM = stateInfo.current.nir.albplwave
                    altitudeList = pressurex.copy()
                    altitudeListFM = pressurex.copy()
                    # altitudeListFM = stateInfo.current.nir.albplwave

                else:
                    raise RuntimeError(
                        f"Unknown type for {speciesInformationFile.filename} constraintType is {speciesInformationFile.constraintType}"
                    )
            else:
                # already made maps, above

                retrievalParameters = range(mm)
                pressureList = np.zeros(mm) - 2
                pressureListFM = np.zeros(nn) - 2
                altitudeList = np.zeros(mm) - 2
                altitudeListFM = np.zeros(nn) - 2

            if (
                species_name == "NIRALBBRDF"
                or species_name == "NIRALBLAMB"
                or species_name == "NIRALBCM"
            ):
                pressureListFM = (stateInfo.current["nir"]["albplwave"]).reshape(
                    len(stateInfo.current["nir"]["albplwave"])
                )

                if speciesInformationFile.constraintType == "Diagonal":
                    val = (speciesInformationFile.sSubaDiagonalValues).split(",")
                    for jj in range(len(val)):
                        val[jj] = np.double(val[jj])
                    sSubaDiagonalValues = val
                    constraintMatrix = np.identity(mm)
                    for jj in range(mm):
                        constraintMatrix[jj, jj] = (
                            1 / sSubaDiagonalValues[jj] / sSubaDiagonalValues[jj]
                        )
                elif speciesInformationFile.constraintType == "Full":
                    file = mpy.constraint_read(speciesInformationFile.sSubaFilename)
                    constraintMatrix = np.invert(file["data"])
                else:
                    raise RuntimeError(
                        f"Unknown type for {speciesInformationFile.filename} constraintType is {speciesInformationFile.constraintType}"
                    )

            # change to log if specified
            if (speciesInformationFile.mapType).upper() == "LOG":
                initialGuessListFM = np.log(initialGuessListFM)
                constraintVector = np.log(constraintVector)
                initialGuessList = np.log(initialGuessList)
                if self.state_info.has_true_values():
                    trueParameterList = np.log(trueParameterList)
                    trueParameterListFM = np.log(trueParameterListFM)

            if "ALB" in species_name:
                ind = (np.where(initialGuessListFM < -990))[0]
                if len(ind) > 0:
                    raise RuntimeError(
                        "Error -999 in albedo. Possibly mismatch between state file inputs and strategy table for ALBLAMB vs. ALBBRDF"
                    )

        elif (
            (species_name == "EMIS")
            or (species_name == "CLOUDEXT")
            or (species_name == "CALSCALE")
            or (species_name == "CALOFFSET")
        ):
            # IDL AT_LINE 243 Get_Species_Information:
            microwindows = []
            for swin in current_strategy_step.spectral_window_dict.values():
                microwindows.extend(swin.muses_microwindows())

            # Select non-UV windows
            ind = []
            for ff in range(len(microwindows)):
                if "UV" not in microwindows[ff]["filter"]:
                    ind.append(ff)

            temp_microwindows = microwindows
            microwindows = []
            for ff in range(len(ind)):
                microwindows.append(temp_microwindows[ind[ff]])

            # Get species specific things, e.g. get frequency grid
            # for EMIS from EMIS pars
            # AT_LINE 250 Get_Species_Information.pro
            if species_name == "EMIS":
                frequencyIn = stateInfo.emisPars["frequency"][
                    0 : int(stateInfo.emisPars["num_frequencies"])
                ]
                # AT_LINE 252 Get_Species_Information.pro
                stepFMSelect = mpy.mw_frequency_needed(microwindows, frequencyIn)

                # AT_LINE 254 Get_Species_Information.pro
                nn = len(stepFMSelect)
                ind = np.where(stepFMSelect != 0)
                ind = ind[0]
                mm = len(ind)
                freq = stateInfo.emisPars["frequency"][0:nn]
                ind = np.where(stepFMSelect != 0)
                ind = ind[0]

                # AT_LINE 258 Get_Species_Information.pro
                freqRet = freq[ind]
                freqRet = np.asarray(freqRet)

                # in this, all frequencies that are between other
                # frequencies are mapped.  Then take out frequencies
                # more than 20 from the retrieved on both sides

                ind = np.where(stepFMSelect != 0)
                ind = ind[0]
                ind = ind + 1

                linearFlag = True
                averageFlag = False
                maps = mpy.make_maps(
                    stateInfo.emisPars["frequency"][0:nn], ind, linearFlag, averageFlag
                )
                mapToState = maps["toState"]  # SPECIES_NAME 'EMIS'
                mapToParameters = maps["toPars"]  # mapToParameters.shape (121, 2)

                # if an EMIS frequency was not retrieved but is in a
                # gap larger than 50 cm-1, then remove it
                # All frequencies bracketed by retrieved frequencies
                # are AUTOMATICALLY interpolated by mapping, above.
                # AT_LINE 274 Get_Species_Information.pro
                num_rows_cleared = 0
                for kk in range(0, nn):
                    ind = np.where(freqRet < freq[kk])
                    if len(ind[0]) > 0:
                        ind = ind[0]
                        ind1 = np.amax(ind)
                        ind1_arr = []
                        ind1_arr.append(ind1)
                        ind1 = np.asarray(
                            ind1_arr
                        )  # Convert a list of 1 element into an array of 1 element.
                    else:
                        ind1 = []  # An empty list since there are none that fit the criteria: freqRet < freq[kk]

                    ind = np.where(freqRet > freq[kk])
                    if len(ind[0]) > 0:
                        ind = ind[0]
                        ind = np.asarray(ind)
                        ind2 = np.amin(ind)
                        ind2_arr = []
                        ind2_arr.append(ind2)
                        ind2 = np.asarray(
                            ind2_arr
                        )  # Convert a list of 1 element into an array of 1 element.
                    else:
                        ind2 = []  # An empty list since there are none that fit the criteria: freqRet > freq[kk]

                    ind = np.where(np.absolute(freqRet - freq[kk]) < 0.001)
                    ind = ind[0]

                    # AT_LINE 278 Get_Species_Information.pro
                    if len(ind) == 0 and len(ind1) > 0 and len(ind2) > 0:
                        # calculate the frequency difference between the
                        # given frequency point and the closest points with larger and smaller
                        # frequency
                        diff1 = (
                            freq[kk] - freqRet[ind1[0]]
                        )  # We only want 1 value from freqRet
                        diff2 = (
                            freqRet[ind2[0]] - freq[kk]
                        )  # We only want 1 value from freqRet
                        if diff1 + diff2 > 50:
                            # zero out interpolation if gap larger
                            # than 50 cm-1
                            mapToState[:, kk] = 0
                            num_rows_cleared = num_rows_cleared + 1
                # end for kk in range(0, nn):

                # Set pars for retrieval
                # AT_LINE 294 Get_Species_Information.pro
                pressureListFM = stateInfo.emisPars["frequency"][0:nn]
                altitudeListFM = (
                    stateInfo.emisPars["frequency"][0:nn] / 100.0
                )  # just for spacing
                constraintVectorFM = stateInfo.constraint["emissivity"][0:nn]
                initialGuessListFM = stateInfo.current["emissivity"][0:nn]
                if self.state_info.has_true_values():
                    trueParameterListFM = stateInfo.true["emissivity"][0:nn]
            # end part of if (species_name == 'EMIS'):
            # AT_LINE 300 Get_Species_Information.pro

            # clouds complicated by capability of having two true
            # clouds and bracket mode for IGR step
            # AT_LINE 305 Get_Species_Information.pro
            if species_name == "CLOUDEXT":
                # get IGR frequency mode
                # AT_LINE 308 Get_Species_Information.pro
                filename = self.retrieval_config["CloudParameterFilename"]
                if not os.path.isfile(filename):
                    raise RuntimeError(f"File not found:  {filename}")

                (_, fileID) = mpy.read_all_tes_cache(filename)
                freqMode = mpy.tes_file_get_preference(
                    fileID, "CLOUDEXT_IGR_Min_Freq_Spacing"
                )
                freqMode = (freqMode.split())[
                    0
                ]  # In case the preference contains multiple tokens, we only wish to get the first one.

                # get which freqs are used in this step... consider
                # step type
                # AT_LINE 318 Get_Species_Information.pro
                frequencyIn = stateInfo.cloudPars["frequency"][
                    0 : int(stateInfo.cloudPars["num_frequencies"])
                ]
                stepType = current_strategy_step.retrieval_type
                stepFMSelect = mpy.mw_frequency_needed(
                    microwindows, frequencyIn, stepType, freqMode
                )

                nn = len(stepFMSelect)
                ind = np.where(stepFMSelect != 0)
                ind = ind[0]
                mm = len(ind)

                averageFlag = False
                if freqMode.lower() == "one":
                    averageFlag = True
                    mm = 1

                ind = np.where(stepFMSelect != 0)[0]
                ind = ind + 1
                linearFlag = True

                # Note: lines 333 and 334 in IDL are unnecessary since mapToState and mapToParameters gets assigned to maps.toState and maps.toPars.
                # AT_LINE 336 Get_Species_Information.pro
                maps = mpy.make_maps(
                    stateInfo.cloudPars["frequency"][0:nn], ind, linearFlag, averageFlag
                )
                maps = mpy.ObjectView(maps)
                mapToState = maps.toState
                mapToParameters = maps.toPars

                # AT_LINE 340 Get_Species_Information.pro
                pressureListFM = stateInfo.cloudPars["frequency"][0:nn]
                altitudeListFM = stateInfo.cloudPars["frequency"][0:nn] / 100.0
                constraintVectorFM = np.log(
                    stateInfo.constraint["cloudEffExt"][0, 0:nn]
                )
                initialGuessListFM = np.log(stateInfo.current["cloudEffExt"][0, 0:nn])
                if self.state_info.has_true_values():
                    trueParameterListFM = np.log(stateInfo.true["cloudEffExt"][0, 0:nn])

                    # get true values for 2 clouds.
                    # try to combine the two clouds into 1, since we'll have
                    # only 1 cloud Jacobian, for the linear estimate
                    # AT_LINE 351 Get_Species_Information.pro
                    if int(stateInfo.true["num_clouds"]) == 2:
                        # take larger cloud; then try to fold smaller cloud in
                        c1 = stateInfo.true["cloudEffExt"][0, 0:nn]
                        c2 = stateInfo.true["cloudEffExt"][1, 0:nn]
                        trueParameterListFM = np.log(c1 + c2)
                # end part of if (species_name == 'CLOUDEXT'):

            # AT_LINE 360 Get_Species_Information.pro
            if species_name == "CALSCALE":
                # get which freqs are used in this step

                # AT_LINE 363 Get_Species_Information.pro
                frequencyIn = stateInfo.calibrationPars["frequency"][
                    0 : int(stateInfo.calibrationPars["num_frequencies"])
                ]
                stepFMSelect = mpy.mw_frequency_needed(
                    microwindows, frequencyIn, stepType, freqMode
                )

                # AT_LINE 365 Get_Species_Information.pro
                nn = len(stepFMSelect)
                ind = np.where(stepFMSelect != 0)
                ind = ind[0]
                mm = len(ind)
                mapToParameters = np.zeros(shape=(nn, mm), dtype=np.float64)
                mapToState = np.zeros(shape=(mm, nn), dtype=np.float64)

                # For CALSCALE species, do NOT allow interpolation.  Just have 1:1
                # mapping.  E.g. if retrieve in 2B1 and 2A1
                # filters, results should not go into the 1B2 filter.
                # ideally interpolation allowed within filters only
                count = 0
                for ik in range(0, nn):
                    if stepFMSelect[ik] == 1:
                        mapToState[count, ik] = 1
                        mapToParameters[ik, count] = 1
                        count = count + 1

                pressureListFM = stateInfo.calibrationPars["frequency"][0:nn]
                altitudeListFM = stateInfo.calibrationPars["frequency"][0:nn] / 100.0
                constraintVectorFM = stateInfo.constraint["calibrationScale"][0:nn]
                initialGuessListFM = stateInfo.current["calibrationScale"][0:nn]
                if self.state_info.has_true_values():
                    trueParameterListFM = stateInfo.true["calibrationScale"][0:nn]
                assert False
            # end part of if (species_name == 'CALSCALE'):

            # AT_LINE 391 Get_Species_Information.pro
            if species_name == "CALOFFSET":
                # get which freqs are used in this step
                frequencyIn = stateInfo.calibrationPars["frequency"][
                    0 : stateInfo.calibrationPars["num_frequencies"]
                ]
                stepFMSelect = mpy.mw_frequency_needed(microwindows, frequencyIn)

                # AT_LINE 397 Get_Species_Information.pro
                nn = len(stepFMSelect)
                ind = np.where(stepFMSelect != 0)
                ind = ind[0]
                mm = len(ind)

                # AT_LINE 400 Get_Species_Information.pro
                mapToParameters = np.zeros(shape=(nn, mm), dtype=np.float64)
                mapToState = np.zeros(shape=(mm, nn), dtype=np.float64)

                # For CALOFFSET species, do NOT allow interpolation.  Just have 1:1
                # mapping

                # AT_LINE 407 Get_Species_Information.pro
                count = 0
                for ik in range(0, nn):
                    if stepFMSelect[ik] == 1:
                        mapToState[count, ik] = 1
                        mapToParameters[ik, count] = 1
                        count = count + 1

                pressureListFM = stateInfo.calibrationPars["frequency"][0:nn]
                altitudeListFM = stateInfo.calibrationPars["frequency"][0:nn] / 100.0
                constraintVectorFM = stateInfo.constraint["calibrationOffset"][0:nn]
                initialGuessListFM = stateInfo.current["calibrationOffset"][0:nn]
                if self.state_info.has_true_values():
                    trueParameterListFM = stateInfo.true["calibrationOffset"][0:nn]
                assert False
            # end part of if (species_name == 'CALOFFSET'):

            # AT_LINE 425 Get_Species_Information.pro

            # now map FM to retrieval grid.  If log parameter, the
            # FM is already in log

            ind = np.where(stepFMSelect != 0)
            ind = ind[0]
            mm = len(ind)
            m = np.asarray(mapToParameters)

            if mm > 1:
                altitudeList = np.matmul(altitudeListFM, m)
                pressureList = np.matmul(pressureListFM, m)
                constraintVector = np.matmul(constraintVectorFM, m)
                initialGuessList = np.matmul(initialGuessListFM, m)
                if self.state_info.has_true_values():
                    trueParameterList = np.matmul(trueParameterListFM, m)
            else:
                altitudeList = np.sum(m * altitudeListFM)
                pressureList = np.sum(m * pressureListFM)
                constraintVector = np.sum(m * constraintVectorFM)
                initialGuessList = np.sum(m * initialGuessListFM)
                if self.state_info.has_true_values():
                    trueParameterList = np.sum(m * trueParameterListFM)

            # AT_LINE 443 Get_Species_Information.pro
            # Now get constraint matrix.
            if constraintType.lower() == "tikhonov":
                constraintMatrix = np.zeros(
                    shape=(mm, mm), dtype=np.float64
                )  # DBLARR(mm,mm)
                myError = (1 / (0.1)) ** 2
                constraintMatrix[0, 0] = myError
                constraintMatrix[mm - 1, mm - 1] = myError
                for ll in range(0, mm - 2):
                    if pressureList[ll + 1] - pressureList[ll] < 30:
                        constraintMatrix[ll, ll] = constraintMatrix[ll, ll] + myError
                        constraintMatrix[ll + 1, ll] = (
                            constraintMatrix[ll + 1, ll] - myError
                        )
                        constraintMatrix[ll, ll + 1] = (
                            constraintMatrix[ll, ll + 1] - myError
                        )
                        constraintMatrix[ll + 1, ll + 1] = (
                            constraintMatrix[ll + 1, ll + 1] + myError
                        )
                    else:
                        constraintMatrix[ll, ll] = constraintMatrix[ll, ll] + myError
                        constraintMatrix[ll + 1, ll + 1] = (
                            constraintMatrix[ll + 1, ll + 1] + myError
                        )
            elif constraintType.lower() == "full":
                sSubaFilename = speciesInformationFile.sSubaFilename
                constraintMatrix = mpy.supplier_constraint_matrix_ssuba(
                    constraintVector,
                    species_name,
                    mapType,
                    mapToParameters,
                    pressureListFM,
                    pressureList,
                    sSubaFilename,
                )
            elif constraintType.lower == "premade":
                raise RuntimeError(
                    "premade type not implemented for spectral type check Supplier_Constraint_Matrix for previous implementation"
                )

                # problem with pre-made pre-inverted
                # constraint... when select some frequency it
                # seems to be ill-conditioned when inverted to
                # covariance...
            else:
                raise RuntimeError(
                    f"Unknown constraint type for {species_name} {constraintType}"
                )

            # done with spectral species

        # end part of elif (species_name == 'EMIS')     or (species_name == 'CLOUDEXT')
        #                  (species_name == 'CALSCALE') or (species_name == 'CALOFFSET'):

        # AT_LINE 489 Get_Species_Information.pro
        elif species_name == "PTGANG":
            # AT_LINE 491 Get_Species_Information.pro
            nn = 1
            mm = 1

            mapToState = 1
            mapToParameters = 1

            # it's difficult to get a nx1 array.  1xn is easy.
            retrievalParameters = [0]
            altitudeList = [-2]
            altitudeListFM = [-2]
            pressureList = [-2]
            pressureListFM = [-2]
            num_retrievalParameters = 1

            # set pars for retrieval
            constraintVector = stateInfo.constraint["tes"]["boresightNadirRadians"]
            constraintVectorFM = stateInfo.constraint["tes"]["boresightNadirRadians"]
            initialGuessList = stateInfo.current["tes"]["boresightNadirRadians"]
            initialGuessListFM = stateInfo.current["tes"]["boresightNadirRadians"]
            if self.state_info.has_true_values():
                trueParameterList = stateInfo.true["tes"]["boresightNadirRadians"]
                trueParameterListFM = stateInfo.true["tes"]["boresightNadirRadians"]

            raise RuntimeError(
                f"Need more coding from developer. species_name {species_name}"
            )
            # Constraint = INVERT(Constraint_Get(errorInitial, species))

        # end part of elif (species_name == 'PTGANG'):

        # AT_LINE 515 Get_Species_Information.pro
        elif species_name == "PCLOUD":
            nn = 1
            mm = 1

            mapToState = 1
            mapToParameters = 1

            retrievalParameters = [0]
            altitudeList = [-2]
            altitudeListFM = [-2]
            pressureList = [-2]
            pressureListFM = [-2]
            num_retrievalParameters = 1

            # set pars for retrieval
            constraintVector = math.log(stateInfo.constraint["PCLOUD"][0])
            constraintVectorFM = math.log(stateInfo.constraint["PCLOUD"][0])
            initialGuessList = math.log(stateInfo.current["PCLOUD"][0])
            initialGuessListFM = math.log(stateInfo.current["PCLOUD"][0])
            if self.state_info.has_true_values():
                trueParameterList = math.log(stateInfo.true["PCLOUD"][0])
                trueParameterListFM = math.log(stateInfo.true["PCLOUD"][0])

            # get constraint
            stepType = current_strategy_step.retrieval_type
            tag_names = mpy.idl_tag_names(speciesInformationFile)
            if "sSubaDiagonalValues" not in tag_names:
                raise RuntimeError(
                    f"Preference 'sSubaDiagonalValues' NOT found in file {speciesInformationFile.filename}"
                )

            # AT_LINE 546 Get_Species_Information.pro
            constraintMatrix = np.float64(speciesInformationFile.sSubaDiagonalValues)
            constraintMatrix = 1 / constraintMatrix / constraintMatrix

            if self.state_info.has_true_values():
                # pick thickest cloud for cloud height
                if stateInfo.true["num_clouds"] == 2:
                    # take larger cloud; then try to fold smaller cloud in
                    c1 = stateInfo.true["cloudEffExt"][0, :]
                    c2 = stateInfo.true["cloudEffExt"][1, :]

                    if np.sum(c1) > np.sum(c2):
                        trueParameterList = math.log(stateInfo.true["PCLOUD"][1])
        # end part of elif (species_name == 'PCLOUD'):
        # AT_LINE 558 Get_Species_Information.pro
        elif species_name == "RESSCALE":
            nn = 2
            mm = 2

            mapToState = np.identity(2)
            mapToParameters = np.identity(2)

            retrievalParameters = [0]
            altitudeList = [2]
            altitudeListFM = [2]
            pressureList = [2]
            pressureListFM = [2]
            num_retrievalParameters = 2

            # set pars for retrieval
            constraintVector = [1, 1]
            constraintVectorFM = [1, 1]
            initialGuessList = [0, 0]
            initialGuessListFM = [1, 1]
            if self.state_info.has_true_values():
                trueParameterList = [1, 1]
                trueParameterListFM = [1, 1]

            sSubaDiagonalValues = np.float64(speciesInformationFile.sSubaDiagonalValues)
            constraintMatrix = 1 / sSubaDiagonalValues / sSubaDiagonalValues
        # end part of elif (species_name == 'RESSCALE'):
        elif species_name == "TSUR":
            # AT_LINE 583 Get_Species_Information.pro
            nn = 1
            mm = 1

            mapToState = 1
            mapToParameters = 1

            # it's difficult to get a nx1 array.  1xn is easy.
            # AT_LINE 592 Get_Species_Information.pro
            retrievalParameters = [0]
            altitudeList = [-2]
            altitudeListFM = [-2]
            pressureList = [-2]
            pressureListFM = [-2]
            num_retrievalParameters = 1

            # set pars for retrieval
            # AT_LINE 600 Get_Species_Information.pro
            constraintVectorFM = stateInfo.constraint["TSUR"]
            constraintVector = stateInfo.constraint["TSUR"]
            initialGuessList = stateInfo.current["TSUR"]
            initialGuessListFM = stateInfo.current["TSUR"]
            if self.state_info.has_true_values():
                trueParameterList = stateInfo.true["TSUR"]
                trueParameterListFM = stateInfo.true["TSUR"]

            tag_names = mpy.idl_tag_names(speciesInformationFile)
            upperTags = [x.upper() for x in tag_names]
            step_name = current_strategy_step.strategy_step.step_name
            full_step_label = ("sSubaDiagonalValues-" + step_name).upper()
            if full_step_label in upperTags:
                sSubaDiagonalValues = np.asarray(
                    speciesInformationFile[upperTags.index(full_step_label)]
                )
                logger.warning(
                    f"using step-dependent constraint for TSUR, value: {1 / (sSubaDiagonalValues * sSubaDiagonalValues)}"
                )
            else:
                sSubaDiagonalValues = np.float64(
                    speciesInformationFile.sSubaDiagonalValues
                )

            constraintMatrix = 1 / (sSubaDiagonalValues * sSubaDiagonalValues)
        # end part of elif (species_name == 'TSUR'):
        elif species_name == "PSUR":
            # AT_LINE 583 Get_Species_Information.pro
            nn = 1
            mm = 1

            mapToState = 1
            mapToParameters = 1

            # it's difficult to get a nx1 array.  1xn is easy.
            # AT_LINE 592 Get_Species_Information.pro
            retrievalParameters = [0]
            altitudeList = [-2]
            altitudeListFM = [-2]
            pressureList = [-2]
            pressureListFM = [-2]
            num_retrievalParameters = 1

            # set pars for retrieval
            # AT_LINE 696 Get_Species_Information.pro
            constraintVector = stateInfo.constraint["pressure"][0]
            constraintVectorFM = stateInfo.constraint["pressure"][0]
            initialGuessList = stateInfo.current["pressure"][0]
            initialGuessListFM = stateInfo.current["pressure"][0]
            if self.state_info.has_true_values():
                trueParameterList = stateInfo.true["pressure"][0]
                trueParameterListFM = stateInfo.true["pressure"][0]

            tag_names = mpy.idl_tag_names(speciesInformationFile)
            upperTags = [x.upper() for x in tag_names]
            step_name = current_strategy_step.strategy_step.step_name
            full_step_label = ("sSubaDiagonalValues-" + step_name).upper()
            if full_step_label in upperTags:
                sSubaDiagonalValues = np.asarray(
                    speciesInformationFile[upperTags.index(full_step_label)]
                )
                logger.warning(
                    f"using step-dependent constraint for PSUR, value: {1 / (sSubaDiagonalValues * sSubaDiagonalValues)}"
                )
            else:
                sSubaDiagonalValues = np.float64(
                    speciesInformationFile.sSubaDiagonalValues
                )

            constraintMatrix = 1 / (sSubaDiagonalValues * sSubaDiagonalValues)
        # end part of elif (species_name == 'PSUR'):
        elif (mapType == "linearscale") or (mapType == "logscale"):
            # AT_LINE 718 Get_Species_Information.pro
            if species_name not in stateInfo.species:
                raise RuntimeError(
                    "Species not found in stateInfo.  This usually means your spectral windows do not include this species OR the L2_Setup does not list this species. Looking for species: {species_name}"
                )
            retrievalParameters = 5  # level 5

            num_retrievalParameters = 1
            mm = num_retrievalParameters
            nn = stateInfo.num_pressures

            pressureList = pressure[retrievalParameters - 1]
            pressureListFM = pressure
            altitudeList = stateInfo.current["heightKm"][retrievalParameters - 1]
            altitudeListFM = stateInfo.current["heightKm"]

            # map isn't used but size is useful
            # maps = {'toPars':np.zeros(shape=(nn,mm), dtype=np.float64),
            #    'toState':np.zeros(shape=(mm,nn), dtype=np.float64)}
            # maps = make_maps(stateInfo.current['pressure'], retrievalParameters)
            # maps = ObjectView(maps)
            mapToParameters = (
                np.zeros(shape=(nn, mm), dtype=np.float64) + 1 / 20
            )  # not sure about this
            mapToState = (
                np.zeros(shape=(mm, nn), dtype=np.float64) + 1
            )  # only for mapping Jacobian

            # PYTHON_NOTE: Because the value of stateInfo.species is a list, we have to convert to numpy array.
            ind = np.where(species_name == np.asarray(stateInfo.species))[0]

            initialGuessFM = stateInfo.current["values"][
                ind, :
            ]  # Keep the array as 2 dimensions so we can multiply them later.
            currentGuessFM = stateInfo.current["values"][ind, :]
            constraintVectorFM = stateInfo.constraint["values"][ind, :]
            constraintMatrix = stateInfo.constraint["values"][ind, :]
            if self.state_info.has_true_values():
                trueStateFM = stateInfo.true["values"][ind, :]

            sSubaDiagonalValues = np.float64(speciesInformationFile.sSubaDiagonalValues)
            constraintMatrix = 1 / (sSubaDiagonalValues * sSubaDiagonalValues)

            # AT_LINE 752 Get_Species_Information.pro

            # since the "true" is relative to the initial guess
            # the "true state" is set to e.g. 0.8 if the initial guess
            # is off by -0.8K
            if mapType == "linearscale":
                constraintVector = 0
                initialGuessList = np.mean(initialGuessFM - constraintVectorFM)
                initialGuessListFM = constraintVectorFM + initialGuessList
                if self.state_info.has_true_values():
                    trueParameterList = np.mean(trueStateFM - initialGuessFM)
                    trueParameterListFM = np.copy(trueStateFM)
            else:
                constraintVector = 1
                initialGuessList = np.mean(constraintVectorFM / initialGuessFM)
                initialGuessListFM = constraintVectorFM * initialGuessList
                if self.state_info.has_true_values():
                    trueParameterList = np.mean(trueStateFM / initialGuessFM)
                    trueParameterListFM = np.copy(trueStateFM)
        # end part of elif (mapType == 'linearscale') or (mapType == 'logscale'):

        elif (mapType == "linearpca") or (mapType == "logpca"):
            # this is state vector of the form:
            # current = apriori + mapToState @ currentGuess for linearpca
            # or
            # log(current) = log(apriori) + mapToState @ log(currentGuess) for logpca
            # Doing current = mapToState @ currentGuess does not work
            # because the maps do not have a good span of the state, e.g. the stratosphere does not have sensitivity.
            # when I tried this I got Tatm = [300, ..., 62, 37, -20, -27]
            # so it must be aprior + offset

            if species_name not in stateInfo.species:
                raise RuntimeError(
                    f"Species not found in stateInfo.  This usually means your spectral windows do not include this species OR the L2_Setup does not list this species. Looking for species: {species_name}"
                )

            mapsFilename = speciesInformationFile.mapsFilename.replace(
                "64_", str(stateInfo.num_pressures) + "_"
            )

            # retrieval "levels"
            levels_tokens = speciesInformationFile.retrievalLevels.split(",")
            int_levels_arr = [int(x) for x in levels_tokens]
            retrievalParameters = int_levels_arr

            num_retrievalParameters = len(retrievalParameters)
            mm = num_retrievalParameters
            nn = stateInfo.num_pressures

            pressureListFM = pressure
            altitudeListFM = stateInfo.current["heightKm"]

            # implemented for OCO-2, but if used for other satellite
            # need to change # of full state levels to read correct file
            # for oco-2:  maps_TATM_Linear_20_3.nc, where 20 is # of full state pressures
            (mapDict, _, _) = mpy.cdf_read_dict(mapsFilename)
            mapToState = np.transpose(mapDict["to_state"])
            mapToParameters = np.transpose(mapDict["to_pars"])

            altitudeList = np.transpose(mapToParameters) @ altitudeListFM
            pressureList = (
                np.transpose(mapToParameters) @ pressure
            )  # nonsense values, e.g. [1595,  1833, 594]
            pressureList[:] = -999
            altitudeList[:] = -999

            filename = speciesInformationFile.constraintFilename
            (constraintStruct, constraintPressure) = mpy.constraint_read(filename)
            constraintMatrix = mpy.constraint_get(constraintStruct)

            # PYTHON_NOTE: Because the value of stateInfo.species is a list, we have to convert to numpy array.
            ind = np.where(species_name == np.asarray(stateInfo.species))[0]

            initialGuessFM = stateInfo.current["values"][ind, :].reshape(
                nn
            )  # Keep the array as 2 dimensions so we can multiply them later.
            currentGuessFM = stateInfo.current["values"][ind, :].reshape(nn)
            constraintVectorFM = stateInfo.constraint["values"][ind, :].reshape(nn)
            if self.state_info.has_true_values():
                trueStateFM = stateInfo.true["values"][ind, :].reshape(nn)

            # since the "true" is relative to the a priori
            # the "true state" is set to e.g. 0.8 if the a priori
            # is off by -0.8K
            # atmospheric parameters
            if mapType == "linearpca":
                constraintVector = np.zeros(mm, dtype=np.float32) + 0
                initialGuessList = np.transpose(mapToParameters) @ (
                    initialGuessFM - constraintVectorFM
                )
                initialGuessListFM = (
                    np.copy(constraintVectorFM)
                    + np.transpose(mapToState) @ initialGuessList
                )

                if self.state_info.has_true_values():
                    trueParameterList = np.transpose(mapToParameters) @ (
                        trueStateFM - constraintVectorFM
                    )
                    trueParameterListFM = np.copy(trueStateFM)
            else:
                constraintVector = np.zeros(mm, dtype=np.float32) + 0
                initialGuessList = np.transpose(mapToParameters) @ (
                    np.log(initialGuessFM) - np.log(constraintVectorFM)
                )
                initialGuessListFM = (
                    np.copy(constraintVectorFM)
                    + np.transpose(np.mapToState) @ initialGuessList
                )
                if self.state_info.has_true_values():
                    trueParameterList = np.transpose(mapToParameters) @ (
                        np.log(trueStateFM) - np.log(constraintVectorFM)
                    )
                    trueParameterListFM = np.copy(trueStateFM)
        # end part of elif (mapType == 'linearpca') or (mapType == 'logpca'):

        else:
            # AT_LINE 629 Get_Species_Information.pro
            # line parameter, e.g. H2O, CO2, O3, TATM, ...

            if species_name not in stateInfo.species:
                raise RuntimeError(
                    f"Species not found in stateInfo.  This usually means your spectral windows do not include this species OR the L2_Setup does not list this species. Looking for species: {species_name}"
                )

            # maps
            if (mapType == "linear") or (mapType == "log"):
                # We read in the retrieval levels and modify for
                # current pressure grid
                # Because the value of speciesInformationFile.retrievalLevels is a long string of:
                #    '1,2,3,4,5,6,7,8,10,12,14,16,18,21,24,27,30,33,36,39,42,45,48,51,53,54,55,58,60,62,64,66'
                # We need to split it up.
                levels_tokens = speciesInformationFile.retrievalLevels.split(",")
                int_levels_arr = [int(x) for x in levels_tokens]
                levels0 = np.asarray(int_levels_arr)
                pinput = self.retrieval_config["pressure_species_input"]
                retrievalParameters = mpy.supplier_retrieval_levels_tes(
                    levels0, pinput, stateInfo.current["pressure"]
                )

                # PYTHON_NOTE: It is possible that some values in i_levels may index passed the size of pressure.
                # The size of pressure may be 63 and one indices may be 64.
                any_values_greater_than_size = (
                    retrievalParameters > pressure.size
                ).any()
                if any_values_greater_than_size:
                    o_cleaned_retrievalParameters = mpy.utilLevels.RemoveIndicesTooBig(
                        retrievalParameters, pressure, "StateElement"
                    )
                    # Reassign retrievalParameters to o_cleaned_retrievalParameters so it will contain indices that are within size of pressure.
                    retrievalParameters = o_cleaned_retrievalParameters

                # AT_LINE 636 Get_Species_Information.pro
                num_retrievalParameters = len(retrievalParameters)
                mm = num_retrievalParameters
                nn = stateInfo.num_pressures

                pressureList = pressure[retrievalParameters - 1]
                pressureListFM = pressure
                altitudeList = stateInfo.current["heightKm"][retrievalParameters - 1]
                altitudeListFM = stateInfo.current["heightKm"]

                maps = mpy.make_maps(stateInfo.current["pressure"], retrievalParameters)
                maps = mpy.ObjectView(maps)
                mapToParameters = maps.toPars
                mapToState = maps.toState
            else:
                # AT_LINE 699 Get_Species_Information.pro
                raise RuntimeError("Only linear/log/pca maps implemented")

            # AT_LINE 652 Get_Species_Information.pro

            # PYTHON_NOTE: Because the value of stateInfo.species is a list, we have to convert to numpy array.
            ind = np.where(species_name == np.asarray(stateInfo.species))[0]

            initialGuessFM = stateInfo.current["values"][
                ind, :
            ]  # Keep the array as 2 dimensions so we can multiply them later.
            currentGuessFM = stateInfo.current["values"][ind, :]
            constraintVectorFM = stateInfo.constraint["values"][ind, :]
            if self.state_info.has_true_values():
                trueStateFM = stateInfo.true["values"][ind, :]

            # AT_LINE 668 Get_Species_Information.pro
            if mapType.lower() == "log":
                if self.state_info.has_true_values():
                    if self.retrieval_config["mapTrueFullStateVector"] == "yes":
                        trueStateFM = np.exp(
                            np.matmul(
                                np.matmul(mapToState, mapToParameters),
                                np.log(trueStateFM),
                            )
                        )

                if self.retrieval_config["mapInitialGuess"] == "yes":
                    initialGuessFM = np.exp(
                        np.matmul(
                            np.log(initialGuessFM),
                            np.matmul(mapToParameters, mapToState),
                        )
                    )
                    currentGuessFM = np.exp(
                        np.matmul(
                            np.log(currentGuessFM),
                            np.matmul(mapToParameters, mapToState),
                        )
                    )
            else:
                if self.state_info.has_true_values():
                    if self.retrieval_config["mapTrueFullStateVector"] == "yes":
                        trueStateFM = np.matmul(
                            np.matmul(mapToState, mapToParameters), trueStateFM
                        )

            if mapType.lower() == "log":
                constraintVector = np.matmul(
                    np.log(constraintVectorFM), mapToParameters
                )
            else:
                constraintVector = np.matmul(constraintVectorFM, mapToParameters)

                # AT_LINE 693 src_ms-2018-12-10/Get_Species_Information.pro
                if (
                    len(constraintVectorFM.shape) >= 2
                    and constraintVectorFM.shape[0] == 1
                ):
                    # Re-shape back to 1-D array: from (1,30) to (30,)
                    constraintVector = np.reshape(
                        constraintVector, (constraintVector.shape[1])
                    )

                if np.amin(constraintVector) < 0 and np.amax(constraintVector) > 0:
                    # fix issue with mapping going to negative #s
                    logger.info(
                        f"Fix negative mapping: constraintVector: species_name: {species_name}"
                    )

                    ind1 = np.where(constraintVector < 0)[0]
                    ind2 = np.where(constraintVector > 0)[0]
                    constraintVector[ind1] = np.amin(constraintVector[ind2])
                # end if np.amin(constraintCector) < 0 and np.amax(constraintVector) > 0:

            # AT_LINE 693 Get_Species_Information.pro

            initialGuessFM = currentGuessFM

            # for jointly retrieved... don't populate
            # constraint check H2O-HDO, if so, get off diagonal also
            locs = [
                (np.where(np.asarray(species_list) == "H2O"))[0],
                (np.where(np.asarray(species_list) == "HDO"))[0],
            ]
            num_retrievalPressures = len(retrievalParameters)

            # AT_LINE 702 Get_Species_Information.pro
            if (
                locs[0] >= 0
                and locs[1] >= 0
                and (species_name == "H2O" or species_name == "HDO")
            ):
                constraintMatrix = np.zeros(
                    shape=(num_retrievalPressures, num_retrievalPressures),
                    dtype=np.float64,
                )
            else:
                # AT_LINE 708 Get_Species_Information.pro
                if constraintType == "premade":
                    filename = speciesInformationFile.constraintFilename
                    if filename[0] == "":
                        raise RuntimeError(
                            f"Name not found for PREMADE constraint: {filename}"
                        )

                    constraintMatrix, pressurex = (
                        mpy.supplier_constraint_matrix_premade(
                            species_name,
                            filename,
                            num_retrievalPressures,
                            i_nh3type=self.state_info.nh3type,
                            i_ch3ohtype=self.state_info.ch3ohtype,
                        )
                    )
                # AT_LINE 727 Get_Species_Information.pro
                elif constraintType == "covariance":
                    filename = speciesInformationFile.sSubaFilename
                    if filename[0] == "":
                        raise RuntimeError(
                            f"Name not found for Covariance constraint In file {speciesInformationFile.filename}"
                        )

                    constraintMatrix = mpy.supplier_constraint_matrix_ssuba(
                        constraintVector,
                        species_name,
                        mapType,
                        mapToParameters,
                        pressureListFM,
                        pressureList,
                        filename,
                        i_nh3type=self.state_info.nh3type,
                        i_ch3ohtype=self.state_info.ch3ohtype,
                    )

                # AT_LINE 747 Get_Species_Information.pro
                elif species_name == "O3" and constraintType == "McPeters":
                    raise RuntimeError(
                        "Constraint type McPeters and O3 species not implemented yet"
                    )
                else:
                    raise RuntimeError(
                        f"Constraint type not implemented: {constraintType}"
                    )
                # end else portion of if constraintType == 'premade':
            # end else portion of if locs[0] >= 0 and locs[1] >= 0 and (species_name == 'H2O' or species_name == 'HDO'):

            # AT_LINE 771 Get_Species_Information.pro
            if constraintType == "Scale":
                if mapType == "linear":
                    initialGuessList = np.mean(initialGuessListFM - constraintVectorFM)
                    initialGuessListFM = constraintVectorFM + initialGuessList
                    if self.state_info.has_true_values():
                        trueParameterList = np.mean(trueStateFM - constraintVectorFM)
                        trueParameterListFM = constraintVectorFM + trueParameterList
                else:
                    initialGuessList = np.mean(initialGuessListFM / constraintVectorFM)
                    initialGuessListFM = constraintVectorFM * initialGuessList
                    if self.state_info.has_true_values():
                        trueParameterList = np.mean(trueStateFM / constraintVectorFM)
                        trueParameterListFM = constraintVectorFM * trueParameterList
            else:
                if mapType == "log":
                    initialGuessList = np.matmul(
                        np.log(initialGuessFM), mapToParameters
                    )
                    initialGuessListFM = np.log(initialGuessFM)
                    if self.state_info.has_true_values():
                        trueParameterList = np.matmul(
                            np.log(trueStateFM), mapToParameters
                        )
                        trueParameterListFM = np.log(trueStateFM)
                elif mapType == "linear":
                    initialGuessList = np.matmul(initialGuessFM, mapToParameters)
                    initialGuessListFM = np.copy(initialGuessFM)
                    if self.state_info.has_true_values():
                        trueParameterList = np.matmul(trueStateFM, mapToParameters)
                        trueParameterListFM = np.copy(trueStateFM)

                    if (
                        len(initialGuessList.shape) >= 2
                        and initialGuessList.shape[0] == 1
                    ):
                        # Re-shape back to 1-D array: from (1,30) to (30,)
                        initialGuessList = np.reshape(
                            initialGuessList, (initialGuessList.shape[1])
                        )

                    if self.state_info.has_true_values():
                        if (
                            len(trueParameterList.shape) >= 2
                            and trueParameterList.shape[0] == 1
                        ):
                            # Re-shape back to 1-D array: from (1,30) to (30,)
                            trueParameterList = np.reshape(
                                trueParameterList, (trueParameterList.shape[1])
                            )

                    # AT_LINE 789 src_ms-2018-12-10/Get_Species_Information.pro Get_Species_Information

                    if np.amin(initialGuessList) < 0 and np.max(initialGuessList) > 0:
                        logger.info(
                            f"Fix negative mapping: initialGuessList: species_name: {species_name}"
                        )
                        # fix issue with mapping going to negative #s
                        ind1 = np.where(initialGuessList < 0)[0]
                        ind2 = np.where(initialGuessList > 0)[0]
                        initialGuessList[ind1] = np.amin(initialGuessList[ind2])
                    # end if np.amin(initialGuessList) < 0 and np.max(initialGuessList) > 0:

                    if self.state_info.has_true_values():
                        if (
                            np.amin(trueParameterList) < 0
                            and np.amax(trueParameterList) > 0
                        ):
                            logger.info(
                                f"Fix negative mapping: initialGuessList: species_name: {species_name}"
                            )
                            # fix issue with mapping going to negative #s
                            ind1 = np.where(trueParameterList < 0)[0]
                            ind2 = np.where(trueParameterList > 0)[0]
                            trueParameterList[ind1] = np.amin(trueParameterList[ind2])
                        # end if np.amin(trueParameterList) < 0 and np.amax(trueParameterList) > 0:
                else:
                    raise RuntimeError(f"mapType not handled: {mapType}")
            # end else part of if constraintType == 'Scale':

            loc = (np.where(np.asarray(stateInfo.species) == species_name))[0]
            if loc.size == 0:
                raise RuntimeError(
                    f"FM species not found {species_name}. Are you running the step this species is in?"
                )

            # AT_LINE 789 Get_Species_Information.pro
            stateInfo.initial["values"][loc, :] = initialGuessFM[:]
            stateInfo.current["values"][loc, :] = currentGuessFM[:]
            if self.state_info.has_true_values():
                stateInfo.true["values"][loc, :] = trueStateFM[:]

        # end else part from AT_LINE 629 Get_Species_Information.pro
        # end else part of if 'OMI' in species_name:

        # ---- MMS This seems to always be executed
        # AT_LINE 799 Get_Species_Information.pro

        # Convert any scalar values to array so we can use the [:] index syntax.
        if np.isscalar(altitudeList):
            altitudeList = np.asanyarray([altitudeList])
        else:
            altitudeList = np.asanyarray(altitudeList)

        if np.isscalar(altitudeListFM):
            altitudeListFM = np.asanyarray([altitudeListFM])
        else:
            altitudeListFM = np.asanyarray(altitudeListFM)

        if np.isscalar(pressureList):
            pressureList = np.asanyarray([pressureList])
        else:
            pressureList = np.asanyarray(pressureList)

        if np.isscalar(pressureListFM):
            pressureListFM = np.asanyarray([pressureListFM])
        else:
            pressureListFM = np.asanyarray(pressureListFM)

        if np.isscalar(constraintVector):
            constraintVector = np.asanyarray([constraintVector])
        else:
            constraintVector = np.asanyarray(constraintVector)

        if np.isscalar(initialGuessList):
            initialGuessList = np.asanyarray([initialGuessList])
        else:
            initialGuessList = np.asanyarray(initialGuessList)

        if np.isscalar(initialGuessListFM):
            initialGuessListFM = np.asanyarray([initialGuessListFM])
        else:
            initialGuessListFM = np.asanyarray(initialGuessListFM)

        if np.isscalar(constraintVectorFM):
            constraintVectorFM = np.asanyarray([constraintVectorFM])
        else:
            constraintVectorFM = np.asanyarray(constraintVectorFM)

        if self.state_info.has_true_values():
            if np.isscalar(trueParameterList):
                trueParameterList = np.asanyarray([trueParameterList])
            else:
                trueParameterList = np.asanyarray(trueParameterList)

            if np.isscalar(trueParameterListFM):
                trueParameterListFM = np.asanyarray([trueParameterListFM])
            else:
                trueParameterListFM = np.asanyarray(trueParameterListFM)

        # check minimum, maximum, maximumChange
        # if parameters missing, set to -999 (do not check min, max, max change)
        # allow single value or value for each retrieval parameter
        # Note below code does not account for differing # of pressure values

        minimum = np.zeros(shape=(mm), dtype=np.float64) - 999
        try:
            ff = (speciesInformationFile.minimum).split(",")
            if len(ff) == 1:
                minimum = minimum * 0 + float(ff[0])
            else:
                for ix in range(len(ff)):
                    minimum[ix] = minimum[ix] * 0 + float(ff[ix])
        except AttributeError:
            pass

        maximum = np.zeros(shape=(mm), dtype=np.float64) - 999
        try:
            ff = (speciesInformationFile.maximum).split(",")
            if len(ff) == 1:
                maximum = maximum * 0 + float(ff[0])
            else:
                for ix in range(len(ff)):
                    maximum[ix] = maximum[ix] * 0 + float(ff[ix])
        except AttributeError:
            pass

        maximum_change = np.zeros(shape=(mm), dtype=np.float64) - 999
        try:
            ff = (speciesInformationFile.maximumChange).split(",")
            if len(ff) == 1:
                maximum_change = maximum_change * 0 + float(ff[0])
            else:
                for ix in range(len(ff)):
                    maximum_change[ix] = maximum_change[ix] * 0 + float(ff[ix])
        except AttributeError:
            pass

        # If we skipped setting true values, go ahead and put a placeholder of
        # zeros, just so we don't need special handling
        if not self.state_info.has_true_values():
            trueParameterList = np.zeros_like(initialGuessList)
            trueParameterListFM = np.zeros_like(initialGuessListFM)

        self.mapType = mapType
        self.pressureList = pressureList.flatten()
        self.altitudeList = altitudeList.flatten()
        self.constraintVector = constraintVector.flatten()
        self.initialGuessList = initialGuessList.flatten()
        self.trueParameterList = trueParameterList.flatten()
        self.pressureListFM = pressureListFM.flatten()
        self.altitudeListFM = altitudeListFM.flatten()
        self.constraintVectorFM = constraintVectorFM.flatten()
        self.initialGuessListFM = initialGuessListFM.flatten()
        self.trueParameterListFM = trueParameterListFM.flatten()
        self.minimum = minimum.flatten()
        self.maximum = maximum.flatten()
        self.maximum_change = maximum_change.flatten()
        self.mapToState = mapToState
        self.mapToParameters = mapToParameters
        self.constraintMatrix = constraintMatrix


class MusesPyStateElementHandleOld(StateElementHandleOld):
    def state_element_object(
        self, state_info: StateInfoOld, name: StateElementIdentifier
    ) -> tuple[
        bool,
        tuple[StateElementOld, StateElementOld, StateElementOld, StateElementOld]
        | None,
    ]:
        return (
            True,
            (
                MusesPyStateElementOld(state_info, name, "initialInitial"),
                MusesPyStateElementOld(state_info, name, "initial"),
                MusesPyStateElementOld(state_info, name, "current"),
                MusesPyStateElementOld(state_info, name, "true"),
            ),
        )


class MusesPyOmiStateElementOld(MusesPyStateElementOld):
    """MUSES-py groups all the OMI state elements together. While we could pull
    this apart, the create_uip depends on finding the OMI stuff in a "omi" key
    in the state info dict. We have no strong reason to pull this out right now.
    Instead this class handles the mapping.

    Note that new StateElement that don't need to map to the muses-py probably
    shouldn't use this class, there is no reason to store this information in
    a separate data structure."""

    def __init__(
        self, state_info: StateInfoOld, name: StateElementIdentifier, step: str
    ):
        super().__init__(state_info, name, step)
        omiInfo = mpy.ObjectView(self.state_info.state_info_obj.current["omi"])
        self.omi_key = mpy.get_omi_key(omiInfo, str(self.name))

    @property
    def value(self):
        return np.array(
            [self.state_info.state_info_dict["current"]["omi"][self.omi_key]]
        )

    @value.setter
    def value(self, v):
        self.state_info.state_info_dict["current"]["omi"][self.omi_key] = v[0]

    @property
    def apriori_value(self):
        return np.array(
            [self.state_info.state_info_dict["constraint"]["omi"][self.omi_key]]
        )

    def update_state(
        self,
        current: np.ndarray | None = None,
        apriori: np.ndarray | None = None,
        initial: np.ndarray | None = None,
        initial_initial: np.ndarray | None = None,
        true: np.ndarray | None = None,
    ) -> None:
        """We have a few places where we want to update a state element other than
        update_initial_guess. This function updates each of the various values passed in.
        A value of 'None' (the default) means skip updating that part of the state."""
        if current is not None:
            self.state_info.state_info_dict["current"]["omi"][self.omi_key] = current[0]
        if apriori is not None:
            if apriori.shape[0] != 1:
                raise RuntimeError("Needs to be a scalar")
            self.state_info.state_info_dict["constraint"]["omi"][self.omi_key] = (
                apriori[0]
            )
        if initial is not None or initial_initial is not None or true is not None:
            raise NotImplementedError

    @apriori_value.setter
    def apriori_value(self, v):
        self.state_info.state_info_dict["constraint"]["omi"][self.omi_key] = v[0]

    def update_state_element(
        self,
        retrieval_info: RetrievalInfo,
        results_list: np.array,
        retrieval_config: RetrievalConfiguration | MeasurementId,
        step: int,
        do_update_fm: np.array,
    ):
        # Note we assume here that all the mappings are linear. I'm pretty
        # sure that is the case, we can put the extra logic in if needed.
        self.value = results_list[
            np.array(retrieval_info.species_list) == str(self.name)
        ]
        ij = retrieval_info.species_names.index(str(self.name))
        ind1 = retrieval_info.retrieval_info_obj.parameterStartFM[ij]
        ind2 = retrieval_info.retrieval_info_obj.parameterEndFM[ij]
        do_update_fm[ind1 : ind2 + 1] = 1

    def update_initial_guess(self, current_strategy_step: CurrentStrategyStep):
        self.mapType = "linear"
        self.pressureList = np.array(
            [
                -2,
            ]
        )
        self.altitudeList = np.array(
            [
                -2,
            ]
        )
        self.pressureListFM = np.array(
            [
                -2,
            ]
        )
        self.altitudeListFM = np.array(
            [
                -2,
            ]
        )
        # Apriori
        self.initialGuessList = self.value
        self.initialGuessListFM = self.initialGuessList
        self.constraintVector = np.array(
            [self.state_info.state_info_dict["constraint"]["omi"][self.omi_key]]
        )
        self.constraintVectorFM = self.constraintVector
        if self.state_info.has_true_values():
            self.trueParameterList = np.array(
                [self.state_info.state_info_dict["true"]["omi"][self.omi_key]]
            )
            self.trueParameterListFM = self.trueParameterList
        else:
            self.trueParameterList = np.array([0.0])
            self.trueParameterListFM = self.trueParameterList

        self.minimum = np.array([-999.0])
        self.maximum = np.array([-999.0])
        self.maximum_change = np.array([-999.0])
        self.mapToState = np.eye(1)
        self.mapToParameters = np.eye(1)
        # Not sure if the is covariance, or sqrt covariance
        # if str(self.name) == "OMIODWAVUV1":
        #    breakpoint()
        sfile = self.species_information_file(current_strategy_step.retrieval_type)
        sSubaDiagonalValues = float(sfile.sSubaDiagonalValues)
        self.constraintMatrix = np.diag(
            [1 / (sSubaDiagonalValues * sSubaDiagonalValues)]
        )


class MusesPyOmiStateElementHandleOld(StateElementHandleOld):
    def state_element_object(
        self, state_info: StateInfoOld, name: StateElementIdentifier
    ) -> tuple[
        bool,
        tuple[StateElementOld, StateElementOld, StateElementOld, StateElementOld]
        | None,
    ]:
        if str(name) not in mpy.ordered_species_list() or not str(name).startswith(
            "OMI"
        ):
            return (False, None)
        return (
            True,
            (
                MusesPyOmiStateElementOld(state_info, name, "initialInitial"),
                MusesPyOmiStateElementOld(state_info, name, "initial"),
                MusesPyOmiStateElementOld(state_info, name, "current"),
                MusesPyOmiStateElementOld(state_info, name, "true"),
            ),
        )


class MusesPyTropomiStateElementOld(MusesPyStateElementOld):
    """MUSES-py groups all the OMI state elements together. While we could pull
    this apart, the create_uip depends on finding the OMI stuff in a "omi" key
    in the state info dict. We have no strong reason to pull this out right now.
    Instead this class handles the mapping.

    Note that new StateElement that don't need to map to the muses-py probably
    shouldn't use this class, there is no reason to store this information in
    a separate data structure."""

    def __init__(
        self, state_info: StateInfoOld, name: StateElementIdentifier, step: str
    ):
        super().__init__(state_info, name, step)
        tropomiInfo = mpy.ObjectView(self.state_info.state_info_obj.current["tropomi"])
        self.tropomi_key = mpy.get_tropomi_key(tropomiInfo, str(self.name))

    @property
    def value(self):
        return np.array(
            [self.state_info.state_info_dict["current"]["tropomi"][self.tropomi_key]]
        )

    @value.setter
    def value(self, v):
        self.state_info.state_info_dict["current"]["tropomi"][self.tropomi_key] = v[0]

    @property
    def apriori_value(self):
        return np.array(
            [self.state_info.state_info_dict["constraint"]["tropomi"][self.tropomi_key]]
        )

    @apriori_value.setter
    def apriori_value(self, v):
        self.state_info.state_info_dict["constraint"]["tropomi"][self.tropomi_key] = v[
            0
        ]

    def update_state(
        self,
        current: np.ndarray | None = None,
        apriori: np.ndarray | None = None,
        initial: np.ndarray | None = None,
        initial_initial: np.ndarray | None = None,
        true: np.ndarray | None = None,
    ) -> None:
        """We have a few places where we want to update a state element other than
        update_initial_guess. This function updates each of the various values passed in.
        A value of 'None' (the default) means skip updating that part of the state."""
        if current is not None:
            self.state_info.state_info_dict["current"]["tropomi"][self.tropomi_key] = (
                current[0]
            )
        if apriori is not None:
            if apriori.shape[0] != 1:
                raise RuntimeError("Needs to be a scalar")
            self.state_info.state_info_dict["constraint"]["tropomi"][
                self.tropomi_key
            ] = apriori[0]
        if initial is not None or initial_initial is not None or true is not None:
            raise NotImplementedError

    def update_state_element(
        self,
        retrieval_info: RetrievalInfo,
        results_list: np.array,
        retrieval_config: RetrievalConfiguration | MeasurementId,
        step: int,
        do_update_fm: np.array,
    ):
        # Note we assume here that all the mappings are linear. I'm pretty
        # sure that is the case, we can put the extra logic in if needed.
        self.value = results_list[
            np.array(retrieval_info.species_list) == str(self.name)
        ]
        ij = retrieval_info.species_names.index(str(self.name))
        ind1 = retrieval_info.retrieval_info_obj.parameterStartFM[ij]
        ind2 = retrieval_info.retrieval_info_obj.parameterEndFM[ij]
        do_update_fm[ind1 : ind2 + 1] = 1

    def update_initial_guess(self, current_strategy_step: CurrentStrategyStep):
        self.mapType = "linear"
        self.pressureList = np.array(
            [
                -2,
            ]
        )
        self.altitudeList = np.array(
            [
                -2,
            ]
        )
        self.pressureListFM = np.array(
            [
                -2,
            ]
        )
        self.altitudeListFM = np.array(
            [
                -2,
            ]
        )
        # Apriori
        self.initialGuessList = self.value
        self.initialGuessListFM = self.initialGuessList
        self.constraintVector = np.array(
            [self.state_info.state_info_dict["constraint"]["tropomi"][self.tropomi_key]]
        )
        self.constraintVectorFM = self.constraintVector
        if self.state_info.has_true_values():
            self.trueParameterList = np.array(
                [self.state_info.state_info_dict["true"]["tropomi"][self.tropomi_key]]
            )
            self.trueParameterListFM = self.trueParameterList
        else:
            self.trueParameterList = np.array([0.0])
            self.trueParameterListFM = self.trueParameterList

        self.minimum = np.array([-999.0])
        self.maximum = np.array([-999.0])
        self.maximum_change = np.array([-999.0])
        self.mapToState = np.eye(1)
        self.mapToParameters = np.eye(1)
        # Not sure if the is covariance, or sqrt covariance
        sfile = self.species_information_file(current_strategy_step.retrieval_type)
        sSubaDiagonalValues = float(sfile.sSubaDiagonalValues)
        self.constraintMatrix = np.diag(
            [1 / (sSubaDiagonalValues * sSubaDiagonalValues)]
        )


class MusesPyTropomiStateElementHandleOld(StateElementHandleOld):
    def state_element_object(
        self, state_info: StateInfoOld, name: StateElementIdentifier
    ) -> tuple[
        bool,
        tuple[StateElementOld, StateElementOld, StateElementOld, StateElementOld]
        | None,
    ]:
        if str(name) not in mpy.ordered_species_list() or not str(name).startswith(
            "TROPOMI"
        ):
            return (False, None)
        return (
            True,
            (
                MusesPyTropomiStateElementOld(state_info, name, "initialInitial"),
                MusesPyTropomiStateElementOld(state_info, name, "initial"),
                MusesPyTropomiStateElementOld(state_info, name, "current"),
                MusesPyTropomiStateElementOld(state_info, name, "true"),
            ),
        )


class StateElementOnLevelsOld(MusesPyStateElementOld):
    """These are things that are reported on our pressure levels."""

    def __init__(
        self, state_info: StateInfoOld, name: StateElementIdentifier, step: str
    ):
        super().__init__(state_info, name, step)
        self._ind = self.state_info.state_element_on_levels.index(str(name))

    def update_state(
        self,
        current: np.ndarray | None = None,
        apriori: np.ndarray | None = None,
        initial: np.ndarray | None = None,
        initial_initial: np.ndarray | None = None,
        true: np.ndarray | None = None,
    ):
        """We have a few places where we want to update a state element other than
        update_initial_guess. This function updates each of the various values passed in.
        A value of 'None' (the default) means skip updating that part of the state."""
        for v, stp in (
            (current, "current"),
            (apriori, "constraint"),
            (initial, "initial"),
            (initial_initial, "initialInitial"),
            (true, "true"),
        ):
            if v is not None:
                self.state_info.state_info_dict[self.step]["values"][self._ind, :] = v

    @property
    def value(self):
        return self.state_info.state_info_dict[self.step]["values"][self._ind, :]

    @property
    def apriori_value(self):
        return self.state_info.state_info_dict["constraint"]["values"][self._ind, :]


class StateElementOnLevelsHandleOld(StateElementHandleOld):
    def state_element_object(
        self, state_info: StateInfoOld, name: StateElementIdentifier
    ) -> tuple[
        bool,
        tuple[StateElementOld, StateElementOld, StateElementOld, StateElementOld]
        | None,
    ]:
        if str(name) not in state_info.state_element_on_levels:
            return (False, None)
        return (
            True,
            (
                StateElementOnLevelsOld(state_info, name, "initialInitial"),
                StateElementOnLevelsOld(state_info, name, "initial"),
                StateElementOnLevelsOld(state_info, name, "current"),
                StateElementOnLevelsOld(state_info, name, "true"),
            ),
        )


class StateElementInDictOld(MusesPyStateElementOld):
    def __init__(
        self, state_info: StateInfoOld, name: StateElementIdentifier, step: str
    ):
        super().__init__(state_info, name, step)

    def update_state(
        self,
        current: np.ndarray | None = None,
        apriori: np.ndarray | None = None,
        initial: np.ndarray | None = None,
        initial_initial: np.ndarray | None = None,
        true: np.ndarray | None = None,
    ):
        """We have a few places where we want to update a state element other than
        update_initial_guess. This function updates each of the various values passed in.
        A value of 'None' (the default) means skip updating that part of the state."""
        # Check to see if the data should be scalar
        vcheck = self.state_info.state_info_dict[self.step][str(self.name)]
        is_scalar = isinstance(vcheck, numbers.Number)
        for v, stp in (
            (current, "current"),
            (apriori, "constraint"),
            (initial, "initial"),
            (initial_initial, "initialInitial"),
            (true, "true"),
        ):
            if v is not None:
                if is_scalar and v.shape[0] != 1:
                    raise "Value set should be a scalar"
                self.state_info.state_info_dict[stp][str(self.name)] = (
                    v[0] if is_scalar else v
                )

    @property
    def value(self):
        v = self.state_info.state_info_dict[self.step][str(self.name)]
        # So we don't need special cases, always have a numpy array. A
        # single value is an array with one value.
        if isinstance(v, numbers.Number):
            return np.array(
                [
                    v,
                ]
            )
        return v

    @property
    def apriori_value(self):
        v = self.state_info.state_info_dict["constraint"][str(self.name)]
        # So we don't need special cases, always have a numpy array. A
        # single value is an array with one value.
        if isinstance(v, numbers.Number):
            return np.array(
                [
                    v,
                ]
            )
        return v


class StateElementInTopDictOld(MusesPyStateElementOld):
    def __init__(
        self, state_info: StateInfoOld, name: StateElementIdentifier, step: str
    ):
        super().__init__(state_info, name, step)

    @property
    def value_str(self) -> str:
        v = self.state_info.state_info_dict[str(self.name)]
        return str(v)

    @property
    def value(self):
        v = self.state_info.state_info_dict[str(self.name)]
        # So we don't need special cases, always have a numpy array. A
        # single value is an array with one value.
        if isinstance(v, numbers.Number):
            return np.array(
                [
                    v,
                ]
            )
        return v

    @property
    def apriori_value(self):
        v = self.state_info.state_info_dict[str(self.name)]
        # So we don't need special cases, always have a numpy array. A
        # single value is an array with one value.
        if isinstance(v, numbers.Number):
            return np.array(
                [
                    v,
                ]
            )
        return v


class StateElementInDictHandleOld(StateElementHandleOld):
    def state_element_object(
        self, state_info: StateInfoOld, name: StateElementIdentifier
    ) -> tuple[
        bool,
        tuple[StateElementOld, StateElementOld, StateElementOld, StateElementOld]
        | None,
    ]:
        if str(name) not in state_info.state_info_dict["current"]:
            return (False, None)
        return (
            True,
            (
                StateElementInDictOld(state_info, name, "initialInitial"),
                StateElementInDictOld(state_info, name, "initial"),
                StateElementInDictOld(state_info, name, "current"),
                StateElementInDictOld(state_info, name, "true"),
            ),
        )


class StateElementInTopDictHandleOld(StateElementHandleOld):
    def state_element_object(
        self, state_info: StateInfoOld, name: StateElementIdentifier
    ) -> tuple[
        bool,
        tuple[StateElementOld, StateElementOld, StateElementOld, StateElementOld]
        | None,
    ]:
        if str(name) not in state_info.state_info_dict:
            return (False, None)
        return (
            True,
            (
                StateElementInTopDictOld(state_info, name, "initialInitial"),
                StateElementInTopDictOld(state_info, name, "initial"),
                StateElementInTopDictOld(state_info, name, "current"),
                StateElementInTopDictOld(state_info, name, "true"),
            ),
        )


class StateElementWithFrequencyOld(MusesPyStateElementOld):
    """Some of the species also have frequencies associated with them.
    We return these as Refractor SpectralDomain objects.

    TODO I'm pretty sure these are in nm, but this would be worth verifying."""

    def __init__(
        self, state_info: StateInfoOld, name: StateElementIdentifier, step: str
    ):
        super().__init__(state_info, name, step)

    @property
    def spectral_domain(self) -> rf.SpectralDomain:
        raise NotImplementedError


class PtgAngState(MusesPyStateElementOld):
    def __init__(self, state_info, step):
        from refractor.muses import StateElementIdentifier

        super().__init__(state_info, StateElementIdentifier("PTGANG"), step)

    @property
    def value(self):
        # Probably to support old IDL, the arrays are larger than the actual data.
        # We need to subset to get the actual data.
        return np.array(
            [
                self.state_info.state_info_dict[self.step]["tes"][
                    "boresightNadirRadians"
                ],
            ]
        )

    @property
    def apriori_value(self):
        # Probably to support old IDL, the arrays are larger than the actual data.
        # We need to subset to get the actual data.
        return np.array(
            [
                self.state_info.state_info_dict["constraint"]["tes"][
                    "boresightNadirRadians"
                ],
            ]
        )


class EmissivityStateOld(StateElementWithFrequencyOld):
    def __init__(self, state_info, step):
        from refractor.muses import StateElementIdentifier

        super().__init__(state_info, StateElementIdentifier("emissivity"), step)

    @property
    def spectral_domain(self):
        # Probably to support old IDL, the arrays are larger than the actual data.
        # We need to subset to get the actual data.
        r = range(0, self.state_info.state_info_dict["emisPars"]["num_frequencies"])
        return rf.SpectralDomain(
            self.state_info.state_info_dict["emisPars"]["frequency"][r], rf.Unit("nm")
        )

    @property
    def value(self):
        # Probably to support old IDL, the arrays are larger than the actual data.
        # We need to subset to get the actual data.
        r = range(0, self.state_info.state_info_dict["emisPars"]["num_frequencies"])
        return self.state_info.state_info_dict[self.step]["emissivity"][r]

    @property
    def apriori_value(self):
        # Probably to support old IDL, the arrays are larger than the actual data.
        # We need to subset to get the actual data.
        r = range(0, self.state_info.state_info_dict["emisPars"]["num_frequencies"])
        return self.state_info.state_info_dict["constraint"]["emissivity"][r]

    @property
    def camel_distance(self):
        # Not sure what this is, but seems worth keeping
        return self.state_info.state_info_dict["emisPars"]["camel_distance"]

    @property
    def prior_source(self):
        """Source of prior."""
        return self.state_info.state_info_dict["emisPars"]["emissivity_prior_source"]


class NativeEmissivityStateOld(StateElementWithFrequencyOld):
    def __init__(self, state_info, step):
        from refractor.muses import StateElementIdentifier

        super().__init__(state_info, StateElementIdentifier("native_emissivity"), step)

    @property
    def spectral_domain(self):
        return rf.SpectralDomain(
            self.state_info.state_info_dict[self.step]["native_emis_wavenumber"],
            rf.Unit("nm"),
        )

    @property
    def value(self):
        return self.state_info.state_info_dict[self.step]["native_emissivity"]


class CloudStateOld(StateElementWithFrequencyOld):
    def __init__(self, state_info, step):
        from refractor.muses import StateElementIdentifier

        super().__init__(state_info, StateElementIdentifier("cloudEffExt"), step)
        self.step = step

    @property
    def spectral_domain(self):
        # Probably to support old IDL, the arrays are larger than the actual data.
        # We need to subset to get the actual data.
        r = range(0, self.state_info.state_info_dict["cloudPars"]["num_frequencies"])
        return rf.SpectralDomain(
            self.state_info.state_info_dict["cloudPars"]["frequency"][r], rf.Unit("nm")
        )

    @property
    def value(self):
        # Probably to support old IDL, the arrays are larger than the actual data.
        # We need to subset to get the actual data.
        r = range(0, self.state_info.state_info_dict["cloudPars"]["num_frequencies"])
        return self.state_info.state_info_dict[self.step]["cloudEffExt"][:, r]

    def update_state(
        self,
        current: np.ndarray | None = None,
        apriori: np.ndarray | None = None,
        initial: np.ndarray | None = None,
        initial_initial: np.ndarray | None = None,
        true: np.ndarray | None = None,
    ):
        """We have a few places where we want to update a state element other than
        update_initial_guess. This function updates each of the various values passed in.
        A value of 'None' (the default) means skip updating that part of the state."""
        for v, stp in (
            (current, "current"),
            (apriori, "constraint"),
            (initial, "initial"),
            (initial_initial, "initialInitial"),
            (true, "true"),
        ):
            if v is not None:
                self.state_info.state_info_dict[stp]["cloudEffExt"] = v


class CalibrationState(StateElementWithFrequencyOld):
    def __init__(self, state_info, step):
        from refractor.muses import StateElementIdentifier

        super().__init__(state_info, StateElementIdentifier("calibrationScale"), step)
        self.step = step

    @property
    def spectral_domain(self):
        # Probably to support old IDL, the arrays are larger than the actual data.
        # We need to subset to get the actual data.
        r = range(
            0, self.state_info.state_info_dict["calibrationPars"]["num_frequencies"]
        )
        return rf.SpectralDomain(
            self.state_info.state_info_dict["calibrationPars"]["frequency"][r],
            rf.Unit("nm"),
        )

    @property
    def value(self):
        # Probably to support old IDL, the arrays are larger than the actual data.
        # We need to subset to get the actual data.
        r = range(
            0, self.state_info.state_info_dict["calibrationPars"]["num_frequencies"]
        )
        return self.state_info.state_info_dict[self.step]["calibrationScale"][r]


class SingleSpeciesHandleOld(StateElementHandleOld):
    def __init__(
        self,
        specname: str,
        state_element_class,
        pass_state=True,
        **kwargs,
    ):
        from refractor.muses import StateElementIdentifier

        self.name = StateElementIdentifier(specname)
        self.state_element_class = state_element_class
        self.pass_state = pass_state
        self.kwargs = kwargs

    def state_element_object(
        self, state_info: StateInfoOld, name: StateElementIdentifier
    ) -> tuple[
        bool,
        tuple[StateElementOld, StateElementOld, StateElementOld, StateElementOld]
        | None,
    ]:
        if name != self.name:
            return (False, None)
        if self.pass_state:
            return (
                True,
                (
                    self.state_element_class(
                        state_info, "initialInitial", **self.kwargs
                    ),
                    self.state_element_class(state_info, "initial", **self.kwargs),
                    self.state_element_class(state_info, "current", **self.kwargs),
                    self.state_element_class(state_info, "true", **self.kwargs),
                ),
            )
        else:
            return (
                True,
                (
                    self.state_element_class(state_info, **self.kwargs),
                    self.state_element_class(state_info, **self.kwargs),
                    self.state_element_class(state_info, **self.kwargs),
                    self.state_element_class(state_info, **self.kwargs),
                ),
            )


StateElementHandleSetOld.add_default_handle(
    SingleSpeciesHandleOld("emissivity", EmissivityStateOld),
    priority_order=2,
)
StateElementHandleSetOld.add_default_handle(
    SingleSpeciesHandleOld("native_emissivity", NativeEmissivityStateOld),
    priority_order=2,
)
StateElementHandleSetOld.add_default_handle(
    SingleSpeciesHandleOld("cloudEffExt", CloudStateOld),
    priority_order=2,
)
StateElementHandleSetOld.add_default_handle(
    SingleSpeciesHandleOld("calibrationScale", CalibrationState),
    priority_order=2,
)
StateElementHandleSetOld.add_default_handle(
    SingleSpeciesHandleOld("PTGANG", PtgAngState),
    priority_order=2,
)
# We have some things that are *both* in the top dictionary and in the state dictionary
# (example I know of is surfaceType). I *think* we want the value from the top dict, so
# I've given this a higher priority
StateElementHandleSetOld.add_default_handle(
    StateElementInTopDictHandleOld(), priority_order=1
)
StateElementHandleSetOld.add_default_handle(StateElementInDictHandleOld())
StateElementHandleSetOld.add_default_handle(StateElementOnLevelsHandleOld())
StateElementHandleSetOld.add_default_handle(
    MusesPyOmiStateElementHandleOld(), priority_order=-1
)
StateElementHandleSetOld.add_default_handle(
    MusesPyTropomiStateElementHandleOld(), priority_order=-1
)
# If nothing else handles a state_element, fall back to the muses-py code.
StateElementHandleSetOld.add_default_handle(
    MusesPyStateElementHandleOld(), priority_order=-2
)


__all__ = [
    "MusesPyStateElementOld",
    "MusesPyOmiStateElementOld",
    "StateElementOnLevelsOld",
    "StateElementOnLevelsHandleOld",
    "StateElementInDictOld",
    "StateElementInDictHandleOld",
    "StateElementWithFrequencyOld",
    "EmissivityStateOld",
    "CloudStateOld",
    "SingleSpeciesHandleOld",
]

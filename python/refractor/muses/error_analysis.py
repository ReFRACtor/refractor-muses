from __future__ import annotations
import refractor.muses.muses_py as mpy  # type: ignore
from .fake_state_info import FakeStateInfo
import copy
import numpy as np
import sys
import os
from pprint import pprint
from scipy.linalg import block_diag  # type: ignore
import typing

if typing.TYPE_CHECKING:
    from .current_state import CurrentState
    from .muses_strategy import CurrentStrategyStep
    from .retrieval_result import RetrievalResult
    from .retrieval_info import RetrievalInfo
    from .identifier import StateElementIdentifier


class ErrorAnalysis:
    """This just groups together some of the error analysis stuff
    together, to put this together. Nothing more than a shuffling
    around of stuff already in muses-py

    """

    def __init__(
        self,
        current_state: CurrentState,
        current_strategy_step: CurrentStrategyStep,
        covariance_state_element_name: list[StateElementIdentifier],
    ) -> None:
        self.error_initial = self.initialize_error_initial(
            current_state, current_strategy_step, covariance_state_element_name
        )
        self.error_current: dict | mpy.ObjectView = copy.deepcopy(self.error_initial)
        # Code seems to assume these are object view.
        self.error_initial = mpy.ObjectView(self.error_initial)
        self.error_current = mpy.ObjectView(self.error_current)

    def snapshot_to_file(self, fname: str | os.PathLike[str]) -> None:
        """py-retrieve is big on having functions with unintended side effects. It can
        be hard to determine what changes when. This writes a complete text dump of
        this object, which we can then diff against other snapshots to see what has
        changed."""
        with np.printoptions(precision=None, threshold=sys.maxsize):
            with open(fname, "w") as fh:
                pprint(
                    {
                        "error_initial": self.error_initial.__dict__,
                        "error_current": self.error_current.__dict__,
                    },
                    stream=fh,
                )

    def initialize_error_initial(
        self,
        current_state: CurrentState,
        current_strategy_step: CurrentStrategyStep,
        covariance_state_element_name: list[StateElementIdentifier],
    ) -> dict | mpy.ObjectView:
        """covariance_state_element_name should be the list of state
        elements we need covariance from. This is all the elements we
        will retrieve, plus any interferents that get added in. This
        list is unique elements, sorted by the order_species sorting

        """
        selem_list = []
        for sname in covariance_state_element_name:
            selem = current_state.full_state_element_old(sname)
            # Note clear why, but we get slightly different results if we
            # update the original state_info. May want to track this down,
            # but as a work around we just copy this. This is just needed
            # to get the mapping type, I don't think anything else is
            # needed. We should be able to pull that out from the full
            # initial guess update at some point, so we don't need to do
            # the full initial guess
            if hasattr(selem, "update_initial_guess"):
                selem = copy.deepcopy(selem)
                selem.update_initial_guess(current_strategy_step)
                selem_list.append(selem)

        # Note the odd seeming "capitalize" here. This is because
        # get_prior_error uses the map type to look up files, and
        # rather than "linear" or "log" it needs "Linear" or "Log"

        pressure_list = []
        species_list = []
        map_list = []

        # Make block diagonal covariance.
        matrix_list = []
        # AT_LINE 25 get_prior_error.pro
        for selem in selem_list:
            matrix, pressureSa = selem.sa_covariance()
            pressure_list.extend(pressureSa)
            species_list.extend([selem.name] * matrix.shape[0])
            matrix_list.append(matrix)
            map_list.extend([selem.map_type] * matrix.shape[0])

        initial = block_diag(*matrix_list)
        # Off diagonal blocks for covariance.
        for i, selem1 in enumerate(selem_list):
            for selem2 in selem_list[i + 1 :]:
                matrix = selem1.sa_cross_covariance(selem2)
                if matrix is not None:
                    initial[np.array(species_list) == selem1.name, :][
                        :, np.array(species_list) == selem2.name
                    ] = matrix
                    initial[np.array(species_list) == selem2.name, :][
                        :, np.array(species_list) == selem1.name
                    ] = np.transpose(matrix)
        return mpy.constraint_data(
            initial, pressure_list, [str(i) for i in species_list], map_list
        )

    def update_retrieval_result(self, retrieval_result: RetrievalResult) -> None:
        """Update the retrieval_result and ErrorAnalysis. The retrieval_result
        are updated in place."""
        # Both these functions update retrieval_result in place, and
        # also returned. We don't need the return value, it is just the
        # same as retrieval_result
        fstate_info = FakeStateInfo(retrieval_result.current_state)
        _ = self.error_analysis(
            retrieval_result.rstep.__dict__,
            fstate_info,
            retrieval_result.current_state.retrieval_info,
            retrieval_result,
        )
        _ = mpy.write_retrieval_summary(
            None,
            retrieval_result.current_state.retrieval_info.retrieval_info_obj,
            fstate_info,
            None,
            retrieval_result,
            {},
            None,
            None,
            None,
            self.error_current,
            writeOutputFlag=False,
            errorInitial=self.error_initial,
        )

    def error_analysis(
        self,
        radiance_step: dict,
        fstate_info: FakeStateInfo,
        retrieval_info: RetrievalInfo,
        retrieval_result: RetrievalResult,
    ) -> mpy.ObjectView:
        """Update results and error_current"""
        # Doesn't seem to be used for anything, but we need to pass in. I think
        # this might have been something that was used in the past?
        radiance_noise = {"radiance": np.zeros_like(radiance_step["radiance"])}
        (results, self.error_current) = mpy.error_analysis_wrapper(
            None,
            None,
            radiance_step,
            radiance_noise,
            retrieval_info.retrieval_info_obj,
            fstate_info,
            self.error_initial,
            self.error_current,
            None,
            retrieval_result,
        )
        return results


__all__ = [
    "ErrorAnalysis",
]

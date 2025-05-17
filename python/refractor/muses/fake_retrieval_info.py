from __future__ import annotations
from . import muses_py as mpy  # type: ignore
import refractor.framework as rf  # type: ignore
import numpy as np
import typing

if typing.TYPE_CHECKING:
    from .current_state import CurrentState
    from .identifier import StateElementIdentifier


class FakeRetrievalInfo:
    """We are moving away from the RetrievalInfo structure used by
    muses-py. There is a lot of hidden functionality in this structure, and
    we want to move this into StateInfo and CurrentState. However there
    are a handful of muses-py functions that we want to call which is
    tightly coupled to the old RetrievalInfo structure. The content of this
    is pretty much what is in CurrentState, but in a format that is
    substantially different. This class produces a "fake" RetrievalInfo that
    reformats the CurrentState data. The purpose of this
    is just to call the old code, an alternative might be to replace
    this old code. But for now, we are massaging our data to call the
    old code.
    """

    def __init__(
        self, current_state: CurrentState, use_state_mapping: bool = False
    ) -> None:
        self.current_state = current_state
        # TODO
        # We are trying to remove the use the map_type in the UIP and ErrorAnalysis, as
        # well has the basis matrix and replace these with StateMapping. This flag allows
        # us to do this with one but not the other. This is a work in progress.
        self.use_state_mapping = use_state_mapping

    @property
    def retrieval_info_systematic(self) -> mpy.ObjectView:
        """Version of retrieval info to use for a creating a systematic UIP. Note that all
        this information is actually available in just the current FakeRetrievalInfo, but
        this reformats things so the systematic forward model looks just like the
        forward model. Could do the same thing with some flag handling in the UIP code, but
        this is the way py-retrieve currently has things set up."""
        return mpy.ObjectView(
            {
                "parameterStartFM": self.parameterStartSys,
                "parameterEndFM": self.parameterEndSys,
                "species": self.speciesSys,
                "n_species": self.n_totalParametersSys,
                "speciesList": self.speciesListSys,
                "speciesListFM": self.speciesListSys,
                "mapTypeListFM": self.map_type_systematic,
                "initialGuessListFM": np.zeros(
                    shape=(self.n_totalParametersSys,), dtype=np.float32
                ),
                "constraintVectorListFM": np.zeros(
                    shape=(self.n_totalParametersSys,), dtype=np.float32
                ),
                "initialGuessList": np.zeros(
                    shape=(self.n_totalParametersSys,), dtype=np.float32
                ),
                "n_totalParametersFM": self.n_totalParametersSys,
            }
        )

    @property
    def speciesListFM(self) -> list[str]:
        return [
            str(i) for i in self.current_state.forward_model_state_vector_element_list
        ]

    @property
    def species(self) -> list[str]:
        return [str(i) for i in self.current_state.retrieval_state_element_id]

    @property
    def n_species(self) -> int:
        return len(self.current_state.retrieval_state_element_id)

    @property
    def n_totalParameters(self) -> int:
        return len(self.current_state.retrieval_state_vector_element_list)

    @property
    def n_totalParametersFM(self) -> int:
        return len(self.current_state.forward_model_state_vector_element_list)

    @property
    def speciesList(self) -> list[str]:
        return [str(i) for i in self.current_state.retrieval_state_vector_element_list]

    @property
    def n_speciesSys(self) -> int:
        return len(self.current_state.systematic_state_element_id)

    @property
    def mapTypeListFM(self) -> list[str | rf.StateMapping]:
        res = [
            "",
        ] * self.current_state.fm_state_vector_size
        for sid, (pstart, plen) in self.current_state.fm_sv_loc.items():
            res[pstart : (pstart + plen)] = [
                self._map_type(sid),
            ] * plen
        return res

    @property
    def parameterStartSys(self) -> list[int]:
        return [
            self.current_state.sys_sv_loc[sid][0]
            for sid in self.current_state.systematic_state_element_id
        ]

    @property
    def parameterEndSys(self) -> list[int]:
        return [
            self.current_state.sys_sv_loc[sid][0]
            + self.current_state.sys_sv_loc[sid][1]
            - 1
            for sid in self.current_state.systematic_state_element_id
        ]

    @property
    def speciesSys(self) -> list[str]:
        return [str(i) for i in self.current_state.systematic_state_element_id]

    @property
    def speciesListSys(self) -> list[str]:
        return [
            str(i)
            for i in self.current_state.systematic_model_state_vector_element_list
        ]

    @property
    def Constraint(self) -> np.ndarray:
        return self.current_state.constraint_matrix

    @property
    def doUpdateFM(self) -> np.ndarray:
        return self.current_state.updated_fm_flag

    def _map_type(self, sid: StateElementIdentifier) -> str | rf.StateMapping:
        smap = self.current_state.state_mapping(sid)
        if self.use_state_mapping:
            return smap
        if isinstance(smap, rf.StateMappingLinear):
            return "linear"
        elif isinstance(smap, rf.StateMappingLog):
            return "log"
        raise RuntimeError(f"Don't recognize state mapping {smap}")

    @property
    def map_type_systematic(self) -> list[str | rf.StateMapping]:
        res = [
            "",
        ] * self.current_state.sys_state_vector_size
        for sid, (pstart, plen) in self.current_state.sys_sv_loc.items():
            res[pstart : (pstart + plen)] = [
                self._map_type(sid),
            ] * plen
        return res

    @property
    def n_totalParametersSys(self) -> int:
        return len(self.current_state.systematic_model_state_vector_element_list)

    @property
    def mapType(self) -> list[str | rf.StateMapping]:
        return [
            self._map_type(sid) for sid in self.current_state.retrieval_state_element_id
        ]

    @property
    def parameterStart(self) -> list[int]:
        return [
            self.current_state.retrieval_sv_loc[sid][0]
            for sid in self.current_state.retrieval_state_element_id
        ]

    @property
    def parameterEnd(self) -> list[int]:
        return [
            self.current_state.retrieval_sv_loc[sid][0]
            + self.current_state.retrieval_sv_loc[sid][1]
            - 1
            for sid in self.current_state.retrieval_state_element_id
        ]

    @property
    def parameterStartFM(self) -> list[int]:
        return [
            self.current_state.fm_sv_loc[sid][0]
            for sid in self.current_state.retrieval_state_element_id
        ]

    @property
    def parameterEndFM(self) -> list[int]:
        return [
            self.current_state.fm_sv_loc[sid][0]
            + self.current_state.fm_sv_loc[sid][1]
            - 1
            for sid in self.current_state.retrieval_state_element_id
        ]

    @property
    def n_parametersFM(self) -> list[int]:
        return [
            self.current_state.fm_sv_loc[sid][1]
            for sid in self.current_state.retrieval_state_element_id
        ]

    @property
    def n_parameters(self) -> list[int]:
        return [
            self.current_state.retrieval_sv_loc[sid][1]
            for sid in self.current_state.retrieval_state_element_id
        ]

    @property
    def initialGuessList(self) -> np.ndarray:
        return self.current_state.initial_guess

    @property
    def initialGuessListFM(self) -> np.ndarray:
        return self.current_state.initial_guess_fm

    @property
    def trueParameterList(self) -> np.ndarray:
        return self.current_state.true_value

    @property
    def trueParameterListFM(self) -> np.ndarray:
        return self.current_state.true_value_fm

    @property
    def constraintVector(self) -> np.ndarray:
        return self.current_state.constraint_vector

    @property
    def constraintVectorListFM(self) -> np.ndarray:
        return self.current_state.constraint_vector_fm

    @property
    def mapToParameters(self) -> np.ndarray | None:
        return self.current_state.map_to_parameter_matrix

    @property
    def mapToState(self) -> np.ndarray | None:
        return self.current_state.basis_matrix

    @property
    def altitudeListFM(self) -> np.ndarray:
        pdata : list[np.ndarray] = []
        # Convention of muses-py is to use [-2] for items that aren't on
        # pressure levels
        for sid in self.current_state.retrieval_state_element_id:
            d = self.current_state.altitude_list_fm(sid)
            if d is not None:
                pdata.append(d)
            else:
                pdata.append(np.array([-2.0]))
        return np.concatenate(pdata)

    @property
    def altitudeList(self) -> np.ndarray:
        pdata : list[np.ndarray] = []
        # Convention of muses-py is to use [-2] for items that aren't on
        # pressure levels
        for sid in self.current_state.retrieval_state_element_id:
            d = self.current_state.altitude_list(sid)
            if d is not None:
                pdata.append(d)
            else:
                pdata.append(np.array([-2.0]))
        return np.concatenate(pdata)

    @property
    def pressureListFM(self) -> np.ndarray:
        pdata : list[np.ndarray] = []
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
    def pressureList(self) -> np.ndarray:
        pdata : list[np.ndarray] = []
        # Convention of muses-py is to use [-2] for items that aren't on
        # pressure levels
        for sid in self.current_state.retrieval_state_element_id:
            d = self.current_state.pressure_list(sid)
            if d is not None:
                pdata.append(d)
            else:
                pdata.append(np.array([-2.0]))
        return np.concatenate(pdata)


__all__ = [
    "FakeRetrievalInfo",
]

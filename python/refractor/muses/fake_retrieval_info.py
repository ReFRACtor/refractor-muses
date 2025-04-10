from __future__ import annotations
from . import muses_py as mpy  # type: ignore
import numpy as np
import typing

if typing.TYPE_CHECKING:
    from .current_state import CurrentState


class FakeRetrievalInfo:
    """We are moving away from the RetrievalInfo structure used by
    muses-py. There is a lot of hidden functionality in this structure, and
    we want to move this into StateInfo and CurrentState. However there
    are a handful of muses-[y functions that we want to call which is
    tightly coupled to the old RetrievalInfo structure. The content of this
    is pretty much what is in CurrentState, but in a format that is
    substantially different. This class produces a "fake" RetrievalInfo that
    reformats the CurrentState data. The purpose of this
    is just to call the old code, an alternative might be to replace
    this old code. But for now, we are massaging our data to call the
    old code.
    """

    def __init__(self, current_state: CurrentState) -> None:
        self.current_state = current_state

    @property
    def retrieval_info_systematic(self) -> mpy.ObjectView:
        """Version of retrieval info to use for a creating a systematic UIP"""
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
        return self.current_state.retrieval_info.retrieval_info_obj.n_totalParameters

    @property
    def n_totalParametersFM(self) -> int:
        return self.current_state.retrieval_info.retrieval_info_obj.n_totalParametersFM
    
    @property
    def speciesList(self) -> list[str]:
        return self.current_state.retrieval_info.retrieval_info_obj.speciesList

    @property
    def speciesListFM(self) -> list[str]:
        return self.current_state.retrieval_info.retrieval_info_obj.speciesListFM

    @property
    def mapTypeListFM(self) -> list[str]:
        return self.current_state.retrieval_info.retrieval_info_obj.mapTypeListFM

    @property
    def parameterStartSys(self) -> list[str]:
        return self.current_state.retrieval_info.retrieval_info_obj.parameterStartSys

    @property
    def parameterEndSys(self) -> list[str]:
        return self.current_state.retrieval_info.retrieval_info_obj.parameterEndSys
    

    @property
    def speciesSys(self) -> list[str]:
        return self.current_state.retrieval_info.retrieval_info_obj.speciesSys

    @property
    def speciesListSys(self) -> list[str]:
        return self.current_state.retrieval_info.retrieval_info_obj.speciesListSys

    @property
    def map_type_systematic(self) -> list[str]:
        return self.current_state.retrieval_info.map_type_systematic
    
    @property
    def n_totalParametersSys(self) -> int:
        return self.current_state.retrieval_info.retrieval_info_obj.n_totalParametersSys
    
    @property
    def mapType(self) -> list[str]:
        return [
            self.current_state.map_type(selem)
            for selem in self.current_state.retrieval_state_element_id
        ]

    @property
    def parameterStart(self) -> list[int]:
        return [
            self.current_state.retrieval_sv_loc[selem][0]
            for selem in self.current_state.retrieval_state_element_id
        ]

    @property
    def parameterEnd(self) -> list[int]:
        return [
            self.current_state.retrieval_sv_loc[selem][0]
            + self.current_state.retrieval_sv_loc[selem][1]
            - 1
            for selem in self.current_state.retrieval_state_element_id
        ]

    @property
    def parameterStartFM(self) -> list[int]:
        return [
            self.current_state.fm_sv_loc[selem][0]
            for selem in self.current_state.retrieval_state_element_id
        ]

    @property
    def parameterEndFM(self) -> list[int]:
        return [
            self.current_state.fm_sv_loc[selem][0]
            + self.current_state.fm_sv_loc[selem][1]
            - 1
            for selem in self.current_state.retrieval_state_element_id
        ]

    @property
    def n_parametersFM(self) -> list[int]:
        return [
            self.current_state.fm_sv_loc[selem][1]
            for selem in self.current_state.retrieval_state_element_id
        ]

    @property
    def n_parameters(self) -> list[int]:
        return [
            self.current_state.retrieval_sv_loc[selem][1]
            for selem in self.current_state.retrieval_state_element_id
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
        return self.current_state.apriori

    @property
    def constraintVectorListFM(self) -> np.ndarray:
        return self.current_state.apriori_fm

    @property
    def mapToParameters(self) -> np.ndarray | None:
        return self.current_state.map_to_parameter_matrix

    @property
    def mapToState(self) -> np.ndarray | None:
        return self.current_state.basis_matrix

    @property
    def altitudeListFM(self) -> np.ndarray:
        return np.concatenate(
            [
                self.current_state.altitude_list_fm(selem)
                for selem in self.current_state.retrieval_state_element_id
            ]
        )

    @property
    def altitudeList(self) -> np.ndarray:
        return np.concatenate(
            [
                self.current_state.altitude_list(selem)
                for selem in self.current_state.retrieval_state_element_id
            ]
        )

    @property
    def pressureListFM(self) -> np.ndarray:
        return np.concatenate(
            [
                self.current_state.pressure_list_fm(selem)
                for selem in self.current_state.retrieval_state_element_id
            ]
        )

    @property
    def pressureList(self) -> np.ndarray:
        return np.concatenate(
            [
                self.current_state.pressure_list(selem)
                for selem in self.current_state.retrieval_state_element_id
            ]
        )


__all__ = [
    "FakeRetrievalInfo",
]

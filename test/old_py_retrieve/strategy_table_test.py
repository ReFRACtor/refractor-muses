from refractor.muses import (
    MusesRunDir,
    CurrentStrategyStep,
    RetrievalConfiguration,
)
from refractor.old_py_retrieve_wrapper import (
    StrategyTable,
    RetrievableStateElementOld,
    StateInfoOld,
    RetrievalInfo,
)
import subprocess
import numpy as np
import pytest

# Add a extra state element, just so we can make sure our StrategyTable functions
# handles this correctly


class EofStateElement(RetrievableStateElementOld):
    def __init__(self, state_info: StateInfoOld, name="OMIEOF1"):
        super().__init__(state_info, name)
        self._value = None

    @property
    def value(self):
        return self._value

    @property
    def apriori_value(self) -> np.ndarray:
        return np.array(
            [
                0.0,
            ]
        )

    def should_write_to_l2_product(self, instruments):
        if "OMI" in instruments:
            return True
        return False

    def net_cdf_variable_name(self):
        return "OMI_EOF_1"

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
        state_info: StateInfoOld,
        retrieval_info: RetrievalInfo,
        results_list: np.ndarray,
        retrieval_config: RetrievalConfiguration,
        step: int,
        do_update_fm: np.ndarray,
    ):
        self._value = results_list[np.array(retrieval_info.species_list) == self._name]

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
        self.constraintVector = np.array(
            [
                0.0,
            ]
        )
        # Normally the same as apriori, but doesn't have to be
        self.initialGuessList = np.array(
            [
                0.0,
            ]
        )
        self.trueParameterList = np.array(
            [
                0.0,
            ]
        )
        self.constraintVectorFM = np.array(
            [
                0.0,
            ]
        )
        self.initialGuessListFM = np.array(
            [
                0.0,
            ]
        )
        self.trueParameterListFM = np.array(
            [
                0.0,
            ]
        )
        self.minimum = np.array(
            [
                -999,
            ]
        )
        self.maximum = np.array(
            [
                -999,
            ]
        )
        self.maximum_change = np.array(
            [
                -999,
            ]
        )
        self.mapToState = 1
        self.mapToParameters = 1
        # Not sure if the is covariance, or sqrt covariance
        self.constraintMatrix = 10 * 10


@pytest.mark.old_py_retrieve_test
def test_strategy_table(isolated_dir, osp_dir, gmao_dir, omi_test_in_dir):
    r = MusesRunDir(omi_test_in_dir, osp_dir, gmao_dir)
    # Modify the Table.asc to add a EOF element. This is just a short cut,
    # so we don't need to make a new strategy table. Eventually a new table
    # will be needed in the OSP directory, but it is too early for that.
    subprocess.run(
        f'sed -i -e "s/OMINRADWAVUV1/OMINRADWAVUV1,OMIEOF1/" {str(r.run_dir / "Table.asc")}',
        shell=True,
    )
    s = StrategyTable(r.run_dir / "Table.asc")
    assert s.retrieval_elements_all_step == [
        "O3",
        "OMICLOUDFRACTION",
        "OMISURFACEALBEDOUV1",
        "OMISURFACEALBEDOUV2",
        "OMISURFACEALBEDOSLOPEUV2",
        "OMINRADWAVUV1",
        "OMINRADWAVUV2",
        "OMIODWAVUV1",
        "OMIODWAVUV2",
        "OMIEOF1",
    ]

    s.table_step = 0
    print(s.spectral_filename)
    assert (
        s.spectral_filename.name
        == "Windows_Nadir_OMICLOUDFRACTION_OMICLOUD_IG_Refine.asc"
    )
    assert s.table_step == 0
    assert s.number_table_step == 2
    assert s.step_name == "OMICLOUDFRACTION"
    assert s.output_directory.name == "20160414_23_394_11_23"

    s.table_step = 1
    assert s.spectral_filename.name == "Windows_Nadir_O3.asc"
    assert s.table_step == 1
    assert s.number_table_step == 2
    assert s.step_name == "O3_OMI"
    assert s.output_directory.name == "20160414_23_394_11_23"

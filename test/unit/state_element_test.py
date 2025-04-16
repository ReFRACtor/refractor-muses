from __future__ import annotations
from refractor.muses import (
    RetrievalStrategy,
    MusesRunDir,
    ProcessLocation,
    StateElementIdentifier,
)
from fixtures.misc_fixture import all_output_disabled
from typing import Any


class RetrievalStrategyStop:
    def notify_update(
        self,
        retrieval_strategy: RetrievalStrategy,
        location: ProcessLocation,
        **kwargs: Any,
    ) -> None:
        if location == ProcessLocation("notify_new_step done"):
            raise StopIteration()


def test_omi_state_element(
    isolated_dir, osp_dir, gmao_dir, vlidort_cli, joint_omi_test_in_dir
):
    try:
        with all_output_disabled():
            r = MusesRunDir(joint_omi_test_in_dir, osp_dir, gmao_dir, path_prefix=".")
            rs = RetrievalStrategy(r.run_dir / "Table.asc")
            rs.clear_observers()
            rs.add_observer(RetrievalStrategyStop())
            rs.retrieval_ms()
    except StopIteration:
        pass
    cstate = rs.current_state
    print([str(i) for i in cstate.full_state_element_id])
    selem = cstate.full_state_element(StateElementIdentifier("OMIODWAVUV1"))
    # Doesn't work yet
    # selem.assert_equal(selem)

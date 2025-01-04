import os
import pytest
from refractor.muses import MusesRunDir, RetrievalStrategy

# Fixtures that set up a full RetrievalStrategy at a given retrieval step, for use
# in testing that is hard to do outside of a full retrieval


def load_step(rs, step_number, dir, include_ret_state=False):
    """Load in the state information and optional retrieval results for the given
    step, and jump to that step."""
    rs.load_state_info(
        f"{dir}/state_info_step_{step_number}.pkl",
        step_number,
        ret_state_file=f"{dir}/retrieval_state_step_{step_number}.json.gz"
        if include_ret_state
        else None,
    )


def set_up_run_to_location(
    dir, step_number, location, osp_dir, gmao_dir, vlidort_cli, include_ret_state=True
):
    """Set up directory and run the given step number to the given location."""
    r = MusesRunDir(dir, osp_dir, gmao_dir)
    rs = RetrievalStrategy(f"{r.run_dir}/Table.asc", vlidort_cli=vlidort_cli)
    rstep, kwargs = run_step_to_location(
        rs, step_number, dir, location, include_ret_state=include_ret_state
    )
    return rs, rstep, kwargs


def run_step_to_location(rs, step_number, dir, location, include_ret_state=True):
    """Load in the given step, and run up to the location we notify at
    (e.g., "retrieval step"). Return the retrieval_strategy_step"""

    class CaptureRs:
        def __init__(self):
            self.retrieval_strategy_step = None

        def notify_update(
            self, retrieval_strategy, loc, retrieval_strategy_step=None, **kwargs
        ):
            if loc != location:
                return
            self.retrieval_strategy_step = retrieval_strategy_step
            self.kwargs = kwargs
            raise StopIteration()

    try:
        rcap = CaptureRs()
        rs.add_observer(rcap)
        load_step(rs, step_number, dir, include_ret_state=include_ret_state)
        try:
            rs.continue_retrieval(stop_after_step=step_number)
        except StopIteration:
            return rcap.retrieval_strategy_step, rcap.kwargs
    finally:
        rs.remove_observer(rcap)


@pytest.fixture(scope="function")
def joint_omi_step_8(
    isolated_dir, joint_omi_test_in_dir, osp_dir, gmao_dir, vlidort_cli
):
    rs, rstep, kwargs = set_up_run_to_location(
        joint_omi_test_in_dir, 8, "retrieval input", osp_dir, gmao_dir, vlidort_cli
    )
    os.chdir(rs.run_dir)
    return rs, rstep, kwargs


@pytest.fixture(scope="function")
def joint_tropomi_step_12(
    isolated_dir,
    joint_tropomi_test_in_dir,
    osp_dir,
    gmao_dir,
    vlidort_cli,
):
    rs, rstep, kwargs = set_up_run_to_location(
        joint_tropomi_test_in_dir, 12, "retrieval input", osp_dir, gmao_dir, vlidort_cli
    )
    os.chdir(rs.run_dir)
    return rs, rstep, kwargs

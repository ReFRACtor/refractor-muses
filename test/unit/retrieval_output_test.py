from refractor.muses import (
    RetrievalJacobianOutput,
    RetrievalL2Output,
    RetrievalRadianceOutput,
    RetrievalIrkOutput,
    RetrievalPickleResult,
    RetrievalPlotRadiance,
    RetrievalPlotResult,
    RetrievalInputOutput,
    ProcessLocation,
)
from fixtures.compare_run import compare_run
import pytest
from pathlib import Path


@pytest.fixture(scope="function")
def joint_tropomi_output(
    joint_tropomi_step_12_output, joint_tropomi_test_refractor_expected_dir
):
    """Common part of out output tests."""
    rs, rstep, kwargs = joint_tropomi_step_12_output
    # Update new state elements with old saved data. This is temporary, until we
    # start saving the new state elements. But short term so we can continue testing
    # just blunt force update the data
    rs.current_state._state_info.update_with_old()
    yield rs, rstep, kwargs
    compare_dir = joint_tropomi_test_refractor_expected_dir
    diff_is_error = True
    # Skip, not fully working. We need to update the captured data, but wait until
    # we finish making changes to the StateInfo stuff
    #compare_run(compare_dir, rs.run_dir.parent, diff_is_error=diff_is_error, from_run_dir=True)


def test_retrieval_radiance_output(joint_tropomi_output):
    rs, rstep, kwargs = joint_tropomi_output
    jout = RetrievalRadianceOutput()
    jout.notify_update(
        rs, ProcessLocation("retrieval step"), retrieval_strategy_step=rstep, **kwargs
    )


def test_retrieval_jacobian_output(joint_tropomi_output):
    rs, rstep, kwargs = joint_tropomi_output
    jout = RetrievalJacobianOutput()
    jout.notify_update(
        rs, ProcessLocation("retrieval step"), retrieval_strategy_step=rstep, **kwargs
    )


def test_retrieval_l2_output(joint_tropomi_output):
    rs, rstep, kwargs = joint_tropomi_output
    jout = RetrievalL2Output()
    jout.notify_update(
        rs, ProcessLocation("retrieval step"), retrieval_strategy_step=rstep, **kwargs
    )
    jout.finalize_file_number()


def test_retrieval_pickle_results(joint_tropomi_output):
    rs, rstep, kwargs = joint_tropomi_output
    jout = RetrievalPickleResult()
    jout.notify_update(
        rs, ProcessLocation("retrieval step"), retrieval_strategy_step=rstep, **kwargs
    )
    # We just check that output exists, no easy way to check that it is the same.
    # Since this is diagnostic output we really don't need this the same
    assert (
        Path(rs.run_dir) / "Step12_H2O,O3,EMIS_TROPOMI/Diagnostics/results.pkl"
    ).exists()


def test_retrieval_plot_radiance(joint_tropomi_output):
    rs, rstep, kwargs = joint_tropomi_output
    jout = RetrievalPlotRadiance()
    jout.notify_update(
        rs, ProcessLocation("retrieval step"), retrieval_strategy_step=rstep, **kwargs
    )
    # We just check that output exists, no easy way to check that it is the same.
    # Since this is diagnostic output we really don't need this the same
    for fname in (
        "radiance_fit_diff.png",
        "radiance_fit_diff_vs_radiance.png",
        "radiance_fit_initial_diff.png",
        "radiance_fit_initial_diff_vs_radiance.png",
        "radiance_fit_initial.png",
        "radiance_fit.png",
    ):
        assert (
            Path(rs.run_dir) / f"Step12_H2O,O3,EMIS_TROPOMI/StepAnalysis/{fname}"
        ).exists()


# Doesn't work, too tightly coupled to StrategyTable. We can perhaps get this
# working, but this is a diagnostic anyways so not overly important
@pytest.mark.skip
def test_retrieval_input_output(joint_tropomi_output):
    rs, rstep, kwargs = joint_tropomi_output
    jout = RetrievalInputOutput()
    jout.notify_update(
        rs, ProcessLocation("retrieval step"), retrieval_strategy_step=rstep, **kwargs
    )
    breakpoint()


def test_retrieval_plot_results(joint_tropomi_output):
    rs, rstep, kwargs = joint_tropomi_output
    jout = RetrievalPlotResult()
    jout.notify_update(
        rs, ProcessLocation("retrieval step"), retrieval_strategy_step=rstep, **kwargs
    )
    # We just check that output exists, no easy way to check that it is the same.
    # Since this is diagnostic output we really don't need this the same
    for fname in ("ak_full.png", "plot_H2O.png", "plot_O3.png"):
        assert (Path(rs.run_dir) / f"Step12_H2O,O3,EMIS_TROPOMI/{fname}").exists()


def test_retrieval_irk_output(airs_irk_step_6, airs_irk_test_expected_dir):
    rs, rstep, kwargs = airs_irk_step_6
    jout = RetrievalIrkOutput()
    jout.notify_update(
        rs, ProcessLocation("IRK step"), retrieval_strategy_step=rstep, **kwargs
    )
    compare_dir = airs_irk_test_expected_dir
    diff_is_error = True
    #compare_run(compare_dir, rs.run_dir.parent, diff_is_error=diff_is_error, from_run_dir=True)

import refractor.muses.muses_py as mpy  # type: ignore
from refractor.muses import CloudResultSummary
from pytest import approx

def test_cloud_result_summary(joint_tropomi_step_12_output):
    rs, rstp, _ = joint_tropomi_step_12_output
    # We can perhaps pull back from needing the full results here, but we are
    # still cleaning up error handling needed before this
    result = rstp.results
    error_analysis = rs.error_analysis
    csum = CloudResultSummary(result, error_analysis)
    #breakpoint()

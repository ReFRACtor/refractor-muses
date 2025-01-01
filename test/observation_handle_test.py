from test_support import *
from refractor.muses import (
    RetrievalStrategy,
    MusesRunDir,
    mpy_radiance_from_observation_list,
)


class RetrievalStrategyStop:
    def notify_update(self, retrieval_strategy, location, **kwargs):
        if location == "initial set up done":
            raise StopIteration()


def test_radiance(isolated_dir, osp_dir, gmao_dir, vlidort_cli):
    # TODO - We should have a constructor for StateInfo. Don't currently,
    # so we just run RetrievalStrategy to the beginning and stop
    try:
        with all_output_disabled():
            r = MusesRunDir(
                joint_omi_test_in_dir,
                # r = MusesRunDir(joint_tropomi_test_in_dir,
                osp_dir,
                gmao_dir,
                path_prefix=".",
            )
            rs = RetrievalStrategy(f"{r.run_dir}/Table.asc")
            rs.clear_observers()
            rs.add_observer(RetrievalStrategyStop())
            rs.retrieval_ms()
    except StopIteration:
        pass
    oset = rs.observation_handle_set
    olist = [oset.observation(iname, None, None, None) for iname in ["AIRS", "OMI"]]
    rad = mpy_radiance_from_observation_list(olist, full_band=True)
    print(rad)

from test_support import *
from refractor.muses import (
    RetrievalConfiguration,
)


def test_retrieval_configuration(osp_dir):
    rconf = RetrievalConfiguration.create_from_strategy_file(
        f"{test_base_path}/omi/in/sounding_1/Table.asc", osp_dir=osp_dir
    )
    print(dict(rconf))

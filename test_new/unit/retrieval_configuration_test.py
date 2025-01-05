from refractor.muses import RetrievalConfiguration


def test_retrieval_configuration(osp_dir, omi_test_in_dir):
    rconf = RetrievalConfiguration.create_from_strategy_file(
        omi_test_in_dir / "Table.asc", osp_dir=osp_dir
    )
    print(dict(rconf))

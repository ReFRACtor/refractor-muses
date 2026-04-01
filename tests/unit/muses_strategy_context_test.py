from refractor.muses import MusesStrategyContext


def test_muses_strategy_context(joint_tropomi_test_in_dir):
    strategy_context = MusesStrategyContext(
        strategy_table_filename=joint_tropomi_test_in_dir / "Table.asc"
    )
    print(strategy_context.measurement_id)
    print(strategy_context.retrieval_config)
    print(strategy_context.strategy)

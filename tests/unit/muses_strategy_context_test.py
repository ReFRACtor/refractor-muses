from refractor.muses import MusesStrategyContext


def test_muses_strategy_context(joint_tropomi_test_in_dir, ifile_hlp):
    strategy_context = MusesStrategyContext(
        strategy_directory=joint_tropomi_test_in_dir,
        ifile_hlp=ifile_hlp,
    )
    print(strategy_context.measurement_id)
    print(strategy_context.retrieval_config)
    print(strategy_context.strategy)
    sproxy = MusesStrategyContext()
    print(sproxy.measurement_id)
    print(sproxy.retrieval_config)
    print(sproxy.strategy)
    sproxy.merge(strategy_context)
    print(sproxy.measurement_id)
    print(sproxy.retrieval_config)
    print(sproxy.strategy)

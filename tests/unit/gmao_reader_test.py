from refractor.muses import GmaoReader, InputFileHelper


def test_read_gmao(gmao_dir):
    res = GmaoReader.read_gmao(gmao_dir, 2004, 9, 20, 12)
    res2 = GmaoReader.read_gmao(gmao_dir, 2016, 4, 2, 9)
    assert res is not None
    assert res2 is not None


def test_gmao(airs_omi_shandle, gmao_dir):
    # We don't need most of airs_omi_shandle, but at the same time
    # it isn't worth making a new fixture. So just use this here
    # for testing, even if we ignore most of it
    _, _, _, _, smeta, _ = airs_omi_shandle
    res = GmaoReader(smeta, gmao_dir, InputFileHelper())
    assert res is not None

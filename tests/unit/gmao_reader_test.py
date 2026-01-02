from refractor.muses import GmaoReader


def test_read_gmao(ifile_hlp):
    res = GmaoReader.read_gmao(ifile_hlp, 2004, 9, 20, 12)
    res2 = GmaoReader.read_gmao(ifile_hlp, 2016, 4, 2, 9)
    assert res is not None
    assert res2 is not None


def test_gmao(airs_omi_shandle, ifile_hlp):
    # We don't need most of airs_omi_shandle, but at the same time
    # it isn't worth making a new fixture. So just use this here
    # for testing, even if we ignore most of it
    _, _, _, _, smeta, _ = airs_omi_shandle
    res = GmaoReader(smeta, ifile_hlp)
    assert res is not None

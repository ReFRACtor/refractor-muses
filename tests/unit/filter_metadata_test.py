from refractor.muses import FileFilterMetadata


def test_file_filter_metadata(ifile_hlp):
    default_fname = (
        ifile_hlp.osp_dir
        / "Strategy_Tables/ops/Defaults/Default_Spectral_Windows_Definition_File_Filters_CrIS_TROPOMI.asc"
    )
    fmeta = FileFilterMetadata(default_fname, ifile_hlp)
    assert fmeta.filter_metadata(None) == {}
    assert fmeta.filter_metadata("FOO") == {}
    assert fmeta.filter_metadata("1A1") == {
        "monoextend": 0.48,
        "monoSpacing": 0.0008,
        "speciesList": "H2O,HDO,CO2,O3,N2O,CO,CH4,NO,NH3,OCS,HCN,C2H2,HCOOH,PAN,ISOP",
        "maxopd": 7.5,
        "spacing": 0.06,
        "num_points": 0,
    }

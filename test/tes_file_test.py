from test_support import *
from refractor.muses import (TesFile,)

def test_tes_file(osp_dir):
    fname = f"{test_base_path}/omi/in/sounding_1/Table.asc"
    tfile = TesFile.create(fname)
    # Check that second call returns same object
    tfile_second = TesFile.create(fname)
    tfile_third = TesFile.create(fname)
    assert tfile is tfile_second
    assert tfile is tfile_third
    # Verify against the mpy version
    tfile2 = TesFile(fname, use_mpy=True)
    assert(dict(tfile) == dict(tfile2))
    assert tfile.table.equals(tfile2.table)

    # Try another file that has no table
    fname = f"{osp_dir}/Strategy_Tables/ops/OSP-OMI-v2/Species-66/OMICLOUDFRACTION.asc"
    tfile = TesFile.create(fname)
    tfile2 = TesFile(fname, use_mpy=True)
    assert(dict(tfile) == dict(tfile2))
    assert(tfile.table is None)
    assert(tfile2.table is None)

    # Check comment handling for end of line
    fname = f"{osp_dir}/Strategy_Tables/ops/OSP-OMI-v2/Cloud/CloudParameters.asc"
    tfile = TesFile.create(fname)
    tfile2 = TesFile(fname, use_mpy=True)
    assert(dict(tfile) == dict(tfile2))
    assert tfile.table.equals(tfile2.table)

    # Check comment handling for whole line
    fname = f"{osp_dir}/Strategy_Tables/ops/OSP-OMI-v2/Species-66/CO2_stepco2_1.asc"
    tfile = TesFile.create(fname)
    tfile2 = TesFile(fname, use_mpy=True)
    assert(dict(tfile) == dict(tfile2))
    assert(tfile.table is None)
    assert(tfile2.table is None)
    
    # Test handling of extra stuff at the end of the file
    fname = f"{osp_dir}/Strategy_Tables/ops/Defaults/Default_Spectral_Windows_Definition_File_Filters_CrIS_TROPOMI.asc"
    tfile = TesFile.create(fname)
    tfile2 = TesFile(fname, use_mpy=True)
    assert(dict(tfile) == dict(tfile2))
    assert tfile.table.equals(tfile2.table)

    # Another example with stuff as the end of the file
    fname = f"{osp_dir}/Strategy_Tables/ops/OSP-OMI-AIRS-v10/MWDefinitions/Windows_Nadir_TATM_H2O_N2O_CH4_HDO_BAR_LAND.asc"
    tfile = TesFile.create(fname)
    tfile2 = TesFile(fname, use_mpy=True)
    assert(dict(tfile) == dict(tfile2))
    assert tfile.table.equals(tfile2.table)
    

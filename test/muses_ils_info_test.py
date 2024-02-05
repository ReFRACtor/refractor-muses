from refractor.muses import (MusesIlsInfo, MusesRunDir, StrategyTable)
from refractor.omi import (OmiFmObjectCreator)
from test_support import *

def test_muses_ils_tropomi(isolated_dir, osp_dir, gmao_dir):
    r = MusesRunDir(tropomi_test_in_dir, osp_dir, gmao_dir)
    for ils_method in ('FASTCONV', 'POSTCONV', 'APPLY'):
        ils_info = MusesIlsInfo(ils_method, "TROPOMI", osp_dir=osp_dir,
                                xtrack_index=3)
        print(f"Checking {ils_method}")
        assert ils_info.ils_method == ils_method
    ils_info = MusesIlsInfo("POSTCONV", "TROPOMI", osp_dir=osp_dir,
                            xtrack_index=3)
    for band_index in range(1,8+1):
        print(ils_info.tropomi_ils_postconv(band_index))


def test_muses_ils_omi(isolated_dir, osp_dir, gmao_dir):
    r = MusesRunDir(omi_test_in_dir, osp_dir, gmao_dir)
    for ils_method in ('FASTCONV', 'POSTCONV', 'APPLY'):
        ils_info = MusesIlsInfo(ils_method, "OMI", osp_dir=osp_dir,
                                atrack_index=3)
        print(f"Checking {ils_method}")
        assert ils_info.ils_method == ils_method
    ils_info = MusesIlsInfo("POSTCONV", "OMI", osp_dir=osp_dir,
                            atrack_index=3)
    print(ils_info.omi_ils_postconv("UV1"))

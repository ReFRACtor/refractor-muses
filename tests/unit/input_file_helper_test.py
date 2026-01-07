from __future__ import annotations
from refractor.muses import InputFileHelper, InputFileLogging


def test_ifile(osp_dir, gmao_dir):
    ifile_hlp = InputFileHelper(osp_dir, gmao_dir)
    ifile_hlp.add_observer(InputFileLogging())
    # Try opening each kind of file, and making sure we get logging
    _ = ifile_hlp.open_ncdf(
        ifile_hlp.osp_dir / "Lite" / "pan_mask-margin2.-cutoff0.004.nc"
    )
    _ = ifile_hlp.open_tes(ifile_hlp.osp_dir / "Lite" / "TES_baseline_66.asc")
    _ = ifile_hlp.open_h5(
        ifile_hlp.osp_dir / "Climatology" / "Climatology_files" / "climatology_PAN.nc"
    )


def test_ifpath(osp_dir, gmao_dir):
    ifile_hlp = InputFileHelper(osp_dir, gmao_dir)
    f = ifile_hlp.osp_dir / "Lite" / "TES_baseline_66.asc"
    assert f == f
    print(hash(f))
    assert f.exists()
    assert f.parent == (ifile_hlp.osp_dir / "Lite")
    assert f.name == "TES_baseline_66.asc"
    assert f.absolute() == f
    assert f.resolve() == f
    print(str(f))
    print(f.as_posix())
    assert (
        f.sub_fname(r"66", "33") == ifile_hlp.osp_dir / "Lite" / "TES_baseline_33.asc"
    )
    # add glob

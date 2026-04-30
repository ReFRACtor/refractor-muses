from refractor.muses import (
    MusesSpectralWindow,
    MusesOmiObservation,
    FileFilterMetadata,
    InstrumentIdentifier,
    FilterIdentifier,
    RetrievalType,
    StateElementIdentifier,
)
import refractor.muses_py as mpy
from fixtures.require_check import require_muses_py
from refractor.old_py_retrieve_wrapper import StrategyTable
import numpy as np
import numpy.testing as npt


def struct_compare(s1, s2):
    for k in s1.keys():
        if k == "THROW_AWAY_WINDOW_INDEX":
            continue
        if isinstance(s1[k], np.ndarray) and np.can_cast(s1[k], np.float64):
            npt.assert_allclose(s1[k], s2[k])
        elif isinstance(s1[k], np.ndarray):
            assert np.all(s1[k] == s2[k])
        elif isinstance(s1[k], float) or isinstance(s2[k], float):
            assert float(s1[k]) == float(s2[k])
        else:
            assert s1[k] == s2[k]


def mw_compare(mw1, mw2):
    assert len(mw1) == len(mw2)
    mw1_s = sorted(mw1, key=lambda t: t["start"])
    mw2_s = sorted(mw2, key=lambda t: t["start"])
    for t1, t2 in zip(mw1_s, mw2_s):
        struct_compare(t1, t2)


# We use the old StrategyTable here (since we don't want to use our higher order
# MusesStrategy, we want to test MusesSpectralWindow without that).
@require_muses_py
def test_muses_spectral_window(ifile_hlp, joint_omi_test_in_dir):
    # This is an observation that has some bad samples in it.
    xtrack_uv1 = 10
    xtrack_uv2 = 20
    atrack = 1139
    filename = (
        joint_omi_test_in_dir.parent
        / "OMI-Aura_L1-OML1BRUG_2016m0401t2215-o62308_v003-2016m0402t041806.he4"
    )
    cld_filename = (
        joint_omi_test_in_dir.parent
        / "OMI-Aura_L2-OMCLDO2_2016m0401t2215-o62308_v003-2016m0402t044340.he5"
    )
    utc_time = "2016-04-01T23:07:33.676106Z"
    calibration_filename = ifile_hlp.osp_dir / "OMI/OMI_Rad_Cal/JPL_OMI_RadCaL_2006.h5"
    stable = StrategyTable(joint_omi_test_in_dir / "Table.asc", ifile_hlp=ifile_hlp)
    obs = MusesOmiObservation.create_from_filename(
        filename,
        xtrack_uv1,
        xtrack_uv2,
        atrack,
        utc_time,
        calibration_filename,
        [FilterIdentifier("UV1"), FilterIdentifier("UV2")],
        cld_filename=cld_filename,
        ifile_hlp=ifile_hlp,
    )
    step_number = 3
    # Note this is off by 1. The table numbering get redone after the BT step. It might
    # be nice to straighten this out - this is actually kind of confusing. Might be better to
    # just have a way to skip steps - but this is at least how the code works. The
    # code mpy.modify_from_bt changes the number of steps
    swin = MusesSpectralWindow(
        stable.spectral_window(InstrumentIdentifier("OMI"), stp=step_number + 1), obs
    )
    spec = obs.spectrum_full(1)
    # Check number of good points
    assert swin.apply(spec, 1).spectral_domain.data.shape[0] == 4
    # Include bad points
    swin.include_bad_sample = True
    assert swin.apply(spec, 1).spectral_domain.data.shape[0] == 4 + 3
    swin.include_bad_sample = False
    assert swin.apply(spec, 1).spectral_domain.data.shape[0] == 4
    # Use raman extended
    swin.do_raman_ext = True
    assert swin.apply(spec, 1).spectral_domain.data.shape[0] == 50
    swin.do_raman_ext = False
    assert swin.apply(spec, 1).spectral_domain.data.shape[0] == 4
    # Check number of full band
    swin.full_band = True
    assert (
        swin.apply(spec, 1).spectral_domain.data.shape[0]
        == spec.spectral_domain.data.shape[0]
    )
    swin.full_band = False
    assert swin.apply(spec, 1).spectral_domain.data.shape[0] == 4


def test_muses_spectral_window_microwindows(ifile_hlp):
    """Test creating a spectral window dictionary and then creating the
    microwindows struct. Compare to using the old muses-py code."""
    # This is just a set of microwindows, sufficient for testing all the functionality.
    default_fname = (
        ifile_hlp.osp_dir
        / "Strategy_Tables/ops/Defaults/Default_Spectral_Windows_Definition_File_Filters_CrIS_TROPOMI.asc"
    )
    viewing_mode = "nadir"
    spectral_dir = (
        ifile_hlp.osp_dir / "Strategy_Tables/ops/OSP-CrIS-TROPOMI-v7/MWDefinitions"
    )
    retrieval_elements = [
        StateElementIdentifier("H2O"),
        StateElementIdentifier("O3"),
        StateElementIdentifier("CLOUDEXT"),
        StateElementIdentifier("PCLOUD"),
        StateElementIdentifier("TSUR"),
        StateElementIdentifier("EMIS"),
        StateElementIdentifier("TROPOMIRINGSFBAND3"),
        StateElementIdentifier("TROPOMISOLARSHIFTBAND3"),
        StateElementIdentifier("TROPOMIRADIANCESHIFTBAND3"),
        StateElementIdentifier("TROPOMISURFACEALBEDOBAND3"),
        StateElementIdentifier("TROPOMISURFACEALBEDOSLOPEBAND3"),
        StateElementIdentifier("TROPOMISURFACEALBEDOSLOPEORDER2BAND3"),
    ]
    step_name = "H2O,O3,EMIS_TROPOMI"
    retrieval_type = RetrievalType("joint")
    # This is the file muses-py ends up with. We need to get that functionality in place,
    # but at a higher level. At this level, we just read "a given file"
    spec_fname = (
        ifile_hlp.osp_dir
        / "Strategy_Tables/ops/OSP-CrIS-TROPOMI-v7/MWDefinitions/Windows_Nadir_H2O_O3_joint.asc"
    )
    fmeta = FileFilterMetadata(default_fname, ifile_hlp)
    swin = MusesSpectralWindow.create_from_file(
        spec_fname,
        InstrumentIdentifier("TROPOMI"),
        ifile_hlp,
        filter_metadata=fmeta,
        different_filter_different_sensor_index=True,
    )
    swin_dict = MusesSpectralWindow.create_dict_from_file(
        spec_fname, ifile_hlp, filter_metadata=fmeta
    )
    assert str(spec_fname) == str(
        MusesSpectralWindow.muses_microwindows_fname(
            viewing_mode,
            spectral_dir,
            retrieval_elements,
            step_name,
            retrieval_type,
            spec_file=None,
        )
    )
    # Can only compare if we have muses-py, skip otherwise
    if mpy.have_muses_py:
        mw = MusesSpectralWindow.muses_microwindows_from_muses_py(
            default_fname,
            viewing_mode,
            spectral_dir,
            retrieval_elements,
            step_name,
            retrieval_type,
            spec_file=None,
        )
        mw2 = MusesSpectralWindow.muses_microwindows_from_dict(swin_dict)
        mw_compare(mw, mw2)
    # check calculation of muses_monochromatic
    mono_list, mono_filter_list, mono_list_length = swin.muses_monochromatic()
    assert len(mono_list) == (337 - 323) * 100
    assert len(mono_filter_list) == (337 - 323) * 100


def test_microwindows_fname(ifile_hlp):
    """Compare our name against the old mpy code."""
    viewing_mode = "nadir"
    spectral_dir = (
        ifile_hlp.osp_dir / "Strategy_Tables/ops/OSP-CrIS-TROPOMI-v7/MWDefinitions"
    )
    retrieval_elements = [
        StateElementIdentifier("H2O"),
        StateElementIdentifier("O3"),
        StateElementIdentifier("CLOUDEXT"),
        StateElementIdentifier("PCLOUD"),
        StateElementIdentifier("TSUR"),
        StateElementIdentifier("EMIS"),
        StateElementIdentifier("TROPOMIRINGSFBAND3"),
        StateElementIdentifier("TROPOMISOLARSHIFTBAND3"),
        StateElementIdentifier("TROPOMIRADIANCESHIFTBAND3"),
        StateElementIdentifier("TROPOMISURFACEALBEDOBAND3"),
        StateElementIdentifier("TROPOMISURFACEALBEDOSLOPEBAND3"),
        StateElementIdentifier("TROPOMISURFACEALBEDOSLOPEORDER2BAND3"),
    ]
    step_name = "H2O,O3,EMIS_TROPOMI"
    retrieval_type = RetrievalType("joint")
    if mpy.have_muses_py:
        spec_fname = MusesSpectralWindow.muses_microwindows_fname_from_muses_py(
            viewing_mode,
            spectral_dir,
            retrieval_elements,
            step_name,
            retrieval_type,
            spec_file=None,
        )
    else:
        spec_fname = str(spectral_dir / "Windows_Nadir_H2O_O3_joint.asc")
    spec_fname2 = MusesSpectralWindow.muses_microwindows_fname(
        viewing_mode,
        spectral_dir,
        retrieval_elements,
        step_name,
        retrieval_type,
        spec_file=None,
    )
    print(spec_fname)
    assert spec_fname == str(spec_fname2)

    viewing_mode = "limb"
    if mpy.have_muses_py:
        spec_fname = MusesSpectralWindow.muses_microwindows_fname_from_muses_py(
            viewing_mode,
            spectral_dir,
            retrieval_elements,
            step_name,
            retrieval_type,
            spec_file=None,
        )
    else:
        spec_fname = str(spectral_dir / "Windows_Limb_H2O_O3_joint.asc")
    spec_fname2 = MusesSpectralWindow.muses_microwindows_fname(
        viewing_mode,
        spectral_dir,
        retrieval_elements,
        step_name,
        retrieval_type,
        spec_file=None,
    )
    print(spec_fname)
    assert spec_fname == str(spec_fname2)

    viewing_mode = "nadir"
    if mpy.have_muses_py:
        spec_fname = MusesSpectralWindow.muses_microwindows_fname_from_muses_py(
            viewing_mode,
            spectral_dir,
            retrieval_elements,
            step_name,
            retrieval_type,
            spec_file="foo",
        )
    else:
        spec_fname = str(spectral_dir / "Windows_Nadir_foo.asc")
    spec_fname2 = MusesSpectralWindow.muses_microwindows_fname(
        viewing_mode,
        spectral_dir,
        retrieval_elements,
        step_name,
        retrieval_type,
        spec_file="foo",
    )
    print(spec_fname)
    assert spec_fname == str(spec_fname2)

    retrieval_type = RetrievalType("bt")
    if mpy.have_muses_py:
        spec_fname = MusesSpectralWindow.muses_microwindows_fname_from_muses_py(
            viewing_mode,
            spectral_dir,
            retrieval_elements,
            step_name,
            retrieval_type,
            spec_file=None,
        )
    else:
        spec_fname = str(spectral_dir / "Windows_Nadir_H2O,O3,EMIS_TROPOMI_bt.asc")
    spec_fname2 = MusesSpectralWindow.muses_microwindows_fname(
        viewing_mode,
        spectral_dir,
        retrieval_elements,
        step_name,
        retrieval_type,
        spec_file=None,
    )
    print(spec_fname)
    assert spec_fname == str(spec_fname2)

    retrieval_type = RetrievalType("forwardmodel")
    if mpy.have_muses_py:
        spec_fname = MusesSpectralWindow.muses_microwindows_fname_from_muses_py(
            viewing_mode,
            spectral_dir,
            retrieval_elements,
            step_name,
            retrieval_type,
            spec_file=None,
        )
    else:
        spec_fname = str(
            spectral_dir / "Windows_Nadir_H2O,O3,EMIS_TROPOMI_forwardmodel.asc"
        )
    spec_fname2 = MusesSpectralWindow.muses_microwindows_fname(
        viewing_mode,
        spectral_dir,
        retrieval_elements,
        step_name,
        retrieval_type,
        spec_file=None,
    )
    print(spec_fname)
    assert spec_fname == str(spec_fname2)

    retrieval_elements = [
        StateElementIdentifier("CLOUDEXT"),
        StateElementIdentifier("PCLOUD"),
        StateElementIdentifier("TSUR"),
        StateElementIdentifier("TROPOMIRINGSFBAND3"),
        StateElementIdentifier("TROPOMISOLARSHIFTBAND3"),
        StateElementIdentifier("TROPOMIRADIANCESHIFTBAND3"),
        StateElementIdentifier("TROPOMISURFACEALBEDOBAND3"),
        StateElementIdentifier("TROPOMISURFACEALBEDOSLOPEBAND3"),
        StateElementIdentifier("TROPOMISURFACEALBEDOSLOPEORDER2BAND3"),
    ]
    retrieval_type = RetrievalType("joint")
    if mpy.have_muses_py:
        spec_fname = MusesSpectralWindow.muses_microwindows_fname_from_muses_py(
            viewing_mode,
            spectral_dir,
            retrieval_elements,
            step_name,
            retrieval_type,
            spec_file=None,
        )
    else:
        spec_fname = str(
            spectral_dir
            / "Windows_Nadir_TSUR_CLOUDEXT_PCLOUD_TROPOMISURFACEALBEDOBAND3_TROPOMISURFACEALBEDOSLOPEBAND3_TROPOMISURFACEALBEDOSLOPEORDER2BAND3_TROPOMISOLARSHIFTBAND3_TROPOMIRADIANCESHIFTBAND3_TROPOMIRINGSFBAND3_joint.asc"
        )
    spec_fname2 = MusesSpectralWindow.muses_microwindows_fname(
        viewing_mode,
        spectral_dir,
        retrieval_elements,
        step_name,
        retrieval_type,
        spec_file=None,
    )
    print(spec_fname)
    assert spec_fname == str(spec_fname2)

    retrieval_elements = [
        StateElementIdentifier("H2O"),
        StateElementIdentifier("O3"),
        StateElementIdentifier("CLOUDEXT"),
        StateElementIdentifier("PCLOUD"),
        StateElementIdentifier("TSUR"),
        StateElementIdentifier("EMIS"),
        StateElementIdentifier("TROPOMIRINGSFBAND3"),
        StateElementIdentifier("TROPOMISOLARSHIFTBAND3"),
        StateElementIdentifier("TROPOMIRADIANCESHIFTBAND3"),
        StateElementIdentifier("TROPOMISURFACEALBEDOBAND3"),
        StateElementIdentifier("TROPOMISURFACEALBEDOSLOPEBAND3"),
        StateElementIdentifier("TROPOMISURFACEALBEDOSLOPEORDER2BAND3"),
    ]
    step_name = "H2O,O3,EMIS_TROPOMI"
    retrieval_type = RetrievalType("default")
    if mpy.have_muses_py:
        spec_fname = MusesSpectralWindow.muses_microwindows_fname_from_muses_py(
            viewing_mode,
            spectral_dir,
            retrieval_elements,
            step_name,
            retrieval_type,
            spec_file=None,
        )
    else:
        spec_fname = str(spectral_dir / "Windows_Nadir_H2O_O3.asc")
    spec_fname2 = MusesSpectralWindow.muses_microwindows_fname(
        viewing_mode,
        spectral_dir,
        retrieval_elements,
        step_name,
        retrieval_type,
        spec_file=None,
    )
    print(spec_fname)
    assert spec_fname == str(spec_fname2)

    step_name = "H2O,O3,EMIS_TROPOMI"
    retrieval_type = RetrievalType("-")
    if mpy.have_muses_py:
        spec_fname = MusesSpectralWindow.muses_microwindows_fname_from_muses_py(
            viewing_mode,
            spectral_dir,
            retrieval_elements,
            step_name,
            retrieval_type,
            spec_file=None,
        )
    else:
        spec_fname = str(spectral_dir / "Windows_Nadir_H2O_O3.asc")
    spec_fname2 = MusesSpectralWindow.muses_microwindows_fname(
        viewing_mode,
        spectral_dir,
        retrieval_elements,
        step_name,
        retrieval_type,
        spec_file=None,
    )
    print(spec_fname)
    assert spec_fname == str(spec_fname2)

    step_name = "H2O,O3,EMIS_TROPOMI"
    retrieval_type = RetrievalType("fullfilter")
    if mpy.have_muses_py:
        spec_fname = MusesSpectralWindow.muses_microwindows_fname_from_muses_py(
            viewing_mode,
            spectral_dir,
            retrieval_elements,
            step_name,
            retrieval_type,
            spec_file=None,
        )
    else:
        spec_fname = str(spectral_dir / "Windows_Nadir_H2O,O3,EMIS_TROPOMI.asc")
    spec_fname2 = MusesSpectralWindow.muses_microwindows_fname(
        viewing_mode,
        spectral_dir,
        retrieval_elements,
        step_name,
        retrieval_type,
        spec_file=None,
    )
    print(spec_fname)
    assert spec_fname == str(spec_fname2)

    step_name = "H2O,O3,EMIS_TROPOMI"
    retrieval_type = RetrievalType("bt_ig_refine")
    if mpy.have_muses_py:
        spec_fname = MusesSpectralWindow.muses_microwindows_fname_from_muses_py(
            viewing_mode,
            spectral_dir,
            retrieval_elements,
            step_name,
            retrieval_type,
            spec_file=None,
        )
    else:
        spec_fname = str(
            spectral_dir
            / "Windows_Nadir_H2O_O3_TSUR_EMIS_CLOUDEXT_PCLOUD_TROPOMISURFACEALBEDOBAND3_TROPOMISURFACEALBEDOSLOPEBAND3_TROPOMISURFACEALBEDOSLOPEORDER2BAND3_TROPOMISOLARSHIFTBAND3_TROPOMIRADIANCESHIFTBAND3_TROPOMIRINGSFBAND3_BT_IG_Refine.asc"
        )
    spec_fname2 = MusesSpectralWindow.muses_microwindows_fname(
        viewing_mode,
        spectral_dir,
        retrieval_elements,
        step_name,
        retrieval_type,
        spec_file=None,
    )
    print(spec_fname)
    assert spec_fname == str(spec_fname2)

    step_name = "TROPOMIwide"
    retrieval_type = RetrievalType("joint")
    if mpy.have_muses_py:
        spec_fname = MusesSpectralWindow.muses_microwindows_fname_from_muses_py(
            viewing_mode,
            spectral_dir,
            retrieval_elements,
            step_name,
            retrieval_type,
            spec_file=None,
        )
    else:
        spec_fname = str(spectral_dir / "Windows_Nadir_H2O_O3wide_joint.asc")
    spec_fname2 = MusesSpectralWindow.muses_microwindows_fname(
        viewing_mode,
        spectral_dir,
        retrieval_elements,
        step_name,
        retrieval_type,
        spec_file=None,
    )
    print(spec_fname)
    assert spec_fname == str(spec_fname2)

    step_name = "TROPOMIBand_1_2_short"
    retrieval_type = RetrievalType("joint")
    if mpy.have_muses_py:
        spec_fname = MusesSpectralWindow.muses_microwindows_fname_from_muses_py(
            viewing_mode,
            spectral_dir,
            retrieval_elements,
            step_name,
            retrieval_type,
            spec_file=None,
        )
    else:
        spec_fname = str(spectral_dir / "Windows_Nadir_H2O_O3Band_1_2_short_joint.asc")
    spec_fname2 = MusesSpectralWindow.muses_microwindows_fname(
        viewing_mode,
        spectral_dir,
        retrieval_elements,
        step_name,
        retrieval_type,
        spec_file=None,
    )
    print(spec_fname)
    assert spec_fname == str(spec_fname2)

    step_name = "TROPOMIBand_1_2"
    retrieval_type = RetrievalType("joint")
    if mpy.have_muses_py:
        spec_fname = MusesSpectralWindow.muses_microwindows_fname_from_muses_py(
            viewing_mode,
            spectral_dir,
            retrieval_elements,
            step_name,
            retrieval_type,
            spec_file=None,
        )
    else:
        spec_fname = str(spectral_dir / "Windows_Nadir_H2O_O3Band_1_2_joint.asc")
    spec_fname2 = MusesSpectralWindow.muses_microwindows_fname(
        viewing_mode,
        spectral_dir,
        retrieval_elements,
        step_name,
        retrieval_type,
        spec_file=None,
    )
    print(spec_fname)
    assert spec_fname == str(spec_fname2)

    step_name = "TROPOMIBand_2"
    retrieval_type = RetrievalType("joint")
    if mpy.have_muses_py:
        spec_fname = MusesSpectralWindow.muses_microwindows_fname_from_muses_py(
            viewing_mode,
            spectral_dir,
            retrieval_elements,
            step_name,
            retrieval_type,
            spec_file=None,
        )
    else:
        spec_fname = str(spectral_dir / "Windows_Nadir_H2O_O3Band_2_joint.asc")
    spec_fname2 = MusesSpectralWindow.muses_microwindows_fname(
        viewing_mode,
        spectral_dir,
        retrieval_elements,
        step_name,
        retrieval_type,
        spec_file=None,
    )
    print(spec_fname)
    assert spec_fname == str(spec_fname2)

    step_name = "TROPOMIBand_1_2_short"
    retrieval_type = RetrievalType("foo")
    if mpy.have_muses_py:
        spec_fname = MusesSpectralWindow.muses_microwindows_fname_from_muses_py(
            viewing_mode,
            spectral_dir,
            retrieval_elements,
            step_name,
            retrieval_type,
            spec_file=None,
        )
    else:
        spec_fname = str(spectral_dir / "Windows_Nadir_H2O_O3_foo.asc")
    spec_fname2 = MusesSpectralWindow.muses_microwindows_fname(
        viewing_mode,
        spectral_dir,
        retrieval_elements,
        step_name,
        retrieval_type,
        spec_file=None,
    )
    print(spec_fname)
    assert spec_fname == str(spec_fname2)


def test_species_list_all(ifile_hlp):
    swin = MusesSpectralWindow.create_dict_from_file(
        ifile_hlp.osp_dir
        / "Strategy_Tables"
        / "ops"
        / "OSP-CrIS-TROPOMI-v7"
        / "MWDefinitions"
        / "Windows_Nadir_H2O_O3_joint.asc",
        ifile_hlp,
    )
    print(list(str(s) for s in swin[InstrumentIdentifier("CRIS")].species_list_all))
    assert swin[InstrumentIdentifier("CRIS")].species_list_all == [
        StateElementIdentifier(s)
        for s in [
            "H2O",
            "CO2",
            "O3",
            "N2O",
            "CH4",
            "NH3",
            "CFC11",
            "CFC12",
            "HDO",
            "CH3OH",
        ]
    ]

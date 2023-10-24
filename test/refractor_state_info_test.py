from test_support import *
from refractor.muses import RefractorStateInfo, RetrievalStrategy, MusesRunDir

class RetrievalStrategyStop:
    def notify_update(self, retrieval_strategy, location, **kwargs):
        if(location == "initial set up done"):
            raise StopIteration()

@require_muses_py
def test_refractor_state_info(isolated_dir, osp_dir, gmao_dir, vlidort_cli):
    # TODO - We should have a constructor for RefractorStateInfo. Don't currently,
    # so we just run RetrievalStrategy to the beginning and stop
    try:
        with all_output_disabled():
            #r = MusesRunDir(joint_omi_test_in_dir,
            r = MusesRunDir(joint_tropomi_test_in_dir,
                            osp_dir, gmao_dir, path_prefix=".")
            rs = RetrievalStrategy(f"{r.run_dir}/Table.asc")
            rs.clear_observers()
            rs.add_observer(RetrievalStrategyStop())
            rs.retrieval_ms()
    except StopIteration:
        pass
    sinfo = rs.state_info
    # Check a single value, just to make sure we can read this
    if False:
        print(sinfo.l1b_file("CRIS").sounding_desc)
        print(sinfo.l1b_file("TROPOMI").sounding_desc)
    assert sinfo.sounding_metadata().wrong_tai_time == pytest.approx(839312679.58409)
    assert sinfo.species_state("emissivity").value[0] == pytest.approx(0.98081997)
    assert sinfo.species_state("emissivity").wavelength[0] == pytest.approx(600)
    assert sinfo.sounding_metadata().latitude.value == pytest.approx(62.8646) 
    assert sinfo.sounding_metadata().longitude.value == pytest.approx(81.0379) 
    assert sinfo.sounding_metadata().surface_altitude.convert("m").value == pytest.approx(169.827)
    assert sinfo.sounding_metadata().tai_time == pytest.approx(839312683.58409)
    assert sinfo.sounding_metadata().sounding_id == "20190807_065_04_08_5"
    assert sinfo.sounding_metadata().is_land
    assert sinfo.species_state("cloudEffExt").value[0,0] == pytest.approx(1e-29)
    assert sinfo.species_state("cloudEffExt").wavelength[0] == pytest.approx(600)
    assert sinfo.species_state("PCLOUD").value[0] == pytest.approx(500.0)
    assert sinfo.species_state("PSUR").value[0] == pytest.approx(0.0)
    assert sinfo.sounding_metadata().local_hour == pytest.approx(11.40252685546875)
    assert sinfo.sounding_metadata().height.value[0] == 0
    assert sinfo.species_state("TATM").value[0] == pytest.approx(293.28302002)

    # Have a mix of species names in the muses-py ordered list and not. Check
    # that we handle the sort correctly.
    species_list = ["Fred", 'TROPOMICLOUDSURFACEALBEDO', "Carl", 'TROPOMITEMPSHIFTBAND7',
                    "Al"]
    assert sinfo.order_species(species_list) == ['TROPOMITEMPSHIFTBAND7', 'TROPOMICLOUDSURFACEALBEDO', "Al", "Carl", "Fred"]
    

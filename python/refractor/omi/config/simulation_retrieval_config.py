import netCDF4  # type: ignore

from refractor import framework as rf  # type: ignore
from refractor.framework import creator, refractor_config  # type: ignore
from refractor.framework.factory import param  # type: ignore

from .retrieval_run_config import retrieval_run_config, get_input

from refractor.omi.level1 import OmiLevel1RadianceFile


class OmiLevel1Simulation(OmiLevel1RadianceFile):
    def __init__(
        self, l1b_filename, simulation_filename, along_track_index, across_track_indexes
    ):
        super().__init__(l1b_filename, along_track_index, across_track_indexes)

        self.simulation_filename = simulation_filename

    def sample_grid(self, chan_index: int) -> rf.SpectralDomain:
        with netCDF4.Dataset(self.simulation_filename, "r") as sim_file:
            grid_var = sim_file[
                f"/Retrieval/Step_1/Initial/Spectrum/Channel_{chan_index + 1}/Convolved/grid"
            ]
            return rf.SpectralDomain(grid_var[:], rf.Unit("nm"))

    def radiance(self, chan_index: int) -> rf.SpectralRange:
        with netCDF4.Dataset(self.simulation_filename, "r") as sim_file:
            rad_var = sim_file[
                f"/Retrieval/Step_1/Initial/Spectrum/Channel_{chan_index + 1}/Convolved/radiance"
            ]
            rad_uncert = sim_file[f"/Observation/Channel_{chan_index + 1}/uncertainty"]
            return rf.SpectralRange(rad_var[:], rf.Unit(rad_var.units), rad_uncert[:])


class OmiSimulatedRadianceCreator(creator.base.Creator):
    input_filename = param.Scalar(str)
    simulation_filename = param.Scalar(str)
    along_track_index = param.Scalar(int)
    across_track_indexes = param.Iterable(int)

    def create(self, **kwargs):
        return OmiLevel1Simulation(
            self.input_filename(),
            self.simulation_filename(),
            self.along_track_index(),
            self.across_track_indexes(),
        )


@refractor_config
def retrieval_config(**kwargs):
    """Simulate radiances using the retrieval initial guess set up. Used for self consistency retrieval tests."""

    config_def = retrieval_run_config(**kwargs)

    config_def["input"]["l1b"]["creator"] = creator.modifier.singleton(
        OmiSimulatedRadianceCreator
    )
    config_def["input"]["l1b"]["simulation_filename"] = get_input("OMI_SIM_FILENAME")

    return config_def

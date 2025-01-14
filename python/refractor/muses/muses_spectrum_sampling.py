from __future__ import annotations
import refractor.framework as rf  # type: ignore


class MusesSpectrumSampling(rf.SpectrumSampling):
    """Return the hires/monochromatic grid to use. This is a fixed grid, or
    if we aren't using a ILS this is just the lowres grid.
    """

    # Note MusesSpectrumSampling doesn't have the logic in place to skip
    # highres wavelengths that we don't need - e.g., because of bad pixels.
    # For now, just follow the logic that py-retrieve uses, but we may
    # want to change this.

    def __init__(self, hres_spec: list[rf.SpectralDomain | None]):
        super().__init__()
        self.hres_spec = hres_spec

    def spectral_domain(
        self,
        sensor_index: int,
        lowres_grid: rf.SpectralDomain,
        edge_extension: rf.DoubleWithUnit,
    ) -> rf.SpectralDomain:
        if self.hres_spec[sensor_index] is not None:
            return self.hres_spec[sensor_index]
        else:
            return lowres_grid

    def desc(self) -> str:
        return "MusesSpectrumSampling"

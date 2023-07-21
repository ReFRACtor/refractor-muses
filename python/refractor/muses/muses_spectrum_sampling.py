import numpy as np
import refractor.framework as rf

class MusesSpectrumSampling(rf.SpectrumSampling):
    "Return the monochromatic grid using OMI uip information"

    def __init__(self, rf_uip, instrument_name):
        # Initalize director
        super().__init__()

        self.rf_uip = rf_uip
        self.instrument_name = instrument_name
        
    def spectral_domain(self, fm_idx: int, lowres_grid: rf.SpectralDomain, edge_extension: rf.DoubleWithUnit) -> rf.SpectralDomain:
        if self.rf_uip.ils_method(fm_idx, self.instrument_name) == "FASTCONV":
            ils_uip_info = self.rf_uip.ils_params(fm_idx)
            return rf.SpectralDomain(ils_uip_info["monochromgrid"], rf.Unit("nm"))
        else:
            return lowres_grid

    def print_desc(self, ostream):
        # A bit clumsy, we should perhaps put a better interface in
        # here.
        ostream.write("MusesSpectrumSampling", len("MusesSpectrumSampling"))


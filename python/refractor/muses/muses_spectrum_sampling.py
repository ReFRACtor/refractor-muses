import numpy as np
import refractor.framework as rf

class MusesSpectrumSampling(rf.SpectrumSampling):
    "Return the monochromatic grid using OMI uip information"

    def __init__(self, instrument_name, rf_uip=None, ils_method=None):
        # Initalize director
        super().__init__()

        self.rf_uip = rf_uip
        self.instrument_name = instrument_name
        self.ils_method = ils_method
        if(ils_method is None):
            self.ils_method = self.rf_uip.ils_method(0, self.instrument_name)
        
    def spectral_domain(self, fm_idx: int, lowres_grid: rf.SpectralDomain, edge_extension: rf.DoubleWithUnit) -> rf.SpectralDomain:
        if self.ils_method in ("FASTCONV", "POSTCONV"):
            if(self.rf_uip is None):
                raise NotImplementedError()
            ils_uip_info = self.rf_uip.ils_params(fm_idx, self.instrument_name)
            return rf.SpectralDomain(ils_uip_info["central_wavelength"], rf.Unit("nm"))
        else:
            return lowres_grid

    def desc(self):
        return f'''MusesSpectrumSampling:
  Instrument name: {self.instrument_name}
  ILS method:      {self.ils_method}
'''  


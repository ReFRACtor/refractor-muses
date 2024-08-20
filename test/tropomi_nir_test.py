# This is similar to the band 7 data, but is based on a test case from Josh. The
# band 7 test case was a major kludge, and I expect it may go away. This is more
# likely to be the replacement for this.

import numpy as np
import numpy.testing as npt
from test_support import *
import refractor.framework as rf
from refractor.tropomi import (TropomiFmObjectCreator, TropomiForwardModelHandle)
from refractor.muses import MusesRunDir
from refractor.old_py_retrieve_wrapper import RefractorMusesIntegration

class SaveSpectrum(rf.ObserverPtrNamedSpectrum):

    def __init__(self, filename):
        super().__init__()

        self.filename = filename

    def notify_update(self, o):
        import pickle
        fn = self.filename.format(name=o.name.replace(" ", "_"))
        with open(fn, "wb") as out:
            print(f"Saving {o.name} to {fn}")
            data = { "name": o.name,
                     "wavelength": o.spectral_domain.wavelength("nm"),
                     "radiance": o.spectral_range.data,
                    }
            pickle.dump(data, out)

@long_test    
@require_muses_py            
def test_nir_retrieval(isolated_dir, osp_dir, gmao_dir, vlidort_cli,
                       clean_up_replacement_function):
    # NOTE - This depends on the newer OSP directory. If you don't have this
    # data, you should set the environment variable MUSES_OSP_PATH to
    # "/tb/sandbox17/laughner/OSP-mine/OSP"
    
    r = RefractorMusesIntegration()
    r.forward_model_handle_set.add_handle(TropomiForwardModelHandle(use_pca=True,
                                       use_raman=False,
                                       use_lrad=False, lrad_second_order=False),
                                       priority_order=100)
    r.register_with_muses_py()
    r = MusesRunDir(tropomi_test_in_dir2, osp_dir, gmao_dir)
    r.run_retrieval(vlidort_cli=vlidort_cli)


    

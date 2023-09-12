# These are the arguments passed to cli function that loads this code.
from py_retrieve.cli import params
from refractor.omi import RefractorOmiFm

# Replace omi_fm in py-retrieve with RefractorOmiFm call
fm = RefractorOmiFm(use_pca=True, use_lrad=False, lrad_second_order=False)
fm.register_with_muses_py()

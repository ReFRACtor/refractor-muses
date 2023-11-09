from test_support import *
from refractor.muses import order_species

def test_order_species():
    # Have a mix of species names in the muses-py ordered list and not. Check
    # that we handle the sort correctly.
    species_list = ["Fred", 'TROPOMICLOUDSURFACEALBEDO', "Carl", 'TROPOMITEMPSHIFTBAND7',
                    "Al"]
    assert order_species(species_list) == ['TROPOMITEMPSHIFTBAND7', 'TROPOMICLOUDSURFACEALBEDO', "Al", "Carl", "Fred"]
    

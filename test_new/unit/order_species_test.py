from refractor.muses import order_species


def test_order_species():
    # Have a mix of species names in the muses-py ordered list and not. Check
    # that we handle the sort correctly.
    species_list = [
        "Fred",
        "TROPOMICLOUDSURFACEALBEDO",
        "Carl",
        "TROPOMITEMPSHIFTBAND11",
        "Al",
    ]
    assert order_species(species_list) == [
        "TROPOMICLOUDSURFACEALBEDO",
        "Al",
        "Carl",
        "Fred",
        "TROPOMITEMPSHIFTBAND11",
    ]

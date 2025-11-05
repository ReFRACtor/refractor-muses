from __future__ import annotations
from . import muses_py as mpy  # type: ignore

if mpy.have_muses_py:
    _ordered_species_list = mpy.ordered_species_list()
    _atmospheric_species_list = mpy.atmospheric_species_list()
else:
    # Hardcoded, just so we don't depend on muses_py being available. This should be
    # same list
    _ordered_species_list = [
        "TATM",
        "H2O",
        "CO2",
        "O3",
        "N2O",
        "CO",
        "CH4",
        "O2",
        "NO",
        "SO2",
        "NO2",
        "NH3",
        "HNO3",
        "OH",
        "HF",
        "HCL",
        "HBR",
        "HI",
        "CLO",
        "OCS",
        "HCOH",
        "HOCL",
        "N2",
        "HCN",
        "CH3CL",
        "H2O2",
        "C2H2",
        "C2H6",
        "PH3",
        "COF2",
        "SF6",
        "H2S",
        "HCOOH",
        "HO2",
        "O",
        "CLONO2",
        "NOPLUS",
        "HOBR",
        "CCL4",
        "CFC11",
        "CFC12",
        "CFC22",
        "TSUR",
        "TSUR_2B1",
        "TSUR_1B2",
        "TSUR_2A1",
        "TSUR_1A1",
        "PTGANG",
        "EMIS",
        "E0",
        "R0",
        "TABSURF",
        "CLOUDOD",
        "CLOUDEXT",
        "PCLOUD",
        "CALSCALE",
        "CALOFFSET",
        "HDO",
        "H2O17",
        "H2O18",
        "CH3OH",
        "C2H4",
        "PAN",
        "RESSCALE",
        "OMICLOUDFRACTION",
        "OMISURFACEALBEDOUV1",
        "OMISURFACEALBEDOUV2",
        "OMISURFACEALBEDOSLOPEUV2",
        "OMINRADWAVUV1",
        "OMINRADWAVUV2",
        "OMIODWAVUV1",
        "OMIODWAVUV2",
        "OMIODWAVSLOPEUV1",
        "OMIODWAVSLOPEUV2",
        "OMIODWAV",
        "OMIRINGSFUV1",
        "OMIRINGSFUV2",
        "OMIRESSCALE",
        "ACET",
        "ISOP",
        "CFC14",
        #'PSUR','OCO2AER','OCO2ALB','OCO2ALBBRDF','OCO2ALBLAMB','OCO2DISP','OCO2EOF','OCO2FLUOR','OCO2WIND',
        # nir quantities for OCO
        "PSUR",
        "NIRAEROD",
        "NIRAERP",
        "NIRAERW",
        "NIRALBPL",
        "NIRALB",
        "NIRALBLAMB",
        "NIRALBBRDF",
        "NIRALBCM",
        "NIRALBLAMBPL",
        "NIRALBBRDFPL",
        "NIRDISP",
        "NIREOF",
        "NIRCLOUD3DOFFSET",
        "NIRCLOUD3DSLOPE",
        "NIRFLUOR",
        "NIRWIND",
        "TROPOMICLOUDFRACTION",
        "TROPOMISURFACEALBEDOBAND1",
        "TROPOMISURFACEALBEDOBAND2",
        "TROPOMISURFACEALBEDOBAND3",
        "TROPOMISURFACEALBEDOBAND3TIGHT",
        "TROPOMISURFACEALBEDOBAND7",
        "TROPOMISURFACEALBEDOSLOPEBAND2",
        "TROPOMISURFACEALBEDOSLOPEBAND3",
        "TROPOMISURFACEALBEDOSLOPEBAND3TIGHT",
        "TROPOMISURFACEALBEDOSLOPEBAND7",
        "TROPOMISURFACEALBEDOSLOPEORDER2BAND2",
        "TROPOMISURFACEALBEDOSLOPEORDER2BAND3",
        "TROPOMISURFACEALBEDOSLOPEORDER2BAND3TIGHT",
        "TROPOMISURFACEALBEDOSLOPEORDER2BAND7",
        "TROPOMISOLARSHIFTBAND1",
        "TROPOMISOLARSHIFTBAND2",
        "TROPOMISOLARSHIFTBAND3",
        "TROPOMISOLARSHIFTBAND7",
        "TROPOMIRADIANCESHIFTBAND1",
        "TROPOMIRADIANCESHIFTBAND2",
        "TROPOMIRADIANCESHIFTBAND3",
        "TROPOMIRADIANCESHIFTBAND7",
        "TROPOMIRADSQUEEZEBAND1",
        "TROPOMIRADSQUEEZEBAND2",
        "TROPOMIRADSQUEEZEBAND3",
        "TROPOMIRADSQUEEZEBAND7",
        "TROPOMIRINGSFBAND1",
        "TROPOMIRINGSFBAND2",
        "TROPOMIRINGSFBAND3",
        "TROPOMIRINGSFBAND7",
        "TROPOMIRESSCALE",
        "TROPOMIRESSCALEO0BAND2",
        "TROPOMIRESSCALEO1BAND2",
        "TROPOMIRESSCALEO2BAND2",
        "TROPOMIRESSCALEO0BAND3",
        "TROPOMIRESSCALEO1BAND3",
        "TROPOMIRESSCALEO2BAND3",
        "TROPOMITEMPSHIFTBAND3",
        "TROPOMITEMPSHIFTBAND3TIGHT",
        "TROPOMIRESSCALEO0BAND7",
        "TROPOMIRESSCALEO1BAND7",
        "TROPOMIRESSCALEO2BAND7",
        "TROPOMITEMPSHIFTBAND7",
        "TROPOMICLOUDSURFACEALBEDO",
    ]
    _atmospheric_species_list = [
        "TATM",
        "H2O",
        "CO2",
        "O3",
        "N2O",
        "CO",
        "CH4",
        "O2",
        "NO",
        "SO2",
        "NO2",
        "NH3",
        "HNO3",
        "OH",
        "HF",
        "HCL",
        "HBR",
        "HI",
        "CLO",
        "OCS",
        "HCOH",
        "HOCL",
        "N2",
        "HCN",
        "CH3CL",
        "H2O2",
        "C2H2",
        "C2H6",
        "PH3",
        "COF2",
        "SF6",
        "H2S",
        "HCOOH",
        "HO2",
        "O",
        "CLONO2",
        "NOPLUS",
        "HOBR",
        "CCL4",
        "CFC11",
        "CFC12",
        "CFC22",
        "HDO",
        "H2O17",
        "H2O18",
        "CH3OH",
        "C2H4",
        "PAN",
        "ACET",
        "ISOP",
        "CFC14",
    ]


def is_atmospheric_species(species_name: str) -> bool:
    """Some species are marked as "atmospheric_species". This is used in the
    determination of the microwindows file name, this wants to filter out things
    like O3_EMIS, O3_TSUR, and just have O3 pass. I don't think this gets used
    anywhere else."""
    return species_name.upper() in _atmospheric_species_list


def order_species(species_list: list[str]) -> list[str]:
    """This provides an order to the given species list. This is used to
    make sure various matrixes etc. have the state elements all ordered the
    same way.

    We use the list order the mpy.ordered_species_list have for items
    found in that list, and then we just alphabetize any species not
    found in that list. We could adapt this logic if needed (e.g.,
    extend ordered_species_list), but this seems sufficient for now.

    """

    # The use of a tuple here is a standard python "trick" to separate
    # out the values sorted by _ordered_species_list
    # vs. alphabetically. In python False < True, so all the items in
    # _ordered_species_list are put first in the sorted list. The
    # second part of the tuple is only sorted for things that are in
    # the same set for the first test, so using integers from index
    # vs. string comparison is separated nicely.
    return sorted(
        species_list,
        key=lambda v: (
            v not in _ordered_species_list,
            _ordered_species_list.index(v) if v in _ordered_species_list else v,
        ),
    )


def compare_species(s1: str, s2: str) -> int:
    """Return -1, 0 or 1 for s1 < s2, s1 == s2, s1 > s2"""
    if s1 in _ordered_species_list and s2 not in _ordered_species_list:
        return -1
    if s1 not in _ordered_species_list and s2 in _ordered_species_list:
        return 1
    t1 = (
        s1 in _ordered_species_list,
        _ordered_species_list.index(s1) if s1 in _ordered_species_list else s1,
    )
    t2 = (
        s2 in _ordered_species_list,
        _ordered_species_list.index(s2) if s2 in _ordered_species_list else s2,
    )
    if t1 < t2:
        return -1
    elif t1 == t2:
        return 0
    return 1


__all__ = [
    "is_atmospheric_species",
    "order_species",
    "compare_species",
]

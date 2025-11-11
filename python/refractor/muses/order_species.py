from __future__ import annotations
from .mpy import (
    have_muses_py,
    mpy_ordered_species_list,
    mpy_atmospheric_species_list,
    mpy_cdf_var_attributes,
    mpy_cdf_var_names,
    mpy_cdf_var_map,
)

# muses-py has a lot of hard coded things related to the species names and netcdf output.
# It would be good a some point to just replace this all with a better thought out output
# format. But for now, we need to support the existing output format.
# TODO - Replace with better thought out output format

# Temp, test hardcoded stuff
if False and have_muses_py:
    _ordered_species_list: list[str] = mpy_ordered_species_list()
    _atmospheric_species_list: list[str] = mpy_atmospheric_species_list()
    cdf_var_attributes: dict[str, dict[str, str | float]] = mpy_cdf_var_attributes
    cdf_var_names: list[list[str]] = mpy_cdf_var_names()
    cdf_var_map: dict[str, str] = mpy_cdf_var_map()
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
    cdf_var_attributes = dict(
        TROPOSPHERICCOLUMN_attr={
            "Longname": "Tropospheric column amount computed from the retrieved profile.",
            "Units": "molecules/m2",
            "FillValue": -999.00,
            "MissingValue": -999.00,
        },
        TROPOSPHERICCOLUMNERROR_attr={
            "Longname": "Tropospheric column error estimate.",
            "Units": "molecules/m2",
            "FillValue": -999.00,
            "MissingValue": -999.00,
        },
        TROPOSPHERICCOLUMNINITIAL_attr={
            "Longname": "Initial tropospheric column.",
            "Units": "molecules/m2",
            "FillValue": -999.00,
            "MissingValue": -999.00,
        },
        O3TROPOSPHERICCOLUMN_attr={
            "Longname": "Tropospheric column amount computed from the retrieved profile.",
            "Units": "molecules/m2",
            "FillValue": -999.00,
            "MissingValue": -999.00,
        },
        O3TROPOSPHERICCOLUMNERROR_attr={
            "Longname": "Tropospheric column error estimate.",
            "Units": "molecules/m2",
            "FillValue": -999.00,
            "MissingValue": -999.00,
        },
        O3TROPOSPHERICCOLUMNINITIAL_attr={
            "Longname": "Initial tropospheric column.",
            "Units": "molecules/m2",
            "FillValue": -999.00,
            "MissingValue": -999.00,
        },
        # AT_LINE 962 TOOLS/cdf_write_tes.pro
        # ===============================
        # Define variables attributes to improve
        # variables and file user-friendliness
        # Extra definitions are fine here, it will only pick up what is needed.
        # ===============================
        ALTITUDE_attr={
            "Longname": "altitude at each target",
            "Units": "meters",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        SURFACEALTITUDE_attr={
            "Longname": "surface altitude at each target",
            "Units": "meters",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        ALTITUDE_FM_attr={
            "Longname": "altitude at each target on FM grid",
            "Units": "meters",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        AIRDENSITY_attr={
            "LongName": "density of air",
            "Units": "molecules/m^3",
            "FillValue": -999.00,
            "MissingValue": -999.00,
        },
        AVERAGE_800_TROPOPAUSE_attr={
            "Longname": "average between 800 hPa and Tropopause",
            "Units": "VMR",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        AVERAGE_800_TROPOPAUSEPRIOR_attr={
            "Longname": "average between 800 hPa and Tropopause for the prior",
            "Units": "VMR",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        XPAN800_attr={
            "Longname": "average between 800 hPa and Tropopause in PPT",
            "Units": "VMR (PPT)",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        XPAN800_PRIOR_attr={
            "Longname": "prior average between 800 hPa and Tropopause in PPT",
            "Units": "VMR (PPT)",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        XPAN_attr={
            "Longname": "column average in PPT",
            "Units": "VMR (PPT)",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        XPAN_PRIOR_attr={
            "Longname": "prior column average in PPT",
            "Units": "VMR (PPT)",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        XPAN800_ERROROBS_attr={
            "Longname": "obs error for xpan800 in PPT",
            "Units": "VMR (PPT)",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        XPAN800_AK_attr={
            "Longname": "column AK for xpan800 in PPT",
            "Units": "VMR (PPT)",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        XPAN800_ERRORSMOOTHING_attr={
            "Longname": "smoothing error for xpan800 in PPT",
            "Units": "VMR (PPT)",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        XPAN_ERROROBS_attr={
            "Longname": "obs error for xpan",
            "Units": "VMR (PPT)",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        XPAN_ERRORSMOOTHING_attr={
            "Longname": "smoothing error for xpan in PPT",
            "Units": "VMR (PPT)",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        XPAN_AK_attr={
            "Longname": "column AK for xpan",
            "Units": "VMR (PPT)",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        AVERAGE_900_200_attr={
            "Longname": "average between 900 and 200 hPa",
            "Units": "VMR",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        RADIANCEMAXIMUMSNR_attr={
            "Longname": "Maximum signal to noise ratio (used in some quality flags)",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        CITY_attr={
            "Longname": "city name",
            "Units": "",
            "FillValue": "",
            "MisingValue": "",
        },
        CITYINDEX_attr={
            "Longname": "city index:  [Los Angeles, Houston, New York City, Mexico City, Sao Paulo, Buenos Aires, Paris, Istanbul, Lagos, Beijing, SW China, Shenzhen, Tokyo, Bangkok, Dhaka, Delhi, Mumbai, Kolkata, Karachi]",
            "Units": "",
            "FillValue": "",
            "MisingValue": "",
        },
        POINTINGANGLE_AIRS_attr={
            "Longname": "",
            "Units": "",
            "FillValue": -999.0,
            "MisingValue": -999.0,
        },
        POINTINGANGLE_TES_attr={
            "Longname": "",
            "Units": "",
            "FillValue": -999.0,
            "MisingValue": -999.0,
        },
        POINTINGANGLE_CRIS_attr={
            "Longname": "",
            "Units": "",
            "FillValue": -999.0,
            "MisingValue": -999.0,
        },
        CRIS_GRANULE_attr={
            "Longname": "",
            "Units": "",
            "FillValue": -999.0,
            "MisingValue": -999.0,
        },
        # CRIS_SCANLINE_attr={'Longname':'The CrIS instrument scans from west to east direction, contains 30 FORs, forming a scan line in the across track direction', 'Units':'', 'FillValue':-999.0, 'MisingValue':-999.0},
        # CRIS_FIELDOFREGARD_attr={'Longname':'field of regard (0,1,2,...), ', 'Units':'', 'FillValue':-999.0, 'MisingValue':-999.0},
        # CRIS_PIXEL_attr={'Longname':'Values can be 0-8', 'Units':'', 'FillValue':-999.0, 'MisingValue':-999.0},
        CRIS_ATRACK_INDEX_attr={
            "Longname": "The CrIS instrument scans from west to east direction, contains 30 FORs, forming a scan line in the across track direction",
            "Units": "",
            "fillValue": -999.0,
            "MissingValue": -999.0,
        },
        CRIS_XTRACK_INDEX_attr={
            "Longname": "field of regard (0,1,2,...), ",
            "Units": "",
            "fillValue": -999.0,
            "MissingValue": -999.0,
        },
        CRIS_PIXEL_INDEX_attr={
            "Longname": "Values can be 0-8",
            "Units": "",
            "FillValue": -999.0,
            "MissingValue": -999.0,
        },
        CRIS_L1B_TYPE_attr={
            "Longname": "Values can be 0-6. 0=suomi_nasa_nsr (nominal spectral resolution), 1=suomi_nasa_fsr (full spectral resolution), 2=suomi_nasa_nomw, 3=jpss1_nasa_fsr, 4=suomi_cspp_fsr, 5=jpss1_cspp_fsr, 6=jpss2_cspp_fsr",
            "Units": "",
            "FillValue": -999.0,
            "MissingValue": -999.0,
        },
        # AT_LINE 1078 TOOLS/cdf_write_tes.pro
        # The CrIS L1B file was organized in the following dimension: Number of scan line x Number of Field of Regard (FOR) x Number of Pixel within a Field of Regard
        # The CrIS instrument scans from west to east direction, contains 30 FORs, forming a scan line in the across track directionn
        # A NOAA CrIS L1 file has a different scan line than that of a NASA CrIS L1B file.
        # A single FOR contains 3 X 3 single pixels CrIS measurements.
        # These indexes start from 0 for the convention of IDL goodness. -?
        FILENAME_attr={
            "Longname": "filename",
            "Units": "",
            "FillValue": "",
            "MisingValue": "",
        },
        UPPERTROPOSPHERICCOLUMN_attr={
            "Longname": "UPPERTROPOSPHERICCOLUMN",
            "Units": "",
            "FillValue": "",
            "MisingValue": "",
        },
        UPPERTROPOSPHERICCOLUMNERROR_attr={
            "Longname": "UPPERTROPOSPHERICCOLUMNERROR",
            "Units": "",
            "FillValue": "",
            "MisingValue": "",
        },
        UPPERTROPOSPHERICCOLUMNINITIAL_attr={
            "Longname": "UPPERTROPOSPHERICCOLUMNINITIAL",
            "Units": "",
            "FillValue": "",
            "MisingValue": "",
        },
        LOWERTROPOSPHERICCOLUMN_attr={
            "Longname": "LOWERTROPOSPHERICCOLUMN",
            "Units": "",
            "FillValue": "",
            "MisingValue": "",
        },
        LOWERTROPOSPHERICCOLUMNERROR_attr={
            "Longname": "LOWERTROPOSPHERICCOLUMNERROR",
            "Units": "",
            "FillValue": "",
            "MisingValue": "",
        },
        LOWERTROPOSPHERICCOLUMNINITIAL_attr={
            "Longname": "LOWERTROPOSPHERICCOLUMNINITIAL",
            "Units": "",
            "FillValue": "",
            "MisingValue": "",
        },
        DOFSTROPOSPHERE_attr={
            "Longname": "DOFSTROPOSPHERE",
            "Units": "",
            "FillValue": "",
            "MisingValue": "",
        },
        DOFSUPPERTROPOSPHERE_attr={
            "Longname": "DOFSUPPERTROPOSPHERE",
            "Units": "",
            "FillValue": "",
            "MisingValue": "",
        },
        DOFSLOWERTROPOSPHERE_attr={
            "Longname": "DOFSLOWERTROPOSPHERE",
            "Units": "",
            "FillValue": "",
            "MisingValue": "",
        },
        OMI_SZA_UV1_attr={
            "Longname": "omi_sza_uv1",
            "Units": "degrees",
            "FillValue": "",
            "MisingValue": "",
        },
        OMI_RAZ_UV1_attr={
            "Longname": "omi_raz_uv1",
            "Units": "degrees",
            "FillValue": "",
            "MisingValue": "",
        },
        OMI_VZA_UV1_attr={
            "Longname": "omi_vza_uv1",
            "Units": "degrees",
            "FillValue": "",
            "MisingValue": "",
        },
        OMI_SCA_UV1_attr={
            "Longname": "omi_sca_uv1",
            "Units": "degrees",
            "FillValue": "",
            "MisingValue": "",
        },
        OMI_SZA_UV2_attr={
            "Longname": "omi_sza_uv2",
            "Units": "degrees",
            "FillValue": "",
            "MisingValue": "",
        },
        OMI_RAZ_UV2_attr={
            "Longname": "omi_raz_uv2",
            "Units": "degrees",
            "FillValue": "",
            "MisingValue": "",
        },
        OMI_VZA_UV2_attr={
            "Longname": "omi_vza_uv2",
            "Units": "degrees",
            "FillValue": "",
            "MisingValue": "",
        },
        OMI_SCA_UV2_attr={
            "Longname": "omi_sca_uv2",
            "Units": "degrees",
            "FillValue": "",
            "MisingValue": "",
        },
        TROPOMI_SZA_BAND1_attr={
            "Longname": "tropomi_sza_band1",
            "Units": "degrees",
            "FillValue": -999.0,
            "MisingValue": "",
        },
        TROPOMI_RAZ_BAND1_attr={
            "Longname": "tropomi_raz_band1",
            "Units": "degrees",
            "FillValue": -999.0,
            "MisingValue": "",
        },
        TROPOMI_VZA_BAND1_attr={
            "Longname": "tropomi_vza_band1",
            "Units": "degrees",
            "FillValue": -999.0,
            "MisingValue": "",
        },
        TROPOMI_SCA_BAND1_attr={
            "Longname": "tropomi_sca_band1",
            "Units": "degrees",
            "FillValue": -999.0,
            "MisingValue": "",
        },
        TROPOMI_SZA_BAND2_attr={
            "Longname": "tropomi_sza_band2",
            "Units": "degrees",
            "FillValue": -999.0,
            "MisingValue": "",
        },
        TROPOMI_RAZ_BAND2_attr={
            "Longname": "tropomi_raz_band2",
            "Units": "degrees",
            "FillValue": -999.0,
            "MisingValue": "",
        },
        TROPOMI_VZA_BAND2_attr={
            "Longname": "tropomi_vza_band2",
            "Units": "degrees",
            "FillValue": -999.0,
            "MisingValue": "",
        },
        TROPOMI_SCA_BAND2_attr={
            "Longname": "tropomi_sca_band2",
            "Units": "degrees",
            "FillValue": -999.0,
            "MisingValue": "",
        },
        TROPOMI_SZA_BAND3_attr={
            "Longname": "tropomi_sza_band3",
            "Units": "degrees",
            "FillValue": -999.0,
            "MisingValue": "",
        },
        TROPOMI_RAZ_BAND3_attr={
            "Longname": "tropomi_raz_band3",
            "Units": "degrees",
            "FillValue": -999.0,
            "MisingValue": "",
        },
        TROPOMI_VZA_BAND3_attr={
            "Longname": "tropomi_vza_band3",
            "Units": "degrees",
            "FillValue": -999.0,
            "MisingValue": "",
        },
        TROPOMI_SCA_BAND3_attr={
            "Longname": "tropomi_sca_band3",
            "Units": "degrees",
            "FillValue": -999.0,
            "MisingValue": "",
        },
        TROPOMI_SZA_BAND7_attr={
            "Longname": "tropomi_sza_band7",
            "Units": "degrees",
            "FillValue": -999.0,
            "MisingValue": "",
        },
        TROPOMI_RAZ_BAND7_attr={
            "Longname": "tropomi_raz_band7",
            "Units": "degrees",
            "FillValue": -999.0,
            "MisingValue": "",
        },
        TROPOMI_VZA_BAND7_attr={
            "Longname": "tropomi_vza_band7",
            "Units": "degrees",
            "FillValue": -999.0,
            "MisingValue": "",
        },
        TROPOMI_SCA_BAND7_attr={
            "Longname": "tropomi_sca_band7",
            "Units": "degrees",
            "FillValue": -999.0,
            "MisingValue": "",
        },
        AIRS_GRANULE_attr={
            "Longname": "airs_granule",
            "Units": "",
            "FillValue": "",
            "MisingValue": "",
        },
        AIRS_ATRACK_INDEX_attr={
            "Longname": "airs_atrack_index",
            "Units": "",
            "FillValue": "",
            "MisingValue": "",
        },
        AIRS_XTRACK_INDEX_attr={
            "Longname": "airs_xtrack_index",
            "Units": "",
            "FillValue": "",
            "MisingValue": "",
        },
        TES_RUN_attr={
            "Longname": "tes_run",
            "Units": "",
            "FillValue": "",
            "MisingValue": "",
        },
        TES_SEQUENCE_attr={
            "Longname": "tes_sequence",
            "Units": "",
            "FillValue": "",
            "MisingValue": "",
        },
        TES_SCAN_attr={
            "Longname": "tes_scan",
            "Units": "",
            "FillValue": "",
            "MisingValue": "",
        },
        OCO2_CO2_RATIO_IDP_attr={
            "Longname": "Contains the ratio of the retrieved CO2 column from the weak Co2 band relative to that from the strong CO2 band. This ratio should be near unity.",
            "Units": "",
            "FillValue": "",
            "MisingValue": "",
        },
        OCO2_H2O_RATIO_IDP_attr={
            "Longname": "Contains the ratio of the retrieved H2O column from the weak CO2 band relative to that from the strong CO2 band. This ratio should be near unity.",
            "Units": "",
            "FillValue": "",
            "MisingValue": "",
        },
        OCO2_DP_ABP_attr={
            "Longname": "OCO-2 preprocessing result for change in surface pressure",
            "Units": "hPa",
            "FillValue": "",
            "MisingValue": "",
        },
        OCO2_ALTITUDE_STDDEV_attr={
            "Longname": "The standard deviation of the surface elevation in the target field of view, in meters.",
            "Units": "m",
            "FillValue": "",
            "MisingValue": "",
        },
        OCO2_MAX_DECLOCKING_FACTOR_WCO2_attr={
            "Longname": "An estimate of the absolute value of the clocking error in the weak CO2 band (used in the clocking correction algorithm that attempts to correct the L1b radiances for clocking errors). Expressed in percent. Typical values range from 0 to 10%.",
            "Units": "%",
            "FillValue": "",
            "MisingValue": "",
        },
        OCO2_MAX_DECLOCKING_FACTOR_SCO2_attr={
            "Longname": "An estimate of the absolute value of the clocking error in the strong CO2 band (used in the clocking correction algorithm that attempts to correct the L1b radiances for clocking errors). Expressed in percent. Typical values range from 0 to 10%.",
            "Units": "%",
            "FillValue": "",
            "MisingValue": "",
        },
        OMI_ATRACK_INDEX_attr={
            "Longname": "omi_atrack_index",
            "Units": "",
            "FillValue": "",
            "MisingValue": "",
        },
        OMI_XTRACK_INDEX_UV1_attr={
            "Longname": "omi_xtrack_index for uv1",
            "Units": "",
            "FillValue": "",
            "MisingValue": "",
        },
        OMI_XTRACK_INDEX_UV2_attr={
            "Longname": "omi_xtrack_index for uv2",
            "Units": "",
            "FillValue": "",
            "MisingValue": "",
        },
        TROPOMI_ATRACK_INDEX_attr={
            "Longname": "tropomi_atrack_index",
            "Units": "",
            "FillValue": "",
            "MisingValue": "",
        },
        TROPOMI_XTRACK_INDEX_BAND1_attr={
            "Longname": "tropomi_xtrack_index for band1",
            "Units": "",
            "FillValue": "",
            "MisingValue": "",
        },
        TROPOMI_XTRACK_INDEX_BAND2_attr={
            "Longname": "tropomi_xtrack_index for band2",
            "Units": "",
            "FillValue": "",
            "MisingValue": "",
        },
        TROPOMI_XTRACK_INDEX_BAND3_attr={
            "Longname": "tropomi_xtrack_index for band3",
            "Units": "",
            "FillValue": "",
            "MisingValue": "",
        },
        TROPOMI_XTRACK_INDEX_BAND7_attr={
            "Longname": "tropomi_xtrack_index for band7",
            "Units": "",
            "FillValue": "",
            "MisingValue": "",
        },
        H2O_QA_attr={
            "Longname": "H2O quality",
            "Units": "",
            "FillValue": -999,
            "MisingValue": "",
        },
        O3_QA_attr={
            "Longname": "O3 quality",
            "Units": "",
            "FillValue": -999,
            "MisingValue": "",
        },
        TATM_QA_attr={
            "Longname": "TATM quality",
            "Units": "",
            "FillValue": -999,
            "MisingValue": "",
        },
        COLUMN750_attr={
            "Longname": "COLUMN750, column above 750 hPa",
            "Units": "VMR (ppb)",
            "FillValue": -999,
            "MisingValue": "",
        },
        COLUMN750_ERROR_attr={
            "Longname": "COLUMN750 error",
            "Units": "VMR (ppb)",
            "FillValue": -999,
            "MisingValue": "",
        },
        COLUMN750_OBSERVATIONERROR_attr={
            "Longname": "COLUMN750 observation error",
            "Units": "VMR (ppb)",
            "FillValue": -999,
            "MisingValue": "",
        },
        COLUMN750_CONSTRAINTVECTOR_attr={
            "Longname": "COLUMN750 prior",
            "Units": "VMR (ppb)",
            "FillValue": -999,
            "MisingValue": "",
        },
        COLUMN750_INITIAL_attr={
            "Longname": "COLUMN750 initial",
            "Units": "VMR (ppb)",
            "FillValue": -999,
            "Missingalue": "",
        },
        COLUMN750_AVERAGINGKERNEL_attr={
            "Longname": "COLUMN750 averaging kernel",
            "Units": "()",
            "FillValue": -999,
            "Missingalue": "",
        },
        COLUMN750_PWF_attr={
            "Longname": "COLUMN750 pressure weighting function",
            "Units": "()",
            "FillValue": -999,
            "Missingalue": "",
        },
        SURFACEPRESSURE_attr={
            "Longname": "surface pressure",
            "Units": "VMR",
            "FillValue": -999,
            "MisingValue": "",
        },
        AVERAGECLOUDEFFOPTICALDEPTH_attr={
            "Longname": "average cloud effective optical depth",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        AVERAGECLOUDEFFOPTICALDEPTHERROR_attr={
            "Longname": "average cloud effective optical depth error",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        CLOUDEFFECTIVEOPTICALDEPTH1000_attr={
            "Longname": "average cloud effective optical depth at 1000 cm-1",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        CLOUDEFFECTIVEOPTICALDEPTHERROR1000_attr={
            "Longname": "average cloud effective optical depth error and 1000 cm-1",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        CLOUDEFFECTIVEOPTICALDEPTH_attr={
            "Longname": "average cloud effective optical depth",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        # AT_LINE 1156 TOOLS/cdf_write_tes.pro
        CLOUDEFFECTIVEOPTICALDEPTHERROR_attr={
            "Longname": "average cloud effective optical depth error",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        # AT_LINE 1161 TOOLS/cdf_write_tes.pro
        AVERAGINGKERNEL_attr={
            "Longname": "The averaging kernel is the sensitivity of the estimated state to variations in the atmospheric state. The rows of the averaging kernel represent the sensitivity of the estimated state at a specific pressure level to variations in the atmospheric state at all levels. The columns of averaging kernel represent the sensitivity of the estimated state at all levels to variations in the atmospheric state at specific pressure level.  For atmospheric species this is the sensitivity of retrieved ln(vmr) to the true ln(vmr).",
            "FullDescription": "see http://tes.jpl.nasa.gov/uploadedfiles/TES_DPS_V11.8.pdf",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        H2O_HDOMEASUREMENTERRORCOVARIANCE_attr={
            "Longname": "Measurement error for H2O/HDO ratio",
            "FullDescription": "see http://tes.jpl.nasa.gov/uploadedfiles/TES_DPS_V11.8.pdf",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        H2O_HDOTOTALERRORCOVARIANCE_attr={
            "Longname": "Total error for H2O/HDO ratio",
            "FullDescription": "see http://tes.jpl.nasa.gov/uploadedfiles/TES_DPS_V11.8.pdf",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        H2O_HDOOBSERVATIONERRORCOVARIANCE_attr={
            "Longname": "Total error for H2O/HDO ratio",
            "FullDescription": "see http://tes.jpl.nasa.gov/uploadedfiles/TES_DPS_V11.8.pdf",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        AVERAGINGKERNELDIAGONAL_attr={
            "Longname": "averaging kernel diagonal",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        AVERAGINGKERNELDIAGONAL_FM_attr={
            "Longname": "averaging kernel diagonal on fm grid",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        CLOUDTOPPRESSURE_attr={
            "Longname": "Pressure of inferred cloud top (species independent)",
            "Units": "hPa",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        CLOUDTOPPRESSUREERROR_attr={
            "Longname": "Cloud top pressure error",
            "Units": "hPa",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        THERMALCONTRAST_attr={
            "Longname": "Thermal contrast=surface temperature - lowest atmospheric temperature for retrieved TES values ",
            "Units": "K",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        THERMALCONTRASTINITIAL_attr={
            "Longname": "Thermal contrast=surface temperature - lowest atmospheric temperature for inital TES values (from GMAO)",
            "Units": "K",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        CLOUDVARIABILITY_QA_attr={
            "Longname": "Quality value calculated from cloud variability",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        O3_CCURVE_QA_attr={
            "Longname": "Flag for if O3 ccurve found.  1=no ccurve found.  0=ccurve found.",
            "Units": "",
            "FillValue": 157,
            "MisingValue": 157,
        },
        O3_SLOPE_QA_attr={
            "Longname": "o3 mean absolute slope (ppb/m)",
            "Units": "",
            "FillValue": -999,
            "MisingValue": -999,
        },
        O3_COLUMNERRORDU_attr={
            "Longname": "o3 column error in DU",
            "Units": "DU",
            "FillValue": -999,
            "MisingValue": -999,
        },
        O3_TROPO_CONSISTENCY_qa_attr={
            "Longname": "o3 tropospheric column divided by the initial guess for this step - 1",
            "Units": "fraction",
            "FillValue": -999,
            "MisingValue": -999,
        },
        TROPOMI_CLOUDFRACTION_attr={
            "Longname": "tropomi cloud fraction",
            "Units": "",
            "FillValue": -999,
            "MisingValue": -999,
        },
        OZONEIRK_attr={
            "Longname": "Ozone IRK",
            "Units": "",
            "FillValue": -999,
            "MisingValue": -999,
        },
        # AT_LINE 1238 TOOLS/cdf_write_tes.pro
        CH4_STRATOSPHERE_QA_attr={
            "Longname": "fraction of the averaging kernel above the tropopause for pressure level 562 hPa.  Lower values are better.",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        CH4_DOFTROP_attr={
            "Longname": "degrees of freedom below the tropopause",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        CH4_DOFSTRAT_attr={
            "Longname": "degrees of freedom above the tropopause",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        CONSTRAINTVECTOR_FM_attr={
            "Longname": "tes apriori volume mixing ratio on fm grid",
            "Units": "mole/mole",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        CONSTRAINTVECTOR_attr={
            "Longname": "tes apriori volume mixing ratio",
            "Units": "mole/mole",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        DEGREESOFFREEDOMFORSIGNAL_attr={
            "Longname": "number of independent parameters for the profile",
            "Othername": "trace of the averaging kernel",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        DEVIATIONVSRETRIEVALCOVARIANCE_attr={
            "Longname": "Deviations vs. retrieval covariance",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        DEVIATION_QA_attr={
            "Longname": "O3 deviation",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        DEVIATIONBAD_QA_attr={
            "Longname": "Too many deviations from prior",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        NUM_DEVIATIONS_QA_attr={
            "Longname": "O3 deviation",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        MAXNUMITERATIONS_attr={
            "Longname": "Maximum number of iterations allowed for convergence",
            "Units": "",
            "FillValue": -99,
            "MisingValue": -99,
        },
        MEASUREMENTERRORCOVARIANCE_attr={
            "Longname": "Propagated measured radiance noise",
            "Units": "ln(vmr)^2",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        NUMBERITERPERFORMED_attr={
            "Longname": "Actual number of iterations performed",
            "Units": "",
            "FillValue": -99,
            "MisingValue": -99,
        },
        PRECISION_attr={
            "Longname": "square root of diagonal elements of the measurement error covariance",
            "Units": "ln(vmr)",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        # AT_LINE 1324 TOOLS/cdf_write_tes.pro
        PRECISION_FM_attr={
            "Longname": "square root of diagonal elements of the measurement error covariance on fm grid",
            "Units": "ln(vmr)",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        OMI_CLOUDFRACTION_attr={
            "Longname": "fraction of scene obscured by cloud",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        OMI_CLOUDFRACTIONCONSTRAINTVECTOR_attr={
            "Longname": "fraction of scene obscured by cloud prior",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        OMI_CLOUDTOPPRESSURE_attr={
            "Longname": "",
            "Units": "hPa",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        OMI_SURFACEALBEDOUV1_attr={
            "Longname": "single value for band",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        OMI_SURFACEALBEDOUV1CONSTRAINTVECTOR_attr={
            "Longname": "",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        OMI_SURFACEALBEDOUV2_attr={
            "Longname": "linear fit",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        OMI_SURFACEALBEDOUV2CONSTRAINTVECTOR_attr={
            "Longname": "prior",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        OMI_SURFACEALBEDOSLOPEUV2_attr={
            "Longname": "",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        OMI_SURFACEALBEDOSLOPEUV2CONSTRAINTVECTOR_attr={
            "Longname": "",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        OMI_NRADWAVUV1_attr={
            "Longname": "",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        OMI_NRADWAVUV2_attr={
            "Longname": "",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        OMI_ODWAVUV1_attr={
            "Longname": "",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        OMI_ODWAVUV2_attr={
            "Longname": "",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        OMI_RINGSFUV1_attr={
            "Longname": "",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        OMI_RINGSFUV2_attr={
            "Longname": "",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        OMI_SZA_attr={
            "Longname": "",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        # AT_LINE 1411 TOOLS/cdf_write_tes.pro
        OMI_RAZ_attr={
            "Longname": "",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        OBSERVATIONERRORCOVARIANCE_attr={
            "Longname": "Measurement + systematic + cross-state errors.  The utility of the observation error is for comparisons with other measurements and for assimilation.  The smoothing error is accounted for when one applies the averaging kernel, so the observation error accounts for everything else. See comment for TotalErrorCovariance.",
            "FullDescription": "see http://tes.jpl.nasa.gov/uploadedfiles/TES_DPS_V11.8.pdf",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        PRESSURE_attr={
            "Longname": "Atmospheric pressure grid used for retrieval",
            "Units": "hPa",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        PRESSURE_FM_attr={
            "Longname": "Full pressure grid used in forward model calculation.  Actual pressure grid varies depending on surface pressure.",
            "Units": "hPa",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        RADIANCERESIDUALMAX_attr={
            "Longname": "Maximum absolute difference between model and data",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        RETRIEVEINLOG_attr={
            "Longname": "Set to 1 if log(VMR, etc) is retrieved, 0 if VMR, etc is retrieved",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        LMRESULTS_COSTTHRESH_attr={
            "Longname": "LM retrieval parameter",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        LMRESULTS_RESNORM_attr={
            "Longname": "LM retrieval parameter",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        LMRESULTS_RESNORMNEXT_attr={
            "Longname": "LM retrieval parameter",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        LMRESULTS_DELTA_attr={
            "Longname": "LM retrieval parameter",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        LMRESULTS_JACRESNORM_attr={
            "Longname": "LM retrieval parameter",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        LMRESULTS_JACRESNORMNEXT_attr={
            "Longname": "LM retrieval parameter",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        LMRESULTS_PNORM_attr={
            "Longname": "LM retrieval parameter",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        LMRESULTS_ITERLIST_attr={
            "Longname": "LM retrieval iteration values for retrieval vector",
            "Units": "same as species",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        PROPAGATED_H2O_QA_attr={
            "Longname": "H2O quality",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        PROPAGATED_O3_QA_attr={
            "Longname": "O3 quality",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        PROPAGATED_TATM_QA_attr={
            "Longname": "TATM quality",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        RESIDUALNORMFINAL_attr={
            "Longname": "residual between model and data",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        RESIDUALNORMINITIAL_attr={
            "Longname": "initial residual between model and data",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        RADIANCERESIDUALMEAN_attr={
            "Longname": "Mean of the data minus model radiance",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        RADIANCERESIDUALRMS_attr={
            "Longname": "Standard deviation of model and data difference",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        RADIANCE_RESIDUAL_STDEV_CHANGE_attr={
            "Longname": "Initial minus final radiance residual standard deviation",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        # AT_LINE 1501 TOOLS/cdf_write_tes.pro
        FILTER_INDEX_attr={
            "Longname": "index of filters used in retrieval, where list is:  0=UV1,1=UV2, VIS,UVIS,NIR1,NIR2,SWIR1,SWIR2,SWIR3,SWIR4,TIR1,TIR2,TIR3,TIR4",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        RADIANCERESIDUALMEAN_FILTER_attr={
            "Longname": "Mean of the data minus model radiance in different filters",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        RADIANCERESIDUALRMSRELATIVECONTINUUM_FILTER_attr={
            "Longname": "Stdev of model and data difference divided by the 5% highest fit radiances in different filters",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        RADIANCE_CONTINUUM_FILTER_attr={
            "Longname": "Mean of the 5% highest fit radiances in different filters",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        RADIANCERESIDUALRMS_FILTER_attr={
            "Longname": "RMS of model and data difference in different filters",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        RADIANCERESIDUALSLOPE_FILTER_attr={
            "Longname": "Slope of:  [radiance / continuum (top 5% radiance) , (observed minus model radiance) divided by NESR], in different filters",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        RADIANCERESIDUALQUADRATIC_FILTER_attr={
            "Longname": "2nd order term of:  [radiance / continuum (top 5% radiance) , (observed minus model radiance) divided by NESR], in different filters",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        # new filter names
        # filters = ['ALL', 'UV1', 'UV2', 'VIS', 'UVIS', 'NIR1','NIR2','SWIR1','SWIR2','SWIR3','SWIR4','TIR1','TIR2','TIR3','TIR4']
        # Radiance Residual Mean + Filter
        RADIANCERESIDUALMEAN_ALL_attr={
            "Longname": "Mean of the model and data radiance difference (per species) in different filters",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        RADIANCERESIDUALMEAN_UV1_attr={
            "Longname": "Mean of the model and data radiance difference (per species) in different filters",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        RADIANCERESIDUALMEAN_UV2_attr={
            "Longname": "Mean of the model and data radiance difference (per species) in different filters",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        RADIANCERESIDUALMEAN_VIS_attr={
            "Longname": "Mean of the model and data radiance difference (per species) in different filters",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        RADIANCERESIDUALMEAN_UVIS_attr={
            "Longname": "Mean of the model and data radiance difference (per species) in different filters",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        RADIANCERESIDUALMEAN_NIR1_attr={
            "Longname": "Mean of the model and data radiance difference (per species) in different filters",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        RADIANCERESIDUALMEAN_NIR2_attr={
            "Longname": "Mean of the model and data radiance difference (per species) in different filters",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        RADIANCERESIDUALMEAN_SWIR1_attr={
            "Longname": "Mean of the model and data radiance difference (per species) in different filters",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        RADIANCERESIDUALMEAN_SWIR2_attr={
            "Longname": "Mean of the model and data radiance difference (per species) in different filters",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        RADIANCERESIDUALMEAN_SWIR3_attr={
            "Longname": "Mean of the model and data radiance difference (per species) in different filters",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        RADIANCERESIDUALMEAN_SWIR4_attr={
            "Longname": "Mean of the model and data radiance difference (per species) in different filters",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        RADIANCERESIDUALMEAN_TIR1_attr={
            "Longname": "Mean of the model and data radiance difference (per species) in different filters",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        RADIANCERESIDUALMEAN_TIR2_attr={
            "Longname": "Mean of the model and data radiance difference (per species) in different filters",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        RADIANCERESIDUALMEAN_TIR3_attr={
            "Longname": "Mean of the model and data radiance difference (per species) in different filters",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        RADIANCERESIDUALMEAN_TIR4_attr={
            "Longname": "Mean of the model and data radiance difference (per species) in different filters",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        # Radiance Residual RMS + Filter
        RADIANCERESIDUALRMS_ALL_attr={
            "Longname": "RMS of model and data difference in different filters",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        RADIANCERESIDUALRMS_UV1_attr={
            "Longname": "RMS of model and data difference in different filters",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        RADIANCERESIDUALRMS_UV2_attr={
            "Longname": "RMS of model and data difference in different filters",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        RADIANCERESIDUALRMS_VIS_attr={
            "Longname": "RMS of model and data difference in different filters",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        RADIANCERESIDUALRMS_UVIS_attr={
            "Longname": "RMS of model and data difference in different filters",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        RADIANCERESIDUALRMS_NIR1_attr={
            "Longname": "RMS of model and data difference in different filters",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        RADIANCERESIDUALRMS_NIR2_attr={
            "Longname": "RMS of model and data difference in different filters",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        RADIANCERESIDUALRMS_SWIR1_attr={
            "Longname": "RMS of model and data difference in different filters",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        RADIANCERESIDUALRMS_SWIR2_attr={
            "Longname": "RMS of model and data difference in different filters",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        RADIANCERESIDUALRMS_SWIR3_attr={
            "Longname": "RMS of model and data difference in different filters",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        RADIANCERESIDUALRMS_SWIR4_attr={
            "Longname": "RMS of model and data difference in different filters",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        RADIANCERESIDUALRMS_TIR1_attr={
            "Longname": "RMS of model and data difference in different filters",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        RADIANCERESIDUALRMS_TIR2_attr={
            "Longname": "RMS of model and data difference in different filters",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        RADIANCERESIDUALRMS_TIR3_attr={
            "Longname": "RMS of model and data difference in different filters",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        RADIANCERESIDUALRMS_TIR4_attr={
            "Longname": "RMS of model and data difference in different filters",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        # legacy filter name names
        RADIANCERESIDUALMEAN_2B1_attr={
            "Longname": "Mean of the model and data radiance difference (per species) in different filters",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        RADIANCERESIDUALMEAN_1B2_attr={
            "Longname": "Mean of the model and data radiance difference (per species) in different filters",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        RADIANCERESIDUALMEAN_2A1_attr={
            "Longname": "Mean of the model and data radiance difference (per species) in different filters",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        RADIANCERESIDUALMEAN_1A1_attr={
            "Longname": "Mean of the model and data radiance difference (per species) in different filters",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        RADIANCERESIDUALRMS_2B1_attr={
            "Longname": "RMS of model and data difference in different filters",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        RADIANCERESIDUALRMS_1B2_attr={
            "Longname": "RMS of model and data difference in different filters",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        RADIANCERESIDUALRMS_2A1_attr={
            "Longname": "RMS of model and data difference in different filters",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        RADIANCERESIDUALRMS_1A1_attr={
            "Longname": "RMS of model and data difference in different filters",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        SPECIESRETRIEVALCONVERGED_attr={
            "Longname": "Indicates whether the non-linear least squares solver converged to a minimum.True=1, False=0",
            "Units": "",
            "FillValue": -99,
            "MisingValue": -99,
        },
        DESERT_EMISS_QA_attr={
            "Longname": "Quality value comparing the retrieved emissivity at 1020 to look for silicate feature",
            "Units": "",
            "FillValue": -999.0,
            "MisingValue": -999.0,
        },
        PAN_DESERT_QA_attr={
            "Longname": "Quality value looking at the linearity of UWIS EMIS between 1040 to 1080 within 2 degrees of current location",
            "Units": "",
            "FillValue": -999.0,
            "MisingValue": -999.0,
        },
        H2O_RETRIEVAL_QA_attr={
            "Longname": "Quality flag propagated from H2O (may not be used in master quality)",
            "Units": "",
            "FillValue": -999.0,
            "MisingValue": -999.0,
        },
        H2O_PROPAGATED_QA_attr={
            "Longname": "Quality flag propagated from H2O (may not be used in master quality)",
            "Units": "",
            "FillValue": -999.0,
            "MisingValue": -999.0,
        },
        O3_RETRIEVAL_QA_attr={
            "Longname": "Quality flag propagated from O3 (may not be used in master quality)",
            "Units": "",
            "FillValue": -999.0,
            "MisingValue": -999.0,
        },
        O3_PROPAGATED_QA_attr={
            "Longname": "Quality flag propagated from O3 (may not be used in master quality)",
            "Units": "",
            "FillValue": -999.0,
            "MisingValue": -999.0,
        },
        TATM_RETRIEVAL_QA_attr={
            "Longname": "Quality flag propagated from atmospheric temperature",
            "Units": "",
            "FillValue": -999.0,
            "MisingValue": -999.0,
        },
        TATM_PROPAGATED_QA_attr={
            "Longname": "Quality flag propagated from atmospheric temperature",
            "Units": "",
            "FillValue": -999.0,
            "MisingValue": -999.0,
        },
        # AT_LINE 1584 TOOLS/cdf_write_tes.pro
        T700_attr={
            "Longname": "NCEP temperature at 700 hPa",
            "Units": "K",
            "FillValue": -999.0,
            "MisingValue": -999.0,
        },
        SURFACEEMISSMEAN_QA_attr={
            "Longname": "Quality value comparing the retrieved emissivity to the initial emissivity. Fill forocean and limb scenes. This will be fill forthe species HDO and H2O",
            "Units": "",
            "FillValue": -999.0,
            "MisingValue": -999.0,
        },
        SURFACEEMISSIONLAYER_QA_attr={
            "Longname": "Quality value comparing the atmospheric temperature to the surface temperature when ozone near the surface is elevated.This field will be fill for CO, CH4,HDO and most H2O",
            "Units": "",
            "FillValue": -999.0,
            "MisingValue": -999.0,
        },
        SURFACETEMPCONSTRAINT_attr={
            "Longname": "Surface temperature value used to constrain the retrieval (species independent)",
            "Units": "K",
            "FillValue": -999.0,
            "MisingValue": -999.0,
        },
        FMOZONEBANDFLUX_attr={
            "Longname": "Ozone band flux calculated from TES",
            "Units": "",
            "FillValue": -999.0,
            "MisingValue": -999.0,
        },
        L1BOZONEBANDFLUX_attr={
            "Longname": "Ozone band flux from radiances",
            "Units": "",
            "FillValue": -999.0,
            "MisingValue": -999.0,
        },
        SURFACETEMPDEGREESOFFREEDOM_attr={
            "Longname": "Surface temperature degrees of freedom",
            "Units": "",
            "FillValue": -999.0,
            "MisingValue": -999.0,
        },
        SURFACETEMPERROR_attr={
            "Longname": "Error in the retrieved Surface temperature",
            "Units": "K",
            "FillValue": -999.0,
            "MisingValue": -999.0,
        },
        SURFACETEMPINITIAL_attr={
            "Longname": "Initial surface temperature at the start of the retrieval process, currently taken from GMAO",
            "Units": "K",
            "FillValue": -999.0,
            "MisingValue": -999.0,
        },
        SURFACETEMPOBSERVATIONERROR_attr={
            "Longname": "Measurement + systematic + cross-state errors. The utility of the observation error is for comparisons with other measurements and for assimilation. The smoothing erroris accounted for when one applies the averaging kernel, so the observation error accounts for everything else.",
            "Units": "K",
            "FillValue": -999.0,
            "MisingValue": -999.0,
        },
        SURFACETEMPPRECISION_attr={
            "Longname": "Square-root of diagonal element of the measurement error covariance",
            "Units": "K",
            "FillValue": -999.0,
            "MisingValue": -999.0,
        },
        SURFACETEMPVSAPRIORI_QA_attr={
            "Longname": "Quality value comparing the surface temperature to a priori value",
            "Units": "K",
            "FillValue": -999.0,
            "MisingValue": -999.0,
        },
        SURFACETEMPERATURE_attr={
            "Longname": "surface temperature",
            "Units": "K",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        TOTALCOLUMNDENSITY_attr={
            "Longname": "Total column amount computed from the retrieved profile. For the Atmospheric Temperature, this will be a fill value",
            "Units": "molecules/m2",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        TOTALCOLUMNDENSITYERROR_attr={
            "Longname": "Error in total column amount computed from total error covarianceFor the Atmospheric Temperature Product this will be a fill value",
            "Units": "molecules/m2",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        TOTALCOLUMNDENSITYINITIAL_attr={
            "Longname": "Total column amount computed from the initial profile. For the Atmospheric Temperature Product this will be a fill value",
            "Units": "molecules/m2",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        TROPOMI_CLOUDFRACTIONCONSTRAINTVECTOR_attr={
            "Longname": "fraction of scene obscured by cloud prior",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        TROPOMI_CLOUDTOPPRESSURE_attr={
            "Longname": "",
            "Units": "hPa",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        TROPOMI_SURFACEALBEDOBAND1_attr={
            "Longname": "single value for band 1",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        TROPOMI_SURFACEALBEDOBAND1CONSTRAINTVECTOR_attr={
            "Longname": "",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        TROPOMI_SURFACEALBEDOBAND2_attr={
            "Longname": "linear fit",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        TROPOMI_SURFACEALBEDOBAND2CONSTRAINTVECTOR_attr={
            "Longname": "prior",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        TROPOMI_SURFACEALBEDOBAND3_attr={
            "Longname": "linear fit",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        TROPOMI_SURFACEALBEDOBAND7_attr={
            "Longname": "linear fit",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        TROPOMI_SURFACEALBEDOBAND3CONSTRAINTVECTOR_attr={
            "Longname": "prior",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        TROPOMI_SURFACEALBEDOBAND7CONSTRAINTVECTOR_attr={
            "Longname": "prior",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        TROPOMI_SURFACEALBEDOSLOPEBAND2_attr={
            "Longname": "",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        TROPOMI_SURFACEALBEDOSLOPEBAND2CONSTRAINTVECTOR_attr={
            "Longname": "",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        TROPOMI_SURFACEALBEDOSLOPEBAND3_attr={
            "Longname": "",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        TROPOMI_SURFACEALBEDOSLOPEBAND7_attr={
            "Longname": "",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        TROPOMI_SURFACEALBEDOSLOPEBAND3CONSTRAINTVECTOR_attr={
            "Longname": "",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        TROPOMI_SURFACEALBEDOSLOPEBAND7CONSTRAINTVECTOR_attr={
            "Longname": "",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        TROPOMI_SURFACEALBEDOSLOPEORDER2BAND2_attr={
            "Longname": "",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        TROPOMI_SURFACEALBEDOSLOPEBAND2ORDER2CONSTRAINTVECTOR_attr={
            "Longname": "",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        TROPOMI_SURFACEALBEDOSLOPEORDER2BAND3_attr={
            "Longname": "",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        TROPOMI_SURFACEALBEDOSLOPEORDER2BAND7_attr={
            "Longname": "",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        TROPOMI_SURFACEALBEDOSLOPEORDER2BAND3CONSTRAINTVECTOR_attr={
            "Longname": "",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        TROPOMI_SURFACEALBEDOSLOPEORDER2BAND7CONSTRAINTVECTOR_attr={
            "Longname": "",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        TROPOMI_SOLARSHIFTBAND1_attr={
            "Longname": "",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        TROPOMI_SOLARSHIFTBAND2_attr={
            "Longname": "",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        TROPOMI_SOLARSHIFTBAND3_attr={
            "Longname": "",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        TROPOMI_SOLARSHIFTBAND7_attr={
            "Longname": "",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        TROPOMI_RADIANCESHIFTBAND1_attr={
            "Longname": "",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        TROPOMI_RADIANCESHIFTBAND2_attr={
            "Longname": "",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        TROPOMI_RADIANCESHIFTBAND3_attr={
            "Longname": "",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        TROPOMI_RADIANCESHIFTBAND7_attr={
            "Longname": "",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        TROPOMI_RADSQUEEZEBAND1_attr={
            "Longname": "",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        TROPOMI_RADSQUEEZEBAND2_attr={
            "Longname": "",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        TROPOMI_RADSQUEEZEBAND3_attr={
            "Longname": "",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        TROPOMI_RADSQUEEZEBAND7_attr={
            "Longname": "",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        TROPOMI_RINGSFBAND1_attr={
            "Longname": "",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        TROPOMI_RINGSFBAND2_attr={
            "Longname": "",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        TROPOMI_RINGSFBAND3_attr={
            "Longname": "",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        TROPOMI_RINGSFBAND7_attr={
            "Longname": "",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        TROPOMI_TEMPSHIFTBAND3_attr={
            "Longname": "",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        TROPOMI_TEMPSHIFTBAND7_attr={
            "Longname": "",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        COLUMN_attr={
            "Longname": "Column amounts computed from the retrieved profile. See columnPressureMin, columnPressureMax for pressure ranges",
            "Units": "molecules/cm2",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        # AT_LINE 1669 TOOLS/cdf_write_tes.pro
        COLUMN_AIR_attr={
            "Longname": "Column amounts computed from air. See columnPressureMin, columnPressureMax for pressure ranges",
            "Units": "molecules/cm2",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        COLUMN_DOFS_attr={
            "Longname": "DOFS for Column amounts for partial/full columns. See columnPressureMin, columnPressureMax for pressure ranges",
            "Units": "",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        COLUMN_ERROR_attr={
            "Longname": "Errors for column amounts. See columnPressureMin, columnPressureMax for pressure ranges",
            "Units": "molecules/cm2",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        COLUMN_INITIAL_attr={
            "Longname": "Column amounts computed for Initial. See columnPressureMin, columnPressureMax for pressure ranges",
            "Units": "molecules/cm2",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        COLUMN_PRESSUREMAX_attr={
            "Longname": "Bounding pressure for column values",
            "Units": "hPa",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        COLUMN_PRESSUREMIN_attr={
            "Longname": "Bounding pressure for column values",
            "Units": "hPa",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        COLUMN_PRIOR_attr={
            "Longname": "Column amounts computed for prior. See columnPressureMin, columnPressureMax for pressure ranges",
            "Units": "molecules/cm2",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        COLUMN_TRUE_attr={
            "Longname": "Column amounts computed for true (if real data usually set to initial guess). See columnPressureMin, columnPressureMax for pressure ranges",
            "Units": "molecules/cm2",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        COLUMN_UNITS_attr={
            "Longname": "multiplier on column values, e.g. *1e-9",
            "Units": "molecules/cm2",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        MICROWINDOW_attr={
            "Longname": "Microwindow start, end frequency",
            "Units": "cm-1",
            "FillValue": -999.0,
            "MisingValue": -999.0,
        },
        MICROWINDOW_INSTRUMENT_attr={
            "Longname": "Microwindow instrument",
            "Units": "",
            "FillValue": "-",
            "MisingValue": "-",
        },
        MICROWINDOW_SPECIES_attr={
            "Longname": "Microwindow species",
            "Units": "",
            "FillValue": "-",
            "MisingValue": "-",
        },
        TOTALERROR_attr={
            "Longname": "Square-root of diagonal elements of output total error covariance (i.e. smoothing, systematic, and measurement error). For atmospheric temperature, it represents the error in (K). For Atmospheric Species, positive error bar=exp(ln(vmr)+err)-vmr, negative error bar=vmr-exp(ln(vmr)-error)",
            "Units": "ln(vmr)",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        TOTALERROR_FM_attr={
            "Longname": "Square-root of diagonal elements of output total error covariance (i.e. smoothing, systematic, and measurement error). For atmospheric temperature, it represents the error in (K). For Atmospheric Species, positive error bar=exp(ln(vmr)+err)-vmr, negative error bar=vmr-exp(ln(vmr)-error).  On fm grid.",
            "Units": "ln(vmr)",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        TOTALERRORCOVARIANCE_attr={
            "Longname": "Sum of smoothing, systematic, and measurement error. For atmospheric temperature, it represents the covariance of the error of temperature. For Atmospheric Species, it is the covariance of the error of ln(vmr)",
            "Units": "ln(vmr)^2",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        YYYYMMDD_attr={
            "Longname": "Human readable UTC time at location",
            "Units": "time as YYYYMMDD",
            "Example": "20060805",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        UT_HOUR_attr={
            "Longname": "UTC hour at location.  fraction corresponds to minutes, seconds",
            "Units": "hours",
            "Example": "19.44000",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        RUN_attr={"Longname": "Associated TES original RUNID"},
        DAYNIGHTFLAG_attr={
            "Longname": "day and night flag (1=day, 0=night)",
            "Units": "",
            "FillValue": -99,
            "MisingValue": -99,
        },
        # AT_LINE 1752 TOOLS/cdf_write_tes.pro
        DOMINANTSURFACETYPE_attr={
            "Longname": "Dominant surface type",
            "FullDescription": "see http://tes.jpl.nasa.gov/uploadedfiles/TES_DPS_V11.8.pdf",
            "Units": "",
            "FillValue": -99,
            "MisingValue": -99,
        },
        LATITUDE_attr={
            "Longname": "latitude",
            "Units": "degrees_north",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        LATITUDE_FOOTPRINT_1_attr={
            "Longname": "latitude for footprint bounding point 1",
            "Units": "degrees_north",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        LATITUDE_FOOTPRINT_2_attr={
            "Longname": "latitude for footprint bounding point 2",
            "Units": "degrees_north",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        LATITUDE_FOOTPRINT_3_attr={
            "Longname": "latitude for footprint bounding point 3",
            "Units": "degrees_north",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        LATITUDE_FOOTPRINT_4_attr={
            "Longname": "latitude for footprint bounding point 4",
            "Units": "degrees_north",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        LOCALSOLARTIME_attr={
            "Longname": "local solar time at location",
            "Units": "hours_of_day.fraction_of_hours_of_day",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        LONGITUDE_attr={
            "Longname": "longitude",
            "Units": "degrees_east",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        LONGITUDE_FOOTPRINT_1_attr={
            "Longname": "longitude for footprint bounding point 1",
            "Units": "degrees_east",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        LONGITUDE_FOOTPRINT_2_attr={
            "Longname": "llongitude for footprint bounding point 2",
            "Units": "degrees_east",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        LONGITUDE_FOOTPRINT_3_attr={
            "Longname": "llongitude for footprint bounding point 3",
            "Units": "degrees_east",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        LONGITUDE_FOOTPRINT_4_attr={
            "Longname": "longitude for footprint bounding point 4",
            "Units": "degrees_east",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        SCAN_attr={
            "Longname": "scan number within a sequence",
            "Units": "",
            "FillValue": -999,
            "MisingValue": -999,
        },
        SEQUENCE_attr={
            "Longname": "sequence number within a run",
            "Units": "",
            "FillValue": -999,
            "MisingValue": -999,
        },
        SOLARZENITHANGLE_attr={
            "Longname": "Solar zenith relative to the local zenith at the spacecraft",
            "Units": "decimal degrees",
            "Calendar": "julian",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        SURFACEELEVSTANDARDDEVIATION_attr={
            "Longname": "standard deviation of average elevation over footprint",
            "Units": "meters",
            "FillValue": -999.0,
            "MisingValue": -999.0,
        },
        SURFACETYPEFOOTPRINT_attr={
            "Longname": "surface type footprint",
            "FullDescription": "1=freshwater,2=saltwater,3=land,4=mixed",
            "Units": "",
            "FillValue": -99,
            "MisingValue": -99,
        },
        TIME_attr={
            "Longname": "Julian date",
            "Units": "seconds since some reference (1993-01-01 00':00':00)",
            "Calendar": "julian",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        # AT_LINE 1846 TOOLS/cdf_write_tes.pro
        # fields for TES/OMI
        O3_CCURVE_TESOMI_attr={
            "Longname": "field for TES/OMI",
            "Units": "",
            "FillValue": -999,
            "MisingValue": "",
        },
        OMI_CLOUDFRACTION_INITIAL_attr={
            "Longname": "field for TES/OMI",
            "Units": "",
            "FillValue": -999,
            "MisingValue": "",
        },
        OMI_NRADWAV_attr={
            "Longname": "field for TES/OMI",
            "Units": "",
            "FillValue": -999,
            "MisingValue": "",
        },
        OMI_NRADWAVCONSTRAINTVECTOR_attr={
            "Longname": "field for TES/OMI",
            "Units": "",
            "FillValue": -999,
            "MisingValue": "",
        },
        OMI_NRADWAV_INITIAL_attr={
            "Longname": "field for TES/OMI",
            "Units": "",
            "FillValue": -999,
            "MisingValue": "",
        },
        OMI_RADIANCERESIDUALMEAN_attr={
            "Longname": "field for TES/OMI",
            "Units": "",
            "FillValue": -999,
            "MisingValue": "",
        },
        OMI_RADIANCERESIDUALMEAN_QA_attr={
            "Longname": "field for TES/OMI",
            "Units": "",
            "FillValue": -999,
            "MisingValue": "",
        },
        OMI_RADIANCERESIDUALRMS_attr={
            "Longname": "field for TES/OMI",
            "Units": "",
            "FillValue": -999,
            "MisingValue": "",
        },
        OMI_RADIANCERESIDUALRMS_QA_attr={
            "Longname": "field for TES/OMI",
            "Units": "",
            "FillValue": -999,
            "MisingValue": "",
        },
        OMI_RINGSF_attr={
            "Longname": "field for TES/OMI",
            "Units": "",
            "FillValue": -999,
            "MisingValue": "",
        },
        OMI_RINGSFCONSTRAINTVECTOR_attr={
            "Longname": "field for TES/OMI",
            "Units": "",
            "FillValue": -999,
            "MisingValue": "",
        },
        OMI_RINGSF_INITIAL_attr={
            "Longname": "field for TES/OMI",
            "Units": "",
            "FillValue": -999,
            "MisingValue": "",
        },
        OMI_SURFALB_attr={
            "Longname": "field for TES/OMI",
            "Units": "",
            "FillValue": -999,
            "MisingValue": "",
        },
        OMI_SURFALBCONSTRAINTVECTOR_attr={
            "Longname": "field for TES/OMI",
            "Units": "",
            "FillValue": -999,
            "MisingValue": "",
        },
        OMI_SURFALBSLOPE_attr={
            "Longname": "field for TES/OMI",
            "Units": "",
            "FillValue": -999,
            "MisingValue": "",
        },
        OMI_SURFALBSLOPECONSTRAINTVECTOR_attr={
            "Longname": "field for TES/OMI",
            "Units": "",
            "FillValue": -999,
            "MisingValue": "",
        },
        OMI_SURFALBSLOPE_INITIAL_attr={
            "Longname": "field for TES/OMI",
            "Units": "",
            "FillValue": -999,
            "MisingValue": "",
        },
        OMI_SURFALB_INITIAL_attr={
            "Longname": "field for TES/OMI",
            "Units": "",
            "FillValue": -999,
            "MisingValue": "",
        },
        OMI_SURFALB_QA_attr={
            "Longname": "field for TES/OMI",
            "Units": "",
            "FillValue": -999,
            "MisingValue": "",
        },
        TES_SCANAVERAGEDCOUNT_attr={
            "Longname": "field for TES/OMI",
            "Units": "",
            "FillValue": -999,
            "MisingValue": "",
        },
        TES_SURFACETEMPINITIAL_attr={
            "Longname": "field for TES/OMI",
            "Units": "",
            "FillValue": -999,
            "MisingValue": "",
        },
        OMI_CLOUD_PRESSURE_attr={
            "Longname": "field for TES/OMI",
            "Units": "",
            "FillValue": -999,
            "MisingValue": "",
        },
        OMI_EARTHSUNDISTANCE_attr={
            "Longname": "field for TES/OMI",
            "Units": "",
            "FillValue": -999,
            "MisingValue": "",
        },
        OMI_LAT_attr={
            "Longname": "field for TES/OMI",
            "Units": "",
            "FillValue": -999,
            "MisingValue": "",
        },
        OMI_LINE_attr={
            "Longname": "field for TES/OMI",
            "Units": "",
            "FillValue": -999,
            "MisingValue": "",
        },
        OMI_LON_attr={
            "Longname": "field for TES/OMI",
            "Units": "",
            "FillValue": -999,
            "MisingValue": "",
        },
        OMI_ORBIT_attr={
            "Longname": "field for TES/OMI",
            "Units": "",
            "FillValue": -999,
            "MisingValue": "",
        },
        OMI_ORBITPHASE_attr={
            "Longname": "field for TES/OMI",
            "Units": "",
            "FillValue": -999,
            "MisingValue": "",
        },
        OMI_PIXEL_attr={
            "Longname": "field for TES/OMI",
            "Units": "",
            "FillValue": -999,
            "MisingValue": "",
        },
        OMI_PIXEL_CLOUD_FRACTION_attr={
            "Longname": "field for TES/OMI",
            "Units": "",
            "FillValue": -999,
            "MisingValue": "",
        },
        OMI_SOLARAZIMUTHANGLE_attr={
            "Longname": "field for TES/OMI",
            "Units": "",
            "FillValue": -999,
            "MisingValue": "",
        },
        OMI_SOLARZENITHANGLE_attr={
            "Longname": "field for TES/OMI",
            "Units": "",
            "FillValue": -999,
            "MisingValue": "",
        },
        OMI_TAI_TIME_attr={
            "Longname": "field for TES/OMI",
            "Units": "",
            "FillValue": -999,
            "MisingValue": "",
        },
        OMI_UTC_TIME_attr={
            "Longname": "field for TES/OMI",
            "Units": "",
            "FillValue": -999,
            "MisingValue": "",
        },
        OMI_UV2_TERRAINHEIGHT_attr={
            "Longname": "field for TES/OMI",
            "Units": "",
            "FillValue": -999,
            "MisingValue": "",
        },
        OMI_VIEWINGAZIMUTHANGLE_attr={
            "Longname": "field for TES/OMI",
            "Units": "",
            "FillValue": -999,
            "MisingValue": "",
        },
        OMI_VIEWINGZENITHANGLE_attr={
            "Longname": "field for TES/OMI",
            "Units": "",
            "FillValue": -999,
            "MisingValue": "",
        },
        OMI_XTRACKQUALITYFLAGS_attr={
            "Longname": "field for TES/OMI",
            "Units": "",
            "FillValue": -999,
            "MisingValue": "",
        },
        TES_DOMINANTSURFACETYPE_attr={
            "Longname": "field for TES/OMI",
            "Units": "",
            "FillValue": -999,
            "MisingValue": "",
        },
        TES_LATITUDE_attr={
            "Longname": "field for TES/OMI",
            "Units": "",
            "FillValue": -999,
            "MisingValue": "",
        },
        TES_LONGITUDE_attr={
            "Longname": "field for TES/OMI",
            "Units": "",
            "FillValue": -999,
            "MisingValue": "",
        },
        TES_OMI_DISTANCE_attr={
            "Longname": "field for TES/OMI",
            "Units": "",
            "FillValue": -999,
            "MisingValue": "",
        },
        TES_OMI_MATCHED_attr={
            "Longname": "field for TES/OMI",
            "Units": "",
            "FillValue": -999,
            "MisingValue": "",
        },
        TES_OMI_TIME_DIFFERENCE_attr={
            "Longname": "field for TES/OMI",
            "Units": "",
            "FillValue": -999,
            "MisingValue": "",
        },
        TES_SURFACETYPEFOOTPRINT_attr={
            "Longname": "field for TES/OMI",
            "Units": "",
            "FillValue": -999,
            "MisingValue": "",
        },
        TES_TAI_TIME_attr={
            "Longname": "field for TES/OMI",
            "Units": "",
            "FillValue": -999,
            "MisingValue": "",
        },
        TES_UTC_TIME_attr={
            "Longname": "field for TES/OMI",
            "Units": "",
            "FillValue": -999,
            "MisingValue": "",
        },
        # GRID_FILTER_attr={'Longname':'Things separated by filter', 'Units':''},
        # GRID_COLUMN_attr={'Longname':'Full column, trop column, upper trop, lower trop', 'Units':''},
        GRID_PRESSURE_FM_attr={
            "Longname": "full pressure grid used in forward model calculation.  Actual pressure grid varies depending on surface pressure.",
            "Units": "hPa",
        },
        # GRID_PRESSURE_attr={'Longname':'full retrieval pressure grid.  Actual pressure grid varies depending on surface pressure.', 'Units':''},
        # GRID_PRESSURE_COMPOSITE_attr={'Longname':'full retrieval pressure grid for stacked HDO and H2O vectors.  Actual pressure grid varies depending on surface pressure', 'Units':''},
        # GRID_RTVMR_LEVELS_attr={'Longname':'number of rtvmr trop. pressure levels', 'Units':''},
        # GRID_RTVMR_MAP_attr={'Longname':'number of rtvmr map levels', 'Units':''},
        # GRID_TARGETS_attr={'Longname':'number of individual points sampled', 'Units':''},
        # GRID_CT_LEVEL_attr={'Longname':'CarbonTracker pressure levels', 'Units':'hPa'},
        # GRID_CT_LAYER_attr={'Longname':'CarbonTracker pressure layers', 'Units':'hPa'},
        # GRID_NCEP_attr={'Longname':'pressure grid for NCEP pressures', 'Units':'hPa'},
        # GRID_CLOUD_attr={'Longname':'wavenumber grid for Cloud effective optical depth', 'Units':'cm-1'},
        # GRID_EMISSIVITY_attr={'Longname':'wavenumber grid for emissivity', 'Units':'cm-1'},
        EMISSIVITY_attr={"Longname": "surface emissivity", "Units": ""},
        EMISSIVITY_INITIAL_attr={
            "Longname": "surface emissivity initial guess",
            "Units": "",
        },
        EMISSIVITY_CONSTRAINT_attr={
            "Longname": "surface emissivity constraint",
            "Units": "",
        },
        EMISSIVITY_WAVENUMBER_attr={
            "Longname": "surface emissivity wavenumber",
            "Units": "cm-1",
        },
        EMISSIVITY_ERROR_attr={
            "Longname": "surface emissivity error",
            "Units": "",
            "FillValue": -999.0,
            "MisingValue": -999.0,
        },
        EMISSIVITY_OFFSET_DISTANCE_attr={
            "Longname": "surface emissivity offset distance",
            "Units": "km",
            "FullDescription": "The distance between the measurement center and the selected grid cell in the surface emissivity database.",
        },
        NATIVE_HSR_EMIS_WAVENUMBER_attr={
            "Longname": "surface emissivity wavenumber at native resolution",
            "Units": "cm-1",
            "Comment": "Right-padded with fill values as needed to ensure all soundings using the same emissivity source have the same native dimension length",
        },
        NATIVE_HSR_EMISSIVITY_INITIAL_attr={
            "Longname": "surface emissivity initial guess at native resolution",
            "Units": "",
            "Comment": "Right-padded with fill values as needed to ensure all soundings using the same emissivity source have the same native dimension length",
        },
        # GRID_ITERS_attr={'Longname':'iterations', 'Units':''},
        # GRID_ITERLIST_attr={'Longname':'retrieval iterations', 'Units':''},
        # GRID_2_attr={'Longname':'omi pixel dependent', 'Units':''},
        # GRID_3_attr={'Longname':'omi pixel dependent', 'Units':''},
        # AT_LINE 1931 TOOLS/cdf_write_tes.pro
        CT_CO2_attr={"Longname": "CarbonTracker CO2 layer value", "Units": "ppm"},
        CT_CO2_AK_attr={
            "Longname": "CarbonTracker CO2 value with AK applied",
            "Units": "ppm",
        },
        CT_PRESSURE_attr={
            "Longname": "CarbonTracker Pressure level value",
            "Units": "hPa",
        },
        CT_LATITUDE_attr={"Longname": "CarbonTracker latitude.", "Units": "degrees"},
        CT_LONGITUDE_attr={"Longname": "CarbonTracker longitude.", "Units": "degrees"},
        CT_YEARFLOAT_attr={"Longname": "CarbonTracker yearfloat", "Units": "year"},
        SONDE_attr={"Longname": "sonde match, if exists", "Units": ""},
        SONDEAK_attr={
            "Longname": "sonde match, if exists, with AK applied",
            "Units": "",
        },
        NCEP_PRESSURE_attr={"Longname": "NCEP pressure", "Units": "ppm"},
        NCEP_TEMPERATURE_attr={"Longname": "NCEP temperature", "Units": "K"},
        NCEP_TEMPERATURESURFACE_attr={
            "Longname": "NCEP temperature at surface",
            "Units": "K",
        },
        CH4_EVS_attr={
            "Longname": "Size of first 10 eVs for CH4",
            "Units": "",
            "FillValue": -999.0,
            "MisingValue": -999.0,
        },
        CH4_EVRATIO_attr={
            "Longname": "ratio of 0th over first 10 eVs for CH4",
            "Units": "",
            "FillValue": -999.0,
            "MisingValue": -999.0,
        },
        H2O_H2O_CORR_QA_attr={
            "Longname": "Water consistency quality assurance",
            "Units": "",
            "FillValue": -999.0,
            "MisingValue": -999.0,
        },
        H2O_HDO_CORR_QA_attr={
            "Longname": "Water consistency quality assurance",
            "Units": "",
            "FillValue": -999.0,
            "MisingValue": -999.0,
        },
        KDOTDL_QA_attr={
            "Longname": "Jacobian dotted into the radiance residual quality assurance",
            "Units": "",
            "FillValue": -999.0,
            "MisingValue": -999.0,
        },
        LDOTDL_QA_attr={
            "Longname": "Radiance dotted into the radiance residual quality assurance",
            "Units": "",
            "FillValue": -999.0,
            "MisingValue": -999.0,
        },
        LDOTDNCEP_TEMPERATURE_attr={
            "Longname": "CarbonTracker CO2 value with AK applied",
            "Units": "ppm",
        },
        L_QA_attr={
            "Longname": "Radiance dotted into the radiance residual quality assurance",
            "Units": "",
            "FillValue": -999.0,
            "MisingValue": -999.0,
        },
        SURFACETEMPVSATMTEMP_QA_attr={
            "Longname": "Quality value equal to the surface temperature minus the atmospheric temperature closest to the surface",
            "Units": "",
            "FillValue": -999.0,
            "MisingValue": -999.0,
        },
        QUALITY_attr={
            "Longname": "Retrieval quality set from lite update, 1=good, 0=bad",
            "Units": "",
            "FillValue": -999.0,
            "MisingValue": -999.0,
        },
        QUALITYAGGREGATE_attr={
            "Longname": "Retrieval quality for monthly average centered around this point, 1=good",
            "Units": "",
            "FillValue": -999.0,
            "MisingValue": -999.0,
        },
        ORIGINAL_QUALITY_attr={
            "Longname": "Retrieval quality, 1=good.  If present, this is the quality in the L2 HDF products, and the quality was updated for the lite products.",
            "Units": "",
            "FillValue": -999.0,
            "MisingValue": -999.0,
        },
        TROPOPAUSEPRESSURE_attr={
            "Longname": "Pressure between the troposphere and stratosphere used to calculate the tropospheric column, from GMAO",
            "Units": "hPa",
            "FillValue": -999.0,
            "MisingValue": -999.0,
        },
        BORESIGHTAZIMUTH_attr={
            "Longname": "boresight (LOS) azimuth angle relative to the local north at SC",
            "Units": "Decimal degrees",
            "FillValue": -999.0,
            "MisingValue": -999.0,
        },
        BORESIGHTNADIRANGLE_attr={
            "Longname": "boresight (LOS) nadir angle relative to local nadir at SC",
            "Units": "Decimal degrees",
            "FillValue": -999.0,
            "MisingValue": -999.0,
        },
        # TODO: These need to be upper case or they will not be selected, e.g. POINTINGANGLE_AIRS_attr
        pointingangle_airs_attr={
            "Longname": "AIRS boresight (LOS) nadir angle relative to local nadir at SC",
            "Units": "Decimal degrees",
            "FillValue": -999.0,
            "MisingValue": -999.0,
        },
        pointingangle_omi_attr={
            "Longname": "OMI boresight (LOS) nadir angle relative to local nadir at SC",
            "Units": "Decimal degrees",
            "FillValue": -999.0,
            "MisingValue": -999.0,
        },
        pointingangle_tes_attr={
            "Longname": "TES boresight (LOS) nadir angle relative to local nadir at SC",
            "Units": "Decimal degrees",
            "FillValue": -999.0,
            "MisingValue": -999.0,
        },
        pointingangle_tropomi_attr={
            "Longname": "TROPOMI boresight (LOS) nadir angle relative to local nadir at SC",
            "Units": "Decimal degrees",
            "FillValue": -999.0,
            "MisingValue": -999.0,
        },
        TGT_SPACECRAFTAZIMUTH_attr={
            "Longname": "boresight (LOS) azimuth angle relative to the local north at the target geolocation.",
            "Units": "Decimal degrees (east of north)",
            "FillValue": -999.0,
            "MisingValue": -999.0,
        },
        TGT_SPACECRAFTZENITH_attr={
            "Longname": "boresight (LOS) zenith angle relative to the local zenith at the target geolocation.",
            "Units": "Decimal degrees",
            "FillValue": -999.0,
            "MisingValue": -999.0,
        },
        YEARFLOAT_attr={
            "Longname": "year plus fraction of the year",
            "Units": "",
            "FillValue": -999.0,
            "MisingValue": -999.0,
        },
        ORIGINAL_SPECIES_attr={
            "Longname": "Uncorrected retrieved values",
            "Units": "(K) for atmospheric temperature, volume mixing ratio (dry) for atmospheric species",
            "FillValue": -999.0,
            "MisingValue": -999.0,
        },
        ORIGINAL_SPECIES_HDO_attr={
            "Longname": "Uncorrected retrieved values for HDO",
            "Units": "volume mixing ratio (dry)",
            "FillValue": -999.0,
            "MisingValue": -999.0,
        },
        ORIGINAL_SPECIES_N2O_attr={
            "Longname": "Uncorrected retrieved values for N2O",
            "Units": "volume mixing ratio (dry)",
            "FillValue": -999.0,
            "MisingValue": -999.0,
        },
        ORIGINAL_CONSTRAINTVECTOR_N2O_attr={
            "Longname": "N2O constraint vector used in retrieval",
            "Units": "VMR",
            "FillValue": -999.0,
            "MisingValue": -999.0,
        },
        ORIGINAL_COLUMN750_attr={
            "Longname": "Column above 750 mb for original CH4",
            "Units": "VMR (ppb)",
            "FillValue": -999.0,
            "MisingValue": -999.0,
        },
        ORIGINAL_SPECIES_FM_attr={
            "Longname": "Uncorrected retrieved values, fm grid",
            "Units": "(K) for atmospheric temperature, volume mixing ratio (dry) for atmospheric species",
            "FillValue": -999.0,
            "MisingValue": -999.0,
        },
        ORIGINAL_SPECIES_FM_HDO_attr={
            "Longname": "Uncorrected retrieved values for HDO, fm grid",
            "Units": "volume mixing ratio (dry)",
            "FillValue": -999.0,
            "MisingValue": -999.0,
        },
        ORIGINAL_SPECIES_N2O_FM_attr={
            "Longname": "Uncorrected retrieved values for N2O, fm grid",
            "Units": "volume mixing ratio (dry)",
            "FillValue": -999.0,
            "MisingValue": -999.0,
        },
        ORIGINAL_CONSTRAINTVECTOR_N2O_FM_attr={
            "Longname": "N2O constraint vector used in retrieval, fm grid",
            "Units": "VMR",
            "FillValue": -999.0,
            "MisingValue": -999.0,
        },
        INITIAL_attr={
            "Longname": "Initial vmr data or temperature data (for retrieved temperature) used in the retrieval",
            "Units": "VMR or K",
            "FillValue": -999.0,
            "MisingValue": -999.0,
        },
        TATM_SPECIES_attr={
            "Longname": "TATM profile result",
            "Units": "K",
            "FillValue": -999.0,
            "MisingValue": -999.0,
        },
        TATM_attr={
            "Longname": "TATM profile result",
            "Units": "K",
            "FillValue": -999.0,
            "MisingValue": -999.0,
        },
        TATM_DEVIATION_attr={
            "Longname": "TATM result maximum deviation from .constraint vector",
            "Units": "K",
            "FillValue": -999.0,
            "MisingValue": -999.0,
        },
        TATM_CONSTRAINTVECTOR_attr={
            "Longname": "TATM profile constraint vector",
            "Units": "K",
            "FillValue": -999.0,
            "MisingValue": -999.0,
        },
        H2O_SPECIES_attr={
            "Longname": "H2O profile result",
            "Units": "VMR",
            "FillValue": -999.0,
            "MissingValue": -999.0,
        },
        H2O_CONSTRAINTVECTOR_attr={
            "Longname": "H2O profile constraint vector",
            "Units": "VMR",
            "FillValue": -999.0,
            "MissingValue": -999.0,
        },
        H2O_attr={
            "Longname": "H2O profile result",
            "Units": "VMR",
            "FillValue": -999.0,
            "MisingValue": -999.0,
        },
        HDO_attr={
            "Longname": "HDO profile result",
            "Units": "VMR",
            "FillValue": -999.0,
            "MisingValue": -999.0,
        },
        HDO_H2O_attr={
            "Longname": "HDO_H2O profile result.  HDO starts at index 0 and H2O starts at index 17",
            "Units": "VMR",
            "FillValue": -999.0,
            "MisingValue": -999.0,
        },
        INITIAL_FM_attr={
            "Longname": "Initial vmr data or temperature data (for retrieved temperature) used in the retrieval on fm grid",
            "Units": "VMR or K",
            "FillValue": -999.0,
            "MisingValue": -999.0,
        },
        TRUE_attr={
            "Longname": "true profile",
            "Units": "VMR or K",
            "FillValue": -999.0,
            "MissingValue": -999.0,
        },
        TRUE_AK_attr={
            "Longname": "true profile with averaging kernel applied, e.g. xa + A@(xt-xa)",
            "Units": "VMR or K",
            "FillValue": -999.0,
            "MissingValue": -999.0,
        },
        N2O_SPECIES_attr={
            "Longname": "retrieved N2O",
            "Units": "VMR",
            "FillValue": -999.0,
            "MisingValue": -999.0,
        },
        N2O_SPECIES_FM_attr={
            "Longname": "retrieved N2O",
            "Units": "VMR",
            "FillValue": -999.0,
            "MisingValue": -999.0,
        },
        # AT_LINE 2008 TOOLS/cdf_write_tes.pro
        N2O_CONSTRAINTVECTOR_attr={
            "Longname": "N2O constraint vector",
            "Units": "VMR",
            "FillValue": -999.0,
            "MisingValue": -999.0,
        },
        N2O_CONSTRAINTVECTOR_FM_attr={
            "Longname": "N2O constraint vector, fm grid",
            "Units": "VMR",
            "FillValue": -999.0,
            "MisingValue": -999.0,
        },
        N2O_DOFS_attr={
            "Longname": "degrees of freedom for N2O",
            "Units": "()",
            "FillValue": -999.0,
            "MisingValue": -999.0,
        },
        N2O_AVERAGINGKERNEL_attr={
            "Longname": "N2O Averaging Kernel",
            "Units": "",
            "FillValue": -999.0,
            "MisingValue": -999.0,
        },
        N2O_OBSERVATIONERRORCOVARIANCE_attr={
            "Longname": "N2O Observation Error Covariance",
            "Units": "ln(vmr)^2 or K^2",
            "FillValue": -999.0,
            "MisingValue": -999.0,
        },
        SPECIES_N2OCORRECTED_attr={
            "Longname": "CH4 corrected by N2O, using log(CH4_corr)=log(CH4) + log(N2O_constraintvector) - log(N2O)",
            "Units": "",
            "FillValue": -999.0,
            "MisingValue": -999.0,
        },
        SPECIES_N2OCORRECTED_FM_attr={
            "Longname": "CH4 corrected by N2O, using log(CH4_corr)=log(CH4) + log(N2O_constraintvector) - log(N2O), fm grid",
            "Units": "",
            "FillValue": -999.0,
            "MisingValue": -999.0,
        },
        DELTA_P_attr={
            "Longname": "Surface pressure difference from prior",
            "Units": "hPa",
            "FillValue": -999.0,
            "MisingValue": -999.0,
        },
        CO2_GRAD_DEL_attr={
            "Longname": "Change (between the retrieved profile and the prior profile) of the co2 dry air mole fraction difference from the surface (level 0) minus that at level 7 (0.631579 Psurf)",
            "Units": "ppm",
            "FillValue": -999.0,
            "MisingValue": -999.0,
        },
        DELTA_T_attr={
            "Longname": "Atmospheric temperature difference from prior near 750 hPa",
            "Units": "K",
            "FillValue": -999.0,
            "MisingValue": -999.0,
        },
        NIR_WINDSPEED_attr={
            "Longname": "NIR windspeed, usually only used over water",
            "Units": "m/s",
            "FillValue": -999.0,
            "MisingValue": -999.0,
        },
        NIR_AEROD_attr={
            "Longname": "Aerosol optical depth for types: total, ice_cloud, wc, DU, SO, strat, oc, SS, BC",
            "Units": "optical depth",
            "FillValue": -999.0,
            "MisingValue": -999.0,
        },
        NIR_AERP_attr={
            "Longname": "Aerosol pressure/psurf for types: total, ice_cloud, wc, DU, SO, strat, oc, SS, BC",
            "Units": "P/Psurf",
            "FillValue": -999.0,
            "MisingValue": -999.0,
        },
        NIR_ALBEDO_attr={
            "Longname": "NIR piecewise linear albedo, converted to Lambertian if retrieved in BRDF (by multiply by 0.07)",
            "Units": "",
            "FillValue": -999.0,
            "MisingValue": -999.0,
        },
        NIR_ALBEDO_POLY2_attr={
            "Longname": "NIR 2nd order polynomial albedo, converted to Lambertian if retrieved in BRDF, by band, with all terms for band1 listed first, etc.",
            "Units": "",
            "FillValue": -999.0,
            "MisingValue": -999.0,
        },
        NIR_FLUOR_REL_attr={
            "Longname": "NIR fluorescence divided by O2A top 5% radiance average in fitted radiance.",
            "Units": "N/A",
            "FillValue": -999.0,
            "MisingValue": -999.0,
        },
        NIR_CLOUD3D_SLOPE_attr={
            "Longname": "NIR 3d-cloud slope by band.",
            "Units": "N/A",
            "FillValue": -999.0,
            "MisingValue": -999.0,
        },
        NIR_CLOUD3D_OFFSET_attr={
            "Longname": "NIR 3d-cloud offset by band.",
            "Units": "N/A",
            "FillValue": -999.0,
            "MisingValue": -999.0,
        },
        DELTA_P_TRUE_attr={
            "Longname": "True surface pressure difference from prior",
            "Units": "hPa",
            "FillValue": -999.0,
            "MisingValue": -999.0,
        },
        CO2_GRAD_DEL_TRUE_attr={
            "Longname": "Change (between the retrieved profile and the prior profile) of the co2 dry air mole fraction difference from the surface (level 0) minus that at level 7 (0.631579 Psurf)",
            "Units": "ppm",
            "FillValue": -999.0,
            "MisingValue": -999.0,
        },
        DELTA_T_TRUE_attr={
            "Longname": "True atmospheric temperature difference from prior near 750 hPa",
            "Units": "K",
            "FillValue": -999.0,
            "MisingValue": -999.0,
        },
        NIR_WINDSPEED_TRUE_attr={
            "Longname": "True NIR windspeed, usually only used over water",
            "Units": "m/s",
            "FillValue": -999.0,
            "MisingValue": -999.0,
        },
        NIR_AEROD_TRUE_attr={
            "Longname": "True aerosol optical depth for types: total, ice_cloud, wc, DU, SO, strat, oc, SS, BC",
            "Units": "optical depth",
            "FillValue": -999.0,
            "MisingValue": -999.0,
        },
        NIR_AERP_TRUE_attr={
            "Longname": "True aerosol pressure/psurf for types: total, ice_cloud, wc, DU, SO, strat, oc, SS, BC",
            "Units": "P/Psurf",
            "FillValue": -999.0,
            "MisingValue": -999.0,
        },
        NIR_ALBEDO_TRUE_attr={
            "Longname": "True NIR piecewise linear albedo, converted to Lambertian if retrieved in BRDF (by multiply by 0.07)",
            "Units": "",
            "FillValue": -999.0,
            "MisingValue": -999.0,
        },
        NIR_ALBEDO_POLY2_TRUE_attr={
            "Longname": "True NIR 2nd order polynomial albedo, converted to Lambertian if retrieved in BRDF, by band, with all terms for band1 listed first, etc.",
            "Units": "",
            "FillValue": -999.0,
            "MisingValue": -999.0,
        },
        NIR_ALBEDO_POLY2_ERROR_TRUE_attr={
            "Longname": "True NIR converted to and back to a 2nd order polynomial albedo minus true (converted to Lambertian if needed)",
            "Units": "",
            "FillValue": -999.0,
            "MisingValue": -999.0,
        },
        NIR_FLUOR_REL_TRUE_attr={
            "Longname": "True NIR fluorescence divided by O2A top 5% radiance average.",
            "Units": "N/A",
            "FillValue": -999.0,
            "MisingValue": -999.0,
        },
        NIR_CLOUD3D_SLOPE_TRUE_attr={
            "Longname": "True NIR 3d-cloud slope by band.",
            "Units": "N/A",
            "FillValue": -999.0,
            "MisingValue": -999.0,
        },
        NIR_CLOUD3D_OFFSET_TRUE_attr={
            "Longname": "True NIR 3d-cloud offset by band.",
            "Units": "N/A",
            "FillValue": -999.0,
            "MisingValue": -999.0,
        },
        # species_data.NIR_ALBEDO_POLY2_TRUE
        # species_data.NIR_FLUOR_REL_TRUE
        # species_data.DELTA_P_TRUE
        # species_data.CO2_GRAD_DEL_TRUE
        # species_data.DELTA_T_TRUE
        # species_data.NIR_WINDSPEED_TRUE
        # species_data.NIR_AEROD_TRUE
        # species_data.NIR_AERP_TRUE
        # species_data.NIR_CLOUD3D_SLOPE_TRUE
        GLOBALSURVEYFLAG_attr={
            "Longname": "If run is a global survey ==1, otherwise ==0",
            "Units": "",
            "FillValue": -999.0,
            "MisingValue": -999.0,
        },
        LANDFLAG_attr={
            "Longname": "If observation is over land ==1, otherwise ==0",
            "Units": "",
            "FillValue": -999.0,
            "MisingValue": -999.0,
        },
        DOFS_attr={
            "Longname": "Same as degreesOfFreedomForSignal; trace of averaging kernel",
            "Units": "",
            "FillValue": -999.0,
            "MisingValue": -999.0,
        },
        SOUNDINGID_attr={
            "Longname": "Unique value for each sounding, in order of measurement, e.g. run_sequence_scan",
            "Units": "",
            "FillValue": -999.0,
            "MisingValue": -999.0,
        },
        VARIABILITYCH4_QA_attr={
            "Longname": "Q/A for methane, the standard deviation/mean below 200 mb for ch4",
            "Units": "",
            "FillValue": -999.0,
            "MisingValue": -999.0,
        },
        VARIABILITYN2O_QA_attr={
            "Longname": "Q/A for methane, the standard deviation/mean below 350 mb for n2o",
            "Units": "",
            "FillValue": -999.0,
            "MisingValue": -999.0,
        },
        BTOBS_attr={
            "Longname": "observed brightness temperature",
            "Units": "Kelvin",
            "FillValue": -999.0,
            "MisingValue": -999.0,
        },
        BTFIT_attr={
            "Longname": "brightness temperature of initial guess",
            "Units": "Kelvin",
            "FillValue": -999.0,
            "MisingValue": -999.0,
        },
        DBT_attr={
            "Longname": "brightness temperature difference between observed and initial guess",
            "Units": "Kelvin",
            "FillValue": -999.0,
            "MisingValue": -999.0,
        },
        KDOTDLSYS_QA_attr={
            "Longname": "same as KdotDL except for interferent species",
            "Units": "",
            "FillValue": -999.0,
            "MisingValue": -999.0,
        },
        CLOUDTOPPRESSUREDOF_attr={
            "Longname": "DOF for the cloud top pressure",
            "Units": "hPa",
            "FillValue": -999.0,
            "MisingValue": -999.0,
        },
        PRIORCOVARIANCE_attr={
            "Longname": "A priori covariance",
            "Units": "",
            "FillValue": -999.0,
            "MisingValue": -999.0,
        },
        RTVMR_attr={
            "Longname": "Representative Tropospheric VMR",
            "Units": "",
            "FillValue": -999.0,
            "MisingValue": -999.0,
        },
        RTVMR_PRESSURE_attr={
            "Longname": "Representative Tropospheric VMR pressure",
            "Units": "hPa",
            "FillValue": -999.0,
            "MisingValue": -999.0,
        },
        RTVMR_PRESSUREBOUNDUPPER_attr={
            "Longname": "Pressure at which the AK is half maximum",
            "Units": "hPa",
            "FillValue": -999.0,
            "MisingValue": -999.0,
        },
        RTVMR_PRESSUREBOUNDLOWER_attr={
            "Longname": "Pressure at which the AK is half maximum",
            "Units": "hPa",
            "FillValue": -999.0,
            "MisingValue": -999.0,
        },
        RTVMR_MAP_attr={
            "Longname": "Map from 4-5 level RTVMR to retrieval grid [RTVMR, Ret grid]",
            "Units": "hPa",
            "FillValue": -999.0,
            "MisingValue": -999.0,
        },
        RTVMR_MAPPRESSURE_attr={
            "Longname": "4-5 pressure levels used for RTVMR",
            "Units": "hPa",
            "FillValue": -999.0,
            "MisingValue": -999.0,
        },
        RTVMR_ERRORMEASUREMENT_attr={
            "Longname": "Measurement error",
            "Units": "Fraction VMR",
            "FillValue": -999.0,
            "MisingValue": -999.0,
        },
        RTVMR_ERROROBSERVATION_attr={
            "Longname": "Observation error",
            "Units": "Fraction VMR",
            "FillValue": -999.0,
            "MisingValue": -999.0,
        },
        RTVMR_ERROTOTAL_attr={
            "Longname": "Total error",
            "Units": "Fraction VMR",
            "FillValue": -999.0,
            "MisingValue": -999.0,
        },
        UTCTIME_attr={
            "Longname": "UTC time of observation",
            "Units": "N/A",
            "FillValue": -999.0,
            "MisingValue": -999.0,
        },
        BIAS2010_attr={
            "Longname": "CO2 bias in ppm applied after 2010",
            "Units": "ppm",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        BIASTIMEDEPENDENT_attr={
            "Longname": "Time dependent bias in ppm",
            "Units": "ppm",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        BIASSPATIAL_attr={
            "Longname": "Spatial bias in ppm (apply with caution as improvements are not conclusive)",
            "Units": "ppm",
            "FillValue": -999.00,
            "MisingValue": -999.00,
        },
        MAP_attr={
            "Longname": "Map from retrieval grid to FM grid [Ret grid, FM grid]",
            "Units": "",
            "FillValue": -999.0,
            "MisingValue": -999.0,
        },
    )
    rootgroup = ""
    retrgroup = "Retrieval"
    chargroup = "Characterization"
    geogroup = "Geolocation"
    cdf_var_names = [
        [rootgroup, "Altitude"],
        [rootgroup, "AveragingKernel"],
        [rootgroup, "Column750"],
        [rootgroup, "Column750_Averagingkernel"],
        [rootgroup, "Column750_constraintvector"],
        [rootgroup, "Column750_Initial"],
        [rootgroup, "Column750_ObservationError"],
        [rootgroup, "Column750_PWF"],
        [rootgroup, "ConstraintVector"],
        [rootgroup, "DOFs"],
        [rootgroup, "DayNightFlag"],
        [rootgroup, "GlobalSurveyFlag"],
        [rootgroup, "LandFlag"],
        [rootgroup, "Latitude"],
        [rootgroup, "Longitude"],
        [rootgroup, "O3_Ccurve_QA"],
        [rootgroup, "ObservationErrorCovariance"],
        [rootgroup, "Pressure"],
        [rootgroup, "Grid_Pressure_FM"],
        [rootgroup, "Quality"],
        [rootgroup, "Run"],
        [rootgroup, "Scan"],
        [rootgroup, "Sequence"],
        [rootgroup, "RTVMR"],
        [rootgroup, "RTVMR_Pressure"],
        [rootgroup, "RTVMR_ErrorObservation"],
        [rootgroup, "Species"],
        [rootgroup, "true"],
        [rootgroup, "true_ak"],
        [rootgroup, "SurfaceAltitude"],
        [rootgroup, "Time"],
        [rootgroup, "UT_Hour"],
        [rootgroup, "UTCTime"],
        [rootgroup, "YearFloat"],
        [rootgroup, "YYYYMMDD"],
        [rootgroup, "SoundingID"],
    ]
    cdf_var_names.extend(
        [
            [retrgroup, "AirDensity"],
            [retrgroup, "Average_800_Tropopause"],
            [retrgroup, "Average_800_TropopausePrior"],
            [retrgroup, "Average_900_200"],
            [retrgroup, "AverageCloudEffOpticalDepth"],
            [retrgroup, "City"],
            [retrgroup, "CityIndex"],
            [retrgroup, "CloudTopPressure"],
            [retrgroup, "Column"],
            [retrgroup, "Column_Air"],
            [retrgroup, "Column_Prior"],
            [retrgroup, "FmOzoneBandFlux"],
            [retrgroup, "LowerTroposphericColumn"],
            [retrgroup, "H2O_ConstraintVector"],
            [retrgroup, "H2O_Species"],
            [retrgroup, "H2O"],
            [retrgroup, "HDO"],
            [retrgroup, "HDO_H2O"],
            [retrgroup, "L1BOzoneBandFlux"],
            [retrgroup, "N2O_DOFS"],
            [retrgroup, "N2O_ConstraintVector"],
            [retrgroup, "N2O_Species"],
            [retrgroup, "Original_Species_N2O"],
            [retrgroup, "Original_Species_HDO"],
            [retrgroup, "Original_Column750"],
            [retrgroup, "Original_Species"],
            [retrgroup, "O3TroposphericColumn"],
            [retrgroup, "O3TroposphericColumnError"],
            [retrgroup, "OzoneIRK"],
            [retrgroup, "SurfaceTemperature"],
            [retrgroup, "TATM_ConstraintVector"],
            [retrgroup, "TATM_Deviation"],
            [retrgroup, "TATM_Species"],
            [retrgroup, "TATM"],
            [retrgroup, "TroposphericColumn"],
            [retrgroup, "TotalColumnDensity"],
            [retrgroup, "XPAN800"],
            [retrgroup, "XPAN800_Prior"],
            [retrgroup, "XPAN"],
            [retrgroup, "XPAN_Prior"],
            # OMI
            [retrgroup, "Omi_CloudTopPressure"],
            [retrgroup, "OMI_NRadWav"],
            [retrgroup, "OMI_NRadWavConstraintVector"],
            [retrgroup, "OMI_RingSF"],
            [retrgroup, "OMI_RingSFConstraintVector"],
            [retrgroup, "OMI_SurfAlb"],
            [retrgroup, "OMI_SurfAlbConstraintVector"],
            [retrgroup, "OMI_SurfAlbSlope"],
            [retrgroup, "OMI_SurfAlbSlopeConstraintVector"],
            [retrgroup, "Omi_CloudFraction"],
            [retrgroup, "Omi_RingSFUV1"],
            [retrgroup, "Omi_RingSFUV2"],
            [retrgroup, "Omi_SurfaceAlbedoUV1"],
            [retrgroup, "Omi_SurfaceAlbedoUV1ConstraintVector"],
            [retrgroup, "Omi_SurfaceAlbedoUV2"],
            [retrgroup, "Omi_SurfaceAlbedoUV2ConstraintVector"],
            [retrgroup, "Omi_SurfaceAlbedoSlopeUV2"],
            [retrgroup, "Omi_SurfaceAlbedoSlopeUV2ConstraintVector"],
            # TROPOMI
            [retrgroup, "TROPOMI_RingSFBAND1"],
            [retrgroup, "TROPOMI_RingSFBAND2"],
            [retrgroup, "TROPOMI_RingSFBAND3"],
            [retrgroup, "TROPOMI_RingSFBAND7"],
            [retrgroup, "TROPOMI_CloudFraction"],
            [retrgroup, "TROPOMI_SurfaceAlbedoBAND1"],
            [retrgroup, "TROPOMI_SurfaceAlbedoBAND1ConstraintVector"],
            [retrgroup, "TROPOMI_SurfaceAlbedoBAND2"],
            [retrgroup, "TROPOMI_SurfaceAlbedoBAND2ConstraintVector"],
            [retrgroup, "TROPOMI_SurfaceAlbedoBAND3"],
            [retrgroup, "TROPOMI_SurfaceAlbedoBAND3ConstraintVector"],
            [retrgroup, "TROPOMI_SurfaceAlbedoBAND7"],
            [retrgroup, "TROPOMI_SurfaceAlbedoBAND7ConstraintVector"],
            [retrgroup, "TROPOMI_SurfaceAlbedoSlopeBAND2"],
            [retrgroup, "TROPOMI_SurfaceAlbedoSlopeBAND2ConstraintVector"],
            [retrgroup, "TROPOMI_SurfaceAlbedoSlopeBAND3"],
            [retrgroup, "TROPOMI_SurfaceAlbedoSlopeBAND3ConstraintVector"],
            [retrgroup, "TROPOMI_SurfaceAlbedoSlopeBAND7"],
            [retrgroup, "TROPOMI_SurfaceAlbedoSlopeBAND7ConstraintVector"],
            [retrgroup, "TROPOMI_SurfaceAlbedoSlopeORDER2BAND2"],
            [retrgroup, "TROPOMI_SurfaceAlbedoSlopeORDER2BAND3"],
            [retrgroup, "TROPOMI_SurfaceAlbedoSlopeORDER2BAND7"],
            [retrgroup, "microwindow"],
            [retrgroup, "microwindow_instrument"],
            [retrgroup, "microwindow_species"],
        ]
    )
    cdf_var_names.extend(
        [
            [chargroup, "Altitude_FM"],
            [chargroup, "AverageCloudEffOpticalDepthError"],
            [chargroup, "AveragingKernelDiagonal"],
            [chargroup, "Bias2010"],
            [chargroup, "BiasSpatial"],
            [chargroup, "BiasTimeDependent"],
            [chargroup, "BTFIT"],
            [chargroup, "BTOBS"],
            [chargroup, "CalFreq"],
            [chargroup, "CalOffset"],
            [chargroup, "CalScale"],
            [chargroup, "CH4_Stratosphere_QA"],
            [chargroup, "CH4_DOFStrat"],
            [chargroup, "CH4_DOFTrop"],
            [chargroup, "CH4_EVs"],
            [chargroup, "CH4_EVRatio"],
            [chargroup, "CloudEffectiveOpticalDepth"],
            [chargroup, "CloudEffectiveOpticalDepth1000"],
            [chargroup, "CloudEffectiveOpticalDepthError"],
            [chargroup, "CloudEffectiveOpticalDepthError1000"],
            [chargroup, "CloudTopPressureDOF"],
            [chargroup, "CloudTopPressureError"],
            [chargroup, "CloudVariability_QA"],
            [chargroup, "Column_DOFS"],
            [chargroup, "Column_Error"],
            [chargroup, "Column_Initial"],
            [chargroup, "Column_PressureMax"],
            [chargroup, "Column_PressureMin"],
            [chargroup, "Column_True"],
            [chargroup, "Column_Units"],
            [chargroup, "Column750_Error"],
            [chargroup, "DBT"],
            [chargroup, "DegreesOfFreedomForSignal"],
            [chargroup, "Desert_Emiss_QA"],
            [chargroup, "Deviation_QA"],
            [chargroup, "DeviationBad_QA"],
            [chargroup, "DOFSLowerTroposphere"],
            [chargroup, "DOFSTroposphere"],
            [chargroup, "DOFSUpperTroposphere"],
            [chargroup, "Emissivity"],
            [chargroup, "Emissivity_Constraint"],
            [chargroup, "Emissivity_Error"],
            [chargroup, "Emissivity_Initial"],
            [chargroup, "Emissivity_Wavenumber"],
            [chargroup, "Emissivity_Offset_Distance"],
            [chargroup, "filter_index"],
            [chargroup, "Native_HSR_Emissivity_Initial"],
            [chargroup, "Native_HSR_Emis_Wavenumber"],
            [chargroup, "Filename"],
            [chargroup, "H2O_H2O_Corr_QA"],
            [chargroup, "H2O_HDO_Corr_QA"],
            [chargroup, "H2O_HDOMeasurementErrorCovariance"],
            [chargroup, "H2O_HDOObservationErrorCovariance"],
            [chargroup, "H2O_HDOTotalErrorCovariance"],
            [chargroup, "H2O_Propagated_QA"],
            [chargroup, "H2O_QA"],
            [chargroup, "H2O_Retrieval_QA"],
            [chargroup, "Initial"],
            [chargroup, "KDotDL_QA"],
            [chargroup, "KDotDLSys_QA"],
            [chargroup, "LDotDL_QA"],
            [chargroup, "LMResults_CostThresh"],
            [chargroup, "LMResults_IterList"],
            [chargroup, "LMResults_JacresNorm"],
            [chargroup, "LMResults_JacResNormNext"],
            [chargroup, "LMResults_PNorm"],
            [chargroup, "LMResults_ResNorm"],
            [chargroup, "LMResults_ResNormNext"],
            [chargroup, "LMResults_delta"],
            [chargroup, "LowerTroposphericColumnError"],
            [chargroup, "LowerTroposphericColumnInitial"],
            [chargroup, "Map"],
            [chargroup, "MeasurementErrorCovariance"],
            [chargroup, "N2O_AveragingKernel"],
            [chargroup, "N2O_ObservationErrorCovariance"],
            # parameters used for OCO-2 QF
            [chargroup, "nir_albedo"],
            [chargroup, "nir_albedo_poly2"],
            [chargroup, "co2_grad_del"],
            [chargroup, "delta_p"],
            [chargroup, "delta_t"],
            [chargroup, "nir_windspeed"],
            [chargroup, "nir_aerod"],
            [chargroup, "nir_aerp"],
            [chargroup, "nir_fluor_rel"],
            [chargroup, "nir_cloud3d_slope"],
            [chargroup, "nir_cloud3d_offset"],
            [chargroup, "nir_albedo_true"],
            [chargroup, "nir_albedo_poly2_error_true"],
            [chargroup, "nir_albedo_poly2_true"],
            [chargroup, "co2_grad_del_true"],
            [chargroup, "delta_p_true"],
            [chargroup, "delta_t_true"],
            [chargroup, "nir_windspeed_true"],
            [chargroup, "nir_aerod_true"],
            [chargroup, "nir_aerp_true"],
            [chargroup, "nir_fluor_rel_true"],
            [chargroup, "nir_cloud3d_slope_true"],
            [chargroup, "nir_cloud3d_offset_true"],
            [chargroup, "Num_Deviations_QA"],
            [chargroup, "O3_CCurve_TESOMI"],
            [chargroup, "O3_ColumnErrorDU"],
            [chargroup, "O3_Propagated_QA"],
            [chargroup, "O3_QA"],
            [chargroup, "O3_Retrieval_QA"],
            [chargroup, "O3_Slope_QA"],
            [chargroup, "O3_Tropo_Consistency_qa"],
            [chargroup, "Original_ConstraintVector_N2O"],
            [chargroup, "Original_Quality"],
            [chargroup, "pan_desert_qa"],
            [chargroup, "Precision"],
            [chargroup, "Pressure_FM"],
            [chargroup, "PriorCovariance"],
            [chargroup, "Propagated_H2O_QA"],
            [chargroup, "Propagated_O3_QA"],
            [chargroup, "Propagated_TATM_QA"],
            [chargroup, "QualityAggregate"],
            [chargroup, "RadianceMaximumSNR"],
            [chargroup, "RadianceResidualMean"],
            [chargroup, "RadianceResidualRMS"],
            # new filter names
            # filters = ['ALL', 'UV1', 'UV2', 'VIS', 'UVIS', 'NIR1','NIR2','SWIR1','SWIR2','SWIR3','SWIR4','TIR1','TIR2','TIR3','TIR4']
            # Radiance Residual Mean + Filter
            [chargroup, "RadianceResidualMean_UV1"],
            [chargroup, "RadianceResidualMean_UV2"],
            [chargroup, "RadianceResidualMean_VIS"],
            [chargroup, "RadianceResidualMean_UVIS"],
            [chargroup, "RadianceResidualMean_NIR1"],
            [chargroup, "RadianceResidualMean_NIR2"],
            [chargroup, "RadianceResidualMean_SWIR1"],
            [chargroup, "RadianceResidualMean_SWIR2"],
            [chargroup, "RadianceResidualMean_SWIR3"],
            [chargroup, "RadianceResidualMean_SWIR4"],
            [chargroup, "RadianceResidualMean_TIR1"],
            [chargroup, "RadianceResidualMean_TIR2"],
            [chargroup, "RadianceResidualMean_TIR3"],
            [chargroup, "RadianceResidualMean_TIR4"],
            # Radiance Residual RMS + Filter
            [chargroup, "RadianceResidualRMS_UV1"],
            [chargroup, "RadianceResidualRMS_UV2"],
            [chargroup, "RadianceResidualRMS_VIS"],
            [chargroup, "RadianceResidualRMS_UVIS"],
            [chargroup, "RadianceResidualRMS_SWIR1"],
            [chargroup, "RadianceResidualRMS_SWIR2"],
            [chargroup, "RadianceResidualRMS_SWIR3"],
            [chargroup, "RadianceResidualRMS_SWIR4"],
            [chargroup, "RadianceResidualRMS_TIR1"],
            [chargroup, "RadianceResidualRMS_TIR2"],
            [chargroup, "RadianceResidualRMS_TIR3"],
            [chargroup, "RadianceResidualRMS_TIR4"],
            # legacy filter names
            [chargroup, "RadianceResidualMean_1A1"],
            [chargroup, "RadianceResidualMean_1B2"],
            [chargroup, "RadianceResidualMean_2A1"],
            [chargroup, "RadianceResidualMean_2B1"],
            [chargroup, "RadianceResidualRMS_1A1"],
            [chargroup, "RadianceResidualRMS_1B2"],
            [chargroup, "RadianceResidualRMS_2A1"],
            [chargroup, "RadianceResidualRMS_2B1"],
            [chargroup, "RadianceResidualMean_Filter"],
            [chargroup, "RadianceResidualRMS_Filter"],
            [chargroup, "RadianceResidualQuadratic_Filter"],
            [chargroup, "RadianceResidualSlope_Filter"],
            [chargroup, "radiance_continuum_filter"],
            [chargroup, "radianceResidualRMSRelativeContinuum_filter"],
            [chargroup, "radiance_residual_stdev_change"],
            [chargroup, "ResidualNormFinal"],
            [chargroup, "ResidualNormInitial"],
            [chargroup, "RetrieveInLog"],
            [chargroup, "RTVMR_ErrorMeasurement"],
            [chargroup, "RTVMR_ErrorTotal"],
            [chargroup, "RTVMR_Map"],
            [chargroup, "RTVMR_MapPressure"],
            [chargroup, "RTVMR_PressureBoundLower"],
            [chargroup, "RTVMR_PressureBoundUpper"],
            [chargroup, "Quality"],
            [chargroup, "SurfaceEmissMean_QA"],
            [chargroup, "SurfacePressure"],
            [chargroup, "SurfaceTempConstraint"],
            [chargroup, "SurfaceTempDegreesOfFreedom"],
            [chargroup, "SurfaceTempError"],
            [chargroup, "SurfaceTempInitial"],
            [chargroup, "SurfaceTempObservationError"],
            [chargroup, "SurfaceTempPrecision"],
            [chargroup, "SurfaceTempVsApriori_QA"],
            [chargroup, "SurfaceTempVsAtmTemp_QA"],
            [chargroup, "SurfaceTypeFootprint"],
            [chargroup, "T700"],
            [chargroup, "TATM_Propagated_QA"],
            [chargroup, "TATM_QA"],
            [chargroup, "TATM_Retrieval_QA"],
            [chargroup, "ThermalContrast"],
            [chargroup, "ThermalContrastInitial"],
            [chargroup, "TotalColumnDensityError"],
            [chargroup, "TotalColumnDensityInitial"],
            [chargroup, "TotalError"],
            [chargroup, "TotalErrorCovariance"],
            [chargroup, "TropopausePressure"],
            [chargroup, "TroposphericColumnError"],
            [chargroup, "TroposphericColumnInitial"],
            [chargroup, "UpperTroposphericColumn"],
            [chargroup, "UpperTroposphericColumnError"],
            [chargroup, "UpperTroposphericColumnInitial"],
            [chargroup, "VariabilityCH4_QA"],
            [chargroup, "VariabilityN2O_QA"],
            [chargroup, "XPAN_AK"],
            [chargroup, "XPAN_ErrorObs"],
            [chargroup, "XPAN_ErrorSmoothing"],
            [chargroup, "XPAN800_AK"],
            [chargroup, "XPAN800_ErrorObs"],
            [chargroup, "XPAN800_ErrorSmoothing"],
            # OCO-2 parameters used in QF
            [chargroup, "oco2_co2_ratio_idp"],
            [chargroup, "oco2_h2o_ratio_idp"],
            [chargroup, "oco2_dp_abp"],
            [chargroup, "oco2_altitude_stddev"],
            [chargroup, "oco2_max_declocking_factor_wco2"],
            [chargroup, "oco2_max_declocking_factor_sco2"],
            # OMI
            [chargroup, "OMI_CloudFraction_Initial"],
            [chargroup, "OMI_CloudFractionConstraintVector"],
            [chargroup, "OMI_NRadWav_Initial"],
            [chargroup, "Omi_NRADWavUV1"],
            [chargroup, "Omi_NRADWavUV2"],
            [chargroup, "Omi_ODWavUV1"],
            [chargroup, "Omi_OdWavUV2"],
            [chargroup, "OMI_RadianceResidualMean"],
            [chargroup, "OMI_RadianceResidualMean_QA"],
            [chargroup, "OMI_RadianceResidualRMS"],
            [chargroup, "OMI_RadianceResidualRMS_QA"],
            [chargroup, "OMI_RingSF_Initial"],
            [chargroup, "OMI_SurfAlb_Initial"],
            [chargroup, "OMI_SurfAlb_QA"],
            [chargroup, "OMI_SurfAlbSlope_Initial"],
            # TROPOMI
            [chargroup, "TROPOMI_CloudFractionConstraintVector"],
            [chargroup, "TROPOMI_SOLARSHIFTBAND1"],
            [chargroup, "TROPOMI_SOLARSHIFTBAND2"],
            [chargroup, "TROPOMI_SOLARSHIFTBAND3"],
            [chargroup, "TROPOMI_SOLARSHIFTBAND7"],
            [chargroup, "TROPOMI_RADIANCESHIFTBAND1"],
            [chargroup, "TROPOMI_RADIANCESHIFTBAND2"],
            [chargroup, "TROPOMI_RADIANCESHIFTBAND3"],
            [chargroup, "TROPOMI_RADIANCESHIFTBAND7"],
            [chargroup, "TROPOMI_RADSQUEEZEBAND1"],
            [chargroup, "TROPOMI_RADSQUEEZEBAND2"],
            [chargroup, "TROPOMI_RADSQUEEZEBAND3"],
            [chargroup, "TROPOMI_RADSQUEEZEBAND7"],
            [chargroup, "TROPOMI_TEMPSHIFTBAND3"],
            [chargroup, "TROPOMI_TEMPSHIFTBAND7"],
        ]
    )

    # AT_LINE 631 TOOLS/cdf_write_tes.pro
    cdf_var_names.extend(
        [
            [chargroup, "CT_CO2"],
            [chargroup, "CT_CO2_AK"],
            [chargroup, "CT_Pressure"],
            [chargroup, "CT_Latitude"],
            [chargroup, "CT_Longitude"],
            [chargroup, "CT_YearFloat"],
            [chargroup, "NCEP_Temperature"],
            [chargroup, "NCEP_TemperatureSurface"],
        ]
    )
    cdf_var_names.extend([[chargroup, "Sonde"], [chargroup, "SondeAK"]])
    cdf_var_names.extend(
        [
            # AIRS
            [geogroup, "Airs_ATrack_Index"],
            [geogroup, "Airs_Granule"],
            [geogroup, "Airs_XTrack_Index"],
            [geogroup, "BoresightAzimuth"],
            [geogroup, "BoresightNadirAngle"],
            # CrIS
            [geogroup, "CrIS_Granule"],
            [geogroup, "CrIS_Atrack_Index"],
            [geogroup, "CrIS_Xtrack_Index"],
            [geogroup, "CrIS_Pixel_Index"],
            [geogroup, "CrIS_L1b_Type"],
            # OMI
            [geogroup, "OMI_ATrack_Index"],
            [geogroup, "OMI_XTrack_Index_uv1"],
            [geogroup, "OMI_XTrack_Index_uv2"],
            [geogroup, "OMI_XTrackQualityFlags"],
            [geogroup, "OMI_SZA_UV1"],
            [geogroup, "OMI_RAZ_UV1"],
            [geogroup, "OMI_VZA_UV1"],
            [geogroup, "OMI_SCA_UV1"],
            [geogroup, "OMI_SZA_UV2"],
            [geogroup, "OMI_RAZ_UV2"],
            [geogroup, "OMI_VZA_UV2"],
            [geogroup, "OMI_SCA_UV2"],
            [geogroup, "OMI_Cloud_Pressure"],
            [geogroup, "OMI_EarthSunDistance"],
            [geogroup, "OMI_Lat"],
            [geogroup, "OMI_Lon"],
            [geogroup, "OMI_Line"],
            [geogroup, "OMI_Orbit"],
            [geogroup, "OMI_OrbitPhase"],
            [geogroup, "OMI_Pixel"],
            [geogroup, "OMI_Pixel_Cloud_Fraction"],
            [geogroup, "Omi_SZA"],
            [geogroup, "Omi_RAZ"],
            [geogroup, "OMI_SolarAzimuthAngle"],
            [geogroup, "OMI_SolarZenithAngle"],
            [geogroup, "OMI_TAI_Time"],
            [geogroup, "OMI_UTC_Time"],
            [geogroup, "OMI_UV2_TerrainHeight"],
            [geogroup, "OMI_ViewingAzimuthAngle"],
            [geogroup, "OMI_ViewingZenithAngle"],
            # TROPOMI
            [geogroup, "TROPOMI_ATrack_Index"],
            [geogroup, "TROPOMI_XTrack_Index_BAND1"],
            [geogroup, "TROPOMI_XTrack_Index_BAND2"],
            [geogroup, "TROPOMI_XTrack_Index_BAND3"],
            [geogroup, "TROPOMI_XTrack_Index_BAND7"],
            [geogroup, "TROPOMI_SZA_BAND1"],
            [geogroup, "TROPOMI_RAZ_BAND1"],
            [geogroup, "TROPOMI_VZA_BAND1"],
            [geogroup, "TROPOMI_SCA_BAND1"],
            [geogroup, "TROPOMI_SZA_BAND2"],
            [geogroup, "TROPOMI_RAZ_BAND2"],
            [geogroup, "TROPOMI_VZA_BAND2"],
            [geogroup, "TROPOMI_SCA_BAND2"],
            [geogroup, "TROPOMI_SZA_BAND3"],
            [geogroup, "TROPOMI_RAZ_BAND3"],
            [geogroup, "TROPOMI_VZA_BAND3"],
            [geogroup, "TROPOMI_SCA_BAND3"],
            [geogroup, "TROPOMI_SZA_BAND7"],
            [geogroup, "TROPOMI_RAZ_BAND7"],
            [geogroup, "TROPOMI_VZA_BAND7"],
            [geogroup, "TROPOMI_SCA_BAND7"],
            [geogroup, "TROPOMI_Cloud_Pressure"],
            [geogroup, "TROPOMI_EarthSunDistance"],
            # TES
            [geogroup, "TES_ScanAveragedCount"],
            [geogroup, "TES_SurfaceTempInitial"],
            [geogroup, "TES_DominantSurfaceType"],
            [geogroup, "TES_Latitude"],
            [geogroup, "TES_Longitude"],
            [geogroup, "TES_OMI_Distance"],
            [geogroup, "TES_OMI_Matched"],
            [geogroup, "TES_OMI_Time_Difference"],
            [geogroup, "TES_SurfaceTypeFootprint"],
            [geogroup, "TES_TAI_Time"],
            [geogroup, "TES_UTC_Time"],
            [geogroup, "TES_Run"],
            [geogroup, "TES_Scan"],
            [geogroup, "TES_Sequence"],
            [geogroup, "PointingAngle_AIRS"],
            [geogroup, "PointingAngle_CrIS"],
            [geogroup, "PointingAngle_OMI"],
            [geogroup, "PointingAngle_TROPOMI"],
            [geogroup, "PointingAngle_TES"],
            [geogroup, "DominantSurfaceType"],
            [geogroup, "Latitude_Footprint_1"],
            [geogroup, "Latitude_Footprint_2"],
            [geogroup, "Latitude_Footprint_3"],
            [geogroup, "Latitude_Footprint_4"],
            [geogroup, "LocalSolarTime"],
            [geogroup, "Longitude_Footprint_1"],
            [geogroup, "Longitude_Footprint_2"],
            [geogroup, "Longitude_Footprint_3"],
            [geogroup, "Longitude_Footprint_4"],
            [geogroup, "SolarZenithAngle"],
            [geogroup, "SurfaceElevStandardDeviation"],
            [geogroup, "Tgt_SpacecraftAzimuth"],
            [geogroup, "Tgt_SpacecraftZenith"],
        ]
    )
    cdf_var_map = {}

    cdf_var_map["AIRDENSITY"] = "AirDensity"

    cdf_var_map["AIRS_ATRACK_INDEX"] = "Airs_ATrack_Index"
    cdf_var_map["AIRS_GRANULE"] = "Airs_Granule"
    cdf_var_map["AIRS_XTRACK_INDEX"] = "Airs_XTrack_Index"

    cdf_var_map["ALTITUDE"] = "Altitude"
    cdf_var_map["ALTITUDE_FM"] = "Altitude_FM"

    cdf_var_map["AVERAGE_800_TROPOPAUSE"] = "Average_800_Tropause"
    cdf_var_map["AVERAGE_800_TROPOPAUSEPRIOR"] = "Average_800_TropopausePrior"
    cdf_var_map["AVERAGE_900_200"] = "Average_900_200"
    cdf_var_map["AVERAGECLOUDEFFOPTICALDEPTH"] = "AverageCloudEffOpticalDepth"
    cdf_var_map["AVERAGECLOUDEFFOPTICALDEPTHERROR"] = "AverageCloudEffOpticalDepthError"

    cdf_var_map["AVERAGINGKERNEL"] = "AveragingKernel"

    cdf_var_map["AVERAGINGKERNELDIAGONAL"] = "AveragingKernelDiagonal"
    cdf_var_map["AVERAGINGKERNELDIAGONAL_FM"] = "AveragingKernelDiagonal_FM"

    cdf_var_map["BIAS2010"] = "Bias2010"
    cdf_var_map["BIASSPATIAL"] = "BiasSpatial"
    cdf_var_map["BIASTIMEDEPENDENT"] = "BiasTimeDependent"
    cdf_var_map["BORESIGHTAZIMUTH"] = "BoreSightAzimuth"
    cdf_var_map["BORESIGHTNADIRANGLE"] = "BoresightNadirAngle"
    cdf_var_map["BTFIT"] = "BTFIT"
    cdf_var_map["BTOBS"] = "BTOBS"
    cdf_var_map["CH4_DOFSTRAT"] = "CH4_DOFStrat"
    cdf_var_map["CH4_DOFTROP"] = "CH4_DOFTrop"
    cdf_var_map["CH4_EVRATIO"] = "CH4_EVRatio"
    cdf_var_map["CH4_EVS"] = "CH4_EVs"
    cdf_var_map["CH4_STRATOSPHERE_QA"] = "CH4_Stratosphere_QA"
    cdf_var_map["CITY"] = "City"
    cdf_var_map["CITYINDEX"] = "CityIndex"
    cdf_var_map["CLOUDEFFECTIVEOPTICALDEPTH1000"] = "CloudEffectiveOpticalDepth1000"
    cdf_var_map["CLOUDEFFECTIVEOPTICALDEPTH"] = "CloudEffectiveOpticalDepth"
    cdf_var_map["CLOUDEFFECTIVEOPTICALDEPTHERROR1000"] = (
        "CloudEffectiveOpticalDepthError1000"
    )
    cdf_var_map["CLOUDEFFECTIVEOPTICALDEPTHERROR"] = "CloudEffectiveOpticalDepthError"
    cdf_var_map["CLOUDTOPPRESSURE"] = "CloudTopPressure"
    cdf_var_map["CLOUDTOPPRESSUREDOF"] = "CloudTopPressureDOF"
    cdf_var_map["CLOUDTOPPRESSUREERROR"] = "CloudTopPressureError"
    cdf_var_map["CLOUDVARIABILITY_QA"] = "CloudVariability_QA"

    cdf_var_map["COLUMN750"] = "Column750"
    cdf_var_map["COLUMN750_AVERAGINGKERNEL"] = "Column750_AveragingKernel"
    cdf_var_map["COLUMN750_CONSTRAINTVECTOR"] = "Column750_ConstraintVector"
    cdf_var_map["COLUMN750_ERROR"] = "Column750_Error"
    cdf_var_map["COLUMN750_INITIAL"] = "Column750_Initial"
    cdf_var_map["COLUMN750_OBSERVATIONERROR"] = "Column750_ObservationError"
    cdf_var_map["COLUMN750_PWF"] = "Column750_PWF"

    cdf_var_map["COLUMN_AIR"] = "Column_Air"
    cdf_var_map["COLUMN"] = "Column"
    cdf_var_map["COLUMN_DOFS"] = "Column_DOFS"
    cdf_var_map["COLUMN_ERROR"] = "Column_Error"
    cdf_var_map["COLUMN_INITIAL"] = "Column_Initial"
    cdf_var_map["COLUMN_PRESSUREMAX"] = "Column_PressureMax"
    cdf_var_map["COLUMN_PRESSUREMIN"] = "Column_PressureMin"
    cdf_var_map["COLUMN_PRIOR"] = "Column_Prior"
    cdf_var_map["COLUMN_TRUE"] = "Column_True"
    cdf_var_map["COLUMN_UNITS"] = "Column_Units"

    cdf_var_map["MICROWINDOW"] = "microwindow"
    cdf_var_map["MICROWINDOW_INSTRUMENT"] = "microwindow_instrument"
    cdf_var_map["MICROWINDOW_SPECIES"] = "microwindow_species"

    cdf_var_map["CONSTRAINTVECTOR"] = "ConstraintVector"
    cdf_var_map["CONSTRAINTVECTOR_FM"] = "ConstraintVector_FM"

    cdf_var_map["CRIS_ATRACK_INDEX"] = "CrIS_Atrack_Index"
    cdf_var_map["CRIS_FIELDOFREGARD"] = "CrIS_FieldOfRegard"
    cdf_var_map["CRIS_GRANULE"] = "CrIS_Granule"
    cdf_var_map["CRIS_PIXEL"] = "CrIS_Pixel"
    cdf_var_map["CRIS_PIXEL_INDEX"] = "CrIS_Pixel_Index"
    cdf_var_map["CRIS_SCANLINE"] = "CrIS_Scanline"
    cdf_var_map["CRIS_XTRACK_INDEX"] = "CrIS_Xtrack_Index"
    cdf_var_map["CRIS_L1B_TYPE"] = "CrIS_L1b_Type"

    cdf_var_map["CT_CO2_AK"] = "CT_CO2_AK"
    cdf_var_map["CT_CO2"] = "CT_CO2"
    cdf_var_map["CT_LATITUDE"] = "CT_Latitude"
    cdf_var_map["CT_LONGITUDE"] = "CT_Longitude"
    cdf_var_map["CT_PRESSURE"] = "CT_Pressure"
    cdf_var_map["CT_YEARFLOAT"] = "CT_YearFloat"
    cdf_var_map["DAYNIGHTFLAG"] = "DayNightFlag"
    cdf_var_map["DBT"] = "DTB"

    cdf_var_map["DEGREESOFFREEDOMFORSIGNAL"] = "DegreesOfFreedomForSignal"

    cdf_var_map["DESERT_EMISS_QA"] = "Desert_Emiss_QA"

    cdf_var_map["DEVIATIONBAD_QA"] = "DeviationBad_QA"
    cdf_var_map["DEVIATION_QA"] = "Deviation_QA"
    cdf_var_map["DEVIATIONVSRETRIEVALCOVARIANCE"] = "DeviationVsRetrievalCovariance"

    cdf_var_map["DOFS"] = "DOFs"
    cdf_var_map["DOFSLOWERTROPOSPHERE"] = "DOFsLowerTroposphere"
    cdf_var_map["DOFSTROPOSPHERE"] = "DOFsTroposphere"
    cdf_var_map["DOFSUPPERTROPOSPHERE"] = "DOFsUpperTroposphere"
    cdf_var_map["DOMINANTSURFACETYPE"] = "DominantSurfaceType"

    cdf_var_map["EMISSIVITY"] = "Emissivity"
    cdf_var_map["EMISSIVITY_CONSTRAINT"] = "Emissivity_Constraint"
    cdf_var_map["EMISSIVITY_ERROR"] = "Emissivity_Error"
    cdf_var_map["EMISSIVITY_INITIAL"] = "Emissivity_Initial"
    cdf_var_map["EMISSIVITY_WAVENUMBER"] = "Emissivity_Wavenumber"
    cdf_var_map["EMISSIVITY_OFFSET_DISTANCE"] = "Emissivity_Offset_Distance"
    cdf_var_map["FILTER_INDEX"] = "filter_index"
    cdf_var_map["NATIVE_HSR_EMISSIVITY_INITIAL"] = "Native_HSR_Emissivity_Initial"
    cdf_var_map["NATIVE_HSR_EMIS_WAVENUMBER"] = "Native_HSR_Emis_Wavenumber"

    cdf_var_map["FMOZONEBANDFLUX"] = "FMOzoneBandFlux"
    cdf_var_map["GLOBALSURVEYFLAG"] = "GlobalSurveyFlag"

    cdf_var_map["GRID_2"] = "Grid_2"
    cdf_var_map["GRID_3"] = "Grid_3"
    cdf_var_map["GRID_CLOUD"] = "Grid_Cloud"
    cdf_var_map["GRID_COLUMN"] = "Grid_Column"
    cdf_var_map["GRID_CT_LAYER"] = "Grid_CT_Layer"
    cdf_var_map["GRID_CT_LEVEL"] = "Grid_CT_Level"
    cdf_var_map["GRID_EMISSIVITY"] = "Grid_Emissivity"
    cdf_var_map["GRID_FILTER"] = "Grid_Filter"
    cdf_var_map["GRID_ITERLIST"] = "Grid_IterList"
    cdf_var_map["GRID_ITERS"] = "Grid_Iters"
    cdf_var_map["GRID_NCEP"] = "Grid_NCEP"
    cdf_var_map["GRID_PRESSURE"] = "Grid_Pressure"
    cdf_var_map["GRID_PRESSURE_COMPOSITE"] = "Grid_Pressure_Composite"
    cdf_var_map["GRID_PRESSURE_FM"] = "Grid_Pressure_FM"
    cdf_var_map["GRID_RTVMR_LEVELS"] = "Grid_RTVMR_Levels"
    cdf_var_map["GRID_RTVMR_MAP"] = "Grid_RTVMR_Map"
    cdf_var_map["GRID_TARGETS"] = "Grid_Targets"

    cdf_var_map["H2O"] = "H2O"
    cdf_var_map["H2O_CONSTRAINTVECTOR"] = "H2O_ConstraintVector"
    cdf_var_map["H2O_H2O_CORR_QA"] = "H2O_H2O_Corr_QA"
    cdf_var_map["H2O_HDOMEASUREMENTERRORCOVARIANCE"] = (
        "H2O_HDOMeasurementErrorCovariance"
    )
    cdf_var_map["H2O_HDOOBSERVATIONERRORCOVARIANCE"] = (
        "H2O_HDOObservationErrorCovariance"
    )
    cdf_var_map["H2O_HDOTOTALERRORCOVARIANCE"] = "H2O_HDOTotalErrorCovariance"
    cdf_var_map["H2O_PROPAGATED_QA"] = "H2O_Propogated_QA"
    cdf_var_map["H2O_QA"] = "H2O_QA"
    cdf_var_map["H2O_RETRIEVAL_QA"] = "H2O_Retrieval_QA"
    cdf_var_map["H2O_SPECIES"] = "H2O_Species"

    cdf_var_map["HDO"] = "H2O"
    cdf_var_map["HDO_H2O"] = "HDO_H2O"

    cdf_var_map["INITIAL"] = "Initial"
    cdf_var_map["INITIAL_FM"] = "Initial_FM"

    cdf_var_map["KDOTDL_QA"] = "KDotDL_QA"
    cdf_var_map["KDOTDLSYS_QA"] = "KDotDLSys_QA"

    cdf_var_map["L1BOZONEBANDFLUX"] = "L1BOzoneBandFlux"

    cdf_var_map["LANDFLAG"] = "LandFlag"

    cdf_var_map["LATITUDE"] = "Latitude"

    cdf_var_map["LATITUDE_FOOTPRINT_1"] = "Latitude_Footprint_1"
    cdf_var_map["LATITUDE_FOOTPRINT_2"] = "Latitude_Footprint_2"
    cdf_var_map["LATITUDE_FOOTPRINT_3"] = "Latitude_Footprint_3"
    cdf_var_map["LATITUDE_FOOTPRINT_4"] = "Latitude_Footprint_4"

    cdf_var_map["LDOTDL_QA"] = "LDotDL_QA"
    cdf_var_map["LDOTDNCEP_TEMPERATURE"] = "LDotDNCEP_Temperature"
    cdf_var_map["LMRESULTS_COSTTHRESH"] = "LmResults_CostThresh"
    cdf_var_map["LMRESULTS_ITERLIST"] = "LmResults_IterList"
    cdf_var_map["LMRESULTS_JACRESNORM"] = "LmResults_JacResNorm"
    cdf_var_map["LMRESULTS_JACRESNORMNEXT"] = "LmResults_JacResNormNext"
    cdf_var_map["LMRESULTS_PNORM"] = "LmResults_PNorm"
    cdf_var_map["LMRESULTS_RESNORM"] = "LmResults_ResNorm"
    cdf_var_map["LMRESULTS_RESNORMNEXT"] = "LmResults_ResNormNext"
    cdf_var_map["LMRESULTS_DELTA"] = "LmResults_delta"

    cdf_var_map["LOCALSOLARTIME"] = "LocalSolarTime"

    cdf_var_map["LONGITUDE"] = "Longitude"

    cdf_var_map["LONGITUDE_FOOTPRINT_1"] = "Longitude_Footprint_1"
    cdf_var_map["LONGITUDE_FOOTPRINT_2"] = "Longitude_Footprint_2"
    cdf_var_map["LONGITUDE_FOOTPRINT_3"] = "Longitude_Footprint_3"
    cdf_var_map["LONGITUDE_FOOTPRINT_4"] = "Longitude_Footprint_4"

    cdf_var_map["LOWERTROPOSPHERICCOLUMN"] = "LowerTroposphericColumn"
    cdf_var_map["LOWERTROPOSPHERICCOLUMNERROR"] = "LowerTroposphericColumnError"
    cdf_var_map["LOWERTROPOSPHERICCOLUMNINITIAL"] = "LowerTroposphericColumnInitial"

    cdf_var_map["L_QA"] = "L_QA"
    cdf_var_map["MAP"] = "Map"
    cdf_var_map["MAXNUMITERATIONS"] = "MaxNumIterations"
    cdf_var_map["MEASUREMENTERRORCOVARIANCE"] = "MeasurementErrorCovariance"

    cdf_var_map["N2O_AVERAGINGKERNEL"] = "N2O_AveragingKernel"
    cdf_var_map["N2O_CONSTRAINTVECTOR"] = "N2O_ConstraintVector"
    cdf_var_map["N2O_CONSTRAINTVECTOR_FM"] = "N2O_ConstraintVector_FM"
    cdf_var_map["N2O_DOFS"] = "N2O_DOFs"
    cdf_var_map["N2O_OBSERVATIONERRORCOVARIANCE"] = "N2O_ObservationErrorCovariance"
    cdf_var_map["N2O_SPECIES"] = "N2O_Species"
    cdf_var_map["N2O_SPECIES_FM"] = "N2O_Species_FM"

    cdf_var_map["DELTA_P"] = "delta_p"
    cdf_var_map["CO2_GRAD_DEL"] = "co2_grad_del"
    cdf_var_map["DELTA_T"] = "delta_t"
    cdf_var_map["NIR_WINDSPEED"] = "nir_windspeed"
    cdf_var_map["NIR_AEROD"] = "nir_aerod"
    cdf_var_map["NIR_AERP"] = "nir_aerp"
    cdf_var_map["NIR_ALBEDO"] = "nir_albedo"
    cdf_var_map["NIR_ALBEDO_POLY2"] = "nir_albedo_poly2"
    cdf_var_map["NIR_FLUOR_REL"] = "nir_fluor_rel"
    cdf_var_map["NIR_CLOUD3D_SLOPE"] = "nir_cloud3d_slope"
    cdf_var_map["NIR_CLOUD3D_OFFSET"] = "nir_cloud3d_offset"

    cdf_var_map["DELTA_P_TRUE"] = "delta_p_true"
    cdf_var_map["CO2_GRAD_DEL_TRUE"] = "co2_grad_del_true"
    cdf_var_map["DELTA_T_TRUE"] = "delta_t_true"
    cdf_var_map["NIR_WINDSPEED_TRUE"] = "nir_windspeed_true"
    cdf_var_map["NIR_AEROD_TRUE"] = "nir_aerod_true"
    cdf_var_map["NIR_AERP_TRUE"] = "nir_aerp_true"
    cdf_var_map["NIR_ALBEDO_POLY2_TRUE"] = "nir_albedo_poly2_true"
    cdf_var_map["NIR_ALBEDO_TRUE"] = "nir_albedo_true"
    cdf_var_map["NIR_ALBEDO_POLY2_ERROR_TRUE"] = "nir_albedo_poly2_error_true"
    cdf_var_map["NIR_FLUOR_REL_TRUE"] = "nir_fluor_rel_true"
    cdf_var_map["NIR_CLOUD3D_SLOPE_TRUE"] = "nir_cloud3d_slope_true"
    cdf_var_map["NIR_CLOUD3D_OFFSET_TRUE"] = "nir_cloud3d_offset_true"

    cdf_var_map["NCEP_PRESSURE"] = "NCEP_Pressure"
    cdf_var_map["NCEP_TEMPERATURE"] = "NCEP_Pressure"
    cdf_var_map["NCEP_TEMPERATURESURFACE"] = "NCEP_Temperature"
    cdf_var_map["NUMBERITERPERFORMED"] = "NumerIterPerformed"
    cdf_var_map["NUM_DEVIATIONS_QA"] = "Num_Deviations_QA"

    cdf_var_map["O3_CCURVE_QA"] = "O3_Ccurve_QA"
    cdf_var_map["O3_CCURVE_TESOMI"] = "O3_CCurve_TESOMI"
    cdf_var_map["O3_COLUMNERRORDU"] = "O3_ColumnErrorDU"
    cdf_var_map["O3_PROPAGATED_QA"] = "O3_Propagated_QA"
    cdf_var_map["O3_QA"] = "O3_QA"
    cdf_var_map["O3_RETRIEVAL_QA"] = "O3_Retrieval_QA"
    cdf_var_map["O3_SLOPE_QA"] = "O3_Slope_QA"
    cdf_var_map["O3_TROPO_CONSISTENCY_QA"] = "O3_Tropo_Consistency_QA"
    cdf_var_map["O3TROPOSPHERICCOLUMN"] = "O3TroposphericColumn"
    cdf_var_map["O3TROPOSPHERICCOLUMNERROR"] = "O3TroposphericColumnError"
    cdf_var_map["O3TROPOSPHERICCOLUMNINITIAL"] = "O3TroposphericColumnInitial"
    cdf_var_map["OBSERVATIONERRORCOVARIANCE"] = "ObservationErrorCovariance"

    cdf_var_map["OCO2_CO2_RATIO_IDP"] = "oco2_co2_ratio_idp"
    cdf_var_map["OCO2_H2O_RATIO_IDP"] = "oco2_h2o_ratio_idp"
    cdf_var_map["OCO2_DP_ABP"] = "oco2_dp_abp"
    cdf_var_map["OCO2_ALTITUDE_STDDEV"] = "oco2_altitude_stddev"
    cdf_var_map["OCO2_MAX_DECLOCKING_FACTOR_WCO2"] = "oco2_max_declocking_factor_wco2"
    cdf_var_map["OCO2_MAX_DECLOCKING_FACTOR_SCO2"] = "oco2_max_declocking_factor_sco2"

    cdf_var_map["OMI_ATRACK_INDEX"] = "OMI_ATrack_Index"
    cdf_var_map["OMI_XTRACK_INDEX_UV1"] = "OMI_XTrack_Index_uv1"
    cdf_var_map["OMI_XTRACK_INDEX_UV2"] = "OMI_XTrack_Index_uv2"

    cdf_var_map["OMI_XTRACKQUALITYFLAGS"] = "OMI_XTrackQualityFlags"

    cdf_var_map["OMI_CLOUDFRACTION"] = "OMI_CloudFraction"
    cdf_var_map["OMI_CLOUDFRACTIONCONSTRAINTVECTOR"] = (
        "OMI_CloudFractionConstraintVector"
    )

    cdf_var_map["OMI_CLOUDFRACTION_INITIAL"] = "OMI_CloudFraction_Initial"

    cdf_var_map["OMI_CLOUD_PRESSURE"] = "OMI_Cloud_Pressure"

    cdf_var_map["OMI_CLOUDTOPPRESSURE"] = "OMI_CloudTopPressure"

    cdf_var_map["OMI_EARTHSUNDISTANCE"] = "OMI_EarthSunDistance"

    cdf_var_map["OMI_LAT"] = "OMI_Lat"
    cdf_var_map["OMI_LON"] = "OMI_Lon"
    cdf_var_map["OMI_LINE"] = "OMI_Line"

    cdf_var_map["OMI_NRADWAV"] = "OMI_NRadWav"
    cdf_var_map["OMI_NRADWAVCONSTRAINTVECTOR"] = "OMI_NRadWavConstraintVector"

    cdf_var_map["OMI_NRADWAV_INITIAL"] = "OMI_NRadWav_Initial"

    cdf_var_map["OMI_NRADWAVUV1"] = "OMI_NRadWavUV1"
    cdf_var_map["OMI_NRADWAVUV2"] = "OMI_NRadWavUV2"

    cdf_var_map["OMI_ODWAVUV1"] = "OMI_ODWavUV1"
    cdf_var_map["OMI_ODWAVUV2"] = "OMI_ODWavUV2"

    cdf_var_map["OMI_ORBIT"] = "OMI_Orbit"
    cdf_var_map["OMI_ORBITPHASE"] = "OMI_OrbitPhase"

    cdf_var_map["OMI_PIXEL"] = "OMI_Pixel"
    cdf_var_map["OMI_PIXEL_CLOUD_FRACTION"] = "OMI_Pixel_Cloud_Fraction"

    cdf_var_map["OMI_RADIANCERESIDUALMEAN"] = "OMI_RadianceResidualMean"
    cdf_var_map["OMI_RADIANCERESIDUALMEAN_QA"] = "OMI_RadianceResidualMean_QA"

    cdf_var_map["OMI_RADIANCERESIDUALRMS"] = "OMI_RadianceResidualRMS"
    cdf_var_map["OMI_RADIANCERESIDUALRMS_QA"] = "OMI_RadianceResidualRMS_QA"

    cdf_var_map["OMI_RAZ"] = "OMI_RAZ"
    cdf_var_map["OMI_RAZ_UV1"] = "OMI_RAZ_UV1"
    cdf_var_map["OMI_RAZ_UV2"] = "OMI_RAZ_UV2"

    cdf_var_map["OMI_RINGSF"] = "OMI_RingSF"
    cdf_var_map["OMI_RINGSFCONSTRAINTVECTOR"] = "OMI_OMI_RingSFConstraintVector"

    cdf_var_map["OMI_RINGSF_INITIAL"] = "OMI_RingSF_Initial"

    cdf_var_map["OMI_RINGSFUV1"] = "OMI_RingSFUV1"
    cdf_var_map["OMI_RINGSFUV2"] = "OMI_RingSFUV2"

    cdf_var_map["OMI_SCA_UV1"] = "OMI_SCA_UV1"
    cdf_var_map["OMI_SCA_UV2"] = "OMI_SCA_UV2"

    cdf_var_map["OMI_SOLARAZIMUTHANGLE"] = "OMI_SolarAzimuthAngle"
    cdf_var_map["OMI_SOLARZENITHANGLE"] = "OMI_SolarZenithAngle"

    cdf_var_map["OMI_SURFACEALBEDOSLOPEUV2"] = "OMI_SurfaceAlbedoSlopeUV2"
    cdf_var_map["OMI_SURFACEALBEDOSLOPEUV2CONSTRAINTVECTOR"] = (
        "OMI_SurfaceAlbedoSlopeUV2ConstraintVector"
    )

    cdf_var_map["OMI_SURFACEALBEDOUV1"] = "OMI_SurfaceAlbedoUV1"
    cdf_var_map["OMI_SURFACEALBEDOUV1CONSTRAINTVECTOR"] = (
        "OMI_SurfaceAlbedoUV1ConstraintVector"
    )

    cdf_var_map["OMI_SURFACEALBEDOUV2"] = "OMI_SurfaceAlbedoUV2"
    cdf_var_map["OMI_SURFACEALBEDOUV2CONSTRAINTVECTOR"] = (
        "OMI_SurfaceAlbedoUV2ConstraintVector"
    )

    cdf_var_map["OMI_SURFALB"] = "OMI_SurfAlb"
    cdf_var_map["OMI_SURFALBCONSTRAINTVECTOR"] = "OMI_SurfAlbConstraintVector"

    cdf_var_map["OMI_SURFALB_INITIAL"] = "OMI_SurfAlb_Initial"

    cdf_var_map["OMI_SURFALB_QA"] = "OMI_SurfAlb_QA"

    cdf_var_map["OMI_SURFALBSLOPE"] = "OMI_SurfAlbSlope"
    cdf_var_map["OMI_SURFALBSLOPECONSTRAINTVECTOR"] = "OMI_SurfAlbSlopeConstraintVector"

    cdf_var_map["OMI_SURFALBSLOPE_INITIAL"] = "OMI_SurfAlbSlope_Initial"

    cdf_var_map["OMI_SZA"] = "OMI_SZA"
    cdf_var_map["OMI_SZA_UV1"] = "OMI_SZA_UV1"
    cdf_var_map["OMI_SZA_UV2"] = "OMI_SZA_UV2"

    cdf_var_map["OMI_TAI_TIME"] = "OMI_TAI_Time"
    cdf_var_map["OMI_UTC_TIME"] = "OMI_UTC_Time"

    cdf_var_map["OMI_UV2_TERRAINHEIGHT"] = "OMI_UV2_TerrainHeight"

    cdf_var_map["OMI_VIEWINGAZIMUTHANGLE"] = "OMI_ViewingAzimuthAngle"
    cdf_var_map["OMI_VIEWINGZENITHANGLE"] = "OMI_ViewingZenithAngle"

    cdf_var_map["OMI_VZA_UV1"] = "OMI_VZA_UV1"
    cdf_var_map["OMI_VZA_UV2"] = "OMI_VZA_UV2"

    # Geolocation group
    cdf_var_map["TROPOMI_ATRACK_INDEX"] = "TROPOMI_ATrack_Index"

    cdf_var_map["TROPOMI_XTRACK_INDEX_BAND1"] = "TROPOMI_XTrack_Index_BAND1"
    cdf_var_map["TROPOMI_XTRACK_INDEX_BAND2"] = "TROPOMI_XTrack_Index_BAND2"
    cdf_var_map["TROPOMI_XTRACK_INDEX_BAND3"] = "TROPOMI_XTrack_Index_BAND3"

    cdf_var_map["TROPOMI_SZA_BAND1"] = "TROPOMI_SZA_BAND1"
    cdf_var_map["TROPOMI_RAZ_BAND1"] = "TROPOMI_RAZ_BAND1"
    cdf_var_map["TROPOMI_VZA_BAND1"] = "TROPOMI_VZA_BAND1"
    cdf_var_map["TROPOMI_SCA_BAND1"] = "TROPOMI_SCA_BAND1"

    cdf_var_map["TROPOMI_SZA_BAND2"] = "TROPOMI_SZA_BAND2"
    cdf_var_map["TROPOMI_RAZ_BAND2"] = "TROPOMI_RAZ_BAND2"
    cdf_var_map["TROPOMI_VZA_BAND2"] = "TROPOMI_VZA_BAND2"
    cdf_var_map["TROPOMI_SCA_BAND2"] = "TROPOMI_SCA_BAND2"

    cdf_var_map["TROPOMI_SZA_BAND3"] = "TROPOMI_SZA_BAND3"
    cdf_var_map["TROPOMI_RAZ_BAND3"] = "TROPOMI_RAZ_BAND3"
    cdf_var_map["TROPOMI_VZA_BAND3"] = "TROPOMI_VZA_BAND3"
    cdf_var_map["TROPOMI_SCA_BAND3"] = "TROPOMI_SCA_BAND3"

    cdf_var_map["TROPOMI_EARTHSUNDISTANCE"] = "TROPOMI_EarthSunDistance"

    # Retrieval group
    cdf_var_map["TROPOMI_CLOUDFRACTION"] = "TROPOMI_CloudFraction"
    cdf_var_map["TROPOMI_CLOUDTOPPRESSURE"] = "TROPOMI_CloudTopPressure"

    cdf_var_map["TROPOMI_RINGSFBAND1"] = "TROPOMI_RingSFBAND1"
    cdf_var_map["TROPOMI_RINGSFBAND2"] = "TROPOMI_RingSFBAND2"
    cdf_var_map["TROPOMI_RINGSFBAND3"] = "TROPOMI_RingSFBAND3"

    cdf_var_map["TROPOMI_SURFACEALBEDOBAND1"] = "TROPOMI_SurfaceAlbedoBAND1"
    cdf_var_map["TROPOMI_SURFACEALBEDOBAND1CONSTRAINTVECTOR"] = (
        "TROPOMI_SurfaceAlbedoBAND1ConstraintVector"
    )

    cdf_var_map["TROPOMI_SURFACEALBEDOBAND2"] = "TROPOMI_SurfaceAlbedoBAND2"
    cdf_var_map["TROPOMI_SURFACEALBEDOBAND2CONSTRAINTVECTOR"] = (
        "TROPOMI_SurfaceAlbedoBAND2ConstraintVector"
    )

    cdf_var_map["TROPOMI_SURFACEALBEDOBAND3"] = "TROPOMI_SurfaceAlbedoBAND3"
    cdf_var_map["TROPOMI_SURFACEALBEDOBAND3CONSTRAINTVECTOR"] = (
        "TROPOMI_SurfaceAlbedoBAND3ConstraintVector"
    )

    cdf_var_map["TROPOMI_SURFACEALBEDOSLOPEBAND2"] = "TROPOMI_SurfaceAlbedoSlopeBAND2"
    cdf_var_map["TROPOMI_SURFACEALBEDOSLOPEBAND2CONSTRAINTVECTOR"] = (
        "TROPOMI_SurfaceAlbedoSlopeBAND2ConstraintVector"
    )

    cdf_var_map["TROPOMI_SURFACEALBEDOSLOPEBAND3"] = "TROPOMI_SurfaceAlbedoSlopeBAND3"
    cdf_var_map["TROPOMI_SURFACEALBEDOSLOPEBAND3CONSTRAINTVECTOR"] = (
        "TROPOMI_SurfaceAlbedoSlopeBAND3ConstraintVector"
    )

    cdf_var_map["TROPOMI_SURFACEALBEDOSLOPEORDER2BAND2"] = (
        "TROPOMI_SurfaceAlbedoSlopeORDER2BAND2"
    )
    cdf_var_map["TROPOMI_SURFACEALBEDOSLOPEORDER2BAND3"] = (
        "TROPOMI_SurfaceAlbedoSlopeORDER2BAND3"
    )

    # Characterization group
    cdf_var_map["TROPOMI_CLOUDFRACTIONCONSTRAINTVECTOR"] = (
        "TROPOMI_CloudFractionConstraintVector"
    )

    cdf_var_map["TROPOMI_SOLARSHIFTBAND1"] = "TROPOMI_SOLARSHIFTBAND1"
    cdf_var_map["TROPOMI_SOLARSHIFTBAND2"] = "TROPOMI_SOLARSHIFTBAND2"
    cdf_var_map["TROPOMI_SOLARSHIFTBAND3"] = "TROPOMI_SOLARSHIFTBAND3"

    cdf_var_map["TROPOMI_RADIANCESHIFTBAND1"] = "TROPOMI_RADIANCESHIFTBAND1"
    cdf_var_map["TROPOMI_RADIANCESHIFTBAND2"] = "TROPOMI_RADIANCESHIFTBAND2"
    cdf_var_map["TROPOMI_RADIANCESHIFTBAND3"] = "TROPOMI_RADIANCESHIFTBAND3"

    cdf_var_map["TROPOMI_RADSQUEEZEBAND1"] = "TROPOMI_RADSQUEEZEBAND1"
    cdf_var_map["TROPOMI_RADSQUEEZEBAND2"] = "TROPOMI_RADSQUEEZEBAND2"
    cdf_var_map["TROPOMI_RADSQUEEZEBAND3"] = "TROPOMI_RADSQUEEZEBAND3"

    cdf_var_map["TROPOMI_TEMPSHIFTBAND3"] = "TROPOMI_TEMPSHIFTBAND3"

    # Root group
    cdf_var_map["ORIGINAL_COLUMN750"] = "Original_Column750"

    cdf_var_map["ORIGINAL_CONSTRAINTVECTOR_N2O"] = "Original_ConstraintVector_N2O"
    cdf_var_map["ORIGINAL_CONSTRAINTVECTOR_N2O_FM"] = "Original_ConstraintVector_N2O_FM"

    cdf_var_map["ORIGINAL_SPECIES"] = "Original_Species"
    cdf_var_map["ORIGINAL_SPECIES_FM"] = "Original_Species_FM"

    cdf_var_map["ORIGINAL_SPECIES_FM_HDO"] = "Original_Species_FM_HDO"
    cdf_var_map["ORIGINAL_SPECIES_HDO"] = "Original_Species_HDO"

    cdf_var_map["ORIGINAL_SPECIES_N2O"] = "Original_Species_N2O"
    cdf_var_map["ORIGINAL_SPECIES_N2O_FM"] = "Original_Species_N2O_FM"

    cdf_var_map["ORIGINAL_QUALITY"] = "Original_Quality"

    cdf_var_map["OZONEIRK"] = "OzoneIRK"

    cdf_var_map["PAN_DESERT_QA"] = "Pan_Desert_QA"

    cdf_var_map["POINTINGANGLE_AIRS"] = "PointingAngle_AIRS"
    cdf_var_map["POINTINGANGLE_CRIS"] = "PointingAngle_CrIS"
    cdf_var_map["POINTINGANGLE_OMI"] = "PointingAngle_OMI"
    cdf_var_map["POINTINGANGLE_TES"] = "PointingAngle_TES"

    cdf_var_map["PRECISION"] = "Precision"
    cdf_var_map["PRECISION_FM"] = "Precision_FM"

    cdf_var_map["PRESSURE"] = "Pressure"
    cdf_var_map["PRESSURE_FM"] = "Pressure_FM"

    cdf_var_map["PRIORCOVARIANCE"] = "PriorCovariance"

    cdf_var_map["PROPAGATED_H2O_QA"] = "Propagated_H2O_QA"
    cdf_var_map["PROPAGATED_O3_QA"] = "Propagated_O3_QA"
    cdf_var_map["PROPAGATED_TATM_QA"] = "Propagated_TATM_QA"

    cdf_var_map["QUALITYAGGREGATE"] = "QualityAggregate"
    cdf_var_map["QUALITY"] = "Quality"

    cdf_var_map["RADIANCEMAXIMUMSNR"] = "RadianceMaximumSNR"

    cdf_var_map["RADIANCERESIDUALMAX"] = "RadianceResidualMax"

    cdf_var_map["RADIANCERESIDUALMEAN"] = "RadianceResidualMean"
    cdf_var_map["RADIANCERESIDUALRMS"] = "RadianceResidualRMS"

    # new filter names
    # filters = ['ALL', 'UV1', 'UV2', 'VIS', 'UVIS', 'NIR1','NIR2','SWIR1','SWIR2','SWIR3','SWIR4','TIR1','TIR2','TIR3','TIR4']
    # Radiance Residual Mean + Filter
    cdf_var_map["RADIANCERESIDUALMEAN_UV1"] = "RadianceResidualMean_UV1"
    cdf_var_map["RADIANCERESIDUALMEAN_UV2"] = "RadianceResidualMean_UV2"

    cdf_var_map["RADIANCERESIDUALMEAN_VIS"] = "RadianceResidualMean_VIS"

    cdf_var_map["RADIANCERESIDUALMEAN_UVIS"] = "RadianceResidualMean_UVIS"

    cdf_var_map["RADIANCERESIDUALMEAN_NIR1"] = "RadianceResidualMean_NIR1"
    cdf_var_map["RADIANCERESIDUALMEAN_NIR2"] = "RadianceResidualMean_NIR2"

    cdf_var_map["RADIANCERESIDUALMEAN_SWIR1"] = "RadianceResidualMean_SWIR1"
    cdf_var_map["RADIANCERESIDUALMEAN_SWIR2"] = "RadianceResidualMean_SWIR2"
    cdf_var_map["RADIANCERESIDUALMEAN_SWIR3"] = "RadianceResidualMean_SWIR3"
    cdf_var_map["RADIANCERESIDUALMEAN_SWIR4"] = "RadianceResidualMean_SWIR4"

    cdf_var_map["RADIANCERESIDUALMEAN_TIR1"] = "RadianceResidualMean_TIR1"
    cdf_var_map["RADIANCERESIDUALMEAN_TIR2"] = "RadianceResidualMean_TIR2"
    cdf_var_map["RADIANCERESIDUALMEAN_TIR3"] = "RadianceResidualMean_TIR3"
    cdf_var_map["RADIANCERESIDUALMEAN_TIR4"] = "RadianceResidualMean_TIR4"

    # Radiance Residual RMS + Filter
    cdf_var_map["RADIANCERESIDUALRMS_UV1"] = "RadianceResidualRMS_UV1"
    cdf_var_map["RADIANCERESIDUALRMS_UV2"] = "RadianceResidualRMS_UV2"

    cdf_var_map["RADIANCERESIDUALRMS_VIS"] = "RadianceResidualRMS_VIS"

    cdf_var_map["RADIANCERESIDUALRMS_UVIS"] = "RadianceResidualRMS_UVIS"

    cdf_var_map["RADIANCERESIDUALRMS_NIR1"] = "RadianceResidualRMS_NIR1"
    cdf_var_map["RADIANCERESIDUALRMS_NIR2"] = "RadianceResidualRMS_NIR2"

    cdf_var_map["RADIANCERESIDUALRMS_SWIR1"] = "RadianceResidualRMS_SWIR1"
    cdf_var_map["RADIANCERESIDUALRMS_SWIR2"] = "RadianceResidualRMS_SWIR2"
    cdf_var_map["RADIANCERESIDUALRMS_SWIR3"] = "RadianceResidualRMS_SWIR3"
    cdf_var_map["RADIANCERESIDUALRMS_SWIR4"] = "RadianceResidualRMS_SWIR4"

    cdf_var_map["RADIANCERESIDUALRMS_TIR1"] = "RadianceResidualRMS_TIR1"
    cdf_var_map["RADIANCERESIDUALRMS_TIR2"] = "RadianceResidualRMS_TIR2"
    cdf_var_map["RADIANCERESIDUALRMS_TIR3"] = "RadianceResidualRMS_TIR3"
    cdf_var_map["RADIANCERESIDUALRMS_TIR4"] = "RadianceResidualRMS_TIR4"

    # legacy filter names
    cdf_var_map["RADIANCERESIDUALMEAN_1A1"] = "RadianceResidualMean_1A1"
    cdf_var_map["RADIANCERESIDUALMEAN_1B2"] = "RadianceResidualMean_1B2"
    cdf_var_map["RADIANCERESIDUALMEAN_2A1"] = "RadianceResidualMean_2A1"
    cdf_var_map["RADIANCERESIDUALMEAN_2B1"] = "RadianceResidualMean_2B1"

    cdf_var_map["RADIANCERESIDUALRMS_1A1"] = "RadianceResidualRMS_1A1"
    cdf_var_map["RADIANCERESIDUALRMS_1B2"] = "RadianceResidualRMS_1B2"
    cdf_var_map["RADIANCERESIDUALRMS_2A1"] = "RadianceResidualRMS_2A1"
    cdf_var_map["RADIANCERESIDUALRMS_2B1"] = "RadianceResidualRMS_2B1"

    cdf_var_map["RADIANCERESIDUALMEAN_FILTER"] = "RadianceResidualMean_Filter"
    cdf_var_map["RADIANCERESIDUALRMS_FILTER"] = "RadianceResidualRMS_Filter"

    cdf_var_map["RADIANCERESIDUALSLOPE_FILTER"] = "RadianceResidualSlope_Filter"
    cdf_var_map["RADIANCERESIDUALQUADRATIC_FILTER"] = "RadianceResidualQuadratic_Filter"

    cdf_var_map["RADIANCE_RESIDUAL_STDEV_CHANGE"] = "radiance_residual_stdev_change"

    cdf_var_map["RADIANCERESIDUALRMSRELATIVECONTINUUM_FILTER"] = (
        "radianceResidualRMSRelativeContinuum_filter"
    )
    cdf_var_map["RADIANCE_CONTINUUM_FILTER"] = "radiance_continuum_filter"

    cdf_var_map["RESIDUALNORMFINAL"] = "ResidualNormFinal"
    cdf_var_map["RESIDUALNORMINITIAL"] = "ResidualNormInitial"

    cdf_var_map["RETRIEVEINLOG"] = "RetrieveInLog"

    cdf_var_map["RTVMR"] = "RTVMR"
    cdf_var_map["RTVMR_ERRORMEASUREMENT"] = "RTVMR_ErrorMeasurement"
    cdf_var_map["RTVMR_ERROROBSERVATION"] = "RTVMR_ErrorObservation"
    cdf_var_map["RTVMR_ERRORTOTAL"] = "RTVMR_ErrorTotal"

    cdf_var_map["RTVMR_MAP"] = "RTVMR_Map"
    cdf_var_map["RTVMR_MAPPRESSURE"] = "RTVMR_MapPressure"
    cdf_var_map["RTVMR_PRESSURE"] = "RTVMR_Pressure"
    cdf_var_map["RTVMR_PRESSUREBOUNDLOWER"] = "RTVMR_PressureBoundLower"
    cdf_var_map["RTVMR_PRESSUREBOUNDUPPER"] = "RTVMR_PressureBoundUpper"

    cdf_var_map["RUN"] = "Run"
    cdf_var_map["SCAN"] = "Scan"
    cdf_var_map["SEQUENCE"] = "Sequence"

    cdf_var_map["SOLARZENITHANGLE"] = "SolarZenithAngle"

    cdf_var_map["SONDEAK"] = "SondeAK"
    cdf_var_map["SONDE"] = "Sonde"

    cdf_var_map["SOUNDINGID"] = "SoundingID"

    cdf_var_map["SPECIES"] = "Species"
    cdf_var_map["TRUE"] = "true"
    cdf_var_map["TRUE_AK"] = "true_ak"
    cdf_var_map["SPECIES_FM"] = "Species_FM"

    cdf_var_map["SPECIES_N2OCORRECTED"] = "Species_N2OCorrected"
    cdf_var_map["SPECIES_N2OCORRECTED_FM"] = "Species_N2OCorrected_FM"

    cdf_var_map["SPECIESRETRIEVALCONVERGED"] = "SpeciesRetrievalConverged"

    cdf_var_map["QUALITY"] = "Quality"

    cdf_var_map["SURFACEALTITUDE"] = "SurfaceAltitude"
    cdf_var_map["SURFACEELEVSTANDARDDEVIATION"] = "SurfaceEvStandardDeviation"
    cdf_var_map["SURFACEEMISSIONLAYER_QA"] = "SurfaceEmissionLayer_QA"
    cdf_var_map["SURFACEEMISSMEAN_QA"] = "SurfaceEmissMean_QA"
    cdf_var_map["SURFACEPRESSURE"] = "SurfacePressure"
    cdf_var_map["SURFACETEMPCONSTRAINT"] = "SurfaceTempConstraint"
    cdf_var_map["SURFACETEMPDEGREESOFFREEDOM"] = "SurfaceTempDegreesOfFreedom"
    cdf_var_map["SURFACETEMPERATURE"] = "SurfaceTemperature"
    cdf_var_map["SURFACETEMPERROR"] = "SurfaceTempError"
    cdf_var_map["SURFACETEMPINITIAL"] = "SurfaceTempInitial"
    cdf_var_map["SURFACETEMPOBSERVATIONERROR"] = "SurfaceTempObservationError"
    cdf_var_map["SURFACETEMPPRECISION"] = "SurfaceTempPrecision"
    cdf_var_map["SURFACETEMPVSAPRIORI_QA"] = "SurfaceTempVsApriori_QA"
    cdf_var_map["SURFACETEMPVSATMTEMP_QA"] = "SurfaceTempVsAtmTemp_QA"
    cdf_var_map["SURFACETYPEFOOTPRINT"] = "SurfaceTypeFootprint"

    cdf_var_map["T700"] = "T7000"

    cdf_var_map["TATM"] = "TATM"
    cdf_var_map["TATM_CONSTRAINTVECTOR"] = "TATM_ConstraintVector"
    cdf_var_map["TATM_DEVIATION"] = "TATM_Deviation"
    cdf_var_map["TATM_PROPAGATED_QA"] = "TATM_Propagated_QA"
    cdf_var_map["TATM_QA"] = "TATM_QA"
    cdf_var_map["TATM_RETRIEVAL_QA"] = "TATM_Retrieval_QA"
    cdf_var_map["TATM_SPECIES"] = "TATM_Species"

    cdf_var_map["TES_DOMINANTSURFACETYPE"] = "TES_DominantSurfaceType"
    cdf_var_map["TES_LATITUDE"] = "TES_Latitude"
    cdf_var_map["TES_LONGITUDE"] = "TES_Longitude"

    cdf_var_map["TES_OMI_DISTANCE"] = "TES_OMI_Distance"
    cdf_var_map["TES_OMI_MATCHED"] = "TES_OMI_Matched"
    cdf_var_map["TES_OMI_TIME_DIFFERENCE"] = "TES_OMI_Time_Difference"

    cdf_var_map["TES_RUN"] = "TES_Run"
    cdf_var_map["TES_SCAN"] = "TES_Scan"
    cdf_var_map["TES_SCANAVERAGEDCOUNT"] = "TES_ScanAveragedCount"
    cdf_var_map["TES_SEQUENCE"] = "TES_Sequence"

    cdf_var_map["TES_SURFACETEMPINITIAL"] = "TES_SurfaceTempInitial"
    cdf_var_map["TES_SURFACETYPEFOOTPRINT"] = "TES_SurfaceTypeFootprint"

    cdf_var_map["TES_TAI_TIME"] = "TES_TAI_Time"
    cdf_var_map["TES_UTC_TIME"] = "TES_UTC_Time"

    cdf_var_map["TGT_SPACECRAFTAZIMUTH"] = "Tgt_SpacecraftAzimuth"
    cdf_var_map["TGT_SPACECRAFTZENITH"] = "Tgt_SpacecraftZenith"

    cdf_var_map["THERMALCONTRAST"] = "ThermalConstrast"
    cdf_var_map["THERMALCONTRASTINITIAL"] = "ThermalConstrastInitial"

    cdf_var_map["TIME"] = "Time"

    cdf_var_map["TOTALCOLUMNDENSITY"] = "TotalColumnDensity"
    cdf_var_map["TOTALCOLUMNDENSITYERROR"] = "TotalColumnDensityError"
    cdf_var_map["TOTALCOLUMNDENSITYINITIAL"] = "TotalColumnDensityInitial"

    cdf_var_map["TOTALERROR"] = "TotalError"
    cdf_var_map["TOTALERRORCOVARIANCE"] = "TotalErrorCovariance"
    cdf_var_map["TOTALERROR_FM"] = "TotalError_FM"

    cdf_var_map["TROPOPAUSEPRESSURE"] = "TropopausePressure"

    cdf_var_map["TROPOSPHERICCOLUMN"] = "TroposphericColumn"
    cdf_var_map["TROPOSPHERICCOLUMNERROR"] = "TroposphericColumnError"
    cdf_var_map["TROPOSPHERICCOLUMNINITIAL"] = "TroposphericColumnInitial"

    cdf_var_map["UPPERTROPOSPHERICCOLUMN"] = "UpperTroposphericColumn"
    cdf_var_map["UPPERTROPOSPHERICCOLUMNERROR"] = "UpperTroposphericColumnError"
    cdf_var_map["UPPERTROPOSPHERICCOLUMNINITIAL"] = "UpperTroposphericColumnInitial"

    cdf_var_map["UTCTIME"] = "UTCTime"
    cdf_var_map["UT_HOUR"] = "UT_Hour"

    cdf_var_map["VARIABILITYCH4_QA"] = "VariabilityCH4_QA"
    cdf_var_map["VARIABILITYN2O_QA"] = "VariabilityN2O_QA"

    cdf_var_map["XPAN800_AK"] = "XPAN800_AK"
    cdf_var_map["XPAN800"] = "XPAN800"
    cdf_var_map["XPAN800_ERROROBS"] = "XPAN800_ErrorObs"
    cdf_var_map["XPAN800_ERRORSMOOTHING"] = "XPAN800_ErrorSmoothing"
    cdf_var_map["XPAN800_PRIOR"] = "XPAN800_Prior"

    cdf_var_map["XPAN_AK"] = "XPAN_AK"
    cdf_var_map["XPAN"] = "XPAN"
    cdf_var_map["XPAN_ERROROBS"] = "XPAN_ErrorObs"
    cdf_var_map["XPAN_ERRORSMOOTHING"] = "XPAN_ErrorSmoothing"
    cdf_var_map["XPAN_PRIOR"] = "XPANPrior"

    cdf_var_map["YEARFLOAT"] = "YearFloat"
    cdf_var_map["YYYYMMDD"] = "YYYYMMDD"


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
    "cdf_var_attributes",
    "cdf_var_names",
    "cdf_var_map",
]

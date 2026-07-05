import tools.frendy as fdy
import tools.depletion_chain as dc

import endf
import h5py
import os
import numpy as np
from scipy.integrate import quad
from concurrent.futures import ThreadPoolExecutor

####################################################################################
##                         MODIFY LINES IN THIS BLOCK
endf_base = "/home/hunter/Documents/nuclear_data/endf_libs/ENDF-B-VIII.0/neutrons/"
tsl_base = "/home/hunter/Documents/nuclear_data/endf_libs/ENDF-B-VIII.0/thermal_scatt/"
lib_name = "ENDF/B-VIII.0"

tendl_endf_base = "/home/hunter/Documents/nuclear_data/endf_libs/TENDL-2025/"
tendl_lib_name = "TENDL-2025"

temperatures = [293.0, 500.0, 600.0, 800.0, 1000.0, 1500.0, 2000.0]

num_threads = 15

lib_fname = "endf8_shem281.h5"

# Set the default group strucutre
fdy.set_default_group_structure("SHEM-281")
fdy.set_default_max_legendre_moments(3)

####################################################################################


# ==============================================================================
# Generate a U235 Fission Spectrum for computing specialized transport xs
def watt(
    E: np.ndarray | float, a: float = 0.988e6, b: float = 2.249e-6
) -> np.ndarray | float:
    """
    Default values from MCNP 6.2.0 manual for U235 with thermal neutron.
    Distribution is not normalized, because of the unknown integration bounds,
    which means the chi vector will need to be normalized anyway.
    """
    return np.exp(-E / a) * np.sinh(np.sqrt(b * E))


# Make the fission spectrum chi
group_bounds = fdy.get_default_group_structure().bounds
chi = np.zeros(group_bounds.size - 1)
for g in range(chi.size):
    chi[g] = quad(watt, group_bounds[g + 1], group_bounds[g])[0]
chi /= np.sum(chi)

# Zero all the low-energy group which contribute little to the fission spectrum
partial_sum = 0.0
for g in range(chi.size):
    partial_sum += chi[-(g + 1)]
    if partial_sum <= 0.0001:
        chi[-(g + 1)] = 0.0
    else:
        break

# Re-normalize the culled fission spectrum
chi /= np.sum(chi)

# ==============================================================================
# Functions used in processing


def get_tsl_temps(tsl_fname) -> list[float]:
    mat = endf.Material(tsl_fname)
    mf7mt4 = mat[7, 4]
    first_beta_data = mf7mt4["beta_data"][0]
    temps = []
    for i in range(len(first_beta_data)):
        temps.append(first_beta_data[i]["T"])
    return temps


def nuc(
    name: str,
    endf: str,
    dilutions: list[float] | None = None,
    chi: np.ndarray | None = None,
):
    fdy.process_nuclide(
        name,
        os.path.join(endf_base, endf),
        lib_name,
        temperatures,
        dilutions=dilutions,
        chi=chi,
    )


def tnuc(
    name: str,
    endf: str,
    dilutions: list[float] | None = None,
    chi: np.ndarray | None = None,
):
    fdy.process_nuclide(
        name,
        os.path.join(tendl_endf_base, endf),
        tendl_lib_name,
        temperatures,
        dilutions=dilutions,
        chi=chi,
    )


def tsl(
    name: str,
    endf: str,
    tsl_endf: str,
    tsl_type: str,
    dilutions: list[float] | None = None,
    chi: np.ndarray | None = None,
):
    tsl_temps = get_tsl_temps(os.path.join(tsl_base, tsl_endf))
    fdy.process_nuclide(
        name,
        os.path.join(endf_base, endf),
        lib_name,
        tsl_temps,
        dilutions=dilutions,
        chi=chi,
        tsl_fname=os.path.join(tsl_base, tsl_endf),
        tsl_type=tsl_type,
    )


# ==============================================================================
# List of all Nuclides

# fmt: off
inf = [1.0E10]
nuc_args = [("H1",      "n-001_H_001.endf",    inf, chi),
            ("H2",      "n-001_H_002.endf",    inf, chi),
            ("He3",     "n-002_He_003.endf",   inf),
            ("He4",     "n-002_He_004.endf",   inf),
            ("Li6",     "n-003_Li_006.endf",   inf),
            ("Li7",     "n-003_Li_007.endf",   inf),
            ("Be9",     "n-004_Be_009.endf",   inf, chi),
            ("B10",     "n-005_B_010.endf",    inf),
            ("B11",     "n-005_B_011.endf",    inf),
            ("C12",     "n-006_C_012.endf",    inf, chi),
            ("C13",     "n-006_C_013.endf",    inf, chi),
            ("N14",     "n-007_N_014.endf",    inf),
            ("N15",     "n-007_N_015.endf",    inf),
            ("O16",     "n-008_O_016.endf",    inf, chi),
            ("O17",     "n-008_O_017.endf",    inf, chi),
            ("O18",     "n-008_O_018.endf",    inf, chi),
            ("Na23",    "n-011_Na_023.endf",   inf),
            ("Mg24",    "n-012_Mg_024.endf",   inf),
            ("Mg25",    "n-012_Mg_025.endf",   inf),
            ("Mg26",    "n-012_Mg_026.endf",   inf),
            ("Al27",    "n-013_Al_027.endf",   inf),
            ("Si28",    "n-014_Si_028.endf",   inf),
            ("Si29",    "n-014_Si_029.endf",   inf),
            ("Si30",    "n-014_Si_030.endf",   inf),
            ("P31",     "n-015_P_031.endf",    inf),
            ("Ar36",    "n-018_Ar_036.endf",   inf),
            ("Ar38",    "n-018_Ar_038.endf",   inf),
            ("Ar40",    "n-018_Ar_040.endf",   inf),
            ("Ti46",    "n-022_Ti_046.endf",   None),
            ("Ti47",    "n-022_Ti_047.endf",   None),
            ("Ti48",    "n-022_Ti_048.endf",   None),
            ("Ti49",    "n-022_Ti_049.endf",   None),
            ("Ti50",    "n-022_Ti_050.endf",   None),
            ("Cr50",    "n-024_Cr_050.endf",   inf),
            ("Cr52",    "n-024_Cr_052.endf",   None),
            ("Cr53",    "n-024_Cr_053.endf",   None),
            ("Cr54",    "n-024_Cr_054.endf",   inf),
            ("Mn55",    "n-025_Mn_055.endf",   inf),
            ("Fe54",    "n-026_Fe_054.endf",   None),
            ("Fe55",    "n-026_Fe_055.endf",   inf),
            ("Fe56",    "n-026_Fe_056.endf",   None),
            ("Fe57",    "n-026_Fe_057.endf",   inf),
            ("Fe58",    "n-026_Fe_058.endf",   inf),
            ("Co59",    "n-027_Co_059.endf",   inf),
            ("Ni58",    "n-028_Ni_058.endf",   inf),
            ("Ni60",    "n-028_Ni_060.endf",   inf),
            ("Ni61",    "n-028_Ni_061.endf",   inf),
            ("Ni62",    "n-028_Ni_062.endf",   inf),
            ("Ni64",    "n-028_Ni_064.endf",   inf),
            ("Cu63",    "n-029_Cu_063.endf",   inf),
            ("Cu65",    "n-029_Cu_065.endf",   inf),
            ("Zn64",    "n-030_Zn_064.endf",   None),
            ("Zn66",    "n-030_Zn_066.endf",   None),
            ("Zn67",    "n-030_Zn_067.endf",   inf),
            ("Zn68",    "n-030_Zn_068.endf",   None),
            ("Zn70",    "n-030_Zn_070.endf",   inf),
            ("Br81",    "n-035_Br_081.endf",   inf),
            ("Kr82",    "n-036_Kr_082.endf",   inf),
            ("Kr83",    "n-036_Kr_083.endf",   inf),
            ("Kr84",    "n-036_Kr_084.endf",   inf),
            ("Kr85",    "n-036_Kr_085.endf",   inf),
            ("Kr86",    "n-036_Kr_086.endf",   inf),
            ("Sr89",    "n-038_Sr_089.endf",   inf),
            ("Sr90",    "n-038_Sr_090.endf",   inf),
            ("Y89",     "n-039_Y_089.endf",    inf),
            ("Y90",     "n-039_Y_090.endf",    inf),
            ("Y91",     "n-039_Y_091.endf",    inf),
            ("Zr90",    "n-040_Zr_090.endf",   None),
            ("Zr91",    "n-040_Zr_091.endf",   None),
            ("Zr92",    "n-040_Zr_092.endf",   None),
            ("Zr93",    "n-040_Zr_093.endf",   inf),
            ("Zr94",    "n-040_Zr_094.endf",   None),
            ("Zr95",    "n-040_Zr_095.endf",   inf),
            ("Zr96",    "n-040_Zr_096.endf",   None),
            ("Nb95",    "n-041_Nb_095.endf",   inf),
            ("Mo92",    "n-042_Mo_092.endf",   None),
            ("Mo94",    "n-042_Mo_094.endf",   None),
            ("Mo95",    "n-042_Mo_095.endf",   None),
            ("Mo96",    "n-042_Mo_096.endf",   None),
            ("Mo97",    "n-042_Mo_097.endf",   None),
            ("Mo98",    "n-042_Mo_098.endf",   None),
            ("Mo99",    "n-042_Mo_099.endf",   inf),
            ("Mo100",   "n-042_Mo_100.endf",   None),
            ("Tc99",    "n-043_Tc_099.endf",   None),
            ("Ru99",    "n-044_Ru_099.endf",   inf),
            ("Ru100",   "n-044_Ru_100.endf",   inf),
            ("Ru101",   "n-044_Ru_101.endf",   inf),
            ("Ru102",   "n-044_Ru_102.endf",   inf),
            ("Ru103",   "n-044_Ru_103.endf",   inf),
            ("Ru104",   "n-044_Ru_104.endf",   inf),
            ("Ru105",   "n-044_Ru_105.endf",   inf),
            ("Ru106",   "n-044_Ru_106.endf",   inf),
            ("Rh103",   "n-045_Rh_103.endf",   inf),
            ("Rh104",   "n-045_Rh_104.endf",   inf),
            ("Rh105",   "n-045_Rh_105.endf",   inf),
            ("Pd104",   "n-046_Pd_104.endf",   inf),
            ("Pd105",   "n-046_Pd_105.endf",   inf),
            ("Pd106",   "n-046_Pd_106.endf",   inf),
            ("Pd107",   "n-046_Pd_107.endf",   inf),
            ("Pd108",   "n-046_Pd_108.endf",   inf),
            ("Pd109",   "n-046_Pd_109.endf",   inf),
            ("Ag107",   "n-047_Ag_107.endf",   None),
            ("Ag109",   "n-047_Ag_109.endf",   None),
            ("Ag110m1", "n-047_Ag_110m1.endf", None),
            ("Ag111",   "n-047_Ag_111.endf",   None),
            ("Cd106",   "n-048_Cd_106.endf",   None),
            ("Cd108",   "n-048_Cd_108.endf",   None),
            ("Cd110",   "n-048_Cd_110.endf",   None),
            ("Cd111",   "n-048_Cd_111.endf",   None),
            ("Cd112",   "n-048_Cd_112.endf",   None),
            ("Cd113",   "n-048_Cd_113.endf",   None),
            ("Cd114",   "n-048_Cd_114.endf",   None),
            ("Cd116",   "n-048_Cd_116.endf",   None),
            ("In113",   "n-049_In_113.endf",   None),
            ("In115",   "n-049_In_115.endf",   None),
            ("Sn112",   "n-050_Sn_112.endf",   inf),
            ("Sn114",   "n-050_Sn_114.endf",   None),
            ("Sn115",   "n-050_Sn_115.endf",   None),
            ("Sn116",   "n-050_Sn_116.endf",   None),
            ("Sn117",   "n-050_Sn_117.endf",   None),
            ("Sn118",   "n-050_Sn_118.endf",   None),
            ("Sn119",   "n-050_Sn_119.endf",   None),
            ("Sn120",   "n-050_Sn_120.endf",   None),
            ("Sn122",   "n-050_Sn_122.endf",   None),
            ("Sn124",   "n-050_Sn_124.endf",   None),
            ("Sb121",   "n-051_Sb_121.endf",   inf),
            ("Sb123",   "n-051_Sb_123.endf",   inf),
            ("Sb125",   "n-051_Sb_125.endf",   inf),
            ("Sb126",   "n-051_Sb_126.endf",   inf),
            ("Te127m1", "n-052_Te_127m1.endf", inf),
            ("Te129m1", "n-052_Te_129m1.endf", inf),
            ("Te132",   "n-052_Te_132.endf",   inf),
            ("I127",    "n-053_I_127.endf",    None),
            ("I128",    "n-053_I_128.endf",    inf),
            ("I129",    "n-053_I_129.endf",    None),
            ("I130",    "n-053_I_130.endf",    inf),
            ("I131",    "n-053_I_131.endf",    inf),
            ("I132",    "n-053_I_132.endf",    inf),
            ("I135",    "n-053_I_135.endf",    inf),
            ("Xe128",   "n-054_Xe_128.endf",   inf),
            ("Xe129",   "n-054_Xe_129.endf",   None),
            ("Xe130",   "n-054_Xe_130.endf",   inf),
            ("Xe131",   "n-054_Xe_131.endf",   None),
            ("Xe132",   "n-054_Xe_132.endf",   inf),
            ("Xe133",   "n-054_Xe_133.endf",   inf),
            ("Xe134",   "n-054_Xe_134.endf",   inf),
            ("Xe135",   "n-054_Xe_135.endf",   inf),
            ("Xe136",   "n-054_Xe_136.endf",   inf),
            ("Cs133",   "n-055_Cs_133.endf",   inf),
            ("Cs134",   "n-055_Cs_134.endf",   inf),
            ("Cs135",   "n-055_Cs_135.endf",   inf),
            ("Cs136",   "n-055_Cs_136.endf",   inf),
            ("Cs137",   "n-055_Cs_137.endf",   inf),
            ("Ba134",   "n-056_Ba_134.endf",   inf),
            ("Ba137",   "n-056_Ba_137.endf",   inf),
            ("Ba140",   "n-056_Ba_140.endf",   inf),
            ("La139",   "n-057_La_139.endf",   inf),
            ("La140",   "n-057_La_140.endf",   inf),
            ("Ce140",   "n-058_Ce_140.endf",   inf),
            ("Ce141",   "n-058_Ce_141.endf",   inf),
            ("Ce142",   "n-058_Ce_142.endf",   inf),
            ("Ce143",   "n-058_Ce_143.endf",   inf),
            ("Ce144",   "n-058_Ce_144.endf",   inf),
            ("Pr141",   "n-059_Pr_141.endf",   inf),
            ("Pr142",   "n-059_Pr_142.endf",   inf),
            ("Pr143",   "n-059_Pr_143.endf",   inf),
            ("Nd142",   "n-060_Nd_142.endf",   inf),
            ("Nd143",   "n-060_Nd_143.endf",   inf),
            ("Nd144",   "n-060_Nd_144.endf",   inf),
            ("Nd145",   "n-060_Nd_145.endf",   inf),
            ("Nd146",   "n-060_Nd_146.endf",   inf),
            ("Nd147",   "n-060_Nd_147.endf",   inf),
            ("Nd148",   "n-060_Nd_148.endf",   inf),
            ("Nd149",   "n-060_Nd_149.endf",   inf),
            ("Nd150",   "n-060_Nd_150.endf",   inf),
            ("Pm147",   "n-061_Pm_147.endf",   inf),
            ("Pm148",   "n-061_Pm_148.endf",   inf),
            ("Pm148m1", "n-061_Pm_148m1.endf", inf),
            ("Pm149",   "n-061_Pm_149.endf",   inf),
            ("Pm150",   "n-061_Pm_150.endf",   inf),
            ("Pm151",   "n-061_Pm_151.endf",   inf),
            ("Sm147",   "n-062_Sm_147.endf",   None),
            ("Sm148",   "n-062_Sm_148.endf",   inf),
            ("Sm149",   "n-062_Sm_149.endf",   None),
            ("Sm150",   "n-062_Sm_150.endf",   None),
            ("Sm151",   "n-062_Sm_151.endf",   None),
            ("Sm152",   "n-062_Sm_152.endf",   None),
            ("Sm153",   "n-062_Sm_153.endf",   None),
            ("Sm154",   "n-062_Sm_154.endf",   None),
            ("Eu151",   "n-063_Eu_151.endf",   None),
            ("Eu152",   "n-063_Eu_152.endf",   None),
            ("Eu153",   "n-063_Eu_153.endf",   None),
            ("Eu154",   "n-063_Eu_154.endf",   None),
            ("Eu155",   "n-063_Eu_155.endf",   None),
            ("Eu156",   "n-063_Eu_156.endf",   inf),
            ("Eu157",   "n-063_Eu_157.endf",   inf),
            ("Gd152",   "n-064_Gd_152.endf",   None),
            ("Gd154",   "n-064_Gd_154.endf",   None),
            ("Gd155",   "n-064_Gd_155.endf",   None),
            ("Gd156",   "n-064_Gd_156.endf",   None),
            ("Gd157",   "n-064_Gd_157.endf",   None),
            ("Gd158",   "n-064_Gd_158.endf",   None),
            ("Gd159",   "n-064_Gd_158.endf",   inf),
            ("Gd160",   "n-064_Gd_160.endf",   None),
            ("Tb159",   "n-065_Tb_159.endf",   inf),
            ("Tb160",   "n-065_Tb_160.endf",   inf),
            ("Tb161",   "n-065_Tb_161.endf",   inf),
            ("Dy160",   "n-066_Dy_160.endf",   None),
            ("Dy161",   "n-066_Dy_161.endf",   None),
            ("Dy162",   "n-066_Dy_162.endf",   None),
            ("Dy163",   "n-066_Dy_163.endf",   None),
            ("Dy164",   "n-066_Dy_164.endf",   None),
            ("Ho165",   "n-067_Ho_165.endf",   inf),
            ("Er162",   "n-068_Er_162.endf",   inf),
            ("Er164",   "n-068_Er_164.endf",   inf),
            ("Er166",   "n-068_Er_166.endf",   inf),
            ("Er167",   "n-068_Er_167.endf",   inf),
            ("Er168",   "n-068_Er_168.endf",   inf),
            ("Er169",   "n-068_Er_169.endf",   inf),
            ("Er170",   "n-068_Er_170.endf",   inf),
            ("Tm169",   "n-069_Tm_169.endf",   inf),
            ("Tm170",   "n-069_Tm_170.endf",   inf),
            ("Tm171",   "n-069_Tm_171.endf",   inf),
            ("Hf174",   "n-072_Hf_174.endf",   None),
            ("Hf176",   "n-072_Hf_176.endf",   None),
            ("Hf177",   "n-072_Hf_177.endf",   None),
            ("Hf178",   "n-072_Hf_178.endf",   None),
            ("Hf179",   "n-072_Hf_179.endf",   None),
            ("Hf180",   "n-072_Hf_180.endf",   None),
            ("Hf181",   "n-072_Hf_181.endf",   None),
            ("Ta181",   "n-073_Ta_181.endf",   None),
            ("Ta182",   "n-073_Ta_182.endf",   inf),
            ("Th230",   "n-090_Th_230.endf",   None),
            ("Th231",   "n-090_Th_231.endf",   None),
            ("Th232",   "n-090_Th_232.endf",   None),
            ("Th233",   "n-090_Th_233.endf",   None),
            ("Th234",   "n-090_Th_234.endf",   None),
            ("Pa231",   "n-091_Pa_231.endf",   None),
            ("Pa232",   "n-091_Pa_232.endf",   None),
            ("Pa233",   "n-091_Pa_233.endf",   None),
            ("U232",    "n-092_U_232.endf",    None),
            ("U233",    "n-092_U_233.endf",    None),
            ("U234",    "n-092_U_234.endf",    None),
            ("U235",    "n-092_U_235.endf",    None),
            ("U236",    "n-092_U_236.endf",    None),
            ("U237",    "n-092_U_237.endf",    None),
            ("U238",    "n-092_U_238.endf",    None),
            ("U239",    "n-092_U_239.endf",    None),
            ("Np236",   "n-093_Np_236.endf",   None),
            ("Np237",   "n-093_Np_237.endf",   None),
            ("Np238",   "n-093_Np_238.endf",   None),
            ("Np239",   "n-093_Np_239.endf",   None),
            ("Pu236",   "n-094_Pu_236.endf",   None),
            ("Pu237",   "n-094_Pu_237.endf",   None),
            ("Pu238",   "n-094_Pu_238.endf",   None),
            ("Pu239",   "n-094_Pu_239.endf",   None),
            ("Pu240",   "n-094_Pu_240.endf",   None),
            ("Pu241",   "n-094_Pu_241.endf",   None),
            ("Pu242",   "n-094_Pu_242.endf",   None),
            ("Pu243",   "n-094_Pu_243.endf",   None),
            ("Pu244",   "n-094_Pu_244.endf",   None),
            ("Am241",   "n-095_Am_241.endf",   None),
            ("Am242",   "n-095_Am_242.endf",   None),
            ("Am242m1", "n-095_Am_242m1.endf", None),
            ("Am243",   "n-095_Am_243.endf",   None),
            ("Am244",   "n-095_Am_244.endf",   None),
            ("Am244m1", "n-095_Am_244m1.endf", None),
            ("Cm242",   "n-096_Cm_242.endf",   None),
            ("Cm243",   "n-096_Cm_243.endf",   None),
            ("Cm244",   "n-096_Cm_244.endf",   None),
            ("Cm245",   "n-096_Cm_245.endf",   None),
            ("Cm246",   "n-096_Cm_246.endf",   None)
           ]

tsl_args = [("H1_H2O", "n-001_H_001.endf", "tsl-HinH2O.endf", "hh2o", inf, chi),
            ("H2_D2O", "n-001_H_002.endf", "tsl-DinD2O.endf", "dd2o", inf, chi)
           ]

tnuc_args = [("Pr144",   "n-Pr144.tendl",  inf),
             ("Ag109m1", "n-Ag109m.tendl", inf),
             ("Ag110",   "n-Ag110.tendl",  inf),
             ("Xe135m1", "n-Xe135m.tendl", inf),
             ("Xe137",   "n-Xe137.tendl",  inf),
             ("Er171",   "n-Er171.tendl",  inf),
             ("Gd161",   "n-Gd161.tendl",  inf),
             ("Eu152m1", "n-Eu152m.tendl", inf),
             ("Rh102",   "n-Rh102.tendl",  inf),
             ("Rh102m1", "n-Rh102m.tendl", inf),
             ("Rh103m1", "n-Rh103m.tendl", inf),
             ("Rh105m1", "n-Rh105m.tendl", inf),
             ("Rh106",   "n-Rh106.tendl",  inf),
             ("Rh106m1", "n-Rh106m.tendl", inf),
             ("Np240",   "n-Np240.tendl",  inf),
             ("Np240m1", "n-Np240m.tendl", inf),
             ("Sm155",   "n-Sm155.tendl",  inf),
             ("Cd115",   "n-Cd115.tendl",  inf),
             ("Tc99m1",  "n-Tc099m.tendl", inf),
             ("Tc100",   "n-Tc100.tendl",  inf),
             ("Nd151",   "n-Nd151.tendl",  inf),
             ("Pa234",   "n-Pa234.tendl",  inf),
             ("Dy165",   "n-Dy165.tendl",  inf),
             ("Te127",   "n-Te127.tendl",  inf),
             ("Br82",    "n-Br082.tendl",  inf),
             ("Sb127",   "n-Sb127.tendl",  inf),
            ]
# fmt: on

# ==============================================================================
# Launch execution of all processing
all_nuclides = []
jobs = []
with ThreadPoolExecutor(max_workers=num_threads) as executor:
    # Submit all nuclide jobs
    for arg in nuc_args:
        jobs.append(executor.submit(nuc, *arg))
        all_nuclides.append(arg[0])
    # Submit all TSL jobs
    for arg in tsl_args:
        jobs.append(executor.submit(tsl, *arg))
        all_nuclides.append(arg[0])
    # Submit all TENDL nuclide jobs
    for arg in tnuc_args:
        jobs.append(executor.submit(tnuc, *arg))
        all_nuclides.append(arg[0])

# Check for errors
for i in range(len(jobs)):
    name = all_nuclides[i]
    if jobs[i].exception() is not None or not os.path.exists(f"{name}.h5"):
        print(f"ERROR: {name} was not processed")
        if jobs[i].exception() is not None:
            print(jobs[i].exception())
        else:
            print(f"No exception was raised, but no file {name}.h5 was found")
        print(end='\n\n')

# ==============================================================================
# Make Depletion Chain File
dep_chain = dc.build_depletion_chain("chain_casl_pwr.xml")
dep_chain.save("depletion_chain.h5")

# ==============================================================================
# Combine all HDF5 files into a single file
h5 = h5py.File(lib_fname, "w")
h5.attrs["group-structure"] = fdy.get_default_group_structure().name
h5.attrs["group-bounds"] = fdy.get_default_group_structure().bounds
h5.attrs["ngroups"] = fdy.get_default_group_structure().ngroups
h5.attrs["condensation-scheme"] = fdy.get_default_group_structure().condensation_scheme
h5.attrs["cmfd-condensation-scheme"] = (
    fdy.get_default_group_structure().cmfd_condensation_scheme
)
h5.attrs["first-resonance-group"] = fdy.get_default_group_structure().first_res_grp
h5.attrs["last-resonance-group"] = fdy.get_default_group_structure().last_res_grp
h5.attrs["library"] = lib_name

# Add all nuclides into the new file
print()
for i, nuc_name in enumerate(all_nuclides):
    if not os.path.exists(f"{nuc_name}.h5"):
        continue
    nuc_h5 = h5py.File(f"{nuc_name}.h5", "r")
    nuc_h5.copy(source=nuc_name, dest=h5)
    nuc_h5.close()
    os.remove(f"{nuc_name}.h5")

# Copy depletion chain
with h5py.File("depletion_chain.h5", "r") as dep_chain_h5:
    dep_chain_h5.copy(source="depletion-chain", dest=h5)
os.remove("depletion_chain.h5")

# Close file
h5.close()

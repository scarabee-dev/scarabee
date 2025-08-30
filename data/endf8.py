import tools.frendy as fdy
import tools.depletion_chain as dc
import h5py


####################################################################################
##                         MODIFY LINES IN THIS BLOCK
base = "/path/to/ENDF-B-VIII.0/neutrons/"
tslbase = "/path/to/ENDF-B-VIII.0/thermal_scatt/"
lib_name = "ENDF/B-VIII.0"

temps = [293.0, 500.0, 600.0, 800.0, 1000.0, 1500.0, 2000.0]

hdf5_file_name = "endf8_shem281.h5"

# Set the default group strucutre
fdy.set_default_group_structure("SHEM-281")
fdy.set_default_max_legendre_moments(3)

####################################################################################


# First we initialize the file with the depletion chain data.
# Must be done first as the save method will delete file first if it already exists !
dep_chain = dc.build_depletion_chain("chain_casl_pwr.xml")
dep_chain.save(hdf5_file_name)

# Initialize HDF5 with the library information
print(fdy.get_default_group_structure())
h5 = h5py.File(hdf5_file_name, "r+")
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

# We process U235 first so that we can steal its fission spectrum for
# performing transport correciton calculations
N = fdy.FrendyMG()
N.name = "U235"
N.endf_file = base + "n-092_U_235.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.process(h5)

chi = N.chi[:]

# Process TSL based evaluations
N = fdy.FrendyMG()
N.name = "H1_H2O"
N.endf_file = base + "n-001_H_001.endf"
N.tsl_file = tslbase + "tsl-HinH2O.endf"
N.tsl_type = "hh2o"
N.label = "H1 in H2O from ENDF/B-8.0"
N.temps = [
    283.6,
    293.6,
    300.0,
    323.6,
    350.0,
    373.6,
    400.0,
    423.6,
    450.0,
    473.6,
    500.0,
    523.6,
    550.0,
    573.6,
    600.0,
    623.6,
    650.0,
    800.0,
]
N.dilutions = [1.0e10]
N.process(h5, chi)  # Processed with in-scatter transport correction !

N = fdy.FrendyMG()
N.name = "H2_D2O"
N.endf_file = base + "n-001_H_002.endf"
N.tsl_file = tslbase + "tsl-DinD2O.endf"
N.tsl_type = "dd2o"
N.label = "H2 in D2O from ENDF/B-8.0"
N.temps = [
    283.6,
    293.6,
    300.0,
    323.6,
    350.0,
    373.6,
    400.0,
    423.6,
    450.0,
    473.6,
    500.0,
    523.6,
    550.0,
    573.6,
    600.0,
    623.6,
    650.0,
]
N.dilutions = [1.0e10]
N.process(h5, chi)  # Processed with in-scatter transport correction !

# Free Gas Nuclides
N = fdy.FrendyMG()
N.name = "H1"
N.endf_file = base + "n-001_H_001.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]
N.process(h5)

N = fdy.FrendyMG()
N.name = "H2"
N.endf_file = base + "n-001_H_002.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]
N.process(h5)

N = fdy.FrendyMG()
N.name = "He3"
N.endf_file = base + "n-002_He_003.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]
N.process(h5)

N = fdy.FrendyMG()
N.name = "He4"
N.endf_file = base + "n-002_He_004.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]
N.process(h5)

N = fdy.FrendyMG()
N.name = "Li6"
N.endf_file = base + "n-003_Li_006.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]
N.process(h5)

N = fdy.FrendyMG()
N.name = "Li7"
N.endf_file = base + "n-003_Li_007.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]
N.process(h5)

N = fdy.FrendyMG()
N.name = "Be9"
N.endf_file = base + "n-004_Be_009.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]
N.process(h5)

N = fdy.FrendyMG()
N.name = "B10"
N.endf_file = base + "n-005_B_010.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]
N.process(h5)

N = fdy.FrendyMG()
N.name = "B11"
N.endf_file = base + "n-005_B_011.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]
N.process(h5)

N = fdy.FrendyMG()
N.name = "C12"
N.endf_file = base + "n-006_C_012.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]
N.process(h5)

N = fdy.FrendyMG()
N.name = "C13"
N.endf_file = base + "n-006_C_013.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]
N.process(h5)

N = fdy.FrendyMG()
N.name = "N14"
N.endf_file = base + "n-007_N_014.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]
N.process(h5)

N = fdy.FrendyMG()
N.name = "N15"
N.endf_file = base + "n-007_N_015.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]
N.process(h5)

N = fdy.FrendyMG()
N.name = "O16"
N.endf_file = base + "n-008_O_016.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]
N.process(h5)

N = fdy.FrendyMG()
N.name = "O17"
N.endf_file = base + "n-008_O_017.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]
N.process(h5)

N = fdy.FrendyMG()
N.name = "O18"
N.endf_file = base + "n-008_O_018.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]
N.process(h5)

N = fdy.FrendyMG()
N.name = "Na23"
N.endf_file = base + "n-011_Na_023.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]
N.process(h5)

N = fdy.FrendyMG()
N.name = "Mg24"
N.endf_file = base + "n-012_Mg_024.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]
N.process(h5)

N = fdy.FrendyMG()
N.name = "Mg25"
N.endf_file = base + "n-012_Mg_025.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]
N.process(h5)

N = fdy.FrendyMG()
N.name = "Mg26"
N.endf_file = base + "n-012_Mg_026.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]
N.process(h5)

N = fdy.FrendyMG()
N.name = "Al27"
N.endf_file = base + "n-013_Al_027.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]
N.process(h5)

N = fdy.FrendyMG()
N.name = "Si28"
N.endf_file = base + "n-014_Si_028.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]
N.process(h5)

N = fdy.FrendyMG()
N.name = "Si29"
N.endf_file = base + "n-014_Si_029.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]
N.process(h5)

N = fdy.FrendyMG()
N.name = "Si30"
N.endf_file = base + "n-014_Si_030.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]
N.process(h5)

N = fdy.FrendyMG()
N.name = "Ar36"
N.endf_file = base + "n-018_Ar_036.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]
N.process(h5)

N = fdy.FrendyMG()
N.name = "Ar38"
N.endf_file = base + "n-018_Ar_038.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]
N.process(h5)

N = fdy.FrendyMG()
N.name = "Ar40"
N.endf_file = base + "n-018_Ar_040.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]
N.process(h5)

N = fdy.FrendyMG()
N.name = "Cr50"
N.endf_file = base + "n-024_Cr_050.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]
N.process(h5)

N = fdy.FrendyMG()
N.name = "Cr52"
N.endf_file = base + "n-024_Cr_052.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.process(h5)

N = fdy.FrendyMG()
N.name = "Cr53"
N.endf_file = base + "n-024_Cr_053.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.process(h5)

N = fdy.FrendyMG()
N.name = "Cr54"
N.endf_file = base + "n-024_Cr_054.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]
N.process(h5)

N = fdy.FrendyMG()
N.name = "Mn55"
N.endf_file = base + "n-025_Mn_055.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]
N.process(h5)

N = fdy.FrendyMG()
N.name = "Fe54"
N.endf_file = base + "n-026_Fe_054.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.process(h5)

N = fdy.FrendyMG()
N.name = "Fe55"
N.endf_file = base + "n-026_Fe_055.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]
N.process(h5)

N = fdy.FrendyMG()
N.name = "Fe56"
N.endf_file = base + "n-026_Fe_056.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.process(h5)

N = fdy.FrendyMG()
N.name = "Fe57"
N.endf_file = base + "n-026_Fe_057.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]
N.process(h5)

N = fdy.FrendyMG()
N.name = "Fe58"
N.endf_file = base + "n-026_Fe_058.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]
N.process(h5)

N = fdy.FrendyMG()
N.name = "Co59"
N.endf_file = base + "n-027_Co_059.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]
N.process(h5)

N = fdy.FrendyMG()
N.name = "Ni58"
N.endf_file = base + "n-028_Ni_058.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]
N.process(h5)

N = fdy.FrendyMG()
N.name = "Ni60"
N.endf_file = base + "n-028_Ni_060.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]
N.process(h5)

N = fdy.FrendyMG()
N.name = "Ni61"
N.endf_file = base + "n-028_Ni_061.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]
N.process(h5)

N = fdy.FrendyMG()
N.name = "Ni62"
N.endf_file = base + "n-028_Ni_062.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]
N.process(h5)

N = fdy.FrendyMG()
N.name = "Ni64"
N.endf_file = base + "n-028_Ni_064.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]
N.process(h5)

N = fdy.FrendyMG()
N.name = "Cu63"
N.endf_file = base + "n-029_Cu_063.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]
N.process(h5)

N = fdy.FrendyMG()
N.name = "Cu65"
N.endf_file = base + "n-029_Cu_065.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]
N.process(h5)

N = fdy.FrendyMG()
N.name = "Br81"
N.endf_file = base + "n-035_Br_081.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]
N.process(h5)

N = fdy.FrendyMG()
N.name = "Kr82"
N.endf_file = base + "n-036_Kr_082.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]
N.process(h5)

N = fdy.FrendyMG()
N.name = "Kr83"
N.endf_file = base + "n-036_Kr_083.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]
N.process(h5)

N = fdy.FrendyMG()
N.name = "Kr84"
N.endf_file = base + "n-036_Kr_084.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]
N.process(h5)

N = fdy.FrendyMG()
N.name = "Kr85"
N.endf_file = base + "n-036_Kr_085.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]
N.process(h5)

N = fdy.FrendyMG()
N.name = "Kr86"
N.endf_file = base + "n-036_Kr_086.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]
N.process(h5)

N = fdy.FrendyMG()
N.name = "Sr89"
N.endf_file = base + "n-038_Sr_089.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]
N.process(h5)

N = fdy.FrendyMG()
N.name = "Sr90"
N.endf_file = base + "n-038_Sr_090.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]
N.process(h5)

N = fdy.FrendyMG()
N.name = "Y89"
N.endf_file = base + "n-039_Y_089.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]
N.process(h5)

N = fdy.FrendyMG()
N.name = "Y90"
N.endf_file = base + "n-039_Y_090.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]
N.process(h5)

N = fdy.FrendyMG()
N.name = "Y91"
N.endf_file = base + "n-039_Y_091.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]
N.process(h5)

N = fdy.FrendyMG()
N.name = "Zr90"
N.endf_file = base + "n-040_Zr_090.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.process(h5)

N = fdy.FrendyMG()
N.name = "Zr91"
N.endf_file = base + "n-040_Zr_091.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.process(h5)

N = fdy.FrendyMG()
N.name = "Zr92"
N.endf_file = base + "n-040_Zr_092.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.process(h5)

N = fdy.FrendyMG()
N.name = "Zr93"
N.endf_file = base + "n-040_Zr_093.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]  # Fission Product
N.process(h5)

N = fdy.FrendyMG()
N.name = "Zr94"
N.endf_file = base + "n-040_Zr_094.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.process(h5)

N = fdy.FrendyMG()
N.name = "Zr95"
N.endf_file = base + "n-040_Zr_095.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]  # Fission Product
N.process(h5)

N = fdy.FrendyMG()
N.name = "Zr96"
N.endf_file = base + "n-040_Zr_096.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.process(h5)

N = fdy.FrendyMG()
N.name = "Nb95"
N.endf_file = base + "n-041_Nb_095.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]  # Fission Product
N.process(h5)

N = fdy.FrendyMG()
N.name = "Mo92"
N.endf_file = base + "n-042_Mo_092.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.process(h5)

N = fdy.FrendyMG()
N.name = "Mo94"
N.endf_file = base + "n-042_Mo_094.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.process(h5)

N = fdy.FrendyMG()
N.name = "Mo95"
N.endf_file = base + "n-042_Mo_095.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.process(h5)

N = fdy.FrendyMG()
N.name = "Mo96"
N.endf_file = base + "n-042_Mo_096.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.process(h5)

N = fdy.FrendyMG()
N.name = "Mo97"
N.endf_file = base + "n-042_Mo_097.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.process(h5)

N = fdy.FrendyMG()
N.name = "Mo98"
N.endf_file = base + "n-042_Mo_098.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.process(h5)

N = fdy.FrendyMG()
N.name = "Mo99"
N.endf_file = base + "n-042_Mo_099.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]  # Fission Product
N.process(h5)

N = fdy.FrendyMG()
N.name = "Mo100"
N.endf_file = base + "n-042_Mo_100.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.process(h5)

N = fdy.FrendyMG()
N.name = "Tc99"
N.endf_file = base + "n-043_Tc_099.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.process(h5)

N = fdy.FrendyMG()
N.name = "Ru99"
N.endf_file = base + "n-044_Ru_099.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]  # Fission Product
N.process(h5)

N = fdy.FrendyMG()
N.name = "Ru100"
N.endf_file = base + "n-044_Ru_100.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]  # Fission Product
N.process(h5)

N = fdy.FrendyMG()
N.name = "Ru101"
N.endf_file = base + "n-044_Ru_101.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]  # Fission Product
N.process(h5)

N = fdy.FrendyMG()
N.name = "Ru102"
N.endf_file = base + "n-044_Ru_102.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]  # Fission Product
N.process(h5)

N = fdy.FrendyMG()
N.name = "Ru103"
N.endf_file = base + "n-044_Ru_103.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]  # Fission Product
N.process(h5)

N = fdy.FrendyMG()
N.name = "Ru104"
N.endf_file = base + "n-044_Ru_104.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]  # Fission Product
N.process(h5)

N = fdy.FrendyMG()
N.name = "Ru105"
N.endf_file = base + "n-044_Ru_105.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]  # Fission Product
N.process(h5)

N = fdy.FrendyMG()
N.name = "Ru106"
N.endf_file = base + "n-044_Ru_106.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]  # Fission Product
N.process(h5)

N = fdy.FrendyMG()
N.name = "Rh103"
N.endf_file = base + "n-045_Rh_103.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]  # Fission Product
N.process(h5)

N = fdy.FrendyMG()
N.name = "Rh105"
N.endf_file = base + "n-045_Rh_105.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]  # Fission Product
N.process(h5)

N = fdy.FrendyMG()
N.name = "Pd104"
N.endf_file = base + "n-046_Pd_104.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]  # Fission Product
N.process(h5)

N = fdy.FrendyMG()
N.name = "Pd105"
N.endf_file = base + "n-046_Pd_105.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]  # Fission Product
N.process(h5)

N = fdy.FrendyMG()
N.name = "Pd106"
N.endf_file = base + "n-046_Pd_106.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]  # Fission Product
N.process(h5)

N = fdy.FrendyMG()
N.name = "Pd107"
N.endf_file = base + "n-046_Pd_107.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]  # Fission Product
N.process(h5)

N = fdy.FrendyMG()
N.name = "Pd108"
N.endf_file = base + "n-046_Pd_108.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]  # Fission Product
N.process(h5)

N = fdy.FrendyMG()
N.name = "Ag107"
N.endf_file = base + "n-047_Ag_107.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.process(h5)

N = fdy.FrendyMG()
N.name = "Ag109"
N.endf_file = base + "n-047_Ag_109.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.process(h5)

N = fdy.FrendyMG()
N.name = "Ag110m1"
N.endf_file = base + "n-047_Ag_110m1.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.process(h5)

N = fdy.FrendyMG()
N.name = "Ag111"
N.endf_file = base + "n-047_Ag_111.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.process(h5)

N = fdy.FrendyMG()
N.name = "Cd106"
N.endf_file = base + "n-048_Cd_106.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.process(h5)

N = fdy.FrendyMG()
N.name = "Cd108"
N.endf_file = base + "n-048_Cd_108.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.process(h5)

N = fdy.FrendyMG()
N.name = "Cd110"
N.endf_file = base + "n-048_Cd_110.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.process(h5)

N = fdy.FrendyMG()
N.name = "Cd111"
N.endf_file = base + "n-048_Cd_111.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.process(h5)

N = fdy.FrendyMG()
N.name = "Cd112"
N.endf_file = base + "n-048_Cd_112.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.process(h5)

N = fdy.FrendyMG()
N.name = "Cd113"
N.endf_file = base + "n-048_Cd_113.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.process(h5)

N = fdy.FrendyMG()
N.name = "Cd114"
N.endf_file = base + "n-048_Cd_114.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.process(h5)

N = fdy.FrendyMG()
N.name = "Cd116"
N.endf_file = base + "n-048_Cd_116.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.process(h5)

N = fdy.FrendyMG()
N.name = "In113"
N.endf_file = base + "n-049_In_113.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.process(h5)

N = fdy.FrendyMG()
N.name = "In115"
N.endf_file = base + "n-049_In_115.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.process(h5)

N = fdy.FrendyMG()
N.name = "Sn112"
N.endf_file = base + "n-050_Sn_112.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]
N.process(h5)

N = fdy.FrendyMG()
N.name = "Sn114"
N.endf_file = base + "n-050_Sn_114.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.process(h5)

N = fdy.FrendyMG()
N.name = "Sn115"
N.endf_file = base + "n-050_Sn_115.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.process(h5)

N = fdy.FrendyMG()
N.name = "Sn116"
N.endf_file = base + "n-050_Sn_116.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.process(h5)

N = fdy.FrendyMG()
N.name = "Sn117"
N.endf_file = base + "n-050_Sn_117.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.process(h5)

N = fdy.FrendyMG()
N.name = "Sn118"
N.endf_file = base + "n-050_Sn_118.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.process(h5)

N = fdy.FrendyMG()
N.name = "Sn119"
N.endf_file = base + "n-050_Sn_119.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.process(h5)

N = fdy.FrendyMG()
N.name = "Sn120"
N.endf_file = base + "n-050_Sn_120.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.process(h5)

N = fdy.FrendyMG()
N.name = "Sn122"
N.endf_file = base + "n-050_Sn_122.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.process(h5)

N = fdy.FrendyMG()
N.name = "Sn124"
N.endf_file = base + "n-050_Sn_124.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.process(h5)

N = fdy.FrendyMG()
N.name = "Sb121"
N.endf_file = base + "n-051_Sb_121.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]  # Fission Product
N.process(h5)

N = fdy.FrendyMG()
N.name = "Sb123"
N.endf_file = base + "n-051_Sb_123.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]  # Fission Product
N.process(h5)

N = fdy.FrendyMG()
N.name = "Sb125"
N.endf_file = base + "n-051_Sb_125.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]  # Fission Product
N.process(h5)

N = fdy.FrendyMG()
N.name = "Te127m1"
N.endf_file = base + "n-052_Te_127m1.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]  # Fission Product
N.process(h5)

N = fdy.FrendyMG()
N.name = "Te129m1"
N.endf_file = base + "n-052_Te_129m1.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]  # Fission Product
N.process(h5)

N = fdy.FrendyMG()
N.name = "Te132"
N.endf_file = base + "n-052_Te_132.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]  # Fission Product
N.process(h5)

N = fdy.FrendyMG()
N.name = "I127"
N.endf_file = base + "n-053_I_127.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.process(h5)

N = fdy.FrendyMG()
N.name = "I129"
N.endf_file = base + "n-053_I_129.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.process(h5)

N = fdy.FrendyMG()
N.name = "I131"
N.endf_file = base + "n-053_I_131.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]  # Fission Product
N.process(h5)

N = fdy.FrendyMG()
N.name = "I135"
N.endf_file = base + "n-053_I_135.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]  # Fission Product
N.process(h5)

N = fdy.FrendyMG()
N.name = "Xe128"
N.endf_file = base + "n-054_Xe_128.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]  # Fission Product
N.process(h5)

N = fdy.FrendyMG()
N.name = "Xe129"
N.endf_file = base + "n-054_Xe_129.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.process(h5)

N = fdy.FrendyMG()
N.name = "Xe130"
N.endf_file = base + "n-054_Xe_130.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]  # Fission Product
N.process(h5)

N = fdy.FrendyMG()
N.name = "Xe131"
N.endf_file = base + "n-054_Xe_131.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.process(h5)

N = fdy.FrendyMG()
N.name = "Xe132"
N.endf_file = base + "n-054_Xe_132.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]  # Fission Product
N.process(h5)

N = fdy.FrendyMG()
N.name = "Xe133"
N.endf_file = base + "n-054_Xe_133.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]  # Fission Product
N.process(h5)

N = fdy.FrendyMG()
N.name = "Xe134"
N.endf_file = base + "n-054_Xe_134.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]  # Fission Product
N.process(h5)

N = fdy.FrendyMG()
N.name = "Xe135"
N.endf_file = base + "n-054_Xe_135.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]  # Fission Product
N.process(h5)

N = fdy.FrendyMG()
N.name = "Xe136"
N.endf_file = base + "n-054_Xe_136.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]  # Fission Product
N.process(h5)

N = fdy.FrendyMG()
N.name = "Cs133"
N.endf_file = base + "n-055_Cs_133.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]  # Fission Product
N.process(h5)

N = fdy.FrendyMG()
N.name = "Cs134"
N.endf_file = base + "n-055_Cs_134.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]  # Fission Product
N.process(h5)

N = fdy.FrendyMG()
N.name = "Cs135"
N.endf_file = base + "n-055_Cs_135.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]  # Fission Product
N.process(h5)

N = fdy.FrendyMG()
N.name = "Cs136"
N.endf_file = base + "n-055_Cs_136.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]  # Fission Product
N.process(h5)

N = fdy.FrendyMG()
N.name = "Cs137"
N.endf_file = base + "n-055_Cs_137.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]  # Fission Product
N.process(h5)

N = fdy.FrendyMG()
N.name = "Ba134"
N.endf_file = base + "n-056_Ba_134.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]  # Fission Product
N.process(h5)

N = fdy.FrendyMG()
N.name = "Ba137"
N.endf_file = base + "n-056_Ba_137.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]  # Fission Product
N.process(h5)

N = fdy.FrendyMG()
N.name = "Ba140"
N.endf_file = base + "n-056_Ba_140.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]  # Fission Product
N.process(h5)

N = fdy.FrendyMG()
N.name = "La139"
N.endf_file = base + "n-057_La_139.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]  # Fission Product
N.process(h5)

N = fdy.FrendyMG()
N.name = "La140"
N.endf_file = base + "n-057_La_140.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]  # Fission Product
N.process(h5)

N = fdy.FrendyMG()
N.name = "Ce140"
N.endf_file = base + "n-058_Ce_140.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]  # Fission Product
N.process(h5)

N = fdy.FrendyMG()
N.name = "Ce141"
N.endf_file = base + "n-058_Ce_141.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]  # Fission Product
N.process(h5)

N = fdy.FrendyMG()
N.name = "Ce142"
N.endf_file = base + "n-058_Ce_142.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]  # Fission Product
N.process(h5)

N = fdy.FrendyMG()
N.name = "Ce143"
N.endf_file = base + "n-058_Ce_143.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]  # Fission Product
N.process(h5)

N = fdy.FrendyMG()
N.name = "Ce144"
N.endf_file = base + "n-058_Ce_144.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]  # Fission Product
N.process(h5)

N = fdy.FrendyMG()
N.name = "Pr141"
N.endf_file = base + "n-059_Pr_141.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]  # Fission Product
N.process(h5)

N = fdy.FrendyMG()
N.name = "Pr143"
N.endf_file = base + "n-059_Pr_143.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]  # Fission Product
N.process(h5)

N = fdy.FrendyMG()
N.name = "Nd142"
N.endf_file = base + "n-060_Nd_142.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]  # Fission Product
N.process(h5)

N = fdy.FrendyMG()
N.name = "Nd143"
N.endf_file = base + "n-060_Nd_143.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]  # Fission Product
N.process(h5)

N = fdy.FrendyMG()
N.name = "Nd144"
N.endf_file = base + "n-060_Nd_144.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]  # Fission Product
N.process(h5)

N = fdy.FrendyMG()
N.name = "Nd145"
N.endf_file = base + "n-060_Nd_145.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]  # Fission Product
N.process(h5)

N = fdy.FrendyMG()
N.name = "Nd146"
N.endf_file = base + "n-060_Nd_146.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]  # Fission Product
N.process(h5)

N = fdy.FrendyMG()
N.name = "Nd147"
N.endf_file = base + "n-060_Nd_147.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]  # Fission Product
N.process(h5)

N = fdy.FrendyMG()
N.name = "Nd148"
N.endf_file = base + "n-060_Nd_148.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]  # Fission Product
N.process(h5)

N = fdy.FrendyMG()
N.name = "Nd150"
N.endf_file = base + "n-060_Nd_150.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]  # Fission Product
N.process(h5)

N = fdy.FrendyMG()
N.name = "Pm147"
N.endf_file = base + "n-061_Pm_147.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]  # Fission Product
N.process(h5)

N = fdy.FrendyMG()
N.name = "Pm148"
N.endf_file = base + "n-061_Pm_148.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]  # Fission Product
N.process(h5)

N = fdy.FrendyMG()
N.name = "Pm148m1"
N.endf_file = base + "n-061_Pm_148m1.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]  # Fission Product
N.process(h5)

N = fdy.FrendyMG()
N.name = "Pm149"
N.endf_file = base + "n-061_Pm_149.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]  # Fission Product
N.process(h5)

N = fdy.FrendyMG()
N.name = "Pm151"
N.endf_file = base + "n-061_Pm_151.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]  # Fission Product
N.process(h5)

N = fdy.FrendyMG()
N.name = "Sm147"
N.endf_file = base + "n-062_Sm_147.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.process(h5)

N = fdy.FrendyMG()
N.name = "Sm148"
N.endf_file = base + "n-062_Sm_148.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]  # Fission Product
N.process(h5)

N = fdy.FrendyMG()
N.name = "Sm149"
N.endf_file = base + "n-062_Sm_149.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.process(h5)

N = fdy.FrendyMG()
N.name = "Sm150"
N.endf_file = base + "n-062_Sm_150.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.process(h5)

N = fdy.FrendyMG()
N.name = "Sm151"
N.endf_file = base + "n-062_Sm_151.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.process(h5)

N = fdy.FrendyMG()
N.name = "Sm152"
N.endf_file = base + "n-062_Sm_152.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.process(h5)

N = fdy.FrendyMG()
N.name = "Sm153"
N.endf_file = base + "n-062_Sm_153.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.process(h5)

N = fdy.FrendyMG()
N.name = "Sm154"
N.endf_file = base + "n-062_Sm_154.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.process(h5)

N = fdy.FrendyMG()
N.name = "Eu151"
N.endf_file = base + "n-063_Eu_151.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.process(h5)

N = fdy.FrendyMG()
N.name = "Eu152"
N.endf_file = base + "n-063_Eu_152.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.process(h5)

N = fdy.FrendyMG()
N.name = "Eu153"
N.endf_file = base + "n-063_Eu_153.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.process(h5)

N = fdy.FrendyMG()
N.name = "Eu154"
N.endf_file = base + "n-063_Eu_154.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.process(h5)

N = fdy.FrendyMG()
N.name = "Eu155"
N.endf_file = base + "n-063_Eu_155.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.process(h5)

N = fdy.FrendyMG()
N.name = "Eu156"
N.endf_file = base + "n-063_Eu_156.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]  # Fission Product
N.process(h5)

N = fdy.FrendyMG()
N.name = "Gd152"
N.endf_file = base + "n-064_Gd_152.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.process(h5)

N = fdy.FrendyMG()
N.name = "Gd154"
N.endf_file = base + "n-064_Gd_154.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.process(h5)

N = fdy.FrendyMG()
N.name = "Gd155"
N.endf_file = base + "n-064_Gd_155.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.process(h5)

N = fdy.FrendyMG()
N.name = "Gd156"
N.endf_file = base + "n-064_Gd_156.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.process(h5)

N = fdy.FrendyMG()
N.name = "Gd157"
N.endf_file = base + "n-064_Gd_157.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.process(h5)

N = fdy.FrendyMG()
N.name = "Gd158"
N.endf_file = base + "n-064_Gd_158.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.process(h5)

N = fdy.FrendyMG()
N.name = "Gd160"
N.endf_file = base + "n-064_Gd_160.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.process(h5)

N = fdy.FrendyMG()
N.name = "Tb159"
N.endf_file = base + "n-065_Tb_159.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]  # Fission Product
N.process(h5)

N = fdy.FrendyMG()
N.name = "Tb160"
N.endf_file = base + "n-065_Tb_160.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]  # Fission Product
N.process(h5)

N = fdy.FrendyMG()
N.name = "Tb161"
N.endf_file = base + "n-065_Tb_161.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]  # Fission Product
N.process(h5)

N = fdy.FrendyMG()
N.name = "Dy160"
N.endf_file = base + "n-066_Dy_160.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.process(h5)

N = fdy.FrendyMG()
N.name = "Dy161"
N.endf_file = base + "n-066_Dy_161.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.process(h5)

N = fdy.FrendyMG()
N.name = "Dy162"
N.endf_file = base + "n-066_Dy_162.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.process(h5)

N = fdy.FrendyMG()
N.name = "Dy163"
N.endf_file = base + "n-066_Dy_163.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.process(h5)

N = fdy.FrendyMG()
N.name = "Dy164"
N.endf_file = base + "n-066_Dy_164.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.process(h5)

N = fdy.FrendyMG()
N.name = "Ho165"
N.endf_file = base + "n-067_Ho_165.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]  # Fission Product
N.process(h5)

N = fdy.FrendyMG()
N.name = "Er162"
N.endf_file = base + "n-068_Er_162.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]  # Fission Product
N.process(h5)

N = fdy.FrendyMG()
N.name = "Er164"
N.endf_file = base + "n-068_Er_164.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]  # Fission Product
N.process(h5)

N = fdy.FrendyMG()
N.name = "Er166"
N.endf_file = base + "n-068_Er_166.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]  # Fission Product
N.process(h5)

N = fdy.FrendyMG()
N.name = "Er167"
N.endf_file = base + "n-068_Er_167.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]  # Fission Product
N.process(h5)

N = fdy.FrendyMG()
N.name = "Er168"
N.endf_file = base + "n-068_Er_168.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]  # Fission Product
N.process(h5)

N = fdy.FrendyMG()
N.name = "Er169"
N.endf_file = base + "n-068_Er_169.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]  # Fission Product
N.process(h5)

N = fdy.FrendyMG()
N.name = "Er170"
N.endf_file = base + "n-068_Er_170.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]  # Fission Product
N.process(h5)

N = fdy.FrendyMG()
N.name = "Tm169"
N.endf_file = base + "n-069_Tm_169.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]  # Fission Product
N.process(h5)

N = fdy.FrendyMG()
N.name = "Tm170"
N.endf_file = base + "n-069_Tm_170.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]  # Fission Product
N.process(h5)

N = fdy.FrendyMG()
N.name = "Tm171"
N.endf_file = base + "n-069_Tm_171.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]  # Fission Product
N.process(h5)

N = fdy.FrendyMG()
N.name = "Hf174"
N.endf_file = base + "n-072_Hf_174.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.process(h5)

N = fdy.FrendyMG()
N.name = "Hf176"
N.endf_file = base + "n-072_Hf_176.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.process(h5)

N = fdy.FrendyMG()
N.name = "Hf177"
N.endf_file = base + "n-072_Hf_177.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.process(h5)

N = fdy.FrendyMG()
N.name = "Hf178"
N.endf_file = base + "n-072_Hf_178.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.process(h5)

N = fdy.FrendyMG()
N.name = "Hf179"
N.endf_file = base + "n-072_Hf_179.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.process(h5)

N = fdy.FrendyMG()
N.name = "Hf180"
N.endf_file = base + "n-072_Hf_180.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.process(h5)

N = fdy.FrendyMG()
N.name = "Hf181"
N.endf_file = base + "n-072_Hf_181.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.process(h5)

N = fdy.FrendyMG()
N.name = "Ta181"
N.endf_file = base + "n-073_Ta_181.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.process(h5)

N = fdy.FrendyMG()
N.name = "Ta182"
N.endf_file = base + "n-073_Ta_182.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.dilutions = [1.0e10]  # Fission Product
N.process(h5)

N = fdy.FrendyMG()
N.name = "Th230"
N.endf_file = base + "n-090_Th_230.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.process(h5)

N = fdy.FrendyMG()
N.name = "Th231"
N.endf_file = base + "n-090_Th_231.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.process(h5)

N = fdy.FrendyMG()
N.name = "Th232"
N.endf_file = base + "n-090_Th_232.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.process(h5)

N = fdy.FrendyMG()
N.name = "Th234"
N.endf_file = base + "n-090_Th_234.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.process(h5)

N = fdy.FrendyMG()
N.name = "Pa231"
N.endf_file = base + "n-091_Pa_231.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.process(h5)

N = fdy.FrendyMG()
N.name = "Pa232"
N.endf_file = base + "n-091_Pa_232.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.process(h5)

N = fdy.FrendyMG()
N.name = "Pa233"
N.endf_file = base + "n-091_Pa_233.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.process(h5)

N = fdy.FrendyMG()
N.name = "U232"
N.endf_file = base + "n-092_U_232.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.process(h5)

N = fdy.FrendyMG()
N.name = "U233"
N.endf_file = base + "n-092_U_233.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.process(h5)

N = fdy.FrendyMG()
N.name = "U234"
N.endf_file = base + "n-092_U_234.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.process(h5)

N = fdy.FrendyMG()
N.name = "U236"
N.endf_file = base + "n-092_U_236.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.process(h5)

N = fdy.FrendyMG()
N.name = "U237"
N.endf_file = base + "n-092_U_237.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.process(h5)

N = fdy.FrendyMG()
N.name = "U238"
N.endf_file = base + "n-092_U_238.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.process(h5)

N = fdy.FrendyMG()
N.name = "Np236"
N.endf_file = base + "n-093_Np_236.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.process(h5)

N = fdy.FrendyMG()
N.name = "Np237"
N.endf_file = base + "n-093_Np_237.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.process(h5)

N = fdy.FrendyMG()
N.name = "Np238"
N.endf_file = base + "n-093_Np_238.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.process(h5)

N = fdy.FrendyMG()
N.name = "Np239"
N.endf_file = base + "n-093_Np_239.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.process(h5)

N = fdy.FrendyMG()
N.name = "Pu236"
N.endf_file = base + "n-094_Pu_236.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.process(h5)

N = fdy.FrendyMG()
N.name = "Pu237"
N.endf_file = base + "n-094_Pu_237.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.process(h5)

N = fdy.FrendyMG()
N.name = "Pu238"
N.endf_file = base + "n-094_Pu_238.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.process(h5)

N = fdy.FrendyMG()
N.name = "Pu239"
N.endf_file = base + "n-094_Pu_239.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.process(h5)

N = fdy.FrendyMG()
N.name = "Pu240"
N.endf_file = base + "n-094_Pu_240.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.process(h5)

N = fdy.FrendyMG()
N.name = "Pu241"
N.endf_file = base + "n-094_Pu_241.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.process(h5)

N = fdy.FrendyMG()
N.name = "Pu242"
N.endf_file = base + "n-094_Pu_242.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.process(h5)

N = fdy.FrendyMG()
N.name = "Pu243"
N.endf_file = base + "n-094_Pu_243.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.process(h5)

N = fdy.FrendyMG()
N.name = "Pu244"
N.endf_file = base + "n-094_Pu_244.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.process(h5)

N = fdy.FrendyMG()
N.name = "Am241"
N.endf_file = base + "n-095_Am_241.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.process(h5)

N = fdy.FrendyMG()
N.name = "Am242m1"
N.endf_file = base + "n-095_Am_242m1.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.process(h5)

N = fdy.FrendyMG()
N.name = "Am243"
N.endf_file = base + "n-095_Am_243.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.process(h5)

N = fdy.FrendyMG()
N.name = "Cm242"
N.endf_file = base + "n-096_Cm_242.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.process(h5)

N = fdy.FrendyMG()
N.name = "Cm243"
N.endf_file = base + "n-096_Cm_243.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.process(h5)

N = fdy.FrendyMG()
N.name = "Cm244"
N.endf_file = base + "n-096_Cm_244.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.process(h5)

N = fdy.FrendyMG()
N.name = "Cm245"
N.endf_file = base + "n-096_Cm_245.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.process(h5)

N = fdy.FrendyMG()
N.name = "Cm246"
N.endf_file = base + "n-096_Cm_246.endf"
N.label = N.name + " from ENDF/B-8.0"
N.temps = temps
N.process(h5)

h5.close()

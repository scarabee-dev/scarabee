from numpy import mod
from scarabee import (
    BoundaryCondition,
    NDLibrary,
    MaterialComposition,
    Material,
    Fraction,
    MixingFraction,
    DensityUnits,
    set_output_file,
    mix_materials
)
from scarabee.reseau import FuelPin, GuideTube, PWRAssembly, Symmetry

name = "F30_20"

set_output_file(name+"_out.txt")

ndl = NDLibrary()

# Define all Materials
Fuel3Comp = MaterialComposition(Fraction.Atoms, name='Fuel 3.0 %')
Fuel3Comp.add_leu(enrichment=3., fraction=1.)
Fuel3Comp.add_element("O", 2.)
Fuel3 = Material(Fuel3Comp, 575., 10.96, DensityUnits.g_cm3, ndl)
Fuel3.max_legendre_order = 3

Gd2O3Comp = MaterialComposition(Fraction.Atoms, name='Gadolinia')
Gd2O3Comp.add_element("Gd", 2.)
Gd2O3Comp.add_element("O", 3.)
Gd2O3 = Material(Gd2O3Comp, 575., 7.07, DensityUnits.g_cm3, ndl)
Gd2O3.max_legendre_order = 3

# Depleted uranium is used in the Gd pins !
Fuel0Comp = MaterialComposition(Fraction.Atoms, name='Fuel 3.0 %')
Fuel0Comp.add_leu(enrichment=0.25, fraction=1.)
Fuel0Comp.add_element("O", 2.)
Fuel0 = Material(Fuel0Comp, 575., 10.96, DensityUnits.g_cm3, ndl)
Fuel0.max_legendre_order = 3

FuelGad = mix_materials([Fuel0, Gd2O3], [0.92, 0.08], MixingFraction.Weight, ndl)
FuelGad.max_legendre_order = 3

CladComp = MaterialComposition(Fraction.Weight, name="Zircaloy 4")
CladComp.add_element('O', 0.00125)
CladComp.add_element('Cr', 0.0010)
CladComp.add_element('Fe', 0.0021)
CladComp.add_element('Zr', 0.98115)
CladComp.add_element('Sn', 0.0145)
Clad = Material(CladComp, 575., 6.55, DensityUnits.g_cm3 , ndl)
Clad.max_legendre_order = 3

HeComp = MaterialComposition(Fraction.Atoms, name="He Gas")
HeComp.add_element("He", 1.)
He = Material(HeComp, 575., 0.0015981, DensityUnits.g_cm3, ndl) 
He.max_legendre_order = 3

# Define a guide tube
gt = GuideTube(inner_radius=0.5725, outer_radius=0.6125, clad=Clad)

# Define fuel pin
fp = FuelPin(fuel=Fuel3, fuel_radius=0.3975, gap=He, gap_radius=0.4125,
             clad=Clad, clad_radius=0.4750)

# Define Gadolinium fuel pin
gp = FuelPin(fuel=FuelGad, fuel_radius=0.3975, gap=He, gap_radius=0.4125,
             clad=Clad, clad_radius=0.4750, num_fuel_rings=15)

# Define assembly
cells = [[fp, fp, fp, fp, fp, fp, fp, fp, fp],
         [fp, fp, fp, fp, fp, fp, fp, fp, fp],
         [gt, fp, fp, gt, fp, fp, gp, fp, fp],
         [fp, gp, fp, fp, fp, gt, fp, fp, fp],
         [fp, fp, fp, fp, gp, fp, fp, fp, fp],
         [gt, fp, fp, gt, fp, fp, gt, fp, fp],
         [fp, fp, gp, fp, fp, fp, fp, fp, fp],
         [fp, fp, fp, fp, fp, gp, fp, fp, fp],
         [gt, fp, fp, gt, fp, fp, gt, fp, fp]]
asmbly = PWRAssembly(
    pitch=1.26,
    assembly_pitch=21.50364,
    shape=(17, 17),
    symmetry=Symmetry.Quarter,
    moderator_pressure=15.5,
    moderator_temp=575.0,
    boron_ppm=960.0,
    cells=cells,
    ndl=ndl,
    moderator_legendre_order=3
)
asmbly.anisotropic = True
asmbly.cmfd = True
#asmbly.cmfd_condensation_scheme = [[0, ndl.ngroups-1]]
asmbly.flux_tolerance = 1.E-14
asmbly.solve()


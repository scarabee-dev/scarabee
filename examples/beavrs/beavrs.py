import scarabee as scb
import scarabee.reseau as scr
import scarabee.coeur as scc
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import copy

if not os.path.isfile("BEAVRS_out.txt"):
    scb.set_output_file("BEAVRS_out.txt")

scb.scarabee_log(scb.LogLevel.Info, "2D Variant of BEAVRS Benchmark")
scb.scarabee_log(scb.LogLevel.Info, "")
scb.scarabee_log(scb.LogLevel.Info, "Assembly Calculations")
scb.scarabee_log(scb.LogLevel.Info, "")

ndl = scb.NDLibrary("/home/hunter/Documents/nuclear_data/scarabee/endf8_shem281.h5")

# Uniform material temperature across the core
T = 575.0
boron_ppm = 975.0
mod_pressure = 15.5132

# Legendre order
L = 3

pin_pitch = 1.25984
asmbly_pitch = 21.50364

# Fuel pin cell radii
fuel_r = 0.39218
gap_r = 0.40005
clad_r = 0.45720

# Guide tube cell radii
inner_gt_r = 0.56134
outer_gt_r = 0.60198

# Pyrex cell radii
air_r = 0.214
inner_pc_r = 0.23051
inner_gap_r = 0.24130
poison_r = 0.42672
outer_gap_r = 0.43688
outer_pc_r = 0.48387

# Baffle geometry
baffle_gap = 0.1627
baffle_width = 2.2225


def run_assembly(name, cells) -> scr.PWRAssembly:
    scb.scarabee_log(scb.LogLevel.Info, "")
    scb.scarabee_log(scb.LogLevel.Info, "")
    scb.scarabee_log(scb.LogLevel.Info, f"Running Assembly {name}")

    # Define assembly
    asmbly = scr.PWRAssembly(
        pitch=pin_pitch,
        assembly_pitch=asmbly_pitch,
        shape=(17, 17),
        moderator={
            "boron-ppm": boron_ppm,
            "temperature": T,
            "pressure": mod_pressure,
            "legendre-order": L,
        },
        independent_quadrant=True,
        cells=cells,
        ndl=ndl,
    )
    asmbly.solve()
    return asmbly


def run_assembly_then_save_tile(name, cells) -> scc.CoreTile:
    asmbly = run_assembly(name, cells)
    assert isinstance(asmbly._asmbly_moc, scb.MOCDriver)

    ct = scc.QuadrantsTile.from_independent_quadrant(
        asmbly.diffusion_data[0], asmbly.form_factors[0]
    )
    pickle.dump(ct, open(f"{name}.pkl", "wb"))
    return ct


def load_or_run_assembly(name, cells) -> scc.CoreTile:
    fname = f"{name}.pkl"
    if os.path.isfile(fname):
        return pickle.loads(open(fname, "rb").read())
    else:
        return run_assembly_then_save_tile(name, cells)


# ==============================================================================
# DEFINE MATERIALS

CladComp = scb.MaterialComposition(scb.Fraction.Weight, name="Zircaloy 4")
CladComp.add_element("O", 0.00125)
CladComp.add_element("Cr", 0.0010)
CladComp.add_element("Fe", 0.0021)
CladComp.add_element("Zr", 0.98115)
CladComp.add_element("Sn", 0.0145)
Clad = scb.Material(CladComp, T, 6.55, scb.DensityUnits.g_cm3, ndl)
Clad.max_legendre_order = L

HeComp = scb.MaterialComposition(scb.Fraction.Atoms, name="He Gas")
HeComp.add_element("He", 1.0)
He = scb.Material(HeComp, T, 0.0015981, scb.DensityUnits.g_cm3, ndl)
He.max_legendre_order = L

SS304Comp = scb.MaterialComposition(scb.Fraction.Weight, name="SS304")
SS304Comp.add_element("Si", 0.0060)
SS304Comp.add_element("Cr", 0.1900)
SS304Comp.add_element("Mn", 0.0200)
SS304Comp.add_element("Fe", 0.6840)
SS304Comp.add_element("Ni", 0.1000)
SS304 = scb.Material(SS304Comp, T, 8.03, scb.DensityUnits.g_cm3, ndl)

BSiGlassComp = scb.MaterialComposition(scb.Fraction.Weight, name="Borosilicate Glass")
BSiGlassComp.add_element("O", 0.5481)
BSiGlassComp.add_element("Si", 0.3787)
BSiGlassComp.add_element("Al", 0.0344)
BSiGlassComp.add_nuclide("B10", 0.0071)
BSiGlassComp.add_nuclide("B11", 0.0317)
BSiGlass = scb.Material(BSiGlassComp, T, 2.26, scb.DensityUnits.g_cm3, ndl)

AirComp = scb.MaterialComposition(scb.Fraction.Atoms, name="Air")
AirComp.add_element("O", 0.2095)
AirComp.add_element("N", 0.7809)
AirComp.add_element("Ar", 0.00933)
AirComp.add_element("C", 0.00027)
Air = scb.Material(AirComp, T, 0.000616, scb.DensityUnits.g_cm3, ndl)

Fuel16Comp = scb.MaterialComposition(scb.Fraction.Atoms, name="Fuel 1.6%")
Fuel16Comp.add_leu(1.6, 1.0)
Fuel16Comp.add_element("O", 2.0)
Fuel16 = scb.Material(Fuel16Comp, T, 10.31341, scb.DensityUnits.g_cm3, ndl)
Fuel16.max_legendre_order = L

Fuel24Comp = scb.MaterialComposition(scb.Fraction.Atoms, name="Fuel 2.4%")
Fuel24Comp.add_leu(2.4, 1.0)
Fuel24Comp.add_element("O", 2.0)
Fuel24 = scb.Material(Fuel24Comp, T, 10.29748, scb.DensityUnits.g_cm3, ndl)
Fuel24.max_legendre_order = L

Fuel31Comp = scb.MaterialComposition(scb.Fraction.Atoms, name="Fuel 3.1%")
Fuel31Comp.add_leu(3.1, 1.0)
Fuel31Comp.add_element("O", 2.0)
Fuel31 = scb.Material(Fuel31Comp, T, 10.30166, scb.DensityUnits.g_cm3, ndl)
Fuel31.max_legendre_order = L

# ==============================================================================
# DEFINE CELLS

# Define all fuel pins
make_fp = lambda fuel: scr.FuelPin(
    fuel=fuel,
    fuel_radius=fuel_r,
    gap=He,
    gap_radius=gap_r,
    clad=Clad,
    clad_radius=clad_r,
)

fp1 = make_fp(Fuel16)
fp2 = make_fp(Fuel24)
fp3 = make_fp(Fuel31)

# Define a guide tube
gt_ = scr.GuideTube(inner_radius=inner_gt_r, outer_radius=outer_gt_r, clad=Clad)

# Define guide tube with a burnable poison rod
bpr = scr.BurnablePoisonRod(
    center=Air,
    clad=SS304,
    gap=He,
    poison=BSiGlass,
    center_radius=air_r,
    inner_clad_radius=inner_pc_r,
    inner_gap_radius=inner_gap_r,
    poison_radius=poison_r,
    outer_gap_radius=outer_gap_r,
    outer_clad_radius=outer_pc_r,
)

bpp = scr.GuideTube(
    inner_radius=inner_gt_r, outer_radius=outer_gt_r, clad=Clad, fill=bpr
)

# ==============================================================================
# DEFINE ASSEMBLIES

# fmt: off

a16_0 = [[fp1, fp1, fp1, fp1, fp1, fp1, fp1, fp1, fp1],
         [fp1, fp1, fp1, fp1, fp1, fp1, fp1, fp1, fp1],
         [gt_, fp1, fp1, gt_, fp1, fp1, fp1, fp1, fp1],
         [fp1, fp1, fp1, fp1, fp1, gt_, fp1, fp1, fp1],
         [fp1, fp1, fp1, fp1, fp1, fp1, fp1, fp1, fp1],
         [gt_, fp1, fp1, gt_, fp1, fp1, gt_, fp1, fp1],
         [fp1, fp1, fp1, fp1, fp1, fp1, fp1, fp1, fp1],
         [fp1, fp1, fp1, fp1, fp1, fp1, fp1, fp1, fp1],
         [gt_, fp1, fp1, gt_, fp1, fp1, gt_, fp1, fp1]]

a24_0 = [[fp2, fp2, fp2, fp2, fp2, fp2, fp2, fp2, fp2],
         [fp2, fp2, fp2, fp2, fp2, fp2, fp2, fp2, fp2],
         [gt_, fp2, fp2, gt_, fp2, fp2, fp2, fp2, fp2],
         [fp2, fp2, fp2, fp2, fp2, gt_, fp2, fp2, fp2],
         [fp2, fp2, fp2, fp2, fp2, fp2, fp2, fp2, fp2],
         [gt_, fp2, fp2, gt_, fp2, fp2, gt_, fp2, fp2],
         [fp2, fp2, fp2, fp2, fp2, fp2, fp2, fp2, fp2],
         [fp2, fp2, fp2, fp2, fp2, fp2, fp2, fp2, fp2],
         [gt_, fp2, fp2, gt_, fp2, fp2, gt_, fp2, fp2]]

a24_12 = [[fp2, fp2, fp2, fp2, fp2, fp2, fp2, fp2, fp2],
          [fp2, fp2, fp2, fp2, fp2, fp2, fp2, fp2, fp2],
          [gt_, fp2, fp2, bpp, fp2, fp2, fp2, fp2, fp2],
          [fp2, fp2, fp2, fp2, fp2, bpp, fp2, fp2, fp2],
          [fp2, fp2, fp2, fp2, fp2, fp2, fp2, fp2, fp2],
          [gt_, fp2, fp2, gt_, fp2, fp2, bpp, fp2, fp2],
          [fp2, fp2, fp2, fp2, fp2, fp2, fp2, fp2, fp2],
          [fp2, fp2, fp2, fp2, fp2, fp2, fp2, fp2, fp2],
          [gt_, fp2, fp2, gt_, fp2, fp2, gt_, fp2, fp2]]

a24_16 = [[fp2, fp2, fp2, fp2, fp2, fp2, fp2, fp2, fp2],
          [fp2, fp2, fp2, fp2, fp2, fp2, fp2, fp2, fp2],
          [bpp, fp2, fp2, bpp, fp2, fp2, fp2, fp2, fp2],
          [fp2, fp2, fp2, fp2, fp2, bpp, fp2, fp2, fp2],
          [fp2, fp2, fp2, fp2, fp2, fp2, fp2, fp2, fp2],
          [gt_, fp2, fp2, gt_, fp2, fp2, bpp, fp2, fp2],
          [fp2, fp2, fp2, fp2, fp2, fp2, fp2, fp2, fp2],
          [fp2, fp2, fp2, fp2, fp2, fp2, fp2, fp2, fp2],
          [gt_, fp2, fp2, gt_, fp2, fp2, bpp, fp2, fp2]]

a31_0 = [[fp3, fp3, fp3, fp3, fp3, fp3, fp3, fp3, fp3],
         [fp3, fp3, fp3, fp3, fp3, fp3, fp3, fp3, fp3],
         [gt_, fp3, fp3, gt_, fp3, fp3, fp3, fp3, fp3],
         [fp3, fp3, fp3, fp3, fp3, gt_, fp3, fp3, fp3],
         [fp3, fp3, fp3, fp3, fp3, fp3, fp3, fp3, fp3],
         [gt_, fp3, fp3, gt_, fp3, fp3, gt_, fp3, fp3],
         [fp3, fp3, fp3, fp3, fp3, fp3, fp3, fp3, fp3],
         [fp3, fp3, fp3, fp3, fp3, fp3, fp3, fp3, fp3],
         [gt_, fp3, fp3, gt_, fp3, fp3, gt_, fp3, fp3]]

a31_15_I = [[fp3, fp3, fp3, fp3, fp3, fp3, fp3, fp3, fp3],
            [fp3, fp3, fp3, fp3, fp3, fp3, fp3, fp3, fp3],
            [bpp, fp3, fp3, bpp, fp3, fp3, fp3, fp3, fp3],
            [fp3, fp3, fp3, fp3, fp3, gt_, fp3, fp3, fp3],
            [fp3, fp3, fp3, fp3, fp3, fp3, fp3, fp3, fp3],
            [bpp, fp3, fp3, bpp, fp3, fp3, gt_, fp3, fp3],
            [fp3, fp3, fp3, fp3, fp3, fp3, fp3, fp3, fp3],
            [fp3, fp3, fp3, fp3, fp3, fp3, fp3, fp3, fp3],
            [gt_, fp3, fp3, bpp, fp3, fp3, gt_, fp3, fp3]]

a31_15_II = [[fp3, fp3, fp3, fp3, fp3, fp3, fp3, fp3, fp3],
             [fp3, fp3, fp3, fp3, fp3, fp3, fp3, fp3, fp3],
             [bpp, fp3, fp3, bpp, fp3, fp3, fp3, fp3, fp3],
             [fp3, fp3, fp3, fp3, fp3, bpp, fp3, fp3, fp3],
             [fp3, fp3, fp3, fp3, fp3, fp3, fp3, fp3, fp3],
             [bpp, fp3, fp3, bpp, fp3, fp3, bpp, fp3, fp3],
             [fp3, fp3, fp3, fp3, fp3, fp3, fp3, fp3, fp3],
             [fp3, fp3, fp3, fp3, fp3, fp3, fp3, fp3, fp3],
             [gt_, fp3, fp3, bpp, fp3, fp3, bpp, fp3, fp3]]

a31_15_IV = [[fp3, fp3, fp3, fp3, fp3, fp3, fp3, fp3, fp3],
             [fp3, fp3, fp3, fp3, fp3, fp3, fp3, fp3, fp3],
             [gt_, fp3, fp3, gt_, fp3, fp3, fp3, fp3, fp3],
             [fp3, fp3, fp3, fp3, fp3, gt_, fp3, fp3, fp3],
             [fp3, fp3, fp3, fp3, fp3, fp3, fp3, fp3, fp3],
             [bpp, fp3, fp3, bpp, fp3, fp3, gt_, fp3, fp3],
             [fp3, fp3, fp3, fp3, fp3, fp3, fp3, fp3, fp3],
             [fp3, fp3, fp3, fp3, fp3, fp3, fp3, fp3, fp3],
             [gt_, fp3, fp3, bpp, fp3, fp3, gt_, fp3, fp3]]

a31_16 = [[fp3, fp3, fp3, fp3, fp3, fp3, fp3, fp3, fp3],
          [fp3, fp3, fp3, fp3, fp3, fp3, fp3, fp3, fp3],
          [bpp, fp3, fp3, bpp, fp3, fp3, fp3, fp3, fp3],
          [fp3, fp3, fp3, fp3, fp3, bpp, fp3, fp3, fp3],
          [fp3, fp3, fp3, fp3, fp3, fp3, fp3, fp3, fp3],
          [gt_, fp3, fp3, gt_, fp3, fp3, bpp, fp3, fp3],
          [fp3, fp3, fp3, fp3, fp3, fp3, fp3, fp3, fp3],
          [fp3, fp3, fp3, fp3, fp3, fp3, fp3, fp3, fp3],
          [gt_, fp3, fp3, gt_, fp3, fp3, bpp, fp3, fp3]]

a31_20 = [[fp3, fp3, fp3, fp3, fp3, fp3, fp3, fp3, fp3],
          [fp3, fp3, fp3, fp3, fp3, fp3, fp3, fp3, fp3],
          [bpp, fp3, fp3, bpp, fp3, fp3, fp3, fp3, fp3],
          [fp3, fp3, fp3, fp3, fp3, bpp, fp3, fp3, fp3],
          [fp3, fp3, fp3, fp3, fp3, fp3, fp3, fp3, fp3],
          [gt_, fp3, fp3, bpp, fp3, fp3, bpp, fp3, fp3],
          [fp3, fp3, fp3, fp3, fp3, fp3, fp3, fp3, fp3],
          [fp3, fp3, fp3, fp3, fp3, fp3, fp3, fp3, fp3],
          [gt_, fp3, fp3, gt_, fp3, fp3, bpp, fp3, fp3]]

a31_6_I = [[fp3, fp3, fp3, fp3, fp3, fp3, fp3, fp3, fp3],
           [fp3, fp3, fp3, fp3, fp3, fp3, fp3, fp3, fp3],
           [gt_, fp3, fp3, bpp, fp3, fp3, fp3, fp3, fp3],
           [fp3, fp3, fp3, fp3, fp3, bpp, fp3, fp3, fp3],
           [fp3, fp3, fp3, fp3, fp3, fp3, fp3, fp3, fp3],
           [gt_, fp3, fp3, gt_, fp3, fp3, bpp, fp3, fp3],
           [fp3, fp3, fp3, fp3, fp3, fp3, fp3, fp3, fp3],
           [fp3, fp3, fp3, fp3, fp3, fp3, fp3, fp3, fp3],
           [gt_, fp3, fp3, gt_, fp3, fp3, gt_, fp3, fp3]]

# Same as a31_0 !
a31_6_II = [[fp3, fp3, fp3, fp3, fp3, fp3, fp3, fp3, fp3],
            [fp3, fp3, fp3, fp3, fp3, fp3, fp3, fp3, fp3],
            [gt_, fp3, fp3, gt_, fp3, fp3, fp3, fp3, fp3],
            [fp3, fp3, fp3, fp3, fp3, gt_, fp3, fp3, fp3],
            [fp3, fp3, fp3, fp3, fp3, fp3, fp3, fp3, fp3],
            [gt_, fp3, fp3, gt_, fp3, fp3, gt_, fp3, fp3],
            [fp3, fp3, fp3, fp3, fp3, fp3, fp3, fp3, fp3],
            [fp3, fp3, fp3, fp3, fp3, fp3, fp3, fp3, fp3],
            [gt_, fp3, fp3, gt_, fp3, fp3, gt_, fp3, fp3]]

# fmt: on

# ==============================================================================
# RUN ASSEMBLIES

# Make the basic assemblies
a1_00_ = load_or_run_assembly("F16_0", a16_0)
a2_00_ = load_or_run_assembly("F24_0", a24_0)
a2_12_ = load_or_run_assembly("F24_12", a24_12)
a2_16_ = load_or_run_assembly("F24_16", a24_16)
a3_16_ = load_or_run_assembly("F31_16", a31_16)
a3_20_ = load_or_run_assembly("F31_20", a31_20)

# Make the 15 poison pin assembly
if os.path.isfile("F31_15II.pkl"):
    a3_152 = pickle.loads(open("F31_15II.pkl", "rb").read())
else:
    quad_I = run_assembly("F31_15_I", a31_15_I)
    q1_dd = quad_I.diffusion_data[0]
    q1_ff = quad_I.form_factors[0]

    q3_dd = copy.deepcopy(q1_dd)
    q3_ff = copy.deepcopy(q1_ff)
    q3_dd.rotate_clockwise().reflect_across_y_axis()
    q3_ff.rotate_clockwise().reflect_across_y_axis()

    quad_II = run_assembly("F31_15_II", a31_15_II)
    q2_dd = copy.deepcopy(quad_II.diffusion_data[0])
    q2_ff = copy.deepcopy(quad_II.form_factors[0])
    q2_dd.reflect_across_y_axis()
    q2_ff.reflect_across_y_axis()

    quad_IV = run_assembly("F31_15_IV", a31_15_IV)
    q4_dd = copy.deepcopy(quad_IV.diffusion_data[0])
    q4_ff = copy.deepcopy(quad_IV.form_factors[0])
    q4_dd.reflect_across_x_axis()
    q4_ff.reflect_across_x_axis()

    ff = scb.FormFactors(q1_ff, q2_ff, q3_ff, q4_ff)
    a3_152 = scc.QuadrantsTile(q1_dd, q2_dd, q3_dd, q4_dd, ff)
    pickle.dump(a3_152, open("F31_15II.pkl", "wb"))

a3_151 = copy.deepcopy(a3_152)
a3_151.rotate_clockwise()

a3_154 = copy.deepcopy(a3_152)
a3_154.rotate_clockwise().rotate_clockwise()

a3_153 = copy.deepcopy(a3_152)
a3_153.rotate_clockwise().rotate_clockwise().rotate_clockwise()

# Make the 6 poison pin assembly
if os.path.isfile("F31_6R.pkl"):
    a3_06R = pickle.loads(open("F31_6R.pkl", "rb").read())
else:
    quad_I = run_assembly("F31_6R_I", a31_6_I)
    q1_dd = quad_I.diffusion_data[0]
    q1_ff = quad_I.form_factors[0]

    q4_dd = copy.deepcopy(q1_dd)
    q4_ff = copy.deepcopy(q1_ff)
    q4_dd.reflect_across_x_axis()
    q4_ff.reflect_across_x_axis()

    quad_II = run_assembly("F31_6R_II", a31_6_II)
    q2_dd = copy.deepcopy(quad_II.diffusion_data[0])
    q2_ff = copy.deepcopy(quad_II.form_factors[0])
    q2_dd.rotate_counterclockwise()
    q2_ff.rotate_counterclockwise()

    q3_dd = copy.deepcopy(q2_dd)
    q3_ff = copy.deepcopy(q2_ff)
    q3_dd.reflect_across_x_axis()
    q3_ff.reflect_across_x_axis()

    ff = scb.FormFactors(q1_ff, q2_ff, q3_ff, q4_ff)
    a3_06R = scc.QuadrantsTile(q1_dd, q2_dd, q3_dd, q4_dd, ff)
    pickle.dump(a3_06R, open(f"F31_6R.pkl", "wb"))

a3_06U = copy.deepcopy(a3_06R)
a3_06U.rotate_counterclockwise()

a3_06D = copy.deepcopy(a3_06R)
a3_06D.rotate_clockwise()

a3_06L = copy.deepcopy(a3_06R)
a3_06L.rotate_clockwise().rotate_clockwise()

# ------------------------------------------------------------------------------
# REFLECTOR CALCULATION

if os.path.isfile("F31_0.pkl") and os.path.isfile("reflector.pkl"):
    a3_00_ = pickle.loads(open("F31_0.pkl", "rb").read())
    rf____ = pickle.loads(open("reflector.pkl", "rb").read())
else:
    asmbly = run_assembly("F31_0", a31_0)

    a3_00_ = scc.QuadrantsTile.from_independent_quadrant(
        asmbly.diffusion_data[0], asmbly.form_factors[0]
    )

    scb.scarabee_log(scb.LogLevel.Info, "")
    scb.scarabee_log(scb.LogLevel.Info, "")
    refl = scr.Reflector(
        asmbly.moc.homogenize(),
        moderator=asmbly.moderator_xs,
        assembly_width=asmbly_pitch,
        gap_width=baffle_gap,
        baffle_width=baffle_width,
        baffle=SS304,
        ndl=ndl,
    )
    refl.anisotropic = True
    refl.solve()

    rf____ = scc.SimpleTile(refl.diffusion_data, refl.form_factors)
    pickle.dump(rf____, open("reflector.pkl", "wb"))

scb.scarabee_log(scb.LogLevel.Info, "")
scb.scarabee_log(scb.LogLevel.Info, 80 * "=")
scb.scarabee_log(scb.LogLevel.Info, "")

# ==============================================================================
# CORE CALCULATION

scb.scarabee_log(scb.LogLevel.Info, "Core Calculation")
scb.scarabee_log(scb.LogLevel.Info, "")

# fmt: off

# Define core geometry
#                R       P       N       M       L       K       J       H       G       F       E       D       C       B       A
tiles = [[[0.    , 0.    , 0.    , 0.    , rf____, rf____, rf____, rf____, rf____, rf____, rf____, rf____, rf____, 0.    , 0.    , 0.    , 0.],

          [0.    , 0.    , rf____, rf____, rf____, a3_00_, a3_06D, a3_00_, a3_06D, a3_00_, a3_06D, a3_00_, rf____, rf____, rf____, 0.    , 0.],     #  1 

          [0.    , rf____, rf____, a3_00_, a3_00_, a3_16_, a1_00_, a3_20_, a1_00_, a3_20_, a1_00_, a3_16_, a3_00_, a3_00_, rf____, rf____, 0.],     #  2 

          [0.    , rf____, a3_00_, a3_154, a2_16_, a1_00_, a2_16_, a1_00_, a2_16_, a1_00_, a2_16_, a1_00_, a2_16_, a3_153, a3_00_, rf____, 0.],     #  3

          [rf____, rf____, a3_00_, a2_16_, a2_00_, a2_16_, a1_00_, a2_12_, a1_00_, a2_12_, a1_00_, a2_16_, a2_00_, a2_16_, a3_00_, rf____, rf____], #  4

          [rf____, a3_00_, a3_16_, a1_00_, a2_16_, a1_00_, a2_12_, a1_00_, a2_12_, a1_00_, a2_12_, a1_00_, a2_16_, a1_00_, a3_16_, a3_00_, rf____], #  5

          [rf____, a3_06R, a1_00_, a2_16_, a1_00_, a2_12_, a1_00_, a2_12_, a1_00_, a2_12_, a1_00_, a2_12_, a1_00_, a2_16_, a1_00_, a3_06L, rf____], #  6

          [rf____, a3_00_, a3_20_, a1_00_, a2_12_, a1_00_, a2_12_, a1_00_, a2_16_, a1_00_, a2_12_, a1_00_, a2_12_, a1_00_, a3_20_, a3_00_, rf____], #  7

          [rf____, a3_06R, a1_00_, a2_16_, a1_00_, a2_12_, a1_00_, a2_16_, a1_00_, a2_16_, a1_00_, a2_12_, a1_00_, a2_16_, a1_00_, a3_06L, rf____], #  8

          [rf____, a3_00_, a3_20_, a1_00_, a2_12_, a1_00_, a2_12_, a1_00_, a2_16_, a1_00_, a2_12_, a1_00_, a2_12_, a1_00_, a3_20_, a3_00_, rf____], #  9

          [rf____, a3_06R, a1_00_, a2_16_, a1_00_, a2_12_, a1_00_, a2_12_, a1_00_, a2_12_, a1_00_, a2_12_, a1_00_, a2_16_, a1_00_, a3_06L, rf____], # 10

          [rf____, a3_00_, a3_16_, a1_00_, a2_16_, a1_00_, a2_12_, a1_00_, a2_12_, a1_00_, a2_12_, a1_00_, a2_16_, a1_00_, a3_16_, a3_00_, rf____], # 11

          [rf____, rf____, a3_00_, a2_16_, a2_00_, a2_16_, a1_00_, a2_12_, a1_00_, a2_12_, a1_00_, a2_16_, a2_00_, a2_16_, a3_00_, rf____, rf____], # 12

          [0.    , rf____, a3_00_, a3_151, a2_16_, a1_00_, a2_16_, a1_00_, a2_16_, a1_00_, a2_16_, a1_00_, a2_16_, a3_152, a3_00_, rf____, 0.],     # 13

          [0.    , rf____, rf____, a3_00_, a3_00_, a3_16_, a1_00_, a3_20_, a1_00_, a3_20_, a1_00_, a3_16_, a3_00_, a3_00_, rf____, rf____, 0.],     # 14

          [0.    , 0.    , rf____, rf____, rf____, a3_00_, a3_06U, a3_00_, a3_06U, a3_00_, a3_06U, a3_00_, rf____, rf____, rf____, 0.    , 0.],     # 15

          [0.    , 0.    , 0.    , 0.    , rf____, rf____, rf____, rf____, rf____, rf____, rf____, rf____, rf____, 0.    , 0.    , 0.    , 0.],
        ]]

# fmt: on

tiles = np.array(tiles)
dz = np.array([20.0])

core_builder = scc.CoreBuilder(21.50364, 17, 1.25984, 17, tiles, dz, 1.0, 1.0)
core_builder.solver.solve()

# -----------------------------------------------------------------------------
# Rasterize the flux and homogeneous power

# x, y, z arrays for rasterizing the flux / power
x_max = np.sum(core_builder.dx)
dx = x_max / 500.0
x = np.arange(start=0.0, stop=x_max + dx, step=dx)
y_max = np.sum(core_builder.dy)
dy = y_max / 500.0
y = np.arange(start=0.0, stop=y_max + dy, step=dy)
z = np.array([0.0])

power = core_builder.solver.power(x, y, z)[:, :, 0]
asmbly_powers = core_builder.compute_assembly_powers()
pin_power = core_builder.compute_pin_powers(np.array([10.0]))

scb.scarabee_log(scb.LogLevel.Info, "")
scb.scarabee_log(scb.LogLevel.Info, f"Max Pin Power: {np.nanmax(pin_power):.5f}")
scb.scarabee_log(
    scb.LogLevel.Info,
    f"Min Pin Power: {np.nanmin(pin_power[np.where(pin_power != 0.)]):.5f}",
)

pin_power[np.where(pin_power == 0.0)] = np.nan

# -----------------------------------------------------------------------------
# Plots

# Plot the flux in each group
flux = core_builder.solver.flux(x, y, z)[:, :, :, 0]
for g in range(flux.shape[0]):
    plt.pcolormesh(y, x, flux[g, :, :], cmap="turbo")
    plt.title(f"Group {g+1} Flux")
    plt.show()

# Plot the homogeneous power
plt.pcolormesh(y, x, power, cmap="turbo")
plt.title("Homogeneous Power Distribution")
plt.show()

# Plot assembly powers
plt.imshow(asmbly_powers, cmap="turbo")
plt.title("Assembly Powers")
plt.show()

# Plot pin powers
plt.pcolormesh(
    core_builder.x_pin_centers,
    core_builder.y_pin_centers,
    pin_power[:, :, 0],
    cmap="turbo",
)
plt.title("Pin Power Distribution")
plt.show()

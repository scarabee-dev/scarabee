from scarabee import FluxCalculator, scarabee_log, LogLevel
import pyPapillonNDL as pndl
import numpy as np


class DilutionTable:
    def __init__(self, dilutions, xs):
        self.dilutions = dilutions
        self.xs = xs  # First index dilution, second is group from high to low energy

        # Ensure that the dilutions are sorted
        if np.all(self.dilutions[:-1] >= self.dilutions[1:]):
            raise RuntimeError("Dilutions are not sorted")

        if self.xs.shape[0] != self.dilutions.shape[0]:
            raise RuntimeError("Number of dilutions and shape of xs array disagree.")

    @property
    def ngroups(self):
        return self.xs.shape[1]

    def group_is_resonant(self, g):
        xs_mid = 0.5 * (
            self.xs[0, g] + self.xs[-1, g]
        )  # Should be the middle value of the xs
        std = np.abs(
            self.xs[0, g] - self.xs[-1, g]
        )  # Should be the absolute difference between min and max dilution

        rel_err = std / xs_mid

        if rel_err <= 0.02:  # 2% threshold for being resonant
            return False

        return True

    def interpolate_xs(self, g, sig_0):
        if g < 0 or g >= self.ngroups:
            raise IndexError("Energy group index out of range.")

        # First, find the LOWER dilution index
        if sig_0 <= self.dilutions[0]:
            return self.xs[0, g]
        elif sig_0 >= self.dilutions[-1]:
            return self.xs[-1, g]

        id = np.searchsorted(self.dilutions, sig_0, side="right") - 1

        sig_0_low = self.dilutions[id]
        sig_0_hi = self.dilutions[id + 1]

        xs_g_low = self.xs[id, g]
        xs_g_hi = self.xs[id + 1, g]

        return ((xs_g_hi - xs_g_low) / (sig_0_hi - sig_0_low)) * (
            sig_0 - sig_0_low
        ) + xs_g_low

    def find_dilution(self, g, xs):
        if g < 0 or g >= self.ngroups:
            raise IndexError("Energy group index is out of range.")

        # If XS is increasing with dilution
        if self.xs[0, g] <= self.xs[-1, g]:
            if xs < self.xs[0, g]:
                raise RuntimeError(
                    "Provided xs is smaller than smallest tabulated dilution."
                )
            elif xs > self.xs[-1, g]:
                raise RuntimeError(
                    "Provided xs is larger than largest tabulated dilution."
                )

            id = np.searchsorted(self.xs[:, g], xs) - 1

        else:
            # XS decreases with dilution
            if xs > self.xs[0, g]:
                raise RuntimeError(
                    "Provided xs is larger than largest tabulated dilution."
                )
            elif xs < self.xs[-1, g]:
                raise RuntimeError(
                    "Provided xs is smaller than smallest tabulated dilution."
                )

            id = np.searchsorted(self.xs[:, g], xs) - 1
            for id in range(self.xs.shape[0] - 1):
                if self.xs[id, g] > xs and xs > self.xs[id + 1, g]:
                    break

        sig_0_low = self.dilutions[id]
        sig_0_hi = self.dilutions[id + 1]

        xs_g_low = self.xs[id, g]
        xs_g_hi = self.xs[id + 1, g]

        return ((sig_0_hi - sig_0_low) / (xs_g_hi - xs_g_low)) * (
            xs - xs_g_low
        ) + sig_0_low


class IRLambdaCalculator:
    # Values taken from ENDF/B-8.0
    AWR_H1 = 0.9991673
    SIG_POT_H1 = 20.436089874720498

    def __init__(
        self,
        Nr,
        awr,
        sig_pot,
        reference_dilution,
        mg_energy_boundaries,
        dilution_table=None,
    ):
        self.Nr = Nr
        self.mg_energy_boundaries = mg_energy_boundaries
        self.dilution_table = dilution_table
        self.reference_dilution = reference_dilution
        self.ir_lambda = None

        # Parameters of nuclide for which we want to generate IR parameters
        self.awr = awr
        self.sig_pot = sig_pot

        # Get parameters for the slowing down calculations
        self.energy_bounds = np.geomspace(1.0e-5, 20.0e6, 500000)
        self.sig_t = np.zeros(self.energy_bounds.size - 1)
        self.sig_s = np.zeros(self.energy_bounds.size - 1)
        for g in range(self.sig_t.size):
            E = 0.5 * (self.energy_bounds[g + 1] + self.energy_bounds[g]) * 1.0e-6
            self.sig_t[g] = self.Nr.total_xs()(E)
            self.sig_s[g] = self.Nr.elastic_xs()(E)

        if self.dilution_table is None:
            self.generate_dilution_table()

    def compute_H1_capture_xs(self, sig_d):
        # Do slowing down calculation
        calc = FluxCalculator(self.energy_bounds, self.sig_t, self.sig_s, self.Nr.awr())
        calc.add_background_nuclide(sig_d, self.AWR_H1)
        calc.solve()

        xs_102 = self.Nr.reaction(102).xs()

        # Setup functions and arrays
        flux = pndl.Tabulated1D(pndl.Interpolation.LinLin, calc.avg_energy, calc.flux)
        sig_mt102_flx = np.array(xs_102.xs())
        enrgy = np.array(xs_102.energy()) * 1.0e6  # Convert energies from MeV to eV
        for i in range(len(enrgy)):
            sig_mt102_flx[i] *= flux(enrgy[i])
        xs_flx = pndl.Tabulated1D(pndl.Interpolation.LinLin, enrgy, sig_mt102_flx)

        # Do integrations for the xs
        mg_sig = np.zeros(self.mg_energy_boundaries.size - 1)
        for g in range(mg_sig.size):
            Ehi = self.mg_energy_boundaries[g]
            Elow = self.mg_energy_boundaries[g + 1]
            mg_sig[g] = xs_flx.integrate(Elow, Ehi) / flux.integrate(Elow, Ehi)

        return mg_sig

    def generate_dilution_table(self):
        scarabee_log(LogLevel.Info, "Generating resonant nuclide dilution table...")
        dilutions = np.geomspace(0.1, 10000.0, 100)
        xs = np.zeros((dilutions.size, self.mg_energy_boundaries.size - 1))

        for i in range(dilutions.size):
            xs[i, :] = self.compute_H1_capture_xs(dilutions[i])

        self.dilution_table = DilutionTable(dilutions, xs)
        scarabee_log(LogLevel.Info, "Resonant nuclide dilution table generated.")

    def solve(self):
        # This ratio is N_H1 / N_r
        R_H1_r = self.reference_dilution / self.SIG_POT_H1

        # We now reduce the ratio of H1 and move it to the nuclide in question
        R_H1 = 0.95 * R_H1_r
        R_nuc = 0.05 * R_H1_r

        # Do slowing down calculation
        calc = FluxCalculator(self.energy_bounds, self.sig_t, self.sig_s, self.Nr.awr())
        calc.add_background_nuclide(R_H1 * self.SIG_POT_H1, self.AWR_H1)
        calc.add_background_nuclide(R_nuc * self.sig_pot, self.awr)
        calc.solve()

        # From flux, compute the MGXS
        xs_102 = self.Nr.reaction(102).xs()

        # Setup functions and arrays
        flux = pndl.Tabulated1D(pndl.Interpolation.LinLin, calc.avg_energy, calc.flux)
        sig_mt102_flx = np.array(xs_102.xs())
        enrgy = np.array(xs_102.energy()) * 1.0e6  # Convert energies from MeV to eV
        for i in range(len(enrgy)):
            sig_mt102_flx[i] *= flux(enrgy[i])
        xs_flx = pndl.Tabulated1D(pndl.Interpolation.LinLin, enrgy, sig_mt102_flx)

        # Do integrations for the xs
        mg_sig = np.zeros(self.mg_energy_boundaries.size - 1)
        for g in range(mg_sig.size):
            Ehi = self.mg_energy_boundaries[g]
            Elow = self.mg_energy_boundaries[g + 1]
            mg_sig[g] = xs_flx.integrate(Elow, Ehi) / flux.integrate(Elow, Ehi)

        # Compute the IR parameters for each group !
        self.ir_lambda = np.zeros(mg_sig.size)
        for g in range(self.ir_lambda.size):
            if (
                self.dilution_table.group_is_resonant(g)
                and self.mg_energy_boundaries[g + 1] > 5.0
            ):  # Only resonant groups and where energy greater than 5 eV
                try:
                    # Get the equivalent background xs
                    sig_0_equiv = self.dilution_table.find_dilution(g, mg_sig[g])
                    self.ir_lambda[g] = (sig_0_equiv - R_H1 * self.SIG_POT_H1) / (
                        R_nuc * self.sig_pot
                    )

                    if self.ir_lambda[g] < 0.0 or 1.0 < self.ir_lambda[g]:
                        Ehi = self.mg_energy_boundaries[g]
                        Elow = self.mg_energy_boundaries[g + 1]
                        mssg = "Reseting IR parameter in group {:d} with energy interval [{:.5E}, {:.5E}] eV from {:.3f} to 1.".format(
                            g + 1, Elow, Ehi, self.ir_lambda[g]
                        )
                        scarabee_log(LogLevel.Warning, mssg)
                        self.ir_lambda[g] = 1.0
                except:
                    mssg = "Group {:d} sig_a: {:.5E}, lowest tabulated sig_a: {:.5E}, highest tabulated sig_a: {:.5E}".format(
                        g + 1,
                        mg_sig[g],
                        self.dilution_table.xs[0, g],
                        self.dilution_table.xs[-1, g],
                    )
                    scarabee_log(LogLevel.Warning, mssg)
                    scarabee_log(LogLevel.Warning, "    Setting IR parameter to 1.")
                    self.ir_lambda[g] = 1.0
            else:
                # Ehi = self.mg_energy_boundaries[g]
                # Elow = self.mg_energy_boundaries[g+1]
                # mssg = "Cannot obtain lambda for group {:d} with energy interval [{:.5E}, {:.5E}] eV. Setting IR parameter to 1.".format(g+1, Elow, Ehi)
                # scarabee_log(LogLevel.Info, mssg)
                self.ir_lambda[g] = 1.0


def generate_U238_U235_ir_lambda(
    U238: pndl.STNeutron,
    U235: pndl.STNeutron,
    awr: float,
    sig_pot: float,
    mg_energy_bounds: np.ndarray,
    dilution: float = 50.0,
):
    lmbda_U238 = IRLambdaCalculator(
        U238,
        awr=awr,
        sig_pot=sig_pot,
        reference_dilution=dilution,
        mg_energy_boundaries=mg_energy_bounds,
    )
    lmbda_U238.solve()

    lmbda_U235 = IRLambdaCalculator(
        U235,
        awr=awr,
        sig_pot=sig_pot,
        reference_dilution=dilution,
        mg_energy_boundaries=mg_energy_bounds,
    )
    lmbda_U235.solve()

    # Start with U238 IR-lambdas as a base
    out_ir_lambda = lmbda_U238.ir_lambda.copy()

    # Now go through and reset with the U235 if we are at 1.
    for g in range(out_ir_lambda.size):
        if out_ir_lambda[g] == 1.0:
            out_ir_lambda[g] = lmbda_U235.ir_lambda[g]

    return out_ir_lambda

import endf
from scarabee import *
from .group_structures import GROUP_STRUCTURES
from .ir_lambda import generate_U238_U235_ir_lambda, NuclideInfo
from .endf_tools import get_potential_scatter_xs, get_fission_energy_release_components
import subprocess
from subprocess import Popen, PIPE, STDOUT, CalledProcessError
import os
import numpy as np
from typing import Optional
from pathlib import Path
import tempfile
import shutil
import h5py
from pathlib import Path

# Global variables initialized for computing IR parameters
_this_src_file_path = Path(__file__).resolve().parents[0]
_U235_ace_path = _this_src_file_path / "U235_600.txt"
_U238_ace_path = _this_src_file_path / "U238_600.txt"

_U235_grids_path = _this_src_file_path / "U235_grids.npy"
_U238_grids_path = _this_src_file_path / "U238_grids.npy"

def _make_grid_nuclide_info(path: Path):
    ace = endf.ace.get_table(path)
    awr = ace.atomic_weight_ratio

    N = ace.interpret()

    tot = N.reactions[1]

    egrid = tot.xs['600K'].x
    tot_xs = tot.xs['600K'].y

    ela = N.reactions[2]
    ela_xs = ela.xs['600K'].y

    cap = N.reactions[102]
    cap_xs = cap.xs['600K'].y

    awr_grid = np.ones(cap_xs.shape) * awr

    return np.stack([egrid, tot_xs, ela_xs, cap_xs, awr_grid])

def _read_nuclide_info_grid(path: Path):
    grid = np.load(str(path))

    egrid = grid[0,:]
    tot_xs = Tab1(egrid, grid[1,:])
    ela_xs = Tab1(egrid, grid[2,:])
    cap_xs = Tab1(egrid, grid[3,:])
    awr = grid[4, 0]

    return NuclideInfo(awr, tot_xs, ela_xs, cap_xs)

if not _U235_grids_path.exists():
    orig_dir = os.getcwd()
    os.chdir(str(Path(__file__).resolve().parents[0]))
    if not _U235_ace_path.exists():
        subprocess.run(["frendy", "process_U235.txt"])
    U235_grid = _make_grid_nuclide_info(_U235_ace_path)
    np.save(_U235_grids_path, U235_grid)
    os.chdir(orig_dir)

if not _U238_grids_path.exists():
    orig_dir = os.getcwd()
    os.chdir(str(Path(__file__).resolve().parents[0]))
    if not _U238_ace_path.exists():
        subprocess.run(["frendy", "process_U238.txt"])
    U238_grid = _make_grid_nuclide_info(_U238_ace_path)
    np.save(_U238_grids_path, U238_grid)
    os.chdir(orig_dir)

_U235 = _read_nuclide_info_grid(_U235_grids_path)
_U238 = _read_nuclide_info_grid(_U238_grids_path)

_DEFAULT_GROUP_STRUCTURE = "SCARABEE-155"
_DEFAULT_MAX_LEGENDRE_MOMENT = 3

def set_default_group_structure(name):
    global _DEFAULT_GROUP_STRUCTURE
    if name not in GROUP_STRUCTURES:
        raise RuntimeError('Uknown group structure "{}".'.format(name))
    _DEFAULT_GROUP_STRUCTURE = name


def set_default_max_legendre_moments(l):
    global _DEFAULT_MAX_LEGENDRE_MOMENT
    if l >= 0 and l <= 3:
        _DEFAULT_MAX_LEGENDRE_MOMENT = l
    else:
        raise RuntimeError("Default max legendre moment must be in range [0, 3].")


def get_default_max_legendre_moments():
    return _DEFAULT_MAX_LEGENDRE_MOMENT


def get_default_group_structure():
    return GROUP_STRUCTURES[_DEFAULT_GROUP_STRUCTURE]


class KRAMXS:
    def __init__(self):
        self.Et = None
        self.Ea = None
        self.Es = None
        self.Es1 = None
        self.Es2 = None
        self.Es3 = None
        self.Ef = None
        self.nu = None
        self.chi = None

    @property
    def ngroups(self):
        if self.Et is None:
            return 0
        return len(self.Et)

    @staticmethod
    def __read_line(fl):
        line = fl.readline()
        line = line.strip().split()
        for i in range(len(line)):
            line[i] = float(line[i])
        line = np.array(line)
        return line

    @staticmethod
    def from_file(fname, max_l):
        fl = open(fname, "r")
        fl.readline()  # Skip the XSN 1 header

        # Read scattering matrix first
        Es = []
        ngroups = 100  # This is a guess to start
        line_num = 0
        while line_num < ngroups:
            line_num += 1
            Es.append(KRAMXS.__read_line(fl))
            ngroups = len(Es[-1])
        Es = np.array(Es)
        Es = np.copy(np.swapaxes(Es, 0, 1))

        # Read vEf
        vEf = KRAMXS.__read_line(fl)

        # Read Ea
        Ea = KRAMXS.__read_line(fl)

        # Read Et
        Et = KRAMXS.__read_line(fl)

        # Read Ef
        Ef = KRAMXS.__read_line(fl)

        # Skip FSP 1 line
        fl.readline()

        # Read chi
        chi = KRAMXS.__read_line(fl)

        # Skip ASC 1 line and 1 line
        fl.readline()
        fl.readline()

        # Read P1-scattering matrix
        Es1 = None
        if max_l >= 1:
            Es1 = []
            line_num = 0
            while line_num < ngroups:
                line_num += 1
                Es1.append(KRAMXS.__read_line(fl))
            Es1 = np.array(Es1)
            Es1 = np.copy(np.swapaxes(Es1, 0, 1))

        # Read P2-scattering matrix
        Es2 = None
        if max_l >= 2:
            Es2 = []
            line_num = 0
            while line_num < ngroups:
                line_num += 1
                Es2.append(KRAMXS.__read_line(fl))
            Es2 = np.array(Es2)
            Es2 = np.copy(np.swapaxes(Es2, 0, 1))

        # Read P3-scattering matrix
        Es3 = None
        if max_l >= 3:
            Es3 = []
            line_num = 0
            while line_num < ngroups:
                line_num += 1
                Es3.append(KRAMXS.__read_line(fl))
            Es3 = np.array(Es3)
            Es3 = np.copy(np.swapaxes(Es3, 0, 1))

        fl.close()

        # Create and return instance
        xs = KRAMXS()
        xs.Et = Et
        xs.Ea = Ea
        xs.Es = Es

        if Es1 is not None:
            xs.Es1 = Es1
        if Es2 is not None:
            xs.Es2 = Es2
        if Es3 is not None:
            xs.Es3 = Es3

        xs.Ef = Ef
        xs.nu = np.divide(vEf, Ef, out=np.zeros_like(vEf), where=Ef != 0.0)
        xs.chi = chi
        return xs


class UltraFineGroupTable:
    """Holds tabulated data from a FRENDY formated Ultra-Fine Group file.

    Attributes
    ----------
    energy : np.ndarray
        1D array with the energies in eV.
    dilutions : np.ndarray
        1D array with the dilutions in barns.
    data : np.ndarray
        2D array, indexed by dilution then energy.
    """

    def __init__(self, energy, dilutions, data):
        self.energy = energy
        self.dilutions = dilutions
        self.data = data

    @staticmethod
    def from_file(fname: str):
        """Reads an Ultra-Fine Group FRENDY formated table.

        Parameters
        ----------
        fname : str
            Name of file containing the table.

        Returns
        -------
        UltraFineGroupTable
        """
        with open(fname, "r") as fl:
            # Read the background cross sections
            line = fl.readline().split()[5:]
            dilutions = np.array([float(sig) for sig in line])

            # Skip next 2 line
            _ = fl.readline()
            _ = fl.readline()

            data = []
            energy = []

            # Read all groups
            for line in fl:
                line = fl.readline().split()
                energy.append(float(line[2]))
                data.append([float(val) for val in line[4:]])

            energy = np.flip(np.array(energy))
            data = np.flip(np.swapaxes(np.array(data), 0, 1), 1)
        return UltraFineGroupTable(energy, dilutions, data)


class FrendyMG:
    def __init__(self, group_structure: Optional[str] = None):
        self.temps = [293.6]
        self.dilutions: list[float] = []
        self.pot_xs: float = 0.
        self.endf_file: str | None = None
        self.tsl_file: str | None = None
        self.tsl_type: str | None = None
        self.label = ""
        self.name = ""
        if group_structure is not None:
            if group_structure not in GROUP_STRUCTURES:
                raise RuntimeError(
                    'Unknown group structure "{}".'.format(group_structure)
                )
            self.group_structure = GROUP_STRUCTURES[group_structure]
        else:
            self.group_structure = GROUP_STRUCTURES[_DEFAULT_GROUP_STRUCTURE]
        self.ngroups = self.group_structure.ngroups
        self.initialized = False
        self.processed = False
        self.resonant = False
        self.delete_files = True
        self.max_legendre_moment = _DEFAULT_MAX_LEGENDRE_MOMENT

        if self.max_legendre_moment > 3:
            raise RuntimeError("Only legendre moments up to L=3 are supported.")

        # Run settings
        self.verbose = True
        self.frendy_exe = 'frendy'
        self.run_dir: Path = Path(os.getcwd())

    def initialize(self):
        if self.dilutions is not None:
            self.dilutions.sort()
            if len(self.dilutions) == 0 or self.dilutions[-1] < 1.0e10:
                self.dilutions.append(1.0e10)

        self._get_endf_info()

        self._allocate_arrays()

        self.initialized = True

    def _allocate_arrays(self):
        if self.dilutions is None or (isinstance(self.dilutions, list) and len(self.dilutions) == 0):
            return

        if len(self.dilutions) > 1:
            self.resonant = True

        self.Dtr = np.zeros((len(self.temps), len(self.dilutions), self.ngroups))
        self.Ea = np.zeros((len(self.temps), len(self.dilutions), self.ngroups))
        self.Es = np.zeros(
            (len(self.temps), len(self.dilutions), self.ngroups, self.ngroups)
        )
        if self.fissile:
            self.Ef = np.zeros((len(self.temps), len(self.dilutions), self.ngroups))

            # Nu and chi are only very weakly dependent on temp and dilution.
            # Because of this, we don't tabulate them on temp or dilution.
            self.nu = np.zeros((self.ngroups))
            self.chi = np.zeros((self.ngroups))
        else:
            self.Ef = None
            self.nu = None
            self.chi = None

        if self.max_legendre_moment >= 1:
            self.Es1 = np.zeros(
                (len(self.temps), len(self.dilutions), self.ngroups, self.ngroups)
            )
        else:
            self.Es1 = None

        if self.max_legendre_moment >= 2:
            self.Es2 = np.zeros(
                (len(self.temps), len(self.dilutions), self.ngroups, self.ngroups)
            )
        else:
            self.Es2 = None

        if self.max_legendre_moment >= 3:
            self.Es3 = np.zeros(
                (len(self.temps), len(self.dilutions), self.ngroups, self.ngroups)
            )
        else:
            self.Es3 = None

        # Depletion related reactions
        self.Egamma = None
        self.En2n = None
        self.En3n = None
        self.Enp = None
        self.Ena = None

    def process(self, h5=None, chi=None):
        if not self.initialized:
            self.initialize()

        # Make sure we have all tsl info
        if (self.tsl_file is not None and self.tsl_type is None) or (
            self.tsl_file is None and self.tsl_type is not None
        ):
            raise RuntimeError("For TSL, must provide both tsl_file and tsl_type.")

        for i in range(len(self.temps)):
            self._process_temp(i, generate_ir_lambda= i == 0)

        self.processed = True

        # Truncate threshold reactions to remove zeros
        self._remove_zeros()

        if chi is not None:
            self.apply_inflow_transport_correction(chi)
        else:
            self.apply_outflow_transport_correction()

        # Apply compression after computing the transport correction !
        self._get_compressed_scatter_layout()
        self._compress_scatter_matrices()

        if h5 is not None:
            self.add_to_hdf5(h5)

    def apply_inflow_transport_correction(self, chi):
        if not self.processed:
            raise RuntimeError("Cannot apply transport corretion to unprocessed data.")

        for iT in range(len(self.temps)):
            for id in range(len(self.dilutions)):
                # Create a temporary xs set with the provided fission spectrum
                Et = self.Ea[iT, id, :] + np.sum(self.Es[iT, id, :, :], axis=1)
                Es = np.array([self.Es[iT, id, :, :], self.Es1[iT, id, :, :]])
                if self.fissile:
                    TempXS = CrossSection(
                        Et,
                        self.Dtr[iT, id, :],
                        self.Ea[iT, id, :],
                        Es,
                        self.Ef[iT, id, :],
                        self.nu[iT, id, :] * self.Ef[iT, id, :],
                        chi,
                    )
                else:
                    TempXS = CrossSection(
                        Et,
                        self.Dtr[iT, id, :],
                        self.Ea[iT, id, :],
                        Es,
                        np.zeros(self.ngroups),
                        np.zeros(self.ngroups),
                        chi,
                    )

                # We now perform a P1 leakage calculation
                P1_spectrum = P1CriticalitySpectrum(TempXS, 0.0001)

                # We now have diffusion coefficients
                D = P1_spectrum.diff_coeff

                # Compute transport xs
                Etr = 1.0 / (3.0 * D)

                # Calculate the delta xs for the transport correction
                self.Dtr[iT, id, :] = Et - Etr

    def apply_outflow_transport_correction(self):
        if not self.processed:
            raise RuntimeError("Cannot apply transport corretion to unprocessed data.")

        if self.Es1 is not None:
            for iT in range(len(self.temps)):
                for id in range(len(self.dilutions)):
                    # Calculate the delta xs for the transport correction
                    for g in range(self.ngroups):
                        self.Dtr[iT, id, g] = np.sum(self.Es1[iT, id, g, :])

    def _get_compressed_scatter_layout(self):
        self.low_grps = []
        self.high_grps = []
        self.data_starts = []

        for g in range(self.ngroups):
            # Find the first outgoing group which isn't 0
            g_low = 0
            for gg in range(self.ngroups):
                not_all_zeros = np.any(self.Es[:, :, g, : gg + 1])
                if not_all_zeros:
                    g_low = gg
                    break

            # Find the last outgoing group which isn't 0
            g_hi = 0
            for gg in range(self.ngroups):
                all_zeros = not np.any(self.Es[:, :, g, gg:])
                if all_zeros:
                    g_hi = gg - 1
                    break
            if g_hi == 0:
                g_hi = self.ngroups - 1

            self.low_grps.append(g_low)
            self.high_grps.append(g_hi)

            if g == 0:
                self.data_starts.append(0)
            else:
                self.data_starts.append(
                    self.data_starts[-1] + (self.high_grps[-2] - self.low_grps[-2]) + 1
                )

        self.len_scatter_matrix_data = (
            self.data_starts[-1] + (self.high_grps[-1] - self.low_grps[-1]) + 1
        )

    def _compress_scatter_matrices(self):
        Es = np.zeros(
            (len(self.temps), len(self.dilutions), self.len_scatter_matrix_data)
        )
        if self.Es1 is not None:
            Es1 = np.zeros(
                (len(self.temps), len(self.dilutions), self.len_scatter_matrix_data)
            )
        if self.Es2 is not None:
            Es2 = np.zeros(
                (len(self.temps), len(self.dilutions), self.len_scatter_matrix_data)
            )
        if self.Es3 is not None:
            Es3 = np.zeros(
                (len(self.temps), len(self.dilutions), self.len_scatter_matrix_data)
            )

        i = 0
        for g in range(self.ngroups):
            g_low = self.low_grps[g]
            g_hi = self.high_grps[g]
            l = g_hi - g_low + 1

            Es[:, :, i : i + l] = self.Es[:, :, g, g_low : g_hi + 1]

            if self.Es1 is not None:
                Es1[:, :, i : i + l] = self.Es1[:, :, g, g_low : g_hi + 1]
            if self.Es2 is not None:
                Es2[:, :, i : i + l] = self.Es2[:, :, g, g_low : g_hi + 1]
            if self.Es3 is not None:
                Es3[:, :, i : i + l] = self.Es3[:, :, g, g_low : g_hi + 1]

            i += l

        self.Es = Es
        if self.Es1 is not None:
            self.Es1 = Es1
        if self.Es2 is not None:
            self.Es2 = Es2
        if self.Es3 is not None:
            self.Es3 = Es3

    def _remove_zeros(self):
        self.Egamma = self._cull_array(self.Egamma)
        self.En2n = self._cull_array(self.En2n)
        self.En3n = self._cull_array(self.En3n)
        self.Enp = self._cull_array(self.Enp)
        self.Ena = self._cull_array(self.Ena)

    def _cull_array(self, a):
        if a is not None:
            gmax = self.ngroups - 1
            for g in range(self.ngroups - 1, -1, -1):
                # Check if group is all zeros
                not_all_zeros = np.any(a[:, :, g])
                if not_all_zeros:
                    gmax = g
                    break
            if gmax != self.ngroups - 1:
                return a[:, :, : gmax + 1]
            else:
                return a

    def add_to_hdf5(self, h5):
        grp = h5.create_group(self.name)

        # Save attributes
        grp.attrs["name"] = self.name
        grp.attrs["fissile"] = self.fissile
        grp.attrs["resonant"] = self.resonant
        grp.attrs["awr"] = self.awr
        grp.attrs["ZA"] = self.ZA
        grp.attrs["label"] = self.label
        grp.attrs["potential-xs"] = self.pot_xs
        grp.attrs["temperatures"] = self.temps
        grp.attrs["dilutions"] = self.dilutions
        if self.fission_energy is not None:
            grp.attrs["fission-energy"] = self.fission_energy
        if self.ir_lambda is not None:
            grp.attrs["ir-lambda"] = self.ir_lambda

        packing = np.zeros((self.ngroups, 3), dtype=np.uint32)
        for g in range(self.ngroups):
            packing[g, 0] = self.data_starts[g]
            packing[g, 1] = self.low_grps[g]
            packing[g, 2] = self.high_grps[g]
        grp.create_dataset("matrix-compression", data=packing)

        # Save the infinite diution cross section data
        grp.create_dataset("inf-transport-correction", data=self.Dtr[:, -1, :])
        grp.create_dataset("inf-absorption", data=self.Ea[:, -1, :])
        grp.create_dataset("inf-scatter", data=self.Es[:, -1, :])
        if self.Es1 is not None:
            grp.create_dataset("inf-p1-scatter", data=self.Es1[:, -1, :])
        if self.Es2 is not None:
            grp.create_dataset("inf-p2-scatter", data=self.Es2[:, -1, :])
        if self.Es3 is not None:
            grp.create_dataset("inf-p3-scatter", data=self.Es3[:, -1, :])
        if self.fissile:
            grp.create_dataset("inf-fission", data=self.Ef[:, -1, :])
            grp.create_dataset("nu", data=self.nu)
            grp.create_dataset("chi", data=self.chi)

        # Depletion data
        if self.Egamma is not None:
            grp.create_dataset("inf-(n,gamma)", data=self.Egamma[:, -1, :])
        if self.En2n is not None:
            grp.create_dataset("inf-(n,2n)", data=self.En2n[:, -1, :])
        if self.En3n is not None:
            grp.create_dataset("inf-(n,3n)", data=self.En3n[:, -1, :])
        if self.Enp is not None:
            grp.create_dataset("inf-(n,p)", data=self.Enp[:, -1, :])
        if self.Ena is not None:
            grp.create_dataset("inf-(n,a)", data=self.Ena[:, -1, :])

        if self.resonant:
            # Get indices for scatterinig matrices
            glow = self.group_structure.first_res_grp
            ghi = self.group_structure.last_res_grp
            ilow = self.data_starts[glow]
            ihi = self.data_starts[ghi + 1]

            grp.create_dataset(
                "res-transport-correction", data=self.Dtr[:, :, glow : ghi + 1]
            )
            grp.create_dataset("res-absorption", data=self.Ea[:, :, glow : ghi + 1])
            grp.create_dataset("res-scatter", data=self.Es[:, :, ilow:ihi])
            if self.Es1 is not None:
                grp.create_dataset("res-p1-scatter", data=self.Es1[:, :, ilow:ihi])
            if self.Es2 is not None:
                grp.create_dataset("res-p2-scatter", data=self.Es2[:, :, ilow:ihi])
            if self.Es3 is not None:
                grp.create_dataset("res-p3-scatter", data=self.Es3[:, :, ilow:ihi])
            if self.fissile:
                grp.create_dataset("res-fission", data=self.Ef[:, :, glow : ghi + 1])
            if self.Egamma is not None:
                grp.create_dataset(
                    "res-(n,gamma)", data=self.Egamma[:, :, glow : ghi + 1]
                )

    def _get_endf_info(self):
        # Get first MAT
        tape = endf.get_materials(self.endf_file)
        mat = tape[0]
        self.mat = mat.MAT

        # Read AWR and ZA from MF1 MT451
        mf1mt451 = mat[1,451]
        self.awr = mf1mt451['AWR']
        self.ZA = mf1mt451['ZA']
        self.isomeric_state = mf1mt451['LISO']
        self.ZAM = self.ZA
        if self.isomeric_state > 0:
            self.ZAM += 300 + (self.isomeric_state * 100)
        self.fissile = mf1mt451['LFI'] == 1
        self.fission_energy = None # Filled from ACE

        if self.fissile:
            self.fission_energy_release = get_fission_energy_release_components(mat)
            
            # Get energy release (in MeV) at 1 eV
            Ein = 1.0
            self.fission_energy = self.fission_energy_release['EFR'](Ein) + self.fission_energy_release['EGP'](Ein) + self.fission_energy_release['EGD'](Ein) + self.fission_energy_release['EB'](Ein) 
            self.fission_energy *= 1.E-6 # Convert from eV to MeV
        
        # This is created later
        self.ir_lambda = None

        # Get potential scattering
        self.pot_xs = get_potential_scatter_xs(mat)

    def _frendy_input(self, temp):
        # Will write (n,2n), (n,3n), (n,gamma), (n,p), and (n,alpha)
        out = "mg_neutron_mode\n"
        out += "mg_edit_option ( KRAMXS MGFlux 1DXS 16, 17, 102, 103, 107 )\n"
        out += f"nucl_file_name ({self.endf_file})\n"
        if self.tsl_file is not None:
            out += f"nucl_file_name_tsl ({self.tsl_file})\n"
            out += f"mg_tsl_data_type {self.tsl_type}\n"
        if self.pot_xs is not None:
            out += f"potential_scat_xs {self.pot_xs}\n"
        out += f"mg_file_name {self.name}\n"
        out += f"temperature {temp}\n"
        out += f"legendre_order {self.max_legendre_moment}\n"
        if self.group_structure.id is not None:
            out += f"mg_structure ( {self.group_structure.id} )\n"
        else:
            ebnd_frmt = len(self.group_structure.bounds) * "{:.7E}  "
            out += (
                "mg_structure ( "
                + ebnd_frmt.format(*np.flip(self.group_structure.bounds))
                + " )\n"
            )
        out += "mg_weighting_spectrum ( fission+1/e+maxwell  )\n"
        out += "process_gas_xs off\n"
        if self.dilutions is None or (isinstance(self.dilutions, list) and len(self.dilutions) == 0):
            out += "sigma_zero_data ( auto 0.005 100 1.E-10 rr linear )"
        else:
            dil_frmt = len(self.dilutions) * "{:.2E} "
            out += "sigma_zero_data ( " + dil_frmt.format(*self.dilutions) + " )\n"
        return out

    def _process_temp(self, itemp, generate_ir_lambda=False):
        temp = self.temps[itemp]
        frendy_input = self._frendy_input(temp)
        with open(self.run_dir / "frendy_input", "w") as fl:
            fl.write(frendy_input)

        #subprocess.run([self.frendy_exe, "frendy_input"])
        frendy_proc = Popen([self.frendy_exe, "frendy_input"], cwd=self.run_dir, stdout=PIPE, stderr=STDOUT, universal_newlines=True)
        
        # Get all output of FRENDY and copy to terminal
        while True:
            line = frendy_proc.stdout.readline()
            if not line and frendy_proc.poll() is not None:
                break
            if self.verbose:
                print(line, end='')

        # Check for an error
        if frendy_proc.returncode != 0:
            raise CalledProcessError(frendy_proc.returncode, self.frendy_exe, f"FRENDY failed to process {self.name}")


        if itemp == 0 and self.dilutions is None or (isinstance(self.dilutions, list) and len(self.dilutions) == 0):
            self._get_dilutions()

        self._read_temp(itemp)

        if generate_ir_lambda:
            self.ir_lambda = generate_U238_U235_ir_lambda(
                _U238,
                _U235,
                awr=self.awr,
                sig_pot=self.pot_xs,
                mg_energy_bounds=self.group_structure.bounds,
                verbose=self.verbose
            )

        if self.delete_files:
            try:
                os.remove(self.run_dir / "frendy_input")
                os.remove(self.run_dir / "FMAlternateInputData.txt")
                os.remove(self.run_dir / Path(os.path.basename(self.endf_file) + ".ace"))
                os.remove(self.run_dir / Path(os.path.basename(self.endf_file) + ".ace.dir"))

                if self.tsl_file is not None:
                    os.remove(self.run_dir / Path(os.path.basename(self.tsl_file) + ".ace"))
                    os.remove(self.run_dir / Path(os.path.basename(self.tsl_file) + ".ace.dir"))
            except:
                pass

            for fl in os.listdir(self.run_dir):
                if self.name + "_" in fl:
                    try:
                        os.remove(self.run_dir / fl)
                    except:
                        pass

    def _get_dilutions(self):
        fname = self.run_dir / Path(self.name + "_MGFlux.mg")
        fl = open(fname, "r")
        fl.readline()
        line = fl.readline()
        fl.close()
        line = line.strip().split()
        line = line[2:]
        for i in range(len(line)):
            line[i] = float(line[i])
        line.sort()
        self.dilutions = line
        self._allocate_arrays()

    def _read_temp(self, itemp):
        # For each dilution, we need to read the KRAMXS file
        for d in range(len(self.dilutions)):
            # Read xs file
            fname = self.run_dir / Path(self.name + "_KRAMXS_MACRO_bg" + str(d) + ".mg")
            xs = KRAMXS.from_file(fname, self.max_legendre_moment)

            # Save values. FRENDY order dilutions from high to low, hence the index
            # shift on d to add them backwards
            self.Ea[itemp, -(d + 1), :] = xs.Ea
            self.Es[itemp, -(d + 1), :, :] = xs.Es
            if xs.Es1 is not None:
                self.Es1[itemp, -(d + 1), :, :] = xs.Es1
            if xs.Es2 is not None:
                self.Es2[itemp, -(d + 1), :, :] = xs.Es2
            if xs.Es3 is not None:
                self.Es3[itemp, -(d + 1), :, :] = xs.Es3
            if self.fissile:
                self.Ef[itemp, -(d + 1), :] = xs.Ef
                if itemp == 0 and d == 0:
                    self.nu = xs.nu
                    self.chi = xs.chi

        # read in (n,gamma) data
        fname = self.run_dir / Path(self.name + "_1DXS_" + str(self.ZAM) + ".00c_MT102.mg")
        if fname.exists():
            if itemp == 0:
                self.Egamma = np.zeros(
                    (len(self.temps), len(self.dilutions), self.ngroups)
                )
            ngamma = read_1dxs(fname, 3)
            self.Egamma[itemp, :, :] = ngamma

        # Check for (n,2n) data
        fname = self.run_dir / Path(self.name + "_1DXS_" + str(self.ZAM) + ".00c_MT16.mg")
        if fname.exists():
            if itemp == 0:
                self.En2n = np.zeros(
                    (len(self.temps), len(self.dilutions), self.ngroups)
                )
            n2n = read_1dxs(fname, 3)
            self.En2n[itemp, :, :] = n2n

        # Check for (n,3n) data
        fname = self.run_dir / Path(self.name + "_1DXS_" + str(self.ZAM) + ".00c_MT17.mg")
        if fname.exists():
            if itemp == 0:
                self.En3n = np.zeros(
                    (len(self.temps), len(self.dilutions), self.ngroups)
                )
            n3n = read_1dxs(fname, 3)
            self.En3n[itemp, :, :] = n3n

        # Check for (n,p) data
        fname = self.run_dir / Path(self.name + "_1DXS_" + str(self.ZAM) + ".00c_MT103.mg")
        if fname.exists():
            if itemp == 0:
                self.Enp = np.zeros(
                    (len(self.temps), len(self.dilutions), self.ngroups)
                )
            nprt = read_1dxs(fname, 3)
            self.Enp[itemp, :, :] = nprt

        # Check for (n,a) data
        fname = self.run_dir / Path(self.name + "_1DXS_" + str(self.ZAM) + ".00c_MT107.mg")
        if fname.exists():
            if itemp == 0:
                self.Ena = np.zeros(
                    (len(self.temps), len(self.dilutions), self.ngroups)
                )
            na = read_1dxs(fname, 3)
            self.Ena[itemp, :, :] = na


def read_1dxs(fname, nskip):
    fl = open(fname, "r")

    # Skip first lines that have headers / dilutions / temperatures
    for i in range(nskip):
        fl.readline()

    array = []

    for line in fl:
        line = line.strip()
        if len(line) == 0:
            continue

        line = line.split()[4:]
        line.reverse()  # Reverse line for dilutions to go from low to high
        for i in range(len(line)):
            line[i] = float(line[i])
        array.append(line)
    fl.close()

    array = np.array(array, dtype=np.float32)
    array = np.copy(np.swapaxes(array, 0, 1))

    # First index on dilution, second on group
    return array

def process_nuclide(
    name: str,
    endf_fname: str,
    lib: str,
    temperatures: list[float],
    tsl_fname: str | None = None,
    tsl_type: str | None = None,
    dilutions: list[float] | None = None,
    chi: np.ndarray | None = None,
    verbose: bool = False
):
    print(f"Starting to process {name}")
    orig_dir = Path(os.getcwd())
    tempdir = tempfile.TemporaryDirectory() 
    tempdir_path = Path(tempdir.name)

    # Make an HDF5 file for just this nuclide
    nuc_h5 = h5py.File(tempdir_path / f"{name}.h5", "w")
    
    # Make nuclide object to do processing
    N = FrendyMG()
    N.delete_files = False
    N.verbose = verbose
    N.run_dir = tempdir_path # Make sure to set run path !
    N.name = name
    N.endf_file = endf_fname
    N.tsl_file = tsl_fname
    N.tsl_type = tsl_type
    N.label = name + " from " + lib
    N.temps = temperatures
    N.dilutions = dilutions
    N.process(nuc_h5, chi)

    # Close HDF5 file
    nuc_h5.close()

    # Move HDF5 to original directory
    shutil.move(src=tempdir_path / f"{name}.h5", dst=orig_dir / f"{name}.h5")

    # Cleanup temporary directory
    tempdir.cleanup()

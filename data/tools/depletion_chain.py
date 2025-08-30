from scarabee import (
    NoTarget,
    SingleTarget,
    Branch,
    BranchingTargets,
    FissionYields,
    ChainEntry,
    DepletionChain,
    NDLibrary,
)
import numpy as np
import xml.etree.ElementTree as ET
from typing import Union


def strip_name(name) -> str:
    if "_" in name:
        return name.replace("_", "")
    return name


def read_fission_yields(node) -> Union[FissionYields, str]:
    if "parent" in node.attrib:
        # raise RuntimeError("Fission yields refers to parent nuclide {:}.".format(node.attrib['parent']))
        return strip_name(node.attrib["parent"])

    energies = node[0].text.split()
    energies = [float(E) for E in energies]
    NE = len(energies)  # Number of energies

    products = None
    yields = []

    for i in range(1, NE + 1):
        fiss_yields = node[i]

        prods = fiss_yields[0].text.split()
        prods = [strip_name(n) for n in prods]
        if products is None:
            products = prods
        else:
            # Make sure products are in the same order !
            if len(products) != len(prods):
                raise RuntimeError(
                    "Product lists at different energies are of different lengths."
                )
            for i in range(len(products)):
                if products[i] != prods[i]:
                    raise RuntimeError(
                        "Product lists at different energies do not agree."
                    )

        i_yields = fiss_yields[1].text.split()
        yields.append([float(y) for y in i_yields])

    yields = np.array(yields)

    return FissionYields(products, energies, yields)


def build_depletion_chain(omc_fname) -> DepletionChain:
    root = ET.parse(omc_fname).getroot()

    dc = DepletionChain()

    # First, go through all nuclides and obtain all fission yields which don't
    # just refer to a different nuclide.
    fiss_yields = {}
    for nuc in root:
        if (
            len(nuc) > 0
            and nuc[-1].tag == "neutron_fission_yields"
            and "parent" not in nuc[-1].attrib
        ):
            nuclide_name = strip_name(nuc.attrib["name"])
            fiss_yields[nuclide_name] = read_fission_yields(nuc[-1])

    # Parse all nuclides in the chain
    for nuc in root:
        nuclide_name = strip_name(nuc.attrib["name"])

        half_life = None
        decay_branches_list = []

        n_gamma = None
        n_2n = None
        n_3n = None
        n_p = None
        n_alpha = None
        n_fission = None

        if "half_life" in nuc.attrib:
            half_life = float(nuc.attrib["half_life"])

        for entry in nuc:
            if entry.tag == "decay":
                if "target" in entry.attrib:
                    branch = Branch()
                    branch.target = strip_name(entry.attrib["target"])
                    branch.branch_ratio = float(entry.attrib["branching_ratio"])
                    if branch.branch_ratio > 0.0:
                        decay_branches_list.append(branch)

            elif entry.tag == "reaction":
                reaction = entry.attrib["type"]
                transmut_target = None
                branch_ratio = None
                if "target" in entry.attrib:
                    transmut_target = strip_name(entry.attrib["target"])
                if "branching_ratio" in entry.attrib:
                    branch_ratio = float(entry.attrib["branching_ratio"])

                if reaction == "(n,gamma)":
                    if branch_ratio is None and transmut_target is None:
                        n_gamma = NoTarget()
                    elif branch_ratio is None:
                        n_gamma = SingleTarget(transmut_target)
                    else:
                        if n_gamma is None:
                            n_gamma = []
                        branch = Branch()
                        branch.target = transmut_target
                        branch.branch_ratio = branch_ratio
                        n_gamma.append(branch)

                elif reaction == "(n,2n)":
                    if branch_ratio is None and transmut_target is None:
                        n_2n = NoTarget()
                    elif branch_ratio is None:
                        n_2n = SingleTarget(transmut_target)
                    else:
                        if n_2n is None:
                            n_2n = []
                        branch = Branch()
                        branch.target = transmut_target
                        branch.branch_ratio = branch_ratio
                        n_2n.append(branch)

                elif reaction == "(n,3n)":
                    if branch_ratio is None and transmut_target is None:
                        n_3n = NoTarget()
                    elif branch_ratio is None:
                        n_3n = SingleTarget(transmut_target)
                    else:
                        if n_3n is None:
                            n_3n = []
                        branch = Branch()
                        branch.target = transmut_target
                        branch.branch_ratio = branch_ratio
                        n_3n.append(branch)

                elif reaction == "(n,p)":
                    if branch_ratio is None and transmut_target is None:
                        n_p = NoTarget()
                    elif branch_ratio is None:
                        n_p = SingleTarget(transmut_target)
                    else:
                        if n_p is None:
                            n_p = []
                        branch = Branch()
                        branch.target = transmut_target
                        branch.branch_ratio = branch_ratio
                        n_p.append(branch)

                elif reaction == "(n,a)":
                    if branch_ratio is None and transmut_target is None:
                        n_alpha = NoTarget()
                    elif branch_ratio is None:
                        n_alpha = SingleTarget(transmut_target)
                    else:
                        if n_alpha is None:
                            n_alpha = []
                        branch = Branch()
                        branch.target = transmut_target
                        branch.branch_ratio = branch_ratio
                        n_alpha.append(branch)
                else:
                    # OpenMC considers others like (n,4n), but Scarabée doesn't
                    # track these other transmutation reactions.
                    pass

            elif entry.tag == "neutron_fission_yields":
                parent = nuclide_name
                if "parent" in entry.attrib:
                    parent = strip_name(entry.attrib["parent"])
                n_fission = fiss_yields[parent]

        # Create decay targets (if there are any)
        decay_targets = None
        if len(decay_branches_list) == 1:
            decay_targets = SingleTarget(decay_branches_list[0].target)
        elif len(decay_branches_list) > 1:
            decay_targets = BranchingTargets(decay_branches_list)
        if half_life is not None and decay_targets is None:
            decay_targets = NoTarget()

        if isinstance(n_gamma, list):
            n_gamma = BranchingTargets(n_gamma)
        if isinstance(n_2n, list):
            n_2n = BranchingTargets(n_2n)
        if isinstance(n_3n, list):
            n_3n = BranchingTargets(n_3n)
        if isinstance(n_p, list):
            n_p = BranchingTargets(n_p)
        if isinstance(n_alpha, list):
            n_alpha = BranchingTargets(n_alpha)

        # Build the chain entry for the nuclide
        ce = ChainEntry()
        ce.half_life = half_life
        ce.decay_targets = decay_targets
        ce.n_gamma = n_gamma
        ce.n_2n = n_2n
        ce.n_3n = n_3n
        ce.n_p = n_p
        ce.n_alpha = n_alpha
        ce.n_fission = n_fission

        dc.insert_entry(nuclide_name, ce)

    # Expunge all short-lives nuclides (i.e. half life less than 24 hrs) that
    # aren't in important chains.
    nuclide_names = dc.nuclides
    for nuc_name in nuclide_names:
        if nuc_name in ["I135", "Xe135"]:
            continue

        nuc = dc.nuclide_data(nuc_name)

        if nuc.half_life is not None and nuc.half_life < 60.0 * 60.0 * 24.0:
            print("Removing {:} from chain".format(nuc_name))
            dc.remove_nuclide(nuc_name)
        elif nuc_name in [
            "Cd115",
            "Rh102",
            "Rh102m1",
            "Sb127",
            "Br82",
        ]:  # No Evals in ENDF-8.0 for these. RIP
            print("Removing {:} from chain".format(nuc_name))
            dc.remove_nuclide(nuc_name)

    return dc


def fix_target_name(name) -> str:
    if name[-2:] == "m1":
        name = name[:-2] + "_m1"
    return name


def print_no_target(name: str, target: NoTarget) -> None:
    print(f'    <reaction type="{name}" Q="0.0" />')


def print_single_target(name: str, target: SingleTarget) -> None:
    target_name = target.target
    target_name = fix_target_name(target_name)
    print(f'    <reaction type="{name}" Q="0.0" target="{target_name}" />')


def print_branching_targets(name: str, target: BranchingTargets) -> None:
    # OpenMC can have a branching reaction target. In this case, we find the largets
    max = 0.0
    max_i = -1
    for i in range(len(target.branches)):
        if target.branches[i].branch_ratio > max:
            max = target.branches[i].branch_ratio
            max_i = i

    branch = target.branches[max_i]
    target_name = branch.target
    target_name = fix_target_name(target_name)
    # print(f'    <reaction type="{name}" Q="0.0" branching_ratio="{branch.branch_ratio}" target="{target_name}"/>')
    print(f'    <reaction type="{name}" Q="0.0" target="{target_name}" />')


def print_fission_yields(
    name: str, nuc: str, target: FissionYields, ndl: NDLibrary
) -> None:
    nucdata = ndl.get_nuclide(nuc)
    Q = nucdata.fission_energy * 1.0e6

    print(f'    <reaction type="{name}" Q="{Q}" />')

    # Now must print fission yields !
    print("    <neutron_fission_yields>")

    # Print list of tabulated energies
    energies_str = "<energies>"
    for E in target.incident_energies:
        energies_str += f"{E:.8E} "
    energies_str = energies_str.strip()
    energies_str = "      " + energies_str + "</energies>"
    print(energies_str)

    # Print yields for each tabulated energy
    for i, E in enumerate(target.incident_energies):
        print(f'      <fission_yields energy="{E}">')

        targets = target.targets
        for i in range(len(targets)):
            targets[i] = fix_target_name(targets[i])

        yields = [0.0 for t in range(len(targets))]
        for i in range(len(targets)):
            yields[i] = target.fission_yield(i, E)

        targets_str = "<products>"
        for tr in targets:
            targets_str += f"{tr} "
        targets_str = targets_str.strip()
        targets_str = "        " + targets_str + "</products>"
        print(targets_str)

        yields_str = "<data>"
        for y in yields:
            yields_str += f"{y:.6E} "
        yields_str = yields_str.strip()
        yields_str = "        " + yields_str + "</data>"
        print(yields_str)

        print(f"      </fission_yields>")

    print("    </neutron_fission_yields>")


def print_reaction(
    name: str,
    nuc: str,
    target: Union[NoTarget, SingleTarget, BranchingTargets, FissionYields],
    ndl: NDLibrary,
) -> None:
    if isinstance(target, NoTarget):
        print_no_target(name, target)
    elif isinstance(target, SingleTarget):
        print_single_target(name, target)
    elif isinstance(target, BranchingTargets):
        print_branching_targets(name, target)
    elif isinstance(target, FissionYields):
        print_fission_yields(name, nuc, target, ndl)
    else:
        raise RuntimeError("Should never get here !")


def print_nuclide(nucname: str, entry: ChainEntry, ndl: NDLibrary) -> None:
    # Compute number of reactions
    NR = 0
    if entry.n_gamma:
        NR += 1
    if entry.n_2n:
        NR += 1
    if entry.n_3n:
        NR += 1
    if entry.n_p:
        NR += 1
    if entry.n_alpha:
        NR += 1
    if entry.n_fission:
        NR += 1

    # Get number of decay modes and number of reactions
    ND = 0
    if entry.half_life:
        if isinstance(entry.decay_targets, NoTarget):
            ND = 1
        elif isinstance(entry.decay_targets, SingleTarget):
            ND = 1
        elif isinstance(entry.decay_targets, BranchingTargets):
            ND = len(entry.decay_targets.branches)
        else:
            raise RuntimeError("Should never get here !")

    # Get half-life
    HL = 0.0
    if ND != 0:
        HL = entry.half_life

    # Print nuclide header
    if ND != 0:
        print(
            f'  <nuclide name="{fix_target_name(nucname)}" decay_modes="{ND}" reactions="{NR}" half_life="{HL:.6E}">'
        )
    else:
        print(
            f'  <nuclide name="{fix_target_name(nucname)}" decay_modes="{ND}" reactions="{NR}">'
        )

    # Print all decays (just say they are all beta- or something)
    if entry.decay_targets is not None:
        if isinstance(entry.decay_targets, NoTarget):
            print(f'    <decay type="beta" branching_ratio="1.0" />')
        elif isinstance(entry.decay_targets, SingleTarget):
            target_name = entry.decay_targets.target
            target_name = fix_target_name(target_name)
            print(
                f'    <decay type="beta" branching_ratio="1.0" target="{target_name}" />'
            )
        elif isinstance(entry.decay_targets, BranchingTargets):
            for i in range(len(entry.decay_targets.branches)):
                branch = entry.decay_targets.branches[i]
                target_name = branch.target
                target_name = fix_target_name(target_name)
                print(
                    f'    <decay type="beta" branching_ratio="{branch.branch_ratio}" target="{target_name}" />'
                )
        else:
            raise RuntimeError("Should never get here !")

    # Print reactions
    if entry.n_gamma:
        print_reaction("(n,gamma)", nucname, entry.n_gamma, ndl)
    if entry.n_2n:
        print_reaction("(n,2n)", nucname, entry.n_2n, ndl)
    if entry.n_3n:
        print_reaction("(n,3n)", nucname, entry.n_3n, ndl)
    if entry.n_p:
        print_reaction("(n,p)", nucname, entry.n_p, ndl)
    if entry.n_alpha:
        print_reaction("(n,a)", nucname, entry.n_alpha, ndl)
    if entry.n_fission:
        print_reaction("fission", nucname, entry.n_fission, ndl)

    # Close nuclide
    print("  </nuclide>")


def print_chain(chain: DepletionChain, ndl: NDLibrary) -> None:
    print('<?xml version="1.0"?>')
    print("<depletion_chain>")

    # List of all nuclides
    nuclides = list(chain.nuclides)

    for nuc in nuclides:
        print_nuclide(nuc, chain.nuclide_data(nuc), ndl)

    print("</depletion_chain>")

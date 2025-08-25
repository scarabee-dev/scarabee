#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <xtensor-python/pytensor.hpp>

#include <data/depletion_chain.hpp>
#include <utils/logging.hpp>
#include <utils/scarabee_exception.hpp>

namespace py = pybind11;

using namespace scarabee;

void init_DepletionChain(py::module& m) {
  //===========================================================================
  // TARGETS
  py::class_<NoTarget>(m, "NoTarget",
                       "Represents the end of a decay / transmutation chain "
                       "with no resulting nuclide.")
      .def(py::init<>());

  py::class_<SingleTarget>(m, "SingleTarget",
                           "Represents a single possible resulting nuclide "
                           "from a decay or transmutation.")
      .def(py::init<const std::string& /*target*/>(),
           "Creates a single nuclide depletion target.\n\n"
           "Parameters\n"
           "----------\n"
           "target : string\n"
           "    Name of the target nuclide.\n",
           py::arg("target"))

      .def_property_readonly("target", &SingleTarget::target,
                             "Name of the target nuclide.");

  py::class_<BranchingTargets::Branch>(
      m, "Branch",
      "Represents a single branch (or target) when multiple nuclide targets "
      "are possible from a decay or nuclear reaction. This should not be used "
      "for fission yields ! Instead use :class:`FissionYields`.")
      .def(py::init<>())
      .def_readwrite("target", &BranchingTargets::Branch::target,
                     "Target nuclide.")
      .def_readwrite("branch_ratio", &BranchingTargets::Branch::branch_ratio,
                     "Branch ratio.");

  py::class_<BranchingTargets>(
      m, "BranchingTargets",
      "Represents a set of possible targets for a sinble decay or reaction.")
      .def(
          py::init<const std::vector<BranchingTargets::Branch>& /*branches*/>(),
          "Creates a BranchingTargets object.\n\n"
          "Parameters\n"
          "----------\n"
          "branches : list of :class:`Branch`\n"
          "    Possible branches for the decay or transmutation.\n",
          py::arg("branches"))
      .def_property_readonly(
          "branches", &BranchingTargets::branches,
          "List of all branches with the target name and branch ratio.");

  //===========================================================================
  // Fission Yields
  py::class_<FissionYields>(
      m, "FissionYields",
      "Represents a set of multiple targets with different yields (not "
      "probabilities) resulting from fission. The yields can be a function of "
      "the incident neutron energy.")
      .def(py::init<const std::vector<std::string>& /*targets*/,
                    const std::vector<double>& /*incident_energies*/,
                    const xt::xtensor<double, 2>& /*yields*/>(),
           "Constructs a tabulated set of fission yields.\n\n"
           "Parameters\n"
           "----------\n"
           "targets : list of string\n"
           "    List of all possible targets.\n"
           "incident_energies : list of float\n"
           "    Sorted list of tabulated incident energies for which yields "
           "    are provided.\n"
           "yields : ndarray\n"
           "    2D Numpy array containing the fission yields where first axis "
           "    is incident energy and second is target.\n",
           py::arg("targets"), py::arg("incident_energies"), py::arg("yields"))
      .def_property_readonly("size", &FissionYields::size, "Number of targets.")
      .def_property_readonly("targets", &FissionYields::targets,
                             "List of possible target nuclides.")
      .def_property_readonly(
          "incident_energies", &FissionYields::incident_energies,
          "List of incident energies at which yields are tabulated.")
      .def("fission_yield", &FissionYields::yield,
           "Computes the fission yield for a specified target and incident "
           "neutron energy.\n\n"
           "Parameters\n"
           "----------\n"
           "t : int\n"
           "    Index of the target nuclide.\n"
           "E : float\n"
           "    Incident energy of the neutron inducing fission.\n\n"
           "Returns\n"
           "-------\n"
           "float\n"
           "    Yield of the specified target at the specified incident "
           "energy.\n",
           py::arg("t"), py::arg("E"));

  //===========================================================================
  // Chain Entry
  py::class_<ChainEntry>(
      m, "ChainEntry",
      "Represents all possible means by which a nuclide can be removed from a "
      "material. This includes radioactive decay and possible transmutations. "
      "For all possible loss paths, the possible resulting nuclide are also "
      "provided as different types of targets.")
      .def(py::init<>())

      .def_property(
          "half_life", [](const ChainEntry& c) { return c.half_life(); },
          [](ChainEntry& c, std::optional<double> hl) {
            if (hl.has_value() && hl.value() <= 0.) {
              const auto mssg = "Half life must be > 0.";
              spdlog::error(mssg);
              throw ScarabeeException(mssg);
            }
            c.half_life() = hl;
          },
          "Half life of the nuclide in seconds.")

      .def_property(
          "decay_targets",
          [](const ChainEntry& c) { return c.decay_targets(); },
          [](ChainEntry& c, std::optional<Target> t) { c.decay_targets() = t; },
          "Targets for the radioactive decay of the nuclide.")

      .def_property(
          "n_gamma", [](const ChainEntry& c) { return c.n_gamma(); },
          [](ChainEntry& c, std::optional<Target> t) { c.n_gamma() = t; },
          "Targets for the (n,gamma) transmutation of the nuclide.")

      .def_property(
          "n_2n", [](const ChainEntry& c) { return c.n_2n(); },
          [](ChainEntry& c, std::optional<Target> t) { c.n_2n() = t; },
          "Targets for the (n,2n) transmutation of the nuclide.")

      .def_property(
          "n_3n", [](const ChainEntry& c) { return c.n_3n(); },
          [](ChainEntry& c, std::optional<Target> t) { c.n_3n() = t; },
          "Targets for the (n,3n) transmutation of the nuclide.")

      .def_property(
          "n_p", [](const ChainEntry& c) { return c.n_p(); },
          [](ChainEntry& c, std::optional<Target> t) { c.n_p() = t; },
          "Targets for the (n,p) transmutation of the nuclide.")

      .def_property(
          "n_alpha", [](const ChainEntry& c) { return c.n_alpha(); },
          [](ChainEntry& c, std::optional<Target> t) { c.n_alpha() = t; },
          "Targets for the (n,alpha) transmutation of the nuclide.")

      .def_property(
          "n_fission", [](const ChainEntry& c) { return c.n_fission(); },
          [](ChainEntry& c, std::optional<FissionYields> f) {
            c.n_fission() = f;
          },
          "Energy dependent fission yields for the nuclide.");

  //===========================================================================
  // Depletion Chain
  py::class_<DepletionChain, std::shared_ptr<DepletionChain>>(
      m, "DepletionChain",
      "Represents a full depletion chain for use with a nuclear data library "
      "when performing depletion calculations. Holds data to account for "
      "radioactive decay and transmutation of nuclides.")

      .def(py::init<>())

      .def("holds_nuclide_data", &DepletionChain::holds_nuclide_data,
           "Returns True if a chain entry for the specified nuclide exists. If "
           "not, False is returned.\n\n"
           "Parameters\n"
           "----------\n"
           "nuclide : string\n"
           "    Name of nuclide.\n\n"
           "Returns\n"
           "-------\n"
           "bool\n"
           "    True if there is data for the nuclide. False otherwise.\n",
           py::arg("nuclide"))

      .def("nuclide_data", &DepletionChain::nuclide_data,
           "Returns the :class:`ChainEntry` for the specified nuclide.\n\n"
           "Parameters\n"
           "----------\n"
           "nuclide : string\n"
           "    Name of nuclide.\n\n"
           "Returns\n"
           "-------\n"
           "ChainEntry\n"
           "    Decay and transmutation data for the nuclide.\n",
           py::arg("nuclide"))

      .def("insert_entry", &DepletionChain::insert_entry,
           "Inserts decay and transmutation data for a new nuclide.\n\n"
           "Parameters\n"
           "----------\n"
           "nuclide : string\n"
           "    Name of nuclide.\n"
           "entry : ChainEntry\n"
           "    Decay and transmutation data for the nuclide.",
           py::arg("nuclide"), py::arg("entry"))

      .def("remove_nuclide", &DepletionChain::remove_nuclide,
           "Removes a nuclide from the chain. All targets referencing this "
           "nuclide are replaced with its radioactive decay targets.\n\n"
           "Parameters\n"
           "---------\n"
           "nuclide : string\n"
           "    Name of nuclide to remove.\n",
           py::arg("nuclide"))

      .def_property_readonly("nuclides", &DepletionChain::nuclides,
                             "Set of all nuclides that have a chain entry.")

      .def("descend_chains", &DepletionChain::descend_chains,
           "Finds the set of all possible nuclides given an initial set of "
           "nuclides.\n\n"
           "Parameters\n"
           "----------\n"
           "nuclides : set of string\n"
           "    Set of initial nuclides.\n"
           "decay_only : bool\n"
           "    Only include target nuclides from radioactive decay. Default "
           "is False.\n\n"
           "Returns\n"
           "-------\n"
           "list of string\n"
           "    Sorted list of all possible nuclides given the initial set of "
           "nuclides.\n",
           py::arg("nuclides"), py::arg("decay_only") = false)

      .def("save", &DepletionChain::save,
           "Saves the depletion chain to a binary file.\n\n"
           "Parameters\n"
           "----------\n"
           "fname : string\n"
           "    Name of the output file.\n",
           py::arg("fname"))

      .def_static("load", &DepletionChain::load,
                  "Loads a DepletionChain from a binary file.\n\n"
                  "Parameters\n"
                  "----------\n"
                  "fname : string\n"
                  "    Name of the binary file.\n\n"
                  "Returns\n"
                  "-------\n"
                  "DepletionChain\n"
                  "    Depletion chain from the file.\n",
                  py::arg("fname"));
}

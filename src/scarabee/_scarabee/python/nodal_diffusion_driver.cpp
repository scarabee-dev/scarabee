#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cereal/archives/portable_binary.hpp>

#include <xtensor-python/pytensor.hpp>

#include <diffusion/nodal_diffusion_driver.hpp>
#include <diffusion/finite_difference.hpp>
#include <diffusion/nem4.hpp>
#include <diffusion/sanm.hpp>

#include <sstream>

namespace py = pybind11;

using namespace scarabee;

template <NodalMethod NM>
struct NodalDiffusionDriverPickler {
  using NeighborInfo = NodalDiffusionDriver<NM>::NeighborInfo;

  static NodalDiffusionDriver<NM> from_state(py::tuple t) {
    NodalDiffusionDriver<NM> nd;

    // Load the binary portion of the tuple
    std::size_t n0, n1;
    py::bytes bytes = t[3].cast<py::bytes>();
    std::istringstream bits_stream(bytes,
                                   std::ios_base::binary | std::ios_base::in);
    {
      cereal::PortableBinaryInputArchive ar(bits_stream);
      ar(n0, n1, nd.nodes_, nd.reconstructed_flux_params_, nd.nodal_solver_,
         nd.surface_indices_, nd.surface_diffusion_coefficients_, nd.flux_,
         nd.NG_, nd.NM_, nd.nonlinear_update_frequency_,
         nd.source_extrapolation_frequency_, nd.max_bicgstab_iterations_,
         nd.keff_, nd.flux_tol_, nd.keff_tol_, nd.Dnl_tol_,
         nd.leakage_corrections_, nd.solved_);
    }

    nd.geom_ = t[0].cast<std::shared_ptr<DiffusionGeometry>>();

    std::vector<NeighborInfo> flat_neighbors =
        t[1].cast<std::vector<NeighborInfo>>();

    std::vector<std::pair<std::shared_ptr<DiffusionData>,
                          std::shared_ptr<DiffusionCrossSection>>>
        temp_mats = t[2].cast<
            std::vector<std::pair<std::shared_ptr<DiffusionData>,
                                  std::shared_ptr<DiffusionCrossSection>>>>();

    // Fill mats_ again
    nd.mats_.resize(temp_mats.size());
    for (std::size_t i = 0; i < temp_mats.size(); i++)
      nd.mats_[i] = {temp_mats[i].first, temp_mats[i].second};

    // Fill neighbors_ again
    nd.neighbors_.resize({n0, n1});
    for (std::size_t i = 0; i < nd.neighbors_.size(); i++) {
      nd.neighbors_.flat(i) = flat_neighbors[i];
    }

    return nd;
  }

  static py::tuple to_state(const NodalDiffusionDriver<NM>& nd) {
    // Mats contains pointers to XS objects that could be in Python, so it
    // shouldn't be serialized with cereal. We make a picklable copy here.
    std::vector<std::pair<std::shared_ptr<DiffusionData>,
                          std::shared_ptr<DiffusionCrossSection>>>
        temp_mats;
    temp_mats.reserve(nd.mats_.size());
    for (const auto& p : nd.mats_) temp_mats.push_back({p.dd, p.xs});

    // We also can't serialize directly an array of NeighborInfo
    const std::size_t n0 = nd.neighbors_.shape()[0];
    const std::size_t n1 = nd.neighbors_.shape()[1];
    std::vector<NeighborInfo> flat_neighbors;
    flat_neighbors.reserve(nd.neighbors_.size());
    for (std::size_t i = 0; i < nd.neighbors_.size(); i++)
      flat_neighbors.push_back(nd.neighbors_.flat(i));

    // Make a binary of things we don't need in the tuple
    std::ostringstream bits_stream(std::ios_base::binary | std::ios_base::out);
    {
      cereal::PortableBinaryOutputArchive ar(bits_stream);
      ar(n0, n1, nd.nodes_, nd.reconstructed_flux_params_, nd.nodal_solver_,
         nd.surface_indices_, nd.surface_diffusion_coefficients_, nd.flux_,
         nd.NG_, nd.NM_, nd.nonlinear_update_frequency_,
         nd.source_extrapolation_frequency_, nd.max_bicgstab_iterations_,
         nd.keff_, nd.flux_tol_, nd.keff_tol_, nd.Dnl_tol_,
         nd.leakage_corrections_, nd.solved_);
    }
    py::bytes bytes(bits_stream.str());

    return py::make_tuple(nd.geom_, flat_neighbors, temp_mats, bytes);
  }
};

template <NodalMethod NM>
void init_NodalDiffusionDriver(py::module& m, const char* class_name,
                               const char* description) {
  using NodalSolver = NodalDiffusionDriver<NM>;

  auto solver =
      py::class_<NodalSolver>(m, class_name, description)

          .def(py::init<std::shared_ptr<DiffusionGeometry> /*geom*/>(),
               "Initializes a nodal diffusion solver.\n\n"
               "Parameters\n"
               "----------\n"
               "geom : DiffusionGeometry\n"
               "       Problem deffinition to solve.")

          .def("solve", &NodalSolver::solve,
               py::call_guard<py::gil_scoped_release>(),
               "Solves the diffusion problem.")

          .def_property_readonly(
              "geometry", &NodalSolver::geometry,
              "The :py:class:`DiffusionGeometry` geometry for the problem.")

          .def_property_readonly("ngroups", &NodalSolver::ngroups,
                                 "Number of energy groups.")

          .def_property_readonly(
              "solved", &NodalSolver::solved,
              "True if the problem has been solved, False otherwise.")

          .def_property_readonly(
              "keff", &NodalSolver::keff,
              "Value of keff. This is 1 by default is solved is False.")

          .def_property(
              "keff_tolerance", &NodalSolver::keff_tolerance,
              &NodalSolver::set_keff_tolerance,
              "Maximum relative error in keff for problem convergence.")

          .def_property(
              "flux_tolerance", &NodalSolver::flux_tolerance,
              &NodalSolver::set_flux_tolerance,
              "Maximum relative error in the flux for problem convergence.")

          .def_property("nonlinear_update_frequency",
                        &NodalSolver::nonlinear_update_frequency,
                        &NodalSolver::set_nonlinear_update_frequency,
                        "Frequency at which nonlinear diffusion coefficients "
                        "are updated.")

          .def_property("source_extrapolation_frequency",
                        &NodalSolver::source_extrapolation_frequency,
                        &NodalSolver::set_source_extrapolation_frequency,
                        "Frequency at which the source is extrapolated to "
                        "accelerate convergence.")

          .def_property("max_inner_iterations",
                        &NodalSolver::max_inner_iterations,
                        &NodalSolver::set_max_inner_iterations,
                        "Maximum number of inner iteration to perform per "
                        "outer iteration.");

  if constexpr (NM::update_currents) {
    solver.def_property(
        "leakage_corrections", &NodalSolver::leakage_corrections,
        &NodalSolver::set_leakage_corrections,
        "Apply leakage corrections to update node cross sections.");
  }

  solver
      .def("flux",
           py::overload_cast<double /*x*/, double /*y*/, double /*z*/,
                             std::size_t /*g*/>(&NodalSolver::flux, py::const_),
           "Calculates the flux at the desired position and group. The "
           "lowest value for any coordinate is 0.\n\n"
           "Parameters\n"
           "----------\n"
           "x : float\n"
           "    Position along the x axis.\n"
           "y : float\n"
           "    Position along the y axis.\n"
           "z : float\n"
           "    Position along the z axis.\n"
           "g : ing\n"
           "    Energy group index.\n\n"
           "Returns\n"
           "-------\n"
           "float\n"
           "      Value of the flux.\n",
           py::arg("x"), py::arg("y"), py::arg("z"), py::arg("g"))

      .def("flux",
           py::overload_cast<const xt::xtensor<double, 1>& /*x*/,
                             const xt::xtensor<double, 1>& /*y*/,
                             const xt::xtensor<double, 1>& /*z*/>(
               &NodalSolver::flux, py::const_),
           "Constructs an array storing the flux at all desired (x,y,z) "
           "points and at all energy groups. The first index is the group, "
           "the second is x, the third is y, and the fourth is z.\n\n"
           "Parameters\n"
           "----------\n"
           "x : array of float\n"
           "    Positions along the x axis.\n"
           "y : array of float\n"
           "    Positions along the y axis.\n"
           "z : array of float\n"
           "    Positions along the z axis.\n\n"
           "Returns\n"
           "-------\n"
           "array of float\n"
           "      Value of the flux at all (g,x,y,z).\n",
           py::arg("x"), py::arg("y"), py::arg("z"))

      .def("avg_flux", &NodalSolver::avg_flux,
           "Constructs an array storing the value of the average flux in "
           "each node. The resulting array is indexed as (g, x, y, z).\n\n"
           "Returns\n"
           "-------\n"
           "array of float\n"
           "      Value of the average flux in each node.\n")

      .def("power",
           py::overload_cast<double /*x*/, double /*y*/, double /*z*/>(
               &NodalSolver::power, py::const_),
           "Calculates the power density at the desired position. The lowest "
           "value for any coordinate is 0.\n\n"
           "Parameters\n"
           "----------\n"
           "x : float\n"
           "    Position along the x axis.\n"
           "y : float\n"
           "    Position along the y axis.\n"
           "z : float\n"
           "    Position along the z axis.\n\n"
           "Returns\n"
           "-------\n"
           "float\n"
           "      Value of the power density.\n",
           py::arg("x"), py::arg("y"), py::arg("z"))

      .def("power",
           py::overload_cast<const xt::xtensor<double, 1>& /*x*/,
                             const xt::xtensor<double, 1>& /*y*/,
                             const xt::xtensor<double, 1>& /*z*/>(
               &NodalSolver::power, py::const_),
           "Constructs an array storing the power density at all desired "
           "(x,y,z) points. The first index is x, the second is y, and the "
           "third is z.\n\n"
           "Parameters\n"
           "----------\n"
           "x : array of float\n"
           "    Positions along the x axis.\n"
           "y : array of float\n"
           "    Positions along the y axis.\n"
           "z : array of float\n"
           "    Positions along the z axis.\n\n"
           "Returns\n"
           "-------\n"
           "array of float\n"
           "      Value of the power density at all (x,y,z).\n",
           py::arg("x"), py::arg("y"), py::arg("z"))

      .def("avg_power", &NodalSolver::avg_power,
           "Constructs an array storing the value of the average power "
           "density in each node. The resulting array is indexed as "
           "(x, y, z).\n\n"
           "Returns\n"
           "-------\n"
           "array of float\n"
           "      Value of the average power density in each node.\n")

      .def(py::pickle(&NodalDiffusionDriverPickler<NM>::to_state,
                      &NodalDiffusionDriverPickler<NM>::from_state));
}

void init_all_NodalDiffusionDrivers(py::module& m) {
  init_NodalDiffusionDriver<FiniteDifference>(
      m, "FDNodalDiffusionDriver",
      "A FDNodalDiffusionDriver solves a diffusion problem using the finite "
      "difference formalism, but inside the CMFD nodal solver shell. "
      "Therefore, a spatial discretization compatible with finite differences "
      "should be used. The geometry is defined with a "
      ":py:class:`DiffusionGeometry` instance.");

  init_NodalDiffusionDriver<NEM4>(
      m, "NEM4DiffusionDriver",
      "Solves a diffusion problem using the 4th order Nodal Expansion Method "
      "with CMFD acceleration. Uses a standard quadratic transverse leakage "
      "approximation. Can be used with assembly or half assembly sized nodes. "
      "The geometry is defined with a :py:class:`DiffusionGeometry` instance.");

  init_NodalDiffusionDriver<SANM>(
      m, "SANMDiffusionDriver",
      "Solves a diffusion problem using the Semi-Analytical Nodal Method "
      "with CMFD acceleration. Uses a standard quadratic transverse leakage "
      "approximation. Can be used with assembly or half assembly sized nodes. "
      "The geometry is defined with a :py:class:`DiffusionGeometry` instance.");
}

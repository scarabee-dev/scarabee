#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cereal/archives/portable_binary.hpp>

#include <xtensor-python/pytensor.hpp>

#include <diffusion/nem_diffusion_driver.hpp>

#include <sstream>

namespace py = pybind11;

using namespace scarabee;

struct NEMDiffusionDriverPickler {
  static NEMDiffusionDriver from_state(py::tuple t) {
    NEMDiffusionDriver nd;

    // Flat neighbors array
    std::vector<NEMDiffusionDriver::NeighborInfo> flat_neighbors;
    std::size_t n0 = nd.neighbors_.shape()[0];
    std::size_t n1 = nd.neighbors_.shape()[0];

    nd.geom_ = t[0].cast<std::shared_ptr<DiffusionGeometry>>();
    flat_neighbors = t[1].cast<std::vector<NEMDiffusionDriver::NeighborInfo>>();
    py::bytes bytes = t[2].cast<py::bytes>();

    // Make a binary of things we don't need in the tuple
    std::istringstream bits_stream(bytes,
                                   std::ios_base::binary | std::ios_base::in);
    {
      cereal::PortableBinaryInputArchive ar(bits_stream);
      ar(n0, n1, nd.NG_, nd.NM_, nd.flux_, nd.j_in_out_, nd.Rmats_, nd.Pmats_,
         nd.Q_, nd.geom_inds_, nd.mats_, nd.diff_datas_, nd.adf_, nd.keff_,
         nd.flux_tol_, nd.leakage_corrections_, nd.solved_);
    }

    // Must re-fill neighbors array
    nd.neighbors_.resize({n0, n1});
    for (std::size_t i = 0; i < nd.neighbors_.size(); i++)
      nd.neighbors_.flat(i) = flat_neighbors[i];

    return nd;
  }

  static py::tuple to_state(const NEMDiffusionDriver& nd) {
    // Make a flat neighbors array
    std::vector<NEMDiffusionDriver::NeighborInfo> flat_neighbors;
    const std::size_t n0 = nd.neighbors_.shape()[0];
    const std::size_t n1 = nd.neighbors_.shape()[0];
    flat_neighbors.reserve(nd.neighbors_.size());
    for (std::size_t i = 0; i < nd.neighbors_.size(); i++)
      flat_neighbors.push_back(nd.neighbors_.flat(i));

    // Make a binary of things we don't need in the tuple
    std::ostringstream bits_stream(std::ios_base::binary | std::ios_base::out);
    {
      cereal::PortableBinaryOutputArchive ar(bits_stream);
      ar(n0, n1, nd.NG_, nd.NM_, nd.flux_, nd.j_in_out_, nd.Rmats_, nd.Pmats_,
         nd.Q_, nd.geom_inds_, nd.mats_, nd.diff_datas_, nd.adf_, nd.keff_,
         nd.flux_tol_, nd.leakage_corrections_, nd.solved_);
    }
    py::bytes bytes(bits_stream.str());

    return py::make_tuple(nd.geom_, flat_neighbors, bytes);
  }
};

void init_NEMDiffusionDriver(py::module& m) {
  py::class_<NEMDiffusionDriver>(
      m, "NEMDiffusionDriver",
      "A NEMDiffusionDriver solves a diffusion problem using the nodal "
      "expansion method with the response matrix interface current "
      "formalism. It is capable of solving 3D problems which are "
      "defined by providing a :py:class:`DiffusionGeometry` instance.\n\n"
      ".. deprecated:: 0.1.2\n"
      "   `NEMDiffusionDriver` is deprecated and will be removed in the "
      "future. It is replaced by `NEM4DiffusionDriver` because the latter "
      "has better performance.")

      .def(py::init<std::shared_ptr<DiffusionGeometry> /*geom*/>(),
           "Initializes a nodal diffusion solver.\n\n"
           "Parameters\n"
           "----------\n"
           "geom : DiffusionGeometry\n"
           "       Problem deffinition to solve.")

      .def("solve", &NEMDiffusionDriver::solve, "Solves the diffusion problem.")

      .def_property_readonly(
          "geometry", &NEMDiffusionDriver::geometry,
          "The :py:class:`DiffusionGeometry` geometry for the problem.")

      .def_property_readonly("ngroups", &NEMDiffusionDriver::ngroups,
                             "Number of energy groups.")

      .def_property_readonly(
          "solved", &NEMDiffusionDriver::solved,
          "True if the problem has been solved, False otherwise.")

      .def_property_readonly(
          "keff", &NEMDiffusionDriver::keff,
          "Value of keff. This is 1 by default is solved is False.")

      .def_property("keff_tolerance", &NEMDiffusionDriver::keff_tolerance,
                    &NEMDiffusionDriver::set_keff_tolerance,
                    "Maximum relative error in keff for problem convergence.")

      .def_property(
          "flux_tolerance", &NEMDiffusionDriver::flux_tolerance,
          &NEMDiffusionDriver::set_flux_tolerance,
          "Maximum relative error in the flux for problem convergence.")

      .def_property("leakage_corrections",
                    &NEMDiffusionDriver::leakage_corrections,
                    &NEMDiffusionDriver::set_leakage_corrections,
                    "Apply leakage corrections to update node cross sections.")

      .def("flux",
           py::overload_cast<double /*x*/, double /*y*/, double /*z*/,
                             std::size_t /*g*/>(&NEMDiffusionDriver::flux,
                                                py::const_),
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
               &NEMDiffusionDriver::flux, py::const_),
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

      .def("avg_flux", &NEMDiffusionDriver::avg_flux,
           "Constructs an array storing the value of the average flux in "
           "each node. The resulting array is indexed as (g, x, y, z).\n\n"
           "Returns\n"
           "-------\n"
           "array of float\n"
           "      Value of the average flux in each node.\n")

      .def("power",
           py::overload_cast<double /*x*/, double /*y*/, double /*z*/>(
               &NEMDiffusionDriver::power, py::const_),
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
               &NEMDiffusionDriver::power, py::const_),
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

      .def("avg_power", &NEMDiffusionDriver::avg_power,
           "Constructs an array storing the value of the average power "
           "density in each node. The resulting array is indexed as "
           "(x, y, z).\n\n"
           "Returns\n"
           "-------\n"
           "array of float\n"
           "      Value of the average power density in each node.\n")

      .def(py::pickle(&NEMDiffusionDriverPickler::to_state,
                      &NEMDiffusionDriverPickler::from_state));
}

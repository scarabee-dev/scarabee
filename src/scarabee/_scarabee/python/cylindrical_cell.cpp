#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <xtensor-python/pytensor.hpp>

#include <cereal/archives/portable_binary.hpp>

#include <cylindrical_cell.hpp>

#include <sstream>

namespace py = pybind11;

using namespace scarabee;

struct CylindricalCellPickler {
  static std::shared_ptr<CylindricalCell> from_state(py::tuple t) {
    std::shared_ptr<CylindricalCell> cc(new CylindricalCell);

    cc->mats_ = t[0].cast<std::vector<std::shared_ptr<CrossSection>>>();

    py::bytes bytes = t[1].cast<py::bytes>();
    std::istringstream bits_stream(bytes,
                                   std::ios_base::binary | std::ios_base::in);
    {
      cereal::PortableBinaryInputArchive ar(bits_stream);
      ar(cc->p_, cc->X_, cc->Y_, cc->Gamma_, cc->radii_, cc->vols_,
         cc->ngroups_, cc->solved_);
    }

    return cc;
  }

  static py::tuple to_state(const std::shared_ptr<CylindricalCell>& cc) {
    std::ostringstream bits_stream(std::ios_base::binary | std::ios_base::out);
    {
      cereal::PortableBinaryOutputArchive ar(bits_stream);
      ar(cc->p_, cc->X_, cc->Y_, cc->Gamma_, cc->radii_, cc->vols_,
         cc->ngroups_, cc->solved_);
    }
    py::bytes bytes(bits_stream.str());
    return py::make_tuple(cc->mats_, bytes);
  }
};

void init_CylindricalCell(py::module& m) {
  py::class_<CylindricalCell, std::shared_ptr<CylindricalCell>>(
      m, "CylindricalCell",
      "A CylindricalCell object represents a one dimensional annular problem "
      "(generally a pin cell), that is solved with the method of collision "
      "probabilities. This object is responsible for computing the collision "
      "probabilities, the different response fluxes, and the multicollision "
      "blackness in each group.")

      .def(py::init<
               const std::vector<double>& /*radii*/,
               const std::vector<std::shared_ptr<CrossSection>>& /*mats*/>(),
           "Creates a CylindricalCell object for a 1D annular pin cell.\n\n"
           "Parameters\n"
           "----------\n"
           "radii : list of float\n"
           "        List of region radii.\n"
           "mats : list of CrossSection\n"
           "       List of :py:class:`CrossSection` for each region.",
           py::arg("radii"), py::arg("mats"))

      .def_property_readonly(
          "solved", &CylindricalCell::solved,
          "True is the system has been solved, False otherwise.")

      .def("solve", &CylindricalCell::solve,
           py::call_guard<py::gil_scoped_release>(),
           "Solves the system for partial flux responses.\n\n"
           "Parameters\n"
           "----------\n"
           "parallel : bool\n"
           "  If True, solves the cell in parallel. Default is False.\n",
           py::arg("parallel") = false)

      .def_property_readonly("ngroups", &CylindricalCell::ngroups,
                             "Number of energy groups.")

      .def_property_readonly("nregions", &CylindricalCell::nregions,
                             "Number of annular regions.")

      .def_property_readonly("Sb", &CylindricalCell::Sb,
                             "Outer surface area (circumference) of cell.")

      .def("volume", &CylindricalCell::volume,
           "Volume of region i.\n\n"
           "Parameters\n"
           "----------\n"
           "i : int\n"
           "    Region index.\n\n"
           "Returns\n"
           "-------\n"
           "float\n"
           "     Volume of the region.",
           py::arg("i"))

      .def("radius", &CylindricalCell::radius,
           "Radius of region i.\n\n"
           "Parameters\n"
           "----------\n"
           "i : int\n"
           "    Region index.\n\n"
           "Returns\n"
           "-------\n"
           "float\n"
           "     Radius of region.",
           py::arg("i"))

      .def("Y", &CylindricalCell::Y,
           "Current response flux. This is the flux in region i cause by a "
           "neutron entering the cell from the outer surface (with albedo a) "
           "in group g.\n\n"
           "Parameters\n"
           "----------\n"
           "a : float\n"
           "    Albedo of outer boundary.\n"
           "g : int\n"
           "    Energy group index.\n"
           "i : int\n"
           "    Region index.\n\n"
           "Returns\n"
           "-------\n"
           "float\n"
           "     Current response flux.",
           py::arg("a"), py::arg("g"), py::arg("i"))

      .def("x", &CylindricalCell::x,
           "First escape probability to the boundary from region i. This is "
           "the probability that a source neutron born in region i will reach "
           "the boundary for the first time.\n\n"
           "Parameters\n"
           "----------\n"
           "g : int\n"
           "    Energy group index.\n"
           "i : int\n"
           "    Region index.\n\n"
           "Returns\n"
           "-------\n"
           "float\n"
           "     First escape probability.",
           py::arg("g"), py::arg("i"))

      .def("X", &CylindricalCell::X,
           "Source response flux. This is the flux in region i cause by a unit "
           "source in region k, in group g.\n\n"
           "Parameters\n"
           "----------\n"
           "a : float\n"
           "    Albedo of outer boundary.\n"
           "g : int\n"
           "    Energy group index.\n"
           "i : int\n"
           "    Region index for response.\n"
           "k : int\n"
           "    Region index for source.\n\n"
           "Returns\n"
           "-------\n"
           "float\n"
           "     Source response flux.",
           py::arg("a"), py::arg("g"), py::arg("i"), py::arg("k"))

      .def("Gamma", &CylindricalCell::Gamma,
           "Multicollision blackness. The probability of neutrons entering "
           "through the outer surface Sb are absorbed.\n\n"
           "Parameters\n"
           "----------\n"
           "g : int\n"
           "    Energy group index.\n\n"
           "Returns\n"
           "-------\n"
           "float\n"
           "     Multicollision blackness.",
           py::arg("g"))

      .def("p", &CylindricalCell::p,
           "Symmetric collision probability for a neutron in region i to "
           "collide in region k. This matrix is symmetric, so p(i -> k) = "
           "p(k -> i).\n\n"
           "Parameters\n"
           "----------\n"
           "g : int\n"
           "    Energy group index.\n"
           "i : int\n"
           "    Region index for origin.\n"
           "k : int\n"
           "    Region index for destination.\n\n"
           "Returns\n"
           "-------\n"
           "float\n"
           "     Symmetric collision probability.",
           py::arg("g"), py::arg("i"), py::arg("k"))

      .def("xs", &CylindricalCell::xs,
           "CrossSection in region i.\n\n"
           "Parameters\n"
           "----------\n"
           "i : float\n"
           "    Region index.\n\n"
           "Returns\n"
           "CrossSection\n"
           "            Cross section in region i.",
           py::arg("i"))

      .def(py::pickle(&CylindricalCellPickler::to_state,
                      &CylindricalCellPickler::from_state));
}

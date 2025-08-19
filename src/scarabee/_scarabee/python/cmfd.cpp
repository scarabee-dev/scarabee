#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include <moc/cmfd.hpp>

namespace py = pybind11;

using namespace scarabee;

void init_CMFD(py::module& m) {
  py::enum_<CMFDSurfaceCrossing::Type>(m, "CMFDSurfaceCrossingType")
      .value("XN", CMFDSurfaceCrossing::Type::XN, "CMFD cell x < 0 side.")
      .value("XP", CMFDSurfaceCrossing::Type::XP, "CMFD cell x > 0 side.")
      .value("YN", CMFDSurfaceCrossing::Type::YN, "CMFD cell y < 0 side.")
      .value("YP", CMFDSurfaceCrossing::Type::YP, "CMFD cell y > 0 side.")
      .value("I", CMFDSurfaceCrossing::Type::I,
             "Corner of CMFD cell in quadrant I.")
      .value("II", CMFDSurfaceCrossing::Type::II,
             "Corner of CMFD cell in quadrant II.")
      .value("III", CMFDSurfaceCrossing::Type::III,
             "Corner of CMFD cell in quadrant III.")
      .value("IV", CMFDSurfaceCrossing::Type::IV,
             "Corner of CMFD cell in quadrant IV.");

  py::class_<CMFDSurfaceCrossing>(m, "CMFDSurfaceCrossing")
      .def_readonly("is_valid", &CMFDSurfaceCrossing::is_valid,
                    "Bool that indicates whether or not the segment end is on "
                    "a valid CMFD surface.")
      .def_readonly("cell_index", &CMFDSurfaceCrossing::cell_index,
                    "Int representing the linear cell index.")
      .def_readonly("crossing", &CMFDSurfaceCrossing::crossing,
                    "Defines which side / corner of the cell the segment end "
                    "passes through.");

  py::class_<CMFD, std::shared_ptr<CMFD>>(m, "CMFD")
      .def(py::init<const std::vector<double>& /*dx*/,
                    const std::vector<double>& /*dy*/,
                    const std::vector<
                        std::pair<std::size_t, std::size_t>>& /*groups*/>(),
           "A Cartesian mesh for accelerating MOC convergence.\n\n"
           "Parameters\n"
           "----------\n"
           "dx : list of float\n"
           "     Widths along x axis.\n"
           "dy : list of float\n"
           "     Widths along y axis.\n"
           "groups : list of 2D tuples of ints\n"
           "         The scheme for condensing energy groups.\n",
           py::arg("dx"), py::arg("dy"), py::arg("groups"))

      .def_property_readonly("nx", &CMFD::nx, "Number of CMFD tiles along x.")
      .def_property_readonly("ny", &CMFD::ny, "Number of CMFD tiles along y.")

      .def_property_readonly("dx", &CMFD::dx,
                             "List of tile widths along the x axis.")
      .def_property_readonly("dy", &CMFD::dy,
                             "List of tile widths along the y axis.")

      .def_property_readonly("condensation_scheme", &CMFD::condensation_scheme,
                             "Condensation scheme to go from the MOC group "
                             "structure to the CMFD group structure.")

      .def_property(
          "keff_tolerance", &CMFD::keff_tolerance, &CMFD::set_keff_tolerance,
          "Maximum relative absolute difference in keff for convergence.")

      .def_property(
          "flux_tolerance", &CMFD::flux_tolerance, &CMFD::set_flux_tolerance,
          "Maximum relative absolute difference in flux for convergence.")

      .def_property(
          "flux_limiting", &CMFD::flux_limiting, &CMFD::set_flux_limiting,
          "Whether or not to use the flux-limiting condition when calculating"
          "the surface and non-linear diffusion coefficients.")

      .def_property(
          "larsen_correction", &CMFD::larsen_correction,
          &CMFD::set_larsen_correction,
          "Flag indicating use of Larsen's corrected diffusion coefficient "
          "for optically thick meshes. Mutally exclusive with the od_cmfd "
          "flag.")

      .def_property(
          "od_cmfd", &CMFD::od_cmfd, &CMFD::set_od_cmfd,
          "Flag indicating use of optimally diffusive CMFD (odCMFD) to modify "
          "the diffusion coeffients. Mutally exclusive with the "
          "larsen_correction flag.")

      .def_property(
          "damping", &CMFD::damping, &CMFD::set_damping,
          "The damping factor used for under-relaxing the nonlinear diffusion "
          "coefficient between iterations.")

      .def_property("skip_moc_iterations", &CMFD::skip_moc_iterations,
                    &CMFD::set_skip_moc_iterations,
                    "Number of MOC iterations to skip before applying CMFD.")

      .def_property("unbounded_cmfd_solves", &CMFD::num_unbounded_solves,
                    &CMFD::set_num_unbounded_solves,
                    "Number of CMFD solves before flux update ratios are "
                    "clamped to the range (0.05, 20).")

      .def_property(
          "check_neutorn_balance", &CMFD::neutron_balance_check,
          &CMFD::set_neutron_balance_check,
          "Flag indicating that the neutron balance should be checked in each "
          "CMFD tile on each CMFD solve. Should only be used for debugging "
          "purposes.")

      .def("get_surface", &CMFD::get_surface,
           "Obtains the CMFD surface crossing info for a provided position and "
           "direction.\n\n"
           "Parameters\n"
           "----------\n"
           "r: Vector\n"
           "    Position of the desired point.\n"
           "u: Direction\n"
           "    Direction vector for disambiguating the cell region.\n\n"
           "Returns\n"
           "-------\n"
           "CMFDSurfaceCrossing\n"
           "    Contains the surface crossing information.\n",
           py::arg("r"), py::arg("u"))

      .def("get_tile", &CMFD::get_tile,
           "Finds the CMFD tile of a provided position and direction.\n\n"
           "Parameters\n"
           "----------\n"
           "r: Vector\n"
           "    Position of the desired point.\n"
           "u: Direction\n"
           "    Direction vector for disambiguating the cell region.\n\n"
           "Returns\n"
           "-------\n"
           "List of two ints\n"
           "    The x and y tile indices.\n",
           py::arg("r"), py::arg("u"))

      .def("get_x_neg_surf", &CMFD::get_x_neg_surf,
           "Obtains the index of negative x surface for tile (i,j).\n\n"
           "Parameters\n"
           "----------\n"
           "i: int.\n"
           "    x index of tile.\n"
           "j: int.\n"
           "    y index of tile.\n\n"
           "Returns\n"
           "-------\n"
           "int\n"
           "    Index of the surface.\n")

      .def("get_x_pos_surf", &CMFD::get_x_pos_surf,
           "Obtains the index of positive x surface for tile (i,j).\n\n"
           "Parameters\n"
           "----------\n"
           "i: int.\n"
           "    x index of tile.\n"
           "j: int.\n"
           "    y index of tile.\n\n"
           "Returns\n"
           "-------\n"
           "int\n"
           "    Index of the surface.\n")

      .def("get_y_neg_surf", &CMFD::get_y_neg_surf,
           "Obtains the index of negative y surface for tile (i,j).\n\n"
           "Parameters\n"
           "----------\n"
           "i: int.\n"
           "    x index of tile.\n"
           "j: int.\n"
           "    y index of tile.\n\n"
           "Returns\n"
           "-------\n"
           "int\n"
           "    Index of the surface.\n")

      .def("get_y_pos_surf", &CMFD::get_y_pos_surf,
           "Obtains the index of positive y surface for tile (i,j).\n\n"
           "Parameters\n"
           "----------\n"
           "i: int.\n"
           "    x index of tile.\n"
           "j: int.\n"
           "    y index of tile.\n\n"
           "Returns\n"
           "-------\n"
           "int\n"
           "    Index of the surface.\n")

      .def("tally_current", &CMFD::tally_current,
           "Tallies the current onto the appropriate CMFD surface(s).\n\n"
           "Parameters\n"
           "----------\n"
           "aflx: float\n"
           "    Value of the angular flux to be tallied.\n"
           "u: Direction\n"
           "    Direction of the angular flux.\n"
           "g: int\n"
           "    CMFD energy group index."
           "surf: CMFDSurfaceCrossing\n"
           "    Information for surfaces on which the current is tallied.\n",
           py::arg("aflx"), py::arg("u"), py::arg("g"), py::arg("surf"))

      .def("current", &CMFD::current,
           "Returns the current on a CMFD cell boundary.\n\n"
           "Parameters\n"
           "----------\n"
           "g: int\n"
           "    CMFD energy group.\n"
           "surf: int\n"
           "    CMFD surface index.\n\n"
           "Returns\n"
           "-------\n"
           "float\n"
           "    Tallied current in CMFD group g on surface surf.\n",
           py::arg("g"), py::arg("surf"))

      .def("flux", &CMFD::flux,
           "Gets the CMFD flux in a desired tile and group.\n\n"
           "Parameters\n"
           "----------\n"
           "i: int.\n"
           "    x index of tile.\n"
           "j: int.\n"
           "    y index of tile.\n"
           "g: int.\n"
           "    Energy group index.\n\n"
           "Returns\n"
           "-------\n"
           "float\n"
           "    The CMFD scalar flux at cell (i,j) in group g.\n",
           py::arg("i"), py::arg("j"), py::arg("g"));
}

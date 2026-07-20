#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <xtensor-python/pytensor.hpp>

#include <moc/moc_driver.hpp>
#include <utils/logging.hpp>
#include <utils/scarabee_exception.hpp>

#ifndef SCARABEE_ON_RTD
#include <moc/moc_plotter.hpp>
#include <ImApp/imapp.hpp>
#endif

#include <tuple>

namespace py = pybind11;

using namespace scarabee;

void init_MOCDriver(py::module& m) {
  py::class_<MOCDriver, std::shared_ptr<MOCDriver>>(m, "MOCDriver")
      .def(py::init<std::shared_ptr<Cartesian2D> /*geometry*/,
                    BoundaryCondition /*xmin = BoundaryCondition::Reflective*/,
                    BoundaryCondition /*xmax = BoundaryCondition::Reflective*/,
                    BoundaryCondition /*ymin = BoundaryCondition::Reflective*/,
                    BoundaryCondition /*ymax = BoundaryCondition::Reflective*/,
                    bool /*anisotropic = false*/>(),
           "Initializes a Method of Characteristics problem.\n\n"
           "Parameters\n"
           "----------\n"
           "geometry : :py:class:`Cartesian2D`\n"
           "           Geometry for the problem.\n"
           "xminbc : BoundaryCondition\n"
           "         Boundary condition at the lower x boundary.\n"
           "xmaxbc : BoundaryCondition\n"
           "         Boundary condition at the upper x boundary.\n"
           "yminbc : BoundaryCondition\n"
           "         Boundary condition at the lower y boundary.\n"
           "ymaxbc : BoundaryCondition\n"
           "         Boundary condition at the upper y boundary.\n"
           "anisotropic : bool\n"
           "             Enable the anisotropic scattering solver.\n",
           py::arg("geometry"),
           py::arg("xminbc") = BoundaryCondition::Reflective,
           py::arg("xmaxbc") = BoundaryCondition::Reflective,
           py::arg("yminbc") = BoundaryCondition::Reflective,
           py::arg("ymaxbc") = BoundaryCondition::Reflective,
           py::arg("anisotropic") = false)

      .def("generate_tracks", &MOCDriver::generate_tracks,
           "Traces tracks across the geometry for the calculation.\n\n"
           "Parameters\n"
           "----------\n"
           "nangles : int\n"
           "          Number of azimuthal angles (must be even).\n"
           "d : float\n"
           "    Max spacing between tracks of a given angle (in cm).\n"
           "polar_quad : PolarQuadrature\n"
           "             Polar quadrature for generating segment lengths.",
           py::arg("nangles"), py::arg("d"), py::arg("polar_quad"))

      .def(
          "generate_tracks",
          [](MOCDriver& md, std::uint32_t na, double d,
             PolarQuadratureType pq) { return md.generate_tracks(na, d, pq); },
          py::call_guard<py::gil_scoped_release>(),
          "Traces tracks across the geometry for the calculation.\n\n"
          "Parameters\n"
          "----------\n"
          "nangles : int\n"
          "          Number of azimuthal angles (must be even).\n"
          "d : float\n"
          "    Max spacing between tracks of a given angle (in cm).\n"
          "polar_quad : PolarQuadrature\n"
          "             Polar quadrature for generating segment lengths.",
          py::arg("nangles"), py::arg("d"), py::arg("polar_quad"))

      .def_property_readonly(
          "drawn", &MOCDriver::drawn,
          "True if geometry has been traced, False otherwise.")

      .def_property(
          "keff_tolerance", &MOCDriver::keff_tolerance,
          &MOCDriver::set_keff_tolerance,
          "Maximum relative absolute difference in keff for convergence.")

      .def_property(
          "flux_tolerance", &MOCDriver::flux_tolerance,
          &MOCDriver::set_flux_tolerance,
          "Maximum relative absolute difference in flux for convergence.")

      .def_property(
          "fsr_area_tolerance", &MOCDriver::fsr_area_tolerance,
          &MOCDriver::set_fsr_area_tolerance,
          "Maximum relative absolute difference between approximate "
          "flat source region area and the true flat source region "
          "area. Only used if check_fsr_areas is True. Default value is 0.05.")

      .def_property("check_fsr_areas", &MOCDriver::check_fsr_areas,
                    &MOCDriver::set_check_fsr_areas,
                    "Checks if the approximate flat source region area (from "
                    "numerical integration) is within fsr_area_tolerance of "
                    "the true flat source region area. It the check does not "
                    "pass, a warning is issued, but the calculation continues. "
                    "Default value is False.")

      .def_property("cmfd", &MOCDriver::cmfd, &MOCDriver::set_cmfd,
                    "CMFD object for convergence acceleration.")

      .def("flux",
           py::overload_cast<const Vector&, const Direction&, std::size_t,
                             std::size_t>(&MOCDriver::flux, py::const_),
           "Returns the scalar flux in group g at position r.\n\n"
           "Parameters\n"
           "----------\n"
           "r : Vector\n"
           "    Position at which to obtain flux.\n"
           "u : Direction\n"
           "    Direction vector for disambiguating the cell region.\n"
           "g : int\n"
           "    Energy group index.\n"
           "lj : int\n"
           "    Spherical harmonic index. Default is zero.\n\n",
           "Returns\n"
           "-------\n"
           "float\n"
           "     Flux at position r and in group g.\n",
           py::arg("r"), py::arg("u"), py::arg("g"), py::arg("lj") = 0)

      .def("flux",
           py::overload_cast<std::size_t, std::size_t, std::size_t>(
               &MOCDriver::flux, py::const_),
           "Returns the scalar flux in group g in Flat Source Region i.\n\n"
           "Parameters\n"
           "----------\n"
           "i : int\n"
           "    Flat Source Region index.\n"
           "g : int\n"
           "    Energy group index.\n"
           "lj : int\n"
           "    Spherical harmonic index. Default is zero.\n\n",
           "Returns\n"
           "-------\n"
           "float\n"
           "     Flux FSR i and in group g.\n",
           py::arg("i"), py::arg("g"), py::arg("lj") = 0)

      .def("volume",
           py::overload_cast<const Vector&, const Direction&>(
               &MOCDriver::volume, py::const_),
           "Returns the volume of the Flat Source Region at position r.\n\n"
           "Parameters\n"
           "----------\n"
           "r : Vector\n"
           "    Position at which to obtain flux.\n"
           "u : Direction\n"
           "    Direction vector for disambiguating the cell region.\n\n"
           "Returns\n"
           "float\n"
           "     Volume of the FSR at r.\n\n",
           py::arg("r"), py::arg("u"))

      .def("volume",
           py::overload_cast<std::size_t>(&MOCDriver::volume, py::const_),
           "Returns the volume of Flat Source Region i.\n\n"
           "Parameters\n"
           "----------\n"
           "i : int\n"
           "    Flat Source Region index.\n\n"
           "Returns\n"
           "-------\n"
           "float\n"
           "     Volume of FSR i.\n\n",
           py::arg("i"))

      .def("xs",
           py::overload_cast<const Vector&, const Direction&>(&MOCDriver::xs,
                                                              py::const_),
           "Returns the CrossSection at position r.\n\n"
           "Parameters\n"
           "----------\n"
           "r : Vector\n"
           "    Position at which to obtain flux.\n"
           "u : Direction\n"
           "    Direction vector for disambiguating the cell region.\n\n"
           "Returns\n"
           "-------\n"
           "CrossSection\n"
           "            Material cross sections at r.\n\n",
           py::arg("r"), py::arg("u"))

      .def("xs", py::overload_cast<std::size_t>(&MOCDriver::xs, py::const_),
           "Returns the CrossSection in Flat Source Region i.\n\n"
           "Parameters\n"
           "----------\n"
           "i : int\n"
           "    Flat Source Region index.\n\n"
           "Returns\n"
           "-------\n"
           "CrossSection\n"
           "            Material cross sections at r.\n\n",
           py::arg("i"))

      .def_property_readonly("keff", &MOCDriver::keff,
                             "Value of keff estimated by solver (1 by default "
                             "if no solution has been obtained).")

      .def_property_readonly("ngroups", &MOCDriver::ngroups,
                             "Number of energy groups.")

      .def_property_readonly(
          "num_spherical_harmonics", &MOCDriver::num_spherical_harmonics,
          "Number of spherical harmonics for storing the flux moments.")

      .def_property_readonly("polar_quadrature", &MOCDriver::polar_quadrature,
                             "Quadrature used for polar angle integration.")

      .def("solve", &MOCDriver::solve, py::call_guard<py::gil_scoped_release>(),
           "Begins iterations to solve problem.")

      .def_property(
          "sim_mode",
          [](const MOCDriver& md) -> SimulationMode { return md.sim_mode(); },
          [](MOCDriver& md, SimulationMode& m) { md.sim_mode() = m; },
          ":py:class:`SimulationMode` describing type of simulation "
          "(fixed-source or keff).")

      .def_property(
          "x_min_bc",
          [](const MOCDriver& md) -> BoundaryCondition {
            return md.x_min_bc();
          },
          [](MOCDriver& md, BoundaryCondition& bc) { md.x_min_bc() = bc; },
          ":py:class:`BoundadaryCondition` at x_min.")

      .def_property(
          "x_max_bc",
          [](const MOCDriver& md) -> BoundaryCondition {
            return md.x_max_bc();
          },
          [](MOCDriver& md, BoundaryCondition& bc) { md.x_max_bc() = bc; },
          ":py:class:`BoundadaryCondition` at x_max.")

      .def_property(
          "y_min_bc",
          [](const MOCDriver& md) -> BoundaryCondition {
            return md.y_min_bc();
          },
          [](MOCDriver& md, BoundaryCondition& bc) { md.y_min_bc() = bc; },
          ":py:class:`BoundadaryCondition` at y_min.")

      .def_property(
          "y_max_bc",
          [](const MOCDriver& md) -> BoundaryCondition {
            return md.y_max_bc();
          },
          [](MOCDriver& md, BoundaryCondition& bc) { md.y_max_bc() = bc; },
          ":py:class:`BoundadaryCondition` at y_max.")

      .def_property_readonly(
          "geometry", &MOCDriver::geometry,
          "The :py:class:`Cartesian2D` geometry for the problem.")

      .def_property_readonly("size", &MOCDriver::size,
                             "Number of flat source regions.")

      .def_property_readonly("nfsr", &MOCDriver::nfsr,
                             "Number of flat source regions.")

      .def_property_readonly("nregions", &MOCDriver::nregions,
                             "Number of flat source regions.")

      .def_property_readonly("max_legendre_order",
                             &MOCDriver::max_legendre_order,
                             "Maximum legendre order for scattering.")

      .def_property_readonly(
          "anisotropic", &MOCDriver::anisotropic,
          "If True, anisotropic scattering will be simulated.")

      .def_property_readonly("x_min", &MOCDriver::x_min,
                             "Minimum value of x in problem domain.")

      .def_property_readonly("x_max", &MOCDriver::x_max,
                             "Maximum value of x in problem domain.")

      .def_property_readonly("y_min", &MOCDriver::y_min,
                             "Minimum value of y in problem domain.")

      .def_property_readonly("y_max", &MOCDriver::y_max,
                             "Maximum value of y in problem domain.")

      .def_property_readonly("solved", &MOCDriver::solved,
                             "True if solve has been run sucessfully (reset to "
                             "false on generate_tracks).")

      .def("get_all_fsr_in_cell", &MOCDriver::get_all_fsr_in_cell,
           "Obtains the index of all Flat Source Regions contained in the Cell "
           "located at position r.\n\n"
           "Parameters\n"
           "----------\n"
           "r : Vector\n"
           "    Position at which to set the source.\n"
           "u : Direction\n"
           "    Direction vector used to disambiguate the FSR.\n\n"
           "Returns\n"
           "-------\n"
           "list of int\n"
           "    Indices of all flat source regions in the cell.\n",
           py::arg("r"), py::arg("u"))

      .def("get_fsr_indx",
           py::overload_cast<std::size_t, std::size_t>(&MOCDriver::get_fsr_indx,
                                                       py::const_),
           "Obtains the index for a given flat source region ID and "
           "instance.\n\n"
           "Parameters\n"
           "----------\n"
           "fsr_id : int\n"
           "    Flat source region ID.\n"
           "instance : int\n"
           "    Desired instance of the provided FSR ID.\n\n"
           "Returns\n"
           "-------\n"
           "int\n"
           "    Index in the MOCDriver of the specified FSR instance.\n",
           py::arg("fsr_id"), py::arg("instance"))

      .def("set_extern_src",
           py::overload_cast<const Vector&, const Direction&, std::size_t,
                             double>(&MOCDriver::set_extern_src),
           "Sets the external source in the Flat Source Region at r.\n\n"
           "Parameters\n"
           "----------\n"
           "r : Vector\n"
           "    Position at which to set the source.\n"
           "u : Direction\n"
           "    Direction vector used to disambiguate the FSR.\n"
           "g : int\n"
           "    Energy group index.\n"
           "src : float\n"
           "      Value of source in the FSR.\n",
           py::arg("r"), py::arg("u"), py::arg("g"), py::arg("src"))

      .def("set_extern_src",
           py::overload_cast<std::size_t, std::size_t, double>(
               &MOCDriver::set_extern_src),
           "Sets the external source in Flat Source Region with index i.\n\n"
           "Parameters\n"
           "----------\n"
           "i : int\n"
           "    Flat Source Region index.\n"
           "g : int\n"
           "    Energy group index.\n"
           "src : float\n"
           "      Value of source in the FSR.\n",
           py::arg("i"), py::arg("g"), py::arg("src"))

      .def("extern_src",
           py::overload_cast<const Vector&, const Direction&, std::size_t>(
               &MOCDriver::extern_src, py::const_),
           "Returns the external source in the Flat Source Region at r.\n\n"
           "Parameters\n"
           "----------\n"
           "r : Vector\n"
           "    Position at which to set the source.\n"
           "u : Direction\n"
           "    Direction vector used to disambiguate the FSR.\n"
           "g : int\n"
           "    Energy group index.\n\n"
           "Returns\n"
           "-------\n"
           "float\n"
           "     Value of the external source at position r.\n",
           py::arg("r"), py::arg("u"), py::arg("g"))

      .def("extern_src",
           py::overload_cast<std::size_t, std::size_t>(&MOCDriver::extern_src,
                                                       py::const_),
           "Returns the external source in Flat Source Region i.\n\n"
           "Parameters\n"
           "----------\n"
           "i : int\n"
           "    Flat Source Region index.\n"
           "g : int\n"
           "    Energy group index.\n\n"
           "Returns\n"
           "-------\n"
           "float\n"
           "     Value of the external source at position r.\n",
           py::arg("i"), py::arg("g"))

      .def("homogenize",
           py::overload_cast<>(&MOCDriver::homogenize, py::const_),
           "Computes a homogenized set of cross sections for the problem "
           "based on the previously computed flux and reaction rates. This "
           "method raises an exception if the problem has not yet been "
           "solved.\n\n"
           "Returns\n"
           "-------\n"
           ":py:class:`CrossSection`\n"
           "            Homogenized cross section.\n")

      .def("homogenize",
           py::overload_cast<const std::vector<std::size_t>&>(
               &MOCDriver::homogenize, py::const_),
           "Computes a homogenized set of cross sections for the set of "
           "provided region indices.\n\n"
           "Parameters\n"
           "----------\n"
           "regions : list of int\n"
           "          List of regions for homogenization.\n"
           "Returns\n"
           "-------\n"
           ":py:class:`CrossSection`\n"
           "                        Homogenized cross section.\n",
           py::arg("regions"))

      .def(
          "homogenize_flux_spectrum",
          py::overload_cast<>(&MOCDriver::homogenize_flux_spectrum, py::const_),
          "Computes a homogenized flux spectrum based on the previously "
          "computed flux, which can be used for energy condensation. This "
          "method will raise an exception if the problem has not yet been "
          "solved.\n\n"
          "Returns\n"
          "-------\n"
          "ndarray of floats\n"
          "                 Homogenized flux spectrum.\n")

      .def("homogenize_flux_spectrum",
           py::overload_cast<const std::vector<std::size_t>&>(
               &MOCDriver::homogenize_flux_spectrum, py::const_),
           "Computes a homogenized flux spectrum based on the list of "
           "provided region indices. This method will raise an exception if "
           "the problem has not yet been solved.\n\n"
           "Parameters\n"
           "----------\n"
           "regions : list of int\n"
           "          List of regions for homogenization.\n"
           "Returns\n"
           "-------\n"
           "ndarray of floats\n"
           "                 Homogenized flux spectrum.",
           py::arg("regions"))

      .def("apply_criticality_spectrum", &MOCDriver::apply_criticality_spectrum,
           "Modifies the flux spectrum of the solved problem by multiplying "
           "the value of the flux by the ratio of the provided criticality "
           "spectrum to homogenized flux spectrum.\n\n"
           "Parameters\n"
           "----------\n"
           "flux : ndarray of floats\n"
           "       Criticality spectrum from a P1 or B1 calculation.\n",
           py::arg("flux"))

      .def("trace_fsr_segments", &MOCDriver::trace_fsr_segments,
           "Starting from a given position and direction, this function traces "
           "across the geometry until leaving the problem domain, returning a "
           "list of FSR index - distance pairs.\n\n"
           "Parameters\n"
           "----------\n"
           "r_start : Vector\n"
           "    Starting position.\n"
           "u : Direction\n"
           "    Direction to trace segments.\n\n"
           "Returns\n"
           "-------\n"
           "list of pairs of int and float\n"
           "    All the FSR index - distance pairs.\n",
           py::arg("r_start"), py::arg("u"))

      .def(
          "plot",
#ifndef SCARABEE_ON_RTD
          [](const MOCDriver& md) {
            ImApp::App guiplotter(1920, 1080, "Scarabee MOC Plotter");
            guiplotter.io().IniFilename = "scarabee_moc_plotter.ini";
            guiplotter.enable_docking();
            guiplotter.push_layer(std::make_unique<MOCPlotter>(&md));
            guiplotter.run();
          },
#else
          [](const MOCDriver& /*md*/) {
            const auto mssg = "Not built with plotting capabilities.";
            spdlog::error(mssg);
            throw ScarabeeException(mssg);
          },
#endif
          "Open the graphical MOC geometry plotting window.")

      .def(
          "rasterize_flux",
          [](const MOCDriver& md, std::size_t nx, std::size_t ny) {
            if (nx == 0) {
              auto mssg = "Rasterizing flux requires nx > 0.";
              spdlog::error(mssg);
              throw ScarabeeException(mssg);
            } else if (ny == 0) {
              auto mssg = "Rasterizing flux requires ny > 0.";
              spdlog::error(mssg);
              throw ScarabeeException(mssg);
            }

            if (md.solved() == false) {
              auto mssg = "Cannot rasterize flux. System has not been solved.";
              spdlog::error(mssg);
              throw ScarabeeException(mssg);
            }

            const double dx =
                (md.x_max() - md.x_min()) / static_cast<double>(nx);
            const double dy =
                (md.y_max() - md.y_min()) / static_cast<double>(ny);
            const Direction u(1., 1.);

            // Initialize arrays to return
            xt::pytensor<double, 3> flux =
                xt::zeros<double>({md.ngroups(), ny, nx});
            xt::pytensor<double, 1> x_bnds = xt::zeros<double>({nx + 1});
            xt::pytensor<double, 1> y_bnds = xt::zeros<double>({ny + 1});

            // Fill bounds
            x_bnds(0) = md.x_min();
            for (std::size_t i = 0; i < nx; i++) {
              x_bnds(i + 1) = x_bnds(i) + dx;
            }

            y_bnds(0) = md.y_min();
            for (std::size_t j = 0; j < ny; j++) {
              y_bnds(j + 1) = y_bnds(j) + dy;
            }

            // Rasterize flux
            for (std::size_t g = 0; g < md.ngroups(); g++) {
#pragma omp parallel for
              for (int ij = 0; ij < static_cast<int>(ny); ij++) {
                const std::size_t j = static_cast<std::size_t>(ij);
                const double y =
                    md.y_min() + (static_cast<double>(j) + 0.5) * dy;
                for (std::size_t i = 0; i < nx; i++) {
                  const double x =
                      md.x_min() + (static_cast<double>(i) + 0.5) * dx;
                  const Vector r(x, y);
                  flux(g, j, i) = md.flux(r, u, g);
                }
              }
            }

            // Return values
            return std::tuple<xt::pytensor<double, 3>, xt::pytensor<double, 1>,
                              xt::pytensor<double, 1>>{flux, x_bnds, y_bnds};
          },
          "Rasterizes the flux in all energy groups for easy plotting.\n\n"
          "Parameters\n"
          "----------\n"
          "nx : int\n"
          "     Number of mesh bins along x.\n"
          "ny : int\n"
          "     Number of mesh bins along y.\n\n"
          "Returns\n"
          "-------\n"
          "flux : ndarray\n"
          "       Values of the flux. First index is group, second is y, third "
          "is x.\n"
          "x : ndarray\n"
          "    Array of bounding x values.\n"
          "y : ndarray\n"
          "    Array of bounding y values.\n",
          py::arg("nx"), py::arg("ny"))

      .def("save", &MOCDriver::save_bin,
           "Saves MOCDriver to a binary file.\n\n"
           "Parameters\n"
           "----------\n"
           "fname : str\n"
           "        Name of file.\n",
           py::arg("fname"))

      .def_static("load", &MOCDriver::load_bin,
                  "Loads MOCDriver from a binary file.\n\n"
                  "Parameters\n"
                  "----------\n"
                  "fname : str\n"
                  "        Name of file.\n",
                  py::arg("fname"));
}

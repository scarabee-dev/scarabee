#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <xtensor-python/pyarray.hpp>

#include <diffusion/diffusion_geometry.hpp>

namespace py = pybind11;

using namespace scarabee;

void init_DiffusionGeometry(py::module& m) {
  // Tile definition
  py::class_<DiffusionGeometry::Tile>(
      m, "DiffusionGeometryTile",
      "A DiffusionGeometryTile represents an element of a cartesian diffusion "
      "mesh. It can have either an albedo entry (float) or a xs entry "
      "(:py:class:`DiffusionData`), but not both.")

      .def_readwrite("albedo", &DiffusionGeometry::Tile::albedo,
                     "The albedo if the tile is a boundary condition.")

      .def_readwrite("xs", &DiffusionGeometry::Tile::xs,
                     "The DiffusionData if the tile represents a material.")

      .def(py::pickle(
          [](const DiffusionGeometry::Tile tile) {
            return py::make_tuple(tile.albedo, tile.xs);
          },
          [](py::tuple t) {
            return DiffusionGeometry::Tile{
                t[0].cast<std::optional<double>>(),
                t[1].cast<std::shared_ptr<DiffusionData>>()};
          }));

  py::enum_<DiffusionGeometry::Neighbor>(m, "Neighbor")
      .value("XN", DiffusionGeometry::Neighbor::XN,
             "Neighbor on the x < 0 side.")
      .value("XP", DiffusionGeometry::Neighbor::XP,
             "Neighbor on the x > 0 side.")
      .value("YN", DiffusionGeometry::Neighbor::YN,
             "Neighbor on the y < 0 side.")
      .value("YP", DiffusionGeometry::Neighbor::YP,
             "Neighbor on the y > 0 side.")
      .value("ZN", DiffusionGeometry::Neighbor::ZN,
             "Neighbor on the z < 0 side.")
      .value("ZP", DiffusionGeometry::Neighbor::ZP,
             "Neighbor on the z > 0 side.");

  py::class_<DiffusionGeometry, std::shared_ptr<DiffusionGeometry>>(
      m, "DiffusionGeometry",
      "A DiffusionGeometry represents a cartesian mesh used to solve diffusion "
      "problems.")

      .def(py::init<const std::vector<DiffusionGeometry::TileFill>& /*tiles*/,
                    const std::vector<double>& /*dx*/,
                    const std::vector<std::size_t>& /*xdivs*/,
                    double /*albedo_xn*/, double /*albedo_xp*/>(),
           "Creates a 1D DiffusionGeometry.\n\n"
           "Parameters\n"
           "----------\n"
           "tiles : list of float or DiffusionData or DiffusionCrossSection\n"
           "        All tiles in the geometry.\n"
           "dx : list of float\n"
           "     Width of each tile.\n"
           "xdivs : list of int\n"
           "        Number of meshes in each tile.\n"
           "albedo_xn : float\n"
           "            Albedo at the negative x boundary.\n"
           "albedo_xp : float\n"
           "            Albedo at the positive x boundary.\n\n",
           py::arg("tiles"), py::arg("dx"), py::arg("xdivs"),
           py::arg("albedo_xn"), py::arg("albedo_xp"))

      .def(py::init<const std::vector<DiffusionGeometry::TileFill>& /*tiles*/,
                    const std::vector<double>& /*dx*/,
                    const std::vector<std::size_t>& /*xdivs*/,
                    const std::vector<double>& /*dy*/,
                    const std::vector<std::size_t>& /*ydivs*/,
                    double /*albedo_xn*/, double /*albedo_xp*/,
                    double /*albedo_yn*/, double /*albedo_yp*/>(),
           "Creates a 2D DiffusionGeometry.\n\n"
           "Parameters\n"
           "----------\n"
           "tiles : list of float or DiffusionData or DiffusionCrossSection\n"
           "        All tiles in the geometry.\n"
           "dx : list of float\n"
           "     Width of each tile along x.\n"
           "xdivs : list of int\n"
           "        Number of x meshes in each tile.\n"
           "dy : list of float\n"
           "     Width of each tile along y.\n"
           "ydivs : list of int\n"
           "        Number of y meshes in each tile.\n"
           "albedo_xn : float\n"
           "            Albedo at the negative x boundary.\n"
           "albedo_xp : float\n"
           "            Albedo at the positive x boundary.\n"
           "albedo_yn : float\n"
           "            Albedo at the negative y boundary.\n"
           "albedo_yp : float\n"
           "            Albedo at the positive y boundary.\n\n",
           py::arg("tiles"), py::arg("dx"), py::arg("xdivs"), py::arg("dy"),
           py::arg("ydivs"), py::arg("albedo_xn"), py::arg("albedo_xp"),
           py::arg("albedo_yn"), py::arg("albedo_yp"))

      .def(py::init<const std::vector<DiffusionGeometry::TileFill>& /*tiles*/,
                    const std::vector<double>& /*dx*/,
                    const std::vector<std::size_t>& /*xdivs*/,
                    const std::vector<double>& /*dy*/,
                    const std::vector<std::size_t>& /*ydivs*/,
                    const std::vector<double>& /*dz*/,
                    const std::vector<std::size_t>& /*zdivs*/,
                    double /*albedo_xn*/, double /*albedo_xp*/,
                    double /*albedo_yn*/, double /*albedo_yp*/,
                    double /*albedo_zn*/, double /*albedo_zp*/>(),
           "Creates a 3D DiffusionGeometry.\n\n"
           "Parameters\n"
           "----------\n"
           "tiles : list of float or DiffusionData or DiffusionCrossSection\n"
           "        All tiles in the geometry.\n"
           "dx : list of float\n"
           "     Width of each tile along x.\n"
           "xdivs : list of int\n"
           "        Number of x meshes in each tile.\n"
           "dy : list of float\n"
           "     Width of each tile along y.\n"
           "ydivs : list of int\n"
           "        Number of y meshes in each tile.\n"
           "dz : list of float\n"
           "     Width of each tile along z.\n"
           "zdivs : list of int\n"
           "        Number of z meshes in each tile.\n"
           "albedo_xn : float\n"
           "            Albedo at the negative x boundary.\n"
           "albedo_xp : float\n"
           "            Albedo at the positive x boundary.\n"
           "albedo_yn : float\n"
           "            Albedo at the negative y boundary.\n"
           "albedo_yp : float\n"
           "            Albedo at the positive y boundary.\n"
           "albedo_zn : float\n"
           "            Albedo at the negative z boundary.\n"
           "albedo_zp : float\n"
           "            Albedo at the positive z boundary.\n\n",
           py::arg("tiles"), py::arg("dx"), py::arg("xdivs"), py::arg("dy"),
           py::arg("ydivs"), py::arg("dz"), py::arg("zdivs"),
           py::arg("albedo_xn"), py::arg("albedo_xp"), py::arg("albedo_yn"),
           py::arg("albedo_yp"), py::arg("albedo_zn"), py::arg("albedo_zp"))

      .def(
          "neighbor", &DiffusionGeometry::neighbor,
          "Obtains the desired neighboring DiffusionGeometryTile and index for "
          "material m. If the neighbor does not exist, an albedo tile is "
          "returned "
          "and the neighbor index is None.\n\n"
          "Parameters\n"
          "----------\n"
          "m : int\n"
          "    Material index.\n"
          "n : Neighbor\n"
          "    Desired neighbor of tile m.\n\n"
          "Returns\n"
          "-------\n"
          "DiffusionGeometryTile\n"
          "                     Tile of the desired neighbor.\n"
          "int (optional)\n"
          "              The material index of the neighbor (if neighbor is a "
          "material).\n",
          py::arg("m"), py::arg("n"))

      .def("mat",
           py::overload_cast<std::size_t>(&DiffusionGeometry::mat, py::const_),
           "Obtains the :py:class:`DiffusionData` for material m.\n\n"
           "Parameters\n"
           "----------\n"
           "m : int\n"
           "    Material index.\n\n"
           "Returns\n"
           "-------\n"
           "DiffusionData\n"
           "    Cross section data and ADFs for material m.\n",
           py::arg("m"))

      .def("volume", &DiffusionGeometry::volume,
           "Obtains the volume for material m.\n\n"
           "Parameters\n"
           "----------\n"
           "m : int\n"
           "    Material index.\n\n"
           "Returns\n"
           "-------\n"
           "float\n"
           "     Volume of material m.\n",
           py::arg("m"))

      .def(
          "geom_indx",
          [](const DiffusionGeometry& geom, std::size_t m) {
            auto inds = geom.geom_indx(m);
            return std::vector<std::size_t>(inds.begin(), inds.end());
          },
          "The geometry indices for material index m.\n\n"
          "Parameters\n"
          "----------\n"
          "m : int\n"
          "    Material index.\n\n"
          "Returns\n"
          "-------\n"
          "list of int\n"
          "           Geometry indices of material index m.\n",
          py::arg("m"))

      .def("dx", &DiffusionGeometry::dx,
           "Width in the x direction of the i mesh along the x axis.\n\n"
           "Parameters\n"
           "----------\n"
           "i : int\n"
           "    Mesh index along x-axis.\n\n"
           "Returns\n"
           "float\n"
           "     Width of mesh along x-axis.\n",
           py::arg("i"))

      .def("dy", &DiffusionGeometry::dy,
           "Width in the y direction of the j mesh along the y axis.\n\n"
           "Parameters\n"
           "----------\n"
           "j : int\n"
           "    Mesh index along y-axis.\n\n"
           "Returns\n"
           "float\n"
           "     Width of mesh along y-axis.\n",
           py::arg("j"))

      .def("dz", &DiffusionGeometry::dz,
           "Width in the z direction of the k mesh along the z axis.\n\n"
           "Parameters\n"
           "----------\n"
           "k : int\n"
           "    Mesh index along z-axis.\n\n"
           "Returns\n"
           "float\n"
           "     Width of mesh along z-axis.\n",
           py::arg("k"))

      .def_property_readonly("ngroups", &DiffusionGeometry::ngroups,
                             "Number of energy groups.")

      .def_property_readonly("ndims", &DiffusionGeometry::ndims,
                             "Number of dimensions (1, 2, or 3).")

      .def_property_readonly("nmats", &DiffusionGeometry::nmats,
                             "Total number of material tiles.")

      .def_property_readonly("nx", &DiffusionGeometry::nx,
                             "Number of tiles along the x-axis.")

      .def_property_readonly("ny", &DiffusionGeometry::ny,
                             "Number of tiles along the y-axis.")

      .def_property_readonly("nz", &DiffusionGeometry::nz,
                             "Number of tiles along the z-axis.")

      .def(py::pickle([](const DiffusionGeometry& d) { return d.to_tuple(); },
                      [](py::tuple t) { return DiffusionGeometry(t); }));
}

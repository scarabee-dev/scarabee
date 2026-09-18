#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <xtensor-python/pyarray.hpp>

#include <cereal/archives/portable_binary.hpp>

#include <diffusion/diffusion_geometry.hpp>
#include <utils/logging.hpp>
#include <utils/scarabee_exception.hpp>

namespace py = pybind11;

using namespace scarabee;

struct DiffusionGeometryTilePickler {
  static DiffusionGeometry::Tile from_state(py::tuple t) {
    return DiffusionGeometry::Tile{t[0].cast<std::optional<double>>(),
                                   t[1].cast<std::shared_ptr<DiffusionData>>()};
  }

  static py::tuple to_state(const DiffusionGeometry::Tile& t) {
    return py::make_tuple(t.albedo, t.xs);
  }
};

struct DiffusionGeometryPickler {
  static std::shared_ptr<DiffusionGeometry> from_state(py::tuple t) {
    std::shared_ptr<DiffusionGeometry> dg(new DiffusionGeometry);

    // Must rebuild tiles_ array
    const std::size_t tsx = t[0].cast<std::size_t>();
    const std::size_t tsy = t[1].cast<std::size_t>();
    const std::size_t tsz = t[2].cast<std::size_t>();
    std::vector<DiffusionGeometry::Tile> flat_tiles =
        t[3].cast<std::vector<DiffusionGeometry::Tile>>();
    dg->xn_ = t[4].cast<DiffusionGeometry::Tile>();
    dg->xp_ = t[5].cast<DiffusionGeometry::Tile>();
    dg->yn_ = t[6].cast<DiffusionGeometry::Tile>();
    dg->yp_ = t[7].cast<DiffusionGeometry::Tile>();
    dg->zn_ = t[8].cast<DiffusionGeometry::Tile>();
    dg->zp_ = t[9].cast<DiffusionGeometry::Tile>();
    dg->tile_dx_ = t[10].cast<std::vector<double>>();
    dg->x_divs_per_tile_ = t[11].cast<std::vector<std::size_t>>();
    dg->tile_dy_ = t[12].cast<std::vector<double>>();
    dg->y_divs_per_tile_ = t[13].cast<std::vector<std::size_t>>();
    dg->tile_dz_ = t[14].cast<std::vector<double>>();
    dg->z_divs_per_tile_ = t[15].cast<std::vector<std::size_t>>();
    dg->x_bounds_ = t[16].cast<std::vector<double>>();
    dg->y_bounds_ = t[17].cast<std::vector<double>>();
    dg->z_bounds_ = t[18].cast<std::vector<double>>();
    dg->nmats_ = t[19].cast<std::size_t>();
    dg->mat_indx_to_flat_geom_indx_ = t[20].cast<std::vector<std::size_t>>();
    dg->nx_ = t[21].cast<std::size_t>();
    dg->ny_ = t[22].cast<std::size_t>();
    dg->nz_ = t[23].cast<std::size_t>();

    if (tsz > 0 && tsy > 0)
      dg->tiles_.resize({tsx, tsy, tsz});
    else if (tsy > 0)
      dg->tiles_.resize({tsx, tsy});
    else
      dg->tiles_.resize({tsx});

    if (dg->tiles_.size() != flat_tiles.size()) {
      std::stringstream mssg;
      mssg << "Could not reconstruct tiles_ array from provided tuple.";
      spdlog::error(mssg.str());
      throw ScarabeeException(mssg.str());
    }

    for (std::size_t j = 0; j < flat_tiles.size(); j++)
      dg->tiles_.flat(j) = flat_tiles[j];

    // Rebuild shape
    const std::size_t ndims = t[24].cast<std::size_t>();
    if (ndims >= 1) dg->geom_shape_.push_back(dg->nx_);
    if (ndims >= 2) dg->geom_shape_.push_back(dg->ny_);
    if (ndims >= 3) dg->geom_shape_.push_back(dg->nz_);

    return dg;
  }

  static py::tuple to_state(const std::shared_ptr<DiffusionGeometry>& dg) {
    const std::size_t ndims = dg->geom_shape_.size();
    std::vector<DiffusionGeometry::Tile> flat_tiles;
    flat_tiles.reserve(dg->tiles_.size());
    for (std::size_t i = 0; i < dg->tiles_.size(); i++)
      flat_tiles.push_back(dg->tiles_.flat(i));

    return py::make_tuple(
        dg->tile_dx_.size(), dg->tile_dy_.size(), dg->tile_dz_.size(),
        flat_tiles, dg->xn_, dg->xp_, dg->yn_, dg->yp_, dg->zn_, dg->zp_,
        dg->tile_dx_, dg->x_divs_per_tile_, dg->tile_dy_, dg->y_divs_per_tile_,
        dg->tile_dz_, dg->z_divs_per_tile_, dg->x_bounds_, dg->y_bounds_,
        dg->z_bounds_, dg->nmats_, dg->mat_indx_to_flat_geom_indx_, dg->nx_,
        dg->ny_, dg->nz_, ndims);
  }
};

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

      .def(py::pickle(&DiffusionGeometryTilePickler::to_state,
                      &DiffusionGeometryTilePickler::from_state));

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

      .def(py::pickle(&DiffusionGeometryPickler::to_state,
                      &DiffusionGeometryPickler::from_state));
}

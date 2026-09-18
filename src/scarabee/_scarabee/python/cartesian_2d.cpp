#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <moc/cartesian_2d.hpp>

#include <memory>

namespace py = pybind11;

using namespace scarabee;

struct Cartesian2DPickler {
  static std::shared_ptr<Cartesian2D> from_state(py::tuple t) {
    Cartesian2D::Tuple ct = t[0].cast<Cartesian2D::Tuple>();
    return std::make_shared<Cartesian2D>(ct);
  }

  static py::tuple to_state(const std::shared_ptr<Cartesian2D>& c) {
    return py::make_tuple(c->to_tuple());
  }
};

void init_Cartesian2D(py::module& m) {
  // Tile
  py::class_<Cartesian2D::Tile>(
      m, "Tile",
      "A Tile represents an elemnent of a :py:class:`Cartesian2D` geometry, "
      "which can contain either a :py:class:`Cell` or another "
      ":py:class:`Cartensian2D` geometry.")
      .def_readwrite("c2d", &Cartesian2D::Tile::c2d,
                     "The optional Cartesian2D which fills the tile.")
      .def_readwrite("cell", &Cartesian2D::Tile::cell,
                     "The optional Cell which fills the tile.")
      .def_property_readonly(
          "valid", &Cartesian2D::Tile::valid,
          "True if the tile is completely filled with Cells, False otherwise.")
      .def(py::pickle(
          [](const Cartesian2D::Tile& t) {
            return py::make_tuple(t.to_tuple());
          },
          [](py::tuple t) {
            return Cartesian2D::Tile::from_tuple(
                t[0].cast<Cartesian2D::Tile::Tuple>());
          }));

  // TileIndex
  py::class_<Cartesian2D::TileIndex>(
      m, "TileIndex",
      "A TileIndex contains the x and y coordinates of a Tile within a "
      "Cartesian2D geometry.")
      .def_readwrite("i", &Cartesian2D::TileIndex::i, "Index along x axis.")
      .def_readwrite("j", &Cartesian2D::TileIndex::j, "Index along y axis.");

  // Cartesian2D
  py::class_<Cartesian2D, std::shared_ptr<Cartesian2D>>(m, "Cartesian2D")
      .def(py::init<const std::vector<double>& /*dx*/,
                    const std::vector<double>& /*dy*/>(),
           "A Cartesian2D represents a cartesian geometry, which is divided "
           "into "
           ":py:class:`Tile` instances, each of which can contain either other "
           "Cartesian2D geometries, or a :py:class:`Cell` which terminates the "
           "geometry tree with materials.\n\n"
           "Parameters\n"
           "----------\n"
           "dx : list of float\n"
           "     List of all x widths.\n"
           "dy : list of float\n"
           "     List of all y heights.\n",
           py::arg("dx"), py::arg("dy"))

      .def_property_readonly("nx", &Cartesian2D::nx,
                             "Number of cells in x direction.")

      .def_property_readonly("ny", &Cartesian2D::ny,
                             "Number of cells in y direction.")

      .def_property_readonly("dx", &Cartesian2D::dx, "Width of geometry in x.")

      .def_property_readonly("dy", &Cartesian2D::dy, "Width of geometry in y.")

      .def_property_readonly("x_min", &Cartesian2D::x_min,
                             "Minimum x coordinate.")

      .def_property_readonly("x_max", &Cartesian2D::x_max,
                             "Maximum x coordinate.")

      .def_property_readonly("y_min", &Cartesian2D::y_min,
                             "Minimum y coordinate.")

      .def_property_readonly("y_max", &Cartesian2D::y_max,
                             "Maximum y coordinate.")

      .def_property_readonly("num_fsrs", &Cartesian2D::num_fsrs,
                             "Number of flat source regions.")

      .def_property_readonly("tiles_valid", &Cartesian2D::tiles_valid,
                             "Returns True if all tiles are populated with a "
                             "Cell, False otherwise.")

      .def("get_tile_index", &Cartesian2D::get_tile_index,
           "Returns the TileIndex for given position and direction.\n\n"
           "Parameters\n"
           "----------\n"
           "r : Vector\n"
           "    Position of desired tile.\n"
           "u : Direction\n"
           "    Direction use in the case of geometry ambiguity.",
           py::arg("r"), py::arg("u"))

      .def("get_xs", &Cartesian2D::get_xs,
           "Returns the CrossSection for given position and direction.\n\n"
           "Parameters\n"
           "----------\n"
           "r : Vector\n"
           "    Position of desired material.\n"
           "u : Direction\n"
           "    Direction use in the case of geometry ambiguity.",
           py::arg("r"), py::arg("u"))

      .def("tile", &Cartesian2D::tile,
           "Returns the Tile for the given TileIndex.\n\n"
           "Parameters\n"
           "----------\n"
           "ti : TileIndex\n"
           "     Index of desired Tile.",
           py::arg("ti"))

      .def("set_tiles", &Cartesian2D::set_tiles,
           "Sets all tiles in the geometry.\n\n"
           "Parameters\n"
           "----------\n"
           "fills : list of Cartesian2D or Cell\n"
           "        Fills for all tiles.",
           py::arg("fills"))

      .def(py::pickle(&Cartesian2DPickler::to_state,
                      &Cartesian2DPickler::from_state));
}

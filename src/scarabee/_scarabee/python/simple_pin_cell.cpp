#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <moc/simple_pin_cell.hpp>

namespace py = pybind11;

using namespace scarabee;

void init_SimplePinCell(py::module& m) {
  py::class_<SimplePinCell, Cell, std::shared_ptr<SimplePinCell>>(
      m, "SimplePinCell")
      .def(py::init<const std::vector<double>& /*rads*/,
                    const std::vector<std::shared_ptr<CrossSection>>& /*mats*/,
                    double /*dx*/, double /*dy*/, PinCellType /*pin_type*/>(),
           "An annular pin centered in a rectangular cell with no angular "
           "segments. Must provide one more CrossSection than radii, as that "
           "material will fill the cell out to the boundary.\n\n"
           "Parameters\n"
           "----------\n"
           "radii : list of float\n"
           "        Radius of each annular region.\n"
           "mats : list of CrossSection\n"
           "       Material cross sections for each region.\n"
           "dx : float\n"
           "     Width of cell along x.\n"
           "dy : float\n"
           "     Width of cell along y.\n"
           "pin_type : PinCellType\n"
           "     Wether is a full or half or quarter pin cell. Default is "
           "Full.\n",
           py::arg("radii"), py::arg("mats"), py::arg("dx"), py::arg("dy"),
           py::arg("pin_type") = PinCellType::Full)

      .def(py::pickle(
          [](std::shared_ptr<SimplePinCell> c) { return c->to_tuple(); },
          [](py::tuple t) { return std::make_shared<SimplePinCell>(t); }));
}

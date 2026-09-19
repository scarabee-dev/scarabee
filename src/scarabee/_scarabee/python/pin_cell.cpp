#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <moc/pin_cell.hpp>

namespace py = pybind11;

using namespace scarabee;

struct PinCellPickler {
  static std::shared_ptr<PinCell> from_state(py::tuple t) {
    PinCell::Tuple pct = t[0].cast<PinCell::Tuple>();
    return std::make_shared<PinCell>(pct);
  }

  static py::tuple to_state(const std::shared_ptr<PinCell>& pc) {
    return py::make_tuple(pc->to_tuple());
  }
};

void init_PinCell(py::module& m) {
  py::class_<PinCell, Cell, std::shared_ptr<PinCell>>(m, "PinCell")
      .def(py::init<const std::vector<double>& /*rads*/,
                    const std::vector<std::shared_ptr<CrossSection>>&
                    /*mats*/,
                    double /*dx*/, double /*dy*/, PinCellType
                    /*pin_type*/>(),
           "An annular pin centered in a rectangular cell, with 8 angular "
           "segments. Must provide one more CrossSection than radius, as that "
           "material will fill the cell out to the boundary.\n\n"
           "Parameters\n"
           "----------\n"
           "mat : CrossSection\n"
           "      The material cross sections for the cell.\n"
           "dx : float\n"
           "     Width of the cell along x.\n"
           "dy : float\n"
           "     Width of the cell along y.\n"
           "pin_type : PinCellType\n"
           "     Wether is a full or half or quarter pin cell. Default is "
           "Full.\n",
           py::arg("radii"), py::arg("mats"), py::arg("dx"), py::arg("dy"),
           py::arg("pin_type") = PinCellType::Full)

      .def(py::pickle(&PinCellPickler::to_state, &PinCellPickler::from_state));
}

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <moc/empty_cell.hpp>

namespace py = pybind11;

using namespace scarabee;

#include <memory>

struct EmptyCellPickler {
  static std::shared_ptr<EmptyCell> from_state(py::tuple t) {
    EmptyCell::Tuple ect = t[0].cast<EmptyCell::Tuple>();
    return std::make_shared<EmptyCell>(ect);
  }

  static py::tuple to_state(const std::shared_ptr<EmptyCell>& ec) {
    return py::make_tuple(ec->to_tuple());
  }
};

void init_EmptyCell(py::module& m) {
  py::class_<EmptyCell, Cell, std::shared_ptr<EmptyCell>>(m, "EmptyCell")
      .def(py::init<const std::shared_ptr<CrossSection>& /*mat*/, double /*dx*/,
                    double /*dy*/>(),
           "An empty cell with only one flat source region.\n\n"
           "Parameters\n"
           "----------\n"
           "mat : CrossSection\n"
           "      The material cross sections for the cell.\n"
           "dx : float\n"
           "     Width of the cell along x.\n"
           "dy : float\n"
           "     Width of the cell along y.\n",
           py::arg("mat"), py::arg("dx"), py::arg("dy"))

      .def(py::pickle(&EmptyCellPickler::to_state,
                      &EmptyCellPickler::from_state));
}

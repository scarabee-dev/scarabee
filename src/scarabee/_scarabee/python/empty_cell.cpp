#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <moc/empty_cell.hpp>

namespace py = pybind11;

using namespace scarabee;

#include <memory>

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

      .def(py::pickle(
          [](std::shared_ptr<EmptyCell> c) { return c->to_tuple(); },
          [](py::tuple t) { return std::make_shared<EmptyCell>(t); }));
}

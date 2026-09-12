#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <moc/simple_bwr_corner_pin_cell.hpp>

namespace py = pybind11;

using namespace scarabee;

void init_SimpleBWRCornerPinCell(py::module& m) {
  py::class_<SimpleBWRCornerPinCell, Cell,
             std::shared_ptr<SimpleBWRCornerPinCell>>(m,
                                                      "SimpleBWRCornerPinCell")
      .def(
          py::init<
              const std::vector<double>& /*pin_rads*/,
              const std::vector<std::shared_ptr<CrossSection>>& /*pin_mats*/,
              double /*inner_gap*/, std::shared_ptr<CrossSection> /*inner_mod*/,
              double /*box_width*/, double /*rc*/,
              std::shared_ptr<CrossSection> /*box_mat*/,
              std::shared_ptr<CrossSection> /*outer_mod*/, double /*dx*/,
              double /*dy*/, BWRCornerType /*corner_type*/>(),
          "A pin cell at the corner of a BWR fuel bundle channel. Models the "
          "channel box, moderator outside the bundle, and the moderator inside "
          "the channel. There can be a gap between the channel box and the "
          "normal pin cell boundary. The corners can also be rounded.\n\n"
          "Parameters\n"
          "----------\n"
          "pin_radii : list of float\n"
          "    Radius of each annular region for the pin.\n"
          "pin_mats : list of CrossSection\n"
          "    Cross sections for each annular region of the pin.\n"
          "inner_gap : float\n"
          "    Width of the moderator gap inside the channel.\n"
          "inner_mod : CrossSection\n"
          "    Cross sections for the moderator inside the channel.\n"
          "box_width : float\n"
          "    Width of the channel box.\n"
          "rc : float\n"
          "    Radius of curvature for the inside of the channel box corner.\n"
          "box_mat : CrossSection\n"
          "    Cross sections for the channel box.\n"
          "outer_mod : CrossSection\n"
          "    Cross sections for the moderator outside the channel.\n"
          "dx : float\n"
          "    Width of cell along x.\n"
          "dy : float\n"
          "    Width of cell along y.\n"
          "corner_type : BWRCornerType\n"
          "    Indicates which corner of the BWR channel is being modeled.\n",
          py::arg("pin_radii"), py::arg("pin_mats"), py::arg("inner_gap"),
          py::arg("inner_mod"), py::arg("box_width"), py::arg("rc"),
          py::arg("box_mat"), py::arg("outer_mod"), py::arg("dx"),
          py::arg("dy"), py::arg("corner_type"))

      .def(py::pickle(
          [](std::shared_ptr<SimpleBWRCornerPinCell> c) {
            return c->to_tuple();
          },
          [](py::tuple t) {
            return std::make_shared<SimpleBWRCornerPinCell>(t);
          }));
}

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <xtensor-python/pytensor.hpp>

#include <cereal/types/memory.hpp>
#include <cereal/archives/portable_binary.hpp>

#include <diffusion/form_factors.hpp>

#include <sstream>

namespace py = pybind11;

using namespace scarabee;

struct FormFactorsPickler {
  static std::shared_ptr<FormFactors> from_state(py::tuple t) {
    py::bytes bytes = t[0].cast<py::bytes>();
    std::istringstream bits_stream(bytes,
                                   std::ios_base::binary | std::ios_base::in);
    std::shared_ptr<FormFactors> p;
    {
      cereal::PortableBinaryInputArchive ar(bits_stream);
      ar(p);
    }
    return p;
  }

  static py::tuple to_state(const std::shared_ptr<FormFactors>& ff) {
    std::ostringstream bits_stream(std::ios_base::binary | std::ios_base::out);
    {
      cereal::PortableBinaryOutputArchive ar(bits_stream);
      ar(ff);
    }
    return py::make_tuple(py::bytes(bits_stream.str()));
  }
};

void init_FormFactors(py::module& m) {
  py::class_<FormFactors, std::shared_ptr<FormFactors>>(
      m, "FormFactors",
      "A FormFactors object contains the form factors used to compute the pin "
      "powers after a full core diffusion simulation. It is essentially a 2D "
      "array where the average value of the non-zero entries is 1. The array "
      "should be provided as \"what you see is what you get\", where the first "
      "row of the array would have a Python index of 0, but actually "
      "represents the largest possible y coordinate.")

      .def(py::init<const xt::xtensor<double, 2>& /*ff*/,
                    const xt::xtensor<double, 1>& /*xw*/,
                    const xt::xtensor<double, 1>& /*yw*/>(),
           "Creates a FormFactors object.\n\n"
           "Parameters\n"
           "----------\n"
           "ff : ndarray\n"
           "    2D array containing the form factors.\n"
           "xw : ndarray\n"
           "    1D array with the widths of each tile along x.\n"
           "yw : ndarray\n"
           "    1D array with the widths of each tile along y.\n\n",
           py::arg("ff"), py::arg("xw"), py::arg("yw"))

      .def(py::init<const FormFactors& /*q1*/, const FormFactors& /*q2*/,
                    const FormFactors& /*q3*/, const FormFactors& /*q4*/,
                    bool /*half_pins*/>(),
           "Builds a FormFactors object from four other FormFactors, which "
           "represent the four quadrants.\n\n"
           "Parameters\n"
           "----------\n"
           "q1 : FormFactors\n"
           "    Form factors for quadrant I.\n"
           "q2 : FormFactors\n"
           "    Form factors for quadrant II.\n"
           "q3 : FormFactors\n"
           "    Form factors for quadrant III.\n"
           "q4 : FormFactors\n"
           "    Form factors for quadrant IV.\n"
           "half_pins : bool\n"
           "    True if the pins along the x and y axis planes are cut in half"
           "    in the four quadrants, and False otherwise. Default value is "
           "True.\n\n",
           py::arg("q1"), py::arg("q2"), py::arg("q3"), py::arg("q4"),
           py::arg("half_pins") = true)

      .def_property_readonly("form_factors", &FormFactors::form_factors,
                             "2D array of form factors.")

      .def_property_readonly("x_widths", &FormFactors::x_widths,
                             "1D array of the widths of tiles along x.")

      .def_property_readonly("y_widths", &FormFactors::y_widths,
                             "1D array of the widths of tiles along y.")

      .def_property_readonly("x_width", &FormFactors::x_width,
                             "Total width along x.")

      .def_property_readonly("y_width", &FormFactors::y_width,
                             "Total width along y.")

      .def("__call__", &FormFactors::operator(),
           "Computes the form factor for the position (x,y), which are given "
           "assuming that the lower-left coorner of the array begins at "
           "(0,0).\n\n"
           "Parameters\n"
           "----------\n"
           "x : float\n"
           "    Relative x position in the assembly.\n"
           "y : float\n"
           "    Relative y position in the assembly.\n\n"
           "Returns\n"
           "-------\n"
           "float\n"
           "    Computed value of the form factor.\n",
           py::arg("x"), py::arg("y"))

      .def("rotate_clockwise", &FormFactors::rotate_clockwise,
           "Rotates the form factors corresponding to a 90 degree "
           "rotation of the assembly in the clockwise direction.",
           py::return_value_policy::reference_internal)

      .def("rotate_counterclockwise", &FormFactors::rotate_counterclockwise,
           "Rotates the form factors corresponding to a 90 degree "
           "rotation of the assembly in the counter clockwise direction.",
           py::return_value_policy::reference_internal)

      .def("reflect_across_x_axis", &FormFactors::reflect_across_x_axis,
           "Performs a reflection of the form factors across the x axis.",
           py::return_value_policy::reference_internal)

      .def("reflect_across_y_axis", &FormFactors::reflect_across_y_axis,
           "Performs a reflection of the form factors across the y axis",
           py::return_value_policy::reference_internal)

      .def("cut_half_pos_x", &FormFactors::cut_half_pos_x,
           "Creates a new FromFactor instance from the +x half of this "
           "instance.\n\n"
           "Returns\n"
           "-------\n"
           "FormFactors\n"
           "    Form factors on the +x half.\n")

      .def("cut_half_neg_x", &FormFactors::cut_half_neg_x,
           "Creates a new FromFactor instance from the -x half of this "
           "instance.\n\n"
           "Returns\n"
           "-------\n"
           "FormFactors\n"
           "    Form factors on the -x half.\n")

      .def("cut_half_pos_y", &FormFactors::cut_half_pos_y,
           "Creates a new FromFactor instance from the +y half of this "
           "instance.\n\n"
           "Returns\n"
           "-------\n"
           "FormFactors\n"
           "    Form factors on the +y half.\n")

      .def("cut_half_neg_y", &FormFactors::cut_half_neg_y,
           "Creates a new FromFactor instance from the -y half of this "
           "instance.\n\n"
           "Returns\n"
           "-------\n"
           "FormFactors\n"
           "    Form factors on the -y half.\n")

      .def("cut_quad_I", &FormFactors::cut_quad_I,
           "Creates a new FromFactor instance from the I quadrant of this "
           "instance.\n\n"
           "Returns\n"
           "-------\n"
           "FormFactors\n"
           "    Form factors on the I quadrant.\n")

      .def("cut_quad_II", &FormFactors::cut_quad_II,
           "Creates a new FromFactor instance from the II quadrant of this "
           "instance.\n\n"
           "Returns\n"
           "-------\n"
           "FormFactors\n"
           "    Form factors on the II quadrant.\n")

      .def("cut_quad_III", &FormFactors::cut_quad_III,
           "Creates a new FromFactor instance from the III quadrant of this "
           "instance.\n\n"
           "Returns\n"
           "-------\n"
           "FormFactors\n"
           "    Form factors on the III quadrant.\n")

      .def("cut_quad_IV", &FormFactors::cut_quad_IV,
           "Creates a new FromFactor instance from the IV quadrant of this "
           "instance.\n\n"
           "Returns\n"
           "-------\n"
           "FormFactors\n"
           "    Form factors on the IV quadrant.\n")

      .def(py::pickle(&FormFactorsPickler::to_state,
                      &FormFactorsPickler::from_state));
}

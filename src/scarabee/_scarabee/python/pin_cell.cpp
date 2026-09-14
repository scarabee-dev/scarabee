#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <moc/pin_cell.hpp>

namespace py = pybind11;

using namespace scarabee;

void init_PinCell(py::module& m) {
  /*
  using V1 = std::vector<double>;
  using V2 = std::vector<std::shared_ptr<scarabee::CrossSection>>;

  static_assert(std::is_same_v<pybind11::detail::make_caster<V2>,
                               pybind11::detail::type_caster<V2>>);

  static_assert(
      std::is_base_of_v<pybind11::detail::list_caster<
                            V2, std::shared_ptr<scarabee::CrossSection>>,
                        pybind11::detail::type_caster<V2>>);

  std::cout << "sizeof(V2 caster) = "
            << sizeof(pybind11::detail::type_caster<V2>) << "\n";

  std::cout << "sizeof(V2 list caster) = "
            << sizeof(pybind11::detail::list_caster<
                      V2, std::shared_ptr<scarabee::CrossSection>>)
            << "\n";

  std::cerr << "\n=== MANUAL CASTER TEST ===\n";

  {
    pybind11::detail::type_caster<V1> c1;
  }

  {
    pybind11::detail::type_caster<V2> c2;
  }
  */

  py::class_<PinCell, Cell, std::shared_ptr<PinCell>>(m, "PinCell")
      .def(  // py::init<const std::vector<double>& /*rads*/,
             //          const std::vector<std::shared_ptr<CrossSection>>&
             //          /*mats*/, double /*dx*/, double /*dy*/, PinCellType
             //          /*pin_type*/>(),
          py::init([](std::vector<double> rads,
                      std::vector<std::shared_ptr<CrossSection>> mats,
                      double dx, double dy, PinCellType pin_type) {
            std::cout << "\n>>> Pre PinCell construction <<<\n";
            auto out = std::make_shared<PinCell>(rads, mats, dx, dy, pin_type);
            std::cout << "\n>>> Post PinCell construction <<<\n";
            return out;
          }),
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

      .def(
          py::pickle([](std::shared_ptr<PinCell> c) { return c->to_tuple(); },
                     [](py::tuple t) { return std::make_shared<PinCell>(t); }));
}

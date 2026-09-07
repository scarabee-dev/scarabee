#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <xtensor-python/pytensor.hpp>

#include <data/depletion_matrix.hpp>

#include <cereal/types/memory.hpp>
#include <cereal/archives/portable_binary.hpp>

#include <memory>
#include <sstream>

namespace py = pybind11;

using namespace scarabee;

void init_DepletionMatrix(py::module& m) {
  py::class_<DepletionMatrix, std::shared_ptr<DepletionMatrix>>(
      m, "DepletionMatrix",
      "Represents a depletion matrix containing transfer terms due to "
      "radioactive decay and transmutation reactions.")

      .def(
          py::init<const std::vector<std::string>& /*nuclides*/>(),
          "Initializes an empty depletion matrix for the provided nuclides.\n\n"
          "Parameters\n"
          "----------\n"
          "nuclides : list of string\n"
          "    All possible nuclides in the material.\n",
          py::arg("nuclides"))

      .def_property_readonly("size", &DepletionMatrix::size,
                             "Number of nuclides in the depletion matrix.")

      .def_property_readonly(
          "nuclides", &DepletionMatrix::nuclides,
          "List of nuclides considered in the depletion matrix.")

      .def("has_nuclide", &DepletionMatrix::has_nuclide,
           "Returns True if the depletion matrix contains the specified "
           "nuclide, and False otherwise.\n\n"
           "Parameters\n"
           "----------\n"
           "nuclide : string\n"
           "    Name of the nuclide.\n\n"
           "Returns\n"
           "-------\n"
           "bool\n"
           "    True if the nuclide is present, False otherwise.\n",
           py::arg("nuclide"))

      .def("get_nuclide_index", &DepletionMatrix::get_nuclide_index,
           "Returns the index of the specified nuclide, if contained in the "
           "depletion matrix.\n\n"
           "Parameters\n"
           "----------\n"
           "nuclide : string\n"
           "    Name of the nuclide.\n\n"
           "Returns\n"
           "-------\n"
           "int\n"
           "    Index of the nuclide.\n",
           py::arg("nuclide"))

      .def("zero", &DepletionMatrix::zero,
           "Sets all entries in the depletion matrix to zero.")

      .def_property_readonly(
          "is_compressed", &DepletionMatrix::is_compressed,
          "True if the matrix is in compressed format, False otherwise.")

      .def("compress", &DepletionMatrix::compress,
           "Puts the matrix into compressed format.")

      .def(
          "exponential_product",
          [](const DepletionMatrix& m, xt::pytensor<double, 1>& N,
             bool cram48 = false) {
            std::span<double> Nspn(N.data(), N.size());
            m.exponential_product(Nspn, cram48);
          },
          py::call_guard<py::gil_scoped_release>(),
          "Computes the matrix product of the input vector N with the "
          "exponential of the depletion matrix. The array N is modified in "
          "place to contain the result of this product.\n\n"
          ".. math:: N' = \\exp{A} N.\n\n"
          "This product is approximated using a Chebyshev Rational "
          "Approximation (CRAM). By default, a 16th order CRAM is used, but a "
          "48th order approximation is available.\n\n"
          "Parameters\n"
          "----------\n"
          "N : ndarray\n"
          "    1D array of the number densities of the nuclides in the "
          "    depletion matrix (in the same order). This array is modified"
          "    in place, and will contain the result of the product after the"
          "    function is complete.\n"
          "cram48 : bool\n"
          "    If True, a 48th order CRAM is used. If False, a 16th order CRAM"
          "    is employed. Default value is False.\n",
          py::arg("N"), py::arg("cram48") = false)

      .def("__getitem__",
           [](const DepletionMatrix& m,
              std::pair<std::size_t, std::size_t> indx) {
             return m.value(indx.first, indx.second);
           })

      .def("__setitem__",
           [](DepletionMatrix& m, std::pair<std::size_t, std::size_t> indx,
              double v) { m.ref(indx.first, indx.second) = v; })

      .def("__imul__", &DepletionMatrix::operator*=)
      .def("__itruediv__", &DepletionMatrix::operator/=)
      .def("__iadd__", &DepletionMatrix::operator+=)
      .def("__isub__", &DepletionMatrix::operator-=)
      .def("__add__", &DepletionMatrix::operator+)
      .def("__sub__", &DepletionMatrix::operator-)
      .def("__mul__", &DepletionMatrix::operator*)
      .def("__rmul__", [](const DepletionMatrix& M, double c) { return M * c; })
      .def("__truediv__", &DepletionMatrix::operator/)
      .def(py::pickle(
          [](const std::shared_ptr<DepletionMatrix>& p) {
            std::ostringstream bits_stream(std::ios_base::binary |
                                           std::ios_base::out);
            {
              cereal::PortableBinaryOutputArchive ar(bits_stream);
              ar(p);
            }
            return py::bytes(bits_stream.str());
          },
          [](py::bytes bites) {
            std::istringstream bits_stream(
                bites, std::ios_base::binary | std::ios_base::in);
            std::shared_ptr<DepletionMatrix> p;
            {
              cereal::PortableBinaryInputArchive ar(bits_stream);
              ar(p);
            }
            return p;
          }));

  // Module function to build a depletion matrix.
  m.def(
      "build_depletion_matrix",
      [](std::shared_ptr<DepletionChain> chain, std::shared_ptr<Material> mat,
         const xt::pytensor<double, 1>& flux, std::shared_ptr<NDLibrary> ndl) {
        std::span<const double> flux_spn(flux.data(), flux.size());
        return build_depletion_matrix(chain, mat, flux_spn, ndl);
      },
      py::call_guard<py::gil_scoped_release>(),
      "Builds a depletion matrix for a given material and flux spectrum.\n\n"
      "Parameters\n"
      "----------\n"
      "chain : DepletionChain\n"
      "    Depletion chain to used when building the matrix.\n"
      "mat : Material\n"
      "    Material for which the chain is being built. Must have loaded "
      "    depletion cross section data.\n"
      "flux : ndarray\n"
      "    1D Numpy array containing the flux spectrum.\n"
      "ndl : NDLibrary\n"
      "    Nuclear data library, needed for the group structure when "
      "    calculating the average fission energy.\n\n"
      "Returns\n"
      "-------\n"
      "DepletionMatrix\n"
      "    Depletion matrix for the provided material and flux spectrum. The "
      "    matrix has not been multiplied by any time step at this point.\n\n",
      py::arg("chain"), py::arg("mat"), py::arg("flux"), py::arg("ndl"));

  m.def(
      "build_depletion_matrix",
      [](std::shared_ptr<DepletionChain> chain,
         std::vector<DepletionReactionRates>& nuc_rrs) {
        return build_depletion_matrix(chain, nuc_rrs);
      },
      py::call_guard<py::gil_scoped_release>(),
      "Builds a depletion matrix for a given set of reaction rates.\n\n"
      "Parameters\n"
      "----------\n"
      "chain : DepletionChain\n"
      "    Depletion chain to used when building the matrix.\n"
      "nuc_rrs : list of DepletionReactionRates\n"
      "    Integrated reaction rates for all nuclides.\n\n"
      "Returns\n"
      "-------\n"
      "DepletionMatrix\n"
      "    Depletion matrix for the provided reaction rates. The matrix\n"
      "    has not been multiplied by any time step at this point.\n\n",
      py::arg("chain"), py::arg("nuc_rrs"));
}

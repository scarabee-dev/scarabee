#ifndef SCARABEE_MICRO_CROSS_SECTIONS_H
#define SCARABEE_MICRO_CROSS_SECTIONS_H

#include <data/xs1d.hpp>
#include <data/xs2d.hpp>
#include <utils/serialization.hpp>

#include <xtensor/containers/xtensor.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

#include <cereal/cereal.hpp>
#include <cereal/types/optional.hpp>
#include <cereal/types/string.hpp>
#include <cereal/archives/portable_binary.hpp>

#include <optional>
#include <sstream>

namespace scarabee {

struct MicroDepletionXS {
  std::optional<XS1D> n_fission{std::nullopt};
  std::optional<XS1D> n_gamma{std::nullopt};
  std::optional<XS1D> n_2n{std::nullopt};
  std::optional<XS1D> n_3n{std::nullopt};
  std::optional<XS1D> n_alpha{std::nullopt};
  std::optional<XS1D> n_p{std::nullopt};

  static MicroDepletionXS from_tuple(py::tuple t) {
    MicroDepletionXS out;
    py::bytes bytes = t[0].cast<py::bytes>();
    std::istringstream bits_stream(bytes,
                                   std::ios_base::binary | std::ios_base::in);
    {
      cereal::PortableBinaryInputArchive ar(bits_stream);
      ar(out);
    }
    return out;
  }

  py::tuple to_tuple() const {
    std::ostringstream bits_stream(std::ios_base::binary | std::ios_base::out);
    {
      cereal::PortableBinaryOutputArchive ar(bits_stream);
      ar(*this);
    }
    py::bytes bytes(bits_stream.str());
    return py::make_tuple(bytes);
  }

  template <class Archive>
  void serialize(Archive& arc) {
    arc(CEREAL_NVP(n_fission), CEREAL_NVP(n_gamma), CEREAL_NVP(n_2n),
        CEREAL_NVP(n_3n), CEREAL_NVP(n_alpha), CEREAL_NVP(n_p));
  }
};

struct MicroNuclideXS {
  XS1D Et;
  XS1D Dtr;
  XS2D Es;
  XS1D Ea;
  XS1D Ef;
  XS1D nu;
  XS1D chi;

  static MicroNuclideXS from_tuple(py::tuple t) {
    MicroNuclideXS out;
    py::bytes bytes = t[0].cast<py::bytes>();
    std::istringstream bits_stream(bytes,
                                   std::ios_base::binary | std::ios_base::in);
    {
      cereal::PortableBinaryInputArchive ar(bits_stream);
      ar(out);
    }
    return out;
  }

  py::tuple to_tuple() const {
    std::ostringstream bits_stream(std::ios_base::binary | std::ios_base::out);
    {
      cereal::PortableBinaryOutputArchive ar(bits_stream);
      ar(*this);
    }
    py::bytes bytes(bits_stream.str());
    return py::make_tuple(bytes);
  }

  template <class Archive>
  void serialize(Archive& arc) {
    arc(CEREAL_NVP(Et), CEREAL_NVP(Dtr), CEREAL_NVP(Es), CEREAL_NVP(Ea),
        CEREAL_NVP(Ef), CEREAL_NVP(nu), CEREAL_NVP(chi));
  }
};

struct ResonantOneGroupXS {
  double Dtr{0.};
  double Ea{0.};
  double Ef{0.};
  xt::xtensor<double, 2>
      Es;  // First index is legendre moment, second is outgoing energy group
  std::size_t gout_min{0};  // Index of first tabulated outgoing energy group

  std::optional<double> n_gamma{std::nullopt};

  // Scarabée assumes that (n,2n), (n,3n), (n,a), and (n,p) are not resonant
  // i.e. not dilution dependent.

  static ResonantOneGroupXS from_tuple(py::tuple t) {
    ResonantOneGroupXS out;
    py::bytes bytes = t[0].cast<py::bytes>();
    std::istringstream bits_stream(bytes,
                                   std::ios_base::binary | std::ios_base::in);
    {
      cereal::PortableBinaryInputArchive ar(bits_stream);
      ar(out);
    }
    return out;
  }

  py::tuple to_tuple() const {
    std::ostringstream bits_stream(std::ios_base::binary | std::ios_base::out);
    {
      cereal::PortableBinaryOutputArchive ar(bits_stream);
      ar(*this);
    }
    py::bytes bytes(bits_stream.str());
    return py::make_tuple(bytes);
  }

  template <class Archive>
  void serialize(Archive& arc) {
    arc(CEREAL_NVP(Dtr), CEREAL_NVP(Ea), CEREAL_NVP(Ef), CEREAL_NVP(Es),
        CEREAL_NVP(gout_min), CEREAL_NVP(n_gamma));
  }
};

struct DepletionReactionRates {
  std::string nuclide{};
  double number_density{0.};
  double n_gamma{0.};
  double n_2n{0.};
  double n_3n{0.};
  double n_p{0.};
  double n_alpha{0.};
  double n_fission{0.};
  double average_fission_energy{0.};

  static DepletionReactionRates from_tuple(py::tuple t) {
    DepletionReactionRates out;
    py::bytes bytes = t[0].cast<py::bytes>();
    std::istringstream bits_stream(bytes,
                                   std::ios_base::binary | std::ios_base::in);
    {
      cereal::PortableBinaryInputArchive ar(bits_stream);
      ar(out);
    }
    return out;
  }

  py::tuple to_tuple() const {
    std::ostringstream bits_stream(std::ios_base::binary | std::ios_base::out);
    {
      cereal::PortableBinaryOutputArchive ar(bits_stream);
      ar(*this);
    }
    py::bytes bytes(bits_stream.str());
    return py::make_tuple(bytes);
  }

  template <class Archive>
  void serialize(Archive& arc) {
    arc(CEREAL_NVP(nuclide), CEREAL_NVP(number_density), CEREAL_NVP(n_gamma),
        CEREAL_NVP(n_2n), CEREAL_NVP(n_3n), CEREAL_NVP(n_p),
        CEREAL_NVP(n_alpha), CEREAL_NVP(n_fission),
        CEREAL_NVP(average_fission_energy));
  }
};

}  // namespace scarabee

#endif

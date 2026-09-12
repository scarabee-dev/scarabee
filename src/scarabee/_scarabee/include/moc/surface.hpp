#ifndef SURFACE_H
#define SURFACE_H

#include <moc/vector.hpp>
#include <moc/direction.hpp>
#include <utils/logging.hpp>
#include <utils/scarabee_exception.hpp>

#include <cereal/cereal.hpp>
#include <cereal/types/array.hpp>
#include <cereal/types/base_class.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

#include <array>

namespace scarabee {

class Surface {
 public:
  enum class Type : char {
    XPlane,
    YPlane,
    Plane,
    Cylinder,
    BWRCornerI,
    BWRCornerII,
    BWRCornerIII,
    BWRCornerIV
  };
  enum class Side : bool { Positive, Negative };

  Surface(py::tuple t) {
    params_ = t[0].cast<std::array<double, 5>>();
    type_ = static_cast<Type>(t[1].cast<int>());
  }

  Side side(const Vector& r, const Direction& u) const;
  double distance(const Vector& r, const Direction& u) const;

  double integrate_x(double xmin, double xmax, const Side side) const;
  double integrate_y(double ymin, double ymax, const Side side) const;

  Type& type() { return type_; }
  const Type& type() const { return type_; }

  // Parameters for XPlane, YPlane, and Cylinder
  double& x0() { return params_[0]; }
  const double& x0() const { return params_[0]; }

  double& y0() { return params_[1]; }
  const double& y0() const { return params_[1]; }

  double& r() { return params_[2]; }
  const double& r() const { return params_[2]; }

  // Parameters for Plane
  double& A() { return params_[0]; }
  const double& A() const { return params_[0]; }

  double& B() { return params_[1]; }
  const double& B() const { return params_[1]; }

  double& C() { return params_[2]; }
  const double& C() const { return params_[2]; }

  // Parameters for a BWR Corner Surface
  double& xl() { return params_[0]; }
  const double& xl() const { return params_[0]; }

  double& xh() { return params_[1]; }
  const double& xh() const { return params_[1]; }

  double& yl() { return params_[2]; }
  const double& yl() const { return params_[2]; }

  double& yh() { return params_[3]; }
  const double& yh() const { return params_[3]; }

  double& rc() { return params_[4]; }
  const double& rc() const { return params_[4]; }

  py::tuple to_tuple() const {
    return py::make_tuple(this->params_, static_cast<int>(this->type_));
  }

 protected:
  Surface(Type type) : params_(), type_(type) {}

 private:
  Surface() : params_(), type_(Type::XPlane) {}

  std::array<double, 5> params_;
  Type type_;

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& arc) {
    arc(CEREAL_NVP(params_), CEREAL_NVP(type_));
  }
};

class XPlane : public Surface {
 public:
  XPlane(double x0) : Surface(Type::XPlane) { this->x0() = x0; }
  XPlane() : Surface(Type::XPlane) { this->x0() = 0.; }

 private:
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& arc) {
    arc(cereal::base_class<Surface>(this));
  }
};

class YPlane : public Surface {
 public:
  YPlane(double y0) : Surface(Type::YPlane) { this->y0() = y0; }
  YPlane() : Surface(Type::YPlane) { this->y0() = 0.; }

 private:
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& arc) {
    arc(cereal::base_class<Surface>(this));
  }
};

class Plane : public Surface {
 public:
  Plane(double A, double B, double C) : Surface(Type::Plane) {
    this->A() = A;
    this->B() = B;
    this->C() = C;
  }

 private:
  Plane() : Surface(Type::Plane) {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& arc) {
    arc(cereal::base_class<Surface>(this));
  }
};

class Cylinder : public Surface {
 public:
  Cylinder(double x0, double y0, double r) : Surface(Type::Cylinder) {
    this->x0() = x0;
    this->y0() = y0;
    this->r() = r;
  }

 private:
  Cylinder() : Surface(Type::Cylinder) {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& arc) {
    arc(cereal::base_class<Surface>(this));
  }
};

class BWRCornerI : public Surface {
 public:
  BWRCornerI(double xl, double xh, double yl, double yh, double rc)
      : Surface(Type::BWRCornerI) {
    if (xh <= xl) {
      const auto mssg = "x width must be > 0 (i.e. xl > xh).";
      spdlog::error(mssg);
      throw ScarabeeException(mssg);
    }

    if (yh <= yl) {
      const auto mssg = "y width must be > 0 (i.e. yl > yh).";
      spdlog::error(mssg);
      throw ScarabeeException(mssg);
    }

    if (rc < 0.) {
      const auto mssg = "Corner radius must be > 0 (i.e. rc > 0).";
      spdlog::error(mssg);
      throw ScarabeeException(mssg);
    }

    const double Rcx = this->xh() - this->rc();
    const double Rcy = this->yh() - this->rc();
    if (Rcx < this->xl() || Rcy < this->yl()) {
      const auto mssg = "Corner radius too large for cell width.";
      spdlog::error(mssg);
      throw ScarabeeException(mssg);
    }

    this->xl() = xl;
    this->xh() = xh;
    this->yl() = yl;
    this->yh() = yh;
    this->rc() = rc;
  }

 private:
  BWRCornerI() : Surface(Type::BWRCornerI) {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& arc) {
    arc(cereal::base_class<Surface>(this));
  }
};

class BWRCornerII : public Surface {
 public:
  BWRCornerII(double xl, double xh, double yl, double yh, double rc)
      : Surface(Type::BWRCornerII) {
    if (xh <= xl) {
      const auto mssg = "x width must be > 0 (i.e. xl > xh).";
      spdlog::error(mssg);
      throw ScarabeeException(mssg);
    }

    if (yh <= yl) {
      const auto mssg = "y width must be > 0 (i.e. yl > yh).";
      spdlog::error(mssg);
      throw ScarabeeException(mssg);
    }

    if (rc < 0.) {
      const auto mssg = "Corner radius must be > 0 (i.e. rc > 0).";
      spdlog::error(mssg);
      throw ScarabeeException(mssg);
    }

    const double Rcx = this->xl() + this->rc();
    const double Rcy = this->yh() - this->rc();
    if (Rcx > this->xh() || Rcy < this->yl()) {
      const auto mssg = "Corner radius too large for cell width.";
      spdlog::error(mssg);
      throw ScarabeeException(mssg);
    }

    this->xl() = xl;
    this->xh() = xh;
    this->yl() = yl;
    this->yh() = yh;
    this->rc() = rc;
  }

 private:
  BWRCornerII() : Surface(Type::BWRCornerII) {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& arc) {
    arc(cereal::base_class<Surface>(this));
  }
};

class BWRCornerIII : public Surface {
 public:
  BWRCornerIII(double xl, double xh, double yl, double yh, double rc)
      : Surface(Type::BWRCornerIII) {
    if (xh <= xl) {
      const auto mssg = "x width must be > 0 (i.e. xl > xh).";
      spdlog::error(mssg);
      throw ScarabeeException(mssg);
    }

    if (yh <= yl) {
      const auto mssg = "y width must be > 0 (i.e. yl > yh).";
      spdlog::error(mssg);
      throw ScarabeeException(mssg);
    }

    if (rc < 0.) {
      const auto mssg = "Corner radius must be > 0 (i.e. rc > 0).";
      spdlog::error(mssg);
      throw ScarabeeException(mssg);
    }

    const double Rcx = this->xl() + this->rc();
    const double Rcy = this->yl() + this->rc();
    if (Rcx > this->xh() || Rcy > this->yh()) {
      const auto mssg = "Corner radius too large for cell width.";
      spdlog::error(mssg);
      throw ScarabeeException(mssg);
    }

    this->xl() = xl;
    this->xh() = xh;
    this->yl() = yl;
    this->yh() = yh;
    this->rc() = rc;
  }

 private:
  BWRCornerIII() : Surface(Type::BWRCornerIII) {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& arc) {
    arc(cereal::base_class<Surface>(this));
  }
};

class BWRCornerIV : public Surface {
 public:
  BWRCornerIV(double xl, double xh, double yl, double yh, double rc)
      : Surface(Type::BWRCornerIV) {
    if (xh <= xl) {
      const auto mssg = "x width must be > 0 (i.e. xl > xh).";
      spdlog::error(mssg);
      throw ScarabeeException(mssg);
    }

    if (yh <= yl) {
      const auto mssg = "y width must be > 0 (i.e. yl > yh).";
      spdlog::error(mssg);
      throw ScarabeeException(mssg);
    }

    if (rc < 0.) {
      const auto mssg = "Corner radius must be > 0 (i.e. rc > 0).";
      spdlog::error(mssg);
      throw ScarabeeException(mssg);
    }

    const double Rcx = this->xh() - this->rc();
    const double Rcy = this->yl() + this->rc();
    if (Rcx < this->xl() || Rcy > this->yh()) {
      const auto mssg = "Corner radius too large for cell width.";
      spdlog::error(mssg);
      throw ScarabeeException(mssg);
    }

    this->xl() = xl;
    this->xh() = xh;
    this->yl() = yl;
    this->yh() = yh;
    this->rc() = rc;
  }

 private:
  BWRCornerIV() : Surface(Type::BWRCornerIV) {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& arc) {
    arc(cereal::base_class<Surface>(this));
  }
};

}  // namespace scarabee

#endif

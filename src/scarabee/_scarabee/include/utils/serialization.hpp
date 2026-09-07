#ifndef SCARABEE_SERIALIZATION_H
#define SCARABEE_SERIALIZATION_H

#include <cereal/cereal.hpp>
#include <cereal/types/vector.hpp>

#include <xtensor/containers/xtensor.hpp>
#include <xtensor/containers/xarray.hpp>

#include <htl/static_vector.hpp>

#include <Eigen/Dense>
#include <Eigen/SparseCore>

namespace cereal {

// xarray
template <class Archive, class T>
void save(Archive& arc, const xt::xarray<T>& a) {
  const std::size_t ndims = a.shape().size();
  arc(CEREAL_NVP(ndims));

  for (std::size_t i = 0; i < ndims; i++) {
    arc(a.shape()[i]);
  }

  const std::size_t size = a.size();
  arc(CEREAL_NVP(size));
  for (std::size_t i = 0; i < size; i++) {
    arc(a.flat(i));
  }
}

template <class Archive, class T>
void load(Archive& arc, xt::xarray<T>& a) {
  std::size_t ndims = 0;
  arc(CEREAL_NVP(ndims));

  std::vector<std::size_t> shape(ndims, 0);
  for (std::size_t i = 0; i < ndims; i++) {
    arc(shape[i]);
  }

  a.resize(shape);
  std::size_t size = 0;
  arc(CEREAL_NVP(size));
  for (std::size_t i = 0; i < size; i++) {
    arc(a.flat(i));
  }
}

// 1D Tensor
template <class Archive, class T>
void save(Archive& arc, const xt::xtensor<T, 1>& a) {
  const std::size_t shape_0 = a.shape()[0];
  arc(CEREAL_NVP(shape_0));
  for (std::size_t i = 0; i < shape_0; i++) {
    arc(a.flat(i));
  }
}

template <class Archive, class T>
void load(Archive& arc, xt::xtensor<T, 1>& a) {
  std::size_t shape_0 = 0;
  arc(CEREAL_NVP(shape_0));
  a.resize({shape_0});

  for (std::size_t i = 0; i < shape_0; i++) {
    arc(a.flat(i));
  }
}

// 2D Tensor
template <class Archive, class T>
void save(Archive& arc, const xt::xtensor<T, 2>& a) {
  const std::size_t shape_0 = a.shape()[0];
  const std::size_t shape_1 = a.shape()[1];
  arc(CEREAL_NVP(shape_0));
  arc(CEREAL_NVP(shape_1));
  for (std::size_t i = 0; i < shape_0 * shape_1; i++) {
    arc(a.flat(i));
  }
}

template <class Archive, class T>
void load(Archive& arc, xt::xtensor<T, 2>& a) {
  std::size_t shape_0 = 0;
  std::size_t shape_1 = 0;
  arc(CEREAL_NVP(shape_0));
  arc(CEREAL_NVP(shape_1));
  a.resize({shape_0, shape_1});

  for (std::size_t i = 0; i < shape_0 * shape_1; i++) {
    arc(a.flat(i));
  }
}

// 3D Tensor
template <class Archive, class T>
void save(Archive& arc, const xt::xtensor<T, 3>& a) {
  const std::size_t shape_0 = a.shape()[0];
  const std::size_t shape_1 = a.shape()[1];
  const std::size_t shape_2 = a.shape()[2];
  arc(CEREAL_NVP(shape_0));
  arc(CEREAL_NVP(shape_1));
  arc(CEREAL_NVP(shape_2));
  for (std::size_t i = 0; i < shape_0 * shape_1 * shape_2; i++) {
    arc(a.flat(i));
  }
}

template <class Archive, class T>
void load(Archive& arc, xt::xtensor<T, 3>& a) {
  std::size_t shape_0 = 0;
  std::size_t shape_1 = 0;
  std::size_t shape_2 = 0;
  arc(CEREAL_NVP(shape_0));
  arc(CEREAL_NVP(shape_1));
  arc(CEREAL_NVP(shape_2));
  a.resize({shape_0, shape_1, shape_2});

  for (std::size_t i = 0; i < shape_0 * shape_1 * shape_2; i++) {
    arc(a.flat(i));
  }
}

// svector
template <class Archive, class T>
void save(Archive& arc, const xt::svector<T>& a) {
  const std::size_t size = a.size();
  arc(CEREAL_NVP(size));
  for (std::size_t i = 0; i < size; i++) {
    arc(a[i]);
  }
}

template <class Archive, class T>
void load(Archive& arc, xt::svector<T>& a) {
  std::size_t size = 0;
  arc(CEREAL_NVP(size));
  a.resize(size);
  for (std::size_t i = 0; i < size; i++) {
    arc(a[i]);
  }
}

// Eigen
template <class Archive, class T, int M, int N>
void serialize(Archive& arc, Eigen::Matrix<T, M, N>& a) {
  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      arc(a(m, n));
    }
  }
}

template <class Archive>
void serialize(Archive& ar, Eigen::VectorXd& v) {
  std::size_t len_v = static_cast<std::size_t>(v.size());
  ar(len_v);

  if constexpr (Archive::is_loading::value) {
    v.resize(len_v);
  }

  for (std::size_t i = 0; i < len_v; i++) ar(v[i]);
}

template <class Archive, class Scalar, int Options, typename StorageIndex>
void serialize(Archive& ar,
               Eigen::SparseMatrix<Scalar, Options, StorageIndex>& mat) {
  Eigen::Index rows = mat.rows();
  Eigen::Index cols = mat.cols();

  ar(rows, cols);

  if constexpr (Archive::is_loading::value) {
    std::vector<StorageIndex> outer;
    std::vector<StorageIndex> inner;
    std::vector<Scalar> values;

    ar(outer, inner, values);

    // Reconstruct the matrix.
    mat.resize(rows, cols);

    // Reserve space for all nonzeros.
    mat.reserve(static_cast<Eigen::Index>(values.size()));

    // Insert the entries using Eigen's storage order.
    for (Eigen::Index outer_idx = 0; outer_idx < mat.outerSize(); ++outer_idx) {
      for (StorageIndex k = outer[outer_idx]; k < outer[outer_idx + 1]; ++k) {
        mat.insertBack(inner[k], outer_idx) = values[k];
      }
    }

    mat.makeCompressed();
  } else {
    // Make sure the matrix has compressed storage before accessing
    // the raw compressed-storage arrays.
    if (!mat.isCompressed()) {
      mat.makeCompressed();
    }

    const Eigen::Index outer_size = mat.outerSize();
    const Eigen::Index nnz = mat.nonZeros();

    std::vector<StorageIndex> outer(mat.outerIndexPtr(),
                                    mat.outerIndexPtr() + outer_size + 1);

    std::vector<StorageIndex> inner(mat.innerIndexPtr(),
                                    mat.innerIndexPtr() + nnz);

    std::vector<Scalar> values(mat.valuePtr(), mat.valuePtr() + nnz);

    ar(outer, inner, values);
  }
}

// HTL Static Vector

template <class Archive, class T, std::size_t C>
void save(Archive& arc, const htl::static_vector<T, C>& v) {
  const std::size_t size = v.size();
  arc(CEREAL_NVP(size));

  for (std::size_t i = 0; i < size; i++) {
    arc(v[i]);
  }
}

template <class Archive, class T, std::size_t C>
void load(Archive& arc, htl::static_vector<T, C>& v) {
  std::size_t size = 0;
  arc(CEREAL_NVP(size));

  v.resize(size);

  for (std::size_t i = 0; i < size; i++) {
    arc(v[i]);
  }
}

}  // namespace cereal

#endif

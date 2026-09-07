#ifndef SCARABEE_DEPLETION_CHAIN_H
#define SCARABEE_DEPLETION_CHAIN_H

#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <variant>
#include <vector>

#include <xtensor/containers/xtensor.hpp>

#include <cereal/cereal.hpp>
#include <cereal/types/map.hpp>
#include <cereal/types/optional.hpp>
#include <cereal/types/variant.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/string.hpp>

#include <highfive/highfive.hpp>
namespace H5 = HighFive;

#include <utils/serialization.hpp>

struct DepletionChainPickler;

namespace scarabee {

class NoTarget {
 public:
  NoTarget() = default;

  void initialize_hdf5_group(H5::Group& grp) const;
  static NoTarget from_hdf5_group(const H5::Group& grp);

 private:
  friend class cereal::access;

  template <class Archive>
  void serialize(Archive& /*arc*/) {}
};

class SingleTarget {
 public:
  SingleTarget(const std::string& target) : target_(target) {}

  SingleTarget() = default;

  const std::string& target() const { return target_; }

  void initialize_hdf5_group(H5::Group& grp) const;
  static SingleTarget from_hdf5_group(const H5::Group& grp);

 private:
  std::string target_;

  friend class cereal::access;

  template <class Archive>
  void serialize(Archive& arc) {
    arc(CEREAL_NVP(target_));
  }
};

class BranchingTargets {
 public:
  struct Branch {
    std::string target;
    double branch_ratio;

    template <class Archive>
    void serialize(Archive& arc) {
      arc(CEREAL_NVP(target), CEREAL_NVP(branch_ratio));
    }
  };

  BranchingTargets(const std::vector<Branch>& branches);

  BranchingTargets() = default;

  const std::vector<Branch>& branches() const { return branches_; }

  void remove_nuclide(const std::string& nuclide) {
    for (auto it = branches_.begin(); it != branches_.end(); it++) {
      if (it->target == nuclide) {
        it = branches_.erase(it);
        if (it == branches_.end()) break;
      }
    }
  }

  void replace_nuclide(const std::string& nuclide,
                       const std::string& new_nuclide) {
    for (auto it = branches_.begin(); it != branches_.end(); it++) {
      if (it->target == nuclide) {
        it->target = new_nuclide;
      }
    }
  }

  void replace_nuclide(const std::string& nuclide,
                       const BranchingTargets& targets) {
    double sum_ratios = 0.;
    for (auto it = branches_.begin(); it != branches_.end(); it++) {
      if (it->target == nuclide) {
        sum_ratios += it->branch_ratio;

        it = branches_.erase(it);
        if (it == branches_.end()) break;
      }
    }

    if (sum_ratios > 0.) {
      for (const auto& branch : targets.branches()) {
        branches_.push_back({branch.target, sum_ratios * branch.branch_ratio});
      }
    }
  }

  void initialize_hdf5_group(H5::Group& grp) const;
  static BranchingTargets from_hdf5_group(const H5::Group& grp);

 private:
  std::vector<Branch> branches_;

  friend class cereal::access;

  template <class Archive>
  void serialize(Archive& arc) {
    arc(CEREAL_NVP(branches_));
  }
};

using Target = std::variant<NoTarget, SingleTarget, BranchingTargets>;

class FissionYields {
 public:
  FissionYields(const std::vector<std::string>& targets,
                const std::vector<double>& incident_energies,
                const xt::xtensor<double, 2>& yields);

  FissionYields() = default;

  std::size_t size() const { return targets_.size(); }
  const std::vector<std::string>& targets() const { return targets_; }
  const std::vector<double>& incident_energies() const {
    return incident_energies_;
  }

  double yield(std::size_t t, double E) const;

  void remove_nuclide(const std::string& nuclide);
  void replace_nuclide(const std::string& nuclide,
                       const std::string& new_nuclide);
  void replace_nuclide(const std::string& nuclide,
                       const BranchingTargets& targets);

  void initialize_hdf5_group(H5::Group& grp) const;
  static FissionYields from_hdf5_group(const H5::Group& grp);

 private:
  std::vector<std::string> targets_;
  std::vector<double> incident_energies_;
  // First index incident energy, second is target
  xt::xtensor<double, 2> yields_;

  friend class cereal::access;

  template <class Archive>
  void serialize(Archive& arc) {
    arc(CEREAL_NVP(targets_), CEREAL_NVP(incident_energies_),
        CEREAL_NVP(yields_));
  }
};

class ChainEntry {
 public:
  ChainEntry()
      : half_life_(),
        decay_targets_(),
        n_gamma_(),
        n_2n_(),
        n_3n_(),
        n_p_(),
        n_alpha_(),
        n_fission_() {}

  std::optional<double>& half_life() { return half_life_; }
  const std::optional<double>& half_life() const { return half_life_; }

  std::optional<Target>& decay_targets() { return decay_targets_; }
  const std::optional<Target>& decay_targets() const { return decay_targets_; }

  std::optional<Target>& n_gamma() { return n_gamma_; }
  const std::optional<Target>& n_gamma() const { return n_gamma_; }

  std::optional<Target>& n_2n() { return n_2n_; }
  const std::optional<Target>& n_2n() const { return n_2n_; }

  std::optional<Target>& n_3n() { return n_3n_; }
  const std::optional<Target>& n_3n() const { return n_3n_; }

  std::optional<Target>& n_p() { return n_p_; }
  const std::optional<Target>& n_p() const { return n_p_; }

  std::optional<Target>& n_alpha() { return n_alpha_; }
  const std::optional<Target>& n_alpha() const { return n_alpha_; }

  std::optional<FissionYields>& n_fission() { return n_fission_; }
  const std::optional<FissionYields>& n_fission() const { return n_fission_; }

  void remove_nuclide(const std::string& nuclide, const Target& new_target);

  void initialize_hdf5_group(H5::Group& grp) const;
  static ChainEntry from_hdf5_group(const H5::Group& grp);

 private:
  // Radioactive Decay
  std::optional<double> half_life_;
  std::optional<Target> decay_targets_;

  // Transmutation Reactions
  std::optional<Target> n_gamma_;
  std::optional<Target> n_2n_;
  std::optional<Target> n_3n_;
  std::optional<Target> n_p_;
  std::optional<Target> n_alpha_;
  std::optional<FissionYields> n_fission_;

  void remove_nuclide(const std::string& nuclide, const NoTarget& new_target);
  void remove_nuclide(const std::string& nuclide,
                      const SingleTarget& new_target);
  void remove_nuclide(const std::string& nuclide,
                      const BranchingTargets& new_targets);

  void remove_nuclide(const std::string& nuclide, const NoTarget& new_target,
                      Target& target);
  void remove_nuclide(const std::string& nuclide,
                      const SingleTarget& new_target, Target& target);
  void remove_nuclide(const std::string& nuclide,
                      const BranchingTargets& new_targets, Target& target);

  // FissionYield removals
  void remove_nuclide(const std::string& nuclide,
                      const NoTarget& /*new_target*/, FissionYields& fy) {
    fy.remove_nuclide(nuclide);
  }
  void remove_nuclide(const std::string& nuclide,
                      const SingleTarget& new_target, FissionYields& fy) {
    fy.replace_nuclide(nuclide, new_target.target());
  }
  void remove_nuclide(const std::string& nuclide,
                      const BranchingTargets& new_targets, FissionYields& fy) {
    fy.replace_nuclide(nuclide, new_targets);
  }

  friend class cereal::access;

  template <class Archive>
  void serialize(Archive& arc) {
    arc(CEREAL_NVP(half_life_), CEREAL_NVP(decay_targets_),
        CEREAL_NVP(n_gamma_), CEREAL_NVP(n_2n_), CEREAL_NVP(n_3n_),
        CEREAL_NVP(n_p_), CEREAL_NVP(n_alpha_), CEREAL_NVP(n_fission_));
  }
};

class DepletionChain {
 public:
  DepletionChain() : data_() {}

  bool holds_nuclide_data(const std::string& nuclide) const;
  const ChainEntry& nuclide_data(const std::string& nuclide) const;

  std::set<std::string> nuclides() const;

  void insert_entry(const std::string& nuclide, const ChainEntry& entry);

  std::vector<std::string> descend_chains(std::set<std::string> nuclides,
                                          bool decay_only = false) const;

  void remove_nuclide(const std::string& nuclide);

  void save(const std::string& fname) const;
  static std::shared_ptr<DepletionChain> load(const std::string& fname);

  void initialize_hdf5_group(H5::Group& grp) const;
  static DepletionChain from_hdf5_group(const H5::Group& grp);

 private:
  std::map<std::string, ChainEntry> data_;

  friend class cereal::access;
  friend struct ::DepletionChainPickler;

  template <class Archive>
  void serialize(Archive& arc) {
    arc(CEREAL_NVP(data_));
  }
};

// Functions to save / load a depletion chain to / from an HDF5 file

Target target_from_hdf5_group(const H5::Group& grp);

}  // namespace scarabee

#endif

#include <data/depletion_chain.hpp>
#include <utils/logging.hpp>
#include <utils/nuclide_names.hpp>
#include <utils/scarabee_exception.hpp>

#include <cereal/archives/portable_binary.hpp>

#include <xtensor/views/xview.hpp>

#include <algorithm>
#include <array>
#include <filesystem>

namespace scarabee {

void NoTarget::initialize_hdf5_group(H5::Group& grp) const {
  grp.createAttribute("type", std::string("no-target"));
}

NoTarget NoTarget::from_hdf5_group(const H5::Group& grp) {
  if (grp.hasAttribute("type") == false) {
    const auto mssg = "HDF5 Group has no attribute named \"type\".";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  std::string type = grp.getAttribute("type").read<std::string>();

  if (type != "no-target") {
    std::stringstream mssg;
    mssg << "Cannot load NoTarget object from HDF5 group of type \"" << type
         << "\".";
    spdlog::error(mssg.str());
    throw ScarabeeException(mssg.str());
  }

  return NoTarget();
}

void SingleTarget::initialize_hdf5_group(H5::Group& grp) const {
  grp.createAttribute("type", std::string("single-target"));
  grp.createAttribute("name", target_);
}

SingleTarget SingleTarget::from_hdf5_group(const H5::Group& grp) {
  if (grp.hasAttribute("type") == false) {
    const auto mssg = "HDF5 Group has no attribute named \"type\".";
    spdlog::error(mssg);
    throw ScarabeeException();
  }

  std::string type = grp.getAttribute("type").read<std::string>();

  if (type != "single-target") {
    std::stringstream mssg;
    mssg << "Cannot load SingleTarget object from HDF5 group of type \"" << type
         << "\".";
    spdlog::error(mssg.str());
    throw ScarabeeException(mssg.str());
  }

  if (grp.hasAttribute("name") == false) {
    const auto mssg =
        "Cannot load SingleTarget object due to missing name attribute.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  std::string name = grp.getAttribute("name").read<std::string>();

  return SingleTarget(name);
}

BranchingTargets::BranchingTargets(const std::vector<Branch>& branches)
    : branches_(branches) {
  if (branches_.size() < 2) {
    const auto mssg =
        "Must have at least 2 targets for a BranchingTargets instance.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  // Make sure ratios are all > 0.
  for (const auto& branch : branches_) {
    if (branch.branch_ratio <= 0.) {
      const auto mssg =
          "Branching ratio for target \"" + branch.target + "\" is <= 0.";
      spdlog::error(mssg);
      throw ScarabeeException(mssg);
    }
  }

  // We don't normalize the branching ratios and don't care if the sum != 1.
  // This is due to the fact that if we remove short lived nuclides, and they
  // don't have a decay target, we then the ratios wouldn't sum to unity
  // anymore ! Normalizing after would artificially increase production of the
  // other possible targets in the branch which we don't want either. Also, a
  // BranchingTargets could come from spontaneous fission, which will have a
  // ratio sum which is quite different from unity.
}

void BranchingTargets::initialize_hdf5_group(H5::Group& grp) const {
  if (branches_.size() == 0) {
    const auto mssg = "List of branches is empty.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  grp.createAttribute("type", std::string("branching-targets"));

  std::vector<std::string> targets;
  std::vector<double> ratios;
  targets.reserve(branches_.size());
  ratios.reserve(branches_.size());

  for (const auto& branch : branches_) {
    targets.push_back(branch.target);
    ratios.push_back(branch.branch_ratio);
  }

  grp.createAttribute("targets", targets);
  grp.createAttribute("ratios", ratios);
}

BranchingTargets BranchingTargets::from_hdf5_group(const H5::Group& grp) {
  if (grp.hasAttribute("type") == false) {
    const auto mssg = "HDF5 Group has no attribute named \"type\".";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  std::string type = grp.getAttribute("type").read<std::string>();

  if (type != "branching-targets") {
    std::stringstream mssg;
    mssg << "Cannot load BranchingTargets object from HDF5 group of type \""
         << type << "\".";
    spdlog::error(mssg.str());
    throw ScarabeeException(mssg.str());
  }

  if (grp.hasAttribute("targets") == false) {
    const auto mssg =
        "Cannot load BranchingTargets object due to missing targets attribute.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }
  std::vector<std::string> targets =
      grp.getAttribute("targets").read<std::vector<std::string>>();

  if (grp.hasAttribute("ratios") == false) {
    const auto mssg =
        "Cannot load BranchingTargets object due to missing ratios attribute.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }
  std::vector<double> ratios =
      grp.getAttribute("ratios").read<std::vector<double>>();

  if (targets.size() != ratios.size()) {
    const auto mssg = "Targets and ratios have different lengths.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  if (targets.size() == 0) {
    const auto mssg = "Targets and ratios arrays are empty.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  std::vector<Branch> branches;
  branches.reserve(targets.size());
  for (std::size_t i = 0; i < targets.size(); i++) {
    branches.push_back({targets[i], ratios[i]});
  }

  return BranchingTargets(branches);
}

FissionYields::FissionYields(const std::vector<std::string>& targets,
                             const std::vector<double>& incident_energies,
                             const xt::xtensor<double, 2>& yields)
    : targets_(targets),
      incident_energies_(incident_energies),
      yields_(yields) {
  // Make sure we have targets and energies
  if (targets_.size() == 0) {
    const auto mssg = "No fission yield targets provided.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  if (incident_energies_.size() == 0) {
    const auto mssg = "No incident energies provided.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  // Make sure energies are sorted
  if (std::is_sorted(incident_energies_.begin(), incident_energies_.end()) ==
      false) {
    const auto mssg = "Incident energies must be sorted.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  if (incident_energies_.front() <= 0.) {
    const auto mssg = "Incident energies must be > 0.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  // Make sure shape of yields is correct
  if (yields_.shape()[0] != incident_energies_.size()) {
    const auto mssg =
        "First dimension of yields must have same length as number of incident "
        "energies.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  if (yields_.shape()[1] != targets_.size()) {
    const auto mssg =
        "Second dimension of yields must have same length as number of "
        "targets.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  // Make sure all yields are >= 0
  const double yield_min = xt::amin(yields_)();
  if (yield_min < 0.) {
    const auto mssg = "Fission yields must be >= 0.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }
}

double FissionYields::yield(std::size_t t, double E) const {
  if (t >= this->size()) {
    const auto mssg = "Target index is out of range.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  if (incident_energies_.size() == 1) {
    return yields_(0, t);
  }

  std::size_t iE = 0;
  double fE = 0.;

  if (E >= incident_energies_.back()) {
    iE = incident_energies_.size() - 2;
    fE = 1.;
  } else if (E > incident_energies_.front()) {
    for (iE = 0; iE < incident_energies_.size() - 1; iE++) {
      const double El = incident_energies_[iE];
      const double Eh = incident_energies_[iE + 1];

      if (El <= E && E <= Eh) {
        fE = (E - El) / (Eh - El);
        break;
      }
    }
  }

  return yields_(iE, t) * (1. - fE) + yields_(iE + 1, t) * fE;
}

void FissionYields::remove_nuclide(const std::string& nuclide) {
  auto has_nuclide = [this, &nuclide]() {
    for (const auto& target : this->targets_) {
      if (target == nuclide) return true;
    }
    return false;
  };

  while (has_nuclide()) {
    // Get the index of the nuclide
    std::size_t i = 0;
    auto it = targets_.begin();
    for (i = 0; i < targets_.size(); i++) {
      if (nuclide == targets_[i]) break;
      it++;
    }
    targets_.erase(it);  // Remove target name

    // We have the index, now we remove it from the array.
    // To do this, we first make a view
    std::array<std::size_t, 1> dropped_ind{i};
    auto view = xt::view(yields_, xt::all(), xt::drop(dropped_ind));
    xt::xtensor<double, 2> temp_yields = view;
    yields_ = temp_yields;
  }
}

void FissionYields::replace_nuclide(const std::string& nuclide,
                                    const std::string& new_nuclide) {
  if (nuclide == new_nuclide) return;

  // First, check if the new_nuclide already exists
  std::size_t new_nuc_i = this->size();
  std::size_t nuc_i = this->size();
  for (std::size_t i = 0; i < this->size(); i++) {
    if (this->targets()[i] == nuclide) {
      nuc_i = i;
    } else if (this->targets()[i] == new_nuclide) {
      new_nuc_i = i;
    }

    if (nuc_i != this->size() && new_nuc_i != this->size()) break;
  }

  // If we didn't find nuclide, we don't need to replace it !
  if (nuc_i == this->size()) return;

  if (nuc_i != this->size() && new_nuc_i != this->size()) {
    // If the new_nuclide already exists, add it's contribution then drop the
    // old index to nuclide
    xt::view(yields_, xt::all(), new_nuc_i) +=
        xt::view(yields_, xt::all(), nuc_i);

    // Drop the old yields column
    std::array<std::size_t, 1> dropped_ind{nuc_i};
    auto view = xt::view(yields_, xt::all(), xt::drop(dropped_ind));
    xt::xtensor<double, 2> temp_yields = view;
    yields_ = temp_yields;

    // Drop the old target
    targets_.erase(targets_.begin() + nuc_i);

    // Once we have dropped, nothing left to do !
    return;
  }

  // If we get here, the new nuclide does not yet exist in targets, so we can
  // just replace the old target with the new target
  targets_[nuc_i] = new_nuclide;
}

void FissionYields::replace_nuclide(const std::string& nuclide,
                                    const BranchingTargets& targets) {
  // Get the total yield of the nuclide we want to remove
  std::vector<double> nuclide_yield_sums(incident_energies_.size(), 0.);
  for (std::size_t i = 0; i < targets_.size(); i++) {
    if (nuclide == targets_[i]) {
      for (std::size_t e = 0; e < incident_energies_.size(); e++) {
        nuclide_yield_sums[e] += yields_(e, i);
      }
    }
  }

  // Remove all instances of the nuclide
  this->remove_nuclide(nuclide);

  // Go through and first add targets which already are present
  std::vector<std::size_t> unused_targets;
  for (std::size_t b = 0; b < targets.branches().size(); b++) {
    const auto& branch = targets.branches()[b];
    bool found_target = false;
    for (std::size_t t = 0; t < this->size(); t++) {
      if (branch.target == this->targets()[t]) {
        for (std::size_t ie = 0; ie < this->incident_energies().size(); ie++) {
          yields_(ie, t) += nuclide_yield_sums[ie] * branch.branch_ratio;
        }
        found_target = true;
        break;
      }
    }
    if (found_target == false) unused_targets.push_back(b);
  }

  // Make copy of yields and then zero yields
  xt::xtensor<double, 2> old_yields = yields_;
  yields_ = xt::zeros<double>(
      {old_yields.shape()[0], old_yields.shape()[1] + unused_targets.size()});

  // Re-fill yields
  for (std::size_t e = 0; e < yields_.shape()[0]; e++) {
    std::size_t ti = 0;
    for (std::size_t i = 0; i < yields_.shape()[1]; i++) {
      if (i < old_yields.shape()[1]) {
        // Use old data if in that region
        yields_(e, i) = old_yields(e, i);
      } else {
        // If in region of new target, get info from branch ratios
        yields_(e, i) = targets.branches()[unused_targets[ti]].branch_ratio *
                        nuclide_yield_sums[e];
        ti++;
      }
    }
  }

  // Add the new targets to target list
  for (const auto ti : unused_targets) {
    targets_.push_back(targets.branches()[ti].target);
  }
}

void FissionYields::initialize_hdf5_group(H5::Group& grp) const {
  grp.createAttribute("type", std::string("fission-yields"));

  grp.createAttribute("targets", targets_);
  grp.createAttribute("incident_energies", incident_energies_);

  std::vector<std::size_t> yields_dims{yields_.shape()[0], yields_.shape()[1]};
  H5::DataSet yields_dset =
      grp.createDataSet<double>("yields", H5::DataSpace(yields_dims));
  yields_dset.write_raw(yields_.data());
}

FissionYields FissionYields::from_hdf5_group(const H5::Group& grp) {
  if (grp.hasAttribute("type") == false) {
    const auto mssg = "HDF5 Group has no attribute named \"type\".";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  std::string type = grp.getAttribute("type").read<std::string>();

  if (type != "fission-yields") {
    std::stringstream mssg;
    mssg << "Cannot load FissionYields object from HDF5 group of type \""
         << type << "\".";
    spdlog::error(mssg.str());
    throw ScarabeeException(mssg.str());
  }

  if (grp.hasAttribute("targets") == false) {
    const auto mssg =
        "Cannot load FissionYields object due to missing targets attribute.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }
  std::vector<std::string> targets =
      grp.getAttribute("targets").read<std::vector<std::string>>();

  if (targets.size() == 0) {
    const auto mssg = "FissionYields targets list is empty.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  if (grp.hasAttribute("incident_energies") == false) {
    const auto mssg =
        "Cannot load FissionYields object due to missing incident_energies "
        "attribute.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }
  std::vector<double> incident_energies =
      grp.getAttribute("incident_energies").read<std::vector<double>>();

  if (incident_energies.size() == 0) {
    const auto mssg = "FissionYields incident_energies list is empty.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  if (grp.exist("yields") == false) {
    const auto mssg =
        "Cannot load FissionYields object due to missing yields data set.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  if (grp.getObjectType("yields") != H5::ObjectType::Dataset) {
    const auto mssg = "yields member is not a data set.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  H5::DataSet yields_dset = grp.getDataSet("yields");
  auto yields_dims = yields_dset.getDimensions();

  if (yields_dims[0] != incident_energies.size()) {
    const auto mssg =
        "First dimension of yields array does not agree with length of "
        "incident_energies.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  if (yields_dims[1] != targets.size()) {
    const auto mssg =
        "Second dimension of yields array does not agree with length of "
        "targets.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  xt::xtensor<double, 2> yields =
      xt::zeros<double>({yields_dims[0], yields_dims[1]});
  yields_dset.read_raw<double>(yields.data());

  return FissionYields(targets, incident_energies, yields);
}

void ChainEntry::remove_nuclide(const std::string& nuclide,
                                const Target& new_target) {
  auto target_eliminator = [this, &nuclide](const auto& t) {
    this->remove_nuclide(nuclide, t);
  };
  std::visit(target_eliminator, new_target);
}

void ChainEntry::remove_nuclide(const std::string& nuclide,
                                const NoTarget& new_target) {
  if (decay_targets_)
    remove_nuclide(nuclide, new_target, decay_targets_.value());
  if (n_gamma_) remove_nuclide(nuclide, new_target, n_gamma_.value());
  if (n_2n_) remove_nuclide(nuclide, new_target, n_2n_.value());
  if (n_3n_) remove_nuclide(nuclide, new_target, n_3n_.value());
  if (n_p_) remove_nuclide(nuclide, new_target, n_p_.value());
  if (n_alpha_) remove_nuclide(nuclide, new_target, n_alpha_.value());
  if (n_fission_) remove_nuclide(nuclide, new_target, n_fission_.value());
}

void ChainEntry::remove_nuclide(const std::string& nuclide,
                                const SingleTarget& new_target) {
  if (decay_targets_)
    remove_nuclide(nuclide, new_target, decay_targets_.value());
  if (n_gamma_) remove_nuclide(nuclide, new_target, n_gamma_.value());
  if (n_2n_) remove_nuclide(nuclide, new_target, n_2n_.value());
  if (n_3n_) remove_nuclide(nuclide, new_target, n_3n_.value());
  if (n_p_) remove_nuclide(nuclide, new_target, n_p_.value());
  if (n_alpha_) remove_nuclide(nuclide, new_target, n_alpha_.value());
  if (n_fission_) remove_nuclide(nuclide, new_target, n_fission_.value());
}

void ChainEntry::remove_nuclide(const std::string& nuclide,
                                const BranchingTargets& new_targets) {
  if (decay_targets_)
    remove_nuclide(nuclide, new_targets, decay_targets_.value());
  if (n_gamma_) remove_nuclide(nuclide, new_targets, n_gamma_.value());
  if (n_2n_) remove_nuclide(nuclide, new_targets, n_2n_.value());
  if (n_3n_) remove_nuclide(nuclide, new_targets, n_3n_.value());
  if (n_p_) remove_nuclide(nuclide, new_targets, n_p_.value());
  if (n_alpha_) remove_nuclide(nuclide, new_targets, n_alpha_.value());
  if (n_fission_) remove_nuclide(nuclide, new_targets, n_fission_.value());
}

void ChainEntry::remove_nuclide(const std::string& nuclide,
                                const NoTarget& new_target, Target& target) {
  if (std::holds_alternative<NoTarget>(target)) {
    // Transfering NoTarget to NoTarget. Nothing to do.
  } else if (std::holds_alternative<SingleTarget>(target)) {
    const auto target_name = std::get<SingleTarget>(target).target();
    if (target_name == nuclide) {
      target = new_target;
    }
  } else {
    // BranchingTargets
    auto& branch_targets = std::get<BranchingTargets>(target);
    branch_targets.remove_nuclide(nuclide);
  }
}

void ChainEntry::remove_nuclide(const std::string& nuclide,
                                const SingleTarget& new_target,
                                Target& target) {
  if (std::holds_alternative<NoTarget>(target)) {
    // Nothing to do. Target is empty so don't need to replace anything.
  } else if (std::holds_alternative<SingleTarget>(target)) {
    const auto& target_name = std::get<SingleTarget>(target).target();
    if (target_name == nuclide) {
      target = new_target;
    }
  } else {
    // BranchingTargets
    auto& branch_targets = std::get<BranchingTargets>(target);
    branch_targets.replace_nuclide(nuclide, new_target.target());
  }
}

void ChainEntry::remove_nuclide(const std::string& nuclide,
                                const BranchingTargets& new_targets,
                                Target& target) {
  if (std::holds_alternative<NoTarget>(target)) {
    // Nothing to do. Target is empty so don't need to replace anything.
  } else if (std::holds_alternative<SingleTarget>(target)) {
    const auto& target_name = std::get<SingleTarget>(target).target();
    if (target_name == nuclide) {
      target = new_targets;
    }
  } else {
    // BranchingTargets
    auto& branch_targets = std::get<BranchingTargets>(target);
    branch_targets.replace_nuclide(nuclide, new_targets);

    if (branch_targets.branches().size() == 0) {
      target = NoTarget();
    }
  }
}

void ChainEntry::initialize_hdf5_group(H5::Group& grp) const {
  if (half_life_) {
    grp.createAttribute("half-life", half_life_.value());
  }

  if (decay_targets_) {
    auto tgrp = grp.createGroup("decay_targets");
    std::visit([&tgrp](const auto& T) { T.initialize_hdf5_group(tgrp); },
               decay_targets_.value());
  }

  if (n_gamma_) {
    auto tgrp = grp.createGroup("(n,gamma)");
    std::visit([&tgrp](const auto& T) { T.initialize_hdf5_group(tgrp); },
               n_gamma_.value());
  }

  if (n_2n_) {
    auto tgrp = grp.createGroup("(n,2n)");
    std::visit([&tgrp](const auto& T) { T.initialize_hdf5_group(tgrp); },
               n_2n_.value());
  }

  if (n_3n_) {
    auto tgrp = grp.createGroup("(n,3n)");
    std::visit([&tgrp](const auto& T) { T.initialize_hdf5_group(tgrp); },
               n_3n_.value());
  }

  if (n_p_) {
    auto tgrp = grp.createGroup("(n,p)");
    std::visit([&tgrp](const auto& T) { T.initialize_hdf5_group(tgrp); },
               n_p_.value());
  }

  if (n_alpha_) {
    auto tgrp = grp.createGroup("(n,alpha)");
    std::visit([&tgrp](const auto& T) { T.initialize_hdf5_group(tgrp); },
               n_alpha_.value());
  }

  if (n_fission_) {
    auto tgrp = grp.createGroup("(n,fission)");
    n_fission_.value().initialize_hdf5_group(tgrp);
  }
}

ChainEntry ChainEntry::from_hdf5_group(const H5::Group& grp) {
  ChainEntry c;

  if (grp.hasAttribute("half-life")) {
    c.half_life_ = grp.getAttribute("half-life").read<double>();
  }

  if (grp.exist("decay_targets")) {
    c.decay_targets_ = target_from_hdf5_group(grp.getGroup("decay_targets"));
  }

  if (grp.exist("(n,gamma)")) {
    c.n_gamma_ = target_from_hdf5_group(grp.getGroup("(n,gamma)"));
  }

  if (grp.exist("(n,2n)")) {
    c.n_2n_ = target_from_hdf5_group(grp.getGroup("(n,2n)"));
  }

  if (grp.exist("(n,3n)")) {
    c.n_3n_ = target_from_hdf5_group(grp.getGroup("(n,3n)"));
  }

  if (grp.exist("(n,p)")) {
    c.n_p_ = target_from_hdf5_group(grp.getGroup("(n,p)"));
  }

  if (grp.exist("(n,alpha)")) {
    c.n_alpha_ = target_from_hdf5_group(grp.getGroup("(n,alpha)"));
  }

  if (grp.exist("(n,fission)")) {
    c.n_fission_ = FissionYields::from_hdf5_group(grp.getGroup("(n,fission)"));
  }

  return c;
}

bool DepletionChain::holds_nuclide_data(const std::string& nuclide) const {
  if (data_.find(nuclide) == data_.end()) {
    return false;
  }

  return true;
}

const ChainEntry& DepletionChain::nuclide_data(
    const std::string& nuclide) const {
  const auto it = data_.find(nuclide);
  if (it == data_.end()) {
    const auto mssg =
        "Nuclide \"" + nuclide + "\" is not present in the depletion chain.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  return it->second;
}

void DepletionChain::insert_entry(const std::string& nuclide,
                                  const ChainEntry& entry) {
  if (this->holds_nuclide_data(nuclide)) {
    const auto mssg = "Nuclide \"" + nuclide +
                      "\" is already present in the depletion chain.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  data_[nuclide] = entry;
}

std::set<std::string> DepletionChain::nuclides() const {
  std::set<std::string> nucs;
  for (const auto& nuc : data_) nucs.insert(nuc.first);
  return nucs;
}

void DepletionChain::remove_nuclide(const std::string& nuclide) {
  // If nuclide isn't in the chain, we can't remove it
  if (this->holds_nuclide_data(nuclide) == false) {
    const auto mssg = "Cannot remove nuclide \"" + nuclide +
                      "\", as not present in depletion chain.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  // Get the ChainEntry for the nuclide
  auto& nuc = this->data_[nuclide];

  // Does the nuclide have decay data ? If not, we can't exactly replace it.
  if (nuc.decay_targets().has_value() == false) {
    const auto mssg = "Nuclide \"" + nuclide + "\" has no decay targets.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  auto& decay_targets = nuc.decay_targets().value();

  // We need to make sure the decay target doesn't contain itself !!
  if (std::holds_alternative<SingleTarget>(decay_targets)) {
    if (std::get<SingleTarget>(decay_targets).target() == nuclide) {
      // If we only decay to ourself, set the decay to no target.
      decay_targets = NoTarget();
    }
  } else if (std::holds_alternative<BranchingTargets>(decay_targets)) {
    // If we have branches, remove ourself
    auto& dt = std::get<BranchingTargets>(decay_targets);
    dt.remove_nuclide(nuclide);

    if (dt.branches().size() == 0) {
      // We once we removed ourself there are no branches,
      // just set to no target.
      decay_targets = NoTarget();
    }
  }

  // Go through all other chain entries (but not ourself!) and replace
  // instances of this nuclide in targets with its decay targets.
  for (auto& entry : data_) {
    if (&entry.second == &nuc) continue;  // Don't do ourself !

    entry.second.remove_nuclide(nuclide, decay_targets);
  }

  // Delete nuclide from the chain
  data_.erase(nuclide);
}

// Helper functions for extracting new targets from various types of targets
void extract_targets(const NoTarget& /*t*/,
                     std::set<std::string>& /*found_nuclides*/,
                     std::set<std::string>& /*next_nuclides*/) {}

void extract_targets(const SingleTarget& t,
                     std::set<std::string>& found_nuclides,
                     std::set<std::string>& next_nuclides) {
  if (found_nuclides.contains(t.target()))
    return;
  else if (next_nuclides.contains(t.target()))
    return;
  next_nuclides.insert(t.target());
}

void extract_targets(const BranchingTargets& bt,
                     std::set<std::string>& found_nuclides,
                     std::set<std::string>& next_nuclides) {
  for (const auto& branch : bt.branches()) {
    if (found_nuclides.contains(branch.target))
      continue;
    else if (next_nuclides.contains(branch.target))
      continue;
    next_nuclides.insert(branch.target);
  }
}

void extract_targets(const FissionYields& fy,
                     std::set<std::string>& found_nuclides,
                     std::set<std::string>& next_nuclides) {
  for (const auto& target : fy.targets()) {
    if (found_nuclides.contains(target))
      continue;
    else if (next_nuclides.contains(target))
      continue;
    next_nuclides.insert(target);
  }
}

std::vector<std::string> DepletionChain::descend_chains(
    std::set<std::string> nuclides, bool decay_only) const {
  std::set<std::string> found_nuclides = nuclides;
  std::set<std::string> next_nuclides;

  // While we have a new set of nuclides to follow
  while (nuclides.empty() == false) {
    // Check all products of each nuclide
    for (const auto& nuc_name : nuclides) {
      // Save this nuclide as having been "found"
      found_nuclides.insert(nuc_name);

      // Get the chain entry for the nuclides. If it doesn't have an entry,
      // go to the next nuclide.
      if (this->holds_nuclide_data(nuc_name) == false) continue;
      const auto& entry = this->nuclide_data(nuc_name);

      // For each decay or transmutation reaction (including fission), save all
      // the newly encountered target nuclides.
      auto target_extractor = [&found_nuclides, &next_nuclides](const auto& t) {
        extract_targets(t, found_nuclides, next_nuclides);
      };

      if (entry.decay_targets())
        std::visit(target_extractor, *entry.decay_targets());
      if (decay_only) continue;

      if (entry.n_gamma()) std::visit(target_extractor, *entry.n_gamma());
      if (entry.n_2n()) std::visit(target_extractor, *entry.n_2n());
      if (entry.n_3n()) std::visit(target_extractor, *entry.n_3n());
      if (entry.n_p()) std::visit(target_extractor, *entry.n_p());
      if (entry.n_alpha()) std::visit(target_extractor, *entry.n_alpha());
      if (entry.n_fission())
        extract_targets(*entry.n_fission(), found_nuclides, next_nuclides);
    }

    nuclides.swap(next_nuclides);
    next_nuclides.clear();
  }

  // Make a vector with the found nuclides
  std::vector<std::string> found_vec(found_nuclides.begin(),
                                     found_nuclides.end());

  // Sort nuclides by Z then A
  std::sort(found_vec.begin(), found_vec.end(),
            [](const std::string& n1, const std::string& n2) {
              return nuclide_name_to_za(n1) < nuclide_name_to_za(n2);
            });

  return found_vec;
}

void DepletionChain::save(const std::string& fname) const {
  if (std::filesystem::exists(fname)) {
    std::filesystem::remove(fname);
  }

  H5::File h5(fname, H5::File::Create);

  auto dc_grp = h5.createGroup("depletion-chain");

  this->initialize_hdf5_group(dc_grp);
}

std::shared_ptr<DepletionChain> DepletionChain::load(const std::string& fname) {
  if (std::filesystem::exists(fname) == false) {
    std::stringstream mssg;
    mssg << "The file \"" << fname << "\" does not exist.";
    spdlog::error(mssg.str());
    throw ScarabeeException(mssg.str());
  }

  H5::File h5(fname, H5::File::ReadOnly);

  if (h5.exist("depletion-chain") == false) {
    const auto mssg = "HDF5 file does not contain a depletion-chain entry.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  if (h5.getObjectType("depletion-chain") != H5::ObjectType::Group) {
    const auto mssg = "The depletion-chain entry is not a group.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  auto dc_grp = h5.getGroup("depletion-chain");

  std::shared_ptr<DepletionChain> dc = std::make_shared<DepletionChain>();

  *dc = DepletionChain::from_hdf5_group(dc_grp);

  return dc;
}

void DepletionChain::initialize_hdf5_group(H5::Group& grp) const {
  for (const auto& entry : data_) {
    auto entry_group = grp.createGroup(entry.first);
    entry.second.initialize_hdf5_group(entry_group);
  }
}

DepletionChain DepletionChain::from_hdf5_group(const H5::Group& grp) {
  DepletionChain dc;

  const auto obj_names = grp.listObjectNames();

  for (const auto& name : obj_names) {
    if (grp.getObjectType(name) != H5::ObjectType::Group) {
      std::stringstream mssg;
      mssg << "HDF5 object of name " << name << " is not a group.";
      spdlog::error(mssg.str());
      throw ScarabeeException(mssg.str());
    }

    auto entry_group = grp.getGroup(name);
    auto entry = ChainEntry::from_hdf5_group(entry_group);

    dc.insert_entry(name, entry);
  }

  return dc;
}

Target target_from_hdf5_group(const H5::Group& grp) {
  if (grp.hasAttribute("type") == false) {
    const auto mssg = "HDF5 Group has no attribute named \"type\".";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  std::string type = grp.getAttribute("type").read<std::string>();

  if (type == "no-target") {
    return NoTarget::from_hdf5_group(grp);
  } else if (type == "single-target") {
    return SingleTarget::from_hdf5_group(grp);
  } else if (type == "branching-targets") {
    return BranchingTargets::from_hdf5_group(grp);
  } else {
    std::stringstream mssg;
    mssg << "Unknown depletion target type \"" << type << "\".";
    spdlog::error(mssg.str());
    throw ScarabeeException(mssg.str());
  }

  // NEVER GETS HERE
  return NoTarget();
}

}  // namespace scarabee

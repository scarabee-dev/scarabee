#include <data/nd_library.hpp>
#include <utils/constants.hpp>
#include <utils/logging.hpp>
#include <utils/scarabee_exception.hpp>

#include <xtensor/containers/xtensor.hpp>

#include <cmath>
#include <cstdlib>
#include <optional>
#include <sstream>
#include "spdlog/spdlog.h"

namespace scarabee {

void NuclideHandle::load_xs_from_hdf5(const NDLibrary& ndl, std::size_t max_l) {
  if (this->loaded()) return;

  // Get the HDF5 Group for the nuclide
  auto grp = ndl.h5()->getGroup(this->name);

  // Load in all of the infinite dilution data
  this->load_inf_data(ndl, grp, max_l);

  // Load dilution dependent data if necessary
  if (this->resonant) {
    this->load_res_data(grp, max_l);
  }
}

void NuclideHandle::load_inf_data(const NDLibrary& ndl, const H5::Group& grp,
                                  std::size_t max_l) {
  // First we read in the packing of the scattering matrices. This is needed
  // to get the length of the scattering arrays
  packing = std::make_shared<xt::xtensor<std::uint32_t, 2>>();
  packing->resize({ndl.ngroups(), static_cast<std::size_t>(3)});
  grp.getDataSet("matrix-compression").read_raw<std::uint32_t>(packing->data());
  const std::size_t len_scat_data = (*packing)(ndl.ngroups() - 1, 0) +
                                    (*packing)(ndl.ngroups() - 1, 2) + 1 -
                                    (*packing)(ndl.ngroups() - 1, 1);

  //==========================================================================
  // Create and allocate arrays for infinite dilution cross sections
  inf_absorption = std::make_shared<xt::xtensor<double, 2>>();
  inf_absorption->resize({temperatures.size(), ndl.ngroups()});

  inf_transport_correction = std::make_shared<xt::xtensor<double, 2>>();
  inf_transport_correction->resize({temperatures.size(), ndl.ngroups()});

  inf_scatter = std::make_shared<xt::xtensor<double, 2>>();
  inf_scatter->resize({temperatures.size(), len_scat_data});

  if (grp.exist("inf-p1-scatter") && max_l >= 1) {
    inf_p1_scatter = std::make_shared<xt::xtensor<double, 2>>();
    inf_p1_scatter->resize({temperatures.size(), len_scat_data});
  }
  if (grp.exist("inf-p2-scatter") && max_l >= 2) {
    inf_p2_scatter = std::make_shared<xt::xtensor<double, 2>>();
    inf_p2_scatter->resize({temperatures.size(), len_scat_data});
  }
  if (grp.exist("inf-p3-scatter") && max_l >= 3) {
    inf_p3_scatter = std::make_shared<xt::xtensor<double, 2>>();
    inf_p3_scatter->resize({temperatures.size(), len_scat_data});
  }

  if (grp.exist("inf-(n,gamma)")) {
    const auto dims = grp.getDataSet("inf-(n,gamma)").getDimensions();
    inf_n_gamma = std::make_shared<xt::xtensor<double, 2>>();
    inf_n_gamma->resize({temperatures.size(), dims[1]});
  }

  if (grp.exist("inf-(n,2n)")) {
    const auto dims = grp.getDataSet("inf-(n,2n)").getDimensions();
    inf_n_2n = std::make_shared<xt::xtensor<double, 2>>();
    inf_n_2n->resize({temperatures.size(), dims[1]});
  }

  if (grp.exist("inf-(n,3n)")) {
    const auto dims = grp.getDataSet("inf-(n,3n)").getDimensions();
    inf_n_3n = std::make_shared<xt::xtensor<double, 2>>();
    inf_n_3n->resize({temperatures.size(), dims[1]});
  }

  if (grp.exist("inf-(n,a)")) {
    const auto dims = grp.getDataSet("inf-(n,a)").getDimensions();
    inf_n_a = std::make_shared<xt::xtensor<double, 2>>();
    inf_n_a->resize({temperatures.size(), dims[1]});
  }

  if (grp.exist("inf-(n,p)")) {
    const auto dims = grp.getDataSet("inf-(n,p)").getDimensions();
    inf_n_p = std::make_shared<xt::xtensor<double, 2>>();
    inf_n_p->resize({temperatures.size(), dims[1]});
  }

  if (this->fissile) {
    inf_fission = std::make_shared<xt::xtensor<double, 2>>();
    inf_fission->resize({temperatures.size(), ndl.ngroups()});
    nu = std::make_shared<xt::xtensor<double, 1>>();
    nu->resize({ndl.ngroups()});
    chi = std::make_shared<xt::xtensor<double, 1>>();
    chi->resize({ndl.ngroups()});
  }

  //==========================================================================
  // Read in data
  grp.getDataSet("inf-absorption").read_raw<double>(inf_absorption->data());
  grp.getDataSet("inf-transport-correction")
      .read_raw<double>(inf_transport_correction->data());
  grp.getDataSet("inf-scatter").read_raw<double>(inf_scatter->data());
  if (grp.exist("inf-p1-scatter") && max_l >= 1) {
    grp.getDataSet("inf-p1-scatter").read_raw<double>(inf_p1_scatter->data());
  }
  if (grp.exist("inf-p2-scatter") && max_l >= 2) {
    grp.getDataSet("inf-p2-scatter").read_raw<double>(inf_p2_scatter->data());
  }
  if (grp.exist("inf-p3-scatter") && max_l >= 3) {
    grp.getDataSet("inf-p3-scatter").read_raw<double>(inf_p3_scatter->data());
  }
  if (this->fissile) {
    grp.getDataSet("inf-fission").read_raw<double>(inf_fission->data());
    grp.getDataSet("nu").read_raw<double>(nu->data());
    grp.getDataSet("chi").read_raw<double>(chi->data());
  }
  if (grp.exist("inf-(n,gamma)")) {
    grp.getDataSet("inf-(n,gamma)").read_raw<double>(inf_n_gamma->data());
  }
  if (grp.exist("inf-(n,2n)")) {
    grp.getDataSet("inf-(n,2n)").read_raw<double>(inf_n_2n->data());
  }
  if (grp.exist("inf-(n,3n)")) {
    grp.getDataSet("inf-(n,3n)").read_raw<double>(inf_n_3n->data());
  }
  if (grp.exist("inf-(n,a)")) {
    grp.getDataSet("inf-(n,a)").read_raw<double>(inf_n_a->data());
  }
  if (grp.exist("inf-(n,p)")) {
    grp.getDataSet("inf-(n,p)").read_raw<double>(inf_n_p->data());
  }
}

void NuclideHandle::load_res_data(const H5::Group& grp, std::size_t max_l) {
  //==========================================================================
  // Create and allocate arrays
  // Start by getting the dimensions. dims[0] should be number of temps
  // dims[1] should be number of dilutions
  // dims[2] should be number of resonant groups
  auto dims = grp.getDataSet("res-absorption").getDimensions();
  res_absorption = std::make_shared<xt::xtensor<double, 3>>();
  res_absorption->resize({dims[0], dims[1], dims[2]});

  res_transport_correction = std::make_shared<xt::xtensor<double, 3>>();
  res_transport_correction->resize({dims[0], dims[1], dims[2]});

  if (this->fissile) {
    res_fission = std::make_shared<xt::xtensor<double, 3>>();
    res_fission->resize({dims[0], dims[1], dims[2]});
  }

  res_scatter = std::make_shared<xt::xtensor<double, 3>>();
  // Get new dimensions as scatter matrices are compressed with odd shape
  dims = grp.getDataSet("res-scatter").getDimensions();
  res_scatter->resize({dims[0], dims[1], dims[2]});

  if (grp.exist("res-p1-scatter") && max_l >= 1) {
    res_p1_scatter = std::make_shared<xt::xtensor<double, 3>>();
    res_p1_scatter->resize({dims[0], dims[1], dims[2]});
  }
  if (grp.exist("res-p2-scatter") && max_l >= 2) {
    res_p2_scatter = std::make_shared<xt::xtensor<double, 3>>();
    res_p2_scatter->resize({dims[0], dims[1], dims[2]});
  }
  if (grp.exist("res-p3-scatter") && max_l >= 3) {
    res_p3_scatter = std::make_shared<xt::xtensor<double, 3>>();
    res_p3_scatter->resize({dims[0], dims[1], dims[2]});
  }

  if (grp.exist("res-(n,gamma)")) {
    const auto dims = grp.getDataSet("res-(n,gamma)").getDimensions();
    res_n_gamma = std::make_shared<xt::xtensor<double, 3>>();
    res_n_gamma->resize({dims[0], dims[1], dims[2]});
  }

  //==========================================================================
  // Read in data
  grp.getDataSet("res-absorption").read_raw<double>(res_absorption->data());
  grp.getDataSet("res-transport-correction")
      .read_raw<double>(res_transport_correction->data());
  grp.getDataSet("res-scatter").read_raw<double>(res_scatter->data());
  if (grp.exist("res-p1-scatter") && max_l >= 1) {
    grp.getDataSet("res-p1-scatter").read_raw<double>(res_p1_scatter->data());
  }
  if (grp.exist("res-p2-scatter") && max_l >= 2) {
    grp.getDataSet("res-p2-scatter").read_raw<double>(res_p2_scatter->data());
  }
  if (grp.exist("res-p3-scatter") && max_l >= 3) {
    grp.getDataSet("res-p3-scatter").read_raw<double>(res_p3_scatter->data());
  }
  if (this->fissile) {
    grp.getDataSet("res-fission").read_raw<double>(res_fission->data());
  }
  if (grp.exist("res-(n,gamma)")) {
    grp.getDataSet("res-(n,gamma)").read_raw<double>(res_n_gamma->data());
  }
}

void NuclideHandle::unload() {
  packing = nullptr;

  chi = nullptr;
  nu = nullptr;

  inf_absorption = nullptr;
  inf_transport_correction = nullptr;
  inf_scatter = nullptr;
  inf_p1_scatter = nullptr;
  inf_p2_scatter = nullptr;
  inf_p3_scatter = nullptr;
  inf_fission = nullptr;
  inf_n_gamma = nullptr;
  inf_n_2n = nullptr;
  inf_n_3n = nullptr;
  inf_n_a = nullptr;
  inf_n_p = nullptr;

  res_absorption = nullptr;
  res_transport_correction = nullptr;
  res_scatter = nullptr;
  res_p1_scatter = nullptr;
  res_p2_scatter = nullptr;
  res_p3_scatter = nullptr;
  res_fission = nullptr;
  res_n_gamma = nullptr;
}

NDLibrary::NDLibrary()
    : fname_(),
      nuclide_handles_(),
      group_bounds_(),
      condensation_scheme_(std::nullopt),
      cmfd_condensation_scheme_(std::nullopt),
      library_(),
      group_structure_(),
      ngroups_(0),
      h5_(nullptr),
      depletion_chain_(nullptr) {
  // Get the environment variable
  const char* ndl_env = std::getenv(NDL_ENV_VAR);
  if (ndl_env == nullptr) {
    auto mssg = "Environment variable " NDL_ENV_VAR " is not set.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  // Get the string for the file name.
  std::string fname(ndl_env);

  // Make sure HDF5 file exists
  if (std::filesystem::exists(fname) == false) {
    std::stringstream mssg;
    mssg << "The file \"" << fname << "\" does not exist.";
    spdlog::error(mssg.str());
    throw ScarabeeException(mssg.str());
  }

  spdlog::info("Loading Nuclear Data Library from {}", fname);

  // Open the HDF5 file
  h5_ = std::make_shared<H5::File>(fname, H5::File::ReadOnly);

  this->init();

  // Save file name
  fname_ = fname;
}

NDLibrary::NDLibrary(const std::string& fname)
    : fname_(fname),
      nuclide_handles_(),
      group_bounds_(),
      condensation_scheme_(std::nullopt),
      cmfd_condensation_scheme_(std::nullopt),
      library_(),
      group_structure_(),
      ngroups_(0),
      h5_(nullptr),
      depletion_chain_(nullptr) {
  // Make sure HDF5 file exists
  if (std::filesystem::exists(fname_) == false) {
    std::stringstream mssg;
    mssg << "The file \"" << fname_ << "\" does not exist.";
    spdlog::error(mssg.str());
    throw ScarabeeException(mssg.str());
  }

  spdlog::info("Loading Nuclear Data Library from {}", fname_);

  // Open the HDF5 file
  h5_ = std::make_shared<H5::File>(fname_, H5::File::ReadOnly);

  this->init();
}

void NDLibrary::init() {
  // Get info on library
  if (h5_->hasAttribute("library"))
    library_ = h5_->getAttribute("library").read<std::string>();
  else
    library_ = "UNKNOWN";
  if (h5_->hasAttribute("group-structure"))
    group_structure_ = h5_->getAttribute("group-structure").read<std::string>();
  else
    group_structure_ = "UNKNOWN";
  if (h5_->hasAttribute("group-bounds"))
    group_bounds_ =
        h5_->getAttribute("group-bounds").read<std::vector<double>>();
  if (h5_->hasAttribute("ngroups"))
    ngroups_ = h5_->getAttribute("ngroups").read<std::size_t>();

  if (h5_->hasAttribute("first-resonance-group")) {
    first_resonant_group_ =
        h5_->getAttribute("first-resonance-group").read<std::size_t>();
  } else {
    const auto mssg =
        "No attribute \"first-resonance-group\" in nuclear data library.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  spdlog::info("Nuclear Data Library based on {} using {} group structure",
               library_, group_structure_);

  if (h5_->hasAttribute("last-resonance-group")) {
    last_resonant_group_ =
        h5_->getAttribute("last-resonance-group").read<std::size_t>();
  } else {
    const auto mssg =
        "No attribute \"last-resonance-group\" in nuclear data library.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  // If a default condensation scheme is provided, load it in now
  if (h5_->hasAttribute("condensation-scheme")) {
    const auto attr = h5_->getAttribute("condensation-scheme");
    const auto dims = attr.getMemSpace().getDimensions();

    // Make sure the dimensions of the array are correct
    if (dims.size() != 2 || dims[1] != 2) {
      const auto mssg =
          "Nuclear data library provided condensation scheme has an invalid "
          "shape.";
      spdlog::error(mssg);
      throw ScarabeeException(mssg);
    }

    // Read in the condensation scheme array
    xt::xtensor<double, 2> cond_scheme = xt::zeros<double>({dims[0], dims[1]});
    attr.read_raw<double>(cond_scheme.data());

    // Reconstruct vector of pairs from the array
    condensation_scheme_ =
        std::vector<std::pair<std::size_t, std::size_t>>(dims[0], {0, 0});
    for (std::size_t G = 0; G < dims[0]; G++) {
      (*condensation_scheme_)[G].first =
          static_cast<std::size_t>(cond_scheme(G, 0));
      (*condensation_scheme_)[G].second =
          static_cast<std::size_t>(cond_scheme(G, 1));
    }
  }

  // If a default CMFD condensation scheme is provided, load it in now
  if (h5_->hasAttribute("cmfd-condensation-scheme")) {
    const auto attr = h5_->getAttribute("cmfd-condensation-scheme");
    const auto dims = attr.getMemSpace().getDimensions();

    // Make sure the dimensions of the array are correct
    if (dims.size() != 2 || dims[1] != 2) {
      const auto mssg =
          "Nuclear data library provided CMFD condensation scheme has an "
          "invalid shape.";
      spdlog::error(mssg);
      throw ScarabeeException(mssg);
    }

    // Read in the condensation scheme array
    xt::xtensor<double, 2> cond_scheme = xt::zeros<double>({dims[0], dims[1]});
    attr.read_raw<double>(cond_scheme.data());

    // Reconstruct vector of pairs from the array
    cmfd_condensation_scheme_ =
        std::vector<std::pair<std::size_t, std::size_t>>(dims[0], {0, 0});
    for (std::size_t G = 0; G < dims[0]; G++) {
      (*cmfd_condensation_scheme_)[G].first =
          static_cast<std::size_t>(cond_scheme(G, 0));
      (*cmfd_condensation_scheme_)[G].second =
          static_cast<std::size_t>(cond_scheme(G, 1));
    }
  }

  // Read all nuclide handles
  auto nuc_names = h5_->listObjectNames();
  for (const auto& nuc : nuc_names) {
    // Make sure we don't try to read the depletion chain like a nuclide !
    if (nuc == "depletion-chain") continue;

    auto grp = h5_->getGroup(nuc);

    nuclide_handles_.emplace(std::make_pair(nuc, NuclideHandle()));
    auto& handle = nuclide_handles_.at(nuc);
    handle.name = nuc;

    // Read nuclide info
    handle.label = grp.getAttribute("label").read<std::string>();
    handle.temperatures =
        grp.getAttribute("temperatures").read<std::vector<double>>();
    handle.awr = grp.getAttribute("awr").read<double>();
    handle.potential_xs = grp.getAttribute("potential-xs").read<double>();

    // Intermediate resonance parameters
    if (grp.hasAttribute("ir-lambda")) {
      handle.ir_lambda =
          grp.getAttribute("ir-lambda").read<std::vector<double>>();
    } else {
      handle.ir_lambda = std::vector<double>(ngroups_, 1.);
    }

    if (grp.hasAttribute("fission-energy")) {
      handle.fission_energy = grp.getAttribute("fission-energy").read<double>();
    } else {
      handle.fission_energy = 0.;
    }

    handle.ZA = grp.getAttribute("ZA").read<std::uint32_t>();
    handle.fissile = grp.getAttribute("fissile").read<bool>();
    handle.resonant = grp.getAttribute("resonant").read<bool>();
    handle.dilutions =
        grp.getAttribute("dilutions").read<std::vector<double>>();
  }

  // Read the depletion chain, if present
  if (h5_->exist("depletion-chain")) {
    if (h5_->getObjectType("depletion-chain") != H5::ObjectType::Group) {
      const auto mssg =
          "The file member \"depletion-chain\" is present, but is not a group.";
      spdlog::error(mssg);
      throw ScarabeeException(mssg);
    }

    depletion_chain_ = std::make_shared<DepletionChain>();
    *depletion_chain_ =
        DepletionChain::from_hdf5_group(h5_->getGroup("depletion-chain"));
  }
}

const NuclideHandle& NDLibrary::get_nuclide(const std::string& name) const {
  if (name == "depletion-chain" ||
      nuclide_handles_.find(name) == nuclide_handles_.end()) {
    std::stringstream mssg;
    mssg << "Could not find nuclide by name of \"" << name << "\".";
    spdlog::error(mssg.str());
    throw ScarabeeException(mssg.str());
  }

  return nuclide_handles_.at(name);
}

NuclideHandle& NDLibrary::get_nuclide(const std::string& name) {
  if (name == "depletion-chain" ||
      nuclide_handles_.find(name) == nuclide_handles_.end()) {
    std::stringstream mssg;
    mssg << "Could not find nuclide by name of \"" << name << "\".";
    spdlog::error(mssg.str());
    throw ScarabeeException(mssg.str());
  }

  return nuclide_handles_.at(name);
}

void NDLibrary::unload() {
  for (auto& nuc_handle : nuclide_handles_) {
    nuc_handle.second.unload();
  }
}

std::pair<MicroNuclideXS, MicroDepletionXS> NDLibrary::infinite_dilution_xs(
    const std::string& name, const double temp, std::size_t max_l) {
  auto& nuc = this->get_nuclide(name);

  // Get temperature interpolation factors
  std::size_t it = 0;  // temperature index
  double f_temp = 0.;  // temperature interpolation factor
  get_temp_interp_params(temp, nuc, it, f_temp);

  if (nuc.loaded() == false) {
    nuc.load_xs_from_hdf5(*this, max_l);
  }

  //--------------------------------------------------------
  // Do transport correction
  xt::xtensor<double, 1> Dtr;
  this->interp_temp(Dtr, *nuc.inf_transport_correction, it, f_temp);

  //--------------------------------------------------------
  // Do absorption interpolation
  xt::xtensor<double, 1> Ea;
  this->interp_temp(Ea, *nuc.inf_absorption, it, f_temp);

  //--------------------------------------------------------
  // Create the scattering matrices
  if (max_l == 3 && nuc.inf_p3_scatter == nullptr) max_l--;
  if (max_l == 2 && nuc.inf_p2_scatter == nullptr) max_l--;
  if (max_l == 1 && nuc.inf_p1_scatter == nullptr) max_l--;
  xt::xtensor<double, 2> Es =
      xt::zeros<double>({max_l + 1, nuc.inf_scatter->shape()[1]});

  //--------------------------------------------------------
  // Do P0 scattering interpolation
  xt::xtensor<double, 1> temp_EsPl;
  this->interp_temp(temp_EsPl, *nuc.inf_scatter, it, f_temp);
  xt::view(Es, 0, xt::all()) = temp_EsPl;

  //--------------------------------------------------------
  // Do P1 scattering interpolation
  if (nuc.inf_p1_scatter) {
    this->interp_temp(temp_EsPl, *nuc.inf_p1_scatter, it, f_temp);
    xt::view(Es, 1, xt::all()) = temp_EsPl;
  }

  //--------------------------------------------------------
  // Do P2 scattering interpolation
  if (nuc.inf_p2_scatter && max_l >= 2) {
    this->interp_temp(temp_EsPl, *nuc.inf_p2_scatter, it, f_temp);
    xt::view(Es, 2, xt::all()) = temp_EsPl;
  }

  //--------------------------------------------------------
  // Do P3 scattering interpolation
  if (nuc.inf_p3_scatter && max_l >= 3) {
    this->interp_temp(temp_EsPl, *nuc.inf_p3_scatter, it, f_temp);
    xt::view(Es, 3, xt::all()) = temp_EsPl;
  }
  XS2D Es_xs2d(Es, *nuc.packing);

  //--------------------------------------------------------
  // Do fission interpolation
  xt::xtensor<double, 1> Ef = xt::zeros<double>({ngroups_});
  xt::xtensor<double, 1> nu = xt::zeros<double>({ngroups_});
  xt::xtensor<double, 1> chi = xt::zeros<double>({ngroups_});
  if (nuc.fissile) {
    this->interp_temp(Ef, *nuc.inf_fission, it, f_temp);
    nu = *nuc.nu;
    chi = *nuc.chi;
  }

  // Reconstruct total
  xt::xtensor<double, 1> Et = xt::zeros<double>({ngroups_});
  for (std::size_t g = 0; g < ngroups_; g++) {
    Et(g) = Ea(g) + Es_xs2d(0, g);
  }

  // Fill the basic XS data for the nuclide
  MicroNuclideXS nuc_xs;
  MicroDepletionXS dep_xs;
  nuc_xs.Et = XS1D(Et);
  nuc_xs.Dtr = XS1D(Dtr);
  nuc_xs.Es = Es_xs2d;
  nuc_xs.Ea = XS1D(Ea);
  nuc_xs.Ef = XS1D(Ef);
  nuc_xs.nu = XS1D(nu);
  nuc_xs.chi = XS1D(chi);

  // Now get supplementary depletion xs data
  if (nuc.fissile) {
    dep_xs.n_fission = nuc_xs.Ef;
  }

  if (nuc.inf_n_gamma) {
    xt::xtensor<double, 1> n_gamma =
        xt::zeros<double>({nuc.inf_n_gamma->shape()[1]});
    this->interp_temp(n_gamma, *nuc.inf_n_gamma, it, f_temp);
    dep_xs.n_gamma = XS1D(n_gamma);
  }

  if (nuc.inf_n_2n) {
    xt::xtensor<double, 1> n_2n = xt::zeros<double>({nuc.inf_n_2n->shape()[1]});
    this->interp_temp(n_2n, *nuc.inf_n_2n, it, f_temp);
    dep_xs.n_2n = XS1D(n_2n);
  }

  if (nuc.inf_n_3n) {
    xt::xtensor<double, 1> n_3n = xt::zeros<double>({nuc.inf_n_3n->shape()[1]});
    this->interp_temp(n_3n, *nuc.inf_n_3n, it, f_temp);
    dep_xs.n_3n = XS1D(n_3n);
  }

  if (nuc.inf_n_a) {
    xt::xtensor<double, 1> n_alpha =
        xt::zeros<double>({nuc.inf_n_a->shape()[1]});
    this->interp_temp(n_alpha, *nuc.inf_n_a, it, f_temp);
    dep_xs.n_alpha = XS1D(n_alpha);
  }

  if (nuc.inf_n_p) {
    xt::xtensor<double, 1> n_p = xt::zeros<double>({nuc.inf_n_p->shape()[1]});
    this->interp_temp(n_p, *nuc.inf_n_p, it, f_temp);
    dep_xs.n_p = XS1D(n_p);
  }

  return {nuc_xs, dep_xs};
}

ResonantOneGroupXS NDLibrary::dilution_xs(const std::string& name,
                                          std::size_t g, const double temp,
                                          const double dil, std::size_t max_l) {
  auto& nuc = this->get_nuclide(name);

  // Make sure nuclide is resonant
  if (nuc.resonant == false) {
    std::stringstream mssg;
    mssg << "Nuclide " << name
         << " is not resonant. Cannot obtain dilution cross section.";
    spdlog::error(mssg.str());
    throw ScarabeeException(mssg.str());
  }

  if (g < this->first_resonant_group() || this->last_resonant_group() < g) {
    std::stringstream mssg;
    mssg << "Group index " << g
         << " is not in the resonant region of the library.";
    spdlog::error(mssg.str());
    throw ScarabeeException(mssg.str());
  }

  // Transorm g from global energy index to resonant energy index
  const std::size_t g_res = g - this->first_resonant_group();

  // Get temperature interpolation factors
  std::size_t it = 0;  // temperature index
  double f_temp = 0.;  // temperature interpolation factor
  get_temp_interp_params(temp, nuc, it, f_temp);

  // Get dilution interpolation factors
  std::size_t id = 0;  // dilution index
  double f_dil = 0.;   // dilution interpolation factor
  get_dil_interp_params(dil, nuc, id, f_dil);

  if (nuc.loaded() == false) {
    nuc.load_xs_from_hdf5(*this, max_l);
  }

  ResonantOneGroupXS out;

  // Interpolate easy cross sections
  out.Dtr = this->interp_temp_dil(*nuc.res_transport_correction, g_res, it,
                                  f_temp, id, f_dil);
  out.Ea =
      this->interp_temp_dil(*nuc.res_absorption, g_res, it, f_temp, id, f_dil);
  out.Ef = 0.;
  if (nuc.res_fission) {
    out.Ef =
        this->interp_temp_dil(*nuc.res_fission, g_res, it, f_temp, id, f_dil);
  }
  // out.n_gamma is initially nullopt
  if (nuc.res_n_gamma) {
    out.n_gamma =
        this->interp_temp_dil(*nuc.res_n_gamma, g_res, it, f_temp, id, f_dil);
  }

  //--------------------------------------------------------
  // Create the scattering matrices
  if (max_l == 3 && nuc.res_p3_scatter == nullptr) max_l--;
  if (max_l == 2 && nuc.res_p2_scatter == nullptr) max_l--;
  if (max_l == 1 && nuc.res_p1_scatter == nullptr) max_l--;
  const std::size_t inf_start = (*nuc.packing)(g, 0);
  const std::size_t res_start =
      inf_start - (*nuc.packing)(first_resonant_group_, 0);
  const std::size_t g_min = (*nuc.packing)(g, 1);
  const std::size_t g_max = (*nuc.packing)(g, 2);
  const std::size_t scat_len = 1 + g_max - g_min;
  out.Es = xt::zeros<double>({max_l + 1, scat_len});
  out.gout_min = g_min;

  //--------------------------------------------------------
  // Do P0 scattering interpolation
  this->interp_temp_dil_views(
      xt::view(out.Es, 0, xt::all()),
      xt::view(*nuc.res_scatter, xt::all(), xt::all(),
               xt::range(res_start, res_start + scat_len)),
      it, f_temp, id, f_dil);

  //--------------------------------------------------------
  // Do P1 scattering interpolation
  if (nuc.res_p1_scatter) {
    this->interp_temp_dil_views(
        xt::view(out.Es, 1, xt::all()),
        xt::view(*nuc.res_p1_scatter, xt::all(), xt::all(),
                 xt::range(res_start, res_start + scat_len)),
        it, f_temp, id, f_dil);
  }

  //--------------------------------------------------------
  // Do P2 scattering interpolation
  if (nuc.res_p2_scatter && max_l >= 2) {
    this->interp_temp_dil_views(
        xt::view(out.Es, 2, xt::all()),
        xt::view(*nuc.res_p2_scatter, xt::all(), xt::all(),
                 xt::range(res_start, res_start + scat_len)),
        it, f_temp, id, f_dil);
  }

  //--------------------------------------------------------
  // Do P3 scattering interpolation
  if (nuc.res_p3_scatter && max_l >= 3) {
    this->interp_temp_dil_views(
        xt::view(out.Es, 3, xt::all()),
        xt::view(*nuc.res_p3_scatter, xt::all(), xt::all(),
                 xt::range(res_start, res_start + scat_len)),
        it, f_temp, id, f_dil);
  }

  return out;
}

ResonantOneGroupXS NDLibrary::two_term_xs(const std::string& name,
                                          std::size_t g, const double temp,
                                          const double b1, const double b2,
                                          const double bg_xs_1,
                                          const double bg_xs_2,
                                          std::size_t max_l) {
  auto& nuc = this->get_nuclide(name);

  // Make sure nuclide is resonant
  if (nuc.resonant == false) {
    std::stringstream mssg;
    mssg << "Nuclide " << name
         << " is not resonant. Cannot obtain dilution cross section.";
    spdlog::error(mssg.str());
    throw ScarabeeException(mssg.str());
  }

  if (g < this->first_resonant_group() || this->last_resonant_group() < g) {
    std::stringstream mssg;
    mssg << "Group index " << g
         << " is not in the resonant region of the library.";
    spdlog::error(mssg.str());
    throw ScarabeeException(mssg.str());
  }

  // See references [1] and [2] to understand this interpolation scheme, in
  // addition to the calculation of the flux based on the pot_xs and sig_a.

  // Get the two cross section sets
  const auto xs_1 = dilution_xs(name, g, temp, bg_xs_1, max_l);
  const auto xs_2 = dilution_xs(name, g, temp, bg_xs_2, max_l);
  const double ir_lambda = nuc.ir_lambda[g];
  const double lmbd_pot_xs = ir_lambda * nuc.potential_xs;
  const double lmbd_Es1 =
      ir_lambda * xt::sum(xt::view(xs_1.Es, 0, xt::all()))();
  const double lmbd_Es2 =
      ir_lambda * xt::sum(xt::view(xs_2.Es, 0, xt::all()))();

  // Calculate the two flux values. This formula is different from that given
  // in [1] or [2]. This is baed on a more standard IR approximation where
  // \varphi(E) = (1/E)*(\lambda\sigma_p + \sigma_0) /
  //                    (\sigma_a(E) + \lambda\sigma_s(E) + \sigma_0)
  // Check Gibson in refs [2, 3] for some details and hints on how to do this
  // derivation for yourself.
  const double flux_1_g =
      (lmbd_pot_xs + bg_xs_1) / (xs_1.Ea + lmbd_Es1 + bg_xs_1);
  const double flux_2_g =
      (lmbd_pot_xs + bg_xs_2) / (xs_2.Ea + lmbd_Es2 + bg_xs_2);

  // Calculate the two weighting factors
  const double f1_g = b1 * flux_1_g / (b1 * flux_1_g + b2 * flux_2_g);
  const double f2_g = b2 * flux_2_g / (b1 * flux_1_g + b2 * flux_2_g);

  // Compute the xs values
  ResonantOneGroupXS out;
  out.Dtr = f1_g * xs_1.Dtr + f2_g * xs_2.Dtr;
  out.Ea = f1_g * xs_1.Ea + f2_g * xs_2.Ea;
  out.Ef = f1_g * xs_1.Ef + f2_g * xs_2.Ef;
  out.Es = f1_g * xs_1.Es + f2_g * xs_2.Es;
  out.gout_min = xs_1.gout_min;
  if (xs_1.n_gamma) {
    out.n_gamma = f1_g * xs_1.n_gamma.value() + f2_g * xs_2.n_gamma.value();
  }

  return out;
}

ResonantOneGroupXS NDLibrary::ring_two_term_xs(
    const std::string& name, std::size_t g, const double temp, const double a1,
    const double a2, const double b1, const double b2, const double mat_pot_xs,
    const double N, const double Rfuel, const double Rin, const double Rout,
    std::size_t max_l) {
  if (Rin >= Rout) {
    auto mssg = "Rin must be < Rout.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  if (Rout > Rfuel) {
    auto mssg = "Rout must be < Rfuel.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  const auto& nuclide = get_nuclide(name);
  const double ir_lambda = nuclide.ir_lambda[g];
  const double lmbd_pot_xs = ir_lambda * nuclide.potential_xs;
  const double macro_lmbd_pot_xs = N * lmbd_pot_xs;

  if (max_l == 3 && nuclide.inf_p3_scatter == nullptr) max_l--;
  if (max_l == 2 && nuclide.inf_p2_scatter == nullptr) max_l--;
  if (max_l == 1 && nuclide.inf_p1_scatter == nullptr) max_l--;

  ResonantOneGroupXS out;
  double denom = 0.;

  for (std::size_t m = 1; m <= 4; m++) {
    const std::pair<double, double> eta_lm = this->eta_lm(m, Rfuel, Rin, Rout);
    const double eta_m = eta_lm.first;
    const double l_m = eta_lm.second;

    // If eta_m is zero, then this m has no contribution to the xs.
    // This happens for the last ring of a pin.
    if (eta_m == 0.) continue;

    // Calculate the background xs
    const double bg_xs_1 =
        l_m > 0. ? (mat_pot_xs - macro_lmbd_pot_xs + a1 / l_m) / N : 1.E10;
    const double bg_xs_2 =
        l_m > 0. ? (mat_pot_xs - macro_lmbd_pot_xs + a2 / l_m) / N : 1.E10;

    // Get the two cross section sets
    const auto xs_1 = dilution_xs(name, g, temp, bg_xs_1, max_l);
    const auto xs_2 = dilution_xs(name, g, temp, bg_xs_2, max_l);
    const double lmbd_Es1 =
        ir_lambda * xt::sum(xt::view(xs_1.Es, 0, xt::all()))();
    const double lmbd_Es2 =
        ir_lambda * xt::sum(xt::view(xs_2.Es, 0, xt::all()))();

    // Calculate the two flux values. This formula is different from that given
    // in [1] or [2]. This is baed on a more standard IR approximation where
    // \varphi(E) = (1/E)*(\lambda\sigma_p + \sigma_0) /
    //                    (\sigma_a(E) + \lambda\sigma_s(E) + \sigma_0)
    // Check Gibson in refs [2, 3] for some details and hints on how to do this
    // derivation for yourself.
    const double flux_1_g =
        (lmbd_pot_xs + bg_xs_1) / (xs_1.Ea + lmbd_Es1 + bg_xs_1);
    const double flux_2_g =
        (lmbd_pot_xs + bg_xs_2) / (xs_2.Ea + lmbd_Es2 + bg_xs_2);

    // Add contributions to the denominator
    denom += eta_m * (b1 * flux_1_g + b2 * flux_2_g);

    // Before adding contributions, must set the scatter array to zero on m = 1
    if (m == 1) {
      out.Es = xt::zeros<double>(xs_1.Es.shape());
    }

    // Add contributions to the xs
    // Compute the xs values
    out.Dtr += eta_m * (b1 * flux_1_g * xs_1.Dtr + b2 * flux_2_g * xs_2.Dtr);
    out.Ea += eta_m * (b1 * flux_1_g * xs_1.Ea + b2 * flux_2_g * xs_2.Ea);
    out.Ef += eta_m * (b1 * flux_1_g * xs_1.Ef + b2 * flux_2_g * xs_2.Ef);
    out.Es += eta_m * (b1 * flux_1_g * xs_1.Es + b2 * flux_2_g * xs_2.Es);
    out.gout_min = xs_1.gout_min;
    if (xs_1.n_gamma && (out.n_gamma.has_value() == false)) out.n_gamma = 0.;
    if (out.n_gamma) {
      (*out.n_gamma) += eta_m * (b1 * flux_1_g * xs_1.n_gamma.value() +
                                 b2 * flux_2_g * xs_2.n_gamma.value());
    }
  }

  const double invs_denom = 1. / denom;
  out.Dtr *= invs_denom;
  out.Ea *= invs_denom;
  out.Ef *= invs_denom;
  out.Es *= invs_denom;
  if (out.n_gamma) {
    (*out.n_gamma) *= invs_denom;
  }

  return out;
}

void NDLibrary::get_temp_interp_params(double temp, const NuclideHandle& nuc,
                                       std::size_t& i, double& f) const {
  if (temp <= nuc.temperatures.front()) {
    i = 0;
    f = 0.;
    return;
  } else if (temp >= nuc.temperatures.back()) {
    i = nuc.temperatures.size() - 2;
    f = 1.;
    return;
  }

  for (i = 0; i < nuc.temperatures.size() - 1; i++) {
    double T_i = nuc.temperatures[i];
    double T_i1 = nuc.temperatures[i + 1];

    if (temp >= T_i && temp <= T_i1) {
      f = (std::sqrt(temp) - std::sqrt(T_i)) /
          (std::sqrt(T_i1) - std::sqrt(T_i));
      break;
    }
  }
  if (f < 0.)
    f = 0.;
  else if (f > 1.)
    f = 1.;
}

void NDLibrary::get_dil_interp_params(double dil, const NuclideHandle& nuc,
                                      std::size_t& i, double& f) const {
  if (dil <= nuc.dilutions.front()) {
    i = 0;
    f = 0.;
    return;
  } else if (dil >= nuc.dilutions.back()) {
    i = nuc.dilutions.size() - 2;
    f = 1.;
    return;
  }

  for (i = 0; i < nuc.dilutions.size() - 1; i++) {
    double d_i = nuc.dilutions[i];
    double d_i1 = nuc.dilutions[i + 1];

    if (dil >= d_i && dil <= d_i1) {
      f = (dil - d_i) / (d_i1 - d_i);
      break;
    }
  }
  if (f < 0.)
    f = 0.;
  else if (f > 1.)
    f = 1.;
}

void NDLibrary::interp_temp(xt::xtensor<double, 1>& E,
                            const xt::xtensor<double, 2>& nE, std::size_t it,
                            double f_temp) const {
  if (f_temp > 0.) {
    E = (1. - f_temp) * xt::view(nE, it, xt::all()) +
        f_temp * xt::view(nE, it + 1, xt::all());
  } else {
    E = xt::view(nE, it, xt::all());
  }
}

double NDLibrary::interp_temp_dil(const xt::xtensor<double, 3>& nE,
                                  std::size_t g, std::size_t it, double f_temp,
                                  std::size_t id, double f_dil) const {
  double E = 0.;
  if (f_temp > 0.) {
    if (f_dil > 0.) {
      E = (1. - f_temp) *
              ((1. - f_dil) * nE(it, id, g) + f_dil * nE(it, id + 1, g)) +
          f_temp * ((1. - f_dil) * nE(it + 1, id, g) +
                    f_dil * nE(it + 1, id + 1, g));
    } else {
      E = (1. - f_temp) * nE(it, id, g) + f_temp * nE(it + 1, id, g);
    }
  } else {
    if (f_dil > 0.) {
      E = (1. - f_dil) * nE(it, id, g) + f_dil * nE(it, id + 1, g);
    } else {
      E = nE(it, id, g);
    }
  }

  return E;
}

std::pair<double, double> NDLibrary::eta_lm(std::size_t m, double Rfuel,
                                            double Rin, double Rout) const {
  if (m == 0 || m > 4) {
    auto mssg = "Invalid m.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  if (Rin >= Rout) {
    auto mssg = "Rin >= Rout.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  // Shouldn't need to check m, as this is a private method
  const double p_i = std::min(Rout / Rfuel, 1.);
  const double p_im = Rin / Rfuel;

  double p = p_i;
  if (m == 3 || m == 4) p = p_im;

  double theta = 0.5 * PI * p;
  if (m == 2 || m == 4) theta = -theta;

  // l = 4V_ring / S_pin = 4 pi (Rout^2 - Rin^2) / (2 pi Rfuel)
  const double l = 2. * (Rout * Rout - Rin * Rin) / Rfuel;

  const double T1 = std::sqrt(1. - p * p);
  const double T2 = Rin > 0. ? std::asin(p) / p : 1.;

  const double lm = (2. * Rfuel / PI) * (T1 + T2 + theta);

  double eta = p * lm / l;
  if (m == 2 || m == 3) eta = -eta;

  return {eta, lm};
}

}  // namespace scarabee

// References
// [1] H. Koike, K. Yamaji, K. Kirimura, D. Sato, H. Matsumoto, and A. Yamamoto,
//     “Advanced resonance self-shielding method for gray resonance treatment in
//     lattice physics code GALAXY,” J. Nucl. Sci. Technol., vol. 49, no. 7,
//     pp. 725–747, 2012, doi: 10.1080/00223131.2012.693885.
//
// [2] R. M. Ferrer and J. M. Hykes, “Spatially Dependent Resonance
//     Self-Shielding in CASMO5,” Nucl Sci Eng, vol. 197, no. 2, pp. 333–350,
//     2023, doi: 10.1080/00295639.2022.2053491.
//
// [3] N. Gibson, “Novel Resonance Self-Shielding Methods for Nuclear Reactor
//     Analysis,” Massachusetts Institute of Technology, 2016.

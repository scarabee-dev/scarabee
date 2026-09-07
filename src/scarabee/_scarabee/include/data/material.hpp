#ifndef SCARABEE_MATERIAL_H
#define SCARABEE_MATERIAL_H

#include <data/cross_section.hpp>
#include <data/micro_cross_sections.hpp>
#include <utils/serialization.hpp>

#include <cereal/cereal.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/vector.hpp>

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

struct MaterialCompositionPickler;
struct MaterialPickler;

namespace scarabee {

struct Nuclide {
  std::string name;
  double fraction;

 private:
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& arc) {
    arc(CEREAL_NVP(name), CEREAL_NVP(fraction));
  }
};

enum class Fraction : std::uint8_t { Atoms, Weight };

struct MaterialComposition {
  std::vector<Nuclide> nuclides;
  Fraction fractions;
  std::string name;

  MaterialComposition(Fraction f = Fraction::Atoms,
                      const std::string& name = "");

  void add_element(const std::string& name, double frac);
  void add_leu(double enrichment, double frac);
  void add_nuclide(const std::string& name, double frac);
  void add_nuclide(const Nuclide& nuc);

 private:
  friend class cereal::access;
  friend struct ::MaterialCompositionPickler;
  template <class Archive>
  void serialize(Archive& arc) {
    arc(CEREAL_NVP(nuclides), CEREAL_NVP(fractions), CEREAL_NVP(name));
  }
};

enum class DensityUnits : std::uint8_t { g_cm3, a_bcm, sum };

// Pre-declared for converting fractions
class NDLibrary;

class Material {
 public:
  Material(const MaterialComposition& comp, double temp,
           std::shared_ptr<NDLibrary> ndl);
  Material(const MaterialComposition& comp, double temp, double density,
           DensityUnits du, std::shared_ptr<NDLibrary> ndl);

  const MaterialComposition& composition() const { return composition_; }

  std::size_t size() const { return composition_.nuclides.size(); }

  const std::string& name() const { return name_; }
  void set_name(const std::string& new_name) { name_ = new_name; }

  std::size_t max_legendre_order() const { return max_l_; }
  void set_max_legendre_order(std::size_t max_l) { max_l_ = max_l; }

  bool has_component(const std::string& name) const;
  double atom_density(const std::string& name) const;

  double atoms_per_bcm() const { return atoms_per_bcm_; }
  double grams_per_cm3() const { return grams_per_cm3_; }
  double fissionable_grams_per_cm3() const;
  double potential_xs() const { return potential_xs_; }
  double temperature() const { return temperature_; }
  void set_temperature(double T);
  double average_molar_mass() const { return average_molar_mass_; }

  void clear_all_micro_xs_data();
  void clear_transport_micro_xs_data();
  void clear_depletion_micro_xs_data();

  bool has_transport_micro_xs_data() const {
    return micro_nuc_xs_data_.size() > 0;
  }
  bool has_depletion_micro_xs_data() const {
    return micro_dep_xs_data_.size() > 0;
  }

  bool fissile() const { return fissile_; }
  bool resonant() const { return resonant_; }

  std::shared_ptr<CrossSection> carlvik_xs(
      double C, double Ee, std::shared_ptr<NDLibrary> ndl,
      std::optional<std::size_t> max_l = std::nullopt);

  std::shared_ptr<CrossSection> roman_xs(
      double C, double Ee, std::shared_ptr<NDLibrary> ndl,
      std::optional<std::size_t> max_l = std::nullopt);

  std::shared_ptr<CrossSection> infinite_medium_xs(
      std::shared_ptr<NDLibrary> ndl,
      std::optional<std::size_t> max_l = std::nullopt);

  std::shared_ptr<CrossSection> dilution_xs(
      const std::vector<double>& dils, std::shared_ptr<NDLibrary> ndl,
      std::optional<std::size_t> max_l = std::nullopt);

  std::shared_ptr<CrossSection> ring_carlvik_xs(
      double C, double Rfuel, double Rin, double Rout,
      std::shared_ptr<NDLibrary> ndl,
      std::optional<std::size_t> max_l = std::nullopt);

  void load_nuclides(std::shared_ptr<NDLibrary> ndl) const;

  double compute_fission_power_density(
      std::span<const double> flux,
      const std::shared_ptr<const NDLibrary> ndl) const;

  std::vector<DepletionReactionRates> compute_depletion_reaction_rates(
      std::span<const double> flux,
      const std::shared_ptr<const NDLibrary> ndl) const;

 private:
  MaterialComposition composition_;
  std::string name_;
  double temperature_;
  double average_molar_mass_;
  double atoms_per_bcm_;
  double grams_per_cm3_;
  double potential_xs_;
  std::size_t max_l_{1};
  bool fissile_;
  bool resonant_;

  // Stored in the same order as in the MaterialComposition
  std::vector<MicroNuclideXS> micro_nuc_xs_data_;
  std::vector<MicroDepletionXS> micro_dep_xs_data_;

  double calc_avg_molar_mass(const NDLibrary& ndl) const;
  void normalize_fractions();

  void initialize_inf_dil_xs(std::shared_ptr<NDLibrary> ndl, std::size_t max_l);
  double lambda_pot_xs(std::shared_ptr<NDLibrary> ndl, std::size_t g);
  std::shared_ptr<CrossSection> create_xs_from_micro_data();
  void assign_resonant_xs(const std::size_t i, const std::size_t g,
                          const ResonantOneGroupXS& res_data);

  std::shared_ptr<CrossSection> two_term_xs(const double a1, const double a2,
                                            const double b1, const double b2,
                                            const double Ee,
                                            std::shared_ptr<NDLibrary> ndl,
                                            std::size_t max_l);

  friend class cereal::access;
  friend struct ::MaterialPickler;
  Material() {}
  template <class Archive>
  void serialize(Archive& arc) {
    arc(CEREAL_NVP(composition_), CEREAL_NVP(name_), CEREAL_NVP(temperature_),
        CEREAL_NVP(average_molar_mass_), CEREAL_NVP(atoms_per_bcm_),
        CEREAL_NVP(grams_per_cm3_), CEREAL_NVP(potential_xs_),
        CEREAL_NVP(max_l_), CEREAL_NVP(fissile_), CEREAL_NVP(resonant_),
        CEREAL_NVP(micro_nuc_xs_data_), CEREAL_NVP(micro_dep_xs_data_));
  }
};

enum class MixingFraction : std::uint8_t { Atoms, Weight, Volume };

std::shared_ptr<Material> mix_materials(
    const std::vector<std::shared_ptr<Material>>& mats,
    std::vector<double> fracs, MixingFraction f,
    std::shared_ptr<NDLibrary> ndl);

}  // namespace scarabee

#endif

#ifndef SCARABEE_NUCLIDE_NAMES_H
#define SCARABEE_NUCLIDE_NAMES_H

#include <cstdint>
#include <string>

namespace scarabee {

std::string nuclide_name_to_internal_name(const std::string& name);

std::string nuclide_name_to_simple_name(const std::string& name);

std::string nuclide_name_to_element_symbol(const std::string& name);

std::uint32_t nuclide_name_to_za(const std::string& name);

}  // namespace scarabee

#endif

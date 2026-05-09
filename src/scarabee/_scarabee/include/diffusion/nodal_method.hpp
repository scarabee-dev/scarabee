#ifndef SCARABEE_NODAL_METHOD_H
#define SCARABEE_NODAL_METHOD_H

#include <diffusion/nodal_cmfd_surface.hpp>

#include <array>
#include <cstddef>
#include <concepts>
#include <span>

namespace scarabee {

class Node;
class DiffusionCrossSection;

// Concept for NodalMethod
template <typename NM>
concept NodalMethod = requires(
    NM nm,                      // Nodal Method
    std::size_t NG,             // Number of groups
    std::span<Node> lnode,      // Node on left (-) side of surface
    std::span<Node> rnode,      // Node on right (+) side of surface
    const Side side,            // Side of lnode which is the treated surface
    std::span<const double> D,  // Physical diffusion coefficients for surface
    std::span<double> Dnl,      // Nonlinear diffusion coefficients to update
    const double B,             // Albedo for the boundary condition
    const std::array<double, 3> ld,    // Widths of the left node
    const std::array<double, 3> rd,    // Widths of the right node
    const double invs_keff,            // Latest estimate of keff
    const DiffusionCrossSection& lxs,  // XS of the left node
    const DiffusionCrossSection& rxs) {
  { NM(NG) };
  { NM::update_currents } -> std::convertible_to<bool>;
  { NM::reconstruct_flux } -> std::convertible_to<bool>;
  {
    nm.compute_keff_nonlinear_diffusion_coefficient(lnode, side, rnode, D, lxs,
                                                    rxs, ld, rd, Dnl, invs_keff)
  } -> std::same_as<double>;  // Two-Node Problem
  {
    nm.compute_keff_nonlinear_diffusion_coefficient(lnode, side, D, B, lxs, ld,
                                                    Dnl, invs_keff)
  } -> std::same_as<double>;  // One-Node Problem
};

}  // namespace scarabee

#endif

//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2023 - 2024 by the ryujin authors
//

#pragma once

#include "hyperbolic_system.h"

#include <compile_time_options.h>
#include <multicomponent_vector.h>
#include <simd.h>

namespace ryujin
{
  namespace ScalarConservation
  {
    template <typename ScalarNumber = double>
    class LimiterParameters : public dealii::ParameterAcceptor
    {
    public:
      LimiterParameters(const std::string &subsection = "/Limiter")
          : ParameterAcceptor(subsection)
      {
        iterations_ = 2;
        add_parameter(
            "iterations", iterations_, "Number of limiter iterations");

        relaxation_factor_ = ScalarNumber(1.);
        add_parameter("relaxation factor",
                      relaxation_factor_,
                      "Factor for scaling the relaxation window with r_i = "
                      "factor * (m_i/|Omega|)^(1.5/d).");
      }

      ACCESSOR_READ_ONLY(iterations);
      ACCESSOR_READ_ONLY(relaxation_factor);

    private:
      unsigned int iterations_;
      ScalarNumber relaxation_factor_;
    };


    /**
     * The convex limiter.
     *
     * @ingroup ScalarConservationEquations
     */
    template <int dim, typename Number = double>
    class Limiter
    {
    public:
      /**
       * @name Typedefs and constexpr constants
       */
      //@{

      using View = HyperbolicSystemView<dim, Number>;

      using ScalarNumber = typename View::ScalarNumber;

      static constexpr auto problem_dimension = View::problem_dimension;

      using state_type = typename View::state_type;

      using flux_contribution_type = typename View::flux_contribution_type;

      using precomputed_type = typename View::precomputed_type;

      using PrecomputedVector = typename View::PrecomputedVector;

      using Parameters = LimiterParameters<ScalarNumber>;

      //@}
      /**
       * @name Computation and manipulation of bounds
       */
      //
      //@{

      /**
       * The number of stored entries in the bounds array.
       */
      static constexpr unsigned int n_bounds = 2;

      /**
       * Array type used to store accumulated bounds.
       */
      using Bounds = std::array<Number, n_bounds>;

      /**
       * Constructor taking a HyperbolicSystem instance as argument
       */
      Limiter(const HyperbolicSystem &hyperbolic_system,
              const Parameters &parameters,
              const PrecomputedVector &precomputed_values)
          : hyperbolic_system(hyperbolic_system)
          , parameters(parameters)
          , precomputed_values(precomputed_values)
      {
      }

      /**
       * Given a state @p U_i and an index @p i return "strict" bounds,
       * i.e., a minimal convex set containing the state.
       */
      Bounds projection_bounds_from_state(const unsigned int i,
                                          const state_type &U_i) const;

      /**
       * Given two bounds bounds_left, bounds_right, this function computes
       * a larger, combined set of bounds that this is a (convex) superset
       * of the two.
       */
      Bounds combine_bounds(const Bounds &bounds_left,
                            const Bounds &bounds_right) const;

      //@}
      /**
       * @name Stencil-based computation of bounds
       *
       * Intended usage:
       * ```
       * Limiter<dim, Number> limiter;
       * for (unsigned int i = n_internal; i < n_owned; ++i) {
       *   // ...
       *   limiter.reset(i, U_i, flux_i);
       *   for (unsigned int col_idx = 1; col_idx < row_length; ++col_idx) {
       *     // ...
       *     limiter.accumulate(js, U_j, flux_j, scaled_c_ij, affine_shift);
       *   }
       *   limiter.bounds(hd_i);
       * }
       * ```
       */
      //@{

      /**
       * Reset temporary storage
       */
      void reset(const unsigned int i,
                 const state_type &U_i,
                 const flux_contribution_type &flux_i);

      /**
       * When looping over the sparsity row, add the contribution associated
       * with the neighboring state U_j.
       */
      void accumulate(const unsigned int *js,
                      const state_type &U_j,
                      const flux_contribution_type &flux_j,
                      const dealii::Tensor<1, dim, Number> &scaled_c_ij,
                      const state_type &affine_shift);

      /**
       * Return the computed bounds (with relaxation applied).
       */
      Bounds bounds(const Number hd_i) const;

      //*}
      /** @name Convex limiter */
      //@{

      /**
       * Given a state \f$\mathbf U\f$ and an update \f$\mathbf P\f$ this
       * function computes and returns the maximal coefficient \f$t\f$,
       * obeying \f$t_{\text{min}} < t < t_{\text{max}}\f$, such that the
       * selected local minimum principles are obeyed.
       */
      std::tuple<Number, bool> limit(const Bounds &bounds,
                                     const state_type &U,
                                     const state_type &P,
                                     const Number t_min = Number(0.),
                                     const Number t_max = Number(1.)) const;

    private:
      //@}
      /** @name Arguments and internal fields */
      //@{

      const HyperbolicSystem &hyperbolic_system;
      const Parameters &parameters;
      const PrecomputedVector &precomputed_values;

      state_type U_i;
      flux_contribution_type flux_i;

      Bounds bounds_;

      Number u_relaxation_numerator;
      Number u_relaxation_denominator;
      //@}
    };


    /*
     * -------------------------------------------------------------------------
     * Inline definitions
     * -------------------------------------------------------------------------
     */


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline auto
    Limiter<dim, Number>::projection_bounds_from_state(
        const unsigned int /*i*/, const state_type &U_i) const -> Bounds
    {
      const auto view = hyperbolic_system.view<dim, Number>();
      const auto u_i = view.state(U_i);
      return {/*u_min*/ u_i, /*u_max*/ u_i};
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline auto Limiter<dim, Number>::combine_bounds(
        const Bounds &bounds_left, const Bounds &bounds_right) const -> Bounds
    {
      const auto &[u_min_l, u_max_l] = bounds_left;
      const auto &[u_min_r, u_max_r] = bounds_right;

      return {std::min(u_min_l, u_min_r), std::max(u_max_l, u_max_r)};
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline void
    Limiter<dim, Number>::reset(const unsigned int /*i*/,
                                const state_type &new_U_i,
                                const flux_contribution_type &new_flux_i)
    {
      U_i = new_U_i;
      flux_i = new_flux_i;

      /* Bounds: */

      auto &[u_min, u_max] = bounds_;

      u_min = Number(std::numeric_limits<ScalarNumber>::max());
      u_max = Number(std::numeric_limits<ScalarNumber>::lowest());

      /* Relaxation: */

      u_relaxation_numerator = Number(0.);
      u_relaxation_denominator = Number(0.);
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline void Limiter<dim, Number>::accumulate(
        const unsigned int * /*js*/,
        const state_type &U_j,
        const flux_contribution_type &flux_j,
        const dealii::Tensor<1, dim, Number> &scaled_c_ij,
        const state_type &affine_shift)
    {
      const auto view = hyperbolic_system.view<dim, Number>();

      /* Bounds: */
      auto &[u_min, u_max] = bounds_;

      const auto u_i = view.state(U_i);
      const auto u_j = view.state(U_j);

      const auto U_ij_bar =
          ScalarNumber(0.5) * (U_i + U_j) -
          ScalarNumber(0.5) * contract(add(flux_j, -flux_i), scaled_c_ij) +
          affine_shift;

      const auto u_ij_bar = view.state(U_ij_bar);

      /* Bounds: */

      u_min = std::min(u_min, u_ij_bar);
      u_max = std::max(u_max, u_ij_bar);

      /* Relaxation: */

      /* Use a uniform weight. */
      const auto beta_ij = Number(1.);
      u_relaxation_numerator += beta_ij * (u_i + u_j);
      u_relaxation_denominator += std::abs(beta_ij);
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline auto
    Limiter<dim, Number>::bounds(const Number hd_i) const -> Bounds
    {
      auto relaxed_bounds = bounds_;
      auto &[u_min, u_max] = relaxed_bounds;

      /* Use r_i = factor * (m_i / |Omega|) ^ (1.5 / d): */

      Number r_i = std::sqrt(hd_i);                              // in 3D: ^ 3/6
      if constexpr (dim == 2)                                    //
        r_i = dealii::Utilities::fixed_power<3>(std::sqrt(r_i)); // in 2D: ^ 3/4
      else if constexpr (dim == 1)                               //
        r_i = dealii::Utilities::fixed_power<3>(r_i);            // in 1D: ^ 3/2
      r_i *= parameters.relaxation_factor();

      constexpr ScalarNumber eps = std::numeric_limits<ScalarNumber>::epsilon();
      const Number u_relaxation =
          std::abs(u_relaxation_numerator) /
          (std::abs(u_relaxation_denominator) + Number(eps));

      u_min = std::max(
          std::min((Number(1.) - r_i) * u_min, (Number(1.) + r_i) * u_min),
          u_min - ScalarNumber(2.) * u_relaxation);

      u_max = std::min(
          std::max((Number(1.) + r_i) * u_max, (Number(1.) - r_i) * u_max),
          u_max + ScalarNumber(2.) * u_relaxation);

      return relaxed_bounds;
    }
  } // namespace ScalarConservation
} // namespace ryujin

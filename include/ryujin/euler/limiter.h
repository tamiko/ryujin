//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/compile_time_options.h>

#include <ryujin/euler/hyperbolic_system.h>

#include <deal.II/base/config.h>
#include <ryujin/base/compile_time_options.h>
#include <ryujin/base/gpu.h>
#include <ryujin/base/newton.h>
#include <ryujin/base/observer_pointer.h>
#include <ryujin/base/simd.h>
#include <ryujin/linear_algebra/multicomponent_vector.h>

namespace ryujin
{
  namespace Euler
  {
    template <int dim,
              typename Number = double,
              typename MemorySpace = dealii::MemorySpace::Host>
    class LimiterView;

    template <typename ScalarNumber = double>
    class Limiter : public dealii::ParameterAcceptor
    {
    public:
      struct Parameters {
        unsigned int iterations;
        double newton_tolerance;
        unsigned int newton_max_iterations;
        double relaxation_factor;
      };

      Limiter(const HyperbolicSystem &hyperbolic_system,
              const std::string &subsection = "/Limiter")
          : ParameterAcceptor(subsection)
          , parameters_("euler_limiter_parameters",
                        TransferPolicy::implicit_transfers_host_resident)
          , hyperbolic_system_(&hyperbolic_system)
      {
        auto &parameters = *parameters_.view();

        parameters.iterations = 2;
        add_parameter("iterations",
                      parameters.iterations,
                      "Number of limiter iterations");

        if constexpr (std::is_same<ScalarNumber, double>::value)
          parameters.newton_tolerance = 1.e-10;
        else
          parameters.newton_tolerance = 1.e-4;
        add_parameter("newton tolerance",
                      parameters.newton_tolerance,
                      "Tolerance for the quadratic newton stopping criterion");

        parameters.newton_max_iterations = 2;
        add_parameter("newton max iterations",
                      parameters.newton_max_iterations,
                      "Maximal number of quadratic newton iterations performed "
                      "during limiting");

        parameters.relaxation_factor = 1.;
        add_parameter("relaxation factor",
                      parameters.relaxation_factor,
                      "Factor for scaling the relaxation window with r_i = "
                      "factor * (m_i/|Omega|)^(1.5/d).");

        ParameterAcceptor::parse_parameters_call_back.connect(
            [this] { parameters_.view(); });
      }

      template <int dim,
                typename Number,
                typename MemorySpace = dealii::MemorySpace::Host>
      auto view() const
      {
        return LimiterView<dim, Number, MemorySpace>{
            hyperbolic_system_->template view<dim, Number, MemorySpace>(),
            *this};
      }

    private:
      Mirrored<Parameters> parameters_;

      dealii::ObserverPointer<const HyperbolicSystem> hyperbolic_system_;

      template <int, typename, typename>
      friend class LimiterView;
    };


    template <int dim, typename Number, typename MemorySpace>
    class LimiterView
    {
    public:
      static_assert(
          std::is_same_v<MemorySpace, dealii::MemorySpace::Host> ||
              std::is_same_v<MemorySpace, dealii::MemorySpace::Default>,
          "Unexpected memory space");

      using View = HyperbolicSystemView<dim, Number, MemorySpace>;

      using ScalarNumber = typename View::ScalarNumber;

      using state_type = typename View::state_type;

      using flux_contribution_type = typename View::flux_contribution_type;

      using precomputed_type = typename View::precomputed_type;

      using PrecomputedVectorView = typename View::PrecomputedVectorView;

      static constexpr unsigned int n_bounds = 3;

      using Bounds = std::array<Number, n_bounds>;

      LimiterView(const View &view, const Limiter<ScalarNumber> &limiter)
          : view_(view)
          , parameters_(limiter.parameters_.template view<MemorySpace>())
      {
      }

      DEAL_II_HOST_DEVICE_ALWAYS_INLINE unsigned int iterations() const
      {
        return parameters_->iterations;
      }

      DEAL_II_HOST_DEVICE_ALWAYS_INLINE ScalarNumber newton_tolerance() const
      {
        return ScalarNumber(parameters_->newton_tolerance);
      }

      DEAL_II_HOST_DEVICE_ALWAYS_INLINE unsigned int
      newton_max_iterations() const
      {
        return parameters_->newton_max_iterations;
      }

      DEAL_II_HOST_DEVICE_ALWAYS_INLINE ScalarNumber relaxation_factor() const
      {
        return ScalarNumber(parameters_->relaxation_factor);
      }

      DEAL_II_HOST_DEVICE Bounds
      projection_bounds_from_state(const PrecomputedVectorView &pv,
                                   const unsigned int i,
                                   const state_type &U_i) const;

      DEAL_II_HOST_DEVICE Bounds combine_bounds(
          const Bounds &bounds_left, const Bounds &bounds_right) const;

      DEAL_II_HOST_DEVICE Bounds fully_relax_bounds(const Bounds &bounds,
                                                    const Number &hd) const;

      DEAL_II_HOST_DEVICE void reset(const PrecomputedVectorView &pv,
                                     const unsigned int i,
                                     const state_type &U_i,
                                     const flux_contribution_type &flux_i);

      DEAL_II_HOST_DEVICE void
      accumulate(const PrecomputedVectorView &pv,
                 const unsigned int *js,
                 const state_type &U_j,
                 const flux_contribution_type &flux_j,
                 const dealii::Tensor<1, dim, Number> &scaled_c_ij,
                 const state_type &affine_shift);

      DEAL_II_HOST_DEVICE Bounds bounds(const Number hd_i) const;

      DEAL_II_HOST_DEVICE std::tuple<Number, bool>
      limit(const Bounds &bounds,
            const state_type &U,
            const state_type &P,
            const Number t_min = Number(0.),
            const Number t_max = Number(1.)) const;

    private:
      const View view_;
      const Limiter<ScalarNumber>::Parameters *const parameters_;

      state_type U_i_;

      Bounds bounds_;

      Number rho_relaxation_numerator_;
      Number rho_relaxation_denominator_;
      Number s_interp_max_;
    };


    template <int dim, typename Number, typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE auto
    LimiterView<dim, Number, MemorySpace>::projection_bounds_from_state(
        const PrecomputedVectorView &pv,
        const unsigned int i,
        const state_type &U_i) const -> Bounds
    {
      const auto rho_i = view_.density(U_i);
      const auto &[s_i, eta_i] =
          pv.template read_tensor<Number, precomputed_type>(i);

      return {rho_i, rho_i, s_i};
    }


    template <int dim, typename Number, typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE auto
    LimiterView<dim, Number, MemorySpace>::combine_bounds(
        const Bounds &bounds_left, const Bounds &bounds_right) const -> Bounds
    {
      const auto &[rho_min_l, rho_max_l, s_min_l] = bounds_left;
      const auto &[rho_min_r, rho_max_r, s_min_r] = bounds_right;

      return {std::min(rho_min_l, rho_min_r),
              std::max(rho_max_l, rho_max_r),
              std::min(s_min_l, s_min_r)};
    }


    template <int dim, typename Number, typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE auto
    LimiterView<dim, Number, MemorySpace>::fully_relax_bounds(
        const Bounds &bounds, const Number &hd) const -> Bounds
    {
      auto relaxed_bounds = bounds;
      auto &[rho_min, rho_max, s_min] = relaxed_bounds;

      Number r = std::sqrt(hd);
      if constexpr (dim == 2)
        r = ryujin::fixed_power<3>(std::sqrt(r));
      else if constexpr (dim == 1)
        r = ryujin::fixed_power<3>(r);
      r *= relaxation_factor();

      constexpr ScalarNumber eps = std::numeric_limits<ScalarNumber>::epsilon();
      rho_min *= std::max(Number(1.) - r, Number(eps));
      rho_max *= (Number(1.) + r);
      s_min *= std::max(Number(1.) - r, Number(eps));

      return relaxed_bounds;
    }


    template <int dim, typename Number, typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE void
    LimiterView<dim, Number, MemorySpace>::reset(const PrecomputedVectorView &,
                                                 const unsigned int,
                                                 const state_type &U_i,
                                                 const flux_contribution_type &)
    {
      U_i_ = U_i;

      auto &[rho_min, rho_max, s_min] = bounds_;

      rho_min = Number(std::numeric_limits<ScalarNumber>::max());
      rho_max = Number(0.);
      s_min = Number(std::numeric_limits<ScalarNumber>::max());

      rho_relaxation_numerator_ = Number(0.);
      rho_relaxation_denominator_ = Number(0.);
      s_interp_max_ = Number(0.);
    }


    template <int dim, typename Number, typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE void
    LimiterView<dim, Number, MemorySpace>::accumulate(
        const PrecomputedVectorView &pv,
        const unsigned int *js,
        const state_type &U_j,
        const flux_contribution_type &,
        const dealii::Tensor<1, dim, Number> &scaled_c_ij,
        const state_type &affine_shift)
    {
      Assert(std::max(affine_shift.norm(), Number(0.)) == Number(0.),
             dealii::ExcNotImplemented());

      auto &[rho_min, rho_max, s_min] = bounds_;

      const auto rho_i = view_.density(U_i_);
      const auto m_i = view_.momentum(U_i_);
      const auto rho_j = view_.density(U_j);
      const auto m_j = view_.momentum(U_j);
      const auto rho_affine_shift = view_.density(affine_shift);

      const auto rho_ij_bar =
          ScalarNumber(0.5) * (rho_i + rho_j + (m_i - m_j) * scaled_c_ij) +
          rho_affine_shift;

      rho_min = std::min(rho_min, rho_ij_bar);
      rho_max = std::max(rho_max, rho_ij_bar);

      const auto &[s_j, eta_j] =
          pv.template read_tensor<Number, precomputed_type>(js);
      s_min = std::min(s_min, s_j);

      const auto beta_ij = Number(1.);
      rho_relaxation_numerator_ += beta_ij * (rho_i + rho_j);
      rho_relaxation_denominator_ += std::abs(beta_ij);

      const Number s_interp =
          view_.specific_entropy((U_i_ + U_j) * ScalarNumber(.5));
      s_interp_max_ = std::max(s_interp_max_, s_interp);
    }


    template <int dim, typename Number, typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE auto
    LimiterView<dim, Number, MemorySpace>::bounds(const Number hd_i) const
        -> Bounds
    {
      const auto &[rho_min, rho_max, s_min] = bounds_;

      auto relaxed_bounds = fully_relax_bounds(bounds_, hd_i);
      auto &[rho_min_relaxed, rho_max_relaxed, s_min_relaxed] = relaxed_bounds;

      constexpr ScalarNumber eps = std::numeric_limits<ScalarNumber>::epsilon();

      const auto rho_relaxation =
          ScalarNumber(2. * relaxation_factor()) *
          std::abs(rho_relaxation_numerator_) /
          (std::abs(rho_relaxation_denominator_) + Number(eps));

      const auto entropy_relaxation =
          relaxation_factor() * (s_interp_max_ - s_min);

      rho_min_relaxed = std::max(rho_min_relaxed, rho_min - rho_relaxation);
      rho_max_relaxed = std::min(rho_max_relaxed, rho_max + rho_relaxation);
      s_min_relaxed = std::max(s_min_relaxed, s_min - entropy_relaxation);

      return relaxed_bounds;
    }


    template <int dim, typename Number, typename MemorySpace>
    DEAL_II_HOST_DEVICE std::tuple<Number, bool>
    LimiterView<dim, Number, MemorySpace>::limit(const Bounds &bounds,
                                                 const state_type &U,
                                                 const state_type &P,
                                                 const Number t_min,
                                                 const Number t_max) const
    {
      bool success = true;
      Number t_r = t_max;

      constexpr ScalarNumber eps = std::numeric_limits<ScalarNumber>::epsilon();
      const auto small = view_.vacuum_state_relaxation_small();
      const auto large = view_.vacuum_state_relaxation_large();
      const ScalarNumber relax_small = ScalarNumber(1. + small * eps);
      const ScalarNumber relax = ScalarNumber(1. + large * eps);

      {
        const auto &rho_U = view_.density(U);
        const auto &rho_P = view_.density(P);

        const auto &rho_min = std::get<0>(bounds);
        const auto &rho_max = std::get<1>(bounds);

        const auto test_min = view_.filter_vacuum_density(
            std::max(Number(0.), rho_U - relax * rho_max));
        const auto test_max = view_.filter_vacuum_density(
            std::max(Number(0.), rho_min - relax * rho_U));
        if (!(test_min == Number(0.) && test_max == Number(0.))) {
#ifdef DEBUG_OUTPUT
          std::cout << std::fixed << std::setprecision(16);
          std::cout << "Bounds violation: low-order density (critical)!"
                    << "\n\t\trho min:         " << rho_min
                    << "\n\t\trho min (delta): "
                    << negative_part(rho_U - rho_min)
                    << "\n\t\trho:             " << rho_U
                    << "\n\t\trho max (delta): "
                    << positive_part(rho_U - rho_max)
                    << "\n\t\trho max:         " << rho_max << "\n"
                    << std::endl;
#endif
          success = false;
        }

        const Number denominator =
            ScalarNumber(1.) / (std::abs(rho_P) + eps * rho_max);

        constexpr auto lt = dealii::SIMDComparison::less_than;

        t_r = ryujin::compare_and_apply_mask<lt>(
            rho_max, rho_U + t_r * rho_P, (rho_max - rho_U) * denominator, t_r);

        t_r = ryujin::compare_and_apply_mask<lt>(
            rho_U + t_r * rho_P, rho_min, (rho_U - rho_min) * denominator, t_r);

        t_r = std::min(t_r, t_max);
        t_r = std::max(t_r, t_min);

#ifdef DEBUG_EXPENSIVE_BOUNDS_CHECK
        const auto rho_new = view_.density(U + t_r * P);
        const auto test_new_min = view_.filter_vacuum_density(
            std::max(Number(0.), rho_new - relax * rho_max));
        const auto test_new_max = view_.filter_vacuum_density(
            std::max(Number(0.), rho_min - relax * rho_new));
        if (!(test_new_min == Number(0.) && test_new_max == Number(0.))) {
#ifdef DEBUG_OUTPUT
          std::cout << std::fixed << std::setprecision(16);
          std::cout << "Bounds violation: high-order density!"
                    << "\n\t\trho min:         " << rho_min
                    << "\n\t\trho min (delta): "
                    << negative_part(rho_new - rho_min)
                    << "\n\t\trho:             " << rho_new
                    << "\n\t\trho max (delta): "
                    << positive_part(rho_new - rho_max)
                    << "\n\t\trho max:         " << rho_max << "\n"
                    << std::endl;
#endif
          success = false;
        }
#endif
      }

      Number t_l = t_min;

      const ScalarNumber gamma = view_.gamma();
      const ScalarNumber gp1 = gamma + ScalarNumber(1.);

      {
        const auto &s_min = std::get<2>(bounds);

#ifdef DEBUG_OUTPUT_LIMITER
        std::cout << std::endl;
        std::cout << std::fixed << std::setprecision(16);
        std::cout << "t_l: (start) " << t_l << std::endl;
        std::cout << "t_r: (start) " << t_r << std::endl;
#endif

        for (unsigned int n = 0; n < newton_max_iterations(); ++n) {

          const auto U_r = U + t_r * P;
          const auto rho_r = view_.density(U_r);
          const auto rho_r_gamma = ryujin::pow(rho_r, gamma);
          const auto rho_e_r = view_.internal_energy(U_r);

          auto psi_r =
              relax_small * rho_r * rho_e_r - s_min * rho_r * rho_r_gamma;

#ifndef DEBUG_EXPENSIVE_BOUNDS_CHECK
          t_l = ryujin::compare_and_apply_mask<
              dealii::SIMDComparison::greater_than>(
              psi_r, Number(0.), t_r, t_l);

          if (t_l == t_r) {
#ifdef DEBUG_OUTPUT_LIMITER
            std::cout << "shortcut: t_l == t_r" << std::endl;
            std::cout << "psi_l:       " << psi_l << std::endl;
            std::cout << "psi_r:       " << psi_r << std::endl;
            std::cout << "t_l: (  " << n << "  ) " << t_l << std::endl;
            std::cout << "t_r: (  " << n << "  ) " << t_r << std::endl;
#endif
            break;
          }
#endif

          const auto U_l = U + t_l * P;
          const auto rho_l = view_.density(U_l);
          const auto rho_l_gamma = ryujin::pow(rho_l, gamma);
          const auto rho_e_l = view_.internal_energy(U_l);

          auto psi_l =
              relax_small * rho_l * rho_e_l - s_min * rho_l * rho_l_gamma;

          const auto lower_bound =
              (ScalarNumber(1.) - relax) * s_min * rho_l * rho_l_gamma;
          if (n == 0 &&
              !(std::min(Number(0.), psi_l - lower_bound) == Number(0.))) {
#ifdef DEBUG_OUTPUT
            std::cout << std::fixed << std::setprecision(16);
            std::cout
                << "Bounds violation: low-order specific entropy (critical)!\n";
            std::cout << "\t\tPsi left: 0 <= " << psi_l << "\n" << std::endl;
#endif
            success = false;
          }

#ifdef DEBUG_EXPENSIVE_BOUNDS_CHECK
          t_l = ryujin::compare_and_apply_mask<
              dealii::SIMDComparison::greater_than>(
              psi_r, Number(0.), t_r, t_l);
#endif

          const Number tolerance(newton_tolerance());
          if (std::max(Number(0.), t_r - t_l - tolerance) == Number(0.)) {
#ifdef DEBUG_OUTPUT_LIMITER
            std::cout << "break: t_l and t_r within tolerance" << std::endl;
            std::cout << "psi_l:       " << psi_l << std::endl;
            std::cout << "psi_r:       " << psi_r << std::endl;
            std::cout << "t_l: (  " << n << "  ) " << t_l << std::endl;
            std::cout << "t_r: (  " << n << "  ) " << t_r << std::endl;
#endif
            break;
          }

          const auto drho = view_.density(P);
          const auto drho_e_l = view_.internal_energy_derivative(U_l) * P;
          const auto drho_e_r = view_.internal_energy_derivative(U_r) * P;
          const auto dpsi_l =
              rho_l * drho_e_l + (rho_e_l - gp1 * s_min * rho_l_gamma) * drho;
          const auto dpsi_r =
              rho_r * drho_e_r + (rho_e_r - gp1 * s_min * rho_r_gamma) * drho;

          quadratic_newton_step(
              t_l, t_r, psi_l, psi_r, dpsi_l, dpsi_r, Number(-1.));

#ifdef DEBUG_OUTPUT_LIMITER
          std::cout << "psi_l:       " << psi_l << std::endl;
          std::cout << "psi_r:       " << psi_r << std::endl;
          std::cout << "dpsi_l:      " << dpsi_l << std::endl;
          std::cout << "dpsi_r:      " << dpsi_r << std::endl;
          std::cout << "t_l: (  " << n << "  ) " << t_l << std::endl;
          std::cout << "t_r: (  " << n << "  ) " << t_r << std::endl;
#endif
        }

#ifdef DEBUG_EXPENSIVE_BOUNDS_CHECK
        {
          const auto U_new = U + t_l * P;
          const auto rho_new = view_.density(U_new);
          const auto rho_new_gamma = ryujin::pow(rho_new, gamma);
          const auto rho_e_new = view_.internal_energy(U_new);

          auto psi_new = relax_small * rho_new * rho_e_new -
                         s_min * rho_new * rho_new_gamma;

          const auto lower_bound =
              (ScalarNumber(1.) - relax) * s_min * rho_new * rho_new_gamma;

          const bool e_valid = std::min(Number(0.), rho_e_new) == Number(0.);
          const bool psi_valid =
              std::min(Number(0.), psi_new - lower_bound) == Number(0.);

          if (!e_valid || !psi_valid) {
#ifdef DEBUG_OUTPUT
            std::cout << std::fixed << std::setprecision(16);
            std::cout << "Bounds violation: high-order specific entropy!\n";
            std::cout << "\t\trho e: 0 <= " << rho_e_new << "\n";
            std::cout << "\t\tPsi:   0 <= " << psi_new << "\n" << std::endl;
#endif
            success = false;
          }
        }
#endif
      }

      return {t_l, success};
    }
  } // namespace Euler
} // namespace ryujin

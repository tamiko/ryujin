//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2024 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include "riemann_solver.h"

#include <newton.h>
#include <simd.h>

// #define DEBUG_RIEMANN_SOLVER

namespace ryujin
{
  namespace EulerAEOS
  {
    /*
     * The RiemannSolver is a guaranteed maximal wavespeed (GMS) estimate
     * for the extended Riemann problem outlined in
     * @cite ClaytonGuermondPopov-2022. For extenstions on handling negative
     * pressures, we follow @cite clayton2023robust (see §4.6).
     *
     * In contrast to the algorithm outlined in above reference the
     * algorithm takes a couple of shortcuts to significantly decrease the
     * computational footprint. These simplifications still guarantee that
     * we have an upper bound on the maximal wavespeed - but the number
     * bound might be larger. In particular:
     *
     *  - We do not check and treat the case phi(p_min) > 0. This
     *    corresponds to two expansion waves, see §5.2 in the reference. In
     *    this case we have
     *
     *      0 < p_star < p_min <= p_max.
     *
     *    And due to the fact that p_star < p_min the wavespeeds reduce to
     *    a left wavespeed v_L - a_L and right wavespeed v_R + a_R. This
     *    implies that it is sufficient to set p_2 to ANY value provided
     *    that p_2 <= p_min hold true in order to compute the correct
     *    wavespeed.
     *
     *    If p_2 > p_min then a more pessimistic bound is computed.
     *
     *  - FIXME: Simplification in p_star_RS
     */


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline Number
    RiemannSolver<dim, Number>::c(const Number &gamma) const
    {
      /*
       * We implement the continuous and monotonic function c(gamma) as
       * defined in (A.3) on page A469 of @cite ClaytonGuermondPopov-2022.
       * But with a simplified quick cut-off for the case gamma > 3:
       *
       *   c(gamma)^2 = 1                                    for gamma <= 5 / 3
       *   c(gamma)^2 = (3 * gamma + 11) / (6 * gamma + 6)   in between
       *   c(gamma)^2 = max(1/2, 5 / 6 - slope (gamma - 3))  for gamma > 3
       *
       * Due to the fact that the function is monotonic we can simply clip
       * the values without checking the conditions:
       */

      constexpr ScalarNumber slope =
          ScalarNumber(-0.34976871477801828189920753948709);

      const Number first_radicand = (ScalarNumber(3.) * gamma + Number(11.)) /
                                    (ScalarNumber(6.) * gamma + Number(6.));

      const Number second_radicand =
          Number(5. / 6.) + slope * (gamma - Number(3.));

      Number radicand = std::min(first_radicand, second_radicand);
      radicand = std::min(Number(1.), radicand);
      radicand = std::max(Number(1. / 2.), radicand);

      return std::sqrt(radicand);
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline Number RiemannSolver<dim, Number>::alpha(
        const Number &rho, const Number &gamma, const Number &a) const
    {
      const auto view = hyperbolic_system.view<dim, Number>();
      const auto interpolation_b = view.eos_interpolation_b();

      const Number numerator =
          ScalarNumber(2.) * a * (Number(1.) - interpolation_b * rho);

      const Number denominator = gamma - Number(1.);

      return safe_division(numerator, denominator);
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline Number
    RiemannSolver<dim, Number>::p_star_RS_full(
        const primitive_type &riemann_data_i,
        const primitive_type &riemann_data_j) const
    {
      const auto view = hyperbolic_system.view<dim, Number>();
      const auto pinf = view.eos_interpolation_pinfty();

      const auto &[rho_i, u_i, p_i, gamma_i, a_i] = riemann_data_i;
      const auto &[rho_j, u_j, p_j, gamma_j, a_j] = riemann_data_j;
      const auto alpha_i = alpha(rho_i, gamma_i, a_i);
      const auto alpha_j = alpha(rho_j, gamma_j, a_j);

      /*
       * First get p_min, p_max.
       *
       * Then, we get gamma_min/max, and alpha_min/max. Note that the
       * *_min/max values are associated with p_min/max and are not
       * necessarily the minimum/maximum of *_i vs *_j.
       */

      const Number p_min = std::min(p_i, p_j);
      const Number p_max = std::max(p_i, p_j);

      const Number gamma_min =
          dealii::compare_and_apply_mask<dealii::SIMDComparison::less_than>(
              p_i, p_j, gamma_i, gamma_j);

      const Number alpha_min =
          dealii::compare_and_apply_mask<dealii::SIMDComparison::less_than>(
              p_i, p_j, alpha_i, alpha_j);

      const Number alpha_hat_min = c(gamma_min) * alpha_min;

      const Number alpha_max = dealii::compare_and_apply_mask<
          dealii::SIMDComparison::greater_than_or_equal>(
          p_i, p_j, alpha_i, alpha_j);

      const Number gamma_m = std::min(gamma_i, gamma_j);
      const Number gamma_M = std::max(gamma_i, gamma_j);

      const Number numerator =
          dealii::compare_and_apply_mask<dealii::SIMDComparison::equal>(
              p_max + pinf,
              Number(0.),
              Number(0.),
              positive_part(alpha_hat_min + alpha_max - (u_j - u_i)));

      /*
       * The admissible set is p_min >= pinf. But numerically let's avoid
       * division by zero and ensure positivity:
       */
      const Number p_ratio = safe_division(p_min + pinf, p_max + pinf);

      /*
       * Here, we use a trick: The r-factor only shows up in the formula
       * for the case \gamma_min = \gamma_m, otherwise the r-factor
       * vanishes. We can accomplish this by using the following modified
       * exponent (where we substitute gamma_m by gamma_min):
       */
      const Number r_exponent =
          (gamma_M - gamma_min) / (ScalarNumber(2.) * gamma_min * gamma_M);

      /*
       * Compute (5.7) first formula for \tilde p_1^\ast and (5.8)
       * second formula for \tilde p_2^\ast at the same time:
       */

      const Number first_exponent =
          (gamma_M - Number(1.)) / (ScalarNumber(2.) * gamma_M);

      const Number first_exponent_inverse =
          safe_division(Number(1.), first_exponent);

      const Number first_denom =
          alpha_hat_min * ryujin::pow(p_ratio, r_exponent - first_exponent) +
          alpha_max;

      const Number p_1_tilde =
          (p_max + pinf) * ryujin::pow(safe_division(numerator, first_denom),
                                       first_exponent_inverse) -
          pinf;

#ifdef DEBUG_RIEMANN_SOLVER
      std::cout << "RS p_1_tilde  = " << p_1_tilde << "\n";
#endif

      /*
       * Compute (5.7) second formula for \tilde p_2^\ast and (5.8) first
       * formula for \tilde p_1^\ast at the same time:
       */

      const Number second_exponent =
          (gamma_m - Number(1.)) / (ScalarNumber(2.) * gamma_m);

      const Number second_exponent_inverse =
          safe_division(Number(1.), second_exponent);

      Number second_denom =
          alpha_hat_min * ryujin::pow(p_ratio, -second_exponent) +
          alpha_max * ryujin::pow(p_ratio, r_exponent);

      const Number p_2_tilde =
          (p_max + pinf) * ryujin::pow(safe_division(numerator, second_denom),
                                       second_exponent_inverse) -
          pinf;

#ifdef DEBUG_RIEMANN_SOLVER
      std::cout << "RS p_2_tilde  = " << p_2_tilde << "\n";
#endif

      return std::min(p_1_tilde, p_2_tilde);
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline Number
    RiemannSolver<dim, Number>::p_star_SS_full(
        const primitive_type &riemann_data_i,
        const primitive_type &riemann_data_j) const
    {
      const auto view = hyperbolic_system.view<dim, Number>();
      const auto pinf = view.eos_interpolation_pinfty();

      const auto &[rho_i, u_i, p_i, gamma_i, a_i] = riemann_data_i;
      const auto &[rho_j, u_j, p_j, gamma_j, a_j] = riemann_data_j;

      const Number gamma_m = std::min(gamma_i, gamma_j);

      const Number alpha_hat_i = c(gamma_i) * alpha(rho_i, gamma_i, a_i);
      const Number alpha_hat_j = c(gamma_j) * alpha(rho_j, gamma_j, a_j);

      /*
       * Compute (5.10) formula for \tilde p_1^\ast:
       *
       * Cost: 2x pow, 4x division, 0x sqrt
       */

      const Number exponent =
          (gamma_m - Number(1.)) / (ScalarNumber(2.) * gamma_m);
      const Number exponent_inverse = Number(1.) / exponent;

      const Number numerator =
          dealii::compare_and_apply_mask<dealii::SIMDComparison::equal>(
              p_j + pinf,
              Number(0.),
              Number(0.),
              positive_part(alpha_hat_i + alpha_hat_j - (u_j - u_i)));

      const Number denominator =
          alpha_hat_i *
              ryujin::pow(safe_division(p_i + pinf, p_j + pinf), -exponent) +
          alpha_hat_j;

      const Number p_1_tilde =
          (p_j + pinf) * ryujin::pow(safe_division(numerator, denominator),
                                     exponent_inverse) -
          pinf;

#ifdef DEBUG_RIEMANN_SOLVER
      std::cout << "SS p_1_tilde  = " << p_1_tilde << "\n";
#endif

      const auto p_2_tilde = p_star_failsafe(riemann_data_i, riemann_data_j);

      return std::min(p_1_tilde, p_2_tilde);
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline Number
    RiemannSolver<dim, Number>::p_star_failsafe(
        const primitive_type &riemann_data_i,
        const primitive_type &riemann_data_j) const
    {
      const auto view = hyperbolic_system.view<dim, Number>();
      const auto interpolation_b = view.eos_interpolation_b();
      const auto pinf = view.eos_interpolation_pinfty();

      const auto &[rho_i, u_i, p_i, gamma_i, a_i] = riemann_data_i;
      const auto &[rho_j, u_j, p_j, gamma_j, a_j] = riemann_data_j;

      /*
       * Compute (5.11) formula for \tilde p_2^\ast:
       *
       * Cost: 0x pow, 3x division, 3x sqrt
       */

      const Number p_max = std::max(p_i, p_j) + pinf;

      const Number radicand_i = safe_division(
          ScalarNumber(2.) * (Number(1.) - interpolation_b * rho_i) * p_max,
          rho_i * ((gamma_i + Number(1.)) * p_max +
                   (gamma_i - Number(1.)) * (p_i + pinf)));

      const Number x_i = std::sqrt(radicand_i);

      const Number radicand_j = safe_division(
          ScalarNumber(2.) * (Number(1.) - interpolation_b * rho_j) * p_max,
          rho_j * ((gamma_j + Number(1.)) * p_max +
                   (gamma_j - Number(1.)) * (p_j + pinf)));

      const Number x_j = std::sqrt(radicand_j);

      const Number a = x_i + x_j;
      const Number b =
          dealii::compare_and_apply_mask<dealii::SIMDComparison::equal>(
              a, Number(0.), Number(0.), u_j - u_i);

      const Number c = -(p_i + pinf) * x_i - (p_j + pinf) * x_j;

      const Number base = safe_division(
          std::abs(-b +
                   std::sqrt(positive_part(b * b - ScalarNumber(4.) * a * c))),
          std::abs(ScalarNumber(2.) * a));

      const Number p_2_tilde = base * base - pinf;

#ifdef DEBUG_RIEMANN_SOLVER
      std::cout << "SS p_2_tilde  = " << p_2_tilde << "\n";
#endif
      return p_2_tilde;
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline Number
    RiemannSolver<dim, Number>::p_star_interpolated(
        const primitive_type &riemann_data_i,
        const primitive_type &riemann_data_j) const
    {
      const auto view = hyperbolic_system.view<dim, Number>();
      const auto pinf = view.eos_interpolation_pinfty();

      const auto &[rho_i, u_i, p_i, gamma_i, a_i] = riemann_data_i;
      const auto &[rho_j, u_j, p_j, gamma_j, a_j] = riemann_data_j;
      const auto alpha_i = alpha(rho_i, gamma_i, a_i);
      const auto alpha_j = alpha(rho_j, gamma_j, a_j);

      /*
       * First get p_min, p_max.
       *
       * Then, we get gamma_min/max, and alpha_min/max. Note that the
       * *_min/max values are associated with p_min/max and are not
       * necessarily the minimum/maximum of *_i vs *_j.
       */

      const Number p_min = std::min(p_i, p_j) + pinf;
      const Number p_max = std::max(p_i, p_j) + pinf;

      const Number gamma_min =
          dealii::compare_and_apply_mask<dealii::SIMDComparison::less_than>(
              p_i, p_j, gamma_i, gamma_j);

      const Number alpha_min =
          dealii::compare_and_apply_mask<dealii::SIMDComparison::less_than>(
              p_i, p_j, alpha_i, alpha_j);

      const Number alpha_hat_min = c(gamma_min) * alpha_min;

      const Number gamma_max = dealii::compare_and_apply_mask<
          dealii::SIMDComparison::greater_than_or_equal>(
          p_i, p_j, gamma_i, gamma_j);

      const Number alpha_max = dealii::compare_and_apply_mask<
          dealii::SIMDComparison::greater_than_or_equal>(
          p_i, p_j, alpha_i, alpha_j);

      const Number alpha_hat_max = c(gamma_max) * alpha_max;

      const Number gamma_m = std::min(gamma_i, gamma_j);
      const Number gamma_M = std::max(gamma_i, gamma_j);

      const Number p_ratio = safe_division(p_min, p_max);

      /*
       * Here, we use a trick: The r-factor only shows up in the formula
       * for the case \gamma_min = \gamma_m, otherwise the r-factor
       * vanishes. We can accomplish this by using the following modified
       * exponent (where we substitute gamma_m by gamma_min):
       */
      const Number r_exponent =
          (gamma_M - gamma_min) / (ScalarNumber(2.) * gamma_min * gamma_M);

      /*
       * Compute a simultaneous upper bound on
       *   (5.7) second formula for \tilde p_2^\ast
       *   (5.8) first formula for \tilde p_1^\ast
       *   (5.11) formula for \tilde p_2^\ast
       */

      const Number exponent =
          (gamma_m - Number(1.)) / (ScalarNumber(2.) * gamma_m);
      const Number exponent_inverse = Number(1.) / exponent;

      const Number numerator =
          positive_part(alpha_hat_min + /*SIC!*/ alpha_max - (u_j - u_i));

      Number denominator = alpha_hat_min * ryujin::pow(p_ratio, -exponent) +
                           alpha_hat_max * ryujin::pow(p_ratio, r_exponent);

      const auto temp = safe_division(numerator, denominator);

      const Number p_tilde = p_max * ryujin::pow(temp, exponent_inverse) - pinf;

#ifdef DEBUG_RIEMANN_SOLVER
      std::cout << "IN p_*_tilde  = " << p_tilde << "\n";
#endif

      return p_tilde;
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline Number
    RiemannSolver<dim, Number>::f(const primitive_type &riemann_data,
                                  const Number p_star) const
    {
      constexpr ScalarNumber min = std::numeric_limits<ScalarNumber>::min();

      const auto view = hyperbolic_system.view<dim, Number>();
      const auto interpolation_b = view.eos_interpolation_b();
      const auto pinf = view.eos_interpolation_pinfty();

      const auto &[rho, u, p, gamma, a] = riemann_data;

      const Number one_minus_b_rho = Number(1.) - interpolation_b * rho;
      const Number gamma_minus_one = gamma - Number(1.);

      const Number Az =
          ScalarNumber(2.) * one_minus_b_rho / (rho * (gamma + Number(1.)));

      const Number Bz = gamma_minus_one / (gamma + Number(1.)) * (p + pinf);

      const Number radicand = safe_division(Az, p_star + pinf + Bz);

      /* true_value is shock case */
      const Number true_value = (p_star - p) * std::sqrt(radicand);

      const auto exponent = ScalarNumber(0.5) * gamma_minus_one / gamma;

      const Number ratio = safe_division(p_star + pinf, p + pinf);
      const Number factor = ryujin::pow(ratio, exponent) - Number(1.);

      /* false_value is rarefaction case */
      const auto false_value = ScalarNumber(2.) * a * one_minus_b_rho * factor /
                               std::max(gamma_minus_one, Number(min));

      return dealii::compare_and_apply_mask<
          dealii::SIMDComparison::greater_than_or_equal>(
          p_star, p, true_value, false_value);
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline Number
    RiemannSolver<dim, Number>::phi(const primitive_type &riemann_data_i,
                                    const primitive_type &riemann_data_j,
                                    const Number p_in) const
    {
      const Number &u_i = riemann_data_i[1];
      const Number &u_j = riemann_data_j[1];

      return f(riemann_data_i, p_in) + f(riemann_data_j, p_in) + u_j - u_i;
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline Number
    RiemannSolver<dim, Number>::phi_of_p_max(
        const primitive_type &riemann_data_i,
        const primitive_type &riemann_data_j) const
    {
      const auto view = hyperbolic_system.view<dim, Number>();
      const auto interpolation_b = view.eos_interpolation_b();
      const auto pinf = view.eos_interpolation_pinfty();

      const auto &[rho_i, u_i, p_i, gamma_i, a_i] = riemann_data_i;
      const auto &[rho_j, u_j, p_j, gamma_j, a_j] = riemann_data_j;

      const Number p_max = std::max(p_i, p_j) + pinf;

      const Number radicand_inverse_i =
          safe_division(ScalarNumber(0.5) * rho_i,
                        Number(1.) - interpolation_b * rho_i) *
          ((gamma_i + Number(1.)) * p_max +
           (gamma_i - Number(1.)) * (p_i + pinf));

      const Number value_i =
          safe_division(p_max - p_i, std::sqrt(radicand_inverse_i));

      const Number radicand_inverse_j =
          safe_division(ScalarNumber(0.5) * rho_j,
                        Number(1.) - interpolation_b * rho_j) *
          ((gamma_j + Number(1.)) * p_max +
           (gamma_j - Number(1.)) * (p_j + pinf));

      const Number value_j =
          safe_division(p_max - p_j, std::sqrt(radicand_inverse_j));

      return value_i + value_j + u_j - u_i;
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline Number
    RiemannSolver<dim, Number>::lambda1_minus(
        const primitive_type &riemann_data, const Number p_star) const
    {
      const auto view = hyperbolic_system.view<dim, Number>();
      const auto pinf = view.eos_interpolation_pinfty();

      const auto &[rho, u, p, gamma, a] = riemann_data;

      const auto factor =
          ScalarNumber(0.5) * (gamma + ScalarNumber(1.)) / gamma;

      const Number tmp = safe_division(positive_part(p_star - p), p + pinf);

      return u - a * std::sqrt(Number(1.) + factor * tmp);
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline Number
    RiemannSolver<dim, Number>::lambda3_plus(const primitive_type &riemann_data,
                                             const Number p_star) const
    {
      const auto view = hyperbolic_system.view<dim, Number>();
      const auto pinf = view.eos_interpolation_pinfty();

      const auto &[rho, u, p, gamma, a] = riemann_data;

      const auto factor =
          ScalarNumber(0.5) * (gamma + ScalarNumber(1.)) / gamma;

      const Number tmp = safe_division(positive_part(p_star - p), p + pinf);

      return u + a * std::sqrt(Number(1.) + factor * tmp);
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline Number
    RiemannSolver<dim, Number>::compute_lambda(
        const primitive_type &riemann_data_i,
        const primitive_type &riemann_data_j,
        const Number p_star) const
    {
      const Number nu_11 = lambda1_minus(riemann_data_i, p_star);
      const Number nu_32 = lambda3_plus(riemann_data_j, p_star);

      return std::max(positive_part(nu_32), negative_part(nu_11));
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline auto
    RiemannSolver<dim, Number>::riemann_data_from_state(
        const state_type &U,
        const Number &p,
        const dealii::Tensor<1, dim, Number> &n_ij) const -> primitive_type
    {
      const auto view = hyperbolic_system.view<dim, Number>();

      const auto rho = view.density(U);
      const auto rho_inverse = ScalarNumber(1.0) / rho;

      const auto m = view.momentum(U);
      const auto proj_m = n_ij * m;

      const auto gamma = view.surrogate_gamma(U, p);

      const auto interpolation_b = view.eos_interpolation_b();
      const auto pinf = view.eos_interpolation_pinfty();
      const auto x = Number(1.) - interpolation_b * rho;
      const auto a = std::sqrt(gamma * (p + pinf) / (rho * x));

#ifdef DEBUG_EXPENSIVE_BOUNDS_CHECK
      AssertThrowSIMD(
          Number(p + pinf),
          [](auto val) { return val >= ScalarNumber(0.); },
          dealii::ExcMessage("Internal error: p + pinf < 0."));

      AssertThrowSIMD(
          x,
          [](auto val) { return val > ScalarNumber(0.); },
          dealii::ExcMessage("Internal error: 1. - b * rho <= 0."));

      AssertThrowSIMD(
          gamma,
          [](auto val) { return val >= ScalarNumber(1.); },
          dealii::ExcMessage("Internal error: gamma < 1."));
#endif

      return {{rho, proj_m * rho_inverse, p, gamma, a}};
    }


    template <int dim, typename Number>
    Number RiemannSolver<dim, Number>::compute(
        const primitive_type &riemann_data_i,
        const primitive_type &riemann_data_j) const
    {
      const auto view = hyperbolic_system.view<dim, Number>();
      const auto pinf = view.eos_interpolation_pinfty();

      const auto &[rho_i, u_i, p_i, gamma_i, a_i] = riemann_data_i;
      const auto &[rho_j, u_j, p_j, gamma_j, a_j] = riemann_data_j;

#ifdef DEBUG_RIEMANN_SOLVER
      std::cout << "rho_left: " << rho_i << std::endl;
      std::cout << "u_left: " << u_i << std::endl;
      std::cout << "p_left: " << p_i << std::endl;
      std::cout << "gamma_left: " << gamma_i << std::endl;
      std::cout << "a_left: " << a_i << std::endl;
      std::cout << "rho_right: " << rho_j << std::endl;
      std::cout << "u_right: " << u_j << std::endl;
      std::cout << "p_right: " << p_j << std::endl;
      std::cout << "gamma_right: " << gamma_j << std::endl;
      std::cout << "a_right: " << a_j << std::endl;
#endif

      const Number p_max = std::max(p_i, p_j) + pinf;
      const Number phi_p_max = phi_of_p_max(riemann_data_i, riemann_data_j);

      if (!view.compute_strict_bounds()) {
#ifdef DEBUG_RIEMANN_SOLVER
        const Number p_star_RS = p_star_RS_full(riemann_data_i, riemann_data_j);
        const Number p_star_SS = p_star_SS_full(riemann_data_i, riemann_data_j);
        const Number p_debug =
            dealii::compare_and_apply_mask<dealii::SIMDComparison::less_than>(
                phi_p_max, Number(0.), p_star_SS, std::min(p_max, p_star_RS));
        std::cout << "   p^*_debug  = " << p_debug << "\n";
        std::cout << "   phi(p_*_d) = "
                  << phi(riemann_data_i, riemann_data_j, p_debug) << "\n";
        std::cout << "-> lambda_deb = "
                  << compute_lambda(riemann_data_i, riemann_data_j, p_debug)
                  << std::endl;
#endif

        const Number p_star_tilde =
            p_star_interpolated(riemann_data_i, riemann_data_j);
        const Number p_star_backup =
            p_star_failsafe(riemann_data_i, riemann_data_j);

        const Number p_2 =
            dealii::compare_and_apply_mask<dealii::SIMDComparison::less_than>(
                phi_p_max,
                Number(0.),
                std::min(p_star_tilde, p_star_backup),
                std::min(p_max, p_star_tilde));

#ifdef DEBUG_RIEMANN_SOLVER
        std::cout << "   p^*_tilde  = " << p_2 << "\n";
        std::cout << "   phi(p_*_t) = "
                  << phi(riemann_data_i, riemann_data_j, p_2) << "\n";
        std::cout << "-> lambda_max = "
                  << compute_lambda(riemann_data_i, riemann_data_j, p_2) << "\n"
                  << std::endl;
#endif

        return compute_lambda(riemann_data_i, riemann_data_j, p_2);
      }

      const Number p_star_RS = p_star_RS_full(riemann_data_i, riemann_data_j);
      const Number p_star_SS = p_star_SS_full(riemann_data_i, riemann_data_j);

      const Number p_2 =
          dealii::compare_and_apply_mask<dealii::SIMDComparison::less_than>(
              phi_p_max, Number(0.), p_star_SS, std::min(p_max, p_star_RS));

#ifdef DEBUG_RIEMANN_SOLVER
      std::cout << "   p^*_tilde  = " << p_2 << "\n";
      std::cout << "   phi(p_*_t) = "
                << phi(riemann_data_i, riemann_data_j, p_2) << "\n";
      std::cout << "-> lambda_max = "
                << compute_lambda(riemann_data_i, riemann_data_j, p_2)
                << std::endl;
#endif

      return compute_lambda(riemann_data_i, riemann_data_j, p_2);
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline Number RiemannSolver<dim, Number>::compute(
        const state_type &U_i,
        const state_type &U_j,
        const unsigned int i,
        const unsigned int *js,
        const dealii::Tensor<1, dim, Number> &n_ij) const
    {
      const auto &[p_i, unused_i, s_i, eta_i] =
          precomputed_values.template get_tensor<Number, precomputed_type>(i);

      const auto &[p_j, unused_j, s_j, eta_j] =
          precomputed_values.template get_tensor<Number, precomputed_type>(js);

      const auto riemann_data_i = riemann_data_from_state(U_i, p_i, n_ij);
      const auto riemann_data_j = riemann_data_from_state(U_j, p_j, n_ij);

      return compute(riemann_data_i, riemann_data_j);
    }


  } // namespace EulerAEOS
} // namespace ryujin

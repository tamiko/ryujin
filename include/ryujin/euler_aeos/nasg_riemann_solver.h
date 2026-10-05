//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/compile_time_options.h>

#include <ryujin/base/gpu.h>
#include <ryujin/base/newton.h>
#include <ryujin/base/simd.h>

#include <deal.II/base/parameter_acceptor.h>

#include <array>

namespace ryujin
{
  namespace EulerAEOS
  {
    struct NASGRiemannSolverOptions {
      bool covolume = true;

      bool pinf = true;

      bool safe_division = true;

      bool variable_gamma = true;
    };


    template <typename Number,
              NASGRiemannSolverOptions options = NASGRiemannSolverOptions{},
              typename MemorySpace = dealii::MemorySpace::Host>
    class NASGRiemannSolverView;


    template <typename ScalarNumber = double,
              NASGRiemannSolverOptions options = NASGRiemannSolverOptions{}>
    class NASGRiemannSolver : public dealii::ParameterAcceptor
    {
    public:
      struct Parameters {
        ScalarNumber covolume_b;
        ScalarNumber pinf;

        bool compute_expensive_bounds;
        double newton_tolerance;
        unsigned int newton_max_iterations;

        ScalarNumber gamma;
        ScalarNumber lambda_factor;
        ScalarNumber rarefaction_exponent;
        ScalarNumber rarefaction_exponent_inverse;
        ScalarNumber half_gamma_minus_one;
        ScalarNumber c_of_gamma;
      };

      NASGRiemannSolver(const std::string &subsection)
          : ParameterAcceptor(subsection)
          , parameters_("nasg_riemann_solver_parameters",
                        TransferPolicy::implicit_transfers_host_resident)
      {
        auto &parameters = *parameters_.view();

        if constexpr (std::is_same<ScalarNumber, double>::value)
          parameters.newton_tolerance = 1.e-10;
        else
          parameters.newton_tolerance = 1.e-4;
        add_parameter("newton tolerance",
                      parameters.newton_tolerance,
                      "Tolerance for the quadratic newton stopping criterion");

        parameters.newton_max_iterations = 0;
        add_parameter("newton max iterations",
                      parameters.newton_max_iterations,
                      "Maximal number of quadratic newton iterations performed "
                      "during limiting");

        parameters.covolume_b = ScalarNumber(0.);
        parameters.pinf = ScalarNumber(0.);
        parameters.compute_expensive_bounds = false;

        parameters.gamma = ScalarNumber(0.);
        parameters.lambda_factor = ScalarNumber(0.);
        parameters.rarefaction_exponent = ScalarNumber(0.);
        parameters.rarefaction_exponent_inverse = ScalarNumber(0.);
        parameters.half_gamma_minus_one = ScalarNumber(0.);
        parameters.c_of_gamma = ScalarNumber(0.);

        ParameterAcceptor::parse_parameters_call_back.connect(
            [this] { parameters_.view(); });
      }

      void set_gamma(const double gamma)
        requires(!options.variable_gamma);

      void set_equation_of_state(const double covolume_b,
                                 const double pinf,
                                 const bool compute_expensive_bounds);

      template <typename Number,
                typename MemorySpace = dealii::MemorySpace::Host>
      auto view() const
      {
        return NASGRiemannSolverView<Number, options, MemorySpace>{*this};
      }

    private:
      Mirrored<Parameters> parameters_;

      template <typename, NASGRiemannSolverOptions, typename>
      friend class NASGRiemannSolverView;
    };


    template <typename Number,
              NASGRiemannSolverOptions options,
              typename MemorySpace>
    class NASGRiemannSolverView
    {
    public:
      static_assert(
          std::is_same_v<MemorySpace, dealii::MemorySpace::Host> ||
              std::is_same_v<MemorySpace, dealii::MemorySpace::Default>,
          "Unexpected memory space");

      using ScalarNumber = typename get_value_type<Number>::type;

      using Parameters =
          typename NASGRiemannSolver<ScalarNumber, options>::Parameters;

      static constexpr unsigned int riemann_data_size = 5;

      using primitive_type = std::array<Number, riemann_data_size>;

      struct RiemannSolution {
        primitive_type riemann_data_left;
        primitive_type riemann_data_right;

        Number p_star;
        Number u_star;
        Number rho_star_left;
        Number rho_star_right;

        Number lambda1_minus;
        Number lambda1_plus;
        Number lambda3_minus;
        Number lambda3_plus;
      };

      NASGRiemannSolverView(
          const NASGRiemannSolver<ScalarNumber, options> &riemann_solver)
          : parameters_(riemann_solver.parameters_.template view<MemorySpace>())
      {
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

      DEAL_II_HOST_DEVICE_ALWAYS_INLINE ScalarNumber covolume_b() const
      {
        return parameters_->covolume_b;
      }

      DEAL_II_HOST_DEVICE_ALWAYS_INLINE ScalarNumber pinf() const
      {
        return parameters_->pinf;
      }

      DEAL_II_HOST_DEVICE_ALWAYS_INLINE bool compute_expensive_bounds() const
      {
        return parameters_->compute_expensive_bounds;
      }

      DEAL_II_HOST_DEVICE Number
      compute(const primitive_type &riemann_data_i,
              const primitive_type &riemann_data_j) const;

      DEAL_II_HOST_DEVICE RiemannSolution
      riemann_solution(const primitive_type &riemann_data_i,
                       const primitive_type &riemann_data_j,
                       const Number p_star) const;

      DEAL_II_HOST_DEVICE RiemannSolution
      solve(const primitive_type &riemann_data_i,
            const primitive_type &riemann_data_j,
            const unsigned int max_iterations = 100) const;

      DEAL_II_HOST_DEVICE primitive_type
      sample(const RiemannSolution &solution,
             const Number &xi,
             const unsigned int max_iterations = 100) const;

      DEAL_II_HOST_DEVICE Number
      rarefaction_fan_pressure(const primitive_type &riemann_data,
                               const Number &xi,
                               const ScalarNumber sign,
                               const unsigned int max_iterations) const;

      DEAL_II_HOST_DEVICE_ALWAYS_INLINE Number
      one_minus_b_rho(const Number &rho) const
      {
        if constexpr (options.covolume)
          return Number(1.) - covolume_b() * rho;
        else
          return Number(1.);
      }

      DEAL_II_HOST_DEVICE_ALWAYS_INLINE Number shift(const Number &p) const
      {
        if constexpr (options.pinf)
          return p + pinf();
        else
          return p;
      }

      DEAL_II_HOST_DEVICE_ALWAYS_INLINE Number unshift(const Number &p) const
      {
        if constexpr (options.pinf)
          return p - pinf();
        else
          return p;
      }

      DEAL_II_HOST_DEVICE_ALWAYS_INLINE Number
      safe_division(const Number &numerator, const Number &denominator) const
      {
        if constexpr (options.safe_division)
          return ryujin::safe_division(numerator, denominator);
        else
          return numerator / denominator;
      }

      DEAL_II_HOST_DEVICE Number speed_of_sound(const Number &rho,
                                                const Number &p,
                                                const Number &gamma) const;

      DEAL_II_HOST_DEVICE Number rho_star(const primitive_type &riemann_data,
                                          const Number &p_star) const;

      template <typename T>
      DEAL_II_HOST_DEVICE_ALWAYS_INLINE static T c(const T &gamma_Z);

      DEAL_II_HOST_DEVICE_ALWAYS_INLINE auto
      gamma_of(const primitive_type &riemann_data) const
      {
        if constexpr (options.variable_gamma)
          return riemann_data[3];
        else
          return parameters_->gamma;
      }

      DEAL_II_HOST_DEVICE_ALWAYS_INLINE auto
      lambda_factor(const primitive_type &riemann_data) const
      {
        if constexpr (options.variable_gamma) {
          const auto &gamma = riemann_data[3];
          return ScalarNumber(0.5) * (gamma + ScalarNumber(1.)) / gamma;
        } else
          return parameters_->lambda_factor;
      }

      DEAL_II_HOST_DEVICE_ALWAYS_INLINE auto
      rarefaction_exponent(const primitive_type &riemann_data) const
      {
        if constexpr (options.variable_gamma) {
          const auto &gamma = riemann_data[3];
          return ScalarNumber(0.5) * (gamma - Number(1.)) / gamma;
        } else
          return parameters_->rarefaction_exponent;
      }

      DEAL_II_HOST_DEVICE_ALWAYS_INLINE auto
      rarefaction_exponent_inverse(const primitive_type &riemann_data) const
      {
        if constexpr (options.variable_gamma) {
          const auto &gamma = riemann_data[3];
          return ScalarNumber(2.) * gamma / (gamma - Number(1.));
        } else
          return parameters_->rarefaction_exponent_inverse;
      }

      DEAL_II_HOST_DEVICE_ALWAYS_INLINE auto
      half_gamma_minus_one(const primitive_type &riemann_data) const
      {
        if constexpr (options.variable_gamma) {
          const auto &gamma = riemann_data[3];
          return ScalarNumber(0.5) * (gamma - Number(1.));
        } else
          return parameters_->half_gamma_minus_one;
      }

      DEAL_II_HOST_DEVICE_ALWAYS_INLINE auto
      c_of_gamma(const primitive_type &riemann_data) const
      {
        if constexpr (options.variable_gamma)
          return c(riemann_data[3]);
        else
          return parameters_->c_of_gamma;
      }

      DEAL_II_HOST_DEVICE Number alpha(const Number &rho,
                                       const Number &gamma,
                                       const Number &a) const;

      DEAL_II_HOST_DEVICE Number f(const primitive_type &riemann_data,
                                   const Number p_star) const;

      DEAL_II_HOST_DEVICE Number df(const primitive_type &riemann_data,
                                    const Number &p_star) const;

      DEAL_II_HOST_DEVICE Number phi(const primitive_type &riemann_data_i,
                                     const primitive_type &riemann_data_j,
                                     const Number p_in) const;

      DEAL_II_HOST_DEVICE Number dphi(const primitive_type &riemann_data_i,
                                      const primitive_type &riemann_data_j,
                                      const Number &p) const;

      DEAL_II_HOST_DEVICE Number
      phi_of_p_max(const primitive_type &riemann_data_i,
                   const primitive_type &riemann_data_j) const;

      DEAL_II_HOST_DEVICE Number lambda1_minus(
          const primitive_type &riemann_data, const Number p_star) const;

      DEAL_II_HOST_DEVICE Number lambda3_plus(
          const primitive_type &riemann_data, const Number p_star) const;

      DEAL_II_HOST_DEVICE Number
      p_star_upper_bound(const primitive_type &riemann_data_i,
                         const primitive_type &riemann_data_j,
                         const Number &phi_p_max) const;

      DEAL_II_HOST_DEVICE Number
      p_star_single_gamma(const primitive_type &riemann_data_i,
                          const primitive_type &riemann_data_j,
                          const Number &phi_p_max) const;

      DEAL_II_HOST_DEVICE Number
      p_star_interpolated(const primitive_type &riemann_data_i,
                          const primitive_type &riemann_data_j) const;

      DEAL_II_HOST_DEVICE Number
      p_star_RS_full(const primitive_type &riemann_data_i,
                     const primitive_type &riemann_data_j) const;

      DEAL_II_HOST_DEVICE Number
      p_star_SS_full(const primitive_type &riemann_data_i,
                     const primitive_type &riemann_data_j) const;

      DEAL_II_HOST_DEVICE Number
      p_star_failsafe(const primitive_type &riemann_data_i,
                      const primitive_type &riemann_data_j) const;

      DEAL_II_HOST_DEVICE Number
      p_star_two_rarefaction(const primitive_type &riemann_data_i,
                             const primitive_type &riemann_data_j) const;

      DEAL_II_HOST_DEVICE void newton_step(const primitive_type &riemann_data_i,
                                           const primitive_type &riemann_data_j,
                                           Number &p_1,
                                           Number &p_2) const;

      DEAL_II_HOST_DEVICE std::array<Number, 2>
      compute_gap(const primitive_type &riemann_data_i,
                  const primitive_type &riemann_data_j,
                  const Number p_1,
                  const Number p_2) const;

      DEAL_II_HOST_DEVICE Number
      compute_lambda_max(const primitive_type &riemann_data_i,
                         const primitive_type &riemann_data_j,
                         const Number p_star) const;


    private:
      const Parameters *parameters_;

      template <typename, NASGRiemannSolverOptions>
      friend class NASGRiemannSolver;
    };


    template <typename ScalarNumber, NASGRiemannSolverOptions options>
    inline void
    NASGRiemannSolver<ScalarNumber, options>::set_gamma(const double gamma)
      requires(!options.variable_gamma)
    {
      auto &parameters = *parameters_.view();

      parameters.gamma = ScalarNumber(gamma);
      parameters.lambda_factor = ScalarNumber(0.5 * (gamma + 1.) / gamma);
      parameters.rarefaction_exponent =
          ScalarNumber(0.5 * (gamma - 1.) / gamma);
      parameters.rarefaction_exponent_inverse =
          ScalarNumber(2. * gamma / (gamma - 1.));
      parameters.half_gamma_minus_one = ScalarNumber(0.5 * (gamma - 1.));
      parameters.c_of_gamma = ScalarNumber(
          NASGRiemannSolverView<ScalarNumber, options>::c(ScalarNumber(gamma)));
    }


    template <typename ScalarNumber, NASGRiemannSolverOptions options>
    inline void NASGRiemannSolver<ScalarNumber, options>::set_equation_of_state(
        const double covolume_b,
        const double pinf,
        const bool compute_expensive_bounds)
    {
      auto &parameters = *parameters_.view();

      parameters.covolume_b = ScalarNumber(covolume_b);
      parameters.pinf = ScalarNumber(pinf);
      parameters.compute_expensive_bounds = compute_expensive_bounds;
    }


    template <typename Number,
              NASGRiemannSolverOptions options,
              typename MemorySpace>
    DEAL_II_HOST_DEVICE Number
    NASGRiemannSolverView<Number, options, MemorySpace>::compute(
        const primitive_type &riemann_data_i,
        const primitive_type &riemann_data_j) const
    {
      const auto &[rho_i, u_i, p_i, gamma_i, a_i] = riemann_data_i;
      const auto &[rho_j, u_j, p_j, gamma_j, a_j] = riemann_data_j;

#ifdef DEBUG_WAVE_SPEED_ESTIMATOR
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

      const Number phi_p_max = phi_of_p_max(riemann_data_i, riemann_data_j);
      Number p_2 =
          p_star_upper_bound(riemann_data_i, riemann_data_j, phi_p_max);

#ifdef DEBUG_WAVE_SPEED_ESTIMATOR
      std::cout << "   p^*_tilde  = " << p_2 << "\n";
      std::cout << "   phi(p_*_t) = "
                << phi(riemann_data_i, riemann_data_j, p_2) << std::endl;
#endif

      if (newton_max_iterations() == 0) {
        const auto lambda_max =
            compute_lambda_max(riemann_data_i, riemann_data_j, p_2);

#ifdef DEBUG_WAVE_SPEED_ESTIMATOR
        std::cout << "-> lambda_max = " << lambda_max << std::endl;
#endif
        return lambda_max;
      }

      const Number p_min = std::min(p_i, p_j);
      const Number p_max = std::max(p_i, p_j);

      Number p_1 =
          ryujin::compare_and_apply_mask<dealii::SIMDComparison::less_than>(
              phi_p_max, Number(0.), p_max, p_min);

      p_1 = ryujin::compare_and_apply_mask<
          dealii::SIMDComparison::less_than_or_equal>(p_1, p_2, p_1, p_2);

      auto [gap, lambda_max] =
          compute_gap(riemann_data_i, riemann_data_j, p_1, p_2);

#ifdef DEBUG_WAVE_SPEED_ESTIMATOR
      std::cout << std::fixed << std::setprecision(16);
      std::cout << "p_1: (start) " << p_1 << std::endl;
      std::cout << "p_2: (start) " << p_2 << std::endl;
      std::cout << "gap: (start) " << gap << std::endl;
      std::cout << "l_m: (start) " << lambda_max << std::endl;
#endif

      for (unsigned int i = 0; i < newton_max_iterations(); ++i) {
        const Number tolerance(newton_tolerance());
        if (std::max(Number(0.), gap - tolerance) == Number(0.)) {
#ifdef DEBUG_WAVE_SPEED_ESTIMATOR
          std::cout << "converged after " << i << " iterations." << std::endl;
#endif
          break;
        }

        newton_step(riemann_data_i, riemann_data_j, p_1, p_2);

        auto [gap_new, lambda_max_new] =
            compute_gap(riemann_data_i, riemann_data_j, p_1, p_2);
        gap = gap_new;
        lambda_max = lambda_max_new;

#ifdef DEBUG_WAVE_SPEED_ESTIMATOR
        std::cout << "p_1: (  " << i << "  ) " << p_1 << std::endl;
        std::cout << "p_2: (  " << i << "  ) " << p_2 << std::endl;
        std::cout << "gap:         " << gap << std::endl;
        std::cout << "l_m:         " << lambda_max << std::endl;
#endif
      }

#ifdef DEBUG_WAVE_SPEED_ESTIMATOR
      std::cout << "-> lambda_max = " << lambda_max << std::endl;
#endif

      return lambda_max;
    }


    template <typename Number,
              NASGRiemannSolverOptions options,
              typename MemorySpace>
    DEAL_II_HOST_DEVICE auto
    NASGRiemannSolverView<Number, options, MemorySpace>::riemann_solution(
        const primitive_type &riemann_data_i,
        const primitive_type &riemann_data_j,
        const Number p_star) const -> RiemannSolution
    {
      const auto &[rho_i, u_i, p_i, gamma_Z_i, a_i] = riemann_data_i;
      const auto &[rho_j, u_j, p_j, gamma_Z_j, a_j] = riemann_data_j;
      const auto gamm_i = gamma_of(riemann_data_i);
      const auto gamm_j = gamma_of(riemann_data_j);

      const Number u_star_left = u_i - f(riemann_data_i, p_star);
      const Number u_star_right = u_j + f(riemann_data_j, p_star);
      const Number u_star = ScalarNumber(0.5) * (u_star_left + u_star_right);

      const Number rho_star_left = rho_star(riemann_data_i, p_star);
      const Number rho_star_right = rho_star(riemann_data_j, p_star);

      const Number lambda1_minus = this->lambda1_minus(riemann_data_i, p_star);
      const Number lambda3_plus = this->lambda3_plus(riemann_data_j, p_star);

      constexpr auto GTE = dealii::SIMDComparison::greater_than_or_equal;
      Number lambda1_plus =
          u_star_left - speed_of_sound(rho_star_left, p_star, Number(gamm_i));
      lambda1_plus = ryujin::compare_and_apply_mask<GTE>(
          p_star, p_i, lambda1_minus, lambda1_plus);

      Number lambda3_minus =
          u_star_right + speed_of_sound(rho_star_right, p_star, Number(gamm_j));
      lambda3_minus = ryujin::compare_and_apply_mask<GTE>(
          p_star, p_j, lambda3_plus, lambda3_minus);

      return RiemannSolution{
          .riemann_data_left = riemann_data_i,
          .riemann_data_right = riemann_data_j,
          .p_star = p_star,
          .u_star = u_star,
          .rho_star_left = rho_star_left,
          .rho_star_right = rho_star_right,
          .lambda1_minus = lambda1_minus,
          .lambda1_plus = lambda1_plus,
          .lambda3_minus = lambda3_minus,
          .lambda3_plus = lambda3_plus,
      };
    }


    template <typename Number,
              NASGRiemannSolverOptions options,
              typename MemorySpace>
    DEAL_II_HOST_DEVICE auto
    NASGRiemannSolverView<Number, options, MemorySpace>::solve(
        const primitive_type &riemann_data_i,
        const primitive_type &riemann_data_j,
        const unsigned int max_iterations) const -> RiemannSolution
    {
      const Number &p_i = riemann_data_i[2];
      const Number &p_j = riemann_data_j[2];

      const Number p_min = std::min(p_i, p_j);
      const Number p_max = std::max(p_i, p_j);
      const Number p_vacuum = unshift(Number(0.));

      const Number phi_p_max = phi_of_p_max(riemann_data_i, riemann_data_j);
      const Number phi_p_min = phi(riemann_data_i, riemann_data_j, p_min);
      const Number phi_p_vacuum = phi(riemann_data_i, riemann_data_j, p_vacuum);

      const Number p_lower = std::max(
          p_vacuum, p_star_two_rarefaction(riemann_data_i, riemann_data_j));

      Number p_1 = ryujin::compare_and_apply_mask<
          dealii::SIMDComparison::less_than_or_equal>(
          phi_p_min, Number(0.), p_min, p_lower);
      Number p_2 = ryujin::compare_and_apply_mask<
          dealii::SIMDComparison::less_than_or_equal>(
          phi_p_min, Number(0.), p_max, p_min);

      const Number p_upper = std::max(
          p_max, p_star_upper_bound(riemann_data_i, riemann_data_j, phi_p_max));

      p_1 = ryujin::compare_and_apply_mask<dealii::SIMDComparison::less_than>(
          phi_p_max, Number(0.), p_max, p_1);
      p_2 = ryujin::compare_and_apply_mask<dealii::SIMDComparison::less_than>(
          phi_p_max, Number(0.), p_upper, p_2);

      p_1 = ryujin::compare_and_apply_mask<
          dealii::SIMDComparison::greater_than_or_equal>(
          phi_p_vacuum, Number(0.), p_vacuum, p_1);
      p_2 = ryujin::compare_and_apply_mask<
          dealii::SIMDComparison::greater_than_or_equal>(
          phi_p_vacuum, Number(0.), p_vacuum, p_2);

      constexpr ScalarNumber eps = std::numeric_limits<ScalarNumber>::epsilon();

      for (unsigned int i = 0; i < max_iterations; ++i) {
        const Number tolerance = ScalarNumber(16. * eps) * shift(p_2);
        if (std::max(Number(0.), p_2 - p_1 - tolerance) == Number(0.))
          break;

        newton_step(riemann_data_i, riemann_data_j, p_1, p_2);
      }

      return riemann_solution(riemann_data_i, riemann_data_j, p_2);
    }


    template <typename Number,
              NASGRiemannSolverOptions options,
              typename MemorySpace>
    DEAL_II_HOST_DEVICE auto
    NASGRiemannSolverView<Number, options, MemorySpace>::sample(
        const RiemannSolution &solution,
        const Number &xi,
        const unsigned int max_iterations) const -> primitive_type
    {
      const auto &riemann_data_left = solution.riemann_data_left;
      const auto &riemann_data_right = solution.riemann_data_right;

      const Number xi_left =
          std::max(solution.lambda1_minus, std::min(xi, solution.lambda1_plus));
      const Number p_fan_left = rarefaction_fan_pressure(
          riemann_data_left, xi_left, ScalarNumber(-1.), max_iterations);
      const Number rho_fan_left = rho_star(riemann_data_left, p_fan_left);
      const Number u_fan_left =
          riemann_data_left[1] - f(riemann_data_left, p_fan_left);

      const Number xi_right =
          std::max(solution.lambda3_minus, std::min(xi, solution.lambda3_plus));
      const Number p_fan_right = rarefaction_fan_pressure(
          riemann_data_right, xi_right, ScalarNumber(1.), max_iterations);
      const Number rho_fan_right = rho_star(riemann_data_right, p_fan_right);
      const Number u_fan_right =
          riemann_data_right[1] + f(riemann_data_right, p_fan_right);

      primitive_type result = riemann_data_right;

      const auto select = [&](const Number &threshold,
                              const Number &rho,
                              const Number &u,
                              const Number &p,
                              const Number &gamma) {
        constexpr auto LT = dealii::SIMDComparison::less_than;
        result[0] =
            ryujin::compare_and_apply_mask<LT>(xi, threshold, rho, result[0]);
        result[1] =
            ryujin::compare_and_apply_mask<LT>(xi, threshold, u, result[1]);
        result[2] =
            ryujin::compare_and_apply_mask<LT>(xi, threshold, p, result[2]);
        result[3] =
            ryujin::compare_and_apply_mask<LT>(xi, threshold, gamma, result[3]);
      };

      select(solution.lambda3_plus,
             rho_fan_right,
             u_fan_right,
             p_fan_right,
             riemann_data_right[3]);
      select(solution.lambda3_minus,
             solution.rho_star_right,
             solution.u_star,
             solution.p_star,
             riemann_data_right[3]);
      select(solution.u_star,
             solution.rho_star_left,
             solution.u_star,
             solution.p_star,
             riemann_data_left[3]);
      select(solution.lambda1_plus,
             rho_fan_left,
             u_fan_left,
             p_fan_left,
             riemann_data_left[3]);
      select(solution.lambda1_minus,
             riemann_data_left[0],
             riemann_data_left[1],
             riemann_data_left[2],
             riemann_data_left[3]);

      Number gamma;
      if constexpr (options.variable_gamma)
        gamma = result[3];
      else
        gamma = Number(gamma_of(riemann_data_left));

      result[4] = speed_of_sound(result[0], result[2], gamma);

      return result;
    }


    template <typename Number,
              NASGRiemannSolverOptions options,
              typename MemorySpace>
    DEAL_II_HOST_DEVICE Number
    NASGRiemannSolverView<Number, options, MemorySpace>::
        rarefaction_fan_pressure(const primitive_type &riemann_data,
                                 const Number &xi,
                                 const ScalarNumber sign,
                                 const unsigned int max_iterations) const
    {
      const auto &[rho_Z, u_Z, p_Z, gamma_Z, a_Z] = riemann_data;
      const Number gamma = gamma_of(riemann_data);

      const Number alpha_Z = alpha(rho_Z, gamma, a_Z);
      const Number a_tilde_Z = a_Z * one_minus_b_rho(rho_Z);

      const Number constant = alpha_Z + sign * (xi - u_Z);
      const Number linear = a_tilde_Z + alpha_Z;

      Number r = std::max(Number(0.), safe_division(constant, linear));

      if constexpr (options.covolume) {
        const Number nonlinear = a_Z - a_tilde_Z;
        const Number k_minus_one =
            safe_division(Number(2.), gamma - Number(1.));
        const Number k = k_minus_one + Number(1.);

        constexpr ScalarNumber eps =
            std::numeric_limits<ScalarNumber>::epsilon();
        const Number tolerance(ScalarNumber(16.) * eps);

        for (unsigned int i = 0; i < max_iterations; ++i) {
          const Number r_power = ryujin::pow(r, k_minus_one);

          const Number minus_g =
              linear * r + nonlinear * r * r_power - constant;
          const Number minus_dg = linear + k * nonlinear * r_power;
          const Number delta = safe_division(minus_g, minus_dg);

          r = std::max(Number(0.), r - delta);
          if (std::max(Number(0.), delta - tolerance) == Number(0.))
            break;
        }
      }

      const Number P_Z = shift(p_Z);
      return unshift(
          P_Z *
          ryujin::pow(r, Number(rarefaction_exponent_inverse(riemann_data))));
    }


    template <typename Number,
              NASGRiemannSolverOptions options,
              typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE Number
    NASGRiemannSolverView<Number, options, MemorySpace>::speed_of_sound(
        const Number &rho, const Number &p, const Number &gamma) const
    {
      return std::sqrt(
          safe_division(gamma * shift(p), rho * one_minus_b_rho(rho)));
    }


    template <typename Number,
              NASGRiemannSolverOptions options,
              typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE Number
    NASGRiemannSolverView<Number, options, MemorySpace>::rho_star(
        const primitive_type &riemann_data, const Number &p_star) const
    {
      const auto &[rho, u, p, gamma_Z, a] = riemann_data;
      const auto gamma = gamma_of(riemann_data);

      const Number one_minus_b_rho = this->one_minus_b_rho(rho);
      const Number b_rho = Number(1.) - one_minus_b_rho;

      const Number P = shift(p);
      const Number P_star = shift(p_star);

      const Number gamma_minus_one_P_star = (gamma - Number(1.)) * P_star;
      const Number gamma_minus_one_P = (gamma - Number(1.)) * P;
      const Number gamma_plus_one_P_star = (gamma + Number(1.)) * P_star;
      const Number gamma_plus_one_P = (gamma + Number(1.)) * P;

      const Number shock_numerator = gamma_plus_one_P_star + gamma_minus_one_P;
      const Number shock_denominator =
          one_minus_b_rho * (gamma_minus_one_P_star + gamma_plus_one_P) +
          b_rho * shock_numerator;

      const Number true_value =
          rho * safe_division(shock_numerator, shock_denominator);

      const Number r = ryujin::pow(safe_division(P_star, P),
                                   Number(ScalarNumber(1.) / gamma));

      const Number false_value =
          rho * safe_division(r, one_minus_b_rho + b_rho * r);

      return ryujin::compare_and_apply_mask<
          dealii::SIMDComparison::greater_than_or_equal>(
          p_star, p, true_value, false_value);
    }


    template <typename Number,
              NASGRiemannSolverOptions options,
              typename MemorySpace>
    template <typename T>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE T
    NASGRiemannSolverView<Number, options, MemorySpace>::c(const T &gamma)
    {
      constexpr ScalarNumber slope =
          ScalarNumber(-0.34976871477801828189920753948709);

      const T first_radicand = (ScalarNumber(3.) * gamma + T(11.)) /
                               (ScalarNumber(6.) * gamma + T(6.));

      const T second_radicand = T(5. / 6.) + slope * (gamma - T(3.));

      T radicand = std::min(first_radicand, second_radicand);
      radicand = std::min(T(1.), radicand);
      radicand = std::max(T(1. / 2.), radicand);

      return std::sqrt(radicand);
    }


    template <typename Number,
              NASGRiemannSolverOptions options,
              typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE Number
    NASGRiemannSolverView<Number, options, MemorySpace>::alpha(
        const Number &rho, const Number &gamma, const Number &a) const
    {
      const Number numerator = ScalarNumber(2.) * a * one_minus_b_rho(rho);

      const Number denominator = gamma - Number(1.);

      return safe_division(numerator, denominator);
    }


    template <typename Number,
              NASGRiemannSolverOptions options,
              typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE Number
    NASGRiemannSolverView<Number, options, MemorySpace>::f(
        const primitive_type &riemann_data, const Number p_star) const
    {
      constexpr ScalarNumber min = std::numeric_limits<ScalarNumber>::min();

      const auto &[rho, u, p, gamma_Z, a] = riemann_data;
      const auto gamma = gamma_of(riemann_data);

      const Number one_minus_b_rho = this->one_minus_b_rho(rho);
      const Number gamma_minus_one = gamma - Number(1.);

      const Number Az =
          ScalarNumber(2.) * one_minus_b_rho / (rho * (gamma + Number(1.)));

      const Number Bz = gamma_minus_one / (gamma + Number(1.)) * shift(p);

      const Number radicand = safe_division(Az, shift(p_star) + Bz);

      const Number true_value = (p_star - p) * std::sqrt(radicand);

      const auto exponent = rarefaction_exponent(riemann_data);

      const Number ratio = safe_division(shift(p_star), shift(p));
      const Number factor = ryujin::pow(ratio, exponent) - Number(1.);

      const auto false_value = ScalarNumber(2.) * a * one_minus_b_rho * factor /
                               std::max(gamma_minus_one, Number(min));

      return ryujin::compare_and_apply_mask<
          dealii::SIMDComparison::greater_than_or_equal>(
          p_star, p, true_value, false_value);
    }


    template <typename Number,
              NASGRiemannSolverOptions options,
              typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE Number
    NASGRiemannSolverView<Number, options, MemorySpace>::df(
        const primitive_type &riemann_data, const Number &p_star) const
    {
      const auto &[rho, u, p, gamma_Z, a] = riemann_data;
      const auto gamma = gamma_of(riemann_data);

      const Number one_minus_b_rho = this->one_minus_b_rho(rho);

      const Number radicand_inverse =
          safe_division(ScalarNumber(0.5) * rho, one_minus_b_rho) *
          ((gamma + Number(1.)) * shift(p_star) +
           (gamma - Number(1.)) * shift(p));
      const Number denominator =
          shift(p_star) +
          ((gamma - Number(1.)) / (gamma + Number(1.)) * shift(p));

      const Number true_value =
          (denominator - ScalarNumber(0.5) * (p_star - p)) /
          (denominator * std::sqrt(radicand_inverse));

      const auto exponent = -lambda_factor(riemann_data);

      const Number ratio = safe_division(shift(p_star), shift(p));

      const auto false_value =
          safe_division(a * one_minus_b_rho * ryujin::pow(ratio, exponent),
                        Number(gamma * shift(p)));

      return ryujin::compare_and_apply_mask<
          dealii::SIMDComparison::greater_than_or_equal>(
          p_star, p, true_value, false_value);
    }


    template <typename Number,
              NASGRiemannSolverOptions options,
              typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE Number
    NASGRiemannSolverView<Number, options, MemorySpace>::phi(
        const primitive_type &riemann_data_i,
        const primitive_type &riemann_data_j,
        const Number p_in) const
    {
      const Number &u_i = riemann_data_i[1];
      const Number &u_j = riemann_data_j[1];

      return f(riemann_data_i, p_in) + f(riemann_data_j, p_in) + u_j - u_i;
    }


    template <typename Number,
              NASGRiemannSolverOptions options,
              typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE Number
    NASGRiemannSolverView<Number, options, MemorySpace>::dphi(
        const primitive_type &riemann_data_i,
        const primitive_type &riemann_data_j,
        const Number &p) const
    {
      return df(riemann_data_i, p) + df(riemann_data_j, p);
    }


    template <typename Number,
              NASGRiemannSolverOptions options,
              typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE Number
    NASGRiemannSolverView<Number, options, MemorySpace>::phi_of_p_max(
        const primitive_type &riemann_data_i,
        const primitive_type &riemann_data_j) const
    {
      const auto &[rho_i, u_i, p_i, gamma_Z_i, a_i] = riemann_data_i;
      const auto &[rho_j, u_j, p_j, gamma_Z_j, a_j] = riemann_data_j;
      const auto gamma_i = gamma_of(riemann_data_i);
      const auto gamma_j = gamma_of(riemann_data_j);

      const Number p_max = std::max(p_i, p_j);

      const Number radicand_inverse_i =
          safe_division(ScalarNumber(0.5) * rho_i, one_minus_b_rho(rho_i)) *
          ((gamma_i + Number(1.)) * shift(p_max) +
           (gamma_i - Number(1.)) * shift(p_i));

      const Number value_i =
          safe_division(p_max - p_i, std::sqrt(radicand_inverse_i));

      const Number radicand_inverse_j =
          safe_division(ScalarNumber(0.5) * rho_j, one_minus_b_rho(rho_j)) *
          ((gamma_j + Number(1.)) * shift(p_max) +
           (gamma_j - Number(1.)) * shift(p_j));

      const Number value_j =
          safe_division(p_max - p_j, std::sqrt(radicand_inverse_j));

      return value_i + value_j + u_j - u_i;
    }


    template <typename Number,
              NASGRiemannSolverOptions options,
              typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE Number
    NASGRiemannSolverView<Number, options, MemorySpace>::lambda1_minus(
        const primitive_type &riemann_data, const Number p_star) const
    {
      const auto &[rho, u, p, gamma, a] = riemann_data;

      const auto factor = lambda_factor(riemann_data);

      const Number p_inverse = safe_division(Number(1.), shift(p));
      const Number tmp = positive_part(p_star - p) * p_inverse;

      return u - a * std::sqrt(Number(1.) + factor * tmp);
    }


    template <typename Number,
              NASGRiemannSolverOptions options,
              typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE Number
    NASGRiemannSolverView<Number, options, MemorySpace>::lambda3_plus(
        const primitive_type &riemann_data, const Number p_star) const
    {
      const auto &[rho, u, p, gamma, a] = riemann_data;

      const auto factor = lambda_factor(riemann_data);

      const Number p_inverse = safe_division(Number(1.), shift(p));
      const Number tmp = positive_part(p_star - p) * p_inverse;

      return u + a * std::sqrt(Number(1.) + factor * tmp);
    }


    template <typename Number,
              NASGRiemannSolverOptions options,
              typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE Number
    NASGRiemannSolverView<Number, options, MemorySpace>::p_star_upper_bound(
        const primitive_type &riemann_data_i,
        const primitive_type &riemann_data_j,
        const Number &phi_p_max) const
    {
      const Number &p_i = riemann_data_i[2];
      const Number &p_j = riemann_data_j[2];

      const Number p_max = std::max(p_i, p_j);

      if constexpr (!options.variable_gamma) {
        const Number p_star_tilde =
            p_star_single_gamma(riemann_data_i, riemann_data_j, phi_p_max);
        const Number p_star_backup =
            p_star_failsafe(riemann_data_i, riemann_data_j);

        return ryujin::compare_and_apply_mask<
            dealii::SIMDComparison::less_than>(
            phi_p_max,
            Number(0.),
            std::min(p_star_tilde, p_star_backup),
            std::min(p_max, p_star_tilde));

      } else if (!compute_expensive_bounds()) {
#ifdef DEBUG_WAVE_SPEED_ESTIMATOR
        const Number p_star_RS = p_star_RS_full(riemann_data_i, riemann_data_j);
        const Number p_star_SS = p_star_SS_full(riemann_data_i, riemann_data_j);
        const Number p_strict =
            ryujin::compare_and_apply_mask<dealii::SIMDComparison::less_than>(
                phi_p_max, Number(0.), p_star_SS, std::min(p_max, p_star_RS));
        std::cout << "   p^*_strict = " << p_strict << "\n";
        std::cout << "   phi(p_*_s) = "
                  << phi(riemann_data_i, riemann_data_j, p_strict) << "\n";
        std::cout << "-> lambda_str = "
                  << compute_lambda_max(
                         riemann_data_i, riemann_data_j, p_strict)
                  << std::endl;
#endif

        const Number p_star_tilde =
            p_star_interpolated(riemann_data_i, riemann_data_j);
        const Number p_star_backup =
            p_star_failsafe(riemann_data_i, riemann_data_j);

        return ryujin::compare_and_apply_mask<
            dealii::SIMDComparison::less_than>(
            phi_p_max,
            Number(0.),
            std::min(p_star_tilde, p_star_backup),
            std::min(p_max, p_star_tilde));

      } else {

        const Number p_star_RS = p_star_RS_full(riemann_data_i, riemann_data_j);
        const Number p_star_SS = p_star_SS_full(riemann_data_i, riemann_data_j);

        return ryujin::compare_and_apply_mask<
            dealii::SIMDComparison::less_than>(
            phi_p_max, Number(0.), p_star_SS, std::min(p_max, p_star_RS));
      }
    }


    template <typename Number,
              NASGRiemannSolverOptions options,
              typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE Number
    NASGRiemannSolverView<Number, options, MemorySpace>::p_star_single_gamma(
        const primitive_type &riemann_data_i,
        const primitive_type &riemann_data_j,
        const Number &phi_p_max) const
    {
      const auto &[rho_i, u_i, p_i, gamma_i, a_i] = riemann_data_i;
      const auto &[rho_j, u_j, p_j, gamma_j, a_j] = riemann_data_j;

      const auto c_gamma = c_of_gamma(riemann_data_i);

      const Number alpha_i = a_i * one_minus_b_rho(rho_i);
      const Number alpha_j = a_j * one_minus_b_rho(rho_j);

      const Number p_min = shift(std::min(p_i, p_j));
      const Number p_max = shift(std::max(p_i, p_j));

      const Number alpha_min =
          ryujin::compare_and_apply_mask<dealii::SIMDComparison::less_than>(
              p_i, p_j, alpha_i, alpha_j);

      const Number alpha_max = ryujin::compare_and_apply_mask<
          dealii::SIMDComparison::greater_than_or_equal>(
          p_i, p_j, alpha_i, alpha_j);

      const Number alpha_hat_min = c_gamma * alpha_min;

      const Number alpha_select =
          ryujin::compare_and_apply_mask<dealii::SIMDComparison::less_than>(
              phi_p_max, Number(0.), c_gamma * alpha_max, alpha_max);

      const auto exponent = rarefaction_exponent(riemann_data_i);
      const auto exponent_inverse =
          rarefaction_exponent_inverse(riemann_data_i);

      const Number numerator =
          positive_part(alpha_hat_min + alpha_select -
                        half_gamma_minus_one(riemann_data_i) * (u_j - u_i));

      const Number denominator =
          alpha_hat_min * ryujin::pow(safe_division(p_min, p_max), -exponent) +
          alpha_select;

      const Number p_tilde =
          unshift(p_max * ryujin::pow(safe_division(numerator, denominator),
                                      exponent_inverse));

#ifdef DEBUG_WAVE_SPEED_ESTIMATOR
      std::cout << "p_star_single_gamma = " << p_tilde << std::endl;
#endif
      return p_tilde;
    }


    template <typename Number,
              NASGRiemannSolverOptions options,
              typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE Number
    NASGRiemannSolverView<Number, options, MemorySpace>::p_star_interpolated(
        const primitive_type &riemann_data_i,
        const primitive_type &riemann_data_j) const
    {
      const auto &[rho_i, u_i, p_i, gamma_i, a_i] = riemann_data_i;
      const auto &[rho_j, u_j, p_j, gamma_j, a_j] = riemann_data_j;
      const auto alpha_i = alpha(rho_i, gamma_i, a_i);
      const auto alpha_j = alpha(rho_j, gamma_j, a_j);

      const Number p_min = shift(std::min(p_i, p_j));
      const Number p_max = shift(std::max(p_i, p_j));

      const Number gamma_min =
          ryujin::compare_and_apply_mask<dealii::SIMDComparison::less_than>(
              p_i, p_j, gamma_i, gamma_j);

      const Number alpha_min =
          ryujin::compare_and_apply_mask<dealii::SIMDComparison::less_than>(
              p_i, p_j, alpha_i, alpha_j);

      const Number alpha_hat_min = c(gamma_min) * alpha_min;

      const Number gamma_max = ryujin::compare_and_apply_mask<
          dealii::SIMDComparison::greater_than_or_equal>(
          p_i, p_j, gamma_i, gamma_j);

      const Number alpha_max = ryujin::compare_and_apply_mask<
          dealii::SIMDComparison::greater_than_or_equal>(
          p_i, p_j, alpha_i, alpha_j);

      const Number alpha_hat_max = c(gamma_max) * alpha_max;

      const Number gamma_m = std::min(gamma_i, gamma_j);
      const Number gamma_M = std::max(gamma_i, gamma_j);

      const Number p_ratio = safe_division(p_min, p_max);

      const Number r_exponent =
          (gamma_M - gamma_min) / (ScalarNumber(2.) * gamma_min * gamma_M);

      const Number exponent =
          (gamma_m - Number(1.)) / (ScalarNumber(2.) * gamma_m);
      const Number exponent_inverse = Number(1.) / exponent;

      const Number numerator =
          positive_part(alpha_hat_min + alpha_max - (u_j - u_i));

      Number denominator = alpha_hat_min * ryujin::pow(p_ratio, -exponent) +
                           alpha_hat_max * ryujin::pow(p_ratio, r_exponent);

      const auto temp = safe_division(numerator, denominator);

      const Number p_tilde =
          unshift(p_max * ryujin::pow(temp, exponent_inverse));

#ifdef DEBUG_WAVE_SPEED_ESTIMATOR
      std::cout << "p_star_interpolated = " << p_tilde << std::endl;
#endif
      return p_tilde;
    }


    template <typename Number,
              NASGRiemannSolverOptions options,
              typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE Number
    NASGRiemannSolverView<Number, options, MemorySpace>::p_star_RS_full(
        const primitive_type &riemann_data_i,
        const primitive_type &riemann_data_j) const
    {
      const auto &[rho_i, u_i, p_i, gamma_i, a_i] = riemann_data_i;
      const auto &[rho_j, u_j, p_j, gamma_j, a_j] = riemann_data_j;
      const auto alpha_i = alpha(rho_i, gamma_i, a_i);
      const auto alpha_j = alpha(rho_j, gamma_j, a_j);

      const Number p_min = std::min(p_i, p_j);
      const Number p_max = std::max(p_i, p_j);

      const Number gamma_min =
          ryujin::compare_and_apply_mask<dealii::SIMDComparison::less_than>(
              p_i, p_j, gamma_i, gamma_j);

      const Number alpha_min =
          ryujin::compare_and_apply_mask<dealii::SIMDComparison::less_than>(
              p_i, p_j, alpha_i, alpha_j);

      const Number alpha_hat_min = c(gamma_min) * alpha_min;

      const Number alpha_max = ryujin::compare_and_apply_mask<
          dealii::SIMDComparison::greater_than_or_equal>(
          p_i, p_j, alpha_i, alpha_j);

      const Number gamma_m = std::min(gamma_i, gamma_j);
      const Number gamma_M = std::max(gamma_i, gamma_j);

      const Number numerator =
          ryujin::compare_and_apply_mask<dealii::SIMDComparison::equal>(
              shift(p_max),
              Number(0.),
              Number(0.),
              positive_part(alpha_hat_min + alpha_max - (u_j - u_i)));

      const Number p_ratio = safe_division(shift(p_min), shift(p_max));

      const Number r_exponent =
          (gamma_M - gamma_min) / (ScalarNumber(2.) * gamma_min * gamma_M);

      const Number first_exponent =
          (gamma_M - Number(1.)) / (ScalarNumber(2.) * gamma_M);

      const Number first_exponent_inverse =
          safe_division(Number(1.), first_exponent);

      const Number first_denom =
          alpha_hat_min * ryujin::pow(p_ratio, r_exponent - first_exponent) +
          alpha_max;

      const Number p_1_tilde = unshift(
          shift(p_max) * ryujin::pow(safe_division(numerator, first_denom),
                                     first_exponent_inverse));

      const Number second_exponent =
          (gamma_m - Number(1.)) / (ScalarNumber(2.) * gamma_m);

      const Number second_exponent_inverse =
          safe_division(Number(1.), second_exponent);

      Number second_denom =
          alpha_hat_min * ryujin::pow(p_ratio, -second_exponent) +
          alpha_max * ryujin::pow(p_ratio, r_exponent);

      const Number p_2_tilde = unshift(
          shift(p_max) * ryujin::pow(safe_division(numerator, second_denom),
                                     second_exponent_inverse));

      const Number p_star = std::min(p_1_tilde, p_2_tilde);

#ifdef DEBUG_WAVE_SPEED_ESTIMATOR
      std::cout << "p_star_RS_full = " << p_star << std::endl;
#endif
      return p_star;
    }


    template <typename Number,
              NASGRiemannSolverOptions options,
              typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE Number
    NASGRiemannSolverView<Number, options, MemorySpace>::p_star_SS_full(
        const primitive_type &riemann_data_i,
        const primitive_type &riemann_data_j) const
    {
      const auto &[rho_i, u_i, p_i, gamma_i, a_i] = riemann_data_i;
      const auto &[rho_j, u_j, p_j, gamma_j, a_j] = riemann_data_j;

      const Number gamma_m = std::min(gamma_i, gamma_j);

      const Number alpha_hat_i = c(gamma_i) * alpha(rho_i, gamma_i, a_i);
      const Number alpha_hat_j = c(gamma_j) * alpha(rho_j, gamma_j, a_j);

      const Number exponent =
          (gamma_m - Number(1.)) / (ScalarNumber(2.) * gamma_m);
      const Number exponent_inverse = Number(1.) / exponent;

      const Number numerator =
          ryujin::compare_and_apply_mask<dealii::SIMDComparison::equal>(
              shift(p_j),
              Number(0.),
              Number(0.),
              positive_part(alpha_hat_i + alpha_hat_j - (u_j - u_i)));

      const Number denominator =
          alpha_hat_i *
              ryujin::pow(safe_division(shift(p_i), shift(p_j)), -exponent) +
          alpha_hat_j;

      const Number p_1_tilde = unshift(
          shift(p_j) *
          ryujin::pow(safe_division(numerator, denominator), exponent_inverse));

      const auto p_2_tilde = p_star_failsafe(riemann_data_i, riemann_data_j);

      const Number p_star = std::min(p_1_tilde, p_2_tilde);

#ifdef DEBUG_WAVE_SPEED_ESTIMATOR
      std::cout << "p_star_SS_full = " << p_star << std::endl;
#endif
      return p_star;
    }


    template <typename Number,
              NASGRiemannSolverOptions options,
              typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE Number
    NASGRiemannSolverView<Number, options, MemorySpace>::p_star_failsafe(
        const primitive_type &riemann_data_i,
        const primitive_type &riemann_data_j) const
    {
      const auto &[rho_i, u_i, p_i, gamma_Z_i, a_i] = riemann_data_i;
      const auto &[rho_j, u_j, p_j, gamma_Z_j, a_j] = riemann_data_j;
      const auto gamma_i = gamma_of(riemann_data_i);
      const auto gamma_j = gamma_of(riemann_data_j);

      const Number p_max = shift(std::max(p_i, p_j));

      const Number radicand_i =
          safe_division(ScalarNumber(2.) * one_minus_b_rho(rho_i) * p_max,
                        rho_i * ((gamma_i + Number(1.)) * p_max +
                                 (gamma_i - Number(1.)) * shift(p_i)));

      const Number x_i = std::sqrt(radicand_i);

      const Number radicand_j =
          safe_division(ScalarNumber(2.) * one_minus_b_rho(rho_j) * p_max,
                        rho_j * ((gamma_j + Number(1.)) * p_max +
                                 (gamma_j - Number(1.)) * shift(p_j)));

      const Number x_j = std::sqrt(radicand_j);

      const Number a = x_i + x_j;
      const Number b =
          ryujin::compare_and_apply_mask<dealii::SIMDComparison::equal>(
              a, Number(0.), Number(0.), u_j - u_i);

      const Number c = -shift(p_i) * x_i - shift(p_j) * x_j;

      const Number base = safe_division(
          std::abs(-b +
                   std::sqrt(positive_part(b * b - ScalarNumber(4.) * a * c))),
          std::abs(ScalarNumber(2.) * a));

      const Number p_2_tilde = unshift(base * base);

#ifdef DEBUG_WAVE_SPEED_ESTIMATOR
      std::cout << "p_star_failsafe = " << p_2_tilde << std::endl;
#endif
      return p_2_tilde;
    }


    template <typename Number,
              NASGRiemannSolverOptions options,
              typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE Number
    NASGRiemannSolverView<Number, options, MemorySpace>::p_star_two_rarefaction(
        const primitive_type &riemann_data_i,
        const primitive_type &riemann_data_j) const
    {
      const auto &[rho_i, u_i, p_i, gamma_Z_i, a_i] = riemann_data_i;
      const auto &[rho_j, u_j, p_j, gamma_Z_j, a_j] = riemann_data_j;
      const auto gamma_i = gamma_of(riemann_data_i);
      const auto gamma_j = gamma_of(riemann_data_j);

      const Number alpha_i = alpha(rho_i, Number(gamma_i), a_i);
      const Number alpha_j = alpha(rho_j, Number(gamma_j), a_j);

      const Number p_min = shift(std::min(p_i, p_j));
      const Number p_max = shift(std::max(p_i, p_j));

      const Number alpha_min =
          ryujin::compare_and_apply_mask<dealii::SIMDComparison::less_than>(
              p_i, p_j, alpha_i, alpha_j);

      const Number alpha_max = ryujin::compare_and_apply_mask<
          dealii::SIMDComparison::greater_than_or_equal>(
          p_i, p_j, alpha_i, alpha_j);

      Number exponent;
      Number exponent_inverse;
      if constexpr (options.variable_gamma) {
        const Number gamma_m = std::min(gamma_i, gamma_j);
        exponent = (gamma_m - Number(1.)) / (ScalarNumber(2.) * gamma_m);
        exponent_inverse = Number(1.) / exponent;
      } else {
        exponent = rarefaction_exponent(riemann_data_i);
        exponent_inverse = rarefaction_exponent_inverse(riemann_data_i);
      }

      const Number numerator =
          positive_part(alpha_min + alpha_max - (u_j - u_i));

      const Number denominator =
          alpha_min +
          alpha_max * ryujin::pow(safe_division(p_min, p_max), exponent);

      const Number p_tilde =
          unshift(p_min * ryujin::pow(safe_division(numerator, denominator),
                                      exponent_inverse));

#ifdef DEBUG_WAVE_SPEED_ESTIMATOR
      std::cout << "p_star_two_rarefaction = " << p_tilde << std::endl;
#endif
      return p_tilde;
    }


    template <typename Number,
              NASGRiemannSolverOptions options,
              typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE void
    NASGRiemannSolverView<Number, options, MemorySpace>::newton_step(
        const primitive_type &riemann_data_i,
        const primitive_type &riemann_data_j,
        Number &p_1,
        Number &p_2) const
    {
      const Number phi_p_1 = phi(riemann_data_i, riemann_data_j, p_1);
      const Number phi_p_2 = phi(riemann_data_i, riemann_data_j, p_2);
      const Number dphi_p_1 = dphi(riemann_data_i, riemann_data_j, p_1);
      const Number dphi_p_2 = dphi(riemann_data_i, riemann_data_j, p_2);

#ifdef DEBUG_WAVE_SPEED_ESTIMATOR
      std::cout << "phi_p_1:     " << phi_p_1 << std::endl;
      std::cout << "phi_p_2:     " << phi_p_2 << std::endl;
      std::cout << "dphi_p_1:    " << dphi_p_1 << std::endl;
      std::cout << "dphi_p_2:    " << dphi_p_2 << std::endl;
#endif

      ryujin::quadratic_newton_step(
          p_1, p_2, phi_p_1, phi_p_2, dphi_p_1, dphi_p_2);
    }


    template <typename Number,
              NASGRiemannSolverOptions options,
              typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE std::array<Number, 2>
    NASGRiemannSolverView<Number, options, MemorySpace>::compute_gap(
        const primitive_type &riemann_data_i,
        const primitive_type &riemann_data_j,
        const Number p_1,
        const Number p_2) const
    {
      const Number nu_11 = lambda1_minus(riemann_data_i, p_2);
      const Number nu_12 = lambda1_minus(riemann_data_i, p_1);

      const Number nu_31 = lambda3_plus(riemann_data_j, p_1);
      const Number nu_32 = lambda3_plus(riemann_data_j, p_2);

      const Number lambda_max =
          std::max(positive_part(nu_32), negative_part(nu_11));

      const Number gap =
          std::max(std::abs(nu_32 - nu_31), std::abs(nu_12 - nu_11));

      return {{gap, lambda_max}};
    }


    template <typename Number,
              NASGRiemannSolverOptions options,
              typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE Number
    NASGRiemannSolverView<Number, options, MemorySpace>::compute_lambda_max(
        const primitive_type &riemann_data_i,
        const primitive_type &riemann_data_j,
        const Number p_star) const
    {
      const Number nu_11 = lambda1_minus(riemann_data_i, p_star);
      const Number nu_32 = lambda3_plus(riemann_data_j, p_star);

      return std::max(positive_part(nu_32), negative_part(nu_11));
    }

  } // namespace EulerAEOS
} // namespace ryujin

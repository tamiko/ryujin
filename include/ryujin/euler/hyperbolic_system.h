//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/compile_time_options.h>

#include <ryujin/base/convenience_macros.h>
#include <ryujin/base/gpu.h>
#include <ryujin/base/loop.h>
#include <ryujin/base/patterns_conversion.h>
#include <ryujin/base/simd.h>
#include <ryujin/discretization/discretization.h>
#include <ryujin/linear_algebra/multicomponent_vector.h>
#include <ryujin/linear_algebra/state_vector.h>

#include <deal.II/base/memory_space.h>
#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/tensor.h>

#include <array>

namespace ryujin
{
  namespace Euler
  {
    template <int dim,
              typename Number,
              typename MemorySpace = dealii::MemorySpace::Host>
    class HyperbolicSystemView;

    class HyperbolicSystem final : public dealii::ParameterAcceptor
    {
    public:
      static inline const std::string problem_name =
          "Compressible Euler equations (polytropic gas EOS, optimized)";

      struct Parameters {
        double gamma;

        double reference_density;
        double vacuum_state_relaxation_small;
        double vacuum_state_relaxation_large;

        double gamma_inverse;
        double gamma_minus_one_inverse;
        double gamma_minus_one_over_gamma_plus_one;
        double gamma_plus_one_inverse;
      };

      HyperbolicSystem(const std::string &subsection = "/HyperbolicSystem");

      template <int dim,
                typename Number = double,
                typename MemorySpace = dealii::MemorySpace::Host>
      using View = HyperbolicSystemView<dim, Number, MemorySpace>;

      template <int dim,
                typename Number,
                typename MemorySpace = dealii::MemorySpace::Host>
      auto view() const
      {
        return View<dim, Number, MemorySpace>{*this};
      }

      template <typename MemorySpace = dealii::MemorySpace::Host,
                int dim,
                typename ScalarNumber>
      void fill_precomputed_values(
          const OfflineData<dim, ScalarNumber> &offline_data,
          typename HyperbolicSystemView<dim, ScalarNumber>::StateVector
              &state_vector,
          const bool skip_constrained_dofs = true) const;

    private:
      void update_parameters();

      Mirrored<Parameters> parameters_;

      template <int, typename, typename>
      friend class HyperbolicSystemView;
    };


    template <int dim, typename Number, typename MemorySpace>
    class HyperbolicSystemView
    {
    public:
      static_assert(
          std::is_same_v<MemorySpace, dealii::MemorySpace::Host> ||
              std::is_same_v<MemorySpace, dealii::MemorySpace::Default>,
          "Unexpected memory space");

      using ScalarNumber = typename get_value_type<Number>::type;

      static constexpr unsigned int problem_dimension = 2 + dim;

      using state_type = dealii::Tensor<1, problem_dimension, Number>;

      using flux_type =
          dealii::Tensor<1, problem_dimension, dealii::Tensor<1, dim, Number>>;

      using flux_contribution_type = flux_type;

      static inline const auto component_names =
          []() -> std::array<std::string, problem_dimension> {
        if constexpr (dim == 1)
          return {"rho", "m", "E"};
        else if constexpr (dim == 2)
          return {"rho", "m_1", "m_2", "E"};
        else if constexpr (dim == 3)
          return {"rho", "m_1", "m_2", "m_3", "E"};
        __builtin_trap();
      }();

      static inline const auto primitive_component_names =
          []() -> std::array<std::string, problem_dimension> {
        if constexpr (dim == 1)
          return {"rho", "v", "p"};
        else if constexpr (dim == 2)
          return {"rho", "v_1", "v_2", "p"};
        else if constexpr (dim == 3)
          return {"rho", "v_1", "v_2", "v_3", "p"};
        __builtin_trap();
      }();

      static constexpr unsigned int n_precomputed_values = 2;

      using precomputed_type = std::array<Number, n_precomputed_values>;

      static inline const auto precomputed_names =
          std::array<std::string, n_precomputed_values>{"s", "eta_h"};

      static constexpr unsigned int n_initial_precomputed_values = 0;

      using initial_precomputed_type =
          std::array<Number, n_initial_precomputed_values>;

      static inline const auto initial_precomputed_names =
          std::array<std::string, n_initial_precomputed_values>{};

      using StateVector = Vectors::StateVector<ScalarNumber>;

      using HyperbolicVector = Vectors::MultiComponentVector<ScalarNumber>;

      using PrecomputedVector = Vectors::MultiComponentVector<ScalarNumber>;

      using PrecomputedVectorView = Vectors::MultiComponentVectorView<
          ScalarNumber,
          n_precomputed_values,
          dealii::VectorizedArray<ScalarNumber>::size(),
          MemorySpace,
          false>;

      using InitialPrecomputedVector =
          Vectors::MultiComponentVector<ScalarNumber>;

      using InitialPrecomputedVectorView = Vectors::MultiComponentVectorView<
          ScalarNumber,
          n_initial_precomputed_values,
          dealii::VectorizedArray<ScalarNumber>::size(),
          MemorySpace,
          false>;

      HyperbolicSystemView(const HyperbolicSystem &hyperbolic_system)
          : parameters_(
                hyperbolic_system.parameters_.template view<MemorySpace>())
      {
      }

      DEAL_II_HOST_DEVICE_ALWAYS_INLINE ScalarNumber gamma() const
      {
        return ScalarNumber(parameters_->gamma);
      }

      DEAL_II_HOST_DEVICE_ALWAYS_INLINE ScalarNumber reference_density() const
      {
        return ScalarNumber(parameters_->reference_density);
      }

      DEAL_II_HOST_DEVICE_ALWAYS_INLINE ScalarNumber
      vacuum_state_relaxation_small() const
      {
        return ScalarNumber(parameters_->vacuum_state_relaxation_small);
      }

      DEAL_II_HOST_DEVICE_ALWAYS_INLINE ScalarNumber
      vacuum_state_relaxation_large() const
      {
        return ScalarNumber(parameters_->vacuum_state_relaxation_large);
      }

      DEAL_II_HOST_DEVICE_ALWAYS_INLINE ScalarNumber gamma_inverse() const
      {
        return ScalarNumber(parameters_->gamma_inverse);
      }

      DEAL_II_HOST_DEVICE_ALWAYS_INLINE ScalarNumber
      gamma_plus_one_inverse() const
      {
        return ScalarNumber(parameters_->gamma_plus_one_inverse);
      }

      DEAL_II_HOST_DEVICE_ALWAYS_INLINE ScalarNumber
      gamma_minus_one_inverse() const
      {
        return ScalarNumber(parameters_->gamma_minus_one_inverse);
      }

      DEAL_II_HOST_DEVICE_ALWAYS_INLINE ScalarNumber
      gamma_minus_one_over_gamma_plus_one() const
      {
        return ScalarNumber(parameters_->gamma_minus_one_over_gamma_plus_one);
      }

      static constexpr bool have_gamma = true;
      static constexpr bool have_covolume_constant = false;
      static constexpr bool have_energy_equation = true;

      static DEAL_II_HOST_DEVICE Number density(const state_type &U);

      DEAL_II_HOST_DEVICE Number filter_vacuum_density(const Number &rho) const;

      static DEAL_II_HOST_DEVICE dealii::Tensor<1, dim, Number>
      momentum(const state_type &U);

      static DEAL_II_HOST_DEVICE Number total_energy(const state_type &U);

      static DEAL_II_HOST_DEVICE Number internal_energy(const state_type &U);

      static DEAL_II_HOST_DEVICE state_type
      internal_energy_derivative(const state_type &U);

      DEAL_II_HOST_DEVICE Number pressure(const state_type &U) const;

      DEAL_II_HOST_DEVICE Number speed_of_sound(const state_type &U) const;

      DEAL_II_HOST_DEVICE Number specific_entropy(const state_type &U) const;

      DEAL_II_HOST_DEVICE Number harten_entropy(const state_type &U) const;

      DEAL_II_HOST_DEVICE state_type
      harten_entropy_derivative(const state_type &U) const;

      DEAL_II_HOST_DEVICE Number
      mathematical_entropy(const state_type &U) const;

      DEAL_II_HOST_DEVICE state_type
      mathematical_entropy_derivative(const state_type &U) const;

      DEAL_II_HOST_DEVICE bool is_admissible(const state_type &U) const;

      template <int component>
      DEAL_II_HOST_DEVICE std::array<state_type, 2> linearized_eigenvector(
          const state_type &U,
          const dealii::Tensor<1, dim, Number> &normal) const;

      template <int component>
      DEAL_II_HOST_DEVICE state_type prescribe_riemann_characteristic(
          const state_type &U,
          const state_type &U_bar,
          const dealii::Tensor<1, dim, Number> &normal) const;

      template <typename Lambda>
      DEAL_II_HOST_DEVICE_ALWAYS_INLINE state_type
      apply_boundary_conditions(const dealii::types::boundary_id id,
                                const state_type &U,
                                const dealii::Tensor<1, dim, Number> &normal,
                                const Lambda &get_dirichlet_data) const;

      DEAL_II_HOST_DEVICE flux_type f(const state_type &U) const;

      DEAL_II_HOST_DEVICE
      flux_contribution_type
      flux_contribution(const PrecomputedVectorView &pv,
                        const InitialPrecomputedVectorView &ipv,
                        const unsigned int i,
                        const state_type &U_i) const;

      DEAL_II_HOST_DEVICE
      flux_contribution_type
      flux_contribution(const PrecomputedVectorView &pv,
                        const InitialPrecomputedVectorView &ipv,
                        const unsigned int *js,
                        const state_type &U_j) const;

      DEAL_II_HOST_DEVICE
      state_type
      flux_divergence(const flux_contribution_type &flux_i,
                      const flux_contribution_type &flux_j,
                      const dealii::Tensor<1, dim, Number> &c_ij) const;

      static constexpr bool have_high_order_flux = false;

      DEAL_II_HOST_DEVICE
      state_type high_order_flux_divergence(
          const flux_contribution_type &flux_i,
          const flux_contribution_type &flux_j,
          const dealii::Tensor<1, dim, Number> &c_ij) const = delete;

      static constexpr bool have_source_terms = false;

      DEAL_II_HOST_DEVICE
      state_type nodal_source(const PrecomputedVectorView &pv,
                              const unsigned int i,
                              const state_type &U_i,
                              const ScalarNumber tau) const = delete;

      DEAL_II_HOST_DEVICE
      state_type nodal_source(const PrecomputedVectorView &pv,
                              const unsigned int *js,
                              const state_type &U_j,
                              const ScalarNumber tau) const = delete;

      template <typename ST>
      DEAL_II_HOST_DEVICE state_type expand_state(const ST &state) const;

      template <typename ST>
      DEAL_II_HOST_DEVICE state_type
      from_initial_state(const ST &initial_state) const;

      DEAL_II_HOST_DEVICE
      state_type from_primitive_state(const state_type &primitive_state) const;

      DEAL_II_HOST_DEVICE
      state_type to_primitive_state(const state_type &state) const;

      template <typename Lambda>
      DEAL_II_HOST_DEVICE state_type apply_galilei_transform(
          const state_type &state, const Lambda &lambda) const;

    private:
      const HyperbolicSystem::Parameters *const parameters_;
    };


    inline HyperbolicSystem::HyperbolicSystem(const std::string &subsection)
        : ParameterAcceptor(subsection)
        , parameters_("euler_hyperbolic_system_parameters",
                      TransferPolicy::implicit_transfers_host_resident)
    {
      auto &parameters = *parameters_.view();

      parameters.gamma = 7. / 5.;
      add_parameter("gamma", parameters.gamma, "The ratio of specific heats");

      parameters.reference_density = 1.;
      add_parameter("reference density",
                    parameters.reference_density,
                    "Problem specific density reference");

      parameters.vacuum_state_relaxation_small = 1.e2;
      add_parameter("vacuum state relaxation small",
                    parameters.vacuum_state_relaxation_small,
                    "Problem specific vacuum relaxation parameter");

      parameters.vacuum_state_relaxation_large = 1.e4;
      add_parameter("vacuum state relaxation large",
                    parameters.vacuum_state_relaxation_large,
                    "Problem specific vacuum relaxation parameter");

      ParameterAcceptor::parse_parameters_call_back.connect(
          [this] { update_parameters(); });

      update_parameters();
    }


    inline void HyperbolicSystem::update_parameters()
    {
      auto &parameters = *parameters_.view();

      const auto gamma = parameters.gamma;
      parameters.gamma_inverse = 1. / gamma;
      parameters.gamma_plus_one_inverse = 1. / (gamma + 1.);
      parameters.gamma_minus_one_inverse = 1. / (gamma - 1.);
      parameters.gamma_minus_one_over_gamma_plus_one =
          (gamma - 1.) / (gamma + 1.);
    }


    template <typename MemorySpace, int dim, typename ScalarNumber>
    inline void HyperbolicSystem::fill_precomputed_values(
        const OfflineData<dim, ScalarNumber> &offline_data,
        typename HyperbolicSystemView<dim, ScalarNumber>::StateVector
            &state_vector,
        const bool skip_constrained_dofs) const
    {
      const unsigned int n_internal = offline_data.n_locally_internal();
      const unsigned int n_owned = offline_data.n_locally_owned();

      const auto sparsity_simd_view =
          offline_data.sparsity_pattern_simd().template view<MemorySpace>();

      using ScalarView = HyperbolicSystemView<dim, ScalarNumber>;

      const auto U_view =
          std::get<0>(std::as_const(state_vector))
              .template view<ScalarView::problem_dimension, MemorySpace>();
      const auto precomputed_view =
          std::get<1>(state_vector)
              .template view<ScalarView::n_precomputed_values, MemorySpace>();

      const auto hyperbolic_system_views =
          make_select_view<dim, ScalarNumber, MemorySpace>(*this);

      const auto body = [=](auto sentinel, unsigned int i) {
        using T = decltype(sentinel);
        using View = HyperbolicSystemView<dim, T, MemorySpace>;
        using precomputed_type = typename View::precomputed_type;

        const unsigned int row_length = sparsity_simd_view.row_length(i);
        if (skip_constrained_dofs && row_length == 1)
          return;

        const auto view = hyperbolic_system_views.template view<T>();

        const auto U_i = U_view.template read_tensor<T>(i);
        const precomputed_type prec_i{view.specific_entropy(U_i),
                                      view.harten_entropy(U_i)};
        precomputed_view.template write_tensor<T>(prec_i, i);
      };

      loop<MemorySpace, ScalarNumber>(
          "hyperbolic_kernel_01b", body, 0, n_internal, n_owned);
    }


    template <int dim, typename Number, typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE Number
    HyperbolicSystemView<dim, Number, MemorySpace>::density(const state_type &U)
    {
      return U[0];
    }


    template <int dim, typename Number, typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE Number
    HyperbolicSystemView<dim, Number, MemorySpace>::filter_vacuum_density(
        const Number &rho) const
    {
      constexpr ScalarNumber eps = std::numeric_limits<ScalarNumber>::epsilon();
      const Number rho_cutoff_large =
          reference_density() * vacuum_state_relaxation_large() * eps;

      return ryujin::compare_and_apply_mask<dealii::SIMDComparison::less_than>(
          std::abs(rho), rho_cutoff_large, Number(0.), rho);
    }


    template <int dim, typename Number, typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE dealii::Tensor<1, dim, Number>
    HyperbolicSystemView<dim, Number, MemorySpace>::momentum(
        const state_type &U)
    {
      dealii::Tensor<1, dim, Number> result;
      for (unsigned int i = 0; i < dim; ++i)
        result[i] = U[1 + i];
      return result;
    }


    template <int dim, typename Number, typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE Number
    HyperbolicSystemView<dim, Number, MemorySpace>::total_energy(
        const state_type &U)
    {
      return U[1 + dim];
    }


    template <int dim, typename Number, typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE Number
    HyperbolicSystemView<dim, Number, MemorySpace>::internal_energy(
        const state_type &U)
    {
      const Number rho_inverse = ScalarNumber(1.) / density(U);
      const auto m = momentum(U);
      const Number E = total_energy(U);
      return E - ScalarNumber(0.5) * m.norm_square() * rho_inverse;
    }


    template <int dim, typename Number, typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE auto
    HyperbolicSystemView<dim, Number, MemorySpace>::internal_energy_derivative(
        const state_type &U) -> state_type
    {
      const Number rho_inverse = ScalarNumber(1.) / density(U);
      const auto u = momentum(U) * rho_inverse;

      state_type result;

      result[0] = ScalarNumber(0.5) * u.norm_square();
      for (unsigned int i = 0; i < dim; ++i) {
        result[1 + i] = -u[i];
      }
      result[dim + 1] = ScalarNumber(1.);

      return result;
    }


    template <int dim, typename Number, typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE Number
    HyperbolicSystemView<dim, Number, MemorySpace>::pressure(
        const state_type &U) const
    {
      return (gamma() - ScalarNumber(1.)) * internal_energy(U);
    }


    template <int dim, typename Number, typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE Number
    HyperbolicSystemView<dim, Number, MemorySpace>::speed_of_sound(
        const state_type &U) const
    {
      const Number rho_inverse = ScalarNumber(1.) / density(U);
      const Number p = pressure(U);
      return std::sqrt(gamma() * p * rho_inverse);
    }


    template <int dim, typename Number, typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE Number
    HyperbolicSystemView<dim, Number, MemorySpace>::specific_entropy(
        const state_type &U) const
    {
      const auto rho_inverse = ScalarNumber(1.) / density(U);
      return internal_energy(U) * ryujin::pow(rho_inverse, gamma());
    }


    template <int dim, typename Number, typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE Number
    HyperbolicSystemView<dim, Number, MemorySpace>::harten_entropy(
        const state_type &U) const
    {
      const Number rho = density(U);
      const auto m = momentum(U);
      const Number E = total_energy(U);

      const Number rho_rho_e = rho * E - ScalarNumber(0.5) * m.norm_square();
      return ryujin::pow(rho_rho_e, gamma_plus_one_inverse());
    }


    template <int dim, typename Number, typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE auto
    HyperbolicSystemView<dim, Number, MemorySpace>::harten_entropy_derivative(
        const state_type &U) const -> state_type
    {
      const Number rho = density(U);
      const auto m = momentum(U);
      const Number E = total_energy(U);

      const Number rho_rho_e = rho * E - ScalarNumber(0.5) * m.norm_square();

      const auto factor =
          gamma_plus_one_inverse() *
          ryujin::pow(rho_rho_e, -gamma() * gamma_plus_one_inverse());

      state_type result;

      result[0] = factor * E;
      for (unsigned int i = 0; i < dim; ++i)
        result[1 + i] = -factor * m[i];
      result[dim + 1] = factor * rho;

      return result;
    }


    template <int dim, typename Number, typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE Number
    HyperbolicSystemView<dim, Number, MemorySpace>::mathematical_entropy(
        const state_type &U) const
    {
      using ScalarNumber = typename get_value_type<Number>::type;
      const auto p = pressure(U);
      return ryujin::pow(p, gamma_inverse());
    }


    template <int dim, typename Number, typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE auto
    HyperbolicSystemView<dim, Number, MemorySpace>::
        mathematical_entropy_derivative(const state_type &U) const -> state_type
    {
      const Number rho = density(U);
      const Number rho_inverse = ScalarNumber(1.) / rho;
      const auto u = momentum(U) * rho_inverse;
      const auto p = pressure(U);

      const auto factor = (gamma() - ScalarNumber(1.0)) * gamma_inverse() *
                          ryujin::pow(p, gamma_inverse() - ScalarNumber(1.));

      state_type result;

      result[0] = factor * ScalarNumber(0.5) * u.norm_square();
      result[dim + 1] = factor;
      for (unsigned int i = 0; i < dim; ++i) {
        result[1 + i] = -factor * u[i];
      }

      return result;
    }


    template <int dim, typename Number, typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE bool
    HyperbolicSystemView<dim, Number, MemorySpace>::is_admissible(
        const state_type &U) const
    {
      const auto rho_new = density(U);
      const auto e_new = internal_energy(U);
      const auto s_new = specific_entropy(U);

      constexpr auto gt = dealii::SIMDComparison::greater_than;
      using T = Number;
      const auto test =
          ryujin::compare_and_apply_mask<gt>(rho_new, T(0.), T(0.), T(-1.)) +
          ryujin::compare_and_apply_mask<gt>(e_new, T(0.), T(0.), T(-1.)) +
          ryujin::compare_and_apply_mask<gt>(s_new, T(0.), T(0.), T(-1.));

#ifdef DEBUG_OUTPUT
      if (!(test == Number(0.))) {
        std::cout << std::fixed << std::setprecision(16);
        std::cout << "Bounds violation: Negative state [rho, e, s] detected!\n";
        std::cout << "\t\trho: " << rho_new << "\n";
        std::cout << "\t\tint: " << e_new << "\n";
        std::cout << "\t\tent: " << s_new << "\n" << std::endl;
      }
#endif

      return (test == Number(0.));
    }


    template <int dim, typename Number, typename MemorySpace>
    template <int component>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE auto
    HyperbolicSystemView<dim, Number, MemorySpace>::linearized_eigenvector(
        const state_type &U, const dealii::Tensor<1, dim, Number> &normal) const
        -> std::array<state_type, 2>
    {
      static_assert(component == 1 || component == problem_dimension,
                    "Only first and last eigenvectors implemented");

      const auto rho = density(U);
      const auto m = momentum(U);
      const auto v = m / rho;
      const auto a = speed_of_sound(U);
      const auto gamma = this->gamma();

      state_type b;
      state_type c;

      const auto e_k = 0.5 * v.norm_square();

      switch (component) {
      case 1:
        b[0] = (gamma - 1.) * e_k + a * v * normal;
        for (unsigned int i = 0; i < dim; ++i)
          b[1 + i] = (1. - gamma) * v[i] - a * normal[i];
        b[dim + 1] = gamma - 1.;
        b /= 2. * a * a;

        c[0] = 1.;
        for (unsigned int i = 0; i < dim; ++i)
          c[1 + i] = v[i] - a * normal[i];
        c[dim + 1] = a * a / (gamma - 1) + e_k - a * (v * normal);

        return {b, c};

      case problem_dimension:
        b[0] = (gamma - 1.) * e_k - a * v * normal;
        for (unsigned int i = 0; i < dim; ++i)
          b[1 + i] = (1. - gamma) * v[i] + a * normal[i];
        b[dim + 1] = gamma - 1.;
        b /= 2. * a * a;

        c[0] = 1.;
        for (unsigned int i = 0; i < dim; ++i)
          c[1 + i] = v[i] + a * normal[i];
        c[dim + 1] = a * a / (gamma - 1) + e_k + a * (v * normal);

        return {b, c};
      }

      __builtin_unreachable();
    }


    template <int dim, typename Number, typename MemorySpace>
    template <int component>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE auto
    HyperbolicSystemView<dim, Number, MemorySpace>::
        prescribe_riemann_characteristic(
            const state_type &U,
            const state_type &U_bar,
            const dealii::Tensor<1, dim, Number> &normal) const -> state_type
    {
      static_assert(component == 1 || component == 2,
                    "component has to be 1 or 2");

      const auto m = momentum(U);
      const auto rho = density(U);
      const auto a = speed_of_sound(U);
      const auto vn = m * normal / rho;

      const auto m_bar = momentum(U_bar);
      const auto rho_bar = density(U_bar);
      const auto a_bar = speed_of_sound(U_bar);
      const auto vn_bar = m_bar * normal / rho_bar;

      const auto R_1 = component == 1
                           ? vn_bar - 2. * a_bar / (gamma() - ScalarNumber(1.))
                           : vn - 2. * a / (gamma() - ScalarNumber(1.));

      const auto R_2 = component == 2
                           ? vn_bar + 2. * a_bar / (gamma() - ScalarNumber(1.))
                           : vn + 2. * a / (gamma() - ScalarNumber(1.));

      const auto p = pressure(U);
      const auto s = p / ryujin::pow(rho, gamma());

      const auto vperp = m / rho - vn * normal;

      const auto vn_new = 0.5 * (R_1 + R_2);

      auto rho_new = 1. / (gamma() * s) *
                     ryujin::fixed_power<2>(ScalarNumber((gamma() - 1.) / 4.) *
                                            (R_2 - R_1));
      rho_new = ryujin::pow(rho_new, 1. / (gamma() - 1.));

      const auto p_new = s * std::pow(rho_new, gamma());

      state_type U_new;
      U_new[0] = rho_new;
      for (unsigned int d = 0; d < dim; ++d) {
        U_new[1 + d] = rho_new * (vn_new * normal + vperp)[d];
      }
      U_new[1 + dim] = p_new / ScalarNumber(gamma() - 1.) +
                       0.5 * rho_new * (vn_new * vn_new + vperp.norm_square());

      return U_new;
    }


    template <int dim, typename Number, typename MemorySpace>
    template <typename Lambda>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE auto
    HyperbolicSystemView<dim, Number, MemorySpace>::apply_boundary_conditions(
        dealii::types::boundary_id id,
        const state_type &U,
        const dealii::Tensor<1, dim, Number> &normal,
        const Lambda &get_dirichlet_data) const -> state_type
    {
      state_type result = U;

      if (id == Boundary::dirichlet) {
        result = get_dirichlet_data();

      } else if (id == Boundary::dirichlet_momentum) {
        const auto m_dirichlet = momentum(get_dirichlet_data());
        const auto rho = density(result);
        const auto m = momentum(result);

        for (unsigned int k = 0; k < dim; ++k)
          result[k + 1] = m_dirichlet[k];
        result[dim + 1] +=
            Number(0.5) / rho * (m_dirichlet.norm_square() - m.norm_square());

      } else if (id == Boundary::dirichlet_velocity) {
        const auto U_dirichlet = get_dirichlet_data();
        const auto rho_dirichlet = density(U_dirichlet);
        const auto v_dirichlet = momentum(U_dirichlet) / rho_dirichlet;
        const auto rho = density(result);
        const auto v = momentum(result) / rho;

        for (unsigned int k = 0; k < dim; ++k)
          result[k + 1] = rho * v_dirichlet[k];
        result[dim + 1] +=
            Number(0.5) * rho * (v_dirichlet.norm_square() - v.norm_square());

      } else if (id == Boundary::slip) {
        auto m = momentum(U);
        m -= 1. * (m * normal) * normal;
        for (unsigned int k = 0; k < dim; ++k)
          result[k + 1] = m[k];

      } else if (id == Boundary::no_slip) {
        for (unsigned int k = 0; k < dim; ++k)
          result[k + 1] = Number(0.);

      } else if (id == Boundary::dynamic) {
        const auto m = momentum(U);
        const auto rho = density(U);
        const auto a = speed_of_sound(U);
        const auto vn = m * normal / rho;

        if (vn < -a) {
          result = get_dirichlet_data();
        }

        if (vn >= -a && vn <= 0.) {
          const auto U_dirichlet = get_dirichlet_data();
          result = prescribe_riemann_characteristic<2>(U_dirichlet, U, normal);
        }

        if (vn > 0. && vn <= a) {
          const auto U_dirichlet = get_dirichlet_data();
          result = prescribe_riemann_characteristic<1>(U, U_dirichlet, normal);
        }
      } else {
        Assert(false, dealii::ExcNotImplemented());
      }

      return result;
    }


    template <int dim, typename Number, typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE auto
    HyperbolicSystemView<dim, Number, MemorySpace>::f(const state_type &U) const
        -> flux_type
    {
      const auto rho_inverse = ScalarNumber(1.) / density(U);
      const auto m = momentum(U);
      const auto p = pressure(U);
      const auto E = total_energy(U);

      flux_type result;

      result[0] = m;
      for (unsigned int i = 0; i < dim; ++i) {
        result[1 + i] = m * (m[i] * rho_inverse);
        result[1 + i][i] += p;
      }
      result[dim + 1] = m * (rho_inverse * (E + p));

      return result;
    }


    template <int dim, typename Number, typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE auto
    HyperbolicSystemView<dim, Number, MemorySpace>::flux_contribution(
        const PrecomputedVectorView &,
        const InitialPrecomputedVectorView &,
        const unsigned int,
        const state_type &U_i) const -> flux_contribution_type
    {
      return f(U_i);
    }


    template <int dim, typename Number, typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE auto
    HyperbolicSystemView<dim, Number, MemorySpace>::flux_contribution(
        const PrecomputedVectorView &,
        const InitialPrecomputedVectorView &,
        const unsigned int *,
        const state_type &U_j) const -> flux_contribution_type
    {
      return f(U_j);
    }


    template <int dim, typename Number, typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE auto
    HyperbolicSystemView<dim, Number, MemorySpace>::flux_divergence(
        const flux_contribution_type &flux_i,
        const flux_contribution_type &flux_j,
        const dealii::Tensor<1, dim, Number> &c_ij) const -> state_type
    {
      return -contract(add(flux_i, flux_j), c_ij);
    }


    template <int dim, typename Number, typename MemorySpace>
    template <typename ST>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE auto
    HyperbolicSystemView<dim, Number, MemorySpace>::expand_state(
        const ST &state) const -> state_type
    {
      using T = typename ST::value_type;
      static_assert(std::is_same_v<Number, T>, "template mismatch");

      constexpr auto dim2 = ST::dimension - 2;
      static_assert(dim >= dim2,
                    "the space dimension of the argument state must not be "
                    "larger than the one of the target state");

      state_type result;
      result[0] = state[0];
      result[dim + 1] = state[dim2 + 1];
      for (unsigned int i = 1; i < dim2 + 1; ++i)
        result[i] = state[i];

      return result;
    }


    template <int dim, typename Number, typename MemorySpace>
    template <typename ST>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE auto
    HyperbolicSystemView<dim, Number, MemorySpace>::from_initial_state(
        const ST &initial_state) const -> state_type
    {
      const auto primitive_state = expand_state(initial_state);
      return from_primitive_state(primitive_state);
    }


    template <int dim, typename Number, typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE auto
    HyperbolicSystemView<dim, Number, MemorySpace>::from_primitive_state(
        const state_type &primitive_state) const -> state_type
    {
      const auto &rho = primitive_state[0];
      const auto u = momentum(primitive_state);
      const auto &p = primitive_state[dim + 1];

      auto state = primitive_state;
      for (unsigned int i = 1; i < dim + 1; ++i)
        state[i] *= rho;
      state[dim + 1] =
          p / (ScalarNumber(gamma() - 1.)) + Number(0.5) * rho * u * u;

      return state;
    }


    template <int dim, typename Number, typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE auto
    HyperbolicSystemView<dim, Number, MemorySpace>::to_primitive_state(
        const state_type &state) const -> state_type
    {
      const auto &rho = state[0];
      const auto rho_inverse = Number(1.) / rho;
      const auto p = pressure(state);

      auto primitive_state = state;
      for (unsigned int i = 1; i < dim + 1; ++i)
        primitive_state[i] *= rho_inverse;
      primitive_state[dim + 1] = p;

      return primitive_state;
    }


    template <int dim, typename Number, typename MemorySpace>
    template <typename Lambda>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE auto
    HyperbolicSystemView<dim, Number, MemorySpace>::apply_galilei_transform(
        const state_type &state, const Lambda &lambda) const -> state_type
    {
      auto result = state;
      const auto M = lambda(momentum(state));
      for (unsigned int d = 0; d < dim; ++d)
        result[1 + d] = M[d];
      return result;
    }

  } // namespace Euler
} // namespace ryujin

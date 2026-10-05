//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/compile_time_options.h>

#include <ryujin/euler/hyperbolic_system.h>
#include <ryujin/euler_aeos/nasg_riemann_solver.h>

#include <ryujin/base/gpu.h>
#include <ryujin/base/observer_pointer.h>
#include <ryujin/base/simd.h>

#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>


namespace ryujin
{
  namespace Euler
  {
    template <int dim,
              typename Number = double,
              typename MemorySpace = dealii::MemorySpace::Host>
    class WaveSpeedEstimatorView;

    inline constexpr EulerAEOS::NASGRiemannSolverOptions
        polytropic_gas_riemann_solver_options{.covolume = false,
                                              .pinf = false,
                                              .safe_division = false,
                                              .variable_gamma = false};

    template <typename ScalarNumber>
    using PGRiemannSolver =
        EulerAEOS::NASGRiemannSolver<ScalarNumber,
                                     polytropic_gas_riemann_solver_options>;

    template <typename ScalarNumber = double>
    class WaveSpeedEstimator : protected PGRiemannSolver<ScalarNumber>
    {
    public:
      WaveSpeedEstimator(const HyperbolicSystem &hyperbolic_system,
                         const std::string &subsection = "/WaveSpeedEstimator")
          : PGRiemannSolver<ScalarNumber>(subsection)
          , hyperbolic_system_(&hyperbolic_system)
      {
        const auto update_parameters = [this] {
          const auto view =
              hyperbolic_system_->template view<1, ScalarNumber>();
          this->set_gamma(view.gamma());
        };

        this->parse_parameters_call_back.connect(update_parameters);
        update_parameters();
      }

      template <int dim,
                typename Number,
                typename MemorySpace = dealii::MemorySpace::Host>
      auto view() const
      {
        return WaveSpeedEstimatorView<dim, Number, MemorySpace>{
            hyperbolic_system_->template view<dim, Number, MemorySpace>(),
            *this};
      }

    private:
      dealii::ObserverPointer<const HyperbolicSystem> hyperbolic_system_;

      template <int, typename, typename>
      friend class WaveSpeedEstimatorView;
    };


    template <int dim, typename Number, typename MemorySpace>
    class WaveSpeedEstimatorView
    {
    public:
      static_assert(
          std::is_same_v<MemorySpace, dealii::MemorySpace::Host> ||
              std::is_same_v<MemorySpace, dealii::MemorySpace::Default>,
          "Unexpected memory space");

      using View = HyperbolicSystemView<dim, Number, MemorySpace>;

      using ScalarNumber = typename View::ScalarNumber;

      using state_type = typename View::state_type;

      using RiemannSolverView = EulerAEOS::NASGRiemannSolverView<
          Number,
          polytropic_gas_riemann_solver_options,
          MemorySpace>;

      static constexpr unsigned int riemann_data_size =
          RiemannSolverView::riemann_data_size;

      using primitive_type = typename RiemannSolverView::primitive_type;

      using PrecomputedVectorView = typename View::PrecomputedVectorView;

      WaveSpeedEstimatorView(const View &view,
                             const WaveSpeedEstimator<ScalarNumber> &wse)
          : view_(view)
          , riemann_solver_view_(
                static_cast<const PGRiemannSolver<ScalarNumber> &>(wse)
                    .template view<Number, MemorySpace>())
      {
      }

      DEAL_II_HOST_DEVICE Number
      compute(const primitive_type &riemann_data_i,
              const primitive_type &riemann_data_j) const;

      DEAL_II_HOST_DEVICE Number
      compute(const PrecomputedVectorView &pv,
              const state_type &U_i,
              const state_type &U_j,
              const unsigned int i,
              const unsigned int *js,
              const dealii::Tensor<1, dim, Number> &n_ij) const;

    protected:
      DEAL_II_HOST_DEVICE primitive_type
      riemann_data_from_state(const state_type &U,
                              const dealii::Tensor<1, dim, Number> &n_ij) const;

    private:
      const View view_;
      const RiemannSolverView riemann_solver_view_;
    };


    template <int dim, typename Number, typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE Number
    WaveSpeedEstimatorView<dim, Number, MemorySpace>::compute(
        const primitive_type &riemann_data_i,
        const primitive_type &riemann_data_j) const
    {
      return riemann_solver_view_.compute(riemann_data_i, riemann_data_j);
    }


    template <int dim, typename Number, typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE Number
    WaveSpeedEstimatorView<dim, Number, MemorySpace>::compute(
        const PrecomputedVectorView &,
        const state_type &U_i,
        const state_type &U_j,
        const unsigned int,
        const unsigned int *,
        const dealii::Tensor<1, dim, Number> &n_ij) const
    {
      const auto riemann_data_i = riemann_data_from_state(U_i, n_ij);
      const auto riemann_data_j = riemann_data_from_state(U_j, n_ij);

      return compute(riemann_data_i, riemann_data_j);
    }


    template <int dim, typename Number, typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE auto
    WaveSpeedEstimatorView<dim, Number, MemorySpace>::riemann_data_from_state(
        const state_type &U, const dealii::Tensor<1, dim, Number> &n_ij) const
        -> primitive_type
    {
      const auto rho = view_.density(U);
      const auto rho_inverse = Number(1.0) / rho;

      const auto m = view_.momentum(U);
      const auto proj_m = n_ij * m;
      const auto perp = m - proj_m * n_ij;

      const auto E = view_.total_energy(U) -
                     Number(0.5) * perp.norm_square() * rho_inverse;

      const auto gamma = view_.gamma();
      const auto internal_energy =
          E - ScalarNumber(0.5) * (proj_m * proj_m) * rho_inverse;
      const auto p = (gamma - ScalarNumber(1.)) * internal_energy;
      const auto a = std::sqrt(gamma * p * rho_inverse);

      return {{rho, proj_m * rho_inverse, p, Number(gamma), a}};
    }
  } // namespace Euler
} // namespace ryujin

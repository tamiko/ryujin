//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/compile_time_options.h>

#include <ryujin/euler/hyperbolic_system.h>

#include <ryujin/base/gpu.h>
#include <ryujin/base/observer_pointer.h>
#include <ryujin/base/simd.h>
#include <ryujin/linear_algebra/multicomponent_vector.h>

#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/vectorization.h>

namespace ryujin
{
  namespace Euler
  {
    template <int dim,
              typename Number = double,
              typename MemorySpace = dealii::MemorySpace::Host>
    class IndicatorView;

    template <typename ScalarNumber = double>
    class Indicator : public dealii::ParameterAcceptor
    {
    public:
      struct Parameters {
        double evc_factor;
      };

      Indicator(const HyperbolicSystem &hyperbolic_system,
                const std::string &subsection = "/Indicator")
          : ParameterAcceptor(subsection)
          , parameters_("euler_indicator_parameters",
                        TransferPolicy::implicit_transfers_host_resident)
          , hyperbolic_system_(&hyperbolic_system)
      {
        auto &parameters = *parameters_.view();

        parameters.evc_factor = 1.;
        add_parameter("evc factor",
                      parameters.evc_factor,
                      "Factor for scaling the entropy viscocity commuator");

        ParameterAcceptor::parse_parameters_call_back.connect(
            [this] { parameters_.view(); });
      }

      template <int dim,
                typename Number,
                typename MemorySpace = dealii::MemorySpace::Host>
      auto view() const
      {
        return IndicatorView<dim, Number, MemorySpace>{
            hyperbolic_system_->template view<dim, Number, MemorySpace>(),
            *this};
      }

    private:
      Mirrored<Parameters> parameters_;

      dealii::ObserverPointer<const HyperbolicSystem> hyperbolic_system_;

      template <int, typename, typename>
      friend class IndicatorView;
    };


    template <int dim, typename Number, typename MemorySpace>
    class IndicatorView
    {
    public:
      static_assert(
          std::is_same_v<MemorySpace, dealii::MemorySpace::Host> ||
              std::is_same_v<MemorySpace, dealii::MemorySpace::Default>,
          "Unexpected memory space");

      using View = HyperbolicSystemView<dim, Number, MemorySpace>;

      using ScalarNumber = typename View::ScalarNumber;

      static constexpr auto problem_dimension = View::problem_dimension;

      using state_type = typename View::state_type;

      using flux_type = typename View::flux_type;

      using precomputed_type = typename View::precomputed_type;

      using PrecomputedVectorView = typename View::PrecomputedVectorView;

      IndicatorView(const View &view, const Indicator<ScalarNumber> &indicator)
          : view_(view)
          , parameters_(indicator.parameters_.template view<MemorySpace>())
      {
      }

      DEAL_II_HOST_DEVICE_ALWAYS_INLINE ScalarNumber evc_factor() const
      {
        return ScalarNumber(parameters_->evc_factor);
      }

      DEAL_II_HOST_DEVICE void reset(const PrecomputedVectorView &pv,
                                     const unsigned int i,
                                     const state_type &U_i);

      DEAL_II_HOST_DEVICE void
      accumulate(const PrecomputedVectorView &pv,
                 const unsigned int *js,
                 const state_type &U_j,
                 const dealii::Tensor<1, dim, Number> &c_ij);

      DEAL_II_HOST_DEVICE Number alpha(const Number h_i) const;


    private:
      const View view_;
      const Indicator<ScalarNumber>::Parameters *const parameters_;

      Number rho_i_inverse_ = 0.;
      Number eta_i_ = 0.;
      flux_type f_i_;
      state_type d_eta_i_;

      Number left_ = 0.;
      state_type right_;
    };


    template <int dim, typename Number, typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE void
    IndicatorView<dim, Number, MemorySpace>::reset(
        const PrecomputedVectorView &pv,
        const unsigned int i,
        const state_type &U_i)
    {
      const auto &[s_i, eta_i] =
          pv.template read_tensor<Number, precomputed_type>(i);

      const auto rho_i = view_.density(U_i);
      rho_i_inverse_ = Number(1.) / rho_i;
      eta_i_ = eta_i;

      d_eta_i_ = view_.harten_entropy_derivative(U_i);
      d_eta_i_[0] -= eta_i_ * rho_i_inverse_;
      f_i_ = view_.f(U_i);

      left_ = 0.;
      right_ = 0.;
    }


    template <int dim, typename Number, typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE void
    IndicatorView<dim, Number, MemorySpace>::accumulate(
        const PrecomputedVectorView &pv,
        const unsigned int *js,
        const state_type &U_j,
        const dealii::Tensor<1, dim, Number> &c_ij)
    {
      const auto &[s_j, eta_j] =
          pv.template read_tensor<Number, precomputed_type>(js);

      const auto rho_j = view_.density(U_j);
      const auto rho_j_inverse = Number(1.) / rho_j;

      const auto m_j = view_.momentum(U_j);
      const auto f_j = view_.f(U_j);

      const auto entropy_flux =
          (eta_j * rho_j_inverse - eta_i_ * rho_i_inverse_) * (m_j * c_ij);

      left_ += entropy_flux;
      for (unsigned int k = 0; k < problem_dimension; ++k) {
        const auto component = (f_j[k] - f_i_[k]) * c_ij;
        right_[k] += component;
      }
    }


    template <int dim, typename Number, typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE Number
    IndicatorView<dim, Number, MemorySpace>::alpha(const Number hd_i) const
    {
      Number numerator = left_;
      Number denominator = std::abs(left_);
      for (unsigned int k = 0; k < problem_dimension; ++k) {
        numerator -= d_eta_i_[k] * right_[k];
        denominator += std::abs(d_eta_i_[k] * right_[k]);
      }

      const auto quotient =
          std::abs(numerator) / (denominator + hd_i * std::abs(eta_i_));

      return std::min(Number(1.), evc_factor() * quotient);
    }
  } // namespace Euler
} // namespace ryujin

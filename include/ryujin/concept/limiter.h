//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#pragma once

#include <ryujin/concept/hyperbolic_system.h>

#include <deal.II/base/tensor.h>

#include <array>
#include <concepts>
#include <string>
#include <tuple>

namespace ryujin
{
  namespace Concept
  {
    template <typename V, typename SystemView, int dim, typename Number>
    concept LimiterAccumulatesFluxes =
        requires(V view,
                 const typename SystemView::PrecomputedVectorView &precomputed,
                 const typename SystemView::state_type &U,
                 const typename SystemView::flux_contribution_type &flux,
                 const dealii::Tensor<1, dim, Number> &scaled_c_ij,
                 const unsigned int *js) {
          view.accumulate(precomputed, js, U, flux, scaled_c_ij, U);
        };

    template <typename V, typename SystemView, int dim, typename Number>
    concept LimiterAccumulatesEquilibratedStates =
        requires(V view,
                 const typename SystemView::PrecomputedVectorView &precomputed,
                 const typename SystemView::state_type &U,
                 const dealii::Tensor<1, dim, Number> &scaled_c_ij) {
          view.accumulate(precomputed, U, U, U, scaled_c_ij, U);
        };


    template <typename V, typename SystemView, int dim, typename Number>
    concept LimiterView =
        std::same_as<typename V::Bounds, std::array<Number, V::n_bounds>> &&
        (LimiterAccumulatesFluxes<V, SystemView, dim, Number> ||
         LimiterAccumulatesEquilibratedStates<V, SystemView, dim, Number>) &&
        requires(V view,
                 const V &const_view,
                 const typename SystemView::PrecomputedVectorView &precomputed,
                 const typename SystemView::state_type &U,
                 const typename SystemView::flux_contribution_type &flux,
                 const typename V::Bounds &bounds,
                 const unsigned int i,
                 const Number hd_i) {
          { const_view.iterations() } -> std::convertible_to<unsigned int>;

          view.reset(precomputed, i, U, flux);
          { const_view.bounds(hd_i) } -> std::same_as<typename V::Bounds>;
          {
            const_view.limit(bounds, U, U)
          } -> std::same_as<std::tuple<Number, bool>>;

          {
            const_view.projection_bounds_from_state(precomputed, i, U)
          } -> std::same_as<typename V::Bounds>;
          {
            const_view.combine_bounds(bounds, bounds)
          } -> std::same_as<typename V::Bounds>;
          {
            const_view.fully_relax_bounds(bounds, hd_i)
          } -> std::same_as<typename V::Bounds>;
        };


    template <typename T, typename System, int dim, typename Number>
    concept Limiter =
        std::constructible_from<T, const System &, const std::string &> &&
        LimiterView<view_type<T, dim, Number>,
                    view_type<System, dim, Number>,
                    dim,
                    Number>;
  } // namespace Concept
} // namespace ryujin

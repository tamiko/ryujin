//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#pragma once

#include <ryujin/concept/hyperbolic_system.h>

#include <deal.II/base/tensor.h>

#include <concepts>
#include <string>

namespace ryujin
{
  namespace Concept
  {
    template <typename V, typename SystemView, int dim, typename Number>
    concept IndicatorView =
        requires(V view,
                 const V &const_view,
                 const typename SystemView::PrecomputedVectorView &precomputed,
                 const typename SystemView::state_type &U,
                 const dealii::Tensor<1, dim, Number> &c_ij,
                 const unsigned int i,
                 const unsigned int *js,
                 const Number hd_i) {
          view.reset(precomputed, i, U);
          view.accumulate(precomputed, js, U, c_ij);
          { const_view.alpha(hd_i) } -> std::same_as<Number>;
        };


    template <typename T, typename System, int dim, typename Number>
    concept Indicator =
        std::constructible_from<T, const System &, const std::string &> &&
        IndicatorView<view_type<T, dim, Number>,
                      view_type<System, dim, Number>,
                      dim,
                      Number>;
  } // namespace Concept
} // namespace ryujin

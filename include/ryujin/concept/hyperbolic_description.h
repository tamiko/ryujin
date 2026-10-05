//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#pragma once

#include <ryujin/concept/hyperbolic_system.h>
#include <ryujin/concept/indicator.h>
#include <ryujin/concept/limiter.h>
#include <ryujin/concept/wave_speed_estimator.h>

#include <deal.II/base/vectorization.h>

namespace ryujin
{
  namespace Concept
  {
    template <typename T, int dim, typename ScalarNumber, typename Number>
    concept HyperbolicKernels =
        HyperbolicSystem<typename T::HyperbolicSystem, dim, Number> &&
        Indicator<typename T::template Indicator<ScalarNumber>,
                  typename T::HyperbolicSystem,
                  dim,
                  Number> &&
        Limiter<typename T::template Limiter<ScalarNumber>,
                typename T::HyperbolicSystem,
                dim,
                Number> &&
        WaveSpeedEstimator<
            typename T::template WaveSpeedEstimator<ScalarNumber>,
            typename T::HyperbolicSystem,
            dim,
            Number>;


    template <typename T, int dim, typename Number>
    concept HyperbolicDescription =
        HyperbolicKernels<T, dim, Number, Number> &&
        HyperbolicKernels<T, dim, Number, dealii::VectorizedArray<Number>>;
  } // namespace Concept
} // namespace ryujin

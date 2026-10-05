//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#pragma once

#include <ryujin/concept/parabolic_system.h>
#include <ryujin/interface/parabolic_module.h>

#include <concepts>
#include <string>

namespace ryujin
{
  class MPIEnsemble;

  template <int dim, typename Number>
  class OfflineData;

  template <typename HyperbolicDescription, int dim, typename Number>
  class InitialValues;


  namespace Concept
  {
    template <typename T, typename HD, int dim, typename Number>
    concept ParabolicDescription =
        ParabolicSystem<typename T::ParabolicSystem> &&
        std::derived_from<typename T::template ParabolicModule<HD, dim, Number>,
                          Interface::ParabolicModule<dim, Number>> &&
        std::constructible_from<
            typename T::template ParabolicModule<HD, dim, Number>,
            const MPIEnsemble &,
            const OfflineData<dim, Number> &,
            const typename HD::HyperbolicSystem &,
            const typename T::ParabolicSystem &,
            const InitialValues<HD, dim, Number> &,
            const std::string &>;
  } // namespace Concept
} // namespace ryujin

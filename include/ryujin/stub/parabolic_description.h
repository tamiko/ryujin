//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#pragma once

#include <ryujin/stub/parabolic_module.h>
#include <ryujin/stub/parabolic_system.h>

namespace ryujin
{
  struct StubParabolicDescription {
    using ParabolicSystem = StubParabolicSystem;

    template <typename HyperbolicDescription, int dim, typename Number = double>
    using ParabolicModule =
        StubParabolicModule<HyperbolicDescription, dim, Number>;
  };

} /* namespace ryujin */

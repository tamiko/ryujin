//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2026 by the ryujin authors
//

// Built once per hyperbolic description. The build defines
// RYUJIN_DESCRIPTION_HEADER and RYUJIN_HYPERBOLIC_DESCRIPTION, see section 11
// of design.h.

#include RYUJIN_DESCRIPTION_HEADER

#include <ryujin/dependent/hyperbolic_module.template.h>

namespace ryujin
{
  using HD = RYUJIN_HYPERBOLIC_DESCRIPTION;

  template class HyperbolicModule<HD, 1, NUMBER>;
  template class HyperbolicModule<HD, 2, NUMBER>;
  template class HyperbolicModule<HD, 3, NUMBER>;

  /* step<stages> for stages = 0, ..., 4: as before. */
} // namespace ryujin

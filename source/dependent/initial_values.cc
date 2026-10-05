//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2026 by the ryujin authors
//

// Built once per hyperbolic description. The build defines
// RYUJIN_DESCRIPTION_HEADER and RYUJIN_HYPERBOLIC_DESCRIPTION, see section 11
// of design.h.

#include RYUJIN_DESCRIPTION_HEADER

#include <ryujin/dependent/initial_values.template.h>

namespace ryujin
{
  using HD = RYUJIN_HYPERBOLIC_DESCRIPTION;

  template class InitialValues<HD, 1, NUMBER>;
  template class InitialValues<HD, 2, NUMBER>;
  template class InitialValues<HD, 3, NUMBER>;
} // namespace ryujin

//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2024 - 2026 by the ryujin authors
//

// Built once per hyperbolic description. The build defines
// RYUJIN_DESCRIPTION_HEADER and RYUJIN_HYPERBOLIC_DESCRIPTION, see section 11
// of design.h.

#include RYUJIN_DESCRIPTION_HEADER

#include <ryujin/dependent/solution_transfer.template.h>

namespace ryujin
{
  using HD = RYUJIN_HYPERBOLIC_DESCRIPTION;

  template class SolutionTransfer<HD, 1, NUMBER>;
  template class SolutionTransfer<HD, 2, NUMBER>;
  template class SolutionTransfer<HD, 3, NUMBER>;
} // namespace ryujin

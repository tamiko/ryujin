//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

// Part of the executable, not of the library. The only file that lists the
// pairings of a hyperbolic and a parabolic description.

#include <ryujin/base/compile_time_options.h>
#include <ryujin/driver/equation_dispatch.h>
#include <ryujin/driver/simulation.h>

#ifdef RYUJIN_EQUATION_EULER
#include <ryujin/euler/hyperbolic_description.h>
#endif

namespace ryujin
{
  void register_equations(EquationDispatch &equation_dispatch)
  {
#ifdef RYUJIN_EQUATION_EULER
    equation_dispatch.add<Euler::HyperbolicDescription>("euler");
#endif

    /*
     * A pairing with a parabolic description names both and is guarded by
     * both definitions:
     *
     *   equation_dispatch.add<Euler::HyperbolicDescription,
     *                         NavierStokes::ParabolicDescription>(
     *       "navier stokes");
     */
  }
} // namespace ryujin

//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#pragma once

#include <ryujin/linear_algebra/multicomponent_vector.h>

#include <deal.II/base/parameter_acceptor.h>

namespace ryujin
{
  namespace Interface
  {
    template <int dim, typename Number = double>
    class InitialValues : public dealii::ParameterAcceptor
    {
    public:
      virtual Vectors::MultiComponentVector<Number>
      interpolate_hyperbolic_vector(Number t = 0) const = 0;

    protected:
      using ParameterAcceptor::ParameterAcceptor;
    };
  } // namespace Interface
} // namespace ryujin

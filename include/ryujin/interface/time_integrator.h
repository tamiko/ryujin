//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2022 - 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/convenience_macros.h>
#include <ryujin/linear_algebra/state_vector.h>

#include <deal.II/base/parameter_acceptor.h>

#include <limits>
#include <string>

namespace ryujin
{
  namespace Interface
  {
    template <int dim, typename Number = double>
    class TimeIntegrator : public dealii::ParameterAcceptor
    {
    public:
      using StateVector = Vectors::StateVector<Number>;

      virtual void prepare() = 0;

      virtual void prepare_state_vector(StateVector &state_vector,
                                        Number t) const = 0;

      virtual Number
      step(StateVector &state_vector,
           Number t,
           Number t_final = std::numeric_limits<Number>::max()) = 0;

      virtual std::string time_stepping_scheme_name() const = 0;

      ACCESSOR_READ_ONLY(efficiency);

    protected:
      using ParameterAcceptor::ParameterAcceptor;

      Number efficiency_;
    };
  } // namespace Interface
} // namespace ryujin

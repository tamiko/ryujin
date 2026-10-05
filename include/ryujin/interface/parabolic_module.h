//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/convenience_macros.h>
#include <ryujin/interface/hyperbolic_module.h>
#include <ryujin/linear_algebra/state_vector.h>

#include <deal.II/base/parameter_acceptor.h>

#include <ostream>
#include <string>

namespace ryujin
{
  namespace Interface
  {
    template <int dim, typename Number = double>
    class ParabolicModule : public dealii::ParameterAcceptor
    {
    public:
      using StateVector = Vectors::StateVector<Number>;

      virtual void prepare() = 0;

      virtual void reinit_state_vector(StateVector &state_vector) const = 0;

      virtual void prepare_state_vector(StateVector &state_vector,
                                        Number t) const = 0;

      virtual void backward_euler_step(const StateVector &old_state_vector,
                                       const Number old_t,
                                       StageVectors<Number> stage_state_vectors,
                                       StageWeights<Number> stage_weights,
                                       StateVector &new_state_vector,
                                       Number tau) const = 0;

      virtual void crank_nicolson_step(const StateVector &old_state_vector,
                                       const Number old_t,
                                       StateVector &new_state_vector,
                                       Number tau) const = 0;

      virtual void print_solver_statistics(std::ostream &output) const = 0;

      void set_id_violation_strategy(const IDViolationStrategy &strategy) const
      {
        id_violation_strategy_ = strategy;
      }

      ACCESSOR_READ_ONLY(n_restarts)

      ACCESSOR_READ_ONLY(n_corrections)

      ACCESSOR_READ_ONLY(n_warnings)

    protected:
      explicit ParabolicModule(const std::string &subsection)
          : ParameterAcceptor(subsection)
          , id_violation_strategy_(IDViolationStrategy::warn)
          , n_restarts_(0)
          , n_corrections_(0)
          , n_warnings_(0)
      {
      }

      mutable IDViolationStrategy id_violation_strategy_;

      mutable unsigned int n_restarts_;
      mutable unsigned int n_corrections_;
      mutable unsigned int n_warnings_;
    };
  } // namespace Interface
} // namespace ryujin

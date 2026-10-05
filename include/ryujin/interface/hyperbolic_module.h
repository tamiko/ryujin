//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/convenience_macros.h>
#include <ryujin/linear_algebra/state_vector.h>

#include <deal.II/base/exceptions.h>
#include <deal.II/base/parameter_acceptor.h>

#include <cstdint>
#include <limits>
#include <string>

namespace ryujin
{
  enum class IDViolationStrategy : std::uint8_t {
    warn,

    raise_exception,
  };


  class Restart final
  {
  public:
    double suggested_tau_max;
  };


  class Correction final
  {
  };


  namespace Interface
  {
    template <int dim, typename Number = double>
    class HyperbolicModule : public dealii::ParameterAcceptor
    {
    public:
      using StateVector = Vectors::StateVector<Number>;

      virtual void prepare() = 0;

      virtual void reinit_state_vector(StateVector &state_vector) const = 0;

      virtual void prepare_state_vector(StateVector &state_vector,
                                        Number t) const = 0;

      virtual void fill_precomputed_values(StateVector &state_vector) const = 0;

      virtual Number
      step(const StateVector &old_state_vector,
           StageVectors<Number> stage_state_vectors,
           StageWeights<Number> stage_weights,
           StateVector &new_state_vector,
           Number tau = Number(0.),
           Number tau_max = std::numeric_limits<Number>::max()) const = 0;

      void set_cfl(Number new_cfl) const
      {
        Assert(cfl_ > Number(0.), dealii::ExcInternalError());
        cfl_ = new_cfl;
      }

      void set_acceptable_tau_max_ratio(Number new_ratio) const
      {
        Assert(new_ratio >= Number(1.), dealii::ExcInternalError());
        acceptable_tau_max_ratio_ = new_ratio;
      }

      void set_id_violation_strategy(const IDViolationStrategy &strategy) const
      {
        id_violation_strategy_ = strategy;
      }

      ACCESSOR_READ_ONLY(cfl)

      ACCESSOR_READ_ONLY(acceptable_tau_max_ratio)

      ACCESSOR_READ_ONLY(n_restarts)

      ACCESSOR_READ_ONLY(n_corrections)

      ACCESSOR_READ_ONLY(n_warnings)

      ACCESSOR_READ_ONLY(problem_name)

      ACCESSOR_READ_ONLY(problem_dimension)

      ACCESSOR_READ_ONLY(n_precomputed_values)

    protected:
      HyperbolicModule(const std::string &problem_name,
                       const unsigned int problem_dimension,
                       const unsigned int n_precomputed_values,
                       const std::string &subsection)
          : ParameterAcceptor(subsection)
          , cfl_(Number(0.2))
          , acceptable_tau_max_ratio_(Number(1.e6))
          , id_violation_strategy_(IDViolationStrategy::warn)
          , n_restarts_(0)
          , n_corrections_(0)
          , n_warnings_(0)
          , problem_name_(problem_name)
          , problem_dimension_(problem_dimension)
          , n_precomputed_values_(n_precomputed_values)
      {
      }

      mutable Number cfl_;
      mutable Number acceptable_tau_max_ratio_;
      mutable IDViolationStrategy id_violation_strategy_;

      mutable unsigned int n_restarts_;
      mutable unsigned int n_corrections_;
      mutable unsigned int n_warnings_;

    private:
      const std::string problem_name_;
      const unsigned int problem_dimension_;
      const unsigned int n_precomputed_values_;
    };
  } // namespace Interface
} // namespace ryujin

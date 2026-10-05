//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2025 - 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/compile_time_options.h>
#include <ryujin/base/mpi_ensemble.h>
#include <ryujin/dependent/initial_values.h>
#include <ryujin/discretization/offline_data.h>
#include <ryujin/interface/parabolic_module.h>
#include <ryujin/stub/parabolic_system.h>

#include <ostream>
#include <string>

namespace ryujin
{
  template <typename HyperbolicDescription, int dim, typename Number = double>
  class StubParabolicModule final
      : public Interface::ParabolicModule<dim, Number>
  {
  public:
    using HyperbolicSystem = typename HyperbolicDescription::HyperbolicSystem;

    using StateVector = Vectors::StateVector<Number>;

    StubParabolicModule(
        const MPIEnsemble &,
        const OfflineData<dim, Number> &,
        const HyperbolicSystem &,
        const StubParabolicSystem &,
        const InitialValues<HyperbolicDescription, dim, Number> &,
        const std::string &subsection = "StubParabolicModule")
        : Interface::ParabolicModule<dim, Number>(subsection)
    {
    }

    void prepare() override {}

    void reinit_state_vector(StateVector &) const override {}

    void prepare_state_vector(StateVector &, Number) const override
    {
      Assert(false,
             dealii::ExcMessage("The parabolic system is the identity. This "
                                "function should have never been called."));
    }

    void backward_euler_step(const StateVector &old_state_vector,
                             const Number,
                             StageVectors<Number>,
                             StageWeights<Number>,
                             StateVector &new_state_vector,
                             Number) const override
    {
      Assert(false,
             dealii::ExcMessage("The parabolic system is the identity. This "
                                "function should have never been called."));
      new_state_vector = old_state_vector;
    }

    void crank_nicolson_step(const StateVector &old_state_vector,
                             const Number,
                             StateVector &new_state_vector,
                             Number) const override
    {
      Assert(false,
             dealii::ExcMessage("The parabolic system is the identity. This "
                                "function should have never been called."));
      new_state_vector = old_state_vector;
    }

    void print_solver_statistics(std::ostream &) const override {}
  };

} /* namespace ryujin */

//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception or LGPL-2.1-or-later
// Copyright (C) 2020 - 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/compile_time_options.h>

#include <ryujin/base/mpi_ensemble.h>
#include <ryujin/base/observer_pointer.h>
#include <ryujin/discretization/offline_data.h>
#include <ryujin/interface/solution_transfer.h>
#include <ryujin/linear_algebra/state_vector.h>

#include <deal.II/base/parameter_acceptor.h>

#include <utility>

namespace ryujin
{
  template <typename HyperbolicDescription, int dim, typename Number = double>
  class SolutionTransfer final : public Interface::SolutionTransfer<dim, Number>
  {
  public:
    using HyperbolicSystem = typename HyperbolicDescription::HyperbolicSystem;

    using View = typename HyperbolicSystem::template View<dim, Number>;

    using Limiter = typename HyperbolicDescription::template Limiter<Number>;

    using LimiterView =
        decltype(std::declval<const Limiter &>().template view<dim, Number>());

    static constexpr auto problem_dimension = View::problem_dimension;

    using state_type = typename View::state_type;

    static constexpr unsigned int n_bounds = LimiterView::n_bounds;

    using Bounds = typename LimiterView::Bounds;

    using StateVector = typename View::StateVector;
    using HyperbolicVector = typename View::HyperbolicVector;

    SolutionTransfer(const MPIEnsemble &mpi_ensemble,
                     const OfflineData<dim, Number> &offline_data,
                     const HyperbolicSystem &hyperbolic_system,
                     const std::string &subsection = "/SolutionTransfer");

    ~SolutionTransfer() override = default;

    void prepare_projection(const StateVector &old_state_vector) override;

    void project(StateVector &new_state_vector) override;

  private:
    Limiter limiter_;

    const MPIEnsemble &mpi_ensemble_;

    dealii::ObserverPointer<const OfflineData<dim, Number>> offline_data_;
    dealii::ObserverPointer<const HyperbolicSystem> hyperbolic_system_;


    state_type read_tensor(const HyperbolicVector &U,
                           const dealii::types::global_dof_index global_i);

    void add_tensor(HyperbolicVector &U,
                    const state_type &U_i,
                    const dealii::types::global_dof_index global_i);
  };
} // namespace ryujin

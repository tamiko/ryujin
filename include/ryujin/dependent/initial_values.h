//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/compile_time_options.h>

#include <ryujin/base/mpi_ensemble.h>
#include <ryujin/base/observer_pointer.h>
#include <ryujin/dependent/initial_state_library.h>
#include <ryujin/discretization/offline_data.h>
#include <ryujin/interface/initial_values.h>

#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/tensor.h>

#include <functional>

namespace ryujin
{
  template <typename HyperbolicDescription, int dim, typename Number = double>
  class InitialValues final : public Interface::InitialValues<dim, Number>
  {
  public:
    using HyperbolicSystem = typename HyperbolicDescription::HyperbolicSystem;

    using View = typename HyperbolicSystem::template View<dim, Number>;

    static constexpr auto problem_dimension = View::problem_dimension;

    using state_type = typename View::state_type;

    static constexpr auto n_initial_precomputed_values =
        View::n_initial_precomputed_values;

    using initial_precomputed_type = typename View::initial_precomputed_type;

    using HyperbolicVector = typename View::HyperbolicVector;

    using InitialPrecomputedVector = typename View::InitialPrecomputedVector;

    InitialValues(const MPIEnsemble &mpi_ensemble,
                  const OfflineData<dim, Number> &offline_data,
                  const HyperbolicSystem &hyperbolic_system,
                  const std::string &subsection = "/InitialValues");

    void parse_parameters_callback();

    DEAL_II_ALWAYS_INLINE inline state_type
    initial_state(const dealii::Point<dim> &point, Number t) const
    {
      return initial_state_(point, t);
    }

    DEAL_II_ALWAYS_INLINE inline initial_precomputed_type
    initial_precomputed(const dealii::Point<dim> &point) const
    {
      return initial_precomputed_(point);
    }

    HyperbolicVector interpolate_hyperbolic_vector(Number t = 0) const override;

    InitialPrecomputedVector interpolate_initial_precomputed_vector() const;

  private:
    std::string configuration_;

    dealii::Point<dim> initial_position_;

    dealii::Tensor<1, dim> initial_direction_;

    Number perturbation_;

    const MPIEnsemble &mpi_ensemble_;

    dealii::ObserverPointer<const OfflineData<dim, Number>> offline_data_;
    dealii::ObserverPointer<const HyperbolicSystem> hyperbolic_system_;


    typename InitialStateLibrary<HyperbolicDescription, dim, Number>::
        initial_state_list_type initial_state_list_;

    std::function<state_type(const dealii::Point<dim> &, Number)>
        initial_state_;

    std::function<initial_precomputed_type(const dealii::Point<dim> &)>
        initial_precomputed_;
  };

} /* namespace ryujin */

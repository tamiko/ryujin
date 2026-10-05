//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/compile_time_options.h>

#include <ryujin/base/convenience_macros.h>
#include <ryujin/base/mpi_ensemble.h>
#include <ryujin/base/observer_pointer.h>
#include <ryujin/dependent/initial_values.h>
#include <ryujin/discretization/offline_data.h>
#include <ryujin/interface/hyperbolic_module.h>
#include <ryujin/linear_algebra/sparse_matrix.h>
#include <ryujin/linear_algebra/state_vector.h>

#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/lac/sparse_matrix.templates.h>
#include <deal.II/lac/vector.h>

#include <functional>
#include <utility>

namespace ryujin
{
  template <typename HyperbolicDescription, int dim, typename Number = double>
  class HyperbolicModule final : public Interface::HyperbolicModule<dim, Number>
  {
  public:
    using HyperbolicSystem = typename HyperbolicDescription::HyperbolicSystem;

    using View = typename HyperbolicSystem::template View<dim, Number>;

    using Indicator =
        typename HyperbolicDescription::template Indicator<Number>;

    using Limiter = typename HyperbolicDescription::template Limiter<Number>;

    using WaveSpeedEstimator =
        typename HyperbolicDescription::template WaveSpeedEstimator<Number>;

    static constexpr auto problem_dimension = View::problem_dimension;

    using state_type = typename View::state_type;

    static constexpr auto n_precomputed_values = View::n_precomputed_values;

    using precomputed_type = typename View::precomputed_type;

    using initial_precomputed_type = typename View::initial_precomputed_type;

    using StateVector = typename View::StateVector;

    using InitialPrecomputedVector = typename View::InitialPrecomputedVector;

    HyperbolicModule(
        const MPIEnsemble &mpi_ensemble,
        const OfflineData<dim, Number> &offline_data,
        const HyperbolicSystem &hyperbolic_system,
        const InitialValues<HyperbolicDescription, dim, Number> &initial_values,
        const std::string &subsection = "/HyperbolicModule");

    void prepare() override;

    void reinit_state_vector(StateVector &state_vector) const override;

    void prepare_state_vector(StateVector &state_vector,
                              Number t) const override;

    void fill_precomputed_values(StateVector &state_vector) const override;

    Number
    step(const StateVector &old_state_vector,
         StageVectors<Number> stage_state_vectors,
         StageWeights<Number> stage_weights,
         StateVector &new_state_vector,
         Number tau = Number(0.),
         Number tau_max = std::numeric_limits<Number>::max()) const override;

    template <int stages>
    Number step(const StateVector &old_state_vector,
                std::array<std::reference_wrapper<const StateVector>, stages>
                    stage_state_vectors,
                const std::array<Number, stages> stage_weights,
                StateVector &new_state_vector,
                Number tau = Number(0.),
                Number tau_max = std::numeric_limits<Number>::max()) const;

    ACCESSOR_READ_ONLY(offline_data)

    ACCESSOR_READ_ONLY(hyperbolic_system)

    ACCESSOR_READ_ONLY(initial_precomputed)

    ACCESSOR_READ_ONLY(alpha)

  private:
    using HyperbolicVector = Vectors::MultiComponentVector<Number>;

    template <typename MemorySpace>
    void apply_boundary_conditions(HyperbolicVector &U, const Number t) const;

    Indicator indicator_;

    Limiter limiter_;

    WaveSpeedEstimator wave_speed_estimator_;

    const MPIEnsemble &mpi_ensemble_;

    dealii::ObserverPointer<const OfflineData<dim, Number>> offline_data_;
    dealii::ObserverPointer<const HyperbolicSystem> hyperbolic_system_;
    dealii::ObserverPointer<
        const InitialValues<HyperbolicDescription, dim, Number>>
        initial_values_;

    InitialPrecomputedVector initial_precomputed_;

    using ScalarVector = typename Vectors::ScalarVector<Number>;
    mutable ScalarVector alpha_;

    static constexpr auto n_bounds =
        decltype(std::declval<const Limiter &>()
                     .template view<dim, Number>())::n_bounds;
    mutable Vectors::MultiComponentVector<Number> bounds_;

    mutable HyperbolicVector r_;

    mutable Mirrored<Number *> boundary_states_{
        "hyperbolic_module_boundary_states"};

    mutable SparseMatrix<Number> dij_matrix_;
    mutable SparseMatrix<Number> lij_matrix_;
    mutable SparseMatrix<Number> lij_matrix_next_;
    mutable SparseMatrix<Number, problem_dimension> pij_matrix_;
  };

} /* namespace ryujin */

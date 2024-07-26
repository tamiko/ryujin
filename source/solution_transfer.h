//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception or LGPL-2.1-or-later
// Copyright (C) 2007 - 2024 by the deal.II authors
// Copyright (C) 2024 - 2024 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include "offline_data.h"
#include "state_vector.h"

#include <deal.II/base/parameter_acceptor.h>

#include <optional>

namespace ryujin
{
  /**
   * The SolutionTransfer class is an adaptation of the
   * parallel::distributed::SolutionTransfer class and method implemented
   * in deal.II. This class is, in contrast to the deal.II version,
   * specifically taylored to our needs:
   *  - it uses the MultiComponentVector directly,
   *  - it implements a local mass matrix projection with convex limiting.
   *
   * @ingroup Mesh
   */
  template <typename Description, int dim, typename Number = double>
  class SolutionTransfer
  {
  public:
    /**
     * @name Typedefs and constexpr constants
     */
    //@{

    using HyperbolicSystem = typename Description::HyperbolicSystem;
    using ParabolicSystem = typename Description::ParabolicSystem;

    using View =
        typename Description::template HyperbolicSystemView<dim, Number>;

    static constexpr auto problem_dimension = View::problem_dimension;

    using state_type = typename View::state_type;

    using StateVector = typename View::StateVector;

    //@}
    /**
     * @name Constructor and setup
     */
    //@{

    /**
     * Constructor
     */
    SolutionTransfer(const MPI_Comm &mpi_communicator,
                     Discretization<dim>::Triangulation &triangulation,
                     const OfflineData<dim, Number> &offline_data,
                     const HyperbolicSystem &hyperbolic_system,
                     const ParabolicSystem &parabolic_system);

    /**
     * Destructor
     */
    ~SolutionTransfer() = default;

    //@}
    /**
     * @name Methods for solution transfer after mesh adaptation and
     * serialization/deserialization.
     */
    //@{

    /**
     * Prepare projection and serialization by registering a callback to
     * all cells of the distributed triangulation.
     */
    void
    prepare_projection_and_serialization(const StateVector &old_state_vector);

    /**
     * Project the stored data onto the new triangulation and store the
     * result in @p new_state_vector.
     *
     * @note After mesh refinement all internal data structures stored in
     * the OfflineData object must be reinitialized with a call to
     * prepare() and the @p new_state_vector must be appropriately
     * initialized with the new Partitioner.
     */
    void project(StateVector &new_state_vector);

    /**
     * Read back a previously stored StateVector via Triangulation::load().
     *
     * @note After reading in the checkpointed mesh the OfflineData object
     * must be reinitialized with a call to prepare() and the @p
     * new_state_vector must be appropriately initialized with the new
     * Partitioner.
     */
    void deserialize(StateVector &new_state_vector);

  private:
    //@}
    /**
     * @name Internal data
     */
    //@{

    const MPI_Comm &mpi_communicator_;

    dealii::SmartPointer<typename Discretization<dim>::Triangulation>
        triangulation_;
    dealii::SmartPointer<const OfflineData<dim, Number>> offline_data_;
    dealii::SmartPointer<const HyperbolicSystem> hyperbolic_system_;
    dealii::SmartPointer<const ParabolicSystem> parabolic_system_;

    const StateVector *old_state_vector_;
    unsigned int handle;

    void register_data_attach();
  };
} // namespace ryujin

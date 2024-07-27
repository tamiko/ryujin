//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception or LGPL-2.1-or-later
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
    using HyperbolicVector = typename View::HyperbolicVector;

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
     * Prepare projection (and serialization) by registering a call back to
     * all cells of the distributed triangulation. The call back attaches
     * all necessary data to the triangulation to perform a projection
     * after mesh adaptation, or a Triangulation::store()/load() operation.
     */
    void prepare_projection(const StateVector &old_state_vector);

    /**
     * Return the handle associated with the call back that was set by
     * prepare_projection() and the associated data attached to the
     * triangulation.
     *
     * @pre Can only be called after a call to prepare_projection().
     */
    unsigned int get_handle() const {
      Assert(handle_ != dealii::numbers::invalid_unsigned_int,
             dealii::ExcMessage("Invalid handle: Cannot retrieve a valid "
                                "handle because get_handle() can only be "
                                "called after a call to prepare_projection()"));
      return handle_;
    }

    /**
     * Set the handle associated data attached to the triangulation. This
     * function has to be called after a Triangulation::load() operation
     * and prior to the SolutionTransfer::project().
     *
     * @pre Cannot be called after a call to prepare_projection().
     */
    void set_handle(unsigned int handle)
    {
      Assert(handle_ == dealii::numbers::invalid_unsigned_int,
             dealii::ExcMessage(
                 "Invalid state: Cannot set handle because we already have a "
                 "valid handle due to a prior call to prepare_projection()."));
      handle_ = handle;
    }

    /**
     * Project the data stored in the triangulation into a new state vector
     * @p new_state_vector. The attached data either comes from a previous
     * call to prepare_projection() prior to mesh adaptation or has been
     * read back in from a checkpoint after a call to
     * Triangulation::load().
     *
     * @note After mesh refinement all internal data structures stored in
     * the OfflineData object must be reinitialized with a call to
     * prepare() and the @p new_state_vector must be appropriately
     * initialized with the new Partitioner.
     */
    void project(StateVector &new_state_vector);

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

    unsigned int handle;

    /**
     * For the local mass projection to work properly we need to repopulate
     * constrained degrees of freedom. Ordinarily this would happen with
     * AffineConstraints<>::distribute() - but this function can not work
     * on our MultiComponentVector. So we implement a small helper to do
     * the operation by hand.
     */
    state_type
    get_tensor_with_constraints_distributed(const HyperbolicVector &U,
                                            const unsigned int local_index);
  };
} // namespace ryujin

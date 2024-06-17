//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception or LGPL-2.1-or-later
// Copyright (C) 2007 - 2024 by the deal.II authors
// Copyright (C) 2024 - 2024 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include "discretization.h"
#include "solution_transfer.h"

#include <deal.II/base/config.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/petsc_block_vector.h>
#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/trilinos_parallel_block_vector.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/vector.h>


namespace ryujin
{
  template <typename Description, int dim, typename Number>
  SolutionTransfer<Description, dim, Number>::SolutionTransfer(
      const MPI_Comm &mpi_communicator,
      const OfflineData<dim, Number> &offline_data,
      const HyperbolicSystem &hyperbolic_system,
      const ParabolicSystem &parabolic_system)
      : mpi_communicator_(mpi_communicator)
      , offline_data_(&offline_data)
      , hyperbolic_system_(&hyperbolic_system)
      , parabolic_system_(&parabolic_system)
      , handle(dealii::numbers::invalid_unsigned_int)
  {
#ifdef DEBUG_OUTPUT
    std::cout << "SolutionTransfer<Description, dim, "
                 "Number>::prepare_projection_and_serialization()"
              << std::endl;
#endif

    AssertThrow(have_distributed_triangulation<dim>,
                dealii::ExcMessage(
                    "The SolutionTransfer class is not implemented for a "
                    "distributed::shared::Triangulation which we use in 1D"));
  }


  template <typename Description, int dim, typename Number>
  void SolutionTransfer<Description, dim, Number>::
      prepare_projection_and_serialization(const StateVector &old_state_vector)
  {
#ifdef DEBUG_OUTPUT
    std::cout << "SolutionTransfer<Description, dim, "
                 "Number>::prepare_projection_and_serialization()"
              << std::endl;
#endif

    AssertThrow(have_distributed_triangulation<dim>,
                dealii::ExcMessage(
                    "The SolutionTransfer class is not implemented for a "
                    "distributed::shared::Triangulation which we use in 1D"));

    register_data_attach();
  }


  template <typename Description, int dim, typename Number>
  void SolutionTransfer<Description, dim, Number>::deserialize(
      StateVector &new_state_vector)
  {
#ifdef DEBUG_OUTPUT
    std::cout << "SolutionTransfer<Description, dim, Number>::deserialize()"
              << std::endl;
#endif

    AssertThrow(have_distributed_triangulation<dim>,
                dealii::ExcMessage(
                    "The SolutionTransfer class is not implemented for a "
                    "distributed::shared::Triangulation which we use in 1D"));

    register_data_attach();
    project(new_state_vector);
  }


  template <typename Description, int dim, typename Number>
  void SolutionTransfer<Description, dim, Number>::project(
      StateVector &new_state_vector)
  {
#ifdef DEBUG_OUTPUT
    std::cout << "SolutionTransfer<Description, dim, Number>::deserialize()"
              << std::endl;
#endif

    AssertThrow(have_distributed_triangulation<dim>,
                dealii::ExcMessage(
                    "The SolutionTransfer class is not implemented for a "
                    "distributed::shared::Triangulation which we use in 1D"));

    // implement me
    __builtin_trap();
  }


  namespace
  {
    /**
     * Pack a vector of local state values into a char vector.
     */
    template <typename state_type>
    std::vector<char>
    pack_dof_values(const std::vector<state_type> &state_values)
    {
      std::vector<char> buffer(sizeof(state_type) * state_values.size());
      std::memcpy(buffer.data(), state_values.data(), buffer.size());
      return buffer;
    }


    /**
     * Unpack a char vector into a vector of local state values.
     */
    template <typename state_type>
    std::vector<state_type> unpack_dof_values(
        const boost::iterator_range<std::vector<char>::const_iterator>
            &data_range)
    {
      const std::size_t n_bytes = data_range.size();
      Assert(n_bytes % sizeof(state_type) == 0, dealii::ExcInternalError());
      std::vector<state_type> state_values(n_bytes / sizeof(state_type));
      std::memcpy(state_values.data(),
                  &data_range[0],
                  state_values.size() * sizeof(state_type));
      return state_values;
    }
  } // namespace


  template <typename Description, int dim, typename Number>
  void SolutionTransfer<Description, dim, Number>::register_data_attach()
  {
    // implement me
    __builtin_trap();
  }
} // namespace ryujin

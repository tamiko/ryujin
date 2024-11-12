//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2024 by the ryujin authors
//

#include "mpi_ensemble.h"

namespace ryujin
{
  MPIEnsemble::MPIEnsemble(const MPI_Comm &mpi_communicator)
      : world_communicator_(mpi_communicator)
      , n_ensembles_(1)
      , ensemble_(0)
      , subrange_communicator_(mpi_communicator)
      , global_tau_max_(true)
  {
  }

  void MPIEnsemble::prepare(const int n_ensembles /* = 1 */,
                            const bool global_tau_max /* = true */)
  {
    const auto n_world_ranks =
        dealii::Utilities::MPI::n_mpi_processes(world_communicator_);

    AssertThrow(n_ensembles > 0, dealii::ExcInternalError());
    AssertThrow(global_tau_max, dealii::ExcNotImplemented());
    AssertThrow(n_world_ranks % n_ensembles == 0,
                dealii::ExcMessage(
                    "The total number of (world) MPI ranks must be a multiple "
                    "of the number of ensembles. But we are scheduled with " +
                    std::to_string(n_world_ranks) + " for running " +
                    std::to_string(n_ensembles) + " ensembles."));

    n_ensembles_ = n_ensembles;
    global_tau_max_ = global_tau_max;

    const auto world_rank =
        dealii::Utilities::MPI::this_mpi_process(world_communicator_);

    // FIXME: use a smarter binning strategy (or add a parameter).
    ensemble_ = world_rank % n_ensembles_;

    MPI_Comm_split(
        world_communicator_, ensemble_, world_rank, &subrange_communicator_);

#ifdef DEBUG_OUTPUT
    const auto n_world_ranks =
        dealii::Utilities::MPI::n_mpi_processes(world_communicator_);
    const auto subrange_rank =
        dealii::Utilities::MPI::this_mpi_process(subrange_communicator_);
    const auto n_subrange_ranks =
        dealii::Utilities::MPI::n_mpi_processes(subrange_communicator_);
    std::cout << "RANK: (g = " << world_rank << ", l = " << subrange_rank
              << ") out of (g = " << n_world_ranks
              << ", l = " << n_subrange_ranks << ") -> COLOR: " << ensemble_
              << std::endl;
#endif
  }
} /* namespace ryujin */

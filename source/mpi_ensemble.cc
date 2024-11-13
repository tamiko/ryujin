//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2024 by the ryujin authors
//

#include "mpi_ensemble.h"

namespace ryujin
{
  MPIEnsemble::MPIEnsemble(const MPI_Comm &mpi_communicator)
      : world_communicator_(mpi_communicator)
      , world_rank_(0)
      , n_world_ranks_(1)
      , ensemble_(0)
      , n_ensembles_(1)
      , ensemble_rank_(0)
      , n_ensemble_ranks_(1)
      , global_tau_max_(true)
  {
  }

  MPIEnsemble::~MPIEnsemble()
  {
    MPI_Group_free(&world_group_);
    for (auto &it : subrange_groups_)
      MPI_Group_free(&it);
    MPI_Group_free(&subrange_leader_group_);

    MPI_Comm_free(&subrange_communicator_);
    MPI_Comm_free(&subrange_leader_communicator_);
    MPI_Comm_free(&peer_communicator_);
  }

  void MPIEnsemble::prepare(
      const int n_ensembles /* = 1 */,
      const bool global_tau_max /* = true */,
      const bool require_uniform_ensemble_partition /* = true */)
  {
    n_ensembles_ = n_ensembles;
    global_tau_max_ = global_tau_max;

    world_rank_ = dealii::Utilities::MPI::this_mpi_process(world_communicator_);
    n_world_ranks_ =
        dealii::Utilities::MPI::n_mpi_processes(world_communicator_);

    AssertThrow(n_ensembles_ > 0, dealii::ExcInternalError());
    if (require_uniform_ensemble_partition)
      AssertThrow(
          n_world_ranks_ % n_ensembles_ == 0,
          dealii::ExcMessage(
              "The total number of (world) MPI ranks must be a multiple "
              "of the number of ensembles. But we are scheduled with " +
              std::to_string(n_world_ranks_) + " for running " +
              std::to_string(n_ensembles_) + " ensembles."));

    ensemble_ = world_rank_ % n_ensembles_;

    /*
     * For each ensemble we create an MPI group with a ensemble-local
     * communicator. This allows to do ensemble-independent time-stepping.
     */

    auto ierr = MPI_Comm_group(world_communicator_, &world_group_);
    AssertThrowMPI(ierr);

    /* subrange communicator: */

    subrange_groups_.resize(n_ensembles_);
    for (int ensemble = 0; ensemble < n_ensembles_; ++ensemble) {
      int ranges[1][3]{{ensemble, n_world_ranks_ - 1, n_ensembles_}}; // NOLINT
      ierr = MPI_Group_range_incl(
          world_group_, 1, ranges, &subrange_groups_[ensemble]);
      AssertThrowMPI(ierr);
    }

    ierr = MPI_Comm_create_group(world_communicator_,
                                 subrange_groups_[ensemble_],
                                 ensemble_,
                                 &subrange_communicator_);
    AssertThrowMPI(ierr);

    /* subrange leader communicator: */

    ensemble_rank_ =
        dealii::Utilities::MPI::this_mpi_process(subrange_communicator_);
    n_ensemble_ranks_ =
        dealii::Utilities::MPI::n_mpi_processes(subrange_communicator_);
    AssertThrow(
        (ensemble_rank_ == 0 && world_rank_ < n_ensembles_) ||
            (ensemble_rank_ > 0 && world_rank_ >= n_ensembles_),
        dealii::ExcMessage("MPI Ensemble: Something went horribly wrong: Could "
                           "not determine subrange leader."));

    int ranges[1][3]{{0, n_ensembles_ - 1, 1}}; // NOLINT
    ierr =
        MPI_Group_range_incl(world_group_, 1, ranges, &subrange_leader_group_);
    AssertThrowMPI(ierr);

    ierr = MPI_Comm_create(world_communicator_,
                           subrange_leader_group_,
                           &subrange_leader_communicator_);

    /* peer communicator: */

    ierr = MPI_Comm_split(
        world_communicator_, ensemble_rank_, ensemble_, &peer_communicator_);
    AssertThrowMPI(ierr);

#define DEBUG_OUTPUT
#ifdef DEBUG_OUTPUT
    const auto peer_rank =
        dealii::Utilities::MPI::this_mpi_process(peer_communicator_);
    const auto n_peer_ranks =
        dealii::Utilities::MPI::n_mpi_processes(peer_communicator_);
    std::cout << "RANK: (w = " << world_rank_ << ", e = " << ensemble_rank_
              << ", p = " << peer_rank << ") out of (w = " << n_world_ranks_
              << ", e = " << n_ensemble_ranks_ << ", p = " << n_peer_ranks
              << ") -> belongig to ensemble: " << ensemble_ << std::endl;
#endif
  }
} /* namespace ryujin */

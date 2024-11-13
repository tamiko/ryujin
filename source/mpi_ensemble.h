//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2024 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include "convenience_macros.h"

#include <deal.II/base/mpi.h>
#include <deal.II/base/utilities.h>

namespace ryujin
{
  /**
   * A class responsible for subdividing a given global MPI communicator
   * into a set of "ensembles" with a coresponding subrange communicator.
   *
   * After @p prepare() is called, all getter functions return valid
   * references.
   *
   * @ingroup Miscellaneous
   */
  class MPIEnsemble final
  {
  public:
    MPIEnsemble(const MPI_Comm &mpi_communicator);

    ~MPIEnsemble();

    /**
     * Prepare the MPI ensemble and split the gobal MPI communicator into
     * @p n_ensembles different subranges of comparable size.
     *
     * @pre The total number of mpi ranks must be an integer multiple of
     * n_ensembles.
     */
    void prepare(const int n_ensembles = 1,
                 const bool global_tau_max = true,
                 const bool require_uniform_ensemble_partition = true);

    /**
     * Return the world communicator.
     */
    ACCESSOR_READ_ONLY_NO_DEREFERENCE(world_communicator);

    /**
     * The (global) world rank of the current MPI process.
     */
    ACCESSOR_READ_ONLY(world_rank);

    /**
     * The total number of (global) MPI processes.
     */
    ACCESSOR_READ_ONLY(n_world_ranks);

    /**
     * Return the ensemble in the interval [0, n_ensembles) that the given
     * MPI process belongs to.
     */
    ACCESSOR_READ_ONLY(ensemble);

    /**
     * Return the total number of ensembles.
     */
    ACCESSOR_READ_ONLY(n_ensembles);

    /**
     * The (local) ensemble rank of the current MPI process.
     */
    ACCESSOR_READ_ONLY(ensemble_rank);

    /**
     * The total number of (local) MPI processes belonging to the ensemble.
     */
    ACCESSOR_READ_ONLY(n_ensemble_ranks);

    /**
     * The corresponding subrange communicator of the ensemble. The
     * communicator is collective over all ranks participating in the
     * ensemble. I.e., it allows for ensemble-local communication.
     */
    ACCESSOR_READ_ONLY_NO_DEREFERENCE(subrange_communicator);

    /**
     * A communicator spanning over all ensemble leaders that have ensemble
     * rank 0. This communicator is collective over all ensemble leaders
     * and invalid for all other ranks.
     */
    ACCESSOR_READ_ONLY_NO_DEREFERENCE(subrange_leader_communicator);

    /**
     * A "peer communicator" that groups all kth ranks of each ensemble
     * together. (Suppose the subrange_communicator() groups rows, then the
     * peer_communicator() groups columns of the ensemble partition). The
     * peer communicator is collective over all world ranks.
     */
    ACCESSOR_READ_ONLY_NO_DEREFERENCE(peer_communicator);

    /**
     * Return whether the ensemble has to be run with a global tau_max
     * constraint in which every ensemble member performs an update with
     * the same time step, or whether synchronization is only performed
     * over the ensemble.
     */
    ACCESSOR_READ_ONLY(global_tau_max);

  private:
    const MPI_Comm &world_communicator_;

    int world_rank_;
    int n_world_ranks_;
    int ensemble_;
    int n_ensembles_;
    int ensemble_rank_;
    int n_ensemble_ranks_;

    MPI_Group world_group_;
    std::vector<MPI_Group> subrange_groups_;
    MPI_Group subrange_leader_group_;

    MPI_Comm subrange_communicator_;
    MPI_Comm subrange_leader_communicator_;
    MPI_Comm peer_communicator_;

    bool global_tau_max_;
  };
} /* namespace ryujin */

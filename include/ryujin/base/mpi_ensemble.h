//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2024 - 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/compile_time_options.h>

#include <ryujin/base/convenience_macros.h>

#include <deal.II/base/mpi.h>
#include <deal.II/base/utilities.h>

namespace ryujin
{
  class MPIEnsemble final
  {
  public:
    MPIEnsemble(const MPI_Comm &mpi_communicator,
                const int n_ensembles = 1,
                const bool global_synchronization = true);

    ~MPIEnsemble();

    ACCESSOR_READ_ONLY_NO_DEREFERENCE(world_communicator);

    ACCESSOR_READ_ONLY(global_synchronization);

    ACCESSOR_READ_ONLY(world_rank);

    ACCESSOR_READ_ONLY(n_world_ranks);

    ACCESSOR_READ_ONLY(ensemble);

    ACCESSOR_READ_ONLY(n_ensembles);

    ACCESSOR_READ_ONLY(ensemble_rank);

    ACCESSOR_READ_ONLY(n_ensemble_ranks);

    ACCESSOR_READ_ONLY_NO_DEREFERENCE(ensemble_communicator);

    DEAL_II_ALWAYS_INLINE inline const MPI_Comm &
    synchronization_communicator() const
    {
      if (global_synchronization_)
        return world_communicator_;
      else
        return ensemble_communicator_;
    }

    ACCESSOR_READ_ONLY_NO_DEREFERENCE(ensemble_leader_communicator);

    ACCESSOR_READ_ONLY_NO_DEREFERENCE(peer_communicator);

  private:
    const MPI_Comm &world_communicator_;

    bool global_synchronization_;

    int world_rank_;
    int n_world_ranks_;
    int ensemble_;
    int n_ensembles_;
    int ensemble_rank_;
    int n_ensemble_ranks_;

    MPI_Group world_group_;
    std::vector<MPI_Group> ensemble_groups_;
    MPI_Group ensemble_leader_group_;

    MPI_Comm ensemble_communicator_;
    MPI_Comm ensemble_leader_communicator_;
    MPI_Comm peer_communicator_;
  };
} /* namespace ryujin */

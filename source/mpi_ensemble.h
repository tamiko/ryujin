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

    /**
     * Prepare the MPI ensemble and split the gobal MPI communicator into
     * @p n_ensembles different subranges of comparable size.
     */
    void prepare(const int n_ensembles = 1, const bool global_tau_max = true);

    /**
     * Return the world communicator.
     */
    ACCESSOR_READ_ONLY_NO_DEREFERENCE(world_communicator);

    /**
     * Return the total number of ensembles.
     */
    ACCESSOR_READ_ONLY(n_ensembles);

    /**
     * Return the ensemble in the interval [0, n_ensembles) that the given
     * rank belongs to.
     */
    ACCESSOR_READ_ONLY(ensemble);

    /**
     * The corresponding subrange communicator for the ensemble.
     */
    ACCESSOR_READ_ONLY_NO_DEREFERENCE(subrange_communicator);

    /**
     * Return whether the ensemble has to be run with a global tau_max
     * constraint in which every ensemble member performs an update with
     * the same time step, or whether synchronization is only performed
     * over the ensemble.
     */
    ACCESSOR_READ_ONLY(global_tau_max);

  private:
    const MPI_Comm &world_communicator_;

    int n_ensembles_;
    int ensemble_;

    MPI_Comm subrange_communicator_;

    bool global_tau_max_;
  };
} /* namespace ryujin */

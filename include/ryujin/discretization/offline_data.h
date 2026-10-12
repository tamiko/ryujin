//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/compile_time_options.h>

#include <ryujin/base/mpi_ensemble.h>
#include <ryujin/interface/offline_data.h>

#include <deal.II/base/partitioner.h>

#include <string>

namespace ryujin
{
  template <int dim, typename Number = double>
  class OfflineData final : public Interface::OfflineData<dim, Number>
  {
  public:
    using typename Interface::OfflineData<dim, Number>::BoundaryDescription;
    using typename Interface::OfflineData<dim, Number>::CouplingDescription;

    OfflineData(const MPIEnsemble &mpi_ensemble,
                const Interface::Discretization<dim> &discretization,
                const std::string &subsection = "/OfflineData");

    void prepare(const unsigned int problem_dimension,
                 const unsigned int n_precomputed_values) override;

  private:
    using typename Interface::OfflineData<dim, Number>::BoundaryMap;
    using typename Interface::OfflineData<dim, Number>::CouplingBoundaryPairs;

    bool treat_fe_nothing_as_boundary_;
    double incidence_relaxation_even_;
    double incidence_relaxation_odd_;

    const MPIEnsemble &mpi_ensemble_;

    void create_dof_handlers();

    void renumber_for_simd();

    void create_constraints_and_sparsity_pattern();

    void ensure_simd_stride_consistency();

    void create_partitioner_and_simd_sparsity(
        const unsigned int problem_dimension,
        const unsigned int n_precomputed_values);

    void create_matrices();

    void create_multigrid_data();

    template <typename ITERATOR1, typename ITERATOR2>
    BoundaryMap construct_boundary_map(
        const ITERATOR1 &begin,
        const ITERATOR2 &end,
        const dealii::Utilities::MPI::Partitioner &partitioner) const;

    template <typename ITERATOR1, typename ITERATOR2>
    CouplingBoundaryPairs collect_coupling_boundary_pairs(
        const ITERATOR1 &begin,
        const ITERATOR2 &end,
        const dealii::Utilities::MPI::Partitioner &partitioner) const;
  };

} /* namespace ryujin */

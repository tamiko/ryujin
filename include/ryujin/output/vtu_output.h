//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/compile_time_options.h>

#include <ryujin/base/mpi_ensemble.h>
#include <ryujin/base/observer_pointer.h>
#include <ryujin/discretization/offline_data.h>
#include <ryujin/interface/selected_components_extractor.h>
#include <ryujin/output/postprocessor.h>

#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/grid/intergrid_map.h>
#include <deal.II/multigrid/mg_transfer_matrix_free.h>

namespace ryujin
{
  template <int dim, typename Number = double>
  class VTUOutput final : public dealii::ParameterAcceptor
  {
  public:
    using StateVector = Vectors::StateVector<Number>;

    VTUOutput(const MPIEnsemble &mpi_ensemble,
              const OfflineData<dim, Number> &offline_data,
              Interface::SelectedComponentsExtractor<dim, Number> &extractor,
              const Postprocessor<dim, Number> &postprocessor,

              const std::string &subsection = "/VTUOutput");

    void prepare();

    void schedule_output(const StateVector &state_vector,
                         std::string name,
                         Number t,
                         unsigned int cycle,
                         bool output_full = true,
                         bool output_cutplanes = true);

  private:
    bool use_mpi_io_;

    std::vector<std::string> manifolds_;

    std::vector<std::string> vtu_output_quantities_;

    const MPIEnsemble &mpi_ensemble_;

    dealii::ObserverPointer<const OfflineData<dim, Number>> offline_data_;
    dealii::ObserverPointer<const Postprocessor<dim, Number>> postprocessor_;

    Interface::SelectedComponentsExtractor<dim, Number>
        &selected_components_extractor_;
  };

} /* namespace ryujin */

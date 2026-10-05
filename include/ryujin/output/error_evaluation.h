//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/compile_time_options.h>

#include <ryujin/base/mpi_ensemble.h>
#include <ryujin/base/observer_pointer.h>
#include <ryujin/discretization/offline_data.h>
#include <ryujin/interface/selected_components_extractor.h>

#include <deal.II/base/parameter_acceptor.h>

#include <ostream>
#include <string>
#include <vector>

namespace ryujin
{
  template <int dim, typename Number = double>
  class ErrorEvaluation final : public dealii::ParameterAcceptor
  {
  public:
    using StateVector = Vectors::StateVector<Number>;
    using ScalarHostVector = Vectors::ScalarHostVector<Number>;

    ErrorEvaluation(
        const MPIEnsemble &mpi_ensemble,
        const OfflineData<dim, Number> &offline_data,
        Interface::SelectedComponentsExtractor<dim, Number> &extractor,
        const std::string &subsection = "/ErrorEvaluation");

    void prepare(const std::string &name);

    std::vector<Number> compute(const StateVector &state_vector,
                                const StateVector &analytic) const;

    void write_out(const StateVector &state_vector,
                   const StateVector &analytic,
                   Number t) const;

    void print_summary(std::ostream &stream,
                       Number t,
                       dealii::types::global_dof_index n_global_dofs,
                       const std::vector<Number> &norms) const;

  private:
    std::vector<std::string> error_quantities_;

    bool error_normalize_;

    std::vector<std::string> error_norms_;

    const MPIEnsemble &mpi_ensemble_;

    dealii::ObserverPointer<const OfflineData<dim, Number>> offline_data_;

    Interface::SelectedComponentsExtractor<dim, Number>
        &selected_components_extractor_;

    std::string base_name_;

    std::string description() const;
  };

} /* namespace ryujin */

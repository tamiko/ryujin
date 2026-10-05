//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/compile_time_options.h>

#include <ryujin/base/convenience_macros.h>
#include <ryujin/base/mpi_ensemble.h>
#include <ryujin/base/observer_pointer.h>
#include <ryujin/discretization/offline_data.h>
#include <ryujin/interface/selected_components_extractor.h>

#include <deal.II/base/parameter_acceptor.h>

namespace ryujin
{
  template <int dim, typename Number = double>
  class Postprocessor final : public dealii::ParameterAcceptor
  {
  public:
    template <typename T>
    using grad_type = dealii::Tensor<1, dim, T>;

    template <typename T>
    using curl_type = dealii::Tensor<1, dim == 2 ? 1 : dim, T>;

    using StateVector = Vectors::StateVector<Number>;

    Postprocessor(
        const MPIEnsemble &mpi_ensemble,
        const OfflineData<dim, Number> &offline_data,
        Interface::SelectedComponentsExtractor<dim, Number> &extractor,
        const std::string &subsection = "/Postprocessor");

    void prepare();

    void compute(const StateVector &state_vector) const;

    void reset_bounds() const
    {
      bounds_.clear();
    }

    unsigned int n_quantities() const
    {
      return quantities_.size();
    }

    ACCESSOR_READ_ONLY(component_names)

    ACCESSOR_READ_ONLY(quantities)

  private:
    bool recompute_bounds_;
    Number beta_;

    std::vector<std::string> schlieren_quantities_;
    std::vector<std::string> vorticity_quantities_;

    const MPIEnsemble &mpi_ensemble_;

    dealii::ObserverPointer<const OfflineData<dim, Number>> offline_data_;
    Interface::SelectedComponentsExtractor<dim, Number>
        &selected_components_extractor_;

    std::vector<std::string> component_names_;
    mutable std::vector<std::pair<Number, Number>> bounds_;
    using ScalarHostVector = Vectors::ScalarHostVector<Number>;
    mutable std::vector<ScalarHostVector> quantities_;
  };

} // namespace ryujin

//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/compile_time_options.h>

#include <ryujin/base/gpu.h>
#include <ryujin/base/mpi_ensemble.h>
#include <ryujin/base/observer_pointer.h>
#include <ryujin/discretization/offline_data.h>
#include <ryujin/interface/selected_components_extractor.h>

#include <deal.II/base/parameter_acceptor.h>

#include <optional>

namespace ryujin
{
  template <int dim, typename Number = double>
  class Quantities final : public dealii::ParameterAcceptor
  {
  public:
    using StateVector = Vectors::StateVector<Number>;

    Quantities(const MPIEnsemble &mpi_ensemble,
               const OfflineData<dim, Number> &offline_data,
               Interface::SelectedComponentsExtractor<dim, Number> &extractor,
               const std::string &subsection = "/Quantities");

    void prepare(const std::string &name);

    void accumulate(const StateVector &state_vector, const Number t);

    void write_out(const StateVector &state_vector,
                   const Number t,
                   unsigned int cycle);

  private:
    using ManifoldPoint =
        typename OfflineData<dim, Number>::BoundaryDescription;

    struct Manifold {
      std::string name;
      bool boundary;
      bool instantaneous;
      bool time_averaged;
      bool space_averaged;

      std::vector<ManifoldPoint> points;

      Mirrored<unsigned int *> indices{"quantities_indices"};
      Mirrored<Number *> masses{"quantities_masses"};
      Number mass_sum;

      Mirrored<Number *> old{"quantities_old"};
      Mirrored<Number *> current{"quantities_current"};
      Mirrored<Number *> sum{"quantities_sum"};
      Number t_old;
      Number t_new;
      Number t_sum;

      std::vector<std::pair<Number, std::vector<Number>>> time_series;
      std::optional<unsigned int> time_series_cycle;
    };

    std::vector<std::string> quantities_;

    unsigned int n_moments_;

    std::vector<std::tuple<std::string, std::string, std::string>>
        interior_manifolds_;

    std::vector<std::tuple<std::string, std::string, std::string>>
        boundary_manifolds_;

    bool clear_temporal_statistics_on_writeout_;

    const MPIEnsemble &mpi_ensemble_;

    dealii::ObserverPointer<const OfflineData<dim, Number>> offline_data_;

    Interface::SelectedComponentsExtractor<dim, Number> &extractor_;

    std::vector<Manifold> manifolds_;

    std::string base_name_;
    bool mesh_files_have_been_written_;

    unsigned int stride() const;

    std::string header(bool averaged) const;

    void write_mesh_files(unsigned int cycle);

    void clear_statistics();

    std::vector<Number> internal_accumulate(Manifold &manifold);

    void internal_write_out(const std::string &file_name,
                            const std::string &time_stamp,
                            const Mirrored<Number *> &values,
                            const Number scale,
                            bool averaged);

    void internal_write_out_time_series(
        const std::string &file_name,
        const std::vector<std::pair<Number, std::vector<Number>>> &values,
        bool append);
  };

} /* namespace ryujin */

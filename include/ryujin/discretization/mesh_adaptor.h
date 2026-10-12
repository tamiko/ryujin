//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2024 - 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/compile_time_options.h>

#include <ryujin/base/mpi_ensemble.h>
#include <ryujin/base/observer_pointer.h>
#include <ryujin/interface/mesh_adaptor.h>
#include <ryujin/interface/offline_data.h>
#include <ryujin/interface/selected_components_extractor.h>

#include <deal.II/lac/vector.h>

#include <random>

namespace ryujin
{
  enum class AdaptationStrategy {
    global_refinement,

    random_adaptation,

    smoothness_indicators,
  };

  enum class MarkingStrategy {
    fixed_threshold,
  };

  enum class TimePointSelectionStrategy {
    fixed_time_points,

    simulation_cycle,
  };
} // namespace ryujin

#ifndef DOXYGEN
DECLARE_ENUM(
    ryujin::AdaptationStrategy,
    LIST({ryujin::AdaptationStrategy::global_refinement, "global refinement"},
         {ryujin::AdaptationStrategy::random_adaptation, "random adaptation"},
         {ryujin::AdaptationStrategy::smoothness_indicators,
          "smoothness indicators"}, ));

DECLARE_ENUM(ryujin::MarkingStrategy,
             LIST({ryujin::MarkingStrategy::fixed_threshold,
                   "fixed threshold"}));

DECLARE_ENUM(ryujin::TimePointSelectionStrategy,
             LIST({ryujin::TimePointSelectionStrategy::fixed_time_points,
                   "fixed time points"},
                  {ryujin::TimePointSelectionStrategy::simulation_cycle,
                   "simulation cycle"}, ));
#endif

namespace ryujin
{
  template <int dim, typename Number = double>
  class MeshAdaptor final : public Interface::MeshAdaptor<dim, Number>
  {
  public:
    using typename Interface::MeshAdaptor<dim, Number>::StateVector;
    using typename Interface::MeshAdaptor<dim, Number>::Transformation;
    using ScalarVector = Vectors::ScalarVector<Number>;

    MeshAdaptor(const MPIEnsemble &mpi_ensemble,
                const Interface::OfflineData<dim, Number> &offline_data,
                Interface::SelectedComponentsExtractor<dim, Number> &extractor,
                const std::string &subsection = "/MeshAdaptor");

    void prepare(const Number t) override;

    void analyze(const StateVector &state_vector,
                 const Number t,
                 unsigned int cycle) override;

    void mark_cells_for_coarsening_and_refinement(
        dealii::Triangulation<dim> &triangulation) const override;

    void
    update_transformation(Transformation & /*transformation*/) const override
    {
    }

  private:
    AdaptationStrategy adaptation_strategy_;
    std::uint_fast64_t random_adaptation_mersenne_twister_seed_;

    MarkingStrategy marking_strategy_;
    double coarsening_threshold_;
    double refinement_threshold_;
    bool absolute_threshold_;
    unsigned int min_refinement_level_;
    unsigned int max_refinement_level_;

    TimePointSelectionStrategy time_point_selection_strategy_;
    std::vector<Number> adaptation_time_points_;
    unsigned int adaptation_cycle_interval_;

    std::vector<std::string> smoothness_selected_quantities_;
    Number smoothness_local_global_ratio_;
    Number smoothness_min_cutoff_;
    Number smoothness_max_cutoff_;
    unsigned int smoothness_widen_stencil_;

    const MPIEnsemble &mpi_ensemble_;

    dealii::ObserverPointer<const Interface::OfflineData<dim, Number>>
        offline_data_;

    Interface::SelectedComponentsExtractor<dim, Number>
        &selected_components_extractor_;

    mutable dealii::Vector<float> indicators_;

    mutable std::mt19937_64 mersenne_twister_;

    mutable ScalarVector smoothness_indicators_;

    void populate_cell_indicators_with_random_values() const;

    void compute_smoothness_indicators(const StateVector &state_vector) const;

    void populate_cell_indicators_from_smoothness_indicators() const;
  };

} // namespace ryujin

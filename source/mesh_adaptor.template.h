//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2024 by the ryujin authors
//

#pragma once

#include "mesh_adaptor.h"
#include <deal.II/base/array_view.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/numerics/error_estimator.h>

namespace ryujin
{
  template <typename Description, int dim, typename Number>
  MeshAdaptor<Description, dim, Number>::MeshAdaptor(
      const MPIEnsemble &mpi_ensemble,
      const OfflineData<dim, Number> &offline_data,
      const HyperbolicSystem &hyperbolic_system,
      const ParabolicSystem &parabolic_system,
      const ScalarVector &alpha,
      const std::string &subsection /*= "MeshAdaptor"*/)
      : ParameterAcceptor(subsection)
      , mpi_ensemble_(mpi_ensemble)
      , offline_data_(&offline_data)
      , hyperbolic_system_(&hyperbolic_system)
      , parabolic_system_(&parabolic_system)
      , need_mesh_adaptation_(false)
      , alpha_(alpha)
  {
    adaptation_strategy_ = AdaptationStrategy::global_refinement;
    add_parameter("adaptation strategy",
                  adaptation_strategy_,
                  "The chosen adaptation strategy. Possible values are: global "
                  "refinement, random adaptation, local refinement");

    marking_strategy_ = MarkingStrategy::fixed_number;
    add_parameter(
        "marking strategy",
        marking_strategy_,
        "The chosen marking strategy. Possible values are: fixed number");

    time_point_selection_strategy_ =
        TimePointSelectionStrategy::fixed_adaptation_time_points;
    add_parameter("time point selection strategy",
                  time_point_selection_strategy_,
                  "The chosen time point selection strategy. Possible values "
                  "are: fixed adaptation time points, simulation cycle based");

    /* Options for various adaptation strategies: */
    enter_subsection("adaptation strategies");
    random_adaptation_mersenne_twister_seed_ = 42u;
    add_parameter("random adaptation: mersenne_twister_seed",
                  random_adaptation_mersenne_twister_seed_,
                  "Seed for 64bit Mersenne Twister used for random refinement");

    std::copy(std::begin(View::component_names),
              std::end(View::component_names),
              std::back_inserter(kelly_options_));

    std::copy(std::begin(View::initial_precomputed_names),
              std::end(View::initial_precomputed_names),
              std::back_inserter(kelly_options_));

    kelly_options_ = {};
    add_parameter(
        "Kelly indicators",
        kelly_options_,
        "List of conserved, primitive or precomputed  "
        "quantities that will be used for the KellyErrorEstimator indicator "
        "for refinement and coarsening.");
    leave_subsection();

    /* Options for various marking strategies: */

    enter_subsection("marking strategies");
    fixed_number_refinement_fraction_ = 0.3;
    add_parameter(
        "fixed number: refinement fraction",
        fixed_number_refinement_fraction_,
        "Fixed number strategy: fraction of cells selected for refinement.");

    fixed_number_coarsening_fraction_ = 0.3;
    add_parameter(
        "fixed number: coarsening fraction",
        fixed_number_coarsening_fraction_,
        "Fixed number strategy: fraction of cells selected for coarsening.");
    leave_subsection();

    /* Options for various time point selection strategies: */

    enter_subsection("time point selection strategies");
    adaptation_time_points_ = {};
    add_parameter("adaptation timepoints",
                  adaptation_time_points_,
                  "List of time points in (simulation) time at which we will "
                  "perform a mesh adaptation cycle.");

    adaptation_simulation_cycle_ = 5;
    add_parameter("adaptation simulation cycle",
                  adaptation_simulation_cycle_,
                  "The nth simulation cycle at which we will "
                  "perform mesh adapation.");
    leave_subsection();

    const auto call_back = [this] {
      /* Initialize Mersenne Twister with configured seed: */
      mersenne_twister_.seed(random_adaptation_mersenne_twister_seed_);
    };

    call_back();
    ParameterAcceptor::parse_parameters_call_back.connect(call_back);
  }


  template <typename Description, int dim, typename Number>
  void MeshAdaptor<Description, dim, Number>::prepare(const Number t)
  {
#ifdef DEBUG_OUTPUT
    std::cout << "MeshAdaptor<dim, Number>::prepare()" << std::endl;
#endif

    if (time_point_selection_strategy_ ==
        TimePointSelectionStrategy::fixed_adaptation_time_points) {
      /* Remove outdated refinement timestamps: */
      const auto new_end = std::remove_if(
          adaptation_time_points_.begin(),
          adaptation_time_points_.end(),
          [&](const Number &t_refinement) { return (t > t_refinement); });
      adaptation_time_points_.erase(new_end, adaptation_time_points_.end());
    }

    if (adaptation_strategy_ == AdaptationStrategy::kelly_estimator) {
      /* Populate quantities mapping based on user-defined Kelly options: */

      quantities_mapping_.clear();

      for (const auto &entry : kelly_options_) {
        /* Conserved quantities: */
        {
          constexpr auto &names = View::component_names;
          const auto pos = std::find(std::begin(names), std::end(names), entry);
          if (pos != std::end(names)) {
            const auto index = std::distance(std::begin(names), pos);
            quantities_mapping_.push_back(std::make_tuple(
                entry,
                [index](ScalarVector &result, const StateVector &state_vector) {
                  const auto &U = std::get<0>(state_vector);
                  U.extract_component(result, index);
                }));
            continue;
          }
        }

        /* Primitive quantities: */
        {
          constexpr auto &names = View::primitive_component_names;
          const auto pos = std::find(std::begin(names), std::end(names), entry);
          if (pos != std::end(names)) {
            const auto index = std::distance(std::begin(names), pos);
            quantities_mapping_.push_back(std::make_tuple(
                entry,
                [this, index](ScalarVector &result,
                              const StateVector &state_vector) {
                  const auto &U = std::get<0>(state_vector);
                  /*
                   * FIXME: We might traverse the same vector multiple
                   * times. This is inefficient.
                   */
                  const unsigned int n_owned = offline_data_->n_locally_owned();
                  for (unsigned int i = 0; i < n_owned; ++i) {
                    const auto view =
                        hyperbolic_system_->template view<dim, Number>();
                    result.local_element(i) =
                        view.to_primitive_state(U.get_tensor(i))[index];
                  }
                }));
            continue;
          }
        }

        /* Precomputed quantities: */
        {
          constexpr auto &names = View::precomputed_names;
          const auto pos = std::find(std::begin(names), std::end(names), entry);
          if (pos != std::end(names)) {
            const auto index = std::distance(std::begin(names), pos);
            quantities_mapping_.push_back(std::make_tuple(
                entry,
                [index](ScalarVector &result, const StateVector &state_vector) {
                  const auto &precomputed = std::get<1>(state_vector);
                  precomputed.extract_component(result, index);
                }));
            continue;
          }
        }

        /* Special indicator value: */
        if (entry == "alpha") {
          quantities_mapping_.push_back(std::make_tuple(
              entry, [this](ScalarVector &result, const StateVector &) {
                result = alpha_;
              }));
          continue;
        }

        AssertThrow(
            false,
            dealii::ExcMessage("Invalid component name »" + entry + "«"));
      }

      kelly_quantities_.resize(quantities_mapping_.size());
      for (auto &it : kelly_quantities_)
        it.reinit(offline_data_->scalar_partitioner());
    }

    /* toggle mesh adaptation flag to off. */
    need_mesh_adaptation_ = false;
  }


  template <typename Description, int dim, typename Number>
  void MeshAdaptor<Description, dim, Number>::analyze(
      const StateVector &state_vector, const Number t, unsigned int cycle)
  {
#ifdef DEBUG_OUTPUT
    std::cout << "MeshAdaptor<dim, Number>::analyze()" << std::endl;
#endif

    switch (time_point_selection_strategy_) {
    case TimePointSelectionStrategy::fixed_adaptation_time_points: {
      /* Remove all refinement points from the vector that lie in the past: */
      const auto new_end = std::remove_if( //
          adaptation_time_points_.begin(),
          adaptation_time_points_.end(),
          [&](const Number &t_refinement) {
            if (t < t_refinement)
              return false;
            need_mesh_adaptation_ = true;
            return true;
          });
      adaptation_time_points_.erase(new_end, adaptation_time_points_.end());
    } break;
    case TimePointSelectionStrategy::simulation_cycle_based: {
      /* Check if simulation time cycle modulo adaptation_time_cycle_  = 0 */
      if (cycle % adaptation_simulation_cycle_ == 0)
        need_mesh_adaptation_ = true;
    } break;

    default:
      AssertThrow(false, dealii::ExcInternalError());
      __builtin_trap();
    }

    switch (adaptation_strategy_) {
    case AdaptationStrategy::global_refinement:
      // do nothing
      break;
    case AdaptationStrategy::random_adaptation:
      // do nothing
      break;
    case AdaptationStrategy::kelly_estimator: {
      /* Populate Kelly quantities: */
      {
        const auto &affine_constraints = offline_data_->affine_constraints();

        Assert(kelly_quantities_.size() == quantities_mapping_.size(),
               dealii::ExcInternalError());
        for (unsigned int d = 0; d < kelly_quantities_.size(); ++d) {
          const auto &lambda = std::get<1>(quantities_mapping_[d]);
          lambda(kelly_quantities_[d], state_vector);
          affine_constraints.distribute(kelly_quantities_[d]);
          kelly_quantities_[d].update_ghost_values();
        }
      }
    } break;

    default:
      AssertThrow(false, dealii::ExcInternalError());
      __builtin_trap();
    }
  }


  template <typename Description, int dim, typename Number>
  void MeshAdaptor<Description, dim, Number>::
      mark_cells_for_coarsening_and_refinement(
          dealii::Triangulation<dim> &triangulation) const
  {
    auto &discretization [[maybe_unused]] = offline_data_->discretization();
    Assert(&triangulation == &discretization.triangulation(),
           dealii::ExcInternalError());

    /*
     * Compute an indicator with the chosen adaptation strategy:
     */

    dealii::Vector<float> indicators;

    switch (adaptation_strategy_) {
    case AdaptationStrategy::global_refinement: {
      /* Simply mark all cells for refinement and return: */
      for (auto &cell : triangulation.active_cell_iterators())
        cell->set_refine_flag();
      return;
    } break;

    case AdaptationStrategy::random_adaptation: {
      indicators.reinit(triangulation.n_active_cells());
      std::generate(std::begin(indicators), std::end(indicators), [&]() {
        static std::uniform_real_distribution<double> distribution(0.0, 10.0);
        return distribution(mersenne_twister_);
      });
    } break;

    case AdaptationStrategy::kelly_estimator: {

      /* Calcuator the KellyErrorEstimator for each defined quantitity */
      std::vector<dealii::Vector<float>> kelly_errors;
      std::vector<dealii::Vector<float> *> ptr_kelly_errors;

      kelly_errors.resize(quantities_mapping_.size());
      for (auto &it : kelly_errors) {
        it.reinit(triangulation.n_active_cells());
        ptr_kelly_errors.push_back(&it);
      }

      auto array_view_kelly_errors = dealii::make_array_view(ptr_kelly_errors);
      std::vector<const dealii::ReadVector<Number> *> ptr_kelly_quantities;
      for (const auto &it : kelly_quantities_) {
        ptr_kelly_quantities.push_back(&it);
      }
      const auto array_view_kelly_quantities =
          dealii::make_array_view(ptr_kelly_quantities);

      /* Populate array_view_kelly_errors */
      dealii::KellyErrorEstimator<dim>::estimate(
          offline_data_->discretization().mapping(),
          offline_data_->dof_handler(),
          offline_data_->discretization().face_quadrature(),
          {},
          array_view_kelly_quantities,
          array_view_kelly_errors);

      /* Accumulate total error per cell */
      indicators.reinit(triangulation.n_active_cells());
      for (const auto &it : kelly_errors)
        indicators += it;

    } break;

    default:
      AssertThrow(false, dealii::ExcInternalError());
      __builtin_trap();
    }

    /*
     * Mark cells with chosen marking strategy:
     */

    switch (marking_strategy_) {
    case MarkingStrategy::fixed_number: {
      dealii::GridRefinement::refine_and_coarsen_fixed_number(
          triangulation,
          indicators,
          fixed_number_refinement_fraction_,
          fixed_number_coarsening_fraction_);
    } break;

    default:
      AssertThrow(false, dealii::ExcInternalError());
      __builtin_trap();
    }

    /*
     * Constrain refinement and coarsening to maximum and minimum levels:
     */
#if 0
    unsigned int refinement_level = discretization.refinement(); // How to fix this line?
    const unsigned int max_refinement_level = refinement_level + 2;
    const unsigned int min_refinement_level = refinement_level - 1;
    if (triangulation.n_levels() > max_refinement_level)
      for (const auto &cell :
           triangulation.active_cell_iterators_on_level(max_refinement_level))
        cell->clear_refine_flag();
    for (const auto &cell :
         triangulation.active_cell_iterators_on_level(min_refinement_level))
      cell->clear_coarsen_flag();
#endif

    return;
  }
} // namespace ryujin

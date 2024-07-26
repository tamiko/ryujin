//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception or LGPL-2.1-or-later
// Copyright (C) 2007 - 2024 by the deal.II authors
// Copyright (C) 2024 - 2024 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include "discretization.h"
#include "solution_transfer.h"

#include <deal.II/base/config.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/grid/cell_status.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/petsc_block_vector.h>
#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/trilinos_parallel_block_vector.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/vector.h>
#include <deal.II/matrix_free/fe_point_evaluation.h>


namespace ryujin
{
  template <typename Description, int dim, typename Number>
  SolutionTransfer<Description, dim, Number>::SolutionTransfer(
      const MPI_Comm &mpi_communicator,
      Discretization<dim>::Triangulation &triangulation,
      const OfflineData<dim, Number> &offline_data,
      const HyperbolicSystem &hyperbolic_system,
      const ParabolicSystem &parabolic_system)
      : mpi_communicator_(mpi_communicator)
      , triangulation_(&triangulation)
      , offline_data_(&offline_data)
      , hyperbolic_system_(&hyperbolic_system)
      , parabolic_system_(&parabolic_system)
      , old_state_vector_(nullptr)
      , handle(dealii::numbers::invalid_unsigned_int)

  {
    AssertThrow(have_distributed_triangulation<dim>,
                dealii::ExcMessage(
                    "The SolutionTransfer class is not implemented for a "
                    "distributed::shared::Triangulation which we use in 1D"));
  }


  template <typename Description, int dim, typename Number>
  void SolutionTransfer<Description, dim, Number>::
      prepare_projection_and_serialization(const StateVector &old_state_vector)
  {
#ifdef DEBUG_OUTPUT
    std::cout << "SolutionTransfer<Description, dim, "
                 "Number>::prepare_projection_and_serialization()"
              << std::endl;
#endif

    AssertThrow(have_distributed_triangulation<dim>,
                dealii::ExcMessage(
                    "The SolutionTransfer class is not implemented for a "
                    "distributed::shared::Triangulation which we use in 1D"));

    old_state_vector_ = &old_state_vector;
    register_data_attach();
  }


  template <typename Description, int dim, typename Number>
  void SolutionTransfer<Description, dim, Number>::project(
      StateVector &new_state_vector [[maybe_unused]])
  {
#ifdef DEBUG_OUTPUT
    std::cout << "SolutionTransfer<Description, dim, Number>::deserialize()"
              << std::endl;
#endif

    AssertThrow(have_distributed_triangulation<dim>,
                dealii::ExcMessage(
                    "The SolutionTransfer class is not implemented for a "
                    "distributed::shared::Triangulation which we use in 1D"));

    Assert(false, dealii::ExcNotImplemented());
  }


  template <typename Description, int dim, typename Number>
  void SolutionTransfer<Description, dim, Number>::deserialize(
      StateVector &new_state_vector)
  {
#ifdef DEBUG_OUTPUT
    std::cout << "SolutionTransfer<Description, dim, Number>::deserialize()"
              << std::endl;
#endif

    AssertThrow(have_distributed_triangulation<dim>,
                dealii::ExcMessage(
                    "The SolutionTransfer class is not implemented for a "
                    "distributed::shared::Triangulation which we use in 1D"));

    register_data_attach();
    project(new_state_vector);

    Assert(false, dealii::ExcNotImplemented());
  }


  namespace
  {
    /**
     * Pack a vector of local state values into a char vector.
     */
    template <typename state_type>
    std::vector<char>
    pack_state_values(const std::vector<state_type> &state_values)
    {
      std::vector<char> buffer(sizeof(state_type) * state_values.size());
      std::memcpy(buffer.data(), state_values.data(), buffer.size());
      return buffer;
    }


    /**
     * Unpack a char vector into a vector of local state values.
     */
    template <typename state_type>
    std::vector<state_type> unpack_state_values(
        const boost::iterator_range<std::vector<char>::const_iterator>
            &data_range)
    {
      const std::size_t n_bytes = data_range.size();
      Assert(n_bytes % sizeof(state_type) == 0, dealii::ExcInternalError());
      std::vector<state_type> state_values(n_bytes / sizeof(state_type));
      std::memcpy(state_values.data(),
                  &data_range[0],
                  state_values.size() * sizeof(state_type));
      return state_values;
    }
  } // namespace


  template <typename Description, int dim, typename Number>
  void SolutionTransfer<Description, dim, Number>::register_data_attach()
  {
    const auto &discretization = offline_data_->discretization();
    const auto &triangulation = discretization.triangulation();
    Assert(triangulation_ == &triangulation,
           dealii::ExcMessage(
               "The attached triangulation object must be the same object that "
               "is stored in Discretization/OfflineData"));

    Assert(handle == dealii::numbers::invalid_unsigned_int,
           dealii::ExcMessage(
               "You can only add one solution per SolutionTransfer object."));

    handle = triangulation_->register_data_attach(
        [this](const auto cell, const dealii::CellStatus status) {

          const auto &dof_handler = offline_data_->dof_handler();
          const auto dof_cell = typename dealii::DoFHandler<dim>::cell_iterator(
              &cell->get_triangulation(),
              cell->level(),
              cell->index(),
              &dof_handler);

          Assert(old_state_vector_ != nullptr, dealii::ExcInternalError());
          const auto &U = std::get<0>(*old_state_vector_);

          /*
           * Collect values for packing:
           */

          const auto n_dofs_per_cell = dof_handler.get_fe().n_dofs_per_cell();
          std::vector<state_type> state_values(n_dofs_per_cell);

          switch(status) {
          case dealii::CellStatus::cell_will_persist:
            [[fallthrough]];
          case dealii::CellStatus::cell_will_be_refined: {
            /* We need state values from the currently active cell: */

            std::vector<dealii::types::global_dof_index> dof_indices(
                n_dofs_per_cell);
            dof_cell->get_dof_indices(dof_indices);

            std::transform(std::begin(dof_indices),
                           std::end(dof_indices),
                           std::begin(state_values),
                           [&](const auto i) {
                             const auto U_i = U.get_tensor(i);
                             return U_i;
                           });
          } break;

          case dealii::CellStatus::children_will_be_coarsened: {
            Assert(dof_cell->has_children(), dealii::ExcInternalError());

            const auto &discretization = offline_data_->discretization();
            const auto &finite_element = discretization.finite_element();
            const auto &mapping = discretization.mapping();
            const auto &quadrature = discretization.quadrature();

            dealii::FEValues<dim> fe_values(
                mapping,
                finite_element,
                quadrature,
                dealii::update_values | dealii::update_JxW_values |
                    dealii::update_quadrature_points);

            dealii::FEPointEvaluation<1, dim> evaluator(
                mapping,
                finite_element,
                dealii::update_values | dealii::update_JxW_values);
            std::vector<dealii::Point<dim>> unit_points(quadrature.size());

            /*
             * We need to project values from the active child cells up to
             * the present parent cell that will become active after
             * coarsening.
             */

            /* Step 1: build up right hand side by iterating over children: */

            std::vector<state_type> state_values(quadrature.size());
            dealii::Vector<double> local_values(n_dofs_per_cell);
            dealii::Vector<double> local_rhs(n_dofs_per_cell);

            for (unsigned int child = 0; child < dof_cell->n_children(); ++child) {

              fe_values.reinit(dof_cell->child(child));

              mapping.transform_points_real_to_unit_cell(
                  dof_cell, fe_values.get_quadrature_points(), unit_points);

              for (unsigned int q = 0; q < quadrature.size(); ++q) {
                for (unsigned int i = 0; i < n_dofs_per_cell; ++i) {
                  const auto U_i = U.get_tensor(i);
                  state_values[q] += U_i * fe_values.shape_value(i, q);
                }
              }

//               evaluator.reinit(cell, unit_points);
//               for (unsigned int q = 0; q < quadrature.size(); ++q) {
//                 evaluator.submit_value(state_values[q] * fe_values.JxW(q), q);
//               }
//               evaluator.test_and_sum(local_values,
//                                      dealii::EvaluationFlags::values);
//               for (unsigned int i = 0; i < n_dofs_per_cell; ++i)
//                 local_rhs(i) += local_values(i);
            }

            /* Step 2: solve with mass matrix on coarse cell: */

            fe_values.reinit(dof_cell);

            dealii::FullMatrix<double> mij(n_dofs_per_cell, n_dofs_per_cell);
            dealii::Vector<double> mi(n_dofs_per_cell);
            for (unsigned int i = 0; i < n_dofs_per_cell; ++i) {
              for (unsigned int j = 0; j < n_dofs_per_cell; ++j) {
                double sum = 0;
                for (unsigned int q = 0; q < quadrature.size(); ++q)
                  sum += fe_values.shape_value(i, q) *
                         fe_values.shape_value(j, q) * fe_values.JxW(q);
                mij(i, j) = sum;
                mi(i) += sum;
              }
            }
#if 0

            mij.gauss_jordan();
            mij.vmult(local_values, local_rhs);
            // for (unsigned int i = 0; i < fe.dofs_per_cell; ++i)
            //   local_values(i) = local_rhs(i) / mi(i);
#endif
          } break;

            case dealii::CellStatus::cell_invalid:
              Assert(false, dealii::ExcInternalError());
              __builtin_trap();
              break;
            }

          return pack_state_values(state_values);
        },
        /* returns_variable_size_data =*/false);
  }
} // namespace ryujin


#if 0
    // create buffer for each individual object
    std::vector<::dealii::Vector<typename VectorType::value_type>> dof_values(
        input_vectors.size());

    const unsigned int dofs_per_cell =
        dof_handler->get_fe(fe_index).n_dofs_per_cell();

    if (dofs_per_cell == 0)
      return {}; // nothing to do for FE_Nothing

    auto it_input = input_vectors.cbegin();
    auto it_output = dof_values.begin();
    for (; it_input != input_vectors.cend(); ++it_input, ++it_output) {
      it_output->reinit(dofs_per_cell);
      cell->get_interpolated_dof_values(*(*it_input), *it_output, fe_index);
    }

    return pack_dof_values<typename VectorType::value_type>(dof_values,
                                                            dofs_per_cell);
  }
#endif

//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/compile_time_options.h>

#include <ryujin/base/convenience_macros.h>
#include <ryujin/base/gpu.h>
#include <ryujin/base/observer_pointer.h>
#include <ryujin/interface/discretization.h>
#include <ryujin/linear_algebra/sparse_matrix.h>
#include <ryujin/linear_algebra/sparsity_pattern.h>
#include <ryujin/linear_algebra/state_vector.h>

#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/partitioner.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>

#include <memory>
#include <tuple>
#include <vector>

namespace ryujin
{
  namespace Interface
  {
    template <int dim, typename Number = double>
    class OfflineData : public dealii::ParameterAcceptor
    {
    public:
      using BoundaryDescription = std::tuple<unsigned int,
                                             dealii::Tensor<1, dim, Number>,
                                             Number,
                                             Number,
                                             dealii::types::boundary_id,
                                             dealii::Point<dim>>;

      struct CouplingDescription {
        unsigned int i;
        unsigned int col_idx;
        unsigned int j;

        bool operator==(const CouplingDescription &) const = default;
      };

      virtual void prepare(const unsigned int problem_dimension,
                           const unsigned int n_precomputed_values) = 0;

      ACCESSOR_READ_ONLY(dof_handler_cg)

      ACCESSOR_READ_ONLY(dof_handler_dg)

      const dealii::DoFHandler<dim> &dof_handler() const
      {
        if (discretization_->have_discontinuous_ansatz()) {
          Assert(dof_handler_dg_, dealii::ExcInternalError());
          return *dof_handler_dg_;
        } else {
          Assert(dof_handler_cg_, dealii::ExcInternalError());
          return *dof_handler_cg_;
        }
      }

      dealii::DoFHandler<dim> &dof_handler()
      {
        if (discretization_->have_discontinuous_ansatz()) {
          Assert(dof_handler_dg_, dealii::ExcInternalError());
          return *dof_handler_dg_;
        } else {
          Assert(dof_handler_cg_, dealii::ExcInternalError());
          return *dof_handler_cg_;
        }
      }

      ACCESSOR_READ_ONLY(affine_constraints_cg)

      ACCESSOR_READ_ONLY(affine_constraints_dg)

      const dealii::AffineConstraints<Number> &affine_constraints() const
      {
        if (discretization_->have_discontinuous_ansatz()) {
          return *affine_constraints_dg_;
        } else {
          return *affine_constraints_cg_;
        }
      }

      ACCESSOR_READ_ONLY_NO_DEREFERENCE(scalar_partitioner)

      ACCESSOR_READ_ONLY_NO_DEREFERENCE(hyperbolic_vector_partitioner)

      ACCESSOR_READ_ONLY_NO_DEREFERENCE(precomputed_vector_partitioner)

      ACCESSOR_READ_ONLY(n_export_indices)

      ACCESSOR_READ_ONLY(n_locally_internal)

      ACCESSOR_READ_ONLY(n_locally_owned)

      ACCESSOR_READ_ONLY(n_locally_relevant)

      ACCESSOR_READ_ONLY(boundary_map)

      ACCESSOR_READ_ONLY(boundary_indices)

      ACCESSOR_READ_ONLY(boundary_slots)

      ACCESSOR_READ_ONLY(coupling_boundary_pairs)

      ACCESSOR_READ_ONLY(level_boundary_map)

      ACCESSOR_READ_ONLY(sparsity_pattern)

      ACCESSOR_READ_ONLY(sparsity_pattern_simd)

      ACCESSOR_READ_ONLY(mass_matrix)

      ACCESSOR_READ_ONLY(mass_matrix_inverse)

      ACCESSOR_READ_ONLY(lumped_mass_matrix)

      ACCESSOR_READ_ONLY(lumped_mass_matrix_inverse)

      ACCESSOR_READ_ONLY(level_lumped_mass_matrix)

      ACCESSOR_READ_ONLY(betaij_matrix)

      ACCESSOR_READ_ONLY(cij_matrix)

      ACCESSOR_READ_ONLY(incidence_matrix)

      ACCESSOR_READ_ONLY(measure_of_omega)

      ACCESSOR_READ_ONLY(discretization)

    protected:
      using ParameterAcceptor::ParameterAcceptor;

      dealii::ObserverPointer<const Discretization<dim>> discretization_;

      std::unique_ptr<dealii::DoFHandler<dim>> dof_handler_cg_;
      std::unique_ptr<dealii::DoFHandler<dim>> dof_handler_dg_;

      std::unique_ptr<dealii::AffineConstraints<Number>> affine_constraints_cg_;
      std::unique_ptr<dealii::AffineConstraints<Number>> affine_constraints_dg_;

      std::shared_ptr<const dealii::Utilities::MPI::Partitioner>
          scalar_partitioner_;

      std::shared_ptr<const dealii::Utilities::MPI::Partitioner>
          hyperbolic_vector_partitioner_;

      std::shared_ptr<const dealii::Utilities::MPI::Partitioner>
          precomputed_vector_partitioner_;

      unsigned int n_export_indices_;
      unsigned int n_locally_internal_;
      unsigned int n_locally_owned_;
      unsigned int n_locally_relevant_;

      using BoundaryMap = std::vector<BoundaryDescription>;
      BoundaryMap boundary_map_;
      std::vector<BoundaryMap> level_boundary_map_;

      std::unique_ptr<Mirrored<unsigned int *>> boundary_indices_;
      std::vector<unsigned int> boundary_slots_;

      using CouplingBoundaryPairs = std::vector<CouplingDescription>;
      std::unique_ptr<Mirrored<CouplingDescription *>> coupling_boundary_pairs_;

      std::unique_ptr<dealii::DynamicSparsityPattern> sparsity_pattern_;
      std::unique_ptr<SparsityPattern<warp_size>> sparsity_pattern_simd_;

      std::unique_ptr<SparseMatrix<Number>> mass_matrix_;
      std::unique_ptr<SparseMatrix<Number>> mass_matrix_inverse_;

      using ScalarVector = Vectors::ScalarVector<Number>;
      std::unique_ptr<ScalarVector> lumped_mass_matrix_;
      std::unique_ptr<ScalarVector> lumped_mass_matrix_inverse_;

      using ScalarHostVectorFloat = Vectors::ScalarHostVector<float>;
      std::vector<ScalarHostVectorFloat> level_lumped_mass_matrix_;

      std::unique_ptr<SparseMatrix<Number>> betaij_matrix_;
      std::unique_ptr<SparseMatrix<Number, dim>> cij_matrix_;
      std::unique_ptr<SparseMatrix<Number>> incidence_matrix_;

      Number measure_of_omega_;
    };
  } // namespace Interface
} // namespace ryujin

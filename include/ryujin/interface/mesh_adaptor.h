//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2024 - 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/convenience_macros.h>
#include <ryujin/linear_algebra/state_vector.h>

#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/vector.h>

#include <memory>

namespace ryujin
{
  namespace Interface
  {
    template <int dim, typename Number = double>
    class MeshAdaptor : public dealii::ParameterAcceptor
    {
    public:
      using StateVector = Vectors::StateVector<Number>;
      using ScalarVector = Vectors::ScalarVector<Number>;

      virtual void prepare(const Number t) = 0;

      virtual void analyze(const StateVector &state_vector,
                           const Number t,
                           unsigned int cycle) = 0;

      virtual void mark_cells_for_coarsening_and_refinement(
          dealii::Triangulation<dim> &triangulation) const = 0;

      virtual void
      compute_smoothness_indicators(const StateVector &state_vector) const = 0;

      ACCESSOR_READ_ONLY(need_mesh_adaptation)

      ACCESSOR_READ_ONLY(indicators);

      ACCESSOR_READ_ONLY(smoothness_indicators);

    protected:
      using ParameterAcceptor::ParameterAcceptor;

      bool need_mesh_adaptation_;

      std::unique_ptr<dealii::Vector<float>> indicators_;

      std::unique_ptr<ScalarVector> smoothness_indicators_;
    };
  } // namespace Interface
} // namespace ryujin

//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/convenience_macros.h>
#include <ryujin/interface/hyperbolic_module.h>
#include <ryujin/interface/initial_values.h>
#include <ryujin/interface/parabolic_module.h>
#include <ryujin/interface/parabolic_system.h>
#include <ryujin/interface/solution_transfer.h>

#include <memory>

namespace ryujin
{
  class MPIEnsemble;

  template <int dim>
  class Discretization;

  template <int dim, typename Number>
  class OfflineData;

  template <int dim, typename Number>
  class TimeIntegrator;

  template <int dim, typename Number>
  class MeshAdaptor;

  template <int dim, typename Number>
  class Postprocessor;

  template <int dim, typename Number>
  class VTUOutput;

  template <int dim, typename Number>
  class Quantities;

  template <int dim, typename Number>
  class ErrorEvaluation;


  namespace Interface
  {
    template <int dim, typename Number = double>
    class Simulation
    {
    public:
      virtual ~Simulation() = default;

      ACCESSOR_READ_ONLY(mpi_ensemble)

      ACCESSOR(discretization)

      ACCESSOR(offline_data)

      ACCESSOR_READ_ONLY(parabolic_system)

      ACCESSOR_READ_ONLY(initial_values)

      ACCESSOR(hyperbolic_module)

      ACCESSOR(parabolic_module)

      ACCESSOR(time_integrator)

      ACCESSOR(mesh_adaptor)

      ACCESSOR(solution_transfer)

      ACCESSOR(postprocessor)

      ACCESSOR(vtu_output)

      ACCESSOR(quantities)

      ACCESSOR(error_evaluation)

    protected:
      Simulation() = default;

      std::unique_ptr<MPIEnsemble> mpi_ensemble_;
      std::unique_ptr<Discretization<dim>> discretization_;
      std::unique_ptr<OfflineData<dim, Number>> offline_data_;

      const ParabolicSystem *parabolic_system_ = nullptr;
      const InitialValues<dim, Number> *initial_values_ = nullptr;

      std::unique_ptr<HyperbolicModule<dim, Number>> hyperbolic_module_;
      std::unique_ptr<ParabolicModule<dim, Number>> parabolic_module_;
      std::unique_ptr<TimeIntegrator<dim, Number>> time_integrator_;
      std::unique_ptr<MeshAdaptor<dim, Number>> mesh_adaptor_;
      std::unique_ptr<SolutionTransfer<dim, Number>> solution_transfer_;
      std::unique_ptr<Postprocessor<dim, Number>> postprocessor_;
      std::unique_ptr<VTUOutput<dim, Number>> vtu_output_;
      std::unique_ptr<Quantities<dim, Number>> quantities_;
      std::unique_ptr<ErrorEvaluation<dim, Number>> error_evaluation_;
    };
  } // namespace Interface
} // namespace ryujin

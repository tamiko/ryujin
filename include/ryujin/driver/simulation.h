//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/compile_time_options.h>
#include <ryujin/base/mpi_ensemble.h>
#include <ryujin/base/mpi_ensemble_container.h>
#include <ryujin/concept/hyperbolic_description.h>
#include <ryujin/concept/parabolic_description.h>
#include <ryujin/dependent/hyperbolic_module.h>
#include <ryujin/dependent/initial_values.h>
#include <ryujin/dependent/selected_components_extractor.h>
#include <ryujin/dependent/solution_transfer.h>
#include <ryujin/discretization/discretization.h>
#include <ryujin/discretization/mesh_adaptor.h>
#include <ryujin/discretization/offline_data.h>
#include <ryujin/driver/time_integrator.h>
#include <ryujin/interface/simulation.h>
#include <ryujin/output/error_evaluation.h>
#include <ryujin/output/postprocessor.h>
#include <ryujin/output/quantities.h>
#include <ryujin/output/vtu_output.h>

#include <deal.II/base/mpi.h>

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace ryujin
{
  template <typename HyperbolicDescription,
            typename ParabolicDescription,
            int dim,
            typename Number = double>
    requires Concept::
                 HyperbolicDescription<HyperbolicDescription, dim, Number> &&
             Concept::ParabolicDescription<ParabolicDescription,
                                           HyperbolicDescription,
                                           dim,
                                           Number>
  class Simulation final : public Interface::Simulation<dim, Number>
  {
  public:
    using HD = HyperbolicDescription;
    using PD = ParabolicDescription;

    using HyperbolicSystem = typename HD::HyperbolicSystem;
    using ParabolicSystem = typename PD::ParabolicSystem;
    using ParabolicModule =
        typename PD::template ParabolicModule<HD, dim, Number>;
    using Extractor = SelectedComponentsExtractor<HD, dim, Number>;

    Simulation(const MPI_Comm &mpi_comm)
    {
      using AdditionalNames = std::vector<std::string>;
      using AdditionalVectors = std::vector<
          std::reference_wrapper<const Vectors::ScalarVector<Number>>>;

      int n_ensembles = 1;
      if constexpr (has_n_mpi_ensembles_v<HD>)
        n_ensembles = HD::n_mpi_ensembles();

      this->mpi_ensemble_ =
          std::make_unique<MPIEnsemble>(mpi_comm, n_ensembles);
      const auto &mpi_ensemble = *this->mpi_ensemble_;

      hyperbolic_system_container_ =
          std::make_unique<MPIEnsembleContainer<HyperbolicSystem>>(
              mpi_ensemble, "/B - Equation");
      const HyperbolicSystem &hyperbolic_system =
          hyperbolic_system_container_->get();

      parabolic_system_container_ =
          std::make_unique<MPIEnsembleContainer<ParabolicSystem>>(
              mpi_ensemble, "/B - Equation");
      const ParabolicSystem &parabolic_system =
          parabolic_system_container_->get();
      this->parabolic_system_ = &parabolic_system;

      this->discretization_ = std::make_unique<Discretization<dim>>(
          mpi_ensemble, "/C - Discretization");

      this->offline_data_ = std::make_unique<OfflineData<dim, Number>>(
          mpi_ensemble, *this->discretization_, "/D - OfflineData");
      const auto &offline_data = *this->offline_data_;

      initial_values_container_ = std::make_unique<
          MPIEnsembleContainer<InitialValues<HD, dim, Number>>>(
          mpi_ensemble,
          "/E - InitialValues",
          mpi_ensemble,
          offline_data,
          hyperbolic_system);
      const InitialValues<HD, dim, Number> &initial_values =
          initial_values_container_->get();
      this->initial_values_ = &initial_values;

      auto hyperbolic_module =
          std::make_unique<HyperbolicModule<HD, dim, Number>>(
              mpi_ensemble,
              offline_data,
              hyperbolic_system,
              initial_values,
              "/F - HyperbolicModule");
      const auto &initial_precomputed =
          hyperbolic_module->initial_precomputed();
      const auto &alpha = hyperbolic_module->alpha();
      this->hyperbolic_module_ = std::move(hyperbolic_module);

      this->parabolic_module_ =
          std::make_unique<ParabolicModule>(mpi_ensemble,
                                            offline_data,
                                            hyperbolic_system,
                                            parabolic_system,
                                            initial_values,
                                            "/G - ParabolicModule");

      this->time_integrator_ = std::make_unique<TimeIntegrator<dim, Number>>(
          mpi_ensemble,
          offline_data,
          *this->hyperbolic_module_,
          *this->parabolic_module_,
          parabolic_system,
          "/H - TimeIntegrator");

      mesh_adaptor_extractor_ =
          std::make_unique<Extractor>(offline_data,
                                      hyperbolic_system,
                                      parabolic_system,
                                      initial_precomputed,
                                      AdditionalNames{"alpha"},
                                      AdditionalVectors{alpha});

      this->mesh_adaptor_ =
          std::make_unique<MeshAdaptor<dim, Number>>(mpi_ensemble,
                                                     offline_data,
                                                     *mesh_adaptor_extractor_,
                                                     "/I - MeshAdaptor");

      this->solution_transfer_ =
          std::make_unique<SolutionTransfer<HD, dim, Number>>(
              mpi_ensemble,
              offline_data,
              hyperbolic_system,
              "/I - MeshAdaptor");

      postprocessor_extractor_ =
          std::make_unique<Extractor>(offline_data,
                                      hyperbolic_system,
                                      parabolic_system,
                                      initial_precomputed);

      this->postprocessor_ = std::make_unique<Postprocessor<dim, Number>>(
          mpi_ensemble,
          offline_data,
          *postprocessor_extractor_,
          "/J - VTUOutput");

      vtu_output_extractor_ = std::make_unique<Extractor>(
          offline_data,
          hyperbolic_system,
          parabolic_system,
          initial_precomputed,
          AdditionalNames{"alpha", "smoothness_indicators"},
          AdditionalVectors{alpha,
                            this->mesh_adaptor_->smoothness_indicators()});

      this->vtu_output_ =
          std::make_unique<VTUOutput<dim, Number>>(mpi_ensemble,
                                                   offline_data,
                                                   *vtu_output_extractor_,
                                                   *this->postprocessor_,
                                                   "/J - VTUOutput");

      quantities_extractor_ = std::make_unique<Extractor>(offline_data,
                                                          hyperbolic_system,
                                                          parabolic_system,
                                                          initial_precomputed);

      this->quantities_ =
          std::make_unique<Quantities<dim, Number>>(mpi_ensemble,
                                                    offline_data,
                                                    *quantities_extractor_,
                                                    "/K - Quantities");

      error_evaluation_extractor_ =
          std::make_unique<Extractor>(offline_data,
                                      hyperbolic_system,
                                      parabolic_system,
                                      initial_precomputed);

      this->error_evaluation_ = std::make_unique<ErrorEvaluation<dim, Number>>(
          mpi_ensemble,
          offline_data,
          *error_evaluation_extractor_,
          "/L - ErrorEvaluation");
    }

    ~Simulation() override
    {
      this->error_evaluation_.reset();
      error_evaluation_extractor_.reset();
      this->quantities_.reset();
      quantities_extractor_.reset();
      this->vtu_output_.reset();
      vtu_output_extractor_.reset();
      this->postprocessor_.reset();
      postprocessor_extractor_.reset();
      this->solution_transfer_.reset();
      this->mesh_adaptor_.reset();
      mesh_adaptor_extractor_.reset();
      this->time_integrator_.reset();
      this->parabolic_module_.reset();
      this->hyperbolic_module_.reset();
      this->initial_values_ = nullptr;
      initial_values_container_.reset();
      this->offline_data_.reset();
      this->discretization_.reset();
      this->parabolic_system_ = nullptr;
      parabolic_system_container_.reset();
      hyperbolic_system_container_.reset();
      this->mpi_ensemble_.reset();
    }

  private:
    std::unique_ptr<MPIEnsembleContainer<HyperbolicSystem>>
        hyperbolic_system_container_;
    std::unique_ptr<MPIEnsembleContainer<ParabolicSystem>>
        parabolic_system_container_;
    std::unique_ptr<MPIEnsembleContainer<InitialValues<HD, dim, Number>>>
        initial_values_container_;
    std::unique_ptr<Extractor> mesh_adaptor_extractor_;
    std::unique_ptr<Extractor> postprocessor_extractor_;
    std::unique_ptr<Extractor> vtu_output_extractor_;
    std::unique_ptr<Extractor> quantities_extractor_;
    std::unique_ptr<Extractor> error_evaluation_extractor_;
  };

} // namespace ryujin

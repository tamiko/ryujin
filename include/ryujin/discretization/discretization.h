//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/compile_time_options.h>

#include <ryujin/base/mpi_ensemble.h>
#include <ryujin/base/patterns_conversion.h>
#include <ryujin/geometries/geometry.h>
#include <ryujin/interface/discretization.h>

#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/distributed/shared_tria.h>
#include <deal.II/distributed/tria.h>

#include <memory>
#include <set>
#include <string>

namespace ryujin
{
  enum class MeshType {
    serial,
    parallel_shared,
    parallel_distributed,
    parallel_fullydistributed
  };
} // namespace ryujin

#ifndef DOXYGEN
DECLARE_ENUM(ryujin::MeshType,
             LIST({ryujin::MeshType::serial, "serial"},
                  {ryujin::MeshType::parallel_shared, "parallel shared"},
                  {ryujin::MeshType::parallel_distributed,
                   "parallel distributed"},
                  {ryujin::MeshType::parallel_fullydistributed,
                   "parallel fullydistributed"}));
#endif

namespace ryujin
{
  template <int dim>
  class Discretization final : public Interface::Discretization<dim>
  {
  public:
    using typename Interface::Discretization<dim>::Collection;

    Discretization(const MPIEnsemble &mpi_ensemble,
                   const std::string &subsection = "/Discretization");

    void prepare(const std::string &base_name) override;

    void update_mapping() override;

  private:
    MeshType mesh_type_;

    std::string geometry_;

    bool mesh_writeout_;
    double mesh_distortion_;

    const MPIEnsemble &mpi_ensemble_;

    std::set<std::shared_ptr<Geometry<dim>>> geometry_list_;
  };
} /* namespace ryujin */

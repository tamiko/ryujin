//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/compile_time_options.h>

#include <ryujin/base/convenience_macros.h>
#include <ryujin/base/mpi_ensemble.h>
#include <ryujin/base/patterns_conversion.h>
#include <ryujin/geometries/geometry.h>

#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/distributed/shared_tria.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/fe/mapping_q_cache.h>
#include <deal.II/hp/fe_collection.h>
#include <deal.II/hp/mapping_collection.h>
#include <deal.II/hp/q_collection.h>

#include <memory>
#include <set>
#include <vector>

namespace ryujin
{
  enum Boundary : dealii::types::boundary_id {
    do_nothing = 0,

    periodic = 1,

    slip = 2,

    no_slip = 3,

    dirichlet = 4,

    dynamic = 5,

    dirichlet_momentum = 6,

    dirichlet_velocity = 7
  };


  enum class Ansatz {
    cg_q1,

    cg_q2,

    cg_q3,

    dg_q1,

    dg_q2,

    dg_q3
  };

  enum class MeshType {
    serial,
    parallel_shared,
    parallel_distributed,
    parallel_fullydistributed
  };
} // namespace ryujin

#ifndef DOXYGEN
DECLARE_ENUM(ryujin::Boundary,
             LIST({ryujin::Boundary::do_nothing, "do nothing"},
                  {ryujin::Boundary::periodic, "periodic"},
                  {ryujin::Boundary::slip, "slip"},
                  {ryujin::Boundary::no_slip, "no slip"},
                  {ryujin::Boundary::dirichlet, "dirichlet"},
                  {ryujin::Boundary::dynamic, "dynamic"},
                  {ryujin::Boundary::dirichlet_momentum, "dirichlet momentum"},
                  {ryujin::Boundary::dirichlet_velocity,
                   "dirichlet velocity"}));

DECLARE_ENUM(ryujin::Ansatz,
             LIST({ryujin::Ansatz::cg_q1, "cG Q1"},
                  {ryujin::Ansatz::cg_q2, "cG Q2"},
                  {ryujin::Ansatz::cg_q3, "cG Q3"},
                  {ryujin::Ansatz::dg_q1, "dG Q1"},
                  {ryujin::Ansatz::dg_q2, "dG Q2"},
                  {ryujin::Ansatz::dg_q3, "dG Q3"}));

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
  class Discretization final : public dealii::ParameterAcceptor
  {
  public:
    struct Collection {
      std::unique_ptr<const dealii::hp::MappingCollection<dim>> mapping;
      std::unique_ptr<const dealii::hp::FECollection<dim>> finite_element_cg;
      std::unique_ptr<const dealii::hp::FECollection<dim>> finite_element_dg;
      std::unique_ptr<const dealii::hp::QCollection<dim>> quadrature;
      std::unique_ptr<const dealii::hp::QCollection<dim>> quadrature_high_order;
      std::unique_ptr<const dealii::hp::QCollection<dim>> nodal_quadrature;
      std::unique_ptr<const dealii::hp::QCollection<1>> quadrature_1d;
      std::unique_ptr<const dealii::hp::QCollection<1>> nodal_quadrature_1d;
      std::unique_ptr<const std::vector<dealii::hp::QCollection<dim - 1>>>
          face_quadrature;
      std::unique_ptr<const std::vector<dealii::hp::QCollection<dim - 1>>>
          face_nodal_quadrature;
    };

    Discretization(const MPIEnsemble &mpi_ensemble,
                   const std::string &subsection = "/Discretization");

    void prepare(const std::string &base_name);

    void update_mapping();

    ACCESSOR_READ_ONLY(selected_geometry)

    ACCESSOR_READ_ONLY(ansatz)

    bool have_discontinuous_ansatz() const
    {
      switch (ansatz_) {
      case Ansatz::cg_q1:
        [[fallthrough]];
      case Ansatz::cg_q2:
        [[fallthrough]];
      case Ansatz::cg_q3:
        return false;

      case Ansatz::dg_q1:
        [[fallthrough]];
      case Ansatz::dg_q2:
        [[fallthrough]];
      case Ansatz::dg_q3:
        return true;
      }

      AssertThrow(false, dealii::ExcInternalError());
      __builtin_trap();
    }

    unsigned int polynomial_degree() const
    {
      switch (ansatz_) {
      case Ansatz::cg_q1:
        [[fallthrough]];
      case Ansatz::dg_q1:
        return 1;
      case Ansatz::cg_q2:
        [[fallthrough]];
      case Ansatz::dg_q2:
        return 2;
      case Ansatz::cg_q3:
        [[fallthrough]];
      case Ansatz::dg_q3:
        return 3;
      }

      AssertThrow(false, dealii::ExcInternalError());
      __builtin_trap();
    }

    ACCESSOR(refinement)

    ACCESSOR(triangulation)

    ACCESSOR_READ_ONLY(triangulation)

    ACCESSOR_CONTAINER_READ_ONLY(collection_, mapping)

    ACCESSOR_CONTAINER_READ_ONLY(collection_, finite_element_cg)

    ACCESSOR_CONTAINER_READ_ONLY(collection_, finite_element_dg)

    const dealii::hp::FECollection<dim> &finite_element() const
    {
      if (have_discontinuous_ansatz())
        return *collection_.finite_element_dg;
      else
        return *collection_.finite_element_cg;
    }

    ACCESSOR_CONTAINER_READ_ONLY(collection_, quadrature)

    ACCESSOR_CONTAINER_READ_ONLY(collection_, quadrature_high_order)

    ACCESSOR_CONTAINER_READ_ONLY(collection_, nodal_quadrature)

    ACCESSOR_CONTAINER_READ_ONLY(collection_, quadrature_1d)

    ACCESSOR_CONTAINER_READ_ONLY(collection_, nodal_quadrature_1d)

    ACCESSOR_CONTAINER_READ_ONLY(collection_, face_quadrature)

    ACCESSOR_CONTAINER_READ_ONLY(collection_, face_nodal_quadrature)

  private:
    Ansatz ansatz_;
    MeshType mesh_type_;

    std::string geometry_;

    unsigned int refinement_;

    bool mesh_writeout_;
    double mesh_distortion_;

    const MPIEnsemble &mpi_ensemble_;

    std::unique_ptr<dealii::Triangulation<dim>> triangulation_;

    Collection collection_;
    std::shared_ptr<dealii::MappingQCache<dim>> mapping_cache_;

    std::set<std::shared_ptr<Geometry<dim>>> geometry_list_;
    std::shared_ptr<Geometry<dim>> selected_geometry_;

    template <typename Discretization, int dim_, typename Number_>
    friend class SolutionTransfer;

    template <int dim_>
    friend class Geometry;
  };
} /* namespace ryujin */

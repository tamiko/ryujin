//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/compile_time_options.h>

#include <ryujin/base/convenience_macros.h>
#include <ryujin/base/patterns_conversion.h>
#include <ryujin/geometries/geometry.h>

#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/fe/mapping_q_cache.h>
#include <deal.II/grid/tria.h>
#include <deal.II/hp/fe_collection.h>
#include <deal.II/hp/mapping_collection.h>
#include <deal.II/hp/q_collection.h>

#include <memory>
#include <string>
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
#endif

namespace ryujin
{
#ifndef DOXYGEN
  template <typename HyperbolicDescription, int dim, typename Number>
  class SolutionTransfer;
#endif

  namespace Interface
  {
    template <int dim>
    class Discretization : public dealii::ParameterAcceptor
    {
    public:
      struct Collection {
        std::unique_ptr<const dealii::hp::MappingCollection<dim>> mapping;
        std::unique_ptr<const dealii::hp::FECollection<dim>> finite_element_cg;
        std::unique_ptr<const dealii::hp::FECollection<dim>> finite_element_dg;
        std::unique_ptr<const dealii::hp::QCollection<dim>> quadrature;
        std::unique_ptr<const dealii::hp::QCollection<dim>>
            quadrature_high_order;
        std::unique_ptr<const dealii::hp::QCollection<dim>> nodal_quadrature;
        std::unique_ptr<const dealii::hp::QCollection<1>> quadrature_1d;
        std::unique_ptr<const dealii::hp::QCollection<1>> nodal_quadrature_1d;
        std::unique_ptr<const std::vector<dealii::hp::QCollection<dim - 1>>>
            face_quadrature;
        std::unique_ptr<const std::vector<dealii::hp::QCollection<dim - 1>>>
            face_nodal_quadrature;
      };

      virtual void prepare(const std::string &base_name) = 0;

      virtual void update_mapping() = 0;

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

    protected:
      using ParameterAcceptor::ParameterAcceptor;

      Ansatz ansatz_;

      unsigned int refinement_;

      std::unique_ptr<dealii::Triangulation<dim>> triangulation_;

      Collection collection_;
      std::shared_ptr<dealii::MappingQCache<dim>> mapping_cache_;

      std::shared_ptr<Geometry<dim>> selected_geometry_;

      template <typename HyperbolicDescription_, int dim_, typename Number_>
      friend class ryujin::SolutionTransfer;

      template <int dim_>
      friend class ryujin::Geometry;
    };
  } // namespace Interface
} // namespace ryujin

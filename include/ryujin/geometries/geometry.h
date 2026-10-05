//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/compile_time_options.h>

#include <ryujin/base/convenience_macros.h>

#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/tria.h>

#include <functional>
#include <string>

namespace ryujin
{
#ifndef DOXYGEN
  template <int dim>
  class Discretization;
#endif

  template <int dim>
  class Geometry : public dealii::ParameterAcceptor
  {
  public:
    Geometry(const std::string &name, const std::string &subsection)
        : ParameterAcceptor(subsection + "/" + name)
        , name_(name)
    {
    }

    virtual void create_coarse_triangulation(
        dealii::Triangulation<dim> &triangulation) const = 0;

    virtual void update_dof_handler(dealii::DoFHandler<dim> &) const {}

    enum class HP_Collection {
      standard_quadrilaterals,
      standard_simplices,
      populated_by_geometry
    };

    virtual HP_Collection populate_hp_collections(
        const unsigned int,
        typename ryujin::Discretization<dim>::Collection &) const
    {
      return HP_Collection::standard_quadrilaterals;
    }

    ACCESSOR_READ_ONLY(transformation)

    ACCESSOR_READ_ONLY(name)

  protected:
    mutable std::function<dealii::Point<dim>(
        const typename dealii::Triangulation<dim>::cell_iterator &,
        const dealii::Point<dim> &)>
        transformation_;

  private:
    const std::string name_;
  };

} /* namespace ryujin */

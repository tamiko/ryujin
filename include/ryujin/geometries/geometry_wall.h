//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2022 - 2025 by the ryujin authors
//

#pragma once

#include <ryujin/base/compile_time_options.h>

#include <ryujin/geometries/geometry_common_includes.h>

namespace ryujin
{
  namespace GridGenerator
  {
    template <int dim, int spacedim, template <int, int> class Triangulation>
    void wall(Triangulation<dim, spacedim> &,
              const double,
              const double,
              const double)
    {
      AssertThrow(false, dealii::ExcNotImplemented());
      __builtin_trap();
    }


#ifndef DOXYGEN
    template <template <int, int> class Triangulation>
    void wall(Triangulation<2, 2> &triangulation,
              const double length,
              const double height,
              const double wall_position)
    {
      using namespace dealii;

      dealii::Triangulation<2, 2> tria1, tria2, tria3;
      tria3.set_mesh_smoothing(triangulation.get_mesh_smoothing());

      GridGenerator::subdivided_hyper_rectangle(
          tria1, {18, 6}, Point<2>(wall_position, 0), Point<2>(length, height));

      GridGenerator::subdivided_hyper_rectangle(
          tria2, {1, 6}, Point<2>(0., 0.), Point<2>(wall_position, height));

      GridGenerator::merge_triangulations(tria1, tria2, tria3);

      triangulation.copy_triangulation(tria3);

      for (auto cell : triangulation.active_cell_iterators()) {
        for (auto f : cell->face_indices()) {
          const auto face = cell->face(f);

          if (!face->at_boundary())
            continue;

          const auto center = face->center();

          if (center[0] > wall_position && center[1] < 1.e-6) {
            face->set_boundary_id(Boundary::slip);

          } else if (center[0] > length - 1.e-6) {

            face->set_boundary_id(Boundary::do_nothing);

          } else {
            face->set_boundary_id(Boundary::dirichlet);
          }
        }
      }
    }
#endif
  } /* namespace GridGenerator */


  namespace Geometries
  {
    template <int dim>
    class Wall : public Geometry<dim>
    {
    public:
      Wall(const std::string &subsection)
          : Geometry<dim>("wall", subsection)
      {
        length_ = 3.2;
        this->add_parameter(
            "length", length_, "length of computational domain");

        height_ = 1.0;
        this->add_parameter(
            "height", height_, "height of computational domain");

        wall_position_ = 1. / 6.;
        this->add_parameter(
            "wall position", wall_position_, "x position of wall");
      }

      void create_coarse_triangulation(
          dealii::Triangulation<dim> &triangulation) const final
      {
        GridGenerator::wall(triangulation, length_, height_, wall_position_);
      }

    private:
      double length_;
      double height_;
      double wall_position_;
    };
  } /* namespace Geometries */
} /* namespace ryujin */

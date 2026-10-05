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
    template <int dim, template <int, int> class Triangulation>
    void step(Triangulation<dim, dim> &,
              const double,
              const double,
              const double,
              const double)
    {
      AssertThrow(false, dealii::ExcNotImplemented());
      __builtin_trap();
    }


#ifndef DOXYGEN
    template <template <int, int> class Triangulation>
    void step(Triangulation<2, 2> &triangulation,
              const double length,
              const double height,
              const double step_position,
              const double step_height)
    {
      using namespace dealii;

      dealii::Triangulation<2, 2> tria1, tria2, tria3;
      tria3.set_mesh_smoothing(triangulation.get_mesh_smoothing());

      GridGenerator::subdivided_hyper_rectangle(
          tria1, {15, 4}, Point<2>(0., step_height), Point<2>(length, height));

      GridGenerator::subdivided_hyper_rectangle(
          tria2,
          {3, 1},
          Point<2>(0., 0.),
          Point<2>(step_position, step_height));

      GridGenerator::merge_triangulations(tria1, tria2, tria3);
      triangulation.copy_triangulation(tria3);

      for (auto cell : triangulation.active_cell_iterators()) {
        for (auto f : cell->face_indices()) {
          const auto face = cell->face(f);

          if (!face->at_boundary())
            continue;

          const auto center = face->center();

          if (center[0] > 0. + 1.e-6 && center[0] < length - 1.e-6)
            face->set_boundary_id(Boundary::slip);

          if (center[0] < 0. + 1.e-06)
            face->set_boundary_id(Boundary::dirichlet);
        }
      }

      triangulation.refine_global(4);

      Point<2> point(step_position + 0.0125, step_height - 0.0125);
      triangulation.set_manifold(1, SphericalManifold<2>(point));

      for (auto cell : triangulation.active_cell_iterators())
        for (unsigned int v : cell->vertex_indices()) {
          double distance =
              (cell->vertex(v) - Point<2>(step_position, step_height)).norm();
          if (distance < 1.e-6) {
            for (auto f : cell->face_indices()) {
              const auto face = cell->face(f);
              if (face->at_boundary())
                face->set_manifold_id(1);
              cell->set_manifold_id(1);
            }
          }
        }

      for (auto cell : triangulation.active_cell_iterators()) {
        if (cell->manifold_id() != 1)
          continue;

        cell->set_manifold_id(0);

        for (unsigned int v : cell->vertex_indices()) {
          auto &vertex = cell->vertex(v);

          if (std::abs(vertex[0] - step_position) < 1.e-6 &&
              vertex[1] > step_height - 1.e-6)
            vertex[0] = step_position + 0.0125 * (1 - std::sqrt(1. / 2.));

          if (std::abs(vertex[1] - step_height) < 1.e-6 &&
              vertex[0] < step_position + 0.005)
            vertex[1] = step_height - 0.0125 * (1 - std::sqrt(1. / 2.));
        }
      }
    }
#endif
  } /* namespace GridGenerator */


  namespace Geometries
  {
    template <int dim>
    class Step : public Geometry<dim>
    {
    public:
      Step(const std::string &subsection)
          : Geometry<dim>("step", subsection)
      {
        length_ = 3.;
        this->add_parameter(
            "length", length_, "length of computational domain");

        height_ = 1.;
        this->add_parameter(
            "height", height_, "height of computational domain");

        step_position_ = 0.6;
        this->add_parameter(
            "step position", step_position_, "x position of step");

        step_height_ = 0.2;
        this->add_parameter("step height", step_height_, "height of step");
      }

      void create_coarse_triangulation(
          dealii::Triangulation<dim> &triangulation) const final
      {
        GridGenerator::step(
            triangulation, length_, height_, step_position_, step_height_);
      }

    private:
      double length_;
      double height_;
      double step_position_;
      double step_height_;
    };
  } /* namespace Geometries */

} /* namespace ryujin */

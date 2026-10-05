//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2025 by the ryujin authors
//

#pragma once

#include <ryujin/base/compile_time_options.h>

#include <ryujin/geometries/geometry_common_includes.h>

namespace ryujin
{
  namespace GridGenerator
  {
    template <int dim, int spacedim, template <int, int> class Triangulation>
    void twotanks(Triangulation<dim, spacedim> &,
                  const double,
                  const double,
                  const double,
                  const double,
                  const unsigned int)
    {
      AssertThrow(false, dealii::ExcNotImplemented());
      __builtin_trap();
    }


#ifndef DOXYGEN
    template <template <int, int> class Triangulation>
    void twotanks(Triangulation<2, 2> &triangulation,
                  const double tank_length,
                  const double tank_width,
                  const double tunnel_length,
                  const double tunnel_width,
                  const unsigned int subdivisions_factor)
    {
      using namespace dealii;

      dealii::Triangulation<2, 2> res1, res2, res3, tank1, tank2, tunnel, final;

      const double tolerance = 1.e-8;

      Assert(
          tank_width - tunnel_width > tolerance,
          dealii::ExcMessage(
              " !!! The tank width must be larger than the tunnel width !!!"));

      const double diff = (tank_width - tunnel_width) / 2.;
      unsigned int sub_x =
          static_cast<int>(std::round(tank_length * subdivisions_factor));
      unsigned int sub_y =
          static_cast<int>(std::round(diff * subdivisions_factor));

      GridGenerator::subdivided_hyper_rectangle(
          res1,
          {sub_x, sub_y},
          Point<2>(-tank_length, -tank_width / 2.),
          Point<2>(0, -tunnel_width / 2.));

      GridGenerator::subdivided_hyper_rectangle(
          res3,
          {sub_x, sub_y},
          Point<2>(-tank_length, tunnel_width / 2.),
          Point<2>(0, tank_width / 2.));

      sub_y = static_cast<int>(std::round(tunnel_width * subdivisions_factor));

      GridGenerator::subdivided_hyper_rectangle(
          res2,
          {sub_x, sub_y},
          Point<2>(-tank_length, -tunnel_width / 2.),
          Point<2>(0, tunnel_width / 2.));

      tank1.set_mesh_smoothing(triangulation.get_mesh_smoothing());
      GridGenerator::merge_triangulations(
          {&res1, &res2, &res3}, tank1, tolerance);

      tank2.copy_triangulation(tank1);
      dealii::Point<2> shift_vector(tunnel_length + tank_length, 0.);
      dealii::GridTools::shift(shift_vector, tank2);

      sub_x = static_cast<int>(std::round(tunnel_length * subdivisions_factor));

      GridGenerator::subdivided_hyper_rectangle(
          tunnel,
          {sub_x, sub_y},
          Point<2>(0., -tunnel_width / 2.),
          Point<2>(tunnel_length, tunnel_width / 2.));


      final.set_mesh_smoothing(triangulation.get_mesh_smoothing());
      GridGenerator::merge_triangulations(
          {&tank1, &tunnel, &tank2}, final, tolerance);


      triangulation.copy_triangulation(final);

      for (auto cell : triangulation.active_cell_iterators()) {
        for (auto f : cell->face_indices()) {
          const auto face = cell->face(f);

          if (!face->at_boundary())
            continue;

          face->set_boundary_id(Boundary::slip);

          const auto center = face->center();
          if (center[0] > tank_length + tunnel_length - tolerance)
            face->set_boundary_id(Boundary::dynamic);

          if (center[0] < -tank_length + tolerance)
            face->set_boundary_id(Boundary::dynamic);
        }
      }
    }
#endif
  } /* namespace GridGenerator */


  namespace Geometries
  {
    template <int dim>
    class TwoTanks : public Geometry<dim>
    {
    public:
      TwoTanks(const std::string &subsection)
          : Geometry<dim>("two tanks", subsection)
      {
        tank_length_ = 100.;
        this->add_parameter(
            "tank length", tank_length_, "length of tanks [units]");

        tank_width_ = 100.;
        this->add_parameter("tank width", tank_width_, "width of tank [units]");

        tunnel_length_ = 10.;
        this->add_parameter(
            "tunnel length", tunnel_length_, "length of tunnel [units]");

        tunnel_width_ = 50.;
        this->add_parameter(
            "tunnel width", tunnel_width_, "width of tunnel [units]");

        subdivisions_factor_ = 1;
        this->add_parameter("subdivisions factor",
                            subdivisions_factor_,
                            "A number used for introducing subdivions in both "
                            "x-y direction. Useful when dealing with "
                            "measurements that are less than 1. ");
      }

      void create_coarse_triangulation(
          dealii::Triangulation<dim> &triangulation) const final
      {
        GridGenerator::twotanks(triangulation,
                                tank_length_,
                                tank_width_,
                                tunnel_length_,
                                tunnel_width_,
                                subdivisions_factor_);
      }

    private:
      double tank_length_;
      double tank_width_;
      double tunnel_length_;
      double tunnel_width_;

      unsigned int subdivisions_factor_;
    };
  } /* namespace Geometries */
} /* namespace ryujin */

//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception or LGPL-2.1-or-later
// Copyright (C) 2007 - 2022 by Martin Kronbichler
// Copyright (C) 2008 - 2022 by David Wells
// Copyright (C) 2020 - 2023 by the ryujin authors
//

#pragma once

#include <ryujin/discretization/transfinite_interpolation.h>

#include <boost/container/small_vector.hpp>

#include <deal.II/base/table.h>

namespace ryujin
{

  namespace internal
  {
    static constexpr double invalid_pull_back_coordinate = 20.0;
  }


  template <int dim, int spacedim>
  TransfiniteInterpolationManifold<dim,
                                   spacedim>::TransfiniteInterpolationManifold()
      : level_coarse(-1)
  {
    AssertThrow(dim > 1, ExcNotImplemented());
  }


  template <int dim, int spacedim>
  std::unique_ptr<Manifold<dim, spacedim>>
  TransfiniteInterpolationManifold<dim, spacedim>::clone() const
  {
    auto ptr = new TransfiniteInterpolationManifold<dim, spacedim>();
    if (triangulation.n_levels() != 0)
      ptr->initialize(triangulation, *chart_manifold);
    return std::unique_ptr<Manifold<dim, spacedim>>(ptr);
  }


  template <int dim, int spacedim>
  void TransfiniteInterpolationManifold<dim, spacedim>::initialize(
      const Triangulation<dim, spacedim> &external_triangulation,
      const Manifold<dim, spacedim> &chart_manifold)
  {
    this->triangulation.clear();
    this->triangulation.copy_triangulation(external_triangulation);

    this->chart_manifold = chart_manifold.clone();

    level_coarse = triangulation.last()->level();
    coarse_cell_is_flat.resize(triangulation.n_cells(level_coarse), false);
    typename Triangulation<dim, spacedim>::active_cell_iterator
        cell = triangulation.begin(level_coarse),
        endc = triangulation.end(level_coarse);
    for (; cell != endc; ++cell) {
      bool cell_is_flat = true;
      for (unsigned int l = 0; l < GeometryInfo<dim>::lines_per_cell; ++l)
        if (cell->line(l)->manifold_id() != cell->manifold_id() &&
            cell->line(l)->manifold_id() != numbers::flat_manifold_id)
          cell_is_flat = false;
      if (dim > 2)
        for (unsigned int q = 0; q < GeometryInfo<dim>::quads_per_cell; ++q)
          if (cell->quad(q)->manifold_id() != cell->manifold_id() &&
              cell->quad(q)->manifold_id() != numbers::flat_manifold_id)
            cell_is_flat = false;
      AssertIndexRange(static_cast<unsigned int>(cell->index()),
                       coarse_cell_is_flat.size());
      coarse_cell_is_flat[cell->index()] = cell_is_flat;
    }
  }


  namespace
  {
    template <typename AccessorType>
    Point<AccessorType::space_dimension> compute_transfinite_interpolation(
        const AccessorType &cell, const Point<1> &chart_point, const bool)
    {
      return cell.vertex(0) * (1. - chart_point[0]) +
             cell.vertex(1) * chart_point[0];
    }

    template <typename AccessorType>
    Point<AccessorType::space_dimension>
    compute_transfinite_interpolation(const AccessorType &cell,
                                      const Point<2> &chart_point,
                                      const bool cell_is_flat)
    {
      const unsigned int dim = AccessorType::dimension;
      const unsigned int spacedim = AccessorType::space_dimension;
      const types::manifold_id my_manifold_id = cell.manifold_id();
      const Triangulation<dim, spacedim> &tria = cell.get_triangulation();

      const std::array<Point<spacedim>, 4> vertices{
          {cell.vertex(0), cell.vertex(1), cell.vertex(2), cell.vertex(3)}};

      std::array<double, 4> weights_vertices{
          {(1. - chart_point[0]) * (1. - chart_point[1]),
           chart_point[0] * (1. - chart_point[1]),
           (1. - chart_point[0]) * chart_point[1],
           chart_point[0] * chart_point[1]}};

      Point<spacedim> new_point;
      if (cell_is_flat)
        for (const unsigned int v : GeometryInfo<2>::vertex_indices())
          new_point += weights_vertices[v] * vertices[v];
      else {
        std::array<double, GeometryInfo<2>::vertices_per_face> weights;
        std::array<Point<spacedim>, GeometryInfo<2>::vertices_per_face> points;
        const auto weights_view =
            make_array_view(weights.begin(), weights.end());
        const auto points_view = make_array_view(points.begin(), points.end());

        for (unsigned int line = 0; line < GeometryInfo<2>::lines_per_cell;
             ++line) {
          const double my_weight =
              (line % 2) ? chart_point[line / 2] : 1 - chart_point[line / 2];
          const double line_point = chart_point[1 - line / 2];

          const types::manifold_id line_manifold_id =
              cell.line(line)->manifold_id();
          if (line_manifold_id == my_manifold_id ||
              line_manifold_id == numbers::flat_manifold_id) {
            weights_vertices[GeometryInfo<2>::line_to_cell_vertices(line, 0)] -=
                my_weight * (1. - line_point);
            weights_vertices[GeometryInfo<2>::line_to_cell_vertices(line, 1)] -=
                my_weight * line_point;
          } else {
            points[0] =
                vertices[GeometryInfo<2>::line_to_cell_vertices(line, 0)];
            points[1] =
                vertices[GeometryInfo<2>::line_to_cell_vertices(line, 1)];
            weights[0] = 1. - line_point;
            weights[1] = line_point;
            new_point +=
                my_weight * tria.get_manifold(line_manifold_id)
                                .get_new_point(points_view, weights_view);
          }
        }

        for (const unsigned int v : GeometryInfo<2>::vertex_indices())
          new_point -= weights_vertices[v] * vertices[v];
      }

      return new_point;
    }

    static constexpr unsigned int face_to_cell_vertices_3d[6][4] = {
        {0, 2, 4, 6},
        {1, 3, 5, 7},
        {0, 4, 1, 5},
        {2, 6, 3, 7},
        {0, 1, 2, 3},
        {4, 5, 6, 7}};

    static constexpr unsigned int face_to_cell_lines_3d[6][4] = {{8, 10, 0, 4},
                                                                 {9, 11, 1, 5},
                                                                 {2, 6, 8, 9},
                                                                 {3, 7, 10, 11},
                                                                 {0, 1, 2, 3},
                                                                 {4, 5, 6, 7}};

    template <typename AccessorType>
    Point<AccessorType::space_dimension>
    compute_transfinite_interpolation(const AccessorType &cell,
                                      const Point<3> &chart_point,
                                      const bool cell_is_flat)
    {
      const unsigned int dim = AccessorType::dimension;
      const unsigned int spacedim = AccessorType::space_dimension;
      const types::manifold_id my_manifold_id = cell.manifold_id();
      const Triangulation<dim, spacedim> &tria = cell.get_triangulation();

      const std::array<Point<spacedim>, 8> vertices{{cell.vertex(0),
                                                     cell.vertex(1),
                                                     cell.vertex(2),
                                                     cell.vertex(3),
                                                     cell.vertex(4),
                                                     cell.vertex(5),
                                                     cell.vertex(6),
                                                     cell.vertex(7)}};

      double linear_shapes[10];
      for (unsigned int d = 0; d < 3; ++d) {
        linear_shapes[2 * d] = 1. - chart_point[d];
        linear_shapes[2 * d + 1] = chart_point[d];
      }

      for (unsigned int d = 6; d < 10; ++d)
        linear_shapes[d] = linear_shapes[d - 6];

      std::array<double, 8> weights_vertices;
      for (unsigned int i2 = 0, v = 0; i2 < 2; ++i2)
        for (unsigned int i1 = 0; i1 < 2; ++i1)
          for (unsigned int i0 = 0; i0 < 2; ++i0, ++v)
            weights_vertices[v] =
                (linear_shapes[4 + i2] * linear_shapes[2 + i1]) *
                linear_shapes[i0];

      Point<spacedim> new_point;
      if (cell_is_flat)
        for (unsigned int v = 0; v < 8; ++v)
          new_point += weights_vertices[v] * vertices[v];
      else {
        std::array<double, GeometryInfo<3>::lines_per_cell> weights_lines;
        std::fill(weights_lines.begin(), weights_lines.end(), 0.0);

        std::array<double, GeometryInfo<2>::vertices_per_cell> weights;
        std::array<Point<spacedim>, GeometryInfo<2>::vertices_per_cell> points;
        const auto weights_view =
            make_array_view(weights.begin(), weights.end());
        const auto points_view = make_array_view(points.begin(), points.end());

        for (const unsigned int face : GeometryInfo<3>::face_indices()) {
          const double my_weight = linear_shapes[face];
          const unsigned int face_even = face - face % 2;

          if (std::abs(my_weight) < 1e-13)
            continue;

          const types::manifold_id face_manifold_id =
              cell.face(face)->manifold_id();
          if (face_manifold_id == my_manifold_id ||
              face_manifold_id == numbers::flat_manifold_id) {
            for (unsigned int line = 0; line < GeometryInfo<2>::lines_per_cell;
                 ++line) {
              const double line_weight = linear_shapes[face_even + 2 + line];
              weights_lines[face_to_cell_lines_3d[face][line]] +=
                  my_weight * line_weight;
            }
            weights_vertices[face_to_cell_vertices_3d[face][0]] -=
                linear_shapes[face_even + 2] *
                (linear_shapes[face_even + 4] * my_weight);
            weights_vertices[face_to_cell_vertices_3d[face][1]] -=
                linear_shapes[face_even + 3] *
                (linear_shapes[face_even + 4] * my_weight);
            weights_vertices[face_to_cell_vertices_3d[face][2]] -=
                linear_shapes[face_even + 2] *
                (linear_shapes[face_even + 5] * my_weight);
            weights_vertices[face_to_cell_vertices_3d[face][3]] -=
                linear_shapes[face_even + 3] *
                (linear_shapes[face_even + 5] * my_weight);
          } else {
            for (const unsigned int v : GeometryInfo<2>::vertex_indices())
              points[v] = vertices[face_to_cell_vertices_3d[face][v]];
            weights[0] =
                linear_shapes[face_even + 2] * linear_shapes[face_even + 4];
            weights[1] =
                linear_shapes[face_even + 3] * linear_shapes[face_even + 4];
            weights[2] =
                linear_shapes[face_even + 2] * linear_shapes[face_even + 5];
            weights[3] =
                linear_shapes[face_even + 3] * linear_shapes[face_even + 5];
            new_point +=
                my_weight * tria.get_manifold(face_manifold_id)
                                .get_new_point(points_view, weights_view);
          }
        }

        const auto weights_view_line =
            make_array_view(weights.begin(), weights.begin() + 2);
        const auto points_view_line =
            make_array_view(points.begin(), points.begin() + 2);
        for (unsigned int line = 0; line < GeometryInfo<3>::lines_per_cell;
             ++line) {
          const double line_point =
              (line < 8 ? chart_point[1 - (line % 4) / 2] : chart_point[2]);
          double my_weight = 0.;
          if (line < 8)
            my_weight = linear_shapes[line % 4] * linear_shapes[4 + line / 4];
          else {
            const unsigned int subline = line - 8;
            my_weight =
                linear_shapes[subline % 2] * linear_shapes[2 + subline / 2];
          }
          my_weight -= weights_lines[line];

          if (std::abs(my_weight) < 1e-13)
            continue;

          const types::manifold_id line_manifold_id =
              cell.line(line)->manifold_id();
          if (line_manifold_id == my_manifold_id ||
              line_manifold_id == numbers::flat_manifold_id) {
            weights_vertices[GeometryInfo<3>::line_to_cell_vertices(line, 0)] -=
                my_weight * (1. - line_point);
            weights_vertices[GeometryInfo<3>::line_to_cell_vertices(line, 1)] -=
                my_weight * (line_point);
          } else {
            points[0] =
                vertices[GeometryInfo<3>::line_to_cell_vertices(line, 0)];
            points[1] =
                vertices[GeometryInfo<3>::line_to_cell_vertices(line, 1)];
            weights[0] = 1. - line_point;
            weights[1] = line_point;
            new_point -= my_weight * tria.get_manifold(line_manifold_id)
                                         .get_new_point(points_view_line,
                                                        weights_view_line);
          }
        }

        for (const unsigned int v : GeometryInfo<dim>::vertex_indices())
          new_point += weights_vertices[v] * vertices[v];
      }
      return new_point;
    }
  } // namespace


  template <int dim, int spacedim>
  Point<spacedim> TransfiniteInterpolationManifold<dim, spacedim>::push_forward(
      const typename Triangulation<dim, spacedim>::cell_iterator &cell,
      const Point<dim> &chart_point) const
  {
    AssertDimension(cell->level(), level_coarse);

    Assert(GeometryInfo<dim>::is_inside_unit_cell(chart_point, 5e-4),
           ExcMessage("chart_point is not in unit interval"));

    return compute_transfinite_interpolation(
        *cell, chart_point, coarse_cell_is_flat[cell->index()]);
  }


  template <int dim, int spacedim>
  DerivativeForm<1, dim, spacedim>
  TransfiniteInterpolationManifold<dim, spacedim>::push_forward_gradient(
      const typename Triangulation<dim, spacedim>::cell_iterator &cell,
      const Point<dim> &chart_point,
      const Point<spacedim> &pushed_forward_chart_point) const
  {
    DerivativeForm<1, dim, spacedim> grad;
    for (unsigned int d = 0; d < dim; ++d) {
      Point<dim> modified = chart_point;
      const double step = chart_point[d] > 0.5 ? -1e-8 : 1e-8;

      modified[d] += step;
      Tensor<1, spacedim> difference =
          compute_transfinite_interpolation(
              *cell, modified, coarse_cell_is_flat[cell->index()]) -
          pushed_forward_chart_point;
      for (unsigned int e = 0; e < spacedim; ++e)
        grad[e][d] = difference[e] / step;
    }
    return grad;
  }


  template <int dim, int spacedim>
  Point<dim> TransfiniteInterpolationManifold<dim, spacedim>::pull_back(
      const typename Triangulation<dim, spacedim>::cell_iterator &cell,
      const Point<spacedim> &point,
      const Point<dim> &initial_guess) const
  {
    Point<dim> outside;
    for (unsigned int d = 0; d < dim; ++d)
      outside[d] = internal::invalid_pull_back_coordinate;

    Point<dim> chart_point =
        GeometryInfo<dim>::project_to_unit_cell(initial_guess);

    Tensor<1, spacedim> residual =
        point - compute_transfinite_interpolation(
                    *cell, chart_point, coarse_cell_is_flat[cell->index()]);
    const double tolerance =
        1e-21 * Utilities::fixed_power<2>(cell->diameter());
    double residual_norm_square = residual.norm_square();
    DerivativeForm<1, dim, spacedim> inv_grad;
    bool must_recompute_jacobian = true;
    for (unsigned int i = 0; i < 100; ++i) {
      if (residual_norm_square < tolerance) {
        Tensor<1, dim> update;
        for (unsigned int d = 0; d < spacedim; ++d)
          for (unsigned int e = 0; e < dim; ++e)
            update[e] += inv_grad[d][e] * residual[d];
        return chart_point + update;
      }

      if (must_recompute_jacobian ||
          (residual_norm_square > 1e4 * tolerance && i % 7 == 0)) {
        DerivativeForm<1, dim, spacedim> grad = push_forward_gradient(
            cell, chart_point, Point<spacedim>(point - residual));
        if (grad.determinant() <= 0.0)
          return outside;
        inv_grad = grad.covariant_form();
        must_recompute_jacobian = false;
      }
      Tensor<1, dim> update;
      for (unsigned int d = 0; d < spacedim; ++d)
        for (unsigned int e = 0; e < dim; ++e)
          update[e] += inv_grad[d][e] * residual[d];

      double alpha = 1.;

      while (!GeometryInfo<dim>::is_inside_unit_cell(
                 chart_point + alpha * update, 0.2) &&
             alpha > 1e-7)
        alpha *= 0.5;

      const Tensor<1, spacedim> old_residual = residual;
      while (alpha > 1e-4) {
        Point<dim> guess = chart_point + alpha * update;
        const Tensor<1, dim> residual_guess =
            point - compute_transfinite_interpolation(
                        *cell, guess, coarse_cell_is_flat[cell->index()]);
        const double residual_norm_new = residual_guess.norm_square();
        if (residual_norm_new < residual_norm_square) {
          residual_norm_square = residual_norm_new;
          chart_point += alpha * update;
          residual = residual_guess;
          break;
        } else
          alpha *= 0.5;
      }
      if (alpha <= 1e-4) {
        if (must_recompute_jacobian == true) {
          return residual_norm_square < std::sqrt(tolerance) ? chart_point
                                                             : outside;
        } else
          must_recompute_jacobian = true;
      }

      const Tensor<1, spacedim> delta_f = old_residual - residual;

      Tensor<1, dim> Jinv_deltaf;
      for (unsigned int d = 0; d < spacedim; ++d)
        for (unsigned int e = 0; e < dim; ++e)
          Jinv_deltaf[e] += inv_grad[d][e] * delta_f[d];

      const Tensor<1, dim> delta_x = alpha * update;

      if (std::abs(delta_x * Jinv_deltaf) > 0.1 * tolerance &&
          !must_recompute_jacobian) {
        const Tensor<1, dim> factor =
            (delta_x - Jinv_deltaf) / (delta_x * Jinv_deltaf);
        Tensor<1, spacedim> jac_update;
        for (unsigned int d = 0; d < spacedim; ++d)
          for (unsigned int e = 0; e < dim; ++e)
            jac_update[d] += delta_x[e] * inv_grad[d][e];
        for (unsigned int d = 0; d < spacedim; ++d)
          for (unsigned int e = 0; e < dim; ++e)
            inv_grad[d][e] += factor[e] * jac_update[d];
      }
    }
    return outside;
  }


  template <int dim, int spacedim>
  std::array<unsigned int, 20> TransfiniteInterpolationManifold<dim, spacedim>::
      get_possible_cells_around_points(
          const ArrayView<const Point<spacedim>> &points) const
  {
    Assert(triangulation.n_levels() != 0, ExcNotInitialized());
    Assert(triangulation.begin_active()->level() == level_coarse,
           ExcInternalError());

    auto cell = triangulation.begin(level_coarse);
    const auto endc = triangulation.end(level_coarse);
    boost::container::small_vector<std::pair<double, unsigned int>, 200>
        distances_and_cells;
    for (; cell != endc; ++cell) {
      if (cell->material_id() == 42)
        continue;

      std::array<Point<spacedim>, GeometryInfo<dim>::vertices_per_cell>
          vertices;
      for (const unsigned int vertex_n : GeometryInfo<dim>::vertex_indices()) {
        vertices[vertex_n] = cell->vertex(vertex_n);
      }

      Point<spacedim> center;
      for (const unsigned int v : GeometryInfo<dim>::vertex_indices())
        center += vertices[v];
      center *= 1. / GeometryInfo<dim>::vertices_per_cell;
      double radius_square = 0.;
      for (const unsigned int v : GeometryInfo<dim>::vertex_indices())
        radius_square =
            std::max(radius_square, (center - vertices[v]).norm_square());
      bool inside_circle = true;
      for (unsigned int i = 0; i < points.size(); ++i)
        if ((center - points[i]).norm_square() > radius_square * 1.5) {
          inside_circle = false;
          break;
        }
      if (inside_circle == false)
        continue;

      double current_distance = 0;
      for (unsigned int i = 0; i < points.size(); ++i) {
        Point<dim> point =
            cell->real_to_unit_cell_affine_approximation(points[i]);
        current_distance += GeometryInfo<dim>::distance_to_unit_cell(point);
      }
      distances_and_cells.push_back(
          std::make_pair(current_distance, cell->index()));
    }
    AssertThrow(distances_and_cells.size() > 0,
                (typename Mapping<dim, spacedim>::ExcTransformationFailed()));
    std::sort(distances_and_cells.begin(), distances_and_cells.end());
    std::array<unsigned int, 20> cells;
    cells.fill(numbers::invalid_unsigned_int);
    for (unsigned int i = 0; i < distances_and_cells.size() && i < cells.size();
         ++i)
      cells[i] = distances_and_cells[i].second;

    return cells;
  }


  template <int dim, int spacedim>
  typename Triangulation<dim, spacedim>::cell_iterator
  TransfiniteInterpolationManifold<dim, spacedim>::compute_chart_points(
      const ArrayView<const Point<spacedim>> &surrounding_points,
      ArrayView<Point<dim>> chart_points) const
  {
    Assert(surrounding_points.size() == chart_points.size(),
           ExcMessage("The chart points array view must be as large as the "
                      "surrounding points array view."));

    std::array<unsigned int, 20> nearby_cells =
        get_possible_cells_around_points(surrounding_points);

    auto guess_chart_point_structdim_2 =
        [&](const unsigned int i) -> Point<dim> {
      Assert(
          surrounding_points.size() == 8 && 2 < i && i < 8,
          ExcMessage("This function assumes that there are eight surrounding "
                     "points around a two-dimensional object. It also assumes "
                     "that the first three chart points have already been "
                     "computed."));
      switch (i) {
      case 0:
      case 1:
      case 2:
        Assert(false, ExcInternalError());
        break;
      case 3:
        return chart_points[1] + (chart_points[2] - chart_points[0]);
      case 4:
        return 0.5 * (chart_points[0] + chart_points[2]);
      case 5:
        return 0.5 * (chart_points[1] + chart_points[3]);
      case 6:
        return 0.5 * (chart_points[0] + chart_points[1]);
      case 7:
        return 0.5 * (chart_points[2] + chart_points[3]);
      default:
        Assert(false, ExcInternalError());
      }

      return Point<dim>();
    };

    auto guess_chart_point_structdim_3 =
        [&](const unsigned int i) -> Point<dim> {
      Assert(
          surrounding_points.size() == 8 && 4 < i && i < 8,
          ExcMessage("This function assumes that there are eight surrounding "
                     "points around a three-dimensional object. It also "
                     "assumes that the first five chart points have already "
                     "been computed."));
      return chart_points[i - 4] + (chart_points[4] - chart_points[0]);
    };

    bool use_structdim_2_guesses = false;
    bool use_structdim_3_guesses = false;
    if (surrounding_points.size() == 8) {
      const Tensor<1, spacedim> v06 =
          surrounding_points[6] - surrounding_points[0];
      const Tensor<1, spacedim> v27 =
          surrounding_points[7] - surrounding_points[2];

      const double cosine = scalar_product(v06, v27) /
                            std::sqrt(v06.norm_square() * v27.norm_square());
      if (0.707 < cosine)
        use_structdim_2_guesses = true;
      else if (spacedim == 3)
        use_structdim_3_guesses = true;
    }
    Assert((!use_structdim_2_guesses && !use_structdim_3_guesses) ||
               (use_structdim_2_guesses ^ use_structdim_3_guesses),
           ExcInternalError());


    auto compute_chart_point =
        [&](const typename Triangulation<dim, spacedim>::cell_iterator &cell,
            const unsigned int point_index) {
          Point<dim> guess;
          bool used_affine_approximation = false;
          if (point_index == 3 && surrounding_points.size() >= 8)
            guess = chart_points[1] + (chart_points[2] - chart_points[0]);
          else if (use_structdim_2_guesses && 3 < point_index)
            guess = guess_chart_point_structdim_2(point_index);
          else if (use_structdim_3_guesses && 4 < point_index)
            guess = guess_chart_point_structdim_3(point_index);
          else if (dim == 3 && point_index > 7 &&
                   surrounding_points.size() == 26) {
            if (point_index < 20)
              guess =
                  0.5 * (chart_points[GeometryInfo<dim>::line_to_cell_vertices(
                             point_index - 8, 0)] +
                         chart_points[GeometryInfo<dim>::line_to_cell_vertices(
                             point_index - 8, 1)]);
            else
              guess =
                  0.25 * (chart_points[GeometryInfo<dim>::face_to_cell_vertices(
                              point_index - 20, 0)] +
                          chart_points[GeometryInfo<dim>::face_to_cell_vertices(
                              point_index - 20, 1)] +
                          chart_points[GeometryInfo<dim>::face_to_cell_vertices(
                              point_index - 20, 2)] +
                          chart_points[GeometryInfo<dim>::face_to_cell_vertices(
                              point_index - 20, 3)]);
          } else {
            guess = cell->real_to_unit_cell_affine_approximation(
                surrounding_points[point_index]);
            used_affine_approximation = true;
          }
          chart_points[point_index] =
              pull_back(cell, surrounding_points[point_index], guess);

          if (chart_points[point_index][0] ==
                  internal::invalid_pull_back_coordinate &&
              !used_affine_approximation) {
            guess = cell->real_to_unit_cell_affine_approximation(
                surrounding_points[point_index]);
            chart_points[point_index] =
                pull_back(cell, surrounding_points[point_index], guess);
          }

          if (chart_points[point_index][0] ==
              internal::invalid_pull_back_coordinate) {
            for (unsigned int d = 0; d < dim; ++d)
              guess[d] = 0.5;
            chart_points[point_index] =
                pull_back(cell, surrounding_points[point_index], guess);
          }
        };

    for (unsigned int c = 0; c < nearby_cells.size(); ++c) {
      typename Triangulation<dim, spacedim>::cell_iterator cell(
          &triangulation, level_coarse, nearby_cells[c]);
      bool inside_unit_cell = true;
      for (unsigned int i = 0; i < surrounding_points.size(); ++i) {
        compute_chart_point(cell, i);

        if (GeometryInfo<dim>::is_inside_unit_cell(chart_points[i], 5e-4) ==
            false) {
          inside_unit_cell = false;
          break;
        }
      }
      if (inside_unit_cell == true) {
        return cell;
      }

      if (c == nearby_cells.size() - 1 ||
          nearby_cells[c + 1] == numbers::invalid_unsigned_int) {
        std::ostringstream message;
        for (unsigned int b = 0; b <= c; ++b) {
          typename Triangulation<dim, spacedim>::cell_iterator cell(
              &triangulation, level_coarse, nearby_cells[b]);
          message << "Looking at cell " << cell->id()
                  << " with vertices: " << std::endl;
          for (const unsigned int v : GeometryInfo<dim>::vertex_indices())
            message << std::setprecision(16) << " " << cell->vertex(v)
                    << "    ";
          message << std::endl;
          message << "Transformation to chart coordinates: " << std::endl;
          for (unsigned int i = 0; i < surrounding_points.size(); ++i) {
            compute_chart_point(cell, i);
            message << std::setprecision(16) << surrounding_points[i] << " -> "
                    << chart_points[i] << std::endl;
          }
        }

        AssertThrow(false,
                    (typename Mapping<dim, spacedim>::ExcTransformationFailed(
                        message.str())));
      }
    }

    Assert(false, ExcInternalError());
    return typename Triangulation<dim, spacedim>::cell_iterator();
  }


  template <int dim, int spacedim>
  Point<spacedim>
  TransfiniteInterpolationManifold<dim, spacedim>::get_new_point(
      const ArrayView<const Point<spacedim>> &surrounding_points,
      const ArrayView<const double> &weights) const
  {
    boost::container::small_vector<Point<dim>, 100> chart_points(
        surrounding_points.size());
    ArrayView<Point<dim>> chart_points_view =
        make_array_view(chart_points.begin(), chart_points.end());
    const auto cell =
        compute_chart_points(surrounding_points, chart_points_view);

    const Point<dim> p_chart =
        chart_manifold->get_new_point(chart_points_view, weights);

    return push_forward(cell, p_chart);
  }


  template <int dim, int spacedim>
  void TransfiniteInterpolationManifold<dim, spacedim>::get_new_points(
      const ArrayView<const Point<spacedim>> &surrounding_points,
      const Table<2, double> &weights,
      ArrayView<Point<spacedim>> new_points) const
  {
    Assert(weights.size(0) > 0, ExcEmptyObject());
    AssertDimension(surrounding_points.size(), weights.size(1));

    boost::container::small_vector<Point<dim>, 100> chart_points(
        surrounding_points.size());
    ArrayView<Point<dim>> chart_points_view =
        make_array_view(chart_points.begin(), chart_points.end());
    const auto cell =
        compute_chart_points(surrounding_points, chart_points_view);

    boost::container::small_vector<Point<dim>, 100> new_points_on_chart(
        weights.size(0));
    chart_manifold->get_new_points(chart_points_view,
                                   weights,
                                   make_array_view(new_points_on_chart.begin(),
                                                   new_points_on_chart.end()));

    for (unsigned int row = 0; row < weights.size(0); ++row)
      new_points[row] = push_forward(cell, new_points_on_chart[row]);
  }

} // namespace ryujin

#pragma once

#include <ryujin/base/compile_time_options.h>

#include <ryujin/geometries/geometry_common_includes.h>

#include <deal.II/grid/grid_in.h>

#include <fstream>

namespace ryujin
{
  namespace Geometries
  {
    template <int dim>
    class Reader : public Geometry<dim>
    {
    public:
      Reader(const std::string &subsection)
          : Geometry<dim>("reader", subsection)
      {
        filename_ = "ryujin.msh";
        this->add_parameter("filename",
                            filename_,
                            "The mesh file to read in via dealii::GridIn. This "
                            "class supports, among others, reading in Gmsh "
                            "*.msh files, and the *.ucd file format.");

        use_simplices_ = false;
        this->add_parameter(
            "simplex mesh",
            use_simplices_,
            "If set to true, the triangulation is assumed to use simplices "
            "instead of quadrangles.");
      }

      void create_coarse_triangulation(
          dealii::Triangulation<dim> &triangulation) const final
      {
        dealii::GridIn<dim> gridin;
        gridin.attach_triangulation(triangulation);
        gridin.read(filename_);
      }

      typename Geometry<dim>::HP_Collection populate_hp_collections(
          const unsigned int,
          typename ryujin::Discretization<dim>::Collection &) const override
      {
        if (use_simplices_) {
          return Geometry<dim>::HP_Collection::standard_simplices;
        } else {
          return Geometry<dim>::HP_Collection::standard_quadrilaterals;
        }
      }

    private:
      std::string filename_;
      bool use_simplices_;
    };
  } // namespace Geometries
} // namespace ryujin

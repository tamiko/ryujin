//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2021 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include <hyperbolic_system.h>
#include <postprocessor.h>

#include "offline_data.h"

#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/grid/intergrid_map.h>
#include <deal.II/multigrid/mg_transfer_matrix_free.h>

#include <future>

namespace ryujin
{

  /**
   * the VTUOutput class implements output of the conserved state vector
   * and a number of postprocessed quantities computed by the Postprocessor
   * class.
   *
   * @ingroup TimeLoop
   */
  template <int dim, typename Number = double>
  class VTUOutput final : public dealii::ParameterAcceptor
  {
  public:
    /**
     * @copydoc HyperbolicSystem::problem_dimension
     */
    // clang-format off
    static constexpr unsigned int problem_dimension = HyperbolicSystem::problem_dimension<dim>;
    // clang-format on

    /**
     * @copydoc OfflineData::scalar_type
     */
    using scalar_type = typename OfflineData<dim, Number>::scalar_type;

    /**
     * @copydoc OfflineData::vector_type
     */
    using vector_type = typename OfflineData<dim, Number>::vector_type;

    /**
     * Constructor.
     */
    VTUOutput(const MPI_Comm &mpi_communicator,
              const ryujin::OfflineData<dim, Number> &offline_data,
              const ryujin::Postprocessor<dim, Number> &postprocessor,
              const std::string &subsection = "VTUOutput");

    /**
     * Prepare VTU output. A call to @ref prepare() allocates temporary
     * storage and is necessary before schedule_output() can be called.
     *
     * Calling prepare() allocates temporary storage for additional (dim +
     * 5) scalar vectors of type OfflineData::scalar_type.
     */
    void prepare();

    /**
     * Given a state vector @p U and a file name prefix @p name, the
     * current time @p t, and the current output cycle @p cycle) schedule a
     * solution output.
     *
     * The function post-processes quantities synchronously, and (depending
     * on configuration options) schedules the write-out asynchronously
     * onto a background worker thread. This implies that @p U can again be
     * modified once schedule_output() returned.
     *
     * The booleans @p output_full controls whether the full vector field
     * is written out. Correspondingly, @p output_cutplanes controls
     * whether cells in the vicinity of predefined cutplanes are written
     * out.
     *
     * The function requires MPI communication and is not reentrant.
     */
    void schedule_output(const vector_type &U,
                         std::string name,
                         Number t,
                         unsigned int cycle,
                         bool output_full = true,
                         bool output_cutplanes = true);

  private:
    /**
     * @name Run time options
     */
    //@{

    bool use_mpi_io_;

    std::vector<std::string> manifolds_;

    //@}
    /**
     * @name Internal data
     */
    //@{

    const MPI_Comm &mpi_communicator_;

    dealii::SmartPointer<const ryujin::OfflineData<dim, Number>> offline_data_;
    dealii::SmartPointer<const ryujin::Postprocessor<dim, Number>>
        postprocessor_;

    std::future<void> background_thread_status;

    std::array<scalar_type, problem_dimension> state_vector_;

    //@}
  };

} /* namespace ryujin */

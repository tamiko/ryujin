#ifndef TIMELOOP_TEMPLATE_H
#define TIMELOOP_TEMPLATE_H

#include "timeloop.h"

#include <helper.h>
#include <indicator.h>
#include <limiter.h>
#include <riemann_solver.h>

#include <deal.II/base/logstream.h>
#include <deal.II/base/revision.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/core/demangle.hpp>

#ifdef CALLGRIND
#include <valgrind/callgrind.h>
#endif

#include <filesystem>
#include <fstream>
#include <iomanip>


using namespace dealii;
using namespace grendel;


namespace ryujin
{

  template <int dim, typename Number>
  TimeLoop<dim, Number>::TimeLoop(const MPI_Comm &mpi_comm)
      : ParameterAcceptor("A - TimeLoop")
      , mpi_communicator(mpi_comm)
      , computing_timer(mpi_communicator,
                        timer_output,
                        TimerOutput::never,
                        TimerOutput::cpu_and_wall_times)
      , discretization(mpi_communicator, computing_timer, "B - Discretization")
      , offline_data(mpi_communicator,
                     computing_timer,
                     discretization,
                     "C - OfflineData")
      , initial_values("D - InitialValues")
      , time_step(mpi_communicator,
                  computing_timer,
                  offline_data,
                  initial_values,
                  "E - TimeStep")
      , postprocessor(mpi_communicator,
                      computing_timer,
                      offline_data,
                      "F - SchlierenPostprocessor")
      , mpi_rank(dealii::Utilities::MPI::this_mpi_process(mpi_communicator))
      , n_mpi_processes(
            dealii::Utilities::MPI::n_mpi_processes(mpi_communicator))
  {
    base_name = "cylinder";
    add_parameter("basename", base_name, "Base name for all output files");

    t_final = Number(5.);
    add_parameter("final time", t_final, "Final time");

    output_granularity = Number(0.01);
    add_parameter(
        "output granularity", output_granularity, "time interval for output");

    update_granularity = 10;
    add_parameter(
        "update granularity",
        update_granularity,
        "number of cycles after which output statistics are recomputed");

    enable_checkpointing = true;
    add_parameter("enable checkpointing",
                  enable_checkpointing,
                  "Write out checkpoints to resume an interrupted computation "
                  "at output granularity intervals");

    resume = false;
    add_parameter("resume", resume, "Resume an interrupted computation");

    write_mesh = false;
    add_parameter("write mesh",
                  write_mesh,
                  "Write out the (distributed) mesh in inp format");

    write_output_files = true;
    add_parameter("write output files",
                  write_output_files,
                  "Write out postprocessed output files in vtu/pvtu format");

    enable_compute_error = false;
    add_parameter("enable compute error",
                  enable_compute_error,
                  "Flag to control whether we compute the Linfty Linf_norm of "
                  "the difference to an analytic solution. Implemented only "
                  "for certain initial state configurations.");
  }


  template <int dim, typename Number>
  void TimeLoop<dim, Number>::run()
  {
    initialize();

    print_parameters();

#ifdef DEBUG_OUTPUT
    deallog << "TimeLoop<dim, Number>::run()" << std::endl;
#endif

    /* Create distributed triangulation and output the triangulation: */
    print_head("create triangulation");
    discretization.prepare();

    if (write_mesh) {
#ifdef DEBUG_OUTPUT
      deallog << "        output triangulation" << std::endl;
#endif
      std::ofstream output(base_name + "-triangulation-p" +
                           std::to_string(mpi_rank) + ".inp");
      GridOut().write_ucd(discretization.triangulation(), output);
    }

    /* Prepare data structures: */

    print_head("compute offline data");

    offline_data.prepare();
    time_step.prepare();
    postprocessor.prepare();

    print_mpi_partition();

    Number t = 0.;
    unsigned int output_cycle = 0;
    vector_type U;

    const auto &partitioner = offline_data.partitioner();
    for (auto &it : U)
      it.reinit(partitioner);

    if (!resume) {
      print_head("interpolate initial values");

      U = interpolate_initial_values();

    } else {

      print_head("resume interrupted computation");

      const auto &triangulation = discretization.triangulation();
      const unsigned int i = triangulation.locally_owned_subdomain();
      std::string name = base_name + "-checkpoint-" +
                         dealii::Utilities::int_to_string(i, 4) + ".archive";
      std::ifstream file(name, std::ios::binary);

      boost::archive::binary_iarchive ia(file);
      ia >> t >> output_cycle;

      for (auto &it1 : U) {
        for (auto &it2 : it1)
          ia >> it2;
        it1.update_ghost_values();
      }
    }

    if (write_output_files)
      output(U, base_name + "-solution", t, output_cycle);

    if (write_output_files && enable_compute_error) {
      const auto analytic = interpolate_initial_values(t);
      output(analytic, base_name + "-analytic_solution", t, output_cycle);
    }

    ++output_cycle;

    print_head("enter main loop");

    /* Loop: */

    unsigned int cycle = 1;
    for (; t < t_final; ++cycle) {

#ifdef DEBUG_OUTPUT
      print_cycle(cycle, t);
#endif

      /* Do a time step: */

      const auto tau = time_step.step(U, t);
      t += tau;

#ifndef DEBUG_OUTPUT
      if (cycle % update_granularity == 0)
        print_cycle_statistics(cycle, t);
#endif

      if (t > output_cycle * output_granularity) {

        if (write_output_files)
          output(U,
                 base_name + "-solution",
                 t,
                 output_cycle,
                 /*checkpoint*/ enable_checkpointing);

        if (write_output_files && enable_compute_error) {
          const auto analytic = interpolate_initial_values(t);
          output(analytic, base_name + "-analytic_solution", t, output_cycle);
        }

        ++output_cycle;

#ifndef DEBUG_OUTPUT
        print_cycle(cycle, t);
#endif
        print_throughput(cycle, t);
      }
    } /* end of loop */

    --cycle; /* We have actually performed one cycle less. */

#ifdef CALLGRIND
    CALLGRIND_DUMP_STATS;
#endif

    /* Wait for output thread: */

    if (output_thread.joinable())
      output_thread.join();

    if (enable_compute_error) {
      /* Output final error: */

      const auto &affine_constraints = offline_data.affine_constraints();
      for (auto &it : U)
        affine_constraints.distribute(it);
      compute_error(U, t);
    }

    computing_timer.print_summary();
    if (mpi_rank == 0) {
#ifdef DEBUG_OUTPUT
      auto &stream = deallog;
#else
      auto &stream = *filestream;
#endif
      stream << timer_output.str() << std::endl;
    }

    print_throughput(cycle, t);

    /* Detach deallog: */
#ifdef DEBUG_OUTPUT
    if (mpi_rank == 0) {
      deallog.detach();
    }
#endif
  }


  /**
   * Set up deallog output, read in parameters and initialize all objects.
   */
  template <int dim, typename Number>
  void TimeLoop<dim, Number>::initialize()
  {
    /* Read in parameters and initialize all objects: */

    if (mpi_rank == 0) {

      std::cout << "[Init] initiating flux capacitor" << std::endl;
      std::cout << "[Init] bringing Warp Core online" << std::endl;

      std::cout << "[Init] read parameters and allocate objects" << std::endl;

      ParameterAcceptor::initialize("ryujin.prm");

    } else {

      ParameterAcceptor::initialize("ryujin.prm");
      return;
    }

    /* Print out parameters to a prm file: */

    std::ofstream output(base_name + "-parameter.prm");
    ParameterAcceptor::prm.print_parameters(output, ParameterHandler::Text);

    /* Prepare and attach logfile: */

    filestream.reset(new std::ofstream(base_name + "-deallog.log"));

#ifdef DEBUG_OUTPUT
    deallog.pop();
    deallog.attach(*filestream);
    deallog.depth_console(4);
    deallog.depth_file(4);
#endif

#ifdef DEBUG
    deallog.push("DEBUG");
#endif
  }


  template <int dim, typename Number>
  typename TimeLoop<dim, Number>::vector_type
  TimeLoop<dim, Number>::interpolate_initial_values(Number t)
  {
#ifdef DEBUG_OUTPUT
    deallog << "TimeLoop<dim, Number>::interpolate_initial_values(t = " << t
            << ")" << std::endl;
#endif
    TimerOutput::Scope timer(computing_timer,
                             "time_loop - setup scratch space");

    vector_type U;

    const auto &partitioner = offline_data.partitioner();
    for (auto &it : U)
      it.reinit(partitioner);

    constexpr auto problem_dimension =
        ProblemDescription<dim, Number>::problem_dimension;

    const auto callable = [&](const auto &p) {
      return initial_values.initial_state(p, t);
    };

    for (unsigned int i = 0; i < problem_dimension; ++i)
      VectorTools::interpolate(offline_data.dof_handler(),
                               to_function<dim, Number>(callable, i),
                               U[i]);

    for (auto &it : U)
      it.update_ghost_values();

    return U;
  }


  template <int dim, typename Number>
  void TimeLoop<dim, Number>::compute_error(
      const typename TimeLoop<dim, Number>::vector_type &U, const Number t)
  {
#ifdef DEBUG_OUTPUT
    deallog << "TimeLoop<dim, Number>::compute_error()" << std::endl;
#endif
    TimerOutput::Scope timer(computing_timer, "time_loop - compute error");

    constexpr auto problem_dimension =
        ProblemDescription<dim, Number>::problem_dimension;

    /* Compute L_inf norm: */

    Vector<float> difference_per_cell(
        discretization.triangulation().n_active_cells());

    Number linf_norm = 0.;
    Number l1_norm = 0;
    Number l2_norm = 0;

    auto analytic = interpolate_initial_values(t);

    for (unsigned int i = 0; i < problem_dimension; ++i) {
      auto &error = analytic[i];

      /* Compute norms of analytic solution: */

      const Number linf_norm_analytic =
          Utilities::MPI::max(error.linfty_norm(), mpi_communicator);

      VectorTools::integrate_difference(offline_data.dof_handler(),
                                        error,
                                        ZeroFunction<dim, double>(),
                                        difference_per_cell,
                                        QGauss<dim>(3),
                                        VectorTools::L1_norm);

      const Number l1_norm_analytic =
          Utilities::MPI::sum(difference_per_cell.l1_norm(), mpi_communicator);

      VectorTools::integrate_difference(offline_data.dof_handler(),
                                        error,
                                        ZeroFunction<dim, double>(),
                                        difference_per_cell,
                                        QGauss<dim>(3),
                                        VectorTools::L2_norm);

      const Number l2_norm_analytic = Number(std::sqrt(Utilities::MPI::sum(
          std::pow(difference_per_cell.l2_norm(), 2), mpi_communicator)));

      /* Compute norms of error: */

      error -= U[i];

      const Number linf_norm_error =
          Utilities::MPI::max(error.linfty_norm(), mpi_communicator);

      VectorTools::integrate_difference(offline_data.dof_handler(),
                                        error,
                                        ZeroFunction<dim, double>(),
                                        difference_per_cell,
                                        QGauss<dim>(3),
                                        VectorTools::L1_norm);

      const Number l1_norm_error =
          Utilities::MPI::sum(difference_per_cell.l1_norm(), mpi_communicator);

      VectorTools::integrate_difference(offline_data.dof_handler(),
                                        error,
                                        ZeroFunction<dim, double>(),
                                        difference_per_cell,
                                        QGauss<dim>(3),
                                        VectorTools::L2_norm);

      const Number l2_norm_error = Number(std::sqrt(Utilities::MPI::sum(
          std::pow(difference_per_cell.l2_norm(), 2), mpi_communicator)));

      linf_norm += linf_norm_error / linf_norm_analytic;
      l1_norm += l1_norm_error / l1_norm_analytic;
      l2_norm += l2_norm_error / l2_norm_analytic;
    }

    if (mpi_rank != 0)
      return;

#ifdef DEBUG_OUTPUT
      auto &stream = deallog;
#else
      auto &stream = *filestream;
#endif

      print_head("compute error");

      stream << "Normalized consolidated Linf, L1, and L2 errors at "
             << "final time" << std::endl;
      stream << "#dofs = " << offline_data.dof_handler().n_dofs() << std::endl;
      stream << "t     = " << t << std::endl;
      stream << "Linf  = " << linf_norm << std::endl;
      stream << "L1    = " << l1_norm << std::endl;
      stream << "L2    = " << l2_norm << std::endl;
  }


  template <int dim, typename Number>
  void TimeLoop<dim, Number>::output(
      const typename TimeLoop<dim, Number>::vector_type &U,
      const std::string &name,
      Number t,
      unsigned int cycle,
      bool checkpoint)
  {
#ifdef DEBUG_OUTPUT
    deallog << "TimeLoop<dim, Number>::output(t = " << t
            << ", checkpoint = " << checkpoint << ")" << std::endl;
#endif

    /*
     * Offload output to a worker thread.
     *
     * We wait for a previous thread to finish before we schedule a new
     * one. This logic also serves as a mutex for the postprocessor class.
     */

    if (output_thread.joinable()) {
      TimerOutput::Scope timer(computing_timer, "time_loop - stalled output");
      output_thread.join();
    }

    postprocessor.compute(U, time_step.alpha());

    /* Output data in vtu format: */

    /* capture name, t, cycle, and checkpoint by value */
    const auto output_worker = [this, name, t, cycle, checkpoint]() {
      constexpr auto problem_dimension =
          ProblemDescription<dim, Number>::problem_dimension;
      constexpr auto n_quantities = Postprocessor<dim, Number>::n_quantities;

      const auto &dof_handler = offline_data.dof_handler();
      const auto &triangulation = discretization.triangulation();
      const auto &mapping = discretization.mapping();

      /* Checkpointing: */

      if (checkpoint) {
#ifdef DEBUG_OUTPUT
        deallog << "        Checkpointing" << std::endl;
#endif

        const unsigned int i = triangulation.locally_owned_subdomain();
        std::string name = base_name + "-checkpoint-" +
                           dealii::Utilities::int_to_string(i, 4) + ".archive";

        if (std::filesystem::exists(name))
          std::filesystem::rename(name, name + "~");

        std::ofstream file(name, std::ios::binary | std::ios::trunc);

        boost::archive::binary_oarchive oa(file);
        oa << t << cycle;
        for (const auto &it1 : postprocessor.U())
          for (const auto &it2 : it1)
            oa << it2;
      }

      dealii::DataOut<dim> data_out;
      data_out.attach_dof_handler(dof_handler);

      for (unsigned int i = 0; i < problem_dimension; ++i)
        data_out.add_data_vector(
            postprocessor.U()[i],
            ProblemDescription<dim, Number>::component_names[i]);

      for (unsigned int i = 0; i < n_quantities; ++i)
        data_out.add_data_vector(postprocessor.quantities()[i],
                                 postprocessor.component_names[i]);

      data_out.build_patches(mapping,
                             discretization.finite_element().degree - 1);

      DataOutBase::VtkFlags flags(
          t, cycle, true, DataOutBase::VtkFlags::best_speed);
      data_out.set_flags(flags);

      const auto name_with_cycle =
          name + "-" + Utilities::int_to_string(cycle, 6);

      const auto filename = [&](const unsigned int i) -> std::string {
        const auto seq = dealii::Utilities::int_to_string(i, 4);
        return name_with_cycle + "-" + seq + ".vtu";
      };

      /* Write out local vtu: */

      const unsigned int i = triangulation.locally_owned_subdomain();
      std::ofstream output(filename(i));
      data_out.write_vtu(output);

      if (mpi_rank == 0) {
        /* Write out pvtu control file: */

        std::vector<std::string> filenames;
        for (unsigned int i = 0; i < n_mpi_processes; ++i)
          filenames.push_back(filename(i));

        std::ofstream output(name_with_cycle + ".pvtu");
        data_out.write_pvtu_record(output, filenames);
      }

#ifdef DEBUG_OUTPUT
      deallog << "        Commit output (cycle = " << cycle << ")" << std::endl;
#endif
    };

    /*
     * And spawn the thread:
     */

#ifdef DEBUG_OUTPUT
    deallog << "        Schedule output (cycle = " << cycle << ")" << std::endl;
#endif
    output_thread = std::move(std::thread(output_worker));
  }

  /*
   * Output and logging related functions:
   */


  template <int dim, typename Number>
  void TimeLoop<dim, Number>::print_parameters()
  {
    if (mpi_rank != 0)
      return;

#ifdef DEBUG_OUTPUT
    auto &stream = deallog;
#else
    auto &stream = *filestream;
#endif

    /* Output commit and library informations: */

    /* clang-format off */
    stream << std::endl;
    stream << "###" << std::endl;
    stream << "#" << std::endl;
    stream << "# deal.II version " << std::setw(8) << DEAL_II_PACKAGE_VERSION
            << "  -  " << DEAL_II_GIT_REVISION << std::endl;
    stream << "# ryujin  version " << std::setw(8) << RYUJIN_VERSION
            << "  -  " << RYUJIN_GIT_REVISION << std::endl;
    stream << "#" << std::endl;
    stream << "###" << std::endl;
    stream << std::endl;

    /* Print compile time parameters: */

    stream << "Compile time parameters:" << std::endl;

    stream << "DIM == " << dim << std::endl;
    stream << "NUMBER == " << typeid(Number).name() << std::endl;

#ifdef USE_SIMD
    stream << "SIMD width == " << VectorizedArray<Number>::n_array_elements << std::endl;
#else
    stream << "SIMD width == " << "(( disabled ))" << std::endl;
#endif

#ifdef USE_CUSTOM_POW
    stream << "serial pow == broadcasted pow(Vec4f)/pow(Vec2d)" << std::endl;
#else
    stream << "serial pow == std::pow"<< std::endl;
#endif

    stream << "Indicator<dim, Number>::indicators_ == ";
    switch (Indicator<dim, Number>::indicator_) {
    case Indicator<dim, Number>::Indicators::zero:
      stream << "Indicator<dim, Number>::Indicators::zero" << std::endl;
      break;
    case Indicator<dim, Number>::Indicators::one:
      stream << "Indicator<dim, Number>::Indicators::one" << std::endl;
      break;
    case Indicator<dim, Number>::Indicators::entropy_viscosity_commutator:
      stream << "Indicator<dim, Number>::Indicators::entropy_viscosity_commutator" << std::endl;
      break;
    case Indicator<dim, Number>::Indicators::smoothness_indicator:
      stream << "Indicator<dim, Number>::Indicators::smoothness_indicator" << std::endl;
    }

    stream << "Indicator<dim, Number>::smoothness_indicator_ == ";
    switch (Indicator<dim, Number>::smoothness_indicator_) {
    case Indicator<dim, Number>::SmoothnessIndicators::rho:
      stream << "Indicator<dim, Number>::SmoothnessIndicators::rho" << std::endl;
      break;
    case Indicator<dim, Number>::SmoothnessIndicators::internal_energy:
      stream << "Indicator<dim, Number>::SmoothnessIndicators::internal_energy" << std::endl;
      break;
    case Indicator<dim, Number>::SmoothnessIndicators::pressure:
      stream << "Indicator<dim, Number>::SmoothnessIndicators::pressure" << std::endl;
    }

    stream << "Indicator<dim, Number>::smoothness_indicator_alpha_0_ == "
            << Indicator<dim, Number>::smoothness_indicator_alpha_0_ << std::endl;

    stream << "Indicator<dim, Number>::smoothness_indicator_power_ == "
            << Indicator<dim, Number>::smoothness_indicator_power_ << std::endl;

    stream << "Indicator<dim, Number>::compute_second_variations_ == "
            << Indicator<dim, Number>::compute_second_variations_ << std::endl;

    stream << "Limiter<dim, Number>::limiter_ == ";
    switch (Limiter<dim, Number>::limiter_) {
    case Limiter<dim, Number>::Limiters::none:
      stream << "Limiter<dim, Number>::Limiters::none" << std::endl;
      break;
    case Limiter<dim, Number>::Limiters::rho:
      stream << "Limiter<dim, Number>::Limiters::rho" << std::endl;
      break;
    case Limiter<dim, Number>::Limiters::specific_entropy:
      stream << "Limiter<dim, Number>::Limiters::specific_entropy" << std::endl;
      break;
    case Limiter<dim, Number>::Limiters::entropy_inequality:
      stream << "Limiter<dim, Number>::Limiters::entropy_inequality" << std::endl;
      break;
    }

    stream << "grendel::newton_max_iter == "
            << grendel::newton_max_iter << std::endl;

    stream << "Limiter<dim, Number>::relax_bounds_ == "
            << Limiter<dim, Number>::relax_bounds_ << std::endl;

    stream << "Limiter<dim, Number>::relaxation_order_ == "
            << Limiter<dim, Number>::relaxation_order_ << std::endl;

    stream << "RiemannSolver<dim, Number>::newton_max_iter_ == "
            <<  RiemannSolver<dim, Number>::newton_max_iter_ << std::endl;

    stream << "RiemannSolver<dim, Number>::greedy_dij_ == "
            <<  RiemannSolver<dim, Number>::greedy_dij_ << std::endl;

    stream << "RiemannSolver<dim, Number>::greedy_threshold_ == "
            <<  RiemannSolver<dim, Number>::greedy_threshold_ << std::endl;

    stream << "RiemannSolver<dim, Number>::greedy_relax_bounds_ == "
            <<  RiemannSolver<dim, Number>::greedy_relax_bounds_ << std::endl;


    stream << "TimeStep<dim, Number>::order_ == ";
    switch (TimeStep<dim, Number>::order_) {
    case TimeStep<dim, Number>::Order::first_order:
      stream << "TimeStep<dim, Number>::Order::first_order" << std::endl;
      break;
    case TimeStep<dim, Number>::Order::second_order:
      stream << "TimeStep<dim, Number>::Order::second_order" << std::endl;
    }

    stream << "TimeStep<dim, Number>::time_step_order_ == ";
    switch (TimeStep<dim, Number>::time_step_order_) {
    case TimeStep<dim, Number>::TimeStepOrder::first_order:
      stream << "TimeStep<dim, Number>::TimeStepOrder::first_order" << std::endl;
      break;
    case TimeStep<dim, Number>::TimeStepOrder::second_order:
      stream << "TimeStep<dim, Number>::TimeStepOrder::second_order" << std::endl;
      break;
    case TimeStep<dim, Number>::TimeStepOrder::third_order:
      stream << "TimeStep<dim, Number>::TimeStepOrder::third_order" << std::endl;
      break;
    }

    stream << "TimeStep<dim, Number>::limiter_iter_ == "
            <<  TimeStep<dim, Number>::limiter_iter_ << std::endl;

    /* clang-format on */

#ifndef DEBUG_OUTPUT
    stream << std::endl;
    stream << "Run time parameters:" << std::endl;
    ParameterAcceptor::prm.print_parameters(
        stream, ParameterHandler::OutputStyle::ShortText);
#endif
  }


  template <int dim, typename Number>
  void TimeLoop<dim, Number>::print_mpi_partition()
  {
    unsigned int dofs[2];
    dofs[0] = offline_data.n_locally_owned();
    dofs[1] = offline_data.n_locally_internal();

    if (mpi_rank > 0) {
      MPI_Send(&dofs, 2, MPI_UNSIGNED, 0, 0, mpi_communicator);

    } else {

#ifdef DEBUG_OUTPUT
      auto &stream = deallog;
#else
      auto &stream = *filestream;
#endif

      stream << "Number of MPI ranks: " << n_mpi_processes << std::endl;
      stream << "Number of threads:   " << MultithreadInfo::n_threads()
             << std::endl;

      /* Print out the DoF distribution: */

      const auto n_dofs = offline_data.dof_handler().n_dofs();

      stream << "Qdofs: " << n_dofs
             << " global DoFs, local DoF distribution:" << std::endl;


      for (unsigned int p = 0; p < n_mpi_processes; ++p) {
        stream << "    Rank " << p << std::flush;

        if (p != 0)
          MPI_Recv(&dofs,
                   2,
                   MPI_UNSIGNED,
                   p,
                   0,
                   mpi_communicator,
                   MPI_STATUS_IGNORE);

        stream << ":\tlocal: " << dofs[0] << std::flush;
        stream << "\tinternal: " << dofs[1] << std::endl;
      } /* p */
    }   /* mpi_rank */
  }


  /**
   * A small function that prints formatted section headings.
   */
  template <int dim, typename Number>
  void TimeLoop<dim, Number>::print_head(const std::string &header,
                                         const std::string &secondary,
                                         bool use_cout)
  {
    if (mpi_rank != 0)
      return;

#ifdef DEBUG_OUTPUT
    auto &stream = deallog;
#else
    std::ostream &stream = use_cout ? std::cout : *filestream;
#endif

    const auto header_size = header.size();
    const auto padded_header = std::string((34 - header_size) / 2, ' ') +
                               header +
                               std::string((35 - header_size) / 2, ' ');

    const auto secondary_size = secondary.size();
    const auto padded_secondary = std::string((34 - secondary_size) / 2, ' ') +
                                  secondary +
                                  std::string((35 - secondary_size) / 2, ' ');

    /* clang-format off */
    stream << std::endl;
    stream << "    ####################################################" << std::endl;
    stream << "    #########                                  #########" << std::endl;
    stream << "    #########"     <<  padded_header   <<     "#########" << std::endl;
    stream << "    #########"     << padded_secondary <<     "#########" << std::endl;
    stream << "    #########                                  #########" << std::endl;
    stream << "    ####################################################" << std::endl;
    stream << std::endl;
    /* clang-format on */

    if (secondary == "")
      std::cout << "[Init] " << header << std::endl;
  }


  /**
   * A small function that prints formatted section headings.
   */
  template <int dim, typename Number>
  void TimeLoop<dim, Number>::print_cycle(unsigned int cycle,
                                          Number t,
                                          bool use_cout)
  {
    std::ostringstream primary;
    primary << "Cycle  " << Utilities::int_to_string(cycle, 6) //
            << "  (" << std::fixed << std::setprecision(1)     //
            << t / t_final * 100 << "%)";

    std::ostringstream secondary;
    secondary << "at time t = " << std::setprecision(8) << std::fixed << t;

    print_head(primary.str(), secondary.str(), use_cout);
  }


  template <int dim, typename Number>
  void TimeLoop<dim, Number>::print_throughput(unsigned int cycle,
                                               Number t,
                                               bool use_cout)
  {
    /* Print Jean-Luc and Martin metrics: */

    const auto cpu_summary_data = computing_timer.get_summary_data(
        TimerOutput::OutputData::total_cpu_time);
    const auto wall_summary_data = computing_timer.get_summary_data(
        TimerOutput::OutputData::total_wall_time);

    double cpu_time =
        std::accumulate(cpu_summary_data.begin(),
                        cpu_summary_data.end(),
                        0.,
                        [](auto sum, auto it) { return sum + it.second; });
    cpu_time = Utilities::MPI::sum(cpu_time, mpi_communicator);

    const double wall_time =
        std::accumulate(wall_summary_data.begin(),
                        wall_summary_data.end(),
                        0.,
                        [](auto sum, auto it) { return sum + it.second; });

    const double cpu_m_dofs_per_sec =
        ((double)cycle) * ((double)offline_data.dof_handler().n_dofs()) / 1.e6 /
        cpu_time;
    const double wall_m_dofs_per_sec =
        ((double)cycle) * ((double)offline_data.dof_handler().n_dofs()) / 1.e6 /
        wall_time;

    /* Query data for number of restarts: */

    const auto n_calls_summary_data =
        computing_timer.get_summary_data(TimerOutput::OutputData::n_calls);
    const auto n_euler_steps =
        n_calls_summary_data.at("time_step - 1 compute d_ij, and alpha_i");

    auto n_restart_euler_steps = n_euler_steps;
    switch (TimeStep<dim, Number>::time_step_order_) {
    case TimeStep<dim, Number>::TimeStepOrder::first_order:
      n_restart_euler_steps -= cycle;
      break;
    case TimeStep<dim, Number>::TimeStepOrder::second_order:
      n_restart_euler_steps -= 2. * cycle;
      break;
    case TimeStep<dim, Number>::TimeStepOrder::third_order:
      n_restart_euler_steps -= 3. * cycle;
      break;
    }

    std::ostringstream head;
    head << std::setprecision(4) << std::endl << std::endl;
    head << "Throughput:  (CPU )  "                            //
         << std::fixed << cpu_m_dofs_per_sec << " MQ/s  ("     //
         << std::scientific << 1. / cpu_m_dofs_per_sec * 1.e-6 //
         << " s/Qdof/cycle)" << std::endl;
    head << "             (WALL)  "                             //
         << std::fixed << wall_m_dofs_per_sec << " MQ/s  ("     //
         << std::scientific << 1. / wall_m_dofs_per_sec * 1.e-6 //
         << " s/Qdof/cycle)  ("                                 //
         << std::fixed << ((double)cycle) / wall_time           //
         << " cycles/s)  (avg dt = "                            //
         << std::scientific << t / ((double)cycle)              //
         << ")" << std::endl;
    head << "                     ["                                    //
         << std::setprecision(0) << std::fixed << n_restart_euler_steps //
         << " rsts  (" << std::setprecision(4) << std::scientific
         << n_restart_euler_steps / ((double)cycle) << " rsts/cycle)]"
         << std::endl;

    if (mpi_rank != 0)
      return;

#ifdef DEBUG_OUTPUT
    auto &stream = deallog;
#else
    std::ostream &stream = use_cout ? std::cout : *filestream;
#endif

    stream << head.str() << std::endl;
  }


  /**
   * A small function that prints formatted section headings.
   */
  template <int dim, typename Number>
  void TimeLoop<dim, Number>::print_cycle_statistics(unsigned int cycle,
                                                     Number t)
  {
    if (mpi_rank == 0) {
      std::ostringstream primary;
      primary << "Cycle  " << Utilities::int_to_string(cycle, 6) //
              << "  (" << std::fixed << std::setprecision(1)     //
              << t / t_final * 100 << "%)";

      std::ostringstream secondary;
      secondary << "at time t = " << std::setprecision(8) << std::fixed << t;

      std::cout << "\033[2J\033[H" << std::endl;

      print_head(primary.str(), secondary.str(), /*use_cout*/ true);

      std::cout << "Information: [" << base_name << "] with "
                << offline_data.dof_handler().n_dofs() << " Qdofs on "
                << n_mpi_processes << " ranks / "
                << MultithreadInfo::n_threads() << " threads" << std::flush;
    }

    print_throughput(cycle, t, /*use_cout*/ true);

    computing_timer.print_summary();
    auto summary = timer_output.str();
    timer_output.str("");

    if (mpi_rank == 0) {
      /* Remove CPU statistics: */
      summary.erase(0, summary.length() / 2 + 1);
      std::cout << summary << std::endl;
    }
  }


} // namespace ryujin

#endif /* TIMELOOP_TEMPLATE_H */

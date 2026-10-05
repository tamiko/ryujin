//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2023 - 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/compile_time_options.h>
#include <ryujin/driver/simulation.h>
#include <ryujin/driver/time_loop.h>
#include <ryujin/stub/parabolic_description.h>

#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_acceptor.h>

#include <algorithm>
#include <functional>
#include <map>
#include <string>

namespace ryujin
{
  inline const std::string dave =
      "\nDave, this conversation can serve no purpose anymore. Goodbye.\n\n";


  template <typename HyperbolicDescription,
            typename ParabolicDescription,
            int dim,
            typename Number>
  void create_prm_files(const std::string &name,
                        bool write_detailed_description);


  class EquationDispatch final
  {
  public:
    class Parameters final : dealii::ParameterAcceptor
    {
    public:
      Parameters()
          : ParameterAcceptor("B - Equation")
      {
        dimension = 0;
        add_parameter("dimension", dimension, "The spatial dimension");
        add_parameter("equation", equation, "The PDE system");
      }

      int dimension;
      std::string equation;
    };

    template <typename HyperbolicDescription,
              typename ParabolicDescription = StubParabolicDescription,
              typename Number = NUMBER>
    void add(const std::string &name)
    {
      using HD = HyperbolicDescription;
      using PD = ParabolicDescription;

      Entry entry;

      entry.create_parameter_files = [name]() {
        create_prm_files<HD, PD, 1, Number>(name, false);
        create_prm_files<HD, PD, 2, Number>(name, true);
        create_prm_files<HD, PD, 3, Number>(name, false);
      };

      entry.run = [](const int dimension,
                     const std::string &parameter_file,
                     const MPI_Comm &mpi_comm) {
        if (dimension == 1)
          run_simulation<HD, PD, 1, Number>(parameter_file, mpi_comm);
        else if (dimension == 2)
          run_simulation<HD, PD, 2, Number>(parameter_file, mpi_comm);
        else
          run_simulation<HD, PD, 3, Number>(parameter_file, mpi_comm);
      };

      const bool inserted = entries_.emplace(name, std::move(entry)).second;
      AssertThrow(inserted,
                  dealii::ExcMessage(dave + "The equation »" + name +
                                     "« has been registered twice.\n"));
    }

    void dispatch(const std::string &parameter_file,
                  const MPI_Comm &mpi_comm) const;

    void create_parameter_files() const;

  private:
    template <typename HD, typename PD, int dim, typename Number>
    static void run_simulation(const std::string &parameter_file,
                               const MPI_Comm &mpi_comm)
    {
      Simulation<HD, PD, dim, Number> simulation(mpi_comm);
      TimeLoop<dim, Number> time_loop(simulation);
      dealii::ParameterAcceptor::initialize(parameter_file);
      time_loop.run();
    }

    struct Entry {
      std::function<void()> create_parameter_files;
      std::function<void(int, const std::string &, const MPI_Comm &)> run;
    };

    std::map<std::string, Entry> entries_;
  };


  void register_equations(EquationDispatch &equation_dispatch);


#ifndef DOXYGEN


  template <typename HyperbolicDescription,
            typename ParabolicDescription,
            int dim,
            typename Number>
  void create_prm_files(const std::string &name,
                        bool write_detailed_description)
  {
    {
      auto &prm = dealii::ParameterAcceptor::prm;
      prm.enter_subsection("A - TimeLoop");
      prm.declare_entry("basename", "test");
      prm.leave_subsection();

      EquationDispatch::Parameters parameters;
      Simulation<HyperbolicDescription, ParabolicDescription, dim, Number>
          simulation(MPI_COMM_SELF);
      TimeLoop<dim, Number> time_loop(simulation);

      prm.enter_subsection("B - Equation");
      prm.declare_entry("dimension",
                        std::to_string(dim),
                        dealii::Patterns::Integer(),
                        "The spatial dimension");
      prm.declare_entry(
          "equation", name, dealii::Patterns::Anything(), "The PDE system");
      prm.set("dimension", std::to_string(dim));
      prm.set("equation", name);
      prm.leave_subsection();

      std::string base_name = name;
      std::ranges::replace(base_name, ' ', '_');
      base_name += "-" + std::to_string(dim) + "d";

      if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_SELF) == 0) {
        const auto full_name =
            "default_parameters-" + base_name + "-description.prm";
        if (write_detailed_description)
          prm.print_parameters(
              full_name,
              dealii::ParameterHandler::OutputStyle::KeepDeclarationOrder);

        const auto short_name = "default_parameters-" + base_name + ".prm";
        prm.print_parameters(
            short_name,
            dealii::ParameterHandler::OutputStyle::Short |
                dealii::ParameterHandler::OutputStyle::KeepDeclarationOrder

        );
      }
    }

    dealii::ParameterAcceptor::clear();
  }
#endif
} // namespace ryujin

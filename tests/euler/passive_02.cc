//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#include <ryujin/driver/environment.h>
#include <ryujin/driver/scheduler.h>
#include <ryujin/driver/simulation.h>
#include <ryujin/euler/hyperbolic_description.h>
#include <ryujin/euler/initial_state_contrast.h>
#include <ryujin/euler/initial_state_uniform.h>
#include <ryujin/stub/parabolic_module.h>
#include <ryujin/typed/hyperbolic_module.template.h>
#include <ryujin/typed/initial_values.template.h>
#include <ryujin/typed/postprocessor.h>
#include <ryujin/typed/solution_transfer.template.h>

#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_acceptor.h>

#include <iostream>
#include <sstream>
#include <string>

//
// Test the compressible Euler equations with one passively advected
// component q: A contact discontinuity q / rho = 1 | 2 at x = 0 in a
// uniform flow with rho = 1, v = (1, 0), p = 1 is transported to x = 0.5
// at t = 0.5. On a uniform mesh of [-1, 1]^2 this implies that q / rho
// > 1.5 holds on (roughly) a quarter of all nodes, and (with a limiter
// relaxation factor of 0) that q / rho stays within [1, 2].
//

using namespace ryujin;
using namespace dealii;

constexpr int dim = 2;
using Number = NUMBER;


class PassiveSystem : public Euler::HyperbolicSystem<1>
{
public:
  PassiveSystem(const std::string &subsection = "/HyperbolicSystem")
      : Euler::HyperbolicSystem<1>(subsection, {"q"})
  {
  }
};


struct PassiveDescription {
  using HyperbolicSystem = PassiveSystem;

  template <int dim, typename Number>
  using InitialValues = ryujin::InitialValues<HyperbolicSystem, dim, Number>;

  template <typename ScalarNumber = double>
  using Indicator = Euler::Indicator<ScalarNumber, 1>;

  template <typename ScalarNumber = double>
  using Limiter = Euler::Limiter<ScalarNumber, 1>;

  template <typename ScalarNumber = double>
  using WaveSpeedEstimator = Euler::WaveSpeedEstimator<ScalarNumber, 1>;

  template <int dim, typename Number>
  using HyperbolicModule =
      ryujin::HyperbolicModule<PassiveDescription, dim, Number>;
};


namespace ryujin
{
  template <>
  auto InitialStateLibrary<PassiveSystem, dim, NUMBER>::defaults()
      -> std::vector<Creator>
  {
    using EulerInitialStates::Contrast;
    using EulerInitialStates::Uniform;
    return {creator<Contrast<PassiveSystem, dim, NUMBER>>(),
            creator<Uniform<PassiveSystem, dim, NUMBER>>()};
  }

  template class Library<InitialState<PassiveSystem, dim, NUMBER>,
                         const PassiveSystem &,
                         const std::string &>;
  template class InitialValues<PassiveSystem, dim, NUMBER>;
  template class HyperbolicModule<PassiveDescription, dim, NUMBER>;
  template class SolutionTransfer<PassiveDescription, dim, NUMBER>;
} // namespace ryujin


/*
 * Run the transport problem until @p final_time and return the fraction
 * of nodes with q / rho > 1.5 and whether q / rho is within [1, 2].
 */
std::pair<double, bool> run(const std::string &name, double final_time)
{
  std::stringstream parameters;
  parameters << "subsection A - Scheduler\n"
             << "  set basename   = " << name << "\n"
             << "  set final time = " << final_time << "\n"
             << "end\n"
             << "subsection C - Discretization\n"
             << "  set geometry        = rectangular domain\n"
             << "  set mesh refinement = 5\n"
             << "  subsection rectangular domain\n"
             << "    set boundary condition bottom = slip\n"
             << "    set boundary condition left   = dirichlet\n"
             << "    set boundary condition right  = dirichlet\n"
             << "    set boundary condition top    = slip\n"
             << "    set position bottom left      = -1, -1\n"
             << "    set position top right        =  1,  1\n"
             << "  end\n"
             << "end\n"
             << "subsection E - InitialValues\n"
             << "  set configuration = contrast\n"
             << "  set direction     = 1, 0\n"
             << "  set position      = 0, 0\n"
             << "  subsection contrast\n"
             << "    set primitive state left  = 1, 1, 1, 1\n"
             << "    set primitive state right = 1, 1, 1, 2\n"
             << "  end\n"
             << "end\n"
             << "subsection H - TimeIntegrator\n"
             << "  set cfl min               = 0.5\n"
             << "  set cfl max               = 0.5\n"
             << "  set cfl recovery strategy = none\n"
             << "  set time stepping scheme  = erk 33\n"
             << "end\n"
             << "subsection F - HyperbolicModule\n"
             << "  subsection limiter\n"
             << "    set relaxation factor = 0\n"
             << "  end\n"
             << "end\n";

  unsigned int n_nodes = 0;
  unsigned int n_right = 0;
  bool within_bounds = true;

  {
    Simulation<PassiveDescription, StubParabolicDescription, dim, Number>
        simulation(MPI_COMM_WORLD);
    Scheduler<Number> scheduler(
        MPI_COMM_WORLD, {&simulation}, {}, [](const std::string &) {});

    ParameterAcceptor::initialize(parameters);
    scheduler.run();

    const auto &final_state = simulation.state_vector().state;
    final_state.template copy_to_memory_space<dealii::MemorySpace::Host>();

    const auto &U = final_state.deal_ii_vector();
    const unsigned int n_comp = final_state.n_comp();
    const unsigned int n_owned = U.locally_owned_size() / n_comp;

    for (unsigned int i = 0; i < n_owned; ++i) {
      const auto rho = U.local_element(i * n_comp);
      const auto q = U.local_element(i * n_comp + 2 + dim);
      const auto phi = q / rho;
      n_nodes += 1;
      n_right += (phi > 1.5);
      within_bounds &= (phi >= 1. - 1.e-12 && phi <= 2. + 1.e-12);
    }

    n_nodes = Utilities::MPI::sum(n_nodes, MPI_COMM_WORLD);
    n_right = Utilities::MPI::sum(n_right, MPI_COMM_WORLD);
    within_bounds = Utilities::MPI::min(within_bounds ? 1 : 0, MPI_COMM_WORLD);
  }

  ParameterAcceptor::clear();
  return {double(n_right) / n_nodes, within_bounds};
}


int main(int argc, char *argv[])
{
  Environment environment(argc, argv);

  const auto [initial_fraction, initial_bounds] = run("initial", 0.);
  const auto [final_fraction, final_bounds] = run("final", 0.5);

  std::cout << "t = 0.0:" << std::endl;
  std::cout << "  q / rho > 1.5 on about 1/2 of the nodes: "
            << (std::abs(initial_fraction - 0.5) < 0.05) << std::endl;
  std::cout << "t = 0.5:" << std::endl;
  std::cout << "  q / rho > 1.5 on about 1/4 of the nodes: "
            << (std::abs(final_fraction - 0.25) < 0.05) << std::endl;
  std::cout << "  q / rho in [1, 2]:                       " << final_bounds
            << std::endl;

  return 0;
}

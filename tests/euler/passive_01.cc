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

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

//
// Test the compressible Euler equations with one passively advected
// component q on a two-dimensional Sod shock tube computed with convex
// limiting:
//
//  - For q = rho (i.e., q / rho = 1) the Euler components have to be
//    identical (up to round-off) to a computation without passive
//    component, and q / rho has to stay at 1.
//
//  - For a contrast of q / rho = 1 and q / rho = 2 and a relaxation
//    factor of 0 the final q / rho has to stay within [1, 2]. With the
//    default relaxation factor the bounds are relaxed in every time step
//    by a factor of order (m_i / |Omega|)^(1.5/d).
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


struct Result {
  std::vector<double> euler_norms;
  double phi_min = std::numeric_limits<double>::max();
  double phi_max = std::numeric_limits<double>::lowest();
};


/*
 * Run a Sod shock tube with the hyperbolic description HD, the passive
 * states @p q_left and @p q_right, and the given limiter relaxation
 * factor.
 */
template <typename HD>
Result run(const std::string &name,
           double q_left,
           double q_right,
           double relaxation_factor = 1.)
{
  constexpr bool passive = std::is_same_v<HD, PassiveDescription>;

  std::stringstream parameters;
  parameters << "subsection A - Scheduler\n"
             << "  set basename   = " << name << "\n"
             << "  set final time = 0.2\n"
             << "end\n"
             << "subsection C - Discretization\n"
             << "  set geometry        = rectangular domain\n"
             << "  set mesh refinement = 4\n"
             << "  subsection rectangular domain\n"
             << "    set boundary condition bottom = slip\n"
             << "    set boundary condition left   = slip\n"
             << "    set boundary condition right  = slip\n"
             << "    set boundary condition top    = slip\n"
             << "    set position bottom left      = -1, -1\n"
             << "    set position top right        =  1,  1\n"
             << "  end\n"
             << "end\n"
             << "subsection E - InitialValues\n"
             << "  set configuration = contrast\n"
             << "  set direction     = 1, 0.3\n"
             << "  set position      = 0, 0\n"
             << "  subsection contrast\n"
             << "    set primitive state left  = 1, 0, 1"
             << (passive ? ", " + std::to_string(q_left) : "") << "\n"
             << "    set primitive state right = 0.125, 0, 0.1"
             << (passive ? ", " + std::to_string(q_right) : "") << "\n"
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
             << "    set relaxation factor = " << relaxation_factor << "\n"
             << "  end\n"
             << "end\n";

  Result result;

  {
    Simulation<HD, StubParabolicDescription, dim, Number> simulation(
        MPI_COMM_WORLD);
    Scheduler<Number> scheduler(
        MPI_COMM_WORLD, {&simulation}, {}, [](const std::string &) {});

    ParameterAcceptor::initialize(parameters);
    scheduler.run();

    const auto &final_state = simulation.state_vector().state;
    final_state.template copy_to_memory_space<dealii::MemorySpace::Host>();

    const auto &U = final_state.deal_ii_vector();
    const unsigned int n_comp = final_state.n_comp();
    const unsigned int n_owned = U.locally_owned_size() / n_comp;

    result.euler_norms.resize(2 + dim, 0.);
    for (unsigned int i = 0; i < n_owned; ++i) {
      for (unsigned int c = 0; c < 2 + dim; ++c)
        result.euler_norms[c] += std::abs(U.local_element(i * n_comp + c));

      if constexpr (passive) {
        const auto rho = U.local_element(i * n_comp);
        const auto q = U.local_element(i * n_comp + 2 + dim);
        result.phi_min = std::min(result.phi_min, q / rho);
        result.phi_max = std::max(result.phi_max, q / rho);
      }
    }

    for (auto &norm : result.euler_norms)
      norm = Utilities::MPI::sum(norm, MPI_COMM_WORLD);
    result.phi_min = Utilities::MPI::min(result.phi_min, MPI_COMM_WORLD);
    result.phi_max = Utilities::MPI::max(result.phi_max, MPI_COMM_WORLD);
  }

  ParameterAcceptor::clear();
  return result;
}


/*
 * Compare the norms of the Euler components up to round-off.
 */
bool identical(const Result &left, const Result &right)
{
  for (unsigned int c = 0; c < left.euler_norms.size(); ++c)
    if (std::abs(left.euler_norms[c] - right.euler_norms[c]) >
        1.e-10 * std::abs(left.euler_norms[c]))
      return false;
  return true;
}


int main(int argc, char *argv[])
{
  Environment environment(argc, argv);

  std::cout << std::scientific << std::setprecision(8);

  const auto plain = run<Euler::HyperbolicDescription>("plain", 0., 0.);

  {
    const auto constant = run<PassiveDescription>("constant", 1., 0.125);

    std::cout << "q / rho = 1:" << std::endl;
    std::cout << "  Euler components identical: " << identical(plain, constant)
              << std::endl;
    std::cout << "  |q / rho - 1| < 1e-12:      "
              << (std::abs(constant.phi_min - 1.) < 1.e-12 &&
                  std::abs(constant.phi_max - 1.) < 1.e-12)
              << std::endl;
  }

  {
    const auto contrast = run<PassiveDescription>("contrast", 1., 0.25, 0.);

    std::cout << "q / rho = 1 | 2 (relaxation factor 0):" << std::endl;
    std::cout << "  q / rho in [1, 2]:          "
              << (contrast.phi_min >= 1. - 1.e-12 &&
                  contrast.phi_max <= 2. + 1.e-12)
              << std::endl;
  }

  {
    const auto contrast = run<PassiveDescription>("contrast", 1., 0.25);

    std::cout << "q / rho = 1 | 2 (relaxation factor 1):" << std::endl;
    std::cout << "  q / rho in [" << std::fixed << std::setprecision(4)
              << contrast.phi_min << ", " << contrast.phi_max << "]"
              << std::endl;
  }

  return 0;
}

//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/compile_time_options.h>

#include <ryujin/base/convenience_macros.h>

#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/tensor.h>

#include <set>
#include <string>

namespace ryujin
{
  template <typename HyperbolicDescription, int dim, typename Number = double>
  class InitialState : public dealii::ParameterAcceptor
  {
  public:
    using View =
        typename HyperbolicDescription::HyperbolicSystem::template View<dim,
                                                                        Number>;

    using state_type = typename View::state_type;

    using initial_precomputed_type = typename View::initial_precomputed_type;

    InitialState(const std::string &name, const std::string &subsection)
        : ParameterAcceptor(subsection + "/" + name)
        , name_(name)
    {
    }

    virtual state_type compute(const dealii::Point<dim> &point, Number t) = 0;

    virtual initial_precomputed_type
    initial_precomputations(const dealii::Point<dim> &)
    {
      return initial_precomputed_type{};
    }

    ACCESSOR_READ_ONLY(name)

  private:
    const std::string name_;
  };


  template <typename HyperbolicDescription, int dim, typename Number>
  class InitialStateLibrary
  {
  public:
    using HyperbolicSystem = typename HyperbolicDescription::HyperbolicSystem;

    using initial_state_list_type = std::set<
        std::unique_ptr<InitialState<HyperbolicDescription, dim, Number>>>;

    static void
    populate_initial_state_list(initial_state_list_type &initial_state_list,
                                const HyperbolicSystem &h,
                                const std::string &s);
  };
} // namespace ryujin

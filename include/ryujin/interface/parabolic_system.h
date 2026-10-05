//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/convenience_macros.h>

#include <deal.II/base/parameter_acceptor.h>

#include <string>
#include <vector>

namespace ryujin
{
  namespace Interface
  {
    class ParabolicSystem : public dealii::ParameterAcceptor
    {
    public:
      ACCESSOR_READ_ONLY(problem_name)

      ACCESSOR_READ_ONLY(component_names)

      unsigned int n_components() const
      {
        return component_names_.size();
      }

      ACCESSOR_READ_ONLY(is_identity)

    protected:
      ParabolicSystem(const std::string &problem_name,
                      const std::vector<std::string> &component_names,
                      const bool is_identity,
                      const std::string &subsection)
          : ParameterAcceptor(subsection)
          , problem_name_(problem_name)
          , component_names_(component_names)
          , is_identity_(is_identity)
      {
      }

    private:
      const std::string problem_name_;
      const std::vector<std::string> component_names_;
      const bool is_identity_;
    };
  } // namespace Interface
} // namespace ryujin

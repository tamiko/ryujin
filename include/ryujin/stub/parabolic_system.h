//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2025 - 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/compile_time_options.h>
#include <ryujin/interface/parabolic_system.h>

#include <string>

namespace ryujin
{
  class StubParabolicSystem final : public Interface::ParabolicSystem
  {
  public:
    StubParabolicSystem(const std::string &subsection = "/ParabolicSystem");
  };


  inline StubParabolicSystem::StubParabolicSystem(const std::string &subsection)
      : Interface::ParabolicSystem("Identity", {}, true, subsection)
  {
  }
} // namespace ryujin

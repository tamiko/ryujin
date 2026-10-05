//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/compile_time_options.h>
#include <ryujin/base/convenience_macros.h>
#include <ryujin/base/simd.h>

#include <deal.II/base/timer.h>

#include <map>
#include <string>

#ifdef DEBUG_OUTPUT
#include <iostream>
#endif

namespace ryujin
{
  class ComputingTimer
  {
  public:
    class Scope
    {
    public:
      Scope(const std::string &section)
          : section_(section)
      {
        timers_[section_].start();
#ifdef DEBUG_OUTPUT
        std::cout << "{scoped timer} \"" << section_ << "\" started"
                  << std::endl;
#endif
      }

      ~Scope()
      {
#ifdef DEBUG_OUTPUT
        std::cout << "{scoped timer} \"" << section_ << "\" stopped"
                  << std::endl;
#endif
        timers_[section_].stop();
      }

    private:
      const std::string section_;
    };

    static dealii::Timer &timer(const std::string &section)
    {
      return timers_[section];
    }

    static const std::map<std::string, dealii::Timer> &timers()
    {
      return timers_;
    }

    static void reinit()
    {
      timers_.clear();
    }

  private:
    static inline std::map<std::string, dealii::Timer> timers_;
  };


  class DeviceTimer
  {
  public:
    class Scope
    {
    public:
      Scope()
      {
        timer_.start();
      }

      ~Scope()
      {
        timer_.stop();
        n_kernels_ += 1;
      }
    };

    static double seconds()
    {
      return timer_.wall_time();
    }

    static std::uint64_t n_kernels()
    {
      return n_kernels_;
    }

    static void reinit()
    {
      timer_.reset();
      n_kernels_ = 0;
    }

  private:
    static inline dealii::Timer timer_;
    static inline std::uint64_t n_kernels_ = 0;
  };
} // namespace ryujin

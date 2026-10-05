//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception or LGPL-2.1-or-later
// Copyright (C) 2020 - 2026 by the ryujin authors
//

#pragma once

#include <ryujin/linear_algebra/state_vector.h>

#include <deal.II/base/exceptions.h>
#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/types.h>

#include <string>

namespace ryujin
{
  namespace Interface
  {
    template <int dim, typename Number = double>
    class SolutionTransfer : public dealii::ParameterAcceptor
    {
    public:
      using StateVector = Vectors::StateVector<Number>;

      virtual void prepare_projection(const StateVector &old_state_vector) = 0;

      virtual void project(StateVector &new_state_vector) = 0;

      unsigned int get_handle() const
      {
        Assert(handle_ != dealii::numbers::invalid_unsigned_int,
               dealii::ExcMessage(
                   "Invalid handle: Cannot retrieve a valid handle. "
                   "get_handle() can only be called after a call to "
                   "prepare_projection(), or set_handle()."));
        return handle_;
      }

      void set_handle(unsigned int handle)
      {
        Assert(handle_ == dealii::numbers::invalid_unsigned_int,
               dealii::ExcMessage(
                   "Invalid state: Cannot set handle because we already "
                   "have a valid handle due to a prior call to "
                   "prepare_projection(), or set_handle()."));
        handle_ = handle;
      }

      void reset_handle()
      {
        handle_ = dealii::numbers::invalid_unsigned_int;
      }

    protected:
      explicit SolutionTransfer(const std::string &subsection)
          : ParameterAcceptor(subsection)
          , handle_(dealii::numbers::invalid_unsigned_int)
      {
      }

      unsigned int handle_;
    };
  } // namespace Interface
} // namespace ryujin

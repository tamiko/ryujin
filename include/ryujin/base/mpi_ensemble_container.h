//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2025 - 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/compile_time_options.h>

#include <ryujin/base/mpi_ensemble.h>

#include <deal.II/base/mpi.h>
#include <deal.II/base/utilities.h>

namespace ryujin
{
  template <typename T>
  class MPIEnsembleContainer
  {
  public:
    template <typename... Args>
    MPIEnsembleContainer(const MPIEnsemble &mpi_ensemble,
                         const std::string &subsection,
                         Args &&...args)
    {
      const auto &ensemble = mpi_ensemble.ensemble();
      const auto &n_ensembles = mpi_ensemble.n_ensembles();
      unsigned int digits = dealii::Utilities::needed_digits(n_ensembles - 1);

      payload_.resize(n_ensembles);
      for (int n = 0; n < n_ensembles; ++n) {
        auto modified = subsection;
        if (n_ensembles > 1)
          modified +=
              "/ensemble " + dealii::Utilities::int_to_string(n, digits);
        payload_[n] =
            std::make_unique<T>(std::forward<Args>(args)..., modified);
      }

      ensemble_payload_ = payload_[ensemble].get();
    }

    const T &get() const
    {
      return *ensemble_payload_;
    }

    operator const T &() const
    {
      return *ensemble_payload_;
    }

  private:
    std::vector<std::unique_ptr<T>> payload_;
    T *ensemble_payload_;
  };


  namespace
  {
    template <typename C>
    static auto test(...) -> std::false_type;

    template <typename C>
    static auto test(int) -> decltype(C::n_mpi_ensembles(), std::true_type());
  } // namespace

  template <typename T>
  constexpr bool has_n_mpi_ensembles_v = decltype(test<T>(0))::value;
} /* namespace ryujin */

//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2024 - 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/compile_time_options.h>
#include <ryujin/base/gpu.h>
#include <ryujin/linear_algebra/multicomponent_vector.h>

#include <deal.II/base/exceptions.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <algorithm>
#include <functional>
#include <initializer_list>
#include <limits>
#include <tuple>
#include <vector>

namespace ryujin
{
#ifndef DOXYGEN
  template <int dim, typename Number>
  class OfflineData;
#endif

  namespace Vectors
  {
    template <typename Number>
    using ScalarHostVector = dealii::LinearAlgebra::distributed::Vector<Number>;


    template <typename Number>
    using ScalarVector = MultiComponentVector<Number>;


    template <typename Number>
    using StateVector = std::tuple<MultiComponentVector<Number>,
                                   MultiComponentVector<Number>,
                                   std::vector<ScalarVector<Number>>>;


    template <typename MemorySpace, typename Number>
    void sadd(StateVector<Number> &dst,
              const Number s,
              const Number b,
              const StateVector<Number> &src)
    {
      auto &dst_U = std::get<0>(dst).template deal_ii_vector<MemorySpace>();
      const auto &src_U =
          std::get<0>(src).template deal_ii_vector<MemorySpace>();
      dst_U.zero_out_ghost_values();
      dst_U.sadd(s, b, src_U);

      auto &dst_V = std::get<2>(dst);
      const auto &src_V = std::get<2>(src);
      AssertDimension(dst_V.size(), src_V.size());

      for (std::size_t k = 0; k < dst_V.size(); ++k) {
        auto &dst_V_k = dst_V[k].deal_ii_vector();
        const auto &src_V_k = src_V[k].deal_ii_vector();
        dst_V_k.zero_out_ghost_values();
        dst_V_k.sadd(s, b, src_V_k);
      }
    }


    template <typename Number, typename OfflineData>
    void debug_poison_invalid_values(
        [[maybe_unused]] StateVector<Number> &state_vector,
        [[maybe_unused]] const OfflineData &offline_data)
    {
#ifdef DEBUG

      auto &[U, prec, V] = state_vector;

      if constexpr (have_separate_memory_spaces) {
        U.template move_to_memory_space<dealii::MemorySpace::Host>();
        prec.template move_to_memory_space<dealii::MemorySpace::Host>();
      }

      constexpr auto nan = std::numeric_limits<Number>::signaling_NaN();

      const unsigned int n_owned = offline_data.n_locally_owned();
      const unsigned int n_relevant = offline_data.n_locally_relevant();
      const auto &partitioner = offline_data.scalar_partitioner();

      const unsigned int prob_dim = U.n_comp();
      const unsigned int prec_dim = prec.n_comp();
      Number *const U_data = U.deal_ii_vector().begin();
      Number *const prec_data = prec.deal_ii_vector().begin();

      for (unsigned int i = 0; i < n_owned; ++i) {
        std::fill_n(prec_data + i * prec_dim, prec_dim, nan);

        if (!offline_data.affine_constraints().is_constrained(
                partitioner->local_to_global(i)))
          continue;
        std::fill_n(U_data + i * prob_dim, prob_dim, nan);
      }

      for (unsigned int i = n_owned; i < n_relevant; ++i) {
        std::fill_n(prec_data + i * prec_dim, prec_dim, nan);

        std::fill_n(U_data + i * prob_dim, prob_dim, nan);
      }
#endif
    }

  } // namespace Vectors


  template <typename Number>
  using StageVectors = std::initializer_list<
      std::reference_wrapper<const Vectors::StateVector<Number>>>;

  template <typename Number>
  using StageWeights = std::initializer_list<Number>;

} // namespace ryujin

//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/simd.h>

#include <deal.II/base/tensor.h>
#include <deal.II/base/types.h>

#include <array>
#include <concepts>
#include <string>
#include <utility>

namespace ryujin
{
  template <int dim, typename Number>
  class OfflineData;


  namespace Concept
  {
    template <typename T, int dim, typename Number>
    using view_type =
        decltype(std::declval<const T &>().template view<dim, Number>());


    template <typename V, int dim, typename Number>
    concept HyperbolicSystemView =
        requires {
          typename V::ScalarNumber;
          typename V::flux_contribution_type;
          typename V::StateVector;
          typename V::HyperbolicVector;
          typename V::InitialPrecomputedVector;
          typename V::PrecomputedVectorView;
          typename V::InitialPrecomputedVectorView;
          { V::have_high_order_flux } -> std::convertible_to<bool>;
          { V::have_source_terms } -> std::convertible_to<bool>;
        } &&
        std::same_as<typename V::state_type,
                     dealii::Tensor<1, V::problem_dimension, Number>> &&
        std::same_as<typename V::precomputed_type,
                     std::array<Number, V::n_precomputed_values>> &&
        std::same_as<typename V::initial_precomputed_type,
                     std::array<Number, V::n_initial_precomputed_values>> &&
        requires {
          {
            V::component_names
          } -> std::convertible_to<
              const std::array<std::string, V::problem_dimension> &>;
          {
            V::primitive_component_names
          } -> std::convertible_to<
              const std::array<std::string, V::problem_dimension> &>;
          {
            V::precomputed_names
          } -> std::convertible_to<
              const std::array<std::string, V::n_precomputed_values> &>;
          {
            V::initial_precomputed_names
          } -> std::convertible_to<
              const std::array<std::string, V::n_initial_precomputed_values> &>;
        } &&
        requires(const V &view,
                 const typename V::state_type &U,
                 const typename V::flux_contribution_type &flux,
                 const typename V::PrecomputedVectorView &precomputed,
                 const typename V::InitialPrecomputedVectorView &initial,
                 const dealii::Tensor<1, dim, Number> &c_ij,
                 const dealii::types::boundary_id id,
                 typename V::state_type (&get_dirichlet_data)(),
                 const unsigned int i,
                 const unsigned int *js) {
          { view.is_admissible(U) } -> std::convertible_to<bool>;
          {
            view.apply_boundary_conditions(id, U, c_ij, get_dirichlet_data)
          } -> std::same_as<typename V::state_type>;
          {
            view.flux_contribution(precomputed, initial, i, U)
          } -> std::same_as<typename V::flux_contribution_type>;
          {
            view.flux_contribution(precomputed, initial, js, U)
          } -> std::same_as<typename V::flux_contribution_type>;
          {
            view.flux_divergence(flux, flux, c_ij)
          } -> std::same_as<typename V::state_type>;
          {
            view.to_primitive_state(U)
          } -> std::same_as<typename V::state_type>;
          {
            view.from_primitive_state(U)
          } -> std::same_as<typename V::state_type>;
        } &&
        (!V::have_high_order_flux ||
         requires(const V &view,
                  const typename V::flux_contribution_type &flux,
                  const dealii::Tensor<1, dim, Number> &c_ij) {
           {
             view.high_order_flux_divergence(flux, flux, c_ij)
           } -> std::same_as<typename V::state_type>;
         }) &&
        (!V::have_source_terms ||
         requires(const V &view,
                  const typename V::state_type &U,
                  const typename V::PrecomputedVectorView &precomputed,
                  const typename V::ScalarNumber tau,
                  const unsigned int i,
                  const unsigned int *js) {
           {
             view.nodal_source(precomputed, i, U, tau)
           } -> std::same_as<typename V::state_type>;
           {
             view.nodal_source(precomputed, js, U, tau)
           } -> std::same_as<typename V::state_type>;
         });


    template <typename T, int dim, typename Number>
    concept HyperbolicSystem =
        std::constructible_from<T, const std::string &> &&
        HyperbolicSystemView<view_type<T, dim, Number>, dim, Number> &&
        requires(
            const T &system,
            const OfflineData<dim, typename get_value_type<Number>::type>
                &offline_data,
            typename view_type<T, dim, Number>::StateVector &state_vector) {
          { T::problem_name } -> std::convertible_to<std::string>;
          system.fill_precomputed_values(offline_data, state_vector);
        };
  } // namespace Concept
} // namespace ryujin

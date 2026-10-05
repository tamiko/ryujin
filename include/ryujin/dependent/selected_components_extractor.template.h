//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2024 - 2026 by the ryujin authors
//

#pragma once

#include <ryujin/dependent/selected_components_extractor.h>

#include <iterator>
#include <string>
#include <variant>
#include <vector>

namespace ryujin
{
  template <typename HyperbolicDescription, int dim, typename Number>
  SelectedComponentsExtractor<HyperbolicDescription, dim, Number>::
      SelectedComponentsExtractor(
          const OfflineData<dim, Number> &offline_data,
          const HyperbolicSystem &hyperbolic_system,
          const Interface::ParabolicSystem &parabolic_system,
          const InitialPrecomputedVector &initial_precomputed,
          const std::vector<std::string> &additional_names,
          const std::vector<std::reference_wrapper<const ScalarVector>>
              &additional_vectors)
      : Interface::SelectedComponentsExtractor<dim, Number>(
            std::vector<std::string>(std::begin(View::component_names),
                                     std::end(View::component_names)),
            std::vector<std::string>(
                std::begin(View::primitive_component_names),
                std::end(View::primitive_component_names)),
            std::vector<std::string>(std::begin(View::precomputed_names),
                                     std::end(View::precomputed_names)),
            std::vector<std::string>(
                std::begin(View::initial_precomputed_names),
                std::end(View::initial_precomputed_names)),
            parabolic_system,
            additional_names)
      , offline_data_(&offline_data)
      , hyperbolic_system_(&hyperbolic_system)
      , initial_precomputed_(initial_precomputed)
      , additional_vectors_(additional_vectors)
  {
    Assert(this->primitive_offset_ == primitive_offset &&
               this->precomputed_offset_ == precomputed_offset &&
               this->initial_offset_ == initial_offset &&
               this->parabolic_offset_ == parabolic_offset,
           dealii::ExcInternalError());

    Assert(additional_names.size() == additional_vectors_.size(),
           dealii::ExcMessage("The number of additional component names does "
                              "not match the number of additional vectors."));
  }


  template <typename HyperbolicDescription, int dim, typename Number>
  void SelectedComponentsExtractor<HyperbolicDescription, dim, Number>::prepare(
      const std::vector<std::string> &selected)
  {
    Interface::SelectedComponentsExtractor<dim, Number>::prepare(selected);

    view_ = std::monostate{};
  }


  template <typename HyperbolicDescription, int dim, typename Number>
  template <typename MemorySpace>
  void SelectedComponentsExtractor<HyperbolicDescription, dim, Number>::
      prepare_extraction(const StateVector &state_vector) const
  {
    ExtractorView<MemorySpace> view(*offline_data_, *hyperbolic_system_);

    view.read_conserved_ = this->read_conserved_;
    view.read_primitive_ = this->read_primitive_;
    view.read_precomputed_ = this->read_precomputed_;
    view.read_initial_ = this->read_initial_;

    if (this->read_conserved_ || this->read_primitive_)
      view.U_view_ = std::get<0>(state_vector)
                         .template view<View::problem_dimension, MemorySpace>();

    if (this->read_precomputed_)
      view.precomputed_view_ =
          std::get<1>(state_vector)
              .template view<View::n_precomputed_values, MemorySpace>();

    if (this->read_initial_)
      view.initial_view_ =
          initial_precomputed_
              .template view<View::n_initial_precomputed_values, MemorySpace>();

    Kokkos::View<typename ExtractorView<MemorySpace>::Entry *,
                 Kokkos::HostSpace>
        entries("selected_components_entries", this->n_selected());

    const auto &parabolic = std::get<2>(state_vector);

    for (unsigned int k = 0; k < this->n_selected(); ++k) {
      const auto offset = this->selection_[k];
      entries[k].offset = offset;

      if (offset < parabolic_offset)
        continue;

      if (offset < this->additional_offset_)
        entries[k].scalar_view = parabolic[offset - parabolic_offset]
                                     .template view<1, MemorySpace>();
      else
        entries[k].scalar_view =
            additional_vectors_[offset - this->additional_offset_]
                .get()
                .template view<1, MemorySpace>();
    }

    view.entries_ = Kokkos::create_mirror_view_and_copy(
        typename MemorySpace::kokkos_space{}, entries);

    view_.template emplace<ExtractorView<MemorySpace>>(std::move(view));
  }


  template <typename HyperbolicDescription, int dim, typename Number>
  auto SelectedComponentsExtractor<HyperbolicDescription, dim, Number>::extract(
      const StateVector &state_vector) const -> std::vector<ScalarHostVector>
  {
    prepare_extraction<dealii::MemorySpace::Host>(state_vector);
    return view<dealii::MemorySpace::Host>().extract();
  }


  template <typename HyperbolicDescription, int dim, typename Number>
  void SelectedComponentsExtractor<HyperbolicDescription, dim, Number>::
      prepare_extraction(const StateVector &state_vector) const
  {
    prepare_extraction<selected_memory_space_t>(state_vector);
  }


  template <typename HyperbolicDescription, int dim, typename Number>
  std::vector<Number>
  SelectedComponentsExtractor<HyperbolicDescription, dim, Number>::
      extract_moments(const unsigned int *indices,
                      const Number *masses,
                      const unsigned int n_points,
                      const unsigned int n_moments,
                      Number *values) const
  {
    using MemorySpace = selected_memory_space_t;

    const auto extractor_view = view<MemorySpace>();
    const unsigned int n_selected = this->n_selected();
    const unsigned int stride = n_moments * n_selected;

    const auto body = [=](auto, const unsigned int p) {
      auto *point = values + p * stride;
      extractor_view.extract_element(
          indices[p],
          [=](const unsigned int k, const Number value) { point[k] = value; });

      for (unsigned int k = 1; k < n_moments; ++k)
        for (unsigned int c = 0; c < n_selected; ++c)
          point[k * n_selected + c] =
              point[(k - 1) * n_selected + c] * point[c];

      const auto mass = masses[p];
      return [=](const unsigned int j) { return mass * point[j]; };
    };

    std::vector<Number> sums(stride, Number(0.));
    reduction_loop<MemorySpace>("quantities_accumulate",
                                body,
                                ArrayReducer<Kokkos::Sum<Number>>(sums),
                                0,
                                n_points);

    return sums;
  }


  template <typename HyperbolicDescription, int dim, typename Number>
  template <typename MemorySpace>
  SelectedComponentsExtractorView<HyperbolicDescription,
                                  dim,
                                  Number,
                                  MemorySpace>
  SelectedComponentsExtractor<HyperbolicDescription, dim, Number>::view() const
  {
    Assert(std::holds_alternative<ExtractorView<MemorySpace>>(view_),
           dealii::ExcMessage(
               "Invalid state: prepare_extraction() has to be called for the "
               "selected memory space before a view can be created."));

    return std::get<ExtractorView<MemorySpace>>(view_);
  }


  template <typename HyperbolicDescription,
            int dim,
            typename Number,
            typename MemorySpace>
  SelectedComponentsExtractorView<HyperbolicDescription,
                                  dim,
                                  Number,
                                  MemorySpace>::
      SelectedComponentsExtractorView(
          const OfflineData<dim, Number> &offline_data,
          const HyperbolicSystem &hyperbolic_system)
      : offline_data_(&offline_data)
      , system_views_(hyperbolic_system)
  {
  }


  template <typename HyperbolicDescription,
            int dim,
            typename Number,
            typename MemorySpace>
  auto SelectedComponentsExtractorView<HyperbolicDescription,
                                       dim,
                                       Number,
                                       MemorySpace>::extract() const
      -> std::vector<ScalarVector>
  {
    const auto &offline_data = *offline_data_;

    std::vector<ScalarVector> extracted_components(n_selected());

    struct Destination {
      Number *data;
    };

    Kokkos::View<Destination *, Kokkos::HostSpace> host_destinations(
        "selected_components_destinations", n_selected());

    for (unsigned int k = 0; k < n_selected(); ++k) {
      extracted_components[k].reinit(offline_data.scalar_partitioner());
      host_destinations[k].data = extracted_components[k].begin();
    }

    const auto destinations = Kokkos::create_mirror_view_and_copy(
        typename MemorySpace::kokkos_space{}, host_destinations);

    const auto view = *this;

    const auto body = [=](auto sentinel, unsigned int i) {
      using T = decltype(sentinel);

      view.template extract_element<T>(
          i, [&](const unsigned int k, const T &value) {
            if constexpr (std::is_same_v<T, dealii::VectorizedArray<Number>>)
              value.store(destinations[k].data + i);
            else
              destinations[k].data[i] = value;
          });
    };

    loop<MemorySpace, Number>("extract_selected_components",
                              body,
                              0,
                              offline_data.n_locally_internal(),
                              offline_data.n_locally_owned());

    for (auto &it : extracted_components)
      it.update_ghost_values();

    return extracted_components;
  }


  template <typename HyperbolicDescription,
            int dim,
            typename Number,
            typename MemorySpace>
  template <typename T, typename Writer>
  DEAL_II_HOST_DEVICE_ALWAYS_INLINE void SelectedComponentsExtractorView<
      HyperbolicDescription,
      dim,
      Number,
      MemorySpace>::extract_element(const unsigned int i,
                                    const Writer &write) const
  {
    T staging[Extractor::parabolic_offset];

    if (read_conserved_ || read_primitive_) {
      const auto U_i = U_view_.template read_tensor<T>(i);
      for (unsigned int d = 0; d < EquationView::problem_dimension; ++d)
        staging[Extractor::conserved_offset + d] = U_i[d];

      if (read_primitive_) {
        const auto primitive_i =
            system_views_.template view<T>().to_primitive_state(U_i);
        for (unsigned int d = 0; d < EquationView::problem_dimension; ++d)
          staging[Extractor::primitive_offset + d] = primitive_i[d];
      }
    }

    if (read_precomputed_) {
      const auto precomputed_i = precomputed_view_.template read_tensor<T>(i);
      for (unsigned int d = 0; d < EquationView::n_precomputed_values; ++d)
        staging[Extractor::precomputed_offset + d] = precomputed_i[d];
    }

    if (read_initial_) {
      const auto initial_i = initial_view_.template read_tensor<T>(i);
      for (unsigned int d = 0; d < EquationView::n_initial_precomputed_values;
           ++d)
        staging[Extractor::initial_offset + d] = initial_i[d];
    }

    for (unsigned int k = 0; k < n_selected(); ++k) {
      const auto &entry = entries_[k];

      if (entry.offset < Extractor::parabolic_offset)
        write(k, staging[entry.offset]);
      else
        write(k, entry.scalar_view.template read_entry<T>(i));
    }
  }
} // namespace ryujin

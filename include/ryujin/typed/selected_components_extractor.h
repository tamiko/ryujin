//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2024 - 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/compile_time_options.h>

#include <ryujin/base/gpu.h>
#include <ryujin/base/loop.h>
#include <ryujin/base/observer_pointer.h>
#include <ryujin/discretization/offline_data.h>
#include <ryujin/interface/selected_components_extractor.h>
#include <ryujin/linear_algebra/state_vector.h>

#include <algorithm>
#include <functional>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

namespace ryujin
{
  template <typename HyperbolicDescription,
            int dim,
            typename Number,
            typename MemorySpace>
  class SelectedComponentsExtractorView;

  template <typename HyperbolicDescription, int dim, typename Number>
  class SelectedComponentsExtractor final
      : public Interface::SelectedComponentsExtractor<dim, Number>
  {
  public:
    using HyperbolicSystem = typename HyperbolicDescription::HyperbolicSystem;

    using View = typename HyperbolicSystem::template View<dim, Number>;

    using StateVector = typename View::StateVector;
    using InitialPrecomputedVector = typename View::InitialPrecomputedVector;

    using ScalarVector = Vectors::ScalarVector<Number>;
    using ScalarHostVector = Vectors::ScalarHostVector<Number>;

    static constexpr unsigned int conserved_offset = 0;
    static constexpr unsigned int primitive_offset = View::problem_dimension;
    static constexpr unsigned int precomputed_offset =
        2 * View::problem_dimension;
    static constexpr unsigned int initial_offset =
        precomputed_offset + View::n_precomputed_values;
    static constexpr unsigned int parabolic_offset =
        initial_offset + View::n_initial_precomputed_values;

    SelectedComponentsExtractor(
        const OfflineData<dim, Number> &offline_data,
        const HyperbolicSystem &hyperbolic_system,
        const Interface::ParabolicSystem &parabolic_system,
        const InitialPrecomputedVector &initial_precomputed,
        const std::vector<std::string> &additional_names = {},
        const std::vector<std::reference_wrapper<const ScalarVector>>
            &additional_vectors = {});

    void prepare(const std::vector<std::string> &selected) override;

    template <typename MemorySpace = dealii::MemorySpace::Host>
    void prepare_extraction(const StateVector &state_vector) const;

    std::vector<ScalarHostVector>
    extract(const StateVector &state_vector) const override;

    void prepare_extraction(const StateVector &state_vector) const override;

    std::vector<Number> extract_moments(const unsigned int *indices,
                                        const Number *masses,
                                        const unsigned int n_points,
                                        const unsigned int n_moments,
                                        Number *values) const override;

    template <typename MemorySpace = dealii::MemorySpace::Host>
    SelectedComponentsExtractorView<HyperbolicDescription,
                                    dim,
                                    Number,
                                    MemorySpace>
    view() const;

  private:
    dealii::ObserverPointer<const OfflineData<dim, Number>> offline_data_;
    dealii::ObserverPointer<const HyperbolicSystem> hyperbolic_system_;


    const InitialPrecomputedVector &initial_precomputed_;

    const std::vector<std::reference_wrapper<const ScalarVector>>
        additional_vectors_;

    template <typename MemorySpace>
    using ExtractorView = SelectedComponentsExtractorView<HyperbolicDescription,
                                                          dim,
                                                          Number,
                                                          MemorySpace>;

    mutable std::variant<std::monostate,
                         ExtractorView<dealii::MemorySpace::Host>,
                         ExtractorView<dealii::MemorySpace::Default>>
        view_;
  };


  template <typename HyperbolicDescription,
            int dim,
            typename Number,
            typename MemorySpace>
  class SelectedComponentsExtractorView
  {
  public:
    static_assert(std::is_same_v<MemorySpace, dealii::MemorySpace::Host> ||
                      std::is_same_v<MemorySpace, dealii::MemorySpace::Default>,
                  "Unexpected memory space");

    using Extractor =
        SelectedComponentsExtractor<HyperbolicDescription, dim, Number>;

    using ScalarVector =
        dealii::LinearAlgebra::distributed::Vector<Number, MemorySpace>;

    std::vector<ScalarVector> extract() const;

    DEAL_II_HOST_DEVICE_ALWAYS_INLINE unsigned int n_selected() const
    {
      return static_cast<unsigned int>(entries_.extent(0));
    }

    template <typename T = Number, typename Writer>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE void
    extract_element(unsigned int i, const Writer &write) const;

  private:
    using HyperbolicSystem = typename HyperbolicDescription::HyperbolicSystem;
    using EquationView = typename HyperbolicSystem::template View<dim, Number>;
    using StateVector = typename EquationView::StateVector;

    template <int n_comp>
    using VectorView =
        decltype(std::declval<const Vectors::MultiComponentVector<Number> &>()
                     .template view<n_comp, MemorySpace>());

    using HyperbolicVectorView = VectorView<EquationView::problem_dimension>;
    using PrecomputedVectorView =
        VectorView<EquationView::n_precomputed_values>;
    using InitialPrecomputedVectorView =
        VectorView<EquationView::n_initial_precomputed_values>;
    using ScalarVectorView = VectorView<1>;

    struct Entry {
      unsigned int offset;
      ScalarVectorView scalar_view;
    };

    SelectedComponentsExtractorView(
        const OfflineData<dim, Number> &offline_data,
        const HyperbolicSystem &hyperbolic_system);

    const OfflineData<dim, Number> *offline_data_;

    bool read_conserved_ = false;
    bool read_primitive_ = false;
    bool read_precomputed_ = false;
    bool read_initial_ = false;

    SelectView<dim, Number, MemorySpace, HyperbolicSystem> system_views_;

    HyperbolicVectorView U_view_;
    PrecomputedVectorView precomputed_view_;
    InitialPrecomputedVectorView initial_view_;

    Kokkos::View<const Entry *, typename MemorySpace::kokkos_space> entries_;

    friend class SelectedComponentsExtractor<HyperbolicDescription,
                                             dim,
                                             Number>;
  };

} // namespace ryujin

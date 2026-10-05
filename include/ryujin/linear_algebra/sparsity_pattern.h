//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/compile_time_options.h>
#include <ryujin/base/convenience_macros.h>

#include <ryujin/base/gpu.h>

#include <deal.II/base/aligned_vector.h>
#include <deal.II/base/config.h>
#include <deal.II/base/partitioner.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>

namespace ryujin
{
  template <int warp_size, typename MemorySpace = dealii::MemorySpace::Host>
  class SparsityPatternView;


  template <int warp_size>
  class SparsityPattern : public MirroredStorage<SparsityPattern<warp_size>>
  {
  public:
    struct ExchangeDescription {
      unsigned int row;
      unsigned int column_index;
    };

    SparsityPattern();

    SparsityPattern(
        const unsigned int n_internal_dofs,
        const dealii::DynamicSparsityPattern &sparsity,
        const std::shared_ptr<const dealii::Utilities::MPI::Partitioner>
            &partitioner,
        bool symmetrize_ghost_range = true,
        const TransferPolicy transfer_policy =
            TransferPolicy::explicit_transfers);

    void reinit(const unsigned int n_internal_dofs,
                const dealii::DynamicSparsityPattern &sparsity,
                const std::shared_ptr<const dealii::Utilities::MPI::Partitioner>
                    &partitioner,
                bool symmetrize_ghost_range = true,
                const TransferPolicy transfer_policy =
                    TransferPolicy::explicit_transfers);

    template <typename MemorySpace = dealii::MemorySpace::Host>
    SparsityPatternView<warp_size, MemorySpace> view() const;

    ACCESSOR_READ_ONLY_NO_DEREFERENCE(partitioner);

    ACCESSOR_READ_ONLY(entries_to_be_sent);

    ACCESSOR_READ_ONLY(send_targets);

    ACCESSOR_READ_ONLY(receive_targets);

  private:
    std::shared_ptr<const dealii::Utilities::MPI::Partitioner> partitioner_;

    unsigned int n_internal_dofs_;
    unsigned int n_locally_owned_dofs_;

    using KokkosHost = dealii::MemorySpace::Host::kokkos_space;
    mutable Kokkos::View<unsigned int *, KokkosHost> row_starts_host_;
    mutable Kokkos::View<unsigned int *, KokkosHost> column_indices_host_;
    mutable Kokkos::View<unsigned int *, KokkosHost> indices_transposed_host_;

    using KokkosDefault = dealii::MemorySpace::Default::kokkos_space;
    mutable Kokkos::View<unsigned int *, KokkosDefault> row_starts_default_;
    mutable Kokkos::View<unsigned int *, KokkosDefault> column_indices_default_;
    mutable Kokkos::View<unsigned int *, KokkosDefault>
        indices_transposed_default_;

    Mirrored<ExchangeDescription *> entries_to_be_sent_{
        "sparsity_pattern_entries_to_be_sent"};

    std::vector<std::pair<unsigned int, unsigned int>> send_targets_;
    std::vector<std::pair<unsigned int, unsigned int>> receive_targets_;

    template <typename MemorySpace>
    void allocate_storage() const;

    template <typename To, typename From>
    void deep_copy_storage() const;

    template <typename MemorySpace>
    void deallocate_storage();


    friend class MirroredStorage<SparsityPattern<warp_size>>;

    template <int, typename>
    friend class SparsityPatternView;
  };


  template <int warp_size, typename MemorySpace>
  class SparsityPatternView
  {
  public:
    SparsityPatternView() = default;

    SparsityPatternView(const SparsityPattern<warp_size> &sparsity_pattern);

    void reinit(const SparsityPattern<warp_size> &sparsity_pattern);

    DEAL_II_HOST_DEVICE
    unsigned int n_internal_dofs() const;

    DEAL_II_HOST_DEVICE
    unsigned int n_locally_owned_dofs() const;

    DEAL_II_HOST_DEVICE
    unsigned int n_rows() const;

    DEAL_II_HOST_DEVICE
    unsigned int n_nonzero_elements() const;

    DEAL_II_HOST_DEVICE
    unsigned int stride_of_row(const unsigned int row) const;

    DEAL_II_HOST_DEVICE
    const unsigned int *columns(const unsigned int row) const;

    DEAL_II_HOST_DEVICE
    unsigned int row_length(const unsigned int row) const;

    DEAL_II_HOST_DEVICE
    unsigned int column_index(const unsigned int row,
                              const unsigned int column) const;

    template <unsigned int n_components = 1>
    DEAL_II_HOST_DEVICE unsigned int
    offset(const unsigned int row,
           const unsigned int column_index,
           const unsigned int component = 0) const;

    template <unsigned int n_components = 1>
    DEAL_II_HOST_DEVICE unsigned int
    offset_internal(const unsigned int row,
                    const unsigned int column_index) const;

    template <unsigned int n_components = 1>
    DEAL_II_HOST_DEVICE unsigned int
    transposed_offset(const unsigned int row,
                      const unsigned int column_index,
                      const unsigned int component = 0) const;

    template <unsigned int n_components = 1>
    DEAL_II_HOST_DEVICE const unsigned int *
    transposed_offset_internal(const unsigned int row,
                               const unsigned int column_index) const;

    template <unsigned int n_components = 1>
    DEAL_II_HOST_DEVICE unsigned int ghost_offset() const;

  private:
    unsigned int n_internal_dofs_;
    unsigned int n_locally_owned_dofs_;

    using KokkosSpace = typename MemorySpace::kokkos_space;
    Kokkos::View<const unsigned int *, KokkosSpace> row_starts_;
    Kokkos::View<const unsigned int *, KokkosSpace> column_indices_;
    Kokkos::View<const unsigned int *, KokkosSpace> indices_transposed_;
  };


#ifndef DOXYGEN


  template <int warp_size>
  template <typename MemorySpace>
  SparsityPatternView<warp_size, MemorySpace>
  SparsityPattern<warp_size>::view() const
  {
    this->template prepare_read_access<MemorySpace>();

    return SparsityPatternView<warp_size, MemorySpace>(*this);
  }


  template <int warp_size>
  template <typename MemorySpace>
  void SparsityPattern<warp_size>::allocate_storage() const
  {
    using HostSpace = dealii::MemorySpace::Host;
    using Aligned = Kokkos::MemoryTraits<Kokkos::Aligned>;

    if constexpr (std::is_same_v<MemorySpace, HostSpace>) {
      row_starts_host_ = Kokkos::View<unsigned int *, KokkosHost, Aligned>(
          Kokkos::view_alloc(Kokkos::WithoutInitializing,
                             "sparsity_pattern_row_starts"),
          row_starts_default_.extent(0));

      column_indices_host_ = Kokkos::View<unsigned int *, KokkosHost, Aligned>(
          Kokkos::view_alloc(Kokkos::WithoutInitializing,
                             "sparsity_pattern_column_indices"),
          column_indices_default_.extent(0));

      indices_transposed_host_ =
          Kokkos::View<unsigned int *, KokkosHost, Aligned>(
              Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                 "sparsity_pattern_indices_transposed"),
              indices_transposed_default_.extent(0));

    } else {
      row_starts_default_ = Kokkos::View<unsigned int *, KokkosDefault>(
          Kokkos::view_alloc(Kokkos::WithoutInitializing,
                             "sparsity_pattern_row_starts"),
          row_starts_host_.extent(0));

      column_indices_default_ = Kokkos::View<unsigned int *, KokkosDefault>(
          Kokkos::view_alloc(Kokkos::WithoutInitializing,
                             "sparsity_pattern_column_indices"),
          column_indices_host_.extent(0));

      indices_transposed_default_ = Kokkos::View<unsigned int *, KokkosDefault>(
          Kokkos::view_alloc(Kokkos::WithoutInitializing,
                             "sparsity_pattern_indices_transposed"),
          indices_transposed_host_.extent(0));
    }
  }


  template <int warp_size>
  template <typename To, typename From>
  void SparsityPattern<warp_size>::deep_copy_storage() const
  {
    using HostSpace = dealii::MemorySpace::Host;

    if constexpr (std::is_same_v<To, HostSpace>) {
      Kokkos::deep_copy(row_starts_host_, row_starts_default_);
      Kokkos::deep_copy(column_indices_host_, column_indices_default_);
      Kokkos::deep_copy(indices_transposed_host_, indices_transposed_default_);
    } else {
      Kokkos::deep_copy(row_starts_default_, row_starts_host_);
      Kokkos::deep_copy(column_indices_default_, column_indices_host_);
      Kokkos::deep_copy(indices_transposed_default_, indices_transposed_host_);
    }
  }


  template <int warp_size>
  template <typename MemorySpace>
  void SparsityPattern<warp_size>::deallocate_storage()
  {
    using HostSpace = dealii::MemorySpace::Host;

    if constexpr (std::is_same_v<MemorySpace, HostSpace>) {
      row_starts_host_ = {};
      column_indices_host_ = {};
      indices_transposed_host_ = {};

    } else {
      row_starts_default_ = {};
      column_indices_default_ = {};
      indices_transposed_default_ = {};
    }
  }


  template <int warp_size, typename MemorySpace>
  SparsityPatternView<warp_size, MemorySpace>::SparsityPatternView(
      const SparsityPattern<warp_size> &sparsity_pattern)
  {
    reinit(sparsity_pattern);
  }


  template <int warp_size, typename MemorySpace>
  void SparsityPatternView<warp_size, MemorySpace>::reinit(
      const SparsityPattern<warp_size> &sparsity_pattern)
  {
    n_internal_dofs_ = sparsity_pattern.n_internal_dofs_;
    n_locally_owned_dofs_ = sparsity_pattern.n_locally_owned_dofs_;

    using HostSpace = dealii::MemorySpace::Host;
    using DefaultSpace = dealii::MemorySpace::Default;

    static_assert(std::is_same_v<MemorySpace, HostSpace> ||
                      std::is_same_v<MemorySpace, DefaultSpace>,
                  "Unexpected memory space");

    if constexpr (have_separate_memory_spaces &&
                  !std::is_same_v<MemorySpace, HostSpace>) {
      row_starts_ = sparsity_pattern.row_starts_default_;
      column_indices_ = sparsity_pattern.column_indices_default_;
      indices_transposed_ = sparsity_pattern.indices_transposed_default_;
    } else {
      row_starts_ = sparsity_pattern.row_starts_host_;
      column_indices_ = sparsity_pattern.column_indices_host_;
      indices_transposed_ = sparsity_pattern.indices_transposed_host_;
    }
  }


  template <int warp_size, typename MemorySpace>
  DEAL_II_HOST_DEVICE_ALWAYS_INLINE unsigned int
  SparsityPatternView<warp_size, MemorySpace>::n_internal_dofs() const
  {
    return n_internal_dofs_;
  }


  template <int warp_size, typename MemorySpace>
  DEAL_II_HOST_DEVICE_ALWAYS_INLINE unsigned int
  SparsityPatternView<warp_size, MemorySpace>::n_locally_owned_dofs() const
  {
    return n_locally_owned_dofs_;
  }


  template <int warp_size, typename MemorySpace>
  DEAL_II_HOST_DEVICE_ALWAYS_INLINE unsigned int
  SparsityPatternView<warp_size, MemorySpace>::n_rows() const
  {
    Assert(row_starts_.size() > 0, dealii::ExcNotInitialized());

    return row_starts_.size() - 1;
  }


  template <int warp_size, typename MemorySpace>
  DEAL_II_HOST_DEVICE_ALWAYS_INLINE unsigned int
  SparsityPatternView<warp_size, MemorySpace>::n_nonzero_elements() const
  {
    Assert(row_starts_.size() > 0, dealii::ExcNotInitialized());

    return row_starts_(row_starts_.size() - 1);
  }


  template <int warp_size, typename MemorySpace>
  DEAL_II_HOST_DEVICE_ALWAYS_INLINE unsigned int
  SparsityPatternView<warp_size, MemorySpace>::stride_of_row(
      const unsigned int row) const
  {
    AssertIndexRange(row, n_rows());

    if (row < n_internal_dofs_)
      return warp_size;
    else
      return 1;
  }


  template <int warp_size, typename MemorySpace>
  DEAL_II_HOST_DEVICE_ALWAYS_INLINE const unsigned int *
  SparsityPatternView<warp_size, MemorySpace>::columns(
      const unsigned int row) const
  {
    AssertIndexRange(row, n_rows());

    if (row < n_internal_dofs_)
      return column_indices_.data() + row_starts_(row / warp_size) +
             row % warp_size;
    else
      return column_indices_.data() + row_starts_(row);
  }


  template <int warp_size, typename MemorySpace>
  DEAL_II_HOST_DEVICE_ALWAYS_INLINE unsigned int
  SparsityPatternView<warp_size, MemorySpace>::row_length(
      const unsigned int row) const
  {
    AssertIndexRange(row, n_rows());

    if (row < n_internal_dofs_) {
      const unsigned int warp = row / warp_size;
      return (row_starts_(warp + 1) - row_starts_(warp)) / warp_size;
    } else {
      return row_starts_(row + 1) - row_starts_(row);
    }
  }


  template <int warp_size, typename MemorySpace>
  DEAL_II_HOST_DEVICE_ALWAYS_INLINE unsigned int
  SparsityPatternView<warp_size, MemorySpace>::column_index(
      const unsigned int row, const unsigned int column) const
  {
    const auto &row_length = this->row_length(row);
    const auto &stride_size = this->stride_of_row(row);

    const unsigned int *js = columns(row);
    for (unsigned int k = 0; k < row_length; ++k)
      if (js[k * stride_size] == column)
        return k;

    Assert(false, dealii::ExcMessage("Column index not found in given row"));
    return -1;
  }


  template <int warp_size, typename MemorySpace>
  template <unsigned int n_components>
  DEAL_II_HOST_DEVICE_ALWAYS_INLINE unsigned int
  SparsityPatternView<warp_size, MemorySpace>::offset(
      const unsigned int row,
      const unsigned int column_index,
      const unsigned int comp) const
  {
    AssertIndexRange(row, n_rows());
    AssertIndexRange(column_index, row_length(row));
    AssertIndexRange(comp, n_components);

    const unsigned int warp = row / warp_size;
    const unsigned int lane = row % warp_size;

    if (row < n_internal_dofs_) {
      const unsigned int scalar_offset =
          row_starts_(warp) + column_index * warp_size;
      return scalar_offset * n_components + comp * warp_size + lane;

    } else {
      const unsigned int scalar_offset = row_starts_(row) + column_index;

      return scalar_offset * n_components + comp;
    }
  }


  template <int warp_size, typename MemorySpace>
  template <unsigned int n_components>
  DEAL_II_HOST_DEVICE_ALWAYS_INLINE unsigned int
  SparsityPatternView<warp_size, MemorySpace>::transposed_offset(
      const unsigned int row,
      const unsigned int column_index,
      const unsigned int component) const
  {
    AssertIndexRange(row, n_rows());
    AssertIndexRange(column_index, row_length(row));
    AssertIndexRange(component, n_components);

    const unsigned int scalar_offset = offset(row, column_index);
    const unsigned int transposed_scalar_offset =
        indices_transposed_(scalar_offset);

    const unsigned int j = column_indices_(scalar_offset);

    unsigned int transposed_offset = transposed_scalar_offset;
    if constexpr (n_components > 1) {
      if (j < n_internal_dofs_) {
        transposed_offset =
            transposed_offset / warp_size * warp_size * n_components +
            transposed_offset % warp_size;
        return transposed_offset + component * warp_size;

      } else {

        transposed_offset *= n_components;
        return transposed_offset + component;
      }

    } else {

      return transposed_offset;
    }
  }


  template <int warp_size, typename MemorySpace>
  template <unsigned int n_components>
  DEAL_II_HOST_DEVICE_ALWAYS_INLINE unsigned int
  SparsityPatternView<warp_size, MemorySpace>::offset_internal(
      const unsigned int row, const unsigned int column_index) const
  {
    AssertIndexRange(row, n_rows());
    AssertIndexRange(column_index, row_length(row));
    AssertIndexRange(row, n_internal_dofs_);

    const unsigned int warp = row / warp_size;
    const unsigned int lane = row % warp_size;

    const unsigned int scalar_offset =
        row_starts_(warp) + column_index * warp_size;

    return scalar_offset * n_components + lane;
  }


  template <int warp_size, typename MemorySpace>
  template <unsigned int n_components>
  DEAL_II_HOST_DEVICE_ALWAYS_INLINE const unsigned int *
  SparsityPatternView<warp_size, MemorySpace>::transposed_offset_internal(
      const unsigned int row, const unsigned int column_index) const
  {
    static_assert(n_components == 1,
                  "Vectorized transposed access to multiple components is not "
                  "yet implemented.");
    AssertIndexRange(row, row_starts_.size() - 1);
    AssertIndexRange(column_index, row_length(row));
    AssertIndexRange(row, n_internal_dofs_);

    const unsigned int warp = row / warp_size;
    const unsigned int lane = row % warp_size;

    const unsigned int scalar_offset =
        row_starts_(warp) + column_index * warp_size;

    return indices_transposed_.data() + scalar_offset + lane;
  }


  template <int warp_size, typename MemorySpace>
  template <unsigned int n_components>
  DEAL_II_HOST_DEVICE_ALWAYS_INLINE unsigned int
  SparsityPatternView<warp_size, MemorySpace>::ghost_offset() const
  {
    const auto scalar_offset = row_starts_(n_locally_owned_dofs_);
    return scalar_offset * n_components;
  }


#endif
} // namespace ryujin

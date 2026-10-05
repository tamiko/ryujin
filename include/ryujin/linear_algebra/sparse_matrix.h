//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/compile_time_options.h>

#include <ryujin/base/gpu.h>
#include <ryujin/linear_algebra/sparsity_pattern.h>

#include <deal.II/base/exceptions.h>
#include <deal.II/base/partitioner.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>

#include <type_traits>

#ifdef DEBUG_MPI_EXCHANGE
#include <chrono>
#endif

namespace ryujin
{
  template <typename Number,
            int n_comp,
            int warp_size,
            int simd_length,
            typename MemorySpace = dealii::MemorySpace::Host,
            bool writable = true>
  class SparseMatrixView;

  template <typename Number,
            int n_comp = 1,
            int warp_size = ryujin::warp_size,
            int simd_length = dealii::VectorizedArray<Number>::size()>
  class SparseMatrix : public MirroredStorage<
                           SparseMatrix<Number, n_comp, warp_size, simd_length>>
  {
  public:
    static_assert(warp_size % simd_length == 0,
                  "The warp size must be an integer multiple of the SIMD "
                  "length");

    SparseMatrix() = default;

    SparseMatrix(const SparsityPattern<warp_size> &sparsity,
                 const TransferPolicy transfer_policy =
                     TransferPolicy::explicit_transfers);

    void reinit(const SparsityPattern<warp_size> &sparsity,
                const TransferPolicy transfer_policy =
                    TransferPolicy::explicit_transfers);

    ACCESSOR_READ_ONLY(sparsity_pattern);

    template <typename MemorySpace = dealii::MemorySpace::Host>
    SparseMatrixView<Number, n_comp, warp_size, simd_length, MemorySpace, true>
    view();

    template <typename MemorySpace = dealii::MemorySpace::Host>
    SparseMatrixView<Number, n_comp, warp_size, simd_length, MemorySpace, false>
    view() const;

    template <typename MemorySpace>
    void zero_out_ghost_rows_on_memory_space();

    template <typename MemorySpace>
    void update_ghost_rows_on_memory_space();

    template <typename MemorySpace>
    void compress_on_memory_space(dealii::VectorOperation::values operation);

    template <typename MemorySpace>
    void populate_exchange_buffer_on_memory_space();

  private:
    const SparsityPattern<warp_size> *sparsity_pattern_ = nullptr;

    using KokkosHost = dealii::MemorySpace::Host::kokkos_space;
    mutable Kokkos::View<Number *, KokkosHost> data_host_;

    using KokkosDefault = dealii::MemorySpace::Default::kokkos_space;
    mutable Kokkos::View<Number *, KokkosDefault> data_default_;

    mutable Kokkos::View<Number *, KokkosHost> exchange_buffer_host_;
    mutable Kokkos::View<Number *, KokkosHost> ghost_buffer_host_;
    mutable Kokkos::View<Number *, KokkosDefault> exchange_buffer_default_;

    template <typename MemorySpace>
    void allocate_storage() const;

    template <typename To, typename From>
    void deep_copy_storage() const;

    template <typename MemorySpace>
    void deallocate_storage();


    friend class MirroredStorage<
        SparseMatrix<Number, n_comp, warp_size, simd_length>>;

    template <typename, int, int, int, typename, bool>
    friend class SparseMatrixView;
  };


  template <typename Number,
            int n_comp,
            int warp_size,
            int simd_length,
            typename MemorySpace,
            bool writable>
  class SparseMatrixView
  {
  public:
    static_assert(warp_size % simd_length == 0,
                  "The warp size must be an integer multiple of the SIMD "
                  "length");

    SparseMatrixView() = default;

    SparseMatrixView(
        SparseMatrix<Number, n_comp, warp_size, simd_length> &sparse_matrix)
      requires(writable);

    SparseMatrixView(const SparseMatrix<Number, n_comp, warp_size, simd_length>
                         &sparse_matrix)
      requires(!writable);

    template <bool other_writable>
    DEAL_II_HOST_DEVICE
    SparseMatrixView(const SparseMatrixView<Number,
                                            n_comp,
                                            warp_size,
                                            simd_length,
                                            MemorySpace,
                                            other_writable> &other)
      requires(!writable && other_writable);

    template <typename SparseMatrix>
    void reinit(SparseMatrix &sparse_matrix)
      requires(writable != std::is_const_v<SparseMatrix>);

    ACCESSOR_READ_ONLY(sparsity_pattern);

    template <typename Number2 = Number>
    DEAL_II_HOST_DEVICE Number2
    read_entry(const unsigned int row, const unsigned int column_index) const;

    template <typename Number2 = Number,
              typename Tensor = dealii::Tensor<1, n_comp, Number2>>
    DEAL_II_HOST_DEVICE Tensor
    read_tensor(const unsigned int row, const unsigned int column_index) const;

    template <typename Number2 = Number>
    DEAL_II_HOST_DEVICE Number2 read_transposed_entry(
        const unsigned int row, const unsigned int column_index) const;

    template <typename Number2 = Number,
              typename Tensor = dealii::Tensor<1, n_comp, Number2>>
    DEAL_II_HOST_DEVICE Tensor read_transposed_tensor(
        const unsigned int row, const unsigned int column_index) const;

    template <typename Number2 = Number>
    DEAL_II_HOST_DEVICE void
    write_entry(const Number2 entry,
                const unsigned int row,
                const unsigned int column_index,
                const bool do_streaming_store = false) const
      requires(writable);

    template <typename Number2 = Number,
              typename Tensor = dealii::Tensor<1, n_comp, Number2>>
    DEAL_II_HOST_DEVICE void
    write_tensor(const Tensor &tensor,
                 const unsigned int row,
                 const unsigned int column_index,
                 const bool do_streaming_store = false) const
      requires(writable);

    template <typename Number2 = Number>
    DEAL_II_HOST_DEVICE void add_entry(const Number2 entry,
                                       const unsigned int row,
                                       const unsigned int column_index) const
      requires(writable);

    template <typename Number2 = Number,
              typename Tensor = dealii::Tensor<1, n_comp, Number2>>
    DEAL_II_HOST_DEVICE void add_tensor(const Tensor &tensor,
                                        const unsigned int row,
                                        const unsigned int column_index) const
      requires(writable);

    void zero_out_ghost_rows() const
      requires(writable);

    void update_ghost_rows() const
      requires(writable);

    void compress(dealii::VectorOperation::values operation) const
      requires(writable);

  private:
    using SM = SparseMatrix<Number, n_comp, warp_size, simd_length>;
    std::conditional_t<writable, SM *, const SM *> sparse_matrix_;

    SparsityPatternView<warp_size, MemorySpace> sparsity_pattern_;

    using KokkosSpace = typename MemorySpace::kokkos_space;
    Kokkos::View<Number *, KokkosSpace> data_;

    template <typename, int, int, int, typename, bool>
    friend class SparseMatrixView;
  };


  template <typename Number,
            int n_comp,
            int warp_size,
            int simd_length,
            typename FullMatrix>
  void distribute_local_to_global(
      const FullMatrix &cell_matrix,
      const std::vector<dealii::types::global_dof_index> &dof_indices_row,
      const std::vector<dealii::types::global_dof_index> &dof_indices_column,
      const dealii::AffineConstraints<Number> &affine_constraints,
      SparseMatrix<Number, n_comp, warp_size, simd_length> &sparse_matrix);


  template <typename Number,
            int n_comp,
            int warp_size,
            int simd_length,
            typename FullMatrix,
            typename Vector>
  void distribute_local_to_global(
      const FullMatrix &cell_matrix,
      const std::vector<dealii::types::global_dof_index> &dof_indices,
      const dealii::AffineConstraints<Number> &affine_constraints,
      SparseMatrix<Number, n_comp, warp_size, simd_length> &sparse_matrix);


#ifndef DOXYGEN


  template <typename Number, int n_components, int warp_size, int simd_length>
  SparseMatrix<Number, n_components, warp_size, simd_length>::SparseMatrix(
      const SparsityPattern<warp_size> &sparsity,
      const TransferPolicy transfer_policy)
  {
    reinit(sparsity, transfer_policy);
  }


  template <typename Number, int n_components, int warp_size, int simd_length>
  void SparseMatrix<Number, n_components, warp_size, simd_length>::reinit(
      const SparsityPattern<warp_size> &sparsity,
      const TransferPolicy transfer_policy)
  {
    this->set_transfer_policy(TransferPolicy::explicit_transfers);

    this->sparsity_pattern_ = &sparsity;

    const auto sparsity_view = sparsity.view();

    using KokkosHost = dealii::MemorySpace::Host::kokkos_space;
    using Aligned = Kokkos::MemoryTraits<Kokkos::Aligned>;

    data_host_ = Kokkos::View<Number *, KokkosHost, Aligned>(
        "sparse_matrix_data",
        sparsity_view.n_nonzero_elements() * n_components);

    const std::size_t n_indices = sparsity.entries_to_be_sent().size();

    exchange_buffer_host_ = Kokkos::View<Number *, KokkosHost, Aligned>(
        "sparse_matrix_exchange_buffer", n_components * n_indices);

    if constexpr (have_separate_memory_spaces) {
      const auto ghost_offset =
          sparsity_view.template ghost_offset<n_components>();
      const auto end_offset = sparsity_view.n_nonzero_elements() * n_components;

      ghost_buffer_host_ = Kokkos::View<Number *, KokkosHost, Aligned>(
          "sparse_matrix_ghost_buffer", end_offset - ghost_offset);
    }

    data_default_ = {};
    exchange_buffer_default_ = {};

    this->reset_residency(true, !have_separate_memory_spaces);

    this->set_transfer_policy(transfer_policy);
  }


  template <typename Number, int n_comp, int warp_size, int simd_length>
  template <typename MemorySpace>
  SparseMatrixView<Number, n_comp, warp_size, simd_length, MemorySpace, true>
  SparseMatrix<Number, n_comp, warp_size, simd_length>::view()
  {
    this->template prepare_write_access<MemorySpace>();

    return SparseMatrixView<Number,
                            n_comp,
                            warp_size,
                            simd_length,
                            MemorySpace,
                            true>(*this);
  }


  template <typename Number, int n_comp, int warp_size, int simd_length>
  template <typename MemorySpace>
  SparseMatrixView<Number, n_comp, warp_size, simd_length, MemorySpace, false>
  SparseMatrix<Number, n_comp, warp_size, simd_length>::view() const
  {
    this->template prepare_read_access<MemorySpace>();

    return SparseMatrixView<Number,
                            n_comp,
                            warp_size,
                            simd_length,
                            MemorySpace,
                            false>(*this);
  }


  template <typename Number, int n_components, int warp_size, int simd_length>
  template <typename MemorySpace>
  void
  SparseMatrix<Number, n_components, warp_size, simd_length>::allocate_storage()
      const
  {
    using HostSpace = dealii::MemorySpace::Host;
    using Aligned = Kokkos::MemoryTraits<Kokkos::Aligned>;

    Assert(sparsity_pattern_ != nullptr, dealii::ExcNotInitialized());

    const auto sparsity_view = sparsity_pattern_->view();

    const std::size_t n_data =
        sparsity_view.n_nonzero_elements() * n_components;
    const std::size_t n_exchange =
        n_components * sparsity_pattern_->entries_to_be_sent().size();

    if constexpr (std::is_same_v<MemorySpace, HostSpace>) {
      data_host_ = Kokkos::View<Number *, KokkosHost, Aligned>(
          Kokkos::view_alloc(Kokkos::WithoutInitializing, "sparse_matrix_data"),
          n_data);

    } else {
      data_default_ = Kokkos::View<Number *, KokkosDefault>(
          Kokkos::view_alloc(Kokkos::WithoutInitializing, "sparse_matrix_data"),
          n_data);

      exchange_buffer_default_ = Kokkos::View<Number *, KokkosDefault>(
          Kokkos::view_alloc(Kokkos::WithoutInitializing,
                             "sparse_matrix_exchange_buffer"),
          n_exchange);
    }
  }


  template <typename Number, int n_components, int warp_size, int simd_length>
  template <typename To, typename From>
  void SparseMatrix<Number, n_components, warp_size, simd_length>::
      deep_copy_storage() const
  {
    using HostSpace = dealii::MemorySpace::Host;

    if constexpr (std::is_same_v<To, HostSpace>) {
      Kokkos::deep_copy(data_host_, data_default_);
    } else {
      Kokkos::deep_copy(data_default_, data_host_);
    }
  }


  template <typename Number, int n_components, int warp_size, int simd_length>
  template <typename MemorySpace>
  void SparseMatrix<Number, n_components, warp_size, simd_length>::
      deallocate_storage()
  {
    using HostSpace = dealii::MemorySpace::Host;

    if constexpr (std::is_same_v<MemorySpace, HostSpace>) {
      data_host_ = {};

    } else {
      data_default_ = {};
      exchange_buffer_default_ = {};
    }
  }


  template <typename Number, int n_components, int warp_size, int simd_length>
  template <typename MemorySpace>
  void SparseMatrix<Number, n_components, warp_size, simd_length>::
      zero_out_ghost_rows_on_memory_space()
  {
    using HostSpace = dealii::MemorySpace::Host;
    using DefaultSpace = dealii::MemorySpace::Default;

    Assert(this->template is_resident<MemorySpace>(),
           dealii::ExcMessage("The chosen memory space is not resident."));

    AssertThrow((std::is_same_v<MemorySpace, HostSpace>),
                dealii::ExcNotImplemented());

    const auto sparsity_view = sparsity_pattern_->view();

    const auto ghost_offset =
        sparsity_view.template ghost_offset<n_components>();
    const auto end_offset = sparsity_view.n_nonzero_elements() * n_components;
    std::fill(data_host_.data() + ghost_offset,
              data_host_.data() + end_offset,
              Number{});
  }


  template <typename Number, int n_components, int warp_size, int simd_length>
  template <typename MemorySpace>
  void SparseMatrix<Number, n_components, warp_size, simd_length>::
      populate_exchange_buffer_on_memory_space()
  {
    using HostSpace = dealii::MemorySpace::Host;

    const auto sparsity_view = sparsity_pattern_->template view<MemorySpace>();

    const auto *entries_to_be_sent =
        sparsity_pattern_->entries_to_be_sent().template view<MemorySpace>();
    const std::size_t n_entries_to_be_sent =
        sparsity_pattern_->entries_to_be_sent().size();

    constexpr bool on_host =
        !have_separate_memory_spaces || std::is_same_v<MemorySpace, HostSpace>;

    const auto data = [&]() {
      if constexpr (on_host)
        return data_host_;
      else
        return data_default_;
    }();

    const auto exchange_buffer = [&]() {
      if constexpr (on_host)
        return exchange_buffer_host_;
      else
        return exchange_buffer_default_;
    }();

    using ExecutionSpace = typename MemorySpace::kokkos_space::execution_space;
    using Policy =
        Kokkos::RangePolicy<ExecutionSpace, Kokkos::IndexType<std::size_t>>;

    const auto exec = ExecutionSpace{};

    Kokkos::parallel_for(
        "sparse_matrix_populate_exchange_buffer",
        Policy(exec, 0, n_entries_to_be_sent),
        KOKKOS_LAMBDA(const std::size_t c) {
          const auto &[row, column_index] = entries_to_be_sent[c];
          for (unsigned int d = 0; d < n_components; ++d) {
            const auto offset = sparsity_view.template offset<n_components>(
                row, column_index, d);
            exchange_buffer(n_components * c + d) = data(offset);
          }
        });

    exec.fence();
  }


  template <typename Number, int n_components, int warp_size, int simd_length>
  template <typename MemorySpace>
  void SparseMatrix<Number, n_components, warp_size, simd_length>::
      update_ghost_rows_on_memory_space()
  {
    using HostSpace = dealii::MemorySpace::Host;

    Assert(this->template is_resident<MemorySpace>(),
           dealii::ExcMessage("The chosen memory space is not resident."));

    constexpr bool on_host =
        !have_separate_memory_spaces || std::is_same_v<MemorySpace, HostSpace>;

    const auto sparsity_view = sparsity_pattern_->view();

    const auto &receive_targets = sparsity_pattern_->receive_targets();
    const auto &send_targets = sparsity_pattern_->send_targets();

    const unsigned int mpi_tag =
        dealii::Utilities::MPI::internal::Tags::partitioner_export_start + 0;
    Assert(mpi_tag <=
               dealii::Utilities::MPI::internal::Tags::partitioner_export_end,
           dealii::ExcInternalError());

    const unsigned int n_requests =
        receive_targets.size() + send_targets.size();
    std::vector<MPI_Request> requests(n_requests);

    const auto ghost_offset =
        sparsity_view.template ghost_offset<n_components>();
    const auto end_offset = sparsity_view.n_nonzero_elements() * n_components;

    Number *const receive_pointer =
        on_host ? data_host_.data() + ghost_offset : ghost_buffer_host_.data();
    Number *const send_pointer = exchange_buffer_host_.data();

    for (unsigned int p = 0; p < receive_targets.size(); ++p) {
      const auto receive_offset =
          n_components * (p == 0 ? 0 : receive_targets[p - 1].second);
      const auto receive_size =
          (receive_targets[p].second * n_components - receive_offset);

#ifdef DEBUG_MPI_EXCHANGE
      const auto mpi_rank =
          dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
      std::cout << "Rank " << mpi_rank << " receive from "
                << receive_targets[p].first << " offset = " << receive_offset
                << " size = " << receive_size << std::endl;
#endif

      const int ierr =
          MPI_Irecv(receive_pointer + receive_offset,
                    receive_size,
                    dealii::Utilities::MPI::mpi_type_id_for_type<Number>,
                    receive_targets[p].first,
                    mpi_tag,
                    sparsity_pattern_->partitioner()->get_mpi_communicator(),
                    &requests[p]);
      AssertThrowMPI(ierr);
    }

    populate_exchange_buffer_on_memory_space<MemorySpace>();

    if constexpr (!on_host)
      Kokkos::deep_copy(exchange_buffer_host_, exchange_buffer_default_);

    for (unsigned int p = 0; p < send_targets.size(); ++p) {
      const auto send_offset =
          n_components * (p == 0 ? 0 : send_targets[p - 1].second);
      const auto send_size =
          (send_targets[p].second * n_components - send_offset);

#ifdef DEBUG_MPI_EXCHANGE
      const auto mpi_rank =
          dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
      std::cout << "Rank " << mpi_rank << " send to " << send_targets[p].first
                << " offset = " << send_offset << " size = " << send_size
                << std::endl;
#endif

      const int ierr =
          MPI_Isend(send_pointer + send_offset,
                    send_size,
                    dealii::Utilities::MPI::mpi_type_id_for_type<Number>,
                    send_targets[p].first,
                    mpi_tag,
                    sparsity_pattern_->partitioner()->get_mpi_communicator(),
                    &requests[receive_targets.size() + p]);
      AssertThrowMPI(ierr);
    }

#ifdef DEBUG_MPI_EXCHANGE
    using namespace std::chrono_literals;
    std::this_thread::sleep_for(200ms);
#endif

    const int ierr =
        MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
    AssertThrowMPI(ierr);

    if constexpr (!on_host) {
      const auto ghost_range =
          Kokkos::subview(data_default_,
                          Kokkos::make_pair(std::size_t(ghost_offset),
                                            std::size_t(end_offset)));
      Kokkos::deep_copy(ghost_range, ghost_buffer_host_);
    }
  }


  template <typename Number, int n_components, int warp_size, int simd_length>
  template <typename MemorySpace>
  void SparseMatrix<Number, n_components, warp_size, simd_length>::
      compress_on_memory_space(dealii::VectorOperation::values operation
                               [[maybe_unused]])
  {
    AssertThrow(operation == dealii::VectorOperation::add,
                dealii::ExcNotImplemented());

    using HostSpace = dealii::MemorySpace::Host;
    using DefaultSpace = dealii::MemorySpace::Default;

    AssertThrow((std::is_same_v<MemorySpace, HostSpace>),
                dealii::ExcNotImplemented());

    Assert(this->template is_resident<MemorySpace>(),
           dealii::ExcMessage("The chosen memory space is not resident."));

    const auto sparsity_view = sparsity_pattern_->view();

    const auto &receive_targets = sparsity_pattern_->receive_targets();
    const auto &send_targets = sparsity_pattern_->send_targets();
    const auto *entries_to_be_sent =
        sparsity_pattern_->entries_to_be_sent().view();
    const auto n_entries_to_be_sent =
        sparsity_pattern_->entries_to_be_sent().size();

    const unsigned int mpi_tag =
        dealii::Utilities::MPI::internal::Tags::partitioner_export_start + 0;
    Assert(mpi_tag <=
               dealii::Utilities::MPI::internal::Tags::partitioner_export_end,
           dealii::ExcInternalError());

    const unsigned int n_requests =
        receive_targets.size() + send_targets.size();
    std::vector<MPI_Request> requests(n_requests);

    for (unsigned int p = 0; p < send_targets.size(); ++p) {
      const auto receive_offset =
          n_components * (p == 0 ? 0 : send_targets[p - 1].second);
      const auto receive_size =
          (send_targets[p].second * n_components - receive_offset);

#ifdef DEBUG_MPI_EXCHANGE
      const auto mpi_rank =
          dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
      std::cout << "Rank " << mpi_rank << " receive from "
                << send_targets[p].first << " offset = " << receive_offset
                << " size = " << receive_size << std::endl;
#endif

      const int ierr =
          MPI_Irecv(exchange_buffer_host_.data() + receive_offset,
                    receive_size,
                    dealii::Utilities::MPI::mpi_type_id_for_type<Number>,
                    send_targets[p].first,
                    mpi_tag,
                    sparsity_pattern_->partitioner()->get_mpi_communicator(),
                    &requests[p]);
      AssertThrowMPI(ierr);
    }

    const auto ghost_offset =
        sparsity_view.template ghost_offset<n_components>();

    for (unsigned int p = 0; p < receive_targets.size(); ++p) {
      const auto send_offset =
          n_components * (p == 0 ? 0 : receive_targets[p - 1].second);
      const auto send_size =
          (receive_targets[p].second * n_components - send_offset);

#ifdef DEBUG_MPI_EXCHANGE
      const auto mpi_rank =
          dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
      std::cout << "Rank " << mpi_rank << " send to "
                << receive_targets[p].first << " offset = " << send_offset
                << " size = " << send_size << std::endl;
#endif

      const int ierr =
          MPI_Isend(data_host_.data() + ghost_offset + send_offset,
                    send_size,
                    dealii::Utilities::MPI::mpi_type_id_for_type<Number>,
                    receive_targets[p].first,
                    mpi_tag,
                    sparsity_pattern_->partitioner()->get_mpi_communicator(),
                    &requests[send_targets.size() + p]);
      AssertThrowMPI(ierr);
    }

#ifdef DEBUG_MPI_EXCHANGE
    using namespace std::chrono_literals;
    std::this_thread::sleep_for(200ms);
#endif

    const int ierr =
        MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
    AssertThrowMPI(ierr);

    for (std::size_t c = 0; c < n_entries_to_be_sent; ++c) {
      const auto &[row, column_index] = entries_to_be_sent[c];
      for (unsigned int d = 0; d < n_components; ++d) {
        const auto offset =
            sparsity_view.template offset<n_components>(row, column_index, d);
        data_host_(offset) += exchange_buffer_host_(n_components * c + d);
      }
    }

    zero_out_ghost_rows_on_memory_space<MemorySpace>();
  }


  template <typename Number,
            int n_comp,
            int warp_size,
            int simd_length,
            typename MemorySpace,
            bool writable>
  SparseMatrixView<Number,
                   n_comp,
                   warp_size,
                   simd_length,
                   MemorySpace,
                   writable>::
      SparseMatrixView(
          SparseMatrix<Number, n_comp, warp_size, simd_length> &sparse_matrix)
    requires(writable)
  {
    reinit(sparse_matrix);
  }


  template <typename Number,
            int n_comp,
            int warp_size,
            int simd_length,
            typename MemorySpace,
            bool writable>
  SparseMatrixView<Number,
                   n_comp,
                   warp_size,
                   simd_length,
                   MemorySpace,
                   writable>::
      SparseMatrixView(
          const SparseMatrix<Number, n_comp, warp_size, simd_length>
              &sparse_matrix)
    requires(!writable)
  {
    reinit(sparse_matrix);
  }


  template <typename Number,
            int n_comp,
            int warp_size,
            int simd_length,
            typename MemorySpace,
            bool writable>
  template <bool other_writable>
  DEAL_II_HOST_DEVICE SparseMatrixView<
      Number,
      n_comp,
      warp_size,
      simd_length,
      MemorySpace,
      writable>::SparseMatrixView(const SparseMatrixView<Number,
                                                         n_comp,
                                                         warp_size,
                                                         simd_length,
                                                         MemorySpace,
                                                         other_writable> &other)
    requires(!writable && other_writable)
      : sparse_matrix_(other.sparse_matrix_)
      , sparsity_pattern_(other.sparsity_pattern_)
      , data_(other.data_)
  {
  }


  template <typename Number,
            int n_comp,
            int warp_size,
            int simd_length,
            typename MemorySpace,
            bool writable>
  template <typename SparseMatrix>
  void SparseMatrixView<Number,
                        n_comp,
                        warp_size,
                        simd_length,
                        MemorySpace,
                        writable>::reinit(SparseMatrix &sparse_matrix)
    requires(writable != std::is_const_v<SparseMatrix>)
  {
    using HostSpace = dealii::MemorySpace::Host;
    using DefaultSpace = dealii::MemorySpace::Default;

    static_assert(std::is_same_v<MemorySpace, HostSpace> ||
                      std::is_same_v<MemorySpace, DefaultSpace>,
                  "Unexpected Kokkos memory space");

    sparse_matrix_ = &sparse_matrix;

    if constexpr (have_separate_memory_spaces &&
                  !std::is_same_v<MemorySpace, HostSpace>) {
      data_ = sparse_matrix.data_default_;
    } else {
      data_ = sparse_matrix.data_host_;
    }

    sparsity_pattern_ =
        sparse_matrix.sparsity_pattern_->template view<MemorySpace>();
  }


  template <typename Number,
            int n_comp,
            int warp_size,
            int simd_length,
            typename MemorySpace,
            bool writable>
  template <typename Number2>
  DEAL_II_HOST_DEVICE_ALWAYS_INLINE Number2
  SparseMatrixView<Number,
                   n_comp,
                   warp_size,
                   simd_length,
                   MemorySpace,
                   writable>::read_entry(const unsigned int row,
                                         const unsigned int column_index) const
  {
    static_assert(
        n_comp == 1,
        "Attempted to write a scalar value into a tensor-valued matrix entry");

    const auto result = read_tensor<Number2>(row, column_index);
    return result[0];
  }


  template <typename Number,
            int n_comp,
            int warp_size,
            int simd_length,
            typename MemorySpace,
            bool writable>
  template <typename Number2, typename Tensor>
  DEAL_II_HOST_DEVICE_ALWAYS_INLINE Tensor
  SparseMatrixView<Number,
                   n_comp,
                   warp_size,
                   simd_length,
                   MemorySpace,
                   writable>::read_tensor(const unsigned int row,
                                          const unsigned int column_index) const
  {
    static_assert(std::is_same_v<Number2, typename Tensor::value_type>,
                  "type mismatch");

    AssertIndexRange(row, sparsity_pattern_.n_rows());
    AssertIndexRange(column_index, sparsity_pattern_.row_length(row));

    Tensor result;

    using VA = dealii::VectorizedArray<Number, simd_length>;
    if constexpr (std::is_same_v<VA, Number2>) {
      Assert(row < sparsity_pattern_.n_internal_dofs(),
             dealii::ExcMessage(
                 "Vectorized access only possible in vectorized part"));
      Assert(row % simd_length == 0,
             dealii::ExcMessage(
                 "Access only supported for rows at the SIMD granularity"));

      const Number *load_pos = data_.data();
      load_pos +=
          sparsity_pattern_.template offset_internal<n_comp>(row, column_index);

      for (unsigned int d = 0; d < n_comp; ++d)
        result[d].load(load_pos + d * warp_size);

    } else {
      for (unsigned int d = 0; d < n_comp; ++d) {
        const auto offset =
            sparsity_pattern_.template offset<n_comp>(row, column_index, d);
        result[d] = data_(offset);
      }
    }

    return result;
  }


  template <typename Number,
            int n_comp,
            int warp_size,
            int simd_length,
            typename MemorySpace,
            bool writable>
  template <typename Number2>
  DEAL_II_HOST_DEVICE_ALWAYS_INLINE Number2
  SparseMatrixView<Number,
                   n_comp,
                   warp_size,
                   simd_length,
                   MemorySpace,
                   writable>::read_transposed_entry(const unsigned int row,
                                                    const unsigned int
                                                        column_index) const
  {
    static_assert(
        n_comp == 1,
        "Attempted to write a scalar value into a tensor-valued matrix entry");

    const auto result = read_transposed_tensor<Number2>(row, column_index);
    return result[0];
  }


  template <typename Number,
            int n_comp,
            int warp_size,
            int simd_length,
            typename MemorySpace,
            bool writable>
  template <typename Number2, typename Tensor>
  DEAL_II_HOST_DEVICE_ALWAYS_INLINE Tensor
  SparseMatrixView<Number,
                   n_comp,
                   warp_size,
                   simd_length,
                   MemorySpace,
                   writable>::read_transposed_tensor(const unsigned int row,
                                                     const unsigned int
                                                         column_index) const
  {
    static_assert(std::is_same_v<Number2, typename Tensor::value_type>,
                  "type mismatch");

    AssertIndexRange(row, sparsity_pattern_.n_rows());
    AssertIndexRange(column_index, sparsity_pattern_.row_length(row));

    dealii::Tensor<1, n_comp, Number2> result;

    using VA = dealii::VectorizedArray<Number, simd_length>;
    if constexpr (std::is_same_v<VA, Number2> && (n_comp == 1)) {
      Assert(row < sparsity_pattern_.n_internal_dofs(),
             dealii::ExcMessage(
                 "Vectorized access only possible in vectorized part"));
      Assert(row % simd_length == 0,
             dealii::ExcMessage(
                 "Access only supported for rows at the SIMD granularity"));

      const auto offsets =
          sparsity_pattern_.template transposed_offset_internal<1>(
              row, column_index);
      result[0].gather(data_.data(), offsets);

    } else if constexpr (std::is_same_v<VA, Number2> && (n_comp != 1)) {
      Assert(false,
             dealii::ExcMessage("Vectorized transposed access to multiple "
                                "components is not implemented."));
      __builtin_trap();

    } else {
      for (unsigned int d = 0; d < n_comp; ++d) {
        const auto offset =
            sparsity_pattern_.template transposed_offset<n_comp>(
                row, column_index, d);
        result[d] = data_(offset);
      }
    }

    return result;
  }


  template <typename Number,
            int n_comp,
            int warp_size,
            int simd_length,
            typename MemorySpace,
            bool writable>
  template <typename Number2>
  DEAL_II_HOST_DEVICE_ALWAYS_INLINE void
  SparseMatrixView<Number,
                   n_comp,
                   warp_size,
                   simd_length,
                   MemorySpace,
                   writable>::write_entry(const Number2 entry,
                                          const unsigned int row,
                                          const unsigned int column_index,
                                          const bool do_streaming_store) const
    requires(writable)
  {
    static_assert(
        n_comp == 1,
        "Attempted to write a scalar value into a tensor-valued matrix entry");

    AssertIndexRange(row, sparsity_pattern_.n_rows());
    AssertIndexRange(column_index, sparsity_pattern_.row_length(row));

    dealii::Tensor<1, n_comp, Number2> tensor;
    tensor[0] = entry;

    write_tensor<Number2>(tensor, row, column_index, do_streaming_store);
  }


  template <typename Number,
            int n_comp,
            int warp_size,
            int simd_length,
            typename MemorySpace,
            bool writable>
  template <typename Number2, typename Tensor>
  DEAL_II_HOST_DEVICE_ALWAYS_INLINE void
  SparseMatrixView<Number,
                   n_comp,
                   warp_size,
                   simd_length,
                   MemorySpace,
                   writable>::write_tensor(const Tensor &tensor,
                                           const unsigned int row,
                                           const unsigned int column_index,
                                           const bool do_streaming_store) const
    requires(writable)
  {
    AssertIndexRange(row, sparsity_pattern_.n_rows());
    AssertIndexRange(column_index, sparsity_pattern_.row_length(row));

    using VA = dealii::VectorizedArray<Number, simd_length>;
    if constexpr (std::is_same_v<VA, Number2>) {
      Assert(row < sparsity_pattern_.n_internal_dofs(),
             dealii::ExcMessage(
                 "Vectorized access only possible in vectorized part"));
      Assert(row % simd_length == 0,
             dealii::ExcMessage(
                 "Access only supported for rows at the SIMD granularity"));

      Number *store_pos = data_.data();
      store_pos +=
          sparsity_pattern_.template offset_internal<n_comp>(row, column_index);

      if (do_streaming_store)
        for (unsigned int d = 0; d < n_comp; ++d)
          tensor[d].streaming_store(store_pos + d * warp_size);
      else
        for (unsigned int d = 0; d < n_comp; ++d)
          tensor[d].store(store_pos + d * warp_size);

    } else {
      for (unsigned int d = 0; d < n_comp; ++d) {
        const auto offset =
            sparsity_pattern_.template offset<n_comp>(row, column_index, d);
        data_(offset) = tensor[d];
      }
    }
  }


  template <typename Number,
            int n_comp,
            int warp_size,
            int simd_length,
            typename MemorySpace,
            bool writable>
  template <typename Number2>
  DEAL_II_HOST_DEVICE_ALWAYS_INLINE void
  SparseMatrixView<Number,
                   n_comp,
                   warp_size,
                   simd_length,
                   MemorySpace,
                   writable>::add_entry(const Number2 entry,
                                        const unsigned int row,
                                        const unsigned int column_index) const
    requires(writable)
  {
    static_assert(
        n_comp == 1,
        "Attempted to write a scalar value into a tensor-valued matrix entry");

    AssertIndexRange(row, sparsity_pattern_.n_rows());
    AssertIndexRange(column_index, sparsity_pattern_.row_length(row));

    dealii::Tensor<1, n_comp, Number2> tensor;
    tensor[0] = entry;

    add_tensor<Number2>(tensor, row, column_index);
  }


  template <typename Number,
            int n_comp,
            int warp_size,
            int simd_length,
            typename MemorySpace,
            bool writable>
  template <typename Number2, typename Tensor>
  DEAL_II_HOST_DEVICE_ALWAYS_INLINE void
  SparseMatrixView<Number,
                   n_comp,
                   warp_size,
                   simd_length,
                   MemorySpace,
                   writable>::add_tensor(const Tensor &tensor,
                                         const unsigned int row,
                                         const unsigned int column_index) const
    requires(writable)
  {
    AssertIndexRange(row, sparsity_pattern_.n_rows());
    AssertIndexRange(column_index, sparsity_pattern_.row_length(row));

    using VA = dealii::VectorizedArray<Number, simd_length>;
    if constexpr (std::is_same_v<VA, Number2>) {
      Assert(row < sparsity_pattern_.n_internal_dofs(),
             dealii::ExcMessage(
                 "Vectorized access only possible in vectorized part"));
      Assert(row % simd_length == 0,
             dealii::ExcMessage(
                 "Access only supported for rows at the SIMD granularity"));

      Number *store_pos = data_.data();
      store_pos +=
          sparsity_pattern_.template offset_internal<n_comp>(row, column_index);

      for (unsigned int d = 0; d < n_comp; ++d) {
        auto temp = tensor[d];
        temp.load(store_pos + d * warp_size);
        temp += tensor[d];
        temp.store(store_pos + d * warp_size);
      }

    } else {
      for (unsigned int d = 0; d < n_comp; ++d) {
        const auto offset =
            sparsity_pattern_.template offset<n_comp>(row, column_index, d);
        data_(offset) += tensor[d];
        ;
      }
    }
  }


  template <typename Number,
            int n_comp,
            int warp_size,
            int simd_length,
            typename MemorySpace,
            bool writable>
  void SparseMatrixView<Number,
                        n_comp,
                        warp_size,
                        simd_length,
                        MemorySpace,
                        writable>::zero_out_ghost_rows() const
    requires(writable)
  {
    using HostSpace = dealii::MemorySpace::Host;
    using DefaultSpace = dealii::MemorySpace::Default;

    static_assert(std::is_same_v<MemorySpace, HostSpace> ||
                      std::is_same_v<MemorySpace, DefaultSpace>,
                  "Unexpected Kokkos memory space");

    Assert(sparse_matrix_->template is_resident<MemorySpace>(),
           dealii::ExcMessage("The chosen memory space is not resident."));

    sparse_matrix_->template zero_out_ghost_rows_on_memory_space<MemorySpace>();
  }


  template <typename Number,
            int n_comp,
            int warp_size,
            int simd_length,
            typename MemorySpace,
            bool writable>
  void SparseMatrixView<Number,
                        n_comp,
                        warp_size,
                        simd_length,
                        MemorySpace,
                        writable>::update_ghost_rows() const
    requires(writable)
  {
    using HostSpace = dealii::MemorySpace::Host;
    using DefaultSpace = dealii::MemorySpace::Default;

    static_assert(std::is_same_v<MemorySpace, HostSpace> ||
                      std::is_same_v<MemorySpace, DefaultSpace>,
                  "Unexpected Kokkos memory space");

    Assert(sparse_matrix_->template is_resident<MemorySpace>(),
           dealii::ExcMessage("The chosen memory space is not resident."));

    sparse_matrix_->template update_ghost_rows_on_memory_space<MemorySpace>();
  }


  template <typename Number,
            int n_comp,
            int warp_size,
            int simd_length,
            typename MemorySpace,
            bool writable>
  void SparseMatrixView<Number,
                        n_comp,
                        warp_size,
                        simd_length,
                        MemorySpace,
                        writable>::compress(dealii::VectorOperation::values
                                                operation) const
    requires(writable)
  {
    using HostSpace = dealii::MemorySpace::Host;
    using DefaultSpace = dealii::MemorySpace::Default;

    static_assert(std::is_same_v<MemorySpace, HostSpace> ||
                      std::is_same_v<MemorySpace, DefaultSpace>,
                  "Unexpected Kokkos memory space");

    Assert(sparse_matrix_->template is_resident<MemorySpace>(),
           dealii::ExcMessage("The chosen memory space is not resident."));

    sparse_matrix_->template compress_on_memory_space<MemorySpace>(operation);
  }


  template <typename Number,
            int n_comp,
            int warp_size,
            int simd_length,
            typename FM>
  void distribute_local_to_global(
      const FM &cell_matrix,
      const std::vector<dealii::types::global_dof_index> &dof_indices_row,
      const std::vector<dealii::types::global_dof_index> &dof_indices_column,
      const dealii::AffineConstraints<Number> &affine_constraints
      [[maybe_unused]],
      SparseMatrix<Number, n_comp, warp_size, simd_length> &sparse_matrix)
  {
    constexpr bool is_matrix = std::is_same_v<FM, dealii::FullMatrix<Number>>;
    constexpr bool is_array =
        std::is_same_v<FM, std::array<dealii::FullMatrix<Number>, n_comp>>;
    static_assert((n_comp == 1 && is_matrix) || is_array, "not implemented");

    if constexpr (is_matrix) {
      Assert(cell_matrix.m() == dof_indices_row.size(),
             dealii::ExcInternalError());
      Assert(cell_matrix.n() == dof_indices_column.size(),
             dealii::ExcInternalError());
    } else if constexpr (is_array) {
      Assert(cell_matrix.size() == n_comp, dealii::ExcInternalError());
      for (unsigned int d = 0; d < n_comp; ++d) {
        Assert(cell_matrix[d].m() == dof_indices_row.size(),
               dealii::ExcInternalError());
        Assert(cell_matrix[d].n() == dof_indices_column.size(),
               dealii::ExcInternalError());
      }
    }

    const auto sparse_matrix_view = sparse_matrix.view();

    const auto &sparsity_pattern = sparse_matrix.sparsity_pattern();
    const auto sparsity_pattern_view = sparsity_pattern.view();
    const auto &partitioner = sparsity_pattern.partitioner();

    const auto insert_entry =
        [&](auto r, auto c, auto i, auto j, auto c_ij) DEAL_II_ALWAYS_INLINE {
          if constexpr (is_matrix) {
            const Number &entry = cell_matrix(r, c);
            if (entry == Number{})
              return;
            const auto col_idx = sparsity_pattern_view.column_index(i, j);
            sparse_matrix_view.add_entry(c_ij * entry, i, col_idx);
          } else if constexpr (is_array) {
            dealii::Tensor<1, n_comp, Number> entry;
            for (unsigned int k = 0; k < n_comp; ++k)
              entry[k] = cell_matrix[k](r, c);
            if (entry == dealii::Tensor<1, n_comp>{})
              return;
            const auto col_idx = sparsity_pattern_view.column_index(i, j);
            sparse_matrix_view.add_tensor(c_ij * entry, i, col_idx);
          }
        };

    const auto iterate_over_row_entries =
        [&](const auto r, const auto i, const auto c_i) DEAL_II_ALWAYS_INLINE {
          for (unsigned int c = 0; c < dof_indices_column.size(); ++c) {
            const auto j_global = dof_indices_column[c];
            if (affine_constraints.is_constrained(j_global)) {
              const auto &line =
                  *affine_constraints.get_constraint_entries(j_global);
              for (const auto &[k_global, c_k] : line) {
                const auto k = partitioner->global_to_local(k_global);
                insert_entry(r, c, i, k, c_i * c_k);
              }
            } else {
              const auto j = partitioner->global_to_local(j_global);
              insert_entry(r, c, i, j, c_i);
            }
          }
        };

    for (unsigned int r = 0; r < dof_indices_row.size(); ++r) {
      const auto i_global = dof_indices_row[r];
      if (affine_constraints.is_constrained(i_global)) {
        const auto &line = *affine_constraints.get_constraint_entries(i_global);
        for (const auto &[k_global, c_k] : line) {
          const auto k = partitioner->global_to_local(k_global);
          iterate_over_row_entries(r, k, c_k);
        }
      } else {
        const auto i = partitioner->global_to_local(i_global);
        iterate_over_row_entries(r, i, Number(1.));
      }
    }
  }


  template <typename Number,
            int n_comp,
            int warp_size,
            int simd_length,
            typename FM>
  void distribute_local_to_global(
      const FM &cell_matrix,
      const std::vector<dealii::types::global_dof_index> &dof_indices,
      const dealii::AffineConstraints<Number> &affine_constraints,
      SparseMatrix<Number, n_comp, warp_size, simd_length> &sparse_matrix)
  {
    constexpr bool is_matrix = std::is_same_v<FM, dealii::FullMatrix<Number>>;
    constexpr bool is_array =
        std::is_same_v<FM, std::array<dealii::FullMatrix<Number>, n_comp>>;
    static_assert((n_comp == 1 && is_matrix) || is_array, "not implemented");

    distribute_local_to_global(cell_matrix,
                               dof_indices,
                               dof_indices,
                               affine_constraints,
                               sparse_matrix);
  }

#endif
} // namespace ryujin

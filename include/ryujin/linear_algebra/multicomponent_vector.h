//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/compile_time_options.h>

#include <ryujin/base/convenience_macros.h>
#include <ryujin/base/gpu.h>
#include <ryujin/base/loop.h>

#include <deal.II/base/mpi.h>
#include <deal.II/base/partitioner.h>
#include <deal.II/base/vectorization.h>
#include <deal.II/lac/la_parallel_vector.h>

namespace ryujin
{
  namespace Vectors
  {
    std::shared_ptr<const dealii::Utilities::MPI::Partitioner>
    create_vector_partitioner(
        const std::shared_ptr<const dealii::Utilities::MPI::Partitioner>
            &scalar_partitioner,
        const unsigned int n_comp);


    template <typename Number,
              int n_comp,
              int simd_length = dealii::VectorizedArray<Number>::size(),
              typename MemorySpace = dealii::MemorySpace::Host,
              bool writable = true>
    class MultiComponentVectorView;


    template <typename Number,
              int simd_length = dealii::VectorizedArray<Number>::size()>
    class MultiComponentVector
        : public MirroredStorage<MultiComponentVector<Number, simd_length>>
    {
    public:
      MultiComponentVector() = default;

      MultiComponentVector(const MultiComponentVector &other);

      MultiComponentVector(MultiComponentVector &&other) noexcept;

      void reinit_with_vector_partitioner(
          const std::shared_ptr<const dealii::Utilities::MPI::Partitioner>
              &vector_partitioner,
          const unsigned int n_comp,
          const TransferPolicy transfer_policy =
              TransferPolicy::explicit_transfers);

      void reinit_with_scalar_partitioner(
          const std::shared_ptr<const dealii::Utilities::MPI::Partitioner>
              &scalar_partitioner,
          const unsigned int n_comp,
          const TransferPolicy transfer_policy =
              TransferPolicy::explicit_transfers);

      ACCESSOR_READ_ONLY(n_comp)

      MultiComponentVector &operator=(const MultiComponentVector &other);

      MultiComponentVector &operator=(MultiComponentVector &&other) noexcept;

      template <int n_comp, typename MemorySpace = dealii::MemorySpace::Host>
      MultiComponentVectorView<Number, n_comp, simd_length, MemorySpace, true>
      view();

      template <int n_comp, typename MemorySpace = dealii::MemorySpace::Host>
      MultiComponentVectorView<Number, n_comp, simd_length, MemorySpace, false>
      view() const;

      template <typename MemorySpace = dealii::MemorySpace::Host>
      dealii::LinearAlgebra::distributed::Vector<Number, MemorySpace> &
      deal_ii_vector();

      template <typename MemorySpace = dealii::MemorySpace::Host>
      const dealii::LinearAlgebra::distributed::Vector<Number, MemorySpace> &
      deal_ii_vector() const;

      template <typename MemorySpace>
      void zero_out_ghost_values_on_memory_space();

      template <typename MemorySpace>
      void update_ghost_values_on_memory_space();

      template <typename MemorySpace>
      void compress_on_memory_space(dealii::VectorOperation::values operation);

    private:
      unsigned int n_comp_ = 0;

      mutable dealii::LinearAlgebra::distributed::
          Vector<Number, dealii::MemorySpace::Host>
              host_vector_;

      mutable dealii::LinearAlgebra::distributed::
          Vector<Number, dealii::MemorySpace::Default>
              default_vector_;

      template <typename MemorySpace>
      void allocate_storage() const;

      template <typename To, typename From>
      void deep_copy_storage() const;

      template <typename MemorySpace>
      void deallocate_storage();


      friend class MirroredStorage<MultiComponentVector<Number, simd_length>>;

      template <typename, int, int, typename, bool>
      friend class MultiComponentVectorView;
    };


    template <typename Number,
              int n_comp,
              int simd_length,
              typename MemorySpace,
              bool writable>
    class MultiComponentVectorView
    {
    public:
      MultiComponentVectorView() = default;

      MultiComponentVectorView(
          MultiComponentVector<Number, simd_length> &multi_component_vector)
        requires(writable);

      MultiComponentVectorView(const MultiComponentVector<Number, simd_length>
                                   &multi_component_vector)
        requires(!writable);

      template <bool other_writable>
      DEAL_II_HOST_DEVICE MultiComponentVectorView(
          const MultiComponentVectorView<Number,
                                         n_comp,
                                         simd_length,
                                         MemorySpace,
                                         other_writable> &other)
        requires(!writable && other_writable);

      template <typename MultiComponentVector>
      void reinit(MultiComponentVector &multi_component_vector)
        requires(writable != std::is_const_v<MultiComponentVector>);

      using ScalarVector =
          dealii::LinearAlgebra::distributed::Vector<Number, MemorySpace>;

      template <typename Functor = std::identity>
      void extract_component(ScalarVector &scalar_vector,
                             unsigned int component,
                             const Functor &functor = std::identity{}) const;

      template <typename Functor = std::identity>
      void insert_component(const ScalarVector &scalar_vector,
                            unsigned int component,
                            const Functor &functor = std::identity{}) const
        requires writable;

      template <typename Functor = std::identity>
      void insert_component(const dealii::Vector<Number> &scalar_vector,
                            unsigned int component,
                            const Functor &functor = std::identity{}) const
        requires writable;

      template <typename Number2 = Number>
      DEAL_II_HOST_DEVICE Number2 read_entry(const unsigned int i) const;

      template <typename Number2 = Number>
      DEAL_II_HOST_DEVICE Number2 read_entry(const unsigned int *js) const;

      template <typename Number2 = Number,
                typename Tensor = dealii::Tensor<1, n_comp, Number2>>
      DEAL_II_HOST_DEVICE Tensor read_tensor(const unsigned int i) const;

      template <typename Number2 = Number,
                typename Tensor = dealii::Tensor<1, n_comp, Number2>>
      DEAL_II_HOST_DEVICE Tensor read_tensor(const unsigned int *js) const;

      template <typename Number2 = Number>
      DEAL_II_HOST_DEVICE void write_entry(const Number2 &entry,
                                           const unsigned int i) const
        requires writable;

      template <typename Number2 = Number,
                typename Tensor = dealii::Tensor<1, n_comp, Number2>>
      DEAL_II_HOST_DEVICE void write_tensor(const Tensor &tensor,
                                            const unsigned int i) const
        requires writable;

      template <typename Number2 = Number>
      DEAL_II_HOST_DEVICE void add_entry(const Number2 &entry,
                                         const unsigned int i) const
        requires writable;

      template <typename Number2 = Number,
                typename Tensor = dealii::Tensor<1, n_comp, Number2>>
      DEAL_II_HOST_DEVICE void add_tensor(const Tensor &tensor,
                                          const unsigned int i) const
        requires writable;

      void zero_out_ghost_values() const
        requires(writable);

      void update_ghost_values() const
        requires(writable);

      void compress(dealii::VectorOperation::values operation) const
        requires(writable);

    private:
      using MCV = MultiComponentVector<Number, simd_length>;
      std::conditional_t<writable, MCV *, const MCV *> multi_component_vector_;

      Number *data_;
      unsigned int n_locally_owned_;
      unsigned int n_locally_relevant_;

      template <typename, int, int, typename, bool>
      friend class MultiComponentVectorView;
    };


#ifndef DOXYGEN


    template <typename Number, int simd_length>
    MultiComponentVector<Number, simd_length>::MultiComponentVector(
        const MultiComponentVector &other)
    {
      *this = other;
    }


    template <typename Number, int simd_length>
    MultiComponentVector<Number, simd_length>::MultiComponentVector(
        MultiComponentVector &&other) noexcept
    {
      *this = other;
    }


    template <typename Number, int simd_length>
    void
    MultiComponentVector<Number, simd_length>::reinit_with_vector_partitioner(
        const std::shared_ptr<const dealii::Utilities::MPI::Partitioner>
            &vector_partitioner,
        const unsigned int n_comp,
        const TransferPolicy transfer_policy)
    {
      n_comp_ = n_comp;
      this->set_transfer_policy(TransferPolicy::explicit_transfers);

      if (n_comp == 0) {
        this->reset_residency(true, true);
        this->set_transfer_policy(transfer_policy);
        return;
      }

      host_vector_.reinit(vector_partitioner);

      if constexpr (have_separate_memory_spaces)
        default_vector_.reinit(0);

      this->reset_residency(true, !have_separate_memory_spaces);

      this->set_transfer_policy(transfer_policy);
    }


    template <typename Number, int simd_length>
    void
    MultiComponentVector<Number, simd_length>::reinit_with_scalar_partitioner(
        const std::shared_ptr<const dealii::Utilities::MPI::Partitioner>
            &scalar_partitioner,
        const unsigned int n_comp,
        const TransferPolicy transfer_policy)
    {
      n_comp_ = n_comp;
      this->set_transfer_policy(TransferPolicy::explicit_transfers);

      if (n_comp == 0) {
        this->reset_residency(true, true);
        this->set_transfer_policy(transfer_policy);
        return;
      }

      if (n_comp == 1) {
        host_vector_.reinit(scalar_partitioner);
      } else {
        auto vector_partitioner =
            create_vector_partitioner(scalar_partitioner, n_comp);
        host_vector_.reinit(vector_partitioner);
      }

      if constexpr (have_separate_memory_spaces)
        default_vector_.reinit(0);

      this->reset_residency(true, !have_separate_memory_spaces);

      this->set_transfer_policy(transfer_policy);
    }


    template <typename Number, int simd_length>
    auto MultiComponentVector<Number, simd_length>::operator=(
        const MultiComponentVector &other) -> MultiComponentVector &
    {
      static_cast<MirroredStorage<MultiComponentVector> &>(*this) = other;
      n_comp_ = other.n_comp_;

      host_vector_ = other.host_vector_;
      if constexpr (have_separate_memory_spaces)
        default_vector_ = other.default_vector_;

      return *this;
    }


    template <typename Number, int simd_length>
    auto MultiComponentVector<Number, simd_length>::operator=(
        MultiComponentVector &&other) noexcept -> MultiComponentVector &
    {
      static_cast<MirroredStorage<MultiComponentVector> &>(*this) = other;
      n_comp_ = other.n_comp_;

      host_vector_ = std::move(other.host_vector_);
      if constexpr (have_separate_memory_spaces)
        default_vector_ = std::move(other.default_vector_);

      return *this;
    }


    template <typename Number, int simd_length>
    template <int n_comp, typename MemorySpace>
    MultiComponentVectorView<Number, n_comp, simd_length, MemorySpace, true>
    MultiComponentVector<Number, simd_length>::view()
    {
      Assert(static_cast<unsigned int>(n_comp) == n_comp_,
             dealii::ExcMessage("The number of components of the view does "
                                "not match the number of components of the "
                                "vector."));

      this->template prepare_write_access<MemorySpace>();

      return MultiComponentVectorView<Number,
                                      n_comp,
                                      simd_length,
                                      MemorySpace,
                                      true>(*this);
    }


    template <typename Number, int simd_length>
    template <int n_comp, typename MemorySpace>
    MultiComponentVectorView<Number, n_comp, simd_length, MemorySpace, false>
    MultiComponentVector<Number, simd_length>::view() const
    {
      Assert(static_cast<unsigned int>(n_comp) == n_comp_,
             dealii::ExcMessage("The number of components of the view does "
                                "not match the number of components of the "
                                "vector."));

      this->template prepare_read_access<MemorySpace>();

      return MultiComponentVectorView<Number,
                                      n_comp,
                                      simd_length,
                                      MemorySpace,
                                      false>(*this);
    }


    template <typename Number, int simd_length>
    template <typename MemorySpace>
    dealii::LinearAlgebra::distributed::Vector<Number, MemorySpace> &
    MultiComponentVector<Number, simd_length>::deal_ii_vector()
    {
      this->template prepare_write_access<MemorySpace>();

      if constexpr (std::is_same_v<MemorySpace, dealii::MemorySpace::Host>)
        return host_vector_;
      else
        return default_vector_;
    }


    template <typename Number, int simd_length>
    template <typename MemorySpace>
    const dealii::LinearAlgebra::distributed::Vector<Number, MemorySpace> &
    MultiComponentVector<Number, simd_length>::deal_ii_vector() const
    {
      this->template prepare_read_access<MemorySpace>();

      if constexpr (std::is_same_v<MemorySpace, dealii::MemorySpace::Host>)
        return host_vector_;
      else
        return default_vector_;
    }


    template <typename Number, int simd_length>
    template <typename MemorySpace>
    void MultiComponentVector<Number, simd_length>::allocate_storage() const
    {
      using HostSpace = dealii::MemorySpace::Host;

      if constexpr (std::is_same_v<MemorySpace, HostSpace>) {
        Assert(default_vector_.size() != 0, dealii::ExcNotInitialized());
        host_vector_.reinit(default_vector_.get_partitioner());
      } else {
        Assert(host_vector_.size() != 0, dealii::ExcNotInitialized());
        default_vector_.reinit(host_vector_.get_partitioner());
      }
    }


    template <typename Number, int simd_length>
    template <typename To, typename From>
    void MultiComponentVector<Number, simd_length>::deep_copy_storage() const
    {
      using HostSpace = dealii::MemorySpace::Host;

      if constexpr (std::is_same_v<To, HostSpace>) {
        host_vector_.import_elements(default_vector_,
                                     dealii::VectorOperation::insert);
      } else {
        default_vector_.import_elements(host_vector_,
                                        dealii::VectorOperation::insert);
      }
    }


    template <typename Number, int simd_length>
    template <typename MemorySpace>
    void MultiComponentVector<Number, simd_length>::deallocate_storage()
    {
      using HostSpace = dealii::MemorySpace::Host;

      if constexpr (std::is_same_v<MemorySpace, HostSpace>) {
        host_vector_.reinit(0);

      } else {
        default_vector_.reinit(0);
      }
    }


    template <typename Number, int simd_length>
    template <typename MemorySpace>
    void
    MultiComponentVector<Number,
                         simd_length>::zero_out_ghost_values_on_memory_space()
    {
      using HostSpace = dealii::MemorySpace::Host;
      using DefaultSpace = dealii::MemorySpace::Default;
      static_assert(std::is_same_v<MemorySpace, HostSpace> ||
                        std::is_same_v<MemorySpace, DefaultSpace>,
                    "Unexpected memory space");

      Assert(this->template is_resident<MemorySpace>(),
             dealii::ExcMessage("The chosen memory space is not resident."));

      if constexpr (have_separate_memory_spaces &&
                    !std::is_same_v<MemorySpace, HostSpace>) {
        default_vector_.zero_out_ghost_values();
      } else {
        host_vector_.zero_out_ghost_values();
      }
    }


    template <typename Number, int simd_length>
    template <typename MemorySpace>
    void
    MultiComponentVector<Number,
                         simd_length>::update_ghost_values_on_memory_space()
    {
      using HostSpace = dealii::MemorySpace::Host;
      using DefaultSpace = dealii::MemorySpace::Default;
      static_assert(std::is_same_v<MemorySpace, HostSpace> ||
                        std::is_same_v<MemorySpace, DefaultSpace>,
                    "Unexpected memory space");

      Assert(this->template is_resident<MemorySpace>(),
             dealii::ExcMessage("The chosen memory space is not resident."));

      if constexpr (have_separate_memory_spaces &&
                    !std::is_same_v<MemorySpace, HostSpace>) {
        default_vector_.update_ghost_values();
      } else {
        host_vector_.update_ghost_values();
      }
    }


    template <typename Number, int simd_length>
    template <typename MemorySpace>
    void MultiComponentVector<Number, simd_length>::compress_on_memory_space(
        dealii::VectorOperation::values operation)
    {
      using HostSpace = dealii::MemorySpace::Host;
      using DefaultSpace = dealii::MemorySpace::Default;
      static_assert(std::is_same_v<MemorySpace, HostSpace> ||
                        std::is_same_v<MemorySpace, DefaultSpace>,
                    "Unexpected memory space");

      Assert(this->template is_resident<MemorySpace>(),
             dealii::ExcMessage("The chosen memory space is not resident."));

      if constexpr (have_separate_memory_spaces &&
                    !std::is_same_v<MemorySpace, HostSpace>) {
        default_vector_.compress(operation);
      } else {
        host_vector_.compress(operation);
      }
    }


    template <typename Number,
              int n_comp,
              int simd_l,
              typename MemorySpace,
              bool writable>
    MultiComponentVectorView<Number, n_comp, simd_l, MemorySpace, writable>::
        MultiComponentVectorView(
            MultiComponentVector<Number, simd_l> &multi_component_vector)
      requires(writable)
    {
      reinit(multi_component_vector);
    }


    template <typename Number,
              int n_comp,
              int simd_l,
              typename MemorySpace,
              bool writable>
    MultiComponentVectorView<Number, n_comp, simd_l, MemorySpace, writable>::
        MultiComponentVectorView(
            const MultiComponentVector<Number, simd_l> &multi_component_vector)
      requires(!writable)
    {
      reinit(multi_component_vector);
    }


    template <typename Number,
              int n_comp,
              int simd_l,
              typename MemorySpace,
              bool writable>
    template <bool other_writable>
    DEAL_II_HOST_DEVICE
    MultiComponentVectorView<Number, n_comp, simd_l, MemorySpace, writable>::
        MultiComponentVectorView(
            const MultiComponentVectorView<Number,
                                           n_comp,
                                           simd_l,
                                           MemorySpace,
                                           other_writable> &other)
      requires(!writable && other_writable)
        : multi_component_vector_(other.multi_component_vector_)
        , data_(other.data_)
        , n_locally_owned_(other.n_locally_owned_)
        , n_locally_relevant_(other.n_locally_relevant_)
    {
    }


    template <typename Number,
              int n_comp,
              int simd_l,
              typename MemorySpace,
              bool writable>
    template <typename MultiComponentVector>
    void
    MultiComponentVectorView<Number, n_comp, simd_l, MemorySpace, writable>::
        reinit(MultiComponentVector &multi_component_vector)
      requires(writable != std::is_const_v<MultiComponentVector>)
    {
      using HostSpace = dealii::MemorySpace::Host;
      using DefaultSpace = dealii::MemorySpace::Default;
      static_assert(std::is_same_v<MemorySpace, HostSpace> ||
                        std::is_same_v<MemorySpace, DefaultSpace>,
                    "Unexpected memory space");

      Assert(multi_component_vector.n_comp() ==
                 static_cast<unsigned int>(n_comp),
             dealii::ExcMessage("The number of components of the view does "
                                "not match the number of components of the "
                                "vector."));

      multi_component_vector_ = &multi_component_vector;

      if constexpr (have_separate_memory_spaces &&
                    !std::is_same_v<MemorySpace, HostSpace>) {
        auto &vector = multi_component_vector_->default_vector_;
        const auto &partitioner = vector.get_partitioner();

        data_ = vector.begin();
        n_locally_owned_ = partitioner->locally_owned_size();
        n_locally_relevant_ = n_locally_owned_ + partitioner->n_ghost_indices();

      } else {
        auto &vector = multi_component_vector_->host_vector_;
        const auto &partitioner = vector.get_partitioner();

        data_ = vector.begin();
        n_locally_owned_ = partitioner->locally_owned_size();
        n_locally_relevant_ = n_locally_owned_ + partitioner->n_ghost_indices();
      }
    }


    template <typename Number,
              int n_comp,
              int simd_length,
              typename MemorySpace,
              bool writable>
    template <typename Functor>
    void MultiComponentVectorView<
        Number,
        n_comp,
        simd_length,
        MemorySpace,
        writable>::extract_component(ScalarVector &scalar_vector,
                                     unsigned int component,
                                     const Functor &functor) const
    {
      Assert(n_comp > 0,
             dealii::ExcMessage(
                 "Cannot extract from a vector with zero components."));
      AssertIndexRange(component, n_comp);

      const auto local_size = static_cast<unsigned int>(
          scalar_vector.get_partitioner()->locally_owned_size());

      Assert(n_comp * local_size == n_locally_owned_,
             dealii::ExcMessage("Called with a scalar_vector argument that has "
                                "incompatible local range."));

      const auto *data = data_;
      auto *destination = scalar_vector.begin();

      const auto body = [=](auto, unsigned int i) {
        destination[i] = functor(data[i * n_comp + component]);
      };

      loop<MemorySpace, Number>("extract_component", body, 0, 0, local_size);

      scalar_vector.update_ghost_values();
    }


    template <typename Number,
              int n_comp,
              int simd_length,
              typename MemorySpace,
              bool writable>
    template <typename Functor>
    void MultiComponentVectorView<
        Number,
        n_comp,
        simd_length,
        MemorySpace,
        writable>::insert_component(const ScalarVector &scalar_vector,
                                    unsigned int component,
                                    const Functor &functor) const
      requires writable
    {
      using HostSpace = dealii::MemorySpace::Host;
      AssertThrow((std::is_same_v<MemorySpace, HostSpace>),
                  dealii::ExcNotImplemented());

      Assert(n_comp > 0,
             dealii::ExcMessage(
                 "Cannot insert into a vector with zero components."));
      AssertIndexRange(component, n_comp);

      const auto local_size =
          scalar_vector.get_partitioner()->locally_owned_size();

      Assert(n_comp * local_size == n_locally_owned_,
             dealii::ExcMessage("Called with a scalar_vector argument that has "
                                "incompatible local range."));

      for (unsigned int i = 0; i < local_size; ++i)
        data_[i * n_comp + component] = functor(scalar_vector.local_element(i));
    }


    template <typename Number,
              int n_comp,
              int simd_length,
              typename MemorySpace,
              bool writable>
    template <typename Functor>
    void MultiComponentVectorView<
        Number,
        n_comp,
        simd_length,
        MemorySpace,
        writable>::insert_component(const dealii::Vector<Number> &scalar_vector,
                                    unsigned int component,
                                    const Functor &functor) const
      requires writable
    {
      using HostSpace = dealii::MemorySpace::Host;
      AssertThrow((std::is_same_v<MemorySpace, HostSpace>),
                  dealii::ExcInternalError());

      Assert(n_comp > 0,
             dealii::ExcMessage(
                 "Cannot insert into a vector with zero components."));
      AssertIndexRange(component, n_comp);

      const auto local_size = scalar_vector.size();

      Assert(n_comp * local_size >= n_locally_owned_,
             dealii::ExcMessage("Called with a scalar_vector argument that has "
                                "incompatible local range."));

      for (unsigned int i = 0; i < local_size; ++i)
        data_[i * n_comp + component] = functor(scalar_vector[i]);
    }


    template <typename Number,
              int n_comp,
              int simd_length,
              typename MemorySpace,
              bool writable>
    template <typename Number2>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE Number2
    MultiComponentVectorView<Number,
                             n_comp,
                             simd_length,
                             MemorySpace,
                             writable>::read_entry(const unsigned int i) const
    {
      static_assert(
          n_comp == 1,
          "Attempted to read a scalar value from a tensor-valued vector entry");

      AssertIndexRange(i, n_locally_relevant_);

      const auto result = read_tensor<Number2>(i);
      return result[0];
    }


    template <typename Number,
              int n_comp,
              int simd_length,
              typename MemorySpace,
              bool writable>
    template <typename Number2, typename Tensor>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE Tensor
    MultiComponentVectorView<Number,
                             n_comp,
                             simd_length,
                             MemorySpace,
                             writable>::read_tensor(const unsigned int i) const
    {
      static_assert(std::is_same_v<Number2, typename Tensor::value_type>,
                    "type mismatch");

      AssertIndexRange(i, n_locally_relevant_);

      Tensor tensor;

      if constexpr (n_comp == 0)
        return tensor;

      using VA = dealii::VectorizedArray<Number>;
      if constexpr (std::is_same_v<VA, Number2>) {
        std::array<unsigned int, VA::size()> indices;
        for (unsigned int k = 0; k < VA::size(); ++k)
          indices[k] = k * n_comp;

        dealii::vectorized_load_and_transpose(
            n_comp, data_ + i * n_comp, indices.data(), &tensor[0]);

      } else {
        for (unsigned int d = 0; d < n_comp; ++d)
          tensor[d] = data_[i * n_comp + d];
      }

      return tensor;
    }


    template <typename Number,
              int n_comp,
              int simd_length,
              typename MemorySpace,
              bool writable>
    template <typename Number2>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE Number2
    MultiComponentVectorView<Number,
                             n_comp,
                             simd_length,
                             MemorySpace,
                             writable>::read_entry(const unsigned int *js) const
    {
      static_assert(
          n_comp == 1,
          "Attempted to read a scalar value from a tensor-valued vector entry");

      const auto result = read_tensor<Number2>(js);
      return result[0];
    }


    template <typename Number,
              int n_comp,
              int simd_length,
              typename MemorySpace,
              bool writable>
    template <typename Number2, typename Tensor>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE Tensor
    MultiComponentVectorView<Number,
                             n_comp,
                             simd_length,
                             MemorySpace,
                             writable>::read_tensor(const unsigned int *js)
        const
    {
      static_assert(std::is_same_v<Number2, typename Tensor::value_type>,
                    "type mismatch");

      Tensor tensor;

      if constexpr (n_comp == 0)
        return tensor;

      using VA = dealii::VectorizedArray<Number>;
      if constexpr (std::is_same_v<VA, Number2>) {
        std::array<unsigned int, VA::size()> indices;
        for (unsigned int k = 0; k < VA::size(); ++k) {
          AssertIndexRange(js[k], n_locally_relevant_);
          indices[k] = js[k] * n_comp;
        }

        dealii::vectorized_load_and_transpose(
            n_comp, data_, indices.data(), &tensor[0]);

      } else {
        AssertIndexRange(*js, n_locally_relevant_);

        for (unsigned int d = 0; d < n_comp; ++d)
          tensor[d] = data_[js[0] * n_comp + d];
      }

      return tensor;
    }


    template <typename Number,
              int n_comp,
              int simd_length,
              typename MemorySpace,
              bool writable>
    template <typename Number2>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE void
    MultiComponentVectorView<Number,
                             n_comp,
                             simd_length,
                             MemorySpace,
                             writable>::write_entry(const Number2 &entry,
                                                    const unsigned int i) const
      requires writable
    {
      static_assert(n_comp == 1,
                    "Attempted to write a scalar value into a tensor-valued "
                    "vector entry");

      AssertIndexRange(i, n_locally_relevant_);

      dealii::Tensor<1, n_comp, Number2> tensor;
      tensor[0] = entry;

      write_tensor<Number2>(tensor, i);
    }


    template <typename Number,
              int n_comp,
              int simd_length,
              typename MemorySpace,
              bool writable>
    template <typename Number2, typename Tensor>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE void
    MultiComponentVectorView<Number,
                             n_comp,
                             simd_length,
                             MemorySpace,
                             writable>::write_tensor(const Tensor &tensor,
                                                     const unsigned int i) const
      requires writable
    {
      static_assert(std::is_same_v<Number2, typename Tensor::value_type>,
                    "type mismatch");

      AssertIndexRange(i, n_locally_relevant_);

      if constexpr (n_comp == 0)
        return;

      using VA = dealii::VectorizedArray<Number>;
      if constexpr (std::is_same_v<VA, Number2>) {
        std::array<unsigned int, VA::size()> indices;
        for (unsigned int k = 0; k < VA::size(); ++k)
          indices[k] = k * n_comp;

        dealii::vectorized_transpose_and_store(
            false, n_comp, &tensor[0], indices.data(), data_ + i * n_comp);
      } else {
        for (unsigned int d = 0; d < n_comp; ++d)
          data_[i * n_comp + d] = tensor[d];
      }
    }


    template <typename Number,
              int n_comp,
              int simd_length,
              typename MemorySpace,
              bool writable>
    template <typename Number2>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE void
    MultiComponentVectorView<Number,
                             n_comp,
                             simd_length,
                             MemorySpace,
                             writable>::add_entry(const Number2 &entry,
                                                  const unsigned int i) const
      requires writable
    {
      static_assert(n_comp == 1,
                    "Attempted to write a scalar value into a tensor-valued "
                    "matrix entry");

      AssertIndexRange(i, n_locally_relevant_);

      dealii::Tensor<1, n_comp, Number2> tensor;
      tensor[0] = entry;

      add_tensor<Number2>(tensor, i);
    }


    template <typename Number,
              int n_comp,
              int simd_length,
              typename MemorySpace,
              bool writable>
    template <typename Number2, typename Tensor>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE void
    MultiComponentVectorView<Number,
                             n_comp,
                             simd_length,
                             MemorySpace,
                             writable>::add_tensor(const Tensor &tensor,
                                                   const unsigned int i) const
      requires writable
    {
      static_assert(std::is_same_v<Number2, typename Tensor::value_type>,
                    "type mismatch");

      AssertIndexRange(i, n_locally_relevant_);

      if constexpr (n_comp == 0)
        return;

      using VA = dealii::VectorizedArray<Number>;
      if constexpr (std::is_same_v<VA, Number2>) {
        std::array<unsigned int, VA::size()> indices;
        for (unsigned int k = 0; k < VA::size(); ++k)
          indices[k] = k * n_comp;

        dealii::vectorized_transpose_and_store(
            true, n_comp, &tensor[0], indices.data(), data_ + i * n_comp);
      } else {
        for (unsigned int d = 0; d < n_comp; ++d)
          data_[i * n_comp + d] += tensor[d];
      }
    }


    template <typename Number,
              int n_comp,
              int simd_l,
              typename MemorySpace,
              bool writable>
    void
    MultiComponentVectorView<Number, n_comp, simd_l, MemorySpace, writable>::
        zero_out_ghost_values() const
      requires(writable)
    {
      using HostSpace = dealii::MemorySpace::Host;
      using DefaultSpace = dealii::MemorySpace::Default;
      static_assert(std::is_same_v<MemorySpace, HostSpace> ||
                        std::is_same_v<MemorySpace, DefaultSpace>,
                    "Unexpected memory space");

      Assert(multi_component_vector_->template is_resident<MemorySpace>(),
             dealii::ExcMessage("The chosen memory space is not resident."));

      multi_component_vector_
          ->template zero_out_ghost_values_on_memory_space<MemorySpace>();
    }


    template <typename Number,
              int n_comp,
              int simd_l,
              typename MemorySpace,
              bool writable>
    void
    MultiComponentVectorView<Number, n_comp, simd_l, MemorySpace, writable>::
        update_ghost_values() const
      requires(writable)
    {
      using HostSpace = dealii::MemorySpace::Host;
      using DefaultSpace = dealii::MemorySpace::Default;
      static_assert(std::is_same_v<MemorySpace, HostSpace> ||
                        std::is_same_v<MemorySpace, DefaultSpace>,
                    "Unexpected memory space");

      Assert(multi_component_vector_->template is_resident<MemorySpace>(),
             dealii::ExcMessage("The chosen memory space is not resident."));

      multi_component_vector_
          ->template update_ghost_values_on_memory_space<MemorySpace>();
    }


    template <typename Number,
              int n_comp,
              int simd_l,
              typename MemorySpace,
              bool writable>
    void
    MultiComponentVectorView<Number, n_comp, simd_l, MemorySpace, writable>::
        compress(dealii::VectorOperation::values operation) const
      requires(writable)
    {
      using HostSpace = dealii::MemorySpace::Host;
      using DefaultSpace = dealii::MemorySpace::Default;
      static_assert(std::is_same_v<MemorySpace, HostSpace> ||
                        std::is_same_v<MemorySpace, DefaultSpace>,
                    "Unexpected memory space");

      Assert(multi_component_vector_->template is_resident<MemorySpace>(),
             dealii::ExcMessage("The chosen memory space is not resident."));

      multi_component_vector_->template compress_on_memory_space<MemorySpace>(
          operation);
    }

#endif
  } // namespace Vectors
} // namespace ryujin

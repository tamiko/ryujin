//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/compile_time_options.h>
#include <ryujin/base/convenience_macros.h>
#include <ryujin/base/simd.h>

#include <deal.II/base/exceptions.h>
#include <deal.II/base/memory_space.h>
#include <deal.II/base/vectorization.h>

#include <string>
#include <type_traits>

namespace ryujin
{
  inline constexpr bool have_separate_memory_spaces =
      !std::is_same_v<dealii::MemorySpace::Host::kokkos_space,
                      dealii::MemorySpace::Default::kokkos_space>;


  inline constexpr unsigned int warp_size =
      have_separate_memory_spaces ? 32
                                  : dealii::VectorizedArray<NUMBER>::size();


  using selected_memory_space_t =
      std::conditional_t<have_separate_memory_spaces,
                         dealii::MemorySpace::Default,
                         dealii::MemorySpace::Host>;


  template <typename MemorySpace>
  using other_space_t =
      std::conditional_t<std::is_same_v<MemorySpace, dealii::MemorySpace::Host>,
                         dealii::MemorySpace::Default,
                         dealii::MemorySpace::Host>;


  enum class TransferPolicy {
    explicit_transfers,
    implicit_transfers,
    implicit_transfers_host_resident,
    implicit_transfers_default_resident,
  };


  inline constexpr bool performs_implicit_transfers(const TransferPolicy policy)
  {
    return policy == TransferPolicy::implicit_transfers ||
           policy == TransferPolicy::implicit_transfers_host_resident ||
           policy == TransferPolicy::implicit_transfers_default_resident;
  }


  template <typename Derived>
  class MirroredStorage
  {
  public:
    template <typename MemorySpace>
    bool is_resident() const;

    template <typename MemorySpace>
    bool is_pinned() const;

    template <typename MemorySpace>
    void copy_to_memory_space() const;

    template <typename MemorySpace>
    void move_to_memory_space();

    TransferPolicy transfer_policy() const;

    void set_transfer_policy(const TransferPolicy transfer_policy);

  protected:
    MirroredStorage() = default;

    template <typename MemorySpace>
    void prepare_read_access() const;

    template <typename MemorySpace>
    void prepare_write_access();

    void reset_residency(const bool host_resident, const bool default_resident);

  private:
    using HostSpace = dealii::MemorySpace::Host;
    using DefaultSpace = dealii::MemorySpace::Default;

    template <typename MemorySpace>
    bool &residency_flag() const;

    Derived &derived();
    const Derived &derived() const;

    mutable bool host_resident_ = false;
    mutable bool default_resident_ = false;

    TransferPolicy transfer_policy_ = TransferPolicy::explicit_transfers;
  };


  template <typename T>
  class Mirrored : public MirroredStorage<Mirrored<T>>
  {
  public:
    static constexpr bool is_array = std::is_pointer_v<T>;

    using value_type = std::remove_pointer_t<T>;

    static_assert(std::is_trivially_copyable_v<value_type>,
                  "The stored type has to be trivially copyable so that we "
                  "can move it into device memory");

    static_assert(!std::is_pointer_v<value_type>,
                  "Only a single level of indirection is supported: a "
                  "Mirrored<T *> object maintains a one dimensional array of "
                  "objects of type T");

    static_assert(!std::is_array_v<T>,
                  "An array of objects is spelled Mirrored<T *>, and not "
                  "Mirrored<T[]>");

    template <typename InitializedMemorySpace = dealii::MemorySpace::Host>
    Mirrored(const std::string &label = "mirrored object",
             const TransferPolicy transfer_policy =
                 TransferPolicy::explicit_transfers,
             InitializedMemorySpace = {})
      requires(!is_array);

    template <typename InitializedMemorySpace = dealii::MemorySpace::Host>
    Mirrored(const std::string &label = "mirrored array",
             const std::size_t size = 0,
             const TransferPolicy transfer_policy =
                 TransferPolicy::explicit_transfers,
             InitializedMemorySpace = {})
      requires(is_array);

    template <typename InitializedMemorySpace = dealii::MemorySpace::Host>
    void reinit(const std::size_t size,
                const TransferPolicy transfer_policy =
                    TransferPolicy::explicit_transfers,
                InitializedMemorySpace = {})
      requires(is_array);

    std::size_t size() const
      requires(is_array);

    template <typename MemorySpace = dealii::MemorySpace::Host>
    value_type *view();

    template <typename MemorySpace = dealii::MemorySpace::Host>
    const value_type *view() const;

  private:
    using HostSpace = dealii::MemorySpace::Host;
    using DefaultSpace = dealii::MemorySpace::Default;

    template <typename MemorySpace>
    using KokkosView = Kokkos::View<T, typename MemorySpace::kokkos_space>;

    template <typename MemorySpace>
    KokkosView<MemorySpace> &storage() const;

    template <typename InitializedMemorySpace>
    void initialize_storage();

    template <typename MemorySpace>
    void allocate_storage() const;

    template <typename To, typename From>
    void deep_copy_storage() const;

    template <typename MemorySpace>
    void deallocate_storage();

    std::string label_;

    std::size_t size_ = 0;

    mutable KokkosView<HostSpace> host_;
    mutable KokkosView<DefaultSpace> default_;

    friend class MirroredStorage<Mirrored<T>>;
  };


  template <int dim, typename Number, typename MemorySpace, typename Object>
  class SelectView
  {
  private:
    using SimdNumber = std::conditional_t<
        std::is_same_v<MemorySpace, dealii::MemorySpace::Host>,
        dealii::VectorizedArray<Number>,
        Number>;

    template <typename T>
    static auto create_view(const Object &object)
    {
      if constexpr (std::is_same_v<MemorySpace, dealii::MemorySpace::Host>)
        return object.template view<dim, T>();
      else if constexpr (requires {
                           object.template view<dim, T, MemorySpace>();
                         })
        return object.template view<dim, T, MemorySpace>();
      else {
        static_assert(!have_separate_memory_spaces || sizeof(T) == 0,
                      "The equation does not support MemorySpace");
        return object.template view<dim, T>();
      }
    }

    using ScalarView =
        decltype(create_view<Number>(std::declval<const Object &>()));
    using SimdView =
        decltype(create_view<SimdNumber>(std::declval<const Object &>()));

  public:
    SelectView(const Object &object)
        : scalar_view_(create_view<Number>(object))
        , simd_view_(create_view<SimdNumber>(object))
    {
    }

    template <typename T>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE const auto &view() const
    {
      if constexpr (std::is_same_v<T, typename get_value_type<T>::type>)
        return scalar_view_;
      else
        return simd_view_;
    }

  private:
    ScalarView scalar_view_;
    SimdView simd_view_;
  };


  template <int dim, typename Number, typename MemorySpace, typename Object>
  auto make_select_view(const Object &object)
  {
    return SelectView<dim, Number, MemorySpace, Object>(object);
  }


#ifndef DOXYGEN


  template <typename Derived>
  template <typename MemorySpace>
  inline bool MirroredStorage<Derived>::is_resident() const
  {
    static_assert(std::is_same_v<MemorySpace, HostSpace> ||
                      std::is_same_v<MemorySpace, DefaultSpace>,
                  "Unexpected memory space");

    if constexpr (!have_separate_memory_spaces) {
      Assert(host_resident_ == default_resident_,
             dealii::ExcMessage(
                 "Internal error: the host and default memory spaces coincide "
                 "but the two residency flags do not. There is only a single "
                 "allocation, so the object has to be resident on both memory "
                 "spaces, or on neither of them."));
      return host_resident_ || default_resident_;
    }

    return residency_flag<MemorySpace>();
  }


  template <typename Derived>
  template <typename MemorySpace>
  inline bool MirroredStorage<Derived>::is_pinned() const
  {
    static_assert(std::is_same_v<MemorySpace, HostSpace> ||
                      std::is_same_v<MemorySpace, DefaultSpace>,
                  "Unexpected memory space");

    if constexpr (!have_separate_memory_spaces) {
      return false;
    }

    if constexpr (std::is_same_v<MemorySpace, HostSpace>) {
      return transfer_policy_ ==
             TransferPolicy::implicit_transfers_host_resident;

    } else {
      return transfer_policy_ ==
             TransferPolicy::implicit_transfers_default_resident;
    }
  }


  template <typename Derived>
  template <typename MemorySpace>
  inline void MirroredStorage<Derived>::copy_to_memory_space() const
  {
    static_assert(std::is_same_v<MemorySpace, HostSpace> ||
                      std::is_same_v<MemorySpace, DefaultSpace>,
                  "Unexpected memory space");

    if constexpr (!have_separate_memory_spaces) {
      Assert(host_resident_ && default_resident_,
             dealii::ExcMessage(
                 "Unable to copy to the chosen memory space: the host and "
                 "default memory spaces coincide but the object is not "
                 "resident on them. The object has not been properly "
                 "initialized."));
      return;
    }

    Assert(!is_pinned<MemorySpace>() || is_resident<MemorySpace>(),
           dealii::ExcMessage(
               "Unable to copy to the chosen memory space: the selected "
               "transfer policy pins the chosen memory space but the object "
               "is not resident on it. The object has not been properly "
               "initialized."));

    if (is_resident<MemorySpace>())
      return;

    using OtherSpace = other_space_t<MemorySpace>;
    Assert(is_resident<OtherSpace>(),
           dealii::ExcMessage(
               "Unable to copy to the chosen memory space: the object has "
               "not been properly initialized."));

    derived().template allocate_storage<MemorySpace>();
    derived().template deep_copy_storage<MemorySpace, OtherSpace>();
    residency_flag<MemorySpace>() = true;
  }


  template <typename Derived>
  template <typename MemorySpace>
  inline void MirroredStorage<Derived>::move_to_memory_space()
  {
    static_assert(std::is_same_v<MemorySpace, HostSpace> ||
                      std::is_same_v<MemorySpace, DefaultSpace>,
                  "Unexpected memory space");

    if constexpr (!have_separate_memory_spaces) {
      Assert(host_resident_ && default_resident_,
             dealii::ExcMessage(
                 "Unable to move to the chosen memory space: the host and "
                 "default memory spaces coincide but the object is not "
                 "resident on them. The object has not been properly "
                 "initialized."));
      return;
    }

    Assert(!is_pinned<MemorySpace>() || is_resident<MemorySpace>(),
           dealii::ExcMessage(
               "Unable to move to the chosen memory space: the selected "
               "transfer policy pins the chosen memory space but the object "
               "is not resident on it. The object has not been properly "
               "initialized."));

    using OtherSpace = other_space_t<MemorySpace>;

    Assert(!is_pinned<OtherSpace>(),
           dealii::ExcMessage(
               "Unable to move to the chosen memory space: the selected "
               "transfer policy requires the data to remain resident on "
               "the other memory space."));

    if (!is_resident<MemorySpace>()) {
      Assert(is_resident<OtherSpace>(),
             dealii::ExcMessage(
                 "Unable to move to the chosen memory space: the object has "
                 "not been properly initialized."));

      derived().template allocate_storage<MemorySpace>();
      derived().template deep_copy_storage<MemorySpace, OtherSpace>();
      residency_flag<MemorySpace>() = true;
    }

    if (residency_flag<OtherSpace>()) {
      derived().template deallocate_storage<OtherSpace>();
      residency_flag<OtherSpace>() = false;
    }
  }


  template <typename Derived>
  inline TransferPolicy MirroredStorage<Derived>::transfer_policy() const
  {
    return transfer_policy_;
  }


  template <typename Derived>
  inline void MirroredStorage<Derived>::set_transfer_policy(
      const TransferPolicy transfer_policy)
  {
    transfer_policy_ = transfer_policy;

    Assert(!is_pinned<HostSpace>() || is_resident<HostSpace>(),
           dealii::ExcMessage(
               "Unable to select the given transfer policy: the policy pins "
               "the host memory space but the object is not resident on it. "
               "Transfer the data to the pinned memory space before selecting "
               "the transfer policy."));

    Assert(!is_pinned<DefaultSpace>() || is_resident<DefaultSpace>(),
           dealii::ExcMessage(
               "Unable to select the given transfer policy: the policy pins "
               "the default memory space but the object is not resident on "
               "it. Transfer the data to the pinned memory space before "
               "selecting the transfer policy."));
  }


  template <typename Derived>
  template <typename MemorySpace>
  inline void MirroredStorage<Derived>::prepare_read_access() const
  {
    if (performs_implicit_transfers(transfer_policy_)) {
      copy_to_memory_space<MemorySpace>();

    } else {
      Assert(is_resident<MemorySpace>(),
             dealii::ExcMessage(
                 "The chosen memory space is not resident. Either call "
                 "copy_to_memory_space() / move_to_memory_space() prior to "
                 "requesting a view, or select one of the "
                 "TransferPolicy::implicit_transfers* policies."));
    }
  }


  template <typename Derived>
  template <typename MemorySpace>
  inline void MirroredStorage<Derived>::prepare_write_access()
  {
    if (performs_implicit_transfers(transfer_policy_)) {
      Assert(!is_pinned<other_space_t<MemorySpace>>(),
             dealii::ExcMessage(
                 "Unable to request a writable view on the chosen memory "
                 "space: the selected transfer policy requires the data to "
                 "remain resident on the other memory space."));

      move_to_memory_space<MemorySpace>();

    } else {
      Assert(is_resident<MemorySpace>(),
             dealii::ExcMessage(
                 "The chosen memory space is not resident. Either call "
                 "copy_to_memory_space() / move_to_memory_space() prior to "
                 "requesting a view, or select one of the "
                 "TransferPolicy::implicit_transfers* policies."));
    }
  }


  template <typename Derived>
  inline void
  MirroredStorage<Derived>::reset_residency(const bool host_resident,
                                            const bool default_resident)
  {
    if constexpr (!have_separate_memory_spaces) {
      Assert(host_resident == default_resident,
             dealii::ExcMessage(
                 "Unable to reset the residency flags: the host and default "
                 "memory spaces coincide, so there is only a single "
                 "allocation and both flags have to coincide as well."));
    }

    host_resident_ = host_resident;
    default_resident_ = default_resident;

    Assert(!is_pinned<HostSpace>() || is_resident<HostSpace>(),
           dealii::ExcMessage(
               "Unable to reset the residency flags: the selected transfer "
               "policy pins the host memory space and requires the data to "
               "remain resident on it."));

    Assert(!is_pinned<DefaultSpace>() || is_resident<DefaultSpace>(),
           dealii::ExcMessage(
               "Unable to reset the residency flags: the selected transfer "
               "policy pins the default memory space and requires the data "
               "to remain resident on it."));
  }


  template <typename Derived>
  template <typename MemorySpace>
  inline bool &MirroredStorage<Derived>::residency_flag() const
  {
    if constexpr (std::is_same_v<MemorySpace, HostSpace>)
      return host_resident_;
    else
      return default_resident_;
  }


  template <typename Derived>
  inline Derived &MirroredStorage<Derived>::derived()
  {
    return static_cast<Derived &>(*this);
  }


  template <typename Derived>
  inline const Derived &MirroredStorage<Derived>::derived() const
  {
    return static_cast<const Derived &>(*this);
  }


  template <typename T>
  template <typename InitializedMemorySpace>
  Mirrored<T>::Mirrored(const std::string &label,
                        const TransferPolicy transfer_policy,
                        InitializedMemorySpace)
    requires(!is_array)
      : label_(label)
  {
    initialize_storage<InitializedMemorySpace>();

    this->set_transfer_policy(transfer_policy);
  }


  template <typename T>
  template <typename InitializedMemorySpace>
  Mirrored<T>::Mirrored(const std::string &label,
                        const std::size_t size,
                        const TransferPolicy transfer_policy,
                        InitializedMemorySpace)
    requires(is_array)
      : label_(label)
  {
    reinit(size, transfer_policy, InitializedMemorySpace{});
  }


  template <typename T>
  template <typename InitializedMemorySpace>
  void Mirrored<T>::reinit(const std::size_t size,
                           const TransferPolicy transfer_policy,
                           InitializedMemorySpace)
    requires(is_array)
  {
    this->set_transfer_policy(TransferPolicy::explicit_transfers);

    size_ = size;

    initialize_storage<InitializedMemorySpace>();

    this->set_transfer_policy(transfer_policy);
  }


  template <typename T>
  inline std::size_t Mirrored<T>::size() const
    requires(is_array)
  {
    return size_;
  }


  template <typename T>
  template <typename InitializedMemorySpace>
  void Mirrored<T>::initialize_storage()
  {
    static_assert(std::is_same_v<InitializedMemorySpace, HostSpace> ||
                      std::is_same_v<InitializedMemorySpace, DefaultSpace>,
                  "Unexpected memory space");

    if constexpr (have_separate_memory_spaces &&
                  !std::is_same_v<InitializedMemorySpace, HostSpace>) {
      deallocate_storage<HostSpace>();
      allocate_storage<DefaultSpace>();

      this->reset_residency(false, true);
    } else {
      if constexpr (have_separate_memory_spaces)
        deallocate_storage<DefaultSpace>();
      allocate_storage<HostSpace>();

      this->reset_residency(true, !have_separate_memory_spaces);
    }
  }


  template <typename T>
  template <typename MemorySpace>
  inline auto Mirrored<T>::view() -> value_type *
  {
    this->template prepare_write_access<MemorySpace>();

    return storage<MemorySpace>().data();
  }


  template <typename T>
  template <typename MemorySpace>
  inline auto Mirrored<T>::view() const -> const value_type *
  {
    this->template prepare_read_access<MemorySpace>();

    return storage<MemorySpace>().data();
  }


  template <typename T>
  template <typename MemorySpace>
  inline auto Mirrored<T>::storage() const -> KokkosView<MemorySpace> &
  {
    static_assert(std::is_same_v<MemorySpace, HostSpace> ||
                      std::is_same_v<MemorySpace, DefaultSpace>,
                  "Unexpected memory space");

    if constexpr (have_separate_memory_spaces &&
                  !std::is_same_v<MemorySpace, HostSpace>)
      return default_;
    else
      return host_;
  }


  template <typename T>
  template <typename MemorySpace>
  inline void Mirrored<T>::allocate_storage() const
  {
    if constexpr (is_array)
      storage<MemorySpace>() = KokkosView<MemorySpace>(label_, size_);
    else
      storage<MemorySpace>() = KokkosView<MemorySpace>(label_);
  }


  template <typename T>
  template <typename To, typename From>
  inline void Mirrored<T>::deep_copy_storage() const
  {
    Kokkos::deep_copy(storage<To>(), storage<From>());
  }


  template <typename T>
  template <typename MemorySpace>
  inline void Mirrored<T>::deallocate_storage()
  {
    storage<MemorySpace>() = KokkosView<MemorySpace>();
  }


#endif
} // namespace ryujin

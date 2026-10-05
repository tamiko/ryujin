//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception or LGPL-2.1-or-later
// Copyright (C) 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/compile_time_options.h>
#include <ryujin/base/computing_timer.h>
#include <ryujin/base/convenience_macros.h>
#include <ryujin/base/instrumentation.h>
#include <ryujin/base/simd.h>

#include <deal.II/base/config.h>
#include <deal.II/base/memory_space.h>
#include <deal.II/base/parallel.h>

#include <concepts>
#include <mutex>
#include <string>
#include <type_traits>
#include <vector>

#ifdef WITH_OPENMP
#include <omp.h>
#endif

namespace ryujin
{
  template <typename ScalarNumber, typename Functor, typename... Args>
  inline void cpu_simd_loop(const std::string &region_name [[maybe_unused]],
                            const Functor &body,
                            const unsigned int left,
                            const unsigned int internal,
                            const unsigned int right,
                            Args &&...args)
  {
    Assert(left <= internal && internal <= right,
           dealii::ExcMessage("Invalid index range: it must hold left <= "
                              "internal, internal <= right"));

    if (!region_name.empty()) {
      LIKWID_MARKER_START(region_name.c_str());
    }

    using VA = dealii::VectorizedArray<ScalarNumber>;

    constexpr unsigned int stride_size = VA::size();
    const unsigned int regular =
        left + (internal - left) / stride_size * stride_size;

#if defined(WITH_OPENMP)

    RYUJIN_PRAGMA(omp parallel default(shared))
    {
      RYUJIN_PRAGMA(omp for nowait)
      for (unsigned int i = left; i < regular; i += stride_size)
        body(VA(), std::forward<Args>(args)..., i);

      RYUJIN_PRAGMA(omp for)
      for (unsigned int i = regular; i < right; i += 1)
        body(ScalarNumber(), std::forward<Args>(args)..., i);
    }

#elif defined(WITH_DEAL_II_THREADS)
    {
      Assert((regular - left) % stride_size == 0, dealii::ExcInternalError());
      dealii::parallel::apply_to_subranges(
          0,
          (regular - left) / stride_size,
          [&](const unsigned int begin, const unsigned int end) {
            for (unsigned int i = begin; i < end; ++i)
              body(VA(), std::forward<Args>(args)..., left + stride_size * i);
          },
          1000);

      dealii::parallel::apply_to_subranges(
          regular,
          right,
          [&](const unsigned int begin, const unsigned int end) {
            for (unsigned int i = begin; i < end; ++i)
              body(ScalarNumber(), std::forward<Args>(args)..., i);
          },
          1000);
    }

#else
    {
      for (unsigned int i = left; i < regular; i += stride_size)
        body(VA(), std::forward<Args>(args)..., i);

      for (unsigned int i = regular; i < right; i += 1)
        body(ScalarNumber(), std::forward<Args>(args)..., i);
    }
#endif

    if (!region_name.empty()) {
      LIKWID_MARKER_STOP(region_name.c_str());
    }
  }


  template <typename ScalarNumber, typename Functor, typename... Args>
  inline void gpu_loop(const std::string &region_name,
                       const Functor &body,
                       const unsigned int left,
                       const unsigned int internal [[maybe_unused]],
                       const unsigned int right,
                       Args &&...args)
  {
    DeviceTimer::Scope scope;

    Assert(left <= internal && internal <= right,
           dealii::ExcMessage("Invalid index range: it must hold left <= "
                              "internal, internal <= right"));

    using MemorySpace = dealii::MemorySpace::Default;
    using ExecutionSpace = typename MemorySpace::kokkos_space::execution_space;
    using Policy =
        Kokkos::RangePolicy<ExecutionSpace, Kokkos::IndexType<unsigned int>>;

    const auto exec = ExecutionSpace{};

    if (!region_name.empty()) {
      NVTX_MARKER_START(region_name.c_str());
    }

    Kokkos::parallel_for(
        region_name,
        Policy(exec, left, right),
        KOKKOS_LAMBDA(const unsigned int i) {
          body(ScalarNumber(), args..., i);
        });

    exec.fence();

    if (!region_name.empty()) {
      NVTX_MARKER_STOP(region_name.c_str());
    }
  }


  template <typename MemorySpace,
            typename ScalarNumber,
            typename Functor,
            typename... Args>
  inline void loop(const std::string &region_name,
                   const Functor &body,
                   const unsigned int left,
                   const unsigned int internal,
                   const unsigned int right,
                   Args &&...args)
  {
    using HostSpace = dealii::MemorySpace::Host;
    using DefaultSpace = dealii::MemorySpace::Default;
    static_assert(std::is_same_v<MemorySpace, HostSpace> ||
                      std::is_same_v<MemorySpace, DefaultSpace>,
                  "Unexpected memory space");

    if constexpr (std::is_same_v<MemorySpace, HostSpace>) {
      cpu_simd_loop<ScalarNumber>(region_name,
                                  body,
                                  left,
                                  internal,
                                  right,
                                  std::forward<Args>(args)...);
    } else {
      gpu_loop<ScalarNumber>(region_name,
                             body,
                             left,
                             internal,
                             right,
                             std::forward<Args>(args)...);
    }
  }


  template <typename ElementReducer>
  struct ArrayReducer {
    using scalar_type = typename ElementReducer::value_type;
    using value_type = scalar_type[];

    const unsigned int value_count;

    ArrayReducer(scalar_type *data, const unsigned int n)
        : value_count(n)
        , data_(data)
        , element_reducer_(typename ElementReducer::result_view_type())
    {
    }

    explicit ArrayReducer(std::vector<scalar_type> &values)
        : ArrayReducer(values.data(), static_cast<unsigned int>(values.size()))
    {
    }

    KOKKOS_INLINE_FUNCTION
    void init(scalar_type *values) const
    {
      for (unsigned int k = 0; k < value_count; ++k)
        element_reducer_.init(values[k]);
    }

    KOKKOS_INLINE_FUNCTION
    void join(scalar_type *destination, const scalar_type *source) const
    {
      for (unsigned int k = 0; k < value_count; ++k)
        element_reducer_.join(destination[k], source[k]);
    }

    template <typename Contribution>
      requires std::invocable<const Contribution &, unsigned int>
    KOKKOS_INLINE_FUNCTION void join(scalar_type *destination,
                                     const Contribution &contribution) const
    {
      for (unsigned int k = 0; k < value_count; ++k)
        element_reducer_.join(destination[k], contribution(k));
    }

    scalar_type *reference() const
    {
      return data_;
    }

  private:
    scalar_type *const data_;

    const ElementReducer element_reducer_;
  };


  namespace internal
  {
    template <typename Reducer>
    struct LocalResult {
      static constexpr bool is_array =
          std::is_array_v<typename Reducer::value_type>;
      using scalar_type = std::remove_extent_t<typename Reducer::value_type>;

      std::conditional_t<is_array, std::vector<scalar_type>, scalar_type>
          storage;

      LocalResult(const Reducer &reducer)
      {
        if constexpr (is_array)
          storage.resize(reducer.value_count);
        reducer.init(get());
      }

      std::conditional_t<is_array, scalar_type *, scalar_type &> get()
      {
        if constexpr (is_array)
          return storage.data();
        else
          return storage;
      }

      auto view()
      {
        using Unmanaged = Kokkos::MemoryTraits<Kokkos::Unmanaged>;
        if constexpr (is_array)
          return Kokkos::View<scalar_type *, Kokkos::HostSpace, Unmanaged>(
              storage.data(), storage.size());
        else
          return Kokkos::View<scalar_type, Kokkos::HostSpace, Unmanaged>(
              &storage);
      }
    };


    template <typename Reducer, typename Body>
    struct ReductionFunctor : Reducer {
      const Body body;

      ReductionFunctor(const Reducer &reducer, const Body &body)
          : Reducer(reducer)
          , body(body)
      {
      }

      KOKKOS_INLINE_FUNCTION
      void operator()(const unsigned int i, auto &&local_result) const
      {
        Reducer::join(local_result, body(i));
      }
    };
  } // namespace internal


  template <typename Reducer, typename Functor, typename... Args>
  inline void cpu_reduction_loop(const std::string &region_name
                                 [[maybe_unused]],
                                 const Functor &body,
                                 const Reducer &reducer,
                                 const unsigned int left,
                                 const unsigned int right,
                                 Args &&...args)
  {
    Assert(
        left <= right,
        dealii::ExcMessage("Invalid index range: it must hold left <= right"));

    if (!region_name.empty()) {
      LIKWID_MARKER_START(region_name.c_str());
    }

    using scalar_type = std::remove_extent_t<typename Reducer::value_type>;

#if defined(WITH_OPENMP)

    RYUJIN_PRAGMA(omp parallel default(shared))
    {
      internal::LocalResult<Reducer> local_result(reducer);

      RYUJIN_PRAGMA(omp for nowait)
      for (unsigned int i = left; i < right; ++i)
        reducer.join(local_result.get(),
                     body(scalar_type(), std::forward<Args>(args)..., i));

      RYUJIN_PRAGMA(omp critical)
      reducer.join(reducer.reference(), local_result.get());
    }

#elif defined(WITH_DEAL_II_THREADS)
    {
      std::mutex mutex;

      dealii::parallel::apply_to_subranges(
          left,
          right,
          [&](const unsigned int begin, const unsigned int end) {
            internal::LocalResult<Reducer> local_result(reducer);

            for (unsigned int i = begin; i < end; ++i)
              reducer.join(local_result.get(),
                           body(scalar_type(), std::forward<Args>(args)..., i));

            std::lock_guard<std::mutex> lock(mutex);
            reducer.join(reducer.reference(), local_result.get());
          },
          1000);
    }

#else
    {
      for (unsigned int i = left; i < right; ++i)
        reducer.join(reducer.reference(),
                     body(scalar_type(), std::forward<Args>(args)..., i));
    }
#endif

    if (!region_name.empty()) {
      LIKWID_MARKER_STOP(region_name.c_str());
    }
  }


  template <typename Reducer, typename Functor, typename... Args>
  inline void gpu_reduction_loop(const std::string &region_name,
                                 const Functor &body,
                                 const Reducer &reducer,
                                 const unsigned int left,
                                 const unsigned int right,
                                 Args &&...args)
  {
    DeviceTimer::Scope scope;

    Assert(
        left <= right,
        dealii::ExcMessage("Invalid index range: it must hold left <= right"));

    using scalar_type = std::remove_extent_t<typename Reducer::value_type>;

    using MemorySpace = dealii::MemorySpace::Default;
    using ExecutionSpace = typename MemorySpace::kokkos_space::execution_space;
    using Policy =
        Kokkos::RangePolicy<ExecutionSpace, Kokkos::IndexType<unsigned int>>;

    const auto exec = ExecutionSpace{};

    const auto kernel = KOKKOS_LAMBDA(const unsigned int i)
    {
      return body(scalar_type(), args..., i);
    };

    const auto functor =
        internal::ReductionFunctor<Reducer, decltype(kernel)>(reducer, kernel);

    internal::LocalResult<Reducer> result(reducer);

    if (!region_name.empty()) {
      NVTX_MARKER_START(region_name.c_str());
    }

    Kokkos::parallel_reduce(
        region_name, Policy(exec, left, right), functor, result.view());

    exec.fence();

    if (!region_name.empty()) {
      NVTX_MARKER_STOP(region_name.c_str());
    }

    reducer.join(reducer.reference(), result.get());
  }


  template <typename MemorySpace,
            typename Reducer,
            typename Functor,
            typename... Args>
  inline void reduction_loop(const std::string &region_name,
                             const Functor &body,
                             const Reducer &reducer,
                             const unsigned int left,
                             const unsigned int right,
                             Args &&...args)
  {
    using HostSpace = dealii::MemorySpace::Host;
    using DefaultSpace = dealii::MemorySpace::Default;
    static_assert(std::is_same_v<MemorySpace, HostSpace> ||
                      std::is_same_v<MemorySpace, DefaultSpace>,
                  "Unexpected memory space");

    if constexpr (std::is_same_v<MemorySpace, HostSpace>) {
      cpu_reduction_loop(
          region_name, body, reducer, left, right, std::forward<Args>(args)...);
    } else {
      gpu_reduction_loop(
          region_name, body, reducer, left, right, std::forward<Args>(args)...);
    }
  }
} // namespace ryujin

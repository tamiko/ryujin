//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/compile_time_options.h>

#include <deal.II/base/function.h>

namespace ryujin
{
#ifndef DOXYGEN
  namespace
  {
    template <int dim, typename Number, typename Callable>
    class ToFunction : public dealii::Function<dim, Number>
    {
    public:
      ToFunction(const Callable &callable, const unsigned int k)
          : dealii::Function<dim, Number>(1)
          , callable_(callable)
          , k_(k)
      {
      }

      Number value(const dealii::Point<dim> &point, unsigned int) const override
      {
        return callable_(point)[k_];
      }

    private:
      const Callable callable_;
      const unsigned int k_;
    };
  } // namespace
#endif

  template <int dim, typename Number, typename Callable>
  ToFunction<dim, Number, Callable> to_function(const Callable &callable,
                                                const unsigned int k)
  {
    return {callable, k};
  }


  template <typename FT,
            int problem_dim = FT::dimension,
            typename TT = typename FT::value_type,
            typename T = typename TT::value_type>
  DEAL_II_HOST_DEVICE_ALWAYS_INLINE dealii::Tensor<1, problem_dim, T>
  contract(const FT &flux_ij, const TT &c_ij)
  {
    dealii::Tensor<1, problem_dim, T> result;
    for (unsigned int k = 0; k < problem_dim; ++k)
      result[k] = flux_ij[k] * c_ij;
    return result;
  }


  template <typename FT, int problem_dim = FT::dimension>
  DEAL_II_HOST_DEVICE_ALWAYS_INLINE FT add(const FT &flux_left_ij,
                                           const FT &flux_right_ij)
  {
    FT result;
    for (unsigned int k = 0; k < problem_dim; ++k)
      result[k] = flux_left_ij[k] + flux_right_ij[k];
    return result;
  }
} // namespace ryujin


#ifndef DOXYGEN
namespace
{
  template <typename T>
  class is_dereferenceable
  {
    template <typename C>
    static auto test(...) -> std::false_type;

    template <typename C>
    static auto test(C *) -> decltype(*std::declval<C>(), std::true_type());

  public:
    using type = decltype(test<T>(nullptr));
    static constexpr auto value = type::value;
  };

  template <typename T, typename>
  auto dereference(T &t) -> decltype(dereference(*t)) &;

  template <typename T>
  auto dereference(T &t) -> T &
    requires(!is_dereferenceable<T>::value)
  {
    return t;
  }

  template <typename T>
  auto dereference(T &t) -> decltype(*t) &
    requires is_dereferenceable<T>::value
  {
    return *t;
  }
} /* anonymous namespace */
#endif

#define ACCESSOR_READ_ONLY(member)                                             \
  inline const auto &member() const                                            \
  {                                                                            \
    return dereference(member##_);                                             \
  }


#define ACCESSOR(member)                                                       \
  inline auto &member()                                                        \
  {                                                                            \
    return dereference(member##_);                                             \
  }


#define ACCESSOR_READ_ONLY_NO_DEREFERENCE(member)                              \
  inline const auto &member() const                                            \
  {                                                                            \
    return member##_;                                                          \
  }


#define ACCESSOR_CONTAINER_READ_ONLY(container, member)                        \
  inline const auto &member() const                                            \
  {                                                                            \
    return dereference(dereference(container).member);                         \
  }


#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__) ||               \
    defined(__SYCL_DEVICE_ONLY__)
#define RYUJIN_DEVICE_COMPILATION_PASS
#endif


#define RYUJIN_PRAGMA(x) _Pragma(#x)


#define RYUJIN_UNLIKELY(x) (__builtin_expect(!!(x), 0))

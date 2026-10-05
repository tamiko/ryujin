//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/compile_time_options.h>
#include <ryujin/base/convenience_macros.h>

#include <deal.II/base/tensor.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/vectorization.h>

#include <cmath>


#define AssertThrowSIMD(variable, condition, exception)                        \
  if constexpr (std::is_same<                                                  \
                    typename std::remove_const<decltype(variable)>::type,      \
                    double>::value ||                                          \
                std::is_same<                                                  \
                    typename std::remove_const<decltype(variable)>::type,      \
                    float>::value) {                                           \
    AssertThrow(condition(variable), exception);                               \
  } else {                                                                     \
    for (unsigned int k = 0; k < decltype(variable)::size(); ++k) {            \
      AssertThrow(condition((variable)[k]), exception);                        \
    }                                                                          \
  }


namespace ryujin
{
  template <typename T>
  struct get_value_type {
    using type = T;
  };


  template <typename T, std::size_t width>
  struct get_value_type<dealii::VectorizedArray<T, width>> {
    using type = T;
  };


#ifndef DOXYGEN
  namespace
  {
    template <typename Functor, size_t... Is>
    auto generate_iterators_impl(Functor f, std::index_sequence<Is...>)
        -> std::array<decltype(f(0)), sizeof...(Is)>
    {
      return {{f(Is)...}};
    }
  } /* namespace */
#endif


  template <unsigned int length, typename Functor>
  DEAL_II_ALWAYS_INLINE inline auto generate_iterators(Functor f)
      -> std::array<decltype(f(0)), length>
  {
    return generate_iterators_impl<>(f, std::make_index_sequence<length>());
  }


  template <typename T>
  DEAL_II_ALWAYS_INLINE inline void increment_iterators(T &iterators)
  {
    for (auto &it : iterators)
      it++;
  }

  template <typename Number>
  DEAL_II_HOST_DEVICE_ALWAYS_INLINE Number positive_part(const Number number)
  {
    return std::max(Number(0.), number);
  }


  template <typename Number>
  DEAL_II_HOST_DEVICE_ALWAYS_INLINE Number negative_part(const Number number)
  {
    return -std::min(Number(0.), number);
  }


  template <typename Number>
  DEAL_II_HOST_DEVICE_ALWAYS_INLINE Number
  safe_division(const Number &numerator, const Number &denominator)
  {
    using ScalarNumber = typename get_value_type<Number>::type;
    constexpr ScalarNumber min = std::numeric_limits<ScalarNumber>::min();

    return std::max(numerator, Number(0.)) / std::max(denominator, Number(min));
  }


  template <dealii::SIMDComparison predicate, typename Number>
  DEAL_II_HOST_DEVICE_ALWAYS_INLINE Number
  compare_and_apply_mask(const Number &left,
                         const Number &right,
                         const Number &true_value,
                         const Number &false_value)
  {
    static_assert(std::is_floating_point_v<Number>,
                  "Only scalar number types are allowed");

    bool mask = false;
    if constexpr (predicate == dealii::SIMDComparison::equal)
      mask = (left == right);
    else if constexpr (predicate == dealii::SIMDComparison::not_equal)
      mask = (left != right);
    else if constexpr (predicate == dealii::SIMDComparison::less_than)
      mask = (left < right);
    else if constexpr (predicate == dealii::SIMDComparison::less_than_or_equal)
      mask = (left <= right);
    else if constexpr (predicate == dealii::SIMDComparison::greater_than)
      mask = (left > right);
    else
      mask = (left >= right);

    return mask ? true_value : false_value;
  }


  template <dealii::SIMDComparison predicate, typename T, std::size_t width>
  DEAL_II_ALWAYS_INLINE inline dealii::VectorizedArray<T, width>
  compare_and_apply_mask(const dealii::VectorizedArray<T, width> &left,
                         const dealii::VectorizedArray<T, width> &right,
                         const dealii::VectorizedArray<T, width> &true_value,
                         const dealii::VectorizedArray<T, width> &false_value)
  {
    return dealii::compare_and_apply_mask<predicate>(
        left, right, true_value, false_value);
  }


  template <int N, typename T>
  DEAL_II_HOST_DEVICE_ALWAYS_INLINE T fixed_power(const T x)
  {
    return dealii::Utilities::fixed_power<N, T>(x);
  }


  template <typename T>
  DEAL_II_HOST_DEVICE T pow(const T x, const T b);


  template <typename T, std::size_t width>
  dealii::VectorizedArray<T, width>
  pow(const dealii::VectorizedArray<T, width> x, const T b);


  template <typename T, std::size_t width>
  dealii::VectorizedArray<T, width>
  pow(const dealii::VectorizedArray<T, width> x,
      const dealii::VectorizedArray<T, width> b);


  template <>
  DEAL_II_HOST_DEVICE_ALWAYS_INLINE float pow(const float x, const float b)
  {
#ifdef RYUJIN_DEVICE_COMPILATION_PASS
    return std::pow(x, b);
#elif DEAL_II_COMPILER_VECTORIZATION_LEVEL >= 1 && defined(__SSE2__)
    return pow(dealii::VectorizedArray<float, 4>(x), b)[0];
#else
    return std::pow(x, b);
#endif
  }


  template <>
  DEAL_II_HOST_DEVICE_ALWAYS_INLINE double pow(const double x, const double b)
  {
#ifdef RYUJIN_DEVICE_COMPILATION_PASS
    return std::pow(x, b);
#elif DEAL_II_COMPILER_VECTORIZATION_LEVEL >= 1 && defined(__SSE2__)
    return pow(dealii::VectorizedArray<double, 2>(x), b)[0];
#else
    return std::pow(x, b);
#endif
  }


  enum class Bias {
    none,

    max,

    min
  };


  template <typename T>
  DEAL_II_HOST_DEVICE T fast_pow(const T x,
                                 const T b,
                                 const Bias bias = Bias::none);


  template <typename T, std::size_t width>
  dealii::VectorizedArray<T, width>
  fast_pow(const dealii::VectorizedArray<T, width> x,
           const T b,
           const Bias bias = Bias::none);


  template <typename T, std::size_t width>
  dealii::VectorizedArray<T, width>
  fast_pow(const dealii::VectorizedArray<T, width> x,
           const dealii::VectorizedArray<T, width> b,
           const Bias bias = Bias::none);


  template <>
  DEAL_II_HOST_DEVICE_ALWAYS_INLINE float
  fast_pow(const float x, const float b, [[maybe_unused]] const Bias bias)
  {
#ifdef RYUJIN_DEVICE_COMPILATION_PASS
    return std::pow(x, b);
#elif DEAL_II_COMPILER_VECTORIZATION_LEVEL >= 1 && defined(__SSE2__)
    return fast_pow(dealii::VectorizedArray<float, 4>(x), b, bias)[0];
#else
    return std::pow(x, b);
#endif
  }


  template <>
  DEAL_II_HOST_DEVICE_ALWAYS_INLINE double
  fast_pow(const double x, const double b, [[maybe_unused]] const Bias bias)
  {
#ifdef RYUJIN_DEVICE_COMPILATION_PASS
    return std::pow(static_cast<float>(x), static_cast<float>(b));
#elif DEAL_II_COMPILER_VECTORIZATION_LEVEL >= 1 && defined(__SSE2__)
    return fast_pow(dealii::VectorizedArray<double, 2>(x), b, bias)[0];
#else
    return std::pow(static_cast<float>(x), static_cast<float>(b));
#endif
  }

  template <typename T, typename V>
  DEAL_II_ALWAYS_INLINE inline T read_entry(const V &vector, unsigned int i)
  {
    static_assert(std::is_same_v<typename get_value_type<T>::type,
                                 typename V::value_type>,
                  "type mismatch");
    T result;

    if constexpr (std::is_same_v<T, typename get_value_type<T>::type>) {
      result = vector.local_element(i);
    } else {
      result.load(vector.get_values() + i);
    }

    return result;
  }


  template <typename T, typename T2>
  DEAL_II_ALWAYS_INLINE inline T read_entry(const std::vector<T2> &vector,
                                            unsigned int i)
  {
    if constexpr (std::is_same_v<typename get_value_type<T>::type, T2>) {
      T result;
      if constexpr (std::is_same_v<T, typename get_value_type<T>::type>) {
        result = vector[i];
      } else {
        result.load(vector.data() + i);
      }
      return result;

    } else {
      T result;
      if constexpr (std::is_same_v<T, typename get_value_type<T>::type>) {
        result = vector[i];
      } else {
        for (unsigned int k = 0; k < T::size(); ++k)
          result[k] = vector[i + k];
      }
      return result;
    }
  }


  template <typename T, typename V>
  DEAL_II_ALWAYS_INLINE inline T read_entry(const V &vector,
                                            const unsigned int *js)
  {
    static_assert(std::is_same_v<typename get_value_type<T>::type,
                                 typename V::value_type>,
                  "type mismatch");
    T result;

    if constexpr (std::is_same_v<T, typename get_value_type<T>::type>) {
      result = vector.local_element(js[0]);
    } else {
      result.gather(vector.get_values(), js);
    }

    return result;
  }


  template <typename T, typename T2>
  DEAL_II_ALWAYS_INLINE inline T read_entry(const std::vector<T2> &vector,
                                            const unsigned int *js)
  {
    static_assert(std::is_same_v<typename get_value_type<T>::type, T2>,
                  "type mismatch");
    T result;

    if constexpr (std::is_same_v<T, typename get_value_type<T>::type>) {
      result = vector[js[0]];
    } else {
      result.load(vector.data(), js);
    }

    return result;
  }


  template <typename T, typename V>
  DEAL_II_ALWAYS_INLINE inline void
  write_entry(V &vector, const T &values, unsigned int i)
  {
    static_assert(std::is_same_v<typename get_value_type<T>::type,
                                 typename V::value_type>,
                  "type mismatch");

    if constexpr (std::is_same_v<T, typename get_value_type<T>::type>) {
      vector.local_element(i) = values;
    } else {
      values.store(vector.get_values() + i);
    }
  }


  template <typename T, typename T2>
  DEAL_II_ALWAYS_INLINE inline void
  write_entry(std::vector<T2> &vector, const T &values, unsigned int i)
  {
    if constexpr (std::is_same_v<typename get_value_type<T>::type, T2>) {
      if constexpr (std::is_same_v<T, typename get_value_type<T>::type>) {
        vector[i] = values;
      } else {
        values.store(vector.data() + i);
      }

    } else {
      if constexpr (std::is_same_v<T, typename get_value_type<T>::type>) {
        vector[i] = values;
      } else {
        for (unsigned int k = 0; k < T::size(); ++k)
          vector[i + k] = values[k];
      }
    }
  }


  template <int rank, int dim, std::size_t width, typename Number>
  DEAL_II_ALWAYS_INLINE inline dealii::Tensor<rank, dim, Number>
  serialize_tensor(
      const dealii::Tensor<rank, dim, dealii::VectorizedArray<Number, width>>
          &vectorized,
      const unsigned int k)
  {
    Assert(k < width, dealii::ExcMessage("Index past VectorizedArray width"));
    dealii::Tensor<rank, dim, Number> result;
    if constexpr (rank == 1) {
      for (unsigned int d = 0; d < dim; ++d)
        result[d] = vectorized[d][k];
    } else {
      for (unsigned int d = 0; d < dim; ++d)
        result[d] = serialize_tensor(vectorized[d], k);
    }
    return result;
  }


  template <int rank, int dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline dealii::Tensor<rank, dim, Number>
  serialize_tensor(const dealii::Tensor<rank, dim, Number> &serial,
                   const unsigned int k [[maybe_unused]])
  {
    Assert(k == 0,
           dealii::ExcMessage(
               "The given index k must be zero for a serial tensor"));
    return serial;
  }


  template <int rank, int dim, std::size_t width, typename Number>
  DEAL_II_ALWAYS_INLINE inline void assign_serial_tensor(
      dealii::Tensor<rank, dim, dealii::VectorizedArray<Number, width>> &result,
      const dealii::Tensor<rank, dim, Number> &serial,
      const unsigned int k)
  {
    Assert(k < width, dealii::ExcMessage("Index past VectorizedArray width"));
    if constexpr (rank == 1) {
      for (unsigned int d = 0; d < dim; ++d)
        result[d][k] = serial[d];
    } else {
      for (unsigned int d = 0; d < dim; ++d)
        assign_serial_tensor(result[d], serial[d], k);
    }
  }


  template <int rank, int dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline void
  assign_serial_tensor(dealii::Tensor<rank, dim, Number> &result,
                       const dealii::Tensor<rank, dim, Number> &serial,
                       const unsigned int k [[maybe_unused]])
  {
    Assert(k == 0,
           dealii::ExcMessage(
               "The given index k must be zero for a serial tensor"));

    result = serial;
  }
} // namespace ryujin

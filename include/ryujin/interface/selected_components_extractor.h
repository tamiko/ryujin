//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2024 - 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/convenience_macros.h>
#include <ryujin/interface/parabolic_system.h>
#include <ryujin/linear_algebra/state_vector.h>

#include <deal.II/base/exceptions.h>

#include <algorithm>
#include <string>
#include <vector>

namespace ryujin
{
  namespace Interface
  {
    template <int dim, typename Number = double>
    class SelectedComponentsExtractor
    {
    public:
      using StateVector = Vectors::StateVector<Number>;

      using ScalarHostVector = Vectors::ScalarHostVector<Number>;

      virtual ~SelectedComponentsExtractor() = default;

      ACCESSOR_READ_ONLY(component_names)

      ACCESSOR_READ_ONLY(primitive_component_names)

      ACCESSOR_READ_ONLY(precomputed_names)

      ACCESSOR_READ_ONLY(initial_precomputed_names)

      virtual void prepare(const std::vector<std::string> &selected);

      unsigned int n_selected() const
      {
        return static_cast<unsigned int>(selection_.size());
      }

      virtual std::vector<ScalarHostVector>
      extract(const StateVector &state_vector) const = 0;

      virtual void
      prepare_extraction(const StateVector &state_vector) const = 0;

      virtual std::vector<Number> extract_moments(const unsigned int *indices,
                                                  const Number *masses,
                                                  const unsigned int n_points,
                                                  const unsigned int n_moments,
                                                  Number *values) const = 0;

    protected:
      SelectedComponentsExtractor(
          const std::vector<std::string> &component_names,
          const std::vector<std::string> &primitive_component_names,
          const std::vector<std::string> &precomputed_names,
          const std::vector<std::string> &initial_precomputed_names,
          const ParabolicSystem &parabolic_system,
          const std::vector<std::string> &additional_names);

      const unsigned int primitive_offset_;
      const unsigned int precomputed_offset_;
      const unsigned int initial_offset_;
      const unsigned int parabolic_offset_;
      const unsigned int additional_offset_;

      std::vector<unsigned int> selection_;

      bool read_conserved_ = false;
      bool read_primitive_ = false;
      bool read_precomputed_ = false;
      bool read_initial_ = false;

    private:
      const std::vector<std::string> component_names_;
      const std::vector<std::string> primitive_component_names_;
      const std::vector<std::string> precomputed_names_;
      const std::vector<std::string> initial_precomputed_names_;
      const std::vector<std::string> parabolic_component_names_;
      const std::vector<std::string> additional_names_;
    };


#ifndef DOXYGEN


    template <int dim, typename Number>
    SelectedComponentsExtractor<dim, Number>::SelectedComponentsExtractor(
        const std::vector<std::string> &component_names,
        const std::vector<std::string> &primitive_component_names,
        const std::vector<std::string> &precomputed_names,
        const std::vector<std::string> &initial_precomputed_names,
        const ParabolicSystem &parabolic_system,
        const std::vector<std::string> &additional_names)
        : primitive_offset_(component_names.size())
        , precomputed_offset_(primitive_offset_ +
                              primitive_component_names.size())
        , initial_offset_(precomputed_offset_ + precomputed_names.size())
        , parabolic_offset_(initial_offset_ + initial_precomputed_names.size())
        , additional_offset_(parabolic_offset_ +
                             parabolic_system.component_names().size())
        , component_names_(component_names)
        , primitive_component_names_(primitive_component_names)
        , precomputed_names_(precomputed_names)
        , initial_precomputed_names_(initial_precomputed_names)
        , parabolic_component_names_(parabolic_system.component_names())
        , additional_names_(additional_names)
    {
    }


    template <int dim, typename Number>
    void SelectedComponentsExtractor<dim, Number>::prepare(
        const std::vector<std::string> &selected)
    {
      std::vector<unsigned int> selection;
      selection.reserve(selected.size());

      for (const auto &entry : selected) {
        const auto search = [&](const auto &names, const unsigned int offset) {
          const auto pos = std::find(std::begin(names), std::end(names), entry);
          if (pos == std::end(names))
            return false;
          const unsigned int index = std::distance(std::begin(names), pos);
          selection.push_back(offset + index);
          return true;
        };

        const bool found =
            search(component_names_, 0u) ||
            search(primitive_component_names_, primitive_offset_) ||
            search(precomputed_names_, precomputed_offset_) ||
            search(initial_precomputed_names_, initial_offset_) ||
            search(parabolic_component_names_, parabolic_offset_) ||
            search(additional_names_, additional_offset_);

        AssertThrow(found,
                    dealii::ExcMessage(
                        "Invalid component name: \"" + entry +
                        "\" is not a valid conserved, primitive, precomputed, "
                        "initial, parabolic, or additional component name."));
      }

      const auto selects = [&](const unsigned int begin,
                               const unsigned int end) {
        return std::any_of(
            selection.begin(), selection.end(), [&](const auto offset) {
              return begin <= offset && offset < end;
            });
      };

      read_conserved_ = selects(0u, primitive_offset_);
      read_primitive_ = selects(primitive_offset_, precomputed_offset_);
      read_precomputed_ = selects(precomputed_offset_, initial_offset_);
      read_initial_ = selects(initial_offset_, parabolic_offset_);

      selection_ = std::move(selection);
    }
#endif
  } // namespace Interface
} // namespace ryujin

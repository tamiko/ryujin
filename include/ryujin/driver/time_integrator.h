//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2022 - 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/compile_time_options.h>

#include <ryujin/base/convenience_macros.h>
#include <ryujin/base/mpi_ensemble.h>
#include <ryujin/base/observer_pointer.h>
#include <ryujin/base/patterns_conversion.h>
#include <ryujin/discretization/offline_data.h>
#include <ryujin/interface/hyperbolic_module.h>
#include <ryujin/interface/parabolic_module.h>
#include <ryujin/interface/parabolic_system.h>

#include <numbers>

namespace ryujin
{
  enum class CFLRecoveryStrategy {
    none,

    bang_bang_control,

    cruise_control,
  };


  enum class TimeSteppingScheme {
    ssprk_22,
    ssprk_33,

    erk_11,

    erk_22,

    erk_33,

    erk_43,

    erk_54,

    strang_ssprk_33_cn,

    strang_erk_33_cn,

    strang_erk_43_cn,

    imex_11,

    imex_22,

    imex_33,
  };
} // namespace ryujin

#ifndef DOXYGEN
DECLARE_ENUM(
    ryujin::CFLRecoveryStrategy,
    LIST({ryujin::CFLRecoveryStrategy::none, "none"},
         {ryujin::CFLRecoveryStrategy::bang_bang_control, "bang bang control"},
         {ryujin::CFLRecoveryStrategy::cruise_control, "cruise control"}, ));

DECLARE_ENUM(
    ryujin::TimeSteppingScheme,
    LIST({ryujin::TimeSteppingScheme::ssprk_22, "ssprk 22"},
         {ryujin::TimeSteppingScheme::ssprk_33, "ssprk 33"},
         {ryujin::TimeSteppingScheme::erk_11, "erk 11"},
         {ryujin::TimeSteppingScheme::erk_22, "erk 22"},
         {ryujin::TimeSteppingScheme::erk_33, "erk 33"},
         {ryujin::TimeSteppingScheme::erk_43, "erk 43"},
         {ryujin::TimeSteppingScheme::erk_54, "erk 54"},
         {ryujin::TimeSteppingScheme::strang_ssprk_33_cn, "strang ssprk 33 cn"},
         {ryujin::TimeSteppingScheme::strang_erk_33_cn, "strang erk 33 cn"},
         {ryujin::TimeSteppingScheme::strang_erk_43_cn, "strang erk 43 cn"},
         {ryujin::TimeSteppingScheme::imex_11, "imex 11"},
         {ryujin::TimeSteppingScheme::imex_22, "imex 22"},
         {ryujin::TimeSteppingScheme::imex_33, "imex 33"}));
#endif

namespace ryujin
{
  template <int dim, typename Number = double>
  class TimeIntegrator final : public dealii::ParameterAcceptor
  {
  public:
    using StateVector = Vectors::StateVector<Number>;

    TimeIntegrator(
        const MPIEnsemble &mpi_ensemble,
        const OfflineData<dim, Number> &offline_data,
        const Interface::HyperbolicModule<dim, Number> &hyperbolic_module,
        const Interface::ParabolicModule<dim, Number> &parabolic_module,
        const Interface::ParabolicSystem &parabolic_system,
        const std::string &subsection = "/TimeIntegrator");

    void prepare();

    void prepare_state_vector(StateVector &state_vector, Number t) const;

    Number step(StateVector &state_vector,
                Number t,
                Number t_final = std::numeric_limits<Number>::max());

    ACCESSOR_READ_ONLY(time_stepping_scheme);

    ACCESSOR_READ_ONLY(efficiency);

  private:
    Number cfl_min_;
    Number cfl_max_;

    CFLRecoveryStrategy cfl_recovery_strategy_;

    Number acceptable_tau_max_ratio_;
    Number tau_max_;

    TimeSteppingScheme time_stepping_scheme_;
    Number efficiency_;

    const MPIEnsemble &mpi_ensemble_;

    dealii::ObserverPointer<const OfflineData<dim, Number>> offline_data_;
    dealii::ObserverPointer<const Interface::HyperbolicModule<dim, Number>>
        hyperbolic_module_;
    dealii::ObserverPointer<const Interface::ParabolicModule<dim, Number>>
        parabolic_module_;
    dealii::ObserverPointer<const Interface::ParabolicSystem> parabolic_system_;

    std::vector<StateVector> temp_;

    Number step_ssprk_22(StateVector &state_vector, Number t, Number tau_max);

    Number step_ssprk_33(StateVector &state_vector, Number t, Number tau_max);

    Number step_erk_11(StateVector &state_vector, Number t, Number tau_max);

    Number step_erk_22(StateVector &state_vector, Number t, Number tau_max);

    Number step_erk_33(StateVector &state_vector, Number t, Number tau_max);

    Number step_erk_43(StateVector &state_vector, Number t, Number tau_max);

    Number step_erk_54(StateVector &state_vector, Number t, Number tau_max);

    Number step_strang_ssprk_33_cn(StateVector &state_vector,
                                   Number t,
                                   Number tau_max);

    Number
    step_strang_erk_33_cn(StateVector &state_vector, Number t, Number tau_max);

    Number
    step_strang_erk_43_cn(StateVector &state_vector, Number t, Number tau_max);

    Number step_imex_11(StateVector &state_vector, Number t, Number tau_max);

    Number step_imex_22(StateVector &state_vector, Number t, Number tau_max);

    Number step_imex_33(StateVector &state_vector, Number t, Number tau_max);
  };

} /* namespace ryujin */

//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2021 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include "convenience_macros.h"
#include "discretization.h"
#include "patterns_conversion.h"
#include "simd.h"

#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/tensor.h>

#include <array>
#include <functional>

namespace ryujin
{
  /**
   * The chosen problem type
   */
  enum class ProblemType {
    /**
     * The compressible Euler equations
     */
    euler,

    /**
     * The compressible Navier-Stokes equations
     */
    navier_stokes,
  };
} // namespace ryujin

DECLARE_ENUM(ryujin::ProblemType,
             LIST({ryujin::ProblemType::euler, "Euler"},
                  {ryujin::ProblemType::navier_stokes, "Navier Stokes"}));

namespace ryujin
{
  /**
   * Description of a @p dim dimensional hyperbolic conservation law.
   *
   * We have a (2 + dim) dimensional state space \f$[\rho, \textbf m,
   * E]\f$, where \f$\rho\f$ denotes the density, \f$\textbf m\f$ is the
   * momentum, and \f$E\f$ is the total energy.
   *
   * @ingroup EulerModule
   */
  class ProblemDescription final : public dealii::ParameterAcceptor
  {
  public:
    /**
     * The dimension of the state space.
     */
    template <int dim>
    static constexpr unsigned int problem_dimension = 2 + dim;

    /**
     * The storage type used for a (conserved) state vector \f$\boldsymbol U\f$.
     */
    template <int dim, typename Number>
    using state_type = dealii::Tensor<1, problem_dimension<dim>, Number>;

    /**
     * An array holding all component names of the conserved state as a string.
     */
    template <int dim>
    static const std::array<std::string, dim + 2> component_names;

    /**
     * The storage type used for a primitive state vector.
     */
    template <int dim, typename Number>
    using primitive_state_type =
        dealii::Tensor<1, problem_dimension<dim>, Number>;

    /**
     * An array holding all component names of the primitive state as a string.
     */
    template <int dim>
    static const std::array<std::string, dim + 2> primitive_component_names;

    /**
     * The storage type used for the flux \f$\mathbf{f}\f$.
     */
    template <int dim, typename Number>
    using flux_type = dealii::
        Tensor<1, problem_dimension<dim>, dealii::Tensor<1, dim, Number>>;

    /**
     * An enum describing the equation of state.
     */
    enum class EquationOfState {
      /**
       * Ideal polytropic gas equation of state described by the specific
       * entropy
       * \f{align}
       *   s(\rho,e) - s_0 =
       *   \log\left(e^{1/(\gamma-1)}\,\rho^{-1}\right).
       * \f}
       */
      ideal_gas,
      /**
       * Van der Waals gas equation of state described by the specific
       * entropy
       * \f{align}
       *   s(\rho,e) - s_0 =
       *   \log\left(e^{1/(\gamma-1)}\,\left(\rho^{-1}-b\right)\right).
       * \f}
       */
      van_der_waals
    };

    /**
     * Constructor.
     */
    ProblemDescription(const std::string &subsection = "ProblemDescription");

    /**
     * Callback for ParameterAcceptor::initialize(). After we read in
     * configuration parameters from the parameter file we have to do some
     * (minor) preparatory work in this class to precompute some common
     * quantities. Do this with a callback.
     */
    void parse_parameters_callback();

    /**
     * @name ProblemDescription compile time options
     */
    //@{

    /**
     * Selected equation of state.
     *
     * @ingroup CompileTimeOptions
     */
    static constexpr EquationOfState equation_of_state_ =
        EquationOfState::ideal_gas;

    //@}
    /**
     * @name Computing derived physical quantities.
     */
    //@{

    /**
     * For a given (2+dim dimensional) state vector <code>U</code>, return
     * the density <code>U[0]</code>
     */
    template <int problem_dim, typename Number>
    static Number density(const dealii::Tensor<1, problem_dim, Number> &U);

    /**
     * For a given (2+dim dimensional) state vector <code>U</code>, return
     * the momentum vector <code>[U[1], ..., U[1+dim]]</code>.
     */
    template <int problem_dim, typename Number>
    static dealii::Tensor<1, problem_dim - 2, Number>
    momentum(const dealii::Tensor<1, problem_dim, Number> &U);

    /**
     * For a given (2+dim dimensional) state vector <code>U</code>, return
     * the total energy <code>U[1+dim]</code>
     */
    template <int problem_dim, typename Number>
    static Number total_energy(const dealii::Tensor<1, problem_dim, Number> &U);


    /**
     * For a given (2+dim dimensional) state vector <code>U</code>, compute
     * and return the internal energy \f$\varepsilon = (\rho e)\f$.
     */
    template <int problem_dim, typename Number>
    static Number
    internal_energy(const dealii::Tensor<1, problem_dim, Number> &U);


    /**
     * For a given (2+dim dimensional) state vector <code>U</code>, compute
     * and return the derivative of the internal energy
     * \f$\varepsilon = (\rho e)\f$.
     */
    template <int problem_dim, typename Number>
    static dealii::Tensor<1, problem_dim, Number>
    internal_energy_derivative(const dealii::Tensor<1, problem_dim, Number> &U);


    /**
     * For a given (2+dim dimensional) state vector <code>U</code>, compute
     * and return the pressure \f$p\f$.
     *
     * We assume that the pressure is given by a polytropic equation of
     * state, i.e.,
     * \f[
     *   p = \frac{\gamma - 1}{1 - b*\rho}\; (\rho e)
     * \f]
     *
     * @note If you want to set the covolume paramete @ref b_ to nonzero
     * you have to enable the @ref covolume_ compile-time option.
     */
    template <int problem_dim, typename Number>
    Number pressure(const dealii::Tensor<1, problem_dim, Number> &U) const;


    /**
     * For a given (2+dim dimensional) state vector <code>U</code>, compute
     * the (physical) speed of sound:
     * \f[
     *   c^2 = \frac{\gamma * p}{\rho\;(1 - b * \rho)}
     * \f]
     */
    template <int problem_dim, typename Number>
    Number
    speed_of_sound(const dealii::Tensor<1, problem_dim, Number> &U) const;


    /**
     * For a given (2+dim dimensional) state vector <code>U</code>, compute
     * and return the (scaled) specific entropy
     * \f[
     *   e^{(\gamma-1)s} = \frac{\rho\,e}{\rho^\gamma}
     *   (1 - b * \rho)^(\gamma -1).
     * \f]
     */
    template <int problem_dim, typename Number>
    Number
    specific_entropy(const dealii::Tensor<1, problem_dim, Number> &U) const;


    /**
     * For a given (2+dim dimensional) state vector <code>U</code>, compute
     * and return the Harten-type entropy
     * \f[
     *   \eta = (\rho^2 e) ^ {1 / (\gamma + 1)}.
     * \f]
     */
    template <int problem_dim, typename Number>
    Number
    harten_entropy(const dealii::Tensor<1, problem_dim, Number> &U) const;


    /**
     * For a given (2+dim dimensional) state vector <code>U</code>, compute
     * and return the derivative \f$\eta'\f$ of the Harten-type entropy
     * \f[
     *   \eta = (\rho^2 e) ^ {1 / (\gamma + 1)}.
     * \f]
     */
    template <int problem_dim, typename Number>
    dealii::Tensor<1, problem_dim, Number> harten_entropy_derivative(
        const dealii::Tensor<1, problem_dim, Number> &U) const;


    /**
     * For a given (2+dim dimensional) state vector <code>U</code>, compute
     * and return the entropy \f$\eta = p^{1/\gamma}\f$.
     */
    template <int problem_dim, typename Number>
    Number
    mathematical_entropy(const dealii::Tensor<1, problem_dim, Number> U) const;


    /**
     * For a given (2+dim dimensional) state vector <code>U</code>, compute
     * and return the derivative \f$\eta'\f$ of the entropy \f$\eta =
     * p^{1/\gamma}\f$.
     */
    template <int problem_dim, typename Number>
    dealii::Tensor<1, problem_dim, Number> mathematical_entropy_derivative(
        const dealii::Tensor<1, problem_dim, Number> U) const;

    /**
     * Returns whether the state @ref U is admissible. If @ref U is a
     * vectorized state then @ref U is admissible if all vectorized values
     * are admissible.
     */
    template <int problem_dim, typename Number>
    bool is_admissible(const dealii::Tensor<1, problem_dim, Number> &U) const;

    //@}
    /**
     * @name Special functions for boundary states
     */
    //@{

    /**
     * For a given state @p U and normal direction @p normal returns the
     * n-th pair of left and right eigenvectors of the linearized normal
     * flux.
     */
    template <int component, int problem_dim, typename Number>
    std::array<dealii::Tensor<1, problem_dim, Number>, 2>
    linearized_eigenvector(
        const dealii::Tensor<1, problem_dim, Number> &U,
        const dealii::Tensor<1, problem_dim - 2, Number> &normal) const;

    /**
     * Decomposes a given state @p U into Riemann invariants and then
     * replaces the first or second Riemann characteristic from the one
     * taken from @p U_bar state. Note that the @p U_bar state is just the
     * prescribed dirichlet values.
     */
    template <int component, int problem_dim, typename Number>
    dealii::Tensor<1, problem_dim, Number> prescribe_riemann_characteristic(
        const dealii::Tensor<1, problem_dim, Number> &U,
        const dealii::Tensor<1, problem_dim, Number> &U_bar,
        const dealii::Tensor<1, problem_dim - 2, Number> &normal) const;

    /**
     * Apply boundary conditions.
     *
     * For the compressible Euler equations we have:
     *
     *  - Dirichlet boundary conditions by prescribing the return value of
     *    get_dirichlet_data() as is.
     *
     *  - Slip boundary conditions where we remove the normal component of
     *    the momentum.
     *
     *  - No slip boundary conditions where we set the momentum to 0.
     *
     *  - "Dynamic boundary" conditions that prescribe different Riemann
     *    invariants from the return value of get_dirichlet_data()
     *    depending on the flow state (supersonic versus subsonic, outflow
     *    versus inflow).
     */
    template <int problem_dim, typename Number, typename Lambda>
    dealii::Tensor<1, problem_dim, Number>
    apply_boundary_conditions(dealii::types::boundary_id id,
                              dealii::Tensor<1, problem_dim, Number> U,
                              const dealii::Tensor<1, problem_dim - 2> &normal,
                              Lambda get_dirichlet_data) const;

    //@}
    /**
     * @name Computing fluxes.
     */
    //@{


    /**
     * Given a state @p U compute the flux
     * \f[
     * \begin{pmatrix}
     *   \textbf m \\
     *   \textbf v\otimes \textbf m + p\mathbb{I}_d \\
     *   \textbf v(E+p)
     * \end{pmatrix},
     * \f]
     */
    template <int problem_dim, typename Number>
    flux_type<problem_dim - 2, Number>
    f(const dealii::Tensor<1, problem_dim, Number> &U) const;

    //@}
    /**
     * @name State transformations (primitive states, expanding
     * dimensionality, etc.)
     */
    //@{

    /*
     * Given a state vector associated with @ref dim2 spatial dimensions
     * return an "expanded" version of the state vector associated with
     * @ref dim1 spatial dimensions where the momentum vector is projected
     * onto the first @ref dim2 unit directions of the @ref dim1
     * dimensional euclidean space.
     *
     * @precondition dim1 has to be larger or equal than dim2.
     */
    template <int dim1,
              int prob_dim2,
              typename Number,
              typename = typename std::enable_if<(dim1 + 2 >= prob_dim2)>::type>
    state_type<dim1, Number>
    expand_state(const dealii::Tensor<1, prob_dim2, Number> &state) const;

    /*
     * Given a primitive state [rho, u_1, ..., u_d, p] return a conserved
     * state
     */
    template <int problem_dim, typename Number>
    dealii::Tensor<1, problem_dim, Number> from_primitive_state(
        const dealii::Tensor<1, problem_dim, Number> &primitive_state) const;

    /*
     * Given a conserved state return a primitive state [rho, u_1, ..., u_d, p]
     */
    template <int problem_dim, typename Number>
    dealii::Tensor<1, problem_dim, Number> to_primitive_state(
        const dealii::Tensor<1, problem_dim, Number> &primitive_state) const;

    //@}

  private:
    /**
     * @name Run time options
     */
    //@{

    ProblemType problem_type_;
    ACCESSOR_READ_ONLY(problem_type)

    double gamma_;
    ACCESSOR_READ_ONLY(gamma)

    double b_;
    ACCESSOR_READ_ONLY(b)

    double mu_;
    ACCESSOR_READ_ONLY(mu)

    double lambda_;
    ACCESSOR_READ_ONLY(lambda)

    double cv_inverse_kappa_;
    ACCESSOR_READ_ONLY(cv_inverse_kappa)

    //@}
    /**
     * @name Precomputed scalar quantitites
     */
    //@{
    double gamma_inverse_;
    double gamma_plus_one_inverse_;

    //@}
  };

  /* Inline definitions */

  template <int problem_dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline Number
  ProblemDescription::density(const dealii::Tensor<1, problem_dim, Number> &U)
  {
    return U[0];
  }


  template <int problem_dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline dealii::Tensor<1, problem_dim - 2, Number>
  ProblemDescription::momentum(const dealii::Tensor<1, problem_dim, Number> &U)
  {
    constexpr int dim = problem_dim - 2;

    dealii::Tensor<1, dim, Number> result;
    for (unsigned int i = 0; i < dim; ++i)
      result[i] = U[1 + i];
    return result;
  }


  template <int problem_dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline Number ProblemDescription::total_energy(
      const dealii::Tensor<1, problem_dim, Number> &U)
  {
    constexpr int dim = problem_dim - 2;
    return U[1 + dim];
  }


  template <int problem_dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline Number ProblemDescription::internal_energy(
      const dealii::Tensor<1, problem_dim, Number> &U)
  {
    /*
     * rho e = (E - 1/2*m^2/rho)
     */

    constexpr int dim = problem_dim - 2;
    using ScalarNumber = typename get_value_type<Number>::type;

    const Number rho_inverse = ScalarNumber(1.) / U[0];
    const auto m = momentum(U);
    const Number E = U[dim + 1];
    return E - ScalarNumber(0.5) * m.norm_square() * rho_inverse;
  }


  template <int problem_dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline dealii::Tensor<1, problem_dim, Number>
  ProblemDescription::internal_energy_derivative(
      const dealii::Tensor<1, problem_dim, Number> &U)
  {
    /*
     * With
     *   rho e = E - 1/2 |m|^2 / rho
     * we get
     *   (rho e)' = (1/2m^2/rho^2, -m/rho , 1 )^T
     */

    constexpr int dim = problem_dim - 2;
    using ScalarNumber = typename get_value_type<Number>::type;

    const Number rho_inverse = ScalarNumber(1.) / U[0];
    const auto u = momentum(U) * rho_inverse;

    dealii::Tensor<1, problem_dim, Number> result;

    result[0] = ScalarNumber(0.5) * u.norm_square();
    for (unsigned int i = 0; i < dim; ++i) {
      result[1 + i] = -u[i];
    }
    result[dim + 1] = ScalarNumber(1.);

    return result;
  }


  template <int problem_dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline Number ProblemDescription::pressure(
      const dealii::Tensor<1, problem_dim, Number> &U) const
  {
    /* p = (gamma - 1) / (1 - b * rho) * (rho e) */

    using ScalarNumber = typename get_value_type<Number>::type;
    return ScalarNumber(gamma_ - 1.) * internal_energy(U);
  }


  template <int problem_dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline Number ProblemDescription::speed_of_sound(
      const dealii::Tensor<1, problem_dim, Number> &U) const
  {
    /* c^2 = gamma * p / rho / (1 - b * rho) */

    using ScalarNumber = typename get_value_type<Number>::type;

    const Number rho_inverse = ScalarNumber(1.) / U[0];
    const Number p = pressure(U);
    return std::sqrt(gamma_ * p * rho_inverse);
  }


  template <int problem_dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline Number ProblemDescription::specific_entropy(
      const dealii::Tensor<1, problem_dim, Number> &U) const
  {
    /* exp((gamma - 1)s) = (rho e) / rho ^ gamma */

    using ScalarNumber = typename get_value_type<Number>::type;

    const auto rho_inverse = ScalarNumber(1.) / U[0];
    return internal_energy(U) * ryujin::pow(rho_inverse, gamma_);
  }


  template <int problem_dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline Number ProblemDescription::harten_entropy(
      const dealii::Tensor<1, problem_dim, Number> &U) const
  {
    /* rho^2 e = \rho E - 1/2*m^2 */

    constexpr int dim = problem_dim - 2;
    using ScalarNumber = typename get_value_type<Number>::type;

    const Number rho = U[0];
    const auto m = momentum(U);
    const Number E = U[dim + 1];

    const Number rho_rho_e = rho * E - ScalarNumber(0.5) * m.norm_square();
    return ryujin::pow(rho_rho_e, gamma_plus_one_inverse_);
  }


  template <int problem_dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline dealii::Tensor<1, problem_dim, Number>
  ProblemDescription::harten_entropy_derivative(
      const dealii::Tensor<1, problem_dim, Number> &U) const
  {
    /*
     * With
     *   eta = (rho^2 e) ^ 1/(gamma+1)
     *   rho^2 e = rho * E - 1/2 |m|^2
     *
     * we get
     *
     *   eta' = 1/(gamma+1) * (rho^2 e) ^ -gamma/(gamma+1) * ( E , -m , rho )^T
     */

    constexpr int dim = problem_dim - 2;
    using ScalarNumber = typename get_value_type<Number>::type;

    const Number rho = U[0];
    const auto m = momentum(U);
    const Number E = U[dim + 1];

    const Number rho_rho_e = rho * E - ScalarNumber(0.5) * m.norm_square();

    const auto factor =
        gamma_plus_one_inverse_ *
        ryujin::pow(rho_rho_e, -gamma_ * gamma_plus_one_inverse_);

    dealii::Tensor<1, problem_dim, Number> result;

    result[0] = factor * E;
    for (unsigned int i = 0; i < dim; ++i) {
      result[1 + i] = -factor * m[i];
    }
    result[dim + 1] = factor * rho;

    return result;
  }


  template <int problem_dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline Number ProblemDescription::mathematical_entropy(
      const dealii::Tensor<1, problem_dim, Number> U) const
  {
    const auto p = pressure(U);
    return ryujin::pow(p, gamma_inverse_);
  }


  template <int problem_dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline dealii::Tensor<1, problem_dim, Number>
  ProblemDescription::mathematical_entropy_derivative(
      const dealii::Tensor<1, problem_dim, Number> U) const
  {
    /*
     * With
     *   eta = p ^ (1/gamma)
     *   p = (gamma - 1) * (rho e)
     *   rho e = E - 1/2 |m|^2 / rho
     *
     * we get
     *
     *   eta' = (gamma - 1)/gamma p ^(1/gamma - 1) *
     *
     *     (1/2m^2/rho^2 , -m/rho , 1 )^T
     */

    constexpr int dim = problem_dim - 2;
    using ScalarNumber = typename get_value_type<Number>::type;

    const Number &rho = U[0];
    const Number rho_inverse = ScalarNumber(1.) / rho;
    const auto u = momentum(U) * rho_inverse;
    const auto p = pressure(U);

    const auto factor = (gamma_ - ScalarNumber(1.0)) * gamma_inverse_ *
                        ryujin::pow(p, gamma_inverse_ - ScalarNumber(1.));

    dealii::Tensor<1, problem_dim, Number> result;

    result[0] = factor * ScalarNumber(0.5) * u.norm_square();
    result[dim + 1] = factor;
    for (unsigned int i = 0; i < dim; ++i) {
      result[1 + i] = -factor * u[i];
    }

    return result;
  }


  template <int problem_dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline bool ProblemDescription::is_admissible(
      const dealii::Tensor<1, problem_dim, Number> &U) const
  {
    const auto rho_new = density(U);
    const auto e_new = internal_energy(U);
    const auto s_new = specific_entropy(U);

    constexpr auto gt = dealii::SIMDComparison::greater_than;
    using T = Number;
    const auto test =
        dealii::compare_and_apply_mask<gt>(rho_new, T(0.), T(0.), T(-1.)) + //
        dealii::compare_and_apply_mask<gt>(e_new, T(0.), T(0.), T(-1.)) +   //
        dealii::compare_and_apply_mask<gt>(s_new, T(0.), T(0.), T(-1.));

#ifdef DEBUG_OUTPUT
    if (test != Number(0.)) {
      std::cout << std::fixed << std::setprecision(16);
      std::cout << "Bounds violation: Negative state [rho, e, s] detected!\n";
      std::cout << "\t\trho: " << rho_new << "\n";
      std::cout << "\t\tint: " << e_new << "\n";
      std::cout << "\t\tent: " << s_new << "\n" << std::endl;
    }
#endif

    return (test == Number(0.));
  }


  template <int component, int problem_dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline std::
      array<dealii::Tensor<1, problem_dim, Number>, 2>
      ProblemDescription::linearized_eigenvector(
          const dealii::Tensor<1, problem_dim, Number> &U,
          const dealii::Tensor<1, problem_dim - 2, Number> &normal) const
  {
    static_assert(component == 1 || component == problem_dim,
                  "Only first and last eigenvectors implemented");

    constexpr int dim = problem_dim - 2;

    const auto rho = density(U);
    const auto m = momentum(U);
    const auto v = m / rho;
    const auto a = speed_of_sound(U);
    const auto gamma = this->gamma();

    state_type<dim, Number> b;
    state_type<dim, Number> c;

    const auto e_k = 0.5 * v.norm_square();

    switch (component) {
    case 1:
      b[0] = (gamma - 1.) * e_k + a * v * normal;
      for (unsigned int i = 0; i < dim; ++i)
        b[1 + i] = (1. - gamma) * v[i] - a * normal[i];
      b[dim + 1] = gamma - 1.;
      b /= 2. * a * a;

      c[0] = 1.;
      for (unsigned int i = 0; i < dim; ++i)
        c[1 + i] = v[i] - a * normal[i];
      c[dim + 1] = a * a / (gamma - 1) + e_k - a * (v * normal);

      return {b, c};

    case problem_dim:
      b[0] = (gamma - 1.) * e_k - a * v * normal;
      for (unsigned int i = 0; i < dim; ++i)
        b[1 + i] = (1. - gamma) * v[i] + a * normal[i];
      b[dim + 1] = gamma - 1.;
      b /= 2. * a * a;

      c[0] = 1.;
      for (unsigned int i = 0; i < dim; ++i)
        c[1 + i] = v[i] + a * normal[i];
      c[dim + 1] = a * a / (gamma - 1) + e_k + a * (v * normal);

      return {b, c};
    }

    __builtin_unreachable();
  }


  template <int component, int problem_dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline dealii::Tensor<1, problem_dim, Number>
  ProblemDescription::prescribe_riemann_characteristic(
      const dealii::Tensor<1, problem_dim, Number> &U,
      const dealii::Tensor<1, problem_dim, Number> &U_bar,
      const dealii::Tensor<1, problem_dim - 2, Number> &normal) const
  {
    static_assert(component == 1 || component == 2,
                  "component has to be 1 or 2");

    constexpr int dim = problem_dim - 2;

    const auto m = momentum(U);
    const auto rho = density(U);
    const auto a = speed_of_sound(U);
    const auto vn = m * normal / rho;

    const auto m_bar = momentum(U_bar);
    const auto rho_bar = density(U_bar);
    const auto a_bar = speed_of_sound(U_bar);
    const auto vn_bar = m_bar * normal / rho_bar;

    /* First Riemann characteristic: v* n - 2 / (gamma - 1) * a */

    const auto R_1 = component == 1 ? vn_bar - 2. * a_bar / (gamma_ - 1.)
                                    : vn - 2. * a / (gamma_ - 1.);

    /* Second Riemann characteristic: v* n + 2 / (gamma - 1) * a */

    const auto R_2 = component == 2 ? vn_bar + 2. * a_bar / (gamma_ - 1.)
                                    : vn + 2. * a / (gamma_ - 1.);

    const auto p = pressure(U);
    const auto s = p / ryujin::pow(rho, gamma_);

    const auto vperp = m / rho - vn * normal;

    const auto vn_new = 0.5 * (R_1 + R_2);

    auto rho_new =
        1. / (gamma_ * s) * ryujin::pow((gamma_ - 1.) / 4. * (R_2 - R_1), 2);
    rho_new = ryujin::pow(rho_new, 1. / (gamma_ - 1.));

    const auto p_new = s * std::pow(rho_new, gamma_);

    state_type<dim, Number> U_new;
    U_new[0] = rho_new;
    for (unsigned int d = 0; d < dim; ++d) {
      U_new[1 + d] = rho_new * (vn_new * normal + vperp)[d];
    }
    U_new[1 + dim] = p_new / (gamma_ - 1.) +
                     0.5 * rho_new * (vn_new * vn_new + vperp.norm_square());

    return U_new;
  }


  template <int problem_dim, typename Number, typename Lambda>
  DEAL_II_ALWAYS_INLINE inline dealii::Tensor<1, problem_dim, Number>
  ProblemDescription::apply_boundary_conditions(
      dealii::types::boundary_id id,
      dealii::Tensor<1, problem_dim, Number> U,
      const dealii::Tensor<1, problem_dim - 2> &normal,
      Lambda get_dirichlet_data) const
  {
    constexpr auto dim = problem_dim - 2;

    if (id == Boundary::dirichlet) {
      U = get_dirichlet_data();

    } else if (id == Boundary::slip) {
      auto m = momentum(U);
      m -= 1. * (m * normal) * normal;
      for (unsigned int k = 0; k < dim; ++k)
        U[k + 1] = m[k];

    } else if (id == Boundary::no_slip) {
      for (unsigned int k = 0; k < dim; ++k)
        U[k + 1] = Number(0.);

    } else if (id == Boundary::dynamic) {
      /*
       * On dynamic boundary conditions, we distinguish four cases:
       *
       *  - supersonic inflow: prescribe full state
       *  - subsonic inflow:
       *      decompose into Riemann invariants and leave R_2
       *      characteristic untouched.
       *  - supersonic outflow: do nothing
       *  - subsonic outflow:
       *      decompose into Riemann invariants and prescribe incoming
       *      R_1 characteristic.
       */
      const auto m = momentum(U);
      const auto rho = density(U);
      const auto a = speed_of_sound(U);
      const auto vn = m * normal / rho;

      /* Supersonic inflow: */
      if (vn < -a) {
        U = get_dirichlet_data();
      }

      /* Subsonic inflow: */
      if (vn >= -a && vn <= 0.) {
        const auto U_dirichlet = get_dirichlet_data();
        U = prescribe_riemann_characteristic<2>(U_dirichlet, U, normal);
      }

      /* Subsonic outflow: */
      if (vn > 0. && vn <= a) {
        const auto U_dirichlet = get_dirichlet_data();
        U = prescribe_riemann_characteristic<1>(U, U_dirichlet, normal);
      }

      /* Supersonic outflow: do nothing, i.e., keep U as is */
    }

    return U;
  }


  template <int problem_dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline ProblemDescription::flux_type<problem_dim - 2,
                                                             Number>
  ProblemDescription::f(const dealii::Tensor<1, problem_dim, Number> &U) const
  {
    constexpr int dim = problem_dim - 2;
    using ScalarNumber = typename get_value_type<Number>::type;

    const Number rho_inverse = ScalarNumber(1.) / U[0];
    const auto m = momentum(U);
    const auto p = pressure(U);
    const Number E = U[dim + 1];

    flux_type<dim, Number> result;

    result[0] = m;
    for (unsigned int i = 0; i < dim; ++i) {
      result[1 + i] = m * (m[i] * rho_inverse);
      result[1 + i][i] += p;
    }
    result[dim + 1] = m * (rho_inverse * (E + p));

    return result;
  }


  template <int dim1, int prob_dim2, typename Number, typename>
  ProblemDescription::state_type<dim1, Number> ProblemDescription::expand_state(
      const dealii::Tensor<1, prob_dim2, Number> &state) const
  {
    constexpr auto dim2 = prob_dim2 - 2;

    state_type<dim1, Number> result;
    result[0] = state[0];
    result[dim1 + 1] = state[dim2 + 1];
    for (unsigned int i = 1; i < dim2 + 1; ++i)
      result[i] = state[i];

    return result;
  }


  template <int problem_dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline dealii::Tensor<1, problem_dim, Number>
  ProblemDescription::from_primitive_state(
      const dealii::Tensor<1, problem_dim, Number> &primitive_state) const
  {
    constexpr auto dim = problem_dim - 2;

    const auto &rho = primitive_state[0];
    /* extract velocity: */
    const auto u = /*SIC!*/ momentum(primitive_state);
    const auto &p = primitive_state[dim + 1];

    auto state = primitive_state;
    /* Fix up momentum: */
    for (unsigned int i = 1; i < dim + 1; ++i)
      state[i] *= rho;
    /* Compute total energy: */
    state[dim + 1] = p / (Number(gamma_ - 1.)) + Number(0.5) * rho * u * u;

    return state;
  }


  template <int problem_dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline dealii::Tensor<1, problem_dim, Number>
  ProblemDescription::to_primitive_state(
      const dealii::Tensor<1, problem_dim, Number> &state) const
  {
    constexpr auto dim = problem_dim - 2;

    const auto &rho = state[0];
    const auto rho_inverse = Number(1.) / rho;
    const auto p = pressure(state);

    auto primitive_state = state;
    /* Fix up velocity: */
    for (unsigned int i = 1; i < dim + 1; ++i)
      primitive_state[i] *= rho_inverse;
    /* Set pressure: */
    primitive_state[dim + 1] = p;

    return primitive_state;
  }

} /* namespace ryujin */

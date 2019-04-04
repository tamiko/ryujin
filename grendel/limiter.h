#ifndef LIMITER_H
#define LIMITER_H

#include "problem_description.h"

#include <deal.II/lac/la_parallel_vector.templates.h>

namespace grendel
{

  template <int dim>
  class Limiter
  {
  public:
    static constexpr unsigned int problem_dimension =
        ProblemDescription<dim>::problem_dimension;

    using rank1_type = typename ProblemDescription<dim>::rank1_type;

    /* Let's allocate 5 doubles for limiter bounds: */
    typedef std::array<double, 5> Bounds;

    typedef std::array<dealii::LinearAlgebra::distributed::Vector<double>, 5>
        vector_type;

    /*
     * Options:
     */

    static constexpr enum class Limiters {
      none,
      rho,
      internal_energy,
      specific_entropy
    } limiter_ = Limiters::specific_entropy;

    /*
     * Accumulate bounds:
     */

    inline DEAL_II_ALWAYS_INLINE void reset();

    inline DEAL_II_ALWAYS_INLINE void accumulate(const rank1_type &U);

    inline DEAL_II_ALWAYS_INLINE const Bounds &bounds() const;

    /*
     * Compute limiter value l_ij for update P_ij:
     */

    static inline DEAL_II_ALWAYS_INLINE double
    limit(const Bounds &bounds, const rank1_type &U, const rank1_type &P_ij);

  private:
    Bounds bounds_;
  };


  template <int dim>
  inline DEAL_II_ALWAYS_INLINE void
  Limiter<dim>::reset()
  {
    auto &[rho_min, rho_max, rho_epsilon_min, s_min, s_laplace] = bounds_;
    rho_min = std::numeric_limits<double>::max();
    rho_max = 0.;
    rho_epsilon_min = std::numeric_limits<double>::max();
    s_min = std::numeric_limits<double>::max();
    s_laplace = 0.;
  }


  template <int dim>
  inline DEAL_II_ALWAYS_INLINE void
  Limiter<dim>::accumulate(const rank1_type &U)
  {
    auto &[rho_min, rho_max, rho_epsilon_min, s_min, s_laplace] = bounds_;

    if constexpr(limiter_ == Limiters::none)
      return;

    const auto rho = U[0];
    rho_min = std::min(rho_min, rho);
    rho_max = std::max(rho_max, rho);

    if constexpr(limiter_ == Limiters::rho)
      return;

    const auto rho_epsilon = ProblemDescription<dim>::internal_energy(U);
    rho_epsilon_min = std::min(rho_epsilon_min, rho_epsilon);

    if constexpr(limiter_ == Limiters::internal_energy)
      return;

    const auto s = ProblemDescription<dim>::specific_entropy(U);
    s_min  = std::min(s_min, (1. - 1.e-7) * s);
  }


  template <int dim>
  inline DEAL_II_ALWAYS_INLINE const typename Limiter<dim>::Bounds &
  Limiter<dim>::bounds() const
  {
    return bounds_;
  }


  template <int dim>
  inline DEAL_II_ALWAYS_INLINE double Limiter<dim>::limit(
      const Bounds &bounds, const rank1_type &U, const rank1_type &P_ij)
  {
    auto &[rho_min, rho_max, rho_epsilon_min, s_min, s_laplace] = bounds;

    double l_ij = 1.;

    if constexpr(limiter_ == Limiters::none)
      return l_ij;

    /*
     * First limit rho.
     *
     * See [Guermond, Nazarov, Popov, Thomas] (4.8):
     */

    const auto &U_i_rho = U[0];
    const auto &P_ij_rho = P_ij[0];

    {
      double t_0 = 1.;

      constexpr double eps_ = std::numeric_limits<double>::epsilon();

      if (U_i_rho + P_ij_rho < rho_min)
        t_0 = std::abs(rho_min - U_i_rho) /
                   (std::abs(P_ij_rho) + eps_ * rho_max);

      else if (rho_max < U_i_rho + P_ij_rho)
        t_0 = std::abs(rho_max - U_i_rho) /
                   (std::abs(P_ij_rho) + eps_ * rho_max);

      l_ij = std::min(l_ij, t_0);

      Assert((U + l_ij * P_ij)[0] > 0.,
             dealii::ExcMessage("I'm sorry, Dave. I'm afraid I can't do that. "
                                "- Negative density."));
    }

    if constexpr(limiter_ == Limiters::rho)
      return l_ij;

    /*
     * Then, limit the internal energy. (We skip this limiting step in case
     *
     * See [Guermond, Nazarov, Popov, Thomas], Section 4.5:
     */

    if constexpr (limiter_ == Limiters::internal_energy) {

      const auto P_ij_m = ProblemDescription<dim>::momentum(P_ij);
      const auto &P_ij_E = P_ij[dim + 1];

      const auto U_i_m = ProblemDescription<dim>::momentum(U);
      const double &U_i_E = U[dim + 1];

      const double c =
          (U_i_E - rho_epsilon_min) * U_i_rho - 1. / 2. * U_i_m.norm_square();

      const double b = (U_i_E - rho_epsilon_min) * P_ij_rho +
                       P_ij_E * U_i_rho - U_i_m * P_ij_m;

      const double a = P_ij_E * P_ij_rho  - 1. / 2. * P_ij_m.norm_square();

      /*
       * Solve the quadratic equation a t^2 + b t + c = 0 by hand. We use the
       * Ciatardauq formula to avoid numerical cancellation and some if
       * statements:
       */

      double t_0 = 1.;

      const double discriminant = b * b - 4. * a * c;

      if (discriminant == 0.) {

        const double x = -b / 2. / a;

        if (x > 0)
          t_0 = x;

      } else if (discriminant > 0.) {

        const double x =
            2. * c / (-b - std::copysign(std::sqrt(discriminant), b));
        const double y = c / a / x;

        /* Select the smallest positive root: */
        if (x > 0.) {
          if (y > 0. && y < x)
            t_0 = y;
          else
            t_0 = x;
        } else if (y > 0.) {
          t_0 = y;
        }
      }

      l_ij = std::min(l_ij, t_0);

#ifdef DEBUG
      const double rho_epsilon =
          ProblemDescription<dim>::internal_energy(U + l_ij * P_ij);
      Assert(rho_epsilon - rho_epsilon_min > 0.,
             dealii::ExcMessage("I'm sorry, Dave. I'm afraid I can't do that. "
                                "- Negative internal energy."));
#endif

      return l_ij;
    }

    /*
     * And finally, limit the specific entropy:
     *
     * See [Guermond, Nazarov, Popov, Thomas], Section 4.6:
     */

    if constexpr (limiter_ == Limiters::specific_entropy)
    {
      /*
       * Prepare a Newton second method:
       */

      double t_l = 0.;
      double t_r = l_ij;

      constexpr unsigned int n_max_iter = 3;
      constexpr double tolerance = 1.e-7;

      for (unsigned int n = 0; n < n_max_iter; ++n) {

        const auto U_r = U + t_r * P_ij;
        const auto psi_r =
            ProblemDescription<dim>::specific_entropy(U_r) - s_min;

        /* Right state is good, cut it short and return: */

        if (psi_r >= 0.)
          return std::min(l_ij, t_r);

        const auto U_l = U + t_l * P_ij;
        const auto psi_l =
            ProblemDescription<dim>::specific_entropy(U_l) - s_min;

        AssertThrow(psi_l >= 0. && psi_r < 0. && psi_l > psi_r,
                    dealii::ExcMessage("Houston, we have a problem!"));

        const auto dpsi_l =
            ProblemDescription<dim>::specific_entropy_derivative(U_l) * P_ij;
        const auto dpsi_r =
            ProblemDescription<dim>::specific_entropy_derivative(U_r) * P_ij;

        AssertThrow(dpsi_l > 0. && dpsi_r > 0.,
                    dealii::ExcMessage("Houston, we have a problem!"));

        /* Compute divided differences: */

        const double dd_11 = dpsi_l;
        const double dd_12 = (psi_r - psi_l) / (psi_r - psi_l);
        const double dd_22 = dpsi_r;

        const double dd_112 = (dd_12 - dd_11) / (psi_r - psi_l);
        const double dd_122 = (dd_22 - dd_12) / (psi_r - psi_l);

        /* Update left point: */
        const double discriminant_l = dpsi_l * dpsi_l - 4. * psi_l * dd_112;
        AssertThrow(discriminant_l > 0.,
                    dealii::ExcMessage("Houston, we have a problem!"));
        t_l = t_l - 2. * psi_l / (dpsi_l + std::sqrt(discriminant_l));

        /* Update right point: */
        const double discriminant_r = dpsi_r * dpsi_r - 4. * psi_r * dd_122;
        AssertThrow(discriminant_r > 0.,
                    dealii::ExcMessage("Houston, we have a problem!"));
        t_r = t_r - 2. * psi_r / (dpsi_r + std::sqrt(discriminant_r));

        if (t_r < t_l || std::abs(t_r - t_l) < tolerance)
          return std::min(l_ij, t_r);
      }

      /* t_l is a good state with psi_l > 0. */
      return std::min(l_ij, t_l);
    }

    __builtin_unreachable();
  }

} /* namespace grendel */

#endif /* LIMITER_H */

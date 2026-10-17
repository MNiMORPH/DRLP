"""
Channel width, folded into m or carried explicitly, and where they part.

Two things worth seeing:

1. With a power-law width B = k_QB Q^P_QB, unit stream power (Q/B)^m S^n is the
   SAME MODEL as K Q^m S^n with m -> m(1 - P_QB) and K -> K k_QB^(-m). The
   profiles agree to machine precision, which is why k_QB cannot be identified
   from a profile: any value of it trades against K.
2. With a width that is NOT a power law of Q - a gorge - the two part company,
   and no choice of m or K in the implicit form reproduces the explicit one.

    python examples/width_both_ways.py
"""
import numpy as np

from drlp import LongProfile

YEAR = 31556926.0
U = 0.3e-3 / YEAR


def _common():
    return (LongProfile()
            .set_x(dx=500.0, nx=60, x0=500.0)
            .set_A(k_xA=1.0, P_xA=7 / 4.0)
            .set_Q(k_AQ=1e-4, P_AQ=0.7))


def main():
    m_unit, n, k_QB, P_QB, K_unit = 1.0, 1.0, 3.0, 0.5, 2e-6

    explicit = (_common().set_B(k_QB=k_QB, P_QB=P_QB)
                .set_K(K_unit, basis="Q/B").set_mn(m=m_unit, n=n)
                .set_uplift_rate(U).set_bl(z=0.0).set_steady_state())
    m_imp = LongProfile.implicit_m_from_unit(m_unit, P_QB)
    K_imp = LongProfile.implicit_K_from_unit(K_unit, k_QB, m_unit)
    implicit = (_common().set_K(K_imp, basis="Q").set_mn(m=m_imp, n=n)
                .set_uplift_rate(U).set_bl(z=0.0).set_steady_state())

    print("1. power-law width: the same model, twice")
    print(f"   explicit   (Q/B)^m S^n   m = {m_unit:.3f}  K = {K_unit:.4e}  "
          f"k_QB = {k_QB}")
    print(f"   implicit     Q^m S^n     m = {m_imp:.3f}  K = {K_imp:.4e}")
    print(f"   width B spans {explicit.B.min():.2f} to "
          f"{explicit.B.max():.2f} m")
    print(f"   max |z_explicit - z_implicit| = "
          f"{np.max(np.abs(explicit.z - implicit.z)):.2e} m")

    for lp in (explicit, implicit):
        lp.set_bl(z=-40.0)
        lp.evolve(nt=30, dt=1e11)
    print(f"   and after a 40 m drop and 30 steps: "
          f"{np.max(np.abs(explicit.z - implicit.z)):.2e} m")

    print()
    print("2. a gorge: width that discharge does not know about")
    gorge = _common()
    B = np.where(gorge.x < 15000.0, 24.0, 6.0)        # narrows four-fold
    gorge.set_B(B=B).set_K(2e-6, basis="Q/B").set_mn(m=1.0, n=1.0)
    gorge.set_uplift_rate(U).set_bl(z=0.0).set_steady_state()

    S = gorge.S
    trend = (gorge.Q / gorge.Q[0]) ** (-1.0)          # the smooth Q part
    print(f"   B steps {B[0]:.0f} m -> {B[-1]:.0f} m at 15 km")
    print(f"   S/S_0 corrected for the discharge trend: "
          f"{(S[-1] / S[0]) / (trend[-1]):.3f}")
    print(f"   predicted (B_narrow/B_wide)^(m/n):       {(6.0 / 24.0):.3f}")
    print()
    print("   A narrower channel concentrates the same discharge, raises unit")
    print("   stream power, and so needs LESS slope to balance uplift.")


if __name__ == "__main__":
    main()

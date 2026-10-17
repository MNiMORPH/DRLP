"""
Two faults, and the steady profile they leave behind.

Uplift stepped across two faults at a third and two thirds of the profile. At
steady state the stream-power law gives

    S = (U / (K Q^m))^(1/n)

so a step in U is a step in S. The ratio is NOT the uplift ratio, though,
because Q varies down the profile too, and the two effects multiply. Dividing
the discharge term back out recovers the uplift ratio exactly - which is the
whole reason a steepness index exists.

    python examples/two_faults.py
"""
import numpy as np

from drlp import LongProfile

YEAR = 31556926.0


def main():
    nx = 90
    U0 = 0.3e-3 / YEAR
    lp = (LongProfile()
          .set_x(dx=500.0, nx=nx, x0=500.0)
          .set_A(k_xA=1.0, P_xA=7 / 4.0)
          .set_Q(R=1e-8)
          .set_K(10**-8.5, basis="Q")
          .set_mn(m=0.5, n=1.0)
          .set_bl(z=0.0))

    U = np.full(nx, U0)
    U[: nx // 3] = 3.0 * U0                    # upslope block rises fastest
    U[nx // 3: 2 * nx // 3] = 2.0 * U0
    lp.set_uplift_rate(U)

    lp.set_z(np.zeros(nx))
    lp.evolve(nt=4000, dt=1e11)

    S = lp.S
    S_predicted = lp.steady_state_slope()
    err = np.max(np.abs(S - S_predicted) / S_predicted)

    print(f"relief after {lp.t / (1000 * YEAR):.0f} kyr from flat: "
          f"{lp.z[0] - lp.z[-1]:.1f} m")
    print(f"worst relative error against (U/(K Q^m))^(1/n): {err:.2e}")
    print()
    Q = lp.driver
    blocks = (("upstream (3 U)", slice(0, nx // 3)),
              ("middle (2 U)", slice(nx // 3, 2 * nx // 3)),
              ("downstream (U)", slice(2 * nx // 3, nx)))
    dn = blocks[-1][1]

    # The steepness index: slope with the discharge dependence divided out.
    # S = (U/(K Q^m))^(1/n), so S Q^(m/n) goes as U^(1/n) and nothing else.
    k_s = S * Q ** (lp.m / lp.n)

    print(f"{'block':>16}  {'U [mm/yr]':>10}  {'Q/Q_dn':>8}  "
          f"{'S/S_dn':>8}  {'k_s/k_s_dn':>11}  {'(U/U_dn)^(1/n)':>15}")
    for name, sl in blocks:
        print(f"{name:>16}  {U[sl][0] * YEAR * 1e3:10.1f}  "
              f"{Q[sl].mean() / Q[dn].mean():8.4f}  "
              f"{S[sl].mean() / S[dn].mean():8.3f}  "
              f"{k_s[sl].mean() / k_s[dn].mean():11.3f}  "
              f"{(U[sl][0] / U[dn][0]) ** (1.0 / lp.n):15.3f}")
    print()
    print("The raw slope ratio is 21 : 3 : 1, not the uplift ratio 3 : 2 : 1,")
    print("because the upstream block carries 7.5% of the outlet's discharge as")
    print("well as three times its uplift. Divide the discharge term out and the")
    print("uplift ratio comes back exactly. Raise n and the same uplift contrast")
    print("gives a weaker steepness contrast, because k_s goes as U^(1/n).")


if __name__ == "__main__":
    main()

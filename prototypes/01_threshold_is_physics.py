"""
Is the erosion threshold a numerical detail or a change to the physics?

The implementation is one term.  Backward Euler on

    dz/dt = U - max(0, K Q**m S**n - w_c)

with D0 = z_old + dt*U - z_r and x = (z - z_r)/D0 gives

    f(x) = x - 1 + alpha*x**n - beta,     beta = dt*w_c / D0

which is Landlab's erode_fn exactly, the beta it carries and DRLP currently
hardcodes to zero.  So adding it touches one line of the solve.

Two consequences that are NOT numerical:

1. THE BRACKET STOPS BEING GUARANTEED.  f(0) = -1 - beta < 0 still, but
   f(1) = alpha - beta, which is negative when the threshold is not exceeded.
   There is then no root in [0, 1] and the node must not be solved at all.
   That is why Landlab evaluates f(1) FIRST and skips the node when it is
   negative.  The handoff's one-line bracket proof does not survive beta > 0.

2. THE STEADY STATE CHANGES, and with it the whole steepness-uplift reading:

       U = K Q**m S**n - w_c   ->   S = ((U + w_c) / (K Q**m))**(1/n)

   so k_sn goes as (U + w_c)**(1/n), not U**(1/n).  A threshold-free model
   predicts a power law through the origin; a thresholded one predicts an
   offset power law with nonzero steepness at zero uplift.  No tuning of K, m
   or n in the threshold-free model can reproduce that shape.

This script checks both numerically.
"""
import numpy as np
from scipy.optimize import brentq

DX = 1000.0
K, M, N = 1e-9, 0.5, 1.0
Q = 10.0


def step_thresholded(x, z, Q, K, m, n, U, dt, x_bl, z_bl, w_c=0.0):
    """DRLP's stepper with Landlab's beta term restored.  Throwaway."""
    nx = x.size
    z_new = np.empty(nx)
    skipped = 0
    for i in range(nx - 1, -1, -1):
        z_r, x_r = (z_bl, x_bl) if i == nx - 1 else (z_new[i + 1], x[i + 1])
        dx = x_r - x[i]
        z_no_erosion = z[i] + dt * U
        D0 = z_no_erosion - z_r
        if D0 <= 0.0:
            z_new[i] = z_no_erosion
            continue
        alpha = dt * K * Q**m * D0 ** (n - 1.0) / dx**n
        beta = dt * w_c / D0
        # Landlab's pre-check: if the steepest available slope cannot exceed
        # the threshold, f(1) <= 0, there is no root, and nothing erodes.
        if alpha - beta <= 0.0:
            z_new[i] = z_no_erosion
            skipped += 1
            continue
        f = lambda v: v - 1.0 + alpha * v**n - beta
        z_new[i] = z_r + brentq(f, 0.0, 1.0) * D0
    return z_new, skipped


def steady_slope(U, w_c, K=K, m=M, n=N, Q=Q):
    return ((U + w_c) / (K * Q**m)) ** (1.0 / n)


print("=" * 72)
print("1. The steady state moves, and the solver confirms the new closed form")
print("=" * 72)
nx = 20
x = DX * np.arange(1, nx + 1)
x_bl, z_bl = DX * (nx + 1), 0.0
U = 1e-3 / 3.15e7                       # 1 mm/yr

for w_c in (0.0, 1e-12, 1e-11, 1e-10):
    z = np.zeros_like(x)
    for _ in range(4000):
        z, _ = step_thresholded(x, z, Q, K, M, N, U, 1e11, x_bl, z_bl, w_c)
    S_meas = -np.diff(np.append(z, z_bl)) / np.diff(np.append(x, x_bl))
    S_pred = steady_slope(U, w_c)
    S_nothresh = steady_slope(U, 0.0)
    print(f"  w_c={w_c:8.1e}  S measured={S_meas.mean():.6e}  "
          f"((U+w_c)/(K Q^m))^(1/n)={S_pred:.6e}  "
          f"ratio to w_c=0: {S_pred/S_nothresh:6.3f}")

print()
print("=" * 72)
print("2. k_sn vs U: a power law through the origin becomes an offset one")
print("=" * 72)
w_c = 1e-11
print(f"  threshold w_c = {w_c:.1e}  (so U = w_c at {w_c*3.15e7*1e3:.3f} mm/yr)")
print()
print(f"  {'U [mm/yr]':>10}  {'S no thresh':>13}  {'S threshold':>13}  {'ratio':>7}")
for U_mmyr in (0.01, 0.1, 0.3, 1.0, 3.0, 10.0):
    Ui = U_mmyr * 1e-3 / 3.15e7
    s0, s1 = steady_slope(Ui, 0.0), steady_slope(Ui, w_c)
    print(f"  {U_mmyr:10.2f}  {s0:13.6e}  {s1:13.6e}  {s1/s0:7.2f}")
print()
print("  As U -> 0 the threshold-free slope -> 0; the thresholded slope tends")
print("  to (w_c/(K Q^m))^(1/n) = "
      f"{steady_slope(0.0, w_c):.6e}, a finite steepness at zero uplift.")

print()
print("=" * 72)
print("3. A no-erosion regime appears: nodes the solver must not solve")
print("=" * 72)
# a gentle profile under a threshold high enough to lock it
z_gentle = np.linspace(2.0, 0.1, nx)
for w_c in (0.0, 1e-11, 1e-9):
    _, skipped = step_thresholded(x, z_gentle, Q, K, M, N, U, 1e11,
                                  x_bl, z_bl, w_c)
    print(f"  w_c={w_c:8.1e}  nodes below threshold (f(1) <= 0, no root): "
          f"{skipped:3d} / {nx}")
print()
print("  At w_c = 0 the bracket is guaranteed for every node and the handoff's")
print("  one-line proof holds. At w_c > 0 it does not: f(1) = alpha - beta can")
print("  be negative, and those nodes have no root in [0, 1] to find.")

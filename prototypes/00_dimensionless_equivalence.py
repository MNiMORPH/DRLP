"""
Is Landlab's dimensionless node equation the same equation as HANDOFF.md's,
and is either form better conditioned?

HANDOFF.md section 4, dimensional, uplift included:

    f_dim(z) = z - z_old - dt*U + dt*K*Q**m * ((z - z_r)/dx)**n
    bracket  [z_r, z_old + dt*U]

Landlab (cfuncs.pyx, erode_fn), dimensionless. Its docstring: "If the threshold
term, beta, is zero, this equation collapses to the form given by Braun and
Willet (2012)." DRLP has no erosion threshold, so beta = 0 throughout:

    f_x(x) = x - 1 + alpha * x**n
    x       = (z - z_r) / D0,   D0 = z_old + dt*U - z_r
    alpha   = dt*K*Q**m * D0**(n-1) / dx**n
    bracket [0, 1]

ALGEBRA. Substituting z = z_r + x*D0 into f_dim:

    f_dim = D0*(x - 1) + dt*K*Q**m * (x*D0/dx)**n
          = D0 * [ x - 1 + (dt*K*Q**m*D0**(n-1)/dx**n) * x**n ]
          = D0 * f_x(x)

with D0 > 0. The SAME equation, the same root, and uplift folds in exactly by
using D0 in place of Landlab's (z_old - z_r). So adopting the dimensionless
form imports no new physics, and nothing of Landlab's 2D grid: the per-node
equation is scalar and has no dimensionality.

CONDITIONING, which is the reason to prefer it. f_dim evaluates
`z - z_old - dt*U`, a difference of absolute elevations, so when the erosion
term is small next to the rounding error of that difference, the sign of
f_dim at the upper bracket is noise. f_x(1) = alpha > 0 exactly, by
construction, and cannot do this.

The draws below add an absolute elevation z_r, which HANDOFF.md's stated
ranges (n, K, Q, m, dx, dt) did not include. That is where the failure lives,
and |z_r| >> D0 is the ordinary case for a real profile: metres of relief
between nodes, hundreds or thousands of metres above datum.
"""
import numpy as np
from scipy.optimize import brentq

rng = np.random.default_rng(20261016)
N = 20000

dim_bracket_failures = 0
nd_bracket_failures = 0
worst_root_diff = 0.0
compared = 0
worst_case = None

for _ in range(N):
    n   = rng.uniform(0.5, 3.0)
    K   = 10 ** rng.uniform(-7, -3)
    Q   = 10 ** rng.uniform(0, 4)
    m   = rng.uniform(0.3, 0.7)
    dx  = 10 ** rng.uniform(0, 3)
    dt  = 10 ** rng.uniform(0, 6)
    U   = 10 ** rng.uniform(-11, -8)
    z_r = rng.uniform(-100, 100)          # absolute elevation, the new axis
    D0  = 10 ** rng.uniform(-2, 3)        # no-erosion relief, > 0
    z_old = z_r + D0 - dt * U
    Z0 = z_old + dt * U

    def f_dim(z):
        return z - z_old - dt * U + dt * K * Q**m * ((z - z_r) / dx) ** n

    alpha = dt * K * Q**m * D0 ** (n - 1.0) / dx**n
    def f_x(x):
        return x - 1.0 + alpha * x**n

    dim_ok = f_dim(z_r) <= 0 <= f_dim(Z0)
    nd_ok = f_x(0.0) <= 0 <= f_x(1.0)
    if not dim_ok:
        dim_bracket_failures += 1
        if worst_case is None:
            worst_case = dict(z_r=z_r, D0=D0, n=n,
                              erosion_at_hi=dt * K * Q**m * (D0 / dx) ** n,
                              rounding=Z0 - z_old - dt * U,
                              f_dim_hi=f_dim(Z0), f_x_1=f_x(1.0))
    if not nd_ok:
        nd_bracket_failures += 1

    # Where both brackets hold, confirm the roots agree.
    # Public scipy defaults: xtol=2e-12, rtol=4*eps, maxiter=100.  NOT
    # Landlab's xtol=1e-12 / rtol=2*eps: the public wrapper enforces
    # rtol >= 4*eps, which is why Landlab reaches for the private _brentq.
    if dim_ok and nd_ok:
        z_star = brentq(f_dim, z_r, Z0, maxiter=100)
        x_star = brentq(f_x, 0.0, 1.0, maxiter=100)
        worst_root_diff = max(worst_root_diff,
                              abs((z_r + x_star * D0) - z_star) / D0)
        compared += 1

print(f"draws                                 : {N}")
print(f"dimensional   bracket failures        : {dim_bracket_failures}")
print(f"dimensionless bracket failures        : {nd_bracket_failures}")
print(f"roots compared where both held        : {compared}")
print(f"worst |z(x*) - z*| / D0               : {worst_root_diff:.3e}")
if worst_case:
    w = worst_case
    print("\nfirst dimensional failure, the mechanism:")
    print(f"  z_r                 = {w['z_r']:.6g} m   (absolute elevation)")
    print(f"  D0                  = {w['D0']:.6g} m   (relief to receiver)")
    print(f"  |z_r| / D0          = {abs(w['z_r'])/w['D0']:.0f}")
    print(f"  erosion term at hi  = {w['erosion_at_hi']:.3e} m")
    print(f"  rounding in z-z_old = {w['rounding']:.3e} m   "
          f"({abs(w['rounding']/w['erosion_at_hi']):.0f}x the erosion term)")
    print(f"  => f_dim(hi)        = {w['f_dim_hi']:.3e}  (should be >= 0)")
    print(f"     f_x(1)           = {w['f_x_1']:.3e}  (> 0, exactly)")

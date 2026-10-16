"""
Implicit stream-power incision on a single channel: the Braun & Willett (2013)
outlet-to-upstream sweep.

    dz/dt = U - K Q**m S**n,      S = (z - z_r)/dx  (positive downstream)

Backward Euler at node ``i`` with receiver ``r``, using the *already-updated*
receiver elevation:

    (z_i - z_i_old)/dt = U_i - K_i Q_i**m ((z_i - z_r)/dx_i)**n

Information flows upstream, so sweeping nodes from the downstream end to the
upstream end solves every receiver before the node that needs it: one pass is
exact.  For a single channel this is a reverse loop over one array -- no graph,
no sweep order, no topology (``HANDOFF-AMENDMENT-1.md``).

CONVENTIONS, chosen to match FluvTree's network sweep so that an adapter can
drive this function reach by reach without a translation layer
(``~/models/FluvTree/src/fluvtree/solvers/advection.py``, read 16 Oct 2026):

- index ``0`` is the UPSTREAM end, the last index is the downstream-most node;
- ``x`` increases downstream, so ``dx = x_r - x_i > 0``;
- the receiver of the last real node is the downstream boundary ghost node at
  ``(x_bl, z_bl)``, after GRLP, where ``set_x_bl`` assigns
  ``self.x_ghost_downstream = self.x_bl``.

In the standalone single-channel case the ghost node is base level.  Under a
FluvTree adapter the same two arguments carry the downstream reach's first
node.  The ghost node IS the receiver, so ``(x_bl, z_bl)`` is the whole
downstream boundary condition.

PROVENANCE OF THE NUMERICS.  See ``design/01-prior-implementations.md`` for the
source readings this is built on, and
``prototypes/00_dimensionless_equivalence.py`` for the measurement that chose
between the two available formulations.

- The ``n = 1`` closed form is Braun & Willett.  fastscapelib-fortran's
  ``StreamPowerLaw.f90`` computes it as ``h(ijk) = f/df`` with
  ``f = elev + sum(fact*h_r)`` and ``df = 1 + sum(fact)``, which for a single
  receiver is exactly the expression used below.
- General ``n`` is solved in Landlab's DIMENSIONLESS variable rather than in
  elevation.  Landlab's ``erode_fn`` (``cfuncs.pyx``) states that with its
  threshold term ``beta = 0`` -- DRLP's case, there being no erosion threshold
  -- "this equation collapses to the form given by Braun and Willet (2012)".
  So the equation below is Braun & Willett's, not Landlab's extension of it,
  and nothing of Landlab's 2D grid comes with it: the per-node equation is
  scalar and has no dimensionality.
- The dimensionless form was chosen over the dimensional bracket
  ``[z_r, z_old + dt*U]`` because the dimensional one is ill-conditioned, not
  merely untidy.  ``f_dim`` evaluates ``z - z_old - dt*U``, a difference of
  absolute elevations; once the erosion term falls below the rounding error of
  that difference, the sign of ``f_dim`` at the upper bracket is noise.
  Measured over 20,000 draws that sweep an absolute elevation: 23 bracket
  failures for the dimensional form, 0 for the dimensionless one, at
  ``|z|/relief`` ratios that are ordinary for a real profile.
"""

import numpy as np
from scipy.optimize import brentq

# fastscapelib's test for the linear case, in `spl.hpp`:
#     m_linear = (std::fabs(value) - 1) <= std::numeric_limits<double>::epsilon()
# Machine epsilon, i.e. exact equality up to representation.  Borrowed rather
# than invented so the fast path triggers on the same inputs theirs does.
_N_LINEAR_EPS = np.finfo(float).eps

# scipy.optimize.brentq's PUBLIC defaults, restated here so the choice is
# visible at the call site.  Landlab passes xtol=1e-12 and rtol=2*eps through
# the private scipy.optimize._zeros._brentq, and comments that those "are the
# defaults if one were to import brentq from scipy.optimize".  As of scipy
# 1.11.4 they are not: the public defaults are the values below, and the public
# wrapper REFUSES rtol < 4*eps with "rtol too small (4.44089e-16 <
# 8.88178e-16)".  Landlab's private import is therefore load-bearing for its
# tolerance, not just a speed choice.  DRLP uses the public interface and the
# tolerance it accepts.
_XTOL = 2e-12
_RTOL = 4.0 * np.finfo(float).eps
_MAXITER = 100


class ConvergenceError(RuntimeError):
    """A node's general-``n`` root-find did not converge.

    Raised rather than returning a half-converged elevation.  On a bracket
    that is guaranteed by construction this should never fire; neither
    Fastscape implementation caps its Newton iterations at all, so there is no
    precedent to copy for the cap itself.
    """


def _as_node_array(value, n_nodes, name):
    """Broadcast a scalar or array parameter to one value per node."""
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return np.full(n_nodes, float(arr))
    if arr.shape != (n_nodes,):
        raise ValueError(
            f"{name} must be a scalar or have shape ({n_nodes},); got {arr.shape}"
        )
    return arr


def step(x, z, Q, K, m, n, U, dt, x_bl, z_bl):
    """
    One implicit stream-power step on a single channel.

    Parameters
    ----------
    x : array
        Downstream distance [m], strictly increasing.  Index 0 is upstream.
    z : array
        Elevation [m] at the start of the step.  Not modified.
    Q : array or float
        Discharge [m**3/s] at each node.
    K : array or float
        Erodibility, units set by ``m`` and ``n``.
    m, n : float
        Discharge and slope exponents.  ``n = 1`` takes an exact closed form.
    U : array or float
        Rock uplift rate [m/s].
    dt : float
        Time step [s].
    x_bl, z_bl : float
        Downstream boundary ghost node: its position [m] and elevation [m].
        ``x_bl`` must exceed ``x[-1]``.

    Returns
    -------
    z_new : array
        Elevation [m] after the step.  A new array; ``z`` is left alone, which
        is what lets the time term read ``z`` while receivers read ``z_new``.
    """
    x = np.asarray(x, dtype=float)
    z = np.asarray(z, dtype=float)
    n_nodes = x.size

    if z.shape != x.shape:
        raise ValueError(f"x and z must have the same shape; got {x.shape}, {z.shape}")
    if n_nodes == 0:
        return np.empty(0)
    if n_nodes > 1 and np.any(np.diff(x) <= 0):
        raise ValueError("x must be strictly increasing (index 0 is upstream)")
    if x_bl <= x[-1]:
        raise ValueError(
            f"x_bl ({x_bl}) must be downstream of the last node ({x[-1]}): "
            "the ghost node is the receiver of that node"
        )
    if dt <= 0:
        raise ValueError(f"dt must be positive; got {dt}")
    if n <= 0:
        raise ValueError(f"n must be positive; got {n}")

    Q = _as_node_array(Q, n_nodes, "Q")
    K = _as_node_array(K, n_nodes, "K")
    U = _as_node_array(U, n_nodes, "U")

    linear = abs(n - 1.0) <= _N_LINEAR_EPS
    z_new = np.empty(n_nodes)

    # Outlet -> upstream.  The receiver of node i is node i+1, already updated,
    # or the ghost node for the downstream-most node.
    for i in range(n_nodes - 1, -1, -1):
        if i == n_nodes - 1:
            z_r, x_r = z_bl, x_bl
        else:
            z_r, x_r = z_new[i + 1], x[i + 1]
        dx = x_r - x[i]

        # The no-erosion elevation and the relief that would exist if nothing
        # eroded this step.  D0 <= 0 means the node sits at or below its
        # receiver: a flat, a filled depression, or a numerical artefact.
        # There is no erosion to do, and S**n must not be evaluated on a
        # negative slope -- for fractional n it is not real.
        z_no_erosion = z[i] + dt * U[i]
        D0 = z_no_erosion - z_r

        if D0 <= 0.0 or Q[i] == 0.0 or K[i] == 0.0:
            z_new[i] = z_no_erosion
            continue

        if linear:
            # Braun & Willett closed form, exact and unconditionally stable.
            c = dt * K[i] * Q[i] ** m / dx
            z_new[i] = (z[i] + dt * U[i] + c * z_r) / (1.0 + c)
        else:
            # Landlab's dimensionless variable: x_nd is the fraction of the
            # no-erosion relief D0 that survives the step.
            #
            #     f(x_nd) = x_nd - 1 + alpha * x_nd**n
            #     alpha   = dt K Q**m D0**(n-1) / dx**n
            #
            # f(0) = -1 < 0 and f(1) = alpha > 0, and f increases
            # monotonically in between for alpha > 0, n > 0.  So [0, 1] is a
            # guaranteed bracket, proved in one line rather than sampled, and
            # Brent on it cannot diverge the way a bare Newton step can when
            # n < 1 sends f' to infinity as x_nd -> 0.
            alpha = dt * K[i] * Q[i] ** m * D0 ** (n - 1.0) / dx**n

            def f(x_nd, alpha=alpha, n=n):
                return x_nd - 1.0 + alpha * x_nd**n

            try:
                x_nd = brentq(f, 0.0, 1.0, xtol=_XTOL, rtol=_RTOL,
                              maxiter=_MAXITER)
            except RuntimeError as exc:  # scipy raises this on maxiter
                raise ConvergenceError(
                    f"node {i}: general-n root-find failed to converge in "
                    f"{_MAXITER} iterations (n={n}, alpha={alpha:g}, "
                    f"D0={D0:g} m)"
                ) from exc

            z_new[i] = z_r + x_nd * D0

    return z_new

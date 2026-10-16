"""
The general-``n`` path: Brent on Landlab's dimensionless relief fraction.

The equation solved per node, with DRLP's threshold absent (Landlab's
``beta = 0``, the case its docstring attributes to Braun & Willett):

    f(x) = x - 1 + alpha * x**n,   alpha = dt K Q**m D0**(n-1) / dx**n
    x    = (z_new - z_r) / D0,     D0 = z_old + dt*U - z_r

See ``design/01-prior-implementations.md`` for the provenance and
``prototypes/00_dimensionless_equivalence.py`` for why this form rather than
the dimensional one.
"""

import numpy as np
import pytest

from drlp.incision import step, _N_LINEAR_EPS


DX = 1000.0
K, M = 1e-9, 0.5
U = 1e-3 / 3.15e7
Q = 10.0

N_VALUES = [2.0 / 3.0, 1.5, 2.0, 0.5, 3.0]


def _reach(nx=20):
    x = DX * np.arange(1, nx + 1)
    return x, DX * (nx + 1), 0.0


def _analytic_steady(x, x_bl, z_bl, n):
    S = (U / (K * Q**M)) ** (1.0 / n)
    return z_bl + S * (x_bl - x)


@pytest.mark.parametrize("n", N_VALUES)
def test_steady_state_is_a_fixed_point(n):
    x, x_bl, z_bl = _reach()
    z = _analytic_steady(x, x_bl, z_bl, n)
    z_new = step(x, z, Q, K, M, n, U, 1e10, x_bl, z_bl)
    # tolerance is set by brentq's xtol on x times the local relief
    assert np.allclose(z_new, z, rtol=1e-9, atol=1e-6)


@pytest.mark.parametrize("n", N_VALUES)
def test_relaxes_to_steady_state_from_flat(n):
    x, x_bl, z_bl = _reach()
    z = np.zeros_like(x)
    # 2500 steps. Measured: n = 3 is still 56% off at 600 steps and converged
    # to 4.9e-13 by 2000, so high n relaxes far more slowly from flat.
    for _ in range(2500):
        z = step(x, z, Q, K, M, n, U, 1e11, x_bl, z_bl)
    assert np.allclose(z, _analytic_steady(x, x_bl, z_bl, n), rtol=1e-9, atol=0)


def test_general_path_reproduces_the_n1_closed_form():
    # The fast path triggers at |n - 1| <= machine epsilon, so nudging n just
    # outside that forces the root-find on a problem whose answer is known in
    # closed form. Algebraically they are the same: solving
    # f(x) = x - 1 + alpha x = 0 gives x = 1/(1 + alpha), hence
    # z = z_r + D0/(1 + alpha) = (z_old + dt U + alpha z_r)/(1 + alpha).
    x, x_bl, z_bl = _reach(nx=30)
    z = np.linspace(400.0, 10.0, x.size)
    dt = 1e10

    z_closed = step(x, z, Q, K, M, 1.0, U, dt, x_bl, z_bl)
    n_forced = 1.0 + 64.0 * _N_LINEAR_EPS
    z_brent = step(x, z, Q, K, M, n_forced, U, dt, x_bl, z_bl)

    assert np.allclose(z_brent, z_closed, rtol=0, atol=1e-6)


@pytest.mark.parametrize("n", N_VALUES)
def test_large_dt_stays_finite_and_ordered(n):
    # Unconditional stability is NOT free at n != 1, so this test establishes
    # what is actually true rather than confirming what is assumed: at dt
    # equal to ~32 Myr the solve must still return a finite, downstream-
    # decreasing profile and never overshoot its receiver.
    x, x_bl, z_bl = _reach()
    z = np.linspace(500.0, 20.0, x.size)
    z_new = step(x, z, Q, K, M, n, U, 1e15, x_bl, z_bl)
    assert np.all(np.isfinite(z_new))
    assert np.all(np.diff(z_new) < 0)
    assert z_new[-1] >= z_bl
    # no node may rise above its no-erosion elevation
    assert np.all(z_new <= z + 1e15 * U + 1e-6)


@pytest.mark.parametrize("n", N_VALUES)
def test_incision_front_migrates_upstream(n):
    # Direction only. At n > 1 celerity rises with slope so the step steepens;
    # at n < 1 it falls with slope so the disturbance spreads and there is no
    # coherent step to track -- the peak-slope tracker used at n = 1 is simply
    # not meaningful there. So track the CENTROID of the elevation change,
    # which is threshold-free and defined for a spreading and a steepening
    # front alike.
    #
    # dt is set from the analytic celerity C = K Q**m n S**(n-1) so that the
    # front crosses a comparable fraction of the domain at every n. At
    # n = 0.5, C is 10x its n = 1 value, and a fixed dt would let the signal
    # traverse the whole domain and re-equilibrate before the first
    # measurement -- measured: the upstream node had already dropped the full
    # 50 m.
    nx = 60
    x, x_bl, z_bl = _reach(nx=nx)
    z0 = _analytic_steady(x, x_bl, z_bl, n)
    z = z0.copy()
    z_bl_new = z_bl - 50.0

    S0 = (U / (K * Q**M)) ** (1.0 / n)
    C = K * Q**M * n * S0 ** (n - 1.0)          # analytic celerity at n
    domain = x_bl - x[0]
    n_steps = 20
    dt = 0.15 * domain / (C * n_steps)          # 15% of the domain per leg

    def front_x(zp):
        """Centroid of the elevation change: no threshold, any front shape."""
        d = z0 - zp
        total = d.sum()
        return float((d * x).sum() / total) if total > 0 else x_bl

    for _ in range(n_steps):
        z = step(x, z, Q, K, M, n, U, dt, x_bl, z_bl_new)
    x0 = front_x(z)
    for _ in range(n_steps):
        z = step(x, z, Q, K, M, n, U, dt, x_bl, z_bl_new)
    x1 = front_x(z)

    assert x1 < x0, f"n={n}: front did not migrate upstream: {x0} -> {x1}"
    # and the signal is still inside the domain, so the measurement is not of
    # an already-equilibrated profile
    assert z[0] == pytest.approx(z0[0], rel=0, abs=1e-6)


@pytest.mark.parametrize("n", N_VALUES)
def test_base_level_drop_is_fully_accounted_for(n):
    # The general form of the n = 1 conservation check. A drop of D partitions
    # into relief still stored inside the domain and elevation already carried
    # out past the upstream end:
    #
    #     relief_added + upstream_node_lowering == D
    #
    # At n = 1 the second term is zero within this run and the identity
    # reduces to the n = 1 test. At n = 0.5 the celerity is 10x higher, the
    # signal traverses the whole domain, and the whole 50 m shows up in the
    # second term instead -- measured 0.0000 + 50.000. Stating both terms
    # makes the test hold at every n with no timing guard.
    x, x_bl, z_bl = _reach(nx=60)
    S0 = (U / (K * Q**M)) ** (1.0 / n)
    z0 = _analytic_steady(x, x_bl, z_bl, n)
    z = z0.copy()
    drop = 50.0
    for _ in range(40):
        z = step(x, z, Q, K, M, n, U, 1e11, x_bl, z_bl - drop)

    dx = np.diff(np.append(x, x_bl))
    S = -np.diff(np.append(z, z_bl - drop)) / dx
    relief_added = np.sum((S - S0) * dx)
    exported = z0[0] - z[0]
    assert relief_added + exported == pytest.approx(drop, rel=1e-6)


@pytest.mark.parametrize("n", N_VALUES)
def test_degenerate_inputs(n):
    x, x_bl, z_bl = _reach(nx=5)
    z = np.array([10.0, 8.0, 6.0, 4.0, 2.0])
    dt = 1e10
    # zero discharge
    assert np.allclose(step(x, z, 0.0, K, M, n, U, dt, x_bl, z_bl), z + dt * U)
    # zero erodibility
    assert np.allclose(step(x, z, Q, 0.0, M, n, U, dt, x_bl, z_bl), z + dt * U)
    # ghost node above the profile: the mechanism acts on the last node alone
    z_new = step(x, z, Q, K, M, n, U, dt, x_bl, 1e4)
    assert z_new[-1] == pytest.approx(z[-1] + dt * U, rel=0, abs=1e-12)
    # a flat profile: no slope anywhere, so uplift only
    z_flat = step(x, np.zeros_like(x), Q, K, M, n, U, dt, x_bl, 0.0)
    assert np.all(np.isfinite(z_flat))


def test_bracket_is_guaranteed_over_random_draws():
    # The [0, 1] bracket is provable -- f(0) = -1 < 0, f(1) = alpha > 0, f
    # monotonic in between -- but the proof assumes exact arithmetic, so check
    # it survives floating point over the handoff's parameter ranges PLUS an
    # absolute elevation, the axis whose omission hid the dimensional form's
    # failure (23 in 20,000; see prototypes/00_dimensionless_equivalence.py).
    rng = np.random.default_rng(20261016)
    failures = 0
    for _ in range(20000):
        n = rng.uniform(0.5, 3.0)
        K_ = 10 ** rng.uniform(-7, -3)
        Q_ = 10 ** rng.uniform(0, 4)
        m = rng.uniform(0.3, 0.7)
        dx = 10 ** rng.uniform(0, 3)
        dt = 10 ** rng.uniform(0, 6)
        D0 = 10 ** rng.uniform(-2, 3)
        alpha = dt * K_ * Q_**m * D0 ** (n - 1.0) / dx**n

        def f(xi, alpha=alpha, n=n):
            return xi - 1.0 + alpha * xi**n

        if not (f(0.0) <= 0.0 <= f(1.0)):
            failures += 1
        # monotonic along the bracket
        xs = np.linspace(0.0, 1.0, 64)
        if np.any(np.diff([f(float(v)) for v in xs]) < 0):
            failures += 1
    assert failures == 0, f"{failures} bracket or monotonicity failures"


def test_solver_agrees_with_scipy_brentq_on_the_same_bracket():
    # Free prediction: the stepper's answer must satisfy the dimensionless
    # equation, checked by solving it independently here with the public
    # scipy.optimize.brentq rather than trusting the stepper's own call.
    from scipy.optimize import brentq

    rng = np.random.default_rng(7)
    for _ in range(300):
        n = rng.uniform(0.5, 3.0)
        nx = 8
        x = DX * np.arange(1, nx + 1)
        x_bl, z_bl = DX * (nx + 1), 0.0
        z = np.sort(rng.uniform(0.0, 500.0, nx))[::-1].copy()
        K_ = 10 ** rng.uniform(-10, -7)
        Q_ = 10 ** rng.uniform(0, 3)
        dt = 10 ** rng.uniform(8, 12)

        z_new = step(x, z, Q_, K_, M, n, U, dt, x_bl, z_bl)

        # reproduce the sweep independently, node by node
        z_ref = np.empty(nx)
        for i in range(nx - 1, -1, -1):
            z_r = z_bl if i == nx - 1 else z_ref[i + 1]
            x_r = x_bl if i == nx - 1 else x[i + 1]
            dx = x_r - x[i]
            D0 = z[i] + dt * U - z_r
            if D0 <= 0.0:
                z_ref[i] = z[i] + dt * U
                continue
            alpha = dt * K_ * Q_**M * D0 ** (n - 1.0) / dx**n
            xi = brentq(lambda v: v - 1.0 + alpha * v**n, 0.0, 1.0)
            z_ref[i] = z_r + xi * D0

        assert np.allclose(z_new, z_ref, rtol=1e-9, atol=1e-9)

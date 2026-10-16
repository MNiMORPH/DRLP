"""
The ``n = 1`` path: Braun & Willett's closed form on a single channel.

Every check here is against something the solver was not fitted to -- a closed
form, or an independently written implementation.
"""

import numpy as np
import pytest

from drlp.incision import step


DX = 1000.0
K, M = 1e-9, 0.5
U = 1e-3 / 3.15e7          # 1 mm/yr in m/s
Q = 10.0


def _reach(nx=20):
    """A reach with the ghost node one dx beyond the downstream-most node."""
    x = DX * np.arange(1, nx + 1)
    return x, DX * (nx + 1), 0.0        # x, x_bl, z_bl


def _analytic_steady(x, x_bl, z_bl, n=1.0, U=U, K=K, Q=Q, m=M):
    """``S = (U/(K Q**m))**(1/n)``, integrated upstream from the ghost node."""
    S = (U / (K * Q**m)) ** (1.0 / n)
    return z_bl + S * (x_bl - x)


def test_steady_state_is_a_fixed_point():
    # The analytic profile and the stepper are separate pieces of code. If the
    # profile moves, one of them is wrong.
    x, x_bl, z_bl = _reach()
    z = _analytic_steady(x, x_bl, z_bl)
    z_new = step(x, z, Q, K, M, 1.0, U, 1e10, x_bl, z_bl)
    assert np.allclose(z_new, z, rtol=0, atol=1e-9)


def test_relaxes_to_steady_state_from_flat():
    # Start flat and run: the slope must converge on the closed form.
    x, x_bl, z_bl = _reach()
    z = np.zeros_like(x)
    for _ in range(400):
        z = step(x, z, Q, K, M, 1.0, U, 1e11, x_bl, z_bl)
    assert np.allclose(z, _analytic_steady(x, x_bl, z_bl), rtol=1e-6, atol=0)


def test_unconditionally_stable_at_huge_dt():
    # At n = 1 the sweep is exact in one pass for any dt. A step of 1e12 s is
    # ~32 kyr; nothing may go non-finite or oscillate.
    x, x_bl, z_bl = _reach()
    z = np.linspace(500.0, 20.0, x.size)
    z_new = step(x, z, Q, K, M, 1.0, U, 1e12, x_bl, z_bl)
    assert np.all(np.isfinite(z_new))
    # monotonic downstream, i.e. no inverted reaches created
    assert np.all(np.diff(z_new) < 0)
    # and bounded by the no-erosion elevation above and the receiver below
    assert np.all(z_new <= z + 1e12 * U + 1e-9)


def _knick_x(z, x, x_bl, z_bl):
    """Position of the steepest reach: where the step is.

    Slopes are taken over all reaches INCLUDING the last node to the ghost
    node, so a knickpoint sitting at the downstream boundary is visible.
    """
    S = -np.diff(np.append(z, z_bl)) / np.diff(np.append(x, x_bl))
    return x[int(np.argmax(S))], S


def test_knickpoint_migrates_upstream_at_the_analytic_celerity():
    # A base-level drop on a steady profile. At n = 1 the equation is linear
    # advection of z at celerity C = K Q**m, independent of slope, so the step
    # travels upstream rather than spreading: dz/dt + K Q**m dz/dx = U.
    #
    # C is derived from the governing equation, not from the solver, so this
    # is a free prediction. dt is chosen so the knickpoint crosses several
    # nodes: C*dt = 316 m per step against dx = 1000 m.
    x, x_bl, z_bl = _reach(nx=60)
    z = _analytic_steady(x, x_bl, z_bl)
    z_bl_new = z_bl - 50.0

    C = K * Q**M                      # analytic celerity [m/s]
    dt, n_steps = 1e11, 20

    for _ in range(n_steps):          # form and advance the knickpoint
        z = step(x, z, Q, K, M, 1.0, U, dt, x_bl, z_bl_new)
    x0, _ = _knick_x(z, x, x_bl, z_bl_new)

    for _ in range(n_steps):          # and advance it again
        z = step(x, z, Q, K, M, 1.0, U, dt, x_bl, z_bl_new)
    x1, S = _knick_x(z, x, x_bl, z_bl_new)

    # upstream is a SMALLER x
    assert x1 < x0, f"knickpoint did not migrate upstream: {x0} -> {x1}"

    # and it moved by C * elapsed time. The measured position is quantised to
    # the node spacing, so the tolerance is 1.5 dx -- set by the grid, not
    # tuned to pass.
    expected = C * dt * n_steps
    assert abs((x0 - x1) - expected) < 1.5 * DX, (
        f"migrated {x0 - x1:.0f} m, analytic C*dt*nt = {expected:.0f} m"
    )

    # it is locally steeper than the steady slope upstream of it: direction
    # only. The scheme is first-order upwind and therefore numerically
    # diffusive, so the step's HEIGHT on a given grid is a discretisation
    # property, not physics -- see test_step_sharpens_under_grid_refinement.
    assert S.max() > (U / (K * Q**M))


def test_base_level_drop_adds_exactly_its_own_relief():
    # An identity the mechanism itself relates, so there is no derived
    # expectation to get wrong: dropping base level by D must add exactly D of
    # relief to the profile, however the knickpoint is smeared across the
    # grid. Integrating the slope excess over the steady slope recovers D.
    x, x_bl, z_bl = _reach(nx=60)
    S0 = U / (K * Q**M)
    z = _analytic_steady(x, x_bl, z_bl)
    drop = 50.0

    dt = 1e11
    for _ in range(40):
        z = step(x, z, Q, K, M, 1.0, U, dt, x_bl, z_bl - drop)

    dx = np.diff(np.append(x, x_bl))
    S = -np.diff(np.append(z, z_bl - drop)) / dx
    relief_added = np.sum((S - S0) * dx)
    assert relief_added == pytest.approx(drop, rel=1e-6)


def test_step_sharpens_under_grid_refinement():
    # The exact n = 1 solution advects a step without changing its shape. On a
    # grid, first-order upwind differencing smears it. If the smearing is
    # numerical rather than physical, halving dx at a fixed Courant number
    # must sharpen the step monotonically -- which is also the evidence that
    # DRLP is advective where GRLP is diffusive.
    S0 = U / (K * Q**M)
    C = K * Q**M
    elapsed = 4e12
    sharpness = []
    for factor in (1, 2, 4):
        dx = DX / factor
        nx = 60 * factor
        x = dx * np.arange(1, nx + 1)
        x_bl = dx * (nx + 1)
        z = 0.0 + S0 * (x_bl - x)
        dt = 0.316 * dx / C                 # fixed Courant number
        for _ in range(int(elapsed / dt)):
            z = step(x, z, Q, K, M, 1.0, U, dt, x_bl, -50.0)
        S = -np.diff(np.append(z, -50.0)) / np.diff(np.append(x, x_bl))
        sharpness.append(S.max() / S0)

    assert sharpness == sorted(sharpness), (
        f"step did not sharpen monotonically under refinement: {sharpness}"
    )
    assert sharpness[-1] > sharpness[0] * 1.2


def test_spatially_variable_K_and_U_broadcast():
    x, x_bl, z_bl = _reach()
    z = _analytic_steady(x, x_bl, z_bl)
    K_arr = np.full(x.size, K)
    U_arr = np.full(x.size, U)
    z_scalar = step(x, z, Q, K, M, 1.0, U, 1e10, x_bl, z_bl)
    z_array = step(x, z, Q, K_arr, M, 1.0, U_arr, 1e10, x_bl, z_bl)
    assert np.array_equal(z_scalar, z_array)


def test_two_faults_steady_profile():
    # Uplift stepped at one third and two thirds of the profile. At steady
    # state S = (U/(K Q**m))**(1/n), so S must step in the same ratio as U.
    nx = 90
    x, x_bl, z_bl = _reach(nx=nx)
    U_arr = np.full(nx, U)
    U_arr[: nx // 3] = 3.0 * U          # upstream block rising fastest
    U_arr[nx // 3 : 2 * nx // 3] = 2.0 * U

    z = np.zeros_like(x)
    for _ in range(3000):
        z = step(x, z, Q, K, M, 1.0, U_arr, 1e11, x_bl, z_bl)

    S = -np.diff(np.append(z, z_bl)) / np.diff(np.append(x, x_bl))
    S_expected = U_arr / (K * Q**M)
    assert np.allclose(S, S_expected, rtol=1e-5, atol=0)


def test_degenerate_inputs_return_finite_elevations():
    x, x_bl, z_bl = _reach(nx=5)
    z = np.array([10.0, 8.0, 6.0, 4.0, 2.0])
    # zero discharge: no erosion, uplift only
    z_new = step(x, z, 0.0, K, M, 1.0, U, 1e10, x_bl, z_bl)
    assert np.allclose(z_new, z + 1e10 * U)
    # ghost node far ABOVE the profile. The mechanism acts on the one node
    # that sees the ghost -- the downstream-most -- which has nothing to erode
    # into and so takes uplift only. Every node upstream still drains to its
    # own (already updated) neighbour and erodes normally, so the assertion
    # belongs on the last node alone.
    z_new = step(x, z, Q, K, M, 1.0, U, 1e10, x_bl, 1e4)
    assert np.all(np.isfinite(z_new))
    assert z_new[-1] == pytest.approx(z[-1] + 1e10 * U, rel=0, abs=1e-12)
    assert np.all(z_new[:-1] < (z + 1e10 * U)[:-1])
    # a flat profile
    z_new = step(x, np.zeros_like(x), Q, K, M, 1.0, U, 1e10, x_bl, 0.0)
    assert np.all(np.isfinite(z_new))


def test_input_validation():
    x, x_bl, z_bl = _reach(nx=5)
    z = np.zeros_like(x)
    with pytest.raises(ValueError, match="strictly increasing"):
        step(x[::-1], z, Q, K, M, 1.0, U, 1e10, x_bl, z_bl)
    with pytest.raises(ValueError, match="must be downstream"):
        step(x, z, Q, K, M, 1.0, U, 1e10, x[-1] - 1.0, z_bl)
    with pytest.raises(ValueError, match="dt must be positive"):
        step(x, z, Q, K, M, 1.0, U, 0.0, x_bl, z_bl)
    with pytest.raises(ValueError, match="n must be positive"):
        step(x, z, Q, K, M, 0.0, U, 1e10, x_bl, z_bl)
    with pytest.raises(ValueError, match="same shape"):
        step(x, z[:-1], Q, K, M, 1.0, U, 1e10, x_bl, z_bl)

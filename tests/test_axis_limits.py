"""
Elevation axis limits, and the invariant that makes the lower one safe.

A plot that clips the thing it is plotting is worse than no plot: the reader
cannot tell a profile that left the axis from one that stopped moving. These
tests pin the bound rather than a margin.
"""

import numpy as np
import pytest

from drlp import LongProfile


YEAR = 31556926.0
K, M = 1e-9, 0.5
U = 0.5e-3 / YEAR


def _lp(nx=60, n=1.0, logK=-9.0, U_rate=U, zbl=0.0):
    return (LongProfile()
            .set_x(dx=500.0, nx=nx, x0=500.0)
            .set_A(k_xA=1.0, P_xA=7 / 4.0)
            .set_Q(R=1e-8)
            .set_K(10**logK, basis="Q")
            .set_mn(m=M, n=n)
            .set_uplift_rate(U_rate)
            .set_bl(z=zbl)
            .set_steady_state())


# ------------------------------------------- the invariant the bound rests on

@pytest.mark.parametrize("n", [0.5, 1.0, 1.5, 2.0, 2.5])
def test_no_node_ever_passes_below_base_level(n):
    # The claim elevation_limits' lower bound is built on. Checked every step,
    # through a base-level drop deep enough to invert the profile if it could.
    lp = _lp(n=n)
    lp.set_bl(z=-400.0)
    for _ in range(300):
        lp.evolve(nt=1, dt=1e11)
        assert np.all(lp.z >= lp.z_bl), f"n={n}: a node went below base level"


def test_the_bound_holds_for_a_sequence_of_drops():
    # Base level stepped down repeatedly, as a reader dragging a slider does.
    lp = _lp()
    for z_bl in (0.0, -50.0, -150.0, -400.0):
        lp.set_bl(z=z_bl)
        for _ in range(40):
            lp.evolve(nt=1, dt=1e11)
            assert np.all(lp.z >= lp.z_bl)


# --------------------------------------------------- elevation_limits itself

def test_limits_contain_the_profile_and_base_level():
    lp = _lp()
    lo, hi = lp.elevation_limits()
    assert lo < lp.z.min() and hi > lp.z.max()
    assert lo < lp.z_bl


def test_floor_reserves_room_for_a_drop_that_has_not_happened():
    # The whole point: a reader can drag base level to the slider's minimum,
    # and the axis set before they do must already hold it.
    lp = _lp()
    lo_without, _ = lp.elevation_limits()
    lo_with, _ = lp.elevation_limits(z_bl_floor=-400.0)
    assert lo_without > -400.0        # the naive limit would clip
    assert lo_with < -400.0           # the reserved one does not

    # and after the drop actually happens, the profile is still inside
    lp.set_bl(z=-400.0)
    for _ in range(400):
        lp.evolve(nt=1, dt=1e11)
    assert lp.z.min() >= lo_with
    assert lp.z_bl >= lo_with


@pytest.mark.parametrize("n", [0.5, 1.0, 2.0, 2.5])
@pytest.mark.parametrize("logK", [-10.0, -9.0, -7.0])
def test_limits_never_clip_across_the_parameter_corners(n, logK):
    # Steady relief spans four orders of magnitude over these settings, so a
    # fixed axis cannot work and the limits have to be computed. Each corner
    # must bracket its own profile.
    lp = _lp(n=n, logK=logK)
    lo, hi = lp.elevation_limits(z_bl_floor=-400.0)
    assert lo <= lp.z.min() and hi >= lp.z.max()
    assert lo <= -400.0


def test_padding_is_proportional_and_never_degenerate():
    lp = _lp()
    lo, hi = lp.elevation_limits(pad_frac=0.0)
    assert lo == pytest.approx(min(lp.z.min(), lp.z_bl))
    assert hi == pytest.approx(max(lp.z.max(), lp.z_bl))

    # a flat profile at base level has zero range; the limits must still be
    # ordered rather than collapsing to a point
    flat = _lp()
    flat.set_z(np.zeros_like(flat.x))
    flat.set_bl(z=0.0)
    lo, hi = flat.elevation_limits()
    assert hi > lo


def test_from_steady_state_a_falling_base_level_never_raises_the_top():
    # The upper limit is safe for as long as the parameters hold. Starting at
    # steady state, erosion balances uplift, so a base level that only falls
    # can only lower the profile: the top never moves up and the limit set at
    # reset keeps holding.
    lp = _lp(U_rate=2e-3 / YEAR)
    top0 = lp.z.max()
    lo, hi = lp.elevation_limits(z_bl_floor=-400.0)
    for z_bl in (-50.0, -200.0, -400.0):
        lp.set_bl(z=z_bl)
        for _ in range(200):
            lp.evolve(nt=1, dt=1e11)
            assert lp.z.max() <= top0 + 1e-6
            assert lo <= lp.z.min() and lp.z.max() <= hi


def test_raising_uplift_mid_run_is_what_breaks_the_upper_limit():
    # The genuinely unbounded direction, and the reason an animating caller
    # must EXPAND rather than trust one call: raise U and the profile grows
    # toward a new, higher steady state that the old limits do not contain.
    lp = _lp(U_rate=0.5e-3 / YEAR)
    _, hi = lp.elevation_limits(z_bl_floor=-400.0)
    lp.set_uplift_rate(4.0 * 0.5e-3 / YEAR)        # four times the uplift
    for _ in range(3000):
        lp.evolve(nt=1, dt=1e11)
    assert lp.z.max() > hi, "four times the uplift should outgrow the old axis"

    # the lower bound is unaffected: it rests on z >= z_bl, which uplift
    # cannot violate
    lo, _ = lp.elevation_limits(z_bl_floor=-400.0)
    assert np.all(lp.z >= lp.z_bl)
    assert lp.z.min() >= lo

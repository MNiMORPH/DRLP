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


# ----------------------------------------------- relax_limits, the tracker

from drlp.longprofile import relax_limits


def test_outward_is_immediate_so_nothing_is_ever_clipped():
    # The invariant that matters: if the target contains the data, so does the
    # result, on the very first call. No amount of relaxation delay is allowed
    # to leave the subject outside its own axis.
    cur = (0.0, 100.0)
    for target in ((-500.0, 100.0), (0.0, 900.0), (-500.0, 900.0)):
        lo, hi = relax_limits(cur, target, relax_frac=0.05)
        assert lo <= target[0] and hi >= target[1]


def test_inward_is_gradual_and_converges():
    # Tightening must not snap, and must not stall short of the fit.
    cur, target = (-1000.0, 1000.0), (-10.0, 10.0)
    frac = 0.05
    lo, hi = cur
    first = relax_limits((lo, hi), target, frac)
    assert first[0] > lo and first[1] < hi          # moved inward
    assert first[0] < target[0] and first[1] > target[1]   # but not all the way
    # Convergence is geometric at (1 - frac) per call, so assert the rate
    # rather than an iteration count: a count is a magic number and this is
    # the actual property. 990 m of gap at 0.95 per call needs 404 calls to
    # reach 1e-6, which is why an earlier draft of this test failed at 400.
    lo, hi = cur
    gap0 = target[0] - lo
    for n in (1, 10, 100):
        lo, hi = cur
        for _ in range(n):
            lo, hi = relax_limits((lo, hi), target, frac)
        assert (target[0] - lo) == pytest.approx(gap0 * (1 - frac) ** n, rel=1e-9)

    lo, hi = cur
    for _ in range(600):
        lo, hi = relax_limits((lo, hi), target, frac)
    assert lo == pytest.approx(target[0], abs=1e-9)
    assert hi == pytest.approx(target[1], abs=1e-9)


def test_monotone_and_never_overshoots_while_tightening():
    cur, target = (-1000.0, 1000.0), (-10.0, 10.0)
    lo, hi = cur
    for _ in range(500):
        n_lo, n_hi = relax_limits((lo, hi), target, 0.05)
        assert n_lo >= lo - 1e-12 and n_lo <= target[0] + 1e-9
        assert n_hi <= hi + 1e-12 and n_hi >= target[1] - 1e-9
        lo, hi = n_lo, n_hi


def test_the_two_degenerate_rates():
    cur, target = (-1000.0, 1000.0), (-10.0, 10.0)
    assert relax_limits(cur, target, 0.0) == cur            # never tightens
    assert relax_limits(cur, target, 1.0) == target         # snaps to the fit
    # but even at rate 0, outward still moves at once
    assert relax_limits(cur, (-5000.0, 1000.0), 0.0)[0] == -5000.0


def test_relax_frac_is_validated():
    with pytest.raises(ValueError, match="relax_frac"):
        relax_limits((0.0, 1.0), (0.0, 1.0), 1.5)
    with pytest.raises(ValueError, match="relax_frac"):
        relax_limits((0.0, 1.0), (0.0, 1.0), -0.1)


def test_tracking_a_live_model_never_clips_and_recovers_the_view():
    # The end-to-end property, against the model rather than numbers: drive
    # uplift up and then down, tracking every step, and require (1) the
    # profile and base level are inside the axis at every single step, and
    # (2) once uplift falls the axis tightens back, which is what the
    # expand-only rule could not do.
    lp = _lp()
    limits = lp.elevation_limits()
    frac = 0.05

    def fraction_used():
        lo, hi = limits
        return (lp.z.max() - min(lp.z.min(), lp.z_bl)) / (hi - lo)

    for U_rate, n_steps in ((2e-3 / YEAR, 1500), (0.1e-3 / YEAR, 4000)):
        lp.set_uplift_rate(U_rate)
        for _ in range(n_steps):
            lp.evolve(nt=1, dt=1e11)
            limits = relax_limits(limits, lp.elevation_limits(), frac)
            assert limits[0] <= lp.z.min() and limits[1] >= lp.z.max()
            assert limits[0] <= lp.z_bl

    # after uplift drops, the view is tight again rather than stranded
    assert fraction_used() > 0.5, f"view never recovered: {fraction_used():.3f}"

"""
Where DRLP stops being a model of a river.

The stream-power law is purely erosional, so ``dz/dt <= U`` everywhere and at
all times. These tests record what the model does under a base-level RISE, so
that the behavior is pinned rather than discovered later and mistaken for a
prediction.
"""

import numpy as np
import pytest

from drlp import LongProfile
from drlp.incision import step


YEAR = 31556926.0
K, M = 1e-9, 0.5


def _lp(U_mmyr, zbl, nx=30):
    return (LongProfile().set_x(dx=500.0, nx=nx, x0=500.0)
            .set_A(k_xA=1.0, P_xA=7 / 4.0).set_Q(R=1e-8)
            .set_K(10**-8.5, basis="Q").set_mn(m=M, n=1.0)
            .set_uplift_rate(U_mmyr * 1e-3 / YEAR).set_bl(z=zbl)
            .set_steady_state())


def test_erosion_never_raises_the_bed():
    # dz/dt <= U is the governing inequality, and it is what makes the model
    # fall-only. No node may rise faster than rock uplift, under any forcing.
    U_mmyr = 0.3
    lp = _lp(U_mmyr, -300.0)
    lp.set_bl(z=0.0)                      # a 300 m RISE
    dt = 1e11
    for _ in range(50):
        z_before = lp.z.copy()
        lp.evolve(nt=1, dt=dt)
        assert np.all(lp.z <= z_before + U_mmyr * 1e-3 / YEAR * dt + 1e-9)


def test_base_level_rise_is_answered_at_the_uplift_rate_not_a_fluvial_one():
    # The bed comes up at exactly U*t: rock uplift, a tectonic mechanism, with
    # nothing fluvial in it. A real river aggrades, and far faster.
    U_mmyr = 0.3
    lp = _lp(U_mmyr, -300.0)
    z0 = lp.z.copy()
    lp.set_bl(z=-100.0)                   # a 200 m rise
    lp.evolve(nt=100, dt=1e11)
    expected = U_mmyr * 1e-3 / YEAR * 100 * 1e11
    assert lp.z[-1] - z0[-1] == pytest.approx(expected, rel=1e-6)


def test_with_no_uplift_the_bed_cannot_rise_at_all():
    # E >= 0 and U = 0 give dz/dt <= 0, so a drowned reach stays drowned for
    # ever. This is the clearest statement of the validity limit.
    lp = _lp(0.0, -300.0)
    z0 = lp.z.copy()
    lp.set_bl(z=-100.0)
    lp.evolve(nt=500, dt=1e11)
    assert np.array_equal(lp.z, z0)
    assert np.all(lp.z < -100.0)          # every node below base level


def test_a_drowned_node_freezes_while_the_reach_above_keeps_incising():
    # The guard acts on the node that sees the risen base level. Nodes above
    # it drain to their own neighbors and go on eroding, and the sediment
    # they produce is not deposited anywhere -- there is no deposition term.
    x = np.array([0.0, 500.0, 1000.0])
    z = np.array([30.0, 20.0, 10.0])
    z_new = step(x, z, 1.0, 1e-9, M, 1.0, 0.0, 1e11, 1500.0, 100.0)
    assert z_new[-1] == z[-1]             # drowned: frozen
    assert np.all(z_new[:-1] < z[:-1])    # subaerial: still incising

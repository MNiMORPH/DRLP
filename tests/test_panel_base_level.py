"""
Base-level semantics in the demo, pinned because they have changed twice.

The rule: base level falls freely, and the ONLY way it rises is by riding the
uplift. Raising the slider does nothing at all -- not a rebuild, not a shift,
nothing -- because ``dz/dt = U - E`` with ``E >= 0`` gives ``dz/dt <= U`` and
there is no process here that raises a bed faster than its rock.

An earlier version rebuilt the river at the higher base level, which looked
like a response and was not one: from 190 kyr into a transient it discarded a
knickpoint and 3242 m of relief and substituted a fresh steady profile at
t = 0.

Local only: the panel needs the ``demo`` extra.
"""

import importlib.util
import pathlib

import numpy as np
import pytest

pytest.importorskip("panel")
pytest.importorskip("bokeh")

_PANEL = pathlib.Path(__file__).resolve().parents[1] / "interactive_demo" / "drlp_panel.py"


@pytest.fixture
def m():
    spec = importlib.util.spec_from_file_location("drlp_panel_bl", _PANEL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.bl_lock.value = False
    mod.U_sl.value, mod.n_sl.value, mod.logK.value = 0.5, 1.0, -9.0
    mod.zbl.value = 0.0
    mod.do_reset()
    return mod


def _into_a_transient(m, frames=60):
    m.zbl.value = -1200.0
    for _ in range(frames):
        m.step()
    return m.sim["t"], m.sim["lp"].z.copy()


def test_raising_the_slider_does_nothing_at_all(m):
    t0, z0 = _into_a_transient(m)
    assert t0 > 0 and m.sim["z_bl"] == -1200.0

    for attempt in (-400.0, -20.0, 0.0):
        m.zbl.value = attempt
        assert m.zbl.value == -1200.0, "the slider did not snap back"
        assert m.sim["z_bl"] == -1200.0, "base level moved"
        assert m.sim["t"] == t0, "the clock was reset"
        assert np.array_equal(m.sim["lp"].z, z0), "the profile was rebuilt"


def test_lowering_the_slider_still_forces_the_running_river(m):
    t0, z0 = _into_a_transient(m)
    m.zbl.value = -1600.0
    assert m.sim["z_bl"] == -1600.0
    m.step()
    assert m.sim["t"] > t0                      # the clock kept running
    assert not np.array_equal(m.sim["lp"].z, z0)


def test_the_lock_is_the_only_route_up(m):
    _into_a_transient(m)
    before = m.sim["z_bl"]
    m.bl_lock.value = True
    for _ in range(800):
        m.step()
    assert m.sim["z_bl"] > before, "the lock did not raise base level"
    # it rose at the uplift rate, not by some other amount
    U = m.U_sl.value * 1e-3 / m.YEAR
    assert m.sim["z_bl"] - before == pytest.approx(U * 800 * m.DT, rel=1e-9)


def test_unlocking_hands_back_a_slider_that_can_fall_again(m):
    _into_a_transient(m)
    m.bl_lock.value = True
    for _ in range(800):
        m.step()
    relief_locked = m.sim["lp"].z[0] - m.sim["z_bl"]

    m.bl_lock.value = False
    assert not m.zbl.disabled
    assert m.zbl.start <= m.zbl.value <= m.zbl.end
    # the datum shift preserved the river exactly
    assert m.sim["lp"].z[0] - m.sim["z_bl"] == pytest.approx(relief_locked)
    # and it can be lowered from here
    target = m.zbl.value - 200.0
    m.zbl.value = target
    assert m.sim["z_bl"] == pytest.approx(target)


def test_with_no_uplift_base_level_cannot_rise_even_locked(m):
    # Not an interface dead end: a landscape that is not rising has no way to
    # bring its base level up, and the model says so.
    m.U_sl.value = 0.0
    m.zbl.value = 0.0
    m.do_reset()
    m.zbl.value = -500.0
    for _ in range(100):
        m.step()
    before = m.sim["z_bl"]
    m.bl_lock.value = True
    for _ in range(500):
        m.step()
    assert m.sim["z_bl"] == pytest.approx(before)


def test_the_slider_cannot_reach_above_its_starting_point(m):
    # end = 0 and the ratchet below it, so there is no upward route at all
    # through the slider itself.
    assert m.zbl.end == 0.0
    m.zbl.value = -300.0
    m.zbl.value = 500.0                 # beyond the slider's own range
    assert m.zbl.value <= 0.0
    assert m.sim["z_bl"] <= 0.0

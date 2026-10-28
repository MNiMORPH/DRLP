"""
Structural checks on the demo's axes.

These assert on what GOVERNS the rendered axis rather than on values the panel
has just written, which is the mistake they exist to catch: ``figure()``
returns a ``DataRange1d`` that auto-ranges to the data in the browser, so
reading back ``fig.y_range.start`` in Python returns the last thing Python
wrote regardless of what the client will actually draw. Every headless check
of the axis tracking passed while the deployed page did not follow the
profile, because they read the assignment instead of the mechanism.

Local only: the panel needs the ``demo`` extra. Skipped where it is absent.
"""

import importlib.util
import pathlib

import pytest

pytest.importorskip("panel")
pytest.importorskip("bokeh")

from bokeh.models import Range1d


_PANEL = pathlib.Path(__file__).resolve().parents[1] / "interactive_demo" / "drlp_panel.py"


@pytest.fixture(scope="module")
def panel_module():
    spec = importlib.util.spec_from_file_location("drlp_panel", _PANEL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_both_axes_are_explicit_ranges_not_data_ranges(panel_module):
    # The bug this pins. A DataRange1d recomputes its own bounds client-side,
    # so the panel cannot drive it; a Range1d is authoritative.
    assert isinstance(panel_module.fig.y_range, Range1d), (
        "y_range must be an explicit Range1d: a DataRange1d auto-ranges in the "
        "browser and ignores the panel's tracking"
    )
    assert isinstance(panel_module.fig.x_range, Range1d)


def test_the_plot_fills_its_design_width(panel_module):
    # A plot narrower than the frame it is scaled inside leaves the demo
    # sitting in the middle of empty space. hillcreep uses 880 against 900.
    assert panel_module.DESIGN_WIDTH - panel_module.fig.width <= 40, (
        f"figure is {panel_module.fig.width} px inside a "
        f"{panel_module.DESIGN_WIDTH} px design width"
    )


def test_x_range_spans_the_grid_including_the_ghost_node(panel_module):
    lp = panel_module.sim["lp"]
    assert panel_module.fig.x_range.start == pytest.approx(lp.x.min() / 1000.0)
    assert panel_module.fig.x_range.end == pytest.approx(lp.x_bl / 1000.0)


def test_tracking_moves_the_authoritative_range(panel_module):
    # Drive the model and require the Range1d itself to follow, which is the
    # check the earlier headless tests only appeared to make.
    m = panel_module
    m.bl_lock.value = False
    m.U_sl.value, m.n_sl.value, m.logK.value = 0.5, 1.0, -9.0
    m.zbl.value = 0.0
    m.do_reset()
    assert isinstance(m.fig.y_range, Range1d)
    tall = m.fig.y_range.end

    m.U_sl.value = 2.0                       # a taller steady state
    for _ in range(2000):
        m.step()
    assert m.fig.y_range.end > tall, "axis did not grow with the profile"
    grown = m.fig.y_range.end

    m.U_sl.value = 0.1                       # and a much shorter one
    for _ in range(4000):
        m.step()
    assert m.fig.y_range.end < grown, "axis did not tighten back"

    # and at no point may the profile sit outside it
    z = m.sim["lp"].z
    assert z.min() >= m.fig.y_range.start - 1e-6
    assert z.max() <= m.fig.y_range.end + 1e-6

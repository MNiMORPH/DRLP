"""
Interactive DRLP demo - a detachment-limited river long profile, in real time.

This is the source for the live, in-browser demo. It is compiled to a
standalone WebAssembly app (no server, no install) with::

    artesian build interactive_demo/drlp_panel.py -o _artesian_build \\
        -p . -p ../artesian -r numpy scipy --serve

`artesian build` runs the app in the *building* environment to discover what it
serves, so `drlp` must be importable here, not merely wheeled for the browser:
`pip install -e .` first.

DRLP then runs entirely in the browser via Pyodide, which carries both numpy
and scipy. Press play and drag the sliders while it runs.

The sibling demo for the gravel-bed, transport-limited case is GRLP's
`interactive_demo/grlp_panel.py`, and this one deliberately mirrors its
layout. The physics is the contrast: GRLP is diffusive, DRLP is advective.
"""
import numpy as np
import panel as pn
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource

from drlp import LongProfile

pn.extension()

# Width in CSS pixels the app is laid out for; the embedding page scales the
# demo above it rather than stretching it. 900 is the deployed convention --
# hillcreep's panels declare it and every GeomorphOnline exercise page carries
# data-design-width="900".
DESIGN_WIDTH = 900

YEAR = 31556926.          # seconds per year
DT = 1e11                 # ~3 kyr advanced per animation frame
NX, DX = 60, 500.         # 0.5 to 30 km
K_XA, P_XA = 1.0, 7 / 4.  # Hack's law:  A = k_xA x**P_xA
RUNOFF = 1e-8             # m/s, about 315 mm/yr


def make_steady(logK, m, n, U_mmyr, zbl):
    """A profile started at the steady state for the given settings."""
    lp = LongProfile()
    lp.set_x(dx=DX, nx=NX, x0=DX)
    lp.set_A(k_xA=K_XA, P_xA=P_XA)
    lp.set_Q(R=RUNOFF)
    lp.set_K(10.**logK, basis="Q")
    lp.set_mn(m=m, n=n)
    lp.set_uplift_rate(U_mmyr * 1e-3 / YEAR)
    lp.set_bl(z=zbl)
    lp.set_steady_state()
    return lp


LOGK0, M0, N0, U0, ZBL0 = -8.5, 0.5, 1.0, 0.3, 0.0
# `sim` (NOT `state`, which collides with panel's global pn.state) holds the
# evolving model between animation frames.
sim = {"lp": make_steady(LOGK0, M0, N0, U0, ZBL0), "t": 0.}
_lp = sim["lp"]


def _bl_xy(zbl):
    return {"x": [_lp.x.min() / 1000., _lp.x.max() / 1000.], "z": [zbl, zbl]}


profile = ColumnDataSource(data={"x": _lp.x / 1000., "z": _lp.z})
baselevel = ColumnDataSource(data=_bl_xy(ZBL0))

fig = figure(height=380, width=680, title="t = 0.0 kyr",
             x_axis_label="Downstream distance [km]",
             y_axis_label="Elevation [m]")
fig.line("x", "z", source=profile, line_width=3)
fig.line("x", "z", source=baselevel, line_width=1, line_dash="dashed",
         color="gray", legend_label="base level")
fig.legend.location = "top_right"

logK = pn.widgets.FloatSlider(
    name="Erodibility  log₁₀ K  (units depend on m and n)",
    start=-10, end=-7, step=0.1, value=LOGK0, format="0.0")
m_sl = pn.widgets.FloatSlider(name="Discharge exponent  m",
                              start=0.1, end=1.0, step=0.05, value=M0,
                              format="0.00")
n_sl = pn.widgets.FloatSlider(name="Slope exponent  n",
                              start=0.5, end=2.5, step=0.1, value=N0,
                              format="0.0")
U_sl = pn.widgets.FloatSlider(name="Rock uplift rate  U  [mm/yr]",
                              start=0.0, end=2.0, step=0.1, value=U0,
                              format="0.0")
zbl = pn.widgets.FloatSlider(name="Base level  [m]",
                             start=-400, end=400, step=20, value=ZBL0)

run = pn.widgets.Toggle(name="▶ Run", value=False)
reset = pn.widgets.Button(name="Set to steady state", button_type="primary")


def _rescale_y(z, zbl_value):
    """Fit the axis to the profile, with headroom.

    The steady-state relief spans orders of magnitude across these sliders --
    n changes how hard the profile answers a given U/K -- so a fixed range
    that suits one corner hides another. Rescale when the profile is reset,
    and hold the range steady while it runs.
    """
    lo, hi = min(z.min(), zbl_value), max(z.max(), zbl_value)
    pad = 0.08 * max(hi - lo, 1.0)
    fig.y_range.start, fig.y_range.end = lo - pad, hi + pad


def step(event=None):
    """Advance one frame, reading the sliders as live settings."""
    lp = sim["lp"]
    lp.set_K(10.**logK.value, basis="Q")
    lp.set_mn(m=m_sl.value, n=n_sl.value)
    lp.set_uplift_rate(U_sl.value * 1e-3 / YEAR)
    lp.set_z_bl(zbl.value)
    lp.evolve(nt=1, dt=DT)
    sim["t"] += DT
    profile.data = {"x": lp.x / 1000., "z": lp.z}
    baselevel.data = _bl_xy(zbl.value)
    fig.title.text = "t = %.1f kyr" % (sim["t"] / (1000. * YEAR))


def do_reset(event=None):
    """Restart from steady state for the current slider settings."""
    sim["lp"] = make_steady(logK.value, m_sl.value, n_sl.value,
                            U_sl.value, zbl.value)
    sim["t"] = 0.
    lp = sim["lp"]
    profile.data = {"x": lp.x / 1000., "z": lp.z}
    baselevel.data = _bl_xy(zbl.value)
    fig.title.text = "t = 0.0 kyr"
    _rescale_y(lp.z, zbl.value)


# A single play/pause button drives the animation via a periodic callback (the
# timer only runs while the toggle is on).
_ticker = pn.state.add_periodic_callback(step, period=33, start=False)  # ~30 fps


def toggle_run(event):
    if event.new:
        _ticker.start()
        run.name = "⏸ Pause"
    else:
        _ticker.stop()
        run.name = "▶ Run"


run.param.watch(toggle_run, "value")
reset.on_click(do_reset)

_rescale_y(_lp.z, ZBL0)

pn.Column(
    pn.pane.Markdown(
        "### A detachment-limited river long profile\n"
        "Press **▶** to run, then drag the sliders while it plays.\n\n"
        "Erosion is set locally by the flow here, not by a divergence of "
        "sediment flux: *dz/dt* = *U* − *K* *Q^m* *S^n*. A perturbation "
        "therefore travels upstream as a step rather than spreading and "
        "flattening. (Its gravel-bed, transport-limited sibling GRLP does the "
        "opposite.)\n\n"
        "**Set to steady state** jumps to the profile that balances the "
        "current uplift against the current erodibility, and refits the "
        "elevation axis. Changing a slider while the model runs leaves the "
        "profile where it is and lets it respond.\n\n"
        "Base level is valid going **down**, not up. Stream power only "
        "erodes, so the bed comes back up at the rock uplift rate and not at "
        "all without one – a real river would aggrade, and that is GRLP's "
        "and SRLP's job. Raising the slider above where you dropped it gives "
        "you a drowned reach rather than a prediction.\n\n"
        "Discharge is not a slider. It comes from Hack's law and a fixed "
        "runoff rate (*A* = *x*¹⋅⁷⁵, *Q* = 10⁻⁸ "
        "*A*), which leaves the erosion law and base level as the things you "
        "change. The clock reads in thousands of years."),
    pn.Row(run, reset),
    logK, m_sl, n_sl, U_sl, zbl,
    fig,
    sizing_mode="stretch_width", max_width=DESIGN_WIDTH,
).servable(title="DRLP interactive demo")

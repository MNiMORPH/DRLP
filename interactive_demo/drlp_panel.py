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
from bokeh.models import ColumnDataSource, Range1d

from drlp import LongProfile
from drlp.longprofile import relax_limits

pn.extension()

# Width in CSS pixels the app is laid out for; the embedding page scales the
# demo above it rather than stretching it. 900 is the deployed convention --
# hillcreep's panels declare it and every GeomorphOnline exercise page carries
# data-design-width="900".
DESIGN_WIDTH = 900

YEAR = 31556926.          # seconds per year
DT = 1e11                 # ~3 kyr advanced per animation frame
# The elevation axis tightens with a time constant rather than a magic rate:
# 1 - exp(-frame/TAU) of the remaining gap per frame, at ~30 fps. Half a
# second reads as the view following the river rather than chasing it.
TAU_Y = 0.5               # seconds
RELAX_Y = 1.0 - np.exp(-0.033 / TAU_Y)
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


# A mountain river rather than a hill: about 2 km of relief at these
# settings, against 390 m at the previous defaults.
LOGK0, M0, N0, U0, ZBL0 = -9.0, 0.5, 1.0, 0.5, 0.0
# `sim` (NOT `state`, which collides with panel's global pn.state) holds the
# evolving model between animation frames.
sim = {"lp": make_steady(LOGK0, M0, N0, U0, ZBL0), "t": 0., "z_bl": ZBL0}
_lp = sim["lp"]


def _bl_xy(z_bl):
    return {"x": [_lp.x.min() / 1000., _lp.x.max() / 1000.], "z": [z_bl, z_bl]}


profile = ColumnDataSource(data={"x": _lp.x / 1000., "z": _lp.z})
baselevel = ColumnDataSource(data=_bl_xy(ZBL0))

# 880 px against a 900 px design width, as hillcreep's panels are: the plot
# should fill the frame it is scaled inside, not sit in the middle of it.
fig = figure(height=420, width=880, title="t = 0.0 kyr",
             x_axis_label="Downstream distance [km]",
             y_axis_label="Elevation [m]")

# BOTH ranges are explicit Range1d, and that is load-bearing. figure() hands
# back a DataRange1d, which auto-ranges to the renderers' data IN THE BROWSER
# with range_padding = 0.1. Assignments to its start and end from Python are
# hints the client recomputes around rather than limits that govern, so a
# per-frame assignment fights the auto-range instead of driving the axis --
# and the axis then does not follow the profile. hillcreep replaces its ranges
# for the same reason ("replaced on every redraw").
#
# This also defeated the headless tests of the tracking: reading back
# fig.y_range.start returns whatever Python last wrote to it, whichever class
# it is, so those checks asserted on my own assignment rather than on what
# renders. The structural assertion is in tests/test_panel_axis.py.
fig.x_range = Range1d(0.0, 1.0)      # set from the grid below
fig.y_range = Range1d(0.0, 1.0)      # replaced on every redraw
fig.line("x", "z", source=profile, line_width=3)
fig.line("x", "z", source=baselevel, line_width=1, line_dash="dashed",
         color="gray", legend_label="base level")
fig.legend.location = "top_right"
fig.x_range.start = float(_lp.x.min() / 1000.)
fig.x_range.end = float(_lp.x_bl / 1000.)

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
# Base level falls, and never rises on a river that already exists: stream
# power erodes and never deposits, so nothing here can bring a drowned bed
# back up.
#
# Lowering it is a forcing applied to the running profile. RAISING it rebuilds
# the river at the new base level instead, which is a different initial
# condition rather than an aggradation the model cannot perform. Refusing the
# drag outright was the first attempt and it was worse: once the slider
# reached its floor, nothing could ever bring it back and the control was
# one-way for the rest of the session.
zbl = pn.widgets.FloatSlider(name="Base level  [m]  (lower to force, raise to rebuild)",
                             start=-2000, end=0, step=20, value=ZBL0)
_bl_floor = {"z": ZBL0}


def on_base_level(event):
    if _syncing["on"] or bl_lock.value:
        return                          # the lock is driving it
    if event.new > _bl_floor["z"]:
        _bl_floor["z"] = event.new
        do_reset()                      # a rise is a rebuild, not a forcing
    else:
        _bl_floor["z"] = event.new
        sim["z_bl"] = event.new

run = pn.widgets.Toggle(label="▶ Run", value=False)
reset = pn.widgets.Button(label="Set to steady state", button_type="primary")

# Locking base level to the uplift rate puts the outlet on the same rising
# rock as the rest of the profile, so the WHOLE long profile goes up at U and
# there is no RELATIVE base-level change left. The river still erodes -- it
# removes rock at U minus whatever it is actually incising -- but it loses its
# steady state, because the forcing that held the steady state is gone. A
# landscape can uplift everywhere and give its rivers nothing to cut into.
#
# Unlocked, base level is independent and the only direction is down, because
# stream power erodes and never deposits.
bl_lock = pn.widgets.Toggle(label="\U0001f513 Base level free",
                            button_type="default", width=210)
_syncing = {"on": False}


def toggle_lock(event):
    if event.new:
        bl_lock.label = "\U0001f512 Locked to uplift"
        bl_lock.button_type = "warning"
        zbl.disabled = True
    else:
        bl_lock.label = "\U0001f513 Base level free"
        bl_lock.button_type = "default"
        # Hand control back without a jump. The lock may have carried base
        # level far above the slider's range -- 19 km after 38 Myr at 0.5
        # mm/yr -- and simply clamping it would drop base level by that much
        # in one frame. So shift the datum instead: subtract the same constant
        # from the profile and from base level, which changes nothing physical
        # (the model has no absolute datum, only relief and slope) and leaves
        # the river exactly as the lock left it.
        target = float(min(max(sim["z_bl"], zbl.start), zbl.end))
        shift = sim["z_bl"] - target
        sim["lp"].z = sim["lp"].z - shift
        sim["z_bl"] = target
        _syncing["on"] = True
        zbl.value = target
        _syncing["on"] = False
        _bl_floor["z"] = target
        zbl.disabled = False
        profile.data = {"x": sim["lp"].x / 1000., "z": sim["lp"].z}
        baselevel.data = _bl_xy(target)
        _fit_y(sim["lp"])



def _fit_y(lp):
    """Fit the elevation axis to the profile, tightly.

    Steady relief spans four orders of magnitude across these sliders -- n
    changes how hard the profile answers a given U/K -- so no fixed range
    works and the axis has to be computed. It is fitted TIGHTLY rather than
    reserving room for a base-level drop: the axis now follows the drop as it
    happens, so holding empty space for one would only waste the plot. The
    reserved version gave the river 44% of the height at these defaults.
    """
    lo, hi = lp.elevation_limits()
    fig.y_range.start, fig.y_range.end = lo, hi


def _track_y(lp):
    """Ease the axis toward a tight fit, in both directions, every frame.

    The limits evolve through time rather than snapping: both edges move
    about 6% of the remaining gap per frame, which is half a second at 30
    fps. Only the PROFILE is guaranteed to stay in frame, so a base-level
    drop pans the view down over the following half-second instead of
    widening the axis to its final extent in one frame -- which is what an
    immediate-outward rule did, changing the axis on 44 of the 300 frames
    after a drop and reading as a jump and then a freeze.
    """
    fig.y_range.start, fig.y_range.end = relax_limits(
        (fig.y_range.start, fig.y_range.end),
        lp.elevation_limits(),
        RELAX_Y,
        contain=(float(lp.z.min()), float(lp.z.max())))


def step(event=None):
    """Advance one frame, reading the sliders as live settings."""
    lp = sim["lp"]
    U = U_sl.value * 1e-3 / YEAR
    lp.set_K(10.**logK.value, basis="Q")
    lp.set_mn(m=m_sl.value, n=n_sl.value)
    lp.set_uplift_rate(U)

    if bl_lock.value:
        sim["z_bl"] += U * DT           # the outlet rides the rising rock
    else:
        sim["z_bl"] = zbl.value
    lp.set_z_bl(sim["z_bl"])

    lp.evolve(nt=1, dt=DT)
    sim["t"] += DT
    profile.data = {"x": lp.x / 1000., "z": lp.z}
    baselevel.data = _bl_xy(sim["z_bl"])
    _track_y(lp)
    fig.title.text = "t = %.1f kyr      base level = %.0f m" % (
        sim["t"] / (1000. * YEAR), sim["z_bl"])


def do_reset(event=None):
    """Restart from steady state for the current slider settings."""
    sim["lp"] = make_steady(logK.value, m_sl.value, n_sl.value,
                            U_sl.value, zbl.value)
    sim["t"] = 0.
    sim["z_bl"] = zbl.value
    lp = sim["lp"]
    profile.data = {"x": lp.x / 1000., "z": lp.z}
    baselevel.data = _bl_xy(sim["z_bl"])
    fig.title.text = "t = 0.0 kyr      base level = %.0f m" % sim["z_bl"]
    _bl_floor["z"] = zbl.value          # a rebuild releases the ratchet
    _fit_y(lp)


# A single play/pause button drives the animation via a periodic callback (the
# timer only runs while the toggle is on).
_ticker = pn.state.add_periodic_callback(step, period=33, start=False)  # ~30 fps


def toggle_run(event):
    if event.new:
        _ticker.start()
        run.label = "⏸ Pause"
    else:
        _ticker.stop()
        run.label = "▶ Run"


run.param.watch(toggle_run, "value")
bl_lock.param.watch(toggle_lock, "value")
zbl.param.watch(on_base_level, "value")
reset.on_click(do_reset)

_fit_y(_lp)

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
    logK, m_sl, n_sl, U_sl,
    pn.Row(zbl, bl_lock, sizing_mode="stretch_width", max_width=DESIGN_WIDTH),
    fig,
    sizing_mode="stretch_width", max_width=DESIGN_WIDTH,
).servable(title="DRLP interactive demo")

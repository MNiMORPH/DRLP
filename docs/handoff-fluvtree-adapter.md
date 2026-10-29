# Handoff: make FluvTree's stream power an adapter to DRLP

**Written 28 October 2026, from DRLP.** Nothing in FluvTree was modified.
Verified against FluvTree at `c266a7f` on `main`, 41 commits ahead of
`origin/main`, with `pytest tests/test_stream_power.py tests/test_bedrock.py`
reporting 7 passed in 0.52 s.

## Why

`processes/grlp.py` already states the convention this note applies:

> A thin adapter that reuses GRLP's solver numerics unchanged. [...]
> Dependency direction: FluvTree depends on published GRLP, never the reverse.

`processes/stream_power.py` (54 lines) and `solvers/advection.py` (105 lines)
are the one process in that family carrying their own numerics instead of
adapting a published model. **DRLP is now published**, GPL-3.0-or-later, at
<https://github.com/MNiMORPH/DRLP>, so the asymmetry can go: `stream_power.py`
becomes a thin adapter and `advection.py`'s node update is deleted rather than
maintained in two places.

What FluvTree gains by doing it: general `n`, and per-node `K` and `U`.
`incise_n1_step` takes both as scalars (`c = dt * K * Q[i] ** m / (x_r -
x[i])`), and `advection.py` is `n = 1` only, its own docstring calling
`n != 1` "a later addition".

**The uplift half of that is a capability FluvTree already owns and cannot
reach.** `common/rates.broadcast_rate` normalizes a rate spec -- scalar, per
segment, or a callable `spec(x)` -- to one array per segment, and
`common/tectonics.uplift_rate` wraps it for exactly this purpose; the
diffusion solver consumes it as `source_rate`. But `StreamPower.__init__`
takes `U=0.0` as a scalar and its docstring says "Per-segment uplift via a
graph field is a later extension". Adapting to `drlp.incision.step`, which
takes `U` per node, wires an existing helper into a process that currently
refuses it. No new machinery.

## The entry point

```python
from drlp.incision import step

z_new = step(x, z, Q, K, m, n, U, dt, x_bl, z_bl)
```

Pure: it returns a new array and leaves `z` alone, which is what lets the time
term read the old elevations while receivers read the new ones. The
conventions were chosen to match FluvTree's sweep so no translation layer is
needed:

- index `0` is the UPSTREAM end, the last index the downstream-most node;
- `x` increases downstream, so `dx = x_r - x_i > 0`;
- the receiver of the last real node is the ghost node `(x_bl, z_bl)`, passed
  in rather than owned.

That last point is the whole reason the signature looks like this. In
`incise_n1_step` the receiver of a reach's downstream-most node is
`network.get_segment_field(d, "z")[0]` and `...("x")[0]` of the downstream
reach, or base level at the outlet. **Those two values go straight into
`x_bl`, `z_bl`.** The sweep order, `sweep_order(network)`, stays in FluvTree,
which is where topology belongs: DRLP is one-dimensional on purpose and has no
`Network` (`HANDOFF-AMENDMENT-1.md`).

So the adapter is the existing loop over reaches with the inner node loop
replaced by one call, and `z_old` kept per reach exactly as now.

## What supersedes the earlier note

`docs/handoff-fluvtree-general-n.md`, written 16 October, specified adding
general `n` *inside* FluvTree. Do not follow it as a work order; it is now a
numerics reference, and two of its conclusions were overtaken by measurement:

1. **Solve in Landlab's dimensionless variable, not in elevation.** Its
   bracket `[z_r, z_old + dt*U]` is ill-conditioned: `f` evaluates
   `z - z_old - dt*U`, a difference of absolute elevations, and once the
   erosion term falls below the rounding error of that difference the sign at
   the upper bracket is noise. Over 20,000 draws that sweep an absolute
   elevation, the dimensional bracket fails 23 times and the dimensionless one
   never. `prototypes/00_dimensionless_equivalence.py` shows the two are the
   same equation, `f_dim = D0 * f_x`.
2. **Its 20,000-draw verification swept `n`, `K`, `Q`, `m`, `dx` and `dt` but
   not an absolute elevation**, which is exactly where the failure lives, and
   `|z| >> relief` is the ordinary case for a real profile.

Both are already handled inside `drlp.incision.step`, which is the argument
for adapting rather than reimplementing.

## Tests FluvTree should keep and add

Its existing seven should pass unchanged: the adapter's `n = 1` path is Braun
and Willett's closed form, which is what `advection.py` computes now. Then the
network cases DRLP cannot cover, because they are about topology rather than
the node update:

- steady slopes following discharge across a confluence, which
  `test_network_steady_state_slopes_follow_discharge` already does;
- a knickpoint crossing a confluence into both upstream branches;
- general `n` on a network, which is new capability rather than a port.

## What should NOT be ported, and why

Scanned DRLP against FluvTree on 28 October 2026. Four things in DRLP exist
*because* DRLP is one-dimensional, and copying them into a network model would
be a regression rather than a gain:

1. **Hack's law, `A = k_xA x**P_xA`.** DRLP needs it because it has no
   network. FluvTree accumulates drainage area through topology --
   `generate/topology.py` carries `segment_length_area_ratio`, `supply_area`
   and `segment_areas` -- which is the real thing Hack's law approximates.
2. **The `A -> Q` power-law hydrology.** FluvTree carries `Q` as a graph field
   set by its generators. Imposing `Q = k_AQ A**P_AQ` over that would replace
   accumulated discharge with a one-dimensional stand-in.
3. **`set_K`'s `basis` switch (`"Q"` / `"A"` / `"Q/B"`).** It lets one scalar
   driver stand in for several quantities DRLP does not carry. FluvTree has
   the actual fields, width included, for the alluvial closures.
4. **`elevation_limits` and `relax_limits`.** Written for an animated browser
   view at 30 fps. `plot.py` draws static matplotlib figures, where autoscale
   is correct and there is nothing to track.

## Two things that flow the other way

1. **`broadcast_rate`'s callable form is better than DRLP's.** DRLP's
   `_as_node_array` takes a scalar or an array; FluvTree's takes a scalar, a
   per-segment value, or `spec(x)`. A callable `U(x)` imposes a fault without
   precomputing an array, and DRLP should adopt it.
2. **`analysis.slope_area` and `plot.slope_area` are the shape DRLP's unbuilt
   Phase 5 should take.** They already handle the distinction DRLP arrived at
   from the other direction: the abscissa is selectable, defaulting to `Q`
   (slope-discharge), and the docstring notes that against `A` the fit's
   negative is the concavity index. Fitting against `Q` gives `m/n`; against
   `A` it gives `theta = m P_AQ / n`, which is not `m/n` unless the hydrology
   is linear. DRLP should follow that interface rather than invent one.

## One gap in both

Neither documents that the stream-power law cannot deposit. `E >= 0` gives
`dz/dt <= U` everywhere, so a base-level **rise** has no fluvial answer: the
bed comes up only at the rock uplift rate, and not at all without one.
`stream_power.py` and `advection.py` say nothing about it, and only
`fixed_bed.py` mentions deposition at all. DRLP states it in its module
docstring, in `set_z_bl`, in `tests/test_validity.py`, and in the browser
demo. **That paragraph is worth adding to `stream_power.py` whether or not the
adapter is written** -- it is prose, not code, and it is the one thing a
reader can get wrong without any error appearing.

## Scope

The sweep order, base-level handling, in-place graph mutation, the diffusion
solver and the alluvial closures do not change. Neither does the dependency
direction: FluvTree depends on published DRLP, never the reverse.

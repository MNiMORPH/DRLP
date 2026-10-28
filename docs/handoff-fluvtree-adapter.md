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

What FluvTree gains by doing it: general `n`, spatially variable `K`, `U` and
`Q`, Hack's law and a power-law hydrology, and a test suite that checks the
node update against closed forms. `advection.py` is `n = 1` only, and its own
docstring calls `n != 1` "a later addition".

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

## Scope

The sweep order, base-level handling, in-place graph mutation, the diffusion
solver and the alluvial closures do not change. Neither does the dependency
direction: FluvTree depends on published DRLP, never the reverse.

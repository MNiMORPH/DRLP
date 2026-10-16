# Handoff: general `n` in FluvTree's stream-power solver

**Prepared 16 October 2026.** Written against FluvTree at `c266a7f` (`main`,
41 commits ahead of `origin/main`). Nothing in the repository was modified.

## Why

`solvers/advection.py` solves `dz/dt = U - K Q**m S**n` for `n = 1` only, and
says so:

> Implements the `n = 1` linear rung [...] (celerity `~ S**(n-1)` is
> slope-independent at `n = 1`, so the sweep is exact in one pass). `n != 1`
> (a Newton step per node) is a later addition.

An ESCI 4701 exercise wants students to vary `n`. The intended consumer is a
**separate repository** that imports `fluvtree`; nothing teaching-specific
should land here.

`m` needs no work. The course will convert discharge to drainage area with a
runoff rule `Q = R A`, so `K_Q Q**m = (K_Q R**m) A**m` and only `K` is
rescaled.

## Current state, verified

    src/fluvtree/solvers/advection.py     105 lines
    src/fluvtree/processes/stream_power.py 54 lines
    tests/test_stream_power.py             85 lines
    tests/test_bedrock.py                  29 lines

    $ python3 -m pytest tests/test_stream_power.py tests/test_bedrock.py -q
    7 passed in 0.45s

The sweep is Braun & Willett (2013): reaches outlet-first, nodes last-to-first,
so every receiver is already updated. At `n = 1` the node update is closed-form:

    c = dt * K * Q[i]**m / (x_r - x[i])
    z[i] = (z_old[i] + dt*U + c*z_r) / (1.0 + c)

## Step 0 — read the field first, before writing anything

Two mature implementations solve exactly this, and both chose a root-finder
rather than plain Newton. Look at what they do and why before committing to an
approach.

- **fastscapelib-fortran** (github.com/fastscape-lem/fastscapelib-fortran).
  The reference implementation of Braun & Willett. Its documentation states
  that values of `n` other than unity make the algorithm perform
  Newton-Raphson iterations, and that this slows it substantially — by a
  factor of order `n` when `n > 1`. Worth reading for the iteration count and
  convergence criteria they settled on.
- **Landlab** `landlab/components/stream_power/`. `FastscapeEroder` calls
  `brent_method_erode_fix` from its Cython `cfuncs` for the implicit solve —
  i.e. **Brent**, bracketing and derivative-free, not Newton. Read why.
- **Hergarten (2020b, 2021)** on implicit schemes for the stream-power model,
  which Landlab's `SharedStreamPower` is built on.
- Also check whether `fastscapelib` (the newer C++/Python one) exposes a 1D or
  single-channel path that could simply be depended upon instead.

If one of these is importable and good, depending on it may beat writing a
solver. Report that conclusion before implementing.

## The change, if written here

One function: `incise_n1_step`. The per-node equation for general `n` is

    f(z) = z - z_old - dt*U + dt*K*Q**m*((z - z_r)/dx)**n  =  0

scalar, with an analytic derivative

    f'(z) = 1 + n*dt*K*Q**m*((z - z_r)/dx)**(n-1) / dx

**The root is bracketed for free, and the bracket is exact.** Take

    lo = z_r                     # zero slope: no erosion term
    hi = z_old + dt*U            # no erosion at all

`f(lo) <= 0 <= f(hi)`, and `f` is monotonically increasing in between, so the
root is unique and any bracketing method is guaranteed. I checked this over
20,000 random parameter draws spanning `n` in [0.5, 3], `K` in [1e-7, 1e-3],
`Q` in [1, 1e4], `m` in [0.3, 0.7], `dx` in [1, 1e3], `dt` in [1, 1e6]: zero
failures, and `f` never decreased along the bracket. That makes **Brent on
`[z_r, z_old + dt*U]`** the safe default — it cannot diverge the way a raw
Newton step can when `n < 1` makes `f'` blow up as `z -> z_r`. Seed with the
`n = 1` answer if a Newton-flavoured hybrid is wanted.

## Edge cases that must be handled

- **`z_old + dt*U <= z_r`.** The node is at or below its receiver — a flat, a
  numerical artefact, or a filled depression. There is no erosion to do: set
  `z = z_old + dt*U` and move on. Do **not** evaluate `S**n` on a negative
  slope; for fractional `n` it is not real.
- **`Q == 0`.** No erosion term; same early return.
- **`n == 1`.** Short-circuit to the existing closed form. It is exact and
  needs no iteration, and keeping that path preserves both performance and the
  current numerical results bit-for-bit.
- **Tolerance and iteration cap.** Converge on `z`, scaled by the local relief,
  not on `f`. Cap iterations and fail loudly rather than silently returning a
  half-converged node.

## API

Add `n=1.0` to `StreamPower`, `AdvectionSolver` and the module-level helper.
Keep `incise_n1_step` and `evolve_streampower_n1` working unchanged — other
code and the existing tests call them — and add the general entry points
alongside. Do not change any default.

## Tests to add

1. **`n = 1` is untouched**: the general path at `n = 1` reproduces
   `incise_n1_step` to machine precision.
2. **Steady state at general `n`**: `S = (U / (K Q**m))**(1/n)`, the analogue of
   the existing `test_single_reach_steady_state_slope_discharge`. Run it for
   `n = 2/3`, `1.5` and `2`.
3. **Knickpoint still migrates upstream** at `n != 1` — the companion to
   `test_knickpoint_migrates_upstream`. At `n > 1` celerity rises with slope, so
   a step should steepen rather than merely translate; assert the direction, not
   a precise shape.
4. **Large `dt`**: the companion to `test_implicit_stability_huge_dt`.
   Unconditional stability is no longer free at `n != 1`, so this test is
   establishing what is actually true, not confirming what is assumed.
5. **Degenerate input**: zero slope, zero `Q`, and a receiver above the node all
   return finite elevations and do not raise.

## Out of scope

The sweep order, base-level handling, in-place graph mutation, the diffusion
solver, and the alluvial closures. None of them need to change.

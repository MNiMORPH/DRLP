# DRLP build plan

**Written 16 October 2026.** Governed by `HANDOFF.md` and
`HANDOFF-AMENDMENT-1.md`; where this plan and those disagree, they win.

DRLP is a general research model. The plan below builds the core first and the
interactive demo last, and every phase closes with a named artifact, a number,
and a commit.

---

## Standing constraints

1. **Core dependency: numpy alone.** Verified after Phase 1 by importing `drlp`
   in a clean subprocess and listing what landed in `sys.modules`.
2. **1D only.** No `Network`, no topology, no networkx (Amendment 1).
3. **Nothing leaves this machine.** No GitHub repository, no push, no tag, no
   release, no PyPI. Deploying anything to `GeomorphOnline.github.io` is an
   outward-facing act in a different repository and is gated separately.
4. **A parameter I chose is a proposal.** Thresholds, tolerances, caps and
   defaults not specified in the handoff are raised in one line, not
   implemented.

---

## Phase 0 – Read the field

`HANDOFF.md` §5 gates implementation on this, so it comes first and reports
before any solver code is written.

- **fastscapelib-fortran** – the Newton path for `n != 1`: iteration count,
  convergence criterion, what it does when the step overshoots.
- **Landlab** `components/stream_power/cfuncs` – `brent_method_erode_fix`.
  What bracket it uses and why Brent was chosen over Newton.
- **Hergarten (2020b, 2021)** – implicit stream-power schemes.
- **fastscapelib** (the C++/Python one) – is there an importable single-channel
  path? If it is good and importable, depending on it beats writing a solver.

**Artifact:** `design/01-prior-implementations.md`, ending in a recommendation
to write or to depend, with the reason.

**Checkpoint.** Report before Phase 2.

---

## Phase 1 – Skeleton and packaging

`src/` layout, after `hillcreep`:

    pyproject.toml          numpy only; extras: test, demo, figures
    src/drlp/__init__.py
    src/drlp/_version.py
    tests/
    examples/
    prototypes/
    docs/
    design/
    interactive_demo/
    CHANGELOG.md  CITATION.cff  README.md  CLAUDE.md  FRAME.md

**Artifact:** `pip install -e .` succeeds and `import drlp` pulls numpy and
nothing else, shown by the subprocess check above. Commit.

---

## Phase 2 – The stepper, `n = 1`

Prototype in `prototypes/` before writing the module: a ten-line script that
runs one step on five nodes and prints the result. The handoff's numerics are
specified well enough to implement, but not well enough to skip running.

`src/drlp/incision.py`:

    step(x, z, Q, K, m, n, U, dt, x_downstream, z_downstream) -> z_new

Pure: it returns a new array rather than mutating, which is what lets the time
term read `z` while receivers read `z_new`. `K` and `U` accept scalars or
arrays. Index 0 is the upstream end, `x` increases downstream, and the receiver
of the last node is the downstream boundary – the same convention as FluvTree's
sweep, so an adapter can drive this per reach without a translation layer.

**Tests**, in `tests/test_incision_n1.py`:

1. Steady state: `S = (U / (K Q**m))**(1/n)` to machine precision.
2. Unconditional stability at `dt = 1e12` s.
3. A knickpoint migrates upstream and stays sharp.
4. **Free prediction:** on a single reach, this reproduces FluvTree's
   independently written and separately tested `incise_n1_step` to machine
   precision. Nothing in DRLP is fitted to it.

Commit each of the four as its own logical change.

---

## Phase 3 – General `n`

Brent on the exact bracket `[z_r, z_old + dt*U]`, with the handoff's edge cases:
`z_old + dt*U <= z_r` returns `z_old + dt*U` and never evaluates `S**n` on a
negative slope; `Q == 0` likewise; `n == 1` short-circuits to the closed form so
that path stays bit-for-bit what Phase 2 produced.

**Brent has to be written here, not imported.** `scipy.optimize.brentq` is
scipy, and the core is numpy-only, so the root-finder is about forty lines of
DRLP. See the open questions – I would like scipy as a *test-only* dependency
so our Brent can be checked against `brentq`, which never ships to the browser.

**Tests**, in `tests/test_incision_general_n.py`:

1. The general path at `n = 1` reproduces the closed form to machine precision.
2. Steady state at `n = 2/3`, `1.5`, `2`.
3. A knickpoint still migrates upstream; at `n > 1` it steepens. Assert the
   direction, not the shape.
4. Large `dt`. Stability is no longer free at `n != 1`, so this test establishes
   what is true rather than confirming what is assumed.
5. Degenerate input: zero slope, zero `Q`, receiver above the node.
6. **Property test:** the handoff's 20,000-draw bracket claim, re-run inside
   DRLP's own suite – `f(lo) <= 0 <= f(hi)` and `f` non-decreasing across the
   bracket, over the stated parameter ranges.
7. **Free prediction:** our Brent agrees with `scipy.optimize.brentq` on the
   same bracket, to tolerance, across the same draws. Skipped if scipy is absent.

Commit per test group.

---

## Phase 4 – The model object

`src/drlp/longprofile.py`, a `LongProfile` wrapping the stepper for the
standalone single-channel case, where `z_downstream` is base level:

- Grid and spatially variable `K`, `U`, `Q`.
- Discharge two ways: `Q` given directly, or `Q = R A` from drainage area.
  `K_Q Q**m = (K_Q R**m) A**m`, so an area-based `K` is a rescaling – both are
  exposed and the docstring says so.
- Time-varying base level `z_bl(t)`, callable or tabulated.
- `set_steady_state()` – the closed form `S = (U/(K Q**m))**(1/n)` integrated
  upstream from base level, usable as an initial condition.
- `evolve(nt, dt)`.

**Tests:**

1. **Free prediction:** the analytic steady-state profile is a fixed point of
   the transient solver. The integrator and the stepper are separate pieces of
   code; if the profile moves, one of them is wrong.
2. A base-level drop produces a knickpoint whose celerity matches the analytic
   `C = K Q**m n S**(n-1)` at `n = 1`.
3. Spatially variable `K` reproduces the two-faults steady profile: a step in
   `U` gives a step in `S`, in the ratio the closed form predicts.

Commit.

---

## Phase 5 – Analysis tools

`src/drlp/analysis.py`: chi coordinate, slope–area, and `k_sn` at a reference
concavity. These are how the model is used, not an afterthought (§6).

**Test, and it is the strong one:** build a steady-state profile from known `U`
and `K`, then recover them. Chi-plot slope must return `k_sn`, and inverting
`k_sn` must return the `U` that was put in. The analysis tools are fitted to
nothing; they either invert the forward model or they do not.

Commit.

---

## Phase 6 – Docs, examples, design notes

- `docs/theory.md` – the equation, and the advective-versus-diffusive contrast
  with GRLP and SRLP that is the reason this model exists.
- `docs/numerics.md` – the bracket and why it is exact, why Brent over Newton,
  what `n = 1` short-circuits.
- `docs/api.md`, `README.md`, `CHANGELOG.md`, `CITATION.cff`.
- `examples/` – a base-level-drop transient, the two-faults steady profile, and
  a chi and slope–area analysis of a synthetic profile.
- `design/02-numerics.md`, `design/03-no-network.md` (Amendment 1's decision,
  recorded where a reader will find it), `design/04-panel-scope.md`.

Commit.

---

## Phase 7 – The interactive demo

Last, and it is thin by construction – a view over Phases 2 to 5, with no
physics of its own.

`interactive_demo/drlp_panel.py`, following `GRLP/interactive_demo/grlp_panel.py`
and `hillcreep/interactive_demo/hillcreep_panel.py`. `panel` and `bokeh` sit in
the `demo` extra, so the core stays numpy-only.

    artesian build interactive_demo/drlp_panel.py -o _artesian_build \
        -p . -p ../artesian -r numpy --serve

`artesian build` imports the package in the building environment, which is why
the core has to be installed and correct before this phase can run at all.

Deployment to `GeomorphOnline.github.io` is a separate repository and an
outward-facing act. I will not touch it without being asked.

---

## Phase 8 – The FluvTree note

`docs/handoff-fluvtree-adapter.md`, written here and handed over: what
`processes/stream_power.py` becomes, the `step` signature it drives, and the
dependency direction. **FluvTree is not edited from this repository.**

---

## Open questions

Each of these is a parameter or a convention I would otherwise be choosing.

1. **Convergence criterion and iteration cap** for the general-`n` root-find.
   The handoff says to converge on `z` scaled by local relief and to fail loudly
   rather than return a half-converged node, which makes the cap a real choice.
2. **`x_downstream` is missing from Amendment 1's signature.** The slope at the
   downstream-most node needs the distance to the boundary, not only its
   elevation – FluvTree's sweep carries both (`z_end, x_end = z_bl, x_bl`).
   Add the argument, or fix a ghost-node convention instead?
3. **scipy as a test-only dependency**, to check our hand-rolled Brent against
   `scipy.optimize.brentq`. The core stays numpy-only either way.
4. **Names.** Package `drlp`, distribution `DRLP`, after GRLP's `grlp` / `GRLP`.
5. **Hack's law** for `A(x)` in the demo: core feature, or demo-only helper?

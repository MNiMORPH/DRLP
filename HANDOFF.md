# DRLP — build handoff

**Written 16 October 2026 for a later session.** Nothing has been built. This
folder holds a licence and two documents; everything else is yours to create.

---

## 1. What DRLP is

**Detachment-limited River Long Profile.** A MNiMORPH long-profile model,
sibling to [GRLP](https://github.com/MNiMORPH/GRLP) (gravel-bed) and
[SRLP](https://github.com/MNiMORPH/SRLP) (sand-bed).

    dz/dt = U - K Q**m S**n

GRLP and SRLP are transport-limited and **diffusive**. DRLP is
detachment-limited and **advective**: erosion is set locally by the flow, not by
a divergence of sediment flux, so a perturbation travels as a step instead of
spreading. That contrast is the reason the model is worth having beside them.

**This is a general research model, not a teaching toy.** A classroom exercise
is one consumer and must not drive the design. Build it the way GRLP is built:
general boundary conditions, spatially and temporally variable parameters,
analysis tools, tests against analytic solutions. If a teaching-facing
simplification and a general capability conflict, the general capability wins
and the simplification goes in a demo script.

## 2. Why this name, and what is reserved

`BRLP` is **deliberately not used**. The stream-power law is a parameterisation
of detachment-limited incision; it does not know whether the bed is granite or
shale. A mechanistic bedrock model — abrasion, plucking, tools and cover, block
size — is planned, and **that** model should take the "bedrock" name. DRLP names
the assumption it actually makes.

## 3. Architecture, already decided

**DRLP is standalone; FluvTree adapts it, never the reverse.** This is not a new
convention — it is the one FluvTree already states in
`src/fluvtree/processes/grlp.py`:

> A thin adapter that reuses GRLP's solver numerics unchanged. […] Dependency
> direction (see the design note): FluvTree depends on published GRLP, never the
> reverse.

`processes/stream_power.py` is currently the one process in that family with no
package behind it: it carries its own numerics instead of adapting a published
model. **Once DRLP exists, that file should become a thin adapter to it**, the
way `processes/grlp.py` adapts GRLP. Do not take that step as part of this
build; it belongs in FluvTree, after DRLP is published, and FluvTree should be
handed a separate note rather than edited from here.

**Core dependency: numpy alone.** `hillcreep` declares numpy and nothing else,
on purpose, because its interactive demo compiles to WebAssembly through
`artesian` and runs in a browser. For comparison, `import fluvtree` eagerly
pulls **numpy, scipy and networkx** (scipy via `solvers/diffusion.py`, networkx
via `network.py`) — verified by import. Keep DRLP's core numpy-only and put
anything heavier behind an optional extra.

## 4. The physics and the numerics

**Governing equation**, SI units throughout, time in seconds:

    dz/dt = U - K Q**m S**n,      S = -dz/dx  (positive downstream)

**Implicit solve.** Use the Braun & Willett (2013) outlet-to-upstream sweep: a
node's receiver is solved before the node that needs it, so one pass is enough.
For a single channel this degenerates to a reverse loop over one array — no
graph, no sweep order, no networkx.

**`n = 1` is closed-form and exact**, and should be kept as its own fast path:

    c = dt * K * Q[i]**m / dx
    z[i] = (z_old[i] + dt*U + c*z_r) / (1 + c)

**General `n` is a scalar root-find per node**, on

    f(z) = z - z_old - dt*U + dt*K*Q**m*((z - z_r)/dx)**n

**The root is bracketed for free, and the bracket is exact.** Take
`lo = z_r` and `hi = z_old + dt*U`: `f(lo) <= 0 <= f(hi)`, and `f` increases
monotonically between them, so the root is unique and any bracketing method is
guaranteed. **Verified over 20,000 random parameter draws** — `n` in [0.5, 3],
`K` in [1e-7, 1e-3], `Q` in [1, 1e4], `m` in [0.3, 0.7], `dx` in [1, 1e3],
`dt` in [1, 1e6]: zero bracket failures, and `f` never decreased anywhere on the
bracket. **Prefer Brent on that bracket to a bare Newton step**, which can
diverge when `n < 1` sends `f'` to infinity as `z -> z_r`.

See `docs/handoff-fluvtree-general-n.md` for the same numerics written up for
FluvTree, including the derivative, the edge cases, and what the two mature
implementations do.

## 5. Step 0 — read the field before writing a solver

- **fastscapelib-fortran** (`github.com/fastscape-lem/fastscapelib-fortran`),
  the reference Braun & Willett implementation. Its docs note that `n != 1`
  triggers Newton–Raphson and slows it by a factor of order `n` when `n > 1`.
- **Landlab** `components/stream_power/`: `FastscapeEroder` calls
  `brent_method_erode_fix` from its Cython `cfuncs` — Brent, not Newton.
- **Hergarten (2020b, 2021)** on implicit stream-power schemes, behind
  Landlab's `SharedStreamPower`.
- The newer C++/Python **fastscapelib** — check whether it exposes anything
  worth depending on rather than writing.

Report what you find before implementing. If something is importable and good,
depending on it may beat writing a solver.

## 6. Scope for a general model

Minimum for a first release, beyond the solver:

- **Discharge**: accept `Q` directly, *or* derive it from drainage area with a
  runoff rule `Q = R A`. Note that `K_Q Q**m = (K_Q R**m) A**m`, so an
  area-based `K` is just a rescaling — expose both and say so.
- **Spatially variable `K`, `U`, `Q`**, and time-varying base level `z_bl(t)`.
  Transient response to a base-level drop is the model's main use.
- **Steady state in closed form**: `S = (U / (K Q**m))**(1/n)`, integrated to a
  profile — both as an initial condition and as a check on the transient solver.
- **Analysis tools**, which are part of how this model is used and not an
  afterthought: chi coordinate, slope–area, and the normalised steepness index
  `k_sn` with reference concavity.
- **Networks**: GRLP has its own `Network`. Decide whether DRLP needs one or
  whether the network case is served by the FluvTree adapter. Ask before
  building a second graph layer.

## 7. Precedents to follow

- `~/models/hillcreep` — closest template for layout and packaging: `src/`
  layout, `tests/`, `examples/`, `prototypes/`, `docs/`, `design/`,
  `interactive_demo/`, `CHANGELOG.md`, `CITATION.cff`, `CLAUDE.md`, `FRAME.md`.
  Read its `CLAUDE.md` for the `artesian build` invocation.
- `~/models/GRLP` — the general-model bar: boundary conditions, network,
  benchmarks, docs.
- `~/models/FluvTree/src/fluvtree/solvers/advection.py` — a correct, tested
  implementation of the `n = 1` sweep on a network. Read it. **Do not vendor
  it**; DRLP's single-channel path is simpler, and the duplication to avoid is
  of *physics*, not of a loop.

## 8. Verified state, 16 October 2026

- FluvTree at `c266a7f` on `main`, 41 commits ahead of `origin/main`.
  `pytest tests/test_stream_power.py tests/test_bedrock.py -q` → **7 passed in
  0.45 s**, including `test_knickpoint_migrates_upstream` and
  `test_implicit_stability_huge_dt`.
- FluvTree's solver is `n = 1` only; its own docstring calls `n != 1` "a later
  addition".
- Nothing in `~/models` imports `fluvtree`; the references in `alluvial-core`,
  `LandGraph` and `GRLP` are documentation only.
- `~/models/Bedrock-river-erosion` has an unmerged branch
  `origin/Stream_power_model_simple` with a 217-line Gooseberry River research
  script by J. Marggraf. **Do not build on it**: it never divides by `dx`, so
  its "slope" is a drop in metres; `m_ini` is computed from the wrong `theta`;
  and it has no base-level-drop mechanism. Useful only as evidence that the
  physics has been wanted here before.

## 9. Before you start

This folder is a local `git init` with no remote. **Do not create a GitHub
repository, push, tag, or release** without Andy's explicit say-so in the
message where he asks for it.

# Changelog

All notable changes to this project are documented here, following
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Nothing released yet. The model runs, 93 tests pass, and the browser demo
compiles and runs headlessly through every slider corner. No browser has yet
rendered the compiled page, and the analysis tools that make a long-profile
model usable on real data - chi, slope-area, and the normalised steepness index
`k_sn` - are not written.

### Added

- `drlp.incision.step`: one implicit stream-power step on a single channel, the
  Braun and Willett (2013) outlet-to-upstream sweep. Pure, returning a new
  array, so the time term reads the old elevations while receivers read the new
  ones and a single pass is exact. `n = 1` takes the closed form; general `n`
  solves Landlab's dimensionless relief fraction on the guaranteed bracket
  `[0, 1]` with `scipy.optimize.brentq`, raising `ConvergenceError` rather than
  returning a half-converged node.
- The downstream boundary is a ghost node at `(x_bl, z_bl)`, after GRLP. It is
  the receiver of the last real node, which is the whole boundary condition,
  and it is passed as an argument rather than owned - so FluvTree's network
  sweep can drive the stepper reach by reach without a translation layer.
- `drlp.LongProfile`: the standalone single-channel model. Geometry, spatially
  variable `K`, `U` and `Q`, `set_bl`/`set_x_bl`/`set_z_bl` after GRLP, a
  callable `z_bl(t)` for a base-level history, `set_steady_state` integrating
  the closed-form slope upstream from the ghost node, and `evolve`.
- Three transfer functions, kept separate: `x -> A` by Hack's law
  (`A = k_xA x^P_xA`), `A -> Q` by a power-law hydrology
  (`Q = k_AQ A^P_AQ`, with `set_Q(R=...)` as the linear special case), and an
  optional `Q -> B` by Leopold (`B = k_QB Q^P_QB`). `P_xQ` exposes GRLP's
  composition identity `P_xA · P_AQ`.
- `set_K(K, basis="Q"|"A"|"Q/B")`: discharge, drainage area alone, or unit
  stream power with explicit width. The basis is stated rather than inferred,
  because it fixes what `K`'s number means and a forgotten rescaling gives a
  plausible wrong profile rather than an error. `K_A_from_K_Q`,
  `implicit_m_from_unit` and `implicit_K_from_unit` expose the conversions.
- `interactive_demo/drlp_panel.py`: the `artesian` browser demo, laid out after
  GRLP's panel so the diffusive and advective cases read side by side. Sliders
  for `log10 K`, `m`, `n`, `U` and base level; a long profile with base level
  dashed; the elevation axis refitted on reset, because steady-state relief
  spans 41 m to 40 km across the slider corners. It prescribes no exercise.
- `design/01-prior-implementations.md`: what fastscapelib-fortran, fastscapelib
  and Landlab each do for `n != 1`, read from source rather than from summary,
  with the tolerances and iteration caps quoted and the write-or-depend
  decision argued.
- `prototypes/`: the throwaway scripts that settled two design questions - that
  the dimensionless and dimensional node equations are the same equation, and
  what an erosion threshold would and would not change.

### Known limits

- No networks. One long profile, deliberately; topology is FluvTree's.
- No deposition, so base-level **fall** only. `E >= 0` gives `dz/dt <= U`, so
  under a rise the bed comes up at the rock uplift rate and not at all when
  `U = 0`. Output under a rise is not a prediction.
- No erosion threshold, by choice: that is an excess-stress model, not the
  stream-power law.
- No dynamic width in either mode.
- No chi, slope-area or `k_sn` analysis yet.

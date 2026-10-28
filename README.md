# DRLP

**Detachment-limited river long profile: a perturbation that travels instead of spreading.**

```
dz/dt = U - K Q^m S^n          S = -dz/dx, positive downstream
```

Erosion here is set locally by the flow, not by a divergence of sediment flux.
That one difference changes the character of the equation and everything that
follows from it: a bedrock river answers a base-level fall **advectively**, by
sending a step upstream that keeps its edge, where a gravel river answers it
**diffusively**, by spreading the disturbance out and flattening it.

DRLP is the detachment-limited member of the MNiMORPH long-profile family,
beside [GRLP](https://github.com/MNiMORPH/GRLP) (gravel-bed) and
[SRLP](https://github.com/MNiMORPH/SRLP) (sand-bed). Those two are
transport-limited and diffusive. Run DRLP next to GRLP on the same base-level
drop and the contrast is the lesson.

The name states the assumption the model makes. The stream-power law is a
parameterization of detachment-limited incision, and it does not know whether
the bed is granite or shale, so `BRLP` is deliberately unused: the "bedrock"
name is reserved for a mechanistic model of abrasion, plucking, tools and
cover, and block size.

## Install and run

```sh
pip install -e ".[test]"     # from a clone
pytest                       # 93 tests
```

```python
from drlp import LongProfile

lp = (LongProfile()
      .set_x(dx=500., nx=60, x0=500.)   # 0.5 to 30 km, index 0 upstream
      .set_A(k_xA=1., P_xA=7/4.)        # x -> A, Hack's law
      .set_Q(R=1e-8)                    # A -> Q, runoff
      .set_K(10**-8.5, basis="Q")
      .set_mn(m=0.5, n=1.)
      .set_uplift_rate(0.3e-3/31556926.)
      .set_bl(z=0.))

lp.set_steady_state()        # the analytic profile, and an exact fixed point
lp.set_bl(z=-200.)           # drop base level
lp.evolve(nt=100, dt=1e11)   # watch the knickpoint go
```

SI units throughout, and time in seconds. Every setter returns the model, so
they chain.

## Three transfer functions

The chain from position to the quantity that erodes is three separate
transfers, and they are different kinds of statement. Keeping them apart
matters, because each one has its own exponent and they do not compose into a
single fitted number without losing the ability to ask about any of them.

```
x -> A    Hack's law, geometric     A = k_xA x^P_xA     P_xA = 7/4
A -> Q    hydrology                 Q = k_AQ A^P_AQ     P_AQ = 0.7
Q -> B    Leopold, optional         B = k_QB Q^P_QB     P_QB = 1/2
```

`P_xA` is the *inverse* Hack exponent (Hack's law is `L ~ A^0.6`, so
`A ~ L^1.67`). `P_AQ` is sub-linear because discharge grows more slowly than
area: channel losses, and storms that do not cover a large basin. Composing the
first two gives `P_xQ = P_xA · P_AQ`, which the model exposes.

Hydrology has a shortcut for the linear case: `set_Q(R=...)` is `Q = R A`, which
is `P_AQ = 1`, and it sets the exponent rather than leaving a stale one in place.

The consequence worth knowing: at steady state the slope-area concavity is

```
theta = -d(ln S)/d(ln A) = m P_AQ / n
```

so `theta` is **not** `m/n` unless the hydrology happens to be linear.

## Drainage area or discharge, your choice

`set_K(K, basis=...)` names the quantity the stream-power term acts on, and so
what `K`'s number means:

- `basis="Q"` (default) erodes with `K Q^m S^n`, GRLP's convention and the form
  above.
- `basis="A"` erodes with `K A^m S^n`, driven by drainage area alone, which is
  the convention of the stream-power literature and of fastscapelib, Landlab
  and fastscapelib-fortran.

The two are a rescaling, `K_A = K_Q R^m` (`LongProfile.K_A_from_K_Q`), so
neither is privileged. The basis is stated rather than inferred from whichever
field you happened to fill, because forgetting the rescaling gives a plausible
wrong profile rather than an error.

## Channel width, both ways

**Implicit, the standard way and the default.** Width folds into `m` through
`B ~ Q^P_QB`, with the width coefficient absorbed into `K`. Leopold's
`P_QB = 1/2` gives `m/n = 1/2`. Nothing is lost when width really is a power law
of discharge, and the coefficient is unidentifiable from a profile anyway.

**Explicit, semi-independent.** `set_B(k_QB=...)` carries width as a field and
`set_K(..., basis="Q/B")` erodes with unit stream power `(Q/B)^m S^n`. With a
power-law `B` this reduces *exactly* to the implicit form under

```
m -> m (1 - P_QB)        implicit_m_from_unit
K -> K k_QB^(-m)         implicit_K_from_unit
```

which the test suite checks to 1e-12 through a transient rather than asserting.
It becomes a genuinely different model only when `B` is not a power law of `Q`:
measured widths, lithologic control, a gorge. What it adds in every case is that
width becomes something you can look at rather than an assumption inside an
exponent.

## The browser demo

`interactive_demo/drlp_panel.py` runs the model in the browser, with no server
and no install, compiled to WebAssembly through
[artesian](https://github.com/MNiMORPH/artesian):

```sh
pip install -e .
artesian build interactive_demo/drlp_panel.py -o _artesian_build \
    -p . -p ../artesian -r numpy -r scipy --strip-wheels --strip-vendored --serve
```

`artesian build` imports the app to discover what it serves, so install the
package first or the build fails on a bare `ModuleNotFoundError`. Pyodide
bundles both numpy and scipy, so the general-`n` root-find works in the browser.

## What this model does not do

- **No networks.** DRLP is one long profile, and the single-channel sweep is a
  reverse loop over one array. Confluences, topology and discharge accumulation
  belong to [FluvTree](https://github.com/MNiMORPH/FluvTree), which adapts DRLP
  rather than the reverse. The stepper takes the downstream boundary elevation
  as an argument for exactly this reason, so an adapter can drive it reach by
  reach.
- **No deposition, so base-level *fall* only.** The law is purely erosional:
  `E >= 0`, hence `dz/dt <= U` everywhere and always. Under a base-level *rise*
  the bed can come up only at the rock uplift rate, and not at all when `U = 0`,
  so the lower reach stays drowned while the reach above it keeps incising and
  the sediment that incision produces goes nowhere. A real river aggrades. Use
  GRLP or SRLP for that.
- **No erosion threshold.** Stream power is `rho g Q S`, and the stream-power
  law is `K A^m S^n`. Adding a threshold makes it an excess-stress model, which
  is a different parameterization; a critical shear stress belongs in the
  mechanistic model, where it follows from grain and block size instead of being
  fitted.
- **No dynamic width.** Width never responds to incision in either mode. GRLP
  carries that (`set_B`, `narrow_by_incision`); DRLP does not.

## Defaults, and where they come from

| | value | source |
|---|---|---|
| `P_xA` | 7/4 | GRLP's inverse Hack exponent |
| `P_AQ` | 0.7 | GRLP's area-discharge exponent |
| `P_QB` | 1/2 | Leopold and Maddock's downstream hydraulic geometry |
| `basis` | `"Q"` | the form the governing equation is written in |
| general-`n` solver | Brent on `[0, 1]` | see below |

The `n = 1` node update is closed-form and exact. For general `n` the problem is
non-dimensionalised first, following Landlab: solving for the fraction of relief
retained, `x = (z - z_r)/(z_old + dt U - z_r)`, turns the per-node equation into

```
f(x) = x - 1 + alpha x^n,     alpha = dt K Q^m D0^(n-1) / dx^n
```

on the bracket `[0, 1]`, where `f(0) = -1 < 0` and `f(1) = alpha > 0` with `f`
monotonic between them. The bracket is therefore guaranteed in one line rather
than sampled, and it is scaled by the local relief by construction. Solving in
elevation instead is ill-conditioned: `z - z_old - dt U` differences two large
numbers, and once the erosion term falls below their rounding error the sign of
`f` at the upper bracket is noise. Over 20,000 random draws that include an
absolute elevation, the dimensional bracket fails 23 times and the dimensionless
one never (`prototypes/00_dimensionless_equivalence.py`).

Brent, not Newton: a bare Newton step can diverge when `n < 1` sends `f'` to
infinity. Tolerances are `scipy.optimize.brentq`'s public defaults, through the
public interface. `design/01-prior-implementations.md` records what
fastscapelib-fortran, fastscapelib and Landlab each do, read from source.

## Examples

```sh
python examples/base_level_drop.py      # the transient, and the traveling front
python examples/two_faults.py           # stepped uplift, and why S is not U
python examples/width_both_ways.py      # the two width treatments, and where they part
```

`base_level_drop.py` takes `--plot` for a figure (needs
`pip install -e ".[figures]"`).

## A note on the handoff

Several files here cite `HANDOFF.md`, the build brief this model was written
from. It is deliberately not published: it names a third party and critiques
their unpublished code, which reads very differently in a public repository
than in a private folder. The physics and numerics it specified are all
restated in this README and in `design/01-prior-implementations.md`, so
nothing load-bearing is missing. `HANDOFF-AMENDMENT-1.md` is published,
because it carries only the decision that DRLP is one-dimensional.

## Layout

```
src/drlp/incision.py       the implicit sweep: one step, n = 1 exact, general n by Brent
src/drlp/longprofile.py    the model: geometry, transfers, boundary conditions, evolution
tests/                     93 tests, each against a closed form or an identity
examples/                  runnable scripts
interactive_demo/          the artesian browser demo
design/                    decisions, with the reading they rest on
prototypes/                the throwaway scripts that settled design questions
```

## References

- Braun, J., and Willett, S. (2013). A very efficient O(n), implicit and
  parallel method to solve the stream power equation governing fluvial incision
  and landscape evolution. *Geomorphology* 180-181, 170-179.
  [doi:10.1016/j.geomorph.2012.10.008](https://doi.org/10.1016/j.geomorph.2012.10.008)
- Hack, J. T. (1957). Studies of longitudinal stream-profiles in Virginia and
  Maryland. *U.S. Geological Survey Professional Paper* 294-B, 45-97.
- Leopold, L. B., and Maddock, T. (1953). The hydraulic geometry of stream
  channels and some physiographic implications. *U.S. Geological Survey
  Professional Paper* 252, 1-57.
- Whipple, K. X., and Tucker, G. E. (1999). Dynamics of the stream-power river
  incision model: implications for height limits of mountain ranges, landscape
  response timescales, and research needs. *Journal of Geophysical Research:
  Solid Earth* 104(B8), 17661-17674.
  [doi:10.1029/1999JB900120](https://doi.org/10.1029/1999JB900120)

## License

GPL-3.0-or-later. See `LICENSE`.

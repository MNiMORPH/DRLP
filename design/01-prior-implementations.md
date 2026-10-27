# 01 - Prior implementations: what they do, and write-or-depend

**Phase 0 of `BUILD-PLAN.md`, run 16 October 2026.** `HANDOFF.md` §5 gates
implementation on this. Everything below was read from source in this session;
nothing is carried over from the handoff's summary of these projects.

**Recommendation: write the solver. Do not depend on fastscapelib.** Adopt
Landlab's dimensionless formulation, with the iteration cap the Fastscape
implementations do not have. Reasons below.

---

## 1. fastscapelib-fortran - the reference Braun & Willett

`src/StreamPowerLaw.f90`. Verbatim, the `n != 1` branch:

    tolp=1.d-3
    omega=0.875d0/n
    ...
    do while (errp.gt.tolp)
      f=h(ijk)-h0
      df=1.d0
      do k=1,mnrec(ijk)
        if (ht(ijk).gt.ht(mrec(k,ijk))) then
          fact = kfint(ijk)*dt*a(ijk)**m/length(ijk)**n
          f=f+fact*max(0.d0,h(ijk)-h(ijr))**n
          df=df+fact*n*max(0.d0,h(ijk)-h(ijr))**(n-1.d0)
        endif
      enddo
      hn=h(ijk)-f/df
      errp=abs(hn-h(ijk))
      h(ijk)=h(ijk)*(1.d0-omega)+hn*omega
    enddo

Four things worth having:

- **Tolerance `1.d-3`**, absolute, in meters, compared against
  `errp = abs(hn - h(ijk))` - the Newton **step size** in elevation.
- **No iteration cap.** The `do while` has no counter. The surrounding
  Gauss-Seidel loop is capped (`nGSStreamPowerLaw .lt. nGSStreamPowerLawMax-1`),
  the inner Newton is not.
- **It is not plain Newton - it is under-relaxed**, by `omega = 0.875/n`. Since
  the relaxation goes as `1/n`, large `n` takes proportionally smaller steps.
  *(My inference, not a documented claim: that is plausibly the mechanism behind
  the "slows by a factor of order n" note the handoff cites.)*
- **Negative slope is clamped, not special-cased**: `max(0.d0, h-h_r)**n`.

The `n = 1` branch is a one-shot closed form, `h(ijk) = f/df` with
`f = elev + sum(fact*h_r)` and `df = 1 + sum(fact)`. For a single receiver that
is exactly `z = (z_old + c*z_r)/(1 + c)`, confirming `HANDOFF.md` §4.

## 2. fastscapelib - the newer C++/Python one

`include/fastscapelib/eroders/spl.hpp`. Constructor:

    spl_eroder(FG& flow_graph, K&& k_coef, double area_exp,
               double slope_exp, double tolerance = 1e-3)

`k_coef` is indexed per node (`m_k_coef(inode)`), so spatially variable `K` is
supported. The non-linear branch:

    double delta_0 = inode_elevation - irec_elevation_next;
    double delta_k = delta_0;
    while (true)
    {
        auto factor_delta_exp = factor * std::pow(delta_k, m_slope_exp);
        auto func = delta_k + factor_delta_exp - delta_0;
        if (func <= m_tolerance) { break; }
        auto func_deriv = 1 + m_slope_exp * factor_delta_exp / delta_k;
        delta_k -= func / func_deriv;
        if (delta_k <= 0) { break; }
    }

- Newton on the elevation **difference** `delta = z - z_r`, not on `z`.
- **Same tolerance, `1e-3`** - but here compared against `func`, the
  **residual**, one-sided (`func <= tol`, not `|func| <= tol`).
- **No iteration cap.** `while (true)`, exiting on the residual test or on
  `delta_k <= 0`.

**So the two Fastscapes agree on the number and disagree on what it measures:**
1e-3 is a step size in the Fortran and a residual in the C++. Both are
dimensional - meters - and neither caps its iterations.

## 3. Landlab - Brent, and a better-posed problem

`src/landlab/components/stream_power/cfuncs.pyx`,
`brent_method_erode_fixed_threshold`. Landlab **non-dimensionalises** before
solving. From `erode_fn`'s docstring, verbatim:

    x = (z_node(t + delta_t) - z_downstream(t + delta_t))
        / (z_node(t) - z_downstream(t + delta_t))

    f = x - 1 + alpha*(x**n) - beta

    alpha = delta_t * K * A**m / (delta_x**n)
            * (z_node(t) - z_downstream(t + delta_t))**(n - 1)

and the call:

    x = brentq(erode_fn, 0.0, 1.0,
               1e-12, 4.4408920985006262e-16, 100,
               (alpha_param, beta_param, n), False, True)

- `x` is **the fraction of the original relief retained**, so the bracket is
  `[0, 1]` for every node, at every scale.
- `xtol = 1e-12`, `rtol = 4.4408920985006262e-16` (2x machine epsilon),
  **`maxiter = 100`**. Landlab's comment says these "are the defaults if one
  were to import brentq from scipy.optimize". **They are not**, as of scipy
  1.11.4 on this machine: the public defaults are `xtol = 2e-12` and
  `rtol = 4*eps = 8.881784197001252e-16`, each a factor of two looser. More
  than a documentation slip - the public wrapper *enforces* `rtol >= 4*eps`
  and raises `ValueError: rtol too small (4.44089e-16 < 8.88178e-16)` on
  Landlab's value. The private `_brentq` does not enforce it, so that private
  import is load-bearing for the tolerance, not merely a speed choice.
- `n = 1` has an analytic solution, `x = (1 + beta)/(1 + alpha)`.
- Nodes at or below their receiver are skipped by the guard
  `if src_id != dst_id and z[src_id] > z[dst_id]`.
- `x <= 0` degenerates to `z = z_downstream + 1.0e-15`.
- It imports `from scipy.optimize._zeros import _brentq as brentq` - the C
  implementation, reached directly for speed. Landlab does **not** vendor Brent.

**The bracket `[0, 1]` is provable in one line.** With no threshold
(`beta = 0`), `f(0) = -1 < 0` and `f(1) = alpha > 0`, and `f` increases
monotonically in `x` for `alpha > 0`, `n > 0`. This supersedes the handoff's
20,000-draw verification of `[z_r, z_old + dt*U]`: same guarantee, obtained
analytically, in a variable that is dimensionless by construction.

**And the dimensional bracket is not merely untidy, it fails.**
`prototypes/00_dimensionless_equivalence.py` shows `f_dim = D0 * f_x` exactly,
so the two are one equation - then measures both over 20,000 draws that add an
absolute elevation `z_r`, an axis the handoff's sweep did not include:

    dimensional   bracket failures : 23
    dimensionless bracket failures : 0
    worst root disagreement        : 5.0e-11 of local relief

`f_dim` evaluates `z - z_old - dt*U`, a difference of absolute elevations, so
once the erosion term drops below the rounding error of that difference the
sign of `f_dim` at the upper bracket is noise. In the first failure the
rounding is 55x the erosion term, at `|z_r|/D0 = 3511`. `f_x(1) = alpha > 0`
exactly and cannot do this. `|z| >> D0` is the ordinary case for a real
profile.

Folding uplift in costs nothing: use `z_old + dt*U` wherever Landlab writes
`z_node(t)`, and `f` is unchanged.

## 4. Write or depend?

fastscapelib **does** expose a 1D grid - `python/src/grid.cpp` line 331,
`py::class_<fs::py_profile_grid> pgrid(m, "ProfileGrid", "A 1-dimensional
profile grid.")` - so the single-channel path §5 asked about is real. Depending
on it is still wrong, for four reasons, the first decisive:

1. **It cannot reach the browser.** fastscapelib is a compiled C++ extension and
   is **absent from Pyodide's package list** (checked against
   `pyodide.org/en/stable/usage/packages-in-pyodide.html`). Phase 7 compiles the
   demo to WebAssembly through `artesian`; a C++ extension with no emscripten
   build cannot be imported there. This is precisely the constraint `HANDOFF.md`
   §3 was protecting.
2. **Its boundary model is the wrong one for us.** Base level is node status on
   a flow graph, not the `(x_bl, z_bl)` ghost node of GRLP that DRLP has
   adopted.
3. **The eroder is the small part.** It supplies no `U`, no `z_bl(t)`, no
   `Q = R A` or Hack's law, and no chi / slope-area / `k_sn`. Phases 4 and 5
   would be written regardless.
4. **No `n = 1` exact path to preserve.** DRLP keeps that closed form
   bit-for-bit; routing through a generic eroder gives it up.

**Depend on it for nothing; cite its numerics for everything.**

## 5. What this means for Phase 3

Adopt **Landlab's formulation** - solve for `x` on `[0, 1]`, Brent, uplift
folded into `z_old`. Two notes on the parameters, which are Andy's to set:

- **"Follow Fastscape" transfers only partly, and I would not take its
  tolerance.** 1e-3 in meters is dimensional and loose, it means different
  quantities in the two Fastscape codes, and the handoff asked for a tolerance
  *scaled by local relief* - which is what `x` already is. Proposal: scipy's
  **public** `brentq` defaults, `xtol = 2e-12` and `rtol = 4*eps`, reached
  through `scipy.optimize.brentq` rather than Landlab's private `_brentq`.
  Brent is superlinear on a guaranteed bracket, so the tight tolerance is
  nearly free, and we do not inherit a tolerance the public API refuses.
- **Keep a cap; neither Fastscape has one.** `maxiter = 100`, Landlab's and
  scipy's default, and **raise** rather than return a half-converged node, per
  `HANDOFF.md` §4. On a guaranteed bracket it should never fire.

**And scipy in the core is safe for the browser after all: scipy *is* in
Pyodide** (same check as above; numpy and bokeh too, `panel` arrives as a
pure-Python wheel through `artesian`). So using `scipy.optimize.brentq`
directly, rather than hand-rolling forty lines of Brent, does not cost Phase 7
anything but download weight. That turns the open dependency question from a
correctness question into a size preference.

"""
LongProfile: the standalone single-channel DRLP model.

Wraps :func:`drlp.incision.step` for the case where the downstream ghost node
is base level.  SI units throughout; time in seconds.

The ``set_*`` API and the parameter names follow GRLP so the two models read
the same way side by side: ``P_xA`` is the inverse Hack exponent, ``P_AQ`` the
area-discharge exponent, ``k_xA`` the Hack coefficient, and
``set_bl``/``set_x_bl``/``set_z_bl`` place the downstream ghost node
(``grlp.grlp``: ``set_x_bl`` assigns ``x_ghost_downstream = x_bl``).

TWO TRANSFER FUNCTIONS, kept separate because they are different kinds of
statement:

    x -> A    Hack's law, geometric:  A = k_xA x**P_xA   (P_xA = 7/4)
    A -> Q    hydrology:              Q = k_AQ A**P_AQ   (P_AQ = 0.7)
    Q -> B    Leopold, optional:      B = k_QB Q**P_QB   (P_QB = 1/2)

Composed, ``Q ~ x**(P_xA P_AQ)`` -- GRLP's ``P_xQ = P_xA * P_AQ``.  The runoff
rule ``Q = R A`` is the LINEAR special case of the second, ``P_AQ = 1``, and
``set_Q(R=...)`` says so rather than leaving ``P_AQ`` set and ignored.

GRLP's ``q_R`` (m/hr, with an ``A_R`` saturation area) is a different
parameterization of the same transfer and is not implemented here.

CHANNEL WIDTH can be handled either way, and the model does not choose for
you:

- IMPLICIT, the standard way and the default.  ``basis="Q"`` folds width into
  ``m`` through ``B ~ Q**P_QB``, with the width coefficient absorbed into
  ``K``.  Leopold's ``P_QB = 1/2`` gives ``m/n = 1/2``.  Exact whenever width
  really is a power law of discharge, and ``k_QB`` is unidentifiable anyway.
- EXPLICIT, semi-independent.  ``basis="Q/B"`` erodes with unit stream power
  ``(Q/B)**m S**n`` and carries ``B`` as a field from :meth:`set_B`.  With a
  power-law ``B`` this reduces exactly to the implicit form, which the test
  suite proves rather than asserts; it is a genuinely different model only
  when ``B`` is not a power law of ``Q``.

Neither mode makes width a dynamic state variable that responds to incision.
GRLP does that (``set_B``, ``narrow_by_incision``); DRLP does not.

VALIDITY: BASE-LEVEL FALL ONLY.  The stream-power law is purely erosional --
``E = K Q**m S**n >= 0``, with no deposition term -- so ``dz/dt <= U``
everywhere and at all times.  On base-level FALL that is the whole point, and
the model is doing what it is for.  On base-level RISE it has no fluvial
mechanism to raise the bed at all: the bed can only come up at the rock uplift
rate ``U``, and with ``U = 0`` it cannot come up by any amount, so the lower
reach stays drowned indefinitely while the reach above it keeps incising and
the sediment that incision produces simply disappears.  A real river aggrades.
Output under a base-level rise is therefore not a prediction, and the
transport-limited siblings GRLP and SRLP are the models that can represent
it.

The model can be driven by discharge or by drainage area alone.  Because
``K_Q Q**m = (K_Q R**m) A**m``, the two are a rescaling of one another, and
``set_K``'s ``basis`` states which one a given ``K`` belongs to.  Area is the
convention of the stream-power literature and of fastscapelib, Landlab and
fastscapelib-fortran; discharge is the form ``HANDOFF.md`` specifies and is
GRLP's convention.  Neither is privileged here, but one of them has to be
named, because it fixes what ``K``'s number means.
"""

import numpy as np

from .incision import step


def relax_limits(current, target, relax_frac):
    """Move axis limits toward ``target``: outward at once, inward gradually.

    Returns ``(lo, hi)``.  The asymmetry is the whole point.  **Outward is
    immediate**, because a plot that clips the thing it is plotting is worse
    than no plot: a profile that has left the frame looks exactly like one
    that stopped moving.  **Inward is gradual**, at ``relax_frac`` of the
    remaining gap per call, because a view that snaps to a tight fit every
    frame jitters, and one that never tightens strands the subject in a
    corner of its own axis.

    ``relax_frac`` of 0 never tightens and 1 snaps straight to the fit.  A
    caller animating at a fixed frame rate can set it from a time constant:
    ``1 - exp(-dt_frame / tau)``.
    """
    lo, hi = (float(v) for v in current)
    t_lo, t_hi = (float(v) for v in target)
    if not 0.0 <= relax_frac <= 1.0:
        raise ValueError(f"relax_frac must be in [0, 1]; got {relax_frac}")
    new_lo = t_lo if t_lo < lo else lo + relax_frac * (t_lo - lo)
    new_hi = t_hi if t_hi > hi else hi + relax_frac * (t_hi - hi)
    return new_lo, new_hi


class LongProfile(object):
    """A detachment-limited river long profile.

    Built by construction then ``set_*``, as GRLP is.  ``z`` is mutated in
    place by :meth:`evolve`.
    """

    def __init__(self):
        self.x = None
        self.z = None
        self.A = None
        self.Q = None
        self.B = None
        self.K = None
        self.K_basis = "Q"
        self.U = 0.0
        self.m = 0.5
        self.n = 1.0
        self.x_bl = None
        self._z_bl = 0.0
        self.t = 0.0
        self.set_hydrologic_constants()

    # ---------------------------------------------------------------- grid

    def set_x(self, x=None, dx=None, nx=None, x0=None):
        """Set the downstream distance [m].  Index 0 is upstream.

        Either ``x`` directly, or a uniform grid from ``dx``, ``nx`` and
        ``x0`` (the position of node 0).  The downstream ghost node defaults
        to one spacing beyond the last node, as in GRLP; ``set_x_bl`` moves it.
        """
        if x is not None:
            self.x = np.asarray(x, dtype=float)
        elif None not in (dx, nx):
            x0 = dx if x0 is None else x0
            self.x = x0 + dx * np.arange(nx, dtype=float)
        else:
            raise ValueError("set_x needs x, or dx and nx")
        if self.x_bl is None:
            self.x_bl = self.x[-1] + (self.x[-1] - self.x[-2])
        return self

    def set_z(self, z):
        """Set elevation [m]."""
        self.z = np.asarray(z, dtype=float).copy()
        return self

    # ------------------------------------------------------ hydrology

    def set_hydrologic_constants(self, P_xA=7 / 4.0, P_AQ=0.7):
        """Exponents of the two transfer functions, with GRLP's names.

        The chain from position to the quantity that erodes is two separate
        transfers, and they are different kinds of statement:

        1. ``x -> A``  **Hack's law**, geometric: ``A = k_xA x**P_xA``.
           ``P_xA`` is the INVERSE Hack exponent (Hack's own law is
           ``L ~ A**0.6``, so ``A ~ L**1.67``; GRLP's default is 7/4).
        2. ``A -> Q``  **hydrology**: ``Q = k_AQ A**P_AQ``.  ``P_AQ`` is
           sub-linear in GRLP's default, 0.7, because discharge grows more
           slowly than area -- channel losses, and storms that do not cover a
           large basin.

        Composing them, ``Q ~ x**(P_xA P_AQ)``, which is GRLP's identity
        ``P_xQ = P_xA * P_AQ``; :attr:`P_xQ` exposes it.
        """
        self.P_xA = P_xA
        self.P_AQ = P_AQ
        return self

    @property
    def P_xQ(self):
        """``P_xA * P_AQ``: the exponent of the composed transfer ``x -> Q``.

        GRLP's identity, which holds whenever both transfers are the power
        laws above.  It does not hold if ``A`` or ``Q`` was set directly from
        an array, so it is a property of the exponents, not a claim about the
        current fields.
        """
        return self.P_xA * self.P_AQ

    def set_A(self, A=None, k_xA=None, P_xA=None):
        """The ``x -> A`` transfer: Hack's law.  Drainage area [m**2].

        ``A = k_xA x**P_xA``, with ``P_xA`` the INVERSE Hack exponent
        (default 7/4, GRLP's).  Or pass ``A`` as an array for a real basin,
        in which case no power law is assumed.
        """
        if A is not None:
            self.A = np.broadcast_to(np.asarray(A, dtype=float),
                                     self.x.shape).copy()
        elif k_xA is not None:
            if P_xA is not None:
                self.P_xA = P_xA
            self.k_xA = k_xA
            self.A = k_xA * self.x**self.P_xA
        else:
            raise ValueError("set_A needs A, or k_xA")
        return self

    def set_Q(self, Q=None, k_AQ=None, P_AQ=None, R=None):
        """The ``A -> Q`` transfer: hydrology.  Discharge [m**3/s].

        Three routes:

        - ``Q``: an array or scalar, directly.
        - ``k_AQ``, optionally ``P_AQ``: the general power law
          ``Q = k_AQ A**P_AQ``, with ``P_AQ`` defaulting to whatever
          :meth:`set_hydrologic_constants` holds (0.7).
        - ``R``: a runoff rate [m/s], which is the LINEAR special case
          ``Q = R A``, i.e. ``k_AQ = R`` with ``P_AQ = 1``.  Passing ``R``
          therefore overrides ``P_AQ`` for this field, and says so by setting
          it to 1.

        Note the units: for ``P_AQ != 1``, ``k_AQ`` carries
        m**3 s**-1 m**(-2 P_AQ), which is unavoidable in a power law and is
        why ``R`` is the dimensionally clean route.  (GRLP sidesteps this with
        its ``q_R``/``A_R`` saturation-area form; DRLP does not implement
        that.)
        """
        if Q is not None:
            self.Q = np.broadcast_to(np.asarray(Q, dtype=float),
                                     self.x.shape).copy()
            return self
        if R is not None:
            if k_AQ is not None or P_AQ is not None:
                raise ValueError(
                    "R is the linear special case of k_AQ and P_AQ; pass "
                    "either R, or k_AQ (with P_AQ), not both"
                )
            k_AQ, P_AQ = R, 1.0
            self.R = R
        if k_AQ is None:
            raise ValueError("set_Q needs Q, or k_AQ, or R")
        if self.A is None:
            raise ValueError(
                "the A -> Q transfer needs A; call set_A first"
            )
        if P_AQ is not None:
            self.P_AQ = P_AQ
        self.k_AQ = k_AQ
        self.Q = k_AQ * self.A**self.P_AQ
        return self

    def set_B(self, B=None, k_QB=None, P_QB=0.5):
        """The ``Q -> B`` transfer: channel width [m], after Leopold.

        ``B = k_QB Q**P_QB``, with ``P_QB`` Leopold & Maddock's downstream
        hydraulic-geometry exponent, 1/2 by default.  Or pass ``B`` as an
        array -- measured widths, or a width that varies for reasons the
        power law does not capture.

        Width is only used when ``K`` is set with ``basis="Q/B"``.  With the
        default ``basis="Q"`` width is implicit in ``m`` and this method need
        never be called; see :meth:`set_K`.
        """
        if B is not None:
            self.B = np.broadcast_to(np.asarray(B, dtype=float),
                                     self.x.shape).copy()
            return self
        if k_QB is None:
            raise ValueError("set_B needs B, or k_QB")
        if self.Q is None:
            raise ValueError("the Q -> B transfer needs Q; call set_Q first")
        self.k_QB = k_QB
        self.P_QB = P_QB
        self.B = k_QB * self.Q**P_QB
        return self

    @staticmethod
    def implicit_m_from_unit(m_unit, P_QB=0.5):
        """``m`` for the implicit form equivalent to unit stream power.

        ``m_implicit = m_unit (1 - P_QB)``.  With Leopold's 1/2 this halves
        the exponent, which is why the standard implicit form has
        ``m/n = 1/2``.
        """
        return m_unit * (1.0 - P_QB)

    @staticmethod
    def implicit_K_from_unit(K_unit, k_QB, m_unit):
        """``K`` for the implicit form equivalent to unit stream power.

        ``K_implicit = K_unit k_QB**(-m_unit)``.  The width coefficient is
        absorbed here, which is exactly why ``k_QB`` is not identifiable from
        a profile: any ``k_QB`` can be traded against ``K``.
        """
        return K_unit * k_QB ** (-m_unit)

    # ------------------------------------------------------- erosion law

    def set_K(self, K, basis="Q"):
        """Set erodibility.  Scalar or per node; units depend on ``m``, ``n``
        and the basis.

        ``basis`` says which quantity the stream-power term acts on, and
        therefore what ``K`` means:

        - ``"Q"`` (default): ``dz/dt = U - K Q**m S**n``, the form
          ``HANDOFF.md`` specifies and GRLP's discharge-based convention.
        - ``"A"``: ``dz/dt = U - K A**m S**n``, driven by drainage area alone.
          This is the convention of the stream-power literature and of all
          three reference implementations -- fastscapelib's
          ``erode(elevation, drainage_area, dt)``, Landlab's required
          ``drainage_area`` field, fastscapelib-fortran's ``a(ijk)``.
        - ``"Q/B"``: ``dz/dt = U - K (Q/B)**m S**n``, UNIT stream power, with
          channel width carried explicitly from :meth:`set_B`.

        WIDTH.  Under ``"Q"`` and ``"A"`` width is implicit in ``m``, through
        downstream hydraulic geometry ``B ~ Q**P_QB``: unit stream power
        ``(Q/B)**m`` becomes ``Q**(m(1-P_QB))`` with the width coefficient
        absorbed into ``K``.  With Leopold's ``P_QB = 1/2`` that is
        ``m/n = 1/2``, the standard choice and the one this model defaults to
        in spirit.  This is the normal way to run a stream-power model and it
        loses nothing when width really is a power law of discharge.

        ``"Q/B"`` is only SEMI-independent: if ``B = k_QB Q**P_QB`` exactly,
        it reduces algebraically to ``"Q"`` with
        ``m -> m(1 - P_QB)`` and ``K -> K k_QB**(-m)``
        (:meth:`implicit_m_from_unit`, :meth:`implicit_K_from_unit`), so
        ``k_QB`` cannot be identified from a profile.  It becomes a different
        model only when ``B`` is NOT a power law of ``Q`` -- measured widths,
        lithologic control, a channel that narrows where it steepens.  What it
        always adds is that width becomes an output you can inspect rather
        than an assumption folded into an exponent.

        The two are related by ``K_A = K_Q R**m`` when ``Q = R A``, so this is
        a rescaling and not a different model.  It is nonetheless a MODEL
        choice, not a bookkeeping one: it fixes what ``K``'s number means and
        what the profile predicts, which is why it is stated here rather than
        inferred from which field happens to be populated.

        (``basis`` is my naming, not borrowed from a precedent.)
        """
        if basis not in ("Q", "A", "Q/B"):
            raise ValueError(
                f'basis must be "Q", "A" or "Q/B"; got {basis!r}'
            )
        self.K = np.broadcast_to(np.asarray(K, dtype=float),
                                 self.x.shape).copy()
        self.K_basis = basis
        return self

    @staticmethod
    def K_A_from_K_Q(K_Q, R, m):
        """``K_A = K_Q R**m``: the area-based erodibility equivalent to a
        discharge-based one under ``Q = R A``."""
        return K_Q * R**m

    def set_mn(self, m=None, n=None):
        """Set the discharge and slope exponents."""
        if m is not None:
            self.m = m
        if n is not None:
            self.n = n
        return self

    def set_uplift_rate(self, U):
        """Set rock uplift rate [m/s].  Scalar or per node."""
        self.U = np.broadcast_to(np.asarray(U, dtype=float),
                                 self.x.shape).copy()
        return self

    # -------------------------------------------------- boundary condition

    def set_x_bl(self, x_bl):
        """Position [m] of the downstream ghost node."""
        self.x_bl = float(x_bl)
        return self

    def set_z_bl(self, z_bl):
        """Elevation [m] of the downstream ghost node: base level.

        A callable ``z_bl(t)`` makes base level a function of model time, which
        is how a base-level history is imposed.

        **Fall only.**  Nothing here forbids a rise, and nothing checks for
        one, but the stream-power law cannot represent it: ``E >= 0`` with no
        deposition term, so the bed comes up only at the rock uplift rate and
        not at all when ``U = 0``.  See the module docstring.
        """
        self._z_bl = z_bl
        return self

    def set_bl(self, x=None, z=None):
        """Set base level, the downstream boundary point ``(x, z)``.

        Pass only one of ``x``, ``z`` to move it in that coordinate alone.
        """
        if x is not None:
            self.set_x_bl(x)
        if z is not None:
            self.set_z_bl(z)
        return self

    @property
    def z_bl(self):
        """Base-level elevation [m] at the current model time."""
        return self._z_bl(self.t) if callable(self._z_bl) else self._z_bl

    # ----------------------------------------------------------- diagnostics

    @property
    def driver(self):
        """The quantity the stream-power term acts on: ``Q`` or ``A``.

        Set by :meth:`set_K`'s ``basis``: ``Q``, ``A``, or ``Q/B`` for unit
        stream power.  This is what is passed into
        :func:`drlp.incision.step`, whose corresponding argument is named
        ``Q`` because that is the form the governing equation and FluvTree's
        sweep are written in.
        """
        if self.K_basis == "Q/B":
            if self.Q is None:
                raise ValueError(
                    'K was set with basis="Q/B", so unit stream power drives '
                    "the model: call set_Q first"
                )
            if self.B is None:
                raise ValueError(
                    'K was set with basis="Q/B", so width is explicit: call '
                    "set_B, or use basis=\"Q\" to fold width into m"
                )
            return self.Q / self.B
        if self.K_basis == "A":
            if self.A is None:
                raise ValueError(
                    'K was set with basis="A", so drainage area drives the '
                    "model: call set_A first"
                )
            return self.A
        if self.Q is None:
            raise ValueError(
                'K was set with basis="Q", so discharge drives the model: '
                "call set_Q, or set_K(..., basis=\"A\") to drive from area"
            )
        return self.Q

    @property
    def dx(self):
        """Distance [m] from each node to its receiver; last is to the ghost."""
        return np.diff(np.append(self.x, self.x_bl))

    @property
    def S(self):
        """Downstream slope at each node, positive downstream."""
        return -np.diff(np.append(self.z, self.z_bl)) / self.dx

    def elevation_limits(self, z_bl_floor=None, pad_frac=0.08):
        """Elevation axis limits [m] that cannot clip the profile.

        Returns ``(lo, hi)``.  ``z_bl_floor`` is the lowest base level the
        caller can still reach without rebuilding the profile -- a slider's
        minimum, say.  Passing it reserves room for a drop that has not
        happened yet, so the profile and the base-level line both stay in view
        when it does.

        The lower limit is safe for a reason rather than by margin: **no node
        ever passes below the thing it drains into.**  The implicit update

            z_new = (z_old + dt U + c z_r) / (1 + c)

        is a weighted average of the node's no-erosion elevation and its
        already-updated receiver, with ``c > 0``, so ``z_new`` lies between
        them whenever ``z_old + dt U >= z_r``; and when it does not, the node
        takes uplift only and does not move down at all.  By induction from
        the ghost node, every node satisfies ``z >= z_bl`` at every step.  So
        the deepest the profile can reach is the deepest base level allowed.

        There is no matching bound above: uplift raises the profile without
        limit, at up to ``U`` per unit time.  A caller animating the model
        should therefore EXPAND these limits when the profile grows rather
        than trusting one call, which is what ``pad_frac`` buys time for.
        """
        lows = [float(self.z.min()), float(self.z_bl)]
        if z_bl_floor is not None:
            lows.append(float(z_bl_floor))
        lo = min(lows)
        hi = max(float(self.z.max()), float(self.z_bl))
        pad = pad_frac * max(hi - lo, 1.0)
        return lo - pad, hi + pad

    def steady_state_slope(self):
        """``S = (U / (K Q**m))**(1/n)``, the closed-form steady slope."""
        return (self.U / (self.K * self.driver**self.m)) ** (1.0 / self.n)

    def set_steady_state(self):
        """Set ``z`` to the analytic steady-state profile.

        Integrates the closed-form slope upstream from the ghost node.  Node
        ``i`` satisfies ``z_i = z_r + S_i dx_i`` exactly, which is the discrete
        fixed point the stepper solves, so this is a fixed point of
        :meth:`evolve` and not merely close to one.
        """
        S = self.steady_state_slope()
        z = np.empty_like(self.x)
        z_r = self.z_bl
        for i in range(self.x.size - 1, -1, -1):
            x_r = self.x_bl if i == self.x.size - 1 else self.x[i + 1]
            z[i] = z_r + S[i] * (x_r - self.x[i])
            z_r = z[i]
        self.z = z
        return self

    # ------------------------------------------------------------- evolution

    def evolve(self, nt=1, dt=None):
        """Advance ``nt`` implicit steps of ``dt`` [s], in place on ``z``.

        Base level is re-read each step, so a callable ``z_bl(t)`` is sampled
        at the time the step starts.
        """
        if dt is None:
            raise ValueError("evolve needs dt [s]")
        for _ in range(int(nt)):
            self.z = step(self.x, self.z, self.driver, self.K, self.m,
                          self.n, self.U, dt, self.x_bl, self.z_bl)
            self.t += dt
        return self

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

Composed, ``Q ~ x**(P_xA P_AQ)`` -- GRLP's ``P_xQ = P_xA * P_AQ``.  The runoff
rule ``Q = R A`` is the LINEAR special case of the second, ``P_AQ = 1``, and
``set_Q(R=...)`` says so rather than leaving ``P_AQ`` set and ignored.

GRLP's ``q_R`` (m/hr, with an ``A_R`` saturation area) is a different
parameterisation of the same transfer and is not implemented here.

CHANNEL WIDTH is not a variable in DRLP.  It enters only through ``m``, via
downstream hydraulic geometry ``w ~ Q**b``: for unit stream power
``m = a c (1 - b)``, so ``b`` is buried inside a number the user supplies for
``m``.  That is the standard stream-power assumption and it is exact for a
static power-law width.  It cannot represent width that adjusts dynamically --
GRLP does carry width explicitly (``set_B``, ``narrow_by_incision``), and a
model that needs that needs a width state variable, not a different ``m``.

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

        The two are related by ``K_A = K_Q R**m`` when ``Q = R A``, so this is
        a rescaling and not a different model.  It is nonetheless a MODEL
        choice, not a bookkeeping one: it fixes what ``K``'s number means and
        what the profile predicts, which is why it is stated here rather than
        inferred from which field happens to be populated.

        (``basis`` is my naming, not borrowed from a precedent.)
        """
        if basis not in ("Q", "A"):
            raise ValueError(f'basis must be "Q" or "A"; got {basis!r}')
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

        Set by :meth:`set_K`'s ``basis``.  This is what is passed into
        :func:`drlp.incision.step`, whose corresponding argument is named
        ``Q`` because that is the form the governing equation and FluvTree's
        sweep are written in.
        """
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

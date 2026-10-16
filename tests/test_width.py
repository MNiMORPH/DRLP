"""
Channel width, both ways.

Implicit (the standard way, and the default): width folded into ``m`` through
``B ~ Q**P_QB``, the coefficient absorbed into ``K``.
Explicit (semi-independent): unit stream power ``(Q/B)**m S**n`` with ``B``
carried as a field from Leopold's ``Q -> B`` transfer.

The point of these tests is to pin down exactly where the two coincide and
where they do not.
"""

import numpy as np
import pytest

from drlp import LongProfile


U = 1e-3 / 3.15e7
K, M = 1e-9, 0.5


def _base(nx=30, **kw):
    return (LongProfile().set_x(dx=1000.0, nx=nx)
            .set_A(k_xA=1.0, P_xA=7 / 4.0)
            .set_Q(k_AQ=1e-6, P_AQ=0.7))


# ------------------------------------------------------ Q -> B  (Leopold)

def test_leopold_is_the_Q_to_B_transfer():
    lp = _base().set_B(k_QB=8.0, P_QB=0.5)
    assert np.allclose(lp.B, 8.0 * lp.Q**0.5, rtol=1e-14)
    assert np.polyfit(np.log(lp.Q), np.log(lp.B), 1)[0] == pytest.approx(
        0.5, rel=1e-10)


def test_P_QB_defaults_to_one_half():
    lp = _base().set_B(k_QB=8.0)
    assert lp.P_QB == 0.5


def test_width_given_directly_assumes_no_power_law():
    lp = _base(nx=10)
    B = np.linspace(5.0, 50.0, 10)
    lp.set_B(B=B)
    assert np.allclose(lp.B, B)


def test_Q_to_B_transfer_needs_Q():
    lp = LongProfile().set_x(dx=1000.0, nx=10)
    with pytest.raises(ValueError, match="needs Q"):
        lp.set_B(k_QB=8.0)


# --------------------------------------- the two modes coincide exactly

def test_power_law_width_reduces_exactly_to_the_implicit_form():
    # The reason explicit width is only SEMI-independent. With
    # B = k_QB Q**P_QB, unit stream power (Q/B)**m is Q**(m(1-P_QB)) with the
    # width coefficient absorbed into K. So the two parameterisations are the
    # same model and k_QB is not identifiable from a profile.
    m_u, n, k_QB, P_QB, K_u = 1.0, 1.0, 12.0, 0.5, 2e-5

    explicit = (_base().set_B(k_QB=k_QB, P_QB=P_QB)
                .set_K(K_u, basis="Q/B").set_mn(m=m_u, n=n)
                .set_uplift_rate(U).set_bl(z=0.0))
    implicit = (_base()
                .set_K(LongProfile.implicit_K_from_unit(K_u, k_QB, m_u),
                       basis="Q")
                .set_mn(m=LongProfile.implicit_m_from_unit(m_u, P_QB), n=n)
                .set_uplift_rate(U).set_bl(z=0.0))

    # the transformation is the documented one
    assert implicit.m == pytest.approx(m_u * (1.0 - P_QB))
    assert implicit.K[0] == pytest.approx(K_u * k_QB ** (-m_u))

    for lp in (explicit, implicit):
        lp.set_steady_state()
    assert np.allclose(explicit.z, implicit.z, rtol=1e-12, atol=1e-12)

    # and they stay identical through a transient, not just at steady state
    for lp in (explicit, implicit):
        lp.set_bl(z=-40.0)
        lp.evolve(nt=30, dt=1e11)
    assert np.allclose(explicit.z, implicit.z, rtol=1e-12, atol=1e-12)


def test_leopold_scaling_gives_the_standard_m_over_n_of_one_half():
    # The standard way: P_QB = 1/2 halves the exponent, so unit stream power
    # with m = n becomes an implicit form with m/n = 1/2.
    assert LongProfile.implicit_m_from_unit(1.0, 0.5) == pytest.approx(0.5)
    assert LongProfile.implicit_m_from_unit(2.0, 0.5) == pytest.approx(1.0)
    # and the width coefficient is traded against K, i.e. unidentifiable
    assert (LongProfile.implicit_K_from_unit(1.0, 4.0, 1.0)
            == pytest.approx(0.25))
    assert (LongProfile.implicit_K_from_unit(4.0, 4.0, 1.0)
            == pytest.approx(1.0))


# ----------------------------------- and where they genuinely diverge

def test_non_power_law_width_is_a_model_the_implicit_form_cannot_express():
    # A gorge: the channel narrows over part of its length for reasons
    # discharge does not know about. Narrower channel -> higher unit stream
    # power -> a GENTLER steady slope, by
    #
    #     S_narrow / S_wide = (B_narrow / B_wide)**(m/n)
    #
    # Discharge is smooth here, so no choice of m or K in the implicit form
    # produces this step. This is the capability explicit width adds.
    nx = 40
    lp = _base(nx=nx)
    B = np.where(lp.x < 20000.0, 20.0, 5.0)      # narrows four-fold
    m, n = 1.0, 1.0
    lp.set_B(B=B).set_K(2e-5, basis="Q/B").set_mn(m=m, n=n)
    lp.set_uplift_rate(U).set_bl(z=0.0).set_steady_state()

    S = lp.S
    # compare the two sides away from the junction, correcting for the smooth
    # discharge trend by using the closed form as the reference
    assert np.allclose(S, lp.steady_state_slope(), rtol=1e-12)
    ratio = (S[-1] / S[0]) / ((lp.Q[-1] / lp.Q[0]) ** (-m / n))
    assert ratio == pytest.approx((5.0 / 20.0) ** (m / n), rel=1e-10)

    # a fixed point of the transient solver too
    z0 = lp.z.copy()
    lp.evolve(nt=20, dt=1e11)
    assert np.allclose(lp.z, z0, rtol=0, atol=1e-6)


def test_width_is_an_inspectable_output():
    # Even where it is only a reparameterisation, the explicit route makes
    # width something you can look at rather than an assumption inside m.
    lp = _base().set_B(k_QB=8.0)
    assert lp.B.shape == lp.x.shape
    assert np.all(np.diff(lp.B) > 0)          # widens downstream
    assert np.all(lp.driver == lp.Q / lp.B) if lp.K_basis == "Q/B" else True


# -------------------------------------------------------------- the basis

def test_unit_basis_errors_are_explicit():
    lp = LongProfile().set_x(dx=1000.0, nx=10)
    with pytest.raises(ValueError, match='basis must be'):
        lp.set_K(K, basis="unit")

    lp.set_K(K, basis="Q/B")
    with pytest.raises(ValueError, match="call set_Q first"):
        _ = lp.driver

    lp2 = _base(nx=10).set_K(K, basis="Q/B")
    with pytest.raises(ValueError, match="call set_B"):
        _ = lp2.driver


def test_driver_is_specific_discharge_under_the_unit_basis():
    lp = _base().set_B(k_QB=8.0).set_K(2e-5, basis="Q/B")
    assert np.allclose(lp.driver, lp.Q / lp.B)

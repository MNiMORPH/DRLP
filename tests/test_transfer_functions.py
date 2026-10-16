"""
The two transfer functions, kept separate because they are different kinds of
statement:

    x -> A    Hack's law, geometric:  A = k_xA x**P_xA
    A -> Q    hydrology:              Q = k_AQ A**P_AQ

and their composition, GRLP's ``P_xQ = P_xA * P_AQ``.
"""

import numpy as np
import pytest

from drlp import LongProfile


U = 1e-3 / 3.15e7
K, M = 1e-9, 0.5


# ------------------------------------------------------- x -> A  (Hack's law)

def test_hacks_law_is_the_x_to_A_transfer():
    lp = (LongProfile().set_x(dx=1000.0, nx=25)
          .set_A(k_xA=2.5, P_xA=7 / 4.0))
    assert np.allclose(lp.A, 2.5 * lp.x ** (7 / 4.0), rtol=1e-14)
    # the exponent is recoverable from the field it produced
    assert np.polyfit(np.log(lp.x), np.log(lp.A), 1)[0] == pytest.approx(
        7 / 4.0, rel=1e-10)


def test_area_given_directly_assumes_no_power_law():
    x = 1000.0 * np.arange(1, 11)
    A = np.linspace(1e6, 9e6, 10)           # not a power law at all
    lp = LongProfile().set_x(x=x).set_A(A=A)
    assert np.allclose(lp.A, A)


# ------------------------------------------------------ A -> Q  (hydrology)

def test_hydrology_honours_P_AQ():
    # REGRESSION. set_hydrologic_constants accepted P_AQ and set_Q ignored it,
    # hardcoding Q = R A, i.e. P_AQ = 1. A parameter taken and dropped.
    lp = (LongProfile().set_x(dx=1000.0, nx=25)
          .set_hydrologic_constants(P_xA=7 / 4.0, P_AQ=0.7)
          .set_A(k_xA=1.0)
          .set_Q(k_AQ=1e-6))
    assert np.allclose(lp.Q, 1e-6 * lp.A**0.7, rtol=1e-14)
    # and the exponent is NOT 1
    measured = np.polyfit(np.log(lp.A), np.log(lp.Q), 1)[0]
    assert measured == pytest.approx(0.7, rel=1e-10)


def test_P_AQ_may_be_set_on_the_call():
    lp = (LongProfile().set_x(dx=1000.0, nx=20).set_A(k_xA=1.0)
          .set_Q(k_AQ=1e-6, P_AQ=0.55))
    assert lp.P_AQ == 0.55
    assert np.allclose(lp.Q, 1e-6 * lp.A**0.55, rtol=1e-14)


def test_runoff_is_the_linear_special_case():
    # Q = R A is k_AQ = R with P_AQ = 1, and set_Q says so rather than
    # leaving a stale P_AQ in place.
    lp = (LongProfile().set_x(dx=1000.0, nx=20)
          .set_hydrologic_constants(P_AQ=0.7)       # deliberately not 1
          .set_A(k_xA=1.0)
          .set_Q(R=1e-8))
    assert lp.P_AQ == 1.0
    assert np.allclose(lp.Q, 1e-8 * lp.A, rtol=1e-14)


def test_runoff_and_the_general_law_are_mutually_exclusive():
    lp = LongProfile().set_x(dx=1000.0, nx=10).set_A(k_xA=1.0)
    with pytest.raises(ValueError, match="not both"):
        lp.set_Q(R=1e-8, k_AQ=1e-6)
    with pytest.raises(ValueError, match="not both"):
        lp.set_Q(R=1e-8, P_AQ=0.7)


def test_hydrology_needs_area():
    lp = LongProfile().set_x(dx=1000.0, nx=10)
    with pytest.raises(ValueError, match="needs A"):
        lp.set_Q(k_AQ=1e-6)


# ------------------------------------------------------------- composition

def test_P_xQ_is_the_composition_of_the_two_exponents():
    lp = (LongProfile().set_x(dx=1000.0, nx=30)
          .set_hydrologic_constants(P_xA=7 / 4.0, P_AQ=0.7)
          .set_A(k_xA=1.0).set_Q(k_AQ=1e-6))
    assert lp.P_xQ == pytest.approx(7 / 4.0 * 0.7)
    # and the composed field really has that exponent in x
    measured = np.polyfit(np.log(lp.x), np.log(lp.Q), 1)[0]
    assert measured == pytest.approx(lp.P_xQ, rel=1e-10)


# -------------------------------------- the hydrology changes the concavity

@pytest.mark.parametrize("P_AQ", [1.0, 0.7, 0.5])
def test_hydrology_exponent_sets_the_slope_area_concavity(P_AQ):
    # Free prediction, and the reason the two transfers must stay separate:
    # at steady state S = (U/(K Q**m))**(1/n) with Q = k_AQ A**P_AQ, so
    #
    #     d(ln S)/d(ln A) = -m P_AQ / n
    #
    # The concavity index theta is NOT m/n unless the hydrology is linear.
    # Choosing P_AQ therefore changes the predicted profile shape, which a
    # model that silently forces P_AQ = 1 cannot express.
    m, n = 0.5, 1.0
    lp = (LongProfile().set_x(dx=1000.0, nx=60)
          .set_A(k_xA=1.0, P_xA=7 / 4.0)
          .set_Q(k_AQ=1e-6, P_AQ=P_AQ)
          .set_K(K, basis="Q").set_mn(m=m, n=n)
          .set_uplift_rate(U).set_bl(z=0.0))
    lp.set_steady_state()

    theta = -np.polyfit(np.log(lp.A), np.log(lp.S), 1)[0]
    assert theta == pytest.approx(m * P_AQ / n, rel=1e-8)

    # and it is a fixed point of the transient solver
    z0 = lp.z.copy()
    lp.evolve(nt=20, dt=1e11)
    assert np.allclose(lp.z, z0, rtol=0, atol=1e-6)

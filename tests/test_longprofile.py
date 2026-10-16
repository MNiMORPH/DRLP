"""
``LongProfile``: the standalone single-channel model.

The checks that carry weight are free predictions -- the analytic profile
against the transient solver, and the analytic knickpoint celerity against the
measured one.
"""

import numpy as np
import pytest

from drlp import LongProfile


U = 1e-3 / 3.15e7          # 1 mm/yr in m/s
K, M = 1e-9, 0.5


def _lp(nx=30, n=1.0, Q=10.0, U=U, K=K):
    return (LongProfile()
            .set_x(dx=1000.0, nx=nx)
            .set_Q(Q=Q)
            .set_K(K)
            .set_mn(m=M, n=n)
            .set_uplift_rate(U)
            .set_bl(z=0.0))


# ------------------------------------------------------------------ geometry

def test_ghost_node_defaults_to_one_spacing_beyond_and_set_x_bl_moves_it():
    lp = _lp(nx=10)
    assert lp.x_bl == pytest.approx(lp.x[-1] + 1000.0)
    lp.set_x_bl(99000.0)
    assert lp.x_bl == 99000.0
    assert lp.dx[-1] == pytest.approx(99000.0 - lp.x[-1])


def test_set_bl_moves_either_coordinate_alone():
    lp = _lp(nx=10)
    x_before = lp.x_bl
    lp.set_bl(z=-25.0)
    assert lp.z_bl == -25.0 and lp.x_bl == x_before
    lp.set_bl(x=20000.0)
    assert lp.x_bl == 20000.0 and lp.z_bl == -25.0


# ----------------------------------------------------------------- hydrology

def test_hacks_law_and_the_runoff_rule():
    lp = (LongProfile().set_x(dx=1000.0, nx=20)
          .set_A(k_xA=1.0, P_xA=7 / 4.0)
          .set_Q(R=1e-8))
    assert np.allclose(lp.A, 1.0 * lp.x ** (7 / 4.0))
    assert np.allclose(lp.Q, 1e-8 * lp.A)


def test_area_basis_and_discharge_basis_agree_under_the_rescaling():
    # K_Q Q**m = (K_Q R**m) A**m, so driving from discharge with K_Q and from
    # area with K_A = K_Q R**m must give identical profiles. The identity the
    # two routes exist to satisfy, and it is checked through the first-class
    # area route -- not by putting area into the discharge slot.
    R = 1e-8
    common = dict(dx=1000.0, nx=25)

    lp_Q = (LongProfile().set_x(**common).set_A(k_xA=1.0).set_Q(R=R)
            .set_K(K, basis="Q").set_mn(m=M, n=1.0)
            .set_uplift_rate(U).set_bl(z=0.0))
    lp_A = (LongProfile().set_x(**common).set_A(k_xA=1.0)
            .set_K(LongProfile.K_A_from_K_Q(K, R, M), basis="A")
            .set_mn(m=M, n=1.0).set_uplift_rate(U).set_bl(z=0.0))

    for lp in (lp_Q, lp_A):
        lp.set_steady_state()
    assert np.allclose(lp_Q.z, lp_A.z, rtol=1e-12, atol=0)
    # and they stay identical through a transient
    for lp in (lp_Q, lp_A):
        lp.set_bl(z=-40.0)
        lp.evolve(nt=30, dt=1e11)
    assert np.allclose(lp_Q.z, lp_A.z, rtol=1e-12, atol=0)


def test_driven_from_drainage_area_alone():
    # A(x) and nothing else: discharge is never set.
    lp = (LongProfile().set_x(dx=1000.0, nx=30)
          .set_A(k_xA=1.0, P_xA=7 / 4.0)
          .set_K(1e-13, basis="A").set_mn(m=0.45, n=1.0)
          .set_uplift_rate(U).set_bl(z=0.0))
    assert lp.Q is None
    assert lp.driver is lp.A

    lp.set_steady_state()
    assert np.allclose(lp.S, lp.steady_state_slope(), rtol=1e-12, atol=0)
    z0 = lp.z.copy()
    lp.evolve(nt=20, dt=1e11)
    assert np.allclose(lp.z, z0, rtol=0, atol=1e-6)


def test_arbitrary_area_function_of_distance():
    # A(x) need not be a power law. A stepwise jump in area -- a tributary
    # junction -- must step the steady slope by (A_down/A_up)**(-m/n).
    nx = 40
    x = 1000.0 * np.arange(1, nx + 1)
    A = np.where(x < 20000.0, 1e7, 4e7)        # area quadruples at a junction
    lp = (LongProfile().set_x(x=x).set_A(A=A)
          .set_K(1e-13, basis="A").set_mn(m=0.5, n=1.0)
          .set_uplift_rate(U).set_bl(z=0.0))
    lp.set_steady_state()

    S = lp.S
    ratio = S[-1] / S[0]
    assert ratio == pytest.approx((4e7 / 1e7) ** (-0.5 / 1.0), rel=1e-12)
    # and it is a fixed point of the transient solver
    z0 = lp.z.copy()
    lp.evolve(nt=20, dt=1e11)
    assert np.allclose(lp.z, z0, rtol=0, atol=1e-6)


def test_basis_errors_are_explicit():
    lp = (LongProfile().set_x(dx=1000.0, nx=10)
          .set_mn(m=M, n=1.0).set_uplift_rate(U).set_bl(z=0.0))
    with pytest.raises(ValueError, match='basis must be'):
        lp.set_K(K, basis="area")
    # basis="A" with no area
    lp.set_K(K, basis="A")
    with pytest.raises(ValueError, match="call set_A first"):
        _ = lp.driver
    # basis="Q" with no discharge
    lp.set_K(K, basis="Q")
    with pytest.raises(ValueError, match="call set_Q"):
        _ = lp.driver


def test_K_A_from_K_Q_helper():
    assert LongProfile.K_A_from_K_Q(1e-9, 1e-8, 0.5) == pytest.approx(1e-9 * 1e-8**0.5)


def test_runoff_rule_requires_area():
    with pytest.raises(ValueError, match="needs A"):
        LongProfile().set_x(dx=1000.0, nx=5).set_Q(R=1e-8)


# -------------------------------------------------------------- steady state

@pytest.mark.parametrize("n", [1.0, 2.0 / 3.0, 1.5, 2.0])
def test_analytic_steady_state_is_a_fixed_point_of_evolve(n):
    # The integrator and the stepper are separate pieces of code. If the
    # profile moves, one of them is wrong.
    lp = _lp(nx=30, n=n).set_steady_state()
    z0 = lp.z.copy()
    lp.evolve(nt=25, dt=1e11)
    assert np.allclose(lp.z, z0, rtol=0, atol=1e-6)


@pytest.mark.parametrize("n", [1.0, 1.5, 2.0])
def test_steady_state_slope_matches_the_closed_form(n):
    lp = _lp(nx=30, n=n).set_steady_state()
    assert np.allclose(lp.S, lp.steady_state_slope(), rtol=1e-12, atol=0)


def test_relaxes_to_the_analytic_profile_from_flat():
    lp = _lp(nx=25)
    lp.set_z(np.zeros(25))
    lp.evolve(nt=3000, dt=1e11)
    target = _lp(nx=25).set_steady_state().z
    assert np.allclose(lp.z, target, rtol=1e-6, atol=0)


def test_spatially_variable_uplift_steps_the_steady_slope():
    # Two faults. S = (U/(K Q**m))**(1/n), so a step in U is a step in S.
    nx = 60
    lp = _lp(nx=nx)
    U_arr = np.full(nx, U)
    U_arr[: nx // 3] = 3.0 * U
    U_arr[nx // 3: 2 * nx // 3] = 2.0 * U
    lp.set_uplift_rate(U_arr)
    lp.set_z(np.zeros(nx)).evolve(nt=4000, dt=1e11)
    assert np.allclose(lp.S, lp.steady_state_slope(), rtol=1e-6, atol=0)


def test_spatially_variable_K_steps_the_steady_slope():
    nx = 40
    lp = _lp(nx=nx)
    K_arr = np.full(nx, K)
    K_arr[nx // 2:] = 4.0 * K          # soft rock downstream
    lp.set_K(K_arr)
    lp.set_z(np.zeros(nx)).evolve(nt=4000, dt=1e11)
    assert np.allclose(lp.S, lp.steady_state_slope(), rtol=1e-6, atol=0)


# ------------------------------------------------------------------ transient

def test_knickpoint_celerity_matches_the_analytic_prediction():
    # Free prediction: at n = 1 the equation is linear advection of z at
    # C = K Q**m, derived from the governing equation and not from the solver.
    nx, dx = 80, 1000.0
    lp = _lp(nx=nx).set_steady_state()
    lp.set_bl(z=-50.0)

    C = K * 10.0**M
    dt, n_steps = 1e11, 20

    def knick_x():
        return lp.x[int(np.argmax(lp.S))]

    lp.evolve(nt=n_steps, dt=dt)
    x0 = knick_x()
    lp.evolve(nt=n_steps, dt=dt)
    x1 = knick_x()

    assert x1 < x0
    expected = C * dt * n_steps
    assert abs((x0 - x1) - expected) < 1.5 * dx


def test_time_varying_base_level_is_sampled_each_step():
    # A callable z_bl(t) is the way a base-level history is imposed.
    rate = -1e-9                        # 1 mm/yr of base-level fall
    lp = _lp(nx=20).set_steady_state()
    lp.set_z_bl(lambda t: rate * t)

    assert lp.z_bl == 0.0               # t = 0
    lp.evolve(nt=10, dt=1e10)
    assert lp.t == pytest.approx(1e11)
    assert lp.z_bl == pytest.approx(rate * 1e11)
    # the profile followed base level down rather than staying put
    assert lp.z[-1] < _lp(nx=20).set_steady_state().z[-1]


def test_model_time_accumulates():
    lp = _lp(nx=10).set_steady_state()
    assert lp.t == 0.0
    lp.evolve(nt=3, dt=1e9)
    lp.evolve(nt=2, dt=1e9)
    assert lp.t == pytest.approx(5e9)


def test_evolve_requires_dt():
    lp = _lp(nx=10).set_steady_state()
    with pytest.raises(ValueError, match="needs dt"):
        lp.evolve(nt=1)


@pytest.mark.parametrize("field", ["K", "U"])
def test_set_steady_state_integrates_a_stepped_slope_field(field):
    # set_steady_state walks z_i = z_r + S_i dx_i upstream, so it must use the
    # slope of the NODE, not of its receiver. With uniform parameters those
    # are equal and the distinction is invisible -- which is how a mutation
    # swapping them survived the rest of this suite. A step in K or U makes
    # S discontinuous and the two choices differ by a whole cell's offset.
    nx = 40
    lp = _lp(nx=nx)
    arr = np.full(nx, K if field == "K" else U)
    arr[nx // 2:] *= 4.0
    (lp.set_K(arr) if field == "K" else lp.set_uplift_rate(arr))

    lp.set_steady_state()
    # the analytic profile reproduces the closed-form slope node by node
    assert np.allclose(lp.S, lp.steady_state_slope(), rtol=1e-12, atol=0)
    # and it is a fixed point of the transient solver
    z0 = lp.z.copy()
    lp.evolve(nt=20, dt=1e11)
    assert np.allclose(lp.z, z0, rtol=0, atol=1e-6)

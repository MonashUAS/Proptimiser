"""
The propeller as constraints of a CasADi problem: IPOPT gives the loads of the JAX solver, with matching derivatives.
"""
import dataclasses

import aerosandbox as asb
import casadi as cas
import jax
import jax.numpy as jnp
import numpy as np

import init  # noqa: F401  sets float64 before the arrays are built
import propmodel
from casadi_prop import PropellerModel
from init import Inputs, Propeller

THRUST_SCALE_N = 30.0


def _prop(**inputs):
    return Propeller.default(inputs=Inputs(coeff_method=3, n_r=8, V_x=[18.0, 0.8], V_yz=[0.0, 0.0], **inputs))


def _solve_opti(model, omega, V_x, V_yz, warm=True):
    opti = asb.Opti()
    out = model.add_to(opti, omega, V_x, V_yz, THRUST_SCALE_N)
    if warm:
        model.warm_start(opti, out, omega, V_x, V_yz, THRUST_SCALE_N)
    sol = opti.solve(verbose=False)
    return sol(out.thrust_N), sol(out.torque_Nm), sol(out.shaft_power_W)


def test_opti_matches_the_jax_solver():
    model = PropellerModel.from_propeller(_prop(n_psi=12))
    for omega, V_x, V_yz in ((588.0, 18.0, 3.0), (650.0, 0.8, 3.0)):
        T, Q, P = _solve_opti(model, omega, V_x, V_yz)
        T_jax, Q_jax, _ = model.solve_jax(omega, V_x, V_yz, THRUST_SCALE_N)
        assert np.isclose(T, T_jax, rtol=1e-6) and np.isclose(Q, Q_jax, rtol=1e-6)
        assert np.isclose(P, Q * omega, rtol=1e-12)


def test_opti_converges_from_the_default_start():
    # No warm start: the momentum-theory start (lambda = 1) is enough for attached flow
    model = PropellerModel.from_propeller(_prop(n_psi=12))
    T, Q, _ = _solve_opti(model, 588.0, 18.0, 3.0, warm=False)
    T_jax, Q_jax, _ = model.solve_jax(588.0, 18.0, 3.0, THRUST_SCALE_N)
    assert np.isclose(T, T_jax, rtol=1e-6) and np.isclose(Q, Q_jax, rtol=1e-6)


def test_one_azimuth_point_is_exact_in_axial_flight():
    # With V_yz = 0 the flow does not change with azimuth: the 1-point model gives the loads of the 12-point model
    prop = _prop(n_psi=12)
    full, small = PropellerModel.from_propeller(prop), PropellerModel.from_propeller(prop, n_psi=1)
    assert small.params.psi_pt.shape[0] == small.params.r.shape[0]
    for omega, V_x in ((588.0, 18.0), (650.0, 0.8)):
        T_full, Q_full, _ = _solve_opti(full, omega, V_x, 0.0)
        T_small, Q_small, _ = _solve_opti(small, omega, V_x, 0.0)
        assert np.isclose(T_small, T_full, rtol=1e-8) and np.isclose(Q_small, Q_full, rtol=1e-8)


def test_implicit_derivatives_match_jax():
    # d(thrust, torque)/d(omega, V_x) at the BEMT root: CasADi by the implicit function theorem on the residuals,
    # JAX by jax.lax.custom_root
    model = PropellerModel.from_propeller(_prop(n_psi=12))
    theta0 = np.array([588.0, 18.0])
    V_yz = 3.0

    lam_sym, theta = cas.MX.sym("lam", model.n_solved, model.n_aero), cas.MX.sym("theta", 2)
    expr = model.expressions(lam_sym, theta[0], theta[1], V_yz, THRUST_SCALE_N)
    x, g = cas.vec(lam_sym), cas.vec(expr.residual)
    loads = cas.vertcat(expr.thrust_N, expr.torque_Nm)
    dg_dx, dg_dtheta = cas.jacobian(g, x), cas.jacobian(g, theta)
    dloads = cas.jacobian(loads, theta) - cas.mtimes(cas.jacobian(loads, x), cas.solve(dg_dx, dg_dtheta))
    load_derivatives = cas.Function("load_derivatives", [lam_sym, theta], [dloads])

    *_, lam = model.solve_jax(*theta0, V_yz, THRUST_SCALE_N)
    got = np.asarray(load_derivatives(lam, theta0))

    def loads_jax(th):
        params = dataclasses.replace(model.params, omega=th[0], V_x=th[1], V_yz=jnp.asarray(V_yz),
                                     thrust_target=jnp.asarray(THRUST_SCALE_N))
        V_i, _ = propmodel.induced_velocity(params)
        return jnp.stack([propmodel.thrust(V_i, params), propmodel.torque(V_i, params)])
    expected = np.asarray(jax.jacfwd(loads_jax)(jnp.asarray(theta0)))

    np.testing.assert_allclose(got, expected, rtol=1e-5)

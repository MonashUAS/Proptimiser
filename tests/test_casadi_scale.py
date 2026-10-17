"""
The geometric scale of the propeller in a CasADi problem: scale s multiplies every length of the blade (radius, chord,
station spacing) with the shape kept. With s a number it gives the loads of a propeller built at the scaled size. With s
an Opti variable the radius is a design variable.
"""
import dataclasses

import aerosandbox as asb
import numpy as np

from proptimiser import init  # noqa: F401  sets float64 before the arrays are built
from proptimiser.casadi_prop import PropellerModel, static_and_axial
from proptimiser.init import Inputs, Propeller

THRUST_SCALE_N = 30.0
OMEGA, V_X = 330.0, 0.0


def _base():
    prop = Propeller.load("MAD_FLUXER_22x6_6")
    return prop.with_inputs(dataclasses.replace(prop.inputs, n_r=12, n_psi=1))


def _scaled_by_construction(prop, s):
    """A Propeller built at s times the size: radii and chord multiplied, twist and sections kept."""
    blade = dataclasses.replace(prop.blade, r_foils=tuple(s * r for r in prop.blade.r_foils),
                                hub_chord=s * prop.blade.hub_chord)
    return Propeller(prop.inputs, blade, c=s * np.asarray(prop.c), beta=prop.beta, cst=prop.cst)


def _loads(model, omega, V_x, scale=1.0):
    thrust, torque, _ = model.solve(omega, V_x, 0.0, THRUST_SCALE_N, scale=scale)
    return thrust, torque


def test_scale_one_is_the_unscaled_model():
    model = PropellerModel.from_propeller(_base(), n_psi=1)
    T0, Q0, _ = model.solve(OMEGA, V_X, 0.0, THRUST_SCALE_N)
    T1, Q1 = _loads(model, OMEGA, V_X, 1.0)
    assert T1 == T0 and Q1 == Q0


def test_numeric_scale_matches_a_propeller_built_at_that_size():
    base = _base()
    for s in (0.8, 1.25):
        scaled = PropellerModel.from_propeller(_scaled_by_construction(base, s), n_psi=1)
        T_ref, Q_ref = _loads(scaled, OMEGA / s, 8.0)                # a different speed than the base, to test the scale and not one point
        model = PropellerModel.from_propeller(base, n_psi=1)
        T, Q = _loads(model, OMEGA / s, 8.0, scale=s)
        assert np.isclose(T, T_ref, rtol=1e-8) and np.isclose(Q, Q_ref, rtol=1e-8)


def test_thrust_follows_the_scale_laws_in_hover_at_equal_tip_speed():
    # Same tip speed and chord Reynolds number change slowly: T goes as s^2, Q as s^3 within a few percent
    model = PropellerModel.from_propeller(_base(), n_psi=1)
    T1, Q1 = _loads(model, OMEGA, 0.0)
    T2, Q2 = _loads(model, OMEGA / 1.2, 0.0, scale=1.2)
    assert np.isclose(T2 / T1, 1.2**2, rtol=0.05)
    assert np.isclose(Q2 / Q1, 1.2**3, rtol=0.05)


def test_scale_derivatives_match_central_differences():
    # Implicit derivative of the thrust with the scale: dT/ds - dT/dlam (dR/dlam)^-1 dR/ds, at the BEMT root
    import casadi as cas
    model = PropellerModel.from_propeller(_base(), n_psi=1)
    s = cas.MX.sym("s")
    x = cas.MX.sym("lam", model.n_solved * model.n_aero)
    expr = model.expressions(cas.reshape(x, model.n_solved, model.n_aero), OMEGA, V_X, 0.0, THRUST_SCALE_N, scale=s)
    residual = cas.vec(expr.residual)
    dres_dx, dres_ds = cas.jacobian(residual, x), cas.jacobian(residual, s)
    dT = cas.jacobian(expr.thrust_N, s) - cas.mtimes(cas.jacobian(expr.thrust_N, x), cas.solve(dres_dx, dres_ds))
    f = cas.Function("dT_ds", [s, x], [dT])
    s0 = 1.1
    _, _, lam = model.solve(OMEGA, V_X, 0.0, THRUST_SCALE_N, scale=s0)
    analytic = float(f(s0, lam.reshape(-1, order="F")))
    h = 1e-4
    numeric = (_loads(model, OMEGA, V_X, s0 + h)[0] - _loads(model, OMEGA, V_X, s0 - h)[0]) / (2 * h)
    assert np.isclose(analytic, numeric, rtol=1e-4)


def test_opti_hover_with_the_scale_as_a_variable():
    # Fixed rotor speed and thrust: the radius that gives 40 N at 330 rad/s. Opti must find the same scale as the bisection
    model = PropellerModel.from_propeller(_base(), n_psi=1)
    target_N = 40.0
    lo, hi = 0.6, 1.8
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if _loads(model, OMEGA, 0.0, mid)[0] < target_N else (lo, mid)
    expected = 0.5 * (lo + hi)

    opti = asb.Opti()
    s = opti.variable(init_guess=1.0, lower_bound=0.5, upper_bound=2.0)
    out = model.add_to(opti, OMEGA, 0.0, 0.0, target_N, scale=s)
    model.warm_start(opti, out, OMEGA, 0.0, 0.0, target_N, scale=1.0)
    opti.subject_to(out.thrust_N == target_N)
    opti.minimize(out.shaft_power_W)
    sol = opti.solve(verbose=False)
    assert np.isclose(sol(s), expected, rtol=1e-6)


def test_static_and_axial_takes_the_scale():
    prop = _base()
    big = static_and_axial(prop, 3000.0, 0.0, scale=1.2)
    assert np.isclose(big.CT, static_and_axial(_scaled_by_construction(prop, 1.2), 3000.0, 0.0).CT, rtol=1e-8)

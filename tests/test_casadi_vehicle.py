"""
The propeller model inside AeroSandbox vehicle problems: a quadcopter in hover, a fixed wing in cruise and a quadplane
at a transition speed. Each problem solves the BEMT of every rotor together with the vehicle equations.
"""
import aerosandbox as asb
import numpy as np

from proptimiser import init  # noqa: F401  sets float64 before the arrays are built
from proptimiser import propmodel
from proptimiser.casadi_prop import PropellerModel
from proptimiser.init import Inputs, Propeller

G_M_S2 = 9.81


def _omega_for_thrust(model, V_x, V_yz, thrust_N, lo=100.0, hi=1500.0):
    """Rotor speed (rad/s) that gives thrust_N, by bisection on the JAX solver. A start for the Opti variable."""
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        T, _, _ = model.solve_jax(mid, V_x, V_yz, thrust_N)
        lo, hi = (mid, hi) if T < thrust_N else (lo, mid)
    return 0.5 * (lo + hi)


def _motor_power_W(prop, torque_Nm, omega_rad_s):
    i = prop.inputs
    return propmodel.motorPower(torque_Nm, omega_rad_s, I_0=i.I_0, R_m=i.R_m, kV=i.kV)


def test_quadcopter_hover():
    # Four rotors. The optimiser chooses the thrust of each rotor: total thrust = weight, minimum electrical power.
    mass_kg = 6.0
    prop = Propeller.default(inputs=Inputs(coeff_method=3, n_r=8, V_x=[0.0, 0.0], V_yz=[0.0, 0.0]), beta_deg=15.0)
    model = PropellerModel.from_propeller(prop, n_psi=1)
    thrust_each_N = mass_kg * G_M_S2 / 4
    omega0 = _omega_for_thrust(model, 0.0, 0.0, thrust_each_N)

    opti = asb.Opti()
    omegas = [opti.variable(init_guess=omega0, lower_bound=100.0, upper_bound=1500.0) for _ in range(4)]
    rotors = [model.add_to(opti, w, 0.0, 0.0, thrust_each_N) for w in omegas]
    for w, rotor in zip(omegas, rotors):
        model.warm_start(opti, rotor, omega0, 0.0, 0.0, thrust_each_N)
    opti.subject_to(sum(r.thrust_N for r in rotors) == mass_kg * G_M_S2)
    opti.minimize(sum(_motor_power_W(prop, r.torque_Nm, w) for w, r in zip(omegas, rotors)))
    sol = opti.solve(verbose=False)

    omega = np.array([sol(w) for w in omegas])
    thrust = np.array([sol(r.thrust_N) for r in rotors])
    assert np.isclose(thrust.sum(), mass_kg * G_M_S2, rtol=1e-8)
    assert np.allclose(thrust, thrust_each_N, rtol=1e-3)                # equal rotors share the weight
    assert np.allclose(omega, omega0, rtol=1e-3)
    T_jax, _, _ = model.solve_jax(omega[0], 0.0, 0.0, thrust_each_N)    # the BEMT root is the JAX root
    assert np.isclose(T_jax, thrust[0], rtol=1e-6)


def _airplane(span_m=1.8):
    airfoil = asb.Airfoil("naca2412")
    wing = asb.Wing(name="Wing", symmetric=True, xsecs=[
        asb.WingXSec(xyz_le=[0, 0, 0], chord=0.25, airfoil=airfoil),
        asb.WingXSec(xyz_le=[0.03, span_m / 2, 0], chord=0.18, airfoil=airfoil),
    ])
    return asb.Airplane(name="uav", wings=[wing], xyz_ref=[0.05, 0, 0])


def test_fixed_wing_cruise_trim():
    # Level flight at 18 m/s: lift = weight, propeller thrust = drag. The optimiser picks the angle of attack for
    # minimum shaft power, and the rotor speed follows from the thrust constraint.
    mass_kg, V_m_s = 4.0, 18.0
    prop = Propeller.default(inputs=Inputs(coeff_method=3, n_r=8, V_x=[V_m_s, V_m_s], V_yz=[0.0, 0.0]), beta_deg=25.0)
    model = PropellerModel.from_propeller(prop, n_psi=1)
    airplane = _airplane()

    opti = asb.Opti()
    alpha_deg = opti.variable(init_guess=4.0, lower_bound=-2.0, upper_bound=10.0)
    omega = opti.variable(init_guess=300.0, lower_bound=100.0, upper_bound=1500.0)
    aero = asb.AeroBuildup(airplane, asb.OperatingPoint(velocity=V_m_s, alpha=alpha_deg)).run()
    rotor = model.add_to(opti, omega, V_m_s, 0.0, 5.0)
    opti.subject_to([aero["L"] == mass_kg * G_M_S2, rotor.thrust_N == aero["D"]])
    opti.minimize(rotor.shaft_power_W)
    sol = opti.solve(verbose=False)

    assert np.isclose(sol(aero["L"]), mass_kg * G_M_S2, rtol=1e-8)
    assert np.isclose(sol(rotor.thrust_N), sol(aero["D"]), rtol=1e-8)
    assert -2.0 < sol(alpha_deg) < 10.0 and 100.0 < sol(omega) < 1500.0 and sol(rotor.shaft_power_W) > 0.0
    T_jax, Q_jax, _ = model.solve_jax(sol(omega), V_m_s, 0.0, 5.0)      # the BEMT root is the JAX root
    assert np.isclose(T_jax, sol(rotor.thrust_N), rtol=1e-6) and np.isclose(Q_jax, sol(rotor.torque_Nm), rtol=1e-6)


def test_quadplane_transition_point():
    # Four lift rotors in edgewise flow (disc horizontal, V_yz = forward speed) and a pusher in axial flow. The wing
    # carries part of the weight. Minimise the total shaft power over the angle of attack and the three rotor speeds.
    mass_kg, V_m_s = 5.0, 10.0
    weight_N = mass_kg * G_M_S2
    lift_prop = Propeller.default(inputs=Inputs(coeff_method=3, n_r=8, n_psi=12, inflow_model=1, V_x=[0.0, 0.0],
                                                V_yz=[V_m_s, V_m_s]), beta_deg=15.0)
    push_prop = Propeller.default(inputs=Inputs(coeff_method=3, n_r=8, V_x=[V_m_s, V_m_s], V_yz=[0.0, 0.0]), beta_deg=25.0)
    lift, push = PropellerModel.from_propeller(lift_prop), PropellerModel.from_propeller(push_prop, n_psi=1)
    airplane = _airplane(span_m=2.0)

    lift_each_N = 0.5 * weight_N / 4
    omega_lift0 = _omega_for_thrust(lift, 0.0, V_m_s, lift_each_N)
    omega_push0 = _omega_for_thrust(push, V_m_s, 0.0, 4.0)

    opti = asb.Opti()
    alpha_deg = opti.variable(init_guess=5.0, lower_bound=-2.0, upper_bound=12.0)
    omega_lift = opti.variable(init_guess=omega_lift0, lower_bound=100.0, upper_bound=1500.0)
    omega_push = opti.variable(init_guess=omega_push0, lower_bound=100.0, upper_bound=1500.0)
    aero = asb.AeroBuildup(airplane, asb.OperatingPoint(velocity=V_m_s, alpha=alpha_deg)).run()
    rotor_lift = lift.add_to(opti, omega_lift, 0.0, V_m_s, lift_each_N)
    rotor_push = push.add_to(opti, omega_push, V_m_s, 0.0, 4.0)
    lift.warm_start(opti, rotor_lift, omega_lift0, 0.0, V_m_s, lift_each_N)
    push.warm_start(opti, rotor_push, omega_push0, V_m_s, 0.0, 4.0)
    opti.subject_to([4 * rotor_lift.thrust_N + aero["L"] == weight_N, rotor_push.thrust_N == aero["D"]])
    opti.minimize(4 * rotor_lift.shaft_power_W + rotor_push.shaft_power_W)
    sol = opti.solve(verbose=False)

    assert np.isclose(4 * sol(rotor_lift.thrust_N) + sol(aero["L"]), weight_N, rtol=1e-8)
    assert np.isclose(sol(rotor_push.thrust_N), sol(aero["D"]), rtol=1e-8)
    state = dict(alpha_deg=sol(alpha_deg), omega_lift=sol(omega_lift), omega_push=sol(omega_push), lift_each_N=sol(rotor_lift.thrust_N),
                 push_N=sol(rotor_push.thrust_N), wing_lift_N=sol(aero["L"]), drag_N=sol(aero["D"]))
    # The wing works at its angle-of-attack limit (12 deg) and the lift rotors carry the rest of the weight
    assert np.isclose(state["alpha_deg"], 12.0, atol=1e-4), state
    assert all(100.0 < state[k] < 1500.0 for k in ("omega_lift", "omega_push")) and 4 * state["lift_each_N"] > 0.2 * weight_N, state
    T_jax, _, _ = lift.solve_jax(sol(omega_lift), 0.0, V_m_s, lift_each_N)
    assert np.isclose(T_jax, sol(rotor_lift.thrust_N), rtol=1e-5)

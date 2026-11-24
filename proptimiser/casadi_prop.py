"""
CasADi interface of the propeller model.

The BEMT physics (airflow.py, airfoil.py, propmodel.py) runs on CasADi expressions when Params.cfg.backend is
"casadi" (backend.py). The geometry of the propeller is then a set of NumPy constants, and the flight state
(omega, V_x, V_yz, ...) can be CasADi symbols or Opti variables.

PropellerModel adds a solved propeller to an AeroSandbox (or CasADi) Opti problem. The induced velocity is not solved
by a loop: its non-dimensional ratios are Opti variables and the BEMT residuals are equality constraints, so IPOPT
solves the BEMT together with the vehicle problem and takes exact derivatives through it.

Units: SI. Angles are radians. The name of each value gives its unit.
"""
import dataclasses
from typing import NamedTuple

import casadi as cas
import numpy as np

from . import propmodel
from ._jax import HAS_JAX, jnp, require_jax


def casadi_params(params, **symbols):
    """
    Params with NumPy constants and the CasADi backend, from the Params of one flight condition.
    Each keyword replaces a field with a CasADi symbol or a number, e.g. omega=opti.variable().
    """
    cfg = dataclasses.replace(params.cfg, backend="casadi")
    static = {"cfg", "airfoil_interp"}
    constants = {f.name: np.asarray(getattr(params, f.name)) for f in dataclasses.fields(params)
                 if f.name not in static}
    return dataclasses.replace(params, **{**constants, **symbols, "cfg": cfg})


class PropExpressions(NamedTuple):
    residual: cas.MX            # scaled BEMT residuals (k, n_aero): zero at the solution
    thrust_N: cas.MX            # thrust along the shaft axis
    torque_Nm: cas.MX           # shaft torque
    shaft_power_W: cas.MX       # torque * omega


class PropOutputs(NamedTuple):
    thrust_N: cas.MX
    torque_Nm: cas.MX
    shaft_power_W: cas.MX
    lam: cas.MX                 # induced velocity over its scale, (k, n_aero): a view of the Opti variable
    variable: cas.MX            # the Opti variable itself, (k * n_aero, 1), column-major


class PropellerModel:
    """
    A propeller with fixed geometry (chord, twist and sections of a Propeller), ready for a CasADi problem.

    The aerodynamic zone is solved (propmodel.aero_zone). The blend zone inboard of it carries no force and is held at
    the momentum-theory seed, as in the JAX solver.
    """
    def __init__(self, prop):
        self.prop = prop
        self.params = prop.condition(0)
        self.cfg = self.params.cfg
        self._base = casadi_params(self.params)
        self.n_solved = len(self.cfg.solve_idx)
        aero = np.asarray(self.params.r) >= self.cfg.R_aero
        self.n_aero = int(aero.sum())
        n_r = aero.size
        # E (n_aero, n_r) places the solved stations in the full station grid; the blend zone stays at lambda = 1
        self._solver = None
        self._embed = np.eye(n_r)[aero]
        self._blend_seed = np.ones((self.n_solved, n_r)) * (~aero)

    @classmethod
    def from_propeller(cls, prop, n_psi=None):
        """
        n_psi is the number of azimuth points of the model. Use 1 for axial flight (V_yz = 0, no tilt): the flow does
        not change with azimuth, and the model is 12 times smaller than with the default grid. n_psi = 1 ignores any
        in-plane inflow.
        """
        if n_psi is not None and n_psi != prop.inputs.n_psi:
            prop = prop.with_inputs(dataclasses.replace(prop.inputs, n_psi=n_psi))
        return cls(prop)

    @staticmethod
    def _flight(omega_rad_s, V_x_m_s, V_yz_m_s, thrust_scale_N, rho_kg_m3, mu_Pa_s, a_m_s, scale=1.0):
        """The Params fields of a flight state. Air properties left as None keep the sea-level defaults."""
        fields = dict(omega=omega_rad_s, V_x=V_x_m_s, V_yz=V_yz_m_s, thrust_target=thrust_scale_N, scale=scale)
        air = dict(rho=rho_kg_m3, mu=mu_Pa_s, a=a_m_s)
        return {**fields, **{name: value for name, value in air.items() if value is not None}}

    def expressions(self, lam, omega_rad_s, V_x_m_s, V_yz_m_s, thrust_scale_N, rho_kg_m3=None, mu_Pa_s=None,
                    a_m_s=None, scale=1.0) -> PropExpressions:
        """
        Residuals and loads for the inflow ratios lam (k, n_aero) and one flight state. The flight state values are
        numbers or CasADi expressions. thrust_scale_N only scales the unknowns and the residuals (it is not a
        constraint): use the thrust you expect. rho, mu and a default to sea-level air (1.225 kg/m^3, 1.798e-5 Pa s,
        343 m/s). scale multiplies every length of the blade (radius, chord, station spacing) with the shape kept:
        the radius is scale * R_tip of the propeller. It is a number or a CasADi expression (an Opti variable).
        """
        params = dataclasses.replace(self._base, **self._flight(omega_rad_s, V_x_m_s, V_yz_m_s, thrust_scale_N,
                                                                rho_kg_m3, mu_Pa_s, a_m_s, scale))
        lam_full = cas.mtimes(lam, self._embed) + self._blend_seed
        V_i = propmodel.inflow_from_lambda(lam_full, params)
        residual = cas.mtimes(propmodel.bemt_residuals(lam_full, params), self._embed.T)
        torque = propmodel.torque(V_i, params)
        return PropExpressions(residual, propmodel.thrust(V_i, params), torque, torque * omega_rad_s)

    def add_to(self, opti, omega_rad_s, V_x_m_s, V_yz_m_s, thrust_scale_N, rho_kg_m3=None, mu_Pa_s=None,
               a_m_s=None, scale=1.0) -> PropOutputs:
        """
        Adds the propeller to opti at one flight state: the inflow ratios as variables and the BEMT residuals as
        equality constraints. Call it once for each rotor and flight state. Returns the loads as expressions.
        scale is the geometric scale of the blade (see expressions): an Opti variable makes the radius a design variable.
        """
        flat = opti.variable(init_guess=np.ones(self.n_solved * self.n_aero))
        lam = cas.reshape(flat, self.n_solved, self.n_aero)
        expr = self.expressions(lam, omega_rad_s, V_x_m_s, V_yz_m_s, thrust_scale_N, rho_kg_m3, mu_Pa_s, a_m_s, scale)
        opti.subject_to(cas.vec(expr.residual) == 0)
        return PropOutputs(expr.thrust_N, expr.torque_Nm, expr.shaft_power_W, lam, flat)

    def solve(self, omega_rad_s, V_x_m_s, V_yz_m_s, thrust_scale_N, rho_kg_m3=None, mu_Pa_s=None, a_m_s=None,
              tol=1e-10, max_iter=100, scale=1.0):
        """
        The BEMT solution at a numeric flight state, by damped Newton iteration on the CasADi residuals (no JAX).
        Returns (thrust_N, torque_Nm, lam) with lam of shape (k, n_aero), the values that start an Opti variable.
        Raises RuntimeError when the iteration does not converge.
        """
        air = [self._base.rho if rho_kg_m3 is None else rho_kg_m3, self._base.mu if mu_Pa_s is None else mu_Pa_s,
               self._base.a if a_m_s is None else a_m_s]
        values = np.array([omega_rad_s, V_x_m_s, V_yz_m_s, thrust_scale_N, *air, scale], dtype=float)
        if self._solver is None:
            flat = cas.MX.sym("lam", self.n_solved * self.n_aero)
            state = cas.MX.sym("state", 8)
            expr = self.expressions(cas.reshape(flat, self.n_solved, self.n_aero), state[0], state[1], state[2],
                                    state[3], state[4], state[5], state[6], state[7])
            residual = cas.vec(expr.residual)
            self._solver = (cas.Function("residual", [flat, state], [residual]),
                            cas.Function("jacobian", [flat, state], [cas.jacobian(residual, flat)]),
                            cas.Function("loads", [flat, state], [expr.thrust_N, expr.torque_Nm]))
        residual, jacobian, loads = self._solver
        norm = lambda x: float(np.max(np.abs(np.asarray(residual(x, values)))))
        x, error = np.ones(self.n_solved * self.n_aero), None
        error = norm(x)
        for _ in range(max_iter):
            if error < tol:
                break
            step = np.linalg.solve(np.asarray(jacobian(x, values)), -np.asarray(residual(x, values)).ravel())
            alpha = 1.0
            while alpha > 1e-4:
                trial = x + alpha * step
                trial_error = norm(trial)
                if np.isfinite(trial_error) and trial_error < error:
                    break
                alpha *= 0.5
            else:
                raise RuntimeError(f"BEMT Newton iteration stalled at residual {error:.2e}")
            x, error = trial, trial_error
        if error >= tol:
            raise RuntimeError(f"BEMT Newton iteration did not converge: residual {error:.2e}")
        thrust, torque = loads(x, values)
        return float(thrust), float(torque), x.reshape(self.n_solved, self.n_aero, order="F")

    def solve_jax(self, omega_rad_s, V_x_m_s, V_yz_m_s, thrust_scale_N, rho_kg_m3=None, mu_Pa_s=None, a_m_s=None,
                  scale=1.0):
        """
        The same solution by the JAX solver (Newton with a bracketed fallback). Needs JAX. Used to check solve().
        """
        require_jax("PropellerModel.solve_jax")
        fields = self._flight(omega_rad_s, V_x_m_s, V_yz_m_s, thrust_scale_N, rho_kg_m3, mu_Pa_s, a_m_s, scale)
        params = dataclasses.replace(self.params, **{k: jnp.asarray(v, dtype=float) for k, v in fields.items()})
        V_i, V_i_0 = propmodel.induced_velocity(params)
        lam = V_i_0[self.cfg.solve_rows] / propmodel.get_velocity_scales(params)
        aero = np.asarray(self.params.r) >= self.cfg.R_aero
        return (float(propmodel.thrust(V_i, params)), float(propmodel.torque(V_i, params)),
                np.asarray(lam)[:, aero])

    def warm_start(self, opti, outputs, omega_rad_s, V_x_m_s, V_yz_m_s, thrust_scale_N, rho_kg_m3=None,
                   mu_Pa_s=None, a_m_s=None, scale=1.0):
        """
        Starts the inflow variables of outputs (from add_to) at the Newton solution for numeric guesses of the flight
        state. This keeps IPOPT on the attached-flow root of the BEMT, and saves iterations.
        """
        *_, lam = self.solve(omega_rad_s, V_x_m_s, V_yz_m_s, thrust_scale_N, rho_kg_m3, mu_Pa_s, a_m_s, scale=scale)
        opti.set_initial(outputs.variable, lam.reshape(-1, order="F"))


## Static and axial-flow loads for one rotor (live call from a vehicle model)

class RotorLoads(NamedTuple):
    thrust_N: float
    torque_Nm: float
    shaft_power_W: float        # torque * omega
    CT: float                   # T / (rho n^2 D^4), n in rev/s
    CP: float                   # P / (rho n^3 D^5)
    J: float                    # V_axial / (n D)


_MODEL_CACHE = {}


def coefficients(thrust_N, torque_Nm, rpm, v_axial_m_s, rho_kg_m3, diameter_m):
    """RotorLoads from the thrust and torque of a rotor. rpm must be above zero."""
    if rpm <= 0.0:
        raise ValueError(f"rpm must be above zero, got {rpm}")
    n = rpm / 60.0
    omega = rpm * np.pi / 30.0
    power = torque_Nm * omega
    return RotorLoads(thrust_N, torque_Nm, power, thrust_N / (rho_kg_m3 * n**2 * diameter_m**4),
                      power / (rho_kg_m3 * n**3 * diameter_m**5), v_axial_m_s / (n * diameter_m))


def static_and_axial(state, rpm, v_axial, rho=1.225, scale=1.0):
    """
    Thrust, torque, shaft power, CT and CP of one rotor at a given rpm and axial inflow speed. It uses the CasADi
    path (damped Newton on the BEMT residuals, no JAX).

    state    a Propeller (init.Propeller.load("MAD_FLUXER_22x6_6")), or a PropellerModel. The design YAML and the
             state JSON of the propeller give the geometry. The model is built once for each Propeller and kept.
    rpm      shaft speed (rev/min), above zero.
    v_axial  axial inflow speed along the shaft (m/s). Zero is static.
    rho      air density (kg/m^3). The viscosity and the speed of sound keep the sea-level values.
    scale    geometric scale of the blade (number): the diameter is 2 * scale * R_tip.

    The model uses one azimuth point, which is exact in axial flow. Raises RuntimeError when the BEMT does not
    converge. Returns RotorLoads (SI units).
    """
    if isinstance(state, PropellerModel):
        model = state
    else:
        model = _MODEL_CACHE.get(id(state))
        if model is None or model[0] is not state:
            model = (state, PropellerModel.from_propeller(state, n_psi=1))
            _MODEL_CACHE[id(state)] = model
        model = model[1]
    diameter = 2.0 * scale * float(model.prop.blade.R_tip)
    omega = rpm * np.pi / 30.0
    thrust_scale = max(0.05 * rho * (rpm / 60.0) ** 2 * diameter**4, 0.1)      # only scales the unknowns
    thrust, torque, _ = model.solve(omega, v_axial, 0.0, thrust_scale, rho_kg_m3=rho, scale=scale)
    return coefficients(thrust, torque, rpm, v_axial, rho, diameter)

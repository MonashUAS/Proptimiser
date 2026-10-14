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
import jax
import jax.numpy as jnp
import numpy as np

from . import propmodel


def casadi_params(params, **symbols):
    """
    Params with NumPy constants and the CasADi backend, from the Params of one flight condition.
    Each keyword replaces a field with a CasADi symbol or a number, e.g. omega=opti.variable().
    """
    cfg = dataclasses.replace(params.cfg, backend="casadi")
    constants = jax.tree_util.tree_map(np.asarray, dataclasses.replace(params, cfg=cfg))
    return dataclasses.replace(constants, **symbols)


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
    def _flight(omega_rad_s, V_x_m_s, V_yz_m_s, thrust_scale_N, rho_kg_m3, mu_Pa_s, a_m_s):
        """The Params fields of a flight state. Air properties left as None keep the sea-level defaults."""
        fields = dict(omega=omega_rad_s, V_x=V_x_m_s, V_yz=V_yz_m_s, thrust_target=thrust_scale_N)
        air = dict(rho=rho_kg_m3, mu=mu_Pa_s, a=a_m_s)
        return {**fields, **{name: value for name, value in air.items() if value is not None}}

    def expressions(self, lam, omega_rad_s, V_x_m_s, V_yz_m_s, thrust_scale_N, rho_kg_m3=None, mu_Pa_s=None,
                    a_m_s=None) -> PropExpressions:
        """
        Residuals and loads for the inflow ratios lam (k, n_aero) and one flight state. The flight state values are
        numbers or CasADi expressions. thrust_scale_N only scales the unknowns and the residuals (it is not a
        constraint): use the thrust you expect. rho, mu and a default to sea-level air (1.225 kg/m^3, 1.798e-5 Pa s,
        343 m/s).
        """
        params = dataclasses.replace(self._base, **self._flight(omega_rad_s, V_x_m_s, V_yz_m_s, thrust_scale_N,
                                                                rho_kg_m3, mu_Pa_s, a_m_s))
        lam_full = cas.mtimes(lam, self._embed) + self._blend_seed
        V_i = propmodel.inflow_from_lambda(lam_full, params)
        residual = cas.mtimes(propmodel.bemt_residuals(lam_full, params), self._embed.T)
        torque = propmodel.torque(V_i, params)
        return PropExpressions(residual, propmodel.thrust(V_i, params), torque, torque * omega_rad_s)

    def add_to(self, opti, omega_rad_s, V_x_m_s, V_yz_m_s, thrust_scale_N, rho_kg_m3=None, mu_Pa_s=None,
               a_m_s=None) -> PropOutputs:
        """
        Adds the propeller to opti at one flight state: the inflow ratios as variables and the BEMT residuals as
        equality constraints. Call it once for each rotor and flight state. Returns the loads as expressions.
        """
        flat = opti.variable(init_guess=np.ones(self.n_solved * self.n_aero))
        lam = cas.reshape(flat, self.n_solved, self.n_aero)
        expr = self.expressions(lam, omega_rad_s, V_x_m_s, V_yz_m_s, thrust_scale_N, rho_kg_m3, mu_Pa_s, a_m_s)
        opti.subject_to(cas.vec(expr.residual) == 0)
        return PropOutputs(expr.thrust_N, expr.torque_Nm, expr.shaft_power_W, lam, flat)

    def solve_jax(self, omega_rad_s, V_x_m_s, V_yz_m_s, thrust_scale_N, rho_kg_m3=None, mu_Pa_s=None, a_m_s=None):
        """
        The BEMT solution at a numeric flight state by the JAX solver (Newton with a bracketed fallback).
        Returns (thrust_N, torque_Nm, lam) with lam of shape (k, n_aero), the values that start an Opti variable.
        """
        fields = self._flight(omega_rad_s, V_x_m_s, V_yz_m_s, thrust_scale_N, rho_kg_m3, mu_Pa_s, a_m_s)
        params = dataclasses.replace(self.params, **{k: jnp.asarray(v, dtype=float) for k, v in fields.items()})
        V_i, V_i_0 = propmodel.induced_velocity(params)
        lam = V_i_0[self.cfg.solve_rows] / propmodel.get_velocity_scales(params)
        aero = np.asarray(self.params.r) >= self.cfg.R_aero
        return (float(propmodel.thrust(V_i, params)), float(propmodel.torque(V_i, params)),
                np.asarray(lam)[:, aero])

    def warm_start(self, opti, outputs, omega_rad_s, V_x_m_s, V_yz_m_s, thrust_scale_N, rho_kg_m3=None,
                   mu_Pa_s=None, a_m_s=None):
        """
        Starts the inflow variables of outputs (from add_to) at the JAX solution for numeric guesses of the flight
        state. This keeps IPOPT on the attached-flow root of the BEMT, and saves iterations.
        """
        *_, lam = self.solve_jax(omega_rad_s, V_x_m_s, V_yz_m_s, thrust_scale_N, rho_kg_m3, mu_Pa_s, a_m_s)
        opti.set_initial(outputs.variable, lam.reshape(-1, order="F"))

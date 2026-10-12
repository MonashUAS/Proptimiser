"""
The BEMT physics gives the same numbers on the JAX and the CasADi backend.
"""
import dataclasses

import casadi as cas
import numpy as np
import pytest

import init  # noqa: F401  sets float64 before the arrays are built
import airfoil
import propmodel
from casadi_prop import casadi_params
from init import BladeSpec, Inputs, Propeller

import jax.numpy as jnp


def _outputs(params):
    """Residuals, thrust, torque and coefficients of the inflow ratios lam, on either backend."""
    def f(lam):
        V_i = propmodel.inflow_from_lambda(lam, params)
        return (propmodel.bemt_residuals(lam, params), propmodel.thrust(V_i, params), propmodel.torque(V_i, params),
                propmodel.dT_blade(V_i, params), *airfoil.aero_coeffs(V_i, params))
    return f


def _compare(prop, idx, rtol=1e-9):
    params = prop.condition(idx)
    k, n_r = len(params.cfg.solve_idx), params.r.shape[0]
    rng = np.random.default_rng(idx)
    lam = 1.0 + 0.3 * rng.uniform(-1.0, 1.0, (k, n_r))

    expected = [np.asarray(x) for x in _outputs(params)(jnp.asarray(lam))]

    lam_sym = cas.MX.sym("lam", k, n_r)
    omega, V_x, V_yz = cas.MX.sym("omega"), cas.MX.sym("V_x"), cas.MX.sym("V_yz")
    symbols = casadi_params(params, omega=omega, V_x=V_x, V_yz=V_yz)
    function = cas.Function("f", [lam_sym, omega, V_x, V_yz], list(_outputs(symbols)(lam_sym)))
    got = [np.asarray(x) for x in function(lam, float(params.omega), float(params.V_x), float(params.V_yz))]

    for name, a, b in zip(("residual", "T", "Q", "dT_blade", "CL", "CD", "CM"), expected, got):
        np.testing.assert_allclose(b.reshape(a.shape), a, rtol=rtol, atol=1e-12, err_msg=name)


@pytest.mark.parametrize("idx", [0, 1])
def test_parametric_section_model(idx):
    prop = Propeller.default(inputs=Inputs(coeff_method=0, n_r=6, n_psi=8, V_yz=[0.0, 3.0]))
    _compare(prop, idx)


@pytest.mark.parametrize("idx", [0, 1])
def test_neuralfoil_with_loss_models_and_skewed_inflow(idx):
    inputs = Inputs(coeff_method=3, n_r=6, n_psi=8, V_yz=[0.0, 3.0], inflow_model=1, tip_loss_model=1,
                    root_loss_model=1, rot_corr_model=1)
    _compare(Propeller.default(inputs=inputs), idx)


def test_swept_blade():
    base = BladeSpec()
    blade = dataclasses.replace(base, offset_r=(base.R_hub, base.R_tip), offset_y=(0.0, -0.01),
                                offset_z=(0.0, 0.02))
    _compare(Propeller.default(inputs=Inputs(coeff_method=0, n_r=6, n_psi=8, V_yz=[2.0, 3.0]), blade=blade), 0)

"""
Inputs.loss_in_solve selects where the Prandtl factors apply: 0 after the BEMT solve, 1 in the momentum side of the
residuals.

The reference numbers come from the code before the flag existed. The "off" numbers are that code as it was. The "on"
numbers are that code with a run-time patch that calls dT_momentum and dQ_momentum with the factor in the residuals.

    python -m pytest -q tests/test_loss_in_solve.py
"""
import dataclasses
from pathlib import Path

import casadi as cas
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from proptimiser import init, propmodel
from proptimiser.casadi_prop import PropellerModel, casadi_params
from proptimiser.init import Inputs, Propeller

# (case, flight condition): (thrust N, torque Nm)
REFERENCE_OFF = {
    ("parametric", 0): (38.690905188908346, 3.3999696962812855),
    ("parametric", 1): (79.12676685593559, 6.39439037266924),
    ("neuralfoil", 0): (19.98555174730907, 0.9324793225041907),
    ("neuralfoil", 1): (64.01973307057189, 1.4747951299152728),
}
REFERENCE_ON = {
    ("parametric", 0): (46.284017338701666, 4.191255201018126),
    ("parametric", 1): (85.92779769478564, 7.082509890038858),
    ("neuralfoil", 0): (23.941183823117136, 1.1692682678356179),
    ("neuralfoil", 1): (68.57237681992368, 1.6641184789221206),
}
CASES = list(REFERENCE_OFF)


def _prop(case, loss_in_solve=0, **changes):
    methods = dict(parametric=dict(coeff_method=0), neuralfoil=dict(coeff_method=3, rot_corr_model=1))
    values = dict(n_r=10, n_psi=6, inflow_model=1, tip_loss_model=1, root_loss_model=1, V_yz=[0.0, 3.0],
                  loss_in_solve=loss_in_solve)
    return Propeller.default(inputs=Inputs(**{**values, **methods[case], **changes}))


def _loads(params):
    V_i, _ = propmodel.induced_velocity(params)
    return float(jnp.squeeze(propmodel.thrust(V_i, params))), float(jnp.squeeze(propmodel.torque(V_i, params)))


@pytest.mark.parametrize("case, idx", CASES)
def test_flag_off_gives_the_results_of_the_code_before_the_flag(case, idx):
    assert _loads(_prop(case).condition(idx)) == pytest.approx(REFERENCE_OFF[case, idx], rel=1e-9)


@pytest.mark.parametrize("case, idx", CASES)
def test_flag_on_matches_the_patched_reference(case, idx):
    assert _loads(_prop(case, 1).condition(idx)) == pytest.approx(REFERENCE_ON[case, idx], rel=1e-6)


def test_flag_on_solves_the_residuals_with_the_factor():
    params = _prop("neuralfoil", 1).condition(0)
    V_i, V_i_0 = propmodel.induced_velocity(params)
    Q_ref = params.thrust_target * params.cfg.R_tip
    aero = np.asarray(propmodel.aero_zone(params))
    dT = propmodel.dT_blade(V_i, params) - propmodel.dT_momentum(V_i, params, apply_tip_loss=True)
    dQ = propmodel.dQ_blade(V_i, params) - propmodel.dQ_momentum(V_i, params, apply_tip_loss=True)
    assert np.abs(np.asarray(dT / params.thrust_target))[aero].max() <= params.cfg.V_i_0_eps
    assert np.abs(np.asarray(dQ / Q_ref))[aero].max() <= params.cfg.V_i_0_eps
    # At the root of the residuals the two sides agree, so the blade element thrust equals the reported thrust
    weights = propmodel.radial_weights(params)
    assert float(jnp.sum(propmodel.dT_blade(V_i, params) * weights)) == pytest.approx(
        float(jnp.squeeze(propmodel.thrust(V_i, params))), rel=1e-6)


def test_tip_factor_only_is_the_root_model_set_to_none():
    params = _prop("parametric", 1, root_loss_model=0).condition(0)
    V_i, _ = propmodel.induced_velocity(params)
    _, phi = propmodel.resultantVelocityComponents(V_i, params)
    tip, root = propmodel.tip_root_loss_factors(phi, params)
    assert np.all(np.asarray(root) == 1.0) and np.asarray(tip).min() < 0.9


@pytest.mark.parametrize("case", ["parametric", "neuralfoil"])
def test_gradient_with_the_factor_in_the_solve_matches_finite_differences(case):
    prop = _prop(case, 1, V_i_0_eps=1e-12)
    params = prop.condition(1)
    start = propmodel.induced_velocity(params)[1]

    def loads(beta, chord):
        p = dataclasses.replace(params, beta=beta, c=chord, V_i_0=start)
        V_i, _ = propmodel.induced_velocity(p)
        return jnp.stack([jnp.squeeze(propmodel.thrust(V_i, p)), jnp.squeeze(propmodel.torque(V_i, p))])

    values = jax.jit(loads)
    jacobians = jax.jit(jax.jacrev(loads, argnums=(0, 1)))(params.beta, params.c)
    rng = np.random.default_rng(0)
    d_beta = jnp.asarray(rng.standard_normal(params.beta.shape))
    d_chord = jnp.asarray(rng.standard_normal(params.c.shape)) * float(jnp.mean(params.c))
    exact = np.asarray(jacobians[0] @ d_beta + jacobians[1] @ d_chord)
    h = 1e-5
    plus = values(params.beta + h * d_beta, params.c + h * d_chord)
    minus = values(params.beta - h * d_beta, params.c - h * d_chord)
    np.testing.assert_allclose(exact, np.asarray(plus - minus) / (2 * h), rtol=1e-5)


def test_post_processing_gives_the_loads_of_the_flag_and_keeps_the_input():
    params = _prop("neuralfoil").condition(0)
    loads = propmodel.loads_with_loss_in_solve(params)
    assert (float(loads.thrust_N), float(loads.torque_Nm)) == pytest.approx(REFERENCE_ON["neuralfoil", 0], rel=1e-6)
    assert float(loads.residual_nd) <= params.cfg.V_i_0_eps
    assert params.cfg.loss_in_solve == 0
    assert _loads(params) == pytest.approx(REFERENCE_OFF["neuralfoil", 0], rel=1e-9)


@pytest.mark.parametrize("idx", [0, 1])
def test_casadi_residuals_and_loads_match_jax_with_the_factor_in_the_solve(idx):
    params = _prop("neuralfoil", 1, n_psi=8).condition(idx)
    k, n_r = len(params.cfg.solve_idx), params.r.shape[0]
    lam = 1.0 + 0.3 * np.random.default_rng(idx).uniform(-1.0, 1.0, (k, n_r))

    def outputs(p, x):
        V_i = propmodel.inflow_from_lambda(x, p)
        return propmodel.bemt_residuals(x, p), propmodel.thrust(V_i, p), propmodel.torque(V_i, p)

    expected = [np.asarray(x) for x in outputs(params, jnp.asarray(lam))]
    lam_sym = cas.MX.sym("lam", k, n_r)
    function = cas.Function("f", [lam_sym], list(outputs(casadi_params(params), lam_sym)))
    got = [np.asarray(x) for x in function(lam)]
    for a, b in zip(expected, got):
        np.testing.assert_allclose(b.reshape(a.shape), a, rtol=1e-9, atol=1e-12)


def test_casadi_newton_solve_matches_jax_with_the_factor_in_the_solve():
    model = PropellerModel.from_propeller(_prop("parametric", 1), n_psi=1)
    thrust, torque, _ = model.solve(588.0, 18.0, 0.0, 40.0)
    thrust_jax, torque_jax, _ = model.solve_jax(588.0, 18.0, 0.0, 40.0)
    assert (thrust, torque) == pytest.approx((thrust_jax, torque_jax), rel=1e-6)


def test_each_saved_design_holds_the_key_at_zero():
    paths = sorted((Path(__file__).resolve().parents[1] / "designs").glob("*.yaml"))
    assert paths
    for path in paths:
        assert init.load_design(path)[0].loss_in_solve == 0, path.name


def test_design_without_the_key_stops_the_load_and_a_bad_value_is_an_error():
    inputs, blade = _prop("parametric").inputs, _prop("parametric").blade
    data = init.design_to_dict(inputs, blade)
    assert data["inputs"]["loss_in_solve"] == 0
    del data["inputs"]["loss_in_solve"]
    with pytest.raises(ValueError, match="loss_in_solve"):
        init.design_from_dict(data)
    with pytest.raises(ValueError, match="loss_in_solve"):
        Inputs(loss_in_solve=2)

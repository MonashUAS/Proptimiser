"""
test_gradient_diagnostics.py

Bisects the propeller BEMT / optimisation pipeline to find the FIRST point
where jax.grad() produces a NaN, given that the forward pass is healthy
(finite objective values, NaN only in the gradient / OptErr).

Drop this next to init.py / propmodel.py / airflow.py / airfoil.py and run:

    python test_gradient_diagnostics.py

or with pytest:

    pytest -q test_gradient_diagnostics.py -s

Each test prints [ok] or [NaN DETECTED] plus the pytree path of the first
NaN leaf it finds, so you can see exactly which stage and which variable
(c, beta, omega, tilt, V_i_0...) breaks first.
"""
import jax
import jax.numpy as jnp
import equinox as eqx
from dataclasses import replace

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", False)  # keep off for these tests: we WANT to see nan, not crash on it

from init import Inputs
import propmodel
import airflow
import airfoil


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _isnan_leaf(leaf):
    return hasattr(leaf, "dtype") and jnp.issubdtype(leaf.dtype, jnp.floating) and bool(jnp.any(jnp.isnan(leaf)))


def report(name, grad_pytree):
    """Print pass/fail for a gradient pytree, and point at the offending leaf(s)."""
    bad_paths = []
    try:
        flat_with_path = jax.tree_util.tree_flatten_with_path(grad_pytree)[0]
        for path, leaf in flat_with_path:
            if _isnan_leaf(leaf):
                bad_paths.append(jax.tree_util.keystr(path))
    except AttributeError:
        # older jax without tree_flatten_with_path: fall back to flat check
        for leaf in jax.tree_util.tree_leaves(grad_pytree):
            if _isnan_leaf(leaf):
                bad_paths.append("<leaf>")

    if bad_paths:
        print(f"[NaN DETECTED] {name}")
        for p in bad_paths:
            print(f"    -> NaN at {p}")
    else:
        print(f"[ok]           {name}")
    return len(bad_paths) > 0


def finite_diff_grad(f, x, eps=1e-6):
    """Central-difference gradient for cross-checking analytic gradients once they're NaN-free."""
    flat, unravel = jax.flatten_util.ravel_pytree(x)
    g = jnp.zeros_like(flat)
    f0 = lambda flat_x: f(unravel(flat_x))
    for i in range(flat.shape[0]):
        bump = jnp.zeros_like(flat).at[i].set(eps)
        g = g.at[i].set((f0(flat + bump) - f0(flat - bump)) / (2 * eps))
    return unravel(g)


def base_params(n_r=None):
    """Build a small, fast default propeller for testing."""
    inputs = Inputs() if n_r is None else Inputs(n_r=n_r)
    return propmodel.build_default_propeller(inputs=inputs)


# ---------------------------------------------------------------------------
# Level 0: airfoil interpolator alone (isolates NeuralFoil vs everything else)
# ---------------------------------------------------------------------------

def test_level0_aero_coeffs():
    params = base_params()
    V_i = airflow.inflow_model(params.V_i_0, params)

    def f(c):
        p = replace(params, c=c)
        C_L, C_D, C_M = airfoil.aero_coeffs(V_i, p)
        return jnp.sum(C_L**2) + jnp.sum(C_D**2)

    val, g = jax.value_and_grad(f)(params.c)
    print(f"  forward value = {val}")
    return report("Level 0: aero_coeffs() d/dc", g)


def test_level0b_aero_coeffs_parametric_swap():
    """Same as level 0 but forcing the safe/analytic ParametricInterpolator.
    If THIS passes while test_level0 fails, NeuralFoilInterpolator is your bug."""
    old = Inputs.coeff_method
    Inputs.coeff_method = 0  # ParametricInterpolator (matches airfoil.py's own __main__ idiom)
    try:
        params = base_params()
        V_i = airflow.inflow_model(params.V_i_0, params)

        def f(c):
            p = replace(params, c=c)
            C_L, C_D, C_M = airfoil.aero_coeffs(V_i, p)
            return jnp.sum(C_L**2) + jnp.sum(C_D**2)

        val, g = jax.value_and_grad(f)(params.c)
        print(f"  forward value = {val}")
        return report("Level 0b: aero_coeffs() d/dc [ParametricInterpolator]", g)
    finally:
        Inputs.coeff_method = old


# ---------------------------------------------------------------------------
# Level 1: resultantVelocityComponents (the sqrt / arctan / division stack)
# ---------------------------------------------------------------------------

def test_level1_resultant_velocity():
    params = base_params()

    def f(V_i_0):
        V_i = airflow.inflow_model(V_i_0, params)
        V_R, phi = airflow.resultantVelocityComponents(V_i, params)
        return jnp.sum(V_R) + jnp.sum(phi**2)

    val, g = jax.value_and_grad(f)(params.V_i_0)
    print(f"  forward value = {val}")
    return report("Level 1: resultantVelocityComponents() d/dV_i_0", g)


# ---------------------------------------------------------------------------
# Level 2: bemt_residuals evaluated ONCE (no Broyden loop -- pure autodiff test)
# ---------------------------------------------------------------------------

def test_level2_bemt_residuals():
    params = base_params()
    V_scale = propmodel.get_velocity_scales(params)
    lambda_0 = params.V_i_0[Inputs.bemt_type] / V_scale

    def f(c, lam):
        p = replace(params, c=c)
        res = propmodel.bemt_residuals(lam, p)
        return jnp.sum(res**2)

    val = f(params.c, lambda_0)
    g_c = jax.grad(f, argnums=0)(params.c, lambda_0)
    g_lam = jax.grad(f, argnums=1)(params.c, lambda_0)
    print(f"  forward value = {val}")
    bad_c = report("Level 2: bemt_residuals() d/dc", g_c)
    bad_lam = report("Level 2: bemt_residuals() d/dlambda_i_0", g_lam)
    return bad_c or bad_lam


# ---------------------------------------------------------------------------
# Level 3: induced_velocity() -- THIS is the implicit-diff-through-Broyden step
# ---------------------------------------------------------------------------

def test_level3_induced_velocity():
    params = base_params()

    def f(c):
        p = replace(params, c=c)
        V_i, V_i_0 = propmodel.induced_velocity(p)
        return jnp.sum(V_i_0**2)

    val, g = jax.value_and_grad(f)(params.c)
    print(f"  forward value = {val}")
    return report("Level 3: induced_velocity() [Broyden implicit-diff] d/dc", g)


# ---------------------------------------------------------------------------
# Level 4: full per-condition objective and cost
# ---------------------------------------------------------------------------

def test_level4_objectives():
    params = base_params()

    def f(c):
        p = replace(params, c=c)
        motor_power, thrust, rms_MIL_err, V_i_0 = propmodel.objectives(p)
        return motor_power + thrust + rms_MIL_err

    val, g = jax.value_and_grad(f)(params.c)
    print(f"  forward value = {val}")
    return report("Level 4: objectives() d/dc", g)


def test_level5_condition_cost():
    params = base_params()

    def f(c):
        p = replace(params, c=c)
        cost, V_i_0 = propmodel.condition_cost(p)
        return jnp.sum(cost)

    val, g = jax.value_and_grad(f)(params.c)
    print(f"  forward value = {val}")
    return report("Level 5: condition_cost() d/dc", g)


# ---------------------------------------------------------------------------
# Level 6: one real solver.step() through jaxopt.LBFGSB (end-to-end)
# ---------------------------------------------------------------------------

def test_level6_solver_step():
    import proptimiser

    params_base = base_params()
    solver = proptimiser.LBFGSBSolver(params_base)
    state = solver.init_state()
    vars_scaled, state, V_i_0 = solver.step(solver.vars_scaled, state, solver.params_struct)

    print(f"  reported solver error (gradient-norm proxy) = {getattr(state, 'error', None)}")
    bad_vars = report("Level 6: LBFGSBSolver.step() output vars_scaled", vars_scaled)
    bad_vi0 = report("Level 6: LBFGSBSolver.step() aux V_i_0", V_i_0)
    return bad_vars or bad_vi0


# ---------------------------------------------------------------------------
# Runner: stop at the first level that goes NaN and report it clearly.
# ---------------------------------------------------------------------------

LEVELS = [
    test_level0_aero_coeffs,
    test_level0b_aero_coeffs_parametric_swap,
    test_level1_resultant_velocity,
    test_level2_bemt_residuals,
    test_level3_induced_velocity,
    test_level4_objectives,
    test_level5_condition_cost,
    test_level6_solver_step,
]

if __name__ == "__main__":
    print("=" * 78)
    print("Gradient NaN bisection")
    print("=" * 78)
    first_failure = None
    for test in LEVELS:
        print("-" * 78)
        try:
            failed = test()
        except Exception as e:
            print(f"[ERROR] {test.__name__} raised: {e!r}")
            failed = None
        if failed and first_failure is None:
            first_failure = test.__name__
    print("=" * 78)
    if first_failure:
        print(f"First NaN gradient appears at: {first_failure}")
        print("-> inspect that function's sqrt/log/division/where usage first.")
    else:
        print("No NaN gradients detected in any level. "
              "The bug may be data-dependent (only appears at certain c/beta/omega "
              "values reached mid-optimisation) -- consider adding a NaN guard "
              "inside the L-BFGS-B loop that dumps params_struct the first time "
              "state.error goes NaN, then feed those exact values into this harness.")

"""
test_propeller_setup.py

Checks for the run-setup refactor: JSON propeller files, Inputs reaching the physics,
spanwise CST sections, the NeuralFoil port, the optimiser design space (radial mask + blend zone),
live plotting of the seed state, verbosity settings, optimiser progress reporting, the Newton BEMT
solve and the normalised constrained problem.

    python test_propeller_setup.py

or with pytest:

    pytest -q test_propeller_setup.py
"""
import tempfile
import warnings
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

import init
from init import Inputs, BladeSpec, Propeller
import geometry
import airfoil
import proptimiser


# ---------------------------------------------------------------------------
# File I/O and run setup
# ---------------------------------------------------------------------------

def test_json_roundtrip_is_exact():
    prop = Propeller.default(blade=BladeSpec(freeze={"c": [(0.1, 0.2)]}))
    prop = prop.with_state(c=jnp.linspace(0.02, 0.04, prop.inputs.n_r))
    with tempfile.TemporaryDirectory() as tmp:
        back = Propeller.load(prop.save(Path(tmp) / "roundtrip"))

    for k in init.STATE_FIELDS:
        assert jnp.array_equal(getattr(prop, k), getattr(back, k)), k
    assert back.blade == prop.blade
    assert back.cfg == prop.cfg
    assert back.to_dict()["inputs"] == prop.to_dict()["inputs"]


def test_loaded_inputs_reach_the_physics():
    prop = Propeller.default(inputs=Inputs(n_r=6, coeff_method=0))
    with tempfile.TemporaryDirectory() as tmp:
        params = Propeller.load(prop.save(Path(tmp) / "inputs")).condition(0)

    assert params.r.shape == (6,)
    assert params.cfg.coeff_method == 0
    assert isinstance(params.airfoil_interp, airfoil.ParametricInterpolator)


def test_save_refuses_to_overwrite():
    prop = Propeller.default()
    with tempfile.TemporaryDirectory() as tmp:
        path = prop.save(Path(tmp) / "once")
        try:
            prop.save(path)
        except FileExistsError:
            pass
        else:
            raise AssertionError("save() replaced an existing file without overwrite=True")
        prop.save(path, overwrite=True)


def test_unknown_file_fields_warn_but_load():
    data = Propeller.default().to_dict()
    data["inputs"]["retired_setting"] = 1
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        Propeller.from_dict(data)
    assert any("retired_setting" in str(w.message) for w in caught)


def test_with_inputs_regrids_the_blade():
    prop = Propeller.default()
    prop = prop.with_state(c=jnp.linspace(0.02, 0.04, prop.inputs.n_r))
    fine = prop.with_inputs(Inputs(n_r=12))

    assert fine.c.shape == (12,) and fine.V_i_0.shape == (fine.inputs.n_cond, 3, 12)
    ends = jnp.array([0, -1])
    assert jnp.allclose(fine.c[ends], prop.c[ends])


def test_legacy_migration_keeps_the_saved_grid():
    if not (init.PROP_DIR / "prop_params_Currawong_V1.eqx").exists():
        return
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prop = Propeller.from_legacy_eqx("Currawong_V1")
    assert prop.inputs.n_r == 50
    assert np.allclose(np.asarray(prop.r), np.linspace(0.0, 0.254, 50))


# ---------------------------------------------------------------------------
# Spanwise CST sections and NeuralFoil
# ---------------------------------------------------------------------------

def test_kulfan_port_matches_aerosandbox():
    from aerosandbox.geometry.airfoil.airfoil_families import get_kulfan_coordinates

    rng = np.random.default_rng(0)
    x = np.asarray(geometry.cosine_spacing(80))
    for _ in range(5):
        cst = np.concatenate([rng.uniform(-0.3, 0.0, 8), rng.uniform(0.0, 0.4, 8), rng.uniform(-0.2, 0.3, 1), rng.uniform(0, 0.02, 1)])
        ref = get_kulfan_coordinates(lower_weights=cst[:8], upper_weights=cst[8:16], leading_edge_weight=cst[16],
                                     TE_thickness=cst[17], n_points_per_side=80)
        y_upper, y_lower = geometry.kulfan_coordinates(jnp.asarray(cst), x)
        ours = np.concatenate([np.stack([x[::-1], np.asarray(y_upper)[::-1]], 1), np.stack([x[1:], np.asarray(y_lower)[1:]], 1)])
        assert np.allclose(ours, ref, atol=1e-12)


def test_cst_interpolation_along_the_span():
    prop = Propeller.default(blade=BladeSpec(airfoils=("arad10", "naca4412", "naca0012")))
    params = prop.condition(0)
    r_cst = params.r_cst

    # Exact sections at the r_foils stations, linear blend from the hub section inside the blend zone
    assert jnp.allclose(geometry.cst_at_stations(r_cst, r_cst, params.cst), params.cst)
    r_mid = 0.5 * (r_cst[0] + r_cst[1])
    mid = geometry.cst_at_stations(jnp.array([r_mid]), r_cst, params.cst)[0]
    assert jnp.allclose(mid, 0.5 * (params.cst[0] + params.cst[1]))


def test_neuralfoil_port_matches_reference():
    import neuralfoil as nf

    kulfan = geometry.kulfan_from_name("arad10")
    alpha_deg = np.array([-4.0, 0.0, 3.0, 8.0])
    Re = 5e5
    ref = nf.get_aero_from_kulfan_parameters(
        kulfan_parameters=dict(lower_weights=kulfan[:8], upper_weights=kulfan[8:16],
                               leading_edge_weight=kulfan[16], TE_thickness=kulfan[17]),
        alpha=alpha_deg, Re=Re, model_size="large",
    )
    CL, CD, CM, log_confidence = airfoil.load_airfoil_data(3).raw(jnp.radians(alpha_deg), jnp.full(4, Re), jnp.asarray(kulfan))

    assert np.allclose(CL, ref["CL"], rtol=1e-8)
    assert np.allclose(CD, ref["CD"], rtol=1e-8)
    assert np.allclose(CM, ref["CM"], rtol=1e-8, atol=1e-12)
    assert np.allclose(np.exp(log_confidence), ref["analysis_confidence"], rtol=1e-8)


def test_post_stall_blend_matches_aerosandbox():
    # 360 deg coefficients: NeuralFoil blended into the Truong post-stall model exactly as AeroSandbox does
    # (KulfanAirfoil.get_aero_from_neuralfoil), built here from its parts because AeroSandbox's own compressibility
    # step changes CL slightly even at M = 0
    import neuralfoil as nf
    import aerosandbox.numpy as anp
    from aerosandbox.aerodynamics.aero_2D.airfoil_polar_functions import airfoil_coefficients_post_stall
    from aerosandbox.library.aerodynamics import Cf_flat_plate

    kulfan = geometry.kulfan_from_name("arad10")
    alpha_deg = np.arange(-180.0, 180.0, 2.5)
    Re = 2e5
    raw = nf.get_aero_from_kulfan_parameters(
        kulfan_parameters=dict(lower_weights=kulfan[:8], upper_weights=kulfan[8:16],
                               leading_edge_weight=kulfan[16], TE_thickness=kulfan[17]),
        alpha=alpha_deg, Re=Re, model_size="large",
    )
    CL_sep, CD_sep, _ = airfoil_coefficients_post_stall(airfoil=None, alpha=alpha_deg)
    separated = anp.softmax(alpha_deg - 20, -20 - alpha_deg) / 3
    CL_ref = anp.blend(separated, CL_sep, raw["CL"])
    CD_ref = np.exp(anp.blend(separated, np.log(CD_sep + Cf_flat_plate(Re_L=Re, method="turbulent")), np.log(raw["CD"])))

    n = alpha_deg.size
    CL, CD, _ = airfoil.load_airfoil_data(3)(jnp.radians(alpha_deg), jnp.full(n, Re), jnp.zeros(n), kulfan=jnp.asarray(kulfan))
    assert np.allclose(CL, CL_ref, rtol=1e-8, atol=1e-10)
    assert np.allclose(CD, CD_ref, rtol=1e-8)


def test_swirl_reduces_the_tangential_velocity():
    # Swirl V_it > 0 turns with the blade (positive momentum torque), so the blade sees omega r - V_it
    import airflow

    params = Propeller.default(inputs=Inputs(coeff_method=0, n_r=8, V_yz=[0.0, 0.0])).condition(0)
    V_i = jnp.zeros((3, 8, params.psi.shape[0])).at[0].set(2.0)
    _, phi_0 = airflow.resultantVelocityComponents(V_i, params)
    V_R, phi = airflow.resultantVelocityComponents(V_i.at[1].set(3.0), params)
    V_t = params.omega * params.r[:, None] - 3.0
    assert np.allclose(np.tan(phi), (params.V_x + 2.0) / V_t)
    assert np.allclose(V_R, np.hypot(params.V_x + 2.0, V_t))
    assert bool(jnp.all(phi > phi_0))


# ---------------------------------------------------------------------------
# Design space: radial mask, derived blend zone, CST variables
# ---------------------------------------------------------------------------

def _masked_prop():
    blade = BladeSpec(freeze={"c": [(0.09, 0.14)], "cst": [(0.27, 0.28)]})
    prop = Propeller.default(blade=blade)
    n_r = prop.inputs.n_r
    return prop.with_state(c=jnp.linspace(0.02, 0.04, n_r), beta=jnp.linspace(0.6, 0.2, n_r))


def test_design_space_mask_and_blend():
    prop = _masked_prop()
    space = proptimiser.DesignSpace.build(prop)
    r = np.asarray(prop.r)
    free = dict(space.free)

    frozen_c = [i for i, r_i in enumerate(r) if 0.09 <= r_i <= 0.14]
    assert frozen_c and not set(frozen_c) & set(free["c"])
    assert 0 not in free["c"] and 0 not in free["beta"]               # blend zone is derived, never free
    assert len(free["cst"]) == 2 * (geometry.N_CST - 1)               # tip station frozen, TE thickness fixed

    # Any design the optimiser picks keeps frozen entries at their seed values
    x = jax.tree_util.tree_map(lambda v: jnp.full_like(v, 0.37), space.to_unit(prop))
    values = space.values(x)
    assert jnp.array_equal(values["c"][np.array(frozen_c)], prop.c[np.array(frozen_c)])
    assert jnp.array_equal(values["cst"][2], prop.cst[2])
    assert jnp.array_equal(values["cst"][:, -1], prop.cst[:, -1])

    # Blend-zone planform follows the hub values and the first aerodynamic station
    c_blend, beta_blend = geometry.blend_planform(r, values["c"], values["beta"], prop.blade.R_hub, prop.blade.R_aero,
                                                  prop.blade.hub_chord, np.radians(prop.blade.hub_twist_deg))
    assert jnp.allclose(values["c"], c_blend) and jnp.allclose(values["beta"], beta_blend)
    assert jnp.isclose(values["c"][0], prop.blade.hub_chord)


def test_design_space_roundtrip():
    prop = _masked_prop()
    space = proptimiser.DesignSpace.build(prop)
    values = space.values(space.to_unit(prop))
    free_c = np.array(dict(space.free)["c"])

    assert jnp.allclose(values["c"][free_c], prop.c[free_c])
    assert jnp.allclose(values["cst"], prop.cst)


def test_total_cost_gradients_are_finite():
    solver = proptimiser.LBFGSBSolver(_masked_prop())
    grads = jax.grad(lambda x: proptimiser.total_cost(x, solver.params_struct, solver.space)[0])(solver.vars_scaled)
    for name, g in grads.items():
        assert bool(jnp.all(jnp.isfinite(g))), name


# ---------------------------------------------------------------------------
# Live plotting, verbosity and solver reporting
# ---------------------------------------------------------------------------

def test_contour_levels_are_strictly_increasing():
    import results

    for values in (np.full((4, 5), 2.5), np.zeros((3, 3)), np.array([[np.nan, 1.0], [1.0, np.nan]]),
                   np.full((2, 2), np.nan), np.linspace(-3.0, 4.0, 12).reshape(3, 4)):
        assert np.all(np.diff(results._contour_levels(values)) > 0), values


def test_live_plot_draws_the_uniform_seed_field():
    # The momentum-theory seed gives a uniform V_ia field, which used to crash the first animation frame
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    import results

    params = Propeller.default(inputs=Inputs(coeff_method=0, n_r=6)).condition(0)
    fig = Figure()
    FigureCanvasAgg(fig)
    plotter = results.LivePropPlotter(params, fig=fig)
    plotter.draw(results.plot_snapshot(params))
    plotter.draw(results.plot_snapshot(params))
    fig.canvas.draw()


def test_legacy_verbose_key_maps_to_bemt_verbosity():
    data = Propeller.default().to_dict()
    del data["inputs"]["V_i_0_verbose"]
    data["inputs"]["verbose"] = True
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        prop = Propeller.from_dict(data)
    assert prop.inputs.V_i_0_verbose == 1
    assert not any("verbose" in str(w.message) for w in caught)


def test_solvers_report_progress_and_stop():
    import dataclasses
    import threading

    base = Propeller.default(inputs=Inputs(coeff_method=0, n_r=6, opt_max_iter=2, opt_print_every=1, opt_verbose=0))
    for solver in proptimiser.SOLVERS:
        prop = base.with_inputs(dataclasses.replace(base.inputs, opt_solver=solver))
        reports = []
        result = proptimiser.run_optimization(prop, callback=reports.append, stop_event=threading.Event())
        assert reports and all(isinstance(p, proptimiser.Progress) for p in reports), solver
        assert reports[-1].status in ("max_iter", "converged") and reports[-1].iteration <= 2, (solver, reports[-1].status)
        assert len(result.error_log) == reports[-1].iteration + 1 == len(result.time_log), solver
        assert reports[-1].condition(1).omega == prop.omega[1]
        assert np.isfinite(reports[-1].objective) and reports[-1].objective > 0

        stop = threading.Event()
        stop.set()
        reports = []
        result = proptimiser.run_optimization(prop, callback=reports.append, stop_event=stop)
        assert reports[-1].status == "stopped" and reports[-1].iteration == 0, solver
        assert result.c.shape == prop.c.shape


# ---------------------------------------------------------------------------
# Newton BEMT solve and the normalised problem
# ---------------------------------------------------------------------------

def test_bemt_jacobian_is_block_diagonal():
    import propmodel

    params = Propeller.default(inputs=Inputs(coeff_method=0, n_r=6)).condition(0)
    lam = params.V_i_0[params.cfg.solve_rows] / propmodel.get_velocity_scales(params)
    J = np.asarray(jax.jacfwd(propmodel.bemt_residuals)(lam, params))           # (k, n_r, k, n_r)
    off_station = ~np.eye(J.shape[1], dtype=bool)[None, :, None, :].repeat(J.shape[0], 0).repeat(J.shape[2], 2)
    assert np.abs(J[off_station]).max() == 0.0


def test_newton_bemt_converges_and_matches_a_tighter_solve():
    import dataclasses
    import propmodel

    prop = Propeller.default(inputs=Inputs(coeff_method=0, n_r=8))
    params = prop.condition(0)
    V_i, V_i_0 = propmodel.induced_velocity(params)
    lam = V_i_0[params.cfg.solve_rows] / propmodel.get_velocity_scales(params)
    # Only the aerodynamic zone is solved; the blend zone keeps its initial guess
    residual = np.abs(np.asarray(propmodel.bemt_residuals(lam, params)))[:, np.asarray(propmodel.aero_zone(params))]
    assert residual.max() <= params.cfg.V_i_0_eps

    tight = prop.with_inputs(dataclasses.replace(prop.inputs, V_i_0_eps=1e-12)).condition(0)
    V_i_tight, _ = propmodel.induced_velocity(tight)
    assert np.isclose(float(propmodel.thrust(V_i, params)), float(propmodel.thrust(V_i_tight, tight)), rtol=1e-7)


def test_bracketed_fallback_finds_the_newton_roots():
    # With no Newton iterations every station goes to the bracketed fallback, which must reach the same roots
    import dataclasses
    import propmodel

    for coeff_method in (0, 3):
        params = Propeller.default(inputs=Inputs(coeff_method=coeff_method, n_r=8)).condition(1)
        _, V_newton = propmodel.induced_velocity(params)
        bracketed = dataclasses.replace(params, cfg=dataclasses.replace(params.cfg, V_i_0_max_iter=0))
        _, V_bracketed = propmodel.induced_velocity(bracketed)
        lam = V_bracketed[params.cfg.solve_rows] / propmodel.get_velocity_scales(params)
        residual = np.abs(np.asarray(propmodel.bemt_residuals(lam, params)))[:, np.asarray(propmodel.aero_zone(params))]
        assert residual.max() <= params.cfg.V_i_0_eps
        assert np.allclose(V_bracketed, V_newton, atol=1e-6)


def test_blend_zone_carries_no_force():
    # The blend zone is not solved: its geometry (set by the hub values) must not change thrust or torque
    import dataclasses
    import propmodel

    inputs = Inputs(coeff_method=0, n_r=16)
    base = Propeller.default(inputs=inputs)
    wide_hub = Propeller.default(inputs=inputs, blade=dataclasses.replace(base.blade, hub_chord=0.06, hub_twist_deg=20.0))
    assert not bool(jnp.all(propmodel.aero_zone(base.condition(0))))
    solver = proptimiser.SLSQPSolver(base)
    solver_wide = proptimiser.SLSQPSolver(wide_hub)
    assert not np.allclose(solver.params_struct.c, solver_wide.params_struct.c)
    J, g, _ = proptimiser.problem_values(solver.vars_scaled, solver.params_struct, solver.space)
    J_wide, g_wide, _ = proptimiser.problem_values(solver_wide.vars_scaled, solver_wide.params_struct, solver_wide.space)
    assert np.isclose(float(J), float(J_wide), rtol=1e-9)
    assert np.allclose(np.asarray(g), np.asarray(g_wide), rtol=1e-9, atol=1e-12)


def test_zero_radius_station_is_handled():
    # A blade whose BEMT grid starts at r = 0 (a zero-area annulus with no momentum equation)
    import propmodel

    prop = Propeller.default(inputs=Inputs(coeff_method=0, n_r=8), blade=BladeSpec(r_foils=(0.0, 0.0, 0.05, 0.15, 0.275)))
    assert float(prop.r[0]) == 0.0
    V_i, V_i_0 = propmodel.induced_velocity(prop.condition(0))
    assert bool(jnp.all(jnp.isfinite(V_i_0)))
    assert np.isfinite(float(propmodel.thrust(V_i, prop.condition(0))))


def test_newton_bemt_escapes_a_stale_warm_start():
    # Currawong_V1's saved induced velocity puts the first station near the hub on a reversed-flow state in
    # forward flight, where Newton stalls at a minimum of the residual norm that is not a root
    import propmodel

    prop = Propeller.load("Currawong_V1")
    p = prop.condition(0)
    _, V_i_0 = propmodel.induced_velocity(p)
    lam = V_i_0[p.cfg.solve_rows] / propmodel.get_velocity_scales(p)
    residual = np.abs(np.asarray(propmodel.bemt_residuals(lam, p)))[:, np.asarray(p.r) > 0]
    assert residual.max() <= p.cfg.V_i_0_eps


def test_cost_gradient_matches_finite_differences():
    solver = proptimiser.LBFGSBSolver(Propeller.default(inputs=Inputs(coeff_method=0, n_r=6)))
    x, ps, space = solver.vars_scaled, solver.params_struct, solver.space
    f = lambda v: proptimiser.total_cost(v, ps, space)[0]
    g = jax.grad(f)(x)
    rng = np.random.default_rng(1)
    d = {k: jnp.asarray(rng.standard_normal(v.shape)) for k, v in x.items()}
    h = 1e-6
    fd = (float(f({k: x[k] + h * d[k] for k in x})) - float(f({k: x[k] - h * d[k] for k in x}))) / (2 * h)
    ad = sum(float(jnp.sum(g[k] * d[k])) for k in x)
    assert abs(ad - fd) <= 1e-5 * abs(fd), (ad, fd)


def test_constraints_match_their_groups():
    for inputs in (Inputs(coeff_method=0, n_r=6), Inputs(coeff_method=3, n_r=6)):
        prop = Propeller.default(inputs=inputs)
        space = proptimiser.DesignSpace.build(prop)
        J, g, _ = proptimiser.problem_values(space.to_unit(prop), prop.batched(), space)
        assert g.shape == (sum(n for _, n in space.constraint_groups(inputs.n_cond)),)
        assert np.isfinite(float(J)) and float(J) > 0


def test_confidence_constraint_is_load_weighted_and_keeps_its_gradient():
    import propmodel

    x = jnp.array([0.0, 0.1, 0.35, 0.4, 1.0])
    assert np.isclose(jnp.sum(proptimiser._trapezoid_weights(x) * jnp.sin(3 * x)), jnp.trapezoid(jnp.sin(3 * x), x=x))

    # One constraint per condition: the confidence averaged over the aerodynamic zone, weighted by V_R^2 c dr dpsi
    prop = Propeller.default(inputs=Inputs(coeff_method=3, n_r=6))
    space = proptimiser.DesignSpace.build(prop)
    ps = prop.batched()
    g = np.asarray(proptimiser.constraint_values(proptimiser.evaluate_all_conditions(ps), ps, space))
    p = prop.condition(0)
    V_i, _ = propmodel.induced_velocity(p)
    V_R, _ = airfoil.resultantVelocityComponents(V_i, p)
    confidence = np.exp(np.asarray(airfoil.aero_coeffs(V_i, p, return_log_confidence=True)[3]))
    w = (np.asarray(V_R) ** 2 * np.asarray(p.c * proptimiser._trapezoid_weights(p.r))[:, None]
         * np.asarray(proptimiser._trapezoid_weights(p.psi))[None, :])[np.array(space.aero_stations)]
    expected = np.sum(w * confidence[np.array(space.aero_stations)]) / np.sum(w)
    assert np.isclose(np.exp(g[-prop.inputs.n_cond]) * prop.inputs.min_confidence, expected, rtol=1e-6)

    # Far outside NeuralFoil's training data the confidence saturates at zero; its log still has a gradient
    raw = airfoil.load_airfoil_data(3).raw
    kulfan = jnp.asarray(geometry.kulfan_from_name("arad10"))
    alpha = jnp.radians(-80.0)
    d_log = jax.grad(lambda a: raw(a, 5e5, kulfan)[3])(alpha)
    d_conf = jax.grad(lambda a: jnp.exp(raw(a, 5e5, kulfan)[3]))(alpha)
    assert np.isfinite(d_log) and abs(d_log) > 1.0 and abs(d_conf) < 1e-6


def test_curvature_scale_is_positive_and_normalised():
    prop = Propeller.default(inputs=Inputs(coeff_method=0, n_r=6))
    space = proptimiser.DesignSpace.build(prop)
    x = space.to_unit(prop)
    s = proptimiser.curvature_scale(x, space.from_unit(x, prop.batched()), space)
    assert np.all(np.isfinite(s)) and np.all(s > 0)
    assert np.isclose(np.exp(np.mean(np.log(s))), 1.0)


def test_format_2_files_take_the_newton_defaults():
    data = Propeller.default().to_dict()
    data["format_version"] = 2
    data["inputs"].update(V_i_0_eps=0.05, V_i_0_max_iter=1000, opt_tol=1e-4)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        prop = Propeller.from_dict(data)
    defaults = Inputs()
    assert (prop.inputs.V_i_0_eps, prop.inputs.V_i_0_max_iter, prop.inputs.opt_tol) == (
        defaults.V_i_0_eps, defaults.V_i_0_max_iter, defaults.opt_tol)
    assert any("Format 2" in str(w.message) for w in caught)


TESTS = [value for name, value in list(globals().items()) if name.startswith("test_")]

if __name__ == "__main__":
    failures = 0
    for test in TESTS:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                test()
            print(f"[ok]     {test.__name__}")
        except Exception as e:
            failures += 1
            print(f"[FAILED] {test.__name__}: {e!r}")
    print(f"\n{len(TESTS) - failures}/{len(TESTS)} passed")
    raise SystemExit(1 if failures else 0)

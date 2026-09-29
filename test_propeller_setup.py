"""
test_propeller_setup.py

Checks for the run-setup refactor: JSON propeller files, Inputs reaching the physics,
spanwise CST sections, the NeuralFoil port, the optimiser design space (radial mask + blend zone),
live plotting of the seed state, verbosity settings and optimiser progress reporting.

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
    CL, CD, CM, confidence = airfoil.load_airfoil_data(3).raw(jnp.radians(alpha_deg), jnp.full(4, Re), jnp.asarray(kulfan))

    assert np.allclose(CL, ref["CL"], rtol=1e-8)
    assert np.allclose(CD, ref["CD"], rtol=1e-8)
    assert np.allclose(CM, ref["CM"], rtol=1e-8, atol=1e-12)
    assert np.allclose(confidence, ref["analysis_confidence"], rtol=1e-8)


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


def test_solver_reports_progress_and_stops():
    import threading

    prop = Propeller.default(inputs=Inputs(coeff_method=0, n_r=6, opt_max_iter=2, opt_print_every=1, opt_verbose=0))
    reports = []
    result = proptimiser.run_optimization(prop, callback=reports.append, stop_event=threading.Event())
    assert reports and all(isinstance(p, proptimiser.Progress) for p in reports)
    assert reports[-1].status == "max_iter" and reports[-1].iteration == 2
    assert len(result.error_log) == 3 and len(result.time_log) == 3
    assert reports[-1].condition(1).omega == prop.omega[1]

    stop = threading.Event()
    stop.set()
    reports = []
    result = proptimiser.run_optimization(prop, callback=reports.append, stop_event=stop)
    assert reports[-1].status == "stopped" and reports[-1].iteration == 0
    assert result.c.shape == prop.c.shape


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

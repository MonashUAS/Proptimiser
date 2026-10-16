"""
Tests for fit_prop.py. They use hand-made numbers. No test reads the manufacturer chart as ground truth.

    python -m pytest -q tests/test_fit_prop.py
"""
import math

import numpy as np
import pytest

from proptimiser import fit_prop as fp

TARGET = fp.Target(diameter_m=0.5, pitch_m=0.2, blades=2, r_hub_m=0.02)


def test_read_chart_skips_comments(tmp_path):
    path = tmp_path / "chart.csv"
    path.write_text("# comment\n# more\nrpm,thrust_gf,input_power_W,motor_esc_efficiency\n"
                    "1000,100,10,0.5\n2000,400,80,0.5\n", encoding="utf-8")
    chart = fp.read_chart(path)
    assert chart["rpm"].tolist() == [1000.0, 2000.0]
    assert chart["thrust_gf"].tolist() == [100.0, 400.0]


def test_read_chart_refuses_a_missing_column(tmp_path):
    path = tmp_path / "chart.csv"
    path.write_text("rpm,thrust_gf\n1000,100\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing columns"):
        fp.read_chart(path)


def test_chart_targets_units():
    chart = dict(rpm=np.array([1000.0]), thrust_gf=np.array([1000.0]), input_power_W=np.array([100.0]),
                 motor_esc_efficiency=np.array([0.8]))
    rpm, thrust, shaft = fp.chart_targets(chart)
    assert math.isclose(thrust[0], 9.80665)          # 1 kgf
    assert math.isclose(shaft[0], 80.0)


def test_blade_family_pitch_and_chord():
    r = np.array([0.1875])                             # 0.75 R for R = 0.25 m
    chord, beta = fp.blade_family((0.1, 0.0, 0.4, 0.35, 0.0), r, TARGET)
    assert math.isclose(chord[0], 0.1 * 0.25)          # c75 = 0.1 R at 0.75 R
    assert math.isclose(math.tan(beta[0]), 0.2 / (2 * math.pi * 0.1875))
    # an offset angle adds to the twist
    _, beta2 = fp.blade_family((0.1, 1.5, 0.4, 0.35, 0.0), r, TARGET)
    assert math.isclose(beta2[0] - beta[0], math.radians(1.5))
    # a pitch gradient changes the pitch away from 0.75 R only
    r2 = np.array([0.1875, 0.25 * 0.95])
    _, beta3 = fp.blade_family((0.1, 0.0, 0.4, 0.35, 0.2), r2, TARGET)
    assert math.isclose(math.tan(beta3[0]), 0.2 / (2 * math.pi * 0.1875))
    assert math.isclose(math.tan(beta3[1]), 0.2 * (1 + 0.2 * 0.2) / (2 * math.pi * 0.2375))


def test_chord_is_positive_and_tapers_to_the_tip():
    r = np.linspace(0.03, 0.2499, 30)
    chord, _ = fp.blade_family(fp.THETA_START, r, TARGET)
    assert np.all(chord > 0)
    assert chord[-1] < chord.max()


def test_prior_residuals_are_zero_at_the_priors():
    theta = (0.12, *(fp.PRIORS[k][0] for k in fp.PARAM_NAMES[1:]))
    assert np.allclose(fp.prior_residuals(theta), 0.0)
    shifted = (0.12, 2.0, *theta[2:])
    assert math.isclose(fp.prior_residuals(shifted)[0], 1.0)       # 2 deg over a 2 deg sigma


def test_residuals_scale_with_the_reading_error():
    res = fp.residuals([105.0], [100.0], [100.0], [100.0], TARGET)
    assert math.isclose(res[0], math.log(1.05) / 0.05)
    assert math.isclose(res[1], 0.0)


def test_error_summary_and_gate():
    s = fp.error_summary([1.05, 0.95, 1.0], [1.0, 1.0, 1.0])
    assert math.isclose(s["rms"], math.sqrt(2 * 0.05**2 / 3))
    assert math.isclose(s["max_abs"], 0.05)
    good = fp.error_summary([1.05, 0.95], [1.0, 1.0])
    bad_max = fp.error_summary([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.15], [1.0] * 10)
    bad_rms = fp.error_summary([1.1, 0.9], [1.0, 1.0])
    assert fp.gate(good, good)
    assert not fp.gate(good, bad_max)           # one point worse than 12 %
    assert not fp.gate(bad_rms, good)           # RMS above 7 %


def test_figure_of_merit_of_an_ideal_rotor_is_one():
    # Ideal hover: P = T sqrt(T / (2 rho A))
    D, rho, T = 1.0, 1.2, 50.0
    A = math.pi * D**2 / 4
    P = T * math.sqrt(T / (2 * rho * A))
    assert math.isclose(fp.figure_of_merit(T, P, D, rho), 1.0)


def test_torque_over_thrust_diameter():
    # P = 100 W, omega = 50 rad/s: Q = 2 N m. T = 10 N, D = 0.5 m: Q / (T D) = 0.4
    assert math.isclose(fp.torque_over_thrust_diameter(10.0, 100.0, 50.0, 0.5), 0.4)


def test_fluxer_design_files_load():
    # The design YAML and the state JSON must load with every key present. No chart value is used.
    from proptimiser import init

    prop = init.Propeller.load(fp.NAME)
    assert math.isclose(2 * prop.blade.R_tip, 22 * 0.0254)
    assert prop.blade.N == 2
    assert float(prop.inputs.V_x[0]) == 0.0
    assert np.all(np.asarray(prop.c) > 0)


def test_power_at_thrust_follows_a_power_law():
    T = np.array([10.0, 20.0, 40.0])
    P = 3.0 * T**1.5
    q = np.array([14.0, 8.0, 50.0])                # inside, below and above the curve
    assert np.allclose(fp.power_at_thrust(T, P, q), 3.0 * q**1.5)


def test_equal_thrust_residual_is_zero_for_the_same_curve_and_signs_a_high_power():
    T = np.array([10.0, 20.0, 40.0])
    P = 3.0 * T**1.5
    assert np.allclose(fp.equal_thrust_residuals(T, P, T, P, TARGET), 0.0)
    assert np.all(fp.equal_thrust_residuals(T, 1.1 * P, T, P, TARGET) > 0)


def test_hover_weights():
    w = fp.hover_weights([2400, 2800, 3300, 3800, 4200])
    assert w.tolist() == [1.0, 2.0, 2.0, 2.0, 1.0]


def test_export_cli_passes_the_rpm_and_vx_lists(monkeypatch):
    from proptimiser import export_performance as ep

    seen = {}
    monkeypatch.setattr(ep, "export", lambda *a, **k: seen.update(k) or "out.h5")
    ep.main(["x", "--author", "A", "--rpm-list", "1500,3000", "--vx-list", "0,2.5"])
    assert np.allclose(seen["omega_list"], [1500 * math.pi / 30, 3000 * math.pi / 30])
    assert seen["vx_list"] == [0.0, 2.5]
    seen.clear()
    ep.main(["x", "--author", "A"])
    assert seen["omega_list"] is None and seen["vx_list"] is None

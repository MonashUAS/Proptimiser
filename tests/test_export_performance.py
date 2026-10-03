"""
Tests for export_performance.py. They run offline on a small grid.

    python -m pytest -q tests/test_export_performance.py
"""
import csv
import hashlib
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))           # the module lives in the repository root

import numpy as np
import yaml

import export_performance as ep
from init import Inputs, Propeller

RHO = 1.225
YAML_KEYS = ["schema", "schema_version", "producer", "kind", "prop_id", "source", "D_m", "blades", "rho",
             "model", "design_points", "motor_assumed", "V_x_min_converged_mps", "generated", "exporter",
             "csv_sha256"]


class StubCase:
    """A case with a made-up solver. It fails to converge at V_x = 0."""
    R_tip, blades, design_omegas = 0.25, 2, [500.0]
    files, note = ["stub.json"], "Stub."

    def solve(self, omega, vx, vyz, rho=RHO, warm=None):
        return 10.0 - vx, 0.5, vx > 0, None         # thrust N, torque Nm, converged, state

    def design_points(self):
        return []

    def model(self):
        return dict(n_r=6, n_psi=8)

    def motor(self):
        return dict(kv=220, rm=0.037, i0=2.0)


def read_csv(path):
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, [{k: float(v) for k, v in row.items()} for row in reader]


def check_coefficients(rows, R_tip, rho=RHO):
    """CT and CP must follow from the thrust and power columns."""
    D = 2 * R_tip
    for row in rows:
        n = row["omega_rad_s"] / (2 * math.pi)
        assert math.isclose(row["J"], row["V_x_mps"] / (n * D), rel_tol=1e-9)
        assert math.isclose(row["CT"], row["thrust_N"] / (rho * n**2 * D**4), rel_tol=1e-9)
        assert math.isclose(row["CP"], row["shaft_power_W"] / (rho * n**3 * D**5), rel_tol=1e-9)
        assert math.isclose(row["CQ"], row["CP"] / (2 * math.pi), rel_tol=1e-9)
        assert math.isclose(row["shaft_power_W"], row["torque_Nm"] * row["omega_rad_s"], rel_tol=1e-9)
        assert math.isclose(row["eta"], row["J"] * row["CT"] / row["CP"], rel_tol=1e-9, abs_tol=1e-12)
        assert math.isclose(row["tip_mach"], row["omega_rad_s"] * R_tip / 343.0, rel_tol=1e-9)


def test_row_follows_the_definitions():
    row = ep.table_row(400.0, 6.0, 20.0, 0.8, True, 0.25)
    check_coefficients([row], 0.25)
    assert row["converged"] == 1
    assert row["rpm"] == 400.0 * 30 / math.pi
    assert ep.table_row(400.0, 0.0, 20.0, 0.8, True, 0.25)["eta"] == 0.0


def test_unconverged_point_stays_in_the_table(tmp_path):
    rows = ep.performance_table(lambda: StubCase(), [500.0], [0.0, 2.0, 4.0])
    assert [r["converged"] for r in rows] == [0, 1, 1]
    assert all(math.isnan(rows[0][c]) for c in ["CT", "CP", "CQ", "thrust_N", "torque_Nm", "shaft_power_W", "eta"])
    assert rows[0]["J"] == 0.0

    # The YAML names the smallest V_x at which every omega converged
    csv_path, yaml_path = ep.export(lambda: StubCase(), "Stub", "abc1234", tmp_path, omega_list=[500.0],
                                    vx_list=[0.0, 2.0, 4.0])
    assert yaml.safe_load(yaml_path.read_text())["V_x_min_converged_mps"] == 2.0
    _, written = read_csv(csv_path)
    assert len(written) == 3 and math.isnan(written[0]["CT"])


def test_default_omegas_have_no_duplicates():
    omegas = ep.default_omegas(StubCase())
    assert omegas == sorted(set(omegas))
    assert 500.0 in omegas and 300.0 in omegas and 750.0 in omegas
    assert len(omegas) == 10 and all(isinstance(w, float) for w in omegas)     # 500 is on the 50 rad/s steps


def test_apc_table_reader():
    table = ep.read_apc_table(ROOT / "apc_prop_data" / "performance_data" / "PER3_20x8E.dat")
    assert sorted(table) == list(range(1000, 12001, 1000))
    assert np.allclose(table[4000][0], [0.0, 0.0718, 0.0212])      # J, CT, CP at J = 0


def test_apc_comparison_of_a_perfect_model(tmp_path):
    # A stub that returns the APC values must give zero difference
    dat = ROOT / "apc_prop_data" / "performance_data" / "PER3_20x8E.dat"
    apc = ep.read_apc_table(dat)

    class PerfectCase(StubCase):
        R_tip = 0.254

        def solve(self, omega, vx, vyz, rho=RHO, warm=None):
            n, D = omega / (2 * math.pi), 2 * self.R_tip
            table = apc[round(omega * 30 / math.pi)]
            ct, cp = (np.interp(vx / (n * D), table[:, 0], table[:, c]) for c in (1, 2))
            return ct * rho * n**2 * D**4, cp * rho * n**3 * D**5 / omega, True, None

    shown, summary = ep.compare_apc(lambda: PerfectCase(), dat, tmp_path, "Perfect")
    assert len(shown) == 9 and {r["J"] for r in shown} == {0.0, 0.2, 0.4}
    assert summary["n_points_per_coefficient"] == 81 and summary["n_points_pooled"] == 162
    assert summary["rms_rel_diff_pct"] < 1e-6
    assert (tmp_path / "Perfect__apc.compare.csv").exists() and (tmp_path / "Perfect__apc.compare.yaml").exists()


def test_export_of_a_small_solved_case(tmp_path):
    # A coarse propeller with the parametric airfoil: fast to compile
    prop = Propeller.default(inputs=Inputs(n_r=6, n_psi=8, coeff_method=0))
    case = ep.make_case(prop, ["default.json"], "Test propeller.")
    csv_path, yaml_path = ep.export(lambda: case, "Test_prop", "abc1234", tmp_path, omega_list=[400.0],
                                    vx_list=[0.0, 5.0, 10.0])

    assert csv_path.name == "Test_prop__abc1234.perf.csv" and yaml_path.name == "Test_prop__abc1234.perf.yaml"
    columns, rows = read_csv(csv_path)
    assert columns == ep.COLUMNS
    assert [r["converged"] for r in rows] == [1, 1, 1]
    check_coefficients(rows, prop.cfg.R_tip)
    assert rows[0]["thrust_N"] > rows[1]["thrust_N"] > rows[2]["thrust_N"] > 0     # thrust falls with V_x

    doc = yaml.safe_load(yaml_path.read_text())
    assert list(doc) == YAML_KEYS
    assert doc["schema"] == "prop-performance" and doc["schema_version"] == 1
    assert doc["producer"] == "MUAS" and doc["kind"] == "muas_bemt"
    assert doc["csv_sha256"] == hashlib.sha256(csv_path.read_bytes()).hexdigest()
    assert math.isclose(doc["D_m"], 2 * prop.cfg.R_tip) and doc["blades"] == 2
    assert doc["V_x_min_converged_mps"] == 0.0
    assert len(doc["design_points"]) == prop.inputs.n_cond
    assert {"kv", "rm", "i0"} == set(doc["motor_assumed"])
    assert {"n_r", "n_psi", "inflow_model", "bemt_type"} <= set(doc["model"])

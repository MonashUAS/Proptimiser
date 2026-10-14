"""
Tests for export_performance.py (contract A, prop-performance v1). They run offline on a small grid.

    python -m pytest -q tests/test_export_performance.py
"""
import math
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

import h5py
import numpy as np
import pytest

from proptimiser import export_performance as ep
from proptimiser import init
from proptimiser.init import Inputs, Propeller

RHO = 1.225
ROOT_ATTRS = {"schema", "schema_version", "prop_id", "design_sha256", "created_utc", "author", "placeholder"}
UNCERTAINTY_FIELDS = {"status", "scope", "basis", "method", "distribution", "effect", "num_sigmas", "coverage",
                      "sigma", "lower", "upper", "per_point_column", "factors", "correlations", "qualitative_level"}


class StubCase:
    """A case with a made-up solver. It fails to converge at V_x = 0."""
    R_tip, blades, design_omegas, sound = 0.25, 2, [500.0], 343.0

    def solve(self, omega, vx, rho=RHO, warm=None):
        return 10.0 - vx, 0.5, 1e5, vx > 0, None     # thrust N, torque Nm, Re at 0.75 R, converged, state


def check_coefficients(rows, R_tip, rho=RHO, thrust=None, torque=None):
    """J, CT, CP, CQ and eta must follow the contract A definitions."""
    D = 2 * R_tip
    for row in rows:
        n = row["rpm"] / 60
        omega = 2 * math.pi * n
        assert math.isclose(row["J"], row["V_mps"] / (n * D), rel_tol=1e-9)
        assert math.isclose(row["CP"], 2 * math.pi * row["CQ"], rel_tol=1e-9)
        assert math.isclose(row["eta"], row["J"] * row["CT"] / row["CP"], rel_tol=1e-9, abs_tol=1e-12)
        assert math.isclose(row["tip_mach"], omega * R_tip / 343.0, rel_tol=1e-9)
        if thrust is not None:
            assert math.isclose(row["CT"], thrust / (rho * n**2 * D**4), rel_tol=1e-9)
            assert math.isclose(row["CQ"], torque / (rho * n**2 * D**5), rel_tol=1e-9)


def test_row_follows_the_definitions():
    row = ep.table_row(400.0, 6.0, 20.0, 0.8, 1e5, True, 0.25)
    check_coefficients([row], 0.25, thrust=20.0, torque=0.8)
    assert row["converged"] is True and row["reynolds_75"] == 1e5
    assert row["rpm"] == 400.0 * 30 / math.pi
    assert ep.table_row(400.0, 0.0, 20.0, 0.8, 1e5, True, 0.25)["eta"] == 0.0
    assert set(row) == set(ep.PERFORMANCE_COLUMNS)


def test_unconverged_point_stays_in_the_table():
    rows = ep.performance_table(StubCase(), [500.0], [0.0, 2.0, 4.0])
    assert [r["converged"] for r in rows] == [False, True, True]
    assert all(math.isnan(rows[0][c]) for c in ["CT", "CP", "CQ", "eta", "reynolds_75"])
    assert rows[0]["J"] == 0.0


def test_default_omegas_have_no_duplicates():
    omegas = ep.default_omegas(StubCase())
    assert omegas == sorted(set(omegas))
    assert 500.0 in omegas and 300.0 in omegas and 750.0 in omegas
    assert len(omegas) == 10 and all(isinstance(w, float) for w in omegas)     # 500 is on the 50 rad/s steps


def test_apc_table_reader():
    table = ep.read_apc_table(ROOT / "apc_prop_data" / "performance_data" / "PER3_20x8E.dat")
    assert sorted(table) == list(range(1000, 12001, 1000))
    assert np.allclose(table[4000][0], [0.0, 0.0718, 0.0212])      # J, CT, CP at J = 0


def test_apc_comparison_of_a_perfect_model():
    # A stub that returns the APC values must give zero difference
    dat = ROOT / "apc_prop_data" / "performance_data" / "PER3_20x8E.dat"
    apc = ep.read_apc_table(dat)

    class PerfectCase(StubCase):
        R_tip = 0.254

        def solve(self, omega, vx, rho=RHO, warm=None):
            n, D = omega / (2 * math.pi), 2 * self.R_tip
            table = apc[round(omega * 30 / math.pi)]
            ct, cp = (np.interp(vx / (n * D), table[:, 0], table[:, c]) for c in (1, 2))
            return ct * rho * n**2 * D**4, cp * rho * n**3 * D**5 / omega, 1e5, True, None

    shown, summary = ep.compare_apc(PerfectCase(), dat)
    assert len(shown) == 9 and {r["J"] for r in shown} == {0.0, 0.2, 0.4}
    assert summary["n_points_per_coefficient"] == 81 and summary["n_points_pooled"] == 162
    assert summary["rms_rel_diff_pct"] < 1e-6


## The contract A file

@pytest.fixture(scope="module")
def solved():
    # A coarse propeller with the parametric airfoil: fast to compile
    prop = Propeller.default(inputs=Inputs(n_r=6, n_psi=8, coeff_method=0))
    case = ep.Case(prop)
    rows = ep.performance_table(case, [400.0, 500.0], [0.0, 5.0, 10.0])
    return prop, case, rows


def _write(path, prop, rows, **changes):
    kwargs = dict(prop_id="Test_prop", design_hash="0" * 64, author="A. Person",
                  generator=dict(name="Proptimiser", commit="a" * 40, dirty=False), prop=prop, rows=rows)
    kwargs.update(changes)
    return ep.write_prop_performance(path, **kwargs)


def _all_items(f):
    items = []
    f.visititems(lambda name, obj: items.append((name, obj)))
    return items


def test_contract_a_file_has_every_path_and_attribute(tmp_path, solved):
    prop, case, rows = solved
    assert [r["converged"] for r in rows] == [True] * 6
    check_coefficients(rows, prop.cfg.R_tip)
    assert rows[0]["CT"] > rows[1]["CT"] > rows[2]["CT"] > 0     # thrust falls with V_x
    assert all(1e3 < r["reynolds_75"] < 1e7 for r in rows)

    path = _write(tmp_path / "a.h5", prop, rows)
    with h5py.File(path, "r") as f:
        assert set(f.attrs) == ROOT_ATTRS
        assert f.attrs["schema"] == "prop-performance" and f.attrs["schema_version"] == 1
        assert f.attrs["prop_id"] == "Test_prop" and f.attrs["author"] == "A. Person"
        assert f.attrs["placeholder"] == False and len(f.attrs["design_sha256"]) == 64
        assert f.attrs["created_utc"].endswith("Z")
        for owner, key in ((f, "schema"), (f, "author"), (f["generator"], "commit")):
            string = h5py.check_string_dtype(owner.attrs.get_id(key).dtype)
            assert string.encoding == "utf-8" and string.length is None       # UTF-8 variable length
        assert dict(f["generator"].attrs) == {"name": "Proptimiser", "commit": "a" * 40, "dirty": False}
        assert set(f) == {"generator", "geometry", "performance", "uncertainty"}

        geo = f["geometry"]
        assert set(geo.attrs) == {"diameter_m", "blades", "hub_radius_m", "frame", "twist_sign"}   # no rotation
        assert math.isclose(geo.attrs["diameter_m"], 2 * prop.cfg.R_tip) and geo.attrs["blades"] == 2
        assert math.isclose(geo.attrs["hub_radius_m"], prop.blade.R_hub) and geo.attrs["twist_sign"] == 1
        assert "e_z" in geo.attrs["frame"]
        stations = geo["stations"]
        assert set(stations) == {"r_m", "chord_m", "twist_deg"}
        assert {k: stations[k].attrs["units"] for k in stations} == {"r_m": "m", "chord_m": "m", "twist_deg": "deg"}
        assert np.allclose(stations["r_m"][()], np.asarray(prop.r))
        assert np.allclose(stations["chord_m"][()], np.asarray(prop.c))
        assert np.allclose(stations["twist_deg"][()], np.degrees(np.asarray(prop.beta)))

        perf = f["performance"]
        assert set(perf) == set(ep.PERFORMANCE_COLUMNS)
        assert {k: perf[k].attrs["units"] for k in perf} == {k: u for k, (u, _) in ep.PERFORMANCE_COLUMNS.items()}
        assert all(perf[k].shape == (6,) for k in perf) and perf["converged"].dtype == bool
        assert set(perf.attrs) == {"rho_kgpm3", "nu_m2ps", "J_range", "rpm_values"}
        assert perf.attrs["rho_kgpm3"] == RHO and math.isclose(perf.attrs["nu_m2ps"], case.mu / RHO)
        assert np.allclose(perf.attrs["J_range"], [0.0, max(r["J"] for r in rows)])
        assert np.allclose(perf.attrs["rpm_values"], [400.0 * 30 / math.pi, 500.0 * 30 / math.pi])
        for k in ("CT", "CP", "CQ", "eta", "J", "V_mps", "rpm", "tip_mach", "reynolds_75"):
            assert np.allclose(perf[k][()], [r[k] for r in rows])

        # Units on every dataset, and no motor values or off-axis inflow anywhere
        for name, obj in _all_items(f):
            if isinstance(obj, h5py.Dataset):
                assert "units" in obj.attrs, name
            keys = [name.split("/")[-1], *obj.attrs]
            assert not any(k in ("I_0", "R_m", "kV", "P_max", "V_yz") for k in keys), name


def test_uncertainty_is_unavailable_with_a_basis(tmp_path, solved):
    prop, _, rows = solved
    with h5py.File(_write(tmp_path / "a.h5", prop, rows), "r") as f:
        unc = f["uncertainty"]
        assert {"CT", "CP"} <= set(unc)
        for item in unc.values():
            assert isinstance(item, h5py.Group) and set(item.attrs) <= UNCERTAINTY_FIELDS
            assert item.attrs["status"] == "unavailable" and item.attrs["scope"] == "result"
            assert len(item.attrs["basis"].strip()) > 20
            assert not {"sigma", "num_sigmas", "coverage", "lower", "upper"} & set(item.attrs)
        assert not {"CT_sigma", "CP_sigma"} & set(f["performance"])


def test_writer_refuses_a_dirty_tree_and_a_missing_author(tmp_path, solved):
    prop, _, rows = solved
    with pytest.raises(RuntimeError, match="dirty"):
        _write(tmp_path / "a.h5", prop, rows, generator=dict(name="Proptimiser", commit="a" * 40, dirty=True))
    for author in ("", "   ", None):
        with pytest.raises(RuntimeError, match="author"):
            _write(tmp_path / "a.h5", prop, rows, author=author)
    assert not (tmp_path / "a.h5").exists()


## Provenance

def _git(repo, *args):
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *args], cwd=repo, check=True,
                   capture_output=True)


@pytest.fixture
def repo(tmp_path):
    _git(tmp_path, "init", "-q")
    (tmp_path / "a.txt").write_text("one\n")
    _git(tmp_path, "add", "a.txt")
    _git(tmp_path, "commit", "-q", "-m", "one")
    return tmp_path


def test_a_dirty_tree_is_refused(repo):
    assert len(ep.require_clean(repo)) == 40
    (repo / "a.txt").write_text("two\n")
    with pytest.raises(RuntimeError, match="dirty"):
        ep.require_clean(repo)
    with pytest.raises(RuntimeError, match="dirty"):
        ep.export("currawong_v2", author="A. Person", repo=repo)
    _git(repo, "checkout", "--", "a.txt")
    (repo / "new.py").write_text("")               # an untracked file also makes the tree dirty
    with pytest.raises(RuntimeError, match="dirty"):
        ep.require_clean(repo)


def test_a_missing_author_is_refused(repo, monkeypatch):
    monkeypatch.setattr(ep, "git_user_name", lambda repo=None: None)
    with pytest.raises(RuntimeError, match="author"):
        ep.resolve_author(None, repo)
    with pytest.raises(RuntimeError, match="author"):
        ep.export("currawong_v2", author="  ", repo=repo)
    assert ep.resolve_author("A. Person", repo) == "A. Person"


def test_untracked_design_files_are_refused(repo):
    (repo / "d.yaml").write_text("x: 1\n")
    with pytest.raises(RuntimeError, match="not committed"):
        ep.require_tracked([repo / "d.yaml"], repo)
    ep.require_tracked([repo / "a.txt"], repo)


def test_design_hash_ignores_layout_and_follows_values(tmp_path):
    design, state = init.design_paths("currawong_v2")
    base = ep.design_sha256(design, state)
    text = design.read_text(encoding="utf-8")
    copy = tmp_path / "d.yaml"
    copy.write_bytes(("# another comment\n" + text).replace("\n", "\r\n").encode("utf-8"))
    assert ep.design_sha256(copy, state) == base
    copy.write_text(text.replace("hub_chord: 0.03", "hub_chord: 0.031"), encoding="utf-8")
    assert ep.design_sha256(copy, state) != base

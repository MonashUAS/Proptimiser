"""
Creo .ibl export: the core insert (export_to_creo_ibl with skin_thickness > 0) and offset_airfoil_profile.
"""
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pytest
from aerosandbox.geometry.airfoil.airfoil_families import get_kulfan_coordinates

from proptimiser import geometry, init

N_SIDE = 60


def _section(name="clarky"):
    """Unit-chord section coordinates (TE_upper -> LE -> TE_lower)."""
    cst = geometry.kulfan_from_name(name)
    return get_kulfan_coordinates(lower_weights=cst[:8], upper_weights=cst[8:16], leading_edge_weight=cst[16],
                                  TE_thickness=cst[17], n_points_per_side=N_SIDE)


def _thickness(x, y):
    le = np.argmin(x)
    xu, yu = x[:le + 1][::-1], y[:le + 1][::-1]
    order = np.argsort(x[le:])
    return np.max(yu - np.interp(xu, x[le:][order], y[le:][order]))


def _inside(px, py, x, y):
    """Crossing-number test: True for each point (px, py) inside the closed polygon (x, y)."""
    x1, y1, x2, y2 = x, y, np.roll(x, -1), np.roll(y, -1)
    crosses = ((y1[None] > py[:, None]) != (y2[None] > py[:, None])) & (
        px[:, None] < x1[None] + (py[:, None] - y1[None]) * (x2 - x1)[None] / np.where(y2 == y1, 1e-30, y2 - y1)[None])
    return crosses.sum(axis=1) % 2 == 1


def _read_sections(path):
    sections = Path(path).read_text().split("Begin section")[1:]
    return [np.array([[float(v) for v in line.split()[1:]] for line in s.splitlines() if line[:1].isdigit()])
            for s in sections]


def test_core_is_inside_the_skin_and_two_skins_thinner():
    chord, skin = 0.05, 0.0005
    coords = _section() * chord
    x, y = coords[:, 0], coords[:, 1]
    core_x, core_y = geometry.offset_airfoil_profile(x, y, skin, n_points_per_side=N_SIDE)

    assert len(core_x) == 2 * N_SIDE
    assert core_x[0] == core_x[-1] and core_y[0] == core_y[-1]                      # closed loop
    assert np.all(_inside(core_x[:-1], core_y[:-1], x, y))
    assert _thickness(core_x[:-1], core_y[:-1]) == pytest.approx(_thickness(x, y) - 2 * skin, abs=0.02 * skin)


def test_section_thinner_than_two_skins_has_no_core():
    chord = 0.01                                                    # clarky: about 1.2 mm thick
    coords = _section() * chord
    assert geometry.offset_airfoil_profile(coords[:, 0], coords[:, 1], 0.001, n_points_per_side=N_SIDE) is None


def _export(chords, skin, beta=0.2):
    n = len(chords)
    r = np.linspace(0.02, 0.2, n)
    coords = np.repeat(_section()[None], n, axis=0)
    tmp = tempfile.mkdtemp()
    skin_path = geometry.export_to_creo_ibl(r, np.array(chords), np.full(n, beta), coords, "skin", pitch_axis=0.25,
                                            out_dir=tmp)
    insert_path = geometry.export_to_creo_ibl(r, np.array(chords), np.full(n, beta), coords, "insert", pitch_axis=0.25,
                                              skin_thickness=skin, out_dir=tmp)
    return r, _read_sections(skin_path), _read_sections(insert_path)


def test_insert_keeps_the_core_stations_with_one_point_count():
    r, skins, inserts = _export([0.05, 0.05, 0.04, 0.01, 0.01], skin=0.001)

    assert len(skins) == 5 and len(inserts) == 3
    assert len({len(s) for s in inserts}) == 1
    for k, (skin, insert) in enumerate(zip(skins, inserts)):
        assert np.allclose(insert[:, 1], r[k] * 1000.0)
        # The twisted core lies inside the twisted skin of the same station
        assert np.all(_inside(insert[:-1, 0], insert[:-1, 2], skin[:, 0], skin[:, 2]))


def test_insert_refuses_a_broken_or_missing_core():
    with pytest.raises(ValueError, match="not continuous"):
        _export([0.05, 0.01, 0.05], skin=0.001)
    with pytest.raises(ValueError, match="No station has a core"):
        _export([0.01, 0.01], skin=0.001)


# Minimum trailing edge
def _prop(chord=0.05):
    inputs = init.Inputs(coeff_method=0, n_r=8)
    return init.Propeller.default(inputs=inputs).with_state(c=np.full(inputs.n_r, chord))


def _te_gap_mm(section):
    # The writer closes the loop: the last row repeats the first row, and the row before it is the lower trailing edge
    return np.linalg.norm(section[0] - section[-2])


def test_exported_trailing_edge_gap_is_at_least_the_minimum():
    # At a chord of 5 mm, the design gap of arad10 is 0.075 mm. So the tip takes the minimum. The hub planform makes the
    # root chord larger, so the root can keep a larger design gap
    prop = _prop(chord=0.005)
    r, c, beta, pitch_axis, coords, offsets = geometry.blade_export_geometry(prop, n_points_per_side=N_SIDE)
    path = geometry.export_to_creo_ibl(r, c, beta, coords, "mold", pitch_axis=pitch_axis, out_dir=tempfile.mkdtemp(),
                                       offsets=offsets)
    outboard = [s for s, r_i in zip(_read_sections(path), r) if r_i > prop.blade.R_hub]
    minimum_mm = geometry.TE_MIN_THICKNESS_M * 1000.0

    assert len(outboard) > 0
    assert min(_te_gap_mm(section) for section in outboard) >= minimum_mm - 1e-5
    assert _te_gap_mm(outboard[-1]) == pytest.approx(minimum_mm, abs=1e-5)


def test_larger_design_gap_does_not_change():
    coords = _section()
    coords_with_gap = geometry.enforce_te_thickness(coords, chord=0.05, te_min=1.0e-9)
    assert coords_with_gap is coords


def test_minimum_gap_is_the_cst_trailing_edge_term():
    chord = 0.02
    cst = geometry.kulfan_from_name("clarky")
    assert cst[17] * chord < geometry.TE_MIN_THICKNESS_M
    expected = get_kulfan_coordinates(lower_weights=cst[:8], upper_weights=cst[8:16], leading_edge_weight=cst[16],
                                      TE_thickness=geometry.TE_MIN_THICKNESS_M / chord, n_points_per_side=N_SIDE)
    assert np.allclose(geometry.enforce_te_thickness(_section(), chord), expected, atol=1e-12)


def test_minimum_gap_keeps_the_camber_line_and_adds_thickness():
    coords, thick = _section(), geometry.enforce_te_thickness(_section(), 0.02)
    assert np.allclose(coords[:, 1] + coords[::-1, 1], thick[:, 1] + thick[::-1, 1], atol=1e-12)
    assert np.allclose(coords[:, 0], thick[:, 0])
    assert np.all(thick[:N_SIDE, 1] >= coords[:N_SIDE, 1])


def test_stadium_hub_does_not_change():
    prop = _prop(chord=0.02)
    r, _, _, _, coords, _ = geometry.blade_export_geometry(prop, n_points_per_side=N_SIDE)
    hub = np.flatnonzero(r <= prop.blade.R_hub)
    assert len(hub) > 0
    for i in hub:
        assert coords[i, 0, 1] - coords[i, -1, 1] == pytest.approx(0.0, abs=1e-12)


def test_warning_names_a_trailing_edge_that_is_a_large_part_of_the_chord():
    with pytest.warns(UserWarning, match="trailing edge is more than 5%"):
        geometry.blade_export_geometry(_prop(chord=0.001), n_points_per_side=N_SIDE)
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*trailing edge is more than.*")
        geometry.blade_export_geometry(_prop(chord=0.05), n_points_per_side=N_SIDE)


def test_all_three_targets_export_with_the_minimum_gap():
    prop = _prop(chord=0.05)
    r, c, beta, pitch_axis, coords, offsets = geometry.blade_export_geometry(prop, n_points_per_side=N_SIDE)
    tmp = tempfile.mkdtemp()
    common = dict(pitch_axis=pitch_axis, out_dir=tmp, offsets=offsets)

    assert len(_read_sections(geometry.export_to_creo_ibl(r, c, beta, coords, "mold", **common))) == len(r)
    assert len(_read_sections(geometry.export_to_creo_ibl(r, c, beta, coords, "insert", skin_thickness=5.0e-4, **common))) > 0
    parting = geometry.export_parting_surface_ibl(r, c, beta, coords, prop.blade.r_foils, "parting", **common)
    assert Path(parting).read_text().count("Begin section") == len(r) + 2      # one pad row at each end


def test_writers_default_to_the_export_folder(monkeypatch, tmp_path):
    monkeypatch.setattr(init, "EXPORT_DIR", tmp_path)
    prop = _prop()
    r, c, beta, pitch_axis, coords, offsets = geometry.blade_export_geometry(prop, n_points_per_side=N_SIDE)
    assert geometry.export_to_creo_ibl(r, c, beta, coords, "mold", pitch_axis=pitch_axis, offsets=offsets).parent == tmp_path
    assert geometry.export_parting_surface_ibl(r, c, beta, coords, prop.blade.r_foils, "parting", pitch_axis=pitch_axis,
                                               offsets=offsets).parent == tmp_path


@pytest.mark.parametrize("module", ["geometry", "airfoil", "init"])
def test_each_module_loads_first(module):
    # init imports geometry at load time, and geometry reads its folders from init only inside its functions
    code = f"import proptimiser.{module}"
    assert subprocess.run([sys.executable, "-c", code], capture_output=True, text=True).returncode == 0


# Standard section layout
def test_standard_export_has_fixed_counts_and_shared_joint_sections(tmp_path):
    prop = _prop(chord=0.05)
    paths, stations = geometry.export_standard_ibl(prop, "std", out_dir=tmp_path, n_points_per_side=N_SIDE)
    files = [_read_sections(path) for path in paths]
    assert [path.name for path in paths] == [f"std_{name}.ibl" for name, _ in geometry.STANDARD_SECTIONS]
    assert [len(sections) for sections in files] == [count for _, count in geometry.STANDARD_SECTIONS]
    # The two files at a joint hold the same section, at r_hub and at r_aero
    np.testing.assert_allclose(files[0][-1], files[1][0], atol=1e-6)
    np.testing.assert_allclose(files[1][-1], files[2][0], atol=1e-6)
    r_hub, r_aero = prop.blade.r_foils[1], prop.blade.r_foils[2]
    assert [float(r[0]) for _, r in stations.zones] == pytest.approx([0.0, r_hub, r_aero])
    for _, r in stations.zones:
        assert np.all(np.diff(r) > 0.0)


def test_another_design_keeps_the_same_section_counts(tmp_path):
    counts = []
    for chord in (0.02, 0.06):
        paths, _ = geometry.export_standard_ibl(_prop(chord=chord), f"c{chord}", out_dir=tmp_path, n_points_per_side=N_SIDE)
        counts.append([len(_read_sections(path)) for path in paths])
    assert counts[0] == counts[1]


def test_tip_rule_ends_the_blade_where_the_chord_falls_to_the_limit():
    inputs = init.Inputs(coeff_method=0, n_r=12)
    base = init.Propeller.default(inputs=inputs)
    r = np.asarray(base.r)
    chord = np.where(r < 0.9 * r[-1], 0.05, 0.05 * (r[-1] - r) / (0.1 * r[-1]) + 1.0e-4)
    prop = base.with_state(c=chord)
    r_tip = geometry.tip_radius(prop)
    assert r[0] < r_tip < r[-1]
    from scipy.interpolate import PchipInterpolator
    r_design, c_design, _ = geometry._design_planform(prop)
    assert float(PchipInterpolator(r_design, c_design)(r_tip)) == pytest.approx(geometry.TIP_MIN_CHORD_FRACTION * 0.05, rel=1e-3)
    assert geometry.standard_stations(prop).zones[-1][1][-1] == pytest.approx(r_tip)


def test_unswept_arc_length_is_the_radius():
    prop = _prop()
    r = np.linspace(0.0, float(prop.blade.R_tip), 7)
    np.testing.assert_allclose(geometry.pitch_axis_arc_length(r, prop.blade), r, atol=1e-9)

"""OML sections and the STEP blade solid. The solid tests need the cad extra (cadquery-ocp)."""
import numpy as np
import pytest

from proptimiser import geometry, init

DESIGNS = ["APC_20x10E", "VTOLCURRAWONG_V0", "MAD_FLUXER_22x6_6_jung"]


@pytest.fixture(scope="module", params=DESIGNS)
def design(request):
    prop = init.Propeller.load(request.param)
    return prop, geometry.from_propeller(prop)


def section_volume_mm3(surface):
    """Trapezoid integral along Y of the section areas (shoelace in the X-Z plane)."""
    points = surface.section_points() * geometry.M_TO_MM
    x, z = points[..., 0], points[..., 2]
    area = 0.5 * np.abs(np.sum(x * np.roll(z, -1, 1) - np.roll(x, -1, 1) * z, axis=1))
    return np.trapezoid(area, surface.origin[:, 1] * geometry.M_TO_MM)


def test_smootherstep_ends():
    t = np.array([-1.0, 0.0, 0.5, 1.0, 2.0])
    assert np.allclose(geometry.smootherstep(t), [0.0, 0.0, 0.5, 1.0, 1.0])


def test_sections_have_trailing_edge_gap(design):
    _, surface = design
    gap_m = (surface.coords[:, 0, 1] - surface.coords[:, -1, 1]) * surface.chord
    assert np.all(gap_m >= geometry.TE_MIN_THICKNESS_M - 1e-12)
    assert geometry.leading_edge_index(surface) == np.argmin(surface.coords[0, :, 0])


def test_stations_are_standard(design):
    prop, surface = design
    zones = geometry.standard_stations(prop).zones
    expected = np.unique(np.concatenate([r for _, r in zones]))
    # A swept station has its pitch-axis point off the Y axis, at the station radius from the spin axis
    assert np.allclose(np.hypot(surface.origin[:, 0], surface.origin[:, 1]), expected)


def test_blade_solid(design):
    pytest.importorskip("OCP")
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib

    _, surface = design
    faces = geometry.blade_faces(surface)
    assert tuple(faces) == geometry.BLADE_FACE_NAMES
    solid = geometry.blade_solid(surface)
    assert BRepCheck_Analyzer(solid).IsValid()
    assert len(geometry.solids(solid)) == 1
    # On 2026-11-08 the volume was within 2.1 % of the section integral. The integral is linear between stations
    assert geometry.shape_volume_mm3(solid) == pytest.approx(section_volume_mm3(surface), rel=0.03)
    # The skin must not swing outside the sections (the old loft went 50 mm outside)
    box = Bnd_Box()
    BRepBndLib.AddOptimal_s(solid, box, False, False)
    points = surface.section_points() * geometry.M_TO_MM
    low, high = box.CornerMin(), box.CornerMax()
    assert np.allclose([low.X(), low.Y(), low.Z()], points.reshape(-1, 3).min(axis=0), atol=1.0)
    assert np.allclose([high.X(), high.Y(), high.Z()], points.reshape(-1, 3).max(axis=0), atol=1.0)


@pytest.mark.parametrize("n_blades", [2, 3, 4, 5])
def test_default_hub_covers_roots(n_blades):
    prop = init.Propeller.load("APC_20x10E")
    surface = geometry.from_propeller(prop)
    radius, z_low, z_high = geometry.default_hub(prop, surface, n_blades)
    r_hub_mm = float(prop.blade.r_foils[1]) * geometry.M_TO_MM
    half_width = 0.5 * float(np.max(surface.chord[surface.origin[:, 1] * geometry.M_TO_MM <= r_hub_mm + 1e-9])) * geometry.M_TO_MM
    crossing = half_width / np.sin(np.pi / n_blades) if n_blades >= 3 else half_width
    assert radius >= r_hub_mm
    assert radius >= geometry.HUB_COVER_MARGIN * crossing - 1e-9
    assert z_high > z_low


def test_export_blade_step(tmp_path):
    pytest.importorskip("OCP")
    prop = init.Propeller.load("APC_20x10E")
    result = geometry.export_propeller_step(prop, tmp_path, "test", n_blades=3, rotor=False)
    text = result["blade"].read_text(encoding="utf-8", errors="replace")
    assert result["blade"].name == "test_blade.step"
    assert "TEST_BLADE" in text
    assert "MILLI" in text
    assert result["rotor"] is None and result["rotor_error"] is None


def test_a_round_end_gets_the_gap_by_a_cut_and_does_not_turn_back():
    from proptimiser import geometry

    # A closed section with a round end: half thickness 0.05 sqrt(1 - x) near the edge, and a camber that falls to the edge
    x = np.concatenate([(1.0 - np.cos(np.linspace(0.0, np.pi, 100))[::-1]) / 2.0, (1.0 - np.cos(np.linspace(0.0, np.pi, 100)))[1:] / 2.0])
    le = 99
    half = 0.08 * np.sqrt(np.clip(x * (1.0 - x), 0.0, None))
    camber = 0.03 * np.sin(np.pi * x) - 0.02 * x
    y = np.where(np.arange(len(x)) <= le, camber + half, camber - half)
    closed = np.column_stack([x, y])
    chord = 0.03
    added = geometry.enforce_te_thickness(closed, chord)
    cut = geometry.cut_to_gap(closed, chord)
    assert geometry.te_turn_deg(added) > geometry.TE_TURN_LIMIT_DEG
    assert geometry.te_turn_deg(cut) < geometry.te_turn_deg(added)
    assert (cut[0, 1] - cut[-1, 1]) * chord == pytest.approx(geometry.TE_MIN_THICKNESS_M, rel=1.0e-3)
    assert cut.shape == closed.shape and int(np.argmin(cut[:, 0])) == le
    assert np.array_equal(cut[:, 0], closed[:, 0])


def test_cad_export_writes_the_creo_files(tmp_path):
    pytest.importorskip("OCP")
    from proptimiser import cad_export

    paths = cad_export.export("APC_20x10E", tmp_path, ibl=True)
    names = [p.name for p in paths]
    assert names[:2] == ["APC_20x10E_blade.step", "APC_20x10E_blade_faces.json"]
    assert len(names) == 2 + len(geometry.STANDARD_SECTIONS) and all(p.exists() for p in paths)

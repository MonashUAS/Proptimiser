"""
Creo .ibl export: the core insert (export_to_creo_ibl with skin_thickness > 0) and offset_airfoil_profile.
"""
import tempfile
from pathlib import Path

import numpy as np
import pytest
from aerosandbox.geometry.airfoil.airfoil_families import get_kulfan_coordinates

from proptimiser import geometry

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

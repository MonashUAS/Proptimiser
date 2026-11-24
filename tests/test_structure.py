"""
test_structure.py

Checks for the structural-load evaluation: the sweep and rake geometry and its aerodynamics here, then section
properties, loads and stress as those modules arrive.

    python test_structure.py

or with pytest:

    pytest -q test_structure.py
"""
import dataclasses
import tempfile
import warnings
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from proptimiser import init
from proptimiser.init import Inputs, BladeSpec, Propeller
from proptimiser import geometry


def _swept_blade(scale=1.0, **kwargs):
    # A blade with sweep and rake growing linearly to the tip (two control points: PCHIP is then linear)
    base = BladeSpec(**kwargs)
    return dataclasses.replace(base, offset_r=(base.R_hub, base.R_tip), offset_y=(0.0, scale * 0.02),
                               offset_z=(0.0, scale * 0.01))


# ---------------------------------------------------------------------------
# Sweep and rake geometry
# ---------------------------------------------------------------------------

def _blade_vectors(frame):
    # The section axes as vectors on the blade frame (e_r, e_t, e_z), from their components on the air basis
    sin_g, cos_g = frame.sin_g, frame.cos_g
    zero, one = jnp.zeros_like(sin_g), jnp.ones_like(sin_g)
    rho = jnp.stack([cos_g, sin_g, zero], -1)
    tau = jnp.stack([-sin_g, cos_g, zero], -1)
    e_z = jnp.stack([zero, zero, one], -1)
    pick = lambda x: x[..., None]
    s = pick(frame.s_rho) * rho + pick(frame.s_tau) * tau + pick(frame.s_z) * e_z
    c = pick(frame.c_rho) * rho + pick(frame.c_t) * (-tau) + pick(frame.c_a) * (-e_z)
    n = pick(frame.n_rho) * rho + pick(frame.n_a) * (-e_z)
    return s, c, n


def test_section_frame_orthonormal_and_pins_handedness():
    r = np.linspace(0.05, 0.3, 11)
    y, z = 0.3 * r ** 2, 0.2 * r ** 2 - 0.01 * r
    frame = geometry.section_frame(jnp.asarray(r), jnp.asarray(np.column_stack([y, z, 0.6 * r, 0.4 * r - 0.01])))
    s, c, n = _blade_vectors(frame)

    dot = lambda a, b: np.asarray(jnp.sum(a * b, -1))
    for a in (s, c, n):
        assert np.allclose(dot(a, a), 1.0, atol=1e-14)
    assert np.allclose(dot(s, c), 0.0, atol=1e-14) and np.allclose(dot(s, n), 0.0, atol=1e-14)
    assert np.allclose(dot(c, n), 0.0, atol=1e-14)
    assert np.allclose(np.asarray(jnp.cross(s, c)), np.asarray(n), atol=1e-14)         # n = s x c
    assert np.all(np.asarray(frame.c_t) > 0)

    # Unswept: s = e_r, the chord at zero twist runs against the blade motion (c = -e_t), and the normal that positive
    # twist rotates the chord toward is downstream (n = -e_z)
    flat = geometry.section_frame(jnp.asarray(r), jnp.zeros((len(r), 4)))
    s0, c0, n0 = _blade_vectors(flat)
    assert np.allclose(np.asarray(s0), [1, 0, 0]) and np.allclose(np.asarray(c0), [0, -1, 0])
    assert np.allclose(np.asarray(n0), [0, 0, -1]) and np.allclose(np.asarray(flat.arc), 1.0)
    assert np.allclose(np.asarray(jnp.cross(s0, c0)), np.asarray(n0))


def test_section_frame_follows_the_pitch_axis_curve():
    # s * ds/dr is the derivative of the pitch-axis position p = (sqrt(r^2 - y^2), y, z) on the blade frame
    r = np.linspace(0.05, 0.3, 11)
    y, z = 0.3 * r ** 2, 0.2 * r ** 2 - 0.01 * r
    dy, dz = 0.6 * r, 0.4 * r - 0.01
    frame = geometry.section_frame(jnp.asarray(r), jnp.asarray(np.column_stack([y, z, dy, dz])))
    s, _, _ = _blade_vectors(frame)
    dp = np.stack([(r - y * dy) / np.sqrt(r ** 2 - y ** 2), dy, dz], -1)
    assert np.allclose(np.asarray(s) * np.asarray(frame.arc)[:, None], dp, atol=1e-12)


def test_section_frame_hand_cases():
    r = jnp.linspace(0.05, 0.3, 7)
    zeros = jnp.zeros_like(r)

    # a constant tangential offset: the pitch axis is parallel to e_r, and the arc is longer than the polar radius
    y0 = 0.02
    frame = geometry.section_frame(r, jnp.stack([zeros + y0, zeros, zeros, zeros], -1))
    s, _, _ = _blade_vectors(frame)
    assert np.allclose(np.asarray(s), [1, 0, 0], atol=1e-14)
    assert np.allclose(np.asarray(frame.arc), np.asarray(r) / np.sqrt(np.asarray(r) ** 2 - y0 ** 2), atol=1e-14)

    # a straight line through the hub centre, y = m r: no spanwise turn, arc = 1
    m = 0.1
    frame = geometry.section_frame(r, jnp.stack([m * r, zeros, zeros + m, zeros], -1))
    assert np.allclose(np.asarray(frame.s_tau), 0.0, atol=1e-14) and np.allclose(np.asarray(frame.arc), 1.0, atol=1e-14)
    assert np.allclose(np.asarray(frame.sin_g), m)


def test_rake_turns_the_normal_velocity():
    # z = m r (a cone toward the thrust side): the normal velocity is (m u_rho + u_a) / sqrt(1 + m^2) and the chordwise
    # velocity is unchanged
    from proptimiser import airflow

    m = 0.15
    base = BladeSpec()
    blade = dataclasses.replace(base, offset_r=(base.R_hub, base.R_tip), offset_z=(m * base.R_hub, m * base.R_tip))
    p = Propeller.default(inputs=Inputs(coeff_method=0, n_r=6), blade=blade).condition(0)
    assert p.cfg.swept and np.allclose(np.asarray(p.offsets[:, 3]), m) and np.allclose(np.asarray(p.offsets[:, 0]), 0.0)

    rng = np.random.default_rng(3)
    u_a, u_t, u_rho = (jnp.asarray(rng.uniform(1.0, 20.0, p.psi_pt.shape)) for _ in range(3))
    u_n, u_c = airflow.swept_section_velocity(u_a, u_t, u_rho, p)
    assert np.allclose(np.asarray(u_n), np.asarray(m * u_rho + u_a) / np.sqrt(1 + m ** 2), atol=1e-13)
    assert np.allclose(np.asarray(u_c), np.asarray(u_t), atol=1e-13)


def test_offset_interpolation_and_guards():
    blade = _swept_blade(r_foils=(0.0, 0.025, 0.05, 0.15, 0.275))
    r = np.array([0.0, 0.0125, 0.025, 0.1, 0.275, 0.3])
    off = np.asarray(geometry.station_offsets(r, blade))
    assert off.shape == (6, 4)
    assert np.allclose(off[:, 0], [0.0, 0.0, 0.0, 0.02 * (0.1 - 0.025) / 0.25, 0.02, 0.02])      # held outboard
    assert np.allclose(off[:, 2], [0.0, 0.0, 0.08, 0.08, 0.08, 0.0])                              # slope of the line, 0 held
    assert not np.any(np.asarray(geometry.station_offsets(r, BladeSpec())))

    # a root offset tapers to zero at the hub centre: a straight radial line inboard of the first control point
    root = dataclasses.replace(BladeSpec(), offset_r=(0.025, 0.275), offset_y=(0.005, 0.02))
    off = np.asarray(geometry.station_offsets(np.array([0.0125, 0.025]), root))
    assert np.allclose(off[:, 0], [0.0025, 0.005]) and np.allclose(off[0, 2], 0.2)

    for bad in (dict(offset_r=(0.05,), offset_y=(0.01,)), dict(offset_r=(0.1, 0.05), offset_y=(0.0, 0.01)),
                dict(offset_r=(0.05, 0.1), offset_y=(0.0, 0.01, 0.02)), dict(offset_y=(0.01,)), dict(fill_factor=1.5),
                dict(mat_density=-1.0)):
        try:
            BladeSpec(**bad)
        except ValueError:
            continue
        raise AssertionError(f"BladeSpec accepted {bad}")

    # a sweep offset that reaches the polar radius is refused, and so is a pitch axis that runs (nearly) tangentially
    try:
        Propeller.default(inputs=Inputs(coeff_method=0, n_r=6), blade=BladeSpec(offset_r=(0.025, 0.275), offset_y=(0.0, 0.26)))
    except ValueError as e:
        assert "of the radius" in str(e)
    else:
        raise AssertionError("Propeller accepted a sweep offset of 0.95 of the radius")
    tangential = dataclasses.replace(BladeSpec(), offset_r=(0.2, 0.21), offset_y=(0.0, 0.08))
    try:
        geometry.station_offsets(np.array([0.205]), tangential)
    except ValueError as e:
        assert "turns" in str(e)
    else:
        raise AssertionError("station_offsets accepted a pitch axis that runs tangentially")


# ---------------------------------------------------------------------------
# Swept aerodynamics
# ---------------------------------------------------------------------------

def test_swept_bemt_converges_and_scales_quadratically():
    from proptimiser import propmodel

    inputs = Inputs(coeff_method=0, n_r=6, V_yz=[0.0, 0.0])
    thrust = {}
    for scale in (0.0, 1.0, 2.0):
        p = Propeller.default(inputs=inputs, blade=_swept_blade(scale)).condition(0)
        assert p.cfg.swept == (scale > 0)
        V_i, V_i_0 = propmodel.induced_velocity(p)
        lam = V_i_0[p.cfg.solve_rows] / propmodel.get_velocity_scales(p)
        residual = np.abs(np.asarray(propmodel.bemt_residuals(lam, p)))[:, np.asarray(propmodel.aero_zone(p))]
        assert residual.max() <= p.cfg.V_i_0_eps
        thrust[scale] = float(propmodel.thrust(V_i, p))
    # sweep acts through the cosine of the angle between the flow and the normal: second order in the offset
    change = lambda scale: thrust[scale] - thrust[0.0]
    assert change(1.0) != 0 and np.isclose(change(2.0) / change(1.0), 4.0, rtol=0.05)


def test_station_subset_slices_offsets():
    # With no Newton iterations every station goes to the bracketed fallback, which solves copies of the blade holding
    # only some stations: their offsets must follow them
    from proptimiser import propmodel

    p = Propeller.default(inputs=Inputs(coeff_method=0, n_r=8), blade=_swept_blade(1.0)).condition(1)
    _, V_newton = propmodel.induced_velocity(p)
    bracketed = dataclasses.replace(p, cfg=dataclasses.replace(p.cfg, V_i_0_max_iter=0))
    _, V_bracketed = propmodel.induced_velocity(bracketed)
    assert np.allclose(V_bracketed, V_newton, atol=1e-6)

    idx = jnp.array([5, 2])
    sub = propmodel._station_subset(p, idx)
    assert np.array_equal(np.asarray(sub.offsets), np.asarray(p.offsets)[[5, 2]])


def test_sectional_loads_integrate_to_the_blade_element_sums():
    from proptimiser import propmodel

    for blade in (BladeSpec(), _swept_blade(1.0)):
        p = Propeller.default(inputs=Inputs(coeff_method=0, n_r=8), blade=blade).condition(1)
        V_i, _ = propmodel.induced_velocity(p)
        loads = propmodel.sectional_forces(V_i, p)
        assert loads.f.shape == (8, 12, 3) and loads.m.shape == (8, 12, 3)
        aero = np.asarray(propmodel.aero_zone(p))
        assert not np.any(np.asarray(loads.f)[~aero]) and not np.any(np.asarray(loads.m)[~aero])

        N = float(p.N[0])
        f = np.asarray(loads.f)
        assert np.allclose((N * f[:, :, 2].mean(axis=1))[aero], np.asarray(propmodel.dT_blade(V_i, p))[aero], atol=1e-10)

        # the moment of the blade-frame force about the rotation axis is -r times the in-plane force dH of the BEMT
        frame = geometry.section_frame(p.r, p.offsets)
        x, y = np.asarray(p.r * frame.cos_g), np.asarray(p.r * frame.sin_g)
        torque_z = x[:, None] * f[:, :, 1] - y[:, None] * f[:, :, 0]
        dQ = np.asarray(propmodel.dQ_blade(V_i, p))
        assert np.allclose((N * torque_z.mean(axis=1))[aero], -dQ[aero], atol=1e-10)


def test_pitching_moment_acts_about_the_span():
    from proptimiser import propmodel

    p = Propeller.default(inputs=Inputs(coeff_method=0, n_r=8), blade=_swept_blade(1.0)).condition(0)
    V_i, _ = propmodel.induced_velocity(p)
    loads = propmodel.sectional_forces(V_i, p)
    frame = geometry.section_frame(p.r, p.offsets)
    s = np.stack([np.asarray(frame.cos_g * frame.s_rho - frame.sin_g * frame.s_tau),
                  np.asarray(frame.sin_g * frame.s_rho + frame.cos_g * frame.s_tau), np.asarray(frame.s_z)], -1)
    m = np.asarray(loads.m)
    aero = np.asarray(propmodel.aero_zone(p))
    magnitude = np.linalg.norm(m, axis=-1)
    along = np.einsum("rpk,rk->rp", m, s)
    assert np.allclose(along[aero], magnitude[aero] * np.sign(np.asarray(loads.C_M))[aero], atol=1e-12)
    # m = 0.5 rho V_R^2 c^2 C_M ds/dr
    expected = 0.5 * p.rho * np.asarray(loads.V_R) ** 2 * np.asarray(p.c)[:, None] ** 2 * np.asarray(loads.C_M) * np.asarray(frame.arc)[:, None]
    assert np.allclose(along[aero], expected[aero], atol=1e-12)


# ---------------------------------------------------------------------------
# CAD, APC and files
# ---------------------------------------------------------------------------

def test_cad_offsets_follow_frame():
    r = np.array([0.0, 0.05, 0.1, 0.2])
    zero = np.zeros((4, 4))
    x0, y0, z0 = geometry.section_origin(r, zero)
    assert np.array_equal(x0, 0 * r) and np.array_equal(y0, r) and np.array_equal(z0, 0 * r)

    offsets = np.array([[0.0, 0.0, 0.0, 0.0], [0.004, -0.002, 0.1, 0.0], [0.01, 0.003, 0.1, 0.0], [0.03, 0.01, 0.1, 0.0]])
    x0, y0, z0 = geometry.section_origin(r, offsets)
    assert np.allclose(x0, -offsets[:, 0]) and np.allclose(z0, offsets[:, 1])        # +y_sw toward the leading edge: -X
    assert np.allclose(x0 ** 2 + y0 ** 2, r ** 2)                                      # the polar radius is kept

    prop = Propeller.default(inputs=Inputs(coeff_method=0, n_r=6), blade=_swept_blade(1.0))
    r_cad, c, beta, pitch_axis, coords, cad_offsets = geometry.blade_export_geometry(prop, n_points_per_side=20)
    assert r_cad[0] == 0.0 and len(r_cad) == 7 and cad_offsets.shape == (7, 4)
    assert np.allclose(cad_offsets[1:], np.asarray(prop.offsets))

    with tempfile.TemporaryDirectory() as tmp:
        path = geometry.export_to_creo_ibl(r_cad, c, beta, coords, "swept", pitch_axis=pitch_axis, out_dir=tmp, offsets=cad_offsets)
        sections = [s for s in Path(path).read_text().split("Begin section")[1:]]
        _, y_expected, _ = geometry.section_origin(r_cad, cad_offsets)
        for section, y in zip(sections, y_expected):
            rows = [line.split() for line in section.splitlines() if line[:1].isdigit()]
            assert np.allclose([float(row[2]) for row in rows], y * 1000.0, atol=1e-3)


def test_cad_twist_puts_the_leading_edge_on_the_thrust_side():
    # The structural frame takes e_z (thrust) as CAD +Z. The CAD code rotates a section by -beta (radians) about the
    # pitch axis: a positive twist (nose-up) must lift the leading edge to +Z and drop the trailing edge
    import aerosandbox as asb
    from aerosandbox.geometry.airfoil.airfoil_families import get_kulfan_coordinates

    cst = geometry.kulfan_from_name("arad10")
    coords = get_kulfan_coordinates(lower_weights=cst[:8], upper_weights=cst[8:16], leading_edge_weight=cst[16],
                                    TE_thickness=cst[17], n_points_per_side=60)
    beta, pa = 0.3, 0.25
    rotated = asb.Airfoil(coordinates=coords).rotate(angle=-beta, x_center=pa, y_center=0.0).coordinates
    le, te = rotated[np.argmin(rotated[:, 0])], rotated[np.argmax(rotated[:, 0])]
    assert np.isclose(le[1], pa * np.sin(beta), atol=0.01) and le[1] > 0.05
    assert te[1] < -0.15


def test_apc_sweep_root_zero_and_sign():
    # APC SWEEP and RAKE are those of the leading edge of the mould parting line; the pitch axis lies 0.25 c behind it
    # along the chord line, which is twisted by the section twist
    inch = 0.0254
    apc = dict(r=jnp.array([0.0, 1.0, 2.0, 3.0]), chord=jnp.array([1.0, 0.8, 0.8, 0.5]),
               twist_deg=jnp.array([0.0, 30.0, 20.0, 10.0]), sweep=jnp.array([0.5, 0.5, 0.6, 0.3]),
               rake=jnp.array([0.0, 0.2, 0.3, 0.2]))
    r, y, z = geometry.apc_offsets(apc, 0.5 * inch, 0.75 * inch)       # stations at 1, 2, 3 in are in the aero zone
    assert np.allclose(r, np.array([1.0, 2.0, 3.0]) * inch)
    c, twist = np.array([0.8, 0.8, 0.5]) * inch, np.radians([30.0, 20.0, 10.0])
    y_expected = np.array([0.5, 0.6, 0.3]) * inch - 0.25 * c * np.cos(twist)
    z_expected = np.array([0.2, 0.3, 0.2]) * inch - 0.25 * c * np.sin(twist)
    assert y[0] == 0.0 and z[0] == 0.0
    assert np.allclose(y, y_expected - y_expected[0]) and np.allclose(z, z_expected - z_expected[0])

    path = Path(__file__).resolve().parent / "apc_prop_data" / "propgeom_data" / "10x6-PERF.PE0"
    if path.exists():
        blade = BladeSpec(r_foils=(0.0, 0.025, 0.05, 0.09, 0.127))       # inside the 5 in radius of the 10x6
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plain = Propeller.from_apc(path, inputs=Inputs(coeff_method=0, n_r=8), blade=blade)
            swept = Propeller.from_apc(path, inputs=Inputs(coeff_method=0, n_r=8), blade=blade, use_apc_sweep=True)
        assert not plain.blade.swept and swept.blade.swept
        assert swept.blade.offset_y[0] == 0.0 and swept.blade.offset_z[0] == 0.0
        assert np.allclose(np.asarray(plain.c), np.asarray(swept.c))


def test_offsets_and_material_json():
    prop = Propeller.default(inputs=Inputs(coeff_method=0, n_r=6), blade=_swept_blade(1.0, mat_density=1500.0, fill_factor=0.6))
    with tempfile.TemporaryDirectory() as tmp:
        back = Propeller.load(prop.save(Path(tmp) / "swept"))
    assert back.blade == prop.blade and back.blade.swept and back.cfg.swept
    assert back.blade.mat_density == 1500.0 and back.blade.fill_factor == 0.6
    assert np.array_equal(np.asarray(back.offsets), np.asarray(prop.offsets))


# ---------------------------------------------------------------------------
# Section properties
# ---------------------------------------------------------------------------

def _polygon_properties(x, z):
    # Area, centroid and second moments about the centroid of a closed polygon (shoelace formulas)
    x1, z1 = np.roll(x, -1), np.roll(z, -1)
    cross = x * z1 - x1 * z
    signed = 0.5 * cross.sum()
    sign, A = np.sign(signed), abs(signed)
    cx, cz = sign * ((x + x1) * cross).sum() / (6 * A), sign * ((z + z1) * cross).sum() / (6 * A)
    I_xx = sign * ((z ** 2 + z * z1 + z1 ** 2) * cross).sum() / 12 - A * cz ** 2
    I_zz = sign * ((x ** 2 + x * x1 + x1 ** 2) * cross).sum() / 12 - A * cx ** 2
    I_xz = sign * ((x * z1 + 2 * x * z + 2 * x1 * z1 + x1 * z) * cross).sum() / 24 - A * cx * cz
    return A, cx, cz, I_xx, I_zz, I_xz


def test_strip_properties_exact_shapes():
    from proptimiser import structure

    x, w = structure.GAUSS_X, structure.GAUSS_W
    assert np.isclose(w.sum(), 1.0, rtol=0, atol=1e-14)

    # rectangle of unit length and height h
    h = 0.1
    A, xc, zc, Ixx, Izz, Ixz = (float(v) for v in structure.strip_properties(
        jnp.asarray(x), jnp.asarray(w), jnp.full(x.shape, h / 2), jnp.full(x.shape, -h / 2)))
    assert np.allclose([A, xc, zc, Ixx, Izz, Ixz], [h, 0.5, 0.0, h ** 3 / 12, h / 12, 0.0], rtol=1e-12, atol=1e-14)

    # right triangle with its right angle at the leading edge: legs 1 (chordwise) and h (normal)
    A, xc, zc, Ixx, Izz, Ixz = (float(v) for v in structure.strip_properties(
        jnp.asarray(x), jnp.asarray(w), jnp.asarray(h * (1 - x)), jnp.zeros(x.shape)))
    assert np.allclose([A, xc, zc, Ixx, Izz, Ixz], [h / 2, 1 / 3, h / 3, h ** 3 / 36, h / 36, -h ** 2 / 72], rtol=1e-12, atol=1e-14)


def test_cst_properties_match_a_fine_polygon():
    from proptimiser import structure
    from aerosandbox.geometry.airfoil.airfoil_families import get_kulfan_coordinates

    prop = Propeller.default(inputs=Inputs(coeff_method=0, n_r=8), c=0.04)
    p = prop.condition(0)
    props = structure.section_properties(p, prop.material())
    pitch_axis = np.asarray(geometry.pitch_axis_fraction(p.r, p.cfg.R_hub, p.cfg.R_aero))
    cst = np.asarray(geometry.cst_at_stations(p.r, p.r_cst, p.cst))

    for i in (1, 4, 7):                                     # blend zone and aerodynamic zone
        c = float(p.c[i])
        coords = get_kulfan_coordinates(lower_weights=cst[i, :8], upper_weights=cst[i, 8:16], leading_edge_weight=cst[i, 16],
                                        TE_thickness=cst[i, 17], n_points_per_side=3000)
        A, cx, cz, Ixx, Izz, Ixz = _polygon_properties(coords[:, 0], coords[:, 1])
        assert np.isclose(float(props.area[i]), A * c ** 2, rtol=1e-6)
        assert np.isclose(float(props.x_c[i]), c * (cx - pitch_axis[i]), rtol=0, atol=1e-6 * c)
        assert np.isclose(float(props.z_c[i]), c * cz, rtol=0, atol=1e-6 * c)
        for got, ref in ((props.I_xx, Ixx), (props.I_zz, Izz), (props.I_xz, Ixz)):
            assert np.isclose(float(got[i]), ref * c ** 4, rtol=1e-5)

        # the outline is the same polygon, taken about the centroid
        outline = np.asarray(props.outline[i])
        A_o, cx_o, cz_o, *_ = _polygon_properties(outline[:, 0], outline[:, 1])
        assert np.isclose(A_o, float(props.area[i]), rtol=1e-3) and abs(cx_o) < 1e-3 * c and abs(cz_o) < 1e-3 * c
        assert np.isclose(float(props.t_max[i]), c * float(geometry.max_thickness(jnp.asarray(cst[i]))), rtol=1e-3)


def test_torsion_constant_of_an_ellipse():
    from proptimiser import structure

    # thin ellipse: half-length a = 1/2, half-thickness b = 0.06 (t/c = 0.12): x = (1 - cos t) / 2, y = 2 b sqrt(x (1 - x))
    b, x, w = 0.06, structure.GAUSS_X, structure.GAUSS_W
    y = 2 * b * np.sqrt(x * (1 - x))
    A, _, _, Ixx, Izz, _ = structure.strip_properties(jnp.asarray(x), jnp.asarray(w), jnp.asarray(y), jnp.asarray(-y))
    a = 0.5
    assert np.isclose(float(A), np.pi * a * b, rtol=1e-12)
    assert np.isclose(float(Ixx), np.pi * a * b ** 3 / 4, rtol=1e-12) and np.isclose(float(Izz), np.pi * a ** 3 * b / 4, rtol=1e-12)
    exact = np.pi * a ** 3 * b ** 3 / (a ** 2 + b ** 2)
    J = float(structure.torsion_constant(A, Ixx, Izz))
    assert np.isclose(J, exact, rtol=0.03) and J < exact


def test_hub_stadium_section_fit():
    from proptimiser import structure

    # The hub section is a stadium fitted by a CST: its area follows the analytic stadium to a fraction of a percent
    prop = Propeller.default(inputs=Inputs(coeff_method=0, n_r=8), c=0.04)
    p = prop.condition(0)
    t = prop.blade.hub_thickness_ratio
    area = float(structure.section_properties(p, prop.material()).area[0]) / float(p.c[0]) ** 2
    assert np.isclose(area, t * (1 - t) + np.pi * t ** 2 / 4, rtol=3e-3)


def test_section_area_matches_apc_outboard():
    # The APC 10x6 geometry file lists the cross-section area of its Clark-Y sections, scaled to a thickness ratio. The
    # CST section of the same thickness holds a constant 94% of it outboard of 3 in (the APC section is a fuller
    # Clark-Y). Inboard, the moulded root blend is not a CST airfoil, and the rounded tip (chord below 0.25 in) is not
    # either, so the model sits lower there.
    from proptimiser import structure

    path = Path(__file__).resolve().parent / "apc_prop_data" / "propgeom_data" / "10x6-PERF.PE0"
    if not path.exists():
        return
    rows, in_table = [], False
    for line in path.read_text().splitlines():
        line = line.strip()
        in_table = in_table or line.startswith("0.0000")
        if in_table:
            if not line:
                break
            values = [float(v) for v in line.split()]
            rows.append(values) if len(values) == 14 else None
    table = np.array(rows)
    r_in, chord_in, tc, area_in2 = table[:, 0], table[:, 1], table[:, 7], table[:, 10]

    clark_y = geometry.kulfan_from_name("clarky")
    base_tc = float(geometry.max_thickness(jnp.asarray(clark_y)))
    x, w = jnp.asarray(structure.GAUSS_X), jnp.asarray(structure.GAUSS_W)
    ratios = []
    for r, c, t, area_apc in zip(r_in, chord_in, tc, area_in2):
        if r < 3.0 or c < 0.25 or area_apc <= 0:
            continue
        y_upper, y_lower = geometry.kulfan_coordinates(jnp.asarray(clark_y * (t / base_tc)), x)
        ratios.append(float(structure.strip_properties(x, w, y_upper, y_lower)[0]) * c ** 2 / area_apc)
    assert len(ratios) > 10 and 0.92 < min(ratios) and max(ratios) < 0.96, (min(ratios), max(ratios))


def test_pitch_axis_fraction_follows_the_planform_blend():
    r = np.linspace(0.025, 0.275, 12)
    blend, inboard, _ = geometry.planform_blend_weights(r, 0.025, 0.05)
    old = np.where(inboard, 0.5 + blend * (geometry.AERO_PITCH_AXIS - 0.5), geometry.AERO_PITCH_AXIS)
    assert np.allclose(np.asarray(geometry.pitch_axis_fraction(r, 0.025, 0.05)), old, atol=1e-14)
    assert np.isclose(old[0], 0.5) and np.isclose(old[-1], 0.25)
    # traced radii: the blend is written in jnp
    traced = jax.jit(lambda radii: geometry.pitch_axis_fraction(radii, 0.025, 0.05))(jnp.asarray(r))
    assert np.allclose(np.asarray(traced), old, atol=1e-14)


def test_material_from_blade():
    from proptimiser import structure

    mat = structure.Layup.from_blade(BladeSpec())
    assert float(mat.core.density) == 1700.0 and float(mat.core.fill) == 1.0 and float(mat.skin_thickness) == 0.0
    assert np.isclose(float(mat.core.tau_allow), float(mat.core.sigma_allow) / np.sqrt(3.0))
    mat = Propeller.default(blade=BladeSpec(mat_tau_allow=25e6, fill_factor=0.4)).material()
    assert float(mat.core.tau_allow) == 25e6 and float(mat.core.fill) == 0.4
    # the outer region is the skin only when the skin has thickness
    mat = structure.Layup.from_blade(BladeSpec(skin_thickness_m=3e-4, skin_density_kg_m3=1500.0, mat_density=100.0, skin_tau_allow_Pa=5e6))
    assert float(mat.outer.density) == 1500.0 and float(mat.skin.fill) == 1.0 and float(mat.skin.tau_allow) == 5e6
    assert float(structure.Layup.from_blade(BladeSpec(mat_density=100.0, skin_density_kg_m3=1500.0)).outer.density) == 100.0


def test_section_properties_vmap_and_gradient():
    from proptimiser import structure

    prop = Propeller.default(inputs=Inputs(coeff_method=0, n_r=6))
    mat = prop.material()
    batched = jax.vmap(structure.section_properties, in_axes=(0, None))(prop.batched(), mat)
    assert batched.area.shape == (2, 6) and batched.outline.shape == (2, 6, 2 * (structure.N_X + 2), 2)
    assert np.allclose(np.asarray(batched.area[0]), np.asarray(batched.area[1]))

    p = prop.condition(0)
    grad = jax.grad(lambda c: jnp.sum(structure.section_properties(eqx.tree_at(lambda q: q.c, p, c), mat).area))(p.c)
    assert np.all(np.isfinite(np.asarray(grad))) and np.all(np.asarray(grad) > 0)


# ---------------------------------------------------------------------------
# Loads along the blade and at the hub
# ---------------------------------------------------------------------------

def _rod_propeller(n_r=24, N=2, beta_deg=0.0, c=0.03, blade=None, **inputs):
    # A blade made of one section (the hub section equals the aero section) at constant chord and twist: a rod
    blade = BladeSpec(N=N) if blade is None else dataclasses.replace(blade, N=N)
    prop = Propeller.default(inputs=Inputs(coeff_method=0, n_r=n_r, **inputs), blade=blade, c=c, beta_deg=beta_deg)
    p = prop.condition(0)
    return prop, eqx.tree_at(lambda q: q.cst, p, jnp.tile(p.cst[1], (p.cst.shape[0], 1)))


def _polar_inertia(p, area, mat, N):
    # N blades of rod sections: N * rho * fill * int A r^2 dr from the root, by the trapezoid rule
    r = np.asarray(p.r)
    return N * float(mat.core.density * mat.core.fill) * np.trapezoid(np.asarray(area) * r ** 2, r)


def test_uniform_load_shear_and_moment():
    from proptimiser import structure

    r = jnp.linspace(0.025, 0.275, 11)
    R, w = float(r[-1]), 3.0
    g = jnp.zeros((4, 11, 3)).at[..., 2].set(w)
    p = jnp.stack([r, 0 * r, 0 * r], -1)
    F, M_ref, M_O = structure.section_resultants(r, p, g, jnp.zeros_like(g))
    r = np.asarray(r)
    assert np.allclose(np.asarray(F)[..., 2], w * (R - r), atol=1e-12)
    assert np.allclose(np.asarray(M_ref)[..., 1], -w * (R - r) ** 2 / 2, atol=1e-12)      # e_r x e_z = -e_t
    assert np.allclose(np.asarray(M_O)[..., 1], -w * (R ** 2 - r ** 2) / 2, atol=1e-12)


def test_two_station_swept_statics():
    from proptimiser import structure

    # a point load of 1 N along e_z at the tip of a swept blade (the trapezoid weight of the tip value is dr / 2)
    r, dr, y, z = jnp.array([0.1, 0.2]), 0.1, 0.02, 0.01
    x = np.sqrt(0.2 ** 2 - y ** 2)
    p = jnp.array([[0.1, 0.0, 0.0], [x, y, z]])
    g = jnp.array([[[0.0, 0.0, 0.0], [0.0, 0.0, 2.0 / dr]]])
    F, M_ref, M_O = structure.section_resultants(r, p, g, jnp.zeros_like(g))
    assert np.allclose(np.asarray(F)[0, 0], [0, 0, 1.0])
    assert np.allclose(np.asarray(M_O)[0, 0], [y, -x, 0.0])             # the torque about the pitch axis e_r is y F_z
    assert np.isclose(float(M_O[0, 0, 0]), 0.02)
    assert np.allclose(np.asarray(M_ref)[0, 0], [y, -x + 0.1, 0.0])
    assert np.allclose(np.asarray(M_ref)[0, 1], 0.0)                    # nothing outboard of the tip station


def test_moment_equilibrium():
    from proptimiser import structure

    rng = np.random.default_rng(5)
    r = jnp.linspace(0.03, 0.28, 9)
    p = jnp.asarray(rng.normal(size=(9, 3)))
    g, m = (jnp.asarray(rng.normal(size=(3, 9, 3))) for _ in range(2))
    F, M_ref, M_O = structure.section_resultants(r, p, g, m)
    assert np.allclose(np.asarray(M_O - jnp.cross(p[None], F)), np.asarray(M_ref), atol=1e-13)
    direct = np.sum(0.5 * (np.asarray(jnp.cross(p[None], g) + m)[:, :-1] + np.asarray(jnp.cross(p[None], g) + m)[:, 1:])
                    * np.diff(np.asarray(r))[None, :, None], axis=1)
    assert np.allclose(np.asarray(M_O)[:, 0], direct, atol=1e-13)


def test_uniform_rod_centrifugal_tension():
    from proptimiser import structure

    prop, p = _rod_propeller(n_r=24)
    mat = prop.material()
    res = structure.structural_loads(p, mat, include_aero=False)
    A = float(res.props.area[0])
    assert np.allclose(np.asarray(res.props.area), A)
    r, omega = np.asarray(p.r), float(p.omega)
    expected = float(mat.core.density) * A * omega ** 2 * (r[-1] ** 2 - r ** 2) / 2     # exact: the trapezoid rule integrates r
    assert np.allclose(np.asarray(res.resultants.N), expected, rtol=1e-11, atol=1e-9)
    assert np.allclose(np.asarray(res.F_aero), 0.0)


def test_centrifugal_twisting_moment():
    from proptimiser import structure

    # steady rotation: the moment of a section about its centroid along the span is
    #   -rho omega^2 [sin(b) cos(b) (I_zz - I_xx) - I_xz cos(2b)]: the classic twisting moment, toward fine pitch
    for beta_deg in (40.0, 15.0, 70.0):
        prop, p = _rod_propeller(beta_deg=beta_deg)
        mat = prop.material()
        props, axes = structure.section_properties(p, mat), structure.blade_axes(p)
        loads = structure.inertial_loads(p, props, None, axes)
        b = np.radians(beta_deg)
        expected = -float(mat.core.density) * float(p.omega) ** 2 * (
            np.sin(b) * np.cos(b) * (np.asarray(props.I_zz) - np.asarray(props.I_xx)) - np.asarray(props.I_xz) * np.cos(2 * b))
        along_span = np.einsum("prk,rk->pr", np.asarray(loads.m_cg), np.asarray(axes.s))
        assert np.allclose(along_span, expected, rtol=1e-9)
        assert np.all(along_span[0] < 0)                       # toward fine pitch: a rotation about -s lowers beta

    # a symmetric section has no twisting moment at zero and quarter-turn twist
    symmetric = BladeSpec(airfoils=("naca0012",))
    reference = None
    for beta_deg in (45.0, 0.0, 90.0):
        prop, p = _rod_propeller(beta_deg=beta_deg, blade=symmetric)
        props, axes = structure.section_properties(p, prop.material()), structure.blade_axes(p)
        loads = structure.inertial_loads(p, props, None, axes)
        along_span = np.abs(np.einsum("prk,rk->pr", np.asarray(loads.m_cg), np.asarray(axes.s)))[0, 3]
        reference = along_span if beta_deg == 45.0 else reference
        assert beta_deg == 45.0 or along_span < 2e-3 * reference


def test_root_moment_is_the_sum_of_section_moments():
    from proptimiser import structure

    prop, p = _rod_propeller(beta_deg=25.0, blade=_swept_blade(1.0))
    mat = prop.material()
    kin = structure.Kinematics(pitch_rate=jnp.asarray(0.4), omega_dot=jnp.asarray(30.0), body_rates=jnp.array([0.2, -0.3, 0.1]))
    res = structure.structural_loads(p, mat, kin, include_aero=False)
    loads = structure.inertial_loads(p, res.props, kin, res.axes)
    per_length = np.asarray(jnp.cross(res.axes.p[None], loads.f) + loads.m)
    dr = np.diff(np.asarray(p.r))
    direct = np.sum(0.5 * (per_length[:, :-1] + per_length[:, 1:]) * dr[None, :, None], axis=1)
    assert np.allclose(np.asarray(res.M_O)[:, 0], direct, rtol=1e-12, atol=1e-12)
    force = np.sum(0.5 * (np.asarray(loads.f)[:, :-1] + np.asarray(loads.f)[:, 1:]) * dr[None, :, None], axis=1)
    assert np.allclose(np.asarray(res.F)[:, 0], force, rtol=1e-12, atol=1e-12)


def test_pitch_rate_coriolis():
    from proptimiser import structure

    prop, p = _rod_propeller(beta_deg=20.0)
    mat = prop.material()
    props, axes = structure.section_properties(p, mat), structure.blade_axes(p)
    rate = 0.8
    loads = structure.inertial_loads(p, props, structure.Kinematics(pitch_rate=jnp.asarray(rate)), axes)
    i = 5
    p_cg = np.asarray(axes.p[i] + props.x_c[i] * axes.x_s[i] + props.z_c[i] * axes.z_s[i])
    mass = float(mat.core.density * props.area[i] * axes.arc[i])
    v_rel = np.cross([rate, 0.0, 0.0], p_cg)
    assert np.allclose(np.asarray(loads.f_coriolis)[0, i], -mass * 2.0 * np.cross([0.0, 0.0, float(p.omega)], v_rel))
    assert np.allclose(np.asarray(loads.f_relative)[0, i], -mass * np.cross([rate, 0.0, 0.0], v_rel))

    # a rigid blade in steady rotation has no Coriolis, Euler or relative terms
    steady = structure.inertial_loads(p, props, None, axes)
    for term in (steady.f_coriolis, steady.f_relative, steady.f_euler):
        assert not np.any(np.asarray(term))
    assert np.allclose(np.asarray(steady.f), np.asarray(steady.f_centrifugal))


def test_spin_acceleration_torque():
    from proptimiser import structure

    prop, p = _rod_propeller(n_r=30, N=2, c=0.01)
    mat = prop.material()
    rate = 50.0
    res = structure.structural_loads(p, mat, structure.Kinematics(omega_dot=jnp.asarray(rate)), include_aero=False)
    expected = _polar_inertia(p, res.props.area, mat, 2) * rate
    assert np.isclose(float(-res.hub.moment[:, 2].mean()), expected, rtol=0.02)          # the shaft supplies I * omega_dot
    doubled = structure.structural_loads(p, mat, structure.Kinematics(omega_dot=jnp.asarray(2 * rate)), include_aero=False)
    assert np.isclose(float(-doubled.hub.moment[:, 2].mean()), 2 * float(-res.hub.moment[:, 2].mean()), rtol=1e-12)


def test_gyroscopic_hub_moment():
    from proptimiser import structure

    # a three-blade rotor on an airframe turning at q about y: the blades apply -I_p omega q about x to the hub
    # (the airframe applies the moment I_p omega q that turns the angular momentum I_p omega e_z)
    prop, p = _rod_propeller(n_r=30, N=3, c=0.01, n_psi=24)
    mat = prop.material()
    q = 0.5
    res = structure.structural_loads(p, mat, structure.Kinematics(body_rates=jnp.array([0.0, q, 0.0])), include_aero=False)
    expected = -_polar_inertia(p, res.props.area, mat, 3) * float(p.omega) * q
    assert np.isclose(float(res.hub.moment[:, 0].mean()), expected, rtol=0.02)
    assert abs(float(res.hub.moment[:, 1].mean())) < 1e-3 * abs(expected)
    assert abs(float(res.hub.moment[:, 2].mean())) < 1e-3 * abs(expected)
    # the loads grow linearly with the airframe rate at first order
    double = structure.structural_loads(p, mat, structure.Kinematics(body_rates=jnp.array([0.0, 2 * q, 0.0])), include_aero=False)
    assert np.isclose(float(double.hub.moment[:, 0].mean()), 2 * float(res.hub.moment[:, 0].mean()), rtol=2e-3)


def test_hub_harmonics():
    from proptimiser import structure

    n = 24
    psi = jnp.linspace(0.0, 2 * jnp.pi, n, endpoint=False)
    zero = jnp.zeros((n, 3))
    # a 1P axial load of each blade sums to zero over three blades, a 3P load survives (times N)
    out = structure.hub_loads(zero.at[:, 2].set(jnp.cos(psi)), zero, 3, psi)
    assert np.allclose(np.asarray(out.force), 0.0, atol=1e-12)
    out = structure.hub_loads(zero.at[:, 2].set(jnp.cos(3 * psi)), zero, 3, psi)
    assert np.allclose(np.asarray(out.force)[:, 2], 3 * np.cos(3 * np.asarray(psi)), atol=1e-12)
    assert np.isclose(float(out.force_amplitude[3, 2]), 3.0) and np.isclose(float(out.force_amplitude[0, 2]), 0.0, atol=1e-12)
    # a steady radial blade force turns with the blade: it cancels for two or three blades and not for one
    radial = zero.at[:, 0].set(5.0)
    for blades in (2, 3):
        assert np.allclose(np.asarray(structure.hub_loads(radial, zero, blades, psi).force), 0.0, atol=1e-12)
    one = np.asarray(structure.hub_loads(radial, zero, 1, psi).force)
    assert np.allclose(one[:, 0], 5 * np.cos(np.asarray(psi))) and np.allclose(one[:, 1], 5 * np.sin(np.asarray(psi)))
    # a steady axial load is just multiplied by the number of blades
    steady = structure.hub_loads(zero.at[:, 2].set(2.0), zero.at[:, 2].set(-0.5), 4, psi)
    assert np.allclose(np.asarray(steady.force)[:, 2], 8.0) and np.allclose(np.asarray(steady.moment)[:, 2], -2.0)


def test_azimuth_shift():
    from proptimiser import structure

    n = 24
    psi = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    gamma = jnp.array([0.0, 0.1, -0.2])
    field = np.cos(psi + 0.3)[:, None, None] * np.ones((1, 3, 2))
    shifted = np.asarray(structure._to_root_azimuth(jnp.asarray(field), gamma))
    expected = np.cos(psi[:, None] + 0.3 + np.asarray(gamma)[None, :])[:, :, None] * np.ones((1, 3, 2))
    assert np.allclose(shifted, expected, atol=1e-13)
    constant = np.full((n, 3, 2), 4.0)
    assert np.allclose(np.asarray(structure._to_root_azimuth(jnp.asarray(constant), gamma)), constant, atol=1e-13)


def test_hub_loads_close_with_thrust_and_torque():
    # In axial flow, without loss models, the loads of the blades at the hub add up to the thrust and torque of the BEMT
    from proptimiser import propmodel
    from proptimiser import structure

    for blade in (BladeSpec(), _swept_blade(1.0)):
        for blades in (2, 3):
            prop = Propeller.default(inputs=Inputs(coeff_method=0, n_r=10, n_psi=24, V_yz=[0.0, 0.0], V_i_0_eps=1e-12),
                                     blade=dataclasses.replace(blade, N=blades), c=0.03)
            p = prop.condition(0)
            V_i, _ = propmodel.induced_velocity(p)
            res = structure.structural_loads(p, prop.material(), V_i=V_i)
            thrust, torque = float(propmodel.thrust(V_i, p)), float(propmodel.torque(V_i, p))
            force, moment = np.asarray(res.hub.force), np.asarray(res.hub.moment)
            assert np.allclose(force[:, 2], thrust, rtol=1e-9)
            assert np.allclose(force[:, :2], 0.0, atol=1e-9 * thrust)
            # a swept blade adds its section pitching moment about the shaft, which the BEMT torque leaves out
            assert np.allclose(-moment[:, 2], torque, rtol=1e-9 if not blade.swept else 2e-2)


def test_stress_pure_tension_is_independent_of_the_fill_factor():
    from proptimiser import structure

    for fill in (1.0, 0.5):
        prop, p = _rod_propeller(blade=BladeSpec(fill_factor=fill))
        mat = prop.material()
        res = structure.structural_loads(p, mat, include_aero=False)
        N = res.resultants.N
        pure = eqx.tree_at(lambda x: x.resultants, res, structure.Resultants(N, *(jnp.zeros_like(N),) * 5))
        stress = structure.stress_fields(pure, mat)
        r, omega = np.asarray(p.r), float(p.omega)
        expected = float(mat.core.density) * omega ** 2 * (r[-1] ** 2 - r ** 2) / 2
        assert np.allclose(np.asarray(stress.sigma), expected[None, :, None], rtol=1e-11, atol=1e-6)
        assert np.allclose(np.asarray(stress.sigma_max), expected[None, :], rtol=1e-11, atol=1e-6)


def test_unsymmetric_bending_matches_the_moments():
    from proptimiser import structure

    prop, p = _rod_propeller(beta_deg=30.0)
    mat = prop.material()
    res = structure.structural_loads(p, mat, include_aero=False)
    props = res.props
    M_x, M_z = 0.8, -0.35
    n_psi, n_r = res.resultants.N.shape
    moments = structure.Resultants(*(jnp.zeros((n_psi, n_r)),) * 3, jnp.full((n_psi, n_r), M_x), jnp.zeros((n_psi, n_r)),
                                   jnp.full((n_psi, n_r), M_z))
    stress = structure.stress_fields(eqx.tree_at(lambda x: x.resultants, res, moments), mat)

    # integrate the stress over the section independently: sigma is linear in the outline coordinates, and the strip
    # integral over z is exact
    i = 4
    c = float(p.c[i])
    cst = np.asarray(geometry.cst_at_stations(p.r, p.r_cst, p.cst))[i]
    x, w = structure.GAUSS_X, structure.GAUSS_W
    yu, yl = (np.asarray(v) * c for v in geometry.kulfan_coordinates(jnp.asarray(cst), jnp.asarray(x)))
    xs, ws = x * c, w * c
    # stress gradient recovered from the two nodes of the outline that carry it: solve sigma = a x + b z on the outline
    outline = np.asarray(props.outline[i])
    sigma = np.asarray(stress.sigma)[0, i]
    a, b = np.linalg.lstsq(outline, sigma, rcond=None)[0]
    assert np.allclose(outline @ np.array([a, b]), sigma, atol=1e-6 * np.abs(sigma).max())
    t = yu - yl
    xr = xs - (np.sum(ws * xs * t) / np.sum(ws * t))                       # x relative to the centroid
    z_first, z_second = 0.5 * (yu ** 2 - yl ** 2), (yu ** 3 - yl ** 3) / 3
    zc = np.sum(ws * z_first) / np.sum(ws * t)
    M_x_back = -np.sum(ws * (a * xr * z_first + b * (z_second - zc * z_first)))
    M_z_back = np.sum(ws * xr * (a * xr * t + b * (z_first - zc * t)))
    assert np.isclose(M_x_back, M_x, rtol=1e-9) and np.isclose(M_z_back, M_z, rtol=1e-9)


def test_torsion_stress_of_an_ellipse():
    import types
    from proptimiser import structure

    a, b = 0.5, 0.06
    x, w = structure.GAUSS_X, structure.GAUSS_W
    y = 2 * b * np.sqrt(x * (1 - x))
    A, xc, zc, Ixx, Izz, Ixz = (jnp.atleast_1d(v) for v in structure.strip_properties(
        jnp.asarray(x), jnp.asarray(w), jnp.asarray(y), jnp.asarray(-y)))
    mat = structure.Layup.from_blade(BladeSpec())
    J = structure.torsion_constant(A, Ixx, Izz)
    weighted = lambda v: structure.SectionSum(v * A, xc, zc, v * Ixx, v * Izz, v * Ixz)
    props = structure.SectionProps(A, xc, zc, Ixx, Izz, Ixz, J, jnp.full(1, 2 * b), jnp.zeros((1, 2, 2)), A, jnp.zeros((1, 2, 2)),
                                   jnp.ones((1, 2), dtype=bool), weighted(mat.core.density), weighted(mat.core.E),
                                   mat.core.G * A, mat.core.G * J)
    T = 1.7
    shape = (1, 1)
    zeros = jnp.zeros(shape)
    resultants = structure.Resultants(zeros, zeros, zeros, zeros, jnp.full(shape, T), zeros)
    stress = structure.stress_fields(types.SimpleNamespace(props=props, resultants=resultants), mat)
    exact = 2 * T / (np.pi * a * b ** 2)                                    # the largest shear stress of a twisted ellipse
    assert np.isclose(float(stress.tau[0, 0]), exact, rtol=0.03)
    assert np.isclose(float(stress.sigma_vm[0, 0]), np.sqrt(3.0) * float(stress.tau[0, 0]), rtol=1e-12)


def test_safety_factor_scales_with_the_square_of_the_speed():
    from proptimiser import structure

    prop, p = _rod_propeller(beta_deg=30.0)
    mat = prop.material()
    base = structure.stress_fields(structure.structural_loads(p, mat, include_aero=False), mat)
    fast_p = eqx.tree_at(lambda q: q.omega, p, 2.0 * p.omega)
    fast = structure.stress_fields(structure.structural_loads(fast_p, mat, include_aero=False), mat)
    inner = slice(0, -1)                                                    # the tip station carries no load
    assert np.allclose(np.asarray(fast.safety_factor)[:, inner], np.asarray(base.safety_factor)[:, inner] / 4.0, rtol=1e-9)
    assert np.allclose(np.asarray(fast.sigma_vm)[:, inner], 4.0 * np.asarray(base.sigma_vm)[:, inner], rtol=1e-9)


def test_structural_loads_vmap_matches_a_loop():
    from proptimiser import structure

    prop = Propeller.default(inputs=Inputs(coeff_method=0, n_r=8), c=0.03, blade=_swept_blade(1.0))
    mat = prop.material()
    batched = jax.vmap(structure.structural_loads, in_axes=(0, None))(prop.batched(), mat)
    for i in range(prop.inputs.n_cond):
        single = structure.structural_loads(prop.condition(i), mat)
        assert np.allclose(np.asarray(batched.F[i]), np.asarray(single.F), rtol=1e-11, atol=1e-11)
        assert np.allclose(np.asarray(batched.hub.moment[i]), np.asarray(single.hub.moment), rtol=1e-11, atol=1e-11)


def test_structural_gradients_match_finite_differences():
    # A smooth maximum of the von Mises stress, through the BEMT solve (implicit gradients), against central differences
    from proptimiser import propmodel
    from proptimiser import structure

    for blade in (BladeSpec(), _swept_blade(1.0)):          # the gradient is finite at zero offsets too
        prop = Propeller.default(inputs=Inputs(coeff_method=0, n_r=8, n_psi=12), c=0.03, blade=blade)
        p, mat = prop.condition(1), prop.material()

        def objective(omega, c):
            q = eqx.tree_at(lambda x: (x.omega, x.c), p, (omega, c))
            V_i, _ = propmodel.induced_velocity(q)
            stress = structure.stress_fields(structure.structural_loads(q, mat, V_i=V_i), mat)
            return jax.nn.logsumexp(stress.sigma_vm / 1e6)

        g_omega, g_c = jax.grad(objective, argnums=(0, 1))(p.omega, p.c)
        assert np.all(np.isfinite(np.asarray(g_c))) and np.isfinite(float(g_omega))
        h = 1e-6
        fd_omega = (float(objective(p.omega + h, p.c)) - float(objective(p.omega - h, p.c))) / (2 * h)
        assert np.isclose(float(g_omega), fd_omega, rtol=1e-5)
        d = jnp.asarray(np.random.default_rng(2).standard_normal(p.c.shape)) * 1e-3
        fd_c = (float(objective(p.omega, p.c + h * d)) - float(objective(p.omega, p.c - h * d))) / (2 * h)
        assert np.isclose(float(jnp.sum(g_c * d)), fd_c, rtol=1e-5)


def test_optimiser_runs_on_a_swept_blade():
    # The offsets are fixed geometry: they travel through the batched parameters, stay untouched, and the optimised
    # design goes straight into the structural loads
    from proptimiser import optimiser
    from proptimiser import structure

    base = Propeller.default(inputs=Inputs(coeff_method=0, n_r=6, opt_max_iter=2, opt_verbose=0), blade=_swept_blade(1.0))
    for solver in optimiser.SOLVERS:
        prop = base.with_inputs(dataclasses.replace(base.inputs, opt_solver=solver))
        reports = []
        result = optimiser.run_optimization(prop, callback=reports.append)
        assert reports and np.isfinite(reports[-1].objective) and reports[-1].objective > 0, solver
        assert result.blade.swept and np.array_equal(np.asarray(result.offsets), np.asarray(base.offsets)), solver
        assert not np.allclose(np.asarray(result.c), np.asarray(base.c)) or not np.allclose(np.asarray(result.beta), np.asarray(base.beta))
        loads = structure.structural_loads(result.condition(0), result.material())
        assert np.all(np.isfinite(np.asarray(loads.F))) and np.all(np.isfinite(np.asarray(loads.hub.moment)))


def test_structural_report():
    from proptimiser import propmodel
    from proptimiser import results

    prop = Propeller.default(inputs=Inputs(coeff_method=0, n_r=8, n_psi=24), c=0.03, blade=_swept_blade(1.0))
    prop, params_list, rows = results.evaluate_conditions(prop)
    data = results.structural_data(params_list[0], prop.material())
    n_psi, n_r = 24, 8
    for key in ("N", "V_x", "V_z", "M_x", "M_z", "T", "T_aero", "T_inertial", "sigma_max", "tau", "sigma_vm", "safety_factor"):
        assert data[key].shape == (n_psi, n_r), key
    for key in ("r", "c", "area", "I_xx", "I_zz", "J", "mass_per_length"):
        assert data[key].shape == (n_r,), key
    assert data["hub_force"].shape == (n_psi, 3) and data["atm"].shape == (n_psi,)

    s = data["summary"]
    assert s["n_blades"] == 2 and 0.0 < s["safety_factor_min"] < float("inf")
    assert np.isclose(s["hub_thrust"], rows[0]["thrust"], rtol=0.02)           # blade-element thrust vs the momentum thrust
    assert np.isclose(s["hold_mean"], -(s["atm_mean"] + s["ctm_mean"]))
    assert s["ctm_mean"] < 0                                                    # the centrifugal twist turns toward fine pitch
    text = results.structural_summary(data, 0)
    assert "Weakest section" in text and "Pitch-change torque" in text and "flight condition 0" in text

    # the material reaches the report: a heavier allowable raises the safety factor by the same factor
    stronger = dataclasses.replace(prop.blade, mat_sigma_allow=2 * prop.blade.mat_sigma_allow, mat_tau_allow=2 * prop.blade.mat_sigma_allow / np.sqrt(3))
    again = results.structural_data(params_list[0], prop.with_blade(stronger).material())
    assert np.isclose(again["summary"]["safety_factor_min"], 2 * s["safety_factor_min"], rtol=1e-9)


# ---------------------------------------------------------------------------
# Mass properties: closed forms
# ---------------------------------------------------------------------------

_MASS_SPEC = dict(r_foils=(0.025, 0.025, 0.0250001, 0.15, 0.275), airfoils=("clarky",))


def _one_section_propeller(chord=None, beta_deg=0.0, n_r=9, N=2, **spec):
    # One section along the whole blade, pitch axis at c/4 (the blend zone has no length), no hub zone
    cst = geometry.kulfan_from_name("clarky")
    blade = BladeSpec(N=N, hub_cst=tuple(float(v) for v in cst), **{**_MASS_SPEC, **spec})
    prop = Propeller.default(inputs=Inputs(coeff_method=0, n_r=n_r), blade=blade, c=0.03, beta_deg=beta_deg)
    return prop if chord is None else prop.with_state(c=chord(np.asarray(prop.r)))


def _unit_section():
    # Unit-chord properties of the section from the polygon formulas, about the centroid
    from aerosandbox.geometry.airfoil.airfoil_families import get_kulfan_coordinates

    cst = geometry.kulfan_from_name("clarky")
    coords = get_kulfan_coordinates(lower_weights=cst[:8], upper_weights=cst[8:16], leading_edge_weight=cst[16],
                                    TE_thickness=cst[17], n_points_per_side=12000)
    return _polygon_properties(coords[:, 0], coords[:, 1])


def _chord_tensor(rho, chord, r0, r1):
    # Closed-form mass, CG and tensor of the blade of chord(r) (a numpy Polynomial) at zero twist, pitch axis at c/4:
    # p = (r, -x, z), with x from the pitch axis, so every integrand is a polynomial in r
    from numpy.polynomial import Polynomial as P

    A, cx, cz, Ixx, Izz, Ixz = _unit_section()
    x_p = cx - 0.25
    r = P([0.0, 1.0])
    integral = lambda f: float(f.integ()(r1) - f.integ()(r0))
    I_x2, I_z2, I_xz2 = A * x_p ** 2 + Izz, A * cz ** 2 + Ixx, A * x_p * cz + Ixz
    mass = rho * A * integral(chord ** 2)
    first = rho * A * np.array([integral(chord ** 2 * r), -x_p * integral(chord ** 3), cz * integral(chord ** 3)])
    T = np.zeros((3, 3))
    T[0, 0] = rho * (I_x2 + I_z2) * integral(chord ** 4)
    T[1, 1] = rho * (A * integral(chord ** 2 * r ** 2) + I_z2 * integral(chord ** 4))
    T[2, 2] = rho * (A * integral(chord ** 2 * r ** 2) + I_x2 * integral(chord ** 4))
    T[0, 1] = T[1, 0] = rho * A * x_p * integral(chord ** 3 * r)
    T[0, 2] = T[2, 0] = -rho * A * cz * integral(chord ** 3 * r)
    T[1, 2] = T[2, 1] = rho * I_xz2 * integral(chord ** 4)
    return mass, first / mass, T


def test_mass_properties_of_a_rectangular_blade():
    from numpy.polynomial import Polynomial as P

    prop = _one_section_propeller()
    got = prop.blade_mass_properties()
    mass, cg, T = _chord_tensor(1700.0, P([0.03]), 0.025, 0.275)
    assert np.isclose(float(got.mass), mass, rtol=1e-7)
    assert np.allclose(np.asarray(got.cg), cg, rtol=5e-6, atol=1e-9)
    assert np.allclose(np.asarray(got.inertia), T, rtol=3e-6, atol=1e-13)
    # the expected values are not trivial: the chord offsets the section from the pitch axis
    assert abs(T[0, 1]) > 1e-3 * T[2, 2] and abs(T[1, 2]) > 0.0


def test_mass_properties_of_a_tapered_blade():
    # The chord is linear in r, so the Gauss rule is exact on any station grid
    from numpy.polynomial import Polynomial as P

    chord = P([0.05, -0.1])
    for n_r in (5, 17):
        got = _one_section_propeller(chord, n_r=n_r).blade_mass_properties()
        mass, cg, T = _chord_tensor(1700.0, chord, 0.025, 0.275)
        assert np.isclose(float(got.mass), mass, rtol=1e-7)
        assert np.allclose(np.asarray(got.cg), cg, rtol=5e-6, atol=1e-9)
        assert np.allclose(np.asarray(got.inertia), T, rtol=3e-6, atol=1e-13)


def test_constant_twist_turns_the_tensor_about_the_pitch_axis():
    flat = _one_section_propeller().blade_mass_properties()
    for beta_deg in (30.0, -20.0, 75.0):
        b = np.radians(beta_deg)
        R = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(b), -np.sin(b)], [0.0, np.sin(b), np.cos(b)]])
        got = _one_section_propeller(beta_deg=beta_deg).blade_mass_properties()
        assert np.isclose(float(got.mass), float(flat.mass), rtol=1e-13)
        assert np.allclose(np.asarray(got.cg), R @ np.asarray(flat.cg), rtol=1e-12, atol=1e-14)
        assert np.allclose(np.asarray(got.inertia), R @ np.asarray(flat.inertia) @ R.T, rtol=1e-11, atol=1e-16)


def test_a_slice_with_linear_twist_is_a_rotated_slice():
    # With twist linear in r the tensor is the sum of slices: each slice is the flat slice turned by its own twist
    from proptimiser import structure

    betas = np.radians(np.linspace(35.0, 5.0, 33))
    prop = _one_section_propeller(n_r=33).with_state(beta=betas)
    p, mat = prop.condition(0), prop.material()
    slices = structure.blade_slices(structure.section_properties(p, mat), structure.blade_axes(p))
    A, cx, cz, Ixx, Izz, Ixz = _unit_section()
    x_p, c = cx - 0.25, 0.03
    for i in (1, 9, 32):                                    # station 0 has the pitch axis of the hub section
        b = betas[i]
        R = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(b), -np.sin(b)], [0.0, np.sin(b), np.cos(b)]])
        flat = 1700.0 * c ** 4 * np.array([[Ixx + Izz, 0.0, 0.0], [0.0, Ixx, Ixz], [0.0, Ixz, Izz]])
        assert np.allclose(np.asarray(slices.inertia[i]), R @ flat @ R.T, rtol=1e-5, atol=1e-12)
        assert np.allclose(np.asarray(slices.p_cg[i]), np.array([float(p.r[i]), 0.0, 0.0]) + R @ (c * np.array([0.0, -x_p, cz])), rtol=1e-5, atol=1e-9)


def test_parallel_axis_and_tensor_properties():
    from proptimiser import structure

    got = _one_section_propeller(beta_deg=25.0).blade_mass_properties()
    T, Tc, m, g = (np.asarray(v) for v in (got.inertia, got.inertia_cg, got.mass, got.cg))
    assert np.allclose(T, T.T, rtol=0, atol=1e-18)
    assert np.allclose(T - Tc, m * (np.dot(g, g) * np.eye(3) - np.outer(g, g)), rtol=1e-14, atol=1e-18)
    for tensor in (T, Tc):
        e = np.linalg.eigvalsh(tensor)
        assert e[0] > 0.0 and e[2] <= e[0] + e[1] + 1e-15             # a real body: positive, triangle inequality

    # hardware: a cylinder off the origin, by the parallel axis theorem
    m_h, a, h, z = 0.01, 0.02, 0.01, 0.005
    hw = structure.hub_hardware(m_h, a, h, z)
    assert np.isclose(float(hw.inertia[2, 2]), 0.5 * m_h * a ** 2)
    assert np.isclose(float(hw.inertia[0, 0]), m_h * (3 * a ** 2 + h ** 2) / 12 + m_h * z ** 2)
    assert np.isclose(float(hw.inertia_cg[0, 0]), m_h * (3 * a ** 2 + h ** 2) / 12)
    both = got.plus(hw)
    assert np.isclose(float(both.mass), float(m) + m_h)
    assert np.allclose(np.asarray(both.cg), (float(m) * g + m_h * np.array([0.0, 0.0, z])) / (float(m) + m_h))
    assert np.allclose(np.asarray(both.inertia), T + np.asarray(hw.inertia))
    point = structure.hub_hardware(m_h)
    assert not np.any(np.asarray(point.inertia)) and not np.any(np.asarray(point.cg))


def test_n_blade_sum():
    from proptimiser import structure

    blade = _one_section_propeller(beta_deg=25.0, offset_r=(0.025, 0.275), offset_y=(0.0, 0.02), offset_z=(0.0, 0.01)).blade_mass_properties()
    T, g, m = (np.asarray(v) for v in (blade.inertia, blade.cg, blade.mass))
    assert abs(T[0, 1]) > 0.0 and abs(T[0, 2]) > 0.0 and abs(T[1, 2]) > 0.0
    for N in range(1, 7):
        rotor = structure.rotor_mass_properties(blade, N)
        direct, first = np.zeros((3, 3)), np.zeros(3)
        for k in range(N):
            a = 2 * np.pi * k / N
            R = np.array([[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]])
            direct += R @ T @ R.T
            first += m * R @ g
        assert np.allclose(np.asarray(rotor.inertia), direct, rtol=1e-12, atol=1e-16), N
        assert np.isclose(float(rotor.mass), N * float(m))
        assert np.allclose(np.asarray(rotor.cg), first / (N * m), rtol=1e-12, atol=1e-15), N
        if N >= 3:
            assert np.isclose(float(rotor.inertia[0, 0]), float(rotor.inertia[1, 1]), rtol=1e-13)
    two = np.asarray(structure.rotor_mass_properties(blade, 2).inertia)
    assert np.isclose(two[1, 1] - two[0, 0], 2 * (T[1, 1] - T[0, 0]), rtol=1e-12)         # the correct difference for N = 2
    assert two[1, 1] > 10 * two[0, 0]                                                      # a blade is a rod along e_r
    assert np.isclose(float(jax.jit(structure.rotor_mass_properties, static_argnums=())(blade, jnp.asarray(3)).mass), 3 * float(m))


def test_a_design_propeller_uses_its_blade_count():
    from proptimiser import structure

    prop = _one_section_propeller(N=3)
    rotor, blade = prop.mass_properties(), prop.blade_mass_properties()
    assert np.isclose(float(rotor.mass), 3 * float(blade.mass))
    assert np.isclose(float(rotor.inertia[2, 2]), 3 * float(blade.inertia[2, 2]))
    hardware = structure.hub_hardware(0.012, 0.015, 0.01)
    assert np.isclose(float(prop.mass_properties(hardware).mass), 3 * float(blade.mass) + 0.012)


def test_hub_zone_is_a_prism_of_the_hub_section():
    # A hub zone from the axis adds a prism of the section at the hub chord and twist
    from numpy.polynomial import Polynomial as P

    prop = _one_section_propeller(r_foils=(0.0, 0.025, 0.0250001, 0.15, 0.275), hub_chord=0.03)
    got = prop.blade_mass_properties()
    mass, _, T = _chord_tensor(1700.0, P([0.03]), 0.025, 0.275)
    A, cx, _, _, _, _ = _unit_section()
    hub_mass = 1700.0 * A * 0.03 ** 2 * 0.025
    assert np.isclose(float(got.mass), mass + hub_mass, rtol=1e-7)
    # the pitch axis of the hub zone is at c/2, so the zone is r^2 plus the section about c/2
    _, cx, cz, Ixx, Izz, _ = _unit_section()
    hub_zz = 1700.0 * (A * 0.03 ** 2 * 0.025 ** 3 / 3 + 0.025 * 0.03 ** 4 * (A * (cx - 0.5) ** 2 + Izz))
    assert np.isclose(float(got.inertia[2, 2]), T[2, 2] + hub_zz, rtol=1e-6)


def test_sweep_closed_forms():
    # A straight axis through the hub centre: the whole blade turns about e_z
    m = 0.15
    flat = _one_section_propeller().blade_mass_properties()
    swept = _one_section_propeller(offset_r=(0.025, 0.275), offset_y=(m * 0.025, m * 0.275)).blade_mass_properties()
    g = np.arcsin(m)
    R = np.array([[np.cos(g), -np.sin(g), 0.0], [np.sin(g), np.cos(g), 0.0], [0.0, 0.0, 1.0]])
    assert np.isclose(float(swept.mass), float(flat.mass), rtol=1e-12)
    assert np.allclose(np.asarray(swept.cg), R @ np.asarray(flat.cg), rtol=1e-11, atol=1e-14)
    assert np.allclose(np.asarray(swept.inertia), R @ np.asarray(flat.inertia) @ R.T, rtol=1e-10, atol=1e-15)

    # A constant tangential offset: the axis is parallel to e_r, and the arc is longer than the polar radius
    y0, c = 0.02, 0.03
    got = _one_section_propeller(n_r=65, offset_r=(0.025, 0.275), offset_y=(y0, y0)).blade_mass_properties()
    A, cx, cz, Ixx, Izz, Ixz = _unit_section()
    x0, x1 = np.sqrt(0.025 ** 2 - y0 ** 2), np.sqrt(0.275 ** 2 - y0 ** 2)
    assert np.isclose(float(got.mass), 1700.0 * A * c ** 2 * (x1 - x0), rtol=1e-6)
    r_cg = float(got.cg[0])
    assert np.isclose(r_cg, 0.5 * (x0 + x1), rtol=1e-6)
    assert np.isclose(float(got.cg[1]), y0 - c * (cx - 0.25), rtol=1e-6)
    expected_zz = 1700.0 * A * c ** 2 * ((x1 ** 3 - x0 ** 3) / 3.0 + (x1 - x0) * (y0 - c * (cx - 0.25)) ** 2) + 1700.0 * c ** 4 * Izz * (x1 - x0)
    assert np.isclose(float(got.inertia[2, 2]), expected_zz, rtol=1e-6)


# ---------------------------------------------------------------------------
# Mass properties: grid convergence
# ---------------------------------------------------------------------------

def _smooth_blade(n_r):
    # An analytic planform and twist, so the grid does not change the blade
    prop = Propeller.default(inputs=Inputs(coeff_method=0, n_r=n_r), blade=BladeSpec(r_foils=(0.0, 0.025, 0.05, 0.15, 0.275), hub_chord=0.03, hub_twist_deg=30.0))
    r = np.asarray(prop.r)
    x = (r - r[0]) / (r[-1] - r[0])
    return prop.with_state(c=0.03 + 0.02 * np.sin(np.pi * x) ** 2, beta=np.radians(30.0 - 22.0 * x))


def test_radial_convergence():
    from proptimiser import structure

    reference = _smooth_blade(257).blade_mass_properties()
    errors = {}
    for n_r in (9, 17, 33, 65, 129):
        got = _smooth_blade(n_r).blade_mass_properties()
        errors[n_r] = (abs(float(got.mass) / float(reference.mass) - 1.0), abs(float(got.inertia[2, 2]) / float(reference.inertia[2, 2]) - 1.0))
    # the chord and twist are linear between stations, so the error is second order in dr
    for key in (0, 1):
        for fine, coarse in ((33, 17), (65, 33)):
            assert 2.5 < errors[coarse][key] / errors[fine][key] < 5.0, (key, errors)
    assert errors[33][0] < 2e-4 and errors[33][1] < 2e-4, errors


def test_chordwise_convergence():
    from proptimiser import structure

    prop = _smooth_blade(17)
    p, mat = prop.condition(0), prop.material()
    def area_and_inertia(n):
        props = structure.section_properties(p, mat, structure._chordwise_quadrature(n))
        return np.asarray(props.mass.total), np.asarray(props.mass.I_zz), np.asarray(props.mass.I_xx)
    reference = area_and_inertia(400)
    for n, tolerance in ((24, 1e-5), (48, 1e-6), (96, 1e-7)):
        for got, ref in zip(area_and_inertia(n), reference):
            assert np.max(np.abs(got / ref - 1.0)) < tolerance, (n, np.max(np.abs(got / ref - 1.0)))


# ---------------------------------------------------------------------------
# Mass properties: independent solid and the loads
# ---------------------------------------------------------------------------

def _closed_stack_properties(loops):
    # Volume, first moment and inertia tensor (density 1) of a closed stack of section loops (n, p, 3) from signed
    # tetrahedra to the origin: this shares no code with the structure module
    a = loops[:-1]
    b = np.roll(loops, -1, axis=1)[:-1]
    c = np.roll(loops, -1, axis=1)[1:]
    d = loops[1:]
    sides = np.concatenate([np.stack([a, b, c], axis=-2).reshape(-1, 3, 3), np.stack([a, c, d], axis=-2).reshape(-1, 3, 3)])
    caps = []
    for loop, flip in ((loops[0], True), (loops[-1], False)):
        centre = np.broadcast_to(loop.mean(axis=0), loop.shape)
        turned = np.roll(loop, -1, axis=0)
        caps.append(np.stack([centre, turned, loop] if flip else [centre, loop, turned], axis=-2))
    t = np.concatenate([sides, *caps])
    v = np.einsum("ij,ij->i", t[:, 0], np.cross(t[:, 1], t[:, 2])) / 6.0
    v = v * np.sign(v.sum())
    s = t.sum(axis=1)
    C = (v[:, None, None] / 20.0 * (np.einsum("mi,mj->mij", s, s) + np.einsum("mai,maj->mij", t, t))).sum(axis=0)
    return v.sum(), (v[:, None] * s / 4.0).sum(axis=0), np.trace(C) * np.eye(3) - C


def _section_loops(r, c, beta, pitch_axis, offsets, loops_xz):
    # Section loops (x, z from the leading edge, m, one array for each station) on the blade frame, by the CAD rules
    x0, y0, z0 = geometry.section_origin(r, offsets)
    out = []
    for i, loop in enumerate(loops_xz):
        x, z = loop[:, 0] - pitch_axis[i] * c[i], loop[:, 1]
        X = x * np.cos(beta[i]) + z * np.sin(beta[i]) + x0[i]
        Z = -x * np.sin(beta[i]) + z * np.cos(beta[i]) + z0[i]
        out.append(np.stack([np.full_like(X, y0[i]), -X, Z], axis=1))     # CAD (X, Y, Z) -> (e_r, e_t, e_z) = (Y, -X, Z)
    return np.array(out)


def _export_grid(prop):
    blade = prop.blade
    return np.unique(np.concatenate([np.linspace(blade.r_foils[0], blade.R_hub, 6), np.linspace(blade.R_hub, blade.R_aero, 25),
                                     np.linspace(blade.R_aero, blade.R_tip, 240)]))


def _design_stack(prop, n_points_per_side=300):
    # The blade as the structure model defines it, from its own definition and not from its code: chord, twist and CST
    # linear between their stations, pitch axis from c/2 to c/4 by the cosine blend. Unswept stations only
    from aerosandbox.geometry.airfoil.airfoil_families import get_kulfan_coordinates

    blade = prop.blade
    r = _export_grid(prop)
    c = np.interp(r, np.asarray(prop.r), np.asarray(prop.c))
    beta = np.interp(r, np.asarray(prop.r), np.asarray(prop.beta))
    knots = np.asarray(blade.r_cst)
    table = np.vstack([np.asarray(blade.hub_cst), np.asarray(prop.cst)])
    weights = np.column_stack([np.interp(r, knots, table[:, k]) for k in range(table.shape[1])])
    fraction = np.clip((r - blade.R_hub) / (blade.R_aero - blade.R_hub), 0.0, 1.0)
    pitch_axis = 0.5 + 0.5 * (1.0 - np.cos(np.pi * fraction)) * (0.25 - 0.5)
    coords = np.array([get_kulfan_coordinates(lower_weights=w[:8], upper_weights=w[8:16], leading_edge_weight=w[16], TE_thickness=w[17],
                                              n_points_per_side=n_points_per_side) for w in weights])
    return r, c, beta, pitch_axis, np.zeros((len(r), 4)), coords * c[:, None, None]


def test_real_design_against_an_independent_solid():
    # 1. The same definition as the model, integrated in 3D by tetrahedra. The tolerance is the error of the integral.
    # 2. The CAD export geometry: real stadium loops, a PCHIP planform and the trailing edge rule (P1), which the
    #    structure model leaves out. The tolerance is the size of these differences.
    prop = Propeller.load("MAD_FLUXER_22x6_6_jung")
    rho = prop.blade.mat_density * prop.blade.fill_factor
    got = prop.blade_mass_properties()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stack = _design_stack(prop)
        V, S, T = _closed_stack_properties(_section_loops(*stack))
        r, c, beta, pitch_axis, coords, offsets = geometry.blade_export_geometry(prop, n_points_per_side=100, r=_export_grid(prop))
        V_cad, S_cad, T_cad = _closed_stack_properties(_section_loops(r, c, beta, pitch_axis, offsets, coords * c[:, None, None]))

    assert abs(float(got.mass) / (rho * V) - 1.0) < 2e-4, (float(got.mass), rho * V)
    assert np.linalg.norm(np.asarray(got.cg) - S / V) < 5e-5, (np.asarray(got.cg), S / V)
    assert np.allclose(np.asarray(got.inertia), rho * T, rtol=1e-3, atol=1e-5 * float(got.inertia[2, 2])), (np.asarray(got.inertia), rho * T)
    assert abs(float(got.inertia[2, 2]) / (rho * T[2, 2]) - 1.0) < 1e-4

    assert abs(float(got.mass) / (rho * V_cad) - 1.0) < 1e-2
    assert np.linalg.norm(np.asarray(got.cg) - S_cad / V_cad) < 5e-4
    assert abs(float(got.inertia[2, 2]) / (rho * T_cad[2, 2]) - 1.0) < 1e-2


def test_swept_design_against_a_stack_of_normal_sections():
    # A design with sweep and rake. The sections stand normal to the bent span, so the stack holds the wedge factor of
    # the bend. The stack shares only blade_axes with the mass properties
    from proptimiser import structure

    prop = Propeller.load("APC_20x10E")
    assert prop.blade.swept
    rho = prop.blade.mat_density * prop.blade.fill_factor
    got = prop.blade_mass_properties()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r, c, beta, pitch_axis, _, coords = _design_stack(prop, n_points_per_side=200)
    frame = dataclasses.replace(prop.condition(0), r=jnp.asarray(r), beta=jnp.asarray(beta), offsets=geometry.station_offsets(r, prop.blade))
    axes = structure.blade_axes(frame)
    p, x_s, z_s = (np.asarray(v) for v in (axes.p, axes.x_s, axes.z_s))
    x = coords[:, :, 0] - pitch_axis[:, None] * c[:, None]
    V, S, T = _closed_stack_properties(p[:, None] + x[..., None] * x_s[:, None] + coords[:, :, 1, None] * z_s[:, None])
    assert abs(float(got.mass) / (rho * V) - 1.0) < 5e-4, (float(got.mass), rho * V)
    assert np.linalg.norm(np.asarray(got.cg) - S / V) < 1e-4, (np.asarray(got.cg), S / V)
    assert abs(float(got.inertia[2, 2]) / (rho * T[2, 2]) - 1.0) < 1e-3, (float(got.inertia[2, 2]), rho * T[2, 2])


def test_inertial_loads_match_the_mass_properties():
    # The root force is m w^2 r_cg, and the root moment is the tensor in the form w^2 (T_yz, -T_xz, 0): the slices of
    # the loads integrated by the same rule
    from proptimiser import structure

    prop = _one_section_propeller(beta_deg=25.0, n_r=33, offset_r=(0.025, 0.275), offset_y=(0.0, 0.02), offset_z=(0.0, 0.01))
    p, mat = prop.condition(0), prop.material()
    res = structure.structural_loads(p, mat, include_aero=False)
    slices = structure.blade_slices(res.props, res.axes)
    r = np.asarray(p.r)
    trapezoid = lambda f: np.trapezoid(np.asarray(f), r, axis=0)
    mass = trapezoid(slices.mass)
    first = trapezoid(slices.mass[:, None] * slices.p_cg)
    T = trapezoid(slices.inertia + structure._point_tensor(slices.mass, slices.p_cg))
    w2 = float(p.omega) ** 2
    assert np.allclose(np.asarray(res.F_inertial)[:, 0], w2 * np.array([first[0], first[1], 0.0]), rtol=1e-12, atol=1e-10)
    assert np.allclose(np.asarray(res.M_O_inertial)[:, 0], w2 * np.array([T[1, 2], -T[0, 2], 0.0]), rtol=1e-11, atol=1e-10)
    assert np.allclose(np.asarray(res.pitch_torque_inertial), w2 * T[1, 2], rtol=1e-11)
    assert T[1, 2] < 0.0                                                           # positive twist: toward fine pitch

    # the rule of the loads tends to the Gauss rule of the mass properties
    gauss = structure.blade_mass_properties(p, mat)
    assert abs(mass / float(gauss.mass) - 1.0) < 1e-3
    assert abs(T[2, 2] / float(gauss.inertia[2, 2]) - 1.0) < 1e-3
    assert abs(T[1, 2] / float(gauss.inertia[1, 2]) - 1.0) < 1e-3

    # spin acceleration: the moment about e_z is -T_zz w_dot
    rate = 40.0
    spin = structure.structural_loads(p, mat, structure.Kinematics(omega_dot=jnp.asarray(rate)), include_aero=False)
    expected = -rate * T[2, 2] + 0.0
    assert np.allclose(np.asarray(spin.M_O_inertial)[:, 0, 2], expected, rtol=1e-11)


def test_mass_properties_jit_vmap_and_gradient():
    from proptimiser import structure

    prop = _one_section_propeller(n_r=9)
    p, mat = prop.condition(0), prop.material()
    grad = jax.grad(lambda c: structure.blade_mass_properties(eqx.tree_at(lambda q: q.c, p, c), mat).inertia[2, 2])(p.c)
    assert np.all(np.isfinite(np.asarray(grad))) and np.all(np.asarray(grad) > 0)
    d = jnp.asarray(np.random.default_rng(1).standard_normal(p.c.shape)) * 1e-3
    h = 1e-5
    fd = (float(structure.blade_mass_properties(eqx.tree_at(lambda q: q.c, p, p.c + h * d), mat).inertia[2, 2])
          - float(structure.blade_mass_properties(eqx.tree_at(lambda q: q.c, p, p.c - h * d), mat).inertia[2, 2])) / (2 * h)
    assert np.isclose(float(jnp.sum(grad * d)), fd, rtol=1e-6)
    prop2 = Propeller.default(inputs=Inputs(coeff_method=0, n_r=9))
    batched = jax.vmap(structure.blade_mass_properties, in_axes=(0, None))(prop2.batched(), prop2.material())
    assert batched.mass.shape == (2,) and batched.inertia.shape == (2, 3, 3)


# ---------------------------------------------------------------------------
# Skin over a core
# ---------------------------------------------------------------------------

_COMPOSITE = dict(skin_thickness_m=4e-4, skin_density_kg_m3=1550.0, skin_E_Pa=45e9, skin_G_Pa=12e9, skin_sigma_allow_Pa=300e6, skin_tau_allow_Pa=60e6,
                  mat_density=80.0, mat_E=80e6, mat_G=30e6, mat_sigma_allow=2e6, mat_tau_allow=1e6)


def _rolling_ball_core(cst, depth, n_x=4000, n_outline=60000):
    # The moments of the core by erosion: at a chord station the top of the core is the lowest point of the circles of
    # radius depth on a very dense outline. It uses no normal and no offset line, and shares no code with the structure module
    t = np.linspace(0.0, np.pi, n_outline)
    xo = 0.5 * (1.0 - np.cos(t))
    yu, yl = (np.asarray(v) for v in geometry.kulfan_coordinates(jnp.asarray(cst), jnp.asarray(np.clip(xo, 1e-14, 1.0))))
    xs = (np.arange(n_x) + 0.5) / n_x
    top, bottom = np.full(n_x, np.inf), np.full(n_x, -np.inf)
    gu, gl = np.interp(xs, xo, yu), np.interp(xs, xo, yl)
    for i, x in enumerate(xs):
        lo, hi = np.searchsorted(xo, x - depth), np.searchsorted(xo, x + depth)
        drop = np.sqrt(np.maximum(depth ** 2 - (xo[lo:hi] - x) ** 2, 0.0))
        if hi > lo:
            top[i] = np.min(yu[lo:hi] - drop)
            bottom[i] = np.max(yl[lo:hi] + drop)
        else:
            top[i], bottom[i] = gu[i] - depth, gl[i] + depth
    exists = (top > bottom) & (xs > depth) & (xs < 1.0 - depth)
    top, bottom = np.where(exists, top, 0.0), np.where(exists, bottom, 0.0)
    dx = 1.0 / n_x
    t_ = top - bottom
    return (np.sum(t_) * dx, np.sum(t_ * xs) * dx, np.sum(0.5 * (top ** 2 - bottom ** 2)) * dx,
            np.sum((top ** 3 - bottom ** 3) / 3.0) * dx, np.sum(t_ * xs ** 2) * dx, np.sum(xs * 0.5 * (top ** 2 - bottom ** 2)) * dx)


def _core_moments(cst, depth):
    from proptimiser import structure

    x, w = jnp.asarray(structure.GAUSS_X), jnp.asarray(structure.GAUSS_W)
    cst = jnp.asarray(cst)[None]
    depth = jnp.asarray([depth])
    x_out = jnp.concatenate([jnp.zeros(1), x, jnp.ones(1)])
    span = (1.0 - 2.0 * depth)[:, None]
    x_core = depth[:, None] + span * x_out
    top, bottom, exists = structure._core_surfaces(cst, depth, x_core, x)
    return [float(v[0]) for v in structure.strip_moments(x_core[:, 1:-1], span * w, top[:, 1:-1], bottom[:, 1:-1])]


def test_core_offset_is_normal_to_the_outline():
    # Against erosion by a rolling ball, for a thin skin and for skins that close the trailing edge and fold at the
    # leading edge. Skin mass is the difference of the outline and the core, so the area tolerance is on the skin.
    from proptimiser import structure

    x, w = jnp.asarray(structure.GAUSS_X), jnp.asarray(structure.GAUSS_W)
    for name, depths in (("clarky", (0.002, 0.005, 0.01, 0.02)), ("naca0012", (0.005, 0.02, 0.04)), ("naca0006", (0.005, 0.02))):
        cst = np.asarray(geometry.kulfan_from_name(name))
        y_u, y_l = geometry.kulfan_coordinates(jnp.asarray(cst), x)
        A0 = float(structure.strip_moments(x, w, y_u, y_l)[0])
        for depth in depths:
            exact = _rolling_ball_core(cst, depth)
            got = _core_moments(cst, depth)
            skin_exact = A0 - exact[0]
            assert abs((A0 - got[0]) / skin_exact - 1.0) < 3e-3, (name, depth, A0 - got[0], skin_exact)
            assert abs(got[1] / got[0] - exact[1] / exact[0]) < 2e-3, (name, depth)
            assert abs(got[2] / got[0] - exact[2] / exact[0]) < 2e-3 * 0.1, (name, depth)
            assert abs(got[3] / exact[3] - 1.0) < 5e-3, (name, depth)
            assert abs(got[4] / exact[4] - 1.0) < 1e-2, (name, depth)


def test_thin_wall_section_closed_form():
    # Thin skin: the area of the skin is P t and its second moments are t times the line integrals round the outline
    from proptimiser import structure

    cst = np.asarray(geometry.kulfan_from_name("naca0012"))
    t = np.linspace(1e-4, np.pi - 1e-4, 200001)
    xo = 0.5 * (1.0 - np.cos(t))
    yu, yl = (np.asarray(v) for v in geometry.kulfan_coordinates(jnp.asarray(cst), jnp.asarray(xo)))
    def line_integral(y):
        mid_x, mid_y = 0.5 * (xo[1:] + xo[:-1]), 0.5 * (y[1:] + y[:-1])
        ds = np.hypot(np.diff(xo), np.diff(y))
        return np.sum(ds), np.sum(ds * mid_x), np.sum(ds * mid_y ** 2), np.sum(ds * mid_x ** 2)
    upper, lower = line_integral(yu), line_integral(yl)
    P, S, I_z2, I_x2 = (u + l for u, l in zip(upper, lower))

    x, w = jnp.asarray(structure.GAUSS_X), jnp.asarray(structure.GAUSS_W)
    y_u, y_l = geometry.kulfan_coordinates(jnp.asarray(cst), x)
    gross = structure.strip_moments(x, w, y_u, y_l)
    depth = 5e-4
    core = _core_moments(cst, depth)
    assert np.isclose(float(gross[0]) - core[0], P * depth, rtol=2e-3)
    assert np.isclose(float(gross[4]) - core[4], I_x2 * depth, rtol=5e-3)
    assert np.isclose(float(gross[3]) - core[3], I_z2 * depth, rtol=1e-2)
    assert np.isclose(float(gross[1]) - core[1], S * depth, rtol=5e-3)


def test_zero_skin_is_the_isotropic_section():
    from proptimiser import structure

    prop = Propeller.default(inputs=Inputs(coeff_method=0, n_r=9), blade=_swept_blade(1.0), c=0.03)
    p = prop.condition(1)
    plain = prop.material()
    other = structure.Layup.from_blade(dataclasses.replace(prop.blade, skin_density_kg_m3=200.0, skin_E_Pa=1e9, skin_G_Pa=2e9, skin_sigma_allow_Pa=7e6))
    a, b = structure.section_properties(p, plain), structure.section_properties(p, other)
    assert np.array_equal(np.asarray(a.core_area), np.asarray(a.area))              # the core is the outline, bit for bit
    assert np.array_equal(np.asarray(a.mass.total), np.asarray(b.mass.total))       # and the skin values have no effect
    assert np.array_equal(np.asarray(a.GJ), np.asarray(b.GJ))
    assert np.allclose(np.asarray(a.mass.total), 1700.0 * np.asarray(a.area), rtol=1e-15)
    assert np.allclose(np.asarray(a.mass.x_c), np.asarray(a.x_c), rtol=1e-13, atol=1e-16)
    assert np.allclose(np.asarray(a.stiffness.I_xx), 19.7e9 * np.asarray(a.I_xx), rtol=1e-13)
    assert np.array_equal(np.asarray(a.core_outline), np.asarray(a.outline))

    stress_a = structure.stress_fields(structure.structural_loads(p, plain), plain)
    stress_b = structure.stress_fields(structure.structural_loads(p, other), other)
    assert np.allclose(np.asarray(stress_a.sigma), np.asarray(stress_b.sigma), rtol=1e-12, atol=1e-6)
    assert np.allclose(np.asarray(stress_a.core_sigma_max), np.asarray(stress_a.sigma_max), rtol=1e-12, atol=1e-6)
    assert np.allclose(np.asarray(stress_a.safety_factor)[:, :-1], np.asarray(stress_b.safety_factor)[:, :-1], rtol=1e-12)


def test_equal_regions_are_the_isotropic_blade():
    # A skin of the same material as the core: the mass, the stiffness and every stress are those of the solid blade
    from proptimiser import structure

    blade = dataclasses.replace(_swept_blade(1.0), N=3)
    prop = Propeller.default(inputs=Inputs(coeff_method=0, n_r=9), blade=blade, c=0.03, beta_deg=20.0)
    p = prop.condition(1)
    solid = prop.material()
    equal = structure.Layup.from_blade(dataclasses.replace(blade, skin_thickness_m=5e-4))
    a, b = structure.propeller_mass_properties(p, solid), structure.propeller_mass_properties(p, equal)
    assert np.isclose(float(a.mass), float(b.mass), rtol=1e-14)
    assert np.allclose(np.asarray(a.cg), np.asarray(b.cg), rtol=1e-13, atol=1e-16)
    assert np.allclose(np.asarray(a.inertia), np.asarray(b.inertia), rtol=1e-13, atol=1e-18)

    sa, sb = structure.section_properties(p, solid), structure.section_properties(p, equal)
    for f in ("total", "x_c", "z_c", "I_xx", "I_zz", "I_xz"):
        assert np.allclose(np.asarray(getattr(sa.stiffness, f)), np.asarray(getattr(sb.stiffness, f)), rtol=1e-13, atol=1e-12), f
    assert np.allclose(np.asarray(sa.GJ), np.asarray(sb.GJ), rtol=1e-3)     # J of the regions: the difference of two Js

    ra, rb = structure.structural_loads(p, solid), structure.structural_loads(p, equal)
    assert np.allclose(np.asarray(ra.F), np.asarray(rb.F), rtol=1e-12, atol=1e-9)
    stress_a, stress_b = structure.stress_fields(ra, solid), structure.stress_fields(rb, equal)
    assert np.allclose(np.asarray(stress_a.sigma), np.asarray(stress_b.sigma), rtol=1e-11, atol=1e-5)


def test_the_skin_closes_at_a_thin_section():
    from proptimiser import structure

    blade = BladeSpec(**_COMPOSITE)
    prop = Propeller.default(inputs=Inputs(coeff_method=0, n_r=9), blade=blade, c=0.03)
    p, mat = prop.condition(1), prop.material()
    # a chord that falls to a section thinner than two skins at the tip
    q = eqx.tree_at(lambda x: x.c, p, jnp.linspace(0.03, 0.0035, 9))
    props = structure.section_properties(q, mat)
    assert float(props.core_area[-1]) == 0.0 and not np.any(np.asarray(props.core_exists)[-1])
    assert float(props.core_area[0]) > 0.5 * float(props.area[0])
    # solid skin there: the mass is the skin density times the outline, and the stiffness the same
    assert np.isclose(float(props.mass.total[-1]), 1550.0 * float(props.area[-1]), rtol=1e-14)
    assert np.isclose(float(props.stiffness.I_zz[-1]), 45e9 * float(props.I_zz[-1]), rtol=1e-13)
    assert np.isclose(float(props.mass.x_c[-1]), float(props.x_c[-1]), rtol=1e-12)
    for value in (props.mass.total, props.mass.I_xx, props.stiffness.I_zz, props.GJ, props.GA):
        assert np.all(np.isfinite(np.asarray(value)))

    # a skin thicker than half the chord is still the solid skin
    thick = structure.Layup.from_blade(dataclasses.replace(blade, skin_thickness_m=0.1))
    full = structure.section_properties(q, thick)
    assert not np.any(np.asarray(full.core_area)) and np.allclose(np.asarray(full.mass.total), 1550.0 * np.asarray(full.area), rtol=1e-14)
    stress = structure.stress_fields(structure.structural_loads(q, thick, include_aero=False), thick)
    assert np.all(np.isfinite(np.asarray(stress.sigma))) and not np.any(np.asarray(stress.core_sigma_max))

    # the core ends before the trailing edge: x stays inside the skin
    props = structure.section_properties(p, mat)
    core = np.asarray(props.core_outline)[0]
    exists = np.asarray(props.core_exists)[0]
    outline = np.asarray(props.outline)[0]
    assert core[exists, 0].max() < outline[:, 0].max() - 4e-4 * 0.99 and core[exists, 0].min() > outline[:, 0].min() + 4e-4 * 0.99


def test_mass_and_stiffness_weighted_centroids():
    from proptimiser import structure

    prop = Propeller.default(inputs=Inputs(coeff_method=0, n_r=9), blade=BladeSpec(**_COMPOSITE), c=0.04, beta_deg=25.0)
    p, mat = prop.condition(1), prop.material()
    props = structure.section_properties(p, mat)
    # a skin is further from the centre than a core: more of the mass is at the leading edge region, away from the centroid
    i = 5
    assert abs(float(props.mass.x_c[i] - props.x_c[i])) > 1e-5
    assert abs(float(props.stiffness.x_c[i] - props.x_c[i])) > 1e-5
    # the section moves by hand: mass weights from the regions
    gross_m = float(props.area[i])
    core_m = float(props.core_area[i])
    expected_mass = 1550.0 * (gross_m - core_m) + 80.0 * core_m
    assert np.isclose(float(props.mass.total[i]), expected_mass, rtol=1e-13)

    # the slices of the inertial loads sit at the mass-weighted centroid
    axes = structure.blade_axes(p)
    slices = structure.blade_slices(props, axes)
    expected = np.asarray(axes.p[i] + props.mass.x_c[i] * axes.x_s[i] + props.mass.z_c[i] * axes.z_s[i])
    assert np.allclose(np.asarray(slices.p_cg[i]), expected)
    loads = structure.inertial_loads(p, props, None, axes)
    w = float(p.omega)
    assert np.allclose(np.asarray(loads.f)[0, i, :2], w ** 2 * float(slices.mass[i]) * expected[:2], rtol=1e-10)


def test_stress_on_the_transformed_section():
    # The stress of each region integrated over the regions gives back the section forces
    from proptimiser import structure

    prop = Propeller.default(inputs=Inputs(coeff_method=0, n_r=9), blade=BladeSpec(**_COMPOSITE), c=0.05, beta_deg=25.0)
    p, mat = prop.condition(1), prop.material()
    res = structure.structural_loads(p, mat, include_aero=False)
    props, n_psi, n_r = res.props, *res.resultants.N.shape
    zeros = jnp.zeros((n_psi, n_r))
    N_load, M_x, M_z = 120.0, 0.8, -0.35
    loads = structure.Resultants(jnp.full((n_psi, n_r), N_load), zeros, zeros, jnp.full((n_psi, n_r), M_x), zeros, jnp.full((n_psi, n_r), M_z))
    stress = structure.stress_fields(eqx.tree_at(lambda x: x.resultants, res, loads), mat)

    i = 4
    c = float(p.c[i])
    cst = np.asarray(geometry.cst_at_stations(p.r, p.r_cst, p.cst))[i]
    # the strain field from the outer stress at the outline: sigma / E_skin = a + b x + d z
    outline = np.asarray(props.outline[i])
    sigma = np.asarray(stress.sigma)[0, i]
    design = np.column_stack([np.ones(len(outline)), outline])
    coeff = np.linalg.lstsq(design, sigma / 45e9, rcond=None)[0]
    assert np.allclose(design @ coeff, sigma / 45e9, atol=1e-6 * np.abs(sigma).max() / 45e9)

    # integrate E strain over the regions: the strip moments of the outline and of the core
    x, w = jnp.asarray(structure.GAUSS_X), jnp.asarray(structure.GAUSS_W)
    y_u, y_l = geometry.kulfan_coordinates(jnp.asarray(cst), x)
    gross = [float(v) for v in structure.strip_moments(x, w, y_u, y_l)]
    core = _core_moments(cst, 4e-4 / c)
    # moments about the outline point coordinates: x_o = c (x - x1), z_o = c (z - z1), from the gross centroid
    A1, x1, z1 = gross[0], gross[1] / gross[0], gross[2] / gross[0]
    def about_centroid(m):
        a, sx, sz, zz, xx, xz = m
        return (a * c ** 2, (sx - a * x1) * c ** 3, (sz - a * z1) * c ** 3, (zz - 2 * z1 * sz + a * z1 ** 2) * c ** 4,
                (xx - 2 * x1 * sx + a * x1 ** 2) * c ** 4, (xz - x1 * sz - z1 * sx + a * x1 * z1) * c ** 4)
    g, k = about_centroid(gross), about_centroid(core)
    region = [45e9 * (gv - kv) + 80e6 * kv for gv, kv in zip(g, k)]
    a0, b0, d0 = coeff
    N_back = a0 * region[0] + b0 * region[1] + d0 * region[2]
    M_z_back = a0 * region[1] + b0 * region[4] + d0 * region[5]
    M_x_back = -(a0 * region[2] + b0 * region[5] + d0 * region[3])
    assert np.isclose(N_back, N_load, rtol=1e-9)
    # the moments are about the stiffness-weighted centroid, where N has no moment
    e = props.stiffness
    arm_x, arm_z = float(e.x_c[i] - props.x_c[i]), float(e.z_c[i] - props.z_c[i])
    assert np.isclose(M_z_back - arm_x * N_load, M_z, rtol=1e-8)
    assert np.isclose(M_x_back + arm_z * N_load, M_x, rtol=1e-8)

    # the core carries the same strain at its own, lower modulus
    assert 0.0 < float(stress.core_sigma_max[0, i]) < 80e6 / 45e9 * 1.02 * float(stress.sigma_max[0, i])
    # the torsion and the shear go to the regions by G
    assert np.isclose(float(res.props.GA[i]), 12e9 * (float(props.area[i]) - float(props.core_area[i])) + 30e6 * float(props.core_area[i]), rtol=1e-13)
    # the safety factor sees the foam and the skin: a weak foam governs
    weak = dataclasses.replace(prop.blade, mat_sigma_allow=1e3, mat_tau_allow=1e3)
    weak_mat = structure.Layup.from_blade(weak)
    strong_sf = np.asarray(structure.stress_fields(res, mat).safety_factor)[:, :-1]
    weak_sf = np.asarray(structure.stress_fields(res, weak_mat).safety_factor)[:, :-1]
    assert np.all(weak_sf < strong_sf)


def test_bredt_torsion_of_a_thin_skin():
    # A thin closed skin: GJ of the difference of two Saint-Venant constants is near the Bredt value 4 A^2 t / P
    from proptimiser import structure

    t = 3e-4
    blade = BladeSpec(**{**_COMPOSITE, "skin_thickness_m": t, "mat_G": 1.0})
    prop = Propeller.default(inputs=Inputs(coeff_method=0, n_r=9), blade=blade, c=0.05)
    p, mat = prop.condition(0), prop.material()
    props = structure.section_properties(p, mat)
    i = 5
    outline = np.asarray(props.outline[i])
    P = np.sum(np.hypot(*np.diff(np.concatenate([outline, outline[:1]]), axis=0).T))
    A = float(props.area[i]) - 0.5 * P * t
    bredt = 4 * A ** 2 * t / P * 12e9
    assert 0.9 < float(props.GJ[i]) / bredt < 1.2, float(props.GJ[i]) / bredt


def test_skin_mass_gradient():
    from proptimiser import structure

    prop = Propeller.default(inputs=Inputs(coeff_method=0, n_r=9), blade=BladeSpec(**_COMPOSITE), c=0.04)
    p, mat = prop.condition(0), prop.material()
    objective = lambda t: structure.blade_mass_properties(p, eqx.tree_at(lambda m: m.skin_thickness, mat, t)).inertia[2, 2]
    h = 1e-7
    fd = (float(objective(mat.skin_thickness + h)) - float(objective(mat.skin_thickness - h))) / (2 * h)
    assert np.isclose(float(jax.grad(objective)(mat.skin_thickness)), fd, rtol=1e-3)
    grad = jax.grad(lambda t: jnp.sum(structure.section_properties(p, eqx.tree_at(lambda m: m.skin_thickness, mat, t)).GJ))(jnp.asarray(0.0))
    assert np.isfinite(float(grad))


def test_real_design_with_a_skin_against_an_independent_two_density_solid():
    # The outline and the core are separate closed stacks. The core comes from geometry.offset_airfoil_profile, which
    # finds its normals by finite differences. The regions then add with their densities
    prop = Propeller.load("MAD_FLUXER_22x6_6_jung")
    rho_s, rho_c = 1550.0, 80.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stack = _design_stack(prop, n_points_per_side=600)
        r, c, beta, pitch_axis, offsets, coords = stack
        V0, S0, T0 = _closed_stack_properties(_section_loops(*stack))
        for thickness in (3e-4, 6e-4):
            layered = prop.with_blade(dataclasses.replace(prop.blade, skin_thickness_m=thickness, skin_density_kg_m3=rho_s, mat_density=rho_c))
            cores, kept = [], []
            for i in range(len(r)):
                core = geometry.offset_airfoil_profile(coords[i, :, 0], coords[i, :, 1], thickness, n_points_per_side=600)
                if core is not None:
                    cores.append(np.column_stack(core)[:-1])
                    kept.append(i)
            kept = np.array(kept)
            assert kept[-1] - kept[0] + 1 == len(kept) and len(kept) > 100
            V1, S1, T1 = _closed_stack_properties(_section_loops(r[kept], c[kept], beta[kept], pitch_axis[kept], offsets[kept], np.array(cores)))
            mass, first, tensor = rho_s * (V0 - V1) + rho_c * V1, rho_s * (S0 - S1) + rho_c * S1, rho_s * (T0 - T1) + rho_c * T1
            got = layered.blade_mass_properties()
            assert abs(float(got.mass) / mass - 1.0) < 2e-3, (thickness, float(got.mass), mass)
            assert np.linalg.norm(np.asarray(got.cg) - first / mass) < 3e-4, (thickness, np.asarray(got.cg), first / mass)
            assert abs(float(got.inertia[2, 2]) / tensor[2, 2] - 1.0) < 2e-3, (thickness, float(got.inertia[2, 2]), tensor[2, 2])
            assert abs(float(got.inertia[0, 0]) / tensor[0, 0] - 1.0) < 5e-3, (thickness, float(got.inertia[0, 0]), tensor[0, 0])


def test_design_files_hold_the_skin_keys():
    from proptimiser.init import design_from_dict, design_to_dict, _design_keys
    import yaml

    designs = sorted((Path(__file__).resolve().parents[1] / "designs").glob("*.yaml"))
    assert designs
    for path in designs:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        _, blade = design_from_dict(data, path.name)
        assert blade.skin_thickness_m == 0.0, path.name                        # one material, as before the regions existed
        # a missing key is an error: no key fills a default
        short = {**data, "blade": {k: v for k, v in data["blade"].items() if k != "skin_thickness_m"}}
        try:
            design_from_dict(short, path.name)
        except ValueError as e:
            assert "skin_thickness_m" in str(e)
        else:
            raise AssertionError("A design without skin_thickness_m loaded")

    # the keys go through design saving, and an invalid skin is refused
    blade = BladeSpec(**_COMPOSITE)
    prop = Propeller.default(inputs=Inputs(coeff_method=0, n_r=6), blade=blade)
    with tempfile.TemporaryDirectory() as tmp:
        back = Propeller.load(prop.save(Path(tmp) / "skin"))
    assert back.blade.skin_thickness_m == 4e-4 and back.blade.skin_E_Pa == 45e9 and back.blade == prop.blade
    for bad in (dict(skin_thickness_m=-1e-3), dict(skin_density_kg_m3=0.0), dict(skin_tau_allow_Pa=-1.0)):
        try:
            BladeSpec(**bad)
        except ValueError:
            continue
        raise AssertionError(f"BladeSpec accepted {bad}")


# ---------------------------------------------------------------------------
# The Rosella lift blade (docs/fluxer_22x6_6.md, Mass and inertia)
# ---------------------------------------------------------------------------

def _scaled_propeller(prop, scale):
    blade = prop.blade
    blade = dataclasses.replace(blade, r_foils=tuple(v * scale for v in blade.r_foils), hub_chord=blade.hub_chord * scale,
                                c_bounds=tuple(v * scale for v in blade.c_bounds))
    return Propeller(prop.inputs, blade, c=np.asarray(prop.c) * scale, beta=prop.beta, cst=prop.cst)


def _rotor_of_mass(prop, rotor_kg, hardware_kg, foam):
    # The rotor with the skin thickness that holds the rotor mass: two blades of skin 1550 kg/m^3 over the foam, and the hardware
    from scipy.optimize import brentq
    from proptimiser import structure

    p = prop.condition(0)
    mat = lambda t: structure.Layup.from_blade(dataclasses.replace(prop.blade, skin_thickness_m=t, skin_density_kg_m3=1550.0, mat_density=foam))
    t = brentq(lambda v: float(structure.blade_mass_properties(p, mat(v)).mass) - 0.5 * (rotor_kg - hardware_kg), 1e-6, 3e-3, xtol=1e-8)
    return t, structure.propeller_mass_properties(p, mat(t), structure.hub_hardware(hardware_kg, 0.015, 0.012))


def test_rosella_blade_inertia():
    prop = Propeller.load("MAD_FLUXER_22x6_6_jung")
    # the design holds the weighed mass: 49 g for the rotor, with 4 g of estimated hardware
    assert np.isclose(float(prop.blade_mass_properties().mass), 0.0225, rtol=1e-3)
    assert prop.blade.skin_thickness_m == 0.0

    # best estimates and the rod of the same mass
    t, rotor = _rotor_of_mass(prop, 0.049, 0.004, 80.0)
    assert np.isclose(t, 0.438e-3, atol=1e-5) and np.isclose(float(rotor.mass), 0.049, rtol=1e-6)
    assert np.isclose(float(rotor.inertia[2, 2]), 9.46e-4, rtol=2e-3)
    assert np.isclose(float(rotor.inertia[1, 1]), 9.32e-4, rtol=2e-3) and np.isclose(float(rotor.inertia[0, 0]), 1.56e-5, rtol=5e-3)
    assert float(rotor.inertia[2, 2]) < 0.8 * 0.049 * 0.5588 ** 2 / 12

    scale, mass = 0.33019978264002237 / 0.2794, 0.06827590134598946
    big = _scaled_propeller(prop, scale)
    t, rotor = _rotor_of_mass(big, mass, 0.004, 80.0)
    assert np.isclose(t, 0.426e-3, atol=1e-5)
    assert np.isclose(float(rotor.inertia[2, 2]), 1.881e-3, rtol=2e-3)
    assert np.isclose(float(rotor.inertia[2, 2]) / (mass * (2 * 0.33019978264002237) ** 2), 0.0632, rtol=2e-3)

    # the ends of the family: the rod stays above all of it
    low = _rotor_of_mass(big, mass, 0.012, 200.0)[1]
    high = _rotor_of_mass(big, mass, 0.0, 1e-3)[1]
    assert np.isclose(float(low.inertia[2, 2]), 1.537e-3, rtol=2e-3) and np.isclose(float(high.inertia[2, 2]), 2.064e-3, rtol=2e-3)
    assert float(high.inertia[2, 2]) < mass * (2 * 0.33019978264002237) ** 2 / 12


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

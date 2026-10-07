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

import init
from init import Inputs, BladeSpec, Propeller
import geometry


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
    import airflow

    m = 0.15
    base = BladeSpec()
    blade = dataclasses.replace(base, offset_r=(base.R_hub, base.R_tip), offset_z=(m * base.R_hub, m * base.R_tip))
    p = Propeller.default(inputs=Inputs(coeff_method=0, n_r=6), blade=blade).condition(0)
    assert p.cfg.swept and np.allclose(np.asarray(p.offsets[:, 3]), m) and np.allclose(np.asarray(p.offsets[:, 0]), 0.0)

    rng = np.random.default_rng(3)
    u_a, u_t, u_rho = (jnp.asarray(rng.uniform(1.0, 20.0, (6, 4))) for _ in range(3))
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
    import propmodel

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
    import propmodel

    p = Propeller.default(inputs=Inputs(coeff_method=0, n_r=8), blade=_swept_blade(1.0)).condition(1)
    _, V_newton = propmodel.induced_velocity(p)
    bracketed = dataclasses.replace(p, cfg=dataclasses.replace(p.cfg, V_i_0_max_iter=0))
    _, V_bracketed = propmodel.induced_velocity(bracketed)
    assert np.allclose(V_bracketed, V_newton, atol=1e-6)

    idx = jnp.array([5, 2])
    sub = propmodel._station_subset(p, idx)
    assert np.array_equal(np.asarray(sub.offsets), np.asarray(p.offsets)[[5, 2]])


def test_sectional_loads_integrate_to_the_blade_element_sums():
    import propmodel

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
    import propmodel

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
    import structure

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
    import structure
    from aerosandbox.geometry.airfoil.airfoil_families import get_kulfan_coordinates

    prop = Propeller.default(inputs=Inputs(coeff_method=0, n_r=8), c=0.04)
    p = prop.condition(0)
    props = structure.section_properties(p)
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
    import structure

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
    import structure

    # The hub section is a stadium fitted by a CST: its area follows the analytic stadium to a fraction of a percent
    prop = Propeller.default(inputs=Inputs(coeff_method=0, n_r=8), c=0.04)
    p = prop.condition(0)
    t = prop.blade.hub_thickness_ratio
    area = float(structure.section_properties(p).area[0]) / float(p.c[0]) ** 2
    assert np.isclose(area, t * (1 - t) + np.pi * t ** 2 / 4, rtol=3e-3)


def test_section_area_matches_apc_outboard():
    # The APC 10x6 geometry file lists the cross-section area of its Clark-Y sections, scaled to a thickness ratio. The
    # CST section of the same thickness holds a constant 94% of it outboard of 3 in (the APC section is a fuller
    # Clark-Y). Inboard, the moulded root blend is not a CST airfoil, and the rounded tip (chord below 0.25 in) is not
    # either, so the model sits lower there.
    import structure

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
    import structure

    mat = structure.Material.from_blade(BladeSpec())
    assert float(mat.density) == 1700.0 and float(mat.fill) == 1.0
    assert np.isclose(float(mat.tau_allow), float(mat.sigma_allow) / np.sqrt(3.0))
    mat = Propeller.default(blade=BladeSpec(mat_tau_allow=25e6, fill_factor=0.4)).material()
    assert float(mat.tau_allow) == 25e6 and float(mat.fill) == 0.4


def test_section_properties_vmap_and_gradient():
    import structure

    prop = Propeller.default(inputs=Inputs(coeff_method=0, n_r=6))
    batched = jax.vmap(structure.section_properties)(prop.batched())
    assert batched.area.shape == (2, 6) and batched.outline.shape == (2, 6, 2 * (structure.N_X + 2), 2)
    assert np.allclose(np.asarray(batched.area[0]), np.asarray(batched.area[1]))

    p = prop.condition(0)
    grad = jax.grad(lambda c: jnp.sum(structure.section_properties(eqx.tree_at(lambda q: q.c, p, c)).area))(p.c)
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
    return N * float(mat.density * mat.fill) * np.trapezoid(np.asarray(area) * r ** 2, r)


def test_uniform_load_shear_and_moment():
    import structure

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
    import structure

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
    import structure

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
    import structure

    prop, p = _rod_propeller(n_r=24)
    mat = prop.material()
    res = structure.structural_loads(p, mat, include_aero=False)
    A = float(res.props.area[0])
    assert np.allclose(np.asarray(res.props.area), A)
    r, omega = np.asarray(p.r), float(p.omega)
    expected = float(mat.density) * A * omega ** 2 * (r[-1] ** 2 - r ** 2) / 2     # exact: the trapezoid rule integrates r
    assert np.allclose(np.asarray(res.resultants.N), expected, rtol=1e-11, atol=1e-9)
    assert np.allclose(np.asarray(res.F_aero), 0.0)


def test_centrifugal_twisting_moment():
    import structure

    # steady rotation: the moment of a section about its centroid along the span is
    #   -rho omega^2 [sin(b) cos(b) (I_zz - I_xx) - I_xz cos(2b)]: the classic twisting moment, toward fine pitch
    for beta_deg in (40.0, 15.0, 70.0):
        prop, p = _rod_propeller(beta_deg=beta_deg)
        mat, props, axes = prop.material(), structure.section_properties(p), structure.blade_axes(p)
        loads = structure.inertial_loads(p, props, mat, None, axes)
        b = np.radians(beta_deg)
        expected = -float(mat.density) * float(p.omega) ** 2 * (
            np.sin(b) * np.cos(b) * (np.asarray(props.I_zz) - np.asarray(props.I_xx)) - np.asarray(props.I_xz) * np.cos(2 * b))
        along_span = np.einsum("prk,rk->pr", np.asarray(loads.m_cg), np.asarray(axes.s))
        assert np.allclose(along_span, expected, rtol=1e-9)
        assert np.all(along_span[0] < 0)                       # toward fine pitch: a rotation about -s lowers beta

    # a symmetric section has no twisting moment at zero and quarter-turn twist
    symmetric = BladeSpec(airfoils=("naca0012",))
    reference = None
    for beta_deg in (45.0, 0.0, 90.0):
        prop, p = _rod_propeller(beta_deg=beta_deg, blade=symmetric)
        mat, props, axes = prop.material(), structure.section_properties(p), structure.blade_axes(p)
        loads = structure.inertial_loads(p, props, mat, None, axes)
        along_span = np.abs(np.einsum("prk,rk->pr", np.asarray(loads.m_cg), np.asarray(axes.s)))[0, 3]
        reference = along_span if beta_deg == 45.0 else reference
        assert beta_deg == 45.0 or along_span < 2e-3 * reference


def test_root_moment_is_the_sum_of_section_moments():
    import structure

    prop, p = _rod_propeller(beta_deg=25.0, blade=_swept_blade(1.0))
    mat = prop.material()
    kin = structure.Kinematics(pitch_rate=jnp.asarray(0.4), omega_dot=jnp.asarray(30.0), body_rates=jnp.array([0.2, -0.3, 0.1]))
    res = structure.structural_loads(p, mat, kin, include_aero=False)
    loads = structure.inertial_loads(p, res.props, mat, kin, res.axes)
    per_length = np.asarray(jnp.cross(res.axes.p[None], loads.f) + loads.m)
    dr = np.diff(np.asarray(p.r))
    direct = np.sum(0.5 * (per_length[:, :-1] + per_length[:, 1:]) * dr[None, :, None], axis=1)
    assert np.allclose(np.asarray(res.M_O)[:, 0], direct, rtol=1e-12, atol=1e-12)
    force = np.sum(0.5 * (np.asarray(loads.f)[:, :-1] + np.asarray(loads.f)[:, 1:]) * dr[None, :, None], axis=1)
    assert np.allclose(np.asarray(res.F)[:, 0], force, rtol=1e-12, atol=1e-12)


def test_pitch_rate_coriolis():
    import structure

    prop, p = _rod_propeller(beta_deg=20.0)
    mat, props, axes = prop.material(), structure.section_properties(p), structure.blade_axes(p)
    rate = 0.8
    loads = structure.inertial_loads(p, props, mat, structure.Kinematics(pitch_rate=jnp.asarray(rate)), axes)
    i = 5
    p_cg = np.asarray(axes.p[i] + props.x_c[i] * axes.x_s[i] + props.z_c[i] * axes.z_s[i])
    mass = float(mat.density * props.area[i] * axes.arc[i])
    v_rel = np.cross([rate, 0.0, 0.0], p_cg)
    assert np.allclose(np.asarray(loads.f_coriolis)[0, i], -mass * 2.0 * np.cross([0.0, 0.0, float(p.omega)], v_rel))
    assert np.allclose(np.asarray(loads.f_relative)[0, i], -mass * np.cross([rate, 0.0, 0.0], v_rel))

    # a rigid blade in steady rotation has no Coriolis, Euler or relative terms
    steady = structure.inertial_loads(p, props, mat, None, axes)
    for term in (steady.f_coriolis, steady.f_relative, steady.f_euler):
        assert not np.any(np.asarray(term))
    assert np.allclose(np.asarray(steady.f), np.asarray(steady.f_centrifugal))


def test_spin_acceleration_torque():
    import structure

    prop, p = _rod_propeller(n_r=30, N=2, c=0.01)
    mat = prop.material()
    rate = 50.0
    res = structure.structural_loads(p, mat, structure.Kinematics(omega_dot=jnp.asarray(rate)), include_aero=False)
    expected = _polar_inertia(p, res.props.area, mat, 2) * rate
    assert np.isclose(float(-res.hub.moment[:, 2].mean()), expected, rtol=0.02)          # the shaft supplies I * omega_dot
    doubled = structure.structural_loads(p, mat, structure.Kinematics(omega_dot=jnp.asarray(2 * rate)), include_aero=False)
    assert np.isclose(float(-doubled.hub.moment[:, 2].mean()), 2 * float(-res.hub.moment[:, 2].mean()), rtol=1e-12)


def test_gyroscopic_hub_moment():
    import structure

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
    import structure

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
    import structure

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
    import propmodel
    import structure

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
    import structure

    for fill in (1.0, 0.5):
        prop, p = _rod_propeller(blade=BladeSpec(fill_factor=fill))
        mat = prop.material()
        res = structure.structural_loads(p, mat, include_aero=False)
        N = res.resultants.N
        pure = eqx.tree_at(lambda x: x.resultants, res, structure.Resultants(N, *(jnp.zeros_like(N),) * 5))
        stress = structure.stress_fields(pure, mat)
        r, omega = np.asarray(p.r), float(p.omega)
        expected = float(mat.density) * omega ** 2 * (r[-1] ** 2 - r ** 2) / 2
        assert np.allclose(np.asarray(stress.sigma), expected[None, :, None], rtol=1e-11, atol=1e-6)
        assert np.allclose(np.asarray(stress.sigma_max), expected[None, :], rtol=1e-11, atol=1e-6)


def test_unsymmetric_bending_matches_the_moments():
    import structure

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
    import structure

    a, b = 0.5, 0.06
    x, w = structure.GAUSS_X, structure.GAUSS_W
    y = 2 * b * np.sqrt(x * (1 - x))
    A, xc, zc, Ixx, Izz, Ixz = (jnp.atleast_1d(v) for v in structure.strip_properties(
        jnp.asarray(x), jnp.asarray(w), jnp.asarray(y), jnp.asarray(-y)))
    props = structure.SectionProps(A, xc, zc, Ixx, Izz, Ixz, structure.torsion_constant(A, Ixx, Izz), jnp.full(1, 2 * b),
                                   jnp.zeros((1, 2, 2)))
    T = 1.7
    shape = (1, 1)
    zeros = jnp.zeros(shape)
    resultants = structure.Resultants(zeros, zeros, zeros, zeros, jnp.full(shape, T), zeros)
    mat = structure.Material.from_blade(BladeSpec())
    stress = structure.stress_fields(types.SimpleNamespace(props=props, resultants=resultants), mat)
    exact = 2 * T / (np.pi * a * b ** 2)                                    # the largest shear stress of a twisted ellipse
    assert np.isclose(float(stress.tau[0, 0]), exact, rtol=0.03)
    assert np.isclose(float(stress.sigma_vm[0, 0]), np.sqrt(3.0) * float(stress.tau[0, 0]), rtol=1e-12)


def test_safety_factor_scales_with_the_square_of_the_speed():
    import structure

    prop, p = _rod_propeller(beta_deg=30.0)
    mat = prop.material()
    base = structure.stress_fields(structure.structural_loads(p, mat, include_aero=False), mat)
    fast_p = eqx.tree_at(lambda q: q.omega, p, 2.0 * p.omega)
    fast = structure.stress_fields(structure.structural_loads(fast_p, mat, include_aero=False), mat)
    inner = slice(0, -1)                                                    # the tip station carries no load
    assert np.allclose(np.asarray(fast.safety_factor)[:, inner], np.asarray(base.safety_factor)[:, inner] / 4.0, rtol=1e-9)
    assert np.allclose(np.asarray(fast.sigma_vm)[:, inner], 4.0 * np.asarray(base.sigma_vm)[:, inner], rtol=1e-9)


def test_structural_loads_vmap_matches_a_loop():
    import structure

    prop = Propeller.default(inputs=Inputs(coeff_method=0, n_r=8), c=0.03, blade=_swept_blade(1.0))
    mat = prop.material()
    batched = jax.vmap(structure.structural_loads, in_axes=(0, None))(prop.batched(), mat)
    for i in range(prop.inputs.n_cond):
        single = structure.structural_loads(prop.condition(i), mat)
        assert np.allclose(np.asarray(batched.F[i]), np.asarray(single.F), rtol=1e-11, atol=1e-11)
        assert np.allclose(np.asarray(batched.hub.moment[i]), np.asarray(single.hub.moment), rtol=1e-11, atol=1e-11)


def test_structural_gradients_match_finite_differences():
    # A smooth maximum of the von Mises stress, through the BEMT solve (implicit gradients), against central differences
    import propmodel
    import structure

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
    import proptimiser
    import structure

    base = Propeller.default(inputs=Inputs(coeff_method=0, n_r=6, opt_max_iter=2, opt_verbose=0), blade=_swept_blade(1.0))
    for solver in proptimiser.SOLVERS:
        prop = base.with_inputs(dataclasses.replace(base.inputs, opt_solver=solver))
        reports = []
        result = proptimiser.run_optimization(prop, callback=reports.append)
        assert reports and np.isfinite(reports[-1].objective) and reports[-1].objective > 0, solver
        assert result.blade.swept and np.array_equal(np.asarray(result.offsets), np.asarray(base.offsets)), solver
        assert not np.allclose(np.asarray(result.c), np.asarray(base.c)) or not np.allclose(np.asarray(result.beta), np.asarray(base.beta))
        loads = structure.structural_loads(result.condition(0), result.material())
        assert np.all(np.isfinite(np.asarray(loads.F))) and np.all(np.isfinite(np.asarray(loads.hub.moment)))


def test_structural_report():
    import propmodel
    import results

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

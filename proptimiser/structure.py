## Structural loads of a rigid propeller blade: section properties, mass properties, loads along the blade and at
## the hub, stress
#
# The section is a set of material regions over the CST outline of every station: a skin of constant thickness over a
# core (Layup, from BladeSpec.skin_* and BladeSpec.mat_*). The blade is a rigid body as far as its loads are concerned:
# aerodynamic loads from the BEMT, and inertial loads from the rotation of the hub, of the aircraft and of the pitch
# change. Frames, signs and equations are set out in the README section "Structural loads".
#
# Blade frame (e_r, e_t, e_z), right-handed: e_r radial at the root azimuth, e_t the direction of blade motion,
# e_z the thrust direction. Section axes (x_s, y_s, z_s) = (chord direction at the section twist, span direction s,
# the normal toward the suction side), right-handed and equal to the CAD axes of the exported blade.

# External Libraries
from functools import partial
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

# Internal Libraries
from . import init                 # first: it sets the JAX configuration (float64)
from . import geometry
from . import propmodel
from .airflow import inflow_model

N_X = 48                    # chordwise Gauss nodes of the section integrals
OFFSET_SUBDIVISION = 4      # segments of the offset lines of the core between two chordwise nodes
N_R_GAUSS = 4               # Gauss nodes in each radial interval of the mass properties. The chord and the CST are
                            # linear in r in an interval, so this rule is exact for the mass and the moments of a slice

def _chordwise_quadrature(n):
    '''
    Gauss-Legendre nodes in the angle theta (x = (1 - cos theta) / 2) and the weights of integral f(x) dx. The CST
    section depends on sqrt(x) at the leading edge, which is analytic in theta, so these converge much faster than
    nodes and weights in x. Returns (x, weights), each of shape (n,).
    '''
    xi, w = np.polynomial.legendre.leggauss(n)
    theta = 0.5 * np.pi * (xi + 1.0)
    return 0.5 * (1.0 - np.cos(theta)), w * 0.5 * np.pi * 0.5 * np.sin(theta)

GAUSS_X, GAUSS_W = _chordwise_quadrature(N_X)


## Material

class Material(eqx.Module):
    '''
    The material of one section region, isotropic or homogenised, with the fraction of the region that is material.
    Mass, E and G scale with the fill factor (a homogenised infill), and the stress in the material is the gross
    stress divided by it. All fields are float scalars (SI units), so a change of material never recompiles anything.
    '''
    density: jnp.ndarray        # kg/m^3
    E: jnp.ndarray              # Pa
    G: jnp.ndarray              # Pa
    sigma_allow: jnp.ndarray    # Pa
    tau_allow: jnp.ndarray      # Pa
    fill: jnp.ndarray           # (0, 1]

    @classmethod
    def of(cls, density, E, G, sigma_allow, tau_allow, fill=1.0) -> "Material":
        '''
        A region from SI values. A zero allowable shear stress means sigma_allow / sqrt(3), the von Mises value.
        '''
        tau = tau_allow if tau_allow > 0 else sigma_allow / np.sqrt(3.0)
        return cls(*(jnp.asarray(v, dtype=float) for v in (density, E, G, sigma_allow, tau, fill)))

class Layup(eqx.Module):
    '''
    The material regions of a section, from the outside in: a skin over a core that fills the rest of the outline.
    The skin thickness is measured normal to the outline. The skin closes where the section is thinner than two skins
    (trailing edge, tip), and the section is then solid skin. A solid isotropic blade has skin_thickness 0: the core
    is then the whole outline, and the skin values have no effect.

    The skin is one homogenised region, not a ply table: E and G are the effective spanwise values of the laminate
    and skin_thickness is its cured thickness.
    '''
    skin: Material
    core: Material
    skin_thickness: jnp.ndarray     # m

    @classmethod
    def from_blade(cls, blade) -> "Layup":
        '''
        The layup of a BladeSpec: the skin_* keys and the mat_* keys (the core, with the fill factor).
        '''
        skin = Material.of(blade.skin_density_kg_m3, blade.skin_E_Pa, blade.skin_G_Pa, blade.skin_sigma_allow_Pa, blade.skin_tau_allow_Pa)
        core = Material.of(blade.mat_density, blade.mat_E, blade.mat_G, blade.mat_sigma_allow, blade.mat_tau_allow, blade.fill_factor)
        return cls(skin, core, jnp.asarray(blade.skin_thickness_m, dtype=float))

    @property
    def outer(self) -> Material:
        '''
        The region at the outline: the skin, or the core when the skin has no thickness.
        '''
        return jax.tree_util.tree_map(lambda s, c: jnp.where(self.skin_thickness > 0.0, s, c), self.skin, self.core)


## Section properties

class SectionSum(NamedTuple):
    '''
    A section property weighted by a region value (density: mass, E: stiffness), each of shape (n_r,).

    total        int w dA over the regions: mass per unit span (kg/m), or the axial stiffness EA (N)
    x_c, z_c     weighted centroid relative to the pitch axis, on the section axes (m)
    I_xx, I_zz   int w z^2 dA and int w x^2 dA about the weighted centroid (kg m, or N m^2)
    I_xz         int w x z dA about the weighted centroid
    '''
    total: jnp.ndarray
    x_c: jnp.ndarray
    z_c: jnp.ndarray
    I_xx: jnp.ndarray
    I_zz: jnp.ndarray
    I_xz: jnp.ndarray

class SectionProps(eqx.Module):
    '''
    Properties of the section at every station, each of shape (n_r,) unless stated. Lengths in m, areas in m^2, second
    moments in m^4. The first group is the geometry of the gross outline, without the materials:

    x_c, z_c     centroid relative to the pitch axis, on the section axes: x_c along the chord from the pitch axis
                 (positive toward the trailing edge), z_c toward the suction side
    I_xx, I_zz   second moments about the centroid: I_xx = int z^2 dA (flap), I_zz = int x^2 dA (edgewise)
    I_xz         product of inertia int x z dA about the centroid
    J            torsion constant, Saint-Venant's approximation A^4 / (40 (I_xx + I_zz)) (within 10% for thin
                 sections); the shear centre is taken at the centroid
    t_max        largest thickness of the section
    outline      (n_r, 2 (N_X + 2), 2): points (x, z) of the outline relative to the centroid, upper surface from the
                 leading to the trailing edge, then the lower surface back, used for the stress at the extreme fibres

    The second group holds the regions:

    core_area    area of the core (the skin area is area - core_area)
    core_outline (n_r, 2 (N_X + 2), 2): points of the core outline relative to the gross centroid, in the same order
    core_exists  (n_r, 2 (N_X + 2)): the point is on the core (False where the skin has closed)
    mass         SectionSum weighted by density and fill: the inertial loads and the mass properties use it
    stiffness    SectionSum weighted by E and fill: the bending uses it
    GA           shear stiffness int G dA (N)
    GJ           torsional stiffness of the regions (N m^2), from Saint-Venant's J of the outline and of the core:
                 G_skin (J - J_core) + G_core J_core
    '''
    area: jnp.ndarray
    x_c: jnp.ndarray
    z_c: jnp.ndarray
    I_xx: jnp.ndarray
    I_zz: jnp.ndarray
    I_xz: jnp.ndarray
    J: jnp.ndarray
    t_max: jnp.ndarray
    outline: jnp.ndarray
    core_area: jnp.ndarray
    core_outline: jnp.ndarray
    core_exists: jnp.ndarray
    mass: SectionSum
    stiffness: SectionSum
    GA: jnp.ndarray
    GJ: jnp.ndarray

def strip_moments(x, w, y_upper, y_lower):
    '''
    Strip integrals of a unit-chord section between y_lower and y_upper, over the chordwise nodes x with weights w
    (arrays that broadcast to (..., n_x)), about the leading edge on the chord line.

    Returns (A, S_x, S_z, I_xx, I_zz, I_xz) = int (1, x, z, z^2, x^2, x z) dA.
    '''
    integral = lambda f: jnp.sum(f * w, axis=-1)
    thickness = y_upper - y_lower
    z_moment = 0.5 * (y_upper ** 2 - y_lower ** 2)
    return (integral(thickness), integral(thickness * x), integral(z_moment),
            integral((y_upper ** 3 - y_lower ** 3) / 3.0), integral(thickness * x ** 2), integral(x * z_moment))

def _about_centroid(moments):
    '''
    (A, x_centroid, z_centroid, I_xx, I_zz, I_xz) with the second moments about the centroid, from strip_moments.
    '''
    A, S_x, S_z, I_xx, I_zz, I_xz = moments
    x_c, z_c = S_x / A, S_z / A
    return A, x_c, z_c, I_xx - A * z_c ** 2, I_zz - A * x_c ** 2, I_xz - A * x_c * z_c

def strip_properties(x, w, y_upper, y_lower):
    '''
    Area, centroid and second moments of a section of unit chord from strip integrals over the chordwise nodes x
    with quadrature weights w, for the outline between y_lower and y_upper (each of shape (..., n_x)).

    Returns (A, x_centroid, z_centroid, I_xx, I_zz, I_xz), the moments about the centroid (all unit-chord values).
    '''
    return _about_centroid(strip_moments(x, w, y_upper, y_lower))

def torsion_constant(area, I_xx, I_zz):
    '''
    Saint-Venant's approximation of the torsion constant of a solid section, A^4 / (40 I_p) with I_p = I_xx + I_zz.
    It is 1.3% low for an ellipse and within 10% for a thin rectangle, and needs no integral over the section.
    '''
    return area ** 4 / (40.0 * (I_xx + I_zz))

def _offset_surfaces(cst, depth, x):
    '''
    The upper and lower surfaces of the CST sections (n_r, N_CST), offset inward along their normals by depth (n_r,)
    (fractions of the chord). The offset of a CST section has no closed form, so this gives polylines. The vertices come
    from each node x (n_x,) with OFFSET_SUBDIVISION segments to the next node, and from each end of the chord. x must not
    hold 0 or 1. An offset line folds where the curvature radius of the outline is below depth (a small leading edge):
    the X of its vertices then turn back.

    Returns (X_upper, Y_upper, X_lower, Y_lower), each (n_r, n_v). The last vertex carries the line past the trailing
    edge, so every query point up to the end of the line is inside a half-open segment.
    '''
    theta = 2.0 * jnp.arcsin(jnp.sqrt(x))   # x = (1 - cos theta) / 2
    fractions = jnp.arange(1, OFFSET_SUBDIVISION)[:, None] / OFFSET_SUBDIVISION
    extra = jnp.sin(0.5 * (theta[:-1] + fractions * (theta[1:] - theta[:-1]))).ravel() ** 2
    x = jnp.sort(jnp.concatenate([x, extra]))
    dx_dtheta = jnp.sqrt(x * (1.0 - x))     # dy/dtheta stays finite at the leading edge, and dy/dx does not
    (y_u, y_l), (dy_u, dy_l) = jax.jvp(lambda xi: geometry.kulfan_coordinates(cst, xi), (x,), (dx_dtheta,))
    length_u, length_l = jnp.hypot(dx_dtheta, dy_u), jnp.hypot(dx_dtheta, dy_l)
    ends_u, ends_l = geometry.kulfan_coordinates(cst, jnp.array([0.0, 1.0]))

    def line(y, y_ends, n_x, n_y):
        d = depth[:, None]
        X = jnp.concatenate([d, x + d * n_x, 1.0 + d * n_x[:, -1:], 2.0 + d * n_x[:, -1:]], axis=1)
        y_te = y_ends[:, 1:] + d * n_y[:, -1:]
        return X, jnp.concatenate([y_ends[:, :1], y + d * n_y, y_te, y_te], axis=1)

    return (*line(y_u, ends_u, dy_u / length_u, -dx_dtheta / length_u),
            *line(y_l, ends_l, -dy_l / length_l, dx_dtheta / length_l))

def _line_at(x_query, X, Y):
    '''
    The values at x_query (n_r, n_q) of polylines (n_r, n_v), one for each segment: arrays (n_r, n_q, n_v - 1), with a
    mask of the segments that span the point. A folded line has more than one such segment at a point.
    '''
    x_q, X0, X1, Y0, Y1 = x_query[..., None], X[:, None, :-1], X[:, None, 1:], Y[:, None, :-1], Y[:, None, 1:]
    run = X1 - X0
    spans = ((X0 <= x_q) & (x_q < X1)) | ((X1 <= x_q) & (x_q < X0))
    return Y0 + (x_q - X0) / jnp.where(run == 0.0, 1.0, run) * (Y1 - Y0), spans

def _core_surfaces(cst, depth, x_query, x):
    '''
    The surfaces of the core at x_query (n_r, n_q): the outline eroded by depth (n_r,). At each point the top of the
    core is the lowest offset line of the upper surface, and the bottom is the highest offset line of the lower surface.
    Where the top is not above the bottom the skin has closed.

    Returns (top, bottom, exists), each (n_r, n_q). Where no core exists, top = bottom = 0.
    '''
    X_u, Y_u, X_l, Y_l = _offset_surfaces(cst, depth, x)
    upper, spans_u = _line_at(x_query, X_u, Y_u)
    lower, spans_l = _line_at(x_query, X_l, Y_l)
    top = jnp.min(jnp.where(spans_u, upper, jnp.inf), axis=-1)
    bottom = jnp.max(jnp.where(spans_l, lower, -jnp.inf), axis=-1)
    inside = jnp.any(spans_u, axis=-1) & jnp.any(spans_l, axis=-1)
    exists = inside & (jnp.where(inside, top, 0.0) >= jnp.where(inside, bottom, 0.0))
    return jnp.where(exists, top, 0.0), jnp.where(exists, bottom, 0.0), exists

def _weighted_sum(weight_skin, weight_core, gross, core, c, pitch_axis) -> SectionSum:
    '''
    A SectionSum from the region weights and the unit-chord strip_moments of the gross outline and of the core. The
    skin is the outline minus the core, so a skin with no thickness adds exactly nothing.
    '''
    moments = tuple(weight_core * m_core + weight_skin * (m_gross - m_core) for m_gross, m_core in zip(gross, core))
    total, x1, z1, I_xx, I_zz, I_xz = _about_centroid(moments)
    return SectionSum(total * c ** 2, c * (x1 - pitch_axis), c * z1, I_xx * c ** 4, I_zz * c ** 4, I_xz * c ** 4)

def _core_torsion_constant(core_moments, c):
    '''
    Saint-Venant's J of the core, and 0 where the skin has closed.
    '''
    exists = core_moments[0] > 0.0
    A, _, _, I_xx, I_zz, _ = _about_centroid((jnp.where(exists, core_moments[0], 1.0), *core_moments[1:]))
    I_xx, I_zz = jnp.where(exists, I_xx, 1.0), jnp.where(exists, I_zz, 1.0)
    return jnp.where(exists, torsion_constant(A * c ** 2, I_xx * c ** 4, I_zz * c ** 4), 0.0)

def section_properties(params, mat, nodes=None) -> SectionProps:
    '''
    Section properties at every station of params (one flight condition, or any leading axes under vmap) for the Layup
    mat, from the CST sections the aerodynamics use, interpolated along the span. The hub-to-aero blend zone
    interpolates the stadium of the hub section toward the first aerodynamic section. nodes is (x, weights) of the
    chordwise rule (default GAUSS_X, GAUSS_W).

    The core has a strip rule of its own on x from depth to 1 - depth, where depth = skin_thickness / c, because the
    inward offset starts at x = depth with a square root, as the outline does at x = 0. With no skin it is the rule
    of the outline, and the core is the outline.
    '''
    x, w = (GAUSS_X, GAUSS_W) if nodes is None else nodes
    x, w = jnp.asarray(x), jnp.asarray(w)
    cst = geometry.cst_at_stations(params.r, params.r_cst, params.cst)                  # (n_r, N_CST)
    y_upper, y_lower = geometry.kulfan_coordinates(cst, x)
    gross = strip_moments(x, w, y_upper, y_lower)
    A1, x1, z1, I_xx1, I_zz1, I_xz1 = _about_centroid(gross)

    c = params.c
    pitch_axis = geometry.pitch_axis_fraction(params.r, params.cfg.R_hub, params.cfg.R_aero)
    area = A1 * c ** 2
    I_xx, I_zz, I_xz = I_xx1 * c ** 4, I_zz1 * c ** 4, I_xz1 * c ** 4
    J = torsion_constant(area, I_xx, I_zz)

    # Outline points, with the end points of the chord added so the extreme fibres are exact
    x_out = jnp.concatenate([jnp.zeros(1), x, jnp.ones(1)])
    y_up, y_lo = geometry.kulfan_coordinates(cst, x_out)
    x_pts = jnp.concatenate([x_out, x_out[::-1]])
    z_pts = jnp.concatenate([y_up, y_lo[..., ::-1]], axis=-1)
    outline = c[:, None, None] * jnp.stack([jnp.broadcast_to(x_pts, z_pts.shape) - x1[:, None], z_pts - z1[:, None]], axis=-1)

    # The core, on its own chordwise range. Its outline points hold its quadrature nodes
    depth = jnp.minimum(mat.skin_thickness / c, 0.5)
    span = (1.0 - 2.0 * depth)[:, None]
    x_core = depth[:, None] + span * x_out
    top, bottom, exists = _core_surfaces(cst, depth, x_core, x)
    core = strip_moments(x_core[:, 1:-1], span * w, top[:, 1:-1], bottom[:, 1:-1])
    core_x = jnp.concatenate([x_core, x_core[:, ::-1]], axis=-1)
    core_z = jnp.concatenate([top, bottom[:, ::-1]], axis=-1)
    core_outline = c[:, None, None] * jnp.stack([core_x - x1[:, None], core_z - z1[:, None]], axis=-1)

    skin, inner = mat.skin, mat.core
    weighted = lambda v_skin, v_core: _weighted_sum(v_skin * skin.fill, v_core * inner.fill, gross, core, c, pitch_axis)
    core_area = core[0] * c ** 2
    G_skin, G_core = skin.G * skin.fill, inner.G * inner.fill
    J_core = _core_torsion_constant(core, c)

    return SectionProps(
        area=area, x_c=c * (x1 - pitch_axis), z_c=c * z1, I_xx=I_xx, I_zz=I_zz, I_xz=I_xz,
        J=J, t_max=c * jnp.max(y_upper - y_lower, axis=-1), outline=outline,
        core_area=core_area, core_outline=core_outline, core_exists=jnp.concatenate([exists, exists[:, ::-1]], axis=-1),
        mass=weighted(skin.density, inner.density), stiffness=weighted(skin.E, inner.E),
        GA=G_core * core_area + G_skin * (area - core_area), GJ=G_core * J_core + G_skin * (J - J_core))


## Kinematics and blade axes

class Kinematics(eqx.Module):
    '''
    Motion of the blade beyond steady rotation at the shaft speed params.omega. All zero by default (SI units).

    omega_dot     shaft spin acceleration (rad/s^2)
    pitch_rate    rate of change of the blade pitch, a rotation about the pitch-change axis (the straight line e_r
                  through the hub centre) that increases the twist beta (rad/s): a constant-speed unit at work
    pitch_accel   its acceleration (rad/s^2)
    body_rates    angular velocity of the airframe (p, q, r) on the hub frame (x at azimuth 0, y at 90 deg, z = e_z)
                  (rad/s), held constant: gyroscopic loads
    '''
    omega_dot: jnp.ndarray = eqx.field(default_factory=lambda: jnp.zeros(()))
    pitch_rate: jnp.ndarray = eqx.field(default_factory=lambda: jnp.zeros(()))
    pitch_accel: jnp.ndarray = eqx.field(default_factory=lambda: jnp.zeros(()))
    body_rates: jnp.ndarray = eqx.field(default_factory=lambda: jnp.zeros(3))

class BladeAxes(NamedTuple):
    '''
    Position of the pitch-axis point and the axes of every station on the blade frame (e_r, e_t, e_z): p, s (span
    direction), x_s (chord direction at the twist) and z_s (toward the suction side), each (n_r, 3); the arc length
    per unit radius and the angle g of the pitch-axis point about the rotation axis, each (n_r,).
    '''
    p: jnp.ndarray
    s: jnp.ndarray
    x_s: jnp.ndarray
    z_s: jnp.ndarray
    arc: jnp.ndarray
    gamma: jnp.ndarray

def blade_axes(params) -> BladeAxes:
    '''
    Blade-frame geometry of the pitch axis and the sections from geometry.section_frame and the twist. Unswept at
    zero twist: p = (r, 0, 0), s = e_r, x_s = -e_t, z_s = e_z.
    '''
    frame = geometry.section_frame(params.r, params.offsets)
    sin_g, cos_g = frame.sin_g, frame.cos_g
    # air basis (rho, t, a) -> blade frame: rho = cos g e_r + sin g e_t, t = sin g e_r - cos g e_t, a = -e_z
    to_blade = lambda rho, t, a: jnp.stack([cos_g * rho + sin_g * t, sin_g * rho - cos_g * t, -a], axis=-1)
    s = jnp.stack([cos_g * frame.s_rho - sin_g * frame.s_tau, sin_g * frame.s_rho + cos_g * frame.s_tau, frame.s_z], axis=-1)
    chord, normal = to_blade(frame.c_rho, frame.c_t, frame.c_a), to_blade(frame.n_rho, jnp.zeros_like(sin_g), frame.n_a)
    cos_b, sin_b = jnp.cos(params.beta)[:, None], jnp.sin(params.beta)[:, None]
    p = jnp.stack([params.r * cos_g, params.offsets[:, 0], params.offsets[:, 1]], axis=-1)
    return BladeAxes(p, s, cos_b * chord + sin_b * normal, sin_b * chord - cos_b * normal, frame.arc, jnp.arcsin(sin_g))

def _hub_to_blade(v, psi):
    '''
    Components, on the blade frame at each root azimuth psi (n_psi,), of a vector v (3,) given on the hub frame.
    Returns (n_psi, 3).
    '''
    cos_p, sin_p = jnp.cos(psi), jnp.sin(psi)
    return jnp.stack([v[0] * cos_p + v[1] * sin_p, -v[0] * sin_p + v[1] * cos_p, v[2] * jnp.ones_like(psi)], axis=-1)

def _hub_frame(v, psi):
    '''
    Components on the hub frame of vectors v (n_psi, 3) given on the blade frame at root azimuth psi (n_psi,).
    '''
    cos_p, sin_p = jnp.cos(psi), jnp.sin(psi)
    return jnp.stack([v[:, 0] * cos_p - v[:, 1] * sin_p, v[:, 0] * sin_p + v[:, 1] * cos_p, v[:, 2]], axis=-1)


## Mass properties

class BladeSlices(NamedTuple):
    '''
    The blade as a chain of rigid slices normal to the span, per unit radius at every station, on the blade frame:
    mass (n_r,) in kg/m, p_cg (n_r, 3) the centre of mass of the slice in m, and inertia (n_r, 3, 3) the tensor of the
    slice about its centre of mass in kg m^2/m. The mass properties and the inertial loads share these values.
    '''
    mass: jnp.ndarray
    p_cg: jnp.ndarray
    inertia: jnp.ndarray

def _dyad(a, b):
    return a[..., :, None] * b[..., None, :]

def blade_slices(props, axes) -> BladeSlices:
    '''
    Slices from the mass-weighted section props.mass and the BladeAxes: with the weighted second moments of the section
    the tensor per unit span on the section axes (x_s, s, z_s) is [[I_xx, 0, -I_xz], [0, I_xx + I_zz, 0], [-I_xz, 0, I_zz]].
    The thickness of the slice along the span adds no term per unit radius: the radial integral holds it.
    '''
    m = props.mass
    p_cg = axes.p + m.x_c[:, None] * axes.x_s + m.z_c[:, None] * axes.z_s
    scalar = lambda v: (v * axes.arc)[:, None, None]
    inertia = (scalar(m.I_xx) * _dyad(axes.x_s, axes.x_s) + scalar(m.I_xx + m.I_zz) * _dyad(axes.s, axes.s)
               + scalar(m.I_zz) * _dyad(axes.z_s, axes.z_s)
               - scalar(m.I_xz) * (_dyad(axes.x_s, axes.z_s) + _dyad(axes.z_s, axes.x_s)))
    return BladeSlices(m.total * axes.arc, p_cg, inertia)

def _point_tensor(mass, p):
    '''
    The inertia tensor of a mass at p about the origin: m (|p|^2 I - p p^T). mass (...,), p (..., 3).
    '''
    square = jnp.sum(p * p, axis=-1)
    return mass[..., None, None] * (square[..., None, None] * jnp.eye(3) - _dyad(p, p))

class MassProps(NamedTuple):
    '''
    Mass properties of a rigid body on the blade frame (e_r, e_t, e_z) of the first blade at azimuth 0 (the hub frame
    there): mass (kg), the centre of gravity cg (3,) from the rotation axis origin (m), and the inertia tensor about the
    origin (3, 3) (kg m^2). I_ij = int (|p|^2 delta_ij - p_i p_j) dm, so a product of inertia carries a minus sign, and
    the spin inertia is inertia[2, 2].
    '''
    mass: jnp.ndarray
    cg: jnp.ndarray
    inertia: jnp.ndarray

    @property
    def inertia_cg(self):
        '''
        The inertia tensor about the centre of gravity (the parallel axis theorem).
        '''
        return self.inertia - _point_tensor(self.mass, self.cg)

    def plus(self, other: "MassProps") -> "MassProps":
        '''
        The mass properties of the two bodies together.
        '''
        mass = self.mass + other.mass
        return MassProps(mass, (self.mass * self.cg + other.mass * other.cg) / mass, self.inertia + other.inertia)

def hub_hardware(mass, radius=0.0, height=0.0, z=0.0) -> MassProps:
    '''
    Hub or hardware mass (kg), given explicitly, as a solid cylinder on the rotation axis of the given radius and
    height (m) with its centre at z (m). The default is a point mass at the origin.
    '''
    mass, radius, height, z = (jnp.asarray(v, dtype=float) for v in (mass, radius, height, z))
    in_plane = mass * (3.0 * radius ** 2 + height ** 2) / 12.0 + mass * z ** 2
    return MassProps(mass, jnp.stack([jnp.zeros_like(z), jnp.zeros_like(z), z]),
                     jnp.diag(jnp.stack([in_plane, in_plane, 0.5 * mass * radius ** 2])))

def _radial_nodes(params):
    '''
    Gauss nodes and weights of the integral over the blade, from the start of the hub zone (r_foils[0]) to the tip. The
    intervals end at every station and at every CST radius, so the rule does not cross a kink of the chord or of the
    section shape. Returns (r, weights), each of shape (n_intervals * N_R_GAUSS,).
    '''
    edges = jnp.sort(jnp.concatenate([jnp.array([params.cfg.R_start]), params.r, params.r_cst]))
    xi, w = np.polynomial.legendre.leggauss(N_R_GAUSS)
    half = 0.5 * (edges[1:] - edges[:-1])[:, None]
    return (0.5 * (edges[1:] + edges[:-1])[:, None] + half * xi).ravel(), (half * w).ravel()

def _offsets_at(r, stations, offsets):
    '''
    The pitch-axis offsets (see geometry.station_offsets) at radii r: a cubic Hermite curve between the stations through
    their values and slopes. Inboard of the first station the axis is a straight line to the hub centre.
    '''
    i = jnp.clip(jnp.searchsorted(stations, r) - 1, 0, stations.shape[0] - 2)
    h = (stations[i + 1] - stations[i])[:, None]
    t = (r - stations[i])[:, None] / h
    y0, y1, m0, m1 = offsets[i, :2], offsets[i + 1, :2], offsets[i, 2:], offsets[i + 1, 2:]
    value = ((1.0 + 2.0 * t) * (1.0 - t) ** 2 * y0 + t * (1.0 - t) ** 2 * h * m0
             + t ** 2 * (3.0 - 2.0 * t) * y1 + t ** 2 * (t - 1.0) * h * m1)
    slope = m0 + t * (m1 - m0)
    line = offsets[0, :2] / stations[0]
    inboard = (r < stations[0])[:, None]
    return jnp.concatenate([jnp.where(inboard, r[:, None] * line, value), jnp.where(inboard, line, slope)], axis=1)

def _blade_at(params, r):
    '''
    params with its stations moved to the radii r: chord and twist linear between the design stations and held at the
    first station inboard of it. The hub zone is thus the prism of the hub section at the hub chord and twist.
    '''
    values = (r, jnp.interp(r, params.r, params.c), jnp.interp(r, params.r, params.beta), _offsets_at(r, params.r, params.offsets))
    return eqx.tree_at(lambda p: (p.r, p.c, p.beta, p.offsets), params, values)

@jax.jit
def blade_mass_properties(params, mat) -> MassProps:
    '''
    Mass, centre of gravity and inertia tensor (MassProps) of one blade with the Layup mat: the integral along the radius
    of the slices (blade_slices), with the twist, the sweep and rake, the offset of the section from the pitch axis
    and the own inertia of each slice. It includes the hub zone from r_foils[0] to r_hub, which the structural loads
    leave out, and ends at the last station. It does not hold the hub hardware (propeller_mass_properties).

    Where the pitch axis bends, the slices are wedges: a point at (x, z) on the section sweeps a length per unit radius
    arc - x (x_s . ds/dr) - z (z_s . ds/dr), so a section centroid on the inside of the bend has less volume (Pappus).
    The radial weights hold this factor at the centroid of each slice. The structural loads use the plain arc.
    '''
    r, weights = _radial_nodes(params)
    blade = _blade_at(params, r)
    axes, props = blade_axes(blade), section_properties(blade, mat)
    _, ds_dr = jax.jvp(lambda radii: blade_axes(_blade_at(params, radii)).s, (r,), (jnp.ones_like(r),))
    bend = (props.mass.x_c * jnp.sum(axes.x_s * ds_dr, axis=-1) + props.mass.z_c * jnp.sum(axes.z_s * ds_dr, axis=-1)) / axes.arc
    weights = weights * (1.0 - bend)
    slices = blade_slices(props, axes)
    tensor = slices.inertia + _point_tensor(slices.mass, slices.p_cg)
    mass = jnp.sum(weights * slices.mass)
    cg = jnp.sum((weights * slices.mass)[:, None] * slices.p_cg, axis=0) / mass
    return MassProps(mass, cg, jnp.sum(weights[:, None, None] * tensor, axis=0))

def rotor_mass_properties(blade: MassProps, N) -> MassProps:
    '''
    Mass properties of N equal blades, spaced evenly about e_z, the first blade at azimuth 0. The sum over the blades
    of a term with n cycles per turn is N times the term when N divides n, and zero otherwise. The first moment and the
    products with z have n = 1. The in-plane difference and the in-plane product have n = 2. So three or more blades have
    I_xx = I_yy and no in-plane product, and two blades keep the difference: I_xx and I_yy of the first blade, times 2.
    '''
    N = jnp.asarray(N)
    one_cycle, two_cycle = jnp.where(N == 1, N, 0), jnp.where(2 % N == 0, N, 0)
    T = blade.inertia
    mean, half_difference = 0.5 * (T[0, 0] + T[1, 1]), 0.5 * (T[0, 0] - T[1, 1])
    I_xy, I_xz, I_yz = two_cycle * T[0, 1], one_cycle * T[0, 2], one_cycle * T[1, 2]
    tensor = jnp.array([[N * mean + two_cycle * half_difference, I_xy, I_xz],
                        [I_xy, N * mean - two_cycle * half_difference, I_yz],
                        [I_xz, I_yz, N * T[2, 2]]])
    return MassProps(N * blade.mass, jnp.array([one_cycle / N * blade.cg[0], one_cycle / N * blade.cg[1], blade.cg[2]]), tensor)

def propeller_mass_properties(params, mat, hardware=None) -> MassProps:
    '''
    Mass properties of the N-blade propeller, with optional hub or hardware mass (a MassProps, see hub_hardware) given
    explicitly. The model has no hub part: without hardware this is the blades alone.
    '''
    rotor = rotor_mass_properties(blade_mass_properties(params, mat), params.N[0])
    return rotor if hardware is None else rotor.plus(hardware)


## Loads on the blade

class InertialLoads(NamedTuple):
    '''
    Inertial (d'Alembert) loads of one blade per unit radius at every root azimuth, each (n_psi, n_r, 3) on the blade
    frame. f (N/m) and m (N m/m, about the pitch-axis point) are the totals. m_cg is the moment about the centre of
    mass of the slice: its gyroscopic moment, which holds the centrifugal twisting moment. The four force terms add up
    to f: Euler (spin and airframe acceleration), centrifugal, Coriolis (2 w x v_rel, from the pitch rate) and the
    acceleration relative to the hub (pitch acceleration and pitch rate squared).
    '''
    f: jnp.ndarray
    m: jnp.ndarray
    m_cg: jnp.ndarray
    f_euler: jnp.ndarray
    f_centrifugal: jnp.ndarray
    f_coriolis: jnp.ndarray
    f_relative: jnp.ndarray

def inertial_loads(params, props, kin=None, axes=None) -> InertialLoads:
    '''
    The blade as a chain of rigid slices (blade_slices, from the mass-weighted section props.mass). The slices follow
    the hub, which spins at params.omega (accelerating at kin.omega_dot) about e_z and turns with the airframe
    (kin.body_rates), and pitch about e_r through the hub centre at kin.pitch_rate. With the total angular velocity
    w_h of the hub and the pitch rotation nu, a slice with its centre of mass at p_cg accelerates by
        a = w_h_dot x p_cg + w_h x (w_h x p_cg) + 2 w_h x (nu x p_cg) + nu_dot x p_cg + nu x (nu x p_cg),
    and turns at w_s = w_h + nu, so its loads are F = -m' a and M_cg = -(J' w_s_dot + w_s x J' w_s).
    The airframe acceleration of the hub centre is neglected. A rigid blade at steady rotation has no Coriolis term:
    it appears with the pitch rate (the CSU at work) and carries into the gyroscopic loads with the airframe rates.
    '''
    kin = Kinematics() if kin is None else kin
    axes = blade_axes(params) if axes is None else axes
    e_r, e_z = jnp.array([1.0, 0.0, 0.0]), jnp.array([0.0, 0.0, 1.0])

    body = _hub_to_blade(kin.body_rates, params.psi)                        # (n_psi, 3)
    w_h = params.omega * e_z + body
    w_h_dot = kin.omega_dot * e_z + params.omega * jnp.cross(body, e_z)
    nu, nu_dot = kin.pitch_rate * e_r, kin.pitch_accel * e_r

    slices = blade_slices(props, axes)
    offset = slices.p_cg - axes.p                                           # centre of mass from the pitch-axis point
    p_cg = slices.p_cg[None]                                                # (1, n_r, 3)
    w, w_dot = w_h[:, None, :], w_h_dot[:, None, :]
    v_rel = jnp.cross(nu, p_cg)
    mass = slices.mass[None, :, None]
    f_euler = -mass * jnp.cross(w_dot, p_cg)
    f_centrifugal = -mass * jnp.cross(w, jnp.cross(w, p_cg))
    f_coriolis = -mass * 2.0 * jnp.cross(w, v_rel)
    f_relative = -mass * (jnp.cross(nu_dot, p_cg) + jnp.cross(nu, v_rel))
    f = f_euler + f_centrifugal + f_coriolis + f_relative

    # Gyroscopic moment of every slice about its centre of mass
    w_s, w_s_dot = w_h + nu, w_h_dot + nu_dot + jnp.cross(w_h, nu)
    apply_inertia = lambda v: jnp.einsum("rij,pj->pri", slices.inertia, v)
    m_cg = -(apply_inertia(w_s_dot) + jnp.cross(w_s[:, None, :], apply_inertia(w_s)))
    m = jnp.cross(offset[None], f) + m_cg
    return InertialLoads(f, m, m_cg, f_euler, f_centrifugal, f_coriolis, f_relative)

def _to_root_azimuth(x, gamma):
    '''
    Resamples a field x (n_psi, n_r, ...) given on the azimuth of each station's pitch-axis point, to the azimuth
    of the blade root: the point of station i is at psi_root + gamma_i. A shift by gamma_i of the sampled periodic
    field, exact for the harmonics the grid resolves.
    '''
    n = x.shape[0]
    harmonics = jnp.arange(n // 2 + 1)
    phase = jnp.exp(1j * harmonics[:, None] * gamma[None, :])
    phase = phase.reshape(phase.shape + (1,) * (x.ndim - 2))
    return jnp.fft.irfft(jnp.fft.rfft(x, axis=0) * phase, n=n, axis=0)

def aero_loads(params, V_i):
    '''
    Aerodynamic force f and pitching moment m of one blade per unit radius at every root azimuth, each
    (n_psi, n_r, 3) on the blade frame (propmodel.sectional_forces, taken to the azimuth of the blade root).
    '''
    loads = propmodel.sectional_forces(V_i, params)
    f, m = jnp.swapaxes(loads.f, 0, 1), jnp.swapaxes(loads.m, 0, 1)
    if params.cfg.swept:
        gamma = blade_axes(params).gamma
        f, m = _to_root_azimuth(f, gamma), _to_root_azimuth(m, gamma)
    return f, m

def integrate_outboard(r, values, include=None):
    '''
    Integral from each station to the tip of values (n_psi, n_r, k) along the stations, by the trapezoid rule
    (a reversed cumulative sum). include (n_r - 1,) weights the intervals: radial_weights' rule keeps the intervals
    that start in the aerodynamic zone.
    '''
    dr = jnp.diff(r) if include is None else jnp.diff(r) * include
    segments = 0.5 * (values[:, :-1] + values[:, 1:]) * dr[None, :, None]
    outboard = jnp.cumsum(segments[:, ::-1], axis=1)[:, ::-1]
    return jnp.concatenate([outboard, jnp.zeros_like(values[:, :1])], axis=1)

def section_resultants(r, p, g, m, include=None):
    '''
    Resultants of the loads outboard of every station, for loads g (force) and m (moment about the pitch-axis point p)
    per unit radius, each (n_psi, n_r, 3): the force F, the moment about the hub centre M_O and the moment about the
    station's own pitch-axis point M_ref = M_O - p x F. F is what the blade outboard of the station applies to the
    blade inboard of it. At the root it is the load on the hub.
    '''
    F = integrate_outboard(r, g, include)
    M_O = integrate_outboard(r, jnp.cross(p[None], g) + m, include)
    return F, M_O - jnp.cross(p[None], F), M_O

class Resultants(NamedTuple):
    '''
    Internal loads of a section on its axes, each (n_psi, n_r): the axial force N (tension positive), the shear
    forces V_x and V_z, the bending moments M_x (flap, about x_s) and M_z (edgewise, about z_s), and the torque T
    about the span, with the moment about the stiffness-weighted centroid.
    '''
    N: jnp.ndarray
    V_x: jnp.ndarray
    V_z: jnp.ndarray
    M_x: jnp.ndarray
    T: jnp.ndarray
    M_z: jnp.ndarray

def _resultants(F, M_c, axes) -> Resultants:
    on = lambda v, axis: jnp.einsum("prk,rk->pr", v, axis)
    return Resultants(on(F, axes.s), on(F, axes.x_s), on(F, axes.z_s), on(M_c, axes.x_s), on(M_c, axes.s), on(M_c, axes.z_s))


## Loads at the hub

class HubLoads(NamedTuple):
    '''
    Loads of the N blades on the hub, on the hub frame (x at azimuth 0, y at 90 deg, z = e_z), as functions of the
    azimuth of the first blade (n_psi, 3), with the moments about the hub centre. Positive force z is thrust.
    The torque the shaft supplies is -moment z. force_amplitude and moment_amplitude (n_psi // 2 + 1, 3) are the
    peak amplitudes of the harmonics per revolution (row 0 is the mean): only multiples of N appear.
    '''
    force: jnp.ndarray
    moment: jnp.ndarray
    force_amplitude: jnp.ndarray
    moment_amplitude: jnp.ndarray

def hub_loads(F_root, M_root, N, psi) -> HubLoads:
    '''
    Sums the root loads of the N blades on the hub frame. F_root and M_root (n_psi, 3) are those of one blade on its
    blade frame, as functions of its root azimuth psi. The blades sit at psi + 2 pi k / N, so only the harmonics of
    the (periodic) hub-frame load that are multiples of N survive the sum, each multiplied by N. This holds for any
    n_psi, resolving the harmonics the grid resolves.
    '''
    n = psi.shape[0]
    harmonic = jnp.round(jnp.fft.fftfreq(n, d=1.0 / n)).astype(int)
    keep = (harmonic % jnp.asarray(N).astype(int)) == 0

    def sum_blades(v):
        spectrum = jnp.where(keep[:, None], jnp.fft.fft(_hub_frame(v, psi), axis=0), 0.0)
        return jnp.fft.ifft(spectrum, axis=0).real * N

    def amplitude(v):
        spectrum = jnp.fft.rfft(v, axis=0)
        a = _norm2(spectrum.real, spectrum.imag) / n
        return a.at[1:].multiply(2.0)

    force, moment = sum_blades(F_root), sum_blades(M_root)
    return HubLoads(force, moment, amplitude(force), amplitude(moment))


## Structural result

class StructuralResult(eqx.Module):
    '''
    Loads of one blade at every root azimuth psi and station r, and the hub loads, for one flight condition.
    Array fields have shape (n_psi, n_r) or (n_psi, n_r, 3) (components on the blade frame), SI units.

    F, M_O, M_c          resultant force on the inboard side, its moment about the hub centre and about the
                         stiffness-weighted centroid
    F_aero, F_inertial   the aerodynamic and the inertial part of F (they add up), and M_O_* the same for M_O
    resultants           internal loads of the section on its axes, for the totals (Resultants)
    T_aero, T_inertial   torque about the span, by part (aerodynamic twisting moment, centrifugal and other inertial)
    pitch_torque_*       torque about the pitch-change axis e_r at the root, (n_psi,): aerodynamic (ATM), inertial
                         (CTM and the pitch-acceleration terms) and the CSU hold torque -(ATM + CTM)
    hub                  HubLoads
    '''
    props: SectionProps
    axes: BladeAxes
    F: jnp.ndarray
    M_O: jnp.ndarray
    M_c: jnp.ndarray
    F_aero: jnp.ndarray
    F_inertial: jnp.ndarray
    M_O_aero: jnp.ndarray
    M_O_inertial: jnp.ndarray
    resultants: Resultants
    T_aero: jnp.ndarray
    T_inertial: jnp.ndarray
    pitch_torque_aero: jnp.ndarray
    pitch_torque_inertial: jnp.ndarray
    hub: HubLoads

    @property
    def pitch_torque_hold(self):
        return -(self.pitch_torque_aero + self.pitch_torque_inertial)

@partial(jax.jit, static_argnames=("include_aero",))
def structural_loads(params, mat, kin=None, V_i=None, include_aero=True) -> StructuralResult:
    '''
    Loads along the blade and at the hub for one flight condition (use jax.vmap for several, with mat and kin shared).
    It solves nothing: the aerodynamic loads use V_i, by default the expanded stored induced velocity
    (inflow_model(params.V_i_0, params)), so run propmodel.induced_velocity first, or pass its V_i to differentiate
    through the solve. include_aero=False leaves the aerodynamic loads out (inertial loads alone). The loads start at
    r_hub: the hub zone inboard of it is outside the BEMT grid, and no moment or force acts across it here.
    '''
    kin = Kinematics() if kin is None else kin
    props = section_properties(params, mat)
    axes = blade_axes(params)
    offset = props.stiffness.x_c[:, None] * axes.x_s + props.stiffness.z_c[:, None] * axes.z_s
    centroid_moment = lambda F, M_ref: M_ref - jnp.cross(offset[None], F)

    inertial = inertial_loads(params, props, kin, axes)
    F_i, M_ref_i, M_O_i = section_resultants(params.r, axes.p, inertial.f, inertial.m)
    if include_aero:
        if V_i is None:
            V_i = inflow_model(params.V_i_0, params)
        f_a, m_a = aero_loads(params, V_i)
        include = propmodel.aero_zone(params)[:-1].astype(params.r.dtype)
        F_a, M_ref_a, M_O_a = section_resultants(params.r, axes.p, f_a, m_a, include)
    else:
        F_a, M_ref_a, M_O_a = jnp.zeros_like(F_i), jnp.zeros_like(F_i), jnp.zeros_like(F_i)

    F, M_O = F_a + F_i, M_O_a + M_O_i
    M_c_a, M_c_i = centroid_moment(F_a, M_ref_a), centroid_moment(F_i, M_ref_i)
    M_c = M_c_a + M_c_i
    along_span = lambda M: jnp.einsum("prk,rk->pr", M, axes.s)
    return StructuralResult(
        props=props, axes=axes, F=F, M_O=M_O, M_c=M_c, F_aero=F_a, F_inertial=F_i, M_O_aero=M_O_a, M_O_inertial=M_O_i,
        resultants=_resultants(F, M_c, axes), T_aero=along_span(M_c_a), T_inertial=along_span(M_c_i),
        pitch_torque_aero=M_O_a[:, 0, 0], pitch_torque_inertial=M_O_i[:, 0, 0],
        hub=hub_loads(F[:, 0], M_O[:, 0], params.N[0], params.psi))


## Stress

def _norm2(a, b):
    '''
    sqrt(a^2 + b^2) with a zero (not NaN) gradient where both vanish: the tip station carries no load, and the
    derivative of a norm at the origin is undefined.
    '''
    square = a * a + b * b
    positive = square > 0.0
    return jnp.where(positive, jnp.sqrt(jnp.where(positive, square, 1.0)), 0.0)

class Stress(NamedTuple):
    '''
    Stress in the regions of the blade at every root azimuth and station (Pa), tension positive. The first group is the
    region at the outline: the skin, or the core when the skin has no thickness.

    sigma         axial stress at the outline points of the section (n_psi, n_r, 2 (N_X + 2))
    sigma_max     largest |sigma| of the outline
    tau           shear stress: torsion at the thickest point plus the mean shear of the section forces, a
                  conservative sum
    sigma_vm      von Mises stress sqrt(sigma_max^2 + 3 tau^2)

    core_sigma_max, core_tau, core_sigma_vm   the same for the core, at its own outline. Zero where the skin has closed.
    safety_factor smaller of sigma_allow / sigma_vm and tau_allow / tau over the regions
    '''
    sigma: jnp.ndarray
    sigma_max: jnp.ndarray
    tau: jnp.ndarray
    sigma_vm: jnp.ndarray
    core_sigma_max: jnp.ndarray
    core_tau: jnp.ndarray
    core_sigma_vm: jnp.ndarray
    safety_factor: jnp.ndarray

def stress_fields(result: StructuralResult, mat: Layup) -> Stress:
    '''
    Stress of a structural_loads result on the transformed section. The axial strain is N / EA + a_x x + a_z z for a
    point (x, z) relative to the stiffness-weighted centroid, with the bending gradients a_x and a_z that carry the
    moments M_x = -int z sigma dA and M_z = int x sigma dA of the section (an unsymmetric section bends out of the plane
    of the moment, through the product of inertia). The stress in a region is its E times the strain: a region with a
    fill factor carries the stress of the gross strain in its material. The shear strain of a thin section is
    approximated by T t_max / GJ for the torque and 4/3 of the mean shear strain of the shear forces, and the stress in
    a region is its G times this strain.

    For a laminate skin these are stresses of a homogenised region, and a laminate has no von Mises stress. Set the
    skin allowables to effective values for this rule, and check the plies apart. Peak values are not fatigue limits:
    no S-N model.
    '''
    props, res, e = result.props, result.resultants, result.props.stiffness
    D = e.I_xx * e.I_zz - e.I_xz ** 2
    a_x = (e.I_xz * res.M_x + e.I_xx * res.M_z) / D
    a_z = -(e.I_zz * res.M_x + e.I_xz * res.M_z) / D
    shift_x, shift_z = (props.x_c - e.x_c)[None, :, None], (props.z_c - e.z_c)[None, :, None]

    def strain(points):
        x, z = points[..., 0][None], points[..., 1][None]                               # (1, n_r, K)
        return (res.N / e.total)[..., None] + a_x[..., None] * (x + shift_x) + a_z[..., None] * (z + shift_z)

    shear_strain = jnp.abs(res.T) * props.t_max / props.GJ + 4.0 / 3.0 * _norm2(res.V_x, res.V_z) / props.GA

    outer, core = mat.outer, mat.core
    sigma = outer.E * strain(props.outline)
    sigma_max = jnp.max(jnp.abs(sigma), axis=-1)
    tau = outer.G * shear_strain
    sigma_vm = _norm2(sigma_max, jnp.sqrt(3.0) * tau)

    core_sigma_max = jnp.max(jnp.abs(jnp.where(props.core_exists[None], core.E * strain(props.core_outline), 0.0)), axis=-1)
    core_tau = jnp.where(jnp.any(props.core_exists, axis=-1)[None], core.G * shear_strain, 0.0)
    core_sigma_vm = _norm2(core_sigma_max, jnp.sqrt(3.0) * core_tau)

    tiny = 1e-9
    safety = lambda m, vm, shear: jnp.minimum(m.sigma_allow / jnp.maximum(vm, tiny), m.tau_allow / jnp.maximum(shear, tiny))
    return Stress(sigma, sigma_max, tau, sigma_vm, core_sigma_max, core_tau, core_sigma_vm,
                  jnp.minimum(safety(outer, sigma_vm, tau), safety(core, core_sigma_vm, core_tau)))

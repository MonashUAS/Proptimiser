## Structural loads of a rigid propeller blade: section properties, loads along the blade and at the hub, stress
#
# The blade is one isotropic material over the CST outline of every station (BladeSpec.mat_*, scaled by the fill
# factor), a rigid body as far as its loads are concerned: aerodynamic loads from the BEMT, and inertial loads from
# the rotation of the hub, of the aircraft and of the pitch change. Frames, signs and equations are set out in the
# README section "Structural loads".
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
import init                 # first: it sets the JAX configuration (float64)
import geometry
import propmodel
from airflow import inflow_model

N_X = 48                    # chordwise Gauss nodes of the section integrals

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
    One isotropic material, with the fraction of the section outline that is material. Mass, E and G and every
    stiffness scale with the fill factor (a homogenised infill), and the stress in the material is the gross stress
    divided by it. All fields are float scalars (SI units), so a change of material never recompiles anything.
    '''
    density: jnp.ndarray        # kg/m^3
    E: jnp.ndarray              # Pa
    G: jnp.ndarray              # Pa
    sigma_allow: jnp.ndarray    # Pa
    tau_allow: jnp.ndarray      # Pa
    fill: jnp.ndarray           # (0, 1]

    @classmethod
    def from_blade(cls, blade) -> "Material":
        '''
        The material of a BladeSpec. A zero allowable shear stress means sigma_allow / sqrt(3), the von Mises value.
        '''
        tau = blade.mat_tau_allow if blade.mat_tau_allow > 0 else blade.mat_sigma_allow / np.sqrt(3.0)
        return cls(*(jnp.asarray(v, dtype=float) for v in (
            blade.mat_density, blade.mat_E, blade.mat_G, blade.mat_sigma_allow, tau, blade.fill_factor)))


## Section properties

class SectionProps(eqx.Module):
    '''
    Geometric properties of the CST section at every station, for the gross outline (no fill factor), each of shape
    (n_r,) unless stated. Lengths in m, areas in m^2, second moments in m^4.

    x_c, z_c     centroid relative to the pitch axis, on the section axes: x_c along the chord from the pitch axis
                 (positive toward the trailing edge), z_c toward the suction side
    I_xx, I_zz   second moments about the centroid: I_xx = int z^2 dA (flap), I_zz = int x^2 dA (edgewise)
    I_xz         product of inertia int x z dA about the centroid
    J            torsion constant, Saint-Venant's approximation A^4 / (40 (I_xx + I_zz)) (within 10% for thin
                 sections); the shear centre is taken at the centroid
    t_max        largest thickness of the section
    outline      (n_r, 2 (N_X + 2), 2): points (x, z) of the outline relative to the centroid, upper surface from the
                 leading to the trailing edge, then the lower surface back, used for the stress at the extreme fibres
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

def strip_properties(x, w, y_upper, y_lower):
    '''
    Area, centroid and second moments of a section of unit chord from strip integrals over the chordwise nodes x
    with quadrature weights w, for the outline between y_lower and y_upper (each of shape (..., n_x)).

    Returns (A, x_centroid, z_centroid, I_xx, I_zz, I_xz), the moments about the centroid (all unit-chord values).
    '''
    thickness = y_upper - y_lower
    z_moment = 0.5 * (y_upper ** 2 - y_lower ** 2)
    A = thickness @ w
    x_c, z_c = (thickness * x) @ w / A, z_moment @ w / A
    I_xx = ((y_upper ** 3 - y_lower ** 3) / 3.0) @ w - A * z_c ** 2
    I_zz = (thickness * x ** 2) @ w - A * x_c ** 2
    I_xz = (x * z_moment) @ w - A * x_c * z_c
    return A, x_c, z_c, I_xx, I_zz, I_xz

def torsion_constant(area, I_xx, I_zz):
    '''
    Saint-Venant's approximation of the torsion constant of a solid section, A^4 / (40 I_p) with I_p = I_xx + I_zz.
    It is 1.3% low for an ellipse and within 10% for a thin rectangle, and needs no integral over the section.
    '''
    return area ** 4 / (40.0 * (I_xx + I_zz))

def section_properties(params) -> SectionProps:
    '''
    Section properties at every station of params (one flight condition, or any leading axes under vmap), from the
    CST sections the aerodynamics use, interpolated along the span. The hub-to-aero blend zone interpolates the
    stadium of the hub section toward the first aerodynamic section.
    '''
    cst = geometry.cst_at_stations(params.r, params.r_cst, params.cst)                  # (n_r, N_CST)
    x, w = jnp.asarray(GAUSS_X), jnp.asarray(GAUSS_W)
    y_upper, y_lower = geometry.kulfan_coordinates(cst, x)
    A1, x1, z1, I_xx1, I_zz1, I_xz1 = strip_properties(x, w, y_upper, y_lower)

    c = params.c
    pitch_axis = geometry.pitch_axis_fraction(params.r, params.cfg.R_hub, params.cfg.R_aero)
    area = A1 * c ** 2
    I_xx, I_zz, I_xz = I_xx1 * c ** 4, I_zz1 * c ** 4, I_xz1 * c ** 4

    # Outline points, with the end points of the chord added so the extreme fibres are exact
    x_out = jnp.concatenate([jnp.zeros(1), x, jnp.ones(1)])
    y_up, y_lo = geometry.kulfan_coordinates(cst, x_out)
    x_pts = jnp.concatenate([x_out, x_out[::-1]])
    z_pts = jnp.concatenate([y_up, y_lo[..., ::-1]], axis=-1)
    outline = c[:, None, None] * jnp.stack([jnp.broadcast_to(x_pts, z_pts.shape) - x1[:, None], z_pts - z1[:, None]], axis=-1)

    return SectionProps(
        area=area, x_c=c * (x1 - pitch_axis), z_c=c * z1, I_xx=I_xx, I_zz=I_zz, I_xz=I_xz,
        J=torsion_constant(area, I_xx, I_zz), t_max=c * jnp.max(y_upper - y_lower, axis=-1), outline=outline)


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


## Loads on the blade

class InertialLoads(NamedTuple):
    '''
    Inertial (d'Alembert) loads of one blade per unit radius at every root azimuth, each (n_psi, n_r, 3) on the blade
    frame. f (N/m) and m (N m/m, about the pitch-axis point) are the totals. m_cg is the moment about the centroid:
    the gyroscopic moment of the section, which holds the centrifugal twisting moment. The four force terms add up
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

def inertial_loads(params, props, mat, kin=None, axes=None) -> InertialLoads:
    '''
    The blade as a chain of rigid slices of mass rho * fill * A * ds/dr per unit radius, each of inertia tensor
    J' = rho * fill * ds/dr * [[I_xx, 0, -I_xz], [0, I_xx + I_zz, 0], [-I_xz, 0, I_zz]] about its centroid, on the
    section axes (x_s, s, z_s). The slices follow the hub, which spins at params.omega (accelerating at
    kin.omega_dot) about e_z and turns with the airframe (kin.body_rates), and pitch about e_r through the hub centre
    at kin.pitch_rate. With the total angular velocity w_h of the hub and the pitch rotation nu, a slice with its
    centroid at p_cg accelerates by
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

    offset = props.x_c[:, None] * axes.x_s + props.z_c[:, None] * axes.z_s       # centroid from the pitch-axis point
    p_cg = (axes.p + offset)[None]                                              # (1, n_r, 3)
    w, w_dot = w_h[:, None, :], w_h_dot[:, None, :]
    v_rel = jnp.cross(nu, p_cg)
    mass = (mat.density * mat.fill * props.area * axes.arc)[None, :, None]
    f_euler = -mass * jnp.cross(w_dot, p_cg)
    f_centrifugal = -mass * jnp.cross(w, jnp.cross(w, p_cg))
    f_coriolis = -mass * 2.0 * jnp.cross(w, v_rel)
    f_relative = -mass * (jnp.cross(nu_dot, p_cg) + jnp.cross(nu, v_rel))
    f = f_euler + f_centrifugal + f_coriolis + f_relative

    # Gyroscopic moment of every slice about its centroid
    w_s, w_s_dot = w_h + nu, w_h_dot + nu_dot + jnp.cross(w_h, nu)
    per_length = mat.density * mat.fill * axes.arc
    def apply_inertia(v):
        ux, uy, uz = (jnp.einsum("pk,rk->pr", v, axis) for axis in (axes.x_s, axes.s, axes.z_s))
        hx = per_length * (props.I_xx * ux - props.I_xz * uz)
        hy = per_length * (props.I_xx + props.I_zz) * uy
        hz = per_length * (props.I_zz * uz - props.I_xz * ux)
        return hx[..., None] * axes.x_s + hy[..., None] * axes.s + hz[..., None] * axes.z_s
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
    about the span, with the moment about the centroid.
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

    F, M_O, M_c          resultant force on the inboard side, its moment about the hub centre and about the centroid
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
    through the solve. include_aero=False leaves the aerodynamic loads out (inertial loads alone).
    '''
    kin = Kinematics() if kin is None else kin
    props = section_properties(params)
    axes = blade_axes(params)
    offset = props.x_c[:, None] * axes.x_s + props.z_c[:, None] * axes.z_s
    centroid_moment = lambda F, M_ref: M_ref - jnp.cross(offset[None], F)

    inertial = inertial_loads(params, props, mat, kin, axes)
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
    Stress in the blade material at every root azimuth and station (Pa), tension positive.

    sigma         axial stress at the outline points of the section (n_psi, n_r, 2 (N_X + 2))
    sigma_max     largest |sigma| of the outline
    tau           shear stress: torsion at the thickest point plus the mean shear of the section forces, a
                  conservative sum
    sigma_vm      von Mises stress sqrt(sigma_max^2 + 3 tau^2)
    safety_factor smaller of sigma_allow / sigma_vm and tau_allow / tau
    '''
    sigma: jnp.ndarray
    sigma_max: jnp.ndarray
    tau: jnp.ndarray
    sigma_vm: jnp.ndarray
    safety_factor: jnp.ndarray

def stress_fields(result: StructuralResult, mat: Material) -> Stress:
    '''
    Stress of a structural_loads result. The axial stress of the gross outline is N / A + a_x x + a_z z for the
    outline point (x, z) relative to the centroid, with the bending gradients a_x and a_z that carry the moments
    M_x = -int z sigma dA and M_z = int x sigma dA of the section (an unsymmetric section bends out of the plane of
    the moment, through the product of inertia). The stress in the material is the gross stress divided by the fill
    factor. The shear stress of a thin section is approximated by T t_max / J for the torque and 4/3 of the mean
    shear stress of the shear forces. Peak values are not fatigue limits: no S-N model.
    '''
    props, res = result.props, result.resultants
    D = props.I_xx * props.I_zz - props.I_xz ** 2
    a_x = (props.I_xz * res.M_x + props.I_xx * res.M_z) / D
    a_z = -(props.I_zz * res.M_x + props.I_xz * res.M_z) / D
    x, z = props.outline[..., 0][None], props.outline[..., 1][None]                     # (1, n_r, K)
    sigma = ((res.N / props.area)[..., None] + a_x[..., None] * x + a_z[..., None] * z) / mat.fill
    sigma_max = jnp.max(jnp.abs(sigma), axis=-1)
    tau = (jnp.abs(res.T) * props.t_max / props.J + 4.0 / 3.0 * _norm2(res.V_x, res.V_z) / props.area) / mat.fill
    sigma_vm = _norm2(sigma_max, jnp.sqrt(3.0) * tau)
    tiny = 1e-9
    safety = jnp.minimum(mat.sigma_allow / jnp.maximum(sigma_vm, tiny), mat.tau_allow / jnp.maximum(tau, tiny))
    return Stress(sigma, sigma_max, tau, sigma_vm, safety)

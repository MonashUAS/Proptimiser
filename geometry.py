## Propeller Geometry Tools and Functions

import glob
from functools import lru_cache
from math import comb
from pathlib import Path
from typing import NamedTuple

import numpy as np
import jax
import jax.numpy as jnp
import aerosandbox as asb

# CST vectors are stored as [lower weights x8, upper weights x8, LE weight, TE thickness]
N_CST_WEIGHTS = 8
N_CST = 2 * N_CST_WEIGHTS + 2
AIRFOIL_DIR = Path(__file__).resolve().parent / "airfoils"
EXPORT_DIR = Path(__file__).resolve().parent / "exports"

### Integrand Functions
def radial_stops(R, r_hub, n_r):
    """
    Return radial node centers and dr for integration, including the endpoints.
    """
    # Includes both r_hub and R as nodes
    r = jnp.linspace(r_hub, R, n_r)
    
    # Grid spacing (distance between inclusive nodes)
    dr = jnp.ones_like(r) * (R - r_hub) / (n_r - 1)
    return r, dr

def angular_stops(n_psi):
    """
    Return angular node boundaries and dpsi for integration with the endpoint excluded.
    """
    # Divides 2*pi into n_psi segments, returning the start of each boundary (drops 2*pi)
    psi = jnp.linspace(0, 2 * jnp.pi, n_psi, endpoint=False)
    dpsi = jnp.ones_like(psi) * (2 * jnp.pi) / n_psi
    return psi, dpsi

### CST (Kulfan) Airfoil Functions
def kulfan_from_name(name, n_weights_per_side=N_CST_WEIGHTS):
    """
    Returns the CST vector [lower, upper, LE weight, TE thickness] of a named airfoil,
    normalised to unit chord. Looks in the AeroSandbox database first, then airfoils/<name>/*.dat.
    """
    return np.array(_kulfan_from_name(name, n_weights_per_side))

@lru_cache(maxsize=None)
def _kulfan_from_name(name, n_weights_per_side):
    airfoil = asb.Airfoil(name)
    if airfoil.coordinates is None:
        files = glob.glob(str(AIRFOIL_DIR / name / "*.dat"))
        if not files:
            raise FileNotFoundError(f"Cannot find airfoil '{name}' in AeroSandbox or {AIRFOIL_DIR / name}")
        airfoil = asb.Airfoil(name=name, coordinates=files[0])

    kulfan = airfoil.normalize().to_kulfan_airfoil(
        n_weights_per_side=n_weights_per_side,
        normalize_coordinates=False,
    )
    p = kulfan.kulfan_parameters
    return tuple(np.concatenate([
        p['lower_weights'],
        p['upper_weights'],
        [p['leading_edge_weight']],
        [p['TE_thickness']]
    ]).tolist())

def get_cst_weights_from_names(airfoil_names):
    """
    Converts a list of airfoil strings into a (N_foils, N_CST) array of
    [lower_weights, upper_weights, LE_weight, TE_thickness] for each airfoil.
    """
    return np.stack([kulfan_from_name(name) for name in airfoil_names])

def cosine_spacing(n):
    """
    Chordwise stations clustered at the leading and trailing edges (matches np.cosspace in AeroSandbox).
    """
    return 0.5 * (1.0 - jnp.cos(jnp.linspace(0.0, jnp.pi, n)))

def kulfan_coordinates(cst, x):
    """
    Upper and lower surface y/c at chordwise stations x for CST vectors of shape (..., N_CST).
    JAX port of aerosandbox get_kulfan_coordinates (N1 = 0.5, N2 = 1.0, with leading-edge modification).

    Returns y_upper, y_lower, each of shape (..., len(x)).
    """
    n = N_CST_WEIGHTS - 1
    i = np.arange(N_CST_WEIGHTS)
    K = np.array([comb(n, k) for k in i], dtype=float)

    x = jnp.asarray(x)
    bernstein = K * x[:, None] ** i * (1.0 - x[:, None]) ** (n - i)     # (n_x, 8)
    class_fn = jnp.sqrt(x) * (1.0 - x)

    lower, upper = cst[..., :N_CST_WEIGHTS], cst[..., N_CST_WEIGHTS:2 * N_CST_WEIGHTS]
    le, te = cst[..., -2:-1], cst[..., -1:]

    lem = le * x * (1.0 - x) ** (N_CST_WEIGHTS + 0.5)
    y_lower = class_fn * (lower @ bernstein.T) - x * te / 2 + lem
    y_upper = class_fn * (upper @ bernstein.T) + x * te / 2 + lem
    return y_upper, y_lower

def max_thickness(cst, n_points=65):
    """
    Maximum thickness-to-chord ratio of CST vectors of shape (..., N_CST).
    """
    y_upper, y_lower = kulfan_coordinates(cst, cosine_spacing(n_points))
    return jnp.max(y_upper - y_lower, axis=-1)

def cst_at_stations(r, r_cst, cst):
    """
    Linearly interpolates CST vectors defined at radii r_cst (n_st, N_CST) onto radial stations r.
    CST is linear in its parameters, so this is equivalent to blending the section coordinates.
    Stations outside [r_cst[0], r_cst[-1]] take the end sections.

    Returns shape (n_r, N_CST).
    """
    return jax.vmap(jnp.interp, in_axes=(None, None, 1), out_axes=1)(r, r_cst, cst)

### Hub and Blend Zone Functions
def stadium_coordinates(template_coords, thickness_ratio):
    """
    Normalised rounded-rectangle (stadium) hub section sharing the x-coordinates of template_coords,
    which must be ordered TE (upper) -> LE -> TE (lower) as Kulfan coordinates are.
    """
    hub_coords = np.zeros_like(template_coords)
    hub_coords[:, 0] = template_coords[:, 0] # Lock X-coordinates to the template to prevent twist

    le_idx = np.argmin(template_coords[:, 0])
    rad = thickness_ratio / 2.0

    for j, x in enumerate(hub_coords[:, 0]):
        sign = 1.0 if j <= le_idx else -1.0

        if x < rad:
            y_val = np.sqrt(max(0, rad**2 - (rad - x)**2))
        elif x > 1.0 - rad:
            y_val = np.sqrt(max(0, rad**2 - (x - (1.0 - rad))**2))
        else:
            y_val = rad

        hub_coords[j, 1] = sign * y_val
    return hub_coords

def fit_hub_cst(thickness_ratio, n_points_per_side=100):
    """
    Fits a CST vector to the stadium hub section, so the blend zone can be interpolated in CST space.
    """
    from aerosandbox.geometry.airfoil.airfoil_families import get_kulfan_coordinates, get_kulfan_parameters

    template = get_kulfan_coordinates(n_points_per_side=n_points_per_side)
    p = get_kulfan_parameters(
        stadium_coordinates(template, thickness_ratio),
        n_weights_per_side=N_CST_WEIGHTS,
        normalize_coordinates=False,
    )
    return np.concatenate([
        p['lower_weights'],
        p['upper_weights'],
        [p['leading_edge_weight']],
        [p['TE_thickness']]
    ])

def planform_blend_weights(r, r_hub, r_aero):
    """
    Static weights for the blend-zone planform on a concrete radial grid r.

    Stations inboard of r_aero are blended (cosine) from the hub values at r_hub to the values at the
    anchor, the first station at or outboard of r_aero. Anchoring on a station rather than on r_aero itself
    keeps the blend from depending on other blended stations.

    Returns (blend, inboard, anchor): blend fraction per station, mask of blended stations, anchor index.
    """
    r = np.asarray(r)
    anchor = int(np.argmax(r >= r_aero))
    r_anchor = r[anchor]

    fraction = np.clip((r - r_hub) / max(r_anchor - r_hub, 1e-12), 0.0, 1.0)
    blend = 0.5 * (1.0 - np.cos(np.pi * fraction))
    inboard = r < r_anchor
    return blend, inboard, anchor

def apply_planform_blend(c, beta, blend, inboard, anchor, hub_chord, hub_twist):
    """
    Replaces chord and twist at the inboard stations with the hub-to-anchor blend (differentiable in c and beta).
    """
    c_new = jnp.where(inboard, hub_chord + blend * (c[anchor] - hub_chord), c)
    beta_new = jnp.where(inboard, hub_twist + blend * (beta[anchor] - hub_twist), beta)
    return c_new, beta_new

def blend_planform(r, c, beta, r_hub, r_aero, hub_chord, hub_twist):
    """
    Convenience wrapper: blend-zone chord and twist for a concrete radial grid r.
    """
    blend, inboard, anchor = planform_blend_weights(r, r_hub, r_aero)
    return apply_planform_blend(c, beta, blend, inboard, anchor, hub_chord, hub_twist)

### Geometry Transformation functions
def params_to_le_te(r, c, beta):
    """
    Calculates 3D coordinates for LE and TE lines.
    Aligns the 1/4 chord line precisely along the radial X-axis (pitch axis).
    """
    x = r.ravel()
    c = c.ravel()
    beta = beta.ravel()
    
    # LE is at -25% chord from the pitch axis; TE is at +75% chord from the pitch axis
    y_le = -0.25 * c * np.sin(beta)
    z_le = -0.25 * c * np.cos(beta)
    
    y_te = 0.75 * c * np.sin(beta)
    z_te = 0.75 * c * np.cos(beta)

    # Stack into clean (N, 3) positional arrays
    le_points = np.column_stack([x, y_le, z_le])
    te_points = np.column_stack([x, y_te, z_te])

    return le_points, te_points

def generate_blade_coordinates(r, r_foils, cst_weights, hub_thickness_ratio=0.1, n_points_per_side=100):
    """
    Generates blade coordinates across 3 distinct zones using a unified r_foils array.
    r_foils format: [r_start, r_hub, r_aero, ... aerodynamic stations ...]
    
    1. r <= r_hub: 100% Flat rounded rectangle (Stadium).
    2. r_hub < r < r_aero: Linear coordinate blend from Stadium to the first CST profile.
    3. r >= r_aero: 100% Aerodynamic CST interpolation.
    """
    import numpy as np
    import aerosandbox as asb
    from aerosandbox.geometry.airfoil.airfoil_families import get_kulfan_coordinates

    # Extract structural boundaries from unified array
    r_hub = r_foils[1]
    r_aero = r_foils[2]
    
    # Extract aerodynamic stations (everything from index 2 onwards)
    r_aero_stations = r_foils[2:]
    
    # Validate the inputs match
    if len(cst_weights) != len(r_aero_stations):
        raise ValueError(f"Expected {len(r_aero_stations)} sets of CST weights to match the aero stations in r_foils, but got {len(cst_weights)}.")

    n_foils, n_cst = cst_weights.shape
    n_weights_per_side = (n_cst - 2) // 2
    
    # 1. Generate the Root Airfoil (used at r_aero)
    root_cst = cst_weights[0]
    root_coords = get_kulfan_coordinates(
        lower_weights=root_cst[:n_weights_per_side],
        upper_weights=root_cst[n_weights_per_side:2*n_weights_per_side],
        leading_edge_weight=root_cst[-2],
        TE_thickness=root_cst[-1],
        n_points_per_side=n_points_per_side
    )
    
    n_actual_points = root_coords.shape[0]
    root_foil = asb.Airfoil(name="Root Aero Airfoil", coordinates=root_coords)
    
    # 2. Generate the Normalized Rounded Rectangle (Stadium) Hub
    hub_coords = stadium_coordinates(root_coords, hub_thickness_ratio)
    hub_foil = asb.Airfoil(name="Hub Section", coordinates=hub_coords)
    
    # 3. Initialize output tensor
    blade_coords = np.zeros((len(r), n_actual_points, 2))
    
    # 4. Process all radial stations
    for i, r_i in enumerate(r):
        if r_i <= r_hub:
            # Zone 1: Pure Hub
            blade_coords[i] = hub_foil.coordinates
            
        elif r_i < r_aero:
            # Zone 2: Blend Region
            fraction = (r_i - r_hub) / (r_aero - r_hub)
            
            # --- FIX APPLIED HERE ---
            blended = hub_foil.blend_with_another_airfoil(
                airfoil=root_foil, 
                blend_fraction=fraction,
                n_points_per_side=n_points_per_side
            )
            blade_coords[i] = blended.coordinates
            
        else:
            # Zone 3: Aerodynamic Span (same spanwise CST interpolation as the aerodynamic model)
            interp_weights = np.asarray(cst_at_stations(jnp.array([r_i]), jnp.asarray(r_aero_stations), jnp.asarray(cst_weights)))[0]

            coords = get_kulfan_coordinates(
                lower_weights=interp_weights[:n_weights_per_side],
                upper_weights=interp_weights[n_weights_per_side:2*n_weights_per_side],
                leading_edge_weight=interp_weights[-2],
                TE_thickness=interp_weights[-1],
                n_points_per_side=n_points_per_side
            )
            blade_coords[i] = coords
            
    return blade_coords

def offset_airfoil_profile(foil_x, foil_y, chord, skin_thickness):
    """
    Offsets a normalized 2D airfoil inward by a physical skin thickness.
    Prevents trailing edge self-intersection by clipping crossed lines.
    Assumes foil coordinates are ordered: TE_upper -> LE -> TE_lower.
    """
    import numpy as np

    # 1. Normalize the physical offset to the local chord
    # Ensure chord is a scalar value (extract from array if necessary)
    chord_val = chord[0] if isinstance(chord, np.ndarray) else chord
    offset_mag = skin_thickness / chord_val
    
    # 2. Calculate gradients to find normals
    dx = np.gradient(foil_x)
    dy = np.gradient(foil_y)
    length = np.hypot(dx, dy)
    
    # Avoid division by zero at sharp edges
    length[length == 0] = 1e-8
    
    # Calculate inward-pointing normals based on the loop direction
    nx = -dy / length
    ny = dx / length
    
    # 3. Apply the inward offset
    off_x = foil_x - offset_mag * nx
    off_y = foil_y - offset_mag * ny
    
    # 4. Filter self-intersections (The "Swallowtail" clip)
    le_idx = np.argmin(off_x)
    
    # Split into upper and lower surfaces
    upper_x, upper_y = off_x[:le_idx+1], off_y[:le_idx+1]
    lower_x, lower_y = off_x[le_idx:], off_y[le_idx:]
    
    # Reverse upper surface so X is strictly increasing (LE -> TE) for interpolation
    rev_upper_x = upper_x[::-1]
    rev_upper_y = upper_y[::-1]
    
    try:
        # Interpolate lower surface Y values onto the upper surface X coordinates
        lower_y_interp = np.interp(rev_upper_x, lower_x, lower_y)
        
        # Valid points are where the offset upper surface remains above the lower surface
        valid_idx = rev_upper_y > lower_y_interp
        
        # Revert the valid upper points back to TE -> LE order
        clean_upper_x = rev_upper_x[valid_idx][::-1]
        clean_upper_y = rev_upper_y[valid_idx][::-1]
        
        # Truncate the lower surface at the new blunt trailing edge
        max_x = np.max(clean_upper_x)
        clean_lower_x = lower_x[lower_x <= max_x]
        clean_lower_y = lower_y[lower_x <= max_x]
        
        # Reconstruct the continuous loop
        clean_core_x = np.concatenate([clean_upper_x, clean_lower_x])
        clean_core_y = np.concatenate([clean_upper_y, clean_lower_y])
        
    except Exception as e:
        print(f"Warning: Intersection clipping failed, using raw offset. {e}")
        clean_core_x, clean_core_y = off_x, off_y

    return clean_core_x, clean_core_y

AERO_PITCH_AXIS = 0.25      # pitch axis of the aerodynamic zone (x/c): the quarter-chord line

def pitch_axis_fraction(r, r_hub, r_aero, aero_pitch_axis=AERO_PITCH_AXIS):
    """
    Pitch axis position (x/c, from the leading edge) at each station of the radial grid r. It moves from the
    structural centre (0.5) at the hub to the aerodynamic axis at the anchor station, with the planform blend
    (planform_blend_weights, written in jnp so that it also works on traced radii). One source for the CAD export and
    the structural model.
    """
    r = jnp.asarray(r)
    r_anchor = r[jnp.argmax(r >= r_aero)]
    fraction = jnp.clip((r - r_hub) / jnp.maximum(r_anchor - r_hub, 1e-12), 0.0, 1.0)
    blend = 0.5 * (1.0 - jnp.cos(jnp.pi * fraction))
    return jnp.where(r < r_anchor, 0.5 + blend * (aero_pitch_axis - 0.5), aero_pitch_axis)

def overwrite_hub_planform(r, c, beta, r_foils, hub_chord=None, hub_twist=0.0, aero_pitch_axis=AERO_PITCH_AXIS):
    """
    Overwrites the chord and twist arrays in the structural root region to
    ensure a flat, untwisted hub that smoothly blends into the aerodynamic planform.
    Uses the same blend as the optimiser (blend_planform), so exported geometry matches the analysed blade.
    Also returns a dynamic pitch_axis array that shifts from 0.5 (hub) to the aero axis.
    """
    r_hub = r_foils[1]
    r_aero = r_foils[2]

    if hub_chord is None:
        hub_chord = c[0]

    blend, inboard, anchor = planform_blend_weights(r, r_hub, r_aero)
    c_new, beta_new = apply_planform_blend(np.asarray(c), np.asarray(beta), blend, inboard, anchor, hub_chord, hub_twist)

    pitch_axis = np.asarray(pitch_axis_fraction(r, r_hub, r_aero, aero_pitch_axis))

    return np.asarray(c_new), np.asarray(beta_new), pitch_axis

### Sweep and rake
# The pitch axis (quarter-chord line of the aero zone) leaves the radial line through the hub centre by a tangential
# offset y_sw and an axial offset z_rk, both functions of the polar radius r (the distance from the rotation axis).
# Blade frame (e_r, e_t, e_z), right-handed: e_r radial at the root azimuth, e_t the direction of blade motion,
# e_z the thrust direction. Positive y_sw is toward e_t (toward the leading edge) and positive z_rk toward e_z.
# The station at polar radius r then sits at p = r (cos g e_r + sin g e_t) + z_rk e_z with sin g = y_sw / r, so the
# annuli of the BEMT keep their radii.
MAX_SWEEP_RATIO = 0.9       # |y_sw| / r stays below it, so the pitch-axis point keeps a well-defined angle g about the axis
MAX_SPAN_TURN = 0.98        # the tangential component of the span direction stays below it (the frame is singular at 1).
                            # A rounded blade tip turns the quarter-chord line this far: the APC 10x6 reaches 0.92

def station_offsets(r, blade):
    """
    Pitch-axis offsets at the radial stations r (m) of a BladeSpec: array (len(r), 4) with columns
    [y_sw, z_rk, dy_sw/dr, dz_rk/dr], zeros without control points.

    The control points (blade.offset_r, offset_y, offset_z) are joined by a PCHIP spline (C1, no overshoot). Outboard of
    the last point the offsets are held. Inboard of the first point the offset tapers linearly to zero at r = 0, so
    the pitch axis is a straight radial line through the hub centre there.
    """
    r = np.asarray(r, dtype=float)
    offsets = np.zeros((r.shape[0], 4))
    if not blade.swept:
        return jnp.asarray(offsets)

    from scipy.interpolate import PchipInterpolator

    knots = np.asarray(blade.offset_r)
    inboard, outboard = r < knots[0], r > knots[-1]
    clipped = np.clip(r, knots[0], knots[-1])
    for column, values in enumerate((blade.offset_y, blade.offset_z)):
        spline = PchipInterpolator(knots, np.asarray(values))
        value, slope = spline(clipped), spline.derivative()(clipped)
        if knots[0] > 0:
            value = np.where(inboard, values[0] * r / knots[0], value)
            slope = np.where(inboard, values[0] / knots[0], slope)
        offsets[:, column] = value
        offsets[:, 2 + column] = np.where(outboard, 0.0, slope)

    # The section frame is singular when the pitch axis runs (nearly) tangentially or the sweep reaches the radius
    y, _, dy, dz = offsets.T
    sin_g = np.divide(y, r, out=np.zeros_like(y), where=r > 1e-9)
    if np.any(np.abs(sin_g) > MAX_SWEEP_RATIO):
        raise ValueError(f"Sweep offset reaches {np.max(np.abs(sin_g)):.2f} of the radius (limit {MAX_SWEEP_RATIO}): reduce offset_y")
    q = (dy - sin_g) / np.sqrt(1.0 - sin_g ** 2)
    span_turn = q / np.sqrt(1.0 + q ** 2 + dz ** 2)
    if np.any(np.abs(span_turn) > MAX_SPAN_TURN):
        raise ValueError(f"The pitch axis turns {np.degrees(np.arcsin(np.max(np.abs(span_turn)))):.0f} deg away from the radial "
                         f"line (limit {np.degrees(np.arcsin(MAX_SPAN_TURN)):.0f} deg): reduce the slope of offset_y")
    return jnp.asarray(offsets)

class SectionFrame(NamedTuple):
    """
    Geometry of the swept section frame at each station (each field has shape (n_r,)).

    sin_g, cos_g   angle g of the pitch-axis point about the rotation axis, from the radial line (sin g = y_sw / r)
    arc            ds/dr: length of the pitch axis per unit polar radius
    s_*            span direction s (unit tangent of the pitch axis), components on (rho, tau, e_z):
                   rho = cos g e_r + sin g e_t (outward in the disc plane), tau = -sin g e_r + cos g e_t (blade motion)
    c_*, n_*       section axes in the plane normal to s, components on the air basis (rho, t, a) with t = -tau
                   (opposite to the blade motion) and a = -e_z (downstream), which is right-handed. The chord line
                   at zero twist runs along c (leading edge to trailing edge), and n = s x c is the normal that
                   positive twist rotates the chord toward. Unswept: c = t, n = a.
    """
    sin_g: jnp.ndarray
    cos_g: jnp.ndarray
    arc: jnp.ndarray
    s_rho: jnp.ndarray
    s_tau: jnp.ndarray
    s_z: jnp.ndarray
    c_rho: jnp.ndarray
    c_t: jnp.ndarray
    c_a: jnp.ndarray
    n_rho: jnp.ndarray
    n_a: jnp.ndarray

def section_frame(r, offsets) -> SectionFrame:
    """
    Section frame of every station from the polar radii r (n_r,) and station_offsets (n_r, 4).
    Differentiable and smooth at zero offset, where it reduces to the unswept frame (arc = 1, c = t, n = a).
    """
    y, dy, dz = offsets[:, 0], offsets[:, 2], offsets[:, 3]
    sin_g = y / jnp.where(r > 1e-9, r, 1.0)
    cos_g = jnp.sqrt(1.0 - sin_g ** 2)
    q = (dy - sin_g) / cos_g                    # r dg/dr
    arc = jnp.sqrt(1.0 + q ** 2 + dz ** 2)
    s_rho, s_tau, s_z = 1.0 / arc, q / arc, dz / arc
    k = jnp.sqrt(jnp.maximum(1.0 - s_tau ** 2, 1e-12))
    return SectionFrame(sin_g, cos_g, arc, s_rho, s_tau, s_z,
                        c_rho=s_tau * s_rho / k, c_t=k, c_a=-s_tau * s_z / k,
                        n_rho=s_z / k, n_a=s_rho / k)

def section_origin(r, offsets):
    """
    CAD position (X, Y, Z) of the pitch-axis point of every station, each of shape (n_r,). The CAD frame has X along
    the chord (leading to trailing edge at zero twist, so X = -e_t), Y radial and Z in the thrust direction.
    Unswept: (0, r, 0).
    """
    r, offsets = np.asarray(r, dtype=float), np.asarray(offsets, dtype=float)
    y, z = offsets[:, 0], offsets[:, 1]
    return -y, np.sqrt(np.maximum(r ** 2 - y ** 2, 0.0)), z

def _cad_origin(r, offsets):
    """
    CAD origin (X, Y, Z) of every station: the pitch-axis point from section_origin, or (0, r, 0) without offsets.
    The exporters cut every section at constant Y, in the plane normal to e_r. The aerodynamic sections are normal
    to the span, so the exported chord of a section whose span makes an angle L with e_r is longer by 1 / cos(L).
    """
    if offsets is None:
        r = np.asarray(r, dtype=float)
        return np.zeros_like(r), r, np.zeros_like(r)
    return section_origin(r, offsets)

def apc_offsets(apc, r_hub, r_aero):
    """
    Sweep and rake control points (offset_r, offset_y, offset_z) in metres from a parsed APC geometry file
    (apcpropanalysis.parse_apc_pe0), for the stations at or outboard of r_hub.

    APC measures SWEEP (Y) and RAKE (Z) at the leading edge of the mould parting line, positive toward the leading
    edge and the thrust side. The pitch axis lies pitch_axis_fraction * chord behind it along the chord line, which is
    twisted by the section twist from the rotation plane. Offsets are taken relative to the first station.
    """
    inch = 0.0254
    r = np.asarray(apc["r"]) * inch
    c = np.asarray(apc["chord"]) * inch
    twist = np.radians(np.asarray(apc["twist_deg"]))
    pa = np.asarray(pitch_axis_fraction(r, r_hub, r_aero))
    y = np.asarray(apc["sweep"]) * inch - pa * c * np.cos(twist)
    z = np.asarray(apc["rake"]) * inch - pa * c * np.sin(twist)
    keep = r >= r_hub - 1e-12
    r, y, z = r[keep], y[keep], z[keep]
    return tuple(r.tolist()), tuple((y - y[0]).tolist()), tuple((z - z[0]).tolist())

def export_to_creo_ibl(r, c, beta, blade_coords, exportname,
                       pitch_axis=0.25, scale_to_mm=True, skin_thickness=0.0, out_dir=EXPORT_DIR, offsets=None):
    """
    Generates a Creo .ibl file with support for a dynamic pitch_axis array.
    Forces coordinate loops to close at the trailing edge for solid CAD imports.
    offsets (n_r, 4), from station_offsets, place every section's pitch-axis point on the swept axis.
    Writes <out_dir>/<exportname>.ibl and returns its path.
    """
    import numpy as np
    import aerosandbox as asb

    n_r, n_foil, _ = blade_coords.shape
    scale = 1000.0 if scale_to_mm else 1.0
    x0, y0, z0 = _cad_origin(r, offsets)

    if np.isscalar(pitch_axis):
        pitch_axis = np.full(n_r, pitch_axis)

    path = Path(out_dir) / f"{exportname}.ibl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        # Note: 'Open' here refers to the loft trajectory (spanwise), not the cross-sections
        f.write("Open\nArclength\n") 
        
        for i, r_i in enumerate(r):
            pa = pitch_axis[i]
            f.write(f"Begin section ! Station {i+1}: r = {r_i:.4f} m\n")
            f.write("Begin curve\n")
            
            foil = asb.Airfoil(coordinates=blade_coords[i])
            
            foil = foil.rotate(angle=-beta[i], x_center=pa, y_center=0.0)
            foil = foil.scale(c[i], c[i])
            
            base_coords = foil.coordinates
            
            if skin_thickness > 0.0:
                core_x, core_y = offset_airfoil_profile(base_coords[:, 0], base_coords[:, 1], 
                                                        c[i], skin_thickness)
            else:
                core_x, core_y = base_coords[:, 0], base_coords[:, 1]
            
            if not (np.isclose(core_x[0], core_x[-1]) and np.isclose(core_y[0], core_y[-1])):
                core_x = np.append(core_x, core_x[0])
                core_y = np.append(core_y, core_y[0])
                
            for j in range(len(core_x)):
                # Apply the dynamic pitch axis translation, then the sweep and rake of the pitch-axis point
                pt_x = (core_x[j] - (pa * c[i]) + x0[i]) * scale
                pt_y = y0[i] * scale
                pt_z = (core_y[j] + z0[i]) * scale
                
                f.write(f"{j+1}  {pt_x:.6f}  {pt_y:.6f}  {pt_z:.6f}\n")
                
    part_type = "Core Insert" if skin_thickness > 0 else "Aerodynamic Skin"
    print(f"Successfully generated Creo geometry: {exportname}.ibl ({n_r} sections) - {part_type}")
    return path

def export_parting_surface_ibl(r, c, beta, blade_coords, r_foils, exportname, 
                               pitch_axis=0.25, scale_to_mm=True,
                               x_margin=5.0, y_margin=5.0, n_camber_points=10, out_dir=EXPORT_DIR, offsets=None):
    """
    Generates a 3-zone parting surface sheet exported as a single Creo .ibl file.
    Isolates the flat extensions into distinct curve blocks to eliminate all ripples.
    offsets (n_r, 4), from station_offsets, place every section's pitch-axis point on the swept axis.
    Writes <out_dir>/<exportname>.ibl and returns its path.
    """
    n_r, n_foil, _ = blade_coords.shape
    scale = 1000.0 if scale_to_mm else 1.0
    r_hub = r_foils[1]
    x0, y0, z0 = _cad_origin(r, offsets)

    if np.isscalar(pitch_axis):
        pitch_axis = np.full(n_r, pitch_axis)

    station_data = []
    global_le_x = []
    global_te_x = []

    # 1. Process each radial station and extract the internal camber path
    for i, r_i in enumerate(r):
        pa = pitch_axis[i]
        foil = asb.Airfoil(coordinates=blade_coords[i])
        foil = foil.rotate(angle=-beta[i], x_center=pa, y_center=0.0)
        foil = foil.scale(c[i], c[i])
        base_coords = foil.coordinates
        
        pt_x = (base_coords[:, 0] - (pa * c[i]) + x0[i]) * scale
        pt_y = y0[i] * scale
        pt_z = (base_coords[:, 1] + z0[i]) * scale

        le_idx = np.argmin(pt_x)
        te_idx = np.argmax(pt_x)

        x_le, z_le = pt_x[le_idx], pt_z[le_idx]
        x_te, z_te = pt_x[te_idx], pt_z[te_idx]

        if r_i <= r_hub:
            z_le, z_te = z0[i] * scale, z0[i] * scale

        global_le_x.append(x_le)
        global_te_x.append(x_te)

        # Separate upper and lower surface paths
        side1_x, side1_z = pt_x[:le_idx+1], pt_z[:le_idx+1]
        side2_x, side2_z = pt_x[le_idx:], pt_z[le_idx:]
        
        s1_sort = np.argsort(side1_x)
        s2_sort = np.argsort(side2_x)
        
        # Build strictly internal camber coordinates
        x_internal = np.linspace(x_le, x_te, n_camber_points + 2)[1:-1]
        z_s1_interp = np.interp(x_internal, side1_x[s1_sort], side1_z[s1_sort])
        z_s2_interp = np.interp(x_internal, side2_x[s2_sort], side2_z[s2_sort])
        
        z_internal = (z_s1_interp + z_s2_interp) / 2.0
        if r_i <= r_hub:
            z_internal = np.full_like(z_internal, z0[i] * scale)
            
        station_data.append({
            'y': pt_y,
            'le': [x_le, z_le],
            'camber': list(zip(x_internal, z_internal)),
            'te': [x_te, z_te]
        })

    # Set outer absolute block boundaries
    x_min = np.min(global_le_x) - x_margin
    x_max = np.max(global_te_x) + x_margin

    sections = []

    # Helper function to structure a single station into 3 independent curve zones
    def build_zoned_section(y_val, target_station):
        x_le, z_le = target_station['le']
        x_te, z_te = target_station['te']
        # Introduce a tiny clearance offset (e.g., 0.5 mm) to push the crease away from the skin
        crease_offset = 0.5  

        # Zone 1: Left Flat Extension (Stops just short of hitting the LE skin)
        left_zone = [
            [x_min, y_val, z_le],
            [x_le - crease_offset, y_val, z_le]
        ]

        # Zone 2: Camber Line + Flat Lead-ins
        camber_zone = [
            [x_le - crease_offset, y_val, z_le], # Start flat in the open air
            [x_le, y_val, z_le]                  # Cross the LE skin smoothly
        ]
        for cx, cz in target_station['camber']:
            camber_zone.append([cx, y_val, cz])
        camber_zone.append([x_te, y_val, z_te])  # Cross the TE skin smoothly
        camber_zone.append([x_te + crease_offset, y_val, z_te]) # End flat in the open air

        # Zone 3: Right Flat Extension (Starts out in the open air)
        right_zone = [
            [x_te + crease_offset, y_val, z_te],
            [x_max, y_val, z_te]
        ]
        
        return {
            'left': left_zone,
            'camber': camber_zone,
            'right': right_zone
        }

    # Assemble the final spanwise matrix (Hub pad -> Blade Stations -> Tip pad)
    y_hub_ext = station_data[0]['y'] - y_margin
    sections.append(build_zoned_section(y_hub_ext, station_data[0]))

    for stat in station_data:
        sections.append(build_zoned_section(stat['y'], stat))

    y_tip_ext = station_data[-1]['y'] + y_margin
    sections.append(build_zoned_section(y_tip_ext, station_data[-1]))

    # Write out the multi-curve IBL file
    path = Path(out_dir) / f"{exportname}.ibl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("Open\n")
        f.write("Arclength\n") 
        
        for s_idx, sec in enumerate(sections):
            f.write(f"Begin section ! Row {s_idx+1}\n")
            
            # --- Curve Block 1: Left Plane ---
            f.write("Begin curve ! Left Extension\n")
            for pt in sec['left']:
                f.write(f"{pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f}\n")
                
            # --- Curve Block 2: Airfoil Camber ---
            f.write("Begin curve ! Internal Camber\n")
            for pt in sec['camber']:
                f.write(f"{pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f}\n")
                
            # --- Curve Block 3: Right Plane ---
            f.write("Begin curve ! Right Extension\n")
            for pt in sec['right']:
                f.write(f"{pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f}\n")
                
    print(f"Successfully generated 3-zone sharp parting surface: {exportname}.ibl")
    return path

def blade_export_geometry(prop, n_points_per_side=100):
    """
    Blade geometry for CAD export and 3D meshing from a Propeller: the radial grid extended to the spin
    centre, chord and twist with the structural hub planform applied, the pitch-axis position (x/c) at
    each station, the section coordinates (n_r, n_points, 2) and the sweep and rake offsets (n_r, 4) of the
    pitch axis (station_offsets, zero for an unswept blade).

    Returns (r, c, beta, pitch_axis, blade_coords, offsets).
    """
    blade = prop.blade
    r = np.array(prop.r)
    c = np.array(prop.c)
    beta = np.array(prop.beta)

    # Make sure r extends all the way to the origin to prevent center gaps
    if r[0] > 0.001:
        r = np.insert(r, 0, 0.0)
        c = np.insert(c, 0, c[0])
        beta = np.insert(beta, 0, beta[0])

    # r_foils = [r_start, r_hub, r_aero, r_mid..., r_tip]
    r_foils = np.array(blade.r_foils)

    # Overwrite the structural root on the 3D arrays
    c, beta, pitch_axis = overwrite_hub_planform(
        r, c, beta, r_foils,
        hub_chord=blade.hub_chord, hub_twist=np.radians(blade.hub_twist_deg), aero_pitch_axis=AERO_PITCH_AXIS
    )

    # Section coordinates from the CST sections at the aerodynamic stations r_foils[2:]
    blade_coords = generate_blade_coordinates(
        r, r_foils, np.array(prop.cst), hub_thickness_ratio=blade.hub_thickness_ratio, n_points_per_side=n_points_per_side
    )
    offsets = np.asarray(station_offsets(r, blade))
    return r, c, beta, pitch_axis, blade_coords, offsets

def build_propeller_mesh(r, c, beta, blade_coords, pitch_axis=0.25, num_blades=2, offsets=None):
    """
    Lofts a 3D propeller blade using exactly the same 2D transforms as the Creo exporter.
    Handles dynamic pitch axes to keep the hub perfectly centered. offsets (n_r, 4), from station_offsets,
    place every section's pitch-axis point on the swept axis.
    """
    import pyvista as pv
    import numpy as np
    import aerosandbox as asb

    n_r, n_foil, _ = blade_coords.shape
    x0, y0, z0 = _cad_origin(r, offsets)

    # Broadcast scalar pitch_axis to array if necessary
    if np.isscalar(pitch_axis):
        pitch_axis = np.full(n_r, pitch_axis)
        
    X_3d = np.zeros((n_r, n_foil))
    Y_3d = np.zeros((n_r, n_foil))
    Z_3d = np.zeros((n_r, n_foil))
    
    for i in range(n_r):
        pa = pitch_axis[i]
        foil = asb.Airfoil(coordinates=blade_coords[i])

        # Rotate around the local dynamic pitch axis
        foil = foil.rotate(angle=-beta[i], x_center=pa, y_center=0.0)
        foil = foil.scale(c[i], c[i])
        
        coords = foil.coordinates
        
        # Map to 3D standard axes, offsetting by the exact physical distance of the pitch axis
        X_3d[i, :] = coords[:, 0] - (pa * c[i]) + x0[i]
        Y_3d[i, :] = y0[i]
        Z_3d[i, :] = coords[:, 1] + z0[i]
        
    base_blade = pv.StructuredGrid(X_3d, Y_3d, Z_3d)
    
    meshes = [base_blade]
    for i in range(1, int(num_blades)):
        angle = i * (360.0 / num_blades)
        rotated_blade = base_blade.copy()
        rotated_blade.rotate_z(angle, inplace=True)
        meshes.append(rotated_blade)
        
    return pv.MultiBlock(meshes).combine()

if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    import pyvista as pv
    from init import Propeller

    # Load the saved aerodynamic propeller (r_foils, hub and CST sections come from its BladeSpec)
    prop = Propeller.load("Currawong_V1")
    r_foils = np.array(prop.blade.r_foils)

    # Hub planform, pitch axis and section coordinates along the full span
    r, c, beta, pitch_axis_array, blade_coords, offsets = blade_export_geometry(prop, n_points_per_side=100)

    # Plot Airfoil Geometries at the specific r_foils control stations
    plt.figure(figsize=(11, 5))
    
    # Map each control station in r_foils to the nearest index in the dense radial grid r
    for i, r_f in enumerate(r_foils):
        idx = np.argmin(np.abs(r - r_f))
        coords = blade_coords[idx]
        plt.plot(coords[:, 0], coords[:, 1], label=f"r = {r[idx]:.3f} m")
        
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.axis("equal")
    plt.xlabel("Chord Station (x/c)")
    plt.ylabel("Thickness Station (y/c)")
    plt.title("Airfoil Geometries at Defined Control Stations (r_foils)")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5)) # Moved legend outside to accommodate labels
    plt.tight_layout()
    plt.show()

    # 3D Model of Propeller
    prop_mesh = build_propeller_mesh(
        r, c, beta, blade_coords,
        pitch_axis=pitch_axis_array, num_blades=prop.blade.N, offsets=offsets
    )
    
    # Export to Creo .ibl files
    print("\nExporting Creo geometries...")
    # Main mold (Aerodynamic Skin)
    export_to_creo_ibl(
        r, c, beta, blade_coords, "propeller_blade_mold",
        pitch_axis=pitch_axis_array, skin_thickness=0.0, offsets=offsets
    )

    # Internal Core Insert
    laminate_thickness_m = 0.0015 # 1.5mm
    export_to_creo_ibl(
        r, c, beta, blade_coords, "propeller_blade_insert",
        pitch_axis=pitch_axis_array, skin_thickness=laminate_thickness_m, offsets=offsets
    )

    # Export Silhouette Curve as .ibl

    export_parting_surface_ibl(
        r, c, beta, blade_coords, r_foils, "propeller_parting_surf",
        pitch_axis=pitch_axis_array, n_camber_points=100, offsets=offsets
    )

    # Fire up the local PyVista testing window
    plotter = pv.Plotter()
    plotter.set_background("#1e1e2f")
    plotter.add_mesh(prop_mesh, color="silver", show_edges=True, edge_color="#444444", smooth_shading=True)
    plotter.add_axes()
    plotter.show()
    plotter.close()
    del plotter
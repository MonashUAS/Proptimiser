## Propeller Geometry Tools and Functions

import glob
import json
import warnings
from dataclasses import dataclass
from functools import lru_cache
from math import comb
from pathlib import Path
from typing import NamedTuple

import numpy as np
from ._jax import jnp
import aerosandbox as asb

# CST vectors are stored as [lower weights x8, upper weights x8, LE weight, TE thickness]
N_CST_WEIGHTS = 8
N_CST = 2 * N_CST_WEIGHTS + 2

# Smallest trailing edge gap (m) in the CAD geometry, so also in the mould and the CFD model. The value comes from the
# manufacturing tolerance. The aerodynamic and structural models keep the design CST.
TE_MIN_THICKNESS_M = 1.0e-4
TE_WARN_CHORD_FRACTION = 0.05       # a larger gap, as a fraction of the chord, can change the airfoil performance

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

def point_layout(r, psi):
    """
    Constants of the flat point layout: a field on the (n_r, n_psi) grid is one vector of p = n_r * n_psi points,
    ordered station by station (point k sits at station k // n_psi, azimuth k % n_psi).

    Returns (E_r, M_psi, r_pt, psi_pt) as NumPy arrays:
        E_r (p, n_r)      expands a per-station vector to the points
        M_psi (n_r, p)    averages the points of each station over one turn
        r_pt, psi_pt (p,) radius and azimuth of each point
    """
    r, psi = np.asarray(r, dtype=float), np.asarray(psi, dtype=float)
    n_r, n_psi = r.shape[0], psi.shape[0]
    E_r = np.kron(np.eye(n_r), np.ones((n_psi, 1)))
    M_psi = E_r.T / n_psi
    return E_r, M_psi, E_r @ r, np.tile(psi, n_r)

def cst_station_weights(r, r_cst):
    """
    Matrix W (n_r, n_cst) with W @ cst equal to cst_at_stations(r, r_cst, cst): the linear interpolation onto
    the stations, with the end sections held outside [r_cst[0], r_cst[-1]]. A constant, so it works on any backend.
    """
    r, r_cst = np.asarray(r, dtype=float), np.asarray(r_cst, dtype=float)
    W = np.zeros((r.shape[0], r_cst.shape[0]))
    for i, ri in enumerate(r):
        for j in range(r_cst.shape[0]):
            e = np.zeros(r_cst.shape[0])
            e[j] = 1.0
            W[i, j] = np.interp(ri, r_cst, e)
    return W

### CST (Kulfan) Airfoil Functions
def kulfan_from_name(name, n_weights_per_side=N_CST_WEIGHTS):
    """
    Returns the CST vector [lower, upper, LE weight, TE thickness] of a named airfoil,
    normalised to unit chord. Looks in the AeroSandbox database first, then airfoils/<name>/*.dat.
    """
    return np.array(_kulfan_from_name(name, n_weights_per_side))

@lru_cache(maxsize=None)
def _kulfan_from_name(name, n_weights_per_side):
    from .init import AIRFOIL_DIR       # Internal Import: init imports this module at load time

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
    return jnp.stack([jnp.interp(r, r_cst, cst[:, k]) for k in range(cst.shape[1])], axis=1)

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

def enforce_te_thickness(coords, chord, te_min=TE_MIN_THICKNESS_M):
    """
    Unit-chord section coordinates (TE_upper -> LE -> TE_lower) with a trailing edge gap of at least te_min (m) at the
    chord (m). This is the CST trailing edge term: it adds x * dy to the upper surface and subtracts it from the lower
    surface, so the camber line does not change. A larger gap does not change.
    """
    gap = coords[0, 1] - coords[-1, 1]
    dy = (te_min / chord - gap) / 2.0
    if dy <= 0.0:
        return coords

    x = coords[:, 0]
    xi = (x - x.min()) / (x.max() - x.min())
    le_idx = np.argmin(x)
    thick = coords.copy()
    thick[:le_idx + 1, 1] += xi[:le_idx + 1] * dy
    thick[le_idx + 1:, 1] -= xi[le_idx + 1:] * dy
    return thick

def generate_blade_coordinates(r, c, r_foils, cst_weights, hub_thickness_ratio=0.1, n_points_per_side=100):
    """
    Generates blade coordinates across 3 distinct zones using a unified r_foils array.
    r_foils format: [r_start, r_hub, r_aero, ... aerodynamic stations ...]
    c is the chord (m) at each station of r.
    
    1. r <= r_hub: 100% Flat rounded rectangle (Stadium).
    2. r_hub < r < r_aero: Linear coordinate blend from Stadium to the first CST profile.
    3. r >= r_aero: 100% Aerodynamic CST interpolation.

    Zones 2 and 3 have a trailing edge gap of at least TE_MIN_THICKNESS_M (enforce_te_thickness). The stadium has a
    round end, so zone 1 does not change. A warning names the stations where the added gap is a large part of the chord.
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
            blade_coords[i] = enforce_te_thickness(blended.coordinates, c[i])
            
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
            blade_coords[i] = enforce_te_thickness(coords, c[i])
            
    # The gap is a unit-chord value. A station with a larger design gap did not change, so it does not get a warning
    thick_edge = [i for i, r_i in enumerate(r) if r_i > r_hub and TE_MIN_THICKNESS_M / c[i] > TE_WARN_CHORD_FRACTION
                  and blade_coords[i, 0, 1] - blade_coords[i, -1, 1] <= TE_MIN_THICKNESS_M / c[i] * (1.0 + 1e-9)]
    if thick_edge:
        warnings.warn(f"The {TE_MIN_THICKNESS_M * 1000:.2f} mm trailing edge is more than "
                      f"{TE_WARN_CHORD_FRACTION:.0%} of the chord at r = "
                      + ", ".join(f"{r[i] * 1000:.1f} mm" for i in thick_edge)
                      + ". This can change the airfoil performance at these stations.")

    return blade_coords

CORE_TOLERANCE = 0.01       # a core thinner than this fraction of the skin thickness counts as no core. The discrete
                            # normals put micrometre noise on a section exactly two skins thick (the stadium hub)

def offset_airfoil_profile(foil_x, foil_y, skin_thickness, n_points_per_side=100):
    """
    Offsets one section inward by the skin thickness (m) and returns the core loop (core_x, core_y).
    The section is in the chord frame, in metres, before the twist rotation: x runs from the leading edge to the
    trailing edge, and the points go TE_upper -> LE -> TE_lower.

    The offset surfaces cross near the edges. The core keeps the contiguous chordwise run where the upper surface
    stays above the lower surface, so the trailing edge becomes blunt. Each side is resampled onto n_points_per_side
    cosine-spaced x values, so every core section has the same point count. The loop is closed (the first point is
    repeated at the end).

    Returns None when the section is thinner than two skin thicknesses, with the CORE_TOLERANCE margin (no core at
    this station).
    """
    foil_x, foil_y = np.asarray(foil_x, dtype=float), np.asarray(foil_y, dtype=float)

    # Unit normals on the left of the direction of travel. They point inward on a counter-clockwise loop.
    dx, dy = np.gradient(foil_x), np.gradient(foil_y)
    length = np.maximum(np.hypot(dx, dy), 1e-12)
    nx, ny = -dy / length, dx / length
    area = 0.5 * np.sum(foil_x * np.roll(foil_y, -1) - np.roll(foil_x, -1) * foil_y)
    inward = 1.0 if area > 0 else -1.0
    off_x = foil_x + inward * skin_thickness * nx
    off_y = foil_y + inward * skin_thickness * ny

    # Split at the leading edge of the original section. Sort each side by x, because the offset can fold near the edges.
    le = int(np.argmin(foil_x))
    upper = np.argsort(off_x[:le + 1])
    lower = np.argsort(off_x[le:])
    upper_x, upper_y = off_x[:le + 1][upper], off_y[:le + 1][upper]
    lower_x, lower_y = off_x[le:][lower], off_y[le:][lower]

    # Core thickness on a fine chordwise grid, then the contiguous run of positive thickness around the thickest point
    x = np.linspace(max(upper_x[0], lower_x[0]), min(upper_x[-1], lower_x[-1]), 4 * n_points_per_side)
    if x.size == 0 or x[-1] <= x[0]:
        return None
    thickness = np.interp(x, upper_x, upper_y) - np.interp(x, lower_x, lower_y)
    k = int(np.argmax(thickness))
    if thickness[k] <= CORE_TOLERANCE * skin_thickness:
        return None
    start, end = k, k
    while start > 0 and thickness[start - 1] > 0.0:
        start -= 1
    while end < x.size - 1 and thickness[end + 1] > 0.0:
        end += 1
    if end == start:
        return None

    # Resample both sides between the core leading edge and the core trailing edge
    xs = x[start] + (x[end] - x[start]) * np.asarray(cosine_spacing(n_points_per_side))
    ys_upper = np.interp(xs, upper_x, upper_y)
    ys_lower = np.interp(xs, lower_x, lower_y)
    core_x = np.concatenate([xs[::-1], xs[1:], xs[-1:]])
    core_y = np.concatenate([ys_upper[::-1], ys_lower[1:], ys_upper[-1:]])
    return core_x, core_y

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

def _ibl_path(out_dir, exportname):
    """
    Path of <out_dir>/<exportname>.ibl, with its folder made. The default out_dir is EXPORT_DIR.
    """
    from .init import EXPORT_DIR        # Internal Import: init imports this module at load time

    path = Path(EXPORT_DIR if out_dir is None else out_dir) / f"{exportname}.ibl"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path

def export_to_creo_ibl(r, c, beta, blade_coords, exportname,
                       pitch_axis=0.25, scale_to_mm=True, skin_thickness=0.0, out_dir=None, offsets=None):
    """
    Generates a Creo .ibl file with support for a dynamic pitch_axis array.
    Forces coordinate loops to close at the trailing edge for solid CAD imports.
    offsets (n_r, 4), from station_offsets, place every section's pitch-axis point on the swept axis.

    skin_thickness > 0 (m) writes the core insert: every section offset inward by the skin thickness
    (offset_airfoil_profile). A station thinner than two skin thicknesses has no core, so the insert keeps only the
    contiguous stations with a core. A ValueError tells when no station has a core, or when the core stations are
    not one contiguous run.
    Writes <out_dir>/<exportname>.ibl and returns its path. The default out_dir is EXPORT_DIR.
    """
    import numpy as np
    import aerosandbox as asb

    n_r, n_foil, _ = blade_coords.shape
    scale = 1000.0 if scale_to_mm else 1.0
    x0, y0, z0 = _cad_origin(r, offsets)

    if np.isscalar(pitch_axis):
        pitch_axis = np.full(n_r, pitch_axis)

    # Section loops (x, z) in metres, relative to the pitch-axis point, one for each kept station
    sections = {}
    for i in range(n_r):
        pa = pitch_axis[i]
        if skin_thickness > 0.0:
            # Offset in the chord frame (scaled, not yet twisted), then twist about the pitch axis
            core = offset_airfoil_profile(blade_coords[i, :, 0] * c[i], blade_coords[i, :, 1] * c[i],
                                          skin_thickness, n_points_per_side=(n_foil + 1) // 2)
            if core is None:
                continue
            foil = asb.Airfoil(coordinates=np.column_stack(core))
            base_coords = foil.rotate(angle=-beta[i], x_center=pa * c[i], y_center=0.0).coordinates
        else:
            foil = asb.Airfoil(coordinates=blade_coords[i])
            foil = foil.rotate(angle=-beta[i], x_center=pa, y_center=0.0)
            foil = foil.scale(c[i], c[i])
            base_coords = foil.coordinates

        core_x, core_y = base_coords[:, 0], base_coords[:, 1]
        if not (np.isclose(core_x[0], core_x[-1]) and np.isclose(core_y[0], core_y[-1])):
            core_x = np.append(core_x, core_x[0])
            core_y = np.append(core_y, core_y[0])
        sections[i] = (core_x - pa * c[i], core_y)

    kept = sorted(sections)
    if skin_thickness > 0.0:
        if not kept:
            thickest = max(range(n_r), key=lambda i: c[i] * np.ptp(blade_coords[i, :, 1]))
            raise ValueError(f"No station has a core with a {skin_thickness * 1000:.2f} mm skin. The thickest section is "
                             f"at r = {r[thickest] * 1000:.1f} mm, about {c[thickest] * np.ptp(blade_coords[thickest, :, 1]) * 1000:.2f} mm thick")
        gaps = [i for i in range(kept[0], kept[-1] + 1) if i not in sections]
        if gaps:
            raise ValueError(f"The core is not continuous with a {skin_thickness * 1000:.2f} mm skin: no core at "
                             + ", ".join(f"r = {r[i] * 1000:.1f} mm" for i in gaps))
        print(f"Core Insert: r = {r[kept[0]] * 1000:.1f} to {r[kept[-1]] * 1000:.1f} mm "
              f"(stations {kept[0] + 1} to {kept[-1] + 1} of {n_r})")

    path = _ibl_path(out_dir, exportname)
    with open(path, "w") as f:
        # Note: 'Open' here refers to the loft trajectory (spanwise), not the cross-sections
        f.write("Open\nArclength\n")

        for i in kept:
            core_x, core_y = sections[i]
            f.write(f"Begin section ! Station {i+1}: r = {r[i]:.4f} m\n")
            f.write("Begin curve\n")

            for j in range(len(core_x)):
                # Apply the dynamic pitch axis translation, then the sweep and rake of the pitch-axis point
                pt_x = (core_x[j] + x0[i]) * scale
                pt_y = y0[i] * scale
                pt_z = (core_y[j] + z0[i]) * scale

                f.write(f"{j+1}  {pt_x:.6f}  {pt_y:.6f}  {pt_z:.6f}\n")

    part_type = "Core Insert" if skin_thickness > 0 else "Aerodynamic Skin"
    print(f"Successfully generated Creo geometry: {exportname}.ibl ({len(kept)} sections) - {part_type}")
    return path

def export_parting_surface_ibl(r, c, beta, blade_coords, r_foils, exportname, 
                               pitch_axis=0.25, scale_to_mm=True,
                               x_margin=5.0, y_margin=5.0, n_camber_points=10, out_dir=None, offsets=None):
    """
    Generates a 3-zone parting surface sheet exported as a single Creo .ibl file.
    Isolates the flat extensions into distinct curve blocks to eliminate all ripples.
    offsets (n_r, 4), from station_offsets, place every section's pitch-axis point on the swept axis.
    Writes <out_dir>/<exportname>.ibl and returns its path. The default out_dir is EXPORT_DIR.
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
    path = _ibl_path(out_dir, exportname)
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

def _design_planform(prop):
    """Design chord and twist on the radial grid of the design, extended to the spin centre."""
    r = np.array(prop.r, dtype=float)
    c = np.array(prop.c, dtype=float)
    beta = np.array(prop.beta, dtype=float)
    # Make sure r extends all the way to the origin to prevent center gaps
    if r[0] > 0.001:
        r = np.insert(r, 0, 0.0)
        c = np.insert(c, 0, c[0])
        beta = np.insert(beta, 0, beta[0])
    return r, c, beta


def blade_export_geometry(prop, n_points_per_side=100, r=None):
    """
    Blade geometry for CAD export and 3D meshing from a Propeller: the radial grid extended to the spin
    centre, chord and twist with the structural hub planform applied, the pitch-axis position (x/c) at
    each station, the section coordinates (n_r, n_points, 2) and the sweep and rake offsets (n_r, 4) of the
    pitch axis (station_offsets, zero for an unswept blade).

    r (m), when given, is the radial grid of the export, for example standard_stations. The design chord and twist
    then come from a PCHIP spline through the design stations (C1, no overshoot). The hub planform anchors on the first
    station at or outboard of r_aero, so a grid for the standard layout must hold r_aero.

    Returns (r, c, beta, pitch_axis, blade_coords, offsets).
    """
    blade = prop.blade
    r_design, c, beta = _design_planform(prop)
    if r is None:
        r = r_design
    else:
        from scipy.interpolate import PchipInterpolator

        r = np.asarray(r, dtype=float)
        c = PchipInterpolator(r_design, c)(np.clip(r, r_design[0], r_design[-1]))
        beta = PchipInterpolator(r_design, beta)(np.clip(r, r_design[0], r_design[-1]))

    # r_foils = [r_start, r_hub, r_aero, r_mid..., r_tip]
    r_foils = np.array(blade.r_foils)

    # Overwrite the structural root on the 3D arrays
    c, beta, pitch_axis = overwrite_hub_planform(
        r, c, beta, r_foils,
        hub_chord=blade.hub_chord, hub_twist=np.radians(blade.hub_twist_deg), aero_pitch_axis=AERO_PITCH_AXIS
    )

    # Section coordinates from the CST sections at the aerodynamic stations r_foils[2:]
    blade_coords = generate_blade_coordinates(
        r, c, r_foils, np.array(prop.cst), hub_thickness_ratio=blade.hub_thickness_ratio, n_points_per_side=n_points_per_side
    )
    offsets = np.asarray(station_offsets(r, blade))
    return r, c, beta, pitch_axis, blade_coords, offsets

### Standard section layout for the CAD blade
# The blade is three blends, one for each zone: hub (the stadium prism), blend (stadium to the root airfoil) and aero.
# The zones meet with a corner (C0): the hub planform blend has zero slope at r_aero, but the aero planform does not, and
# the section shape blend is linear in r. One smooth blend through all the sections overshoots at these corners, so
# each zone is a separate .ibl file. The two files at a joint hold the same section.
# Each zone has a fixed section count, so the Creo blade part keeps its topology when the design changes. Creo then
# reads a new design through the ATB link of each curve to its file, with no change to the features.
# The stations are spaced by arc length along the pitch axis, not by radius. A swept tip, and later a winglet, then
# get the same spacing along the span.
STANDARD_SECTIONS = (("hub", 2), ("blend", 6), ("aero", 35))    # sections in each file, joint sections included
TIP_MIN_CHORD_FRACTION = 0.25   # the aero zone ends where the chord falls to this fraction of the largest aero chord.
                                # A section narrower than this makes faces that Fluent cannot tessellate at the
                                # trailing edge size, and that a mould cannot hold. The blade ends with a flat tip face.
                                # At 0.05, the APC_20x10E trailing edge turned back between the last two design
                                # stations (chord 13.3 to 0.67 mm), and Creo showed a hook at the tip (2026-11-08)
ARC_GRID = 4001                 # points of the fine grid that gives the arc length of the pitch axis


class StandardStations(NamedTuple):
    """Radii (m) of the standard layout. zones holds (name, radii) for each file, in order from the hub."""
    zones: tuple
    r_tip: float                # radius of the last section, from the tip rule
    r_design_tip: float         # radius of the last design station


def pitch_axis_arc_length(r, blade):
    """Arc length (m) of the pitch axis from the spin centre to each radius r, from the swept frame (ds/dr)."""
    r = np.asarray(r, dtype=float)
    fine = np.linspace(0.0, float(np.max(r)), ARC_GRID)
    arc = np.asarray(section_frame(jnp.asarray(fine), jnp.asarray(station_offsets(fine, blade))).arc)
    s = np.concatenate([[0.0], np.cumsum(0.5 * (arc[1:] + arc[:-1]) * np.diff(fine))])
    return np.interp(r, fine, s)


def tip_radius(prop, fraction=TIP_MIN_CHORD_FRACTION):
    """
    Radius (m) where the design chord outboard of r_aero first falls to fraction of the largest aero chord, else the
    last design station. The chord comes from the same PCHIP spline as blade_export_geometry.
    """
    from scipy.interpolate import PchipInterpolator

    r_design, c, _ = _design_planform(prop)
    r_aero = float(prop.blade.r_foils[2])
    fine = np.linspace(r_aero, r_design[-1], ARC_GRID)
    chord = PchipInterpolator(r_design, c)(fine)
    limit = fraction * np.max(chord)
    below = np.nonzero(chord < limit)[0]
    if below.size == 0:
        return float(r_design[-1])
    k = int(below[0])
    if k == 0:
        raise ValueError(f"The chord at r_aero = {r_aero * 1000:.1f} mm is below {fraction:.0%} of the largest aero chord")
    # Linear root between the last station above the limit and the first one below it
    t = (chord[k - 1] - limit) / (chord[k - 1] - chord[k])
    return float(fine[k - 1] + t * (fine[k] - fine[k - 1]))


def standard_stations(prop, layout=STANDARD_SECTIONS, fraction=TIP_MIN_CHORD_FRACTION):
    """
    Radii of the standard CAD layout. Each zone is spaced evenly in pitch-axis arc length. The aero zone is closer at
    the tip (sine spacing), where the chord and the sweep change fastest. The first section of each zone is the last
    section of the zone before it, so sections fall exactly at r_hub and r_aero.
    """
    blade = prop.blade
    r_hub, r_aero = float(blade.r_foils[1]), float(blade.r_foils[2])
    r_design_tip = float(_design_planform(prop)[0][-1])
    r_tip = tip_radius(prop, fraction)
    counts = dict(layout)
    if set(counts) != {"hub", "blend", "aero"} or min(counts.values()) < 2:
        raise ValueError(f"The layout must give hub, blend and aero, each with 2 or more sections: {layout}")

    fine = np.linspace(0.0, r_tip, ARC_GRID)
    s_fine = pitch_axis_arc_length(fine, blade)
    s_hub, s_aero, s_tip = np.interp([r_hub, r_aero, r_tip], fine, s_fine)
    t_aero = np.sin(0.5 * np.pi * np.linspace(0.0, 1.0, counts["aero"]))
    spans = {
        "hub": np.linspace(0.0, s_hub, counts["hub"]),
        "blend": np.linspace(s_hub, s_aero, counts["blend"]),
        "aero": s_aero + (s_tip - s_aero) * t_aero,
    }
    zones = []
    for name, _ in layout:
        r = np.interp(spans[name], s_fine, fine)
        ends = {"hub": (0.0, r_hub), "blend": (r_hub, r_aero), "aero": (r_aero, r_tip)}[name]
        r[0], r[-1] = ends      # exact joints, free of the round trip through the arc length
        zones.append((name, r))
    return StandardStations(tuple(zones), r_tip, r_design_tip)


def export_standard_ibl(prop, exportname, out_dir=None, n_points_per_side=100):
    """
    Write the mould skin as one .ibl file for each zone: <exportname>_hub.ibl, _blend.ibl and _aero.ibl. Every section
    has the same point count. Return (paths, stations).
    """
    stations = standard_stations(prop)
    grid = np.unique(np.concatenate([r for _, r in stations.zones]))
    r, c, beta, pitch_axis, coords, offsets = blade_export_geometry(prop, n_points_per_side=n_points_per_side, r=grid)
    paths = []
    for name, radii in stations.zones:
        rows = np.searchsorted(r, radii)
        paths.append(export_to_creo_ibl(r[rows], c[rows], beta[rows], coords[rows], f"{exportname}_{name}",
                                        pitch_axis=pitch_axis[rows], out_dir=out_dir, offsets=offsets[rows]))
    if stations.r_tip < stations.r_design_tip:
        print(f"Tip rule: the blade ends at r = {stations.r_tip * 1000:.3f} mm, "
              f"{(stations.r_design_tip - stations.r_tip) * 1000:.3f} mm inboard of the last design station")
    return paths, stations

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
    from .init import Propeller

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


### Outer mould line (OML) and STEP solid
# Outer mould line (OML) of a lifting surface, and its STEP solid.
#
# The aerodynamic tools define the OML. Creo imports the STEP solid and uses it for the parts and the assembly.
#
# A LiftingSurface is a set of sections along a spine. Each section has its own frame, so the same model holds a straight
# blade, a swept tip, a winglet or a wing:
#     origin      the pitch-axis point of the section (m, CAD frame)
#     x_axis      the chord direction at zero twist, leading edge to trailing edge (unit vector)
#     y_axis      the thickness direction of the section (unit vector, normal to x_axis)
#     chord, twist (rad), pitch_axis (x/c), coords (unit-chord loop, TE_upper -> LE -> TE_lower)
# Twist turns the section in its own plane about the pitch-axis point. The sections need not be at constant radius.
#
# from_propeller makes a blade from a Propeller design. It uses a smooth definition, so a spline loft through the sections
# does not overshoot:
# - The hub planform and the hub section shape blend into the aero trend with a C2 weight (smootherstep). The aero trend
#   continues inboard of r_aero, so chord, twist and section shape keep their slope at r_aero.
# - The stations come from standard_stations: arc length spacing along the pitch axis, and the tip rule.
# This hub blend is for the OML only. The structure model keeps blend_planform, so the two differ between r_hub and
# r_aero. The BEMT uses the aero zone only, so the performance does not change.
# Every OML section has a trailing edge gap of at least TE_MIN_THICKNESS_M, the stadium hub too (the .ibl hub has
# a round end). The blade solid then has five faces: upper, lower, trailing edge, root and tip (blade_faces).
#
# The CAD frame is the .ibl frame: origin on the rotation axis, +Y radial along the blade, +Z thrust and spin axis, mm in STEP.
# OpenCascade (the cad extra, cadquery-ocp) is imported inside the functions that need it.
M_TO_MM = 1000.0
LOFT_TOLERANCE_MM = 1.0e-4      # tolerance of the section curve interpolation
MIN_TRIM_MM = 1.0e-9
SEW_TOLERANCE_MM = 1.0e-4       # the end faces join the skin within this distance
# A larger turn between two segments near the trailing edge is a turn back. The added gap (P1 rule) makes it at a blend
# section, where the round stadium end is thicker than the gap just before the edge. Fluent Meshing could not mesh the
# fold that it gave at r = 40 to 72 mm of APC_20x10E (2026-11-08). The largest turn of a good section was 7.8 deg
TE_TURN_LIMIT_DEG = 15.0
TE_TURN_REGION = 0.9            # the turn test covers x/c above this value
# The skin goes this far past each trailing edge line, so the upper and lower faces trim a larger surface. Creo 11 put the
# trailing edge trim of the lower face at u = 1.0003, past the end of the surface. The face then folded near the tip, and
# Fluent Meshing could not mesh it (2026-11-08)
TE_EXTENSION_MM = 0.1


@dataclass(frozen=True)
class LiftingSurface:
    """Sections of a blade or a wing. Every array has the station count n as its first size."""
    name: str
    origin: np.ndarray          # (n, 3) m
    x_axis: np.ndarray          # (n, 3)
    y_axis: np.ndarray          # (n, 3)
    chord: np.ndarray           # (n,) m
    twist: np.ndarray           # (n,) rad, positive turns the chord from x_axis toward y_axis... see section_points
    pitch_axis: np.ndarray      # (n,) x/c
    coords: np.ndarray          # (n, p, 2) unit chord, TE_upper -> LE -> TE_lower

    @property
    def n_sections(self):
        return self.chord.shape[0]

    def section_points(self):
        """
        Section loops (n, p, 3) in m. The loop turns by -twist about the pitch-axis point in the (x_axis, y_axis) plane,
        which is the turn of the .ibl writer (asb.Airfoil.rotate(angle=-beta)).
        """
        x = (self.coords[..., 0] - self.pitch_axis[:, None]) * self.chord[:, None]
        y = self.coords[..., 1] * self.chord[:, None]
        cos, sin = np.cos(-self.twist)[:, None], np.sin(-self.twist)[:, None]
        xr, yr = x * cos - y * sin, x * sin + y * cos
        return self.origin[:, None, :] + xr[..., None] * self.x_axis[:, None, :] + yr[..., None] * self.y_axis[:, None, :]


def smootherstep(t):
    """C2 blend weight: 0 at t <= 0, 1 at t >= 1, with zero first and second slope at both ends."""
    t = np.clip(np.asarray(t, dtype=float), 0.0, 1.0)
    return t * t * t * (t * (6.0 * t - 15.0) + 10.0)


def _aero_cst(r, r_aero_stations, cst):
    """CST weights at radii r: the BEMT interpolation (linear) in the aero zone, and its first slope continued inboard."""
    r = np.asarray(r, dtype=float)
    inside = np.asarray(cst_at_stations(jnp.asarray(np.maximum(r, r_aero_stations[0])), jnp.asarray(r_aero_stations), jnp.asarray(cst)))
    if len(r_aero_stations) < 2:
        return inside
    slope = (cst[1] - cst[0]) / (r_aero_stations[1] - r_aero_stations[0])
    inboard = r < r_aero_stations[0]
    return np.where(inboard[:, None], cst[0][None, :] + (r - r_aero_stations[0])[:, None] * slope[None, :], inside)


def _kulfan(weights, n_points_per_side):
    from aerosandbox.geometry.airfoil.airfoil_families import get_kulfan_coordinates

    n = (len(weights) - 2) // 2
    return get_kulfan_coordinates(lower_weights=weights[:n], upper_weights=weights[n:2 * n], leading_edge_weight=weights[-2],
                                  TE_thickness=weights[-1], n_points_per_side=n_points_per_side)


def propeller_sections(prop, r, n_points_per_side=100):
    """
    Smooth chord (m), twist (rad), pitch axis (x/c) and unit-chord section loops at radii r for a Propeller design.
    """
    from scipy.interpolate import PchipInterpolator

    blade = prop.blade
    r = np.asarray(r, dtype=float)
    r_foils = np.asarray(blade.r_foils, dtype=float)
    r_hub, r_aero = r_foils[1], r_foils[2]
    r_design, c_design, beta_design = _design_planform(prop)
    inside = np.clip(r, r_design[0], r_design[-1])
    c_aero = PchipInterpolator(r_design, c_design)(inside)
    beta_aero = PchipInterpolator(r_design, beta_design)(inside)

    w = smootherstep((r - r_hub) / max(r_aero - r_hub, 1e-12))
    hub_chord = float(blade.hub_chord) if blade.hub_chord is not None else float(c_design[0])
    hub_twist = np.radians(float(blade.hub_twist_deg))
    chord = hub_chord + w * (c_aero - hub_chord)
    twist = hub_twist + w * (beta_aero - hub_twist)
    pitch_axis = 0.5 + w * (AERO_PITCH_AXIS - 0.5)

    cst = np.asarray(prop.cst, dtype=float)
    weights = _aero_cst(r, r_foils[2:], cst)
    root = _kulfan(cst[0], n_points_per_side)
    hub = stadium_coordinates(root, blade.hub_thickness_ratio)
    hub = _resample_like(hub, root)
    coords = np.empty((len(r), root.shape[0], 2))
    for i in range(len(r)):
        aero = _kulfan(weights[i], n_points_per_side)
        section = (1.0 - w[i]) * hub + w[i] * aero
        # Every section gets the gap, the stadium hub too. So the trailing edge is one flat face from root to tip.
        # The P1 rule adds the gap. Where the added gap makes the edge turn back (a blend section), a cut makes the gap
        gapped = enforce_te_thickness(section, chord[i])
        coords[i] = gapped if te_turn_deg(gapped) <= TE_TURN_LIMIT_DEG else cut_to_gap(section, chord[i])
    return chord, twist, pitch_axis, coords


def te_turn_deg(coords):
    """The largest turn (deg) between two segments of a unit-chord section at x/c above TE_TURN_REGION, on each side."""
    le = int(np.argmin(coords[:, 0]))
    largest = 0.0
    for side in (coords[:le + 1], coords[le:]):
        steps = np.diff(side, axis=0)
        steps /= np.linalg.norm(steps, axis=1, keepdims=True)
        turns = np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", steps[1:], steps[:-1]), -1.0, 1.0)))
        near = side[1:-1, 0] > TE_TURN_REGION
        largest = max(largest, float(turns[near].max()) if near.any() else 0.0)
    return largest


def cut_to_gap(coords, chord, te_min=TE_MIN_THICKNESS_M):
    """
    A unit-chord section with a trailing edge gap of te_min (m), made by a cut and not by an added thickness. The cut is at
    the x/c where the section is te_min thick, and x/c then goes back to 0 to 1. Each point keeps its x/c, so the point
    count and the leading edge index do not change. The thickness falls toward the cut, so the edge does not turn back.
    """
    le = int(np.argmin(coords[:, 0]))
    x = coords[:, 0]
    upper_x, upper_y = x[:le + 1][::-1], coords[:le + 1, 1][::-1]       # LE -> TE, x increasing
    lower_x, lower_y = x[le:], coords[le:, 1]
    grid = np.linspace(x[le], 1.0, 4001)
    thickness = np.interp(grid, upper_x, upper_y) - np.interp(grid, lower_x, lower_y)
    want = te_min / chord
    thick = np.nonzero(thickness >= want)[0]
    if len(thick) == 0:
        raise ValueError("The section is nowhere as thick as the trailing edge gap")
    k = int(thick[-1])
    if k == len(grid) - 1:
        return enforce_te_thickness(coords, chord, te_min)     # the edge is already thick enough
    # The thickness falls through want between grid k and k + 1
    fraction = (thickness[k] - want) / (thickness[k] - thickness[k + 1])
    x_cut = grid[k] + fraction * (grid[k + 1] - grid[k])
    source = x[le] + (x - x[le]) * (x_cut - x[le]) / (1.0 - x[le])
    cut = coords.copy()
    cut[:le + 1, 1] = np.interp(source[:le + 1], upper_x, upper_y)
    cut[le:, 1] = np.interp(source[le:], lower_x, lower_y)
    return cut


def _resample_like(loop, template):
    """The loop with one point for each template point, at the same x/c on the same side (upper or lower)."""
    if loop.shape == template.shape:
        return loop
    le_t, le_l = int(np.argmin(template[:, 0])), int(np.argmin(loop[:, 0]))
    upper = np.interp(template[:le_t + 1, 0][::-1], loop[:le_l + 1, 0][::-1], loop[:le_l + 1, 1][::-1])[::-1]
    order = np.argsort(loop[le_l:, 0])
    lower = np.interp(template[le_t:, 0], loop[le_l:, 0][order], loop[le_l:, 1][order])
    return np.column_stack([template[:, 0], np.concatenate([upper, lower[1:]])])


def from_propeller(prop, stations=None, n_points_per_side=100, name=None):
    """
    The OML of one blade of a Propeller design, in the CAD frame. stations (m) are the section radii. The default is
    standard_stations with the joint radii once.
    """
    if stations is None:
        zones = standard_stations(prop).zones
        stations = np.unique(np.concatenate([r for _, r in zones]))
    r = np.asarray(stations, dtype=float)
    chord, twist, pitch_axis, coords = propeller_sections(prop, r, n_points_per_side)
    offsets = np.asarray(station_offsets(r, prop.blade))
    x0, y0, z0 = section_origin(r, offsets)
    n = len(r)
    # Sections at constant Y, as the .ibl writer and the moulds use them
    return LiftingSurface(name=name or "blade", origin=np.column_stack([x0, y0, z0]),
                          x_axis=np.tile([1.0, 0.0, 0.0], (n, 1)), y_axis=np.tile([0.0, 0.0, 1.0], (n, 1)),
                          chord=chord, twist=twist, pitch_axis=pitch_axis, coords=coords)


### Solid (OpenCascade)
def leading_edge_index(surface):
    """The point index of the leading edge (smallest x/c). It must be the same in every section."""
    index = np.argmin(surface.coords[..., 0], axis=1)
    if np.any(index != index[0]):
        raise ValueError("The sections do not have the leading edge at the same point index")
    return int(index[0])


def shared_parameters(loops_mm):
    """
    One parameter set (p,) in [0, 1] for all open loops (TE_upper -> LE -> TE_lower): the mean chord-length position of
    each point. All section curves then have the same knots, so the loft surface keeps one pole for each point and each
    section.
    """
    steps = np.linalg.norm(np.diff(loops_mm, axis=1), axis=-1)
    fraction = np.cumsum(steps, axis=1) / steps.sum(axis=1, keepdims=True)
    return np.concatenate([[0.0], fraction.mean(axis=0)])


def _section_curve(points_mm, parameters):
    """Open cubic B-spline through one loop, at the shared parameters. Return the OpenCascade curve."""
    from OCP.GeomAPI import GeomAPI_Interpolate
    from OCP.collections import HArray1_double, HArray1_gp_Pnt
    from OCP.gp import gp_Pnt

    points = HArray1_gp_Pnt(1, len(points_mm))
    values = HArray1_double(1, len(points_mm))
    for k, (p, u) in enumerate(zip(points_mm, parameters), start=1):
        points.SetValue(k, gp_Pnt(*map(float, p)))
        values.SetValue(k, float(u))
    interpolate = GeomAPI_Interpolate(points, values, False, LOFT_TOLERANCE_MM)
    interpolate.Perform()
    if not interpolate.IsDone():
        raise RuntimeError("A section curve did not interpolate")
    return interpolate.Curve()


def extend_ends(loops_mm, length_mm=TE_EXTENSION_MM):
    """Each open loop (n, p, 3) with one more point at each end, length_mm past the end along the end segment."""
    start = loops_mm[:, 0] - loops_mm[:, 1]
    end = loops_mm[:, -1] - loops_mm[:, -2]
    start *= length_mm / np.linalg.norm(start, axis=-1, keepdims=True)
    end *= length_mm / np.linalg.norm(end, axis=-1, keepdims=True)
    return np.concatenate([(loops_mm[:, 0] + start)[:, None], loops_mm, (loops_mm[:, -1] + end)[:, None]], axis=1)


def _curve_poles(curve):
    return np.array([[curve.Pole(i).X(), curve.Pole(i).Y(), curve.Pole(i).Z()] for i in range(1, curve.NbPoles() + 1)])


def span_parameters(surface):
    """Spanwise parameter (n,) in [0, 1]: the distance between the section origins, summed and scaled."""
    steps = np.linalg.norm(np.diff(surface.origin, axis=0), axis=1)
    if np.any(steps <= 0.0):
        raise ValueError("Two sections have the same origin")
    v = np.concatenate([[0.0], np.cumsum(steps)])
    return v / v[-1]


def _bspline_surface(poles, u_knots, u_mults, u_degree, v_knots, v_mults, v_degree):
    """An OpenCascade B-spline surface from poles (n_u, n_v, 3) and knot lists."""
    from OCP.Geom import Geom_BSplineSurface
    from OCP.collections import Array1_double, Array1_int, Array2_gp_Pnt
    from OCP.gp import gp_Pnt

    grid = Array2_gp_Pnt(1, poles.shape[0], 1, poles.shape[1])
    for i in range(poles.shape[0]):
        for j in range(poles.shape[1]):
            grid.SetValue(i + 1, j + 1, gp_Pnt(*map(float, poles[i, j])))

    def array(values, kind, cast):
        out = kind(1, len(values))
        for k, value in enumerate(values, start=1):
            out.SetValue(k, cast(value))
        return out

    return Geom_BSplineSurface(grid, array(u_knots, Array1_double, float), array(v_knots, Array1_double, float),
                               array(u_mults, Array1_int, int), array(v_mults, Array1_int, int), int(u_degree), int(v_degree),
                               False, False)


def skin_surfaces(surface):
    """
    The blade skin as B-spline surfaces. Return a dict:
        skin            cubic around each section (u, TE_upper -> LE -> TE_lower) and a cubic spline along the span (v).
                        It goes TE_EXTENSION_MM past each trailing edge line
        trailing_edge   the flat trailing edge face: linear in u between the two trailing edge lines of the skin
        u_le            the skin u of the leading edge. The upper face is u_te[0] <= u <= u_le, the lower face
                        u_le <= u <= u_te[1]
        u_te            the skin u of the upper and the lower trailing edge line
    Every section curve has the same knots (shared_parameters), so the skin poles are the section poles, interpolated pole
    by pole along the span. A cubic in v does not swing like the high-degree loft of BRepOffsetAPI_ThruSections, which
    went 50 mm outside two blades on 2026-11-08. The skin is open in u. A periodic skin made a loop where the stadium
    trailing edge closes, and Fluent Meshing could not tessellate its seam on 2026-11-08.
    """
    from scipy.interpolate import make_interp_spline

    loops = surface.section_points() * M_TO_MM
    gaps = np.linalg.norm(loops[:, 0] - loops[:, -1], axis=-1)
    if np.any(gaps < MIN_TRIM_MM):
        raise ValueError("A section has no trailing edge gap, so the trailing edge face has no width")
    loops = extend_ends(loops)
    u = shared_parameters(loops)
    curves = [_section_curve(points, u) for points in loops]
    first = curves[0]
    n_u = first.NbPoles()
    poles = np.array([[[c.Pole(i).X(), c.Pole(i).Y(), c.Pole(i).Z()] for i in range(1, n_u + 1)] for c in curves])
    u_knots = [first.Knot(i) for i in range(1, first.NbKnots() + 1)]
    u_mults = [first.Multiplicity(i) for i in range(1, first.NbKnots() + 1)]
    for c in curves[1:]:
        if c.NbPoles() != n_u or not np.allclose([c.Knot(i) for i in range(1, c.NbKnots() + 1)], u_knots):
            raise RuntimeError("The section curves do not share their knots")

    v = span_parameters(surface)
    spline = make_interp_spline(v, poles, k=3, axis=0)
    v_poles = np.swapaxes(np.asarray(spline.c), 0, 1)      # (n_u, n_v, 3)
    v_knots, v_mults = np.unique(np.round(spline.t, 14), return_counts=True)
    skin = _bspline_surface(v_poles, u_knots, u_mults, first.Degree(), v_knots, v_mults, 3)
    # The trailing edge lines are the u iso curves of the skin at the second and the second last point. An iso curve has
    # the v knots of the skin, so a linear face between the two iso curves meets the skin exactly
    u_te = (float(u[1]), float(u[-2]))
    lines = [skin.UIso(value) for value in u_te]
    for line in lines:
        if line.NbPoles() != v_poles.shape[1] or not np.allclose([line.Knot(i) for i in range(1, line.NbKnots() + 1)], v_knots):
            raise RuntimeError("A trailing edge line does not have the v knots of the skin")
    edge_poles = np.stack([_curve_poles(lines[1]), _curve_poles(lines[0])])
    trailing_edge = _bspline_surface(edge_poles, [0.0, 1.0], [2, 2], 1, v_knots, v_mults, 3)
    return dict(skin=skin, trailing_edge=trailing_edge, u_le=float(u[leading_edge_index(surface) + 1]), u_te=u_te)


BLADE_FACE_NAMES = ("upper", "lower", "trailing_edge", "root", "tip")


def blade_faces(surface):
    """The five faces of the blade solid, as a dict of OpenCascade faces (mm) with the keys BLADE_FACE_NAMES."""
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakeWire

    parts = skin_surfaces(surface)
    skin, trailing_edge, u_le = parts["skin"], parts["trailing_edge"], parts["u_le"]
    u_upper, u_lower = parts["u_te"]
    # Each face gets its own surface, cut exactly from the skin. On 2026-11-08 Creo 11 lost the trim of a face that used
    # a part of the shared skin: the upper face got the area of the whole skin, so the import gave a quilt, not a solid.
    # Each surface still goes TE_EXTENSION_MM past the trailing edge trim (skin_surfaces)
    upper, lower = skin.Copy(), skin.Copy()
    upper.Segment(0.0, u_le, 0.0, 1.0)
    lower.Segment(u_le, 1.0, 0.0, 1.0)
    faces = dict(upper=BRepBuilderAPI_MakeFace(upper, u_upper, u_le, 0.0, 1.0, 1.0e-7).Face(),
                 lower=BRepBuilderAPI_MakeFace(lower, u_le, u_lower, 0.0, 1.0, 1.0e-7).Face(),
                 trailing_edge=BRepBuilderAPI_MakeFace(trailing_edge, 0.0, 1.0, 0.0, 1.0, 1.0e-7).Face())
    for name, v in (("root", 0.0), ("tip", 1.0)):
        wire = BRepBuilderAPI_MakeWire()
        wire.Add(BRepBuilderAPI_MakeEdge(skin.VIso(v), u_upper, u_le).Edge())
        wire.Add(BRepBuilderAPI_MakeEdge(skin.VIso(v), u_le, u_lower).Edge())
        wire.Add(BRepBuilderAPI_MakeEdge(trailing_edge.VIso(v), 0.0, 1.0).Edge())
        if not wire.IsDone():
            raise RuntimeError(f"The {name} edges do not make a closed wire")
        cap = BRepBuilderAPI_MakeFace(wire.Wire(), True)
        if not cap.IsDone():
            raise RuntimeError(f"The {name} face is not flat")
        faces[name] = cap.Face()
    return faces


def _face_centre(face):
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps

    props = GProp_GProps()
    BRepGProp.SurfaceProperties_s(face, props)
    centre = props.CentreOfMass()
    return np.array([centre.X(), centre.Y(), centre.Z()]), props.Mass()


def shape_faces(shape):
    """The faces of a shape."""
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopExp import TopExp_Explorer

    found, explorer = [], TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        found.append(explorer.Current())
        explorer.Next()
    return found


def match_face_names(shape, named_faces):
    """
    [(face of shape, name)]: each face of shape gets the name of the named face with the nearest area centre. The sewing
    and the solid fix make new faces, so the faces of the solid are not the input faces. Raise a ValueError when two
    faces get one name, or when the areas do not agree.
    """
    reference = [(name, *_face_centre(face)) for name, face in named_faces.items()]
    matched = []
    for face in shape_faces(shape):
        centre, area = _face_centre(face)
        name, _, ref_area = min(reference, key=lambda item: np.linalg.norm(item[1] - centre))
        if not np.isclose(area, ref_area, rtol=1.0e-3):
            raise ValueError(f"The face named {name} changed its area from {ref_area:.4f} to {area:.4f} mm2")
        matched.append((face, name))
    names = [name for _, name in matched]
    if sorted(names) != sorted(named_faces):
        raise ValueError(f"The solid faces do not match the named faces: {names}")
    return matched


def blade_solid_named(surface):
    """One closed solid from the five blade faces, and [(face, name)] for its faces (BLADE_FACE_NAMES). Shape in mm."""
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeSolid, BRepBuilderAPI_Sewing
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.ShapeFix import ShapeFix_Solid
    from OCP.TopAbs import TopAbs_SHELL
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS

    faces = blade_faces(surface)
    sewing = BRepBuilderAPI_Sewing(SEW_TOLERANCE_MM)
    for face in faces.values():
        sewing.Add(face)
    sewing.Perform()
    explorer = TopExp_Explorer(sewing.SewedShape(), TopAbs_SHELL)
    if not explorer.More():
        raise RuntimeError("The skin and the end faces do not close")
    solid = BRepBuilderAPI_MakeSolid(TopoDS.Shell(explorer.Current())).Solid()
    fix = ShapeFix_Solid(solid)
    fix.Perform()
    shape = fix.Solid()
    if not BRepCheck_Analyzer(shape).IsValid():
        raise RuntimeError("The blade solid is not valid")
    return shape, match_face_names(shape, faces)


def blade_solid(surface):
    """One closed solid from the five blade faces (blade_faces). Return the OpenCascade shape (mm)."""
    return blade_solid_named(surface)[0]


def hub_solid(radius_mm, z_low_mm, z_high_mm):
    """A hub cylinder on the spin axis (+Z), from z_low_mm to z_high_mm."""
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeCylinder
    from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt

    axis = gp_Ax2(gp_Pnt(0.0, 0.0, float(z_low_mm)), gp_Dir(0.0, 0.0, 1.0))
    return BRepPrimAPI_MakeCylinder(axis, float(radius_mm), float(z_high_mm - z_low_mm)).Shape()


def solids(shape):
    """The solids in a shape."""
    from OCP.TopAbs import TopAbs_SOLID
    from OCP.TopExp import TopExp_Explorer

    found, explorer = [], TopExp_Explorer(shape, TopAbs_SOLID)
    while explorer.More():
        found.append(explorer.Current())
        explorer.Next()
    return found


FUSE_FUZZY_MM = 1.0e-3          # tolerance of the trim at the hub
FUSE_FUZZY_TRIES_MM = (0.0, 1.0e-3, 1.0e-2)   # join tolerances to try. On 2026-11-08 each one failed for some rotor
HUB_OVERLAP_MM = 0.5            # each trimmed blade starts this far inside the hub, so the join has a volume overlap


def trimmed_blade(blade, hub_radius_mm):
    """The blade outboard of a cylinder of radius hub_radius_mm - HUB_OVERLAP_MM about +Z."""
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeCylinder
    from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt

    (_, _, z_low), (_, _, z_high) = shape_box_mm(blade)
    margin = 10.0 + (z_high - z_low)
    core = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(0.0, 0.0, z_low - margin), gp_Dir(0.0, 0.0, 1.0)),
                                    hub_radius_mm - HUB_OVERLAP_MM, (z_high - z_low) + 2.0 * margin).Shape()
    cut = BRepAlgoAPI_Cut(blade, core)
    cut.SetFuzzyValue(FUSE_FUZZY_MM)
    cut.Build()
    if not cut.IsDone() or len(solids(cut.Shape())) != 1:
        raise RuntimeError("The trim of the blade at the hub failed")
    return cut.Shape()


def rotor_solid(blade, n_blades, hub, hub_radius_mm):
    """
    The hub joined with n_blades copies of the blade, turned by 360 / n_blades deg about +Z. Each blade is first trimmed
    at the hub radius, so the blades touch the hub and not each other. The stadium faces of neighbour blades lie on one
    plane where the roots overlap, and a join of whole blades failed for 4 of 12 cases on 2026-11-08. Raise a
    RuntimeError when the result is not one valid solid.
    """
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Copy, BRepBuilderAPI_Transform
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.collections import List_TopoDS_Shape
    from OCP.gp import gp_Ax1, gp_Dir, gp_Pnt, gp_Trsf

    if int(n_blades) != n_blades or n_blades < 1:
        raise ValueError(f"The blade count must be a positive integer, not {n_blades}")
    outer = trimmed_blade(blade, hub_radius_mm)
    tried = []
    for fuzzy in FUSE_FUZZY_TRIES_MM:
        # A join with a tolerance changes the tolerances of its input shapes, so every try starts from new copies
        arguments, tools = List_TopoDS_Shape(), List_TopoDS_Shape()
        arguments.Append(BRepBuilderAPI_Copy(hub).Shape())
        for k in range(int(n_blades)):
            turn = gp_Trsf()
            turn.SetRotation(gp_Ax1(gp_Pnt(0.0, 0.0, 0.0), gp_Dir(0.0, 0.0, 1.0)), 2.0 * np.pi * k / n_blades)
            tools.Append(BRepBuilderAPI_Transform(BRepBuilderAPI_Copy(outer).Shape(), turn, True).Shape())
        fuse = BRepAlgoAPI_Fuse()
        fuse.SetArguments(arguments)
        fuse.SetTools(tools)
        if fuzzy > 0.0:
            fuse.SetFuzzyValue(fuzzy)
        fuse.Build()
        if not fuse.IsDone():
            tried.append(f"{fuzzy} mm: not done")
            continue
        rotor = fuse.Shape()
        count = len(solids(rotor))
        if count == 1 and BRepCheck_Analyzer(rotor).IsValid():
            return rotor
        tried.append(f"{fuzzy} mm: {count} solids, valid {BRepCheck_Analyzer(rotor).IsValid()}")
    raise RuntimeError(f"The join of the hub and {n_blades} blades gave no valid solid: " + "; ".join(tried))


def shape_volume_mm3(shape):
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps

    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, props)
    return props.Mass()


def shape_box_mm(shape):
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib

    box = Bnd_Box()
    BRepBndLib.Add_s(shape, box)
    low, high = box.CornerMin(), box.CornerMax()
    return (low.X(), low.Y(), low.Z()), (high.X(), high.Y(), high.Z())


def write_step(shape, path, name, face_names=()):
    """
    Write the shape to a STEP file in mm, with name as the product name. face_names is [(face, name)] for faces of shape.
    Each name goes in the name field of its ADVANCED_FACE. Return the path.
    """
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.Interface import Interface_Static
    from OCP.STEPCAFControl import STEPCAFControl_Writer
    from OCP.STEPControl import STEPControl_AsIs
    from OCP.TCollection import TCollection_ExtendedString
    from OCP.TDataStd import TDataStd_Name
    from OCP.TDocStd import TDocStd_Document
    from OCP.XCAFDoc import XCAFDoc_DocumentTool

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    document = TDocStd_Document(TCollection_ExtendedString("MDTV-XCAF"))
    shapes = XCAFDoc_DocumentTool.ShapeTool_s(document.Main())
    label = shapes.AddShape(shape, False)
    TDataStd_Name.Set_s(label, TCollection_ExtendedString(name))
    for face, face_name in face_names:
        TDataStd_Name.Set_s(shapes.AddSubShape(label, face), TCollection_ExtendedString(face_name))
    writer = STEPCAFControl_Writer()       # the writer registers the parameters below, so it comes first
    Interface_Static.SetIVal_s("write.stepcaf.subshapes.name", 1 if face_names else 0)
    Interface_Static.SetCVal_s("write.step.unit", "MM")
    Interface_Static.SetCVal_s("write.step.schema", "AP214IS")
    writer.SetNameMode(True)
    if not writer.Transfer(document, STEPControl_AsIs):
        raise RuntimeError("The STEP transfer failed")
    if writer.Write(str(path)) != IFSelect_RetDone:
        raise RuntimeError(f"The STEP write to {path} failed")
    return path


HUB_HEIGHT_MARGIN = 1.2         # the hub cylinder is this much taller than the hub section, so no hub face lies on a blade face
HUB_RADIAL_MARGIN_MM = 1.0     # blade points this far outside the hub radius also set the hub height
HUB_COVER_MARGIN = 1.1          # the hub radius is this much larger than the radius where two root sections cross


def default_hub(prop, surface, n_blades=None):
    """
    (radius, z_low, z_high) in mm of the default hub. The radius is the larger of r_hub and the radius where the root
    sections of two neighbour blades cross: w / sin(pi / N) for a root half width w. With 3 or more blades the roots
    cross away from the axis, and a hub smaller than this gave an invalid join (tested on 2026-11-08 for 3 and 5 blades).
    Two blades meet only at the axis. The height is the height of the hub sections with a margin.
    """
    count = int(prop.blade.N if n_blades is None else n_blades)
    r_hub = float(prop.blade.r_foils[1]) * M_TO_MM
    points = surface.section_points() * M_TO_MM
    inside = surface.origin[:, 1] * M_TO_MM <= r_hub + 1e-9
    half_width = 0.5 * float(np.max(surface.chord[inside])) * M_TO_MM
    crossing = half_width / np.sin(np.pi / count) if count >= 3 else half_width
    radius = max(r_hub, HUB_COVER_MARGIN * crossing)
    # The hub holds every blade point inside its radius, so a blade enters the hub through the side face only
    near = np.hypot(points[..., 0], points[..., 1]) <= radius + HUB_RADIAL_MARGIN_MM
    z = points[..., 2][near]
    middle, half = 0.5 * (z.max() + z.min()), 0.5 * (z.max() - z.min()) * HUB_HEIGHT_MARGIN
    return radius, middle - half, middle + half


FACE_TABLE_FORMAT = 1


def _turned(shape, angle_rad):
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
    from OCP.gp import gp_Ax1, gp_Dir, gp_Pnt, gp_Trsf

    turn = gp_Trsf()
    turn.SetRotation(gp_Ax1(gp_Pnt(0.0, 0.0, 0.0), gp_Dir(0.0, 0.0, 1.0)), float(angle_rad))
    return BRepBuilderAPI_Transform(shape, turn, True).Shape()


def _true_box_mm(shape):
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib

    box = Bnd_Box()
    BRepBndLib.AddOptimal_s(shape, box, False, False)
    low, high = box.CornerMin(), box.CornerMax()
    return [[low.X(), low.Y(), low.Z()], [high.X(), high.Y(), high.Z()]]


def face_table(face_names, n_blades, hub_mm, name):
    """
    The named faces of each blade instance, for a CAD check after import. Blade k turns by 360 k / n_blades deg about +Z, as the
    Creo assembly places it. Each face has its area (mm2) and its true box (mm) in the CAD frame. A CFD mesh can match its
    zones to these faces by box, because no CAD reader keeps the STEP face names (Fluent 2026 R1, 2026-11-08).
    """
    faces = []
    for k in range(int(n_blades)):
        angle = 2.0 * np.pi * k / n_blades
        for face, face_name in face_names:
            faces.append(dict(blade=k, name=face_name, area_mm2=_face_centre(face)[1], box_mm=_true_box_mm(_turned(face, angle))))
    return dict(format_version=FACE_TABLE_FORMAT, name=name, n_blades=int(n_blades), angle_unit="deg",
                blade_angles_deg=[360.0 * k / n_blades for k in range(int(n_blades))],
                hub_mm=dict(radius=hub_mm[0], z_low=hub_mm[1], z_high=hub_mm[2]), faces=faces)


def export_propeller_step(prop, out_dir, exportname, n_blades=None, stations=None, hub=None, rotor=True):
    """
    Write <exportname>_blade.step, one blade in the CAD frame, for a Propeller design. This is the main output: the Creo
    assembly places the blades and the hub. With rotor=True, also try <exportname>_rotor.step: the hub and n_blades blades
    joined in one solid. The OpenCascade join failed for 5 of 12 design and blade count cases on 2026-11-08, so a failed
    join gives no rotor file and its reason. n_blades defaults to the design blade count. hub is (radius, z_low, z_high)
    in mm, default default_hub.
    Also write <exportname>_blade_faces.json, the face table (face_table).
    Return a dict: blade (path), faces (path), rotor (path or None), rotor_error (text or None), hub (radius, z_low,
    z_high), surface.
    """
    surface = from_propeller(prop, stations=stations, name=exportname)
    blade, face_names = blade_solid_named(surface)
    count = int(prop.blade.N if n_blades is None else n_blades)
    hub = tuple(float(v) for v in (hub or default_hub(prop, surface, count)))
    out_dir = Path(out_dir)
    result = dict(blade=write_step(blade, out_dir / f"{exportname}_blade.step", f"{exportname.upper()}_BLADE", face_names),
                  rotor=None, rotor_error=None, hub=hub, surface=surface)
    table = face_table(face_names, count, hub, exportname)
    result["faces"] = out_dir / f"{exportname}_blade_faces.json"
    result["faces"].write_text(json.dumps(table, indent=1) + "\n", encoding="utf-8")
    if rotor:
        try:
            joined = rotor_solid(blade, count, hub_solid(*hub), hub[0])
            result["rotor"] = write_step(joined, out_dir / f"{exportname}_rotor.step", f"{exportname.upper()}_ROTOR_{count}")
        except RuntimeError as error:
            result["rotor_error"] = str(error)
    return result

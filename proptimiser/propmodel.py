## Define All Physical Properties for Given Propeller Geometry

# External Libraries
import dataclasses
from typing import NamedTuple

import numpy as np

from ._jax import jax, jnp, eqx, jit

# Internal Libraries
from . import geometry
from .init import Inputs, BladeSpec, Params, Propeller
from .airfoil import aero_coeffs
from .airflow import tip_root_loss_factors, inflow_model, equivelantActuatorDiskEfficiency, resultantVelocityComponents, high_induction_correction, station_to_points, to_grid, section_frame_of
from .backend import JAX_OPS, ops


def motorPower(torque: float, omega: float, I_0: float = 0.7, R_tot: float = 0.120, kV: float = 150,
               kappa: float = 1.0, Q0: float = 0.0):
    """
    Electrical input power of the motor and ESC that drive the propeller. This is the Propalyser chain
    (estimate/chain.py), a Drela first-order motor behind an ESC that works as a chopper:

        I = I_0 + (torque + Q0) * Kv           Kv = kV * 2 pi / 60 (rad/s per volt)
        V_term = omega / (kappa * Kv) + R_tot * I
        P_in = V_term * I

    P_in is the power at the ESC input when V_term is the equivalent bus voltage (the duty is V_term / V_bus).
    With kappa = 1 and Q0 = 0 the result is the old first-order motor model.

    Inputs:
        torque: shaft torque of the propeller (Nm)
        omega: rotational rate of the motor (rad/s)
        I_0: zero load current (A)
        R_tot: resistance of the motor, ESC and leads (ohms)
        kV: motor speed constant (RPM/volt)
        kappa: ESC voltage factor (nd)
        Q0: friction torque at no load (Nm)
    """
    return motor_voltage(torque, omega, I_0, R_tot, kV, kappa, Q0) * motor_current(torque, I_0, kV, Q0)

def motor_current(torque, I_0, kV, Q0=0.0):
    """
    Motor current (A) for a shaft torque (Nm).
    """
    return I_0 + (torque + Q0) * kV * 2 * jnp.pi / 60

def motor_voltage(torque, omega, I_0, R_tot, kV, kappa=1.0, Q0=0.0):
    """
    Equivalent bus voltage (V) that the motor and ESC need: back EMF over kappa, plus the resistive drop.
    """
    return omega / (kappa * kV * 2 * jnp.pi / 60) + R_tot * motor_current(torque, I_0, kV, Q0)

## Azimuthal averaging

def azimuth_mean(x: jnp.ndarray, params: Params) -> jnp.ndarray:
    """
    Average over one rotation of a field on the flat point layout (p,), giving one value for each station (n_r,).
    psi is a uniform periodic grid that stops one step short of 2 pi, so the rule for a full turn is the plain
    mean (params.M_psi). jnp.trapezoid(x, x=psi) would end at psi[-1] and drop the last interval: a constant field
    would integrate to (n_psi - 1) / n_psi of its true value. The integral over a turn is 2 pi * azimuth_mean(x).
    """
    return ops(params.cfg).matmul(params.M_psi, x)


## Blade Element Theory

class BladeForces(NamedTuple):
    """
    Aerodynamic loads of ONE blade per unit polar radius at every point (station, azimuth) of the flat point layout,
    each of shape (p,). The force components are on the air basis (rho, t, a) of the swept section frame
    (geometry.SectionFrame), with f_z the thrust (along e_z = -a). They are N/m, and the pitching moment m_s about
    the pitch axis (nose-up positive, along the span direction s) is N m/m.
    """
    f_rho: jnp.ndarray
    f_t: jnp.ndarray
    f_z: jnp.ndarray
    m_s: jnp.ndarray
    C_L: jnp.ndarray
    C_D: jnp.ndarray
    C_M: jnp.ndarray
    V_R: jnp.ndarray
    phi: jnp.ndarray

def _force_components(V_i: jnp.ndarray, params: Params) -> BladeForces:
    """
    Blade-element loads of one blade from the section flow (V_R, phi) and coefficients. The section sees the flow
    normal to the span, with lift normal to it and drag along it; the load per unit polar radius carries the factor
    ds/dr of the span. The pitching moment uses the section chord and C_M about the quarter chord, which is the pitch
    axis of the aerodynamic zone. Unswept (arc = 1, c = t, n = a), these reduce to dT_blade and dH_blade.
    """
    xp = ops(params.cfg)
    V_R, phi = resultantVelocityComponents(V_i, params)
    C_L, C_D, C_M = aero_coeffs(V_i, params)

    frame = section_frame_of(params)
    pt = lambda x: station_to_points(x, params)
    c_pt = params.scale * pt(params.c)
    q_c = 0.5 * params.rho * c_pt * V_R ** 2 * pt(frame.arc)
    cos_phi, sin_phi = xp.cos(phi), xp.sin(phi)

    # drag along d = cos(phi) c + sin(phi) n, lift along l = sin(phi) c - cos(phi) n
    f_rho = q_c * (C_L * (sin_phi * pt(frame.c_rho) - cos_phi * pt(frame.n_rho))
                   + C_D * (cos_phi * pt(frame.c_rho) + sin_phi * pt(frame.n_rho)))
    f_t = q_c * pt(frame.c_t) * (C_L * sin_phi + C_D * cos_phi)
    f_a = q_c * (C_L * (sin_phi * pt(frame.c_a) - cos_phi * pt(frame.n_a))
                 + C_D * (cos_phi * pt(frame.c_a) + sin_phi * pt(frame.n_a)))
    m_s = q_c * c_pt * C_M
    return BladeForces(f_rho, f_t, -f_a, m_s, C_L, C_D, C_M, V_R, phi)

class SectionalLoads(NamedTuple):
    """
    Loads of ONE blade per unit polar radius at every (station, azimuth), on the blade frame (e_r, e_t, e_z):
    f (n_r, n_psi, 3) in N/m, m (n_r, n_psi, 3) in N m/m (the pitching moment about the pitch axis, along the span
    direction), and the section flow and coefficients on the (n_r, n_psi) grid. The azimuth grid is that of the
    pitch-axis point of each station: the blade root is at psi - g(r), with g the sweep angle of the station. Zero
    in the blend zone.
    """
    f: jnp.ndarray
    m: jnp.ndarray
    C_L: jnp.ndarray
    C_D: jnp.ndarray
    C_M: jnp.ndarray
    V_R: jnp.ndarray
    phi: jnp.ndarray

def sectional_forces(V_i: jnp.ndarray, params: Params) -> SectionalLoads:
    """
    Aerodynamic loads of one blade at every station and azimuth for the induced velocity V_i (3, p), as the
    blade-element side of the BEMT sees them (no Prandtl loss factor). The aerodynamic zone only: the blend zone
    carries no force. Averaged over azimuth and times N, f gives dT_blade and dH_blade. JAX only: the loads are
    reshaped to the (n_r, n_psi) grid.
    """
    b = _force_components(V_i, params)
    grid = lambda x: to_grid(x, params)
    f_rho, f_t, f_z, m_s = grid(b.f_rho), grid(b.f_t), grid(b.f_z), grid(b.m_s)
    frame = section_frame_of(params)
    sin_g, cos_g = frame.sin_g[:, None], frame.cos_g[:, None]

    # air basis (rho, t, a) -> blade frame: rho = cos g e_r + sin g e_t, t = sin g e_r - cos g e_t, a = -e_z
    f = jnp.stack([cos_g * f_rho + sin_g * f_t, sin_g * f_rho - cos_g * f_t, f_z], axis=-1)

    # span direction on the blade frame, from its components on (rho, tau, e_z) with tau = -sin g e_r + cos g e_t
    s = jnp.stack([frame.cos_g * frame.s_rho - frame.sin_g * frame.s_tau,
                   frame.sin_g * frame.s_rho + frame.cos_g * frame.s_tau, frame.s_z], axis=-1)
    m = m_s[..., None] * s[:, None, :]

    aero = aero_zone(params)[:, None, None]
    return SectionalLoads(jnp.where(aero, f, 0.0), jnp.where(aero, m, 0.0),
                          grid(b.C_L), grid(b.C_D), grid(b.C_M), grid(b.V_R), grid(b.phi))

def dT_blade(V_i: jnp.ndarray, params: Params, debug: bool = False):
    """
    Calculates instantaneous thrust at each point, and calculates the average thrust in each annulus over a rotation.

    Inputs:
        V_i         <- Induced velocity on the flat point layout, shape (3, p) [axial, tangential, radial]
        params      <- Params object used in the optimisation loop, remains unmodified by function execution
        debug       <- Boolean to select whether or not to print debug information when function is called

    Outputs:
        dT_blade_ave w/ size (Nr, )                        <- Thrust per annulus

    Assumes:
        |V_yz| < |V_x + V_ia|   <- Assumption that the annuli do not interact with each other

    Uses formulation from (M. Nahon et al., 2015) equations 5 and 6 modified with
    the inclusion of the prandtl loss parameter F
    """
    xp = ops(params.cfg)
    N = params.N
    rho = params.rho # air density

    if params.cfg.swept:
        # Thrust of the section in the swept frame, per unit polar radius
        dT_blade_instantaneous = _force_components(V_i, params).f_z
    else:
        # Calculate Local Flow Properties
        V_R, phi = resultantVelocityComponents(V_i, params)

        # Get Aerodynamic Coefficients
        C_L, C_D, _ = aero_coeffs(V_i, params)

        # Blade Element Theory instantaneous thrust element
        dT_blade_instantaneous = 0.5 * rho * (params.scale * station_to_points(params.c, params)) * (V_R**2) * (C_L * xp.cos(phi) - C_D * xp.sin(phi))

    # Average thrust in each annulus over one rotation, summed over the N blades
    dT_blade_ave = N * azimuth_mean(dT_blade_instantaneous, params)

    if params.cfg.V_i_0_verbose >= 2 and xp is JAX_OPS:
        jax.debug.print("  [BEMT] dT_blade: {dT_blade_ave}", dT_blade_ave=dT_blade_ave)

    return dT_blade_ave

def dH_blade(V_i: jnp.ndarray, params: Params, debug: bool = False):
    """
    Calculates instantaneous hub force at each point, and calculates the average hub force in each annulus over a
    rotation.

    The hub forces are defined as the force element coplanar with the rotor disk
    and normal to the radial direction.

    Inputs:
        V_i         <- Induced velocity on the flat point layout, shape (3, p) [axial, tangential, radial]
        params      <- Params object used in the optimisation loop, remains unmodified by function execution
        debug       <- Boolean to select whether or not to print debug information when function is called

    Outputs:
        dH_blade_ave w/ size (Nr, )                        <- Force per annulus

    Assumes:
        |V_yz| < |V_x + V_ia|   <- Assumption that the annuli do not interact with each other

    Uses formulation from (M. Nahon et al., 2015) equation 11. No Prandtl loss factor enters the blade-element
    side; it is applied to the momentum thrust and torque in thrust() and torque().
    """
    xp = ops(params.cfg)
    N, c = params.N, params.scale * params.c # geometric values
    rho = params.rho # air density

    if params.cfg.swept:
        # In-plane force of the section normal to the polar radius, in the swept frame, per unit polar radius
        dH_blade_ave = N * azimuth_mean(_force_components(V_i, params).f_t, params)
    else:
        # Calculate Local Flow Properties
        V_R, phi = resultantVelocityComponents(V_i, params)

        # Get Aerodynamic Coefficients
        C_L, C_D, _ = aero_coeffs(V_i, params)

        # Inner integral term
        inner = (V_R**2) * (C_L * xp.sin(phi) + C_D * xp.cos(phi))

        # Average hub force per annulus over one rotation, summed over the N blades
        dH_blade_ave = N / 2 * rho * c * azimuth_mean(inner, params)

    if debug:
        print("------dH_blade DEBUG------")
        print("dH Shapes (ave.):", jnp.shape(dH_blade_ave))
        print("V_i shape:", jnp.shape(V_i))
        print("------dH_blade DEBUG------")

    return dH_blade_ave


def dQ_blade(V_i: jnp.ndarray, params: Params, debug: bool = False):
    """
    Calculates instantaneous torque at each point, and calculates the average torque in each annulus over a rotation.

    Inputs:
        V_i         <- Induced velocity on the flat point layout, shape (3, p) [axial, tangential, radial]
        params      <- Params object used in the optimisation loop, remains unmodified by function execution
        debug       <- Boolean to select whether or not to print debug information when function is called

    Outputs:
        dQ_blade_ave w/ size (Nr, )                        <- Torque per annulus

    Assumes:
        |V_yz| < |V_x + V_ia|   <- Assumption that the annuli do not interact with each other

    Uses formulation from (M. Nahon et al., 2015) equation 12
    """
    r = params.scale * params.r

    # Calculate the hub force elements dH
    dH_blade_ave = dH_blade(V_i, params, debug)

    # Use equation 12 to calculate the torque elements from the hub elements
    dQ_blade_ave = dH_blade_ave * r

    if debug:
        print("------dQ_blade DEBUG------")
        print("dQ Shapes (ave.):", jnp.shape(dQ_blade_ave))
        print("V_i shape:", jnp.shape(V_i))
        print("------dQ_blade DEBUG------")

    return dQ_blade_ave


### Momentum Theory
def dT_momentum(V_i: jnp.ndarray, params: Params, debug: bool = False, apply_tip_loss: bool = True):
    """
    Calculates instantaneous thrust at each point, and calculates the average thrust in each annulus over a rotation.

    dT_momentum uses the momentum change in the air as it passes through the
    blade element and is accelerated rather than the forces on the blade element to determine
    thrust.

    Inputs:
        V_i         <- Induced velocity on the flat point layout, shape (3, p) [axial, tangential, radial]
        params      <- Params object used in the optimisation loop, remains unmodified by function execution
        debug       <- Boolean to select whether or not to print debug information when function is called

    Outputs:
        dT_momentum_ave w/ size (Nr, )                     <- Thrust per annulus

    Assumes:
        |V_yz| < |V_x + V_ia|   <- Assumption that the annuli do not interact with each other

    Uses formulation from (M. Nahon et al., 2015) equation 9
    """
    xp = ops(params.cfg)

    # Unpack axial induced velocity component
    V_ia = xp.row(V_i, 0)

    # Momentum theory about annular element
    r_pt = station_to_points(params.r, params)
    V_x, V_yz = params.V_x, params.V_yz
    rho = params.rho

    V_R, phi = resultantVelocityComponents(V_i, params)

    if apply_tip_loss:
        tip_loss, root_loss = tip_root_loss_factors(phi, params)
        F = tip_loss * root_loss
        F = xp.clip(F, 0.05, 1.0)
    else:
        F = xp.ones_like(V_ia)   # no factor in the solve (Inputs.loss_in_solve = 0)

    # Compute baseline classical momentum element locally
    V_prime = xp.sqrt((V_x + V_ia) ** 2 + V_yz ** 2)
    dT_element_classical = 2 * rho * (params.scale * r_pt) * V_ia * V_prime * F

    # Route through the configured model for conditional modifications
    dT_element_corrected = high_induction_correction(dT_element_classical, V_ia, F, params)

    # Integrate over one full turn of psi to get the thrust per annulus
    dT_momentum_ave = 2 * jnp.pi * azimuth_mean(dT_element_corrected, params)

    if debug:
        print("------dT_momentum DEBUG------")
        print("dT Shapes (ave.):", jnp.shape(dT_momentum_ave))
        print("V_i shape:", jnp.shape(V_i))
        print("F shape:", jnp.shape(F))
        print("------dT_momentum DEBUG------")

    return dT_momentum_ave

def dQ_momentum(V_i, params, debug=False, apply_tip_loss=True):
    """
    Calculates instantaneous torque at each point, and calculate the average torque in each annulus over a rotation

    Inputs:
        V_i         <- Induced velocity on the flat point layout, shape (3, p) [axial, tangential, radial]
        params      <- Params object used in the optimisation loop, remains unmodified by function execution
        debug       <- Boolean to select whether or not to print debug information when function is called

    Outputs:
        dQ_momentum_ave w/ size (Nr, )                        <- Torque per annulus

    Assumes:
        |V_yz| < |V_x + V_ia|   <- Assumption that the annuli do not interact with each other

    Uses formulation from (M. Nahon et al., 2015) adapted for angular momentum
    """
    xp = ops(params.cfg)

    # Extract velocity components
    V_ia = xp.row(V_i, 0)  # shape (p,)
    V_it = xp.row(V_i, 1)  # shape (p,)

    # Momentum theory about annular element
    r_pt = station_to_points(params.r, params)
    V_x, V_yz = params.V_x, params.V_yz
    rho = params.rho

    V_R, phi = resultantVelocityComponents(V_i, params)

    if apply_tip_loss:
        tip_loss, root_loss = tip_root_loss_factors(phi, params)
        F = tip_loss * root_loss
        F = xp.clip(F, 0.05, 1.0)
    else:
        F = xp.ones_like(V_ia)   # no factor in the solve (Inputs.loss_in_solve = 0)

    # Compute baseline classical momentum element locally
    # Uses V_prime to account for skewed wake mass flux (Nahon 2015)
    V_prime = xp.sqrt((V_x + V_ia) ** 2 + V_yz ** 2)

    # dQ element per radian (omitting dr to match dT_momentum format)
    dQ_element_classical = 2 * rho * ((params.scale * r_pt) ** 2) * V_it * V_prime * F

    # Integrate over one full turn of psi to get the torque per annulus
    dQ_momentum_ave = 2 * jnp.pi * azimuth_mean(dQ_element_classical, params)

    if debug:
        print("------dQ_momentum DEBUG------")
        print("dQ Shapes (ave.):", jnp.shape(dQ_momentum_ave))
        print("V_ia shape:", jnp.shape(V_ia))
        print("V_it shape:", jnp.shape(V_it))
        print("F shape:", jnp.shape(F))
        print("------dQ_momentum DEBUG------")

    return dQ_momentum_ave

def thrust_residual(V_i, params):
    """
    Calculates the 1D thrust residual (BEMT vs Momentum) across radial stops.
    Returns array of shape (n_r,)
    """
    dT_BEMT_avg = dT_blade(V_i, params, debug=False)
    # Inputs.loss_in_solve selects where the Prandtl factors apply: 0 after the solve, 1 in the momentum side of the solve
    dT_MMNT_avg = dT_momentum(V_i, params, debug=False, apply_tip_loss=params.cfg.loss_in_solve == 1)
    return dT_BEMT_avg - dT_MMNT_avg

def torque_residual(V_i, params):
    """
    Calculates the 1D torque residual (BEMT vs Momentum) across radial stops.
    Returns array of shape (n_r,)
    """
    dQ_BEMT_avg = dQ_blade(V_i, params, debug=False)
    dQ_MMNT_avg = dQ_momentum(V_i, params, debug=False, apply_tip_loss=params.cfg.loss_in_solve == 1)
    return dQ_BEMT_avg - dQ_MMNT_avg

def inflow_from_lambda(lambda_i_0_guess, params):
    """
    The induced velocity field V_i (3, p) of the non-dimensional inflow ratios lambda_i_0_guess (k, n_r), one row for
    each solved component. The components that are not solved are zero.
    """
    xp = ops(params.cfg)
    n_r = params.r.shape[0]

    # Retrieve scales and Re-dimensionalise
    V_scale = get_velocity_scales(params)
    V_i_0_guess_dim = lambda_i_0_guess * xp.matmul(V_scale, np.ones((1, n_r)))

    # Introduce zero for non-solved components to maintain shape (3, n_r)
    V_i_0_full = xp.matmul(np.eye(3)[:, list(params.cfg.solve_idx)], V_i_0_guess_dim)

    # Expand to the point layout
    return inflow_model(V_i_0_full, params)

def bemt_residuals(lambda_i_0_guess, params):
    """
    Evaluates the coupled BEMT residuals.
    lambda_i_0_guess: non-dimensional inflow ratios of shape (k, n_r), one row for each solved component
    (params.cfg.solve_idx). Returns the scaled residuals of the same shape.
    """
    xp = ops(params.cfg)
    solve_idx = params.cfg.solve_idx
    V_i = inflow_from_lambda(lambda_i_0_guess, params)

    # Axial Residual (Scale by Thrust Target); Tangential Residual (Scale by Approximate Ideal Torque, to make O(1))
    Q_ref = params.thrust_target * params.scale * params.cfg.R_tip
    rows = []
    for i in solve_idx:
        if i == 0:
            rows.append(thrust_residual(V_i, params) / params.thrust_target)
        elif i == 1:
            rows.append(torque_residual(V_i, params) / Q_ref)
        else:
            rows.append(xp.zeros_like(xp.row(lambda_i_0_guess, 0)))     # Radial Residual (Already O(1)) TODO:

    return xp.stack(rows, axis=0)

def get_velocity_scales(params):
    """
    Computes characteristic velocity scales for the 3 inflow components
    that non-dimensionalise the unknowns of the induced velocity solve.
    Returns: V_scale -> shape (k, 1) for the k solved components, for broadcasting against (k, n_r) arrays.
    """
    xp = ops(params.cfg)
    R_tip = params.scale * params.cfg.R_tip
    A_disk = jnp.pi * (R_tip ** 2)

    # Axial Scale: Mean Momentum Inflow
    v_h = xp.sqrt(params.thrust_target / (2.0 * params.rho * A_disk))
    V_ia_mean = -params.V_x / 2.0 + xp.sqrt((params.V_x / 2.0)**2 + v_h**2)

    # Tangential Scale: Ideal Mean Swirl
    P_ideal = params.thrust_target * (params.V_x + V_ia_mean)
    Q_ideal = P_ideal / params.omega
    V_it_mean = Q_ideal / (params.rho * A_disk * (params.V_x + V_ia_mean) * R_tip)

    # Radial Scale: Tip Speed (Centrifugal reference)
    V_tip = params.omega * R_tip

    # Only keep the components that are being solved for
    scales = (V_ia_mean, V_it_mean, V_tip)
    return xp.vstack_scalars([scales[i] for i in params.cfg.solve_idx])

def _station_blocks(linear_fn, shape):
    '''
    The (n_r, k, k) diagonal blocks of a linear map on (k, n_r) arrays whose stations are independent:
    one JVP per solved component gives that column of every station's block at once.
    '''
    k = shape[0]
    columns = [linear_fn(jnp.zeros(shape).at[c].set(1.0)) for c in range(k)]
    return jnp.transpose(jnp.stack(columns, axis=-1), (1, 0, 2))

def _station_solve(blocks, rhs, active):
    '''
    Solves every station's block system (rhs of shape (k, n_r)). The pseudo-inverse gives the minimum-norm solution
    when a block is rank deficient (e.g. a component whose residual is not modelled); inactive stations get zero.
    '''
    x = jnp.einsum("nij,jn->in", jnp.linalg.pinv(blocks), rhs)
    return jnp.where(active[None, :], x, 0.0)

FALLBACK_STATIONS = 4   # stations solved together in one pass of the bracketed fallback
MAX_HALVINGS = 10       # Newton step halvings before a station is handed to the bracketed fallback

def _station_subset(params, idx):
    '''
    Params holding only the radial stations idx. The BEMT residuals of a station depend on no other station, so
    the subset gives the same residuals for those stations. The point layout constants shrink to the first
    len(idx) stations: they are the same for any choice of stations.
    '''
    n_sub, n_psi = idx.shape[0], params.psi.shape[0]
    n_pt = n_sub * n_psi
    return eqx.tree_at(lambda p: (p.r, p.dr, p.c, p.beta, p.offsets, p.V_i_0, p.E_r, p.M_psi, p.psi_pt, p.W_cst), params,
                       (params.r[idx], params.dr[idx], params.c[idx], params.beta[idx], params.offsets[idx],
                        params.V_i_0[:, idx], params.E_r[:n_pt, :n_sub], params.M_psi[:n_sub, :n_pt],
                        params.psi_pt[:n_pt], params.W_cst[idx]))

def _bracketed_root(f, x0, aux0, todo, ftol, h0=0.25, max_expand=12, max_iter=100):
    '''
    Root of a scalar residual at every station in todo, kept bracketed. f(x) returns (residual, aux), each shaped
    like x (n,); aux carries a by-product of the evaluation (the inner root when root finders are nested).

    The bracket grows about x0 (half-width h0, doubled up to max_expand times) until f changes sign, then Illinois
    (modified regula falsi) iterations shrink it until |f| <= ftol or it reaches round-off width. Each step
    evaluates f once, vmapped over three points (x0 and x0 -/+ h while the bracket grows, the Illinois iterate
    after), so f is traced only once: nested root finders then compile to one copy of the residual rather than one
    per call site. A station with no sign change in the largest bracket, or not in todo, keeps x0 and aux0.
    Returns (x, aux).
    '''
    f3 = jax.vmap(f)
    sign_change = lambda fa, fb: jnp.isfinite(fa) & jnp.isfinite(fb) & (jnp.sign(fa) != jnp.sign(fb))
    EXPAND, ILLINOIS, DONE = 0, 1, 2

    def cond(s):
        return (s["n"] < max_expand + max_iter) & jnp.any(s["phase"] != DONE)

    def body(s):
        a, fa, b, fb, h = s["a"], s["fa"], s["b"], s["fb"], s["h"]
        expanding, iterating = s["phase"] == EXPAND, s["phase"] == ILLINOIS

        # Illinois iterate from the bracket; bisection if the secant leaves it
        x = (a * fb - b * fa) / (fb - fa)
        x = jnp.where(jnp.isfinite(x) & (x > jnp.minimum(a, b)) & (x < jnp.maximum(a, b)), x, 0.5 * (a + b))
        points = jnp.where(expanding[None, :], jnp.stack([x0, x0 - h, x0 + h]), jnp.stack([x, x, x]))
        values, aux = f3(points)

        # Growing bracket: stop at a root at x0, or at a sign change on either side
        f0, fl, fr = values
        at_root = expanding & jnp.isfinite(f0) & (jnp.abs(f0) <= ftol)
        left = expanding & ~at_root & sign_change(fl, f0)
        right = expanding & ~at_root & ~left & sign_change(f0, fr)
        exhausted = expanding & ~(at_root | left | right) & ((s["n"] + 1 >= max_expand) | ~jnp.isfinite(f0))
        new_a = jnp.where(left, x0 - h, jnp.where(right, x0, a))
        new_fa = jnp.where(left, fl, jnp.where(right, f0, fa))
        new_b = jnp.where(left | at_root, x0, jnp.where(right, x0 + h, b))
        new_fb = jnp.where(left | at_root, f0, jnp.where(right, fr, fb))
        new_aux = jnp.where(left | at_root, aux[0], jnp.where(right, aux[2], s["aux"]))

        # Illinois: b is the latest iterate; a is the retained end, whose residual is halved each time it is kept
        fx = values[0]
        valid = iterating & jnp.isfinite(fx)
        flip = jnp.sign(fx) != jnp.sign(fb)
        new_a = jnp.where(valid, jnp.where(flip, b, a), new_a)
        new_fa = jnp.where(valid, jnp.where(flip, fb, 0.5 * fa), new_fa)
        new_b = jnp.where(valid, x, new_b)
        new_fb = jnp.where(valid, fx, new_fb)
        new_aux = jnp.where(valid, aux[0], new_aux)
        width = jnp.abs(new_b - new_a) <= 4 * jnp.finfo(x.dtype).eps * (1.0 + jnp.abs(new_b))
        converged = valid & ((jnp.abs(new_fb) <= ftol) | width)

        phase = jnp.where(left | right, ILLINOIS, s["phase"])
        phase = jnp.where(at_root | exhausted | converged | (iterating & ~valid), DONE, phase)
        return dict(n=s["n"] + 1, phase=phase, found=s["found"] | at_root | left | right,
                    a=new_a, fa=new_fa, b=new_b, fb=new_fb, aux=new_aux,
                    h=jnp.where(expanding & ~(at_root | left | right), 2 * h, h))

    init = dict(n=0, phase=jnp.where(todo, EXPAND, DONE), found=jnp.zeros_like(todo),
                a=x0, fa=jnp.zeros_like(x0), b=x0, fb=jnp.zeros_like(x0), aux=aux0, h=jnp.full_like(x0, h0))
    s = jax.lax.while_loop(cond, body, init)
    return jnp.where(s["found"], s["b"], x0), jnp.where(s["found"], s["aux"], aux0)

def _bracketed_bemt(residual_fn, lam, todo, tol, solve_idx):
    '''
    Fallback for the stations in todo where Newton stalls (a stationary point of the residual norm that is not a
    root, typically on the post-stall side of the lift curve, where the residual folds). The axial and tangential
    residuals are solved as nested scalar roots, each kept bracketed (_bracketed_root): for a trial axial inflow the
    tangential one balances the torque, and the axial inflow is then moved to balance the thrust. Both residuals
    change sign between large negative and large positive induced velocity (the drag terms grow as V^2 while the
    momentum terms grow as V|V|), so a bracket exists; the nearest root to the stalled point is taken.
    '''
    ia = solve_idx.index(0) if 0 in solve_idx else None
    it = solve_idx.index(1) if 1 in solve_idx else None
    if ia is None and it is None:
        return lam      # only the radial component, whose residual is not modelled

    if ia is None or it is None:
        row = it if ia is None else ia
        f = lambda x: (residual_fn(lam.at[row].set(x))[row], x)
        x, _ = _bracketed_root(f, lam[row], lam[row], todo, 0.5 * tol)
        return lam.at[row].set(x)

    # Nested: the torque balance to a tighter tolerance, so the thrust residual is smooth in the axial inflow. The
    # inner root finder carries the thrust residual as its by-product (NaN where it finds no torque root, which the
    # outer one then treats as unusable), so the residual is evaluated in one place only
    with_rows = lambda a, t: lam.at[ia].set(a).at[it].set(t)
    def thrust_residual(a):
        torque_and_thrust = lambda t: (lambda r: (r[it], r[ia]))(residual_fn(with_rows(a, t)))
        t, thrust = _bracketed_root(torque_and_thrust, lam[it], jnp.full_like(a, jnp.nan), todo, 0.1 * tol)
        return thrust, t
    a, t = _bracketed_root(thrust_residual, lam[ia], lam[it], todo, 0.5 * tol)
    return with_rows(a, t)

def _newton_bemt(residual_fn, lam, active, tol, max_iter, verbose, params):
    '''
    Damped blockwise Newton on residual_fn(lam) = 0. Each station backtracks (halving its step) until its own
    residual norm decreases, so a hard station never slows the others. A station already within tol takes no
    step: at round-off level its residual cannot decrease, and backtracking would run to machine epsilon.

    Near a regular root the full Newton step is accepted. A station whose residual norm does not decrease within
    MAX_HALVINGS halvings of its step is heading into a stationary point that is not a root (a fold of the residual
    past stall, or a reversed-flow state near the hub in forward flight), where Newton creeps with ever smaller
    steps and never arrives. The station stops there and, once the other stations converge, is solved by the bracketed
    fallback (_bracketed_bemt), as is any station still above tol after max_iter. The fallback runs only when a
    station needs it. Converged when the largest scaled residual over the active stations is below tol.
    '''
    def station_residual(l):
        # Non-finite residuals count as unconverged (a NaN would otherwise end the loop as if converged)
        res = jnp.abs(residual_fn(l))
        return jnp.max(jnp.where(jnp.isfinite(res), res, jnp.inf), axis=0)

    def newton_step(state):
        l, it, _, stuck = state
        r, linear_fn = jax.linearize(residual_fn, l)
        step = _station_solve(_station_blocks(linear_fn, l.shape), -r, active)

        # A station whose Newton step is undefined (non-finite residual or derivative, e.g. far outside the airfoil
        # model's range) restarts from the momentum-theory guess, lambda = 1, rather than spreading NaN
        restart = active & ~stuck & ~jnp.all(jnp.isfinite(step), axis=0)
        step = jnp.where(restart[None, :], 0.0, step)
        r = jnp.where(restart[None, :], 0.0, r)
        done = jnp.all(jnp.abs(r) <= tol, axis=0) | stuck
        step = jnp.where(done[None, :], 0.0, step)
        r0 = jnp.sum(r**2, axis=0)

        def backtrack_cond(carry):
            _, accepted, n = carry
            return (~jnp.all(accepted)) & (n < MAX_HALVINGS)
        def backtrack(carry):
            alpha, accepted, n = carry
            decreased = jnp.sum(residual_fn(l + alpha * step)**2, axis=0) < r0
            accepted = accepted | decreased
            return jnp.where(accepted, alpha, 0.5 * alpha), accepted, n + 1
        alpha, accepted, _ = jax.lax.while_loop(backtrack_cond, backtrack, (jnp.ones(l.shape[1]), done | ~active, 0))

        stuck = stuck | ~accepted
        l = jnp.where(restart[None, :], 1.0, jnp.where(stuck[None, :], l, l + alpha * step))
        err = jnp.max(jnp.where(active & ~stuck, station_residual(l), 0.0))
        if verbose >= 2:
            jax.debug.print("    [BEMT] Newton iter {i:3d} | residual {e:.2e} | smallest step fraction {a:.2e} | stalled {s}",
                            i=it + 1, e=err, a=jnp.min(jnp.where(active & ~done, alpha, 1.0)), s=jnp.sum(stuck))
        return l, it + 1, err, stuck

    def not_converged(state):
        _, it, err, _ = state
        return (it < max_iter) & (err > tol)

    no_station = jnp.zeros_like(active)
    err0 = jnp.max(jnp.where(active, station_residual(lam), 0.0))
    l, it, _, stuck = jax.lax.while_loop(not_converged, newton_step, (lam, 0, err0, no_station))

    # Bracketed fallback on the stations Newton could not solve, FALLBACK_STATIONS at a time: the stations are
    # independent, so each pass solves a small copy of the blade holding only those stations, at a fraction of the
    # cost of the full grid. A while_loop rather than lax.cond, which under vmap would run for every condition.
    todo = active & (stuck | (station_residual(l) > tol))
    n_sub = min(FALLBACK_STATIONS, l.shape[1])
    def fallback(carry):
        l, remaining = carry
        idx = jnp.argsort(~remaining)[:n_sub]          # remaining stations first
        chosen = remaining[idx]
        sub = _station_subset(params, idx)
        sub_residual = lambda l_sub: bemt_residuals(l_sub, sub)
        l_sub = _bracketed_bemt(sub_residual, l[:, idx], chosen, tol, params.cfg.solve_idx)
        worst_new, worst_old = jax.vmap(lambda l_sub: jnp.max(jnp.abs(sub_residual(l_sub)), axis=0))(
            jnp.stack([l_sub, l[:, idx]]))
        better = chosen & (worst_new < worst_old)
        l = l.at[:, idx].set(jnp.where(better[None, :], l_sub, l[:, idx]))
        return l, remaining.at[idx].set(False)
    l, _ = jax.lax.while_loop(lambda carry: jnp.any(carry[1]), fallback, (l, todo))

    if verbose >= 1:
        err = jnp.max(jnp.where(active, station_residual(l), 0.0))
        jax.debug.print("  [BEMT] T_target {t:7.2f} N | omega {w:7.1f} rad/s | Newton iters {n:3d} | "
                        "bracketed stations {b:2d} | residual {e:.2e}",
                        t=params.thrust_target, w=params.omega, n=it, b=jnp.sum(todo), e=err)
    return l

def aero_zone(params):
    '''
    Mask of the radial stations in the aerodynamic zone (r >= r_aero). The blend zone inboard of it is a structural
    transition whose chord and twist follow from the hub and the first aero station: it is not solved and carries
    no force, so it cannot drive the first aero station's design.
    '''
    return params.r >= params.cfg.R_aero

def radial_weights(params):
    '''
    Trapezoid weights over the aerodynamic zone: sum(w * f) integrates f from the first aero station to the tip,
    and blend-zone stations get zero weight.
    '''
    segment = jnp.diff(params.r) * aero_zone(params)[:-1]
    zero = jnp.zeros(1)
    return (jnp.concatenate([zero, segment]) + jnp.concatenate([segment, zero])) / 2

@jit
def induced_velocity(params):
    '''
    Calculates the coupled 3D induced velocity field using BEMT.

    Solves the non-dimensional residuals (bemt_residuals) for the solved components with blockwise Newton, from
    params.V_i_0 as the initial guess. jax.lax.custom_root makes derivatives of the solution exact by implicit
    differentiation, so gradients do not depend on the iteration history. Only the aerodynamic zone is solved
    (aero_zone): blend-zone stations keep the initial guess and carry no thrust or torque.
    '''
    cfg = params.cfg
    V_scale = get_velocity_scales(params)
    lam_0 = params.V_i_0[cfg.solve_rows] / V_scale
    active = aero_zone(params)

    residual_fn = lambda lam: bemt_residuals(lam, params)
    solve = lambda f, guess: _newton_bemt(f, guess, active, cfg.V_i_0_eps, cfg.V_i_0_max_iter, cfg.V_i_0_verbose, params)
    tangent_solve = lambda linear_fn, rhs: _station_solve(_station_blocks(linear_fn, rhs.shape), rhs, active)
    lam = jax.lax.custom_root(residual_fn, lam_0, solve, tangent_solve)

    # Re-dimensionalise, with zeros for the components that are not solved
    V_i_0 = jnp.zeros_like(params.V_i_0).at[cfg.solve_rows].set(lam * V_scale)

    # Expand to 3D inflow grid
    V_i = inflow_model(V_i_0, params)

    return V_i, V_i_0

def torque(V_i, params, debug=False):
    """
    Calculates the total torque on the propeller by integrating the average torque per annulus per rotation
    along the radial direction
    
    Inputs:
        V_i        <- Induced velocity at each radial and azimuthal station
        params      <- Params object used in the optimisation loop, remains unmodified by function execution
        debug       <- Boolean to select whether or not to print debug information when function is called
        apply_tip_loss <- Boolean to select whether or not to apply tip loss correction when calculating torque

    Outputs:
        Q <- Net torque per rotation of the propeller
    
    Assumes: 
        |V_yz| < |V_x + V_ia|   <- Assumption that the annuli do not interact with each other 
    """
    
    # Get the average torque per rotation in each annulus using the dQ_blade function
    dQ_blade_ave = dQ_momentum(V_i, params, debug, apply_tip_loss=True)   # the loads always carry the factor

    # Use trapezoidal integration over the aerodynamic zone to compute the total torque
    xp = ops(params.cfg)
    Q = params.scale * xp.sum(dQ_blade_ave * xp.const(radial_weights(params)))

    return Q

def thrust(V_i, params):
    '''
    Calculate total thrust from solved induced velocity
    '''
    dT = dT_momentum(V_i, params, apply_tip_loss=True)   # the loads always carry the factor

    # Numerically Compute Double Integral (over the aerodynamic zone) to Solve for Thrust
    xp = ops(params.cfg)
    T = params.scale * xp.sum(dT * xp.const(radial_weights(params)))

    return T

class SolvedLoads(NamedTuple):
    """
    Loads of one BEMT solve, with the unit in each name (nd: dimensionless).
    """
    thrust_N: jnp.ndarray
    torque_Nm: jnp.ndarray
    residual_nd: jnp.ndarray        # largest scaled BEMT residual over the aerodynamic zone
    V_i_0_m_s: jnp.ndarray          # solved induced velocity (3, n_r)

@jit
def _solved_loads(params) -> SolvedLoads:
    V_i, V_i_0 = induced_velocity(params)
    lam = V_i_0[params.cfg.solve_rows] / get_velocity_scales(params)
    residual = jnp.max(jnp.where(aero_zone(params), jnp.abs(bemt_residuals(lam, params)), 0.0))
    return SolvedLoads(jnp.squeeze(thrust(V_i, params)), jnp.squeeze(torque(V_i, params)), residual, V_i_0)

def loads_with_loss_in_solve(params) -> SolvedLoads:
    """
    Post-processing: the loads of a solved state with the Prandtl factors in the BEMT solve, for any value of
    Inputs.loss_in_solve in the design. params.V_i_0 is the start of the solve. The design and params do not change.
    The factors follow tip_loss_model and root_loss_model. JAX only, for one flight condition (Propeller.condition).
    """
    return _solved_loads(dataclasses.replace(params, cfg=dataclasses.replace(params.cfg, loss_in_solve=1)))

def mechanicalPower(V_i, params):
    '''
    Calculates and returns the shaft power (W) applied to the propeller 
    '''

    # Unpack Params
    omega = params.omega
    # Calculate Local Axial Flow Velocity

    # net torque over one rotation
    Q = torque(V_i, params, debug=False)

    # Net mechanical power (omega in rad/s, torque in Nm)
    P = Q * omega

    return jnp.squeeze(P)

def actuator_disk_power(T, params):
    '''
    Calculate Ideal Actuator Disk Power for same thrust
    Provides theoretical local minimum power using actuator disc model
    T is from momentum theory, 
    '''
    rho = params.rho
    R   = params.scale * params.cfg.R_tip
    V_x = params.V_x # freestream velocity

    A = jnp.pi*R**2 # disc area
    v_induced = ops(params.cfg).sqrt(T/(2*rho*A) + V_x**2/4) - V_x/2  # induced velocity through actuator disc
    P_act = T*(V_x+v_induced) # Ideal actuator disk power
    return jnp.squeeze(P_act)

def apc_model(filepath, plot=True):
    '''
    Loads an APC propeller geometry from a .PE0 file, interpolates it onto the
    solver's computational grid, solves the induced velocity, and plots the results.

    Returns (prop, params, V_i_0, T) where prop is the Propeller built from the APC file.
    '''
    import queue
    from . import results
    from matplotlib.animation import FuncAnimation

    print(f"Loading APC Propeller geometry from: {filepath}")
    prop = Propeller.from_apc(filepath)
    params = prop.condition(0)

    # Solve Induced Velocity
    V_i, V_i_0 = induced_velocity(params)
    
    # Update params with solved state
    params = eqx.tree_at(lambda p: p.V_i_0, params, V_i_0)

    # Calculate overall performance metrics
    T = thrust(V_i, params)
    P_mech = mechanicalPower(V_i, params)

    # Plotting via LivePropPlotter
    if plot:
        print("Launching LivePropPlotter canvas...")
        gui_queue = queue.Queue()
        plotter = results.LivePropPlotter(params)

        gui_queue.put(params)
        
        ani = FuncAnimation(
            plotter.fig, 
            plotter.update, 
            fargs=(gui_queue,), 
            interval=200, 
            cache_frame_data=False,
            blit=False
        )

        import matplotlib.pyplot as plt
        plt.show()

        fig = results.debug_diagnostics_plot(params, 0)

        dummy = input("Done with plots? \n")

    return prop, params, V_i_0, T

if __name__ == "__main__":
    prop, params, V_i_0, T = apc_model("apc_prop_data/propgeom_data/20x8-PERF.PE0", plot=True)
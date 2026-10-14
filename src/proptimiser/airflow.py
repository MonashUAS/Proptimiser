# external imports
from ._jax import jnp

# internal imports
from . import geometry
from .backend import ops

##  Dimensionless Constants
def calc_mach(V, a):
    '''
    Computes the Mach number as the ratio of velocity to speed of sound.
    '''
    return V / a

def calc_reynolds(rho, V, c, mu):
    '''
    Computes the Reynolds number for fluid flow.
    '''
    return (rho * V * c) / mu

def equivelantActuatorDiskEfficiency(T, params):
    '''
    Computes equivalent actuator disk efficiency across all regimes (hover to forward flight).

    Uses general momentum theory to solve for induced velocity v_i,
    avoiding division by zero at V_x = 0.
    '''
    xp = ops(params.cfg)
    R = params.cfg.R_tip
    A_disk = jnp.pi * (R**2)

    # Ideal hover induced velocity: v_0 = sqrt(T / (2 * rho * A))
    v_0 = xp.sqrt(T / (2.0 * params.rho * A_disk))

    # Free-stream velocity
    V_x = params.V_x

    # Solve for total axial velocity at disk
    v_i = xp.sqrt((V_x / 2.0)**2 + v_0**2) - (V_x / 2.0)

    # Ideal Power required by actuator disk
    P_ideal = T * (V_x + v_i)

    # At V_x = 0 (Hover), we return Figure of Merit (v_0 / v_i = 1.0 for ideal actuator disk)
    # At V_x > 0, we return classic propulsive efficiency (T * V_x / P_ideal)

    # Smooth blending or non-zero efficiency formulation:
    eta_actuator_disk = xp.where(
        V_x < 1e-5,
        1.0,  # Ideal actuator disk efficiency at hover (FM = 1.0)
        (T * V_x) / P_ideal
    )

    return eta_actuator_disk

def section_frame_of(params):
    """
    geometry.section_frame of the stations of params, as constants of the backend array type.
    """
    xp = ops(params.cfg)
    return geometry.SectionFrame(*(xp.const(v) for v in geometry.section_frame(params.r, params.offsets)))

def station_to_points(x, params):
    '''
    Expands a per-station quantity (n_r,) to the flat point layout (p,), p = n_r * n_psi (see geometry.point_layout).
    '''
    return ops(params.cfg).matmul(params.E_r, x)

def to_grid(x, params):
    """
    Reshapes the last axis of a field on the flat point layout (p,) to the (n_r, n_psi) grid. JAX only: the
    CasADi backend keeps the flat layout.
    """
    return x.reshape(x.shape[:-1] + (params.r.shape[0], params.psi.shape[0]))

def swept_section_velocity(V_axial_net, V_norml_net, V_radial_net, params):
    '''
    Relative air velocity in the plane normal to the span of a swept blade, on the section axes of
    geometry.section_frame: returns (u_n, u_c), the components along the normal n and the chord direction c.

    The inputs are the components of the relative air velocity on the air basis (rho, t, a) at the polar radius of
    each point: V_radial_net outward in the disc plane (the radial in-plane inflow, which an unswept blade
    ignores), V_norml_net opposite to the blade motion (omega r + in-plane inflow - swirl) and V_axial_net downstream.
    The component along the span is dropped (independence principle): only the velocity normal to the
    pitch axis makes lift. Unswept: (u_n, u_c) = (V_axial_net, V_norml_net).
    '''
    frame = section_frame_of(params)
    pt = lambda x: station_to_points(x, params)
    u_c = pt(frame.c_rho) * V_radial_net + pt(frame.c_t) * V_norml_net + pt(frame.c_a) * V_axial_net
    u_n = pt(frame.n_rho) * V_radial_net + pt(frame.n_a) * V_axial_net
    return u_n, u_c

def resultantVelocityComponents(V_i, params, plotting=False):
    '''
    Calculates the magnitude of the relative velocity at each point of the propeller, and the inflow angle phi

    As per equation (2) in "A propeller model for general forward flight conditions" [Waqas Khan and Meyer Nolan 2015]

    V_i has shape (3, p) on the flat point layout: [axial, tangential, radial] induced velocities at each point.
    V_R and phi have shape (p,).

    A swept blade (params.cfg.swept) takes the velocity normal to its span instead (swept_section_velocity). Its
    grid azimuth psi is that of the pitch-axis point, so the in-plane inflow needs no shift.

    Implemented here for purposes of reducing repeated code (helps with debugging!)

    TODO Debug plot implementation
    '''
    xp = ops(params.cfg)

    # Unpack params
    V_x, omega, V_yz = params.V_x, params.omega, params.V_yz
    r_pt, psi_pt = station_to_points(params.r, params), params.psi_pt

    # Extract the induced velocity components from the state vector
    V_i_a = xp.row(V_i, 0) # Axial induced velocity
    V_i_t = xp.row(V_i, 1) # Tangential induced velocity
    V_i_r = xp.row(V_i, 2) # Radial induced velocity (unused in 2D section V_R and phi)

    # Calculate the normal component of V_yz from the total V_yz
    V_yz_perp = V_yz * xp.sin(psi_pt)

    # Calculate V_R from the magnitude of the vector sum of the individual components
    V_axial_net = V_x + V_i_a

    # Swirl V_i_t > 0 turns with the blade (the momentum torque 2 rho r^2 V_i_t V' is then positive), so it reduces
    # the tangential velocity the blade sees: W_t = omega r (1 - a') for a propeller. With + V_i_t a drag-free blade
    # would absorb less shaft power than the axial momentum flux it produces, which breaks energy conservation.
    V_norml_net = omega * r_pt + V_yz_perp - V_i_t

    if params.cfg.swept:
        V_axial_net, V_norml_net = swept_section_velocity(V_axial_net, V_norml_net, V_yz * xp.cos(psi_pt), params)

    V_R = xp.sqrt(V_axial_net ** 2 + V_norml_net ** 2)

    # Four-quadrant inflow angle: continuous where the tangential velocity changes sign (reversed flow)
    phi = xp.arctan2(V_axial_net, V_norml_net)

    if plotting:
        raise(NotImplementedError("Debug plotting not currently supported by the 'resultantVelocityComponents' function (airflow.py)"))

    return V_R, phi

def simple_inflow(V_i_0, params):
    '''
    Applies a simple uniform inflow model across all azimuthal angles.

    Takes in V_i_0 of shape (3, N_r)
    Output: V_i of shape (3, p)
    '''
    return ops(params.cfg).matmul(V_i_0, params.E_r.T)

def pitt_peters(V_i_0, params):
    '''
    Pitt Peters inflow model for skewed inflow conditions.

    Takes in V_i_0 of shape (3, N_r)

    Output: V_i of shape (3, p)
    '''
    xp = ops(params.cfg)
    V_x, V_yz, psi_pt = params.V_x, params.V_yz, params.psi_pt
    R = params.cfg.R_tip
    r_pt = station_to_points(params.r, params)

    V_i_pt = simple_inflow(V_i_0, params)                       # shape (3, p)
    V_i_a, V_i_t, V_i_r = (xp.row(V_i_pt, i) for i in range(3))
    xi = xp.arctan2(V_yz, V_x + V_i_a)                      # shape (p,), local per point

    skew_factor = 1.0 + ((15.0*jnp.pi)/32.0) * xp.tan(xi/2.0) * (r_pt/R) * xp.cos(psi_pt)

    return xp.stack([V_i_a * skew_factor, V_i_t, V_i_r], axis=0)

def peters_he_model(V_i_0, params):
    '''
    Expands the induced velocity up to a 3D state vector, allowing complete modelling of the propeller.

    CURRENTLY NOT IMPLEMENTED, TOO EXPENSIVE. PITT_PETERS IS BEST MODEL FOR NOW.
    '''

    return simple_inflow(V_i_0, params)

def inflow_model(V_i_0: jnp.ndarray, params) -> jnp.ndarray:
    '''
    Selects and executes the configured inflow velocity calculation model.

    Args:
        V_i_0: Initial induced velocity array of shape (3, n_r).
        params: Propeller parameter structure containing flight state.

    Returns:
        Calculated induced velocity field of shape (3, p).
    '''
    return (simple_inflow, pitt_peters, peters_he_model)[params.cfg.inflow_model](V_i_0, params)


## Tip and Root Loss Models (0: No loss, 1: Prandtl)
def no_loss(phi, params):
    '''
    Returns a unity loss factor (no loss applied).
    Outputs shape matching phi: (p,)
    '''
    return ops(params.cfg).ones_like(phi)

def _smooth_floor(f, xp, eps=1e-2):
    '''
    Smooth, strictly positive stand-in for |f| in the Prandtl exponent, so d(arccos(exp(-f)))/df stays finite at f = 0.
    Leaves F within 1e-4 of the exact Prandtl factor for f > 0.3 and lets it fall to ~0.09 as f -> 0 (the exact
    limit is 0).
    '''
    return xp.sqrt(f**2 + eps**2)

def prandtl_tip_loss(phi, params):
    '''
    Calculates the Prandtl tip loss according to "Propeller blade element momentum theory with vortex wake deflection"
    M. K. Rwigema : https://www.icas.org/icas_archive/ICAS2010/PAPERS/434.PDF
    '''
    xp = ops(params.cfg)
    B = params.N
    r_pt = station_to_points(params.r, params)
    R_tip = params.cfg.R_tip * 1.05

    abs_phi = xp.abs(phi)
    sin_phi_safe = xp.maximum(xp.sin(abs_phi), 1e-3)   #guard low-phi (rotation-dominated) stations

    f_tip = (B / 2.0) * (R_tip - r_pt) / (r_pt * sin_phi_safe)

    f = xp.exp(-_smooth_floor(f_tip, xp))
    F_tip = (2.0 / jnp.pi) * xp.arccos(xp.clip(f, -1.0 + 1e-6, 1.0 - 1e-6))

    return F_tip

def prandtl_root_loss(phi, params):
    '''
    Calculates the Prandtl root loss according to "Propeller blade element momentum theory with vortex wake deflection"
    M. K. Rwigema : https://www.icas.org/icas_archive/ICAS2010/PAPERS/434.PDF
    '''
    xp = ops(params.cfg)
    B = params.N
    r_pt = station_to_points(params.r, params)
    # The wake vortex sheet starts at the inboard end of the loaded blade; the blend zone inboard of it carries no force
    R_root = params.cfg.R_aero

    abs_phi = xp.abs(phi)
    sin_phi_safe = xp.maximum(xp.sin(abs_phi), 1e-3)   #guard low-phi (rotation-dominated) stations

    f_root = (B / 2.0) * (r_pt - R_root) / (r_pt * sin_phi_safe)

    f = xp.exp(-_smooth_floor(f_root, xp))

    F_root = (2.0 / jnp.pi) * xp.arccos(xp.clip(f, -1.0 + 1e-6, 1.0 - 1e-6))

    return F_root


def tip_root_loss_factors(phi, params):
    '''
    Computes and returns both tip and root loss factors via model selection.

    Takes in phi of shape (p,)
    '''
    tip_loss = (no_loss, prandtl_tip_loss)[params.cfg.tip_loss_model](phi, params)
    root_loss = (no_loss, prandtl_root_loss)[params.cfg.root_loss_model](phi, params)

    return tip_loss, root_loss


## High Induction Correction Model (DEPRECIATED)
def no_high_induction(dT_classical, V_ia, F, params):
    '''
    Returns the classical momentum thrust element without modifications.
    '''
    return dT_classical

def buhl_high_induction(dT_classical, V_ia, F, params):
    '''
    Blends the classical thrust element with the Buhl (2005) high-induction
    modification when the local induction factor exceeds 0.4.
    '''
    xp = ops(params.cfg)
    r_pt = station_to_points(params.r, params)
    V_x, V_yz = params.V_x, params.V_yz
    rho = params.rho

    # Reference velocity for local induction factor
    V_inf = xp.sqrt(V_x**2 + V_yz**2)
    V_ref = xp.where(V_inf > 1e-3, V_inf, 1e-3)

    # Calculate local induction factor 'a'
    a = V_ia / V_ref

    # Buhl empirical heavily-loaded modification
    a_c = 0.4
    dT_buhl = 2 * rho * r_pt * (V_ref**2) * F * (a_c**2 + (4.0 - 2.0 * a_c) * a)

    # Element-wise conditional blending
    return xp.where(a <= a_c, dT_classical, dT_buhl)

def high_induction_correction(dT_classical, V_ia, F, params):
    '''
    Routes the classical thrust element through the configured model.

    DEPRECIATED
    '''
    return (no_high_induction, buhl_high_induction)[params.cfg.high_induction_model](dT_classical, V_ia, F, params)

if __name__ == "__main__":
    '''
    Debug/Testing of Functions for Airflow Models
    '''

# external imports
import jax.numpy as jnp
import jax
from jax import lax

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
    R = params.cfg.R_tip
    A_disk = jnp.pi * (R**2)
    
    # Ideal hover induced velocity: v_0 = sqrt(T / (2 * rho * A))
    v_0 = jnp.sqrt(T / (2.0 * params.rho * A_disk))
    
    # Free-stream velocity
    V_x = params.V_x
    
    # Solve for total axial velocity at disk
    v_i = jnp.sqrt((V_x / 2.0)**2 + v_0**2) - (V_x / 2.0)
    
    # Ideal Power required by actuator disk
    P_ideal = T * (V_x + v_i)
    
    # At V_x = 0 (Hover), we return Figure of Merit (v_0 / v_i = 1.0 for ideal actuator disk)
    # At V_x > 0, we return classic propulsive efficiency (T * V_x / P_ideal)
    
    # Smooth blending or non-zero efficiency formulation:
    eta_actuator_disk = jnp.where(
        V_x < 1e-5,
        1.0,  # Ideal actuator disk efficiency at hover (FM = 1.0)
        (T * V_x) / P_ideal
    )
    
    return eta_actuator_disk

def resultantVelocityComponents(V_i, params, plotting=False):
    '''
    Calculates the magnitude of the relative velocity at each station of the propeller, and the inflow angle phi
    
    As per equation (2) in "A propeller model for general forward flight conditions" [Waqas Khan and Meyer Nolan 2015]
    
    Updated to accept V_i (n_r, n_psi, 3) representing [axial, tangential, radial] induced velocities.
    
    Implemented here for purposes of reducing repeated code (helps with debugging!)

    TODO Debug plot implementation
    '''

    # Unpack params
    V_x, omega, r, V_yz, psi = params.V_x, params.omega, params.r, params.V_yz, params.psi

    # Extract the 3D induced velocity components from the state vector
    # V_i shape: [3, n_r, n_psi]
    V_i_a = V_i[0, :, :] # Axial induced velocity
    V_i_t = V_i[1, :, :] # Tangential induced velocity
    V_i_r = V_i[2, :, :] # Radial induced velocity (unused in 2D section V_R and phi)

    # Calculate the normal component of V_yz from the total V_yz
    V_yz_perp = V_yz * jnp.sin(psi[None, :])

    # Calculate V_R from the magnitude of the vector sum of the individual components
    V_axial_net = V_x + V_i_a
    
    # Swirl V_i_t > 0 turns with the blade (the momentum torque 2 rho r^2 V_i_t V' is then positive), so it reduces
    # the tangential velocity the blade sees: W_t = omega r (1 - a') for a propeller. With + V_i_t a drag-free blade
    # would absorb less shaft power than the axial momentum flux it produces, which breaks energy conservation.
    V_norml_net = omega * r[:, None] + V_yz_perp - V_i_t

    V_R = jnp.sqrt(jnp.square(V_axial_net) + jnp.square(V_norml_net))

    # Four-quadrant inflow angle: continuous where the tangential velocity changes sign (reversed flow)
    phi = jnp.arctan2(V_axial_net, V_norml_net)

    if plotting:
        raise(NotImplementedError("Debug plotting not currently supported by the 'resultantVelocityComponents' function (airflow.py)"))

    return V_R, phi

def simple_inflow(V_i_0, params):
    ''' 
    Applies a simple uniform inflow model across all azimuthal angles.

    Takes in V_i_0 of shape (3, N_r)
    Output: V_i of shape (3, N_r, N_psi)
    '''
    N_psi = params.psi.shape[0]

    # Zero-copy memory broadcast in JAX
    V_i = jnp.broadcast_to(V_i_0[:, :, None], (3, V_i_0.shape[1], N_psi))

    return V_i

def pitt_peters(V_i_0, params):
    '''
    Pitt Peters inflow model for skewed inflow conditions.

    Takes in V_i_0 of shape (3, N_r)

    Output: V_i of shape (3, N_r, N_psi)
    '''
    V_x, V_yz, r, psi = params.V_x, params.V_yz, params.r, params.psi
    R = params.cfg.R_tip

    V_ia_0 = V_i_0[0]
    xi = jnp.arctan2(V_yz, V_x + V_ia_0)          # shape (n_r,), local per station

    skew_factor = 1.0 + ((15.0*jnp.pi)/32.0) * jnp.tan(xi[:, None]/2.0) * (r[:, None]/R) * jnp.cos(psi[None, :])

    N_r, N_psi = V_i_0.shape[1], psi.shape[0]

    V_i_a = V_i_0[0:1, :, None] * skew_factor[None, :, :]                      # shape (1, N_r, N_psi)
    V_i_t = jnp.broadcast_to(V_i_0[1:2, :, None], (1, N_r, N_psi))             # shape (1, N_r, N_psi)
    V_i_r = jnp.broadcast_to(V_i_0[2:3, :, None], (1, N_r, N_psi))             # shape (1, N_r, N_psi)

    return jnp.concatenate([V_i_a, V_i_t, V_i_r], axis=0)

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
        V_i_0: Initial 3D induced velocity array of shape (3, n_r).
        params: Propeller parameter structure containing flight state.

    Returns:
        Calculated 3D induced velocity field of shape (3, n_r, n_psi).
    '''
    V_i = lax.switch(params.cfg.inflow_model, [simple_inflow, pitt_peters, peters_he_model], *(V_i_0, params))
    return V_i


## Tip and Root Loss Models (0: No loss, 1: Prandtl) 
def no_loss(phi, params):
    ''' 
    Returns a unity loss factor (no loss applied).
    Outputs shape matching phi: (N_r, N_psi)
    '''
    return jnp.ones_like(phi)

def prandtl_tip_loss(phi, params):
    ''' 
    Calculates the Prandtl tip loss according to "Propeller blade element momentum theory with vortex wake deflection"
    M. K. Rwigema : https://www.icas.org/icas_archive/ICAS2010/PAPERS/434.PDF
    '''
    eps = 1e-3
    B = params.N
    r = params.r
    R_tip = params.cfg.R_tip * 1.05

    abs_phi = jnp.abs(phi)
    sin_phi_safe = jnp.maximum(jnp.sin(abs_phi), 1e-3)   #guard low-phi (rotation-dominated) stations

    f_tip = (B / 2.0) * (R_tip - r[:, None]) / (r[:, None] * sin_phi_safe)

    # Soft-saturate f_tip itself (smooth floor) rather than hard-clipping exp(-f_tip).
    # This keeps d(F)/d(phi) finite AND continuous, instead of exactly zero past a hard cutoff.
    f_tip_safe = jax.nn.softplus(f_tip - eps) + eps   # smooth floor near f_tip -> 0, ~linear elsewhere

    f = jnp.exp(-f_tip_safe)
    F_tip = (2.0 / jnp.pi) * jnp.arccos(jnp.clip(f, -1.0 + 1e-6, 1.0 - 1e-6))

    return F_tip

def prandtl_root_loss(phi, params):
    ''' 
    Calculates the Prandtl root loss according to "Propeller blade element momentum theory with vortex wake deflection"
    M. K. Rwigema : https://www.icas.org/icas_archive/ICAS2010/PAPERS/434.PDF
    '''
    eps = 1e-2
    B = params.N
    r = params.r
    R_hub = params.cfg.R_hub / 2.0

    abs_phi = jnp.abs(phi)
    sin_phi_safe = jnp.maximum(jnp.sin(abs_phi), 1e-3)   #guard low-phi (rotation-dominated) stations


    f_root = (B / 2.0) * (r[:, None] - R_hub) / (r[:, None] * sin_phi_safe)
    
    # Clip to guarantee arccos operates within valid domain [-1, 1]
    f_root_safe = jax.nn.softplus(f_root - eps) + eps   # smooth floor near f_tip -> 0, ~linear elsewhere

    f = jnp.exp(-f_root_safe)

    F_root = (2.0 / jnp.pi) * jnp.arccos(jnp.clip(f, -1.0 + 1e-6, 1.0 - 1e-6))

    return F_root


def tip_root_loss_factors(phi, params):
    ''' 
    Computes and returns both tip and root loss factors via model selection.
    
    Takes in phi of shape (N_r, N_psi)
    '''
    tip_loss = lax.switch(params.cfg.tip_loss_model, [no_loss, prandtl_tip_loss], phi, params)
    root_loss = lax.switch(params.cfg.root_loss_model, [no_loss, prandtl_root_loss], phi, params)

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
    r = params.r[:, None]  # Broadcast to match (Nr, Npsi)
    V_x, V_yz = params.V_x, params.V_yz
    rho = params.rho
    
    # Reference velocity for local induction factor
    V_inf = jnp.sqrt(V_x**2 + V_yz**2)
    V_ref = jnp.where(V_inf > 1e-3, V_inf, 1e-3)
    
    # Calculate local induction factor 'a'
    a = V_ia / V_ref
    
    # Buhl empirical heavily-loaded modification
    a_c = 0.4
    dT_buhl = 2 * rho * r * (V_ref**2) * F * (a_c**2 + (4.0 - 2.0 * a_c) * a)
    
    # Element-wise conditional blending
    return jnp.where(a <= a_c, dT_classical, dT_buhl)

def high_induction_correction(dT_classical, V_ia, F, params):
    ''' 
    Routes the classical thrust element through lax.switch based on model selection.

    DEPRECIATED
    '''
    return lax.switch(
        params.cfg.high_induction_model,
        [no_high_induction, buhl_high_induction],
        dT_classical, V_ia, F, params
    )

if __name__ == "__main__":
    '''
    Debug/Testing of Functions for Airflow Models
    '''

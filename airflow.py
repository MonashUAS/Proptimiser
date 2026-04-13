from dataclasses import replace
import jax.numpy as jnp
from init import Inputs
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


def calc_phi(V_ia, params):
    ''' 
    Computes the inflow angle phi based on local velocity components.
    TODO FInish deprecation of this function
    '''
    raise(NotImplementedError("This function has been deprecated, please modify code to use the 'resultantVelocityComponents' function"))


##  Inflow Model Definitions (0: Simple, 1: Pitt-Peters, 2: DTU) 
def simple_inflow(V_ia_0, params):
    ''' 
    Applies a simple uniform inflow model across all azimuthal angles.

    Takes in V_ia_0 which is expected to be size N_r

    Output: 
        V_ia: Inflow velocity at each azimuthal and radial station (size N_r * N_psi)
    '''

    V_ia = V_ia_0 * jnp.ones_like(params.psi)

    return V_ia


def pitt_peters(V_ia_0, params):
    ''' 
    Computes skewed rotor inflow using the Pitt & Peters (1981) model.

    Takes in V_ia_0 which is expected to be size N_r

    Output: 
        V_ia: Inflow velocity at each azimuthal and radial station (size N_r * N_psi)
    '''
    V_x, V_yz, r, psi, R = params.V_x, params.V_yz, params.r, params.psi, params.R
    
    xi = jnp.arctan(V_yz / (V_x + V_ia_0))  # wake skew angle
    
    V_ia = V_ia_0 * (1 + ((15 * jnp.pi) / 32) * jnp.tan(xi / 2) * (r / R) * jnp.cos(psi))

    return V_ia


def dtu_model(V_ia_0, params):
    ''' 
    Placeholder for the DTU inflow model; currently defaults to uniform inflow (simple model).
    TODO Implement DTU model
    '''

    return simple_inflow(V_ia_0, params)


def inflow_model(V_ia_0, params):
    ''' 
    Switches between inflow models based on global configuration.
    '''

    V_ia_0 = V_ia_0[:,None]

    V_ia = lax.switch(Inputs.inflow_model, [simple_inflow, pitt_peters, dtu_model], *(V_ia_0, params))

    return V_ia


def resultantVelocityComponents(V_ia, params, plotting=False):
    '''
    Calculates the magnitude of the relative velocity at each station of the propeller, and the inflow angle phi
    
    As per equation (2) in "A propeller model for general forward flight conditions" [Waqas Khan and Meyer Nolan 2015]
    
    Implemented here for purposes of reducing repeated code (helps with debugging!)

    TODO Debug plot implementation
    '''

    # Unpack params
    V_x, omega, r, V_yz, psi = params.V_x, params.omega, params.r, params.V_yz, params.psi

    # Calculate the normal component of V_yz from the tocal V_yz
    V_yz_perp = V_yz * jnp.sin(psi)

    # Calculate V_R from the magnitude of the vector sum of the individual components
    V_axial_net = V_x + V_ia
    V_norml_net = omega*r + V_yz_perp
    V_R = jnp.sqrt(jnp.square(V_axial_net) + jnp.square(V_norml_net))

    # Using the same components we can efficiently calculate phi
    phi = jnp.arctan(V_axial_net / V_norml_net)

    if plotting:
        raise(NotImplementedError("Debug plotting not currently supported by the 'resultantVelocityComponents' function (airflow.py)"))

    return V_R, phi


##  Tip and Root Loss Models (0: No loss, 1: Prandtl) 
def no_loss(V_ia, params):
    ''' 
    Returns a unity loss factor (no loss applied).
    '''
    return jnp.ones_like(V_ia)


def prandtl_tip_loss(V_ia, params):
    ''' 
    calculates the prandtl tip loss according to "Propeler bade element momentum theory with vortex wake deflection"
    M. K. Rwigema : https://www.icas.org/icas_archive/ICAS2010/PAPERS/434.PDF

    Notation adapted from that source, with variables reallocated for consistency with literature
    '''

    eps = 1e-6
    B = params.N
    R = params.R
    r = params.r
    phi = jnp.abs(resultantVelocityComponents(V_ia, params)[1]) # get only second output

    f_tip = B/2 * (R - r) / (r * jnp.sin(phi)) 
    
    # exp(-f_root) can exceed 1.0 slightly due to floating point error, we 
    # need to clip this to guarantee that arccos functions as expected
    f = jnp.clip(jnp.exp(-f_tip), -1.0, 1.0)
    #f = f.at[len(f)].set(0)
    
    F_tip = (2/jnp.pi) * jnp.arccos(f)

    return F_tip


def prandtl_root_loss(V_ia, params):
    ''' 
    calculates the prandtl tip loss according to "Propeler bade element momentum theory with vortex wake deflection"
    M. K. Rwigema : https://www.icas.org/icas_archive/ICAS2010/PAPERS/434.PDF

    Notation adapted from that source, with variables reallocated for consistency with literature
    '''

    eps = 1e-6
    B = params.N
    R_hub = 0.05
    r = params.r
    phi = jnp.abs(resultantVelocityComponents(V_ia, params)[1]) # get only second output

    f_root = B/2 * (r - R_hub) / (r * jnp.sin(phi) + eps) 
    
    # exp(-f_root) can exceed 1.0 slightly due to floating point error, we 
    # need to clip this to guarantee that arccos functions as expected
    f = jnp.abs(jnp.clip(jnp.exp(-f_root), -1.0, 1.0))
    #f = f.at[1].set(0)

    F_root = (2/jnp.pi) * jnp.arccos(f)

    return F_root


def tip_root_loss_factors(V_ia, params):
    ''' 
    Computes and returns both tip and root loss factors via model selection.
    '''
    tip_loss = lax.switch(Inputs.tip_loss_model, [no_loss, prandtl_tip_loss],*(V_ia,params))
    root_loss = lax.switch(Inputs.root_loss_model, [no_loss, prandtl_root_loss], *(V_ia,params))
    return tip_loss, root_loss


def equivelantActuatorDiskEfficiency(T,params):
    '''
    Computes the equivalent actuator effiency for the given conditions, useful as a measure 
    of the optimality, with this effiency representing the theoretical maximum effiency (never attainable).
    '''
    massFlowRate = jnp.pi*(params.R)**2*params.rho*params.V_x #Mass flow rate (continuity eq.)
    actuatorDistWakeVelocity = T/massFlowRate - params.V_x
    equivActEfficiency = params.V_x/ (params.V_x + ((1/2) * (actuatorDistWakeVelocity + params.V_x)))
    return equivActEfficiency
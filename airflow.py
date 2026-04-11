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
    '''
    V_x, V_yz, omega, r, psi = params.V_x, params.V_yz, params.omega, params.r, params.psi
    V_yz_perp = V_yz * jnp.sin(psi)
    return jnp.arctan((V_x + V_ia) / (omega * r + V_yz_perp))

##  Inflow Model Definitions (0: Simple, 1: Pitt-Peters, 2: DTU) 
def simple(V_ia_0, params):
    ''' 
    Applies a simple uniform inflow model across all azimuthal angles.
    '''
    return jnp.tile(V_ia_0, (1, Inputs.n_psi))

def pitt_peters(V_ia_0, params):
    ''' 
    
    Computes skewed rotor inflow using the Pitt & Peters (1981) model.
    '''
    V_x, V_yz, r, psi, R = params.V_x, params.V_yz, params.r, params.psi, params.R
    xi = jnp.arctan(V_yz / (V_x + V_ia_0))  # wake skew angle
    return V_ia_0 * (1 + ((15 * jnp.pi) / 32) * jnp.tan(xi / 2) * (r / R) * jnp.cos(psi))

def dtu_model(V_ia_0, params):
    ''' 
    Placeholder for the DTU inflow model; currently defaults to uniform inflow.
    '''
    return jnp.tile(V_ia_0, (1, Inputs.n_psi))

def inflow_model(V_ia_0, params):
    ''' 
    Switches between inflow models based on global configuration.
    '''
    return lax.switch(Inputs.inflow_model, [simple, pitt_peters, dtu_model], V_ia_0, params)

##  Tip and Root Loss Models (0: No loss, 1: Prandtl) 
def no_loss(test):
    ''' 
    Returns a unity loss factor (no loss applied).
    '''
    V_ia,params = test
    return jnp.ones_like(V_ia)
def prandtl_tip_loss(test):
    eps = 1e-6
    V_ia, params = test
    f_tip = params.N/2 * (params.R - params.r) / (params.r * (jnp.abs(jnp.sin(calc_phi(V_ia, params))) + eps))
    F_tip = 2/jnp.pi * jnp.arccos(f_tip)
    return F_tip

def prandtl_root_loss(test):
    eps = 1e-6
    V_ia, params = test
    f_root = params.N/2 * (params.r - 0.05) / (0.05 * (jnp.abs(jnp.sin(calc_phi(V_ia, params))) + eps))
    
    # exp(-f_root) can exceed 1.0 slightly due to floating point
    arg = jnp.clip(jnp.exp(-f_root), -1.0, 1.0)
    F_root = 2/jnp.pi * jnp.arccos(arg)
    return F_root

def tip_root_loss_factors(V_ia,params):
    ''' 
    Computes and returns both tip and root loss factors via model selection.
    '''
    tip_loss = lax.switch(Inputs.tip_loss_model, [no_loss, prandtl_tip_loss],(V_ia,params))
    root_loss = lax.switch(Inputs.root_loss_model, [no_loss, prandtl_root_loss], (V_ia,params))
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
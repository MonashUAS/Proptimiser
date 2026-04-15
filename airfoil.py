### Establish Coefficients for Given Airfoil
import jax
from jax import numpy as jnp
from jax import lax
import matplotlib.pyplot as plt
from init import Inputs,Params
from airflow import resultantVelocityComponents

## ---Parametric Coefficients---
# (good for approximate analysis, use LUT for more accuracy)

def parametric_coeffs(V_ia, params):

    '''
    Return parametric estimations of lift, drag, and moment coefficients based on angle of attack

    Modelled on the ARAD10 airfoil

    Calculate effective angle of attack for the airfoil.

    For plots and a rundown of the theory, see (M. Nahon, 2015)
    '''
    
    # Constants:
    C_La = 5.7     # Lift curve slope per rad (ARAD10)
    C_D90 = 1.98    # Drag coefficient at 90 degrees angle of attack (flat plate)
    C_D0 = 0.0175    # Zero lift drag coefficient (ARA-D 10)
    AoA_0 = jnp.radians(10) # zero lift angle of attack

    # load relevant data from inputs
    geometric_angle = params.beta[:, None]

    # calculate aerodynamic incidence
    _, phi = resultantVelocityComponents(V_ia, params) # get only second output

    # Angle of attack
    AoA_eff = geometric_angle - phi + AoA_0

    # Do some conditioning of the effective AoA
    #AoA_eff = jnp.where(AoA_eff==jnp.pi/2, AoA_eff+jnp.radians(0.05), AoA_eff)
    #AoA_eff = jnp.where(AoA_eff==jnp.pi, AoA_eff-jnp.radians(0.05), AoA_eff)

    # Coefficients in non-stall region
    C_L_unstall = C_La * AoA_eff                                                            # Lift Coefficient in linear region
    C_T_unstall = C_D0 * jnp.cos(AoA_eff)                                                   # Tangential Force Coefficient
    C_N_unstall = (C_L_unstall + C_T_unstall*jnp.sin(AoA_eff)) / (jnp.cos(AoA_eff))  # Normal Force Coefficient
    C_D_unstall = C_N_unstall * jnp.sin(AoA_eff) + C_T_unstall * jnp.cos(AoA_eff)            # Drag Coefficient
    C_M_unstall = -C_N_unstall*(0.25-0.175*(1-2*AoA_eff/jnp.pi))                            # Moment Coefficient
    
    # Coefficients in stall region based on flat plate model
    C_N_stall = C_D90 * (jnp.sin(AoA_eff)) / (0.56+0.44*jnp.sin(AoA_eff))       # Normal Force Coefficient
    C_T_stall = 0.5*C_D0 *jnp.cos(AoA_eff)                                      # Tangential Force Coefficient
    C_L_stall = C_N_stall*jnp.cos(AoA_eff)-C_T_stall*jnp.sin(AoA_eff)           # Lift Coefficient
    C_D_stall = C_N_stall*jnp.sin(AoA_eff)+C_T_stall*jnp.cos(AoA_eff)           # Drag Coefficient
    C_M_stall = -C_N_stall*(0.25-0.175*(1-2*AoA_eff/jnp.pi))                   # Moment Coefficient

    # Set piecewise function between 20 and 160 degrees
    stall_low  = jnp.abs(AoA_eff) >= jnp.radians(20)
    stall_high = jnp.abs(AoA_eff) >  jnp.radians(160) # Don't model high stall angle, inverse flow is complex
    stall      = stall_low & ~stall_high   # 20°–160°
    
    # Piecewise coefficients (stall for 20°–160°)
    C_L = jnp.where(stall, C_L_stall, C_L_unstall)
    C_D = jnp.where(stall, C_D_stall, C_D_unstall)
    C_M = jnp.where(stall, C_M_stall, C_M_unstall)
    
    return C_L, C_D, C_M


def lookup_coeffs(V_ia, params):
    '''
    Placeholder for future lookup table implementation, 
    use parametric for now to allow compile
    '''
    #raise(NotImplementedError("FUnction 'lookup_coeffs' in airfoil.py is not implemented yet"))
    # Look up/estimate CL/CD from CFD data tables

    # Input parameters of alpha, Re, Ma, airfoil type. Return airfoil with suitable coefficients that minimise chord.
    return parametric_coeffs(V_ia, params)

def aero_coeffs(V_ia, params):
    # Select Method to Establish Aerodynamic Coefficients
    C_L, C_D, C_M = lax.switch(Inputs.coeff_method,
                    [parametric_coeffs, lookup_coeffs],
                    V_ia, params)
    return C_L, C_D, C_M


if __name__ == "__main__":
    # perform this import locally as we only need it for unit tests and could cause
    # a circular import error if loaded at the top of the file

    import propmodel as pmod
    # Geometry setup
    psi, dpsi = pmod.angular_stops(Inputs.n_psi)
    r, dr = pmod.radial_stops(0.225, 0.05, Inputs.n_r)

    init_c = jnp.full((len(r), 1), 0.01)
    init_beta = jnp.full((len(r), 1), jnp.radians(10))

    params_static = Params(
        N=2,
        R=jnp.array([0.35]),
        c=init_c,
        beta=init_beta,
        r=r[:, None], dr=dr,
        psi=psi[None, :], dpsi=dpsi,
        V_x=jnp.array([10]), V_yz=jnp.array([2]),
        omega=jnp.array([2*jnp.pi*8000/60]),
        V_ia_0=jnp.full_like(r[:, None], 14)
    )

    parametric_coeffs(jnp.full_like(r[:, None], 0), params_static, True)
### Establish Coefficients for Given Airfoil
import jax
from jax import numpy as jnp
from jax import lax
import matplotlib.pyplot as plt
from init import Inputs,Params
from airflow import resultantVelocityComponents

## ---Parametric Coefficients---
# (good for approximate analysis, use LUT for more accuracy)

def parametric_coeffs(V_ia, params, plot=False):

    '''
    Return parametric estimations of lift, drag, and moment coefficients based on angle of attack

    Modelled on the ARAD10 airfoil

    Calculate effective angle of attack for the airfoil.

    For plots and a rundown of the theory, see (M. Nahon, 2015)
    '''
    
    geometric_angle = params.beta

    _, phi = resultantVelocityComponents(V_ia, params) # get only second output
    
    # Coefficient parameters
    alphaabs = jax.lax.abs(geometric_angle)  # Use absolute value of alpha for symmetry

    # Set piecewise function between 20 and 160 degrees
    stall_low  = alphaabs >= jnp.radians(20)
    stall_high = alphaabs >  jnp.radians(180) # Don't model high stall angle, inverse flow is complex
    stall      = stall_low & ~stall_high   # 20°–160°
    
    C_La = 5.7     # Lift curve slope per rad (ARAD10)
    C_D90 = 1.98    # Drag coefficient at 90 degrees angle of attack (flat plate)
    C_D0 = 0.035    # Zero lift drag coefficient (ARAD10)

    # Coefficients in non-stall region
    C_L_unstall = C_La * alphaabs                                                          # Lift Coefficient
    C_T_unstall = C_D0 * jnp.cos(alphaabs)
    C_L_unstall = jnp.minimum(C_La * alphaabs, 1.2) # 1.2 is a typical max Cl for thin airfoils                                                 # Tangential Force Coefficient
    C_N_unstall = (C_L_unstall + C_T_unstall*jnp.sin(alphaabs)) / (jnp.cos(alphaabs))      # Normal Force Coefficient
    C_D_unstall = C_N_unstall * jnp.sin(alphaabs) + C_T_unstall* jnp.cos(alphaabs)         # Drag Coefficient
    C_M_unstall = -C_N_unstall*(0.25-0.175*(1-2*alphaabs/jnp.pi))                          # Moment Coefficient
    
    # Coefficients in stall region based on flat plate model
    C_N_stall = C_D90 * (jnp.sin(alphaabs)) / (0.56+0.44*jnp.sin(alphaabs))          # Normal Force Coefficient
    C_T_stall = 0.5*C_D0 *jnp.cos(alphaabs)                                                # Tangential Force Coefficient
    C_L_stall = C_N_stall*jnp.cos(alphaabs)-C_T_stall*jnp.sin(alphaabs)                       # Lift Coefficient
    C_D_stall = C_N_stall*jnp.sin(alphaabs)+C_T_stall*jnp.cos(alphaabs)                       # Drag Coefficient
    C_M_stall = -C_N_stall*(0.25-0.175*(1-2*alphaabs/jnp.pi))                              # Moment Coefficient

    # Piecewise coefficients (stall for 20°–160°)
    C_L = jnp.sign(geometric_angle)*jnp.where(stall, C_L_stall, C_L_unstall)
    C_D = jnp.where(stall, C_D_stall, C_D_unstall)
    C_M = jnp.sign(geometric_angle)*jnp.where(stall, C_M_stall, C_M_unstall)
    
    return C_L, C_D, C_M


def lookup_coeffs(V_ia, params):
    '''
    Placeholder for future lookup table implementation, 
    use parametric for now to allow compile
    '''
    C_L, C_D, C_M  = parametric_coeffs(V_ia, params)
    # Look up/estimate CL/CD from CFD data tables

    # Input parameters of alpha, Re, Ma, airfoil type. Return airfoil with suitable coefficients that minimise chord.


    return C_L, C_D, C_M

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
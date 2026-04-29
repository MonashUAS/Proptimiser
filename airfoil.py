### Establish Coefficients for Given Airfoil
import jax
from jax import numpy as jnp
from jax import lax
import jax.scipy as jsp
import matplotlib.pyplot as plt
from init import Inputs,Params
from airflow import resultantVelocityComponents
import numpy as np

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
    AoA_0 = jnp.radians(-10) # zero lift angle of attack

    # load relevant data from inputs
    geometric_angle = params.beta[:, None]

    # calculate aerodynamic incidence
    _, phi = resultantVelocityComponents(V_ia, params) # get only second output

    # Angle of attack
    AoA_eff = geometric_angle - phi - AoA_0

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
    # Unpack params
    V_x, omega, r, V_yz, psi = params.V_x, params.omega, params.r, params.V_yz, params.psi
    
    # Constants:
    C_D90 = 1.98    # Drag coefficient at 90 degrees angle of attack (flat plate)
    C_D0 = 0.0175    # Zero lift drag coefficient (ARA-D 10)
    AoA_0 = jnp.radians(-5) # zero lift angle of attack

    # load relevant data from inputs
    geometric_angle = params.beta[:, None]

    # load interpolators
    clInterp = params.clInterpolator
    cdInterp = params.cdInterpolator

    # calculate aerodynamic incidence
    _, phi = resultantVelocityComponents(V_ia, params) # get only second output

    # Angle of attack
    AoA_eff = geometric_angle - phi - AoA_0

    # Calculate the normal component of V_yz from the tocal V_yz
    V_yz_perp = V_yz * jnp.sin(psi)

    # Calculate V_R from the magnitude of the vector sum of the individual components
    V_axial_net = V_x + V_ia
    V_norml_net = omega*r + V_yz_perp
    V_R = jnp.sqrt(jnp.square(V_axial_net) + jnp.square(V_norml_net))

    # calculate the reynolds and mach numbers
    Re = V_R * params.c[:, None] * params.rho / params.mu
    Ma = V_R / params.v_sound

    # Coefficients in non-stall region
    flowParams = jnp.array([jnp.ravel(AoA_eff),jnp.ravel(Re),jnp.ravel(Ma)])
    #flowParams = jnp.transpose(flowParams)
    pts = jnp.array([5, 200000, 0.2])
    C_L_unstall = clInterp((5, 200000, 0.3))
    C_D_unstall = C_L_unstall
    
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
    C_M = C_M_stall
    
    return C_L, C_D, C_M


def aero_coeffs(V_ia, params):
    # Select Method to Establish Aerodynamic Coefficients
    C_L, C_D, C_M = lax.switch(Inputs.coeff_method,
                    [parametric_coeffs, lookup_coeffs],
                    V_ia, params)
    return C_L, C_D, C_M


def loadAirfoilData(filename: str, shape: np.ndarray = np.array([81, 14, 6]), method:str='linear'):
    """
    Inputs:
    
    Outputs:

    Function description
    """
    array = np.loadtxt(filename, dtype='float', delimiter=',')

    numbloks = 5    
    blocksize = int(np.shape(array)[0]/numbloks)

    # Load the blocks of data from the input file
    alp = array[0           :   blocksize,      :]
    rey = array[blocksize   :   blocksize*2,    :]
    mac = array[blocksize*2 :   blocksize*3,    :]
    cl  = array[blocksize*3 :   blocksize*4,    :]
    cd  = array[blocksize*4 :   blocksize*5,    :]

    # Reshape the blocks of data from 2D to 3D, breaking up only the last two dims:
    alp = jnp.reshape(alp, shape)
    rey = jnp.reshape(rey, shape)
    mac = jnp.reshape(mac, shape)
    cl  = jnp.reshape(cl, shape)
    cd  = jnp.reshape(cd, shape)
    
    # Generate a 3D scipy style interpolator object which can be called like a function to interpolate input points
    cl_interpolator = jsp.interpolate.RegularGridInterpolator((alp, rey, mac), cl, method=method)
    cd_interpolator = jsp.interpolate.RegularGridInterpolator((alp, rey, mac), cd, method=method)

    test = cl_interpolator(jnp.array([5, 200000, 0.3]), "linear")
    test = cd_interpolator((5, 200000, 0.3), "linear")

    return cl_interpolator, cd_interpolator




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

    #parametric_coeffs(jnp.full_like(r[:, None], 0), params_static, True)
### Establish Coefficients for Given Airfoil
import jax
from jax import numpy as jnp
from jax import lax
import jax.scipy as jsp
import matplotlib.pyplot as plt
from init import Inputs,Params
from airflow import resultantVelocityComponents
import numpy as np
import scipy as scipy
from Foil_Interpolators import RationalFunctionInterp, RationalFunction

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
    return C_L, C_D, C_M_stall


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
    clInterpolator = params.clInterpolator
    cdInterpolator = params.cdInterpolator

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

    # Coefficients in non-stall region
    C_L_unstall = jnp.zeros(jnp.shape(AoA_eff))
    C_D_unstall = jnp.zeros(jnp.shape(AoA_eff))
    
    for ii in range(len(params.r)):
        C_L_unstall = C_L_unstall.at[ii, :].set(RationalFunction(phi[ii, :], 
                                                                 clInterpolator[ii, 0:Inputs.P_Cl], 
                                                                 clInterpolator[ii, Inputs.P_Cl:(Inputs.P_Cl + Inputs.Q_Cl - 1)]))
        
        C_D_unstall = C_D_unstall.at[ii, :].set(RationalFunction(phi[ii, :], 
                                                                 cdInterpolator[ii, 0:Inputs.P_Cd], 
                                                                 cdInterpolator[ii, Inputs.P_Cd:(Inputs.P_Cd + Inputs.Q_Cd - 1)]))
    
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


def loadAirfoilData(filename: str, 
                    Re_list: jnp.ndarray, Ma_list: jnp.ndarray, nr: int,
                    shape: jnp.ndarray = jnp.array([81, 14, 6]), numInterpPoints: int = 50,
                    method:str='linear') -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Inputs:
     -> filename:           string containing the relative path to the file containing the airfoil data
     -> Re_list:            A list corresponding to the local reynolds number at each radial position of the propeller
     -> Ma_list:            A list corresponding to the local mach number at each radial position of the propeller
     -> nr:                 The number of radial stations in the propeller
     -> shape:              The shape of each data "block" in the input file. The input file is expected to be made up of one of these blocks for each of alpha, reynolds number, mach number, CL and CD and to be in that order
     -> numInterpPoints:    The number of interpolation points to generate from the input file prior to applying the rational function fitting
     -> method:             The method for performing the initial interpolation, prior to applying the rational function interpolation. Used to interpolate within the alphaInterpRange 
    
    Hidden Inputs: (Accessible through init.py):     
     -> CL_P:               Order of numerator polynomial in CL rational function
     -> CL_Q:               Order of denominator polynomial in CL rational function
     -> CD_P:               Order of numerator polynomial in CD rational function
     -> CD_Q:               Order of denominator polynomial in CD rational function

    Outputs:
     -> cl_interpolators:   an array of size [nr, (p + q)] with each radially spaced element consisting of [CL_P, p0, p1, ..., pn, q1, q2, ..., qn]
     -> cd_interpolators:   an array of size [nr, (p + q)] with each radially spaced element consisting of [CD_P, p0, p1, ..., pn, q1, q2, ..., qn]
    
    >>FUNCTION DESCRIPTION<<
    ...
    ... Description to go here
    ...
    >>FUNCTION DESCRIPTION<<
    """

    # Load the data from file
    array = np.loadtxt(filename, dtype='float', delimiter=',')

    # Establish the block size
    numbloks = 5    
    blocksize = int(np.shape(array)[0]/numbloks)

    # Load the blocks of data from the input file
    alp = array[0           :   blocksize,      :]
    rey = array[blocksize   :   blocksize*2,    :]
    mac = array[blocksize*2 :   blocksize*3,    :]
    cl  = array[blocksize*3 :   blocksize*4,    :]
    cd  = array[blocksize*4 :   blocksize*5,    :]

    # Reshape the blocks of data from 2D to 3D, breaking up only the last two dims:
    alp = np.unique(alp)
    rey = np.unique(rey)
    mac = np.unique(mac)
    cl  = np.reshape(cl, shape)
    cd  = np.reshape(cd, shape)
    
    # Create the cl and cd interpolator arrays
    cl_interpolator = jnp.zeros((nr, Inputs.P_Cl + Inputs.Q_Cl - 1))
    cd_interpolator = jnp.zeros((nr, Inputs.P_Cd + Inputs.Q_Cd - 1))
    
    # Generate a 3D scipy style interpolator object which can be called like a function to interpolate input points
    cli_temp = jsp.interpolate.RegularGridInterpolator((alp, rey, mac), cl, method=method)
    cdi_temp = jsp.interpolate.RegularGridInterpolator((alp, rey, mac), cd, method=method)

    # Iterate through the radial stations
    for r in range(nr):
        # Use the temporary 3d interpolator objects to generate 1D arrays for CL and CD
        Cl_temp = cli_temp((alp, Re_list[r], Ma_list[r]))
        Cd_temp = cdi_temp((alp, Re_list[r], Ma_list[r]))

        Cl_temp = jnp.where(jnp.isfinite(Cl_temp), Cl_temp, 0.01*jnp.sign(alp))
        Cd_temp = jnp.where(jnp.isfinite(Cd_temp), Cd_temp, 0.5)

        # Perform rational fitting on the temporary cl and cd arrays
        cl_params = RationalFunctionInterp(alp, Cl_temp, Inputs.P_Cl, Inputs.Q_Cl, verbose=False, extendedChecking=True)
        cd_params = RationalFunctionInterp(alp, Cd_temp, Inputs.P_Cd, Inputs.Q_Cd, verbose=False, extendedChecking=True)

        # store the data in the interpolator arrays
        cl_interpolator = cl_interpolator.at[r].set(cl_params)
        cd_interpolator = cd_interpolator.at[r].set(cd_params)
    
    # Return the interpolator arrays
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
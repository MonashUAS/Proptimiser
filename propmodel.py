## Define All Aerodynamic Properties for Given Propeller Geometry

# External Libraries
import jax
import jax.numpy as jnp
from jax import jit, vmap, grad, lax

import optax
import jaxopt

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import replace

# Internal Libraries
from init import Inputs, Params
from airfoil import parametric_coeffs
from airflow import tip_root_loss_factors, inflow_model, equivelantActuatorDiskEfficiency, resultantVelocityComponents


### Integrand Functions
def radial_stops(R, r_hub, n_r):
    '''
    Return radial node centers and dr for integration.
    '''
    r = jnp.linspace(r_hub + (R - r_hub) / (2 * n_r), R - (R - r_hub) / (2 * n_r), n_r)
    # Simple Homogeneous radial spacing
    dr = jnp.ones_like(r)*(R - r_hub) / n_r
    return r, dr


def angular_stops(n_psi):
    '''
    Return angular node centers and dpsi for integration
    '''
    psi = jnp.linspace(0, (2 * jnp.pi), n_psi)
    dpsi = jnp.ones_like(psi)*(2 * jnp.pi) / n_psi
    return psi, dpsi


### Thrust Calculations
def dT_blade(V_ia, params):
    """
    Calculate instantaneous thrust force for each radial and azimuthal element

    dT_blade_instantaneous
    Uses formulation from (M. Nahon et al., 2015) equation 5 modified with the inclusion of the prandtl loss parameter 'F'

    dT_blade
    Uses formulation from (M. Nahon et al., 2015) equation 6 modified with the inclusion of the prandtl loss parameter 'F'
    
    Outputs:
        dT_blade w/ size (Nr, 1) 
        dT_blade_instantaneous w/ size (Nr, Npsi)

    Assumes: |V_yz| < |V_x + V_ia| -> Assumption that the annuli do not interact with each other 
    """
    
    # Unpack parameters
    N, c, dr, psi = params.N, params.c, params.dr, params.psi # geometric and positional values
    rho = params.rho # air density

    # Calculate Local Flow Properties
    V_R, phi = resultantVelocityComponents(V_ia, params)
    
    # Get Aerodynamic Coefficients
    C_L, C_D, _ = parametric_coeffs(V_ia, params)

    # Tip and Root Loss Factors
    tip_loss, root_loss = tip_root_loss_factors(V_ia, params)

    # Calculate the total prandtl loss correction factor
    F = jnp.multiply(tip_loss, root_loss)

    # Blade Element Theory instantaneous thrust element
    dT_blade_instantaneous = 0.5 * rho * c * (V_R**2) * (C_L*jnp.cos(phi) - C_D*jnp.sin(phi)) * F * dr[:, None]
    
    # Average thrust in each annulus over one rotation (using formulation from the text prior to statement of eqn. 6)
    # Average thrust per annulus is the integral of the instantaneous thrust element divided by 2*pi
    dT_blade = 1/(2*jnp.pi) * jnp.trapezoid(dT_blade_instantaneous, psi, axis=1)

    return dT_blade, dT_blade_instantaneous


def dT_momentum(V_ia, params):
    """
    Calculate thrust integrand across each annulus for each radial station using momentum theory

    Uses formulation from (M. Nahon et al., 2015) equation 9

    Outputs: dT_momentum (Nr, 1)
             dT_momentum_instantaneous w/ size (Nr, Npsi)

    Assumes: |V_yz| < |V_x + V_ia| -> Assumption that the annuli do not interact with each other 
    """
    # Unpack parameters
    r, dr, psi = params.r, params.dr, params.psi
    V_x, V_yz = params.V_x, params.V_yz
    rho = params.rho

    # total velocity through each differential segment
    V_prime = jnp.sqrt((V_x + V_ia) ** 2 + V_yz ** 2)

    # Tip and Root Loss Factors
    tip_loss, root_loss = tip_root_loss_factors(V_ia, params)

    # Calculate the total prandtl loss correction factor
    F = jnp.multiply(tip_loss, root_loss) 

    # Momentum theory about annular element 
    dT_momentum_instantaneous = 2 * rho * r * V_ia * V_prime * F * dr[:, None]  
    dT_momentum = jnp.trapezoid(dT_momentum_instantaneous, psi, axis=1)

    return dT_momentum, dT_momentum_instantaneous


def thrust_residual(V_ia_0_guess, params):
    """
    Calculate thrust residual at radial stops (difference between blade+momentum theory)

    Takes the difference at each azimuthal stop, then integrates in the azimuthal direction

    (may extrapolate to radial+azimuthal stops later, 
    but for now skew rotor inflow is most complex and 
    dimensionality of problem is reduced)
    """

    # Establish induced velocity from inflow model
    V_ia = inflow_model(V_ia_0_guess, params)

    # Calculate the instantaneous blade element and momentum theory thrust elements:

    dT_BEMT_average, _ = dT_blade(V_ia, params) # get the average elements (1st output) only

    dT_MMNT_average, _ = dT_momentum(V_ia, params) # get the average elements (1st output) only

    dT_residual = dT_BEMT_average - dT_MMNT_average
    
    return dT_residual


### Outputs
def induced_velocity(params):
    '''
    Calculates the axial induced velocity (neglects the radial and tangential induced velocity) for
    the propeller. 
    
    Refer to figure 3 of "A propeller model for general forward flight conditions"
    [Waqas Khan and Meyer Nolan 2015] for insight into variables. 
    
    Same paper used for equations
    '''

    # Unpack initial guess for induced velocity distribution
    V_ia_init = inflow_model(params.V_ia_0, params)

    # Debug Check Initial Residuals
    if Inputs.verbose:
        res_check = thrust_residual(V_ia_init, params)
        print(res_check)

        print("Residual Shape", jnp.shape(res_check))
        print("Residual min/max:", res_check.min(), res_check.max())

    # Newton Solver for Induced Velocity
    solver = jaxopt.Broyden(fun=thrust_residual, maxiter=100, tol=Inputs.newton_eps, max_stepsize=1.0, implicit_diff=True)
    # Input initial guess for induced velocity and solve. (Induced velocity is parametrised per radial stop)
    (V_ia_0, info) = solver.run(V_ia_init, params)

    # Debug Check Final Residuals
    if Inputs.verbose:
        res_check = thrust_residual(params.V_ia_0, params)
        print("Residuals Check:", res_check)

    V_ia = inflow_model(V_ia_0, params)
    return V_ia


def dQ_blade(V_ia,params):
    '''
    Calculate Torque Integrand using Blade Element Theory at radial and aizmuthal stops.

    Uses formulation from (M. Nahon et al., 2015)

    Outputs: dQ_momentum (Nr, Npsi)
    """
    '''
    # Unpack parameters
    N, c, r, psi = params.N, params.c, params.r, params.psi
    V_x, V_yz, omega = params.V_x, params.V_yz, params.omega
    rho = params.rho

    # Calculate Local Flow Properties
    V_yz_perp = V_yz * jnp.sin(psi)                                     # Perpendicular component of velocity in yz plane
    phi = jnp.arctan((V_x + V_ia)/(omega * r + V_yz_perp))              # Angle of attack
    V_rel = jnp.sqrt((V_x + V_ia) ** 2 + (omega * r + V_yz_perp) ** 2)  # Relative velocity
    # Get Aerodynamic Coefficients
    C_L, C_D, C_M = parametric_coeffs(V_ia, params)

    # Tip and Root Loss Factors
    tip_loss, root_loss = tip_root_loss_factors(V_ia, params)

    # Blade Element Theory about annular element
    dQ_blade =( 
        (N / (4 * jnp.pi)) * rho 
        * c * V_rel ** 2 
        * (C_L * jnp.cos(phi) + C_D * jnp.sin(phi))
        # tip and root loss factors
        * tip_loss * root_loss
        )
    return dQ_blade


def torque(V_ia, params):
    """
    Integrate to find Total Torque of Propeller
    """
    r, psi = params.r, params.psi
    dQ = dQ_blade(V_ia, params)

    # Numerically Compute Double Integral to Solve for Torque
    inner = jnp.trapezoid(dQ, psi, axis=1)
    inner_r = jnp.squeeze(inner) * jnp.squeeze(r)
    Q = jnp.trapezoid(jnp.squeeze(inner_r), jnp.squeeze(r))

    return Q


def thrust(V_ia, params):
    '''
    Calculate total thrust from solved induced velocity
    '''
    r = jnp.squeeze(params.r)
    dT = dT_momentum(V_ia,params)[0]

    # Numerically Compute Double Integral to Solve for Thrust
    T = jnp.trapezoid(dT, r)

    return T


def power(V_ia, params):
    '''
    Calculates the aerodynamic Power of Propeller (W) using Momentum Theory.

    DO NOT USE THIS FUNCTION FOR DETERMINING THE POWER INPUT FOR THE PROPELLER
    '''

    # Unpack Params
    r, psi = params.r, params.psi
    # Calculate Local Axial Flow Velocity
    V_x = params.V_x
    V = V_ia+V_x 

    # Elemental torque
    dT = dT_momentum(V_ia, params)

    dP = dT*V

    P = jnp.trapezoid(jnp.trapezoid(dP,r,axis=0),psi)
    return P


def mechanicalPower(V_ia, params):
    '''
    Calculates and returns the shaft power (W) applied to the propeller 
    '''    

    # Unpack Params
    omega = params.omega
    # Calculate Local Axial Flow Velocity

    # net torque over one rotation
    Q = torque(V_ia, params)

    # Net mechanical power (omega in rad/s, torque in Nm)
    P = Q * omega

    return jnp.squeeze(P)



def actuator_disk_power(V_ia, power):
    '''
    Calculate Ideal Actuator Disk Power for same thrust

    UNFINISHED CODE

    TODO Implement actuator thrust
    TODO FInish this code
    '''

    raise(NotImplementedError("This function is yet to be implemented"))

    T = actuator_thrust(V_ia,params)
    V_x = params.V_x

    V_ind_mean = jnp.mean(V_ia)

    return P_actuator


def flatblade(R, c, beta, N, plot=False):
    '''
    Flat Blade Demo Model. Inputs a blade of fixed chord and radial distribution 
    and solves+plots induced velocity.

    TODO Fix function (chord and beta not declared properly)
    '''
    # 
    # Debug NAN's and enable 64 bit precision
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_debug_nans", True)
    # Populate radial/azimuthal stops and initial guess
    psi, dpsi = angular_stops(Inputs.n_psi)
    r, dr = radial_stops(R, 0.05, Inputs.n_r)
    V_ia_0 = jnp.full_like(r,jnp.array([10.0]))

    # Update Parameters datalass
    global params

    params = Params(
                N = N,
                R = jnp.array([R]),
                c = jnp.array(c)[:,None],
                beta = jnp.radians(beta)[:,None],

                r = r[:,None],
                dr = dr,

                psi = psi[None,:],
                dpsi = dpsi,

                V_x = jnp.array([10.0]),
                V_yz = jnp.array([10.0]),
                omega = jnp.array([400]),

                V_ia_0 = V_ia_0[:,None]
            )
    # Register dataclass as a PyTree
    #jax.tree_util.register_dataclass(Params) (depreciated for now)
    
    # Solve Induced Velocity
    V_ia = induced_velocity(params)  # shape (Nr, Npsi)

    # Calculate total thrust
    T = thrust(V_ia, params)
    
    print("Total Thrust:", T)

    actuatorEffiency = equivelantActuatorDiskEfficiency(T,params)
    print("Actuator effiency:", actuatorEffiency)
    if plot==True:
        # Plot heatmap
        fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
        plt.contourf(psi, r, V_ia, shading='auto', cmap='viridis')
        plt.colorbar(label='Induced Velocity [m/s]')
        plt.xlabel('X [m]')
        plt.ylabel('Y [m]')
        plt.title('Induced Velocity over Propeller Disk')
        plt.show()

    return params, V_ia, T


if __name__ == "__main__":
    ## Demo Cases of BEMT
    # single is single annulus solver
    # flat is simple flat blade solver
    # input allows importing of a json file 
    demo_solve = "single"

    # Single Annulus Solver
    if demo_solve == "single":

        psi,dpsi = angular_stops(Inputs.n_psi)
        jax.config.update("jax_enable_x64", True)
        jax.config.update("jax_debug_nans", True)
        #Update Parameters Class
        params = Params(
                    N = 2,
                    R = jnp.array([0.5]),
                    c = jnp.array([0.05]),
                    beta = jnp.array([jnp.radians(20)]),

                    r = jnp.array([0.3]),
                    dr = jnp.array([0.05]),

                    psi = psi,
                    dpsi = dpsi,

                    V_x = jnp.array([10.0]),
                    V_yz = jnp.array([3.0]),
                    omega = jnp.array([400]),

                    V_ia_0 = jnp.array([20.0])[:,None]
                )
        # Register dataclass as a PyTree
        #jax.tree_util.register_dataclass(Params)

        # Solve
        V_ia = induced_velocity(params)

        # Plot Induced Velocity Distribution about singular annulus
        plt.plot(jnp.degrees(params.psi), V_ia[1])
        plt.title("Induced Velocity Distribution around Propeller Annulus")
        plt.xlabel("Angle (degrees)")
        plt.ylabel("Induced Velocity (m/s)")
        plt.show()

    elif demo_solve == "flat":
        c = np.ones_like(Inputs.n_r)
        flatblade(1, 0.1, np.pi/8, 2, plot=True)
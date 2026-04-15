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
from airfoil import aero_coeffs
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


### Calculate thrust, hub and torque elements
def dT_blade(V_ia: jnp.ndarray, params: Params, debug: bool=False):
    """
    Calculates instantaneous thrust at each radial and azimuthal station, 
    and calculate the average thrust in each annulus over a rotation

    Inputs:
        V_ia        <- Induced velocity at each radial and azimuthal station
        params      <- Params object used in the optimisation loop, remains unmodified by function execution
        debug       <- Boolean to select whether or not to print debug information when function is called

    Outputs:
        dT_blade_ave w/ size (Nr, 1)                        <- Thrust per annulus 
        dT_blade_instantaneous w/ size (Nr, Npsi)       <- Thrust per differential element

    Assumes: 
        |V_yz| < |V_x + V_ia|   <- Assumption that the annuli do not interact with each other 

    Uses formulation from (M. Nahon et al., 2015) equations 5 and 6 modified with 
    the inclusion of the prandtl loss parameter 'F'
    """
    
    # Unpack parameters
    N, c, psi = params.N, params.c, params.psi # geometric and positional values
    rho = params.rho # air density

    # Calculate Local Flow Properties
    V_R, phi = resultantVelocityComponents(V_ia, params)
    
    # Get Aerodynamic Coefficients
    C_L, C_D, _ = aero_coeffs(V_ia, params)

    # Tip and Root Loss Factors
    tip_loss, root_loss = tip_root_loss_factors(V_ia, params)

    # Calculate the total prandtl loss correction factor
    F = jnp.multiply(tip_loss, root_loss)

    # Blade Element Theory instantaneous thrust element
    dT_blade_instantaneous = 0.5 * rho * c[:, None] * (V_R**2) * (C_L*jnp.cos(phi) - C_D*jnp.sin(phi)) * F 

    # Average thrust in each annulus over one rotation (using formulation from the text prior to statement of eqn. 6)
    # Average thrust per annulus is the integral of the instantaneous thrust element divided by 2*pi
    dT_blade_ave = N/(2*jnp.pi) * jnp.trapezoid(dT_blade_instantaneous, x=psi, axis=1)

    if debug:
        print("------dT_blade DEBUG------")

        print("dT Shapes (inst., ave.):")
        print((jnp.shape(dT_blade_instantaneous), jnp.shape(dT_blade_ave)))

        print("V_ia shape:")
        print(jnp.shape(V_ia))

        print("F shape:")
        print(jnp.shape(F))
        
        print("------dT_blade DEBUG------")

    return dT_blade_ave, dT_blade_instantaneous


def dH_blade(V_ia, params, debug=False):
    """
    Calculates instantaneous hub force at each radial and azimuthal station, 
    and calculate the average hub force in each annulus over a rotation

    The hub forces are defined as the force element coplanar with the rotor disk 
    and normal to the radial direction

    Inputs:
        V_ia        <- Induced velocity at each radial and azimuthal station
        params      <- Params object used in the optimisation loop, remains unmodified by function execution
        debug       <- Boolean to select whether or not to print debug information when function is called

    Outputs:
        dH_blade_ave w/ size (Nr, 1)                        <- Force per annulus 
        dH_blade_instantaneous w/ size (Nr, Npsi)       <- Force per differential element

    Assumes: 
        |V_yz| < |V_x + V_ia|   <- Assumption that the annuli do not interact with each other 

    Uses formulation from (M. Nahon et al., 2015) equation 11 modified with 
    the inclusion of the prandtl loss parameter 'F'
    """
    
    # Unpack parameters
    N, c, psi = params.N, params.c, params.psi # geometric and positional values
    rho = params.rho # air density

    # Calculate Local Flow Properties
    V_R, phi = resultantVelocityComponents(V_ia, params)
    
    # Get Aerodynamic Coefficients
    C_L, C_D, _ = aero_coeffs(V_ia, params)

    # Tip and Root Loss Factors
    tip_loss, root_loss = tip_root_loss_factors(V_ia, params)

    # Calculate the total prandtl loss correction factor
    F = jnp.multiply(tip_loss, root_loss)

    # Inner integral term
    inner = (V_R**2) * (C_L*jnp.sin(phi) + C_D*jnp.cos(phi))

    # Average thrust in each annulus over one rotation (using formulation from the text prior to statement of eqn. 6)
    # Average thrust per annulus is the integral of the instantaneous thrust element divided by 2*pi
    dH_blade_ave = N/(4*jnp.pi) * rho * c * jnp.trapezoid(inner, x=psi, axis=1)

    if debug:
        print("------dH_blade DEBUG------")

        print("dH Shapes (ave.):")
        print((jnp.shape(dH_blade_ave)))

        print("V_ia shape:")
        print(jnp.shape(V_ia))

        print("F shape:")
        print(jnp.shape(F))
        
        print("------dH_blade DEBUG------")

    return dH_blade_ave


def dQ_blade(V_ia, params, debug=False):
    """
    Calculates instantaneous torque at each radial and azimuthal station, 
    and calculate the average torque in each annulus over a rotation

    Inputs:
        V_ia        <- Induced velocity at each radial and azimuthal station
        params      <- Params object used in the optimisation loop, remains unmodified by function execution
        debug       <- Boolean to select whether or not to print debug information when function is called

    Outputs:
        dQ_blade w/ size (Nr, 1)                        <- Torque per annulus 
        dQ_blade_instantaneous w/ size (Nr, Npsi)       <- Torque per differential element

    Assumes: 
        |V_yz| < |V_x + V_ia|   <- Assumption that the annuli do not interact with each other 

    Uses formulation from (M. Nahon et al., 2015) equation 12
    """
    
    r = params.r

    # Calculate the hub force elements dH
    dH_blade_ave = dH_blade(V_ia, params, debug)

    # Use equation 12 to calculate the torque elements from the hub elements
    # Technically these should be the negative, but its easier to work with if they arent
    dQ_blade_ave = dH_blade_ave * jnp.squeeze(r)

    if debug:
        print("------dQ_blade DEBUG------")

        print("dQ Shapes (inst., ave.):")
        print((jnp.shape(dQ_blade_ave)))

        print("V_ia shape:")
        print(jnp.shape(V_ia))
        
        print("------dT_blade DEBUG------")

    return dQ_blade_ave


### Calculate thrust elements using momentum imparted to the air
def dT_momentum(V_ia, params, debug=False):
    """
    Calculates instantaneous thrust at each radial and azimuthal station, 
    and calculate the average thrust in each annulus over a rotation

    Unlike dT_blade, dT_momentum uses the momentum change in the air as it passes through the 
    blade element and is accelerated rather than the forces on the blade element to determine
    thrust 

    Inputs:
        V_ia        <- Induced velocity at each radial and azimuthal station
        params      <- Params object used in the optimisation loop, remains unmodified by function execution
        debug       <- Boolean to select whether or not to print debug information when function is called

    Outputs:
        dT_momentum w/ size (Nr, 1)                        <- Thrust per annulus 
        dT_momentum_instantaneous w/ size (Nr, Npsi)       <- Thrust per differential element

    Assumes: 
        |V_yz| < |V_x + V_ia|   <- Assumption that the annuli do not interact with each other 

    Uses formulation from (M. Nahon et al., 2015) equation 9
    """
    # Unpack parameters
    r, psi = params.r, params.psi
    V_x, V_yz = params.V_x, params.V_yz
    rho = params.rho

    # total velocity through each differential segment
    V_prime = jnp.sqrt((V_x + V_ia) ** 2 + V_yz ** 2)

    # Tip and Root Loss Factors
    tip_loss, root_loss = tip_root_loss_factors(V_ia, params)

    # Calculate the total prandtl loss correction factor
    F = jnp.multiply(tip_loss, root_loss) 

    # Momentum theory about annular element
    dT_momentum = 2 * rho * jnp.squeeze(r) * jnp.trapezoid(V_ia * V_prime * F, x=psi, axis=1)

    if debug:
        print("------dT_momentum DEBUG------")

        print("dT Shapes (ave.):")
        print(jnp.shape(dT_momentum))

        print("V_ia shape:")
        print(jnp.shape(V_ia))

        print("F shape:")
        print(jnp.shape(F))
        
        print("------dT_momentum DEBUG------")

    return dT_momentum


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

    dT_BEMT_average, _ = dT_blade(V_ia, params, debug=False) # get the average elements (1st output) only

    dT_MMNT_average = dT_momentum(V_ia, params, debug=False) # get the average elements (1st output) only

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

    Returns:
        V_ia: Induced velocity at each azimuthal and radial point (size[n_r, n_psi])
        V_ia_0: Velocity induced by each annulus at the center of the propeller (size[n_r, ])
    '''

    # Debug Check Initial Residuals
    if Inputs.verbose:
        res_check = thrust_residual(params.V_ia_0, params)
        print(res_check)

        print("Residual Shape", jnp.shape(res_check))
        print("Residual min/max:", res_check.min(), res_check.max())

    # Newton Solver for Induced Velocity
    solver = jaxopt.Broyden(fun=thrust_residual, maxiter=100, tol=Inputs.newton_eps, max_stepsize=5.0, implicit_diff=True)
    # Input initial guess for induced velocity and solve. (Induced velocity is parametrised per radial stop)
    (V_ia_0, info) = solver.run(params.V_ia_0, params)

    # Debug Check Final Residuals
    if Inputs.verbose:
        res_check = thrust_residual(params.V_ia_0, params)
        print("Residuals Check:", res_check)

    V_ia = inflow_model(V_ia_0, params)
    return V_ia, V_ia_0


def torque(V_ia, params, debug=False):
    """
    Calculates the total torque on the propeller by integrating the average torque per annulus per rotation
    along the radial direction
    
    Inputs:
        V_ia        <- Induced velocity at each radial and azimuthal station
        params      <- Params object used in the optimisation loop, remains unmodified by function execution
        debug       <- Boolean to select whether or not to print debug information when function is called

    Outputs:
        Q <- Net torque per rotation of the propeller
    
    Assumes: 
        |V_yz| < |V_x + V_ia|   <- Assumption that the annuli do not interact with each other 
    """
    
    r = jnp.squeeze(params.r)

    # Get the average torque per rotation in each annulus using the dQ_blade function
    dQ_blade_ave = dQ_blade(V_ia, params, debug)

    # Use trapezoidal integration in the radial direction to compute the total torque
    Q = jnp.trapezoid(dQ_blade_ave, x=r)

    return Q


def thrust(V_ia, params):
    '''
    Calculate total thrust from solved induced velocity
    '''
    r = jnp.squeeze(params.r)
    dT = dT_blade(V_ia,params)[0]

    # Numerically Compute Double Integral to Solve for Thrust
    T = jnp.trapezoid(dT, x=r)

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

    P = jnp.trapezoid(jnp.trapezoid(dP,x=r,axis=0),x=psi)
    return P


def mechanicalPower(V_ia, params):
    '''
    Calculates and returns the shaft power (W) applied to the propeller 
    '''    

    # Unpack Params
    omega = params.omega
    # Calculate Local Axial Flow Velocity

    # net torque over one rotation
    Q = torque(V_ia, params, debug=False)

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
    V_ia, _ = induced_velocity(params)  # shape (Nr, Npsi)

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
        V_ia, _ = induced_velocity(params)

        # Plot Induced Velocity Distribution about singular annulus
        plt.plot(jnp.degrees(params.psi), V_ia[1])
        plt.title("Induced Velocity Distribution around Propeller Annulus")
        plt.xlabel("Angle (degrees)")
        plt.ylabel("Induced Velocity (m/s)")
        plt.show()

    elif demo_solve == "flat":
        c = np.ones_like(Inputs.n_r)
        flatblade(1, 0.1, np.pi/8, 2, plot=True)
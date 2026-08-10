## Define All Aerodynamic Properties for Given Propeller Geometry

# External Libraries
import jax
import jax.numpy as jnp
import equinox as eqx
import jaxopt
from dataclasses import replace

import time

import numpy as np
import matplotlib.pyplot as plt

# Internal Libraries
from init import build_default_propeller, Inputs, Params
from airfoil import aero_coeffs
from airflow import tip_root_loss_factors, inflow_model, equivelantActuatorDiskEfficiency, resultantVelocityComponents, high_induction_correction

## Blade Element Theory 
def dT_blade(V_i: jnp.ndarray, params: Params, debug: bool = False):
    """
    Calculates instantaneous thrust at each radial and azimuthal station, 
    and calculates the average thrust in each annulus over a rotation.

    Inputs:
        V_i         <- 3D Induced velocity grid of shape (3, Nr, Npsi) [axial, tangential, radial]
        params      <- Params object used in the optimisation loop, remains unmodified by function execution
        debug       <- Boolean to select whether or not to print debug information when function is called

    Outputs:
        dT_blade_ave w/ size (Nr, )                        <- Thrust per annulus 
        dT_blade_instantaneous w/ size (Nr, Npsi)          <- Thrust per differential element

    Assumes: 
        |V_yz| < |V_x + V_ia|   <- Assumption that the annuli do not interact with each other 

    Uses formulation from (M. Nahon et al., 2015) equations 5 and 6 modified with 
    the inclusion of the prandtl loss parameter 'F'
    """
    # Unpack parameters
    N, c, psi = params.N, params.c, params.psi # geometric and positional values
    rho = params.rho # air density

    # Calculate Local Flow Properties
    V_R, phi = resultantVelocityComponents(V_i, params)

    # Get Aerodynamic Coefficients
    C_L, C_D, _ = aero_coeffs(V_i, params)

    # Blade Element Theory instantaneous thrust element
    dT_blade_instantaneous = 0.5 * rho * c[:, None] * (V_R**2) * (C_L * jnp.cos(phi) - C_D * jnp.sin(phi))
    
    # Average thrust in each annulus over one rotation
    dT_blade_ave = N / (2 * jnp.pi) * jnp.trapezoid(dT_blade_instantaneous, x=psi, axis=1)

    if debug:
        print("------dT_blade DEBUG------")
        print("dT Shapes (inst., ave.):", (jnp.shape(dT_blade_instantaneous), jnp.shape(dT_blade_ave)))
        print("V_i shape:", jnp.shape(V_i))
        print("------dT_blade DEBUG------")

    return dT_blade_ave


def dH_blade(V_i: jnp.ndarray, params: Params, debug: bool = False):
    """
    Calculates instantaneous hub force at each radial and azimuthal station, 
    and calculates the average hub force in each annulus over a rotation.

    The hub forces are defined as the force element coplanar with the rotor disk 
    and normal to the radial direction.

    Inputs:
        V_i         <- 3D Induced velocity grid of shape (3, Nr, Npsi) [axial, tangential, radial]
        params      <- Params object used in the optimisation loop, remains unmodified by function execution
        debug       <- Boolean to select whether or not to print debug information when function is called

    Outputs:
        dH_blade_ave w/ size (Nr, )                        <- Force per annulus 

    Assumes: 
        |V_yz| < |V_x + V_ia|   <- Assumption that the annuli do not interact with each other 

    Uses formulation from (M. Nahon et al., 2015) equation 11 modified with 
    the inclusion of the prandtl loss parameter 'F'
    """
    # Unpack parameters
    N, c, psi = params.N, params.c, params.psi # geometric and positional values
    rho = params.rho # air density

    # Calculate Local Flow Properties
    V_R, phi = resultantVelocityComponents(V_i, params)
    
    # Get Aerodynamic Coefficients
    C_L, C_D, _ = aero_coeffs(V_i, params)

    # Tip and Root Loss Factors
    tip_loss, root_loss = tip_root_loss_factors(phi, params)
    
    # Calculate the total prandtl loss correction factor
    F = jnp.multiply(tip_loss, root_loss)
    F = jnp.clip(F, min=0.05, max=1.0) # clip to prevent div by zero

    # Inner integral term
    inner = (V_R**2) * (C_L * jnp.sin(phi) + C_D * jnp.cos(phi))

    # Average hub force per annulus over one rotation
    dH_blade_ave = N / (4 * jnp.pi) * rho * c * jnp.trapezoid(inner, x=psi, axis=1)

    if debug:
        print("------dH_blade DEBUG------")
        print("dH Shapes (ave.):", jnp.shape(dH_blade_ave))
        print("V_i shape:", jnp.shape(V_i))
        print("F shape:", jnp.shape(F))
        print("------dH_blade DEBUG------")

    return dH_blade_ave


def dQ_blade(V_i: jnp.ndarray, params: Params, debug: bool = False):
    """
    Calculates instantaneous torque at each radial and azimuthal station, 
    and calculates the average torque in each annulus over a rotation.

    Inputs:
        V_i         <- 3D Induced velocity grid of shape (3, Nr, Npsi) [axial, tangential, radial]
        params      <- Params object used in the optimisation loop, remains unmodified by function execution
        debug       <- Boolean to select whether or not to print debug information when function is called

    Outputs:
        dQ_blade_ave w/ size (Nr, )                        <- Torque per annulus 

    Assumes: 
        |V_yz| < |V_x + V_ia|   <- Assumption that the annuli do not interact with each other 

    Uses formulation from (M. Nahon et al., 2015) equation 12
    """
    r = params.r

    # Calculate the hub force elements dH
    dH_blade_ave = dH_blade(V_i, params, debug)

    # Use equation 12 to calculate the torque elements from the hub elements
    dQ_blade_ave = dH_blade_ave * r

    if debug:
        print("------dQ_blade DEBUG------")
        print("dQ Shapes (ave.):", jnp.shape(dQ_blade_ave))
        print("V_i shape:", jnp.shape(V_i))
        print("------dQ_blade DEBUG------")

    return dQ_blade_ave


### Momentum Theory
def dT_momentum(V_i: jnp.ndarray, params: Params, debug: bool = False):
    """
    Calculates instantaneous thrust at each radial and azimuthal station, 
    and calculates the average thrust in each annulus over a rotation.

    dT_momentum uses the momentum change in the air as it passes through the 
    blade element and is accelerated rather than the forces on the blade element to determine
    thrust.

    Inputs:
        V_i         <- 3D Induced velocity grid of shape (3, Nr, Npsi) [axial, tangential, radial]
        params      <- Params object used in the optimisation loop, remains unmodified by function execution
        debug       <- Boolean to select whether or not to print debug information when function is called

    Outputs:
        dT_momentum_ave w/ size (Nr, )                     <- Thrust per annulus 

    Assumes: 
        |V_yz| < |V_x + V_ia|   <- Assumption that the annuli do not interact with each other 

    Uses formulation from (M. Nahon et al., 2015) equation 9
    """
    # Unpack axial induced velocity component
    V_ia = V_i[0]

    # Momentum theory about annular element
    r, psi = params.r, params.psi
    V_x, V_yz = params.V_x, params.V_yz
    rho = params.rho

    V_R, phi = resultantVelocityComponents(V_i, params)

    # Tip and Root Loss Factors
    tip_loss, root_loss = tip_root_loss_factors(phi, params)
    F = jnp.multiply(tip_loss, root_loss) 
    F = jnp.clip(F, min=0.05, max=1.0) # clip to prevent div by zero

    # Compute baseline classical momentum element locally
    V_prime = jnp.sqrt((V_x + V_ia) ** 2 + V_yz ** 2)
    dT_element_classical = 2 * rho * r[:, None] * V_ia * V_prime * F

    # Route through lax.switch inside airflow.py for conditional modifications
    dT_element_corrected = high_induction_correction(dT_element_classical, V_ia, F, params)

    # Integrate over psi to get the average thrust per annulus
    dT_momentum_ave = jnp.trapezoid(dT_element_corrected, x=psi, axis=1)

    if debug:
        print("------dT_momentum DEBUG------")
        print("dT Shapes (ave.):", jnp.shape(dT_momentum_ave))
        print("V_i shape:", jnp.shape(V_i))
        print("F shape:", jnp.shape(F))
        print("------dT_momentum DEBUG------")

    return dT_momentum_ave

def dQ_momentum(V_i, params, debug=False):
    """
    Calculates instantaneous torque at each radial and azimuthal station, 
    and calculate the average torque in each annulus over a rotation

    Inputs:
        V_i         <- Induced velocity grid (3, n_r, n_psi) [axial, tangential, radial]
        params      <- Params object used in the optimisation loop, remains unmodified by function execution
        debug       <- Boolean to select whether or not to print debug information when function is called

    Outputs:
        dQ_momentum_ave w/ size (Nr, )                        <- Torque per annulus 

    Assumes: 
        |V_yz| < |V_x + V_ia|   <- Assumption that the annuli do not interact with each other 

    Uses formulation from (M. Nahon et al., 2015) adapted for angular momentum
    """
    # Extract velocity components from 3D grid
    V_ia = V_i[0]  # shape (n_r, n_psi)
    V_it = V_i[1]  # shape (n_r, n_psi)

    # Momentum theory about annular element
    r, psi = params.r, params.psi
    V_x, V_yz = params.V_x, params.V_yz
    rho = params.rho

    V_R, phi = resultantVelocityComponents(V_i, params)

    # Tip and Root Loss Factors
    tip_loss, root_loss = tip_root_loss_factors(phi, params)
    F = jnp.multiply(tip_loss, root_loss) 
    F = jnp.clip(F, min=0.05, max=1.0) # clip to prevent div by zero

    # Compute baseline classical momentum element locally
    # Uses V_prime to account for skewed wake mass flux (Nahon 2015)
    V_prime = jnp.sqrt((V_x + V_ia) ** 2 + V_yz ** 2)
    
    # dQ element per radian (omitting dr to match dT_momentum format)
    dQ_element_classical = 2 * rho * (r[:, None] ** 2) * V_it * V_prime * F

    # Integrate over psi to get the average torque per annulus
    dQ_momentum_ave = jnp.trapezoid(dQ_element_classical, x=psi, axis=1)

    if debug:
        print("------dQ_momentum DEBUG------")
        print("dQ Shapes (ave.):", jnp.shape(dQ_momentum_ave))
        print("V_ia shape:", jnp.shape(V_ia))
        print("V_it shape:", jnp.shape(V_it))
        print("F shape:", jnp.shape(F))
        print("------dQ_momentum DEBUG------")

    return dQ_momentum_ave

def thrust_residual(V_i, params):
    """
    Calculates the 1D thrust residual (BEMT vs Momentum) across radial stops.
    Returns array of shape (n_r,)
    """
    dT_BEMT_avg = dT_blade(V_i, params, debug=False)
    dT_MMNT_avg = dT_momentum(V_i, params, debug=False)
    
    return dT_BEMT_avg - dT_MMNT_avg

def torque_residual(V_i, params):
    """
    Calculates the 1D torque residual (BEMT vs Momentum) across radial stops.
    Returns array of shape (n_r,)
    """
    dQ_BEMT_avg  = dQ_blade(V_i, params, debug=False)
    dQ_MMNT_avg = dQ_momentum(V_i, params, debug=False)
    
    return dQ_BEMT_avg - dQ_MMNT_avg

def bemt_residuals(lambda_i_0_guess, params):
    """
    Evaluates the coupled BEMT residuals.
    lambda_i_0_guess: non-dimensional inflow ratios of shape (3, n_r)
    """
    # Retrieve scales and Re-dimensionalise
    V_scale = get_velocity_scales(params)
    V_i_0_guess_dim = lambda_i_0_guess * V_scale

    # Expand to 3D grid
    V_i = inflow_model(V_i_0_guess_dim, params)

    # Axial Residual (Scale by Thrust Target)
    dT_res_physical = thrust_residual(V_i, params)
    dT_res = dT_res_physical / params.thrust_target
    
    # Radial Residual (Already O(1))
    dR_res = lambda_i_0_guess[2] - 0.0

    # Tangential Residual Branches
    def a_dQ_res(V_i, params):
        return lambda_i_0_guess[1] - 0.0

    def at_dQ_res(V_i, params):
        dQ_res_physical = torque_residual(V_i, params)
        # Scale by an approximate ideal torque to make O(1)
        Q_ref = params.thrust_target * Inputs.R_tip 
        return dQ_res_physical / Q_ref

    dQ_res = jax.lax.switch(
        Inputs.bemt_type, 
        [a_dQ_res, at_dQ_res], 
        *(V_i, params)
    )

    return jnp.stack([dT_res, dQ_res, dR_res], axis=0)

def get_velocity_scales(params):
    """
    Computes characteristic velocity scales for the 3 inflow components
    to act as a diagonal preconditioner for the Broyden solver.
    Returns: V_scale -> shape (3, 1) for broadcasting against (3, n_r) arrays.
    """
    A_disk = jnp.pi * (Inputs.R_tip ** 2)
    
    # Axial Scale: Mean Momentum Inflow
    v_h = jnp.sqrt(params.thrust_target / (2.0 * params.rho * A_disk))
    V_ia_mean = -params.V_x / 2.0 + jnp.sqrt((params.V_x / 2.0)**2 + v_h**2)
    
    # Tangential Scale: Ideal Mean Swirl
    P_ideal = params.thrust_target * (params.V_x + V_ia_mean)
    Q_ideal = P_ideal / params.omega
    V_it_mean = Q_ideal / (params.rho * A_disk * (params.V_x + V_ia_mean) * Inputs.R_tip)
    
    # Radial Scale: Tip Speed (Centrifugal reference)
    V_tip = params.omega * Inputs.R_tip

    # Protect against zero-torque states (e.g., if thrust target is 0)
    V_it_mean = jnp.maximum(V_it_mean, 1e-4)

    # Stack and reshape to (3, 1) so it broadcasts perfectly with (3, n_r)
    return jnp.array([V_ia_mean, V_it_mean, V_tip])[:, None]

@jax.jit
def induced_velocity(params):
    '''
    Calculates the coupled 3D induced velocity field using BEMT.
    '''
    V_i_0_guess_physical = params.V_i_0

    # Non-Dimensionalise Initial Guess Component-Wise
    V_scale = get_velocity_scales(params)
    initial_guess_lambda = V_i_0_guess_physical / V_scale

    if Inputs.verbose:
        res_check = bemt_residuals(initial_guess_lambda, params)
        jax.debug.print("Initial Residual Shape {s}", s=jnp.shape(res_check))

    # Broyden Solver Call
    solver = jaxopt.Broyden(
        fun=bemt_residuals, 
        maxiter=Inputs.V_i_0_max_iter, 
        tol=Inputs.V_i_0_eps, 
        implicit_diff=True,
        verbose = Inputs.verbose
    )
    
    (lambda_i_0, info) = solver.run(initial_guess_lambda, params)

    if Inputs.verbose:
        final_res = bemt_residuals(lambda_i_0, params)
        jax.debug.print("Final Residuals: {res}", res=final_res)

    # Re-dimensionalise the converged solution
    V_i_0 = lambda_i_0 * V_scale

    V_i = inflow_model(V_i_0, params)

    return V_i, V_i_0

def torque(V_i, params, debug=False):
    """
    Calculates the total torque on the propeller by integrating the average torque per annulus per rotation
    along the radial direction
    
    Inputs:
        V_i        <- Induced velocity at each radial and azimuthal station
        params      <- Params object used in the optimisation loop, remains unmodified by function execution
        debug       <- Boolean to select whether or not to print debug information when function is called

    Outputs:
        Q <- Net torque per rotation of the propeller
    
    Assumes: 
        |V_yz| < |V_x + V_ia|   <- Assumption that the annuli do not interact with each other 
    """
    
    r = params.r

    # Get the average torque per rotation in each annulus using the dQ_blade function
    dQ_blade_ave = dQ_blade(V_i, params, debug)

    # Use trapezoidal integration in the radial direction to compute the total torque
    Q = jnp.trapezoid(dQ_blade_ave, x=r)

    return Q

def thrust(V_i, params):
    '''
    Calculate total thrust from solved induced velocity
    '''
    r = params.r
    dT = dT_momentum(V_i, params)

    # Numerically Compute Double Integral to Solve for Thrust
    T = jnp.trapezoid(dT, x=r)

    return T

def mechanicalPower(V_i, params):
    '''
    Calculates and returns the shaft power (W) applied to the propeller 
    '''

    # Unpack Params
    omega = params.omega
    # Calculate Local Axial Flow Velocity

    # net torque over one rotation
    Q = torque(V_i, params, debug=False)

    # Net mechanical power (omega in rad/s, torque in Nm)
    P = Q * omega

    return jnp.squeeze(P)

def actuator_disk_power(T, params):
    '''
    Calculate Ideal Actuator Disk Power for same thrust
    Provides theoretical local minimum power using actuator disc model
    T is from momentum theory, 
    '''
    rho = params.rho
    R   = Inputs.R_tip
    V_x = params.V_x # freestream velocity

    A = jnp.pi*R**2 # disc area
    v_induced = jnp.sqrt(T/(2*rho*A) + V_x**2/4) - V_x/2  # induced velocity through actuator disc
    P_act = T*(V_x+v_induced) # Ideal actuator disk power
    return jnp.squeeze(P_act)

def single(c, beta, N, plot=False):
    '''
    Single Annulus Solve
    '''
    inputs = Inputs(
        n_r=1
    )
    # Build the default propeller profile
    beta_deg = float(jnp.degrees(beta))
    params = build_default_propeller(c=c, beta=beta_deg, inputs=inputs)

    # Match the specific geometry overrides from your original test case
    params = eqx.tree_at(
        lambda p: (p.r, p.dr, p.N, p.V_ia_0),
        params,
        (jnp.array([0.3]), jnp.array([0.05]), jnp.array([N]), jnp.full((1,), 1.0))
    )

    # Solve Induced Velocity
    V_ia, _ = induced_velocity(params)

    if plot:
        # FIX: Changed V_ia[1] to V_ia[0] since there is only 1 radial node
        plt.plot(jnp.degrees(params.psi), V_ia[0])
        plt.title("Induced Velocity Distribution around Propeller Annulus")
        plt.xlabel("Angle (degrees)")
        plt.ylabel("Induced Velocity (m/s)")
        plt.show()


def flatblade(c, beta, N, plot=False):
    '''
    Flat Blade Demo Model. Inputs a blade of fixed chord and radial distribution 
    and solves+plots induced velocity.

    Used for benchmarking the induced velocity solver

    TODO: Fix plotting
    '''

    # Utilize your factory function to build the base configuration
    params = build_default_propeller(c=c, beta=beta)

    # Override remaining parameter states cleanly for JAX
    params = eqx.tree_at(
        lambda p: (p.N),
        params,
        (jnp.array([N]))
    )
    
    import os
    import jax.profiler

    # Create a cross-platform path to ./logs/tensorboard
    log_dir = os.path.join(".", "logs", "tensorboard")

    # Warm up
    _ = induced_velocity(params)
    V_i, _ = induced_velocity(params)
    V_i.block_until_ready() # Ensure warm-up is totally finished

    # Profile
    jax.profiler.start_trace(log_dir)
    V_i, _ = induced_velocity(params)
    V_i.block_until_ready()
    jax.profiler.stop_trace()

    # Calculate total thrust
    T = thrust(V_i, params)
    print("Total Thrust:", T)

    actuatorEffiency = equivelantActuatorDiskEfficiency(T, params)
    print("Actuator efficiency:", actuatorEffiency)

    if plot == True:
        # Plot heatmap using structured params properties
        fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
        plt.contourf(params.psi, params.r, V_ia, cmap='viridis')
        plt.colorbar(label='Induced Velocity [m/s]')
        plt.xlabel('X [m]')
        plt.ylabel('Y [m]')
        plt.title('Induced Velocity over Propeller Disk')
        plt.show()

    return params, V_i, T

def apc_model(filepath, plot=True):
    '''
    Loads an APC propeller geometry from a .PE0 file, interpolates it onto the 
    solver's computational grid, solves the induced velocity, and plots the results.
    '''
    import queue
    import results
    import apcpropanalysis
    from matplotlib.animation import FuncAnimation

    print(f"Loading APC Propeller geometry from: {filepath}")
    
    # Load raw APC Data dictionary
    apc_geom = apcpropanalysis.parse_apc_pe0(filepath)
    
    # Unpack the specific geometry fields needed
    r_apc_in = apc_geom["r"]
    c_apc_in = apc_geom["chord"]
    beta_apc_deg = apc_geom["twist_deg"]
    airfoil_apc = "clarky"
    
    # Convert inches to meters and degrees to radians to match JAX solver SI units
    r_apc_m = jnp.array(r_apc_in) * 0.0254
    c_apc_m = jnp.array(c_apc_in) * 0.0254
    beta_apc_rad = jnp.radians(jnp.array(beta_apc_deg))

    # Configure default inputs to match the physical boundaries of the APC prop
    inputs = Inputs()
    inputs = eqx.tree_at(
        lambda i: (i.R_hub, i.R_tip), 
        inputs, 
        (r_apc_m.min(), r_apc_m.max())
    )
    
    # Build baseline structure
    params = build_default_propeller(inputs=inputs, airfoil_name=airfoil_apc)

    # Interpolate APC geometry onto the solver's discretized radial array
    c_interp = jnp.interp(params.r, r_apc_m, c_apc_m)
    beta_interp = jnp.interp(params.r, r_apc_m, beta_apc_rad)

    # Inject the mapped geometry into the Params tree
    params = eqx.tree_at(
        lambda p: (p.c, p.beta),
        params,
        (c_interp, beta_interp)
    )
    
    print("Solving Induced Velocity Field...")
    # Solve Induced Velocity
    V_ia, V_ia_0 = induced_velocity(params)
    
    # Update params with solved state
    params = eqx.tree_at(lambda p: p.V_ia_0, params, V_ia_0)

    # Calculate overall performance metrics
    T = thrust(V_ia, params)
    P_mech = mechanicalPower(V_ia, params)
    print(f"--- Solve Complete ---")
    print(f"Thrust: {T:.4f} N | Mechanical Power: {P_mech:.4f} W")

    # Plotting via LivePropPlotter
    if plot:
        print("Launching LivePropPlotter canvas...")
        gui_queue = queue.Queue()
        plotter = results.LivePropPlotter(params)
        tip_loss, root_loss = tip_root_loss_factors(V_ia, params)
        F_curr = jnp.multiply(tip_loss, root_loss)
        # Build the frame data dictionary expected by LivePropPlotter.update()
        frame_data = {
            'c': params.c * 1000,
            'beta': jnp.degrees(params.beta),
            'error_log': params.error_log,
            'V_ia': V_ia,
            'F_loss': F_curr  # Map this key to your Prandtl tip/root total loss array
        }
        gui_queue.put(frame_data)
        
        ani = FuncAnimation(
            plotter.fig, 
            plotter.update, 
            fargs=(gui_queue,), 
            interval=200, 
            cache_frame_data=False,
            blit=False

        )
        plt.show()

        fig = results.debug_diagnostics_plot(params, 0)

        dummy = input("Done with plots?")

    return params, V_ia, T

if __name__ == "__main__":

    flatblade(0.05, 0.1, 2)
#     import apcpropanalysis
#     apc_model("apc_prop_data/propgeom_data/20x8-PERF.PE0")
#     rpm_data = apcpropanalysis.parse_apc_dat("apc_prop_data/performance_data/PER3_20x8.dat")
#     # Extract as raw scalar floats instead of numpy arrays
#     target_rpm = float(np.squeeze(Inputs.omega / 0.1047))
#     target_V = float(np.squeeze(Inputs.V_x))

#     if rpm_data:
#         # Find the closest RPM block (broadcasting works flawlessly with scalar targets)
#         available_rpms = np.array(list(rpm_data.keys()))
#         closest_rpm = available_rpms[np.argmin(np.abs(available_rpms - target_rpm))]
        
#         # Extract the data matrix for the closest RPM
#         matrix = rpm_data[closest_rpm]
        
#         # Convert the file's velocity column (mph) to m/s
#         V_ms = matrix[:, 0] * 0.44704
        
#         # Find the closest velocity row within this specific RPM block
#         closest_v_idx = np.argmin(np.abs(V_ms - target_V))
        
#         # Extract performance metrics
#         matched_V = V_ms[closest_v_idx]
#         extracted_thrust = matrix[closest_v_idx, 10]  
#         extracted_eff = matrix[closest_v_idx, 2]      
#         extracted_power = matrix[closest_v_idx, 8]    

#         # --- 4. Print Summary Cleanly ---
#         print("\n" + "="*45)
#         print("      APC PERFORMANCE DATA EXTRACTION")
#         print("="*45)
#         print(f"Target Conditions:  {target_rpm:.2f} RPM  |  {target_V:.2f} m/s")
#         print(f"Closest APC Match:  {closest_rpm} RPM  |  {matched_V:.2f} m/s")
#         print("-"*45)
#         print(f"Extracted Thrust:   {extracted_thrust:.4f} N")
#         print(f"Extracted Efficiency: {extracted_eff * 100:.2f}% (Pe)")
#         print(f"Extracted Power:    {extracted_power:.2f} W")
#         print("="*45)
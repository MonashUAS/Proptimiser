## Define All Physical Properties for Given Propeller Geometry

# External Libraries
import jax
import jax.numpy as jnp
import equinox as eqx
import jaxopt
from dataclasses import replace

import numpy as np
import matplotlib.pyplot as plt

# Internal Libraries
from init import build_default_propeller, Inputs, Params
from airfoil import aero_coeffs
from airflow import tip_root_loss_factors, inflow_model, equivelantActuatorDiskEfficiency, resultantVelocityComponents, high_induction_correction

def minimumInducedLossCost(V_i, params):
    """
    Returns a DIMENSIONLESS cost measuring deviation from the Betz uniform-inflow
    condition (coefficient of variation, weighted toward the tip).
    Inputs:
        V_i: shape = (3, Nr, Npsi)

    TODO: Evaluate if depreciated with new power implementations, update to V_i
    """
    # Unpack Induced Velocity
    V_ia = V_i[0]
    # Global mean over disk (scalar)
    mean_V_ia = jnp.mean(V_ia)

    # Safe denominator: avoid divide-by-zero for near-unloaded blades
    mean_V_ia_safe = jnp.maximum(jnp.abs(mean_V_ia), 1e-3)

    # Normalised squared error at every (r, psi) station — dimensionless
    sq_err_normalised = ((V_ia - mean_V_ia) / mean_V_ia_safe) ** 2

    # Collapse azimuthal axis → radial profile (Nr,)
    radial_sq_err = jnp.mean(sq_err_normalised, axis=1)

    # Outer annuli carry more of the wake energy
    r_norm = params.r / jnp.max(params.r)
    weighted_sq_err = radial_sq_err * r_norm

    # RMS over radial stations — O(0.1–0.5) for a realistic propeller in forward flight
    rms_err = jnp.sqrt(jnp.mean(weighted_sq_err))

    return rms_err

def motorPower(torque: float, omega: float, I_0: float = 0.7, R_m: float = 0.120, kV: float = 150):
    """
    Calcualtes Power Consumption required of the motor to drive the propeller

    Inputs:
        torque: torque of the motor (Nm)
        omega: rotational rate of the motor (rad/s)
        I_0: zero load current (A)
        R_m: Coil resistance (ohms)
        kV: motor speed constant (RPM/volt)
    """
    P_out = torque*omega

    K_t = 60 / (2*jnp.pi*kV) # Calculate the torque constant (Nm/A)

    I = torque / K_t # calculate the current consumed by the motor

    V = omega*60/(2*jnp.pi) / kV

    P_in = P_out + R_m * (I+I_0)**2 + V*I_0

    return P_in

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

    # Introduce zero for non-solved components to maintain shape (3, n_r)

    V_i_0_guess_dim = jnp.zeros_like(params.V_i_0).at[Inputs.bemt_type].set(V_i_0_guess_dim)


    # Expand to 3D grid
    V_i = inflow_model(V_i_0_guess_dim, params)

    # Axial Residual (Scale by Thrust Target)
    dT_res_physical = thrust_residual(V_i, params)
    dT_res = dT_res_physical / params.thrust_target
    
    # Radial Residual (Already O(1))
    dR_res = lambda_i_0_guess[2] - 0.0

    # Tangential Residual (Scale by Approximate Ideal Torque)
    dQ_res_physical = torque_residual(V_i, params)

    # Scale by an approximate ideal torque to make O(1)
    Q_ref = params.thrust_target * Inputs.R_tip 
    dQ_res = dQ_res_physical / Q_ref

    # residual vector of shape (3, n_r)
    res = jnp.stack([dT_res, dQ_res, dR_res], axis=0)

    # filter out non-solved components
    res_filtered = res[Inputs.bemt_type]
    
    return res_filtered

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

    # Stack and reshape to (3, 1) so it broadcasts perfectly with (3, n_r)
    V_scale = jnp.array([V_ia_mean, V_it_mean, V_tip])[:, None]
    V_scale = V_scale[Inputs.bemt_type]  # Only keep the components that are being solved for
    return V_scale

@jax.jit
def induced_velocity(params):
    '''
    Calculates the coupled 3D induced velocity field using BEMT.
    '''
    V_i_0_guess_physical = params.V_i_0

    # Apply Induced Velocity Solver Type (remove non-solve components)
    V_i_0_guess_physical = V_i_0_guess_physical[Inputs.bemt_type]

    # Non-Dimensionalise Initial Guess Component-Wise
    V_scale = get_velocity_scales(params)
    initial_guess_lambda = V_i_0_guess_physical / V_scale

    # Broyden Solver Call
    solver = jaxopt.Broyden(
        fun=bemt_residuals, 
        maxiter=Inputs.V_i_0_max_iter, 
        tol=Inputs.V_i_0_eps,
        implicit_diff=True,
        implicit_diff_solve=jaxopt.linear_solve.solve_lu,
        jit = True,
        verbose = Inputs.verbose
    )
    
    (lambda_i_0, info) = solver.run(initial_guess_lambda, params)

    # Re-dimensionalise the converged solution
    V_i_0 = lambda_i_0 * V_scale

    # re-add non solver components to the solution vector
    V_i_0 = jnp.zeros_like(params.V_i_0).at[Inputs.bemt_type].set(V_i_0)

    if Inputs.verbose:
        final_res = bemt_residuals(lambda_i_0, params)
        jax.debug.print("Final Residuals: {res}", res=final_res)

    # Expand to 3D inflow grid
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
    V_i_a = V_i[0,:,:]
    # Calculate total thrust
    T = thrust(V_i, params)
    print("Total Thrust:", T)

    actuatorEffiency = equivelantActuatorDiskEfficiency(T, params)
    print("Actuator efficiency:", actuatorEffiency)

    if plot == True:
        # Plot heatmap using structured params properties
        fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
        plt.contourf(params.psi, params.r, V_i_a, cmap='viridis')
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
    
    # Solve Induced Velocity
    V_i, V_i_0 = induced_velocity(params)
    
    # Update params with solved state
    params = eqx.tree_at(lambda p: p.V_i_0, params, V_i_0)

    # Calculate overall performance metrics
    T = thrust(V_i, params)
    P_mech = mechanicalPower(V_i, params)

    # Plotting via LivePropPlotter
    if plot:
        print("Launching LivePropPlotter canvas...")
        gui_queue = queue.Queue()
        plotter = results.LivePropPlotter(params)

        gui_queue.put(params)
        
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

    return inputs, params, V_i_0, T

if __name__ == "__main__":
    params, V_i_0, T = apc_model("apc_prop_data/propgeom_data/20x8-PERF.PE0", plot=True)
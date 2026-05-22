import jax
# Set jax debugging
jax.config.update("jax_traceback_filtering", "quiet_remove_frames")
jax.config.update("jax_check_tracer_leaks", "off")
jax.config.update("jax_disable_jit", False)
jax.config.update("jax_debug_nans", False)
jax.config.update("jax_enable_x64", True)

# External Imports
import time
import jax.numpy as jnp
import jaxopt
import matplotlib.pyplot as plt
import scipy.optimize as scipyopt
from dataclasses import replace
from jax import lax
import equinox as eqx

# Internal Imports
from init import Params, Inputs
from airfoil import load_airfoil_data
import propmodel
from results import propinfoplot

def minimumInducedLossCost(V_ia_0, scale=1):
    """
    Returns the cost function of minimum induced loss, that is the RMS of the 
    variation of the induced velocity at the center of the propeller by each annulus
    """

    mean_V_ia = jnp.mean(V_ia_0)

    squared_err = (mean_V_ia - V_ia_0)**2

    rms_err = jnp.sqrt(jnp.sum(squared_err))

    return rms_err*scale

def motorPower(torque: float, omega: float, I_0: float = 0.7, R_m: float = 0.120, k_V: float = 150):
    """
    Calcualtes Power Consumption required of the motor to drive the propeller

    Inputs:
        torque: torque of the motor (Nm)
        omega: rotational rate of the motor (rad/s)
        I_0: zero load current (A)
        R_m: Coil resistance (ohms)
        K_v: motor speed constant (RPM/volt)
    """
    P_out = torque*omega

    K_t = 60 / (2*jnp.pi*k_V) # Calculate the torque constant (Nm/A)

    I = torque / K_t # calculate the current consumed by the motor

    V = omega*60/(2*jnp.pi) / k_V

    P_in = P_out + R_m * (I+I_0)**2 + V*I_0

    return P_in

@jax.jit
def objectives(params):
    '''
    Return the Base Objective Values of the Solver given an input of parameters
    Values of interest include power, thrust error, minimum induced loss error,
    '''
    # replace params object (works in function)
    V_ia, V_ia_0   = propmodel.induced_velocity(params)             # Solve induced velocity field for current guess
    thrust = jnp.squeeze(propmodel.thrust(V_ia, params))            # return the total thrust the propeller produce
    torque = jnp.squeeze(propmodel.torque(V_ia, params))            # return total torque propeller produces
    rms_MIL_err = minimumInducedLossCost(V_ia_0)                    # minimum induced loss condition root mean square error
    
    # calculate the motor power
    motor_power = motorPower(torque, params.omega, I_0 = Inputs.I_0, R_m = Inputs.R_m, k_V = Inputs.k_V)

    # Return V_ia_0 so we can catch it!
    return motor_power, thrust, rms_MIL_err, V_ia_0 

def omega_thrust_residual(omega_guess, params):
    '''
    Calculate thrust residual for Given Parameters and a guess of omega

    Returns:
        thrust_residual: current thrust residual for given parameters and rotational speed
    '''
    params = eqx.tree_at(lambda p: p.omega, params, omega_guess)
    V_ia, _ = propmodel.induced_velocity(params)
    thrust = propmodel.thrust(V_ia, params)

    return thrust-params.thrust_target

@jax.custom_vjp
def solve_omega(params):
    '''
    Root find the rotational velocity that matches the thrust residual exactly for ONE condition
    '''
    # We turn OFF implicit_diff in jaxopt here because we are manually defining the 1D gradient below!
    solver = jaxopt.Broyden(fun=omega_thrust_residual, maxiter=Inputs.omega_max_iter, tol=Inputs.omega_eps, implicit_diff=False)
    
    (omega, info) = solver.run(params.omega, params)
    return omega

# Define the Custom Vector Jacobian Product to Prevent NotImplementedErrors
def solve_omega_fwd(params):
    # Forward pass: Return the primal omega, and save (omega, params) for the backward pass
    omega = solve_omega(params)
    return omega, (omega, params)

def solve_omega_bwd(res, g):
    # Backward pass: Calculate exact 1D implicit differentiation
    omega, params = res
    
    # Derivative of residual w.r.t omega (a simple scalar)
    dR_domega = jax.grad(omega_thrust_residual, argnums=0)(omega, params)
    
    # Prevent division by zero if gradient goes flat
    dR_domega = jnp.where(jnp.abs(dR_domega) < 1e-10, 1e-10, dR_domega)
    
    # Scale the incoming outer gradient (g) by the implicit function theorem
    scalar_factor = -g / dR_domega
    
    # Pull the scaled gradient back through the parameters structure
    _, vjp_params = jax.vjp(lambda p: omega_thrust_residual(omega, p), params)
    grad_params = vjp_params(scalar_factor)[0]
    
    return (grad_params,)

# Bind the custom forward and backward passes to the solver
solve_omega.defvjp(solve_omega_fwd, solve_omega_bwd)

def cost_fn(params):
    ''' 
    Computes cost and returns updated V_ia_0 for exactly ONE condition 
    '''
    motor_power, thrust, rms_MIL_err, V_ia_0 = objectives(params)
    
    # Calculate base cost using this condition's specific targets
    cost = (
        params.objective_thrust * (thrust - params.thrust_target)**2 + 
        params.objective_MIL * rms_MIL_err**2 + 
        params.objective_power * motor_power**2
    )
    
    # Scale by the relative importance weight of this flight condition
    weighted_cost = cost * params.objective_weight
    return weighted_cost, V_ia_0


@jax.jit
def objectives_fn(opt_vars_scaled, params_list): 
    '''
    Vectorized objective function mapping over an ensemble of single-condition Params
    '''
    # Unpack and Scale Optimisation Variables
    c_scaled, beta_scaled = opt_vars_scaled
    c = params_list.c_scale[:, None] * c_scaled[None, :]
    beta = params_list.beta_scale[:, None] * beta_scaled[None, :]

    # Update parameters list with new geometry
    params_list = eqx.tree_at(lambda p: (p.c, p.beta), params_list, (c, beta))

    # vmap omega solving over parameters list
    solve_omega_vmap = jax.vmap(solve_omega, in_axes=0)
    omega = solve_omega_vmap(params_list)

    # Update parameters list with new rotational speeds
    params_list = eqx.tree_at(lambda p: p.omega, params_list, omega)

    # vmap cost function over parameters list
    cost_fn_vmap = jax.vmap(cost_fn, in_axes=0)
    costs, V_ia_0 = cost_fn_vmap(params_list)

    aux = (V_ia_0, omega)
    
    # Sum up total scalar cost across conditions, pass the array of updated V_ia_0 states as aux data
    return jnp.sum(costs), aux

def compute_gradient_scales(init_c, init_beta, params_static):
    '''
    Scale Gradients of variables using upper bounds

    CURRENTLY DEPRECIATED - JUST USE MAX BOUNDS
    '''
    
    c_scale = 0.1            # Max chord is 0.1m
    beta_scale = jnp.radians(50)  # Max twist is 50 degrees

    return c_scale, beta_scale

def run_optimization(params_base):
    '''
    The Meat and Bones of the Optimisation Algorithm.
    Iterate until solution is stable or criteria are met.
    '''
    # Debug Print to check initial thrust residual
    if Inputs.verbose:
        res_check = propmodel.thrust_residual(params_base.V_ia_0, params_base)
        jax.debug.print("Initial Residuals Check: {dT_res}", dT_res=res_check)

    vars_scaled = (
        params_base.c / params_base.c_scale,
        params_base.beta / params_base.beta_scale
    )

    bounds = (
        (jnp.full_like(params_base.c, 1e-4), jnp.full_like(params_base.beta, 0)),
        (jnp.full_like(params_base.c, 1), jnp.full_like(params_base.beta, 1))
    )

    conditions_list = [
        replace(
            params_base,
            V_x=Inputs.V_x[0], V_yz=Inputs.V_yz[0], omega=Inputs.omega[0],
            thrust_target=Inputs.thrust_target[0], objective_weight=Inputs.objective_weight[0],
            objective_thrust=1.0/Inputs.thrust_target[0], objective_MIL=1.0, objective_power=0.1/5000,
            V_ia_0=jnp.full((Inputs.n_r,), 4.0)
        ),
        replace(
            params_base,
            V_x=Inputs.V_x[1], V_yz=Inputs.V_yz[1], omega=Inputs.omega[1],
            thrust_target=Inputs.thrust_target[1], objective_weight=Inputs.objective_weight[1],
            objective_thrust=1.0/Inputs.thrust_target[1], objective_MIL=1.0, objective_power=0.1/5000,
            V_ia_0=jnp.full((Inputs.n_r,), 4.0)
        )
    ]
    
    # Convert list into a stacked PyTree object that JAX natively unrolls via vmap
    params_struct = jax.tree_util.tree_map(lambda *args: jnp.stack(args), *conditions_list)

    # Start the clock
    start_time = time.time()

    # Initialise Timings and Log Arrays
    iter_time = start_time+10
    print_time = iter_time

    error_log = jnp.array([1.0])
    time_log = jnp.array([0])

    ## Solver Selection - Full solver regime must be specified here since many solvers have different inputs
    if Inputs.opt_solver == "L-BFGS-B":
        solver = jaxopt.LBFGSB(
            fun=objectives_fn, 
            has_aux=True,
            implicit_diff=True, 
            jit=True
        )
        print("Initialising Solver...")
        state = solver.init_state(init_params=vars_scaled, bounds=bounds, params_list=params_struct)
        print(f"{'Iter':<6} | {'Cond':<5} | {'Power':<12} | {'Thrust':<12} | {'T-Err':<12} | {'MIL':<10} | {'OptErr':<11} | {'Time'}")
        print("-" * 90)

        for i in range(Inputs.opt_max_iter): 
            vars_scaled, state = solver.update(vars_scaled, 
                                               state=state, 
                                               bounds=bounds, 
                                               params_list=params_struct)
            # state.aux captures the updated tracking profiles (shape: [2, n_r])
            # We assign them directly back to keep Newton warm-starts efficient

            # Scale Variables
            c_scaled, beta_scaled = vars_scaled    
            c = params_struct.c_scale[:, None] * c_scaled[None, :]
            beta = params_struct.beta_scale[:, None] * beta_scaled[None, :]
            # Extract the updated V_ia_0, omega from state.aux and update params_struct
            V_ia_0, omega = state.aux
            params_struct = eqx.tree_at(lambda p: (p.c, p.beta, p.omega, p.V_ia_0), params_struct, (c, beta, omega, V_ia_0))

            # Print Statements for this Iteration
            is_terminal = state.error < Inputs.opt_tol
            is_checkpoint = (time.time() - print_time > 5) or (i % 5 == 0)

            if is_terminal or is_checkpoint:
                vmapped_objectives = jax.vmap(objectives, in_axes=0)
                powers, thrusts, rms_MIL_errs, _ = vmapped_objectives(params_struct)
                
                for idx, (pwr, thr, mil) in enumerate(zip(powers, thrusts, rms_MIL_errs)):
                    t_err = thr - params_struct.thrust_target[idx]
                    
                    # Only print iteration metadata on the first condition row
                    iter_info = f"{i:<6} |" if idx == 0 else f"{'':<6} |"
                    opt_err = f"{state.error:<11.3f}" if idx == 0 else f"{'':<11}"
                    timing = f"{time.time() - iter_time:<4.1f}s" if idx == 0 else ""
                    
                    print(f"{iter_info} C{idx:<3} | {pwr:<12.4f} | {thr:<12.4f} | {t_err:<12.4f} | {mil:<10.4f} | {opt_err} | {timing}")

                print_time = time.time()
                
                if is_terminal:
                    print(f"\nStopping: Convergence reached at step {i}")
                    break

            # Update Error Log, Iter Time
            error_log = jnp.append(error_log, state.error)
            time_log = jnp.append(time_log, time.time() - iter_time)
            iter_time = time.time()

    else:
        # Final terminal backup evaluation if max iterations is hit
        vmapped_objectives = jax.vmap(objectives, in_axes=0)
        _, thrusts, _, _ = vmapped_objectives(params_struct)
        for cond_idx in range(len(thrusts)):
            err = float(thrusts[cond_idx] - params_struct.thrust_target[cond_idx])
            print(f"Warning: max iterations reached. Final thrust error [Cond {cond_idx}]: {err:.2e}")
    
    # Clock Time of solve
    solve_time = time.time() - start_time
    print(f"\nTotal Solve Time: {solve_time:.2f} seconds")

    # Extract 1D base slice representations of the final arrays
    final_c = params_struct.c[0]
    final_beta = params_struct.beta[0]
    final_omega = params_struct.omega[0]
    final_V_ia_0 = params_struct.V_ia_0[0]

    return replace(params_base, 
                   c=final_c, 
                   beta=final_beta, 
                   omega=final_omega,
                   V_ia_0=final_V_ia_0,
                   error_log=error_log, 
                   time_log=time_log)

if __name__ == "__main__":
    # Enable LaTeX formatting for pretty titles
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "axes.labelsize": 12,
        "axes.titlesize": 14
    })

    # Complete Optimisation Run
    params_opt = run_optimization()

    # Save/Export Solver State

    # Plotting
    propinfoplot(params_opt)
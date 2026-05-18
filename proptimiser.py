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

# Internal Imports
from init import Params, Inputs
from airfoil import load_airfoil_data
import propmodel
from results import propinfoplot

# Curvature Penalty (depreciated)
def curvature_penalty(x):
    d2 = x[2:] - 2.0 * x[1:-1] + x[:-2]
    return jnp.mean(d2**2)

def minimumInducedLossCost(V_ia_0, scale=1):
    """
    Returns the cost function of minimum induced loss, that is the RMS of the 
    variation of the induced velocity at the center of the propeller by each annulus
    """

    mean_V_ia = jnp.mean(V_ia_0)

    squared_err = (mean_V_ia - V_ia_0)**2

    rms_err = jnp.sqrt(jnp.sum(squared_err))

    return rms_err*scale

def objectives(params):
    '''
    Return the Base Objective Values of the Solver given an input of parameters
    Values of interest include power, thrust error, minimum induced loss error,
    '''
    # replace params object (works in function)
    V_ia, V_ia_0   = propmodel.induced_velocity(params)             # Solve induced velocity field for current guess
    
    power  = jnp.squeeze(propmodel.mechanicalPower(V_ia, params))   # return the total power produced by the propeller
    thrust = jnp.squeeze(propmodel.thrust(V_ia, params))            # return the total thrust the propeller produces

    thrust_err = (thrust - Inputs.thrust)           # thrust serror for this iteration

    rms_MIL_err = minimumInducedLossCost(V_ia_0)    # minimum induced loss 

    # Return V_ia_0 so we can catch it!
    return power, thrust_err, thrust, rms_MIL_err, V_ia_0 

@jax.jit
def objectives_fn(opt_vars_scaled, state_params): 
    '''
    Objectives Function for the total weight of the objective
    add together all target values with toggles dependent on
    input solver settings.
    ''' 
    # TODO Change below logic so the "optimisation filter"
    # decides which variables are to be optimised.

    # Scale Variables
    c_scaled, beta_scaled = opt_vars_scaled
    c = c_scaled*state_params.c_scale
    beta = beta_scaled*state_params.beta_scale

    # Update params with new geometry
    state_params = replace(state_params, c=c, beta=beta)

    # return the objective function dependent on objective states
    power, thrust_err, thrust, rms_MIL_err, V_ia_0 = objectives(state_params)


    cost = state_params.objective_thrust*thrust_err**2 + state_params.objective_MIL*rms_MIL_err**2 + state_params.objective_power*power**2

    # Return (cost, aux_data)
    return cost, V_ia_0

def compute_gradient_scales(init_c, init_beta, params_static):
    '''
    Scale Gradients of variables using upper bounds

    CURRENTLY DEPRECIATED - JUST USE MAX BOUNDS
    '''
    
    c_scale = 0.1            # Max chord is 0.1m
    beta_scale = jnp.radians(50)  # Max twist is 50 degrees

    return c_scale, beta_scale

def run_optimization(params_init):
    '''
    The Meat and Bones of the Optimisation Algorithm.
    Iterate until solution is stable or criteria are met.
    '''
    # Debug Print to check initial thrust residual
    if Inputs.verbose:
        res_check = propmodel.thrust_residual(params_init.V_ia_0, params_init)
        jax.debug.print("Initial Residuals Check: {dT_res}", dT_res=res_check)

    # Scale initial variables
    vars_scaled = (
        params_init.c/params_init.c_scale,
        params_init.beta/params_init.beta_scale
    )

    # Scaled Bounds (chord + beta pairs)
    bounds = (
        (   jnp.full_like(params_init.c, 1e-4),
            jnp.full_like(params_init.beta, 0),    # optimal angle may be negative
        ),
        (   jnp.full_like(params_init.c, 1),
            jnp.full_like(params_init.beta, 1),
        ))

    # Solver title/header
    print(f"{'Iter':<6} | {'Power':<12} | {'Thrust (N)':<12} | {'Thrust Err':<12} | {'MIL rms':<10} |{'State Error':<6} | {'Iter Time (s)'}")
    print("-" * 95)

    # Start up the parameters
    params = params_init

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
            has_aux=True,      # allows us to parse V_ia_0
            implicit_diff=True, 
            jit=True
        )
        # Bundle parameters into solver state
        state = solver.init_state(
            init_params=vars_scaled, # solver state has to be defined
            bounds=bounds, 
            state_params=params_init
        )
        for i in range(Inputs.opt_max_iter): 

            vars_scaled, state = solver.update(
                vars_scaled,
                state=state,
                bounds=bounds,
                state_params=params
            )
            
            # Scale Variables
            c_scaled, beta_scaled = vars_scaled
            c = c_scaled*params.c_scale
            beta = beta_scaled*params.beta_scale

            # Extract the updated V_ia_0 from state.aux and update params_static
            params = replace(params, V_ia_0=state.aux, c=c, beta=beta)

            if state.error < Inputs.opt_tol: 
                # stop condition first
                power, thrust_err, thrust, rms_MIL_err, _ = objectives(params)
                print(f"{i:<6} | {float(power):<12.4f} | {float(thrust):<12.4f} | {float(thrust_err):<12.4f} | {rms_MIL_err:<10.4f} | {state.error:10.4f} | {time.time()-iter_time:3.1f}")
                print(f"Stopping: Gradient stationarity reached at step {i}")
                break
            
            if (time.time()-print_time > 5) | (i%5==0):
                # Print Solver State Every Increment or Five seconds, whichever comes first
                power, thrust_err, thrust, rms_MIL_err, _ = objectives(params)
                print(f"{i:<6} | {float(power):<12.4f} | {float(thrust):<12.4f} | {float(thrust_err):<12.4f} | {rms_MIL_err:<10.4f} | {state.error:10.4f} | {time.time()-iter_time:3.1f}")
                #if Inputs.verbose==True:
                # res_check = propmodel.thrust_residual(state.aux, params_static)
                # jax.debug.print("Residuals Check: {dT_res}", dT_res=res_check)
                print_time = time.time()

            # Update Error Log, Iter Time
            error_log = jnp.append(error_log, state.error)
            time_log = jnp.append(time_log, time.time()-iter_time)
            iter_time = time.time()

    elif Inputs.opt_solver == "scipy": # Not currently implemented
            solver = scipyopt.minimise(
            fun=objectives_fn, 
            has_aux=True,      # allows us to parse V_ia_0
            implicit_diff=True, 
            jit=True
        )

    else:
        print(f"\nWarning: max iterations reached. Final thrust error: {thrust_err:.2e}")
    
    # Clock Time of solve
    solve_time = time.time()-start_time
    print(f"\nTotal Solve Time: {solve_time}")

    return replace(params, c=vars_scaled[0]*params.c_scale, beta=vars_scaled[1]*params.beta_scale, error_log=error_log, time_log=time_log)

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
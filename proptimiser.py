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
from jax.scipy.special import gammaln

# Bezier Curve Function for Smoothing

def jax_comb(n, k):
    """Continuous binomial coefficient for JAX differentiability."""
    return jnp.exp(gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1))

def get_bezier_basis_matrix(r, R_hub, R_tip, n_cp):
    t = (r - R_hub) / (R_tip - R_hub)
    n = n_cp - 1
    i = jnp.arange(n + 1)
    t_expand = t[:, None]
    binom = jax_comb(n, i)
    basis = binom * (t_expand ** i) * ((1 - t_expand) ** (n - i))
    return basis # Shape: (n_r, n_cp)

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

def cost_fn(params):
    ''' 
    Computes cost and returns updated V_ia_0 for exactly ONE condition 
    '''
    # unpack objectives and induced velocity
    motor_power, thrust, rms_MIL_err, V_ia_0 = objectives(params)
    # calculate scaled variables
    thrust_err_rel = (thrust - params.thrust_target)/params.thrust_target
    # efficiency compared to actuator disk
    actuator_disk_power = propmodel.actuator_disk_power(params.thrust_target, params)
    efficiency = motor_power/actuator_disk_power

    # Calculate base cost using this condition's specific targets
    cost = (
        params.objective_thrust * thrust_err_rel**2 + 
        params.objective_MIL * rms_MIL_err**2 + 
        params.objective_power * efficiency**2
    )
    
    # Scale by the relative importance weight of this flight condition
    weighted_cost = cost * params.objective_weight
    return weighted_cost, V_ia_0


@jax.jit
def objectives_fn(opt_vars_scaled, params_list, spline_matrix): 
    '''
    Vectorized objective function mapping over an ensemble of single-condition Params
    '''
    # Unpack Scaled Control Points    
    c_scaled, beta_scaled, omega_scaled = opt_vars_scaled
    
    if Inputs.opt_smoothing == "bezier":
        c_cp = params_list.c_scale[:, None] * c_scaled[None, :]
        beta_cp = params_list.beta_scale[:, None] * beta_scaled[None, :]
        c = jnp.dot(c_cp, spline_matrix.T)
        beta = jnp.dot(beta_cp, spline_matrix.T)
    else:
        # Direct pointwise variable assignment
        c = params_list.c_scale[:, None] * c_scaled[None, :]
        beta = params_list.beta_scale[:, None] * beta_scaled[None, :]
        # Keep PyTree structure identical
        c_cp = params_list.c_cp
        beta_cp = params_list.beta_cp

    omega = lax.switch(Inputs.solve_omega, [lambda _: params_list.omega, lambda _: params_list.omega_max * omega_scaled], None)

    # Update parameters list with new geometry and control points
    params_list = eqx.tree_at(
        lambda p: (p.c, p.beta, p.c_cp, p.beta_cp, p.omega), 
        params_list, 
        (c, beta, c_cp, beta_cp, omega)
    )

    # vmap cost function over parameters list
    cost_fn_vmap = jax.vmap(cost_fn, in_axes=0)
    costs, V_ia_0 = cost_fn_vmap(params_list)

    aux = (V_ia_0)
    return jnp.sum(costs), aux

@jax.jit
def evaluate_all_conditions(params_struct):
    vmapped_obj = jax.vmap(objectives, in_axes=0)
    return vmapped_obj(params_struct)

def run_optimization(params_base):
    '''
    The Meat and Bones of the Optimisation Algorithm.
    Iterate until solution is stable or criteria are met.
    '''
    # Debug Print to check initial thrust residual
    if Inputs.verbose:
        res_check = propmodel.thrust_residual(params_base.V_ia_0, params_base)
        jax.debug.print("Initial Residuals Check: {dT_res}", dT_res=res_check)

    conditions_list = [
    replace(
            params_base,
            V_x=Inputs.V_x[i],
            V_yz=Inputs.V_yz[i],
            omega=Inputs.omega[i],
            thrust_target=Inputs.thrust_target[i],
            objective_weight=Inputs.objective_weight[i],
            V_ia_0=jnp.full((Inputs.n_r,), 4.0)
        )
        for i in range(Inputs.n_cond)
    ]
    
    # Convert list into a stacked PyTree object that JAX natively unrolls via vmap
    params_struct = jax.tree_util.tree_map(lambda *args: jnp.stack(args), *conditions_list)

    # Define optimisation variables and bounds
    if Inputs.opt_smoothing == "bezier":
        vars_scaled = (
            params_base.c_cp / params_base.c_scale,
            params_base.beta_cp / params_base.beta_scale,
            params_struct.omega / params_base.omega_max # Shape (2,)
        )
    elif Inputs.opt_smoothing == "pointwise":
        vars_scaled = (
            params_base.c / params_base.c_scale,
            params_base.beta / params_base.beta_scale,
            params_struct.omega / params_base.omega_max # Shape (2,)
        )
    else:
        raise ValueError("Inputs.opt_smoothing must be 'bezier' or 'pointwise'")

    lower_bounds = (
        jnp.full_like(vars_scaled[0], 1e-4),
        jnp.full_like(vars_scaled[1], 0.0),
        jnp.full_like(vars_scaled[2], 300/params_base.omega_max)
    )
    
    upper_bounds = (
        jnp.full_like(vars_scaled[0], 1.0),
        jnp.full_like(vars_scaled[1], 1.0),
        jnp.full_like(vars_scaled[2], 1.0)
    )
    
    bounds = (lower_bounds, upper_bounds)

    # Set up control point to spline matrix
    spline_matrix = get_bezier_basis_matrix(params_struct.r[0], Inputs.R_hub, Inputs.R_tip, Inputs.n_cp)

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
        state = solver.init_state(init_params=vars_scaled, bounds=bounds, params_list=params_struct, spline_matrix=spline_matrix)
        print(f"{'Iter':<6} | {'Cond':<5} | {'Omega':<10} | {'Power':<12} | {'Thrust':<12} | {'T-Err':<12} | {'MIL':<10} | {'OptErr':<11} | {'Time'}")
        print("-" * 103)
        try:
            for i in range(Inputs.opt_max_iter): 
                vars_scaled, state = solver.update(vars_scaled, 
                                                state=state, 
                                                bounds=bounds, 
                                                params_list=params_struct,
                                                spline_matrix=spline_matrix
                                                )
                # state.aux captures the updated tracking profiles (shape: [2, n_r])
                # We assign them directly back to keep Newton warm-starts efficient

                c_scaled, beta_scaled, omega_scaled = vars_scaled
                # toggle for omega
                omega = lax.switch(Inputs.solve_omega, [lambda _: params_struct.omega, lambda _: params_struct.omega_max * omega_scaled], None)
                
                if Inputs.opt_smoothing == "bezier":
                    c_cp = params_struct.c_scale[:, None] * c_scaled[None, :]
                    beta_cp = params_struct.beta_scale[:, None] * beta_scaled[None, :]
                    c = jnp.dot(c_cp, spline_matrix.T)
                    beta = jnp.dot(beta_cp, spline_matrix.T)
                else: # pointwise
                    c = params_struct.c_scale[:, None] * c_scaled[None, :]
                    beta = params_struct.beta_scale[:, None] * beta_scaled[None, :]
                    c_cp = params_struct.c_cp # Pass through dummy CPs to keep eqx.tree_at happy
                    beta_cp = params_struct.beta_cp

                # Extract the updated V_ia_0 from state.aux and update params_struct
                V_ia_0 = state.aux
                params_struct = eqx.tree_at(
                    lambda p: (p.c, p.beta, p.c_cp, p.beta_cp, p.omega, p.V_ia_0), 
                    params_struct, 
                    (c, beta, c_cp, beta_cp, omega, V_ia_0)
                )

                # Print Statements for this Iteration
                is_terminal = state.error < Inputs.opt_tol
                is_checkpoint = (time.time() - print_time > 5) or (i % 5 == 0)

                if is_terminal or is_checkpoint:
                    powers, thrusts, rms_MIL_errs, _ = evaluate_all_conditions(params_struct)
                    
                    for idx, (pwr, thr, mil) in enumerate(zip(powers, thrusts, rms_MIL_errs)):
                        t_err = thr - params_struct.thrust_target[idx]
                        omega_cond = omega[idx]
                        # Only print iteration metadata on the first condition row
                        iter_info = f"{i:<6} |" if idx == 0 else f"{'':<6} |"
                        opt_err = f"{state.error:<11.3f}" if idx == 0 else f"{'':<11}"
                        timing = f"{time.time() - iter_time:<4.1f}s" if idx == 0 else ""
                        
                        print(f"{iter_info} C{idx:<3} | {omega_cond:<5.2f} | {pwr:<12.4f} | {thr:<12.4f} | {t_err:<12.4f} | {mil:<10.4f} | {opt_err} | {timing}")

                    print_time = time.time()
                    
                    if is_terminal:
                        print(f"\nStopping: Convergence reached at step {i}")
                        break

                # Update Error Log, Iter Time
                error_log = jnp.append(error_log, state.error)
                time_log = jnp.append(time_log, time.time() - iter_time)
                iter_time = time.time()
        except KeyboardInterrupt:
            print(f"\n[Interrupt] Optimization paused manually by user at iteration {i}.")
            print("Extracting current state variables and returning progress...")
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

        # CHANGED: Final return extraction
        final_c = params_struct.c[0]
        final_beta = params_struct.beta[0]
        final_c_cp = params_struct.c_cp[0]          
        final_beta_cp = params_struct.beta_cp[0]   
        final_omega = params_struct.omega[0]
        final_V_ia_0 = params_struct.V_ia_0[0]

        return replace(params_base, 
                    c=final_c, 
                    beta=final_beta, 
                    c_cp=final_c_cp,               
                    beta_cp=final_beta_cp,          
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
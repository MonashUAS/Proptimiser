# External Imports
import jax
import time
import jax.numpy as jnp
import jaxopt
import matplotlib.pyplot as plt
from jax.flatten_util import ravel_pytree
import optimistix as optx
from slsqp_jax import SLSQP, SLSQPConfig, ToleranceConfig
from dataclasses import replace
from jax import lax
import equinox as eqx

# Internal Imports
from init import Params, Inputs
from airfoil import load_airfoil_data
import airflow
import propmodel
from jax.scipy.special import gammaln



def minimumInducedLossCost(V_ia, params):
    """
    Returns a DIMENSIONLESS cost measuring deviation from the Betz uniform-inflow
    condition (coefficient of variation, weighted toward the tip).

    V_ia shape: (Nr, Npsi)
    """
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


def objectives(params):
    '''
    Return the Base Objective Values of the Solver given an input of parameters
    Values of interest include power, thrust error, minimum induced loss error,
    '''
    # replace params object (works in function)
    V_ia, V_ia_0   = propmodel.induced_velocity(params)             # Solve induced velocity field for current guess
    thrust = jnp.squeeze(propmodel.thrust(V_ia, params))            # return the total thrust the propeller produce
    torque = jnp.squeeze(propmodel.torque(V_ia, params))            # return total torque propeller produces
    rms_MIL_err = minimumInducedLossCost(V_ia, params)            # minimum induced loss condition root mean square error
    
    # calculate the motor power
    motor_power = propmodel.motorPower(torque, params.omega, I_0 = Inputs.I_0, R_m = Inputs.R_m, k_V = Inputs.k_V)

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
        (params.objective_thrust * thrust_err_rel)**2 + 
        (params.objective_MIL * rms_MIL_err)**2 + 
        (params.objective_power * efficiency)**2
    )
    
    # Scale by the relative importance weight of this flight condition
    weighted_cost = cost * params.objective_weight
    return weighted_cost, V_ia_0

def objectives_fn(opt_vars_scaled, params_list): 
    '''
    Vectorized objective function mapping over an ensemble of single-condition Params
    '''
    # Unpack Scaled Control Points    
    c_scaled, beta_scaled, omega_scaled = opt_vars_scaled
    
    # Direct pointwise variable assignment
    c = params_list.c_scale[:, None] * c_scaled[None, :]
    beta = params_list.beta_scale[:, None] * beta_scaled[None, :]

    omega = lax.switch(Inputs.solve_omega, [lambda _: params_list.omega, lambda _: params_list.omega_max * omega_scaled], None)

    # Update parameters list with new geometry and control points
    params_list = eqx.tree_at(
        lambda p: (p.c, p.beta, p.omega), 
        params_list, 
        (c, beta, omega)
    )

    # vmap cost function over parameters list
    cost_fn_vmap = jax.vmap(cost_fn, in_axes=0)
    costs, V_ia_0 = cost_fn_vmap(params_list)

    aux = (V_ia_0)
    return jnp.sum(costs), aux

def apply_vars_to_struct(vars_scaled, params_struct):
    """
    Helper function to apply scaled variables to the batched parameter structure.
    """
    c_scaled, beta_scaled, omega_scaled = vars_scaled
    
    # Direct pointwise variable assignment
    c = params_struct.c_scale[:, None] * c_scaled[None, :]
    beta = params_struct.beta_scale[:, None] * beta_scaled[None, :]

    omega = omega_scaled * params_struct.omega_max

    # Update parameters list with new geometry and control points
    return replace(params_struct, c=c, beta=beta, omega=omega)

def slsqp_objective(vars_scaled, params_struct):
    '''
    Pure performance cost function (Power & MIL) for the slsqp solver.
    '''
    # Debug: Print shapes of input variables
    jax.debug.print("Objective - vars_scaled shapes: {x}", 
                    x=jax.tree_util.tree_map(jnp.shape, vars_scaled))

    # Pass PyTree directly to the struct applier
    params_updated = apply_vars_to_struct(vars_scaled, params_struct)
    
    # Evaluate objectives across all conditions
    motor_power, _, rms_MIL_err, _ = evaluate_all_conditions(params_updated)
    
    # Debug: Print shape of outputs
    jax.debug.print("Objective - motor_power shape: {m}, rms_MIL_err shape: {r}", 
                    m=jnp.shape(motor_power), r=jnp.shape(rms_MIL_err))

    # Calculate efficiency compared to actuator disk
    actuator_disk_power = jax.vmap(propmodel.actuator_disk_power)(params_updated.thrust_target, params_updated)
    efficiency = motor_power / actuator_disk_power
    
    # Optimize strictly for power efficiency and minimum induced loss
    cost = (
        params_updated.objective_MIL * rms_MIL_err**2 + 
        params_updated.objective_power * efficiency**2
    )
    
    # Scale by the relative importance weight of this flight condition
    weighted_cost = cost * params_updated.objective_weight
    
    total_cost = jnp.sum(weighted_cost)
    jax.debug.print("Objective - final cost shape: {s}", s=jnp.shape(total_cost))
    
    return total_cost

def slsqp_eq_constraints(vars_scaled, params_struct):
    '''
    Equality constraints for slsqp-jax. Returns an array of relative thrust errors.
    '''
    # Debug: Print shapes of input variables
    jax.debug.print("Constraint - vars_scaled shapes: {x}", 
                    x=jax.tree_util.tree_map(jnp.shape, vars_scaled))

    params_updated = apply_vars_to_struct(vars_scaled, params_struct)
    
    # We only need thrust for the constraint
    _, thrust, _, _ = evaluate_all_conditions(params_updated)
    
    # Relative thrust error (Shape: (n_cond,))
    thrust_err_rel = (thrust - params_updated.thrust_target) / params_updated.thrust_target
    
    # Debug: Print shape of constraint output
    jax.debug.print("Constraint - thrust_err_rel shape: {s}", s=jnp.shape(thrust_err_rel))
    
    return thrust_err_rel

@jax.jit
def evaluate_all_conditions(params_struct):
    vmapped_obj = jax.vmap(objectives, in_axes=0)
    return vmapped_obj(params_struct)

def get_manual_unflatten(c_shape, b_shape, o_shape):
    """
    Returns a closure that unflattened flat_x into the required tuple.
    We pass shapes as arguments to keep the closure clean for JIT.
    """
    c_size = int(jnp.prod(jnp.array(c_shape)))
    b_size = int(jnp.prod(jnp.array(b_shape)))
    o_size = int(jnp.prod(jnp.array(o_shape)))
    
    def manual_unflatten(flat_x):
        # 1. Pointer arithmetic for slicing
        # Ensure we are working with concrete integers
        c_end = c_size
        b_end = c_end + b_size
        
        # 2. Extract and Reshape
        c_new = flat_x[:c_end].reshape(c_shape)
        beta_new = flat_x[c_end:b_end].reshape(b_shape)
        omega_new = flat_x[b_end:].reshape(o_shape)
        
        return (c_new, beta_new, omega_new)
    
    return manual_unflatten

def run_optimization(params_base, update_queue=None):
    '''
    The Meat and Bones of the Optimisation Algorithm.
    Iterate until solution is stable or criteria are met.
    '''
    # Debug Print to check initial thrust residual
    if Inputs.verbose:
        res_check = propmodel.thrust_residual(params_base.V_ia_0, params_base)
        jax.debug.print("Initial Residuals Check: {dT_res}", dT_res=res_check)

    ## Construct params_struct for all flight conditions
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

    vars_scaled = (
        params_base.c / params_base.c_scale,
        params_base.beta / params_base.beta_scale,
        params_struct.omega / params_base.omega_max # Shape (2,)
    )

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

    # Start the clock
    start_time = time.time()

    # Initialise Timings and Log Arrays
    iter_time = start_time+10
    print_time = iter_time

    error_log = jnp.array([1.0])
    time_log = jnp.array([0])


    if Inputs.opt_solver == "SLSQP":
        print("Initialising SciPy SLSQP Solver...")
        import scipy.optimize as spo
        import numpy as np
        from jax.flatten_util import ravel_pytree
        
        # 1. FLATTEN STATE AND BOUNDS FOR SCIPY
        flat_vars, unflatten_fn = ravel_pytree(vars_scaled)
        flat_lower, _ = ravel_pytree(lower_bounds)
        flat_upper, _ = ravel_pytree(upper_bounds)
        
        # SciPy expects bounds as a list of tuples or a Bounds object
        bounds_scipy = list(zip(np.array(flat_lower), np.array(flat_upper)))
        flat_vars_np = np.array(flat_vars, dtype=np.float64)

        # 2. DEFINE MUTABLE CONTAINER FOR PYTHON STATE & WARM-STARTS
        state_container = {
            "params_struct": params_struct,
            "last_jac": None
        }

        # 3. DEFINE JIT-COMPILED CORE JAX FUNCTIONS
        @jax.jit
        def compute_obj_val_and_grad(flat_x, current_params):
            def f(x):
                vars_scl = unflatten_fn(x)
                params_updated = apply_vars_to_struct(vars_scl, current_params)
                motor_power, _, rms_MIL_err, final_V_ia_0 = evaluate_all_conditions(params_updated)
                
                actuator_disk_power = jax.vmap(propmodel.actuator_disk_power)(params_updated.thrust_target, params_updated)
                efficiency = motor_power / actuator_disk_power
                
                cost = (
                    params_updated.objective_MIL * rms_MIL_err**2 + 
                    params_updated.objective_power * efficiency**2
                )
                weighted_cost = cost * params_updated.objective_weight
                total_cost = jnp.sum(weighted_cost)
                return total_cost, final_V_ia_0

            (cost, final_V_ia_0), grad = jax.value_and_grad(f, has_aux=True)(flat_x)
            return cost, grad, final_V_ia_0

        @jax.jit
        def compute_con_val_and_jac(flat_x, current_params):
            def c_fn(x):
                vars_scl = unflatten_fn(x)
                params_updated = apply_vars_to_struct(vars_scl, current_params)
                _, thrust, _, final_V_ia_0 = evaluate_all_conditions(params_updated)
                thrust_err_rel = (thrust - params_updated.thrust_target) / params_updated.thrust_target
                return thrust_err_rel, final_V_ia_0

            val, final_V_ia_0 = c_fn(flat_x)
            # Shared graph computation via XLA means c_fn evaluates efficiently here
            jac = jax.jacobian(lambda x: c_fn(x)[0])(flat_x)
            return val, jac, final_V_ia_0

        # 4. DEFINE SCIPY-COMPATIBLE WRAPPERS (Converts JAX types to NumPy)
        def scipy_objective(flat_x_np):
            flat_x = jnp.array(flat_x_np, dtype=jnp.float64)
            current_params = state_container["params_struct"]
            cost, grad, final_V_ia_0 = compute_obj_val_and_grad(flat_x, current_params)
            
            # Catch the updated inner-solver state and feed it forward as a warm start
            state_container["params_struct"] = eqx.tree_at(
                lambda p: p.V_ia_0, current_params, final_V_ia_0
            )
            return float(cost), np.array(grad, dtype=np.float64)

        def scipy_constraint(flat_x_np):
            flat_x = jnp.array(flat_x_np, dtype=jnp.float64)
            current_params = state_container["params_struct"]
            val, jac, final_V_ia_0 = compute_con_val_and_jac(flat_x, current_params)
            
            # Sync the warm-start state and cache the Jacobian array 
            state_container["params_struct"] = eqx.tree_at(
                lambda p: p.V_ia_0, current_params, final_V_ia_0
            )
            state_container["last_jac"] = jac
            return np.array(val, dtype=np.float64)

        def scipy_constraint_jac(flat_x_np):
            # Eliminates duplicate execution by returning the pre-cached Jacobian 
            return np.array(state_container["last_jac"], dtype=np.float64)

        # 5. DEFINE ITERATION CALLBACK FOR LIVE TERMINAL DISPLAY
        iteration_counter = [0]
        print(f"\n{'Iter':<5} | {'Cond':<5} | {'Omega':<10} | {'Power':<12} | {'Thrust':<12} | {'T-Err':<12} | {'MIL':<10} | {'Time'}")
        print("-" * 90)

        def scipy_callback(xk):
            iteration_counter[0] += 1
            flat_x = jnp.array(xk, dtype=jnp.float64)
            current_params = state_container["params_struct"]
            vars_scl = unflatten_fn(flat_x)
            params_updated = apply_vars_to_struct(vars_scl, current_params)
            
            powers, thrusts, rms_MIL_errs, _ = evaluate_all_conditions(params_updated)
            
            for idx, (pwr, thr, mil) in enumerate(zip(powers, thrusts, rms_MIL_errs)):
                t_err = thr - params_updated.thrust_target[idx]
                omega_cond = params_updated.omega[idx]
                iter_str = f"{iteration_counter[0]:<5}" if idx == 0 else f"{'':<5}"
                timing = f"{time.time() - start_time:<4.1f}s" if idx == 0 else ""
                print(f"{iter_str} | C{idx:<3} | {omega_cond:<10.2f} | {pwr:<12.4f} | {thr:<12.4f} | {t_err:<12.4f} | {mil:<10.4f} | {timing}")
            print("-" * 90)
            if update_queue is not None:
                c_np = np.array(params_updated.c[0]) * 1000
                beta_np = np.degrees(np.array(params_updated.beta[0]))
                
                # Pass current error log structure
                current_errors = np.array(error_log) 
                
                V_ia_0_curr = params_updated.V_ia_0[0]
                V_ia_curr = np.array(propmodel.inflow_model(V_ia_0_curr, params_base))
                
                update_queue.put({
                    'c': c_np,
                    'beta': beta_np,
                    'error_log': current_errors,
                    'V_ia': V_ia_curr
                })
        # 6. EXECUTE SCIPY MINIMIZATION
        constraints = [
            {
                'type': 'eq',
                'fun': scipy_constraint,
                'jac': scipy_constraint_jac
            }
        ]

        res = spo.minimize(
            fun=scipy_objective,
            x0=flat_vars_np,
            method='SLSQP',
            jac=True,  # Tells SciPy that scipy_objective returns (cost, grad)
            bounds=bounds_scipy,
            constraints=constraints,
            callback=scipy_callback,
            options={'maxiter': Inputs.opt_max_iter, 'ftol': Inputs.opt_tol}
        )

        # 7. RECONSTRUCT THE PYTREE RESULTS FROM THE FINAL SCIPY STATE
        final_flat_vars = jnp.array(res.x)
        vars_scaled_opt = unflatten_fn(final_flat_vars)
        
        params_struct = state_container["params_struct"]
        params_struct = apply_vars_to_struct(vars_scaled_opt, params_struct)
        
        powers, thrusts, rms_MIL_errs, final_V_ia_0 = evaluate_all_conditions(params_struct)
        params_struct = eqx.tree_at(lambda p: p.V_ia_0, params_struct, final_V_ia_0)
        
        print(f"\nOptimization Complete. Status: {res.message}")
        time_log = jnp.append(time_log, time.time() - start_time)

    elif Inputs.opt_solver == "L-BFGS-B":
        import numpy as np
        solver = jaxopt.LBFGSB(
            fun=objectives_fn, 
            has_aux=True,
            implicit_diff=True, 
            jit=True
        )
        ## Initialisation Prints
        print("Initialising Solver...")
        state = solver.init_state(init_params=vars_scaled, bounds=bounds, params_list=params_struct)
        print(f"{'Iter':<6} | {'Cond':<5} | {'Omega':<10} | {'Power':<12} | {'Thrust':<12} | {'T-Err':<12} | {'MIL':<10} | {'OptErr':<11} | {'Time'}")
        print("-" * 103)
        ## Solver Loop
        try:
            for i in range(Inputs.opt_max_iter): 
                vars_scaled, state = solver.update(vars_scaled, 
                                                state=state, 
                                                bounds=bounds, 
                                                params_list=params_struct
                                                )
                # state.aux captures the updated tracking profiles (shape: [2, n_r])
                # We assign them directly back to keep Newton warm-starts efficient

                ## Unpack and Rescale Variables
                c_scaled, beta_scaled, omega_scaled = vars_scaled
                # toggle for omega solving
                omega = lax.switch(Inputs.solve_omega, [lambda _: params_struct.omega, lambda _: params_struct.omega_max * omega_scaled], None)
                c = params_struct.c_scale[:, None] * c_scaled[None, :]
                beta = params_struct.beta_scale[:, None] * beta_scaled[None, :]

                # Extract and unpack induced velocity
                V_ia_0 = state.aux

                ## Insert all new Guesses into params_struct
                params_struct = eqx.tree_at(
                    lambda p: (p.c, p.beta, p.omega, p.V_ia_0), 
                    params_struct, 
                    (c, beta, omega, V_ia_0)
                )

                # Print Statements for this Iteration
                is_terminal = state.error < Inputs.opt_tol
                is_checkpoint = (time.time() - print_time > 5) or (i % 5 == 0)

                if is_terminal or is_checkpoint:
                    # Calculate State for Printing
                    powers, thrusts, rms_MIL_errs, _ = evaluate_all_conditions(params_struct)

                    # Vectorized array math for errors and costs
                    t_errs = thrusts - params_struct.thrust_target

                    # vmap the reference power calculation over the batched thrust_targets
                    vmapped_actuator_pwr = jax.vmap(propmodel.actuator_disk_power, in_axes=(0, None))
                    actuator_pwr_refs = vmapped_actuator_pwr(params_struct.thrust_target, params_base)
                    cost_Ts = params_struct.objective_thrust * (t_errs / params_struct.thrust_target)**2
                    cost_Ps = params_struct.objective_power * (powers / actuator_pwr_refs)**2

                    # Clean zip over the pre-calculated batched arrays
                    for idx, (pwr, thr, t_err, mil, om, c_T, c_P) in enumerate(zip(
                        powers, thrusts, t_errs, rms_MIL_errs, omega, cost_Ts, cost_Ps
                    )):
                        # Only print iteration metadata on the first condition row
                        iter_info = f"{i:<6} |" if idx == 0 else f"{'':<6} |"
                        opt_err = f"{state.error:<11.3f}" if idx == 0 else f"{'':<11}"
                        timing = f"{time.time() - iter_time:<4.1f}s" if idx == 0 else ""
                        
                        print(f"{iter_info} C{idx:<3} | {om:<5.2f} | {pwr:<12.4f} | {thr:<12.4f} | {t_err:<12.4f} | {mil:<10.4f} | {opt_err} | {timing}")
                    print_time = time.time()
                    
                    ## Plotting Thread Update
                    if update_queue is not None:
                        # Extract Condition 0 (params_hover) from the batched PyTree
                        params_hover = jax.tree_util.tree_map(
                            lambda leaf: leaf[0] if hasattr(leaf, 'ndim') and leaf.ndim > 0 else leaf,
                            params_struct
                        )
                        
                        # Inject the current error log into the params object
                        params_hover = eqx.tree_at(
                            lambda p: p.error_log, 
                            params_hover, 
                            error_log
                        )
                        
                        # params object to queue
                        update_queue.put(params_hover)
                    
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

    # return optimised variables
    final_c = params_struct.c[0]
    final_beta = params_struct.beta[0]
    final_omega = jnp.atleast_1d(params_struct.omega[0])
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
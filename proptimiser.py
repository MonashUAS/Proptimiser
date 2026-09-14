# External Imports
import jax
import time
import sys
import signal
import threading
import jax.numpy as jnp
import jaxopt
from dataclasses import replace
import equinox as eqx
from abc import ABC, abstractmethod

# Internal Imports
from init import Inputs
import propmodel

_stop_event = threading.Event()
_sigint_installed = False
 
 
def _sigint_handler(signum, frame):
    if not _stop_event.is_set():
        _stop_event.set()
        # Write + flush immediately so the message shows up in VSCode's
        # integrated terminal without waiting on buffered/batched output.
        sys.stdout.write("\n[Ctrl+C] Interrupt received — stopping after the current iteration...\n")
        sys.stdout.flush()
    else:
        # A second Ctrl+C means "I really mean it, stop now".
        sys.stdout.write("\n[Ctrl+C] Second interrupt received — forcing immediate exit.\n")
        sys.stdout.flush()
        raise KeyboardInterrupt
 
 
def install_sigint_handler():
    """
    Installs the custom SIGINT handler once, from the main thread only.
    Safe to call multiple times (idempotent).
    """
    global _sigint_installed
    if _sigint_installed:
        return
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGINT, _sigint_handler)
        _sigint_installed = True
 
install_sigint_handler()


def objectives(params):
    '''
    Return the Base Objective Values of the Solver given an input of parameters
    Values of interest include power, thrust error, minimum induced loss error,
    '''
    # replace params object (works in function)
    V_i, V_i_0   = propmodel.induced_velocity(params)             # Solve induced velocity field for current guess
    thrust = jnp.squeeze(propmodel.thrust(V_i, params))            # return the total thrust the propeller produce
    torque = jnp.squeeze(propmodel.torque(V_i, params))            # return total torque propeller produces
    rms_MIL_err = propmodel.minimumInducedLossCost(V_i, params)            # minimum induced loss condition root mean square error
    
    # calculate the motor power
    motor_power = propmodel.motorPower(torque, params.omega, I_0 = Inputs.I_0, R_m = Inputs.R_m, kV = Inputs.kV)

    # Return V_i_0 so we can catch it!
    return motor_power, thrust, rms_MIL_err, V_i_0 

def condition_cost(params):
    ''' 
    Computes cost and returns updated V_i_0 for exactly ONE condition 
    '''
    # unpack objectives and induced velocity
    motor_power, thrust, rms_MIL_err, V_i_0 = objectives(params)
    # calculate scaled variables
    thrust_err_rel = (thrust - params.thrust_target)/params.thrust_target
    power_rel = (Inputs.P_max-motor_power)/Inputs.P_max

    # Calculate base cost using this condition's specific targets
    cost = (
        thrust_err_rel**2 +
        power_rel
    )
    
    # Scale by the relative importance weight of this flight condition
    weighted_cost = cost * params.objective_weight
    return weighted_cost, V_i_0

@jax.jit
def total_cost(vars_scaled, params_struct): 
    '''
    Total cost of all n conditions specified in Inputs.
    '''
    # Update parameters struct with new geometry and control points
    params_struct = apply_vars_to_struct(vars_scaled, params_struct)

    # vmap cost function over parameters list
    cost_fn_vmap = jax.vmap(condition_cost, in_axes=0)
    costs, V_i_0 = cost_fn_vmap(params_struct)

    aux = (V_i_0)
    return jnp.sum(costs), aux

def apply_vars_to_struct(vars_scaled_dict, params_struct):
    """
    Applies scaled variables to the params_struct dynamically using dictionaries.
    """
    # Define mapping for how each variable scales up from [0, 1] bounds
    scales = {
        'c': params_struct.c_scale[:, None],
        'beta': params_struct.beta_scale[:, None],
        'omega': params_struct.omega_max,
        'tilt': params_struct.tilt_scale
    }
    
    # Define mapping for condition-specific toggles (reference Inputs)
    masks = {
        'omega': Inputs.solve_omega,
        'tilt': Inputs.solve_tilt
    }
    
    # Iteratively evaluate and inject variables back into the Equinox module
    for key, scaled_value in vars_scaled_dict.items():
        # Scale the variable
        real_value = scaled_value * scales[key]
        
        # Apply conditional masking if a mask is defined and exists
        if key in masks and masks[key] is not None:
            mask = masks[key]
            # Ensure 1D masks broadcast correctly over 2D variable arrays
            mask = mask[:, None] if real_value.ndim > mask.ndim else mask
            base_value = getattr(params_struct, key)
            real_value = jnp.where(mask, real_value, base_value)
            
        # Dynamically inject the updated variable back into the struct
        params_struct = eqx.tree_at(
            lambda p, k=key: getattr(p, k), 
            params_struct, 
            real_value
        )
    return params_struct

@jax.jit
def evaluate_all_conditions(params_struct):
    '''
    Helper Function to evaluate all conditions but not cost.
    '''
    vmapped_obj = jax.vmap(objectives, in_axes=0)
    return vmapped_obj(params_struct)

class BaseSolver(ABC):
    def __init__(self, params_base, update_queue=None):
        self.params_base = params_base
        self.update_queue = update_queue
        self.params_struct, self.vars_scaled, self.bounds = self._prepare_problem()
        
    def _prepare_problem(self):
        """Universal setup: builds the batched params_struct and optimization bounds."""
        params_list = []

        for index in range(Inputs.n_cond):
            p_temp = replace(
                self.params_base,
                V_x=Inputs.V_x[index],
                V_yz=Inputs.V_yz[index],
                omega=Inputs.omega[index],
                tilt=Inputs.tilt[index],
                thrust_target=Inputs.thrust_target[index],
                objective_weight=Inputs.objective_weight[index]
            )
            V_scale = propmodel.get_velocity_scales(p_temp)

                # Introduce zero for non-solved components to maintain shape (3, n_r)
            V_i_0 = jnp.zeros_like(p_temp.V_i_0).at[Inputs.bemt_type].set(V_scale)
            p_temp = replace(p_temp, V_i_0=V_i_0)
            params_list.extend([p_temp])

        params_struct = jax.tree_util.tree_map(lambda *args: jnp.stack(args), *params_list)

        vars_scaled = {
            'c': self.params_base.c / self.params_base.c_scale,
            'beta': self.params_base.beta / self.params_base.beta_scale,
            'omega': params_struct.omega / self.params_base.omega_max,
            'tilt': params_struct.tilt / self.params_base.tilt_scale
        }

        def get_lower_bound(k, v):
            if k == 'c':
                return 0.005
            elif k == 'omega':
                return self.params_base.omega_max / 10.0
            return 0.0

        lower_bounds = {k: jnp.full_like(v, get_lower_bound(k, v)) for k, v in vars_scaled.items()}
        upper_bounds = {k: jnp.full_like(v, 1.0) for k, v in vars_scaled.items()}
        bounds = (lower_bounds, upper_bounds)
        
        return params_struct, vars_scaled, bounds

    @abstractmethod
    def init_state(self):
        """Initializes and returns the solver-specific state."""
        pass

    @abstractmethod
    def step(self, vars_scaled, state, params_struct):
        """Performs a single update step. Must return (new_vars, new_state, aux_data)."""
        pass

    def solve(self):
        """The main optimization loop returning params_solved."""
        # Reset the stop event at the beginning of each solve to ensure clean state
        _stop_event.clear()
 
        state = self.init_state()
        vars_scaled = self.vars_scaled
        params_struct = self.params_struct
        
        start_time = time.time()
        iter_time = start_time + 10
        print_time = iter_time
        error_log = jnp.array([1.0])
        time_log = jnp.array([0])

        print(f"{'Iter':<6} | {'Cond':<5} | {'Omega':<10} | {'Power':<12} | {'Thrust':<12} | {'T-Err':<12} | {'MIL':<10} | {'OptErr':<11} | {'Time'}")
        print("-" * 103)

        for i in range(Inputs.opt_max_iter):
            if _stop_event.is_set():
                # Stop event check
                print(f"\nStopping: interrupted by user at step {i}", flush=True)
                break

            # Step delegated to the specific solver
            vars_scaled, state, V_i_0 = self.step(vars_scaled, state, params_struct)

            # Update and rescale params
            params_struct = apply_vars_to_struct(vars_scaled, params_struct)
            
            V_i_0 = jnp.where(jnp.isfinite(V_i_0), V_i_0, params_struct.V_i_0)
            # Update the V_i_0 field
            params_struct = eqx.tree_at(
                lambda p: (p.V_i_0), 
                params_struct, 
                (V_i_0)
            )

            is_terminal = getattr(state, 'error', 1.0) < Inputs.opt_tol
            is_checkpoint = (time.time() - print_time > 5) or (i % 5 == 0)
            
            if is_terminal or is_checkpoint:
                # Calculate State for Printing
                powers, thrusts, rms_MIL_errs, _ = evaluate_all_conditions(params_struct)

                # Vectorized array math for errors and costs
                t_errs = thrusts - params_struct.thrust_target

                # Extract omega
                omegas = params_struct.omega

                # vmap the reference power calculation over the batched thrust_targets
                vmapped_actuator_pwr = jax.vmap(propmodel.actuator_disk_power, in_axes=(0, None))
                actuator_pwr_refs = vmapped_actuator_pwr(params_struct.thrust_target, self.params_base)
                cost_Ts = params_struct.objective_thrust * (t_errs / params_struct.thrust_target)**2
                cost_Ps = params_struct.objective_power * (powers / actuator_pwr_refs)**2

                # Clean zip over the pre-calculated batched arrays
                for idx, (pwr, thr, t_err, mil, om, c_T, c_P) in enumerate(zip(
                    powers, thrusts, t_errs, rms_MIL_errs, omegas, cost_Ts, cost_Ps
                )):
                    # Only print iteration metadata on the first condition row
                    iter_info = f"{i:<6} |" if idx == 0 else f"{'':<6} |"
                    opt_err = f"{state.error:<11.3f}" if idx == 0 else f"{'':<11}"
                    timing = f"{time.time() - iter_time:<4.1f}s" if idx == 0 else ""
                    
                    print(f"{iter_info} C{idx:<3} | {om:<5.2f} | {pwr:<12.4f} | {thr:<12.4f} | {t_err:<12.4f} | {mil:<10.4f} | {opt_err} | {timing}")
                    print(f"Cond {idx} Internal Costs -> Thrust: {float(c_T):.4e} | Power: {float(c_P):.4e} | Induced: {float(mil):.4e}")
                print_time = time.time()
                
                ## Plotting Thread Update
                if self.update_queue is not None:
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
                    self.update_queue.put(params_hover)
                
                if is_terminal:
                    print(f"\nStopping: Convergence reached at step {i}",flush=True)
                    break

            error_log = jnp.append(error_log, getattr(state, 'error', 1.0))
            time_log = jnp.append(time_log, time.time() - iter_time)
            iter_time = time.time()

            if is_terminal:
                print(f"\nStopping: Convergence reached at step {i}",flush=True)
                break

        print(f"\nTotal Solve Time: {time.time() - start_time:.2f} seconds", flush=True)

        final_c = params_struct.c[0]
        final_beta = params_struct.beta[0]
        final_omega = jnp.atleast_1d(params_struct.omega[0])
        final_V_i_0 = params_struct.V_i_0[0]

        return replace(self.params_base, 
                    c=final_c, 
                    beta=final_beta,   
                    omega=final_omega,
                    V_i_0=final_V_i_0,
                    error_log=error_log, 
                    time_log=time_log)

class LBFGSBSolver(BaseSolver):
    def __init__(self, params_base, update_queue=None):
        super().__init__(params_base, update_queue)
        self.solver = jaxopt.LBFGSB(
            fun=total_cost, 
            has_aux=True,
            implicit_diff=True, 
            jit=True,
            verbose=1
        )

    def init_state(self):
        print("Initialising L-BFGS-B Solver...")
        return self.solver.init_state(
            init_params=self.vars_scaled, 
            bounds=self.bounds, 
            params_struct=self.params_struct
        )

    def step(self, vars_scaled, state, params_struct):
        vars_scaled, state = self.solver.update(
            vars_scaled, 
            state=state, 
            bounds=self.bounds, 
            params_struct=params_struct
        )
        # JAXopt specific auxiliary extraction
        V_i_0 = state.aux
        return vars_scaled, state, V_i_0

def run_optimization(params_base, update_queue=None):
    if Inputs.opt_solver == "L-BFGS-B":
        solver = LBFGSBSolver(params_base, update_queue)
    else:
        raise ValueError(f"Solver {Inputs.opt_solver} not implemented.")
        
    params_solved = solver.solve()
    return params_solved

if __name__ == "__main__":
    ## Testing of optimiser functions
    # generate basic params
    params_base = propmodel.build_default_propeller()

    weighted_cost, V_i_0 = condition_cost(params_base)
    print(weighted_cost)
    params_list = []
    for index in range(Inputs.n_cond):
        #import ic's
        p_temp = replace(
            params_base,
            V_x=Inputs.V_x[index],
            V_yz=Inputs.V_yz[index],
            omega=Inputs.omega[index],
            thrust_target=Inputs.thrust_target[index],
            objective_weight=Inputs.objective_weight[index]
        )
        # calculate first guess for V_i_0 field
        v_scale = propmodel.get_velocity_scales(p_temp)

        V_i_0_guess_dim = jnp.zeros_like(p_temp.V_i_0).at[Inputs.bemt_type].set(V_i_0_guess_dim)
        p_temp = replace(p_temp, V_i_0=V_i_0)
        params_list.extend([p_temp])
    
    # Convert list into a stacked PyTree object that JAX natively unrolls via vmap
    params_struct = jax.tree_util.tree_map(lambda *args: jnp.stack(args), *params_list)

    vars_scaled = (
        params_base.c / params_base.c_scale,
        params_base.beta / params_base.beta_scale,
        params_struct.omega / params_base.omega_max # Shape (2,)
    )

    costs, V_i_0 = total_cost(vars_scaled, params_struct)
    print(costs)
    print(V_i_0)
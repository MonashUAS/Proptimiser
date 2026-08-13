import jax
# Set jax debugging
jax.config.update("jax_traceback_filtering", "off")
jax.config.update("jax_check_tracer_leaks", "on")
jax.config.update("jax_disable_jit", False)
jax.config.update("jax_debug_nans", False)
jax.config.update("jax_enable_x64", True)

from dataclasses import dataclass
import equinox as eqx
import jax.numpy as jnp

# Internal Imports
import geometry

def enable_gpu_acceleration():
    """
    Configures JAX to execute calculations on the GPU.

    Inputs:
        None

    Returns:
        str: Active JAX device platform name
    """
    gpu_device = jax.devices("gpu")[0]
    jax.config.update("jax_default_device", gpu_device)
    return gpu_device.platform

class Inputs(eqx.Module):
    # Outer-loop Optimisation (Currently only L-BFGS-B works)
    opt_solver: str = "L-BFGS-B"
    opt_max_iter: int = 1000
    opt_tol: float = 1e-2 

    # Induced Velocity Settings
    V_ia_0_max_iter: int = 500
    V_ia_0_eps: float = 5e-3

    # Solver Type
    solvertype: str = "T"
    solve_omega: int = 0 # 0: No. 1: Objective

    # Discretisation settings
    n_r: int = 25
    n_psi: int = 24

    # Diagnostics
    verbose: bool = False
    jax_tracing: bool = False

    # Geometry Setup
    R_tip: float = 0.275
    R_hub: float = 0.025

    # Flight Conditions
    n_cond: int = 2
    thrust_target = jnp.array([15.9,100.0])
    V_x = jnp.array([18.0,0.8])
    V_yz = jnp.array([3.0,3.0])
    omega = jnp.array([588.0,727.0])
    objective_weight = jnp.array([0.9,0.1])

    # Motor Information
    I_0: float = 2.0
    R_m: float = 0.037
    kV: float = 220
    W_max: float = 3300

    # Aerodynamic Models Selection
    inflow_model: int = 1
    coeff_method: int = 3 # 0: Parametric, 1: 2D Interp, 2: 3D Interp, 3: NeuralFoil
    tip_loss_model: int = 0 # 0: No Loss 1: Prandtl
    root_loss_model: int = 0
    rot_corr_model: int = 0 # 3D inflow correction model
    high_induction_model: int = 0 # (0: Classical Momentum, 1: Buhl Correction) NOT USEFUL FOR REGULAR FLIGHT

    def save_state(self, filename: str = "inputs_state.eqx"):
        """Saves the Inputs configuration to an Equinox file."""
        if not filename.endswith(".eqx"): filename += ".eqx"
        filename = "propellers/" + filename
        eqx.tree_serialise_leaves(filename, self)
        print(f"Inputs state successfully written to: {filename}")
    
    def load_state(self, filename: str = "inputs_state.eqx"):
        """
        Loads and returns a new Inputs instance from an Equinox file.
        Uses 'self' as the structural skeleton.
        """
        if not filename.endswith(".eqx"): filename += ".eqx"
        filename = "propellers/" + filename
        return eqx.tree_deserialise_leaves(filename, self)


class Params(eqx.Module):
    '''
    Parameters for a given blade solution.
    Ensure that all inputs are 1D arrays upon assignment.

    shape(r, dr, c, beta, V_ia_0) = (N_r,)
    shape(psi, dpsi) = (N_psi,)
    '''
    # Domain of Integration
    r: jnp.ndarray          # radial stations
    dr: jnp.ndarray         # radial spacing (m)
    psi: jnp.ndarray        # azimuthal stations
    dpsi: jnp.ndarray       # azimuthal spacing (m)

    # Geometry & discretization
    N: jnp.ndarray          # number of blades
    c: jnp.ndarray          # chord length at radial stations (m)
    beta: jnp.ndarray       # blade twist angle (rad)

    c_scale: jnp.ndarray    # scale of chord
    beta_scale: jnp.ndarray # scale of angle

    # Inflow / rotor state
    V_x: jnp.ndarray        # axial inflow velocity (ms^{-1})
    V_yz: jnp.ndarray       # In plane inflow velocity (ms^{-1})

    omega: jnp.ndarray      # rotational speed (rad/s)
    omega_max: jnp.ndarray  # max subsonic omega

    # Cost Weights
    objective_thrust: jnp.ndarray
    objective_MIL:  jnp.ndarray     
    objective_power: jnp.ndarray
    objective_weight: jnp.ndarray 

    # Target objective
    thrust_target: jnp.ndarray

    V_ia_0: jnp.ndarray     # initial guess/storage for V_ia_0 (ms^{-1})

    # Solver Logging Info
    error_log: jnp.ndarray
    time_log: jnp.ndarray

    # Airfoil Selection
    airfoil: str = eqx.field(static=True) # Strings must be static
    airfoil_interp: callable = eqx.field(static=True)

    # Aerodynamic Constants
    rho: float = 1.225          # air density (kgm^{-3})
    mu: float = 1.798e-5        # air viscosity (kgm^-1s^-1)
    a: float = 343.0            # speed of sound (m/s)      

    def save_state(self, filename: str = "params_state.eqx"):
        """Saves the array parameters to an Equinox file."""
        if not filename.endswith(".eqx"): filename += ".eqx"
        filename = "propellers/" + filename
        eqx.tree_serialise_leaves(filename, self)

    def load_state(self, filename: str = "params_state.eqx"):
        """
        Loads saved binary array weights back into a Params class instance.n

        Acts on 'self' as the structural skeleton and returns a populated copy.
        """
        if not filename.endswith(".eqx"): filename += ".eqx"
        filename = "propellers/" + filename
        return eqx.tree_deserialise_leaves(filename, self)


def build_default_propeller(c: float = 0.02, beta: float = 10.0, inputs: Inputs = None, airfoil_name: str = "arad10") -> Params:
    """
    Factory function to build a default Params configuration.
    """
    # Internal Imports
    from airfoil import load_airfoil_data
    if inputs is None:
        inputs = Inputs()

    # Geometry constraints
    c_scale = 0.2                   
    beta_scale = jnp.radians(50)    

    # Build Grids
    psi, dpsi = geometry.angular_stops(inputs.n_psi)
    r, dr = geometry.radial_stops(inputs.R_tip, inputs.R_hub, inputs.n_r)

    # Constant chord across the whole blade
    init_c = jnp.full((Inputs.n_r,), c) 
    # Constant twist across the whole blade
    init_beta = jnp.full((Inputs.n_r,), jnp.radians(beta))


    # Airfoil Data
    airfoil_interp = load_airfoil_data(airfoil_name)
    # Construct Params
    return Params(
        airfoil=airfoil_name,
        airfoil_interp=airfoil_interp,
        N=jnp.array([2]), # Ensure standard shape
        c=init_c,      
        beta=init_beta,
        r=r, dr=dr,    
        psi=psi, dpsi=dpsi, 
        V_x=inputs.V_x[0], 
        V_yz=inputs.V_yz[0],
        omega=inputs.omega[0],
        omega_max=343.0 / inputs.R_tip,
        V_ia_0=jnp.full((inputs.n_r,), 8.0), 
        c_scale=c_scale,
        beta_scale=beta_scale,
        error_log=jnp.array([1.0]),
        time_log=jnp.array([0.0]),
        objective_thrust=500.0,
        objective_MIL=0.0,
        objective_power=0.5,
        objective_weight=inputs.objective_weight[0],
        thrust_target=inputs.thrust_target[0],
    )

if __name__ == "__main__":

    #enable_gpu_acceleration() #a bit of work; you need linux for this

    from datetime import datetime
    import threading
    import queue
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    # Internal Imports
    import proptimiser
    import results
    import geometry
    import propmodel

    params_init, _, _ = propmodel.apc_model("apc_prop_data/propgeom_data/20x8-PERF.PE0", plot=False)

    print(f"Initialised airfoil interpolation with airfoil {params_init.airfoil}")

    # Live Plotting Setup
    print("Setting up live visualization canvas...")
    gui_queue = queue.Queue()
    plotter = results.LivePropPlotter(params_init)

    if Inputs.jax_tracing:
        print("Starting Profiler...")
        jax.profiler.start_trace("/tmp/jax_trace")

    thread_results = {}
    
    def optimization_worker():
        # Run optimization on background thread and pass the queue
        optimized_state = proptimiser.run_optimization(params_init, update_queue=gui_queue)
        thread_results['params_opt'] = optimized_state

    # Run Optimisation in Background Thread
    opt_thread = threading.Thread(target=optimization_worker, daemon=True)
    opt_thread.start()
    
    # This automatically polls the queue and refreshes the canvas
    ani = FuncAnimation(
        plotter.fig, 
        plotter.update, 
        fargs=(gui_queue,), 
        interval=200, 
        cache_frame_data=False,
        blit=False
    )
    
    # Hands control to the GUI until the window is closed
    plt.show()

    # Re-join thread to ensure clean closure 
    opt_thread.join()
    params_opt_hover = thread_results.get('params_opt', params_init)

    if Inputs.jax_tracing:
        jax.block_until_ready(params_opt_hover) 
        jax.profiler.stop_trace()
        print("Profiler stopped. Trace saved to /tmp/jax_trace")

    # Save Results/Export Geometry if Required
    user_input = input('Save Propeller? (y/n): ')
    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    if user_input.lower() == 'y':
        paramsfile = "prop_params_" + timestamp
        inputsfile = "prop_inputs_" + timestamp
        print(f'Saving Propeller Inputs/Parameters to {paramsfile} and {inputsfile}')
        params_opt_hover.save_state(paramsfile)
        Inputs().save_state(inputsfile)

    print('Proptimisation Complete')
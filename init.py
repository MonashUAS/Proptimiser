import jax
# Set jax debugging
jax.config.update("jax_traceback_filtering", "off")
jax.config.update("jax_check_tracer_leaks", "on")
jax.config.update("jax_disable_jit", False)
jax.config.update("jax_debug_nans", False)
jax.config.update("jax_enable_x64", True)

# External Imports
from dataclasses import dataclass
import equinox as eqx
import jax.numpy as jnp

# Freeze configuration parameters at runtime
class Inputs(eqx.Module):
    ## Optimisation / design settings which remain constant during execution
    # Outer-loop Optimisation (L-BFGS or ADAM)
    opt_solver: str = "L-BFGS-B" # Toggle Optimisation Loop (type of optimisation algorithm.)
    opt_smoothing: str = "pointwise" # pointwise or bezier. Bezier often breaks propeller shape.
    opt_max_iter: int = 1000
    opt_tol: float = 1e-3   # Optimiser Stability Criteria

    # Induced Velocity Settings
    V_ia_0_max_iter: int = 100
    V_ia_0_eps: float = 1e-3

    # Solver Type
    solvertype: str = "T"
    solve_omega: int = 0 # 0 for fixed omega, 1 for omega solving

    ## Discretisation settings
    # Radial discretisation
    n_r: int = 50
    # Azimuthal discretisation (for skewed inflow)
    n_psi: int = 16
    # Number of Control Points
    n_cp: int = 12

    # diagnostics/debugging
    verbose: bool = False
    jax_tracing: bool = False

    ## Input Parameters
    # Geometry Setup (universal across propeller)
    R_tip = 0.25
    R_hub = 0.025

    ## --- FLIGHT CONDITIONS (Batched as Arrays) ---
    n_cond: int = 1
    # Index 0: Hover | Index 1: Forward Flight | etc...            
    thrust_target = jnp.array([22.5])
    
    V_x = jnp.array([0])
    V_yz = jnp.array([30.0])
    omega= jnp.array([400.0])
    
    objective_weight = jnp.array([1.0])

    # Motor Information
    I_0: float = 0.7           # Zero Load Current (amps)
    R_m: float = 0.120         # Coil Resistance   (ohms)
    k_V: float = 150           # Motor kV          (RPM/Volt)

    # Aerodynamic Models Selection
    inflow_model: int = 1 # 0 = basic, 1 = Pitt & Peters, 2 = DTU (WIP)
    coeff_method: int = 2 # 0 = parametric, 1 = alpha/reynolds, 2 = alpha/mach/reynolds
    tip_loss_model: int = 1 # 0 = none, 1 = Prandtl
    root_loss_model: int = 1 # 0 = none, 1 = Prandtl
    def save_state(self, filename: str = "inputs_state.eqx"):
        """Saves the Inputs configuration to an Equinox file."""
        if not filename.endswith(".eqx"):
            filename += ".eqx"
        # make sure files are saved to propellers
        filename = "propellers/"+filename
        eqx.tree_serialise_leaves(filename, self)
        print(f"Inputs state successfully written to: {filename}")
    
    def load_state(self, filename: str = "inputs_state.eqx"):
        """
        Loads and returns a new Inputs instance from an Equinox file.
        Uses 'self' as the structural skeleton.
        """
        if not filename.endswith(".eqx"):
            filename += ".eqx"
        # make sure files are loaded from propellers
        filename = "propellers/"+filename
        # Returns a NEW instance with the loaded data
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

    c_cp: jnp.ndarray       # chord control points
    beta_cp: jnp.ndarray    # twist control points

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
        # ensure extension is used
        if not filename.endswith(".eqx"):
            filename += ".eqx"
        # make sure files are saved to propellers
        filename = "propellers/"+filename
        eqx.tree_serialise_leaves(filename, self)
        print(f"Params state successfully written to: {filename}")

    def load_state(self, filename: str = "params_state.eqx"):
        """
        Loads saved binary array weights back into a Params class instance.n

        Acts on 'self' as the structural skeleton and returns a populated copy.
        """
        if not filename.endswith(".eqx"):
            filename += ".eqx"
        # make sure files are loaded from propellers/ folder
        filename = "propellers/"+filename
        # Returns a NEW instance with the loaded data overlaid on the current object's skeleton
        return eqx.tree_deserialise_leaves(filename, self)
    
if __name__ == "__main__":
    '''
    Complete a Propeller Optimisation Run from the Terminal.

    Modify Values in Init()
    '''
    from datetime import datetime
    # Internal Imports
    import propmodel
    import proptimiser
    import results
    import geometry
    from airfoil import load_airfoil_data

    ## SOLVER SPECS
    airfoil = "arad10"

    c_scale = 0.2                   # Max chord (m)
    beta_scale = jnp.radians(50)    # Max twist (rad)

    # Initial Geometry setup
    psi, dpsi = propmodel.angular_stops(Inputs.n_psi)
    
    r, dr = propmodel.radial_stops(Inputs.R_tip, Inputs.R_hub, Inputs.n_r)

    # Twist distribution starting guess (in degrees): 
    init_beta_cp = jnp.radians(jnp.array([18.0,16.0,16.0,16.0,14.0,7.0,7.0,4.0,4.0,4.0,4.0,4.0]))

    # Chord distribution (in meters): 
    init_c_cp = jnp.array([0.05,0.05,0.05,0.05,0.05,0.05,0.04,0.03,0.03,0.03,0.03,0.03])

    # Interpolate airfoil data
    airfoil_interp = load_airfoil_data("arad10")

    spline_matrix_init = proptimiser.get_bezier_basis_matrix(r, Inputs.R_hub, Inputs.R_tip, Inputs.n_cp)
    init_c = jnp.dot(init_c_cp, spline_matrix_init.T)
    init_beta = jnp.dot(init_beta_cp, spline_matrix_init.T)

    # Initialise Params Class (will be duplicated for inputs in run_optimisation)

    params_init = Params(
        airfoil = airfoil,
        airfoil_interp=airfoil_interp,
        N=2,
        c=init_c,      
        beta=init_beta,
        c_cp=init_c_cp,
        beta_cp=init_beta_cp,
        r=r, dr=dr,    
        psi=psi, dpsi=dpsi, 
        V_x=jnp.array([Inputs.V_x[0]]), V_yz=jnp.array([Inputs.V_yz[0]]),
        omega=jnp.array([Inputs.omega[0]]),
        omega_max = Params.a / Inputs.R_tip,
        V_ia_0=jnp.full((Inputs.n_r), 4.0), # Induced Velocity Field guess
        c_scale = c_scale,
        beta_scale = beta_scale,
        error_log = jnp.array([1.0]),
        time_log = jnp.array([0]),

        # Cost Weights
        objective_thrust=jnp.array([50.0]),
        objective_MIL=jnp.array([1.0]),
        objective_power=jnp.array([0.0]),
        objective_weight=Inputs.objective_weight,
        thrust_target=Inputs.thrust_target,
    )

    print(f"Initialised Airfoil Fit with airfoil {params_init.airfoil}")

    if Inputs.jax_tracing:
        print("Starting Profiler...")
        # This will create a directory called /tmp/jax_trace (or a folder of your choice)
        jax.profiler.start_trace("/tmp/jax_trace")
        
    # Run Optimiser
    params_opt_hover = proptimiser.run_optimization(params_init)
    
    if Inputs.jax_tracing:
        # Block until all asynchronous JAX operations finish
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

        # Use the instantiated object references (params_opt and an Inputs instance)
        params_opt_hover.save_state(paramsfile)
        Inputs().save_state(inputsfile)
    
    user_input = input('Export Propeller Geometry to .pts files? (y/n): ')
    if user_input.lower() == 'y':
        exportname = f"prop_pts_{timestamp}"
        foilpoints = geometry.upperlowerairfoil(params_opt_hover.airfoil)
        geometry.ptsfilegenerator(params_opt_hover.r, params_opt_hover.c, params_opt_hover.beta, foilpoints, exportname)
        print(f'Saving Propeller Geometry to {exportname}.pts')

    print('Proptimisation Complete')

    # Plot Info
    results.propinfoplot(params_opt_hover)

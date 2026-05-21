import jax
# Set jax debugging
jax.config.update("jax_traceback_filtering", "quiet_remove_frames")
jax.config.update("jax_check_tracer_leaks", "off")
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
    opt_max_iter: int = 100 
    opt_tol: float = 1e-2   # Optimiser Stability Criteria
    opt_weights: int = 2    # Number of Weights for the Cost Function

    # Induced Velocity Settings
    V_ia_0_max_iter: int = 100
    V_ia_0_eps: float = 5e-2

    # Omega Solver Settings
    omega_max_iter: int = 20
    omega_eps: float = 0.5

    # Solver Type (Thrust Constraint)
    solvertype: str = "T"

    ## Discretisation settings
    # Radial discretisation
    n_r: int = 25
    # Azimuthal discretisation (for skewed inflow)
    n_psi: int = 12

    # diagnostics/debugging
    verbose: bool = False

    ## Input Parameters
    # Geometry Setup (universal across propeller)
    R_tip = 0.25
    R_hub = 0.025

    ## --- FLIGHT CONDITIONS (Batched as Arrays) ---
    # Index 0: Hover | Index 1: Forward Flight
    thrust_target = jnp.array([22.0, 30.0])
    
    V_x = jnp.array([0.0, 0.0])
    V_yz = jnp.array([10.0, 0.0])
    omega= jnp.array([640.0, 640.0])
    
    objective_weight = jnp.array([0.7, 0.3])

    # Motor Information
    I_0: float = 0.7           # Zero Load Current (amps)
    R_m: float = 0.120         # Coil Resistance   (ohms)
    k_V: float = 150            # Motor kV          (RPM/Volt)

    # Aerodynamic Models Selection
    inflow_model: int = 1 # 0 = basic, 1 = Pitt & Peters, 2 = DTU (WIP)
    coeff_method: int = 0 # 0 = parametric, 1 = alpha/reynolds, 2 = alpha/mach/reynolds
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

    c_scale: jnp.ndarray    # scale of chord
    beta_scale: jnp.ndarray # scale of angle

    # Inflow / rotor state
    V_x: jnp.ndarray        # axial inflow velocity (ms^{-1})
    V_yz: jnp.ndarray       # In plane inflow velocity (ms^{-1})
    omega: jnp.ndarray      # rotational speed (rad/s)

    # Cost Weights
    objective_thrust: jnp.ndarray
    objective_MIL:  jnp.ndarray     
    objective_power: jnp.ndarray    # Use MIL instead of power consumption
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

    def save_state(self, filename: str = "params_state.eqx"):
        """Saves the array parameters to an Equinox file."""
        # ensure extension is used
        if not filename.endswith(".eqx"):
            filename += ".eqx"
        # make sure files are saved to propellers
        filename = "propellers/"+filename
        eqx.tree_serialise_leaves(filename, self)
        print(f"Params state successfully written to: {filename}")

    def load_state(self, filename: str = "propellers/params_state.eqx"):
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

    c_scale = 0.1                   # Max chord (m)
    beta_scale = jnp.radians(50)    # Max twist (rad)

    # Initial Geometry setup
    psi, dpsi = propmodel.angular_stops(Inputs.n_psi)
    r, dr = propmodel.radial_stops(Inputs.R_tip, Inputs.R_hub, Inputs.n_r)

    init_c = jnp.full((Inputs.n_r,), c_scale/10)
    init_beta = jnp.full((Inputs.n_r,), beta_scale/10)

    # Interpolate airfoil data
    airfoil_interp = load_airfoil_data("arad10")

    # Initialise Params Class
    params_init = Params(
        airfoil = airfoil,
        airfoil_interp=airfoil_interp,
        N=2,
        c=init_c,      
        beta=init_beta,
        r=r, dr=dr,    
        psi=psi, dpsi=dpsi, 
        V_x=jnp.array([Inputs.V_x]), V_yz=jnp.array([Inputs.V_yz]),
        omega=jnp.array([Inputs.omega]), 
        V_ia_0=jnp.full((len(Inputs.thrust_target), Inputs.n_r), 4.0), # Induced Velocity Field for Each Thrust Conditon
        c_scale = c_scale,
        beta_scale = beta_scale,
        error_log = jnp.array([1.0]),
        time_log = jnp.array([0]),

        # Cost Weights
        objective_thrust=jnp.array([1.0]),
        objective_MIL=jnp.array([0.3]),
        objective_power=jnp.array([0.005]),
        objective_weight=Inputs.objective_weight,
        thrust_target=Inputs.thrust_target,
        
    )

    print(f"Initialised Airfoil Fit with airfoil {params_init.airfoil}")

    # Run Optimiser
    params_opt = proptimiser.run_optimization(params_init)

    # Take first flight condition for plotting
    params_opt_hover = jax.tree_map(lambda x: x[0], params_opt)

    # Plot Info
    results.propinfoplot(params_opt)

    # check interpolated plot
    import numpy as np
    import matplotlib.pyplot as plt
    # Fit high order polynomial to data, then establish leading and trailing edge using fit.
    polyorder = 5
    c_coeffs = np.polyfit(params_opt.r, params_opt.c, polyorder)
    c_p = np.poly1d(c_coeffs)
    c_poly = c_p(r)

    beta_coeffs = np.polyfit(params_opt.r, params_opt.beta, polyorder)
    beta_p = np.poly1d(beta_coeffs)
    beta_poly = beta_p(r)

    # Let the plotting begin
    fig = plt.figure(figsize=(14, 10))

    # Optimum Chord & Twist Distribution (Combined)
    # Chord Distribution
    ax1 = fig.add_subplot(1, 2, 1)
    line1 = ax1.plot(r, c_poly, "o-", markersize=4, color="tab:blue", label=r"Chord ($c$)")
    ax1.set_title(r"\textbf{Optimum Chord \& Twist Distribution}")
    ax1.set_ylabel(r"Chord (mm)", color="tab:blue")
    ax1.set_xlabel(r"Radius ($r$) [mm]")
    ax1.set_xlim(left=0)
    ax1.tick_params(axis='y', labelcolor="tab:blue")
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Beta Distribution
    ax2 = fig.add_subplot(1, 2, 2)
    line2 = ax2.plot(r, beta_poly, 'o-', color="tab:red", markersize=4, label=r"Twist ($\beta$)")
    ax2.set_xlim(left=0)
    ax2.set_ylabel(r"Twist ($^\circ$)", color="tab:red")
    ax2.tick_params(axis='y', labelcolor="tab:red")
    plt.show()

    # Save Results/Export Geometry if Required
    user_input = input('Save Propeller? (y/n): ')
    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    if user_input.lower() == 'y':
        paramsfile = "prop_params_" + timestamp
        inputsfile = "prop_inputs_" + timestamp
        print(f'Saving Propeller Inputs/Parameters to {paramsfile} and {inputsfile}')

        # Use the instantiated object references (params_opt and an Inputs instance)
        params_opt.save_state(paramsfile)
        Inputs().save_state(inputsfile)
    
    user_input = input('Export Propeller Geometry to .pts files? (y/n): ')

    if user_input.lower() == 'y':
        exportname = f"prop_pts_{timestamp}"
        foilpoints = geometry.upperlowerairfoil(params_opt.airfoil)
        geometry.ptsfilegenerator(params_opt.r, params_opt.c, params_opt.beta, foilpoints, exportname)
        print(f'Saving Propeller Geometry to {exportname}.pts')

    print('Proptimisation Complete')
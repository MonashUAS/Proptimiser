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

    # Newton solver settings (Broyden for Newton-Like Induced Velocity Solving)
    newton_max_iter: int = 200
    newton_eps: float = 1e-2
    newton_damping: float = 0.1

    # Thrust Target Constraint
    solvertype: str = "T"
    thrust: float = 50 # Newtons

    ## Discretisation settings
    # Radial discretisation
    n_r: int = 50
    # Azimuthal discretisation (for skewed inflow)
    n_psi: int = 36

    # Diagnostics / debugging
    verbose: bool = False

    ## Solver Model Selection

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

    c_scale: jnp.ndarray    # scale of chord
    beta_scale: jnp.ndarray # scale of angle

    # Inflow / rotor state
    V_x: jnp.ndarray        # axial inflow velocity (ms^{-1})
    V_yz: jnp.ndarray       # In plane inflow velocity (ms^{-1})
    omega: jnp.ndarray      # rotational speed (rad/s)

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

    # Optimiser Weights (look into modifying these as ADAM parameters)
    objective_thrust: float = 1 # set to one for now
    objective_MIL:  float = 1   # 
    objective_power: float = 0  # Use MIL instead of power consumption

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
        Loads saved binary array weights back into a Params class instance.
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

    Modify Values in 
    '''
    from datetime import datetime
    # Internal Imports
    import propmodel
    import proptimiser
    import results
    import geometry
    from airfoil import load_airfoil_data

    ## SOLVER SPECS
    R = 0.25
    R_hub = 0.025

    V_x = 2
    V_yz = 0.5

    omega = 640
    airfoil = "arad10"

    c_scale = 0.1                   # Max chord (m)
    beta_scale = jnp.radians(50)    # Max twist (rad)

    # Initial Geometry setup
    psi, dpsi = propmodel.angular_stops(Inputs.n_psi)
    r, dr = propmodel.radial_stops(R, R_hub, Inputs.n_r)

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
        V_x=jnp.array([V_x]), V_yz=jnp.array([V_yz]),
        omega=jnp.array([omega]), 
        V_ia_0 = jnp.full((Inputs.n_r,), jnp.array([4.0])),
        c_scale = c_scale,
        beta_scale = beta_scale,
        error_log = jnp.array([1.0]),
        time_log = jnp.array([0])
    )

    print(f"Initialised Airfoil Fit with airfoil {params_init.airfoil}")

    # Run Optimiser
    params_opt = proptimiser.run_optimization(params_init)

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

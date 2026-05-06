from dataclasses import dataclass
import equinox as eqx
import jax.numpy as jnp
import jax.scipy as jsp

# Freeze configuration parameters at runtime
class Inputs(eqx.Module):
    # Optimisation / design settings which remain constant during execution

    # ----- Geometric parameters -----
    hub_diameter: float = 0.050 # DIAMETER in meters
    tip_diameter: float = 0.450 # DIAMETER in meters

    # max and min chords in meters
    max_chord: float = 0.050
    min_chord: float = 0.005
    
    # max and min twists in degrees
    max_twist: float = 45
    min_twist: float = -15

    # ----- Solution parameters -----
    solvertype: str = "T" # Thrust constraint (T) or torque constraint (Q)
    thrust: float = 100 # Newtons


    # ----- Optimiser parameters -----
    # Newton solver settings
    opt_max_iter: int = 1000
    opt_tol: float = 1e-2
    newton_max_iter: int = 100
    newton_eps: float = 1e-3

    #Augmented Lagrangian params
    eta_tol = 1e-2


    # ----- Discretisation parameters -----
    # Radial discretisation
    n_r: int = 20
    # Azimuthal discretisation (for skewed inflow)
    n_psi: int = 36


    # ----- Airflow model selections -----
    inflow_model: int = 1 # 0 = basic, 1 = Pitt & Peters, 2 = DTU
    tip_loss_model: int = 1 # 0 = none, 1 = Prandtl
    root_loss_model: int = 1 # 0 = none, 1 = Prandtl


    # ----- Chord and twist smoothing parameters -----
    # Set the highest power in the numerator and denominator of the chord smoothing function
    Ord_Denominator_c = 4
    Ord_Numerator_c = 2

    # Set the highest power in the numerator and denominator of the beta smoothing function
    Ord_Denominator_beta = 2
    Ord_Numerator_beta = 1


    # ----- Lookup method parameters -----
    P_Cl: int = 3 # Cl numerator order
    Q_Cl: int = 2 # Cd denominator order
    P_Cd: int = 2 # Cl numerator order
    Q_Cd: int = 2 # Cd denominator order
    
    # Lookup blending steps
    Iterations_initialising: int = 2
    Iterations_blending: int = 13
    Iterations_solving: int = 10
    

    # ----- Siagnostic and debugging flags -----
    verbose: bool = False
    record_convergence_history: bool = False
    interIterationDebug: bool = False
    Smoothing_log = False # Set whether the smoothing function reports on quality of the smoothing attempt

    

class Params(eqx.Module):
    '''
    Parameters for a given blade solution.
    Ensure that all inputs have specified shape upon assignment.
    ([None,:] for azimuthal stops, [:,None] for radial stops)

    '''
    # Domain of Integration
    r: jnp.ndarray          # radial stations
    dr: jnp.ndarray         # radial spacing (m)
    psi: jnp.ndarray        # azimuthal stations
    dpsi: jnp.ndarray       # azimuthal spacing (m)

    # Geometry & discretization
    N: jnp.ndarray          # number of blades
    R: jnp.ndarray          # propeller radius (m)
    c: jnp.ndarray          # chord length at radial stations (m)
    beta: jnp.ndarray       # blade twist angle (rad)

    # Inflow / rotor state
    V_x: jnp.ndarray        # axial inflow velocity (ms^{-1})
    V_yz: jnp.ndarray       # In plane inflow velocity (ms^{-1})
    omega: jnp.ndarray      # rotational speed (rad/s)
    #V_R: jnp.ndarray        # Net resultant velocity magnitude (ms^-1) at each radial and azimuthal station

    V_ia_0: float     # initial guess/storage for V_ia_0 (ms^{-1})

    # Physical properties
    rho: float = 1.225      # air density (kgm^{-3})
    mu: float = 1.7965e-5    # kinematic viscosity of air at 20c (N s/m**2)
    v_sound:float = 343           # speed of sound in air at standard atm (m/s)
    
    # Interpolators
    cdInterpolator: jsp.interpolate.RegularGridInterpolator = None
    clInterpolator: jsp.interpolate.RegularGridInterpolator = None
    blendingParam: float = 0

    #Blending parameter, ranges between 0->1 (full parametric -> full lookup)
    blend_alpha: float = 0.0



def blade_design_mask(params: Params):
    """
    Mask specifying which Params fields are optimized.
    True  -> optimized
    False -> frozen
    """
    return Params(
        # Domain of integration
        r=False,
        dr=False,
        psi=False,
        dpsi=False,

        # Geometry & discretization (DESIGN VARIABLES)
        N=False,
        R=False,
        c=True,
        beta=True,

        # Inflow / rotor state
        V_x=False,
        V_yz=False,
        omega=False,
        V_ia_0=False,

        # Physical properties
        rho=False
    )


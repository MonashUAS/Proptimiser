from dataclasses import dataclass
import equinox as eqx
import jax.numpy as jnp
import jax.scipy as jsp

# Freeze configuration parameters at runtime
class Inputs(eqx.Module):
    # Optimisation / design settings which remain constant during execution

    # Outer-loop optimisation (Optax / similar)
    opt_max_iter: int = 100
    opt_tol: float = 1e-3

    # Newton solver settings
    newton_max_iter: int = 1000
    newton_eps: float = 1e-3
    newton_damping: float = 0.1

    # Thrust constraint
    solvertype: str = "T"
    thrust: float = 100 # Newtons

    # Normalisation for Power (update with flat-plate guess)
    power_scale: float = 1000

    # Smoothness Constraints
    lambda_beta_smooth: float = 1e-1*power_scale
    lambda_chord_smooth: float = 1e-1*power_scale

    ## Discretisation settings
    # Radial discretisation
    n_r: int = 20
    # Azimuthal discretisation (for skewed inflow)
    n_psi: int = 18

    # Diagnostics / debugging
    verbose: bool = False
    record_convergence_history: bool = False
    interIterationDebug: bool = False

    # Model Selection
    inflow_model: int = 0 # 0 = basic, 1 = Pitt & Peters, 2 = DTU
    coeff_method: int = 2 # 0 = parametric, 1 = lookup table, 2 = blending mode
    tip_loss_model: int = 1 # 0 = none, 1 = Prandtl
    root_loss_model: int = 1 # 0 = none, 1 = Prandtl

    ## Curve smoothing parameters
    # Set the highest power in the numerator and denominator of the chord smoothing function
    Ord_Denominator_c = 4
    Ord_Numerator_c = 2
    # Set the highest power in the numerator and denominator of the beta smoothing function
    Ord_Denominator_beta = 2
    Ord_Numerator_beta = 1
    # Set whether the smoothing function reports on quality of the smoothing attempt
    Smoothing_log = False

    # Foil interpolation orders
    P_Cl: int = 3
    Q_Cl: int = 2
    P_Cd: int = 2
    Q_Cd: int = 2
    switchingDelay: int = 2
    switchingTime: int = 13
    

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


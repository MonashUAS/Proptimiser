### Establish Coefficients for Given Airfoil
import glob
import re
import numpy as np

import jax
from jax import numpy as jnp
from jax import lax
import matplotlib.pyplot as plt
import interpax

# Note: Assuming init.py and airflow.py contain the necessary structures
from init import Inputs
from airflow import resultantVelocityComponents, calc_reynolds

# Interpolator Class
class InterpaxInterpolator:
    def __init__(self, alpha_grid, re_grid_log, cl_grid, cd_grid, cm_grid):
        self.alpha_grid = jnp.array(alpha_grid)
        self.re_grid_log = jnp.array(re_grid_log)
        self.cl_grid = jnp.array(cl_grid)
        self.cd_grid = jnp.array(cd_grid)
        self.cm_grid = jnp.array(cm_grid)

    def __call__(self, alpha, re):
        # Capture original shape to restore later
        input_shape = jnp.shape(alpha)
        
        # Flatten inputs to 1D vectors to satisfy interpax internal logic
        alpha_flat = jnp.ravel(alpha)
        re_flat = jnp.ravel(re)

        # Wrap alpha to [-pi, pi]
        alpha_flat = jnp.mod(alpha_flat + jnp.pi, 2 * jnp.pi) - jnp.pi
        
        # Clip Reynolds and convert to Log10 (Add eps buffer to prevent boundary gradient NaN/collapse)
        log_re_flat = jnp.log10(re_flat)
        eps = 1e-5
        log_re_clipped = jnp.clip(log_re_flat, self.re_grid_log.min() + eps, self.re_grid_log.max() - eps)
        
        # Interpolate using flattened vectors
        # Changed method from 'cubic' to 'linear' to stabilize Broyden solver gradients
        Cl = interpax.interp2d(alpha_flat, log_re_clipped, self.alpha_grid, self.re_grid_log, self.cl_grid, method='linear')
        Cd = interpax.interp2d(alpha_flat, log_re_clipped, self.alpha_grid, self.re_grid_log, self.cd_grid, method='linear')
        Cm = interpax.interp2d(alpha_flat, log_re_clipped, self.alpha_grid, self.re_grid_log, self.cm_grid, method='linear')
        
        # Reshape back to the original input structure (Nr, Npsi)
        return Cl.reshape(input_shape), Cd.reshape(input_shape), Cm.reshape(input_shape)

def load_airfoil_data(airfoilname):
    files = glob.glob(f"airfoils/xf-{airfoilname}-il-*.csv")
    if not files:
        raise FileNotFoundError(f"No CSVs for {airfoilname}")

    alpha_grid = jnp.linspace(-jnp.pi, jnp.pi, 500)
    raw_data = []
    
    # Analytical Constants
    Cd90 = 1.98
    Cd0_default = 0.015 

    # --- HOISTED BACKGROUND ANALYTICAL MODEL ---
    # Calculate background model once (independent of Reynolds number)
    sin_a = jnp.sin(alpha_grid)
    cos_a = jnp.cos(alpha_grid)
    Cn_fp = Cd90 * sin_a / (0.56 + 0.44 * jnp.abs(sin_a))
    Ct_fp = 0.5 * Cd0_default * cos_a
    bg_cl = Cn_fp * cos_a - Ct_fp * sin_a
    bg_cd = Cn_fp * sin_a + Ct_fp * cos_a
    bg_cm = -Cn_fp * (0.25 - 0.175 * (1 - 2 * jnp.abs(alpha_grid) / jnp.pi))

    for f in files:
        match = re.search(r'-(\d+)\.csv', f)
        if match:
            Re = float(match.group(1))
            data = np.genfromtxt(f, skip_header=11, delimiter=',')
            
            # --- DEBUG POINT 1: RAW CSV LOAD ---
            if np.isnan(data).any():
                print(f"DEBUG: Found NaNs in raw CSV file: {f}")

            alpha_data = jnp.radians(data[:, 0])
            x_cl, x_cd, x_cm = jnp.array(data[:, 1]), jnp.array(data[:, 2]), jnp.array(data[:, 4])

            # 1D Interpolation with explicit NaN boundaries
            xf_cl_interp = jnp.interp(alpha_grid, alpha_data, x_cl, left=jnp.nan, right=jnp.nan)
            xf_cd_interp = jnp.interp(alpha_grid, alpha_data, x_cd, left=jnp.nan, right=jnp.nan)
            xf_cm_interp = jnp.interp(alpha_grid, alpha_data, x_cm, left=jnp.nan, right=jnp.nan)

            # Sigmoid Weighting
            min_a, max_a = alpha_data.min(), alpha_data.max()
            k = 40.0
            buffer = jnp.radians(2.0)
            mask_low = 1.0 / (1.0 + jnp.exp(-k * (alpha_grid - (min_a + buffer))))
            mask_high = 1.0 / (1.0 + jnp.exp(k * (alpha_grid - (max_a - buffer))))
            weight = mask_low * mask_high

            # Combine: If weight is 0 or interp is NaN, use background
            cl_blended = jnp.where(jnp.isnan(xf_cl_interp), bg_cl, weight * xf_cl_interp + (1 - weight) * bg_cl)
            cd_blended = jnp.where(jnp.isnan(xf_cd_interp), bg_cd, weight * xf_cd_interp + (1 - weight) * bg_cd)
            cm_blended = jnp.where(jnp.isnan(xf_cm_interp), bg_cm, weight * xf_cm_interp + (1 - weight) * bg_cm)

            # debug check: blended data
            if jnp.isnan(cl_blended).any():
                nan_count = jnp.isnan(cl_blended).sum()
                print(f"DEBUG: {nan_count} NaNs propagated into blended Cl for Re={Re}")
            
            raw_data.append((Re, cl_blended, cd_blended, cm_blended))

    raw_data.sort(key=lambda x: x[0])
    re_array = jnp.array([x[0] for x in raw_data])
    
    # --- JAX VECTORIZED 2D GRID ASSEMBLY ---
    # Stack the blended data into (N_re, N_alpha) grids
    cl_array = jnp.stack([x[1] for x in raw_data])
    cd_array = jnp.stack([x[2] for x in raw_data])
    cm_array = jnp.stack([x[3] for x in raw_data])

    re_grid_log = jnp.linspace(jnp.log10(re_array.min()), jnp.log10(re_array.max()), 15)
    log_re_array = jnp.log10(re_array)
    
    # Use jax.vmap to vectorize jnp.interp over the alpha dimension (axis 1 of our arrays)
    # jnp.interp(x, xp, fp)
    # x: re_grid_log (shared, so in_axes=None)
    # xp: log_re_array (shared, so in_axes=None)
    # fp: mapped over alpha axis (axis 1, so in_axes=1)
    interp_vmap = jax.vmap(jnp.interp, in_axes=(None, None, 1))

    cl_grid = interp_vmap(re_grid_log, log_re_array, cl_array)
    cd_grid = interp_vmap(re_grid_log, log_re_array, cd_array)
    cm_grid = interp_vmap(re_grid_log, log_re_array, cm_array)

    # final grid check/debug
    if jnp.isnan(cl_grid).any():
        print("DEBUG: Final CL Grid contains NaNs. Check Reynolds number log-spacing.")
        
    return InterpaxInterpolator(alpha_grid, re_grid_log, cl_grid, cd_grid, cm_grid)

## Parametric Function and Coefficient Method
def parametric_functions(alpha_rad, 
                        Cl_alpha=5.7,   # Lift curve slope (1/rad)
                        Cd0=0.015,      # Zero-lift drag coefficient
                        alpha_stall= jnp.radians(15), # Stall angle
                        k_smooth=30):  # Sigmoid sharpness (30-50)
    """
    Calculates Cl, Cd, and Cm for a propeller blade section using the 
    formulation from (M. Nahon, 2015) paper with sigmoid smoothing.

    Valid for Re=60,000
    """
    Cd90 = 1.98
    
    # Unstalled Regime
    Cl_lin = Cl_alpha * alpha_rad
    Ct_lin = Cd0 * jnp.cos(alpha_rad)
    Cn_lin = (Cl_lin + Ct_lin * jnp.sin(alpha_rad)) / jnp.cos(alpha_rad)
    Cd_lin = Cn_lin * jnp.sin(alpha_rad) + Ct_lin * jnp.cos(alpha_rad)
    
    # High Angle Regime
    sin_a = jnp.sin(alpha_rad)
    Cn_stall = Cd90 * sin_a / (0.56 + 0.44 * jnp.abs(sin_a))
    Ct_stall = 0.5 * Cd0 * jnp.cos(alpha_rad)
    
    Cl_stall = Cn_stall * jnp.cos(alpha_rad) - Ct_stall * jnp.sin(alpha_rad)
    Cd_stall = Cn_stall * jnp.sin(alpha_rad) + Ct_stall * jnp.cos(alpha_rad)
    
    # blend Using a sigmoid function to ensure differentiability
    transition = 1 / (1 + jnp.exp(-k_smooth * (jnp.abs(alpha_rad) - alpha_stall)))
    
    Cl = (1 - transition) * Cl_lin + transition * Cl_stall
    Cd = (1 - transition) * Cd_lin + transition * Cd_stall
    
    # Moment Coefficient (same for both formulations)
    Cn_final = (1 - transition) * Cn_lin + transition * Cn_stall
    Cm = -Cn_final * (0.25 - 0.175 * (1 - 2 * jnp.abs(alpha_rad) / jnp.pi))
    
    return Cl, Cd, Cm

def parametric_coeffs(V_ia, params):
    '''
    Calculates aerodynamic coefficients using parametric functions
    for a given induced velocity and params, including flat-plate estimate.
    '''
    # Constants:
    C_La = 5.7     # Lift curve slope per rad (ARAD10)
    C_D0 = 0.035    # Zero lift drag coefficient (ARAD10)
    AoA_0 = jnp.radians(-10) # zero lift angle of attack

    # load relevant data from inputs
    geometric_angle = params.beta[:, None]

    # calculate aerodynamic incidence
    _, phi = resultantVelocityComponents(V_ia, params) # get only second output

    # Angle of attack
    AoA_eff = geometric_angle - phi - AoA_0

    C_L, C_D, C_M = parametric_functions(AoA_eff, 
                        Cl_alpha=C_La,   # Lift curve slope (1/rad)
                        Cd0=C_D0,        # Zero-lift drag coefficient
                        alpha_stall=jnp.radians(20),  # Stall angle (degrees)
                        k_smooth=30)
    return C_L, C_D, C_M


## Interpolated Function and Coefficient Method
def interpolated_functions(AoA_eff, Re, airfoil_interp):
    '''
    Aerodynamic coefficients from Xfoil lookup tables.
    The post-stall flatplate logic is now entirely handled by the 
    offline interpolation grid, vastly simplifying this JAX loop.
    '''
    C_L, C_D, C_M = airfoil_interp(AoA_eff, Re)
    return C_L, C_D, C_M


def interpolated_coeffs(V_ia, params):

    AoA_0 = jnp.radians(-2) # zero lift angle of attack

    # load relevant data from inputs
    geometric_angle = params.beta[:, None]

    # calculate aerodynamic incidence
    V_R, phi = resultantVelocityComponents(V_ia, params) # get only second output

    # Angle of attack
    AoA_eff = geometric_angle - phi - AoA_0
    # Reynolds Number
    Re = calc_reynolds(params.rho, jnp.maximum(V_R, 1e-6), params.c[:, None], params.mu)
    # Import Airfoil Interpolator (initialise with correct airfoil at runtime)
    airfoil_interp = params.airfoil_interp

    C_L, C_D, C_M = interpolated_functions(AoA_eff, Re, airfoil_interp)
    
    return C_L, C_D, C_M


def aero_coeffs(V_ia, params):
    # Select Method to Establish Aerodynamic Coefficients
    C_L, C_D, C_M = lax.switch(Inputs.coeff_method,
                    [parametric_coeffs, interpolated_coeffs],
                    V_ia, params)
    return C_L, C_D, C_M

if __name__ == "__main__":
    print("Loading airfoil data and initializing interpolation grid...")
    try:
        # Assumes you have your CSV files in the same directory
        airfoil_interp = load_airfoil_data("arad10")
    except FileNotFoundError as e:
        print(f"Bypassing load error for demonstration purposes: {e}")
        # Create a dummy interpolator if the CSV files aren't physically present
        alpha_dummy = jnp.linspace(-jnp.pi, jnp.pi, 50)
        re_dummy = jnp.array([5.0, 6.0]) # log10(1e5), log10(1e6)
        airfoil_interp = InterpaxInterpolator(
            alpha_dummy, re_dummy, 
            jnp.zeros((50,2)), jnp.zeros((50,2)), jnp.zeros((50,2))
        )

    # Setup plotting domains
    aoa_deg = np.linspace(-180, 180, 400)
    aoa_rad = jnp.radians(aoa_deg)
    re_fixed = 1e6

    # Evaluate both methods
    cl_param, cd_param, cm_param = parametric_functions(aoa_rad)
    cl_interp, cd_interp, cm_interp = interpolated_functions(aoa_rad, re_fixed, airfoil_interp)

    # ---------------------------------------------------------
    # Plot 1: 2D Plot at Fixed Reynolds Number
    # ---------------------------------------------------------
    fig, axs = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    fig.suptitle(f'Aerodynamic Coefficients vs AoA (Re = {re_fixed:.0e})', fontsize=14)

    # Cl
    axs[0].plot(aoa_deg, cl_interp, 'b-', label='Interpolated (XFoil+FlatPlate)')
    axs[0].plot(aoa_deg, cl_param, 'b:', label='Parametric Estimate')
    axs[0].set_ylabel('$C_L$')
    axs[0].grid(True)
    axs[0].legend()

    # Cd
    axs[1].plot(aoa_deg, cd_interp, 'r-')
    axs[1].plot(aoa_deg, cd_param, 'r:')
    axs[1].set_ylabel('$C_D$')
    axs[1].grid(True)

    # Cm
    axs[2].plot(aoa_deg, cm_interp, 'g-')
    axs[2].plot(aoa_deg, cm_param, 'g:')
    axs[2].set_ylabel('$C_M$')
    axs[2].set_xlabel('Angle of Attack (Degrees)')
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()

    # ---------------------------------------------------------
    # Plot 2: 3D Surface Plots (AoA vs Re)
    # ---------------------------------------------------------
    # Create meshgrid for Surface Plot
    re_vals = np.logspace(4, 6.5, 50) # Re from 10,000 to ~3,000,000
    AOA_GRID, RE_GRID = np.meshgrid(aoa_rad, re_vals)
    
    # Vectorize the interpolation evaluation
    vmap_interp = jax.vmap(jax.vmap(airfoil_interp, in_axes=(0, 0)), in_axes=(0, 0))
    CL_surf, CD_surf, CM_surf = vmap_interp(AOA_GRID, RE_GRID)

    fig = plt.figure(figsize=(18, 6))
    
    # Cl Surface
    ax1 = fig.add_subplot(131, projection='3d')
    surf1 = ax1.plot_surface(np.degrees(AOA_GRID), np.log10(RE_GRID), CL_surf, cmap='viridis', edgecolor='none')
    ax1.set_title('$C_L$ Surface')
    ax1.set_xlabel('AoA (deg)')
    ax1.set_ylabel('Log10(Re)')
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)

    # Cd Surface
    ax2 = fig.add_subplot(132, projection='3d')
    surf2 = ax2.plot_surface(np.degrees(AOA_GRID), np.log10(RE_GRID), CD_surf, cmap='plasma', edgecolor='none')
    ax2.set_title('$C_D$ Surface')
    ax2.set_xlabel('AoA (deg)')
    ax2.set_ylabel('Log10(Re)')
    fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)

    # Cm Surface
    ax3 = fig.add_subplot(133, projection='3d')
    surf3 = ax3.plot_surface(np.degrees(AOA_GRID), np.log10(RE_GRID), CM_surf, cmap='cividis', edgecolor='none')
    ax3.set_title('$C_M$ Surface')
    ax3.set_xlabel('AoA (deg)')
    ax3.set_ylabel('Log10(Re)')
    fig.colorbar(surf3, ax=ax3, shrink=0.5, aspect=5)

    plt.tight_layout()
    plt.show()
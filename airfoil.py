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
class AlphaReInterpolator:
    def __init__(self, alpha_grid, re_grid_log, cl_grid, cd_grid, cm_grid):
        self.alpha_grid = jnp.array(alpha_grid)
        self.re_grid_log = jnp.array(re_grid_log)
        self.cl_grid = jnp.array(cl_grid)
        self.cd_grid = jnp.array(cd_grid)
        self.cm_grid = jnp.array(cm_grid)

    def __call__(self, alpha, mach, re):
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

class AlphaMachReInterpolator:
    '''
    Interpolator Class for Alpha, Mach and Re. 
    Currently set up to work with xflr5 files.
    '''

    def __init__(self, alpha_grid, mach_grid, re_grid_log, cl_grid, cd_grid, cm_grid):
        self.alpha_grid = jnp.array(alpha_grid)
        self.mach_grid = jnp.array(mach_grid)
        self.re_grid_log = jnp.array(re_grid_log)
        
        # Grids are shape (N_alpha, N_mach, N_re)
        self.cl_grid = jnp.array(cl_grid)
        self.cd_grid = jnp.array(cd_grid)
        self.cm_grid = jnp.array(cm_grid)

    def __call__(self, alpha, mach, re):
        # Capture original shape to restore later
        input_shape = jnp.shape(alpha)
        
        # Flatten inputs to 1D vectors for interpax
        alpha_flat = jnp.ravel(alpha)
        mach_flat = jnp.ravel(mach)
        re_flat = jnp.ravel(re)

        # Wrap alpha to [-pi, pi]
        alpha_flat = jnp.mod(alpha_flat + jnp.pi, 2 * jnp.pi) - jnp.pi
        
        # Clip inputs to prevent NaNs at extrapolation boundaries
        eps = 1e-5
        log_re_flat = jnp.log10(re_flat)
        log_re_clipped = jnp.clip(log_re_flat, self.re_grid_log.min() + eps, self.re_grid_log.max() - eps)
        mach_clipped = jnp.clip(mach_flat, self.mach_grid.min() + eps, self.mach_grid.max() - eps)
        
        # 3D Interpolation (Alpha, Mach, Reynolds)
        Cl = interpax.interp3d(alpha_flat, mach_clipped, log_re_clipped, 
                               self.alpha_grid, self.mach_grid, self.re_grid_log, 
                               self.cl_grid, method='linear')
        Cd = interpax.interp3d(alpha_flat, mach_clipped, log_re_clipped, 
                               self.alpha_grid, self.mach_grid, self.re_grid_log, 
                               self.cd_grid, method='linear')
        Cm = interpax.interp3d(alpha_flat, mach_clipped, log_re_clipped, 
                               self.alpha_grid, self.mach_grid, self.re_grid_log, 
                               self.cm_grid, method='linear')
        return Cl.reshape(input_shape), Cd.reshape(input_shape), Cm.reshape(input_shape)


@jax.jit
def jax_idw_3d_grid(scattered_points, scattered_values, grid_X, grid_Y, power=2.0, eps=1e-8):
    """
    Vectorized Inverse Distance Weighting (IDW) interpolation in pure JAX.
    Interpolates unstructured data onto a structured grid for multiple slices (AoA) simultaneously.
    """
    # Normalize coordinates to prevent skewed distance metrics (Mach vs Log(Re) scales)
    p_min = jnp.min(scattered_points, axis=0)
    p_max = jnp.max(scattered_points, axis=0)
    p_scale = jnp.maximum(p_max - p_min, 1e-5)

    norm_points = (scattered_points - p_min) / p_scale
    norm_grid_X = (grid_X - p_min[0]) / p_scale[0]
    norm_grid_Y = (grid_Y - p_min[1]) / p_scale[1]

    # Flatten grid for pairwise distance calculation
    flat_grid = jnp.stack([norm_grid_X.ravel(), norm_grid_Y.ravel()], axis=-1)

    # Calculate squared distances: shape (N_grid_points, N_scattered_points)
    diff = flat_grid[:, None, :] - norm_points[None, :, :]
    dist_sq = jnp.sum(diff**2, axis=-1)

    # Calculate weights: shape (N_grid, N_pts)
    weights = 1.0 / (dist_sq**(power / 2.0) + eps)
    weights_sum = jnp.sum(weights, axis=-1)

    # Einsum dynamically multiplies weights by values for ALL alpha slices at once:
    # 'gp' (grid_points) * 'ap' (alpha_points) -> 'ag' (alpha_grid)
    interp_flat = jnp.einsum('gp,ap->ag', weights, scattered_values) / weights_sum[None, :]

    # Reshape back to (N_alpha, N_Mach, N_Re)
    return interp_flat.reshape(scattered_values.shape[0], grid_X.shape[0], grid_X.shape[1])

def load_airfoil_data(airfoilname):
    if Inputs.coeff_method==1:
        '''
        Fit 2D Alpha/Reynolds Interpolator
        '''
        files = glob.glob(f"airfoils/xf-{airfoilname}-il-*.csv")
        if not files:
            raise FileNotFoundError(f"No CSVs for {airfoilname}")

        alpha_grid = jnp.linspace(-jnp.pi, jnp.pi, 500)
        raw_data = []
        
        # Analytical Constants
        Cd90 = 1.98
        Cd0_default = 0.015 

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
                if jnp.isnan(data).any():
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

        interp_vmap = jax.vmap(jnp.interp, in_axes=(None, None, 1))

        cl_grid = interp_vmap(re_grid_log, log_re_array, cl_array)
        cd_grid = interp_vmap(re_grid_log, log_re_array, cd_array)
        cm_grid = interp_vmap(re_grid_log, log_re_array, cm_array)

        # final grid check/debug
        if jnp.isnan(cl_grid).any():
            print("DEBUG: Final CL Grid contains NaNs. Check Reynolds number log-spacing.")
            
        return AlphaReInterpolator(alpha_grid, re_grid_log, cl_grid, cd_grid, cm_grid)
    
    elif Inputs.coeff_method == 2:
        '''
        Create 3D Alpha/Mach/Reynolds interpolator
        '''
        files = glob.glob(f"airfoils/{airfoilname}/Polars/*.csv")
        if not files:
            raise FileNotFoundError(f"No CSVs found for 3D interpolation in airfoils/{airfoilname}/Polars/")

        alpha_grid = jnp.linspace(-jnp.pi, jnp.pi, 500)
        raw_data = []

        # Background Analytical Model (Flat Plate)
        Cd90 = 1.98
        Cd0_default = 0.015 
        sin_a = jnp.sin(alpha_grid)
        cos_a = jnp.cos(alpha_grid)
        Cn_fp = Cd90 * sin_a / (0.56 + 0.44 * jnp.abs(sin_a))
        Ct_fp = 0.5 * Cd0_default * cos_a
        bg_cl = Cn_fp * cos_a - Ct_fp * sin_a
        bg_cd = Cn_fp * sin_a + Ct_fp * cos_a
        bg_cm = -Cn_fp * (0.25 - 0.175 * (1 - 2 * jnp.abs(alpha_grid) / jnp.pi))

        for f in files:
            match = re.search(r'Re([\d\.]+)_M([\d\.]+)', f)
            if match:
                Re = float(match.group(1)) * 1e6 
                Mach = float(match.group(2))
                
                # File I/O remains standard NumPy (JAX doesn't read CSVs natively)
                data = np.genfromtxt(f, skip_header=11, delimiter=',')
                if np.isnan(data).any() or len(data) == 0:
                    continue

                alpha_data = jnp.radians(jnp.array(data[:, 0]))
                x_cl, x_cd, x_cm = jnp.array(data[:, 1]), jnp.array(data[:, 2]), jnp.array(data[:, 4])

                # 1D Interpolation onto unified alpha grid
                xf_cl_interp = jnp.interp(alpha_grid, alpha_data, x_cl, left=jnp.nan, right=jnp.nan)
                xf_cd_interp = jnp.interp(alpha_grid, alpha_data, x_cd, left=jnp.nan, right=jnp.nan)
                xf_cm_interp = jnp.interp(alpha_grid, alpha_data, x_cm, left=jnp.nan, right=jnp.nan)

                # Sigmoid Weighting for Flat-Plate Blending
                min_a, max_a = alpha_data.min(), alpha_data.max()
                k = 40.0
                buffer = jnp.radians(2.0)
                mask_low = 1.0 / (1.0 + jnp.exp(-k * (alpha_grid - (min_a + buffer))))
                mask_high = 1.0 / (1.0 + jnp.exp(k * (alpha_grid - (max_a - buffer))))
                weight = mask_low * mask_high

                cl_blended = jnp.where(jnp.isnan(xf_cl_interp), bg_cl, weight * xf_cl_interp + (1 - weight) * bg_cl)
                cd_blended = jnp.where(jnp.isnan(xf_cd_interp), bg_cd, weight * xf_cd_interp + (1 - weight) * bg_cd)
                cm_blended = jnp.where(jnp.isnan(xf_cm_interp), bg_cm, weight * xf_cm_interp + (1 - weight) * bg_cm)

                raw_data.append((Mach, Re, cl_blended, cd_blended, cm_blended))

        if not raw_data:
            raise ValueError("No valid data extracted from XFLR5 CSVs.")

        # --- JAX-NATIVE 3D GRID ASSEMBLY ---
        
        mach_points = jnp.array([d[0] for d in raw_data])
        re_points = jnp.array([d[1] for d in raw_data])
        log_re_points = jnp.log10(re_points)

        # Create structured 2D grid boundaries for (Mach, Log_Re)
        mach_grid = jnp.linspace(mach_points.min(), mach_points.max(), 10)
        re_grid_log = jnp.linspace(log_re_points.min(), log_re_points.max(), 15)
        M_mesh, Re_mesh = jnp.meshgrid(mach_grid, re_grid_log, indexing='ij')

        # Unstructured coordinate list (N_pts, 2)
        points = jnp.column_stack((mach_points, log_re_points))

        # Stack values so they are shape (N_alpha, N_pts)
        cl_vals_all = jnp.stack([d[2] for d in raw_data], axis=-1)
        cd_vals_all = jnp.stack([d[3] for d in raw_data], axis=-1)
        cm_vals_all = jnp.stack([d[4] for d in raw_data], axis=-1)

        # Perform JIT-compiled Inverse Distance Weighting for all Alpha slices simultaneously
        cl_grid = jax_idw_3d_grid(points, cl_vals_all, M_mesh, Re_mesh)
        cd_grid = jax_idw_3d_grid(points, cd_vals_all, M_mesh, Re_mesh)
        cm_grid = jax_idw_3d_grid(points, cm_vals_all, M_mesh, Re_mesh)

        return AlphaMachReInterpolator(alpha_grid, mach_grid, re_grid_log, cl_grid, cd_grid, cm_grid)
    
    else:
        print("Incorrect coefficient method selected (check Inputs)")
        return None

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
def interpolated_functions(AoA_eff, Mach, Re, airfoil_interp):
    '''
    Aerodynamic coefficients from Xfoil/XFLR5 lookup tables.
    '''
    C_L, C_D, C_M = airfoil_interp(AoA_eff, Mach, Re)
    return C_L, C_D, C_M

def xfoil_coeffs(V_ia, params):
    AoA_0 = jnp.radians(-2) # zero lift angle of attack

    geometric_angle = params.beta[:, None]
    V_R, phi = resultantVelocityComponents(V_ia, params) 

    AoA_eff = geometric_angle - phi - AoA_0
    Re = calc_reynolds(params.rho, jnp.maximum(V_R, 1e-6), params.c[:, None], params.mu)
    
    # Calculate Mach number (Assume speed of sound is present in params, fallback to 343.0 m/s)
    speed_of_sound = getattr(params, 'a', 343.0) 
    Mach = V_R / speed_of_sound

    airfoil_interp = params.airfoil_interp
    C_L, C_D, C_M = interpolated_functions(AoA_eff, Mach, Re, airfoil_interp)
    
    return C_L, C_D, C_M

def xflr5_coeffs(V_ia, params):
    AoA_0 = jnp.radians(-2) # zero lift angle of attack

    geometric_angle = params.beta[:, None]
    V_R, phi = resultantVelocityComponents(V_ia, params) 

    AoA_eff = geometric_angle - phi - AoA_0
    Re = calc_reynolds(params.rho, jnp.maximum(V_R, 1e-6), params.c[:, None], params.mu)
    
    # Calculate Mach number (Assume speed of sound is present in params, fallback to 343.0 m/s)
    speed_of_sound = getattr(params, 'a', 343.0) 
    Mach = V_R / speed_of_sound

    airfoil_interp = params.airfoil_interp
    C_L, C_D, C_M = interpolated_functions(AoA_eff, Mach, Re, airfoil_interp)

    return C_L, C_D, C_M

def aero_coeffs(V_ia, params):
    # Select Method to Establish Aerodynamic Coefficients
    C_L, C_D, C_M = lax.switch(Inputs.coeff_method,
                    [parametric_coeffs, xfoil_coeffs, xflr5_coeffs],
                    V_ia, params)
    return C_L, C_D, C_M

if __name__ == "__main__":
    # mmm pretty latex
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "axes.labelsize": 12,
        "axes.titlesize": 14
    })
    print("Loading airfoil data and initializing interpolation grid...")
    try:
        # Assumes you have your CSV files in the same directory
        airfoil_interp = load_airfoil_data("arad10")
    except FileNotFoundError as e:
        print(f"Bypassing load error for demonstration purposes: {e}")
        # Create a dummy interpolator if the CSV files aren't physically present
        alpha_dummy = jnp.linspace(-jnp.pi, jnp.pi, 50)
        re_dummy = jnp.array([5.0, 6.0]) # log10(1e5), log10(1e6)
        airfoil_interp = AlphaReInterpolator(
            alpha_dummy, re_dummy, 
            jnp.zeros((50,2)), jnp.zeros((50,2)), jnp.zeros((50,2))
        )

    # Setup plotting domains
    aoa_deg = np.linspace(-180, 180, 400)
    aoa_rad = jnp.radians(aoa_deg)
    re_fixed = 1e6
    mach_fixed = 0.2

    # Evaluate both methods
    cl_param, cd_param, cm_param = parametric_functions(aoa_rad)
    cl_interp, cd_interp, cm_interp = interpolated_functions(aoa_rad, re_fixed, mach_fixed, airfoil_interp)

    # Plot 1: 2D Plot at Fixed Reynolds Number
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

    # Plot 2: 3D Surface Plots (AoA vs Re)
    # Create meshgrid for Surface Plot
    re_vals = np.logspace(4, 6.5, 50) # Re from 10,000 to ~3,000,000
    mach_constant = 0.3
    
    # Generate 2D meshgrids exactly like your 2D interpolation script
    AOA_GRID, RE_GRID = np.meshgrid(aoa_rad, re_vals)
    
    # Broadcast the constant Mach number to match the 2D grid shape (50, 500)
    MACH_GRID = jnp.full_like(AOA_GRID, mach_constant)
    
    # Vectorize the interpolation evaluation across the rows and columns
    vmap_interp = jax.vmap(jax.vmap(airfoil_interp, in_axes=(0, 0, 0)), in_axes=(0, 0, 0))
    CL_surf, CD_surf, CM_surf = vmap_interp(AOA_GRID, MACH_GRID, RE_GRID)

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
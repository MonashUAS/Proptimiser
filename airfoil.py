### Establish Coefficients for Given Airfoil
import glob
import re
import numpy as np

import jax
from jax import numpy as jnp
from jax import lax
import matplotlib.pyplot as plt
import interpax
from scipy.ndimage import gaussian_filter
import neuralfoil as nf
import aerosandbox as asb
from pathlib import Path

# Note: Assuming init.py and airflow.py contain the necessary structures
from init import Inputs
from airflow import resultantVelocityComponents, calc_reynolds

def apply_mach_corrections(cl_inc, cd_inc, cm_inc, mach, t_over_c=0.10):
    """
    Advanced compressibility correction handling high-subsonic drag rise.
    
    Parameters:
    -----------
    cl_inc, cd_inc, cm_inc : JAX arrays
        Incompressible coefficients.
    mach : JAX array
        Local section Mach number.
    t_over_c : float
        Airfoil thickness-to-chord ratio (e.g., 0.10 for a 10% thick blade section).
    """
    mach_abs = jnp.abs(mach)
    mach_safe = jnp.clip(mach_abs, 0.0, 0.98)
    beta = jnp.sqrt(1.0 - mach_safe**2)
    
    # Lift and Moment corrections (Standard Karman-Tsien / PG)
    cl_comp = cl_inc / (beta + (mach_safe**2 / (1.0 + beta)) * (jnp.abs(cl_inc) / 2.0))
    cm_comp = cm_inc / beta
    
    # High-Subsonic Drag Correction
    # Subsonic scaling with a Lift-Dependent Penalty
    cd_subsonic = cd_inc * (1.0 / beta + 1.8 * (mach_safe**2) * (cl_inc**2))

    # Estimate the Critical Mach Number (M_crit) where local flow hits Mach 1.
    # Thicker airfoils and higher operating lift coefficients lower M_crit.
    m_crit_0 = 0.90 - 1.4 * t_over_c  # Zero-lift critical Mach (e.g., 0.76 for 10% thickness)
    m_crit = m_crit_0 - 0.12 * jnp.abs(cl_comp)
    
    # Smoothly calculate how far past M_crit the flow is operating
    delta_m = mach_abs - m_crit
    delta_m_positive = jnp.maximum(0.0, delta_m)
    
    # Empirical wave drag penalty (cubic rise once past M_crit)
    cd_wave = 2.5 * (delta_m_positive ** 3)
    
    # Combine baseline profile drag scaling and compressible wave drag
    cd_comp = cd_subsonic + cd_wave
    
    return cl_comp, cd_comp, cm_comp

### Interpolator Classes 

class AlphaReInterpolator:
    def __init__(self, alpha_grid, re_grid_log, cl_grid, cd_grid, cm_grid):
        self.alpha_grid = jnp.array(alpha_grid)
        self.re_grid_log = jnp.array(re_grid_log)
        self.cl_grid = jnp.array(cl_grid)
        self.cd_grid = jnp.array(cd_grid)
        self.cm_grid = jnp.array(cm_grid)

    def __call__(self, alpha, re, mach):
        # Capture original shape to restore later
        input_shape = jnp.shape(alpha)
        
        # Flatten inputs to 1D vectors to satisfy interpax internal logic
        alpha_flat = jnp.ravel(alpha)
        re_flat = jnp.ravel(re)
        mach_flat = jnp.ravel(mach)

        # Wrap alpha to [-pi, pi]
        alpha_flat = jnp.mod(alpha_flat + jnp.pi, 2 * jnp.pi) - jnp.pi
        
        # Clip Reynolds and convert to Log10 (Add eps buffer to prevent boundary gradient NaN/collapse)
        log_re_flat = jnp.log10(re_flat)
        eps = 1e-5
        log_re_clipped = jnp.clip(log_re_flat, self.re_grid_log.min() + eps, self.re_grid_log.max() - eps)
        
        # Interpolate using flattened vectors
        Cl = interpax.interp2d(alpha_flat, log_re_clipped, self.alpha_grid, self.re_grid_log, self.cl_grid, method='linear')
        Cd = interpax.interp2d(alpha_flat, log_re_clipped, self.alpha_grid, self.re_grid_log, self.cd_grid, method='linear')
        Cm = interpax.interp2d(alpha_flat, log_re_clipped, self.alpha_grid, self.re_grid_log, self.cm_grid, method='linear')
        # Apply Compressibility corrections
        cl_comp, cd_comp, cm_comp = apply_mach_corrections(Cl, Cd, Cm, mach_flat)
        
        # Reshape back to original input shape
        cl_out = jnp.reshape(cl_comp, input_shape)
        cd_out = jnp.reshape(cd_comp, input_shape)
        cm_out = jnp.reshape(cm_comp, input_shape)
        # Reshape back to the original input structure (Nr, Npsi)
        return cl_out, cd_out, cm_out

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

    def __call__(self, alpha, re, mach):
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
                               self.cl_grid, method='cubic')
        Cd = interpax.interp3d(alpha_flat, mach_clipped, log_re_clipped, 
                               self.alpha_grid, self.mach_grid, self.re_grid_log, 
                               self.cd_grid, method='cubic')
        Cm = interpax.interp3d(alpha_flat, mach_clipped, log_re_clipped, 
                               self.alpha_grid, self.mach_grid, self.re_grid_log, 
                               self.cm_grid, method='cubic')
        return Cl.reshape(input_shape), Cd.reshape(input_shape), Cm.reshape(input_shape)

class NeuralFoilInterpolator:
    '''
    A pure-JAX implementation of the NeuralFoil multi-layer perceptron (MLP) for aerodynamic coefficient prediction.
    By staying in JAX, this interpolator is 100% compatible with JIT compilation and autodiff.

    Implementation mostly cloned from https://github.com/peterdsharpe/NeuralFoil/blob/master/neuralfoil/main.py
    '''
    def __init__(self, airfoilname, model_size="large"):
        # Attempt to load the airfoil geometry via aerosandbox (or locally from .dat)
        try:
            self.airfoil = asb.Airfoil(airfoilname).normalize()
        except Exception:
            try:
                # Fall back to local .dat file in the airfoils directory
                files = glob.glob(f"airfoils/{airfoilname}/*.dat")
                if files:
                    self.airfoil = asb.Airfoil(coordinates=files[0]).normalize()
                else:
                    raise FileNotFoundError(f"Cannot find airfoil {airfoilname}")
            except Exception as e:
                raise ValueError(f"Failed to load airfoil {airfoilname}: {e}")
        
        # Extract and scale the geometric "Kulfan" parameters, which the neural net uses as shape inputs
        normalization_outputs = self.airfoil.normalize(return_dict=True)
        normalized_airfoil = normalization_outputs["airfoil"].to_kulfan_airfoil(
            n_weights_per_side=8,
            normalize_coordinates=False, 
        )
        self.kulfan_parameters = {k: jnp.array(v) for k, v in normalized_airfoil.kulfan_parameters.items()}
        self.delta_alpha = jnp.array(normalization_outputs["rotation_angle"])
        
        # Calculate coordinate translations to correct the moment coefficient (CM) later
        self.x_translation_qc = jnp.array(-normalization_outputs["x_translation"] + 0.25 * (1 / normalization_outputs["scale_factor"] * jnp.cos(jnp.radians(normalization_outputs["rotation_angle"]))) - 0.25)
        self.y_translation_qc = jnp.array(-normalization_outputs["y_translation"] + 0.25 * (1 / normalization_outputs["scale_factor"] * jnp.sin(jnp.radians(-normalization_outputs["rotation_angle"]))))
        self.scale = jnp.array(normalization_outputs["scale_factor"])
        
        # Load the pre-trained neural network weights and input scaling distributions from the neuralfoil package
        nn_weights_dir = Path(nf.__file__).parent / "nn_weights_and_biases"
        self._scaled_input_distribution = {k: jnp.array(v) for k, v in dict(np.load(nn_weights_dir / "scaled_input_distribution.npz")).items()}
        
        nn_params_raw = dict(np.load(nn_weights_dir / f"nn-{model_size}.npz"))
        self.nn_params = {k: jnp.array(v) for k, v in nn_params_raw.items()}
        
        self.layer_indices = sorted(list(set([int(key.split(".")[1]) for key in nn_params_raw.keys()])))
        self.N_inputs = len(self._scaled_input_distribution["mean_inputs_scaled"])

    def __call__(self, alpha, re, mach):
        # 1. Wrap alpha to [-pi, pi] preserving original shape
        alpha = jnp.mod(alpha + jnp.pi, 2 * jnp.pi) - jnp.pi
        
        # Broadcast arrays together to unify their shapes automatically
        alpha, re = jnp.broadcast_arrays(alpha, re)
        
        re_scaled = re / self.scale
        alpha_deg = jnp.degrees(alpha) + self.delta_alpha
        
        k = self.kulfan_parameters
        
        # 2. Assemble the 25-dimensional input vector along the LAST axis (-1)
        # Using jnp.full_like preserves the broadcasted leading dimensions dynamically
        x = jnp.stack([
            *[jnp.full_like(alpha_deg, k["upper_weights"][i]) for i in range(8)],
            *[jnp.full_like(alpha_deg, k["lower_weights"][i]) for i in range(8)],
            jnp.full_like(alpha_deg, k["leading_edge_weight"]),
            jnp.full_like(alpha_deg, k["TE_thickness"] * 50),
            jnp.sin(jnp.radians(2 * alpha_deg)),
            jnp.cos(jnp.radians(alpha_deg)),
            1 - jnp.cos(jnp.radians(alpha_deg)) ** 2,
            (jnp.log(re_scaled) - 12.5) / 3.5,
            jnp.zeros_like(alpha_deg), # n_crit scaling
            jnp.ones_like(alpha_deg),  # forced transition upper
            jnp.ones_like(alpha_deg),  # forced transition lower
        ], axis=-1) # Shape: (*input_shape, 25)
        
        # 3. Leading-dimension safe MLP using jnp.matmul
        def net(x_in):
            layer_indices_to_iterate = self.layer_indices.copy()
            while len(layer_indices_to_iterate) != 0:
                i = layer_indices_to_iterate.pop(0)
                w = self.nn_params[f"net.{i}.weight"]
                b = self.nn_params[f"net.{i}.bias"]
                
                # Using jnp.matmul and w.T allows broadcasting across all leading shapes
                x_in = jnp.matmul(x_in, w.T) + b
                if len(layer_indices_to_iterate) != 0:
                    x_in = jax.nn.swish(x_in)
            return x_in # Shape: (*input_shape, out_features)
            
        def sq_mahalanobis(x_in):
            mean = self._scaled_input_distribution["mean_inputs_scaled"]
            inv_cov = self._scaled_input_distribution["inv_cov_inputs_scaled"]
            
            x_minus_mean = x_in - mean # Broadcasts mean perfectly across trailing axis
            # Compute Mahalanobis distance strictly focusing on the final feature axis
            return jnp.sum(jnp.matmul(x_minus_mean, inv_cov) * x_minus_mean, axis=-1)

        # Evaluate the network
        y = net(x)
        y = y.at[..., 0].set(y[..., 0] - sq_mahalanobis(x) / (2 * self.N_inputs))
        
        # 4. Enforce symmetric consistency using slice syntax (...)
        x_flipped = x \
            .at[..., :8].set(x[..., 8:16] * -1) \
            .at[..., 8:16].set(x[..., :8] * -1) \
            .at[..., 16].set(-1 * x[..., 16]) \
            .at[..., 18].set(-1 * x[..., 18]) \
            .at[..., 23].set(x[..., 24]) \
            .at[..., 24].set(x[..., 23])
        
        y_flipped = net(x_flipped)
        y_flipped = y_flipped.at[..., 0].set(y_flipped[..., 0] - sq_mahalanobis(x_flipped) / (2 * self.N_inputs))
        
        y_unflipped = y_flipped \
            .at[..., 1].set(y_flipped[..., 1] * -1) \
            .at[..., 3].set(y_flipped[..., 3] * -1) \
            .at[..., 4].set(y_flipped[..., 5]) \
            .at[..., 5].set(y_flipped[..., 4])
        
        y_fused = (y + y_unflipped) / 2
        
        # Extract coefficients
        CL_raw = y_fused[..., 1] / 2
        CD_raw = jnp.exp((y_fused[..., 2] - 2) * 2)
        CM_raw = y_fused[..., 3] / 20
        
        # Coordinate translations
        extra_CM = -CL_raw * self.x_translation_qc + CD_raw * self.y_translation_qc
        CM_raw = CM_raw + extra_CM
        
        # Flat-Plate Blending
        alpha_trust_low = jnp.radians(-11.0)
        alpha_trust_high = jnp.radians(13.0)
        
        Cd90 = 1.98
        Cd0_default = 0.015 
        sin_a = jnp.sin(alpha)
        cos_a = jnp.cos(alpha)
        Cn_fp = Cd90 * sin_a / (0.56 + 0.44 * jnp.abs(sin_a))
        Ct_fp = 0.5 * Cd0_default * cos_a
        
        bg_cl = Cn_fp * cos_a - Ct_fp * sin_a
        bg_cd = Cn_fp * sin_a + Ct_fp * cos_a
        bg_cm = -Cn_fp * (0.25 - 0.175 * (1 - 2 * jnp.abs(alpha) / jnp.pi))
        
        k_smooth = 40.0 
        mask_low = 1.0 / (1.0 + jnp.exp(-k_smooth * (alpha - alpha_trust_low)))
        mask_high = 1.0 / (1.0 + jnp.exp(k_smooth * (alpha - alpha_trust_high)))
        weight = mask_low * mask_high
        
        CL_inc = weight * CL_raw + (1 - weight) * bg_cl
        CD_inc = weight * CD_raw + (1 - weight) * bg_cd
        CM_inc = weight * CM_raw + (1 - weight) * bg_cm

        # Apply Mach corrections (which handles shapes smoothly natively)
        return apply_mach_corrections(CL_inc, CD_inc, CM_inc, mach)

# ==========================================
# 2. GRID UTILITIES & DATA LOADING
# ==========================================

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
    '''
    Loads airfoil data and constructs interpolator function.
    '''
    if Inputs.coeff_method<2:
        '''
        Fit 2D Alpha/Reynolds Interpolator
        '''
        files = glob.glob(f"airfoils/ARAD10/xfoil/xf-{airfoilname}-il-*.csv")
        if not files:
            raise FileNotFoundError(f"No CSVs for {airfoilname}")

        # Define regions of interest (in degrees, then convert to radians)
        alpha_deg = jnp.concatenate([
            jnp.linspace(-180, -30, 10),    # Deep negative stall (Coarse)
            jnp.linspace(-29, -12, 8),     # Negative stall transition (Medium)
            jnp.linspace(-11, 15, 27),     # Linear lift & Positive Stall (Fine: ~1 deg spacing)
            jnp.linspace(16, 35, 10),      # Post-stall recovery (Medium)
            jnp.linspace(36, 180, 11)      # Deep positive stall (Coarse)
        ])

        # Remove any accidental duplicates at the boundaries and sort
        alpha_grid = jnp.unique(jnp.radians(alpha_deg))
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

        # Define regions of interest (in degrees, then convert to radians)
        alpha_deg = jnp.concatenate([
            jnp.linspace(-180, -30, 10),    # Deep negative stall (Coarse)
            jnp.linspace(-29, -12, 8),     # Negative stall transition (Medium)
            jnp.linspace(-11, 15, 27),     # Linear lift & Positive Stall (Fine: ~1 deg spacing)
            jnp.linspace(16, 35, 10),      # Post-stall recovery (Medium)
            jnp.linspace(36, 180, 11)      # Deep positive stall (Coarse)
        ])

        # Remove any accidental duplicates at the boundaries and sort
        alpha_grid = jnp.unique(jnp.radians(alpha_deg))
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
        
        # Smooth along AoA and Reynolds Axes
        cl_grid_smooth = gaussian_filter(cl_grid, sigma=(1.5, 0.8, 1.0))
        cd_grid_smooth = gaussian_filter(cd_grid, sigma=(1.5, 0.8, 1.0))
        cm_grid_smooth = gaussian_filter(cm_grid, sigma=(1.5, 0.8, 1.0))

        return AlphaMachReInterpolator(alpha_grid, mach_grid, re_grid_log, cl_grid_smooth, cd_grid_smooth, cm_grid_smooth)
    
    elif Inputs.coeff_method == 3:
        '''
        NeuralFoil implementation using native JAX multi-layer perceptron.
        '''
        return NeuralFoilInterpolator(airfoilname)

    else:
        print("Incorrect coefficient method selected (check Inputs)")
        return None

# Aerodynamic Methods

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

def interpolated_coeffs(V_ia, params):
    '''
    Aerodynamic coefficients from Xfoil, XFLR5, or NeuralFoil solvers.
    This unifies the pipeline as the underlying JAX interpolator abstractly handles the method.
    '''
    geometric_angle = params.beta[:, None]
    V_R, phi = resultantVelocityComponents(V_ia, params) 

    # No zero-lift offset applied to geometric solvers
    AoA_eff = geometric_angle - phi
    
    Re = calc_reynolds(params.rho, jnp.maximum(V_R, 1e-6), params.c[:, None], params.mu)
    
    # Calculate Mach number (Assume speed of sound is present in params, fallback to 343.0 m/s)
    speed_of_sound = getattr(params, 'a', 343.0) 
    Mach = V_R / speed_of_sound

    airfoil_interp = params.airfoil_interp
    C_L, C_D, C_M = airfoil_interp(AoA_eff, Re, Mach)
    
    return C_L, C_D, C_M

## 3D ROTATIONAL STALL DELAY CORRECTIONS

def no_3d_correction(operands):
    '''
    Pass-through function for when rotational_correction == 0.
    '''
    C_L_2D, C_D_2D, C_M, V_ia, params = operands
    return C_L_2D, C_D_2D, C_M

def du_selig_correction(operands):
    '''
    Applies the Du-Selig 3D Rotational Stall Delay correction.
    Modifies C_L and C_D based on local chord-to-radius ratios.
    '''
    C_L_2D, C_D_2D, C_M, V_ia, params = operands
    
    # Extract geometry and prevent division by zero at the exact root
    c = params.c[:, None]
    r = params.r[:, None]
    R = Inputs.R_tip*1.05
    
    r_safe = jnp.maximum(r, 1e-6)
    c_over_r = c / r_safe
    r_over_R = r_safe / R
    
    # Calculate Effective Angle of Attack (AoA_eff)
    geometric_angle = params.beta[:, None]
    _, phi = resultantVelocityComponents(V_ia, params)
    AoA_eff = geometric_angle - phi
    
    # Calculate Du-Selig empirical factor (f1 = f2)
    denominator = 0.1267 * c_over_r + (1 - r_over_R**2) / r_over_R
    f1 = (1 / (2 * jnp.pi)) * (1.6 * c_over_r) / jnp.maximum(denominator, 1e-6)

    tip_fade = 1.0 - (r_over_R ** 2)
    f1 = f1 * tip_fade
    
    # Correct Lift
    # C_L_inv is the theoretical 2D inviscid lift (2 * pi * AoA_eff)
    C_L_inv = 2 * jnp.pi * (AoA_eff + 0.035) # Estimate zero-lift angle
    C_L_3D = C_L_2D + f1 * (C_L_inv - C_L_2D)
    
    # Correct Drag
    C_D0 = 0.015 # zero-lift drag
    C_D_3D = C_D_2D - f1 * (C_D_2D - C_D0)
    
    # Ensure physical drag bounds (prevents f1 from forcing drag below baseline)
    C_D_3D = jnp.maximum(C_D_3D, C_D0)
    
    return C_L_3D, C_D_3D, C_M

## jax dispatch pipeline

def aero_coeffs(V_ia, params):
    # Route to parametric (0) or interpolated (1) based on static config.
    # Using int() ensures this is a Python-level constant at JIT trace-time,
    # which is required for lax.switch — a traced jnp.where index would fail.
    method_idx = int(Inputs.coeff_method > 0)
    
    C_L_2D, C_D_2D, C_M = lax.switch(
        method_idx,
        [parametric_coeffs, interpolated_coeffs],
        V_ia, params 
    )
    
    # Apply 3D Rotational Corrections
    operands = (C_L_2D, C_D_2D, C_M, V_ia, params)
    C_L_final, C_D_final, C_M_final = lax.switch(
        Inputs.rot_corr_model,
        [no_3d_correction, du_selig_correction],
        operands
    )
    
    return C_L_final, C_D_final, C_M_final

## Testing and Plotting

if __name__ == "__main__":
    # mmm pretty LaTeX formatting
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "legend.fontsize": 10
    })
    
    print("Initializing interpolator architectures...")
    # Setup plotting domains
    aoa_deg_360 = np.linspace(-180, 180, 400)
    aoa_rad_360 = jnp.radians(aoa_deg_360)
    re_fixed = 1e6
    mach_fixed = 0.2

    # Evaluate Parametric
    cl_param, cd_param, cm_param = parametric_functions(aoa_rad_360)

    # Evaluate XFoil / XFLR5
    Inputs.coeff_method = 2
    xfoil_interp = load_airfoil_data("arad10")
    cl_xf, cd_xf, cm_xf = xfoil_interp(aoa_rad_360, re_fixed, mach_fixed)

    # Evaluate NeuralFoil
    Inputs.coeff_method = 3
    nf_interp = load_airfoil_data("arad10")
    cl_nf, cd_nf, cm_nf = nf_interp(aoa_rad_360, re_fixed, mach_fixed)

    active_interp = nf_interp

    ## PLOT 1: Full 360-Degree AoA Sweeps
    fig, axs = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    fig.suptitle(f'360° Aerodynamic Coefficients vs AoA (Re = {re_fixed:.0e}, M={mach_fixed})', fontsize=14)

    # Cl
    axs[0].plot(aoa_deg_360, cl_xf, 'b-', label='XFLR5 Data')
    axs[0].plot(aoa_deg_360, cl_nf, 'r--', label='NeuralFoil')
    axs[0].plot(aoa_deg_360, cl_param, 'k:', label='Parametric Baseline')
    axs[0].set_ylabel('$C_L$')
    axs[0].grid(True)
    axs[0].legend()

    # Cd
    axs[1].plot(aoa_deg_360, cd_xf, 'b-')
    axs[1].plot(aoa_deg_360, cd_nf, 'r--')
    axs[1].plot(aoa_deg_360, cd_param, 'k:')
    axs[1].set_ylabel('$C_D$')
    axs[1].grid(True)

    # Cm
    axs[2].plot(aoa_deg_360, cm_xf, 'b-')
    axs[2].plot(aoa_deg_360, cm_nf, 'r--')
    axs[2].plot(aoa_deg_360, cm_param, 'k:')
    axs[2].set_ylabel('$C_M$')
    axs[2].set_xlabel(r'Angle of Attack $\alpha$ (Degrees)')
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()

    ## Drag Polar (Cl vs Cd) & Aerodynamic Efficiency (L/D) vs Alpha
    print("Generating Polar and Efficiency Sweeps across Reynolds Numbers...")
    
    # Define an attached/mildly stalled alpha range for high fidelity comparison
    aoa_polar_deg = np.linspace(-20, 20, 150)
    aoa_polar_rad = jnp.radians(aoa_polar_deg)
    mach_fixed = 0.2
    
    # Suitable range of Reynolds numbers for propeller operating profiles
    re_list = [1e5, 5e5, 1e6, 2.5e6]
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3'] # Distinct high-contrast palette

    fig1, (ax_polar, ax_ld) = plt.subplots(1, 2, figsize=(12, 5.5))
    
    for re_val, col in zip(re_list, colors):
        # Broadcasters for vector safety
        re_vec = jnp.full_like(aoa_polar_rad, re_val)
        mach_vec = jnp.full_like(aoa_polar_rad, mach_fixed)
        
        # Evaluate both models to check cross-compatibility bounds
        cl_xf, cd_xf, _ = xfoil_interp(aoa_polar_rad, re_vec, mach_vec)
        cl_nf, cd_nf, _ = nf_interp(aoa_polar_rad, re_vec, mach_vec)
        
        # 1A. Plot Drag Polar (Cd on X, Cl on Y per aerodynamic standard)
        ax_polar.plot(cd_xf, cl_xf, color=col, linestyle='-', alpha=0.4, label=f'XFoil (Re={re_val:.1e})' if col == colors[0] else "")
        ax_polar.plot(cd_nf, cl_nf, color=col, linestyle='--', linewidth=1.8, label=f'Re = {re_val:.1e}')
        
        # 1B. Plot Aerodynamic Efficiency (L/D vs Alpha)
        # We protect against division by zero nearby cd=0 with standard array mechanics
        ld_xf = cl_xf / jnp.maximum(cd_xf, 1e-4)
        ld_nf = cl_nf / jnp.maximum(cd_nf, 1e-4)
        
        ax_ld.plot(aoa_polar_deg, ld_xf, color=col, linestyle='-', alpha=0.4)
        ax_ld.plot(aoa_polar_deg, ld_nf, color=col, linestyle='--', linewidth=1.8)

    # Polish Drag Polar layout
    ax_polar.set_title('Drag Polar Profile ($C_L$ vs $C_D$)')
    ax_polar.set_xlabel('$C_D$ (Drag Coefficient)')
    ax_polar.set_ylabel('$C_L$ (Lift Coefficient)')
    ax_polar.grid(True, which='both', linestyle=':', alpha=0.5)
    ax_polar.set_xlim(0, 0.06) # Constrain to attached flight regime drag bounds
    ax_polar.legend(loc='lower right')

    # Polish L/D layout
    ax_ld.set_title('Aerodynamic Efficiency ($C_L / C_D$)')
    ax_ld.set_xlabel(r'$\alpha$ (Angle of Attack, deg)')
    ax_ld.set_ylabel('$C_L / C_D$')
    ax_ld.grid(True, which='both', linestyle=':', alpha=0.5)
    
    # Proxy lines for the global line-style legend
    ax_ld.plot([], [], 'k-', alpha=0.4, label='XFoil Baseline')
    ax_ld.plot([], [], 'k--', label='NeuralFoil Blended')
    ax_ld.legend(loc='upper left')

    plt.tight_layout()
    plt.show()

    # PLOT 2: Transonic Drag Divergence (Cd vs Mach)
    print("Evaluating Transonic Drag Divergence profiles...")
    
    # Sweep Mach cleanly up to the near-sonic limits typical of propeller tips
    mach_sweep = jnp.linspace(0.0, 0.85, 200)
    re_mach_fixed = 1e6
    
    # Evaluate at a couple of constant operational angles of attack
    aoa_targets_deg = [0.0, 4.0, 8.0]
    
    fig2, ax_mach = plt.subplots(figsize=(7, 5))
    
    for aoa_target in aoa_targets_deg:
        aoa_vec = jnp.full_like(mach_sweep, jnp.radians(aoa_target))
        re_vec = jnp.full_like(mach_sweep, re_mach_fixed)
        
        # Evaluate using the active selected model
        _, cd_m, _ = active_interp(aoa_vec, re_vec, mach_sweep)
        
        ax_mach.plot(mach_sweep, cd_m, label=r'$\alpha = ' + f'{aoa_target}^\circ$')
        
    ax_mach.set_title(f'Transonic Drag Divergence (Re = {re_mach_fixed:.0e})')
    ax_mach.set_xlabel('Freestream Mach Number ($M$)')
    ax_mach.set_ylabel('$C_D$ (Compressible)')
    ax_mach.grid(True, which='both', linestyle=':', alpha=0.5)
    ax_mach.axvline(0.6, color='gray', linestyle=':', alpha=0.7, label='Target Blade Speed ($M=0.6$)')
    ax_mach.legend()
    
    plt.tight_layout()
    plt.show()

    # 3D Surface Plots
    print("Building bounded 3D multi-variable surface spaces (+/- 20 deg)...")
    
    aoa_deg_3d = np.linspace(-20, 20, 80)
    aoa_rad_3d = jnp.radians(aoa_deg_3d)
    re_vals_3d = np.logspace(4.5, 6.5, 40) # 31,000 to 3,160,000
    mach_constant = 0.4
    
    AOA_GRID, RE_GRID = np.meshgrid(aoa_rad_3d, re_vals_3d)
    MACH_GRID = jnp.full_like(AOA_GRID, mach_constant)
    
    # Double vmap transformation over the custom bounded space 
    vmap_interp = jax.vmap(jax.vmap(active_interp, in_axes=(0, 0, 0)), in_axes=(0, 0, 0))
    CL_surf, CD_surf, CM_surf = vmap_interp(AOA_GRID, RE_GRID, MACH_GRID)

    fig3 = plt.figure(figsize=(18, 5.5))
    coeff_method_name = "NeuralFoil" if active_interp == nf_interp else "XFoil"
    fig3.suptitle(f'Bounded Aerodynamic Surfaces ({coeff_method_name} | $M = {mach_constant}$)', fontsize=14, y=0.98)
    
    # Cl Surface
    ax1 = fig3.add_subplot(131, projection='3d')
    surf1 = ax1.plot_surface(np.degrees(AOA_GRID), np.log10(RE_GRID), CL_surf, cmap='viridis', edgecolor='none', alpha=0.9)
    ax1.set_title('$C_L$ Space Bounds')
    ax1.set_xlabel(r'$\alpha$ (deg)')
    ax1.set_ylabel('$\log_{10}(\mathrm{Re})$')
    fig3.colorbar(surf1, ax=ax1, shrink=0.5, aspect=7)
    ax1.view_init(elev=25, azim=-60)

    # Cd Surface
    ax2 = fig3.add_subplot(132, projection='3d')
    surf2 = ax2.plot_surface(np.degrees(AOA_GRID), np.log10(RE_GRID), CD_surf, cmap='plasma', edgecolor='none', alpha=0.9)
    ax2.set_title('$C_D$ Space Bounds')
    ax2.set_xlabel(r'$\alpha$ (deg)')
    ax2.set_ylabel('$\log_{10}(\mathrm{Re})$')
    fig3.colorbar(surf2, ax=ax2, shrink=0.5, aspect=7)
    ax2.view_init(elev=25, azim=-60)

    # Cm Surface
    ax3 = fig3.add_subplot(133, projection='3d')
    surf3 = ax3.plot_surface(np.degrees(AOA_GRID), np.log10(RE_GRID), CM_surf, cmap='cividis', edgecolor='none', alpha=0.9)
    ax3.set_title('$C_M$ Space Bounds')
    ax3.set_xlabel(r'$\alpha$ (deg)')
    ax3.set_ylabel('$\log_{10}(\mathrm{Re})$')
    fig3.colorbar(surf3, ax=ax3, shrink=0.5, aspect=7)
    ax3.view_init(elev=25, azim=-60)

    plt.tight_layout()
    plt.show()

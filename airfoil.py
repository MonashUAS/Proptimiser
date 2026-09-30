### Establish Coefficients for Given Airfoil
import glob
import re
from functools import lru_cache
import numpy as np

import jax
from jax import numpy as jnp
from jax import lax
import matplotlib.pyplot as plt
import interpax
from scipy.ndimage import gaussian_filter
import neuralfoil as nf
from pathlib import Path

from airflow import resultantVelocityComponents, calc_reynolds
from geometry import N_CST, cst_at_stations, max_thickness

from aerosandbox.geometry.airfoil.airfoil_families import (
    get_kulfan_parameters,
    get_kulfan_coordinates
)

### HELPTER UTILITIES

def fit_and_plot_kulfan(image_path: str, n_weights: int = 8, plot: bool = True):
    """
    Reads an airfoil image, downsamples the boundary using linear bin averaging,
    and fits Kulfan (CST) parameters using AeroSandbox. Plots all Kulfan 
    components on a single standard matplotlib axis.
    
    Parameters:
        image_path (str): Path to the image file.
        n_weights (int): Number of Kulfan weights per side (default: 8).
        plot (bool): If True, plots the airfoil and all Kulfan modes on one axis.
        
    Returns:
        dict: The Kulfan parameters required by NeuralFoil.
    """
    # Load and binarize the image
    arr = plt.imread(image_path)
    if arr.ndim >= 3:
        arr = np.mean(arr[..., :3], axis=2)
        
    binary = arr < np.mean(arr)
    rows, cols = np.where(binary)
    
    if len(cols) == 0:
        raise ValueError("No airfoil shape detected in the image.")
        
    # Extract raw boundaries (one min/max per pixel column)
    x_unique = np.unique(cols)
    y_upper_raw = np.array([rows[cols == x].min() for x in x_unique], dtype=float)
    y_lower_raw = np.array([rows[cols == x].max() for x in x_unique], dtype=float)
    
    # Normalize X domain
    x_min, x_max = x_unique.min(), x_unique.max()
    chord = float(x_max - x_min)
    x_norm = (x_unique - x_min) / chord
    
    # Set Y origin to LE center and normalize
    le_idx = np.argmin(x_norm)
    y_center = (y_upper_raw[le_idx] + y_lower_raw[le_idx]) / 2.0
    y_u_norm = -(y_upper_raw - y_center) / chord
    y_l_norm = -(y_lower_raw - y_center) / chord

    # Downsample to ~60 linearly spaced averaged elements
    num_elements = 80
    bins = np.linspace(0, 1, num_elements + 1)
    x_sampled = 0.5 * (bins[:-1] + bins[1:])  # Bin centers
    
    y_u_sampled = np.zeros(num_elements)
    y_l_sampled = np.zeros(num_elements)
    
    for i in range(num_elements):
        mask = (x_norm >= bins[i]) & (x_norm <= bins[i+1])
        if np.any(mask):
            y_u_sampled[i] = np.mean(y_u_norm[mask])
            y_l_sampled[i] = np.mean(y_l_norm[mask])
        else:
            # Fallback to interpolation if a bin is empty 
            y_u_sampled[i] = np.interp(x_sampled[i], x_norm, y_u_norm)
            y_l_sampled[i] = np.interp(x_sampled[i], x_norm, y_l_norm)
    
    # Construct ordered Counter-Clockwise (CCW) array: TE -> LE -> TE
    x_upper_ccw = x_sampled[::-1]
    y_upper_ccw = y_u_sampled[::-1]
    x_lower_ccw = x_sampled[1:] 
    y_lower_ccw = y_l_sampled[1:]
    
    coords = np.column_stack((
        np.concatenate([x_upper_ccw, np.array([0]), x_lower_ccw]),
        np.concatenate([y_upper_ccw, np.array([0]), y_lower_ccw])
    ))

    # Fit Kulfan parameters via AeroSandbox
    params = get_kulfan_parameters(
        coordinates=coords,
        n_weights_per_side=n_weights
    )

    # Plotting on a single axis
    if plot:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot the Averaged Image Points
        ax.scatter(coords[:, 0], coords[:, 1], color='black', s=15, 
                   label='Averaged Extracted Points', zorder=5)
        
        # Plot the Final Reconstructed Airfoil
        base_airfoil = get_kulfan_coordinates(
            lower_weights=params["lower_weights"],
            upper_weights=params["upper_weights"],
            leading_edge_weight=params["leading_edge_weight"],
            TE_thickness=params["TE_thickness"],
            n_points_per_side=200
        )
        ax.plot(base_airfoil[:, 0], base_airfoil[:, 1], color='black', 
                linewidth=2, label='Full Kulfan Fit', zorder=4)
    
        # Plot Upper Modes
        for i in range(n_weights):
            w_u = np.zeros(n_weights)
            w_u[i] = params["upper_weights"][i]
            mode_u = get_kulfan_coordinates(
                lower_weights=np.zeros(n_weights), upper_weights=w_u,
                leading_edge_weight=0.0, TE_thickness=0.0, n_points_per_side=200
            )
            label = 'Upper Modes' if i == 0 else None
            ax.plot(mode_u[:, 0], mode_u[:, 1], color='tab:orange', alpha=0.5, label=label, zorder=2)

        # Plot Lower Modes
        for i in range(n_weights):
            w_l = np.zeros(n_weights)
            w_l[i] = params["lower_weights"][i]
            mode_l = get_kulfan_coordinates(
                lower_weights=w_l, upper_weights=np.zeros(n_weights),
                leading_edge_weight=0.0, TE_thickness=0.0, n_points_per_side=200
            )
            label = 'Lower Modes' if i == 0 else None
            ax.plot(mode_l[:, 0], mode_l[:, 1], color='tab:blue', alpha=0.5, label=label, zorder=2)

        # Plot LEM
        mode_lem = get_kulfan_coordinates(
            lower_weights=np.zeros(n_weights), upper_weights=np.zeros(n_weights),
            leading_edge_weight=params["leading_edge_weight"], TE_thickness=0.0, n_points_per_side=200
        )
        ax.plot(mode_lem[:, 0], mode_lem[:, 1], color='tab:red', linewidth=2, 
                linestyle='--', label='LE Modification', zorder=3)

        # Plot TE Mode
        mode_te = get_kulfan_coordinates(
            lower_weights=np.zeros(n_weights), upper_weights=np.zeros(n_weights),
            leading_edge_weight=0.0, TE_thickness=params["TE_thickness"], n_points_per_side=200
        )
        ax.plot(mode_te[:, 0], mode_te[:, 1], color='tab:green', linewidth=2, 
                linestyle=':', label='TE Thickness', zorder=3)

        # Axis styling
        ax.set_title(f"Airfoil Kulfan Fit and Component Breakdown ({n_weights*2 + 2} parameters)")
        ax.set_xlabel("x/c")
        ax.set_ylabel("y/c")
        ax.axis('equal')
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # Format Legend (moving it outside the plot so it doesn't obscure lines)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
        
        plt.tight_layout()
        plt.show()

    return params
### Interpolator Classes 
class ParametricInterpolator:
    def __init__(self, Cl_alpha=5.7, Cd0=0.035, AoA_0=-10.0, alpha_stall=20.0, k_smooth=30.0):
        self.Cl_alpha = Cl_alpha
        self.Cd0 = Cd0
        self.AoA_0 = jnp.radians(AoA_0)
        self.alpha_stall = jnp.radians(alpha_stall)
        self.k_smooth = k_smooth

    def __call__(self, alpha, re, mach, kulfan=None, t_over_c=0.10, return_log_confidence=False):
        # Apply the zero-lift offset internally
        alpha_eff = alpha - self.AoA_0
        Cd90 = 1.98
        
        # Unstalled Regime
        Cl_lin = self.Cl_alpha * alpha_eff
        Ct_lin = self.Cd0 * jnp.cos(alpha_eff)
        Cn_lin = (Cl_lin + Ct_lin * jnp.sin(alpha_eff)) / jnp.cos(alpha_eff)
        Cd_lin = Cn_lin * jnp.sin(alpha_eff) + Ct_lin * jnp.cos(alpha_eff)
        
        # High Angle Regime
        sin_a = jnp.sin(alpha_eff)
        Cn_stall = Cd90 * sin_a / (0.56 + 0.44 * jnp.abs(sin_a))
        Ct_stall = 0.5 * self.Cd0 * jnp.cos(alpha_eff)
        
        Cl_stall = Cn_stall * jnp.cos(alpha_eff) - Ct_stall * jnp.sin(alpha_eff)
        Cd_stall = Cn_stall * jnp.sin(alpha_eff) + Ct_stall * jnp.cos(alpha_eff)
        
        # Sigmoid blending
        transition = 1 / (1 + jnp.exp(-self.k_smooth * (jnp.abs(alpha_eff) - self.alpha_stall)))
        
        Cl = (1 - transition) * Cl_lin + transition * Cl_stall
        Cd = (1 - transition) * Cd_lin + transition * Cd_stall
        
        Cn_final = (1 - transition) * Cn_lin + transition * Cn_stall
        Cm = -Cn_final * (0.25 - 0.175 * (1 - 2 * jnp.abs(alpha_eff) / jnp.pi))

        return _with_confidence((Cl, Cd, Cm), return_log_confidence)

class AlphaReInterpolator:
    def __init__(self, airfoilname):
        files = glob.glob(f"airfoils/{airfoilname}/xfoil/xf-{airfoilname}-il-*.csv")
        if not files:
            raise FileNotFoundError(f"No CSVs found for 2D interpolation in airfoils/{airfoilname}/xfoil/")

        alpha_deg = jnp.concatenate([
            jnp.linspace(-180, -30, 10),
            jnp.linspace(-29, -12, 8),
            jnp.linspace(-11, 15, 27),
            jnp.linspace(16, 35, 10),
            jnp.linspace(36, 180, 11)
        ])

        self.alpha_grid = jnp.unique(jnp.radians(alpha_deg))
        raw_data = []
        
        Cd90, Cd0_default = 1.98, 0.015 
        sin_a, cos_a = jnp.sin(self.alpha_grid), jnp.cos(self.alpha_grid)
        Cn_fp = Cd90 * sin_a / (0.56 + 0.44 * jnp.abs(sin_a))
        Ct_fp = 0.5 * Cd0_default * cos_a
        bg_cl = Cn_fp * cos_a - Ct_fp * sin_a
        bg_cd = Cn_fp * sin_a + Ct_fp * cos_a
        bg_cm = -Cn_fp * (0.25 - 0.175 * (1 - 2 * jnp.abs(self.alpha_grid) / jnp.pi))

        for f in files:
            match = re.search(r'-(\d+)\.csv', f)
            if match:
                Re = float(match.group(1))
                data = np.genfromtxt(f, skip_header=11, delimiter=',')
                if jnp.isnan(data).any() or len(data) == 0:
                    continue

                alpha_data = jnp.radians(data[:, 0])
                x_cl, x_cd, x_cm = jnp.array(data[:, 1]), jnp.array(data[:, 2]), jnp.array(data[:, 4])

                xf_cl_interp = jnp.interp(self.alpha_grid, alpha_data, x_cl, left=jnp.nan, right=jnp.nan)
                xf_cd_interp = jnp.interp(self.alpha_grid, alpha_data, x_cd, left=jnp.nan, right=jnp.nan)
                xf_cm_interp = jnp.interp(self.alpha_grid, alpha_data, x_cm, left=jnp.nan, right=jnp.nan)

                min_a, max_a = alpha_data.min(), alpha_data.max()
                mask_low = 1.0 / (1.0 + jnp.exp(-40.0 * (self.alpha_grid - (min_a + jnp.radians(2.0)))))
                mask_high = 1.0 / (1.0 + jnp.exp(40.0 * (self.alpha_grid - (max_a - jnp.radians(2.0)))))
                weight = mask_low * mask_high

                cl_blended = jnp.where(jnp.isnan(xf_cl_interp), bg_cl, weight * xf_cl_interp + (1 - weight) * bg_cl)
                cd_blended = jnp.where(jnp.isnan(xf_cd_interp), bg_cd, weight * xf_cd_interp + (1 - weight) * bg_cd)
                cm_blended = jnp.where(jnp.isnan(xf_cm_interp), bg_cm, weight * xf_cm_interp + (1 - weight) * bg_cm)
                
                raw_data.append((Re, cl_blended, cd_blended, cm_blended))

        raw_data.sort(key=lambda x: x[0])
        re_array = jnp.array([x[0] for x in raw_data])
        
        cl_array = jnp.stack([x[1] for x in raw_data])
        cd_array = jnp.stack([x[2] for x in raw_data])
        cm_array = jnp.stack([x[3] for x in raw_data])

        self.re_grid_log = jnp.linspace(jnp.log10(re_array.min()), jnp.log10(re_array.max()), 15)
        log_re_array = jnp.log10(re_array)
        interp_vmap = jax.vmap(jnp.interp, in_axes=(None, None, 1))

        # Apply nan_to_num to guarantee safe behavior
        self.cl_grid = jnp.nan_to_num(interp_vmap(self.re_grid_log, log_re_array, cl_array))
        self.cd_grid = jnp.nan_to_num(interp_vmap(self.re_grid_log, log_re_array, cd_array))
        self.cm_grid = jnp.nan_to_num(interp_vmap(self.re_grid_log, log_re_array, cm_array))

    def __call__(self, alpha, re, mach, kulfan=None, t_over_c=0.10, return_log_confidence=False):
        input_shape = jnp.shape(alpha)
        alpha_flat, re_flat, mach_flat = jnp.ravel(alpha), jnp.ravel(re), jnp.ravel(mach)

        alpha_flat = jnp.mod(alpha_flat + jnp.pi, 2 * jnp.pi) - jnp.pi
        log_re_clipped = jnp.clip(jnp.log10(re_flat), self.re_grid_log.min() + 1e-5, self.re_grid_log.max() - 1e-5)
        
        Cl = interpax.interp2d(alpha_flat, log_re_clipped, self.alpha_grid, self.re_grid_log, self.cl_grid, method='linear')
        Cd = interpax.interp2d(alpha_flat, log_re_clipped, self.alpha_grid, self.re_grid_log, self.cd_grid, method='linear')
        Cm = interpax.interp2d(alpha_flat, log_re_clipped, self.alpha_grid, self.re_grid_log, self.cm_grid, method='linear')
        
        cl_comp, cd_comp, cm_comp = apply_mach_corrections(Cl, Cd, Cm, mach_flat)
        coeffs = jnp.reshape(cl_comp, input_shape), jnp.reshape(cd_comp, input_shape), jnp.reshape(cm_comp, input_shape)
        return _with_confidence(coeffs, return_log_confidence)

class AlphaMachReInterpolator:
    def __init__(self, airfoilname):
        files = glob.glob(f"airfoils/{airfoilname}/Polars/*.csv")
        if not files:
            raise FileNotFoundError(f"No CSVs found for 3D interpolation in airfoils/{airfoilname}/Polars/")

        alpha_deg = jnp.concatenate([
            jnp.linspace(-180, -30, 10),
            jnp.linspace(-29, -12, 8),
            jnp.linspace(-11, 15, 27),
            jnp.linspace(16, 35, 10),
            jnp.linspace(36, 180, 11)
        ])

        self.alpha_grid = jnp.unique(jnp.radians(alpha_deg))
        raw_data = []

        Cd90, Cd0_default = 1.98, 0.015 
        sin_a, cos_a = jnp.sin(self.alpha_grid), jnp.cos(self.alpha_grid)
        Cn_fp = Cd90 * sin_a / (0.56 + 0.44 * jnp.abs(sin_a))
        Ct_fp = 0.5 * Cd0_default * cos_a
        bg_cl = Cn_fp * cos_a - Ct_fp * sin_a
        bg_cd = Cn_fp * sin_a + Ct_fp * cos_a
        bg_cm = -Cn_fp * (0.25 - 0.175 * (1 - 2 * jnp.abs(self.alpha_grid) / jnp.pi))

        for f in files:
            match = re.search(r'Re([\d\.]+)_M([\d\.]+)', f)
            if match:
                Re = float(match.group(1)) * 1e6 
                Mach = float(match.group(2))
                
                data = np.genfromtxt(f, skip_header=11, delimiter=',')
                if np.isnan(data).any() or len(data) == 0:
                    continue

                alpha_data = jnp.radians(jnp.array(data[:, 0]))
                x_cl, x_cd, x_cm = jnp.array(data[:, 1]), jnp.array(data[:, 2]), jnp.array(data[:, 4])

                xf_cl_interp = jnp.interp(self.alpha_grid, alpha_data, x_cl, left=jnp.nan, right=jnp.nan)
                xf_cd_interp = jnp.interp(self.alpha_grid, alpha_data, x_cd, left=jnp.nan, right=jnp.nan)
                xf_cm_interp = jnp.interp(self.alpha_grid, alpha_data, x_cm, left=jnp.nan, right=jnp.nan)

                min_a, max_a = alpha_data.min(), alpha_data.max()
                mask_low = 1.0 / (1.0 + jnp.exp(-40.0 * (self.alpha_grid - (min_a + jnp.radians(2.0)))))
                mask_high = 1.0 / (1.0 + jnp.exp(40.0 * (self.alpha_grid - (max_a - jnp.radians(2.0)))))
                weight = mask_low * mask_high

                cl_blended = jnp.where(jnp.isnan(xf_cl_interp), bg_cl, weight * xf_cl_interp + (1 - weight) * bg_cl)
                cd_blended = jnp.where(jnp.isnan(xf_cd_interp), bg_cd, weight * xf_cd_interp + (1 - weight) * bg_cd)
                cm_blended = jnp.where(jnp.isnan(xf_cm_interp), bg_cm, weight * xf_cm_interp + (1 - weight) * bg_cm)

                raw_data.append((Mach, Re, cl_blended, cd_blended, cm_blended))

        mach_points = jnp.array([d[0] for d in raw_data])
        re_points = jnp.array([d[1] for d in raw_data])
        log_re_points = jnp.log10(re_points)

        self.mach_grid = jnp.linspace(mach_points.min(), mach_points.max(), 10)
        self.re_grid_log = jnp.linspace(log_re_points.min(), log_re_points.max(), 15)
        M_mesh, Re_mesh = jnp.meshgrid(self.mach_grid, self.re_grid_log, indexing='ij')

        points = jnp.column_stack((mach_points, log_re_points))
        cl_vals_all = jnp.stack([d[2] for d in raw_data], axis=-1)
        cd_vals_all = jnp.stack([d[3] for d in raw_data], axis=-1)
        cm_vals_all = jnp.stack([d[4] for d in raw_data], axis=-1)

        cl_grid = jax_idw_3d_grid(points, cl_vals_all, M_mesh, Re_mesh)
        cd_grid = jax_idw_3d_grid(points, cd_vals_all, M_mesh, Re_mesh)
        cm_grid = jax_idw_3d_grid(points, cm_vals_all, M_mesh, Re_mesh)
        
        # Apply nan_to_num
        self.cl_grid = jnp.nan_to_num(gaussian_filter(cl_grid, sigma=(1.5, 0.8, 1.0)))
        self.cd_grid = jnp.nan_to_num(gaussian_filter(cd_grid, sigma=(1.5, 0.8, 1.0)))
        self.cm_grid = jnp.nan_to_num(gaussian_filter(cm_grid, sigma=(1.5, 0.8, 1.0)))

    def __call__(self, alpha, re, mach, kulfan=None, t_over_c=0.10, return_log_confidence=False):
        input_shape = jnp.shape(alpha)
        alpha_flat, mach_flat, re_flat = jnp.ravel(alpha), jnp.ravel(mach), jnp.ravel(re)

        alpha_flat = jnp.mod(alpha_flat + jnp.pi, 2 * jnp.pi) - jnp.pi
        log_re_clipped = jnp.clip(jnp.log10(re_flat), self.re_grid_log.min() + 1e-5, self.re_grid_log.max() - 1e-5)
        mach_clipped = jnp.clip(mach_flat, self.mach_grid.min() + 1e-5, self.mach_grid.max() - 1e-5)
        
        Cl = interpax.interp3d(alpha_flat, mach_clipped, log_re_clipped, 
                               self.alpha_grid, self.mach_grid, self.re_grid_log, 
                               self.cl_grid, method='cubic')
        Cd = interpax.interp3d(alpha_flat, mach_clipped, log_re_clipped, 
                               self.alpha_grid, self.mach_grid, self.re_grid_log, 
                               self.cd_grid, method='cubic')
        Cm = interpax.interp3d(alpha_flat, mach_clipped, log_re_clipped,
                               self.alpha_grid, self.mach_grid, self.re_grid_log,
                               self.cm_grid, method='cubic')
        coeffs = Cl.reshape(input_shape), Cd.reshape(input_shape), Cm.reshape(input_shape)
        return _with_confidence(coeffs, return_log_confidence)

class NeuralFoilInterpolator:
    '''
    A pure-JAX implementation of the NeuralFoil multi-layer perceptron (MLP) for aerodynamic coefficient prediction.
    By staying in JAX, this interpolator is 100% compatible with JIT compilation and autodiff.

    The section shape is passed on every call as CST vectors [lower x8, upper x8, LE weight, TE thickness]
    (normalised to unit chord), so each radial station can have its own airfoil and the shape can be optimised.

    Implementation mostly cloned from https://github.com/peterdsharpe/NeuralFoil/blob/master/neuralfoil/main.py
    '''
    def __init__(self, model_size="large"):
        # Load the pre-trained neural network weights and input scaling distributions from the neuralfoil package
        nn_weights_dir = Path(nf.__file__).parent / "nn_weights_and_biases"
        self._scaled_input_distribution = {k: jnp.array(v) for k, v in dict(np.load(nn_weights_dir / "scaled_input_distribution.npz")).items()}
        
        nn_params_raw = dict(np.load(nn_weights_dir / f"nn-{model_size}.npz"))
        self.nn_params = {k: jnp.array(v) for k, v in nn_params_raw.items()}
        
        self.layer_indices = sorted(list(set([int(key.split(".")[1]) for key in nn_params_raw.keys()])))
        self.N_inputs = len(self._scaled_input_distribution["mean_inputs_scaled"])

    def raw(self, alpha, re, kulfan):
        '''
        Network outputs before flat-plate blending and Mach correction.
        kulfan (..., N_CST) broadcasts against alpha along a trailing axis.

        Returns CL, CD, CM and the log of the analysis confidence (confidence 0-1), each shaped like alpha.
        The log is taken from the logit (log_sigmoid), so it keeps a useful gradient where the confidence itself
        saturates to zero far outside the training data.
        '''
        alpha, re = jnp.broadcast_arrays(alpha, re)
        alpha_deg = jnp.degrees(alpha)
        k = jnp.broadcast_to(kulfan, alpha_deg.shape + (N_CST,))

        # 1. Assemble the 25-dimensional input vector along the LAST axis (-1)
        # NeuralFoil takes the upper surface weights first
        x = jnp.concatenate([
            k[..., 8:16],               # upper weights
            k[..., 0:8],                # lower weights
            k[..., 16:17],              # leading edge weight
            k[..., 17:18] * 50,         # TE thickness
            jnp.stack([
                jnp.sin(jnp.radians(2 * alpha_deg)),
                jnp.cos(jnp.radians(alpha_deg)),
                1 - jnp.cos(jnp.radians(alpha_deg)) ** 2,
                (jnp.log(re) - 12.5) / 3.5,
                jnp.zeros_like(alpha_deg), # n_crit scaling
                jnp.ones_like(alpha_deg),  # forced transition upper
                jnp.ones_like(alpha_deg),  # forced transition lower
            ], axis=-1),
        ], axis=-1) # Shape: (*input_shape, 25)
        
        # 2. Leading-dimension safe MLP using jnp.matmul
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
        
        # 3. Enforce symmetric consistency using slice syntax (...)
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
        log_confidence = jax.nn.log_sigmoid(y_fused[..., 0])
        CL_raw = y_fused[..., 1] / 2
        CD_raw = jnp.exp((y_fused[..., 2] - 2) * 2)
        CM_raw = y_fused[..., 3] / 20
        return CL_raw, CD_raw, CM_raw, log_confidence

    def __call__(self, alpha, re, mach, kulfan, t_over_c=0.10, return_log_confidence=False):
        # 1. Wrap alpha to [-pi, pi] preserving original shape
        alpha = jnp.mod(alpha + jnp.pi, 2 * jnp.pi) - jnp.pi
        
        # Broadcast arrays together to unify their shapes automatically
        alpha, re = jnp.broadcast_arrays(alpha, re)

        CL_raw, CD_raw, CM_raw, log_confidence = self.raw(alpha, re, kulfan)
        
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
        coeffs = apply_mach_corrections(CL_inc, CD_inc, CM_inc, mach, t_over_c)
        return coeffs + (log_confidence,) if return_log_confidence else coeffs

def _with_confidence(coeffs, return_log_confidence):
    '''
    Table and parametric models have no confidence estimate; report full confidence (log 0) when one is requested.
    '''
    return coeffs + (jnp.zeros_like(coeffs[0]),) if return_log_confidence else coeffs

### Data Loading/Grid Utilities

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

@lru_cache(maxsize=None)
def load_airfoil_data(coeff_method, airfoilname="arad10"):
    '''
    Instantiates the aerodynamic interpolator for coeff_method (0: Parametric, 1: 2D Interp, 2: 3D Interp, 3: NeuralFoil).
    airfoilname selects the polar tables for methods 1 and 2; NeuralFoil takes section shapes per call instead.
    Cached, so propellers sharing a configuration share one interpolator (and one jit trace).
    '''
    if coeff_method == 0:
        return ParametricInterpolator() # Relies on default params, can be extended later
    elif coeff_method == 1:
        return AlphaReInterpolator(airfoilname)
    elif coeff_method == 2:
        return AlphaMachReInterpolator(airfoilname)
    elif coeff_method == 3:
        return NeuralFoilInterpolator()
    else:
        raise ValueError(f"Incorrect coefficient method {coeff_method} selected in Inputs.")

## AERODYNAMIC CORRECTIONS
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
    C_L_2D, C_D_2D, C_M, V_i, params = operands
    
    # Extract geometry and prevent division by zero at the exact root
    c = params.c[:, None]
    r = params.r[:, None]
    R = params.cfg.R_tip*1.05
    
    r_safe = jnp.maximum(r, 1e-6)
    c_over_r = c / r_safe
    r_over_R = r_safe / R
    
    # Calculate Effective Angle of Attack (AoA_eff)
    geometric_angle = params.beta[:, None]
    _, phi = resultantVelocityComponents(V_i, params)
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

# Aerodynamic Coefficient Pipeline
def aero_coeffs(V_i, params, return_log_confidence=False):
    '''
    Universal Aerodynamic pipeline.

    Each radial station uses its own section, interpolated along the span from the CST sections in params.cst.
    With return_log_confidence=True the log NeuralFoil analysis confidence (n_r, n_psi) is returned as a 4th output.
    '''
    geometric_angle = params.beta[:, None]
    V_R, phi = resultantVelocityComponents(V_i, params)
    AoA_eff = geometric_angle - phi

    Re = calc_reynolds(params.rho, jnp.maximum(V_R, 1e-6), params.c[:, None], params.mu)
    Mach = V_R / params.a

    # Section shape and thickness at each radial station
    kulfan = cst_at_stations(params.r, params.r_cst, params.cst)
    t_over_c = max_thickness(kulfan)[:, None]

    # Natively calls the instantiated class
    outputs = params.airfoil_interp(AoA_eff, Re, Mach, kulfan=kulfan[:, None, :], t_over_c=t_over_c,
                                    return_log_confidence=return_log_confidence)
    C_L_2D, C_D_2D, C_M = outputs[:3]

    # Apply 3D Rotational Corrections
    operands = (C_L_2D, C_D_2D, C_M, V_i, params)
    C_L_final, C_D_final, C_M_final = lax.switch(
        params.cfg.rot_corr_model,
        [no_3d_correction, du_selig_correction],
        operands
    )

    if return_log_confidence:
        return C_L_final, C_D_final, C_M_final, outputs[3]
    return C_L_final, C_D_final, C_M_final


if __name__ == "__main__":
    #mmm pretty plots
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "legend.fontsize": 10
    })
    
    print("Initializing interpolator architectures...")
    
    aoa_deg_360 = np.linspace(-180, 180, 400)
    aoa_rad_360 = jnp.radians(aoa_deg_360)
    re_fixed = 1e6
    mach_fixed = 0.2

    # Evaluate Parametric
    param_interp = load_airfoil_data(0, "arad10")
    cl_param, cd_param, cm_param = param_interp(aoa_rad_360, re_fixed, mach_fixed)

    # Evaluate XFoil 2D
    xfoil_interp = load_airfoil_data(1, "arad10")
    cl_xf, cd_xf, cm_xf = xfoil_interp(aoa_rad_360, re_fixed, mach_fixed)

    # Evaluate NeuralFoil (section shape passed as CST weights)
    from functools import partial
    from geometry import kulfan_from_name
    nf_interp = partial(load_airfoil_data(3), kulfan=jnp.asarray(kulfan_from_name("arad10")))
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

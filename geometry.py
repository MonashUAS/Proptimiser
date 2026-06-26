## Propeller Geometry Tools and Functions

import numpy as np
import jax.numpy as jnp
import aerosandbox as asb
from aerosandbox.geometry.airfoil.airfoil_families import get_kulfan_coordinates

### Integrand Functions
def radial_stops(R, r_hub, n_r):
    '''
    Return radial node centers and dr for integration.
    '''
    r = jnp.linspace(r_hub + (R - r_hub) / (2 * n_r), R - (R - r_hub) / (2 * n_r), n_r)
    # Simple Homogeneous radial spacing
    dr = jnp.ones_like(r)*(R - r_hub) / n_r
    return r, dr

def angular_stops(n_psi):
    '''
    Return angular node centers and dpsi for integration
    '''
    psi = jnp.linspace(0, (2 * jnp.pi), n_psi)
    dpsi = jnp.ones_like(psi)*(2 * jnp.pi) / n_psi
    return psi, dpsi

### Geometry Transformation functions
def params_to_le_te(r, c, beta):
    """
    Calculates 3D coordinates for LE and TE lines.
    Aligns the 1/4 chord line precisely along the radial X-axis (pitch axis).
    """
    x = r.ravel()
    c = c.ravel()
    beta = beta.ravel()
    
    # LE is at -25% chord from the pitch axis; TE is at +75% chord from the pitch axis
    y_le = -0.25 * c * np.sin(beta)
    z_le = -0.25 * c * np.cos(beta)
    
    y_te = 0.75 * c * np.sin(beta)
    z_te = 0.75 * c * np.cos(beta)

    # Stack into clean (N, 3) positional arrays
    le_points = np.column_stack([x, y_le, z_le])
    te_points = np.column_stack([x, y_te, z_te])

    return le_points, te_points

def get_cst_weights_from_names(airfoil_names):
    """
    Converts a list of airfoil strings into a (N_foils, N_CST_weights) array.
    Outputs a stacked numpy array containing [lower_weights, upper_weights, LE_weight, TE_thickness]
    for each airfoil, which can be fed directly to JAX.
    """
    weights_list = []
    
    for name in airfoil_names:
        # KulfanAirfoil calculates the CST parameterization on instantiation
        kf = asb.KulfanAirfoil(name)
        p = kf.kulfan_parameters
        
        # Flatten the dictionary into a single 1D array
        flat_weights = np.concatenate([
            p['lower_weights'],
            p['upper_weights'],
            [p['leading_edge_weight']],
            [p['TE_thickness']]
        ])
        weights_list.append(flat_weights)
        
    return np.array(weights_list)

def generate_blade_coordinates(r, r_foils, cst_weights, ellipse_thickness=1.0, n_points_per_side=100):
    """
    Transforms the optimized CST weights into a full (n_r, N_points, 2) coordinate array.
    Linearly interpolates CST weights along the aerodynamic span, and coordinate-blends 
    the root region to an ellipse.
    """
    n_foils, n_cst = cst_weights.shape
    
    # Kulfan parameters usually pack 2 singular weights at the end (LE and TE)
    n_weights_per_side = (n_cst - 2) // 2
    
    # 1. Generate the Root Airfoil to determine exact coordinate array dimensions
    root_cst = cst_weights[0]
    root_coords = get_kulfan_coordinates(
        lower_weights=root_cst[:n_weights_per_side],
        upper_weights=root_cst[n_weights_per_side:2*n_weights_per_side],
        leading_edge_weight=root_cst[-2],
        TE_thickness=root_cst[-1],
        n_points_per_side=n_points_per_side
    )
    
    n_actual_points = root_coords.shape[0]
    root_foil = asb.Airfoil(coordinates=root_coords)
    
    # 2. Generate the mounting ellipse matching the exact point count
    theta = np.linspace(0, 2*np.pi, n_actual_points)
    ellipse_x = 0.5 * (1 - np.cos(theta))
    ellipse_y = (ellipse_thickness / 2) * np.sin(theta)
    ellipse_foil = asb.Airfoil(coordinates=np.column_stack((ellipse_x, ellipse_y)))
    
    # 3. Initialize the output tensor
    blade_coords = np.zeros((len(r), n_actual_points, 2))
    
    # 4. Interpolate and Generate
    for i, r_i in enumerate(r):
        if r_i < r_foils[0]:
            # Structural Root: Coordinate blend between Ellipse and Root Airfoil
            fraction = np.clip(r_i / r_foils[0], 0, 1) if r_foils[0] > 0 else 1.0
            blended = ellipse_foil.blend_with(root_foil, fraction)
            blade_coords[i] = blended.coordinates
            
        else:
            # Aerodynamic Span: Interpolate the CST weights directly
            interp_weights = np.zeros(n_cst)
            for j in range(n_cst):
                interp_weights[j] = np.interp(r_i, r_foils, cst_weights[:, j])
                
            coords = get_kulfan_coordinates(
                lower_weights=interp_weights[:n_weights_per_side],
                upper_weights=interp_weights[n_weights_per_side:2*n_weights_per_side],
                leading_edge_weight=interp_weights[-2],
                TE_thickness=interp_weights[-1],
                n_points_per_side=n_points_per_side
            )
            blade_coords[i] = coords
            
    return blade_coords

def offset_airfoil_profile(foil_x, foil_y, chord, skin_thickness):
    """
    Offsets a normalized 2D airfoil inward by a physical skin thickness.
    Prevents trailing edge self-intersection by clipping crossed lines.
    Assumes foil coordinates are ordered: TE_upper -> LE -> TE_lower.
    """
    import numpy as np

    # 1. Normalize the physical offset to the local chord
    # Ensure chord is a scalar value (extract from array if necessary)
    chord_val = chord[0] if isinstance(chord, np.ndarray) else chord
    offset_mag = skin_thickness / chord_val
    
    # 2. Calculate gradients to find normals
    dx = np.gradient(foil_x)
    dy = np.gradient(foil_y)
    length = np.hypot(dx, dy)
    
    # Avoid division by zero at sharp edges
    length[length == 0] = 1e-8
    
    # Calculate inward-pointing normals based on the loop direction
    nx = -dy / length
    ny = dx / length
    
    # 3. Apply the inward offset
    off_x = foil_x - offset_mag * nx
    off_y = foil_y - offset_mag * ny
    
    # 4. Filter self-intersections (The "Swallowtail" clip)
    le_idx = np.argmin(off_x)
    
    # Split into upper and lower surfaces
    upper_x, upper_y = off_x[:le_idx+1], off_y[:le_idx+1]
    lower_x, lower_y = off_x[le_idx:], off_y[le_idx:]
    
    # Reverse upper surface so X is strictly increasing (LE -> TE) for interpolation
    rev_upper_x = upper_x[::-1]
    rev_upper_y = upper_y[::-1]
    
    try:
        # Interpolate lower surface Y values onto the upper surface X coordinates
        lower_y_interp = np.interp(rev_upper_x, lower_x, lower_y)
        
        # Valid points are where the offset upper surface remains above the lower surface
        valid_idx = rev_upper_y > lower_y_interp
        
        # Revert the valid upper points back to TE -> LE order
        clean_upper_x = rev_upper_x[valid_idx][::-1]
        clean_upper_y = rev_upper_y[valid_idx][::-1]
        
        # Truncate the lower surface at the new blunt trailing edge
        max_x = np.max(clean_upper_x)
        clean_lower_x = lower_x[lower_x <= max_x]
        clean_lower_y = lower_y[lower_x <= max_x]
        
        # Reconstruct the continuous loop
        clean_core_x = np.concatenate([clean_upper_x, clean_lower_x])
        clean_core_y = np.concatenate([clean_upper_y, clean_lower_y])
        
    except Exception as e:
        print(f"Warning: Intersection clipping failed, using raw offset. {e}")
        clean_core_x, clean_core_y = off_x, off_y

    return clean_core_x, clean_core_y

def get_cst_weights_from_names(airfoil_names):
    """
    Converts a list of airfoil strings into a (N_foils, N_CST_weights) array.
    Outputs a stacked numpy array containing [lower_weights, upper_weights, LE_weight, TE_thickness]
    for each airfoil, which can be fed directly to JAX.
    """
    weights_list = []
    
    for name in airfoil_names:
        # KulfanAirfoil calculates the CST parameterization on instantiation
        kf = asb.KulfanAirfoil(name)
        p = kf.kulfan_parameters
        
        # Flatten the dictionary into a single 1D array
        flat_weights = np.concatenate([
            p['lower_weights'],
            p['upper_weights'],
            [p['leading_edge_weight']],
            [p['TE_thickness']]
        ])
        weights_list.append(flat_weights)
        
    return np.array(weights_list)

# Example usage:
# r_foils = np.array([R_hub, R_mid, R_tip])
# cst_weights = get_cst_weights_from_names(["naca4412", "naca2412", "naca0012"])

def export_to_creo_ibl(r, c, beta, blade_coords, exportname, 
                       pitch_axis=0.25, scale_to_mm=True, skin_thickness=0.0):
    """
    Generates a Creo .ibl file from a pre-computed (n_r, N_points, 2) blade coordinate array,
    using AeroSandbox's native 2D geometry manipulation for elegant 3D positioning.
    
    Parameters:
    * pitch_axis: Non-dimensional chord fraction to rotate around (0.25 = quarter chord).
    """
    n_r, n_foil, _ = blade_coords.shape
    scale = 1000.0 if scale_to_mm else 1.0

    with open(f"{exportname}.ibl", "w") as f:
        f.write("Open\nArclength\n")
        
        for i, r_i in enumerate(r):
            f.write(f"Begin section ! Station {i+1}: r = {r_i:.4f} m\n")
            f.write("Begin curve\n")
            
            # 1. Instantiate the ASB Airfoil from the JAX-optimized coordinates
            foil = asb.Airfoil(coordinates=blade_coords[i])
            
            # 2. ELEGANT TRANSFORMS: Rotate and Scale natively in 2D
            # Rotate around the pitch axis (beta is in radians). 
            # Note: adjust the sign on beta[i] depending on your CAD coordinate handedness.
            foil = foil.rotate(angle=-beta[i], x_center=pitch_axis, y_center=0.0)
            
            # Scale the airfoil by the local chord
            foil = foil.scale(c[i], c[i])
            
            # Extract the transformed coordinates
            base_coords = foil.coordinates
            
            # 3. Handle Core/Skin Normal Offsets
            # (Note: ASB does not have a native inward boundary offset for core generation, 
            # so you still rely on standard normal-offset logic here if skin_thickness > 0)
            if skin_thickness > 0.0:
                core_x, core_y = offset_airfoil_profile(base_coords[:, 0], base_coords[:, 1], 
                                                        c[i], skin_thickness)
            else:
                core_x, core_y = base_coords[:, 0], base_coords[:, 1]
                
            # 4. Direct 3D Mapping and Translation
            for j in range(len(core_x)):
                # Map 2D to 3D standard aerospace axes: X=Chord, Y=Span, Z=Thickness
                # We subtract (pitch_axis * c[i]) to ensure the pitch axis aligns perfectly 
                # on the CAD Y-axis (X=0, Z=0) across all radial stations.
                pt_x = (core_x[j] - (pitch_axis * c[i])) * scale
                pt_y = r_i * scale  
                pt_z = core_y[j] * scale
                
                f.write(f"{j+1}  {pt_x:.6f}  {pt_y:.6f}  {pt_z:.6f}\n")
                
    part_type = "Core Insert" if skin_thickness > 0 else "Aerodynamic Skin"
    print(f"Successfully generated Creo geometry: {exportname}.ibl ({n_r} sections) - {part_type}")

def build_propeller_mesh(geompoints, foilpoints, num_blades=2):
    """
    Lofts a 3D propeller blade by interpolating between the Leading Edge (LE) 
    and Trailing Edge (TE) lines, scaling the thickness profile dynamically.
    """
    import pyvista as pv
    # Unpack clean tuple signatures
    le_points, te_points = geompoints
    upper_points, lower_points = foilpoints
    
    # Create continuous loop coordinates around the profile perimeter
    foil_data = np.vstack([upper_points[::-1], lower_points])
    foil_x = foil_data[:, 0]  # Normalized chord station [0 to 1]
    foil_y = foil_data[:, 1]  # Normalized thickness station
    
    # Calculate the raw structural chord vectors and absolute lengths
    chord_vectors = te_points - le_points  # Shape: (n_r, 3)
    chords = np.linalg.norm(chord_vectors, axis=1, keepdims=True)  # Shape: (n_r, 1)
    
    # cross product captures thickness
    u_span = np.array([1.0, 0.0, 0.0])
    u_chord = chord_vectors / chords
    u_thick = np.cross(u_span, u_chord)  # Shape: (n_r, 3)
    
    # Transform arrays to (n_r, 1, 3) to broadcast against the profile points (1, n_foil, 1)
    LE = le_points[:, None, :]
    CHORD_VEC = chord_vectors[:, None, :]
    UTHICK = u_thick[:, None, :]
    CHORDS = chords[:, None, :]
    
    FX = foil_x[None, :, None]
    FY = foil_y[None, :, None]
    
    # Position = Start at LE + move down the chord line + project out normal thickness
    mesh_points = LE + (FX * CHORD_VEC) + (FY * CHORDS * UTHICK)
    
    # Extract structural coordinate grids for PyVista
    X_3d = mesh_points[:, :, 0]
    Y_3d = mesh_points[:, :, 1]
    Z_3d = mesh_points[:, :, 2]
    
    base_blade = pv.StructuredGrid(X_3d, Y_3d, Z_3d)
    
    # Duplicate and rotate across the assembly hub
    meshes = [base_blade]
    for i in range(1, int(num_blades)):
        angle = i * (360.0 / num_blades)
        rotated_blade = base_blade.copy()
        rotated_blade.rotate_z(angle, inplace=True)
        meshes.append(rotated_blade)
        
    return pv.MultiBlock(meshes).combine()

if __name__ == "__main__":
    '''
    Create and Plot Demo Flatblade Propeller Blade Geometry
    '''
    import matplotlib.pyplot as plt
    import pyvista as pv
    import results
    # Generate Flat Blade Parameters
    inputs, params = results.loadprop('redback_v1')
    airfoil = "ARAD10"

    r, c, beta = params.r, params.c, params.beta

    upper_points, lower_points = upperlowerairfoil(airfoil)

    #Trailing Edge -> Leading Edge -> Trailing Edge
    foil_data = np.vstack([upper_points[::-1], lower_points])

    le_points, te_points = params_to_le_te(r, c, beta)

    # Extract x and y coordinates
    foil_x = foil_data[:, 0]
    foil_y = foil_data[:, 1]

    # Plot Airfoil
    plt.figure(figsize=(8, 4))
    plt.plot(foil_x, foil_y, 'b-', label=f"{airfoil} Profile")
    plt.grid(True)
    plt.axis("equal")
    plt.xlabel("Chord Station (x/c)")
    plt.ylabel("Thickness Station (y/c)")
    plt.title(f"Airfoil Geometry Loop Check: {airfoil}")
    plt.legend()
    plt.show()

    ## 3D Render of Propeller
    geompoints = (le_points, te_points)
    foilpoints = (upper_points, lower_points)

    prop_mesh = build_propeller_mesh(geompoints, foilpoints, num_blades=2)

    # 1. Export Full Aerodynamic Propeller (For the negative molds)
    export_to_creo_ibl(r, c, beta, (upper_points, lower_points), "propeller_blade_mold", skin_thickness=0.0)

    # 2. Export Internal Core Insert (1.5mm offset for laminate skins)
    laminate_thickness_m = 0.0015 # 1.5mm in meters
    export_to_creo_ibl(r, c, beta, (upper_points, lower_points), "propeller_blade_insert", skin_thickness=laminate_thickness_m)

    # Fire up the local PyVista testing window
    plotter = pv.Plotter()
    plotter.set_background("#1e1e2f") #dark mode for vibes
    plotter.add_mesh(prop_mesh, color="silver", show_edges=True, edge_color="#444444", smooth_shading=True)
    plotter.add_axes()
    plotter.show()
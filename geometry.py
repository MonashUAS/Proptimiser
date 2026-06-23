## Propeller Geometry Tools and Functions

import numpy as np
from datetime import datetime
import scipy.optimize as opt

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

def upperlowerairfoil(airfoil):
    '''
    Takes airfoil geometry .dat file (contained in airfoils) and returns arrays of the upper 
    and lower points.
    '''
    airfoil_path = "airfoils/"+airfoil+"/"+airfoil+".dat"
    # Load data using numpy, skipping the header rows
    foil_data = np.loadtxt(airfoil_path, skiprows=1)
    
    # Find index where x-coordinates drop back to 0.0 
    # (indicating the transition from upper surface TE to lower surface LE)
    split_idx = np.where(np.diff(foil_data[:, 0]) < 0)[0]
    
    if len(split_idx) > 0:
        split_point = split_idx[0] + 1
        upper_points = foil_data[:split_point]
        lower_points = foil_data[split_point:]
    else:
        # Fallback to value-filtering only if index split fails
        upper_points = foil_data[foil_data[:, 1] >= 0]
        lower_points = foil_data[foil_data[:, 1] <= 0]

    return upper_points, lower_points


def ptsfilegenerator(r, c, beta, foilpoints, exportname, polyorder=6):
    '''
    Creates geometry files which defines the propellers shape
    '''
    # Unpack the tuple of upper and lower points
    upper_points, lower_points = foilpoints
    
    # Reverse upper points (Trailing Edge -> Leading Edge) before stacking 
    # with lower points (Leading Edge -> Trailing Edge) for a continuous perimeter loop
    foil_data = np.vstack([upper_points[::-1], lower_points])
    if polyorder>0:
        # Fit high order polynomial to data, then establish leading and trailing edge using fit.
        c_coeffs = np.polyfit(r, c, polyorder)
        c_p = np.poly1d(c_coeffs)
        c_poly = c_p(r)

        beta_coeffs = np.polyfit(r, beta, polyorder)
        beta_p = np.poly1d(beta_coeffs)
        beta_poly = beta_p(r)

        le_points, te_points = params_to_le_te(r, c_poly, beta_poly)
    else:
        le_points, te_points = params_to_le_te(r, c, beta)

    # Extract x and y coordinates from the unpacked NumPy arrays (Column 0 and 1)
    foil_x = foil_data[:, 0]
    foil_y = foil_data[:, 1]
    m = foil_x.shape[0]

    propgeom_airfoil_coords = np.stack([
        -foil_x, 
        foil_y,
        np.zeros(m)
    ])

    # Convert from m to mm for LE and TE points
    propgeom_LE_coords      = le_points * 1000
    propgeom_TE_coords      = te_points * 1000
    propgeom_airfoil_coords = propgeom_airfoil_coords.T

    # Write files with the timestamp appended
    np.savetxt(f"{exportname}_LE.pts",  propgeom_LE_coords,      delimiter=" ", fmt="%.7f")
    np.savetxt(f"{exportname}_TE.pts",  propgeom_TE_coords,      delimiter=" ", fmt="%.7f")
    np.savetxt(f"{exportname}_airfoil.pts",   propgeom_airfoil_coords, delimiter=" ", fmt="%.7f")

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
def export_to_creo_ibl(r, c, beta, foilpoints, exportname, scale_to_mm=True, skin_thickness=0.0):
    """
    Combines all radial stations and airfoil profiles into a single unified .ibl 
    file for direct multi-curve import into Creo Parametric.
    
    If skin_thickness > 0, the internal profile is offset inward to create core geometry.
    """
    import numpy as np

    # Unpack normalized airfoil coordinates
    upper_points, lower_points = foilpoints
    foil_data = np.vstack([upper_points[::-1], lower_points])
    base_foil_x = foil_data[:, 0]
    base_foil_y = foil_data[:, 1]
    
    # Calculate absolute 3D LE and TE locations
    le_points, te_points = params_to_le_te(r, c, beta)
    
    chord_vectors = te_points - le_points  # Shape: (n_r, 3)
    chords = np.linalg.norm(chord_vectors, axis=1, keepdims=True)  # Shape: (n_r, 1)
    
    u_span = np.array([1.0, 0.0, 0.0])
    u_chord = chord_vectors / chords
    u_thick = np.cross(u_span, u_chord)  # Normal thickness vector (n_r, 3)
    
    n_r = len(r)
    scale = 1000.0 if scale_to_mm else 1.0  # Convert meters to mm if true

    # Write out the structural text file
    with open(f"{exportname}.ibl", "w") as f:
        # Creo File Headers
        f.write("Open\n")
        f.write("Arclength\n")
        
        # Loop through every radial station
        for i in range(n_r):
            # Creo section comment markers
            f.write(f"Begin section ! Station {i+1}: r = {r[i]:.4f} m\n")
            f.write("Begin curve\n")
            
            # Determine if we are building the skin or the core
            if skin_thickness > 0.0:
                core_x, core_y = offset_airfoil_profile(base_foil_x, base_foil_y, chords[i][0], skin_thickness)
            else:
                core_x, core_y = base_foil_x, base_foil_y
                
            n_foil = len(core_x)
            
            # Generate and write out the 3D coordinates for this profile loop
            for j in range(n_foil):
                # 3D transformation logic
                pt = le_points[i] + (core_x[j] * chord_vectors[i]) + (core_y[j] * chords[i] * u_thick[i])
                pt_scaled = pt * scale
                
                # Creo requires: [Index (1-based)] [X] [Y] [Z]
                f.write(f"{j+1}  {pt_scaled[0]:.6f}  {pt_scaled[1]:.6f}  {pt_scaled[2]:.6f}\n")
                
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
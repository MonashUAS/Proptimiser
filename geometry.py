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

def export_to_creo_ibl(r, c, beta, foilpoints, exportname, scale_to_mm=True):
    """
    Combines all radial stations and airfoil profiles into a single unified .ibl 
    file for direct multi-curve import into Creo Parametric.

    .ibl file documentation: https://support.ptc.com/help/creo/creo_pma/r12/usascii/index.html#page/part_modeling/part_modeling/Example_Blend_File_Format.html#wwID0EHAEU
    """
    import numpy as np

    # Unpack normalized airfoil coordinates
    upper_points, lower_points = foilpoints
    foil_data = np.vstack([upper_points[::-1], lower_points])
    foil_x = foil_data[:, 0]
    foil_y = foil_data[:, 1]
    
    # Calculate absolute 3D LE and TE locations
    le_points, te_points = params_to_le_te(r, c, beta)
    
    chord_vectors = te_points - le_points  # Shape: (n_r, 3)
    chords = np.linalg.norm(chord_vectors, axis=1, keepdims=True)  # Shape: (n_r, 1)
    
    u_span = np.array([1.0, 0.0, 0.0])
    u_chord = chord_vectors / chords
    u_thick = np.cross(u_span, u_chord)  # Normal thickness vector (n_r, 3)
    
    n_r = len(r)
    n_foil = len(foil_x)
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
            
            # Generate and write out the 3D coordinates for this profile loop
            for j in range(n_foil):
                # 3D transformation logic matching your mesh builder
                pt = le_points[i] + (foil_x[j] * chord_vectors[i]) + (foil_y[j] * chords[i] * u_thick[i])
                pt_scaled = pt * scale
                
                # Creo requires: [Index (1-based)] [X] [Y] [Z]
                f.write(f"{j+1}  {pt_scaled[0]:.6f}  {pt_scaled[1]:.6f}  {pt_scaled[2]:.6f}\n")
                
    print(f"Successfully generated Creo geometry: {exportname}.ibl ({n_r} sections)")

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

    export_to_creo_ibl(r, c, beta, (upper_points, lower_points), "propeller_blade_sections")

    # Fire up the local PyVista testing window
    plotter = pv.Plotter()
    plotter.set_background("#1e1e2f") #dark mode for vibes
    plotter.add_mesh(prop_mesh, color="silver", show_edges=True, edge_color="#444444", smooth_shading=True)
    plotter.add_axes()
    plotter.show()
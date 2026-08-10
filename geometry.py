## Propeller Geometry Tools and Functions

import numpy as np
import jax.numpy as jnp
import aerosandbox as asb

### Integrand Functions
def radial_stops(R, r_hub, n_r):
    """
    Return radial node centers and dr for integration, including the endpoints.
    """
    # Includes both r_hub and R as nodes
    r = jnp.linspace(r_hub, R, n_r)
    
    # Grid spacing (distance between inclusive nodes)
    dr = jnp.ones_like(r) * (R - r_hub) / (n_r - 1)
    return r, dr

def angular_stops(n_psi):
    """
    Return angular node boundaries and dpsi for integration with the endpoint excluded.
    """
    # Divides 2*pi into n_psi segments, returning the start of each boundary (drops 2*pi)
    psi = jnp.linspace(0, 2 * jnp.pi, n_psi, endpoint=False)
    dpsi = jnp.ones_like(psi) * (2 * jnp.pi) / n_psi
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

def generate_blade_coordinates(r, r_foils, cst_weights, hub_thickness_ratio=0.1, n_points_per_side=100):
    """
    Generates blade coordinates across 3 distinct zones using a unified r_foils array.
    r_foils format: [r_start, r_hub, r_aero, ... aerodynamic stations ...]
    
    1. r <= r_hub: 100% Flat rounded rectangle (Stadium).
    2. r_hub < r < r_aero: Linear coordinate blend from Stadium to the first CST profile.
    3. r >= r_aero: 100% Aerodynamic CST interpolation.
    """
    import numpy as np
    import aerosandbox as asb
    from aerosandbox.geometry.airfoil.airfoil_families import get_kulfan_coordinates

    # Extract structural boundaries from unified array
    r_hub = r_foils[1]
    r_aero = r_foils[2]
    
    # Extract aerodynamic stations (everything from index 2 onwards)
    r_aero_stations = r_foils[2:]
    
    # Validate the inputs match
    if len(cst_weights) != len(r_aero_stations):
        raise ValueError(f"Expected {len(r_aero_stations)} sets of CST weights to match the aero stations in r_foils, but got {len(cst_weights)}.")

    n_foils, n_cst = cst_weights.shape
    n_weights_per_side = (n_cst - 2) // 2
    
    # 1. Generate the Root Airfoil (used at r_aero)
    root_cst = cst_weights[0]
    root_coords = get_kulfan_coordinates(
        lower_weights=root_cst[:n_weights_per_side],
        upper_weights=root_cst[n_weights_per_side:2*n_weights_per_side],
        leading_edge_weight=root_cst[-2],
        TE_thickness=root_cst[-1],
        n_points_per_side=n_points_per_side
    )
    
    n_actual_points = root_coords.shape[0]
    root_foil = asb.Airfoil(name="Root Aero Airfoil", coordinates=root_coords)
    
    # 2. Generate the Normalized Rounded Rectangle (Stadium) Hub
    hub_coords = np.zeros_like(root_coords)
    hub_coords[:, 0] = root_coords[:, 0] # Lock X-coordinates to root foil to prevent twist
    
    le_idx = np.argmin(root_coords[:, 0]) 
    rad = hub_thickness_ratio / 2.0
    
    for j, x in enumerate(hub_coords[:, 0]):
        sign = 1.0 if j <= le_idx else -1.0
        
        if x < rad:
            y_val = np.sqrt(max(0, rad**2 - (rad - x)**2))
        elif x > 1.0 - rad:
            y_val = np.sqrt(max(0, rad**2 - (x - (1.0 - rad))**2))
        else:
            y_val = rad
            
        hub_coords[j, 1] = sign * y_val
        
    hub_foil = asb.Airfoil(name="Hub Section", coordinates=hub_coords)
    
    # 3. Initialize output tensor
    blade_coords = np.zeros((len(r), n_actual_points, 2))
    
    # 4. Process all radial stations
    for i, r_i in enumerate(r):
        if r_i <= r_hub:
            # Zone 1: Pure Hub
            blade_coords[i] = hub_foil.coordinates
            
        elif r_i < r_aero:
            # Zone 2: Blend Region
            fraction = (r_i - r_hub) / (r_aero - r_hub)
            
            # --- FIX APPLIED HERE ---
            blended = hub_foil.blend_with_another_airfoil(
                airfoil=root_foil, 
                blend_fraction=fraction,
                n_points_per_side=n_points_per_side
            )
            blade_coords[i] = blended.coordinates
            
        else:
            # Zone 3: Aerodynamic Span
            interp_weights = np.zeros(n_cst)
            for j in range(n_cst):
                # Interpolate CST weights based on the slice of aerodynamic stations
                interp_weights[j] = np.interp(r_i, r_aero_stations, cst_weights[:, j])
                
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

def overwrite_hub_planform(r, c, beta, r_foils, hub_chord=None, hub_twist=0.0, aero_pitch_axis=0.25):
    """
    Overwrites the chord and twist arrays in the structural root region to 
    ensure a flat, untwisted hub that smoothly blends into the aerodynamic planform.
    Also returns a dynamic pitch_axis array that shifts from 0.5 (hub) to the aero axis.
    """
    import numpy as np
    
    r_hub = r_foils[1]
    r_aero = r_foils[2]
    
    c_new = np.copy(c)
    beta_new = np.copy(beta)
    pitch_axis = np.full_like(r, aero_pitch_axis, dtype=float)
    
    if hub_chord is None:
        hub_chord = c[0]
    
    c_aero = np.interp(r_aero, r, c)
    beta_aero = np.interp(r_aero, r, beta)
    
    for i, r_i in enumerate(r):
        if r_i <= r_hub:
            # Zone 1: Pure Hub
            c_new[i] = hub_chord
            beta_new[i] = hub_twist
            pitch_axis[i] = 0.5  # Centered structurally
            
        elif r_i < r_aero:
            # Zone 2: Blend Region (Cosine interpolation)
            linear_fraction = (r_i - r_hub) / (r_aero - r_hub)
            blend_fraction = 0.5 * (1 - np.cos(np.pi * linear_fraction))
            
            c_new[i] = hub_chord + blend_fraction * (c_aero - hub_chord)
            beta_new[i] = hub_twist + blend_fraction * (beta_aero - hub_twist)
            pitch_axis[i] = 0.5 + blend_fraction * (aero_pitch_axis - 0.5)
            
    return c_new, beta_new, pitch_axis

def export_to_creo_ibl(r, c, beta, blade_coords, exportname, 
                       pitch_axis=0.25, scale_to_mm=True, skin_thickness=0.0):
    """
    Generates a Creo .ibl file with support for a dynamic pitch_axis array.
    Forces coordinate loops to close at the trailing edge for solid CAD imports.
    """
    import numpy as np
    import aerosandbox as asb
    
    n_r, n_foil, _ = blade_coords.shape
    scale = 1000.0 if scale_to_mm else 1.0

    if np.isscalar(pitch_axis):
        pitch_axis = np.full(n_r, pitch_axis)

    with open(f"exports/{exportname}.ibl", "w") as f:
        # Note: 'Open' here refers to the loft trajectory (spanwise), not the cross-sections
        f.write("Open\nArclength\n") 
        
        for i, r_i in enumerate(r):
            pa = pitch_axis[i]
            f.write(f"Begin section ! Station {i+1}: r = {r_i:.4f} m\n")
            f.write("Begin curve\n")
            
            foil = asb.Airfoil(coordinates=blade_coords[i])
            
            foil = foil.rotate(angle=-beta[i], x_center=pa, y_center=0.0)
            foil = foil.scale(c[i], c[i])
            
            base_coords = foil.coordinates
            
            if skin_thickness > 0.0:
                core_x, core_y = offset_airfoil_profile(base_coords[:, 0], base_coords[:, 1], 
                                                        c[i], skin_thickness)
            else:
                core_x, core_y = base_coords[:, 0], base_coords[:, 1]
            
            if not (np.isclose(core_x[0], core_x[-1]) and np.isclose(core_y[0], core_y[-1])):
                core_x = np.append(core_x, core_x[0])
                core_y = np.append(core_y, core_y[0])
                
            for j in range(len(core_x)):
                # Apply the dynamic pitch axis translation
                pt_x = (core_x[j] - (pa * c[i])) * scale
                pt_y = r_i * scale  
                pt_z = core_y[j] * scale
                
                f.write(f"{j+1}  {pt_x:.6f}  {pt_y:.6f}  {pt_z:.6f}\n")
                
    part_type = "Core Insert" if skin_thickness > 0 else "Aerodynamic Skin"
    print(f"Successfully generated Creo geometry: {exportname}.ibl ({n_r} sections) - {part_type}")

def export_parting_surface_ibl(r, c, beta, blade_coords, r_foils, exportname, 
                               pitch_axis=0.25, scale_to_mm=True, 
                               x_margin=5.0, y_margin=5.0, n_camber_points=10):
    """
    Generates a 3-zone parting surface sheet exported as a single Creo .ibl file.
    Isolates the flat extensions into distinct curve blocks to eliminate all ripples.
    """    
    n_r, n_foil, _ = blade_coords.shape
    scale = 1000.0 if scale_to_mm else 1.0
    r_hub = r_foils[1]

    if np.isscalar(pitch_axis):
        pitch_axis = np.full(n_r, pitch_axis)

    station_data = []
    global_le_x = []
    global_te_x = []

    # 1. Process each radial station and extract the internal camber path
    for i, r_i in enumerate(r):
        pa = pitch_axis[i]
        foil = asb.Airfoil(coordinates=blade_coords[i])
        foil = foil.rotate(angle=-beta[i], x_center=pa, y_center=0.0)
        foil = foil.scale(c[i], c[i])
        base_coords = foil.coordinates
        
        pt_x = (base_coords[:, 0] - (pa * c[i])) * scale
        pt_y = r_i * scale              
        pt_z = base_coords[:, 1] * scale 
        
        le_idx = np.argmin(pt_x)
        te_idx = np.argmax(pt_x)
        
        x_le, z_le = pt_x[le_idx], pt_z[le_idx]
        x_te, z_te = pt_x[te_idx], pt_z[te_idx]
        
        if r_i <= r_hub:
            z_le, z_te = 0.0, 0.0

        global_le_x.append(x_le)
        global_te_x.append(x_te)

        # Separate upper and lower surface paths
        side1_x, side1_z = pt_x[:le_idx+1], pt_z[:le_idx+1]
        side2_x, side2_z = pt_x[le_idx:], pt_z[le_idx:]
        
        s1_sort = np.argsort(side1_x)
        s2_sort = np.argsort(side2_x)
        
        # Build strictly internal camber coordinates
        x_internal = np.linspace(x_le, x_te, n_camber_points + 2)[1:-1]
        z_s1_interp = np.interp(x_internal, side1_x[s1_sort], side1_z[s1_sort])
        z_s2_interp = np.interp(x_internal, side2_x[s2_sort], side2_z[s2_sort])
        
        z_internal = (z_s1_interp + z_s2_interp) / 2.0
        if r_i <= r_hub:
            z_internal = np.zeros_like(z_internal)
            
        station_data.append({
            'y': pt_y,
            'le': [x_le, z_le],
            'camber': list(zip(x_internal, z_internal)),
            'te': [x_te, z_te]
        })

    # Set outer absolute block boundaries
    x_min = np.min(global_le_x) - x_margin
    x_max = np.max(global_te_x) + x_margin

    sections = []

    # Helper function to structure a single station into 3 independent curve zones
    def build_zoned_section(y_val, target_station):
        x_le, z_le = target_station['le']
        x_te, z_te = target_station['te']
        # Introduce a tiny clearance offset (e.g., 0.5 mm) to push the crease away from the skin
        crease_offset = 0.5  

        # Zone 1: Left Flat Extension (Stops just short of hitting the LE skin)
        left_zone = [
            [x_min, y_val, z_le],
            [x_le - crease_offset, y_val, z_le]
        ]

        # Zone 2: Camber Line + Flat Lead-ins
        camber_zone = [
            [x_le - crease_offset, y_val, z_le], # Start flat in the open air
            [x_le, y_val, z_le]                  # Cross the LE skin smoothly
        ]
        for cx, cz in target_station['camber']:
            camber_zone.append([cx, y_val, cz])
        camber_zone.append([x_te, y_val, z_te])  # Cross the TE skin smoothly
        camber_zone.append([x_te + crease_offset, y_val, z_te]) # End flat in the open air

        # Zone 3: Right Flat Extension (Starts out in the open air)
        right_zone = [
            [x_te + crease_offset, y_val, z_te],
            [x_max, y_val, z_te]
        ]
        
        return {
            'left': left_zone,
            'camber': camber_zone,
            'right': right_zone
        }

    # Assemble the final spanwise matrix (Hub pad -> Blade Stations -> Tip pad)
    y_hub_ext = station_data[0]['y'] - y_margin
    sections.append(build_zoned_section(y_hub_ext, station_data[0]))

    for stat in station_data:
        sections.append(build_zoned_section(stat['y'], stat))

    y_tip_ext = station_data[-1]['y'] + y_margin
    sections.append(build_zoned_section(y_tip_ext, station_data[-1]))

    # Write out the multi-curve IBL file
    with open(f"exports/{exportname}.ibl", "w") as f:
        f.write("Open\n")
        f.write("Arclength\n") 
        
        for s_idx, sec in enumerate(sections):
            f.write(f"Begin section ! Row {s_idx+1}\n")
            
            # --- Curve Block 1: Left Plane ---
            f.write("Begin curve ! Left Extension\n")
            for pt in sec['left']:
                f.write(f"{pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f}\n")
                
            # --- Curve Block 2: Airfoil Camber ---
            f.write("Begin curve ! Internal Camber\n")
            for pt in sec['camber']:
                f.write(f"{pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f}\n")
                
            # --- Curve Block 3: Right Plane ---
            f.write("Begin curve ! Right Extension\n")
            for pt in sec['right']:
                f.write(f"{pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f}\n")
                
    print(f"Successfully generated 3-zone sharp parting surface: {exportname}.ibl")

def build_propeller_mesh(r, c, beta, blade_coords, pitch_axis=0.25, num_blades=2):
    """
    Lofts a 3D propeller blade using exactly the same 2D transforms as the Creo exporter.
    Handles dynamic pitch axes to keep the hub perfectly centered.
    """
    import pyvista as pv
    import numpy as np
    import aerosandbox as asb
    
    n_r, n_foil, _ = blade_coords.shape
    
    # Broadcast scalar pitch_axis to array if necessary
    if np.isscalar(pitch_axis):
        pitch_axis = np.full(n_r, pitch_axis)
        
    X_3d = np.zeros((n_r, n_foil))
    Y_3d = np.zeros((n_r, n_foil))
    Z_3d = np.zeros((n_r, n_foil))
    
    for i, r_i in enumerate(r):
        pa = pitch_axis[i]
        foil = asb.Airfoil(coordinates=blade_coords[i])
        
        # Rotate around the local dynamic pitch axis
        foil = foil.rotate(angle=-beta[i], x_center=pa, y_center=0.0)
        foil = foil.scale(c[i], c[i])
        
        coords = foil.coordinates
        
        # Map to 3D standard axes, offsetting by the exact physical distance of the pitch axis
        X_3d[i, :] = coords[:, 0] - (pa * c[i])
        Y_3d[i, :] = r_i  
        Z_3d[i, :] = coords[:, 1]
        
    base_blade = pv.StructuredGrid(X_3d, Y_3d, Z_3d)
    
    meshes = [base_blade]
    for i in range(1, int(num_blades)):
        angle = i * (360.0 / num_blades)
        rotated_blade = base_blade.copy()
        rotated_blade.rotate_z(angle, inplace=True)
        meshes.append(rotated_blade)
        
    return pv.MultiBlock(meshes).combine()

if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    import pyvista as pv
    import results 

    # Load the saved aerodynamic propeller
    prop_filename = "Redback V2" 
    inputs, params = results.loadprop(prop_filename)
    
    r = np.array(params.r)
    c = np.array(params.c)
    beta = np.array(params.beta)

    # Make sure r extends all the way to the origin to prevent center gaps
    if r[0] > 0.001:
        r = np.insert(r, 0, 0.0)
        c = np.insert(c, 0, c[0])
        beta = np.insert(beta, 0, beta[0])

    # Define r_foils =  [r_start, r_hub, r_aero, r_mid..., r_tip]
    R_hub = 0.015
    R_aero = 0.05
    r_foils = np.array([0.0, R_hub, R_aero, r[len(r)//2], r[-1]])
    
    # Overwrite the structural root on the 3D arrays
    HUB_CHORD = 0.03
    c, beta, pitch_axis_array = overwrite_hub_planform(
        r, c, beta, r_foils, 
        hub_chord=HUB_CHORD, hub_twist=0.0, aero_pitch_axis=0.25
    )

    # Generate Aerodynamic Weights (TEMP FUNCTION. WILL BE MOVED TO INIT WHEN CST OPTIMISATION IS IMPLEMENTED.)
    # Because r_foils has 5 elements above, we provide 3 airfoil weights.
    airfoil_names = ["clarky", "clarky", "clarky"] 
    cst_weights = get_cst_weights_from_names(airfoil_names)
    
    # Generate Coordinates using the simplified function signature
    blade_coords = generate_blade_coordinates(r, r_foils, cst_weights, hub_thickness_ratio=0.1, n_points_per_side=100)

    # Plot Airfoil Geometries at the specific r_foils control stations
    plt.figure(figsize=(11, 5))
    
    # Map each control station in r_foils to the nearest index in the dense radial grid r
    for i, r_f in enumerate(r_foils):
        idx = np.argmin(np.abs(r - r_f))
        coords = blade_coords[idx]
        plt.plot(coords[:, 0], coords[:, 1], label=f"r = {r[idx]:.3f} m")
        
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.axis("equal")
    plt.xlabel("Chord Station (x/c)")
    plt.ylabel("Thickness Station (y/c)")
    plt.title("Airfoil Geometries at Defined Control Stations (r_foils)")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5)) # Moved legend outside to accommodate labels
    plt.tight_layout()
    plt.show()

    # 3D Model of Propeller
    prop_mesh = build_propeller_mesh(
        r, c, beta, blade_coords, 
        pitch_axis=pitch_axis_array, num_blades=2
    )
    
    # Export to Creo .ibl files
    print("\nExporting Creo geometries...")
    # Main mold (Aerodynamic Skin)
    export_to_creo_ibl(
        r, c, beta, blade_coords, "propeller_blade_mold", 
        pitch_axis=pitch_axis_array, skin_thickness=0.0
    )

    # Internal Core Insert
    laminate_thickness_m = 0.0015 # 1.5mm
    export_to_creo_ibl(
        r, c, beta, blade_coords, "propeller_blade_insert", 
        pitch_axis=pitch_axis_array, skin_thickness=laminate_thickness_m
    )

    # Export Silhouette Curve as .ibl

    export_parting_surface_ibl(
        r, c, beta, blade_coords, r_foils, "propeller_parting_surf",
        pitch_axis=pitch_axis_array, n_camber_points=100
    )

    # Fire up the local PyVista testing window
    plotter = pv.Plotter()
    plotter.set_background("#1e1e2f")
    plotter.add_mesh(prop_mesh, color="silver", show_edges=True, edge_color="#444444", smooth_shading=True)
    plotter.add_axes()
    plotter.show()
    plotter.close()
    del plotter
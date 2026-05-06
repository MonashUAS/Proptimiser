## Propeller Geometry Tools and Functions

import jax.numpy as jnp
import numpy as np
import pandas as pd

def params_to_le_te(r, beta, chord):
    # Calculate Chord Offset
    chord_offset = 0.3 * np.max(chord)

    # 1. Calculate the components
    x = r.ravel()
    
    y_le = ((chord_offset + chord) * np.sin(beta)).ravel()
    z_le = ((chord_offset + chord) * np.cos(beta)).ravel()
    
    y_te = ((-chord_offset + chord) * np.sin(beta)).ravel()
    z_te = ((-chord_offset + chord) * np.cos(beta)).ravel()

    # 2. Stack them into (N, 3) arrays
    # We cast to np.array to ensure PyVista can read the memory
    le_points = np.column_stack([x, y_le, z_le])
    te_points = np.column_stack([x, y_te, z_te])

    return le_points, te_points

def UpperLowerPoints(airfoil_data_path):
    '''
    Takes airfoil geometry .dat file (contained in +AirfoilData) and returns arrays of the upper 
    and lower points. 

    Importantly, the code assumes that the first row is information like 'ARA-D 10% AIRFOIL half camber'. If this
    line is not present, the skiprows and header flag will need to be modified (so just keep the same file format :))
    '''
    foil_data = pd.read_csv(airfoil_data_path, sep=r'\s+',header=None,skiprows=1)
    
    upper_points = foil_data[foil_data.iloc[:, 1] >= 0]
    lower_points = foil_data[foil_data.iloc[:, 1] <= 0]

    return upper_points, lower_points

def PropGeomGenerator(chord, beta, r, foilpoints, exportname):
    '''
    TODO Write better documentation.
    Creates geometry files which defines the propellers shape. Jax/Python implementation of the routine
    already implemented in PropGeomGenerator.m for Propcalc. 
    '''

    #Create arrays for r,chord and beta
    r_arr     = jnp.array(r).flatten()
    chord_arr = jnp.array(chord).flatten()
    beta_arr  = jnp.array(beta).flatten()

    n = r_arr.shape[0]
    propgeom_chord_offset = (1/5) * float(chord_arr.max())

    #Array of LE coords (x,y,z)
    propgeom_LE_coords = jnp.stack([
        r_arr,
        jnp.zeros(n),
        jnp.full(n, -propgeom_chord_offset)
    ])
    #Array of TE coords (x,y,z)
    propgeom_TE_coords = jnp.stack([
        r_arr,
        -chord_arr * jnp.sin(beta_arr),
        chord_arr  * jnp.cos(beta_arr) - propgeom_chord_offset
    ])

    #Arrays for tracking x and y coordinates of foil points.
    foil_x = jnp.array(foilpoints.iloc[:, 0].values)
    foil_y = jnp.array(foilpoints.iloc[:, 1].values)
    m = foil_x.shape[0]

    propgeom_airfoil_coords = jnp.stack([
        -foil_x, #Not sure if this should be negative (probably not)?
        foil_y,
        jnp.zeros(m)
    ])

    #Convert from m to mm
    propgeom_LE_coords      = propgeom_LE_coords.T      * 1000
    propgeom_TE_coords      = propgeom_TE_coords.T      * 1000
    propgeom_airfoil_coords = propgeom_airfoil_coords.T

    #Write files
    np.savetxt("prop_LE_points.pts",  np.array(propgeom_LE_coords),      delimiter=" ", fmt="%.7f")
    np.savetxt("prop_TE_points.pts",  np.array(propgeom_TE_coords),      delimiter=" ", fmt="%.7f")
    np.savetxt(f"{exportname}.pts",   np.array(propgeom_airfoil_coords), delimiter=" ", fmt="%.7f")
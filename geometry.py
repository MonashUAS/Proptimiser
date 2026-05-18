## Propeller Geometry Tools and Functions

import numpy as np
from datetime import datetime

def params_to_le_te(r, beta, c):
    # Calculate Chord Offset
    chord_offset = 0.3 * np.max(c)

    # Calculate the components
    x = r.ravel()
    
    y_le = ((chord_offset + c) * np.sin(beta)).ravel()
    z_le = ((chord_offset + c) * np.cos(beta)).ravel()
    
    y_te = ((-chord_offset + c) * np.sin(beta)).ravel()
    z_te = ((-chord_offset + c) * np.cos(beta)).ravel()

    # Stack them into (N, 3) arrays
    le_points = np.column_stack([x, y_le, z_le])
    te_points = np.column_stack([x, y_te, z_te])

    return le_points, te_points

def upperlowerairfoil(airfoil):
    '''
    Takes airfoil geometry .dat file (contained in airfoils) and returns arrays of the upper 
    and lower points.

    Importantly, the code assumes that the first row is information like 'ARA-D 10% AIRFOIL half camber'. If this
    line is not present, the skiprows flag will need to be modified.
    '''

    arifoil_path = "airfoils/"+airfoil
    # Load data using numpy, skipping the header row
    foil_data = np.loadtxt(airfoil, skiprows=1)
    
    # Filter rows based on the y-coordinate (column index 1)
    upper_points = foil_data[foil_data[:, 1] >= 0]
    lower_points = foil_data[foil_data[:, 1] <= 0]

    return upper_points, lower_points

from datetime import datetime
import numpy as np

def ptsfilegenerator(r, c, beta, foilpoints, exportname):
    '''
    Creates geometry files which defines the propellers shape
    '''
    # Unpack the tuple of upper and lower points
    upper_points, lower_points = foilpoints
    foil_data = np.vstack([upper_points, lower_points])

    # Convert parameters to xyz coordinates
    le_points, te_points = params_to_le_te(r,c,beta)

    # Extract x and y coordinates from the unpacked NumPy arrays (Column 0 and 1)
    foil_x = foil_data[:, 0]
    foil_y = foil_data[:, 1]
    m = foil_x.shape[0]

    propgeom_airfoil_coords = np.stack([
        -foil_x, 
        foil_y,
        np.zeros(m)
    ])

    # Convert from m to mm for LE and TE points, and transpose arrays
    propgeom_LE_coords      = le_points.T      * 1000
    propgeom_TE_coords      = te_points.T      * 1000
    propgeom_airfoil_coords = propgeom_airfoil_coords.T


    # Write files with the timestamp appended
    np.savetxt(f"{exportname}_LE.pts",  propgeom_LE_coords,      delimiter=" ", fmt="%.7f")
    np.savetxt(f"{exportname}_TE.pts",  propgeom_TE_coords,      delimiter=" ", fmt="%.7f")
    np.savetxt(f"{exportname}_airfoil.pts",   propgeom_airfoil_coords, delimiter=" ", fmt="%.7f")
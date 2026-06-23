import jax.numpy as jnp
import numpy as np

def parse_apc_pe0(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    data_lines = []
    in_data_block = False

    for line in lines:
        # The data block predictably starts with the 0.0000 station
        if line.strip().startswith("0.0000"):
            in_data_block = True
            
        if in_data_block:
            # An empty line signals the end of the geometry table
            if line.strip() == "": 
                break
            
            # Split the whitespace-delimited row into floats
            row_data = [float(val) for val in line.split()]
            if len(row_data) == 14:
                data_lines.append(row_data)

    raw_data = np.array(data_lines)
    
    # Map to JAX arrays for BEMT and spatial transformations
    geometry = {
        "r": jnp.array(raw_data[:, 0]),             # Radial Station (in)
        "chord": jnp.array(raw_data[:, 1]),         # Chord length (in)
        "pitch_prather": jnp.array(raw_data[:, 4]), # Lower surface pitch interpretation (in)
        "sweep": jnp.array(raw_data[:, 5]),         # Sweep Y (in)
        "rake": jnp.array(raw_data[:, 6]),          # Rake Z (in)
        "twist_deg": jnp.array(raw_data[:, 8]),     # Twist defined with LE/TE datums (deg)
        "thickness_ratio": jnp.array(raw_data[:, 7])# Scaling factor for the Clark-Y sections
    }
    
    return geometry

# Example usage:
if __name__ == "__main__":
    from init import Inputs, Params
    from datetime import datetime
    # Internal Imports
    import propmodel
    import proptimiser
    import results
    import geometry
    from airfoil import load_airfoil_data

    ### Import Blade Geometry
    
    blade_geom = parse_apc_pe0('apc_prop_data/propgeom_data/20x8-PERF.PE0')

    # (propeller used clark-y airfoil btw)
    
    '''
    The below is copied from init.py; the goal is to modify the
    below parameters/code so it uses the imported geom and tests
    with the clark-y airfoil.

    '''
    airfoil = "arad10"

    c_scale = 0.2                   # Max chord (m)
    beta_scale = jnp.radians(50)    # Max twist (rad)

    # Initial Geometry setup
    psi, dpsi = propmodel.angular_stops(Inputs.n_psi)
    
    r, dr = propmodel.radial_stops(Inputs.R_tip, Inputs.R_hub, Inputs.n_r)

    # Twist distribution starting guess (in degrees): 
    init_beta_cp = jnp.radians(jnp.array([18.0,16.0,16.0,16.0,14.0,7.0,7.0,4.0,4.0,4.0,4.0,4.0]))

    # Chord distribution (in meters): 
    init_c_cp = jnp.array([0.05,0.05,0.05,0.05,0.05,0.05,0.04,0.03,0.03,0.03,0.03,0.03])

    # Interpolate airfoil data
    airfoil_interp = load_airfoil_data("arad10")

    spline_matrix_init = proptimiser.get_bezier_basis_matrix(r, Inputs.R_hub, Inputs.R_tip, Inputs.n_cp)
    init_c = jnp.dot(init_c_cp, spline_matrix_init.T)
    init_beta = jnp.dot(init_beta_cp, spline_matrix_init.T)

    # Initialise Params Class (will be duplicated for inputs in run_optimisation)

    params_init = Params(
        airfoil = airfoil,
        airfoil_interp=airfoil_interp,
        N=2,
        c=init_c,      
        beta=init_beta,
        c_cp=init_c_cp,
        beta_cp=init_beta_cp,
        r=r, dr=dr,    
        psi=psi, dpsi=dpsi, 
        V_x=jnp.array([Inputs.V_x[0]]), V_yz=jnp.array([Inputs.V_yz[0]]),
        omega=jnp.array([Inputs.omega[0]]),
        omega_max = Params.a / Inputs.R_tip,
        V_ia_0=jnp.full((Inputs.n_r), 4.0), # Induced Velocity Field guess
        c_scale = c_scale,
        beta_scale = beta_scale,
        error_log = jnp.array([1.0]),
        time_log = jnp.array([0]),

        # Cost Weights
        objective_thrust=jnp.array([50.0]),
        objective_MIL=jnp.array([1.0]),
        objective_power=jnp.array([0.0]),
        objective_weight=Inputs.objective_weight,
        thrust_target=Inputs.thrust_target,
    )

    print(f"Initialised Airfoil Fit with airfoil {params_init.airfoil}")


    # Find Blade Data

import jax.numpy as jnp
import numpy as np
def parse_apc_pe0(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    data_lines = []
    in_data_block = False
    airfoil_name = "Unknown"

    for line in lines:
        cleaned_line = line.strip()

        # 1. Look for the airfoil name anywhere in the file
        if "AIRFOIL1:" in cleaned_line:
            parts = cleaned_line.split(',')
            if len(parts) > 1:
                airfoil_name = parts[1].strip().split()[0]

        # 2. Track when we enter the main data table
        if cleaned_line.startswith("0.0000"):
            in_data_block = True
            
        if in_data_block:
            # An empty line signals the end of the geometry table,
            # but we switch the flag instead of breaking out of the loop completely
            if cleaned_line == "": 
                in_data_block = False
                continue
            
            # Split the whitespace-delimited row into floats
            row_data = [float(val) for val in cleaned_line.split()]
            if len(row_data) == 14:
                data_lines.append(row_data)

    raw_data = np.array(data_lines)
    
    # Map to JAX arrays
    geometry = {
        "r": jnp.array(raw_data[:, 0]),             # Radial Station (in)
        "chord": jnp.array(raw_data[:, 1]),         # Chord length (in)
        "pitch_prather": jnp.array(raw_data[:, 4]), # Lower surface pitch interpretation (in)
        "sweep": jnp.array(raw_data[:, 5]),         # Sweep Y (in)
        "rake": jnp.array(raw_data[:, 6]),          # Rake Z (in)
        "twist_deg": jnp.array(raw_data[:, 8]),     # Twist defined with LE/TE datums (deg)
        "thickness_ratio": jnp.array(raw_data[:, 7]),# Scaling factor for the sections
        "airfoil": airfoil_name                     # Extracted airfoil string identifier
    }
    
    return geometry

import numpy as np
import matplotlib.pyplot as plt

def parse_apc_dat(filepath):
    '''
    Loads the data from an APC .dat file and plots flight performance.
    '''
    with open(filepath, 'r') as f:
        lines = f.readlines()

    rpm_data = {}
    current_rpm = None
    current_block_lines = []

    for line in lines:
        cleaned = line.strip()
        
        if not cleaned:
            continue

        # Detect the start of an RPM performance block
        if "PROP RPM =" in cleaned:
            # Save the previous block before starting a new one
            if current_rpm is not None and current_block_lines:
                rpm_data[current_rpm] = np.array(current_block_lines)
                current_block_lines = []
            
            # Extract the RPM value
            try:
                current_rpm = int(float(cleaned.split('=')[1].strip().split()[0]))
            except (IndexError, ValueError):
                current_rpm = None
            continue

        # Parse data rows directly based on numeric conversion
        if current_rpm is not None:
            parts = cleaned.split()
            # APC tables typically have at least 6-8 columns of data
            if len(parts) >= 6:
                try:
                    # If this succeeds, it's a data row. If it hits a text header, it throws a ValueError
                    row_data = [float(x) for x in parts]
                    current_block_lines.append(row_data)
                except ValueError:
                    pass # Skip text headers seamlessly

    # Catch the final block remaining at the end of the file loop
    if current_rpm is not None and current_block_lines:
        rpm_data[current_rpm] = np.array(current_block_lines)

    # --- Plotting the Performance Sweeps ---
    if not rpm_data:
        print("No performance data blocks found in file.")
        return rpm_data

    # Create a figure with 2 subplots: Thrust and Power vs Velocity
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"APC Propeller Flight Performance ({filepath.split('/')[-1]})", fontsize=14, fontweight='bold')

    available_rpms = sorted(list(rpm_data.keys()))

    for rpm in available_rpms:
        matrix = rpm_data[rpm]
        
        # Standard APC PER3 Column Map:
        # col 0: V (mph)
        # col 1: J (Advance Ratio)
        # col 2: Pe (Efficiency)
        # col 3: Ct (Thrust Coeff)
        # col 4: Cp (Power Coeff)
        # col 9: PWR (w)
        # col 10: Torque (N/m)
        # col 11: Thrust (N)
        
        # Ensure matrix has enough columns before unpacking
        if matrix.shape[1] >= 8:
            V = matrix[:, 0]*0.44704 # convert to m/s
            Power = matrix[:, 8]
            Thrust = matrix[:, 10]
        else:
            print(f"Skipping plot for {rpm} RPM: Unexpected table dimensions.")
            continue

        # Plot Thrust vs V
        axes[0].plot(V, Thrust, label=f"{rpm} RPM", linewidth=1.5)
        # Plot Power vs V
        axes[1].plot(V, Power, label=f"{rpm} RPM", linewidth=1.5)

    # Format Subplot 1: Thrust
    axes[0].set_title("Thrust vs. Inflow Velocity")
    axes[0].set_xlabel("Velocity (m/s)")
    axes[0].set_ylabel("Thrust (N)")
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].legend(fontsize='small', loc='upper right', ncol=2)

    # Format Subplot 2: Power
    axes[1].set_title("Power vs. Inflow Velocity")
    axes[1].set_xlabel("Velocity (m/s)")
    axes[1].set_ylabel("Power (W)")
    axes[1].grid(True, linestyle='--', alpha=0.6)
    axes[1].legend(fontsize='small', loc='upper right', ncol=2)

    plt.tight_layout()
    plt.show()

    return rpm_data


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
    rpm_data = parse_apc_dat("apc_prop_data/performance_data/PER3_20x8.dat")


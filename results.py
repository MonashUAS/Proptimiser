## Results and Data Visualisation Functions

# External Imports
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp


# Internal Imports
import propmodel
import init
import geometry

def propinfoplot(params, show=True):
    '''
    Plot Propeller Solve Info, Including Geometry, Solver Info,
    Induced Velocity Field and Loss Information.
    '''
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "axes.labelsize": 12,
        "axes.titlesize": 14
    })
    # Extract Params
    r_m = params.r                    # Radial station in m (Needed for polar contour)
    r = params.r * 1000               # Radial station in mm
    c = params.c * 1000               # Chord in mm
    beta = np.degrees(params.beta)    # Twist in Degrees
    V_ia_0 = params.V_ia_0            # Induced velocity
    psi = params.psi                  # Azimuthal stops (for polar plot)
    
    error_log = params.error_log      # State Error Per iteration
    time_log = params.time_log        # Time Per Iteration

    # Calculate derived metrics
    V_ia = propmodel.inflow_model(V_ia_0, params)
    f_root, f_tip = propmodel.tip_root_loss_factors(V_ia, params)
    prandtl_total = f_root * f_tip * 100

    # Let the plotting begin
    fig = plt.figure(figsize=(14, 10))

    # Optimum Chord & Twist Distribution (Combined)
    # Chord Distribution
    ax1 = fig.add_subplot(2, 2, 1)
    line1 = ax1.plot(r, c, "o-", markersize=4, color="tab:blue", label=r"Chord ($c$)")
    ax1.set_title(r"\textbf{Optimum Chord \& Twist Distribution}")
    ax1.set_ylabel(r"Chord (mm)", color="tab:blue")
    ax1.set_xlabel(r"Radius ($r$) [mm]")
    ax1.set_xlim(left=0)
    ax1.tick_params(axis='y', labelcolor="tab:blue")
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Beta Distribution
    ax1_twin = ax1.twinx()
    line2 = ax1_twin.plot(r, beta, 'o-', color="tab:red", markersize=4, label=r"Twist ($\beta$)")
    ax1_twin.set_xlim(left=0)
    ax1_twin.set_ylabel(r"Twist ($^\circ$)", color="tab:red")
    ax1_twin.tick_params(axis='y', labelcolor="tab:red")
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="best")

    # Convergence & Time Plot (Combined)
    ax2 = fig.add_subplot(2, 2, 2)
    iters = range(len(error_log))
    line3 = ax2.plot(iters, error_log, 'o-', color="orange", markersize=4, label=r"State Error")
    ax2.set_title(r"\textbf{Convergence \& Computation Time}")
    ax2.set_ylabel(r"State Error", color="orange")
    ax2.set_xlabel(r"Iterations")
    ax2.tick_params(axis='y', labelcolor="orange")
    ax2.grid(True, linestyle='--', alpha=0.6)

    ax2_twin = ax2.twinx()
    line4 = ax2_twin.plot(iters, time_log, 'o-', color="purple", markersize=4, label=r"Time / Iter")
    ax2_twin.set_ylabel(r"Time per Iteration (s)", color="purple")
    ax2_twin.tick_params(axis='y', labelcolor="purple")

    lines_conv = line3 + line4
    labels_conv = [l.get_label() for l in lines_conv]
    ax2.legend(lines_conv, labels_conv, loc="best")

    # Induced Velocity over Propeller Disk (Polar)
    ax3 = fig.add_subplot(2, 2, 3, projection="polar")
    # using r_m (meters) so the contour X/Y labels make sense mathematically
    contour = ax3.contourf(psi, r_m, V_ia, cmap='viridis')
    cbar = plt.colorbar(contour, ax=ax3, pad=0.1)
    cbar.set_label(r"Induced Velocity $V_{ia}$ [m/s]")
    ax3.set_title(r"\textbf{Induced Velocity over Propeller Disk}")
    ax3.set_xlabel(r"$X$ [m]")
    ax3.set_ylabel(r"$Y$ [m]", labelpad=30)

    # Induced Loss & Prandtl Factors (Existing Twin Axis)
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(r, V_ia_0, 'o-', color="green", label=r"Induced Loss", markersize=4)
    ax4.set_title(r"\textbf{Loss Metrics by Radius}")
    ax4.set_ylabel(r"Induced Velocity (m/s)", color="green")
    ax4.tick_params(axis='y', labelcolor='green')
    ax4.set_xlabel(r"Radius ($r$) [mm]")
    ax4.grid(True, linestyle='--', alpha=0.6)

    ax4_twin = ax4.twinx()
    ax4_twin.plot(r, prandtl_total, color='blue', linestyle='--', label=r"Prandtl Loss Factors")
    ax4_twin.set_ylabel(r"Prandtl Factor (\%)", color='blue')
    ax4_twin.tick_params(axis='y', labelcolor='blue')

    if show:
        plt.tight_layout()
        plt.show()
    return fig

def is_serialisable_leaf(x):
    '''
    Determines if a PyTree leaf is a data type that Equinox serialises by default.
    '''
    if isinstance(x, (jax.Array, np.ndarray, bool, int, float, complex)):
        return True
    return False

def load_perfect_skeleton(filename, base_skeleton):
    '''
    Reads an .eqx file to discover the exact shapes of stored arrays,
    then updates a dummy base_skeleton PyTree to match those shapes perfectly,
    skipping non-serialisable configuration fields like strings or functions.
    '''
    if not filename.endswith(".eqx"):
        filename += ".eqx"
        
    # 1. Read all array/scalar shapes sequentially from the binary file stream
    file_shapes = []
    with open("propellers/"+filename, "rb") as f:
        while True:
            try:
                # np.load automatically advances the file pointer across saved entries
                arr = np.load(f)
                file_shapes.append(arr.shape)
            except Exception:
                break

    # 2. Flatten the dummy skeleton to get its structural blueprint
    dummy_leaves, treedef = jax.tree_util.tree_flatten(base_skeleton)

    # 3. Match file shapes ONLY to fields Equinox actually serialises
    perfect_leaves = []
    file_idx = 0
    
    for leaf in dummy_leaves:
        if is_serialisable_leaf(leaf):
            if file_idx >= len(file_shapes):
                raise RuntimeError(
                    f"Structure mismatch for {filename}: the file contains fewer serialized items "
                    f"({len(file_shapes)}) than the class definition expects."
                )
            shape = file_shapes[file_idx]
            file_idx += 1
            
            # FIX: For 0D scalar shapes, preserve the original Python primitive type (int, float, bool)
            # so structural configurations (like n_r, n_psi) remain concrete Python constants.
            if shape == () and isinstance(leaf, (bool, int, float, complex)) and not isinstance(leaf, jax.Array):
                perfect_leaves.append(type(leaf)(0))
            else:
                perfect_leaves.append(jnp.zeros(shape))
        else:
            # Leave non-serialisable elements (strings, functions) completely untouched
            perfect_leaves.append(leaf)

    if file_idx != len(file_shapes):
        raise RuntimeError(
            f"Structure mismatch for {filename}: the file contains {len(file_shapes)} serialised items, "
            f"but the skeleton layout identified {file_idx} serialisable fields."
        )

    # 4. Reconstruct the perfectly shaped skeleton PyTree
    perfect_skeleton = jax.tree_util.tree_unflatten(treedef, perfect_leaves)
    return perfect_skeleton


def loadprop(prop_timestamp):
    '''
    Load and plot a propeller from saved files using just the timestamp.
    Automatically detects and handles all array dimensions (including dynamic logs).
    '''
    inputs_filename = f"prop_inputs_{prop_timestamp}"
    params_filename = f"prop_params_{prop_timestamp}"

    # --- 1. Load Inputs ---
    print(f"Loading {inputs_filename}.eqx...")
    dummy_inputs = init.Inputs()
    perfect_inputs_skeleton = load_perfect_skeleton(inputs_filename, dummy_inputs)
    inputs = perfect_inputs_skeleton.load_state(inputs_filename)

    # --- 2. Load Params ---
    print(f"Loading {params_filename}.eqx...")
    
    # Define a fast dummy airfoil interpolator to bypass slow loading & NaN errors
    def dummy_airfoil_interp(*args, **kwargs):
        return jnp.zeros(1), jnp.zeros(1), jnp.zeros(1)

    # Initialize a baseline structure using generic 0D scalar arrays
    dummy_scalar = jnp.array(0.0)
    base_params = init.Params(
        airfoil="arad10",
        airfoil_interp=dummy_airfoil_interp,
        N=2,
        c=dummy_scalar, beta=dummy_scalar, c_cp=dummy_scalar, beta_cp=dummy_scalar,
        r=dummy_scalar, dr=dummy_scalar, psi=dummy_scalar, dpsi=dummy_scalar,
        V_x=dummy_scalar, V_yz=dummy_scalar, omega=dummy_scalar, omega_max=0.0,
        V_ia_0=dummy_scalar, c_scale=0.0, beta_scale=0.0,
        error_log=dummy_scalar, time_log=dummy_scalar,
        objective_thrust=dummy_scalar, objective_MIL=dummy_scalar, objective_power=int(1),
        objective_weight=dummy_scalar, thrust_target=dummy_scalar,
    )

    # Automatically adapt the base_params structure to match the file's exact shapes
    perfect_params_skeleton = load_perfect_skeleton(params_filename, base_params)
    params = perfect_params_skeleton.load_state(params_filename)

    return inputs, params

def iterativesmoothing(params):
    return params


if __name__ == "__main__":
    # pass the base name/timestamp
    timestamp = "20260531-1754"
    inputs, params = loadprop(timestamp)

    ## Prompts for Data Processing
    user_input = input("Would you like to plot solver outputs? (y/n) \n")
    if user_input.lower() == "y":
        print("Plotting Solver Ouput...")
        # Plot the results
        propinfoplot(params)

    user_input = input("Does your propeller need control-point based smoothing? \n")

    user_input = input('Export Propeller Geometry to .pts files? (y/n): \n')
    if user_input.lower() == 'y':
        exportname = f"prop_pts_{timestamp}"
        foilpoints = geometry.upperlowerairfoil(params.airfoil)
        geometry.ptsfilegenerator(params.r, params.c, params.beta, foilpoints, exportname)
        print(f'Saving Propeller Geometry to {exportname}.pts')

    
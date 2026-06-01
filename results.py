## Results and Data Visualisation Functions

# External Imports
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import equinox as eqx
from scipy.interpolate import make_interp_spline

# Internal Imports
import propmodel
import proptimiser
import init
import geometry
import airfoil

class DraggableSplineEditor:
    def __init__(self, r_dense, y_dense, num_cps=6, name="Value"):
        self.r_dense = r_dense
        self.y_dense = y_dense
        self.name = name
        
        # 1. Initialize Control Points (CPs) evenly spaced across the radius
        self.cp_r = np.linspace(r_dense.min(), r_dense.max(), num_cps)
        self.cp_y = np.interp(self.cp_r, r_dense, y_dense)
        
        # 2. Setup the plot
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.ax.set_title(f"Interactive Editor: {name}\nDrag the red points up/down/left/right. Close window when finished.")
        self.ax.set_xlabel("Radius (r)")
        self.ax.set_ylabel(name)
        
        # Plot original data as a faint background reference
        self.ax.plot(r_dense, y_dense, '.', color='gray', alpha=0.5, label='Original Data')
        
        # Plot the active spline
        self._spline_line, = self.ax.plot(r_dense, self._get_spline(), 'b-', linewidth=2, label='Smoothed Spline')
        
        # Plot the draggable control points
        self._cp_scatter = self.ax.scatter(self.cp_r, self.cp_y, c='red', s=100, zorder=5, label='Control Points')
        
        self.ax.legend()
        self.ax.grid(True, linestyle='--', alpha=0.6)
        
        # State variable to track which point is currently being dragged
        self._dragged_idx = None 
        
        # 3. Connect Matplotlib Event Listeners
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        
    def _get_spline(self):
        # Generate the cubic spline based on current CP positions
        spline = make_interp_spline(self.cp_r, self.cp_y, k=3)
        return spline(self.r_dense)
        
    def on_press(self, event):
        if event.inaxes != self.ax: return
        
        # Calculate normalized 2D distance for intuitive grabbing
        x_range = self.cp_r[-1] - self.cp_r[0]
        y_range = max(self.y_dense.max() - self.y_dense.min(), 1e-6)
        
        dx = (self.cp_r - event.xdata) / x_range
        dy = (self.cp_y - event.ydata) / y_range
        dist = np.sqrt(dx**2 + dy**2)
        idx = np.argmin(dist)
        
        # If clicked reasonably close to a point (5% of plot area), "grab" it
        if dist[idx] < 0.05: 
            self._dragged_idx = idx

    def on_release(self, event):
        # Release the point
        self._dragged_idx = None

    def on_motion(self, event):
        if self._dragged_idx is None: return
        if event.inaxes != self.ax: return
        
        # Update the Y-value for all points freely
        self.cp_y[self._dragged_idx] = event.ydata
        
        # Update the X-value ONLY for internal points 
        # (Endpoints must stay locked at the hub and tip boundaries)
        if 0 < self._dragged_idx < len(self.cp_r) - 1:
            # Constrain X movement so it cannot pass its neighboring points
            # This ensures X remains strictly increasing, preventing spline crashes
            min_x = self.cp_r[self._dragged_idx - 1] + 1e-5
            max_x = self.cp_r[self._dragged_idx + 1] - 1e-5
            
            self.cp_r[self._dragged_idx] = np.clip(event.xdata, min_x, max_x)
        
        # Redraw the scatter points and the new spline
        self._cp_scatter.set_offsets(np.c_[self.cp_r, self.cp_y])
        self._spline_line.set_ydata(self._get_spline())
        self.fig.canvas.draw_idle()
        
    def get_final_data(self):
        # Called after the window is closed to retrieve the final array
        return self._get_spline()

def iterativesmoothing(params):
    '''
    Opens interactive graphical editors to manually smooth Chord and Beta.
    '''
    print("Opening Interactive Smoother for Chord...")
    chord_editor = DraggableSplineEditor(params.r, params.c, num_cps=10, name="Chord (c)")
    plt.show() # Pauses execution until the user closes the plot window
    new_c = chord_editor.get_final_data()
    
    print("Opening Interactive Smoother for Twist (Beta)...")
    beta_editor = DraggableSplineEditor(params.r, params.beta, num_cps=10, name="Twist (Beta)")
    plt.show()
    new_beta = beta_editor.get_final_data()
    
    # Safely inject the modified arrays back into the PyTree
    params = eqx.tree_at(
        lambda p: (p.c, p.beta), 
        params, 
        (new_c, new_beta)
    )
    
    return params

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
    Reads an .eqx file to discover the exact shapes and data types of stored arrays,
    then updates a dummy base_skeleton PyTree to match perfectly,
    skipping non-serialisable configuration fields like strings or functions.
    '''
    if not filename.endswith(".eqx"):
        filename += ".eqx"
        
    # 1. Read all array/scalar shapes and dtypes sequentially from the binary file stream
    file_shapes = []
    file_dtypes = []
    with open("propellers/"+filename, "rb") as f:
        while True:
            try:
                # np.load automatically advances the file pointer across saved entries
                arr = np.load(f)
                file_shapes.append(arr.shape)
                file_dtypes.append(arr.dtype)
            except Exception:
                break

    # 2. Flatten the dummy skeleton to get its structural blueprint
    dummy_leaves, treedef = jax.tree_util.tree_flatten(base_skeleton)

    # 3. Match file shapes AND dtypes ONLY to fields Equinox actually serialises
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
            dtype = file_dtypes[file_idx]
            file_idx += 1
            
            # For 0D scalar shapes, use the saved dtype to cast a 0 into the correct 
            # native Python primitive (int, float, bool)
            if shape == () and isinstance(leaf, (bool, int, float, complex)) and not isinstance(leaf, jax.Array):
                # .item() safely casts a 0D numpy scalar to the nearest Python type
                perfect_leaves.append(np.zeros((), dtype=dtype).item())
            else:
                # Use the extracted dtype for the dummy JAX array
                perfect_leaves.append(jnp.zeros(shape, dtype=dtype))
        else:
            # Leave non-serialisable elements (strings, functions) completely untouched
            perfect_leaves.append(leaf)

    if file_idx != len(file_shapes):
        raise RuntimeError(
            f"Structure mismatch for {filename}: the file contains {len(file_shapes)} serialised items, "
            f"but the skeleton layout identified {file_idx} serialisable fields."
        )

    # 4. Reconstruct the perfectly shaped/typed skeleton PyTree
    perfect_skeleton = jax.tree_util.tree_unflatten(treedef, perfect_leaves)
    return perfect_skeleton


def loadprop(prop_timestamp):
    '''
    Load and plot a propeller from saved files using just the timestamp.
    Automatically detects and handles all array dimensions (including dynamic logs).
    '''
    inputs_filename = f"prop_inputs_{prop_timestamp}"
    params_filename = f"prop_params_{prop_timestamp}"

    # Load Inputs
    print(f"Loading {inputs_filename}.eqx...")
    dummy_inputs = init.Inputs()
    perfect_inputs_skeleton = load_perfect_skeleton(inputs_filename, dummy_inputs)
    inputs = perfect_inputs_skeleton.load_state(inputs_filename)

    # Load Params
    print(f"Loading {params_filename}.eqx...")
    
    # Define a fast dummy airfoil interpolator to bypass slow loading & NaN errors
    airfoil_interp = airfoil.load_airfoil_data("arad10")

    # Initialize a baseline structure using generic 0D scalar arrays
    dummy_scalar = jnp.array(0.0)
    base_params = init.Params(
        airfoil="arad10",
        airfoil_interp=airfoil_interp,
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


def propgeomblender(timestamps, weights):
    '''
    Averages chord and beta across multiple timestamps, 
    returning params with the updated geometry.
    '''
    # Collect data into arrays of shape (len(timestamps), N_radial_stops)
    all_c = []
    all_beta = []
    
    for ts in timestamps:
        inputs, params = loadprop(ts)
        all_c.append(params.c)
        all_beta.append(params.beta)
    
    C_matrix = np.array(all_c)
    Beta_matrix = np.array(all_beta)
    
    avg_c = weights.dot(C_matrix)
    avg_beta = weights.dot(Beta_matrix)

    params = eqx.tree_at(
    lambda p: (p.c, p.beta), 
    params, 
    (avg_c, avg_beta)
    )
    return inputs, params


if __name__ == "__main__":
    # pass the base name/timestamp
    timestamps = ["20260531-1734"]
    weights = np.array([1.0])
    # Blending with multiple propeller geometries
    inputs, params = propgeomblender(timestamps, weights)
    # Specify Inflow Conditions for Plotting/Solving
    omega = params.omega
    V_x = jnp.array([8.60])
    V_yz = jnp.array([12.29])

    params = eqx.tree_at(
    lambda p: (p.omega, p.V_x, p.V_yz), 
    params, 
    (omega, V_x, V_yz)
    )
    # Solve Induced Velocity
    _, V_ia_0 = propmodel.induced_velocity(params)

    params = eqx.tree_at(
    lambda p: (p.V_ia_0), 
    params, 
    (V_ia_0))

    ## Propeller Post-Processing Loop
    exit_input = "n"
    while exit_input != "y":
        user_input = input("Would you like to plot solver outputs? (y/n) \n")
        if user_input.lower() == "y":
            print("Plotting Solver Ouput...")
            # Plot the results
            propinfoplot(params)

        user_input = input("Does your propeller need control-point based smoothing? (y/n)\n")
        if user_input.lower() == 'y':
            params = iterativesmoothing(params)

        user_input = input('Export Propeller Geometry to .pts files? (y/n): \n')
        if user_input.lower() == 'y':
            propname = input('Enter Propeller Name: ')
            exportname = f"prop_pts_{propname}"
            foilpoints = geometry.upperlowerairfoil(params.airfoil)
            geometry.ptsfilegenerator(params.r, params.c, params.beta, foilpoints, exportname)
            print(f'Saving Propeller Geometry to {exportname}.pts')
        # Save Results/Export Geometry if Required
        user_input = input('Save Propeller? (y/n): ')
        if user_input.lower() == 'y':
            propname = input('Enter Propeller Name:')
            paramsfile = "prop_params_" + propname
            inputsfile = "prop_inputs_" + propname
            print(f'Saving Propeller Inputs/Parameters to {paramsfile} and {inputsfile}')

            # Use the instantiated object references (params_opt and an Inputs instance)
            params.save_state(paramsfile)
            inputs.save_state(inputsfile)
            
        # Debug Print Statement (Solve for Current Propeller)
        pwr, thr, mil, V_ia_0 = proptimiser.objectives(params)
        omega, pwr, thr, mil = np.squeeze(omega), np.squeeze(pwr), np.squeeze(thr), np.squeeze(mil)

        iter = len(params.error_log)
        print(f"Iter: {iter:<3} | Omega: {omega:<5.2f} | Motor Power: {pwr:<8.4f} | Thrust: {thr:<8.4f} | MIL RMS: {mil:<10.4f}")

        exit_input = input('Would you like to exit the editor (y) or loop through (n)? (y/n) \n')
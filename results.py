## Results and Data Visualisation Functions

# External Imports
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import equinox as eqx
from scipy.interpolate import make_interp_spline
import queue

# Internal Imports
import propmodel
import init
#import geometry
import airfoil
import airflow

class DraggableSplineEditor:
    def __init__(self, r_dense, y_dense, num_cps=6, name="Value"):
        self.r_dense = r_dense
        self.y_dense = y_dense
        self.name = name
        
        # Initialize Control Points evenly spaced across the radius
        self.cp_r = np.linspace(r_dense.min(), r_dense.max(), num_cps)
        self.cp_y = np.interp(self.cp_r, r_dense, y_dense)
        
        # Setup the plot
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
        
        # Connect Matplotlib Event Listeners
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

class LivePropPlotter:
    def __init__(self, params_base):
        # 1. Setup Figure
        self.fig = plt.figure(figsize=(14, 8))
        self.r = np.array(params_base.r * 1000)
        self.r_m = np.array(params_base.r)
        self.psi = np.array(params_base.psi)

        # Subplot 1: Chord & Twist
        self.ax1 = self.fig.add_subplot(2, 2, 1)
        self.line_c, = self.ax1.plot(self.r, np.zeros_like(self.r), 'b-o', label='Chord (mm)')
        self.ax1_twin = self.ax1.twinx()
        self.line_beta, = self.ax1_twin.plot(self.r, np.zeros_like(self.r), 'r-o', label='Beta (deg)')
        self.ax1.set_title("Chord & Twist")
        self.ax1.legend(loc="upper left")
        self.ax1_twin.legend(loc="upper right")

        # Subplot 2: Convergence
        self.ax2 = self.fig.add_subplot(2, 2, 2)
        self.line_err, = self.ax2.plot([], [], 'orange', marker='o', label='State Error')
        self.ax2.set_title("Convergence")
        self.ax2.set_yscale('log') # Usually better for error tracking

        # Subplot 3: Polar Map
        self.ax3 = self.fig.add_subplot(2, 2, 3, projection="polar")
        self.ax3.set_title("Induced Velocity [m/s]")
        dummy_V_ia = np.zeros((len(self.r), len(self.psi)))
        self.contour = self.ax3.contourf(self.psi, self.r_m, dummy_V_ia, cmap='viridis')
        self.cbar = self.fig.colorbar(self.contour, ax=self.ax3, pad=0.1)

        plt.tight_layout()

    def update(self, frame_data, q):
        """
        FuncAnimation calls this repeatedly. We drain the queue to get the newest data.
        """
        latest_data = None
        # Drain the queue to only plot the absolute latest state (prevents UI lag)
        while not q.empty():
            try:
                latest_data = q.get_nowait()
            except queue.Empty:
                break
        
        if latest_data is None:
            return # Nothing new to plot

        # 1. Update Lines
        self.line_c.set_ydata(latest_data['c'])
        self.line_beta.set_ydata(latest_data['beta'])
        
        iters = np.arange(len(latest_data['error_log']))
        self.line_err.set_data(iters, latest_data['error_log'])

        # 2. Re-scale Axes dynamically
        self.ax1.relim(); self.ax1.autoscale_view()
        self.ax1_twin.relim(); self.ax1_twin.autoscale_view()
        self.ax2.relim(); self.ax2.autoscale_view()

        # 3. Update Polar Contour (Requires clearing old collections)
        for coll in self.ax3.collections:
            coll.remove()
        self.contour = self.ax3.contourf(self.psi, self.r_m, latest_data['V_ia'], cmap='viridis')

        return self.line_c, self.line_beta, self.line_err

def debug_diagnostics_plot(params, cond_idx, show=True):
    '''
    Generates a 2x2 diagnostic plot window for a specific flight condition.
    Displays Twist/Inflow, Lift/Drag Coefficients, Sectional Forces, and Reynolds Number.
    '''
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "axes.labelsize": 12,
        "axes.titlesize": 14
    })
    
    # Extract Geometry
    r = params.r * 1000               # Radius in mm
    beta_deg = np.degrees(params.beta)
    
    # Resolve Flow Kinematics
    V_ia = propmodel.inflow_model(params.V_ia_0, params)
    V_R, phi = airflow.resultantVelocityComponents(V_ia, params)
    
    # Average across the azimuthal axis for 1D plotting
    phi_mean_deg = np.degrees(jnp.mean(phi, axis=1))
    VR_mean = jnp.mean(V_R, axis=1)
    
    # Extract Aerodynamic Coefficients
    C_L, C_D, _ = airfoil.aero_coeffs(V_ia, params)
    CL_mean = jnp.mean(C_L, axis=1)
    CD_mean = jnp.mean(C_D, axis=1)
    
    # Extract Sectional Forces
    dT_ave, _ = propmodel.dT_blade(V_ia, params)
    dQ_ave = propmodel.dQ_blade(V_ia, params)
    
    # Extract Flow Regime (Reynolds Number)
    Re = airflow.calc_reynolds(params.rho, jnp.maximum(VR_mean, 1e-6), params.c, params.mu)
    
    # Create the Figure
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(rf"\textbf{{Diagnostic Diagnostics: Flight Condition {cond_idx}}}", fontsize=16)

    # Plot 1: Angles (Twist and Inflow)
    axs[0, 0].plot(r, beta_deg, 'o-', color="tab:red", label=r"Twist ($\beta$)")
    axs[0, 0].plot(r, phi_mean_deg, 's-', color="tab:cyan", label=r"Inflow Angle ($\phi$)")
    axs[0, 0].fill_between(r, beta_deg, phi_mean_deg, color='gray', alpha=0.2, label=r"Approx AoA ($\alpha$)")
    axs[0, 0].set_title(r"Angle Distribution")
    axs[0, 0].set_ylabel(r"Angle ($^\circ$)")
    axs[0, 0].grid(True, linestyle='--', alpha=0.6)
    axs[0, 0].legend()

    # Plot 2: Aerodynamic Coefficients
    axs[0, 1].plot(r, CL_mean, 'o-', color="tab:blue", label=r"$C_L$")
    axs[0, 1].set_title(r"Sectional Aerodynamic Coefficients")
    axs[0, 1].set_ylabel(r"$C_L$", color="tab:blue")
    axs[0, 1].tick_params(axis='y', labelcolor="tab:blue")
    axs[0, 1].grid(True, linestyle='--', alpha=0.6)
    
    ax01_twin = axs[0, 1].twinx()
    ax01_twin.plot(r, CD_mean, 's-', color="tab:orange", label=r"$C_D$")
    ax01_twin.set_ylabel(r"$C_D$", color="tab:orange")
    ax01_twin.tick_params(axis='y', labelcolor="tab:orange")
    
    lines_coeff = axs[0, 1].get_lines() + ax01_twin.get_lines()
    axs[0, 1].legend(lines_coeff, [l.get_label() for l in lines_coeff], loc="center right")

    # Plot 3: Sectional Forces (Thrust & Torque)
    axs[1, 0].plot(r, dT_ave, 'o-', color="tab:green", label=r"Thrust ($dT$)")
    axs[1, 0].set_title(r"Sectional Load Distribution")
    axs[1, 0].set_ylabel(r"Thrust (N/m)", color="tab:green")
    axs[1, 0].set_xlabel(r"Radius ($r$) [mm]")
    axs[1, 0].tick_params(axis='y', labelcolor="tab:green")
    axs[1, 0].grid(True, linestyle='--', alpha=0.6)

    ax10_twin = axs[1, 0].twinx()
    ax10_twin.plot(r, dQ_ave, 's-', color="purple", label=r"Torque ($dQ$)")
    ax10_twin.set_ylabel(r"Torque (Nm/m)", color="purple")
    ax10_twin.tick_params(axis='y', labelcolor="purple")
    
    lines_forces = axs[1, 0].get_lines() + ax10_twin.get_lines()
    axs[1, 0].legend(lines_forces, [l.get_label() for l in lines_forces], loc="center right")

    # Plot 4: Reynolds Number
    axs[1, 1].plot(r, Re, 'o-', color="black", label=r"Reynolds Number")
    axs[1, 1].set_title(r"Local Flow Regime")
    axs[1, 1].set_ylabel(r"$Re$")
    axs[1, 1].set_xlabel(r"Radius ($r$) [mm]")
    axs[1, 1].set_yscale('log') # Log scale is better for Re
    axs[1, 1].grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    if show:
        plt.show(block=False) # Non-blocking so all windows open together
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
        
    # Read all array/scalar shapes and dtypes sequentially from the binary file stream
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

    # Flatten the dummy skeleton to get its structural blueprint
    dummy_leaves, treedef = jax.tree_util.tree_flatten(base_skeleton)

    # Match file shapes AND dtypes ONLY to fields Equinox actually serialises
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
    import proptimiser
    import geometry
    # pass the base name/timestamp
    timestamps = ["redback_v1"]
    weights = np.array([1.0])
    
    # Blending with multiple propeller geometries
    inputs, params_base = propgeomblender(timestamps, weights)

    print(f"--- Evaluating {inputs.n_cond} Flight Conditions ---")
    omega = jnp.array([480.0,451.91])

    # Execution & Processing Loop
    exit_input = "n"
    while exit_input != "y":
        evaluated_conditions = []

        print("\nResolving Induced Velocity Field for all conditions...")
        for idx in range(inputs.n_cond):
            # Unpack unique operating points and batched variables
            cond_omega = omega[idx] # manually input omega above (solver export limitations)
            cond_Vx = 0 #inputs.V_x[idx]
            cond_Vyz = 0 #inputs.V_yz[idx]

            # Evaluate at operating conditons
            params_cond = eqx.tree_at(
                lambda p: (p.omega, p.V_x, p.V_yz), 
                params_base, 
                (jnp.array([cond_omega]), jnp.array([cond_Vx]), jnp.array([cond_Vyz]))
            )
            
            # This guarantees that geometry changes (e.g. from smoothing) propagate into V_ia.
            _, V_ia_0_solved = propmodel.induced_velocity(params_cond)

            # write solved V_ia_0 into params
            params_cond = eqx.tree_at(
                lambda p: p.V_ia_0, 
                params_cond, 
                V_ia_0_solved
            )
            
            evaluated_conditions.append(params_cond)
            
            # Extract and print objective metrics
            pwr, thr, mil, _ = proptimiser.objectives(params_cond)
            cond_omega, pwr, thr, mil = jnp.squeeze(cond_omega), jnp.squeeze(pwr), jnp.squeeze(thr), jnp.squeeze(mil)
            print(f"Cond: {idx:<3} | Omega: {float(cond_omega):<6.2f} | Motor Pwr: {float(pwr):<8.4f} | Thrust: {float(thr):<8.4f} | MIL RMS: {float(mil):<10.4f}")

        # Post-Processing User Directives
        user_input = input("\nWould you like to plot solver outputs? (y/n): ")
        if user_input.lower() == "y":
            print("Plotting General Info for Condition 0...")
            propinfoplot(evaluated_conditions[0], show=False) 
            
            print("Plotting Detailed Diagnostics for all conditions...")
            for idx, p_cond in enumerate(evaluated_conditions):
                debug_diagnostics_plot(p_cond, idx, show=False)
            
            plt.show()  # Blocks until all window instances are closed by the user

        user_input = input("\nDoes your propeller need control-point based smoothing? (y/n): ")
        if user_input.lower() == 'y':
            # Smoothing modifies the global underlying geometry skeleton
            params_base = iterativesmoothing(params_base)
            print("Geometry updated. Induced velocity fields will automatically refresh on the next loop pass.")

        user_input = input('\nExport Propeller Geometry to .pts files? (y/n): ')
        if user_input.lower() == 'y':
            propname = input('Enter Propeller Name: ')
            exportname = f"prop_pts_{propname}"
            foilpoints = geometry.upperlowerairfoil(params_base.airfoil)
            geometry.ptsfilegenerator(params_base.r, params_base.c, params_base.beta, foilpoints, exportname)
            print(f'Saving Propeller Geometry to {exportname}.pts')
            
        exit_input = input('\nExit configuration loop completely? (y/n): ')
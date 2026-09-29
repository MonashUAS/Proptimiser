## Results and Data Visualisation Functions

# External Imports
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import equinox as eqx
from scipy.interpolate import make_interp_spline
import queue

# Internal Imports
import propmodel
from init import Inputs, Propeller
import geometry
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
        # Compute a safe top limit with 10% padding above maximum data point
        y_max = float(np.max(y_dense))
        top_limit = y_max * 1.1 if y_max > 0 else 1.0

        # Explicitly set bounds (grounds bottom at 0, top at y_max + padding)
        self.ax.set_ylim(bottom=0, top=top_limit)
        
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
        y_range = max(self.y_dense.max(), 1e-6)
        
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

def iterativesmoothing(prop):
    '''
    Opens interactive graphical editors to manually smooth Chord and Beta; returns the updated Propeller.
    '''
    r = np.array(prop.r)

    print("Opening Interactive Smoother for Chord...")
    chord_editor = DraggableSplineEditor(r, np.array(prop.c), num_cps=10, name="Chord (c)")
    plt.show() # Pauses execution until the user closes the plot window
    new_c = chord_editor.get_final_data()

    print("Opening Interactive Smoother for Twist (Beta)...")
    beta_editor = DraggableSplineEditor(r, np.array(prop.beta), num_cps=10, name="Twist (Beta)")
    plt.show()
    new_beta = beta_editor.get_final_data()

    return prop.with_state(c=new_c, beta=new_beta)


class LivePropPlotter:
    '''
    Live Updating Solver Plot with Debugging information
    '''
    def __init__(self, params_base):
        # Setup Figure
        self.fig = plt.figure(figsize=(18, 9))
        self.r = np.array(params_base.r * 1000)  # Radius in mm
        self.r_m = np.array(params_base.r)      # Radius in m
        self.psi = np.array(params_base.psi)
        self.r_cst = np.array(params_base.r_cst)
        self.x_foil = geometry.cosine_spacing(80)

        # Subplot: Chord & Twist
        self.ax1 = self.fig.add_subplot(2, 3, 1)
        self.line_c, = self.ax1.plot(self.r, np.zeros_like(self.r), 'b-o', label='Chord (mm)')
        self.ax1_twin = self.ax1.twinx()
        self.line_beta, = self.ax1_twin.plot(self.r, np.zeros_like(self.r), 'r-o', label='Beta (deg)')
        self.ax1.set_title("Chord and Twist Distribution")
        self.ax1.set_xlabel("Radius (mm)")
        self.ax1.set_ylabel("Chord (mm)", color='b')
        self.ax1_twin.set_ylabel("Beta (deg)", color='r')
        self.ax1.legend(loc="upper left")
        self.ax1_twin.legend(loc="upper right")
        self.ax1.grid(True, alpha=0.3)

        # Subplot 2: Convergence
        self.ax2 = self.fig.add_subplot(2, 3, 2)
        self.line_err, = self.ax2.plot([], [], 'orange', marker='o', label='State Error')
        self.ax2.set_title("Solver Convergence History")
        self.ax2.set_xlabel("Iteration")
        self.ax2.set_ylabel("Residual Error")
        self.ax2.set_yscale('log') 
        self.ax2.grid(True, which="both", alpha=0.3)

        # Subplot 3: CST sections at the hub and aerodynamic stations
        self.ax_foil = self.fig.add_subplot(2, 3, 3)
        self.lines_foil = [
            self.ax_foil.plot([], [], label=f"r = {r_f * 1000:.0f} mm")[0] for r_f in self.r_cst
        ]
        self.ax_foil.set_title("Airfoil Sections (r_foils stations)")
        self.ax_foil.set_xlabel("x/c")
        self.ax_foil.set_ylabel("y/c")
        self.ax_foil.set_aspect("equal", adjustable="datalim")
        self.ax_foil.legend(loc="upper right", fontsize=8)
        self.ax_foil.grid(True, alpha=0.3)

        dummy_V_i_a = np.zeros((len(self.r), len(self.psi)))
        # Wrap data periodically for 360-degree polar contouring
        psi_plot = np.append(self.psi, 2 * np.pi)
        dummy_V_i_a = np.hstack([dummy_V_i_a, dummy_V_i_a[:, :1]])

        # Subplot 4: Polar Map (Induced Velocity+swirl vectors)
        self.ax3 = self.fig.add_subplot(2, 3, 4, projection="polar")
        self.ax3.set_title("Induced Velocity Map [m/s]\n", pad=15)
        self.contour = self.ax3.contourf(psi_plot, self.r_m, dummy_V_i_a, cmap='viridis')
        self.cbar = self.fig.colorbar(self.contour, ax=self.ax3, pad=0.1)

        # Subplot 5: Induced Velocity Profile & Loss Metrics
        self.ax4 = self.fig.add_subplot(2, 3, 5)
        self.line_via_avg, = self.ax4.plot(self.r, np.zeros_like(self.r), 'g-', linewidth=2, label=r'Avg $V_{i,a}$ [m/s]')
        self.line_vit_avg, = self.ax4.plot(self.r, np.zeros_like(self.r), 'c-', linewidth=2, label=r'Avg $V_{i,t}$ [m/s]')

        self.ax4_twin = self.ax4.twinx()
        self.line_loss, = self.ax4_twin.plot(self.r, np.zeros_like(self.r), 'm--', linewidth=1.5, label='Loss Factor ($F$)')
        
        self.ax4.set_title("Induced Velocity Profile and Prandtl Loss Metrics")
        self.ax4.set_xlabel("Radius (mm)")
        self.ax4.set_ylabel(r"Induced Velocity $V_{i}$ (m/s)", color='g')
        self.ax4_twin.set_ylabel("Prandtl Factor ($F$)", color='m')
        self.ax4_twin.set_ylim(-0.05, 1.05) # Bound the loss factor naturally between 0 and 1
        
        # Update the legend to include the new line
        lines = [self.line_via_avg, self.line_vit_avg, self.line_loss]
        labels = [l.get_label() for l in lines]
        self.ax4.legend(lines, labels, loc='upper right')

        # Subplot 6: Section thickness along the span
        self.ax_tc = self.fig.add_subplot(2, 3, 6)
        self.line_tc, = self.ax_tc.plot(self.r, np.zeros_like(self.r), 'k-o', label='t/c at BEMT stations')
        self.points_tc, = self.ax_tc.plot(self.r_cst * 1000, np.zeros_like(self.r_cst), 'rs', label='r_foils stations')
        self.ax_tc.set_title("Section Thickness Distribution")
        self.ax_tc.set_xlabel("Radius (mm)")
        self.ax_tc.set_ylabel("Max t/c")
        self.ax_tc.legend(loc="upper right")
        self.ax_tc.grid(True, alpha=0.3)

        plt.tight_layout()

    def update(self, frame, q):
        """
        FuncAnimation calls this repeatedly. We drain the queue to get the newest params data.
        """
        latest_params = None
        while not q.empty():
            try:
                latest_params = q.get_nowait()
            except queue.Empty:
                break
        
        if latest_params is None:
            return self.line_c, self.line_beta, self.line_err, self.line_via_avg, self.line_vit_avg, self.line_loss

        # --- Compute aerodynamic flow states internally ---
        # This keeps the main execution loop entirely free of plotting logic
        V_i = propmodel.inflow_model(latest_params.V_i_0, latest_params)
        _, phi = airflow.resultantVelocityComponents(V_i, latest_params)
        tip_loss, root_loss = airflow.tip_root_loss_factors(phi, latest_params)
        F_curr = jnp.multiply(tip_loss, root_loss) 

        # Update Subplot 1 Lines (Geometry)
        self.line_c.set_ydata(latest_params.c * 1000)
        self.line_beta.set_ydata(jnp.degrees(latest_params.beta))
        
        # Update Subplot 2 Line (Convergence)
        error_log = latest_params.error_log
        iters = np.arange(len(error_log))
        self.line_err.set_data(iters, error_log)

        # Update Subplot 3 Lines (CST sections, closed TE -> LE -> TE outlines)
        y_upper, y_lower = geometry.kulfan_coordinates(latest_params.cst, self.x_foil)
        x_loop = np.concatenate([self.x_foil[::-1], self.x_foil[1:]])
        for line, y_u, y_l in zip(self.lines_foil, np.array(y_upper), np.array(y_lower)):
            line.set_data(x_loop, np.concatenate([y_u[::-1], y_l[1:]]))

        # Update Subplot 6 Lines (Thickness)
        kulfan_r = geometry.cst_at_stations(latest_params.r, latest_params.r_cst, latest_params.cst)
        self.line_tc.set_ydata(geometry.max_thickness(kulfan_r))
        self.points_tc.set_ydata(geometry.max_thickness(latest_params.cst))

        # Update Subplot 4 Lines (Induced Velocity Profile & Losses)
        V_ia = V_i[0]
        V_it = V_i[1]
        via_avg_profile = np.mean(np.array(V_ia), axis=1) # Average over azimuthal stations (psi)
        vit_avg_profile = np.mean(np.array(V_it), axis=1) # Average over azimuthal stations (psi)
        self.line_via_avg.set_ydata(via_avg_profile) 
        self.line_vit_avg.set_ydata(vit_avg_profile)
        # Wrap data periodically for 360-degree polar contouring
        psi_plot = np.append(self.psi, 2 * np.pi)
        V_ia_plot = np.hstack([V_ia, V_ia[:, :1]])
        
        F_factors = np.mean(np.array(F_curr), axis=1) # Average over azimuthal stations (psi)
        self.line_loss.set_ydata(F_factors)

        # Dynamically Re-scale XY Cartesian Axes limits
        self.ax4.set_autoscaley_on(True)
        
        for ax in [self.ax1, self.ax1_twin, self.ax2, self.ax_foil, self.ax4, self.ax_tc]:
            ax.relim()
            ax.autoscale_view()
            
        # Update Polar Contour and Colorbar Dynamically
        for coll in self.ax3.collections:
            coll.remove()
            
        # Determine dynamic scales for colorbar bounding
        vmax = max(1e-3, np.max(V_ia))
        vmin = max(0.0, np.min(V_ia))
        levels = np.linspace(vmin, vmax, 15)

        # Draw new contour
        self.contour = self.ax3.contourf(psi_plot, self.r_m, V_ia_plot, levels=levels, cmap='viridis', extend='both')
        # Clear old quiver arrows if they exist to prevent lag
        if hasattr(self, 'quiver_overlay'):
            self.quiver_overlay.remove()

        # Unpack in-plane velocity components
        V_it = V_i[1]  # Tangential (Swirl)
        V_it_plot = np.hstack([V_it, V_it[:, :1]])
        V_ir = V_i[2]  # Radial
        V_ir_plot = np.hstack([V_ir, V_ir[:, :1]])

        # Mesh 1D radial and azimuthal grids into 2D matrices
        PSI, R = np.meshgrid(psi_plot, self.r_m)

        # Transform Cylindrical Velocity (V_ir, V_it) into Cartesian (Vx, Vy)
        U_x = V_ir_plot * np.cos(PSI) - V_it_plot * np.sin(PSI)
        U_y = V_ir_plot * np.sin(PSI) + V_it_plot * np.cos(PSI)

        # Overlay vector field using Cartesian displacement vectors
        self.quiver_overlay = self.ax3.quiver(
            PSI, 
            R, 
            U_x, 
            U_y, 
            color='white', 
            angles='uv',       # Calculates arrow directions directly from (U_x, U_y)
            scale_units='xy',  # Scales vector length relative to axes coordinates
            pivot='tail',      # Roots arrow base at the grid point
            alpha=0.7
        )

        # Clear the colorbar axis and redraw inside it cleanly
        self.cbar.ax.clear()
        self.cbar = self.fig.colorbar(self.contour, cax=self.cbar.ax)
        self.cbar.set_label("Induced Velocity [m/s]")

        return self.line_c, self.line_beta, self.line_err, self.line_via_avg, self.line_loss

def debug_diagnostics_plot(params, cond_idx, show=True):
    '''
    Generates a 2x2 diagnostic plot window for a specific flight condition.
    Displays Twist/Inflow, Lift/Drag Coefficients, Sectional Forces, and Reynolds Number.
    '''
    with plt.rc_context({
        "text.usetex": True,
        "font.family": "serif",
        "axes.labelsize": 12,
        "axes.titlesize": 14
    }):
        
        # Extract Geometry
        r = params.r * 1000               # Radius in mm
        beta_deg = np.degrees(params.beta)
        
        # Resolve Flow Kinematics
        V_i = propmodel.inflow_model(params.V_i_0, params)
        V_R, phi = airflow.resultantVelocityComponents(V_i, params)
        
        # Average across the azimuthal axis for 1D plotting
        phi_mean_deg = np.degrees(jnp.mean(phi, axis=1))
        VR_mean = jnp.mean(V_R, axis=1)
        
        # Extract Aerodynamic Coefficients
        C_L, C_D, _ = airfoil.aero_coeffs(V_i, params)
        CL_mean = jnp.mean(C_L, axis=1)
        CD_mean = jnp.mean(C_D, axis=1)
        
        # Extract Sectional Forces
        dT_ave = propmodel.dT_blade(V_i, params)
        dQ_ave = propmodel.dQ_blade(V_i, params)
        
        # Extract Flow Regime (Reynolds Number)
        Re = airflow.calc_reynolds(params.rho, jnp.maximum(VR_mean, 1e-6), params.c, params.mu)
        
        # Create the Figure
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(rf"\textbf{{Diagnostics: Flight Condition {cond_idx}}}", fontsize=16)

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

def propgeomblender(prop_names, weights):
    '''
    Weighted average of chord, beta and CST sections across several saved propellers,
    returned on the last propeller's Inputs and BladeSpec.
    '''
    props = [Propeller.load(name) for name in prop_names]
    base = props[-1]
    for name, prop in zip(prop_names, props):
        if prop.c.shape != base.c.shape or prop.cst.shape != base.cst.shape:
            raise ValueError(f"{name} has a different radial grid or number of CST stations to {prop_names[-1]}")

    weights = np.asarray(weights)
    avg = {
        k: jnp.tensordot(weights, jnp.stack([getattr(prop, k) for prop in props]), axes=1)
        for k in ("c", "beta", "cst")
    }
    return base.with_state(**avg)

if __name__ == "__main__":

    import proptimiser
    from matplotlib.animation import FuncAnimation

    # propellers/<name>.json to evaluate, blended by weight
    prop_names = ["Currawong_V1"]
    weights = np.array([1.0])
    # unit test propeller
    #prop, _, _, _ = propmodel.apc_model("apc_prop_data/propgeom_data/20x8-PERF.PE0", plot=False)
    # geometry blend of n propellers
    prop = propgeomblender(prop_names, weights)
    # prop = prop.with_inputs(Inputs())  # uncomment to evaluate at the flight conditions in init.py instead of the saved ones

    print(f"--- Evaluating {prop.inputs.n_cond} Flight Conditions ---")

    # Execution & Processing Loop
    exit_input = "n"
    while exit_input != "y":
        evaluated_conditions = []

        print("\nResolving Induced Velocity Field for all conditions...")
        for idx in range(prop.inputs.n_cond):
            # Evaluate at operating conditons
            params_cond = prop.condition(idx)
            
            # This guarantees that geometry changes (e.g. from smoothing) propagate into V_i.
            _, V_i_0_solved = propmodel.induced_velocity(params_cond)

            # write solved V_i_0 into params
            params_cond = eqx.tree_at(
                lambda p: p.V_i_0, 
                params_cond, 
                V_i_0_solved
            )
            
            evaluated_conditions.append(params_cond)
            
            # Extract and print objective metrics
            pwr, thr, _ = proptimiser.objectives(params_cond)
            print(f"Cond: {idx:<3} | Omega: {float(params_cond.omega):<6.2f} | Motor Pwr: {float(pwr):<8.4f} | Thrust: {float(thr):<8.4f}")

        # Keep the solved induced velocity as the warm start for the next pass (and for saving)
        prop = prop.with_state(V_i_0=jnp.stack([p.V_i_0 for p in evaluated_conditions]))

        # Post-Processing User Directives
        user_input = input("\nWould you like to plot solver outputs? (y/n): ")
        if user_input.lower() == "y":
            print("Plotting General Info for Condition 0...")
                    
            # Pull the first flight condition specifically to feed the live plotter
            p_cond_0 = evaluated_conditions[0]
            
            # Plotting via LivePropPlotter
            print("Launching LivePropPlotter canvas...")
            gui_queue = queue.Queue()
            plotter = LivePropPlotter(p_cond_0)
            
            # Pass the entire params object into the queue elegantly
            gui_queue.put(p_cond_0)
            
            ani = FuncAnimation(
                plotter.fig, 
                plotter.update, 
                fargs=(gui_queue,), 
                interval=200, 
                cache_frame_data=False,
                blit=False
            )
            
            print("Plotting Detailed Diagnostics for all conditions...")
            for idx, p_cond in enumerate(evaluated_conditions):
                debug_diagnostics_plot(p_cond, idx, show=False)
            
            plt.show()  # Blocks until all window instances (LivePlotter & Diagnostics) are closed

        user_input = input("\nDoes your propeller need control-point based smoothing? (y/n): ")
        if user_input.lower() == 'y':
            # Smoothing modifies the underlying geometry; conditions are rebuilt from it on the next pass
            prop = iterativesmoothing(prop)
            print("Geometry updated. Induced velocity fields will automatically refresh on the next loop pass.")

        # Save Results/Export Geometry if Required
        user_input = input('Save Propeller? (y/n): ')
        if user_input.lower() == 'y':
            propname = input('Input Prop Name: ')
            try:
                prop.save(propname)
            except FileExistsError:
                if input(f"{propname} already exists. Overwrite? (y/n): ").lower() == 'y':
                    prop.save(propname, overwrite=True)

        exit_input = input('\nExit post-processing loop completely? (y/n): ')

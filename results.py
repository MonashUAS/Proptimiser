## Results and Data Visualisation Functions

# External Imports
import dataclasses
import queue

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import jax
import jax.numpy as jnp
import equinox as eqx
from scipy.interpolate import make_interp_spline

# Internal Imports
import propmodel
import proptimiser
from init import Inputs, Propeller
import geometry
import airfoil
import airflow

class DraggableSplineEditor:
    '''
    Cubic spline through draggable control points, fitted to (r_dense, y_dense).
    Draws into ax when given (e.g. an axes embedded in a GUI), otherwise opens its own figure.
    on_change(y) is called with the new spline after every drag.
    '''
    def __init__(self, r_dense, y_dense, num_cps=6, name="Value", ax=None, on_change=None):
        self.r_dense = np.asarray(r_dense, dtype=float)
        self.y_dense = np.asarray(y_dense, dtype=float)
        self.name = name
        self.num_cps = max(4, int(num_cps))     # a cubic spline needs at least 4 points
        self.on_change = on_change

        # Setup the plot
        if ax is None:
            self.fig, self.ax = plt.subplots(figsize=(10, 6))
            self.ax.set_title(f"Interactive Editor: {name}\nDrag the red points up/down/left/right. Close window when finished.")
        else:
            self.ax, self.fig = ax, ax.figure
            self.ax.set_title(name)
        self.ax.set_xlabel("Radius (r)")
        self.ax.set_ylabel(name)
        # Compute a safe top limit with 10% padding above maximum data point
        y_max = float(np.max(self.y_dense))
        top_limit = y_max * 1.1 if y_max > 0 else 1.0

        # Explicitly set bounds (grounds bottom at 0, top at y_max + padding)
        self.ax.set_ylim(bottom=0, top=top_limit)

        # Initialize Control Points evenly spaced across the radius
        self._init_control_points()

        # Plot original data as a faint background reference
        self.ax.plot(self.r_dense, self.y_dense, '.', color='gray', alpha=0.5, label='Original Data')

        # Plot the active spline
        self._spline_line, = self.ax.plot(self.r_dense, self._get_spline(), 'b-', linewidth=2, label='Smoothed Spline')

        # Plot the draggable control points
        self._cp_scatter = self.ax.scatter(self.cp_r, self.cp_y, c='red', s=100, zorder=5, label='Control Points')

        self.ax.legend()
        self.ax.grid(True, linestyle='--', alpha=0.6)

        # State variable to track which point is currently being dragged
        self._dragged_idx = None

        # Connect Matplotlib Event Listeners
        self._cids = [
            self.fig.canvas.mpl_connect('button_press_event', self.on_press),
            self.fig.canvas.mpl_connect('button_release_event', self.on_release),
            self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion),
        ]

    def _init_control_points(self):
        self.cp_r = np.linspace(self.r_dense.min(), self.r_dense.max(), self.num_cps)
        self.cp_y = np.interp(self.cp_r, self.r_dense, self.y_dense)

    def _get_spline(self):
        # Generate the cubic spline based on current CP positions
        spline = make_interp_spline(self.cp_r, self.cp_y, k=3)
        return spline(self.r_dense)

    def _redraw(self):
        # Redraw the scatter points and the new spline
        self._cp_scatter.set_offsets(np.c_[self.cp_r, self.cp_y])
        spline = self._get_spline()
        self._spline_line.set_ydata(spline)
        self.fig.canvas.draw_idle()
        if self.on_change is not None:
            self.on_change(spline)

    def reset(self):
        '''
        Returns the control points to the original data.
        '''
        self._init_control_points()
        self._redraw()

    def disconnect(self):
        '''
        Stops listening to mouse events (call before discarding an editor on a shared canvas).
        '''
        for cid in self._cids:
            self.fig.canvas.mpl_disconnect(cid)
        self._cids = []

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

        self._redraw()

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


def _without_logs(params):
    '''
    Params (single or batched) with fixed-size solver logs, so jitted helpers are not retraced as the logs grow.
    '''
    fixed = lambda log: jnp.zeros(jnp.shape(log)[:-1] + (1,))
    return eqx.tree_at(lambda p: (p.error_log, p.time_log), params, (fixed(params.error_log), fixed(params.time_log)))

def _contour_levels(values, n=15):
    '''
    Strictly increasing contour levels spanning the finite values. A uniform field (e.g. the momentum-theory
    seed under the simple inflow model) gets a small band around its value, as contourf needs increasing levels.
    '''
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.linspace(0.0, 1.0, n)

    vmin, vmax = float(finite.min()), float(finite.max())
    if vmax - vmin < 1e-9 * max(1.0, abs(vmax)):
        pad = max(1e-3, 0.05 * abs(vmax))
        vmin, vmax = vmin - pad, vmax + pad
    return np.linspace(vmin, vmax, n)


@dataclasses.dataclass
class FlowSnapshot:
    '''
    Plain numpy data for one flight condition, ready to draw (built by plot_snapshot).
    '''
    r: np.ndarray           # radial stations (m)
    psi: np.ndarray         # azimuthal stations (rad)
    r_cst: np.ndarray       # radii of the CST sections (m)
    c: np.ndarray           # chord (m)
    beta: np.ndarray        # twist (rad)
    error_log: np.ndarray   # optimiser error history
    foil_x: np.ndarray      # closed TE -> LE -> TE section outline x/c (n_x,)
    foil_y: np.ndarray      # section outlines y/c at each r_cst station (n_cst, n_x)
    tc: np.ndarray          # max t/c at the BEMT stations (n_r,)
    tc_cst: np.ndarray      # max t/c at the r_cst stations (n_cst,)
    V_i: np.ndarray         # induced velocity (3, n_r, n_psi) [axial, tangential, radial]
    F: np.ndarray           # azimuth-averaged tip x root loss factor (n_r,)

@jax.jit
def _flow_fields(params):
    V_i = propmodel.inflow_model(params.V_i_0, params)
    _, phi = airflow.resultantVelocityComponents(V_i, params)
    tip_loss, root_loss = airflow.tip_root_loss_factors(phi, params)

    x_foil = geometry.cosine_spacing(80)
    y_upper, y_lower = geometry.kulfan_coordinates(params.cst, x_foil)
    kulfan_r = geometry.cst_at_stations(params.r, params.r_cst, params.cst)
    return dict(
        V_i=V_i,
        F=jnp.mean(tip_loss * root_loss, axis=1),
        y_upper=y_upper,
        y_lower=y_lower,
        tc=geometry.max_thickness(kulfan_r),
        tc_cst=geometry.max_thickness(params.cst),
    )

def plot_snapshot(params) -> FlowSnapshot:
    '''
    Flow and geometry data of one flight condition (Params) for LivePropPlotter.draw.
    '''
    fields = {k: np.asarray(v) for k, v in _flow_fields(_without_logs(params)).items()}

    # Closed TE -> LE -> TE outlines of the CST sections
    x_foil = np.asarray(geometry.cosine_spacing(80))
    foil_x = np.concatenate([x_foil[::-1], x_foil[1:]])
    foil_y = np.concatenate([fields["y_upper"][:, ::-1], fields["y_lower"][:, 1:]], axis=1)

    return FlowSnapshot(
        r=np.asarray(params.r),
        psi=np.asarray(params.psi),
        r_cst=np.asarray(params.r_cst),
        c=np.asarray(params.c),
        beta=np.asarray(params.beta),
        error_log=np.atleast_1d(np.asarray(params.error_log)),
        foil_x=foil_x,
        foil_y=foil_y,
        tc=fields["tc"],
        tc_cst=fields["tc_cst"],
        V_i=fields["V_i"],
        F=fields["F"],
    )


class LivePropPlotter:
    '''
    Live Updating Solver Plot with Debugging information.

    base supplies the grids (anything with r, psi and r_cst: Params or FlowSnapshot).
    Draws into fig when given (e.g. a figure embedded in a GUI), otherwise opens a pyplot figure.
    '''
    def __init__(self, base, fig=None):
        # Setup Figure
        self.fig = plt.figure(figsize=(18, 9)) if fig is None else fig
        self.r = np.array(base.r) * 1000        # Radius in mm
        self.r_m = np.array(base.r)             # Radius in m
        self.psi = np.array(base.psi)
        self.r_cst = np.array(base.r_cst)

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

        # Wrap data periodically for 360-degree polar contouring
        self.psi_plot = np.append(self.psi, 2 * np.pi)
        dummy_V_i_a = np.zeros((len(self.r), len(self.psi_plot)))

        # Subplot 4: Polar Map (Induced Velocity+swirl vectors)
        self.ax3 = self.fig.add_subplot(2, 3, 4, projection="polar")
        self.ax3.set_title("Induced Velocity Map [m/s]\n", pad=15)
        self.contour = self.ax3.contourf(self.psi_plot, self.r_m, dummy_V_i_a, levels=_contour_levels(dummy_V_i_a), cmap='viridis')
        self.cbar = self.fig.colorbar(self.contour, ax=self.ax3, pad=0.1)
        self.quiver_overlay = None

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

        self.fig.tight_layout()

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

        if latest_params is not None:
            self.draw(plot_snapshot(latest_params))
        return self.line_c, self.line_beta, self.line_err, self.line_via_avg, self.line_vit_avg, self.line_loss

    def draw(self, snap: FlowSnapshot):
        '''
        Redraws every panel from a FlowSnapshot (matplotlib only, no JAX work).
        '''
        # Update Subplot 1 Lines (Geometry)
        self.line_c.set_ydata(snap.c * 1000)
        self.line_beta.set_ydata(np.degrees(snap.beta))

        # Update Subplot 2 Line (Convergence)
        self.line_err.set_data(np.arange(len(snap.error_log)), snap.error_log)

        # Update Subplot 3 Lines (CST sections, closed TE -> LE -> TE outlines)
        for line, y in zip(self.lines_foil, snap.foil_y):
            line.set_data(snap.foil_x, y)

        # Update Subplot 6 Lines (Thickness)
        self.line_tc.set_ydata(snap.tc)
        self.points_tc.set_ydata(snap.tc_cst)

        # Update Subplot 5 Lines (Induced Velocity Profile & Losses), averaged over azimuthal stations (psi)
        V_ia, V_it, V_ir = snap.V_i
        self.line_via_avg.set_ydata(np.mean(V_ia, axis=1))
        self.line_vit_avg.set_ydata(np.mean(V_it, axis=1))
        self.line_loss.set_ydata(snap.F)

        # Dynamically Re-scale XY Cartesian Axes limits
        for ax in [self.ax1, self.ax1_twin, self.ax2, self.ax_foil, self.ax4, self.ax_tc]:
            ax.relim()
            ax.autoscale_view()

        # Update Polar Contour: wrap data periodically for 360-degree polar contouring
        wrap = lambda a: np.hstack([a, a[:, :1]])
        self.contour.remove()
        self.contour = self.ax3.contourf(self.psi_plot, self.r_m, wrap(V_ia), levels=_contour_levels(V_ia), cmap='viridis', extend='both')

        # Clear old quiver arrows if they exist to prevent lag
        if self.quiver_overlay is not None:
            self.quiver_overlay.remove()

        # Transform Cylindrical Velocity (V_ir, V_it) into Cartesian (Vx, Vy) on the 2D polar grid
        PSI, R = np.meshgrid(self.psi_plot, self.r_m)
        U_x = wrap(V_ir) * np.cos(PSI) - wrap(V_it) * np.sin(PSI)
        U_y = wrap(V_ir) * np.sin(PSI) + wrap(V_it) * np.cos(PSI)

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
        self.cbar.locator = MaxNLocator(nbins=6)     # a tick per contour level crowds small figures
        self.cbar.update_ticks()
        self.cbar.set_label("Induced Velocity [m/s]")

        self.fig.canvas.draw_idle()


## Flight Condition Evaluation
@jax.jit
def _condition_performance(params):
    '''
    Motor power, thrust, shaft power, ideal (actuator disk) power and solved V_i_0 of one flight condition.
    '''
    motor_power, thrust, V_i_0 = proptimiser.objectives(params)
    V_i = propmodel.inflow_model(V_i_0, params)
    shaft_power = propmodel.mechanicalPower(V_i, params)
    return motor_power, thrust, shaft_power, propmodel.actuator_disk_power(thrust, params), V_i_0

def evaluate_conditions(prop):
    '''
    Solves the induced velocity at every flight condition of prop.

    Returns (prop, params_list, rows): prop with the solved V_i_0 kept as its warm start, the solved Params
    of each condition, and one dict of performance figures per condition.
    '''
    solved, rows = [], []
    for idx in range(prop.inputs.n_cond):
        # Solving from the current geometry guarantees that geometry changes (e.g. from smoothing) propagate into V_i
        params = prop.condition(idx)
        motor_power, thrust, shaft_power, ideal_power, V_i_0 = (
            np.asarray(a) for a in _condition_performance(_without_logs(params))
        )
        solved.append(V_i_0)
        rows.append(dict(
            omega=float(params.omega),
            thrust=float(thrust),
            thrust_target=float(params.thrust_target),
            thrust_err_pct=100.0 * float((thrust - params.thrust_target) / params.thrust_target),
            motor_power=float(motor_power),
            shaft_power=float(shaft_power),
            ideal_ratio=float(ideal_power / shaft_power) if shaft_power > 0 else float("nan"),
        ))

    # Keep the solved induced velocity as the warm start for the next evaluation (and for saving)
    prop = prop.with_state(V_i_0=np.stack(solved))
    params_list = [prop.condition(idx) for idx in range(prop.inputs.n_cond)]
    return prop, params_list, rows

@jax.jit
def _diagnostics_fields(params):
    V_i = propmodel.inflow_model(params.V_i_0, params)
    V_R, phi = airflow.resultantVelocityComponents(V_i, params)
    C_L, C_D, _ = airfoil.aero_coeffs(V_i, params)

    VR_mean = jnp.mean(V_R, axis=1)
    phi_deg = jnp.degrees(jnp.mean(phi, axis=1))
    beta_deg = jnp.degrees(params.beta)
    return dict(
        r=params.r,
        c=params.c,
        beta_deg=beta_deg,
        phi_deg=phi_deg,
        alpha_deg=beta_deg - phi_deg,
        V_R=VR_mean,
        Mach=VR_mean / params.a,
        Re=airflow.calc_reynolds(params.rho, jnp.maximum(VR_mean, 1e-6), params.c, params.mu),
        CL=jnp.mean(C_L, axis=1),
        CD=jnp.mean(C_D, axis=1),
        dT=propmodel.dT_blade(V_i, params),
        dQ=propmodel.dQ_blade(V_i, params),
    )

def diagnostics_data(params) -> dict:
    '''
    Azimuth-averaged sectional data (numpy, one value per radial station) of one solved flight condition:
    r, c, beta_deg, phi_deg, alpha_deg, V_R, Mach, Re, CL, CD, dT (N/m), dQ (Nm/m).
    '''
    return {k: np.asarray(v) for k, v in _diagnostics_fields(_without_logs(params)).items()}

def debug_diagnostics_plot(params, cond_idx, show=True, fig=None, usetex=True, data=None):
    '''
    Generates a 2x2 diagnostic plot window for a specific flight condition.
    Displays Twist/Inflow, Lift/Drag Coefficients, Sectional Forces, and Reynolds Number.

    Draws into fig when given (cleared first); data may pass precomputed diagnostics_data(params).
    usetex=False avoids needing a LaTeX installation.
    '''
    data = diagnostics_data(params) if data is None else data

    with plt.rc_context({
        "text.usetex": usetex,
        "font.family": "serif",
        "axes.labelsize": 12,
        "axes.titlesize": 14
    }):
        # Create the Figure
        if fig is None:
            fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        else:
            fig.clear()
            axs = fig.subplots(2, 2)
        title = rf"\textbf{{Diagnostics: Flight Condition {cond_idx}}}" if usetex else f"Diagnostics: Flight Condition {cond_idx}"
        fig.suptitle(title, fontsize=16)

        r = data["r"] * 1000               # Radius in mm

        # Plot 1: Angles (Twist and Inflow)
        axs[0, 0].plot(r, data["beta_deg"], 'o-', color="tab:red", label=r"Twist ($\beta$)")
        axs[0, 0].plot(r, data["phi_deg"], 's-', color="tab:cyan", label=r"Inflow Angle ($\phi$)")
        axs[0, 0].fill_between(r, data["beta_deg"], data["phi_deg"], color='gray', alpha=0.2, label=r"Approx AoA ($\alpha$)")
        axs[0, 0].set_title(r"Angle Distribution")
        axs[0, 0].set_ylabel(r"Angle ($^\circ$)")
        axs[0, 0].grid(True, linestyle='--', alpha=0.6)
        axs[0, 0].legend()

        # Plot 2: Aerodynamic Coefficients
        axs[0, 1].plot(r, data["CL"], 'o-', color="tab:blue", label=r"$C_L$")
        axs[0, 1].set_title(r"Sectional Aerodynamic Coefficients")
        axs[0, 1].set_ylabel(r"$C_L$", color="tab:blue")
        axs[0, 1].tick_params(axis='y', labelcolor="tab:blue")
        axs[0, 1].grid(True, linestyle='--', alpha=0.6)

        ax01_twin = axs[0, 1].twinx()
        ax01_twin.plot(r, data["CD"], 's-', color="tab:orange", label=r"$C_D$")
        ax01_twin.set_ylabel(r"$C_D$", color="tab:orange")
        ax01_twin.tick_params(axis='y', labelcolor="tab:orange")

        lines_coeff = axs[0, 1].get_lines() + ax01_twin.get_lines()
        axs[0, 1].legend(lines_coeff, [l.get_label() for l in lines_coeff], loc="center right")

        # Plot 3: Sectional Forces (Thrust & Torque)
        axs[1, 0].plot(r, data["dT"], 'o-', color="tab:green", label=r"Thrust ($dT$)")
        axs[1, 0].set_title(r"Sectional Load Distribution")
        axs[1, 0].set_ylabel(r"Thrust (N/m)", color="tab:green")
        axs[1, 0].set_xlabel(r"Radius ($r$) [mm]")
        axs[1, 0].tick_params(axis='y', labelcolor="tab:green")
        axs[1, 0].grid(True, linestyle='--', alpha=0.6)

        ax10_twin = axs[1, 0].twinx()
        ax10_twin.plot(r, data["dQ"], 's-', color="purple", label=r"Torque ($dQ$)")
        ax10_twin.set_ylabel(r"Torque (Nm/m)", color="purple")
        ax10_twin.tick_params(axis='y', labelcolor="purple")

        lines_forces = axs[1, 0].get_lines() + ax10_twin.get_lines()
        axs[1, 0].legend(lines_forces, [l.get_label() for l in lines_forces], loc="center right")

        # Plot 4: Reynolds Number
        axs[1, 1].plot(r, data["Re"], 'o-', color="black", label=r"Reynolds Number")
        axs[1, 1].set_title(r"Local Flow Regime")
        axs[1, 1].set_ylabel(r"$Re$")
        axs[1, 1].set_xlabel(r"Radius ($r$) [mm]")
        axs[1, 1].set_yscale('log') # Log scale is better for Re
        axs[1, 1].grid(True, linestyle='--', alpha=0.6)

        fig.tight_layout()
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
        print("\nResolving Induced Velocity Field for all conditions...")
        # The solved induced velocity is kept as the warm start for the next pass (and for saving)
        prop, evaluated_conditions, rows = evaluate_conditions(prop)
        for idx, row in enumerate(rows):
            print(f"Cond: {idx:<3} | Omega: {row['omega']:<6.2f} | Motor Pwr: {row['motor_power']:<8.4f} | "
                  f"Thrust: {row['thrust']:<8.4f} | T-Err: {row['thrust_err_pct']:+.2f}%")

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

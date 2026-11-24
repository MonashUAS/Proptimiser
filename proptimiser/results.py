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
from . import propmodel
from . import optimiser
from .init import Inputs, Propeller
from . import geometry
from . import airfoil
from . import airflow
from . import structure

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
        V_i=airflow.to_grid(V_i, params),
        F=jnp.mean(airflow.to_grid(tip_loss * root_loss, params), axis=1),
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
        self.line_err, = self.ax2.plot([], [], 'orange', marker='o', label="Relative change")
        self.ax2.set_title("Solver Convergence History")
        self.ax2.set_xlabel("Iteration")
        self.ax2.set_ylabel("Relative objective change")
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
    motor_power, thrust, V_i_0 = optimiser.objectives(params)
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

    grid = lambda x: airflow.to_grid(x, params)
    VR_mean = jnp.mean(grid(V_R), axis=1)
    phi_deg = jnp.degrees(jnp.mean(grid(phi), axis=1))
    beta_deg = jnp.degrees(params.beta)
    flow = dict(
        phi_deg=phi_deg,
        alpha_deg=beta_deg - phi_deg,
        V_R=VR_mean,
        Mach=VR_mean / params.a,
        Re=airflow.calc_reynolds(params.rho, jnp.maximum(VR_mean, 1e-6), params.c, params.mu),
        CL=jnp.mean(grid(C_L), axis=1),
        CD=jnp.mean(grid(C_D), axis=1),
        dT=propmodel.dT_blade(V_i, params),
        dQ=propmodel.dQ_blade(V_i, params),
    )
    # The blend zone is not solved (propmodel.aero_zone): its flow quantities are NaN, which plots leave as a gap
    aero = propmodel.aero_zone(params)
    return dict(r=params.r, c=params.c, beta_deg=beta_deg, **{k: jnp.where(aero, v, jnp.nan) for k, v in flow.items()})

def diagnostics_data(params) -> dict:
    '''
    Azimuth-averaged sectional data (numpy, one value per radial station) of one solved flight condition:
    r, c, beta_deg, phi_deg, alpha_deg, V_R, Mach, Re, CL, CD, dT (N/m), dQ (Nm/m).
    The flow quantities are NaN in the blend zone, which the BEMT does not solve.
    '''
    return {k: np.asarray(v) for k, v in _diagnostics_fields(_without_logs(params)).items()}

## Structural loads
@jax.jit
def _structural_fields(params, mat, kin):
    result = structure.structural_loads(params, mat, kin)
    stress = structure.stress_fields(result, mat)
    props, loads, hub = result.props, result.resultants, result.hub
    blade, rotor = structure.blade_mass_properties(params, mat), structure.propeller_mass_properties(params, mat)
    return dict(
        r=params.r, psi=params.psi, c=params.c, beta_deg=jnp.degrees(params.beta), aero=propmodel.aero_zone(params),
        mass_per_length=props.mass.total * result.axes.arc,
        area=props.area, I_xx=props.I_xx, I_zz=props.I_zz, I_xz=props.I_xz, J=props.J, t_max=props.t_max,
        x_c=props.x_c, z_c=props.z_c,
        N=loads.N, V_x=loads.V_x, V_z=loads.V_z, M_x=loads.M_x, M_z=loads.M_z, T=loads.T,
        T_aero=result.T_aero, T_inertial=result.T_inertial,
        sigma_max=stress.sigma_max, tau=stress.tau, sigma_vm=stress.sigma_vm, safety_factor=stress.safety_factor,
        core_sigma_max=stress.core_sigma_max, core_sigma_vm=stress.core_sigma_vm,
        blade_mass=blade.mass, blade_cg=blade.cg, blade_inertia_cg=blade.inertia_cg,
        propeller_mass=rotor.mass, propeller_inertia=rotor.inertia,
        atm=result.pitch_torque_aero, ctm=result.pitch_torque_inertial, hold=result.pitch_torque_hold,
        hub_force=hub.force, hub_moment=hub.moment,
        hub_force_amplitude=hub.force_amplitude, hub_moment_amplitude=hub.moment_amplitude,
        N_blades=params.N[0],
    )

def structural_data(params, mat, kin=None) -> dict:
    '''
    Loads along the blade and at the hub of one solved flight condition (numpy), from structure.structural_loads and
    structure.stress_fields. params is a Params whose V_i_0 is solved (evaluate_conditions), mat a structure.Layup
    (Propeller.material()) and kin an optional structure.Kinematics.

    The dict holds, per station (n_r,): r, c, beta_deg, aero (aerodynamic zone mask), the section properties (area,
    I_xx, I_zz, I_xz, J, t_max, x_c, z_c) and mass_per_length; per azimuth and station (n_psi, n_r), SI units: the
    internal loads N, V_x, V_z, M_x, M_z, T, and T_aero and T_inertial (the parts of T), the stresses sigma_max, tau,
    sigma_vm and safety_factor; per azimuth (n_psi,): the root pitch torques atm, ctm and hold; and the hub loads
    hub_force, hub_moment (n_psi, 3) with their harmonics hub_force_amplitude and hub_moment_amplitude; the mass
    properties blade_mass, blade_cg, blade_inertia_cg, propeller_mass and propeller_inertia (structure.MassProps, without
    hub hardware), with a 'summary' dict of the figures structural_summary reports. Blade loads are those of one blade
    at root azimuth psi.
    '''
    fields = {k: np.asarray(v) for k, v in _structural_fields(_without_logs(params), mat, kin).items()}
    psi = fields["psi"]
    n_blades = int(round(float(fields.pop("N_blades"))))

    # Largest loads on the blade root (station 0) over the azimuth, and the weakest section (the tip carries no load)
    peak = lambda a: float(np.max(np.abs(a)))
    sf = fields["safety_factor"][:, :-1]
    psi_i, r_i = np.unravel_index(np.argmin(sf), sf.shape)
    hub_force, hub_moment = fields["hub_force"], fields["hub_moment"]
    amplitude = fields["hub_force_amplitude"]
    fields["summary"] = dict(
        n_blades=n_blades, r_root=float(fields["r"][0]),
        root_N=float(np.max(fields["N"][:, 0])), root_V_x=peak(fields["V_x"][:, 0]), root_V_z=peak(fields["V_z"][:, 0]),
        root_M_x=peak(fields["M_x"][:, 0]), root_M_z=peak(fields["M_z"][:, 0]), root_T=peak(fields["T"][:, 0]),
        atm_mean=float(fields["atm"].mean()), ctm_mean=float(fields["ctm"].mean()), hold_mean=float(fields["hold"].mean()),
        atm_peak=peak(fields["atm"]), ctm_peak=peak(fields["ctm"]), hold_peak=peak(fields["hold"]),
        safety_factor_min=float(sf[psi_i, r_i]), safety_factor_r=float(fields["r"][r_i]),
        safety_factor_psi=float(np.degrees(psi[psi_i])), sigma_vm_max=float(fields["sigma_vm"].max()),
        sigma_allow=float(mat.outer.sigma_allow), tau_allow=float(mat.outer.tau_allow),
        skin_thickness=float(mat.skin_thickness), core_sigma_vm_max=float(fields["core_sigma_vm"].max()),
        blade_mass=float(fields["blade_mass"]), blade_cg_r=float(fields["blade_cg"][0]),
        propeller_mass=float(fields["propeller_mass"]), propeller_I_spin=float(fields["propeller_inertia"][2, 2]),
        hub_thrust=float(hub_force[:, 2].mean()), hub_torque=float(-hub_moment[:, 2].mean()),
        hub_side_force=float(np.max(np.hypot(hub_force[:, 0], hub_force[:, 1]))),
        hub_moment_xy=float(np.max(np.hypot(hub_moment[:, 0], hub_moment[:, 1]))),
        hub_thrust_amplitude=float(amplitude[n_blades, 2]) if n_blades < amplitude.shape[0] else float("nan"),
    )
    return fields

def structural_summary(data, cond_idx=None) -> str:
    '''
    Plain-text report of structural_data: the blade-root loads, the pitch-change torques of a constant-speed unit, the
    weakest section, the hub loads and the mass properties. Blade-root values are the largest over the azimuth, in N, N m
    and MPa.
    '''
    s = data["summary"]
    title = "Structural loads" + ("" if cond_idx is None else f", flight condition {cond_idx}")
    return "\n".join([
        f"{title} ({s['n_blades']} blades, rigid blade, "
        + (f"{1000 * s['skin_thickness']:.2f} mm skin over a core, largest core von Mises {s['core_sigma_vm_max'] / 1e6:.1f} MPa)"
           if s["skin_thickness"] > 0.0 else "one material section)"),
        f"  Blade root (r = {1000 * s['r_root']:.1f} mm): retention force {s['root_N']:.1f} N | shear x/z {s['root_V_x']:.1f} / {s['root_V_z']:.1f} N | "
        f"flap moment {s['root_M_x']:.3f} N m | edge moment {s['root_M_z']:.3f} N m | torque {s['root_T']:.3f} N m",
        f"  Pitch-change torque (about the blade axis): aerodynamic {s['atm_mean']:+.3f} N m | centrifugal {s['ctm_mean']:+.3f} N m | "
        f"hold {s['hold_mean']:+.3f} N m (mean), peak hold {s['hold_peak']:.3f} N m",
        f"  Weakest section: safety factor {s['safety_factor_min']:.2f} at r = {1000 * s['safety_factor_r']:.1f} mm, "
        f"azimuth {s['safety_factor_psi']:.0f} deg (von Mises {s['sigma_vm_max'] / 1e6:.1f} MPa, allowable {s['sigma_allow'] / 1e6:.1f} MPa)",
        f"  Hub: thrust {s['hub_thrust']:.2f} N | shaft torque {s['hub_torque']:.3f} N m | peak side force {s['hub_side_force']:.2f} N | "
        f"peak moment about x/y {s['hub_moment_xy']:.3f} N m | thrust at {s['n_blades']}/rev {s['hub_thrust_amplitude']:.3f} N",
        f"  Mass: blade {1000 * s['blade_mass']:.1f} g with its CG at r = {1000 * s['blade_cg_r']:.1f} mm | propeller {1000 * s['propeller_mass']:.1f} g | "
        f"propeller spin inertia {s['propeller_I_spin']:.3e} kg m^2 (no hub hardware)",
    ])

def structural_peak(values, psi_index=None):
    '''
    One value per station from a (n_psi, n_r) array of blade loads: the entry of the chosen azimuth, or with
    psi_index None the value of largest magnitude over the azimuth (sign kept).
    '''
    values = np.asarray(values)
    if psi_index is not None:
        return values[psi_index]
    return values[np.argmax(np.abs(values), axis=0), np.arange(values.shape[1])]

def structural_plot(data, cond_idx=None, show=True, fig=None, psi_index=None):
    '''
    Six panels of structural_data: axial and shear force, bending moments, torque about the span, von Mises stress
    with the safety factor, hub loads against azimuth, and section properties. Blade loads show the mean over the
    azimuth with the band of their range, or the single azimuth psi_index. The blend zone is shaded.
    Draws into fig when given (cleared first).
    '''
    fig = plt.figure(figsize=(16, 9)) if fig is None else fig
    fig.clear()
    axs = fig.subplots(2, 3)
    fig.suptitle("Structural loads" + ("" if cond_idx is None else f": flight condition {cond_idx}"), fontsize=14)

    r = data["r"] * 1000
    blend = ~data["aero"]
    azimuth = np.degrees(data["psi"])
    mode = "mean and range over the azimuth" if psi_index is None else f"azimuth {azimuth[psi_index]:.0f} deg"

    def trace(ax, key, label, color, scale=1.0, style="-"):
        a = np.asarray(data[key]) * scale
        if psi_index is None:
            ax.fill_between(r, a.min(axis=0), a.max(axis=0), color=color, alpha=0.2)
            ax.plot(r, a.mean(axis=0), style, color=color, label=label)
        else:
            ax.plot(r, a[psi_index], style, color=color, label=label)

    def finish(ax, title, ylabel, xlabel="Radius [mm]", shade=True):
        if shade and blend.any():
            ax.axvspan(r[0], r[blend][-1], color="0.9", zorder=0)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        ax.grid(True, linestyle="--", alpha=0.5)

    def twin(ax, ylabel, color):
        other = ax.twinx()
        other.set_ylabel(ylabel, color=color)
        other.tick_params(axis="y", labelcolor=color)
        return other

    def legend(*axes):
        lines = [line for ax in axes for line in ax.get_lines()]
        axes[0].legend(lines, [line.get_label() for line in lines], fontsize=8, loc="best")

    # Axial force (tension) and shear forces
    trace(axs[0, 0], "N", "Axial force N", "tab:blue")
    shear = twin(axs[0, 0], "Shear [N]", "tab:orange")
    trace(shear, "V_x", "Shear along the chord", "tab:orange")
    trace(shear, "V_z", "Shear normal to the chord", "tab:green")
    finish(axs[0, 0], "Axial and shear force", "Axial force [N]")
    legend(axs[0, 0], shear)

    # Bending moments
    trace(axs[0, 1], "M_x", "Flap moment (about the chord)", "tab:red")
    trace(axs[0, 1], "M_z", "Edgewise moment", "tab:purple")
    finish(axs[0, 1], "Bending moments", "Moment [N m]")
    axs[0, 1].legend(fontsize=8)

    # Torque about the span
    trace(axs[0, 2], "T", "Total", "black")
    trace(axs[0, 2], "T_aero", "Aerodynamic", "tab:cyan", style="--")
    trace(axs[0, 2], "T_inertial", "Centrifugal and inertial", "tab:olive", style="--")
    finish(axs[0, 2], "Torque about the span", "Torque [N m]")
    axs[0, 2].legend(fontsize=8)

    # Von Mises stress against the allowable, and the safety factor
    s = data["summary"]
    trace(axs[1, 0], "sigma_vm", "Von Mises stress", "tab:red", scale=1e-6)
    axs[1, 0].axhline(s["sigma_allow"] / 1e6, color="tab:red", linestyle=":", label="Allowable normal stress")
    factor = twin(axs[1, 0], "Safety factor", "tab:blue")
    safety = data["safety_factor"].min(axis=0) if psi_index is None else data["safety_factor"][psi_index]
    safety = np.where(safety > 1e3, np.nan, safety)             # the unloaded tip
    factor.plot(r, safety, "-", color="tab:blue", label="Safety factor (lowest over the azimuth)" if psi_index is None else "Safety factor")
    factor.set_ylim(0, min(20.0, float(np.nanmax(safety)) * 1.1) if np.isfinite(safety).any() else 1.0)
    finish(axs[1, 0], f"Stress ({mode})", "Stress [MPa]")
    legend(axs[1, 0], factor)

    # Hub loads of all blades against the azimuth of the first blade
    forces = {"Force x": 0, "Force y": 1, "Thrust (force z)": 2}
    for (label, k), color in zip(forces.items(), ("tab:orange", "tab:green", "tab:blue")):
        axs[1, 1].plot(azimuth, data["hub_force"][:, k], color=color, label=label)
    moments = twin(axs[1, 1], "Moment [N m]", "tab:red")
    for (label, k), style in zip({"Moment x": 0, "Moment y": 1, "Moment z": 2}.items(), ("--", "-.", ":")):
        moments.plot(azimuth, data["hub_moment"][:, k], style, color="tab:red", label=label)
    finish(axs[1, 1], "Hub loads", "Force [N]", xlabel="Azimuth of the first blade [deg]", shade=False)
    legend(axs[1, 1], moments)

    # Section properties
    axs[1, 2].plot(r, data["mass_per_length"] * 1000, color="tab:brown", label="Mass per length [g/m]")
    stiffness = twin(axs[1, 2], "Second moment [mm^4]", "tab:gray")
    stiffness.plot(r, data["I_xx"] * 1e12, color="tab:gray", label="I_xx (flap)")
    stiffness.plot(r, data["I_zz"] * 1e12, color="tab:gray", linestyle="--", label="I_zz (edgewise)")
    stiffness.set_yscale("log")
    finish(axs[1, 2], "Section properties", "Mass per length [g/m]")
    legend(axs[1, 2], stiffness)

    fig.tight_layout()
    if show:
        plt.show(block=False)
    return fig

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

    # designs/<name>.yaml (with propellers/<name>.state.json) to evaluate, blended by weight
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
        material = prop.material()
        for idx, params in enumerate(evaluated_conditions):
            print(structural_summary(structural_data(params, material), idx))

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

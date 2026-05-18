## Results and Data Visualisation Functions


# External Imports
import numpy as np
import matplotlib.pyplot as plt


# Internal Imports
import propmodel
import init


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

def loadplotprop(propfilename):
    '''
    load and plot a propeller from a saved file
    load_state assumes propellers are contained in the path /propellers
    '''
    params = init.Params.load_state(propfilename)

    propinfoplot(params)


if "__name__" == "__main__":
    loadplotprop("params_state")
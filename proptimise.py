import jax
import jax.numpy as jnp
import numpy as np
import jaxopt
import matplotlib.pyplot as plt
from dataclasses import replace
from init import Params, Inputs
import propmodel
import airfoil
import pandas as pd

from Foil_Interpolators import RationalFunctionInterp, RationalFunction
from airflow import resultantVelocityComponents, tip_root_loss_factors
import airflow as afl


# Set jax debugging
jax.config.update("jax_traceback_filtering", "quiet_remove_frames")
jax.config.update("jax_check_tracer_leaks", "off")
jax.config.update("jax_disable_jit", False)


def minimumInducedLossCost(V_ia_0):
    """
    Returns the cost function of minimum induced loss, that is the RMS of the 
    variation of the induced velocity at the center of the propeller by each annulus
    """

    mean_V_ia = jnp.mean(V_ia_0)

    squared_err = (mean_V_ia - V_ia_0)**2

    rms_err = jnp.sqrt(jnp.sum(squared_err))

    return rms_err


#Returns the power, thrust error and smoothness penalty
def base_objective_single(c, beta, params_static, thrust_target):
    """
    Evaluates propeller performance for ONE flight condition.
    thrust_target is passed explicitly so hover/cruise can differ.
    """
    params = replace(params_static, c=c, beta=beta)
    V_ia, V_ia_0 = propmodel.induced_velocity(params)

    power  = jnp.squeeze(propmodel.mechanicalPower(V_ia, params))
    thrust = jnp.squeeze(propmodel.thrust(V_ia, params))

    thrust_err   = thrust - thrust_target
    rms_MIL_err  = minimumInducedLossCost(V_ia_0)

    return power, thrust_err, thrust, rms_MIL_err, V_ia_0


#Scaled Aug. lagrangian, returns the actual Lagrangian 
def objective_fn(
    opt_vars_scaled,
    params_hover, params_cruise,
    c_scale, beta_scale,
    scale_MIL_err=0.3,
    scale_THR_err=15.0,
    scale_POW_cost=0.065,
    w_hover=0.5,
    w_cruise=0.5,
):
    c_scaled, beta_scaled = opt_vars_scaled
    c = c_scaled * c_scale
    beta = beta_scaled * beta_scale

    #Calculate the cost for both hover and cruise conditions. 
    # Hover cost
    pow_h, thr_err_h, _, mil_h, _ = base_objective_single(
        c, beta, params_hover, Inputs.thrust_hover
    )
    cost_h = jnp.sqrt(
        (thr_err_h*scale_THR_err) ** 2
        + (mil_h*scale_MIL_err) ** 2
        + (pow_h*scale_POW_cost) ** 2
    )

    # Cruise cost
    pow_c, thr_err_c, _, mil_c, _ = base_objective_single(
        c, beta, params_cruise, Inputs.thrust_cruise
    )

    cost_c = jnp.sqrt(
        (thr_err_c*scale_THR_err) ** 2
        + (mil_c*scale_MIL_err) ** 2
        + (pow_c*scale_POW_cost) ** 2
    )

    # Weighted combination of cruise/hover
    return w_hover * cost_h + w_cruise * cost_c


#Gradient scaling (compute c and beta scales)
# TODO: FInish implementation of this function or deprecate it
def compute_gradient_scales(init_c, init_beta, params_static):
    c_scale    = 1
    beta_scale = 20

    return c_scale, beta_scale


def debugPlotting(V_ia_0_new, params):
    # Unpack relevant props from params
    r, c, beta = params.r, params.c, params.beta

    # Calculate the new thrust per annulus for blade and momentum equations
    V_ia = propmodel.inflow_model(V_ia_0_new, params)
    dT_bld = propmodel.dT_blade(V_ia, params, debug=False)[0]
    dT_mom = propmodel.dT_momentum(V_ia, params, debug=False)
    dQ_bld = propmodel.dQ_blade(V_ia, params, debug=False)

    # Tip and Root Loss Factors
    tip_loss, root_loss = tip_root_loss_factors(V_ia, params)

    # Calculate the total prandtl loss correction factor
    F = jnp.multiply(tip_loss, root_loss) 

    # Calculate the average value of F across each annulus
    F_means_annuli = jnp.mean(F, axis=1)

    # calculate aerodynamic incidence
    _, phi = resultantVelocityComponents(V_ia, params) # get only second output

    # load relevant data from inputs
    geometric_angle = beta[:, None]

    # Angle of attack
    AoA_eff = geometric_angle - phi

    # Calculate the average value of V_ia across each annulus
    V_ia_means = jnp.mean(V_ia, axis=1)

    fig, axes = plt.subplots(3, 2, figsize=(4, 4))

    axes[0][0].plot(r, c)
    axes[0][0].stem(r, c)
    axes[0][0].set_ylabel("Chord (m)")
    axes[0][0].set_xlabel("r (m)")
    axes[0][0].set_title("Propeller chord dist.")

    axes[0][1].plot(r, jnp.degrees(beta), 'b', label="Twist angle")
    axes[0][1].plot(r, jnp.degrees(AoA_eff), 'r', label="Aerodynamic angle")
    axes[0][1].set_ylabel("Angle (deg)")
    axes[0][1].set_xlabel("r (m)")
    axes[0][1].set_title("Propeller twist (blue) and AoA dist. (red)")

    axes[1][0].plot(r, dT_mom, label="Momentum")
    axes[1][0].plot(r, dT_bld, '--', label="BEMT")
    axes[1][0].set_ylabel("Local thrust (N/m)")
    axes[1][0].set_xlabel("r (m)")
    axes[1][0].legend()
    axes[1][0].set_title("Propeller annulus thrust dist.")

    axes[1][1].plot(r, dQ_bld)
    axes[1][1].set_ylabel("Local torque (Nm/m)")
    axes[1][1].set_xlabel("r (m)")
    axes[1][1].set_title("Propeller annulus torque dist.")

    axes[2][0].plot(r, F_means_annuli)
    axes[2][0].set_ylabel("Local loss factor F")
    axes[2][0].set_xlabel("r (m)")
    axes[2][0].set_title("Propeller annulus average Prandtl factor (F)")

    axes[2][1].plot(r, V_ia_means)
    axes[2][1].set_ylabel("Local induced axial velocity (m/s)")
    axes[2][1].set_xlabel("r (m)")
    axes[2][1].set_title("Propeller annulus average axial induced velocity")

    plt.tight_layout()
    plt.show()


# Optimisation
def run_optimization():
    jax.config.update("jax_enable_x64", True)

    psi, dpsi = propmodel.angular_stops(Inputs.n_psi)
    r,   dr   = propmodel.radial_stops(
        Inputs.tip_diameter/2, Inputs.hub_diameter/2, Inputs.n_r
    )

    #Initialise chord and beta with linearly decreasing profiles
    init_c    = jnp.linspace(0.030, 0.015, Inputs.n_r)
    init_beta = jnp.linspace(jnp.radians(15), jnp.radians(5), Inputs.n_r)

    input_filename = "compiledData-ARAD10.txt"

    P_c    = Inputs.Ord_Numerator_c
    Q_c    = Inputs.Ord_Denominator_c
    P_beta = Inputs.Ord_Denominator_beta
    Q_beta = Inputs.Ord_Numerator_beta


    # Initial Re/Ma guesses per condition (will be updated each iteration)
    Re_hover = 150000 * jnp.ones(Inputs.n_r)
    Ma_hover = 0.10   * jnp.ones(Inputs.n_r)

    Re_cruise = 150000 * jnp.ones(Inputs.n_r)
    Ma_cruise = 0.15   * jnp.ones(Inputs.n_r) #In cruise expected to be slightly higher

    cl_h, cd_h = airfoil.loadAirfoilData(input_filename, Re_hover,  Ma_hover,  Inputs.n_r)
    cl_c, cd_c = airfoil.loadAirfoilData(input_filename, Re_cruise, Ma_cruise, Inputs.n_r)

    # Both conditions share the same propeller geoemtry, 
    _common = dict(
        N=2, R=jnp.max(r),
        c=init_c[:, None], beta=init_beta[:, None],
        r=r[:, None], dr=dr,
        psi=psi[None, :], dpsi=dpsi,
    )

    # Create params object for both cruise and hover. 
    params_hover = Params(
        **_common,
        V_x=jnp.array([Inputs.V_x_hover]),
        V_yz=jnp.array([Inputs.V_yz_hover]),
        omega=jnp.array([Inputs.omega_hover]),
        V_ia_0=jnp.full((Inputs.n_r,), 4.0),
        clInterpolator=cl_h, cdInterpolator=cd_h,
    )

    params_cruise = Params(
        **_common,
        V_x=jnp.array([Inputs.V_x_cruise]),
        V_yz=jnp.array([Inputs.V_yz_cruise]),
        omega=jnp.array([Inputs.omega_cruise]),
        V_ia_0=jnp.full((Inputs.n_r,), 2.0),  # cruise has lower inflow perturbation
        clInterpolator=cl_c, cdInterpolator=cd_c,
    )

    c_scale, beta_scale = compute_gradient_scales(init_c, init_beta, params_hover)

    vars_scaled = (init_c / c_scale, init_beta / beta_scale)

    bounds = (
        (
            jnp.full_like(params_hover.c, Inputs.min_chord / c_scale),
            jnp.full_like(params_hover.beta, jnp.radians(Inputs.min_twist) / beta_scale),
        ),
        (
            jnp.full_like(params_hover.c, Inputs.max_chord / c_scale),
            jnp.full_like(params_hover.beta, jnp.radians(Inputs.max_twist) / beta_scale),
        ),
    )

    solver = jaxopt.ScipyBoundedMinimize(
        fun=objective_fn, method="L-BFGS-B",
        tol=Inputs.opt_tol, maxiter=Inputs.opt_max_iter
    )

    global_best_cost = jnp.inf
    global_best_c = init_c
    global_best_beta = init_beta

    total_iter = (
        Inputs.Iterations_initialising
        + Inputs.Iterations_blending
        + Inputs.Iterations_solving
    )

    for i in range(total_iter):
        print(f"\n--> Running solver (iter {i})")

        sol = solver.run(
            vars_scaled,
            bounds=bounds,
            params_hover=params_hover,
            params_cruise=params_cruise,
            c_scale=c_scale,
            beta_scale=beta_scale,
            scale_MIL_err=0.25,
            scale_THR_err=4.5,
            w_hover=Inputs.w_hover,
            w_cruise=Inputs.w_cruise,
        )

        vars_scaled = sol.params
        c    = vars_scaled[0] * c_scale
        beta = vars_scaled[1] * beta_scale

        # Smooth geometry 
        c_fit    = RationalFunctionInterp(r, jnp.squeeze(c),    P_c,    Q_c,    method='trf')
        beta_fit = RationalFunctionInterp(r, jnp.squeeze(beta), P_beta, Q_beta, method='trf')

        c    = RationalFunction(r, list(c_fit[0:P_c]),       list(c_fit[P_c:]))
        beta = RationalFunction(r, list(beta_fit[0:P_beta]), list(beta_fit[P_beta:]))


        for label, params_static, Re_list, Ma_list in [
            ("hover",  params_hover,  Re_hover,  Ma_hover),
            ("cruise", params_cruise, Re_cruise, Ma_cruise),
        ]:
            params_warm = replace(params_static, c=jnp.squeeze(c), beta=jnp.squeeze(beta))
            V_ia_conv, V_ia_0_new = propmodel.induced_velocity(params_warm)

            # Re/Ma update
            V_R, _ = resultantVelocityComponents(V_ia_conv, params_warm)
            V_R_aved = jnp.mean(V_R, axis=1)
            Re_new = params_static.rho * c * V_R_aved / params_static.mu
            Ma_new = V_R_aved / params_static.v_sound

            cl_interp = params_static.clInterpolator
            cd_interp = params_static.cdInterpolator

            for rr in range(Inputs.n_r):
                re_drift = jnp.abs((Re_new[rr] - Re_list[rr]) / Re_list[rr])
                ma_drift = jnp.abs(Ma_new[rr] - Ma_list[rr])
                if re_drift > 0.1 or ma_drift > 0.05:
                    print(f"  [{label}] Regenerating interpolator at r={rr+1}/{Inputs.n_r}")
                    Re_list = Re_list.at[rr].set(Re_new[rr])
                    Ma_list = Ma_list.at[rr].set(Ma_new[rr])
                    cl_p, cd_p = airfoil.loadAirfoilDataSingle(
                        input_filename, Re_list[rr], Ma_list[rr]
                    )
                    cl_interp = cl_interp.at[rr].set(cl_p)
                    cd_interp = cd_interp.at[rr].set(cd_p)

            # Write updated state back into the correct params object
            updated = replace(
                params_static,
                c=jnp.squeeze(c), beta=jnp.squeeze(beta),
                V_ia_0=V_ia_0_new,
                clInterpolator=cl_interp,
                cdInterpolator=cd_interp,
            )
            if label == "hover":
                params_hover = updated
                Re_hover = Re_list
                Ma_hover = Ma_list
            else:
                params_cruise = updated
                Re_cruise = Re_list
                Ma_cruise = Ma_list

        # Blending
        if Inputs.Iterations_initialising <= i <= Inputs.Iterations_initialising + Inputs.Iterations_blending:
            blend = (i - Inputs.Iterations_initialising) / Inputs.Iterations_blending
            params_hover  = replace(params_hover,  blendingParam=blend)
            params_cruise = replace(params_cruise, blendingParam=blend)

        pow_h, thr_err_h, thr_h, mil_h, _ = base_objective_single(
            c, beta, params_hover, Inputs.thrust_hover
        )
        pow_c, thr_err_c, thr_c, mil_c, _ = base_objective_single(
            c, beta, params_cruise, Inputs.thrust_cruise
        )
        # Calculate cost to be displayed, ensure scale factors are actually what's being used (consolidate in future!)
        current_cost = objective_fn(
            (c / c_scale, beta / beta_scale),
            params_hover, params_cruise,
            c_scale, beta_scale,
            scale_MIL_err=0.25, scale_THR_err=1.5,
            w_hover=Inputs.w_hover, w_cruise=Inputs.w_cruise,
        )

        print(f"\n{'Iter':<5} | {'Cond':<8} | {'Power':>10} | {'Thrust':>10} | {'Thr Err':>10} | {'MIL rms':>10}")
        print("-" * 65)
        print(f"{i:<5} | {'hover':<8} | {float(pow_h):>10.3f} | {float(thr_h):>10.3f} | {float(thr_err_h):>10.4f} | {float(mil_h):>10.3f}")
        print(f"{i:<5} | {'cruise':<8} | {float(pow_c):>10.3f} | {float(thr_c):>10.3f} | {float(thr_err_c):>10.4f} | {float(mil_c):>10.3f}")
        print(f"       Combined cost: {float(current_cost):.5f}  |  Blending: {params_hover.blendingParam:.2f}")

        vars_scaled = (c / c_scale, beta / beta_scale)

        if current_cost <= global_best_cost:
            global_best_cost = current_cost
            global_best_c    = c
            global_best_beta = beta

        # Converge only when BOTH thrust errors are within tolerance and ensure full CFD data is being used.
        both_converged = (
            abs(thr_err_h) < Inputs.eta_tol
            and abs(thr_err_c) < Inputs.eta_tol
            and params_hover.blendingParam == 1.0
        )
        if both_converged:
            print(f"\nConverged at iteration {i}")
            break

    else:
        print(f"\nWarning: max iterations reached.")

    return (
        replace(params_hover,  c=global_best_c, beta=global_best_beta),
        replace(params_cruise, c=global_best_c, beta=global_best_beta),
    )

#Run + plotting (two plots, one for each condition)
if __name__ == "__main__":
    params_hover_opt, params_cruise_opt = run_optimization()

    print("\n--- Hover ---")
    V_ia_h, V_ia_0_h = propmodel.induced_velocity(params_hover_opt)
    debugPlotting(V_ia_0_h, params_hover_opt)

    print("\n---Cruise ---")
    V_ia_c, V_ia_0_c = propmodel.induced_velocity(params_cruise_opt)
    debugPlotting(V_ia_0_c, params_cruise_opt)
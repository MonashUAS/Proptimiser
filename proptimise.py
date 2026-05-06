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

#Curvature penalty 
def curvature_penalty(x):
    d2 = x[2:] - 2.0 * x[1:-1] + x[:-2]
    return jnp.mean(d2**2)

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
def base_objective(c, beta, params_static):

    params = replace(params_static, c=c, beta=beta)
    V_ia, V_ia_0   = propmodel.induced_velocity(params)

    power  = jnp.squeeze(propmodel.mechanicalPower(V_ia, params))
    thrust = jnp.squeeze(propmodel.thrust(V_ia, params))

    thrust_err = thrust - Inputs.thrust

    rms_MIL_err = minimumInducedLossCost(V_ia_0)

    return power, thrust_err, thrust, rms_MIL_err


#Scaled Aug. lagrangian, returns the actual Lagrangian 
def objective_fn(opt_vars_scaled, params_static, c_scale, beta_scale, scale_MIL_err=0.3, scale_THR_err=15.0, scale_POW_cost=0.0005):
    c_scaled, beta_scaled = opt_vars_scaled
    
    c = c_scaled*c_scale
    beta = beta_scaled*beta_scale

    power, thrust_err, _, rms_MIL_err = base_objective(c, beta, params_static)
    #print(power)

    rms_MIL_err = rms_MIL_err * scale_MIL_err
    thrust_err = thrust_err * scale_THR_err
    power_cost = power*scale_POW_cost

    return jnp.sqrt(thrust_err**2 + rms_MIL_err**2 + power_cost**2)


#Gradient scaling (compute c and beta scales)
def compute_gradient_scales(init_c, init_beta, params_static):
    c_scale    = 500
    beta_scale = 1

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
    jax.config.update("jax_enable_x64", True) # Use 64 Bit Precision

    # Geometry setup
    psi, dpsi = propmodel.angular_stops(Inputs.n_psi)
    r, dr = propmodel.radial_stops(0.225, 0.05, Inputs.n_r)

    init_c = jnp.full(Inputs.n_r, 0.020)
    # Linearly decreasing profile to start
    init_beta = jnp.linspace(jnp.radians(15), jnp.radians(5), Inputs.n_r)
    
    print("--> Initialising")

    # Define the input tables
    input_filename = "compiledData-ARAD10.txt"

    # Generate the reynolds and mach lists
    Re_list = 150000 * jnp.ones(jnp.size(r))
    Ma_list = 0.15 * jnp.ones(jnp.size(r))
    Re_list_new = 150000 * jnp.ones(jnp.size(r))
    Ma_list_new = 0.15 * jnp.ones(jnp.size(r))

    # Generate the interpolators
    cl_interpolator, cd_interpolator = airfoil.loadAirfoilData(input_filename, Re_list, Ma_list, Inputs.n_r)

    params_static = Params(
        N=2,
        R=jnp.max(r),
        c=init_c[:, None],
        beta=init_beta[:,None],
        r=r[:, None], dr=dr,
        psi=psi[None, :], dpsi=dpsi,
        V_x=jnp.array([0]), V_yz=jnp.array([0]),
        omega=jnp.array([8000/60*2*3.14159]), # rad/s
        V_ia_0 = jnp.full((Inputs.n_r,), jnp.array([4.0])),
        cdInterpolator=cd_interpolator,
        clInterpolator=cl_interpolator
    )

    print("--> Initialisation complete!")

    P_c = Inputs.Ord_Numerator_c
    Q_c = Inputs.Ord_Denominator_c

    P_beta = Inputs.Ord_Denominator_beta
    Q_beta = Inputs.Ord_Numerator_beta

    Smoothing_log = Inputs.Smoothing_log


    #Calculate the gradient scales
    c_scale, beta_scale = compute_gradient_scales(init_c, init_beta, params_static)

    # Scale initial variables
    vars_scaled = (
        init_c/c_scale,
        init_beta/beta_scale
    )

    #Bounds (chord + beta pairs) (scaled)
    bounds = (
        (
            jnp.full_like(params_static.c, 0.005/c_scale),
            jnp.full_like(params_static.beta, jnp.radians(0) / beta_scale),
        ),
        (
            jnp.full_like(params_static.c, 0.05/c_scale),
            jnp.full_like(params_static.beta, jnp.radians(45) / beta_scale),
        )
    )

    #Solver
    solver = jaxopt.ScipyBoundedMinimize(fun=objective_fn, method="L-BFGS-B", tol=1e-4, maxiter=1000)

    #Augmented Lagrangian params
    lmbda = 0
    mu = 1
    eta_tol = 1e-2

    #Initialise the global best cost variable
    global_best_cost = jnp.inf
    global_best_c = init_c
    global_best_beta = init_beta
    
    total_iter = Inputs.n_iter_parametric + Inputs.n_iter_blended + Inputs.n_iter_lookup

    for i in range(total_iter):

        print("--> Running solver")

        sol = solver.run(
            vars_scaled,
            bounds=bounds,
            params_static=params_static,
            c_scale=c_scale,
            beta_scale=beta_scale,
            scale_MIL_err=0.25,
            scale_THR_err=1.5
        )
        
        print("--> Solver complete!")
        
        vars_scaled = sol.params

        #Unscale C and beta
        c = vars_scaled[0] * c_scale
        beta = vars_scaled[1] * beta_scale

        # Use the foil interpolators to generate rational functions for the chord and beta distribution
        # Refer to https://www.itl.nist.gov/div898/handbook/pmd/section8/pmd812b.htm for choosing P and Q
        c_fit_params = RationalFunctionInterp(r, jnp.squeeze(c), P_c, Q_c, verbose=Smoothing_log, extendedChecking=True, method='trf')
        beta_fit_params = RationalFunctionInterp(r, jnp.squeeze(beta), P_beta, Q_beta, verbose=Smoothing_log, extendedChecking=True, method='trf')

        # Use the fit params to calculate smoothed beta and chord distributions
        c_smoothed = RationalFunction(r, list(c_fit_params[0:P_c]), list(c_fit_params[P_c:len(c_fit_params)]))
        beta_smoothed = RationalFunction(r, list(beta_fit_params[0:P_beta]), list(beta_fit_params[P_beta:len(beta_fit_params)]))

        c = c_smoothed
        beta = beta_smoothed

        params_warm = replace(params_static, c=jnp.squeeze(c), beta=jnp.squeeze(beta))

        V_ia_converged, V_ia_0_new = propmodel.induced_velocity(params_warm)


        print("--> Regenerating interpolators")

        # Generate the new reynolds and mach lists
        V_R, _ = resultantVelocityComponents(V_ia_converged, params_warm)
        V_R_aved = jnp.mean(V_R, axis=1)
        Re_list_new = params_static.rho * c * V_R_aved / params_static.mu
        Ma_list_new = V_R_aved / params_static.v_sound

        if (i >= Inputs.switchingDelay) and (i <= Inputs.switchingDelay + Inputs.switchingTime):
            params_static = replace(params_static, blendingParam = (i - Inputs.switchingDelay)/Inputs.switchingTime)

        # Generate the interpolators
        for rr in range(Inputs.n_r):
            if (jnp.abs((Re_list_new[rr]-Re_list[rr])/Re_list[rr])>0.1) or (jnp.abs((Ma_list_new[rr]-Ma_list[rr])/Ma_list[rr])>0.1):
                print("Regenerating interpolator at radial position " + str(rr) + " of " + str(Inputs.n_r))

                Re_list = Re_list.at[rr].set(Re_list_new[rr])
                Ma_list = Ma_list.at[rr].set(Ma_list_new[rr])

                # check if the mach or reynolds has changed more than 10%
                cl_params, cd_params = airfoil.loadAirfoilDataSingle(input_filename, Re_list[rr], Ma_list[rr])
                
                # store the data in the interpolator arrays
                cl_interpolator = cl_interpolator.at[rr].set(cl_params)
                cd_interpolator = cd_interpolator.at[rr].set(cd_params)

        params_static = replace(params_static, clInterpolator=cl_interpolator, cdInterpolator=cd_interpolator)

        print("--> Interpolators complete!")


        params_static = replace(params_static, V_ia_0=V_ia_0_new)

        power, thrust_err, thrust, rms_MIL_err = base_objective(c, beta, params_static)

        #The cost as calculated by the solver (sol.state.fun_val) doesnt capture the smoothed geometry. 
        current_smoothed_cost = objective_fn(
            (c / c_scale, beta / beta_scale), 
            params_static, 
            c_scale, 
            beta_scale, 
            scale_MIL_err=0.25, 
            scale_THR_err=1.5
        )

        print(f"{'Iter':<6} | {'Power':<12} | {'Thrust (N)':<12} | {'Thrust Err':<12} | {'MIL rms':<10} | {'Lambda':<12} | {'Blending Parameter':<10}")
        print("-" * 90)
        print(f"{i:<6} | {float(power):<12.4f} | {float(thrust):<12.4f} | {float(thrust_err):<12.4f} | {rms_MIL_err:<10.1f} | {current_smoothed_cost:<12.4f} | {params_static.blendingParam:<10.1f}")

        vars_scaled = (c/c_scale,beta_smoothed/beta_scale)

        #Check if current (smoothed) cost is better than global.
        if current_smoothed_cost <= global_best_cost:
            global_best_cost = current_smoothed_cost
            global_best_c = c
            global_best_beta = beta

        # Convergence, don't want to prematurely return a solution
        if abs(thrust_err) < eta_tol and params_static.blendingParam == 1:
            print(f"\nConverged at iteration {i} with thrust error {thrust_err:.2e}")
            break

        #AL update

        if Inputs.interIterationDebug:
            debugPlotting(V_ia_0_new, params_warm)
        
    

    else:
        print(f"\nWarning: max iterations reached. Final thrust error: {thrust_err:.2e}")

    return replace(params_static, c=global_best_c, beta=global_best_beta)

#Run + plotting
if __name__ == "__main__":
    params_opt = run_optimization()

    V_ia_converged, V_ia_0_new = propmodel.induced_velocity(params_opt)
    debugPlotting(V_ia_0_new, params_opt)
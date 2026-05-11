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
def base_objective(c, beta, params_static):

    params = replace(params_static, c=c, beta=beta)
    V_ia, V_ia_0   = propmodel.induced_velocity(params)

    power  = jnp.squeeze(propmodel.mechanicalPower(V_ia, params))
    thrust = jnp.squeeze(propmodel.thrust(V_ia, params))

    thrust_err = thrust - params. thrust_target

    rms_MIL_err = minimumInducedLossCost(V_ia_0)

    return power, thrust_err, thrust, rms_MIL_err


#Scaled Aug. lagrangian, returns the actual Lagrangian 
def objective_fn(opt_vars_scaled, 
                 params_hover,
                 params_cruise, 
                 params_max_thrust,
                 c_scale, 
                 beta_scale, 
                 scale_MIL_err=0.3, 
                 scale_THR_err=5.0, 
                 scale_POW_cost=0.05,
                 scale_hover_thrust=25.0,
                 scale_cruise_power=0.05,
                 scale_max_thrust=2.5,
                 scale_hover_power=0.05,
                 scale_cruise_thrust = 25.0
                 ):
    
    # Variable scaling to increase stability of optimiser
    c_scaled, beta_scaled = opt_vars_scaled
    c = c_scaled*c_scale
    beta = beta_scaled*beta_scale

    #Separate penalties for each of the conditions 
    power_hover,thrust_err_hover,_,rms_MIL_hover  = base_objective(c, beta, params_hover)           
    power_cruise,thrust_err_cruise,_, rms_MIL_cruise  = base_objective(c, beta, params_cruise)
    _,thrust_err_max_thrust,_,rms_MIL_max     = base_objective(c, beta, params_max_thrust)

    # Cost for each of the three conditions (hover, cruise and max_thrust)
    hover_cost  = (thrust_err_hover * scale_hover_thrust)**2 + (power_hover * scale_hover_power)**2 
    cruise_cost  = (thrust_err_cruise * scale_cruise_thrust)**2 + (scale_cruise_power * power_cruise)**2 
    max_thrust_cost = jnp.where(thrust_err_max_thrust >= 0, 0.0, (thrust_err_max_thrust * scale_max_thrust)**2)
    MIL_cost = ((rms_MIL_hover + rms_MIL_cruise) * scale_MIL_err) ** 2

    return jnp.sqrt(hover_cost + cruise_cost + max_thrust_cost + MIL_cost)


#Gradient scaling (compute c and beta scales)
# TODO: FInish implementation of this function or deprecate it
def compute_gradient_scales(init_c, init_beta, params_static):
    c_scale    = 1.0
    beta_scale = 20.0

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
    r, dr = propmodel.radial_stops(Inputs.tip_diameter/2, Inputs.hub_diameter/2, Inputs.n_r)

    init_c = jnp.linspace(0.045, 0.015, Inputs.n_r)
    # Linearly decreasing profile to start
    init_beta = jnp.linspace(jnp.radians(15), jnp.radians(5), Inputs.n_r)
    
    print("\n--> Initialising")

    # Define the input tables
    input_filename = "compiledData-ARAD10.txt"

    # Generate the reynolds and mach lists
    #Re_list = 150000 * jnp.ones(jnp.size(r))
    #Ma_list = 0.15 * jnp.ones(jnp.size(r))
    #Re_list_new = 150000 * jnp.ones(jnp.size(r))
    #Ma_list_new = 0.15 * jnp.ones(jnp.size(r))

    Re_list_hover  = Re_list_cruise  = Re_list_max  = 150000 * jnp.ones(jnp.size(r))
    Ma_list_hover  = Ma_list_cruise  = Ma_list_max  = 0.15   * jnp.ones(jnp.size(r))

    # Generate the interpolators
    #cl_interpolator, cd_interpolator = airfoil.loadAirfoilData(input_filename, Re_list, Ma_list, Inputs.n_r)

    cl_interp_hover, cd_interp_hover   = airfoil.loadAirfoilData(input_filename, Re_list_hover,  Ma_list_hover,  Inputs.n_r)
    cl_interp_cruise, cd_interp_cruise = airfoil.loadAirfoilData(input_filename, Re_list_cruise, Ma_list_cruise, Inputs.n_r)
    cl_interp_max, cd_interp_max       = airfoil.loadAirfoilData(input_filename, Re_list_max,    Ma_list_max,    Inputs.n_r)

    # Treat as hover condition
    params_static = Params(  
    N=2,
    R=jnp.max(r),
    c=init_c[:, None],
    beta=init_beta[:,None],
    r=r[:, None], dr=dr,
    psi=psi[None, :], dpsi=dpsi,
    V_x=jnp.array([0.0]),                          
    V_yz=jnp.array([0]),
    omega=jnp.array([Inputs.omega_hover*2*jnp.pi/60]),
    V_ia_0=jnp.full((Inputs.n_r,), jnp.array([4.0])),
    cdInterpolator=cd_interp_hover,
    clInterpolator=cl_interp_hover,
    thrust_target=Inputs.thrust_hover,
    )   

    # Set up cruise and max_thrust
    params_cruise = replace(params_static, V_x=jnp.array([Inputs.V_cruise]),omega=jnp.array([Inputs.omega_cruise*2*jnp.pi/60]),thrust_target=Inputs.thrust_cruise)
    params_max_thrust = replace(params_static, omega=jnp.array([Inputs.omega_max*2*jnp.pi/60]), thrust_target=Inputs.thrust_max)
    
    print("\t--> Initialisation complete!")

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
            jnp.full_like(params_static.c, Inputs.min_chord/c_scale),
            jnp.full_like(params_static.beta, jnp.radians(Inputs.min_twist) / beta_scale),
        ),
        (
            jnp.full_like(params_static.c, Inputs.max_chord/c_scale),
            jnp.full_like(params_static.beta, jnp.radians(Inputs.max_twist) / beta_scale),
        )
    )

    #Solver
    solver = jaxopt.ScipyBoundedMinimize(fun=objective_fn, method="L-BFGS-B", tol=Inputs.opt_tol, maxiter=Inputs.opt_max_iter)

    #Initialise the global best cost variable
    global_best_cost = jnp.inf
    global_best_c = init_c
    global_best_beta = init_beta
    
    total_iter = Inputs.Iterations_initialising + Inputs.Iterations_blending + Inputs.Iterations_solving

    for i in range(total_iter):

        print("\n--> Running solver")

        sol = solver.run(
            vars_scaled,
            bounds=bounds,
            params_hover=params_static,       
            params_cruise=params_cruise,      
            params_max_thrust=params_max_thrust, 
            c_scale=c_scale,
            beta_scale=beta_scale,
            scale_MIL_err=0.25,
            scale_THR_err=1.5
        )
        
        print("\t--> Solver complete!")
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

        params_warm_hover = replace(params_static, c=jnp.squeeze(c), beta=jnp.squeeze(beta))
        params_warm_cruise = replace(params_cruise, c=jnp.squeeze(c), beta=jnp.squeeze(beta))
        params_warm_max = replace(params_max_thrust, c=jnp.squeeze(c), beta=jnp.squeeze(beta))

        V_ia_hover, V_ia_0_hover = propmodel.induced_velocity(params_warm_hover)
        V_ia_cruise, V_ia_0_cruise = propmodel.induced_velocity(params_warm_cruise)
        V_ia_max, V_ia_0_max = propmodel.induced_velocity(params_warm_max)

        ## New stuff
        #Generate resultant velocity components for each of the conditions
        V_R_hover,  _ = resultantVelocityComponents(V_ia_hover, params_warm_hover)
        V_R_cruise, _ = resultantVelocityComponents(V_ia_cruise, params_warm_cruise)
        V_R_max,    _ = resultantVelocityComponents(V_ia_max, params_warm_max)

        #New Re and Mach lists
        Re_list_new_hover  = params_static.rho * c * jnp.mean(V_R_hover,  axis=1) / params_static.mu
        Re_list_new_cruise = params_static.rho * c * jnp.mean(V_R_cruise, axis=1) / params_static.mu
        Re_list_new_max    = params_static.rho * c * jnp.mean(V_R_max,    axis=1) / params_static.mu

        Ma_list_new_hover  = jnp.mean(V_R_hover,  axis=1) / params_static.v_sound
        Ma_list_new_cruise = jnp.mean(V_R_cruise, axis=1) / params_static.v_sound
        Ma_list_new_max    = jnp.mean(V_R_max,    axis=1) / params_static.v_sound

        print("\n--> Regenerating interpolators")

        interp_conditions = [
        ("Hover",      Re_list_new_hover,  Ma_list_new_hover,  Re_list_hover,  Ma_list_hover,  cl_interp_hover,  cd_interp_hover),
        ("Cruise",     Re_list_new_cruise, Ma_list_new_cruise, Re_list_cruise, Ma_list_cruise, cl_interp_cruise, cd_interp_cruise),
        ("Max Thrust", Re_list_new_max,    Ma_list_new_max,    Re_list_max,    Ma_list_max,    cl_interp_max,    cd_interp_max),
    ]

        updated = {}
        for name, Re_new, Ma_new, Re_curr, Ma_curr, cl_interp, cd_interp in interp_conditions:
            for rr in range(Inputs.n_r):
                if (jnp.abs((Re_new[rr]-Re_curr[rr])/Re_curr[rr]) > 0.1) or (jnp.abs(Ma_new[rr]-Ma_curr[rr]) > 0.05):
                    print(f"  [{name}] Regenerating interpolator at radial position {rr+1} of {Inputs.n_r}")
                    Re_curr = Re_curr.at[rr].set(Re_new[rr])
                    Ma_curr = Ma_curr.at[rr].set(Ma_new[rr])
                    cl_params, cd_params = airfoil.loadAirfoilDataSingle(input_filename, Re_curr[rr], Ma_curr[rr])
                    cl_interp = cl_interp.at[rr].set(cl_params)
                    cd_interp = cd_interp.at[rr].set(cd_params)
            updated[name] = (Re_curr, Ma_curr, cl_interp, cd_interp)

        Re_list_hover,  Ma_list_hover,  cl_interp_hover,  cd_interp_hover  = updated["Hover"]
        Re_list_cruise, Ma_list_cruise, cl_interp_cruise, cd_interp_cruise = updated["Cruise"]
        Re_list_max,    Ma_list_max,    cl_interp_max,    cd_interp_max    = updated["Max Thrust"]

        print("\t--> Interpolators complete!")

        params_static = replace(params_static, clInterpolator=cl_interp_hover, cdInterpolator=cd_interp_hover)

        params_cruise = replace(params_cruise,
                                c=jnp.squeeze(c), 
                                beta=jnp.squeeze(beta),
                                blendingParam=params_static.blendingParam,
                                V_ia_0=V_ia_0_cruise,
                                clInterpolator=cl_interp_cruise,
                                cdInterpolator=cd_interp_cruise)
                                
        params_max_thrust = replace(params_max_thrust, 
                                    c=jnp.squeeze(c), 
                                    beta=jnp.squeeze(beta),
                                    blendingParam=params_static.blendingParam,
                                    V_ia_0=V_ia_0_max,
                                    clInterpolator=cl_interp_max,
                                    cdInterpolator=cd_interp_max)

        params_static = replace(params_static, V_ia_0=V_ia_0_hover)
        
        power_hover, thrust_err_hover, thrust_hover, rms_MIL_err = base_objective(c, beta, params_static)
        power_cruise, thrust_err_cruise, thrust_cruise, _ = base_objective(c, beta, params_cruise)
        _, thrust_err_max, thrust_max, _       = base_objective(c, beta, params_max_thrust)

        #The cost as calculated by the solver (sol.state.fun_val) doesnt capture the smoothed geometry. 
        current_smoothed_cost = objective_fn(
            (c / c_scale, beta / beta_scale), 
            params_hover=params_static, 
            params_cruise=params_cruise, 
            params_max_thrust=params_max_thrust,
            c_scale=c_scale, 
            beta_scale=beta_scale, 
            scale_MIL_err=0.25, 
            scale_THR_err=1.5
        )
        print("\n")
    
        print(f"{'Iter':<6} | {'Power (Hover)':<13} | {'Power (Cruise)':<13} | {'Hvr Thr':<10} {'Hvr Err':<11} | {'Crs Thr':<10} {'Crs Err':<11} | {'Max Thr':<10} {'Max Err':<11} | {'Cost':<12}")
        print("-" * 110)
        print(f"{i:<6} | {float(power_hover):<13.4f} | {float(power_cruise):<13.4f} | {float(thrust_hover):<10.4f} {float(thrust_err_hover):<+11.4f} | {float(thrust_cruise):<10.4f} {float(thrust_err_cruise):<+11.4f} | {float(thrust_max):<10.4f} {float(thrust_err_max):<+11.4f} | {float(sol.state.fun_val):<12.6f}")

        print("\n")

        vars_scaled = (c/c_scale,beta_smoothed/beta_scale)

        #Check if current (smoothed) cost is better than global.
        if current_smoothed_cost <= global_best_cost:
            global_best_cost = current_smoothed_cost
            global_best_c = c
            global_best_beta = beta
            global_best_iter = i

            #Retain solution errors
            global_thrust_err_hover = thrust_err_hover
            global_thrust_err_cruise = thrust_err_cruise
            global_thrust_err_max = thrust_err_max
            global_power_cruise = power_cruise
            global_power_hover = power_hover
        # Convergence, don't want to prematurely return a solution

        if (abs(thrust_err_hover) < Inputs.eta_tol and
            thrust_err_cruise < Inputs.eta_tol and
            thrust_err_max > -Inputs.eta_tol and
            params_static.blendingParam == 1):
            
            print(f"\nConverged at iteration {i}: hover err: {thrust_err_hover:.2e}, cruise err: {thrust_err_cruise:.2e}, max thr err: {thrust_err_max:.2e}")
            break

        if Inputs.interIterationDebug:
            debugPlotting(V_ia_0_new, params_warm_hover)
        
    else:
        print(f"\nWarning: max iterations reached. Using solution from iteration {i}")
        print(f"\nBest solution at iteration {global_best_iter}: Power (cruise): {global_power_cruise:.2e}\nPower (hover) {global_power_hover:.2e}\nThrust error (cruise): {global_thrust_err_cruise:.2e}\nThrust error (hover): {global_thrust_err_hover:.2e}")
        
    return replace(params_static, c=global_best_c, beta=global_best_beta)

#Run + plotting
if __name__ == "__main__":
    params_opt = run_optimization()

    V_ia_converged, V_ia_0_new = propmodel.induced_velocity(params_opt)
    debugPlotting(V_ia_0_new, params_opt)
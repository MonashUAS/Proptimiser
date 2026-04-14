import jax
import jax.numpy as jnp
import numpy as np
import jaxopt
import matplotlib.pyplot as plt
from dataclasses import replace
from init import Params, Inputs
import propmodel

from Foil_Interpolators import RationalFunctionInterp, RationalFunction


# Set jax debugging
jax.config.update("jax_traceback_filtering", "quiet_remove_frames")
jax.config.update("jax_check_tracer_leaks", "off")
jax.config.update("jax_disable_jit", False)

#Curvature penalty 
def curvature_penalty(x):
    d2 = x[2:] - 2.0 * x[1:-1] + x[:-2]
    return jnp.mean(d2**2)

def minimumInducedLossCost(V_ia_0, scale=1):
    """
    Returns the cost function of minimum induced loss, that is the RMS of the 
    variation of the induced velocity at the center of the propeller by each annulus
    """

    mean_V_ia = jnp.mean(V_ia_0)

    squared_err = (mean_V_ia - V_ia_0)**2

    rms_err = jnp.sqrt(jnp.sum(squared_err))

    return rms_err*scale

#Returns the power, thrust error and smoothness penalty
def base_objective(c, beta, params_static, milScale=1, thrScale=1):

    params = replace(params_static, c=c, beta=beta)
    V_ia, V_ia_0   = propmodel.induced_velocity(params)

    power  = jnp.squeeze(propmodel.mechanicalPower(V_ia, params))
    thrust = jnp.squeeze(propmodel.thrust(V_ia, params))

    thrust_err = thrust - Inputs.thrust

    rms_MIL_err = minimumInducedLossCost(V_ia_0, scale=milScale)

    return power, thrust_err*thrScale, thrust, rms_MIL_err


#Scaled Aug. lagrangian, returns the actual Lagrangian 
def objective_fn(opt_vars_scaled, params_static, c_scale, beta_scale, scale_MIL_err=1, scale_THR_err=1):
    c_scaled, beta_scaled = opt_vars_scaled
    
    c = c_scaled*c_scale
    beta = beta_scaled*beta_scale

    power, thrust_err, _, rms_MIL_err = base_objective(c, beta, params_static, scale_MIL_err, scale_THR_err)
    #print(power)

    return jnp.sqrt(thrust_err**2 + rms_MIL_err**2)



#Gradient scaling (compute c and beta scales)
def compute_gradient_scales(init_c, init_beta, params_static):
    c_scale    = 1
    beta_scale = 1

    return c_scale, beta_scale


# Optimisation
def run_optimization():
    jax.config.update("jax_enable_x64", True) # Use 64 Bit Precision

    # Geometry setup
    psi, dpsi = propmodel.angular_stops(Inputs.n_psi)
    r, dr = propmodel.radial_stops(0.225, 0.05, Inputs.n_r)

    init_c = jnp.full(Inputs.n_r, 0.010)
    init_beta = jnp.full(Inputs.n_r, jnp.radians(2))

    params_static = Params(
        N=2,
        R=jnp.max(r),
        c=init_c[:, None],
        beta=init_beta[:,None],
        r=r[:, None], dr=dr,
        psi=psi[None, :], dpsi=dpsi,
        V_x=jnp.array([0]), V_yz=jnp.array([0]),
        omega=jnp.array([840]), # rad/s
        V_ia_0 = jnp.full((Inputs.n_r,), jnp.array([4.0]))
    )
    
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
            jnp.full_like(params_static.beta, jnp.radians(30) / beta_scale),
        )
    )

    #Solver
    solver = jaxopt.ScipyBoundedMinimize(fun=objective_fn, method="L-BFGS-B", tol=1e-8, maxiter=1000)

    #Augmented Lagrangian params
    lmbda = 0
    mu = 1
    eta_tol = 1e-3

    #Set arbtitrarily high to initiate solver
    print(f"{'Iter':<6} | {'Power':<12} | {'Thrust (N)':<12} | {'Thrust Err':<12} | {'MIL rms':<10} | {'Lambda':<12} | {'Mu':<10}")
    print("-" * 117)

    for i in range(50):

        sol = solver.run(
            vars_scaled,
            bounds=bounds,
            params_static=params_static,
            c_scale=c_scale,
            beta_scale=beta_scale,
            scale_MIL_err=1,
            scale_THR_err=1
        )

        vars_scaled = sol.params
        #Unscale
        c = vars_scaled[0] * c_scale
        beta = vars_scaled[1] * beta_scale

        params_warm = replace(params_static, c=jnp.squeeze(c), beta=jnp.squeeze(beta))

        V_ia_converged, V_ia_0_new = propmodel.induced_velocity(params_warm)
        params_static = replace(params_static, V_ia_0=V_ia_0_new)
        power, thrust_err, thrust, rms_MIL_err = base_objective(c, beta, params_static)

        print(f"{i:<6} | {float(power):<12.4f} | {float(thrust):<12.4f} | {float(thrust_err):<12.4f} | {rms_MIL_err:<10.1f} | {lmbda:<12.4f} | {mu:<10.1f}")

    #TODO Add over-relaxation param for chord and beta using last chord and beta and new chord and beta guesses, add this after the new chord and beta are estimated and before the smoothing occurs
        # The dirtiest, jankiest, most intrusive fix ever part 1:
        # Record the shape of the c_final and beta_final arrays
        c_shape = c_final.shape
        beta_shape = beta_final.shape

        # Use the foil interpolators to generate rational functions for the chord and beta distribution
        # Refer to https://www.itl.nist.gov/div898/handbook/pmd/section8/pmd812b.htm for choosing P and Q
        c_fit_params = RationalFunctionInterp(r, jnp.squeeze(c_final), P_c, Q_c, verbose=Smoothing_log, extendedChecking=True, method='dogbox')
        beta_fit_params = RationalFunctionInterp(r, jnp.squeeze(beta_final), P_beta, Q_beta, verbose=Smoothing_log, extendedChecking=True, method='trf')

        # Use the fit params to calculate smoothed beta and chord distributions
        c_final = RationalFunction(r, list(c_fit_params[0:P_c]), list(c_fit_params[P_c:len(c_fit_params)]))
        beta_final = RationalFunction(r, list(beta_fit_params[0:P_beta]), list(beta_fit_params[P_beta:len(beta_fit_params)]))

        # The dirtiest, jankiest, most intrusive fix ever part 2:
        # Enforce the recorded shape of the c_final and beta_final arrays
        c_final = jnp.reshape(c_final, c_shape)
        beta_final = jnp.reshape(beta_final, beta_shape)
        # Convergence
        if abs(thrust_err) < eta_tol:
            print(f"\nConverged at iteration {i} with thrust error {thrust_err:.2e}")
            break

        #AL update

        prev_err = abs(thrust_err)

        """
        # Calculate the new thrust per annulus for blade and momentum equations
        V_ia = propmodel.inflow_model(V_ia_0_new, params_warm)
        dT_bld = propmodel.dT_blade(V_ia, params_warm, debug=False)[0]
        dT_mom = propmodel.dT_momentum(V_ia, params_warm, debug=False)[0]
        dQ_bld = propmodel.dQ_blade(V_ia, params_warm, debug=False)[0]

        fig, axes = plt.subplots(2, 2, figsize=(4, 4))

        axes[0][0].plot(r, c)
        axes[0][0].stem(r, c)
        axes[0][0].set_ylabel("Chord (m)")
        axes[0][0].set_xlabel("r (m)")

        axes[0][1].plot(r, jnp.degrees(beta))
        axes[0][1].stem(r, jnp.degrees(beta))
        axes[0][1].set_ylabel("Twist (deg)")
        axes[0][1].set_xlabel("r (m)")

        axes[1][0].plot(r, dT_mom, label="Momentum")
        axes[1][0].plot(r, dT_bld, '--', label="BEMT")
        axes[1][0].set_ylabel("Local thrust (N/m)")
        axes[1][0].set_xlabel("r (m)")
        axes[1][0].legend()
        
        axes[1][1].plot(r, dQ_bld)
        axes[1][1].set_ylabel("Local torque (Nm/m)")
        axes[1][1].set_xlabel("r (m)")

        plt.tight_layout()
        plt.show()
        """
        

    else:
        print(f"\nWarning: max iterations reached. Final thrust error: {thrust_err:.2e}")
        
    return replace(params_static, c=c, beta=beta)


#Run + plotting
if __name__ == "__main__":
    params_opt = run_optimization()

    r = params_opt.r * 1000 # get r in mm
    c = params_opt.c * 1000 # get c in mm
    beta = params_opt.beta

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(r, c)
    axes[0].stem(r, c)
    axes[0].set_title("Chord (mm)")
    axes[0].set_xlabel("r (mm)")

    axes[1].plot(r, jnp.degrees(beta))
    axes[1].stem(r, jnp.degrees(beta))
    axes[1].set_title("Twist (deg)")
    axes[0].set_xlabel("r (mm)")

    plt.tight_layout()
    plt.show()
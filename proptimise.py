import jax
import jax.numpy as jnp
import numpy as np
import jaxopt
import matplotlib.pyplot as plt
from dataclasses import replace
from init import Params, Inputs
import propmodel


# Set jax debugging
jax.config.update("jax_traceback_filtering", "quiet_remove_frames")
jax.config.update("jax_check_tracer_leaks", "on")
jax.config.update("jax_disable_jit", True)

#Curvature penalty 
def curvature_penalty(x):
    d2 = x[2:] - 2.0 * x[1:-1] + x[:-2]
    return jnp.mean(d2**2)


#Returns the power, thrust error and smoothness penalty
def base_objective(c, beta, params_static):
    params = replace(params_static, c=c, beta=beta)
    V_ia   = propmodel.induced_velocity(params)

    power  = jnp.squeeze(propmodel.mechanicalPower(V_ia, params))
    thrust = jnp.squeeze(propmodel.thrust(V_ia, params))

    thrust_err = thrust - Inputs.thrust

    #torque = jnp.squeeze(propmodel.torque(V_ia, params))

    smooth_penalty = (
        Inputs.lambda_beta_smooth  * curvature_penalty(beta[:,0]) +
        Inputs.lambda_chord_smooth * curvature_penalty(c[:,0])
    )

    return power, thrust_err, smooth_penalty


#Scaled Aug. lagrangian, returns the actual Lagrangian 
def objective_fn(opt_vars_scaled, params_static, c_scale, beta_scale, lmbda, mu):
    c_scaled, beta_scaled = opt_vars_scaled

    c = c_scaled*c_scale
    beta = beta_scaled*beta_scale

    power, thrust_err, smooth_penalty = base_objective(c, beta, params_static)

    return (
        power
        + lmbda * thrust_err
        + 0.5 * mu * thrust_err**2
        + smooth_penalty
    )


#Gradient scaling (compute c and beta scales)
def compute_gradient_scales(init_c, init_beta, params_static):
    c_scale    = 1000
    beta_scale = 1

    return c_scale, beta_scale


# Optimisation
def run_optimization():
    jax.config.update("jax_enable_x64", True)

    # Geometry setup
    psi, dpsi = propmodel.angular_stops(Inputs.n_psi)
    r, dr = propmodel.radial_stops(0.225, 0.05, Inputs.n_r)

    init_c = jnp.full((Inputs.n_r, 1), 0.02)
    init_beta = jnp.full((Inputs.n_r, 1), jnp.radians(0))

    params_static = Params(
        N=2,
        R=jnp.array([0.35]),
        c=init_c,
        beta=init_beta,
        r=r[:, None], dr=dr,
        psi=psi[None, :], dpsi=dpsi,
        V_x=jnp.array([10]), V_yz=jnp.array([2]),
        omega=jnp.array([2*jnp.pi*8000/60]),
        V_ia_0=jnp.full_like(r[:, None], jnp.array([14.0])) # make sure to enter a floating point number (eg 14.0)
                                # Tiling and shape adjustment has been moved to inflow model calcs
    )

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
            jnp.full_like(params_static.c, 0.001/c_scale),
            jnp.full_like(params_static.beta, jnp.radians(2) / beta_scale),
        ),
        (
            jnp.full_like(params_static.c, 0.05/c_scale),
            jnp.full_like(params_static.beta, jnp.radians(70) / beta_scale),
        )
    )
    #Solver
    solver = jaxopt.ScipyBoundedMinimize(fun=objective_fn,method="L-BFGS-B",tol=1e-8,maxiter=1000
    )

    #Augmented Lagrangian params
    lmbda = 0
    mu = 1
    mu_max = 1e7
    mu_scale = 2
    eta_tol = 1e-3

    #Set arbtitrarily high to initiate solver 
    prev_err = 1e12
    print(f"{'Iter':<6} | {'Power':<12} | {'Thrust Err':<12} | {'Lambda':<12} | {'Mu':<10}")
    print("-" * 65)

    for i in range(100):
        sol = solver.run(
            vars_scaled,
            bounds=bounds,
            params_static=params_static,
            c_scale=c_scale,
            beta_scale=beta_scale,
            lmbda=lmbda,
            mu=mu
        )

        vars_scaled = sol.params

        #Unscale
        c = vars_scaled[0] * c_scale
        beta = vars_scaled[1] * beta_scale

        power, thrust_err, _ = base_objective(c, beta, params_static)

        print(f"{i:<6} | {float(power):<12.4f} | {float(thrust_err):<12.4e} | {lmbda:<12.4f} | {mu:<10.1f}")

        # Convergence
        if abs(thrust_err) < eta_tol:
            print(f"\nConverged at iteration {i} with thrust error {thrust_err:.2e}")
            break

        #AL update
        if abs(thrust_err) > 0.25 * prev_err:
            mu = min(mu * mu_scale, mu_max)

        lmbda += mu * thrust_err
        prev_err = abs(thrust_err)


        V_ia   = propmodel.induced_velocity(params_static)

        dT_mom = propmodel.dT_momentum(V_ia, params_static)[0]
        dT_bld = propmodel.dT_blade(V_ia, params_static)[0]

        fig, axes = plt.subplots(3, 1, figsize=(6, 10))

        axes[0].plot(r, c)
        axes[0].stem(r, c)
        axes[0].set_title("Chord (m)")
        axes[0].set_xlabel("r/R")

        axes[1].plot(r, jnp.degrees(beta))
        axes[1].stem(r, jnp.degrees(beta))
        axes[1].set_title("Twist (deg)")
        axes[1].set_xlabel("r/R")

        axes[2].plot(r, dT_mom, label="Momentum")
        axes[2].plot(r, dT_bld, label="BEMT")
        axes[2].legend()

        plt.tight_layout()
        plt.show()

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

    axes[0].plot(r / params_opt.R, c / params_opt.R)
    axes[0].stem(r / params_opt.R, c / params_opt.R)
    axes[0].set_title("Chord (m)")
    axes[0].set_xlabel("r/R")

    axes[1].plot(r / params_opt.R, jnp.degrees(beta))
    axes[1].stem(r / params_opt.R, jnp.degrees(beta))
    axes[1].set_title("Twist (deg)")
    axes[1].set_xlabel("r/R")

    plt.tight_layout()
    plt.show()
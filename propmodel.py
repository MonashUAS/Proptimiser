## Define All Physical Properties for Given Propeller Geometry

# External Libraries
import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np

import matplotlib.pyplot as plt

# Internal Libraries
from init import Inputs, BladeSpec, Params, Propeller
from airfoil import aero_coeffs
from airflow import tip_root_loss_factors, inflow_model, equivelantActuatorDiskEfficiency, resultantVelocityComponents, high_induction_correction

jax.config.update("jax_enable_x64", True)

def motorPower(torque: float, omega: float, I_0: float = 0.7, R_m: float = 0.120, kV: float = 150):
    """
    Calcualtes Power Consumption required of the motor to drive the propeller

    Inputs:
        torque: torque of the motor (Nm)
        omega: rotational rate of the motor (rad/s)
        I_0: zero load current (A)
        R_m: Coil resistance (ohms)
        kV: motor speed constant (RPM/volt)
    """
    P_out = torque*omega

    K_t = 60 / (2*jnp.pi*kV) # Calculate the torque constant (Nm/A)

    I = torque / K_t # calculate the current consumed by the motor

    V = omega*60/(2*jnp.pi) / kV

    P_in = P_out + R_m * (I+I_0)**2 + V*I_0

    return P_in

## Blade Element Theory

def dT_blade(V_i: jnp.ndarray, params: Params, debug: bool = False):
    """
    Calculates instantaneous thrust at each radial and azimuthal station, 
    and calculates the average thrust in each annulus over a rotation.

    Inputs:
        V_i         <- 3D Induced velocity grid of shape (3, Nr, Npsi) [axial, tangential, radial]
        params      <- Params object used in the optimisation loop, remains unmodified by function execution
        debug       <- Boolean to select whether or not to print debug information when function is called

    Outputs:
        dT_blade_ave w/ size (Nr, )                        <- Thrust per annulus 
        dT_blade_instantaneous w/ size (Nr, Npsi)          <- Thrust per differential element

    Assumes: 
        |V_yz| < |V_x + V_ia|   <- Assumption that the annuli do not interact with each other 

    Uses formulation from (M. Nahon et al., 2015) equations 5 and 6 modified with 
    the inclusion of the prandtl loss parameter 'F'
    """
    # Unpack parameters
    N, c, psi = params.N, params.c, params.psi # geometric and positional values
    rho = params.rho # air density

    # Calculate Local Flow Properties
    V_R, phi = resultantVelocityComponents(V_i, params)

    # Get Aerodynamic Coefficients
    C_L, C_D, _ = aero_coeffs(V_i, params)

    # Blade Element Theory instantaneous thrust element
    dT_blade_instantaneous = 0.5 * rho * c[:, None] * (V_R**2) * (C_L * jnp.cos(phi) - C_D * jnp.sin(phi))
    
    # Average thrust in each annulus over one rotation
    dT_blade_ave = N / (2 * jnp.pi) * jnp.trapezoid(dT_blade_instantaneous, x=psi, axis=1)

    if params.cfg.V_i_0_verbose >= 2:
        jax.debug.print("  [BEMT] dT_blade: {dT_blade_ave}", dT_blade_ave=dT_blade_ave)
    
    return dT_blade_ave

def dH_blade(V_i: jnp.ndarray, params: Params, debug: bool = False):
    """
    Calculates instantaneous hub force at each radial and azimuthal station, 
    and calculates the average hub force in each annulus over a rotation.

    The hub forces are defined as the force element coplanar with the rotor disk 
    and normal to the radial direction.

    Inputs:
        V_i         <- 3D Induced velocity grid of shape (3, Nr, Npsi) [axial, tangential, radial]
        params      <- Params object used in the optimisation loop, remains unmodified by function execution
        debug       <- Boolean to select whether or not to print debug information when function is called

    Outputs:
        dH_blade_ave w/ size (Nr, )                        <- Force per annulus 

    Assumes: 
        |V_yz| < |V_x + V_ia|   <- Assumption that the annuli do not interact with each other 

    Uses formulation from (M. Nahon et al., 2015) equation 11 modified with 
    the inclusion of the prandtl loss parameter 'F'
    """
    # Unpack parameters
    N, c, psi = params.N, params.c, params.psi # geometric and positional values
    rho = params.rho # air density

    # Calculate Local Flow Properties
    V_R, phi = resultantVelocityComponents(V_i, params)
    
    # Get Aerodynamic Coefficients
    C_L, C_D, _ = aero_coeffs(V_i, params)

    # Tip and Root Loss Factors
    tip_loss, root_loss = tip_root_loss_factors(phi, params)
    
    # Calculate the total prandtl loss correction factor
    F = jnp.multiply(tip_loss, root_loss)

    # Inner integral term
    inner = (V_R**2) * (C_L * jnp.sin(phi) + C_D * jnp.cos(phi))

    # Average hub force per annulus over one rotation
    dH_blade_ave = N / (4 * jnp.pi) * rho * c * jnp.trapezoid(inner, x=psi, axis=1)

    if debug:
        print("------dH_blade DEBUG------")
        print("dH Shapes (ave.):", jnp.shape(dH_blade_ave))
        print("V_i shape:", jnp.shape(V_i))
        print("F shape:", jnp.shape(F))
        print("------dH_blade DEBUG------")

    return dH_blade_ave


def dQ_blade(V_i: jnp.ndarray, params: Params, debug: bool = False):
    """
    Calculates instantaneous torque at each radial and azimuthal station, 
    and calculates the average torque in each annulus over a rotation.

    Inputs:
        V_i         <- 3D Induced velocity grid of shape (3, Nr, Npsi) [axial, tangential, radial]
        params      <- Params object used in the optimisation loop, remains unmodified by function execution
        debug       <- Boolean to select whether or not to print debug information when function is called

    Outputs:
        dQ_blade_ave w/ size (Nr, )                        <- Torque per annulus 

    Assumes: 
        |V_yz| < |V_x + V_ia|   <- Assumption that the annuli do not interact with each other 

    Uses formulation from (M. Nahon et al., 2015) equation 12
    """
    r = params.r

    # Calculate the hub force elements dH
    dH_blade_ave = dH_blade(V_i, params, debug)

    # Use equation 12 to calculate the torque elements from the hub elements
    dQ_blade_ave = dH_blade_ave * r

    if debug:
        print("------dQ_blade DEBUG------")
        print("dQ Shapes (ave.):", jnp.shape(dQ_blade_ave))
        print("V_i shape:", jnp.shape(V_i))
        print("------dQ_blade DEBUG------")

    return dQ_blade_ave


### Momentum Theory
def dT_momentum(V_i: jnp.ndarray, params: Params, debug: bool = False, apply_tip_loss: bool = True):
    """
    Calculates instantaneous thrust at each radial and azimuthal station, 
    and calculates the average thrust in each annulus over a rotation.

    dT_momentum uses the momentum change in the air as it passes through the 
    blade element and is accelerated rather than the forces on the blade element to determine
    thrust.

    Inputs:
        V_i         <- 3D Induced velocity grid of shape (3, Nr, Npsi) [axial, tangential, radial]
        params      <- Params object used in the optimisation loop, remains unmodified by function execution
        debug       <- Boolean to select whether or not to print debug information when function is called

    Outputs:
        dT_momentum_ave w/ size (Nr, )                     <- Thrust per annulus 

    Assumes: 
        |V_yz| < |V_x + V_ia|   <- Assumption that the annuli do not interact with each other 

    Uses formulation from (M. Nahon et al., 2015) equation 9
    """
    # Unpack axial induced velocity component
    V_ia = V_i[0]

    # Momentum theory about annular element
    r, psi = params.r, params.psi
    V_x, V_yz = params.V_x, params.V_yz
    rho = params.rho

    V_R, phi = resultantVelocityComponents(V_i, params)

    if apply_tip_loss:
        tip_loss, root_loss = tip_root_loss_factors(phi, params)
        F = jnp.multiply(tip_loss, root_loss)
        F = jnp.clip(F, min=0.05, max=1.0)
    else:
        F = jnp.ones_like(V_ia)   # smooth, well-scaled system for the solve

    # Compute baseline classical momentum element locally
    V_prime = jnp.sqrt((V_x + V_ia) ** 2 + V_yz ** 2)
    dT_element_classical = 2 * rho * r[:, None] * V_ia * V_prime * F

    # Route through lax.switch inside airflow.py for conditional modifications
    dT_element_corrected = high_induction_correction(dT_element_classical, V_ia, F, params)

    # Integrate over psi to get the average thrust per annulus
    dT_momentum_ave = jnp.trapezoid(dT_element_corrected, x=psi, axis=1)

    if debug:
        print("------dT_momentum DEBUG------")
        print("dT Shapes (ave.):", jnp.shape(dT_momentum_ave))
        print("V_i shape:", jnp.shape(V_i))
        print("F shape:", jnp.shape(F))
        print("------dT_momentum DEBUG------")

    return dT_momentum_ave

def dQ_momentum(V_i, params, debug=False, apply_tip_loss=True):
    """
    Calculates instantaneous torque at each radial and azimuthal station, 
    and calculate the average torque in each annulus over a rotation

    Inputs:
        V_i         <- Induced velocity grid (3, n_r, n_psi) [axial, tangential, radial]
        params      <- Params object used in the optimisation loop, remains unmodified by function execution
        debug       <- Boolean to select whether or not to print debug information when function is called

    Outputs:
        dQ_momentum_ave w/ size (Nr, )                        <- Torque per annulus 

    Assumes: 
        |V_yz| < |V_x + V_ia|   <- Assumption that the annuli do not interact with each other 

    Uses formulation from (M. Nahon et al., 2015) adapted for angular momentum
    """
    # Extract velocity components from 3D grid
    V_ia = V_i[0]  # shape (n_r, n_psi)
    V_it = V_i[1]  # shape (n_r, n_psi)

    # Momentum theory about annular element
    r, psi = params.r, params.psi
    V_x, V_yz = params.V_x, params.V_yz
    rho = params.rho

    V_R, phi = resultantVelocityComponents(V_i, params)
   
    if apply_tip_loss:
        tip_loss, root_loss = tip_root_loss_factors(phi, params)
        F = jnp.multiply(tip_loss, root_loss)
        F = jnp.clip(F, min=0.05, max=1.0)
    else:
        F = jnp.ones_like(V_ia)   # smooth, well-scaled system for the solve

    # Compute baseline classical momentum element locally
    # Uses V_prime to account for skewed wake mass flux (Nahon 2015)
    V_prime = jnp.sqrt((V_x + V_ia) ** 2 + V_yz ** 2)
    
    # dQ element per radian (omitting dr to match dT_momentum format)
    dQ_element_classical = 2 * rho * (r[:, None] ** 2) * V_it * V_prime * F

    # Integrate over psi to get the average torque per annulus
    dQ_momentum_ave = jnp.trapezoid(dQ_element_classical, x=psi, axis=1)

    if debug:
        print("------dQ_momentum DEBUG------")
        print("dQ Shapes (ave.):", jnp.shape(dQ_momentum_ave))
        print("V_ia shape:", jnp.shape(V_ia))
        print("V_it shape:", jnp.shape(V_it))
        print("F shape:", jnp.shape(F))
        print("------dQ_momentum DEBUG------")

    return dQ_momentum_ave

def thrust_residual(V_i, params):
    """
    Calculates the 1D thrust residual (BEMT vs Momentum) across radial stops.
    Returns array of shape (n_r,)
    """
    dT_BEMT_avg = dT_blade(V_i, params, debug=False)
    dT_MMNT_avg = dT_momentum(V_i, params, debug=False, apply_tip_loss=False)   # smooth for the solve
    return dT_BEMT_avg - dT_MMNT_avg

def torque_residual(V_i, params):
    """
    Calculates the 1D torque residual (BEMT vs Momentum) across radial stops.
    Returns array of shape (n_r,)
    """
    dQ_BEMT_avg = dQ_blade(V_i, params, debug=False)
    dQ_MMNT_avg = dQ_momentum(V_i, params, debug=False, apply_tip_loss=False)
    return dQ_BEMT_avg - dQ_MMNT_avg

def bemt_residuals(lambda_i_0_guess, params):
    """
    Evaluates the coupled BEMT residuals.
    lambda_i_0_guess: non-dimensional inflow ratios of shape (3, n_r)
    """
    # Retrieve scales and Re-dimensionalise
    V_scale = get_velocity_scales(params)
    V_i_0_guess_dim = lambda_i_0_guess * V_scale

    # Introduce zero for non-solved components to maintain shape (3, n_r)
    V_i_0_guess_dim = jnp.zeros_like(params.V_i_0).at[params.cfg.solve_rows].set(V_i_0_guess_dim)

    # Expand to 3D grid
    V_i = inflow_model(V_i_0_guess_dim, params)

    # Axial Residual (Scale by Thrust Target)
    dT_res_physical = thrust_residual(V_i, params)
    dT_res = dT_res_physical / params.thrust_target
    
    # Radial Residual (Already O(1)) TODO: 
    dR_res = jnp.zeros_like(lambda_i_0_guess[0])

    # Tangential Residual (Scale by Approximate Ideal Torque)
    dQ_res_physical = torque_residual(V_i, params)

    # Scale by an approximate ideal torque to make O(1)
    Q_ref = params.thrust_target * params.cfg.R_tip 
    dQ_res = dQ_res_physical / Q_ref

    res_components = {0: dT_res, 1: dQ_res, 2: dR_res}
    res = jnp.stack([res_components[i] for i in range(3)], axis=0)
    # filter out non-solved components
    res_filtered = res[params.cfg.solve_rows]
    
    return res_filtered

def get_velocity_scales(params):
    """
    Computes characteristic velocity scales for the 3 inflow components
    that non-dimensionalise the unknowns of the induced velocity solve.
    Returns: V_scale -> shape (3, 1) for broadcasting against (3, n_r) arrays.
    """
    R_tip = params.cfg.R_tip
    A_disk = jnp.pi * (R_tip ** 2)
    
    # Axial Scale: Mean Momentum Inflow
    v_h = jnp.sqrt(params.thrust_target / (2.0 * params.rho * A_disk))
    V_ia_mean = -params.V_x / 2.0 + jnp.sqrt((params.V_x / 2.0)**2 + v_h**2)
    
    # Tangential Scale: Ideal Mean Swirl
    P_ideal = params.thrust_target * (params.V_x + V_ia_mean)
    Q_ideal = P_ideal / params.omega
    V_it_mean = Q_ideal / (params.rho * A_disk * (params.V_x + V_ia_mean) * R_tip)
    
    # Radial Scale: Tip Speed (Centrifugal reference)
    V_tip = params.omega * R_tip

    # Stack and reshape to (3, 1) so it broadcasts perfectly with (3, n_r)
    V_scale = jnp.array([V_ia_mean, V_it_mean, V_tip])[:, None]
    V_scale = V_scale[params.cfg.solve_rows]  # Only keep the components that are being solved for
    return V_scale

def _station_blocks(linear_fn, shape):
    '''
    The (n_r, k, k) diagonal blocks of a linear map on (k, n_r) arrays whose stations are independent:
    one JVP per solved component gives that column of every station's block at once.
    '''
    k = shape[0]
    columns = [linear_fn(jnp.zeros(shape).at[c].set(1.0)) for c in range(k)]
    return jnp.transpose(jnp.stack(columns, axis=-1), (1, 0, 2))

def _station_solve(blocks, rhs, active):
    '''
    Solves every station's block system (rhs of shape (k, n_r)). The pseudo-inverse gives the minimum-norm solution
    when a block is rank deficient (e.g. a component whose residual is not modelled); inactive stations get zero.
    '''
    x = jnp.einsum("nij,jn->in", jnp.linalg.pinv(blocks), rhs)
    return jnp.where(active[None, :], x, 0.0)

def _newton_bemt(residual_fn, lam, active, tol, max_iter, verbose, params):
    '''
    Damped blockwise Newton on residual_fn(lam) = 0. Each station backtracks (halving its step) until its own
    residual norm decreases, so a hard station never slows the others. A station already within tol takes no
    step: at round-off level its residual cannot decrease, and backtracking would run to machine epsilon.
    Converged when the largest scaled residual over the active stations is below tol.
    '''
    def max_residual(l):
        # Non-finite residuals count as unconverged (a NaN would otherwise end the loop as if converged)
        res = jnp.abs(residual_fn(l))
        return jnp.max(jnp.where(active[None, :], jnp.where(jnp.isfinite(res), res, jnp.inf), 0.0))

    def newton_step(state):
        l, it, _ = state
        r, linear_fn = jax.linearize(residual_fn, l)
        step = _station_solve(_station_blocks(linear_fn, l.shape), -r, active)

        # A station whose Newton step is undefined (non-finite residual or derivative, e.g. far outside the airfoil
        # model's range) restarts from the momentum-theory guess, lambda = 1, rather than spreading NaN
        restart = active & ~jnp.all(jnp.isfinite(step), axis=0)
        step = jnp.where(restart[None, :], 0.0, step)
        r = jnp.where(restart[None, :], 0.0, r)
        done = jnp.all(jnp.abs(r) <= tol, axis=0)
        step = jnp.where(done[None, :], 0.0, step)
        r0 = jnp.sum(r**2, axis=0)

        # Backtracking stops at the precision of the residual: halving past machine epsilon cannot change l
        max_halvings = int(-np.log2(np.finfo(l.dtype).eps))
        def backtrack_cond(carry):
            _, accepted, n = carry
            return (~jnp.all(accepted)) & (n < max_halvings)
        def backtrack(carry):
            alpha, accepted, n = carry
            decreased = jnp.sum(residual_fn(l + alpha * step)**2, axis=0) < r0
            accepted = accepted | decreased
            return jnp.where(accepted, alpha, 0.5 * alpha), accepted, n + 1
        alpha, accepted, _ = jax.lax.while_loop(backtrack_cond, backtrack, (jnp.ones(l.shape[1]), done | ~active, 0))

        # No decrease down to machine epsilon: the station sits at a stationary point of its residual norm that is
        # not a root (e.g. a reversed-flow state near the hub in forward flight); it restarts from the
        # momentum-theory guess too
        restart = restart | ~accepted
        l = jnp.where(restart[None, :], 1.0, l + alpha * step)
        err = max_residual(l)
        if verbose >= 2:
            jax.debug.print("    [BEMT] Newton iter {i:3d} | residual {e:.2e} | smallest step fraction {a:.2e}",
                            i=it + 1, e=err, a=jnp.min(jnp.where(active & ~done, alpha, 1.0)))
        return l, it + 1, err

    def not_converged(state):
        _, it, err = state
        return (it < max_iter) & (err > tol)

    l, it, err = jax.lax.while_loop(not_converged, newton_step, (lam, 0, max_residual(lam)))
    if verbose >= 1:
        jax.debug.print("  [BEMT] T_target {t:7.2f} N | omega {w:7.1f} rad/s | Newton iters {n:3d} | residual {e:.2e}",
                        t=params.thrust_target, w=params.omega, n=it, e=err)
    return l

@jax.jit
def induced_velocity(params):
    '''
    Calculates the coupled 3D induced velocity field using BEMT.

    Solves the non-dimensional residuals (bemt_residuals) for the solved components with blockwise Newton, from
    params.V_i_0 as the initial guess. jax.lax.custom_root makes derivatives of the solution exact by implicit
    differentiation, so gradients do not depend on the iteration history. A station at r = 0 is a zero-area
    annulus with no momentum equation to satisfy, so its induced velocity is left at the initial guess
    (it carries no thrust or torque).
    '''
    cfg = params.cfg
    V_scale = get_velocity_scales(params)
    lam_0 = params.V_i_0[cfg.solve_rows] / V_scale
    active = params.r > 0.0

    residual_fn = lambda lam: bemt_residuals(lam, params)
    solve = lambda f, guess: _newton_bemt(f, guess, active, cfg.V_i_0_eps, cfg.V_i_0_max_iter, cfg.V_i_0_verbose, params)
    tangent_solve = lambda linear_fn, rhs: _station_solve(_station_blocks(linear_fn, rhs.shape), rhs, active)
    lam = jax.lax.custom_root(residual_fn, lam_0, solve, tangent_solve)

    # Re-dimensionalise, with zeros for the components that are not solved
    V_i_0 = jnp.zeros_like(params.V_i_0).at[cfg.solve_rows].set(lam * V_scale)

    # Expand to 3D inflow grid
    V_i = inflow_model(V_i_0, params)

    return V_i, V_i_0

def torque(V_i, params, debug=False):
    """
    Calculates the total torque on the propeller by integrating the average torque per annulus per rotation
    along the radial direction
    
    Inputs:
        V_i        <- Induced velocity at each radial and azimuthal station
        params      <- Params object used in the optimisation loop, remains unmodified by function execution
        debug       <- Boolean to select whether or not to print debug information when function is called
        apply_tip_loss <- Boolean to select whether or not to apply tip loss correction when calculating torque

    Outputs:
        Q <- Net torque per rotation of the propeller
    
    Assumes: 
        |V_yz| < |V_x + V_ia|   <- Assumption that the annuli do not interact with each other 
    """
    
    r = params.r

    # Get the average torque per rotation in each annulus using the dQ_blade function
    dQ_blade_ave = dQ_momentum(V_i, params, debug, apply_tip_loss=True)   # accurate, applied post-solve

    # Use trapezoidal integration in the radial direction to compute the total torque
    Q = jnp.trapezoid(dQ_blade_ave, x=r)

    return Q

def thrust(V_i, params):
    '''
    Calculate total thrust from solved induced velocity
    '''
    r = params.r
    
    dT = dT_momentum(V_i, params, apply_tip_loss=True)   # accurate, applied post-solve

    # Numerically Compute Double Integral to Solve for Thrust
    T = jnp.trapezoid(dT, x=r)

    return T

def mechanicalPower(V_i, params):
    '''
    Calculates and returns the shaft power (W) applied to the propeller 
    '''

    # Unpack Params
    omega = params.omega
    # Calculate Local Axial Flow Velocity

    # net torque over one rotation
    Q = torque(V_i, params, debug=False)

    # Net mechanical power (omega in rad/s, torque in Nm)
    P = Q * omega

    return jnp.squeeze(P)

def actuator_disk_power(T, params):
    '''
    Calculate Ideal Actuator Disk Power for same thrust
    Provides theoretical local minimum power using actuator disc model
    T is from momentum theory, 
    '''
    rho = params.rho
    R   = params.cfg.R_tip
    V_x = params.V_x # freestream velocity

    A = jnp.pi*R**2 # disc area
    v_induced = jnp.sqrt(T/(2*rho*A) + V_x**2/4) - V_x/2  # induced velocity through actuator disc
    P_act = T*(V_x+v_induced) # Ideal actuator disk power
    return jnp.squeeze(P_act)

def single(c, beta, N, plot=False):
    '''
    Single Annulus Solve
    '''
    # Build the default propeller profile, then cut it down to one annulus
    beta_deg = float(jnp.degrees(beta))
    params = Propeller.default(inputs=Inputs(n_r=2), blade=BladeSpec(N=N), c=c, beta_deg=beta_deg).condition(0)

    # Match the specific geometry overrides from your original test case
    params = eqx.tree_at(
        lambda p: (p.r, p.dr, p.c, p.beta, p.V_i_0),
        params,
        (jnp.array([0.3]), jnp.array([0.05]), params.c[:1], params.beta[:1], params.V_i_0[:, :1])
    )

    # Solve Induced Velocity
    V_i, _ = induced_velocity(params)

    if plot:
        # Axial component at the single radial node
        plt.plot(jnp.degrees(params.psi), V_i[0, 0])
        plt.title("Induced Velocity Distribution around Propeller Annulus")
        plt.xlabel("Angle (degrees)")
        plt.ylabel("Induced Velocity (m/s)")
        plt.show()


def flatblade(c, beta, N, plot=False):
    '''
    Flat Blade Demo Model. Inputs a blade of fixed chord and radial distribution 
    and solves+plots induced velocity.

    Used for benchmarking the induced velocity solver

    TODO: Fix plotting
    '''

    # Utilize your factory function to build the base configuration
    params = Propeller.default(blade=BladeSpec(N=N), c=c, beta_deg=beta).condition(0)

    import os
    import jax.profiler

    # Create a cross-platform path to ./logs/tensorboard
    log_dir = os.path.join(".", "logs", "tensorboard")

    # Warm up
    _ = induced_velocity(params)
    V_i, _ = induced_velocity(params)
    V_i.block_until_ready() # Ensure warm-up is totally finished

    # Profile
    jax.profiler.start_trace(log_dir)
    V_i, _ = induced_velocity(params)
    V_i.block_until_ready()
    jax.profiler.stop_trace()
    V_i_a = V_i[0,:,:]
    # Calculate total thrust
    T = thrust(V_i, params)
    print("Total Thrust:", T)

    actuatorEffiency = equivelantActuatorDiskEfficiency(T, params)
    print("Actuator efficiency:", actuatorEffiency)

    # Wrap data periodically for 360-degree polar contouring
    psi_plot = np.append(params.psi, 2 * np.pi)
    V_ia_plot = np.hstack([V_ia, V_ia[:, :1]])

    if plot == True:
        # Plot heatmap using structured params properties
        fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
        plt.contourf(params.psi, params.r, V_i_a, cmap='viridis')
        plt.colorbar(label='Induced Velocity [m/s]')
        plt.xlabel('X [m]')
        plt.ylabel('Y [m]')
        plt.title('Induced Velocity over Propeller Disk')
        plt.show()

    return params, V_i, T

def apc_model(filepath, plot=True):
    '''
    Loads an APC propeller geometry from a .PE0 file, interpolates it onto the
    solver's computational grid, solves the induced velocity, and plots the results.

    Returns (prop, params, V_i_0, T) where prop is the Propeller built from the APC file.
    '''
    import queue
    import results
    from matplotlib.animation import FuncAnimation

    print(f"Loading APC Propeller geometry from: {filepath}")
    prop = Propeller.from_apc(filepath)
    params = prop.condition(0)

    # Solve Induced Velocity
    V_i, V_i_0 = induced_velocity(params)
    
    # Update params with solved state
    params = eqx.tree_at(lambda p: p.V_i_0, params, V_i_0)

    # Calculate overall performance metrics
    T = thrust(V_i, params)
    P_mech = mechanicalPower(V_i, params)

    # Plotting via LivePropPlotter
    if plot:
        print("Launching LivePropPlotter canvas...")
        gui_queue = queue.Queue()
        plotter = results.LivePropPlotter(params)

        gui_queue.put(params)
        
        ani = FuncAnimation(
            plotter.fig, 
            plotter.update, 
            fargs=(gui_queue,), 
            interval=200, 
            cache_frame_data=False,
            blit=False
        )

        plt.show()

        fig = results.debug_diagnostics_plot(params, 0)

        dummy = input("Done with plots? \n")

    return prop, params, V_i_0, T

if __name__ == "__main__":
    prop, params, V_i_0, T = apc_model("apc_prop_data/propgeom_data/20x8-PERF.PE0", plot=True)
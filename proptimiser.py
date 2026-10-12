# External Imports
import jax
import time
from dataclasses import dataclass
from typing import NamedTuple
import sys
import signal
import threading
import warnings
import numpy as np
from scipy.special import logsumexp
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import jaxopt
import equinox as eqx
from abc import ABC, abstractmethod

# Internal Imports
import airflow
import airfoil
import geometry
import propmodel

_stop_event = threading.Event()
_sigint_installed = False


def _sigint_handler(signum, frame):
    if not _stop_event.is_set():
        _stop_event.set()
        # Write + flush immediately so the message shows up in VSCode's
        # integrated terminal without waiting on buffered/batched output.
        sys.stdout.write("\n[Ctrl+C] Interrupt received — stopping after the current iteration...\n")
        sys.stdout.flush()
    else:
        # A second Ctrl+C means "I really mean it, stop now".
        sys.stdout.write("\n[Ctrl+C] Second interrupt received — forcing immediate exit.\n")
        sys.stdout.flush()
        raise KeyboardInterrupt


def install_sigint_handler():
    """
    Installs the custom SIGINT handler once, from the main thread only.
    Safe to call multiple times (idempotent).
    """
    global _sigint_installed
    if _sigint_installed:
        return
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGINT, _sigint_handler)
        _sigint_installed = True

install_sigint_handler()


def objectives(params):
    """
    Return the Base Objective Values of the Solver given an input of parameters:
    motor power, thrust and the solved induced velocity V_i_0 for one flight condition.
    """
    V_i, V_i_0   = propmodel.induced_velocity(params)             # Solve induced velocity field for current guess
    thrust = jnp.squeeze(propmodel.thrust(V_i, params))            # return the total thrust the propeller produce
    torque = jnp.squeeze(propmodel.torque(V_i, params))            # return total torque propeller produces

    # calculate the motor power
    cfg = params.cfg
    motor_power = propmodel.motorPower(torque, params.omega, I_0 = cfg.I_0, R_m = cfg.R_m, kV = cfg.kV)

    # Return V_i_0 so we can catch it!
    return motor_power, thrust, V_i_0


## Problem formulation
# Every quantity the optimiser sees is non-dimensionalised by a physical reference taken from the specification,
# so no weights are needed to balance them:
#   objective    J = sum_i w_i P_i / sum_i w_i P_ideal,i   motor power relative to the ideal (actuator disc) power of
#                                                            the thrust targets; ~1/efficiency, proportional to energy
#   constraints  g >= 0, each relative to its own limit:    thrust within +/- thrust_tol of the target, motor power
#                                                            below P_max, section t/c above min_tc, load-weighted
#                                                            NeuralFoil confidence above min_confidence (see
#                                                            constraint_values)

class ConditionOutputs(NamedTuple):
    power: jnp.ndarray          # motor power (W)
    thrust: jnp.ndarray         # thrust (N)
    ideal_power: jnp.ndarray    # actuator disc power at the thrust target (W)
    log_load: jnp.ndarray       # per station: log of the section dynamic-pressure weight, sum over psi of V_R^2 dr dpsi
    log_load_confidence: jnp.ndarray  # per station: log of the same sum weighted by the NeuralFoil confidence
    V_i_0: jnp.ndarray          # solved induced velocity (warm start for the next evaluation)

def _trapezoid_weights(x):
    """
    Weights w with sum(w * f) == jnp.trapezoid(f, x), the quadrature the forces are integrated with.
    """
    dx = jnp.diff(x)
    return jnp.concatenate([dx[:1], dx[1:] + dx[:-1], dx[-1:]]) / 2

def _periodic_weights(x):
    """
    Weights of the full-turn rule on a uniform periodic grid that stops one step short of its period (the azimuth
    grid): every sample carries one step. _trapezoid_weights would halve the first and last samples.
    """
    return jnp.full_like(x, 2 * jnp.pi / x.shape[0])

def condition_outputs(params) -> ConditionOutputs:
    """
    Physical outputs of ONE flight condition.
    """
    V_i, V_i_0 = propmodel.induced_velocity(params)
    thrust = jnp.squeeze(propmodel.thrust(V_i, params))
    torque = jnp.squeeze(propmodel.torque(V_i, params))
    cfg = params.cfg
    motor_power = propmodel.motorPower(torque, params.omega, I_0=cfg.I_0, R_m=cfg.R_m, kV=cfg.kV)

    if cfg.coeff_method == 3:
        _, _, _, log_confidence = airfoil.aero_coeffs(V_i, params, return_log_confidence=True)
        log_confidence = airflow.to_grid(log_confidence, params)
    else:
        log_confidence = jnp.zeros((params.r.shape[0], params.psi.shape[0]))

    # Each blade element's dynamic-pressure weight V_R^2 dr dpsi, independent of the (uncertain) coefficients and of
    # the chord: a chord factor would let the optimiser drop a poorly modelled station from the average by shrinking
    # it; the floor on V_R matches the Reynolds number in aero_coeffs
    V_R, _ = airfoil.resultantVelocityComponents(V_i, params)
    V_R = airflow.to_grid(V_R, params)
    log_load = (2 * jnp.log(jnp.maximum(V_R, 1e-6)) + jnp.log(_trapezoid_weights(params.r))[:, None]
                + jnp.log(_periodic_weights(params.psi))[None, :])
    return ConditionOutputs(motor_power, thrust, propmodel.actuator_disk_power(params.thrust_target, params),
                            jax.nn.logsumexp(log_load, axis=1), jax.nn.logsumexp(log_load + log_confidence, axis=1),
                            V_i_0)

@jax.jit
def evaluate_all_conditions(params_struct) -> ConditionOutputs:
    """
    ConditionOutputs of every flight condition (fields stacked along a leading condition axis).
    """
    return jax.vmap(condition_outputs)(params_struct)

def energy_ratio(outputs, params_struct):
    """
    Objective: weighted motor power over the weighted ideal power of the thrust targets. The denominator depends
    only on the flight conditions, so this is the mission energy (objective_weight = time fractions) made
    dimensionless; it is roughly the inverse of the overall propulsive efficiency.
    """
    w = params_struct.objective_weight
    return jnp.sum(w * outputs.power) / jnp.sum(w * outputs.ideal_power)

def constraint_values(outputs, params_struct, space):
    """
    Inequality constraints g >= 0, each relative to its own limit, concatenated in the order of
    space.constraint_groups(). A value of -0.01 means the limit is exceeded by 1% of itself.

    NeuralFoil confidence is constrained once per condition, as the mean over the aerodynamic zone weighted by
    each blade element's dynamic pressure (V_R^2 dr dpsi). The chord is left out of the weight on purpose: with it
    a poorly modelled station can leave the average by shrinking to the minimum chord. A per-station limit would
    also bind at elements that carry almost no load, and can be impossible to meet: near the hub in forward flight
    the inflow angle approaches 90 deg, far beyond the blade's twist bounds, for every section shape.

    It is held as log(confidence / min_confidence) >= 0: the same feasible set, and equal to the relative form
    to first order at the limit. NeuralFoil's confidence is a sigmoid that saturates at zero far outside its
    training data, where the relative form has no gradient; the log tends to the network's logit there, which
    still points back towards the data. relative_margins() maps it back for reporting.
    """
    cfg = params_struct.cfg
    rel_thrust = outputs.thrust / params_struct.thrust_target - 1.0
    g = [cfg.thrust_tol - rel_thrust, cfg.thrust_tol + rel_thrust, 1.0 - outputs.power / cfg.P_max]
    if space.cst_active:
        g.append(geometry.max_thickness(params_struct.cst[0, 1:]) / space.min_tc - 1.0)
    if space.confidence_active:
        aero = np.array(space.aero_stations)
        g.append(jax.nn.logsumexp(outputs.log_load_confidence[:, aero], axis=1)
                 - jax.nn.logsumexp(outputs.log_load[:, aero], axis=1) - np.log(cfg.min_confidence))
    return jnp.concatenate(g)

def relative_margins(g, space, n_cond):
    """
    Constraint values as margins relative to their limits (numpy; -0.01: limit exceeded by 1% of itself), with the
    log-form confidence constraint mapped back.
    """
    g, start = np.array(g, dtype=float), 0
    for name, size in space.constraint_groups(n_cond):
        if name == "confidence":
            g[start:start + size] = np.expm1(g[start:start + size])
        start += size
    return g

def planform_penalty(params_struct):
    """
    Chord smoothness penalty: w_smooth times the mean, over aerodynamic-zone stations, of the squared second
    difference of log(chord). Zero for a geometric (exponential) taper.
    """
    cfg = params_struct.cfg
    log_c, r = jnp.log(params_struct.c[0]), params_struct.r[0]      # geometry is shared by every flight condition
    triples = (r[:-2] >= cfg.R_aero).astype(log_c.dtype)            # a triple lies in the aero zone when its first station does
    second_difference = log_c[2:] - 2 * log_c[1:-1] + log_c[:-2]
    return cfg.w_smooth * jnp.sum(triples * second_difference ** 2) / jnp.maximum(jnp.sum(triples), 1.0)

@jax.jit
def problem_values(vars_scaled, params_struct, space):
    """
    (objective, constraints, V_i_0) of the design vars_scaled, for constrained solvers. The objective is the energy
    ratio plus the chord smoothness penalty (planform_penalty); Progress reports the energy ratio alone.
    """
    params_struct = space.from_unit(vars_scaled, params_struct)
    outputs = evaluate_all_conditions(params_struct)
    objective = energy_ratio(outputs, params_struct) + planform_penalty(params_struct)
    return objective, constraint_values(outputs, params_struct, space), outputs.V_i_0

def penalty(g, params_struct, space):
    """
    Quadratic penalty on constraint violations for bound-constrained solvers (L-BFGS-B): thrust and motor power
    weighted by w_thrust, thickness by w_thickness, confidence by w_confidence.
    """
    cfg = params_struct.cfg
    weights = {"thrust (upper)": cfg.w_thrust, "thrust (lower)": cfg.w_thrust, "motor power": cfg.w_thrust,
               "thickness": cfg.w_thickness, "confidence": cfg.w_confidence}
    total, start = 0.0, 0
    for name, size in space.constraint_groups(params_struct.omega.shape[0]):
        shortfall = jax.nn.relu(-g[start:start + size]) ** 2
        total = total + weights[name] * jnp.sum(shortfall)
        start += size
    return total

@jax.jit
def total_cost(vars_scaled, params_struct, space):
    """
    Penalised cost for bound-constrained solvers: the energy ratio plus the constraint penalties.
    Returns (cost, V_i_0).
    """
    objective, g, V_i_0 = problem_values(vars_scaled, params_struct, space)
    return objective + penalty(g, space.from_unit(vars_scaled, params_struct), space), V_i_0

@jax.jit
def infeasibility(vars_scaled, params_struct, space):
    """
    Sum of squared relative constraint violations: zero exactly when every constraint is satisfied. As each
    constraint is already relative to its own limit, no weights are needed. Returns (value, V_i_0).
    """
    _, g, V_i_0 = problem_values(vars_scaled, params_struct, space)
    return jnp.sum(jax.nn.relu(-g) ** 2), V_i_0

def condition_cost(params):
    """
    Diagnostic cost of ONE condition: weighted power relative to its ideal power plus its thrust penalty.
    Returns (cost, V_i_0).
    """
    out = condition_outputs(params)
    cfg = params.cfg
    violation = jax.nn.relu(jnp.abs(out.thrust / params.thrust_target - 1.0) - cfg.thrust_tol)
    return params.objective_weight * (out.power / out.ideal_power + cfg.w_thrust * violation**2), out.V_i_0


class DesignSpace(eqx.Module):
    '''
    Maps the optimiser's unit-box variables onto the blade and operating state.

    Only free entries are design variables (physical value = lo + x * (hi - lo)). Stations frozen by
    BladeSpec.freeze keep their seed values, and blend-zone chord/twist are derived every step from the
    hub values and the first aerodynamic station.
    '''
    lo: dict                    # physical lower bound of each free entry
    hi: dict                    # physical upper bound of each free entry
    seed: dict                  # full seed arrays: c, beta (n_r,), cst (n_aero, N_CST), omega, tilt (n_cond,)
    hub_cst: jnp.ndarray        # hub section, fixed
    blend: jnp.ndarray          # blend fraction per station (0 at r_hub, 1 at the anchor station)
    inboard: jnp.ndarray        # stations whose chord/twist are derived from the blend
    free: tuple = eqx.field(static=True)        # ((name, flat indices of the free entries), ...)
    anchor: int = eqx.field(static=True)
    hub_chord: float = eqx.field(static=True)
    hub_twist: float = eqx.field(static=True)
    cst_active: bool = eqx.field(static=True)
    min_tc: float = eqx.field(static=True)
    aero_stations: tuple = eqx.field(static=True)       # radial stations in the aerodynamic zone (r >= r_aero)
    confidence_active: bool = eqx.field(static=True)    # NeuralFoil confidence constraint in use

    @classmethod
    def build(cls, prop):
        blade, inputs = prop.blade, prop.inputs
        r = np.asarray(prop.r)
        blend, inboard, anchor = geometry.planform_blend_weights(r, blade.R_hub, blade.R_aero)

        free, lo, hi = [], {}, {}

        def add(name, idx, lower, upper):
            if len(idx) == 0:
                return
            free.append((name, tuple(int(i) for i in idx)))
            lo[name] = jnp.broadcast_to(jnp.asarray(lower, dtype=float), (len(idx),))
            hi[name] = jnp.broadcast_to(jnp.asarray(upper, dtype=float), (len(idx),))

        # Chord and twist: aerodynamic-zone stations outside the freeze ranges
        for name, (lower, upper) in (("c", blade.c_bounds), ("beta", np.radians(blade.beta_bounds_deg))):
            add(name, [i for i, r_i in enumerate(r) if not inboard[i] and not blade.is_frozen(name, r_i)], lower, upper)
            if any(blade.is_frozen(name, r_i) for r_i in r[inboard]):
                warnings.warn(f"freeze['{name}'] covers blend-zone stations; their {name} is derived from the hub and r_aero values")

        # CST weights and LE weight at unfrozen aero stations (TE thickness stays at its seed value)
        cst_active = blade.optimise_cst and inputs.coeff_method == 3
        if blade.optimise_cst and not cst_active:
            warnings.warn("CST optimisation needs NeuralFoil (Inputs.coeff_method = 3); airfoil sections stay fixed")
        if cst_active:
            stations = [j for j, r_j in enumerate(blade.r_foils[2:]) if not blade.is_frozen("cst", r_j)]
            idx = [j * geometry.N_CST + k for j in stations for k in range(geometry.N_CST - 1)]
            seed_values = np.asarray(prop.cst).ravel()[idx]
            add("cst", idx, seed_values - blade.cst_bound, seed_values + blade.cst_bound)

        # Operating state: conditions flagged in solve_omega / solve_tilt
        omega_max = 343.0 / blade.R_tip
        add("omega", np.flatnonzero(np.asarray(inputs.solve_omega)), 0.1 * omega_max, omega_max)
        add("tilt", np.flatnonzero(np.asarray(inputs.solve_tilt)), 0.0, np.radians(30.0))

        if not free:
            raise ValueError("Nothing to optimise: every design variable is frozen")

        return cls(
            lo=lo, hi=hi,
            seed=dict(c=prop.c, beta=prop.beta, cst=prop.cst, omega=prop.omega, tilt=prop.tilt),
            hub_cst=jnp.asarray(blade.hub_cst),
            blend=jnp.asarray(blend),
            inboard=jnp.asarray(inboard),
            free=tuple(free),
            anchor=anchor,
            hub_chord=blade.hub_chord,
            hub_twist=float(np.radians(blade.hub_twist_deg)),
            cst_active=cst_active,
            min_tc=blade.min_tc,
            aero_stations=tuple(int(i) for i in np.flatnonzero(r >= blade.R_aero)),
            confidence_active=inputs.coeff_method == 3 and inputs.min_confidence > 0,
        )

    def constraint_groups(self, n_cond):
        """
        (name, count) of each constraint group, in the order of constraint_values.
        """
        groups = [("thrust (upper)", n_cond), ("thrust (lower)", n_cond), ("motor power", n_cond)]
        if self.cst_active:
            groups.append(("thickness", len(self.seed["cst"])))
        if self.confidence_active:
            groups.append(("confidence", n_cond))
        return groups

    def to_unit(self, prop):
        '''
        Unit-box variables for the propeller's current design, clipped to the bounds.
        '''
        x = {}
        for name, idx in self.free:
            value = jnp.ravel(getattr(prop, name))[np.array(idx)]
            unit = (value - self.lo[name]) / (self.hi[name] - self.lo[name])
            if bool(jnp.any((unit < 0.0) | (unit > 1.0))):
                warnings.warn(f"Seed {name} lies outside its bounds and has been clipped")
            x[name] = jnp.clip(unit, 0.0, 1.0)
        return x

    def values(self, x):
        '''
        Full physical design arrays (c, beta, cst, omega, tilt) for unit-box variables x.
        '''
        full = dict(self.seed)
        for name, idx in self.free:
            physical = self.lo[name] + x[name] * (self.hi[name] - self.lo[name])
            base = self.seed[name]
            full[name] = base.ravel().at[np.array(idx)].set(physical).reshape(base.shape)

        # Blend-zone planform follows the hub values and the first aerodynamic station
        full["c"], full["beta"] = geometry.apply_planform_blend(
            full["c"], full["beta"], self.blend, self.inboard, self.anchor, self.hub_chord, self.hub_twist
        )
        return full

    def from_unit(self, x, params_struct):
        '''
        Writes the design for unit-box variables x into the batched params_struct.
        '''
        full = self.values(x)
        n_cond = params_struct.omega.shape[0]
        cst_all = jnp.concatenate([self.hub_cst[None, :], full["cst"]])

        # Geometry is shared by every flight condition
        shared = lambda a: jnp.broadcast_to(a, (n_cond,) + a.shape)
        return eqx.tree_at(
            lambda p: (p.c, p.beta, p.cst, p.omega, p.tilt),
            params_struct,
            (shared(full["c"]), shared(full["beta"]), shared(cst_all), full["omega"], full["tilt"])
        )


@dataclass
class Progress:
    '''
    State of an optimisation, reported every Inputs.opt_print_every iterations and once more when solve() ends.
    Per-condition fields are numpy arrays of shape (n_cond,).
    '''
    iteration: int              # iterations completed
    max_iter: int
    objective: float            # energy ratio J: weighted motor power / weighted ideal power
    violation: float            # largest constraint violation, relative to its limit (0 when feasible)
    error: float                # relative change of the solver's merit function in the last iteration
    elapsed: float              # seconds since solve() started
    status: str                 # "running", "converged", "stopped", "max_iter" or "failed"
    message: str                # solver detail for a failed run
    omega: np.ndarray
    thrust: np.ndarray
    thrust_target: np.ndarray
    power: np.ndarray           # motor power (W)
    ideal_power: np.ndarray     # actuator disc power at the thrust target (W)
    confidence: np.ndarray      # load-weighted NeuralFoil confidence over the aerodynamic zone (nan if unused)
    min_station_confidence: np.ndarray  # lowest single-station (load-weighted over psi) confidence (nan if unused)
    constraints: dict           # constraint group -> smallest value (g >= 0 is feasible)
    error_log: np.ndarray       # relative change of the merit function per iteration (seed entry = 1)
    params_struct: object       # batched Params of the current design

    @property
    def thrust_err_pct(self) -> np.ndarray:
        return 100.0 * (self.thrust - self.thrust_target) / self.thrust_target

    @property
    def power_ratio(self) -> np.ndarray:
        return self.power / self.ideal_power

    def condition(self, idx: int = 0):
        '''
        Params of flight condition idx at the current design, carrying the optimiser error log.
        '''
        params = jax.tree_util.tree_map(
            lambda leaf: leaf[idx] if hasattr(leaf, 'ndim') and leaf.ndim > 0 else leaf,
            self.params_struct
        )
        return eqx.tree_at(lambda p: p.error_log, params, jnp.asarray(self.error_log))


class ProgressPrinter:
    '''
    Prints optimisation progress as an aligned table (ASCII only, so any console encoding can show it).

    verbose 0: banner and final summary, 1: + one row group per report, 2: + constraint groups and confidence.
    '''
    HEADER_EVERY = 20       # reports between repeated headers

    def __init__(self, verbose: int = 1):
        self.verbose = verbose
        self.n_reports = 0
        self.tol = None

    def banner(self, solver_name, space, n_cond, max_iter, tol, notes=()):
        self.tol = tol
        counts = [f"{name} {len(idx)}" for name, idx in space.free]
        total = sum(len(idx) for _, idx in space.free)
        n_con = sum(size for _, size in space.constraint_groups(n_cond))
        print(f"\n{solver_name} | design variables: {', '.join(counts)} ({total} total) | {n_con} constraints | "
              f"{n_cond} flight condition{'s' if n_cond != 1 else ''} | max {max_iter} iterations, tol {tol:.1e}", flush=True)
        for note in notes:
            print(note, flush=True)

    def note(self, text):
        print(text, flush=True)

    def report(self, progress: Progress):
        if self.verbose < 1:
            return
        if self.n_reports % self.HEADER_EVERY == 0:
            self._header()
        self.n_reports += 1
        self._rows(progress)

    def summary(self, progress: Progress):
        n = progress.iteration
        if progress.status == "converged":
            reason = f"Converged at iteration {n} (relative tolerance {self.tol:.1e})"
        elif progress.status == "stopped":
            reason = f"Stopped by user after {n} iteration{'s' if n != 1 else ''}"
        elif progress.status == "failed":
            reason = f"Solver stopped at iteration {n}: {progress.message}"
        else:
            reason = f"Reached the iteration limit ({n}) without converging (last change {progress.error:.1e} > tol {self.tol:.1e})"
        feasible = "all constraints satisfied" if progress.violation <= 0 else f"largest constraint violation {100 * progress.violation:.2f}%"
        per_iter = f" ({progress.elapsed / n:.2f} s/iteration)" if n else ""

        print(f"\n{reason}\n{n} iterations in {progress.elapsed:.1f} s{per_iter}; {feasible}\nFinal design:")
        self._header()
        self._rows(progress)
        if self.verbose >= 1:
            self._constraints(progress)
        print(flush=True)

    def _header(self):
        names = (f"{'Iter':>6}  {'Objective':>9}  {'Violation':>9}  {'Change':>8} | {'Cond':>4}  {'Omega':>7}  {'Thrust':>8}  "
                 f"{'Target':>8}  {'T-Err':>7}  {'Power':>8}  {'P/Pid':>6} | {'Time':>7}")
        units = (f"{'':>6}  {'[P/Pid]':>9}  {'[%]':>9}  {'[rel]':>8} | {'':>4}  {'[rad/s]':>7}  {'[N]':>8}  "
                 f"{'[N]':>8}  {'[%]':>7}  {'[W]':>8}  {'':>6} | {'[s]':>7}")
        print(f"\n{names}\n{units}\n{'-' * len(names)}")

    def _rows(self, p: Progress):
        blank = " " * 40      # width of the Iter/Objective/Violation/Change columns
        err_pct, ratio = p.thrust_err_pct, p.power_ratio
        for idx in range(len(p.omega)):
            lead = f"{p.iteration:>6}  {p.objective:>9.5f}  {100 * p.violation:>9.3f}  {p.error:>8.1e}" if idx == 0 else blank
            time_col = f"{p.elapsed:>7.1f}" if idx == 0 else ""
            print(f"{lead} | {'C' + str(idx):>4}  {p.omega[idx]:>7.1f}  {p.thrust[idx]:>8.3f}  {p.thrust_target[idx]:>8.3f}  "
                  f"{err_pct[idx]:>+7.2f}  {p.power[idx]:>8.1f}  {ratio[idx]:>6.3f} | {time_col}".rstrip())
        if self.verbose >= 2:
            self._constraints(p)
        sys.stdout.flush()

    def _constraints(self, p: Progress):
        blank = " " * 40
        groups = "  ".join(f"{name} {100 * value:+.2f}%" for name, value in p.constraints.items())
        print(f"{blank} | margins (min over group): {groups}")
        if np.all(np.isfinite(p.confidence)):
            print(f"{blank} | load-weighted NeuralFoil confidence (lowest station): "
                  + "  ".join(f"C{i} {c:.3f} ({m:.3f})" for i, (c, m) in enumerate(zip(p.confidence, p.min_station_confidence))))


class _StopRequested(Exception):
    '''Raised inside a solver callback when the stop event is set.'''


class _BudgetUsed(Exception):
    '''Raised when phase 1 used the whole iteration budget, so SLSQP does not start.'''


class BaseSolver(ABC):
    name = "Solver"

    def __init__(self, prop, callback=None, stop_event=None):
        '''
        callback(progress) receives a Progress at every report and when solve() ends.
        stop_event, when set, ends solve() after the current iteration (defaults to the Ctrl+C event).
        '''
        self.prop = prop
        self.inputs = prop.inputs
        self.callback = callback
        self.stop_event = _stop_event if stop_event is None else stop_event
        self.space, self.params_struct, self.vars_scaled, self.bounds = self._prepare_problem()

    def _prepare_problem(self):
        """Universal setup: builds the design space, batched params_struct and optimization bounds."""
        space = DesignSpace.build(self.prop)
        vars_scaled = space.to_unit(self.prop)

        # Start from the design space's view of the blade (blend zone applied to the seed)
        params_struct = space.from_unit(vars_scaled, self.prop.batched())

        lower_bounds = jax.tree_util.tree_map(jnp.zeros_like, vars_scaled)
        upper_bounds = jax.tree_util.tree_map(jnp.ones_like, vars_scaled)
        bounds = (lower_bounds, upper_bounds)

        return space, params_struct, vars_scaled, bounds

    @abstractmethod
    def solve(self):
        """Runs the optimisation and returns the optimised Propeller."""

    ## Shared helpers for the solver loops
    def _begin(self, notes=()):
        if self.stop_event is _stop_event:
            # The shared Ctrl+C event may still be set by a previous solve; a caller's own event is left alone
            _stop_event.clear()
        inputs = self.inputs
        self.printer = ProgressPrinter(inputs.opt_verbose)
        self.printer.banner(self.name, self.space, inputs.n_cond, inputs.opt_max_iter, inputs.opt_tol, notes)
        self.printer.note("Compiling the solver (the first iteration can take a while)...")
        self.start_time = time.time()
        self.last_report = time.time()

    def _should_report(self, iteration):
        return (iteration == 1 or iteration % self.inputs.opt_print_every == 0
                or time.time() - self.last_report > 10.0)

    def _report(self, iteration, vars_scaled, params_struct, error_log, status="running", message=""):
        progress = self._progress(iteration, vars_scaled, params_struct, error_log, status, message)
        if status == "running":
            self.printer.report(progress)
        else:
            self.printer.summary(progress)
        if self.callback is not None:
            self.callback(progress)
        self.last_report = time.time()
        return progress

    def _progress(self, iteration, vars_scaled, params_struct, error_log, status, message) -> Progress:
        params_struct = self.space.from_unit(vars_scaled, params_struct)
        outputs = evaluate_all_conditions(params_struct)
        g = relative_margins(constraint_values(outputs, params_struct, self.space), self.space, self.inputs.n_cond)
        groups, start = {}, 0
        for name, size in self.space.constraint_groups(self.inputs.n_cond):
            groups[name] = float(g[start:start + size].min())
            start += size

        as_np = lambda a: np.asarray(a, dtype=float)
        if self.space.confidence_active:
            aero = np.array(self.space.aero_stations)
            log_load, log_load_conf = as_np(outputs.log_load)[:, aero], as_np(outputs.log_load_confidence)[:, aero]
            confidence = np.exp(logsumexp(log_load_conf, axis=1) - logsumexp(log_load, axis=1))
            min_station = np.exp((log_load_conf - log_load).min(axis=1))
        else:
            confidence = min_station = np.full(self.inputs.n_cond, np.nan)
        return Progress(
            iteration=iteration,
            max_iter=self.inputs.opt_max_iter,
            objective=float(energy_ratio(outputs, params_struct)),
            violation=float(max(0.0, -g.min())),
            error=float(error_log[-1]),
            elapsed=time.time() - self.start_time,
            status=status,
            message=message,
            omega=as_np(params_struct.omega),
            thrust=as_np(outputs.thrust),
            thrust_target=as_np(params_struct.thrust_target),
            power=as_np(outputs.power),
            ideal_power=as_np(outputs.ideal_power),
            confidence=confidence,
            min_station_confidence=min_station,
            constraints=groups,
            error_log=np.asarray(error_log, dtype=float),
            params_struct=params_struct,
        )

    def _result(self, vars_scaled, params_struct, error_log, time_log):
        # Geometry is shared by every condition; operating state is kept per condition
        params_struct = self.space.from_unit(vars_scaled, params_struct)
        return self.prop.with_state(
            c=params_struct.c[0],
            beta=params_struct.beta[0],
            cst=params_struct.cst[0, 1:],
            omega=params_struct.omega,
            tilt=params_struct.tilt,
            V_i_0=params_struct.V_i_0,
            error_log=jnp.array(error_log),
            time_log=jnp.array(time_log),
        )


def _relative_change(previous, current):
    '''
    |previous - current| relative to the larger magnitude (the objective is dimensionless and O(1)).
    '''
    return abs(previous - current) / max(abs(previous), abs(current), np.finfo(float).tiny)


class LBFGSBSolver(BaseSolver):
    '''
    Bound-constrained quasi-Newton on the penalised cost (total_cost). Constraints enter as quadratic penalties
    weighted by Inputs.w_thrust / w_thickness / w_confidence; use SLSQP to treat them exactly.
    Converged when the penalised cost changes by less than Inputs.opt_tol (relative) in one iteration.
    '''
    name = "L-BFGS-B"

    def __init__(self, prop, callback=None, stop_event=None):
        super().__init__(prop, callback, stop_event)
        self.solver = jaxopt.LBFGSB(
            fun=total_cost,
            has_aux=True,
            implicit_diff=True,
            jit=True,
            verbose=self.inputs.opt_verbose >= 2
        )

    def init_state(self):
        return self.solver.init_state(
            init_params=self.vars_scaled,
            bounds=self.bounds,
            params_struct=self.params_struct,
            space=self.space,
        )

    def step(self, vars_scaled, state, params_struct):
        vars_scaled, state = self.solver.update(
            vars_scaled,
            state=state,
            bounds=self.bounds,
            params_struct=params_struct,
            space=self.space,
        )
        # JAXopt specific auxiliary extraction
        V_i_0 = state.aux
        return vars_scaled, state, V_i_0

    def solve(self):
        """The main optimization loop returning the optimised Propeller."""
        self._begin()
        inputs = self.inputs
        state = self.init_state()
        vars_scaled = self.vars_scaled
        params_struct = self.params_struct

        error_log, time_log = [1.0], [0.0]
        status, message, iteration = "max_iter", "", 0
        previous = float(state.value)

        for i in range(inputs.opt_max_iter):
            if self.stop_event.is_set():
                status = "stopped"
                break

            iter_start = time.time()
            candidate, state, V_i_0 = self.step(vars_scaled, state, params_struct)
            if bool(state.failed_linesearch) or not np.isfinite(float(state.value)):
                status, message = "failed", "the line search could not reduce the cost"
                break
            vars_scaled, iteration = candidate, i + 1

            # Keep the solved induced velocity as the warm start for the next step
            V_i_0 = jnp.where(jnp.isfinite(V_i_0), V_i_0, params_struct.V_i_0)
            params_struct = eqx.tree_at(lambda p: p.V_i_0, params_struct, V_i_0)

            current = float(state.value)
            error_log.append(_relative_change(previous, current))
            time_log.append(time.time() - iter_start)
            previous = current

            if error_log[-1] < inputs.opt_tol:
                status = "converged"
                break
            if self._should_report(iteration):
                self._report(iteration, vars_scaled, params_struct, error_log)

        self._report(iteration, vars_scaled, params_struct, error_log, status, message)
        return self._result(vars_scaled, params_struct, error_log, time_log)


def curvature_scale(vars_scaled, params_struct, space):
    '''
    Jacobi scaling of the design variables: s_j = 1 / sqrt|d2J/dx_j2| of the energy ratio at vars_scaled, so a unit
    step in every scaled variable changes the objective by the same amount to second order. Computed once from
    the exact Hessian diagonal (one Hessian-vector product per variable) and normalised to a geometric mean of 1,
    so it changes the shape of the problem but not its overall size. Variables with no curvature (to machine
    precision) keep the largest scale.
    '''
    x0, unravel = ravel_pytree(vars_scaled)
    grad = jax.grad(lambda x: problem_values(unravel(x), params_struct, space)[0])
    hessian_column = lambda e: jax.jvp(grad, (x0,), (e,))[1]
    diag = jnp.abs(jnp.diagonal(jax.jit(jax.vmap(hessian_column))(jnp.eye(x0.size))))
    diag = np.asarray(diag)
    floor = diag.max() * np.finfo(diag.dtype).eps if diag.max() > 0 else 1.0
    scale = 1.0 / np.sqrt(np.maximum(diag, floor))
    return scale / np.exp(np.mean(np.log(scale)))


class SLSQPSolver(BaseSolver):
    '''
    Sequential quadratic programming (SciPy SLSQP) on the physically normalised problem: minimise the energy ratio
    subject to constraint_values >= 0, with no penalty weights.

    Phase 1 (only when the seed violates a constraint by more than Inputs.opt_tol) restores feasibility by
    minimising the sum of squared relative violations with bound-constrained L-BFGS-B. Phase 2 runs SLSQP from
    there, with the design variables Jacobi-scaled by the objective curvature at that point (curvature_scale), so
    SLSQP's initial identity Hessian has the right size in every direction. When a subproblem breaks down it restarts
    from the best feasible design, as long as the previous attempt improved it.
    Converged when the objective changes by less than Inputs.opt_tol with the constraints satisfied.
    '''
    name = "SLSQP"

    def _restore_feasibility(self, vars_scaled, params_struct, error_log, time_log):
        '''
        Phase 1: bound-constrained descent on infeasibility() until it is within opt_tol**2 (every violation below
        opt_tol), it stops decreasing, or the stop event is set. Returns (vars_scaled, params_struct, iterations).
        '''
        inputs = self.inputs
        solver = jaxopt.LBFGSB(fun=infeasibility, has_aux=True, implicit_diff=True, jit=True)
        state = solver.init_state(init_params=vars_scaled, bounds=self.bounds, params_struct=params_struct, space=self.space)
        previous, iteration = float(state.value), 0
        while (float(state.value) > inputs.opt_tol**2 and iteration < inputs.opt_max_iter
               and not self.stop_event.is_set()):
            iter_start = time.time()
            candidate, state = solver.update(vars_scaled, state=state, bounds=self.bounds,
                                             params_struct=params_struct, space=self.space)
            if bool(state.failed_linesearch) or not np.isfinite(float(state.value)):
                break
            vars_scaled, iteration = candidate, iteration + 1
            params_struct = eqx.tree_at(lambda p: p.V_i_0, params_struct,
                                        jnp.where(jnp.isfinite(state.aux), state.aux, params_struct.V_i_0))
            error_log.append(_relative_change(previous, float(state.value)))
            time_log.append(time.time() - iter_start)
            previous = float(state.value)
            if self._should_report(iteration):
                self._report(iteration, vars_scaled, params_struct, error_log)
            if error_log[-1] < inputs.opt_tol:
                break
        return vars_scaled, params_struct, iteration

    def solve(self):
        from scipy.optimize import minimize

        inputs, space = self.inputs, self.space
        self._begin()
        error_log, time_log = [1.0], [0.0]

        # Phase 1: reach the feasible region first when the seed is far outside it
        vars_scaled, params_struct, n_phase1 = self.vars_scaled, self.params_struct, 0
        _, g0, _ = problem_values(vars_scaled, params_struct, space)
        if float(jnp.max(jax.nn.relu(-g0))) > inputs.opt_tol:
            worst = max(0.0, -relative_margins(g0, space, inputs.n_cond).min())
            self.printer.note(f"Phase 1: the seed violates a constraint by {100 * worst:.1f}%; restoring feasibility...")
            vars_scaled, params_struct, n_phase1 = self._restore_feasibility(vars_scaled, params_struct, error_log, time_log)
            _, g1, _ = problem_values(vars_scaled, params_struct, space)
            worst = max(0.0, -relative_margins(g1, space, inputs.n_cond).min())
            self.printer.note(f"Phase 1 done after {n_phase1} iterations (largest violation {100 * worst:.2f}%)")

        # Phase 2: SLSQP in curvature-scaled variables
        x0, unravel = ravel_pytree(vars_scaled)
        t = time.time()
        remaining = inputs.opt_max_iter - n_phase1
        if self.stop_event.is_set() or remaining <= 0:
            s = np.ones(x0.size)
        else:
            s = curvature_scale(vars_scaled, params_struct, space)
            self.printer.note(f"Phase 2: SLSQP with the design variables scaled by the objective curvature ({time.time() - t:.1f} s)")

        def values(x, params_struct):
            J, g, V_i_0 = problem_values(unravel(x), params_struct, space)
            v = jnp.concatenate([J[None], g])
            return v, (v, V_i_0)
        values_and_jacobian = jax.jit(jax.jacrev(values, has_aux=True))

        state = dict(params_struct=params_struct, key=None)
        def evaluate(y):
            # One fused evaluation per new point (SciPy asks for the objective, constraints and their Jacobians
            # separately); the solved induced velocity becomes the next warm start
            key = y.tobytes()
            if key != state["key"]:
                jac, (v, V_i_0) = values_and_jacobian(jnp.asarray(y * s), state["params_struct"])
                ps = state["params_struct"]
                state["params_struct"] = eqx.tree_at(lambda p: p.V_i_0, ps, jnp.where(jnp.isfinite(V_i_0), V_i_0, ps.V_i_0))
                state.update(key=key, v=np.asarray(v), jac=np.asarray(jac) * s[None, :])
            return state["v"], state["jac"]

        feasible = lambda v: max(0.0, -v[1:].min()) <= inputs.opt_tol
        y0 = np.asarray(x0) / s
        v0, _ = evaluate(y0)
        track = dict(iteration=n_phase1, y=y0, previous=float(v0[0]), iter_start=time.time(),
                     best=y0 if feasible(v0) else None, best_objective=float(v0[0]) if feasible(v0) else np.inf)

        def callback(y):
            v, _ = evaluate(y)
            track["iteration"] += 1
            track["y"] = np.array(y)
            error_log.append(_relative_change(track["previous"], v[0]))
            time_log.append(time.time() - track["iter_start"])
            track["previous"], track["iter_start"] = v[0], time.time()
            if feasible(v) and v[0] < track["best_objective"]:
                track["best"], track["best_objective"] = np.array(y), v[0]
            if self.stop_event.is_set():
                raise _StopRequested
            if self._should_report(track["iteration"]):
                self._report(track["iteration"], unravel(jnp.asarray(y * s)), state["params_struct"], error_log)

        # SciPy's SLSQP has no elastic mode: when a linearisation of the constraints is incompatible, or its line
        # search fails, it stops. It then restarts from the best feasible design with a fresh (identity) Hessian,
        # where the first subproblem is always compatible (d = 0 satisfies g + J d >= 0). Restarts continue only
        # while each attempt improves the best feasible objective, so they end.
        status, message, y = "max_iter", "", track["y"]
        try:
            if self.stop_event.is_set():
                raise _StopRequested
            while True:
                remaining = inputs.opt_max_iter - track["iteration"]
                if remaining <= 0:
                    raise _BudgetUsed
                best_before = track["best_objective"]
                result = minimize(
                    lambda y: evaluate(y)[0][0], y, jac=lambda y: evaluate(y)[1][0], method="SLSQP",
                    bounds=list(zip(np.zeros_like(s), 1.0 / s)),
                    constraints=[{"type": "ineq", "fun": lambda y: evaluate(y)[0][1:], "jac": lambda y: evaluate(y)[1][1:]}],
                    callback=callback, options=dict(maxiter=remaining, ftol=inputs.opt_tol),
                )
                y = result.x
                if result.success:
                    status = "converged"
                    break
                if track["iteration"] >= inputs.opt_max_iter:
                    break
                message = result.message
                if track["best"] is None or not track["best_objective"] < best_before:
                    status = "failed"
                    break
                self.printer.note(f"SLSQP: {message} at iteration {track['iteration']}; restarting from the best "
                                  f"feasible design (objective {track['best_objective']:.5f})")
                y, message = track["best"], ""
                track["previous"] = track["best_objective"]
        except _StopRequested:
            status, y = "stopped", track["y"]
        except _BudgetUsed:
            y = track["y"]

        # An SQP iterate may be infeasible when the run ends early; prefer the best feasible design visited
        v, _ = evaluate(y)
        if not feasible(v) and track["best"] is not None:
            y = track["best"]
            message = (message + "; " if message else "") + "returned the best feasible design found"
        vars_scaled = unravel(jnp.asarray(y * s))
        self._report(track["iteration"], vars_scaled, state["params_struct"], error_log, status, message)
        return self._result(vars_scaled, state["params_struct"], error_log, time_log)


# Outer-loop solvers selectable through Inputs.opt_solver
SOLVERS = {"SLSQP": SLSQPSolver, "L-BFGS-B": LBFGSBSolver}

def run_optimization(prop, update_queue=None, callback=None, stop_event=None):
    '''
    Optimises the propeller's design variables for its Inputs; returns the optimised Propeller.

    callback(progress) receives a Progress at every report and when the solve ends. update_queue (for the
    matplotlib live plot) receives the Params of condition 0 at the same points. stop_event, when set,
    ends the solve after the current iteration (defaults to the Ctrl+C event).
    '''
    solver_cls = SOLVERS.get(prop.inputs.opt_solver)
    if solver_cls is None:
        raise ValueError(f"Solver {prop.inputs.opt_solver} not implemented.")

    if update_queue is not None:
        user_callback = callback

        def callback(progress):
            update_queue.put(progress.condition(0))
            if user_callback is not None:
                user_callback(progress)

    solver = solver_cls(prop, callback=callback, stop_event=stop_event)
    return solver.solve()

if __name__ == "__main__":
    ## Testing of optimiser functions
    from init import Propeller

    # generate basic propeller
    prop = Propeller.default()

    weighted_cost, V_i_0 = condition_cost(prop.condition(0))
    print(weighted_cost)

    solver = LBFGSBSolver(prop)
    costs, V_i_0 = total_cost(solver.vars_scaled, solver.params_struct, solver.space)
    print(costs)
    print(V_i_0)

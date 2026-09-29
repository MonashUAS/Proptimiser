# External Imports
import jax
import time
from dataclasses import dataclass
import sys
import signal
import threading
import warnings
import numpy as np
import jax.numpy as jnp
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
    '''
    Return the Base Objective Values of the Solver given an input of parameters:
    motor power, thrust and the solved induced velocity V_i_0 for one flight condition.
    '''
    V_i, V_i_0   = propmodel.induced_velocity(params)             # Solve induced velocity field for current guess
    thrust = jnp.squeeze(propmodel.thrust(V_i, params))            # return the total thrust the propeller produce
    torque = jnp.squeeze(propmodel.torque(V_i, params))            # return total torque propeller produces

    # calculate the motor power
    cfg = params.cfg
    motor_power = propmodel.motorPower(torque, params.omega, I_0 = cfg.I_0, R_m = cfg.R_m, kV = cfg.kV)

    # Return V_i_0 so we can catch it!
    return motor_power, thrust, V_i_0

def confidence_penalty(V_i_0, params):
    '''
    Penalises NeuralFoil analysis confidence below cfg.min_confidence at the aerodynamic stations (r >= r_aero),
    keeping the optimised sections and their operating angles inside the network's trusted region.
    '''
    cfg = params.cfg
    if cfg.coeff_method != 3 or cfg.w_confidence == 0:
        return jnp.zeros(())

    V_i = airflow.inflow_model(V_i_0, params)
    _, _, _, confidence = airfoil.aero_coeffs(V_i, params, return_confidence=True)
    shortfall = jax.nn.relu(cfg.min_confidence - confidence) ** 2

    # Hub and blend-zone sections are fixed, so only the aerodynamic zone is penalised
    aero = (params.r >= params.r_cst[1])[:, None]
    return cfg.w_confidence * jnp.sum(jnp.where(aero, shortfall, 0.0)) / (jnp.sum(aero) * shortfall.shape[1])

def condition_terms(params):
    '''
    Unweighted cost terms (thrust, power, confidence) for exactly ONE condition,
    along with (motor power, thrust, V_i_0).
    '''
    # unpack objectives and induced velocity
    motor_power, thrust, V_i_0 = objectives(params)
    cfg = params.cfg
    # Thrust constraint: zero cost inside the tolerance band, steep quadratic penalty outside it
    thrust_err_rel = (thrust - params.thrust_target)/params.thrust_target
    thrust_violation = jax.nn.relu(jnp.abs(thrust_err_rel) - cfg.thrust_tol)
    thrust_penalty = cfg.w_thrust * thrust_violation**2

    # Power objective: minimised freely once the thrust constraint is satisfied
    power_rel = jnp.abs(motor_power/cfg.P_max)

    terms = (thrust_penalty, power_rel, confidence_penalty(V_i_0, params))
    return terms, (motor_power, thrust, V_i_0)

def condition_cost(params):
    '''
    Computes cost and returns updated V_i_0 for exactly ONE condition
    '''
    terms, (_, _, V_i_0) = condition_terms(params)
    cost = sum(terms)

    # Scale by the relative importance weight of this flight condition
    weighted_cost = cost * params.objective_weight
    return weighted_cost, V_i_0

@jax.jit
def total_cost(vars_scaled, params_struct, space):
    '''
    Total cost of all n conditions specified in Inputs, plus the airfoil thickness penalty.
    '''
    # Update parameters struct with new geometry and control points
    params_struct = space.from_unit(vars_scaled, params_struct)

    # vmap cost function over parameters list
    cost_fn_vmap = jax.vmap(condition_cost, in_axes=0)
    costs, V_i_0 = cost_fn_vmap(params_struct)

    aux = (V_i_0)
    return jnp.sum(costs) + space.thickness_penalty(params_struct), aux

@jax.jit
def evaluate_all_conditions(params_struct):
    '''
    Helper Function to evaluate cost terms and objectives for all conditions.
    '''
    vmapped_terms = jax.vmap(condition_terms, in_axes=0)
    return vmapped_terms(params_struct)


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
    w_thickness: float = eqx.field(static=True)

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
            w_thickness=inputs.w_thickness,
        )

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

    def thickness_penalty(self, params_struct):
        '''
        Penalises aerodynamic sections thinner than min_tc (zero while every section is thick enough).
        '''
        if not self.cst_active or self.w_thickness == 0:
            return jnp.zeros(())
        t_over_c = geometry.max_thickness(params_struct.cst[0, 1:])
        return self.w_thickness * jnp.sum(jax.nn.relu((self.min_tc - t_over_c) / self.min_tc) ** 2)


@dataclass
class Progress:
    '''
    State of an optimisation, reported every Inputs.opt_print_every iterations and once more when solve() ends.
    Per-condition fields are numpy arrays of shape (n_cond,).
    '''
    iteration: int              # iterations completed
    max_iter: int
    cost: float                 # total cost of the current design
    error: float                # solver stopping criterion (projected gradient norm for L-BFGS-B)
    elapsed: float              # seconds since solve() started
    status: str                 # "running", "converged", "stopped" or "max_iter"
    omega: np.ndarray
    thrust: np.ndarray
    thrust_target: np.ndarray
    power: np.ndarray           # motor power (W)
    cost_terms: dict            # unweighted per-condition cost terms: "thrust", "power", "confidence"
    thickness_penalty: float
    error_log: np.ndarray       # stopping criterion at the seed and after every iteration
    params_struct: object       # batched Params of the current design

    @property
    def thrust_err_pct(self) -> np.ndarray:
        return 100.0 * (self.thrust - self.thrust_target) / self.thrust_target

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

    verbose 0: banner and final summary, 1: + one row group per report, 2: + cost terms of every condition.
    '''
    HEADER_EVERY = 20       # reports between repeated headers

    def __init__(self, verbose: int = 1, error_name: str = "OptErr"):
        self.verbose = verbose
        self.error_name = error_name
        self.n_reports = 0
        self.tol = None

    def banner(self, solver_name, space, n_cond, max_iter, tol):
        self.tol = tol
        counts = [f"{name} {len(idx)}" for name, idx in space.free]
        total = sum(len(idx) for _, idx in space.free)
        print(f"\n{solver_name} | design variables: {', '.join(counts)} ({total} total) | "
              f"{n_cond} flight condition{'s' if n_cond != 1 else ''} | max {max_iter} iterations, tol {tol:.1e}", flush=True)

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
            reason = f"Converged at iteration {n}: {self.error_name} {progress.error:.2e} < tol {self.tol:.1e}"
        elif progress.status == "stopped":
            reason = f"Stopped by user after {n} iteration{'s' if n != 1 else ''}"
        else:
            reason = f"Reached the iteration limit ({n}) without converging: {self.error_name} {progress.error:.2e} > tol {self.tol:.1e}"
        per_iter = f" ({progress.elapsed / n:.2f} s/iteration)" if n else ""

        print(f"\n{reason}\n{n} iterations in {progress.elapsed:.1f} s{per_iter}\nFinal design:")
        self._header()
        self._rows(progress)
        print(flush=True)

    def _header(self):
        names = f"{'Iter':>6}  {'Cost':>11}  {self.error_name:>9} | {'Cond':>4}  {'Omega':>8}  {'Thrust':>8}  {'Target':>8}  {'T-Err':>7}  {'Power':>8} | {'Time':>7}"
        units = f"{'':>6}  {'':>11}  {'':>9} | {'':>4}  {'[rad/s]':>8}  {'[N]':>8}  {'[N]':>8}  {'[%]':>7}  {'[W]':>8} | {'[s]':>7}"
        print(f"\n{names}\n{units}\n{'-' * len(names)}")

    def _rows(self, p: Progress):
        blank = " " * 30      # width of the Iter/Cost/error columns
        err_pct = p.thrust_err_pct
        for idx in range(len(p.omega)):
            lead = f"{p.iteration:>6}  {p.cost:>11.4e}  {p.error:>9.2e}" if idx == 0 else blank
            time_col = f"{p.elapsed:>7.1f}" if idx == 0 else ""
            print(f"{lead} | {'C' + str(idx):>4}  {p.omega[idx]:>8.1f}  {p.thrust[idx]:>8.3f}  {p.thrust_target[idx]:>8.3f}  "
                  f"{err_pct[idx]:>+7.2f}  {p.power[idx]:>8.1f} | {time_col}".rstrip())
            if self.verbose >= 2:
                terms = "  ".join(f"{name} {values[idx]:.2e}" for name, values in p.cost_terms.items())
                print(f"{blank} |       cost terms: {terms}")
        if self.verbose >= 2:
            print(f"{blank} |       thickness penalty: {p.thickness_penalty:.2e}")
        sys.stdout.flush()


class BaseSolver(ABC):
    name = "Solver"
    error_name = "OptErr"       # label of the solver's stopping criterion (state.error)

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
    def init_state(self):
        """Initializes and returns the solver-specific state."""
        pass

    @abstractmethod
    def step(self, vars_scaled, state, params_struct):
        """Performs a single update step. Must return (new_vars, new_state, aux_data)."""
        pass

    def solve(self):
        """The main optimization loop returning the optimised Propeller."""
        if self.stop_event is _stop_event:
            # The shared Ctrl+C event may still be set by a previous solve; a caller's own event is left alone
            _stop_event.clear()

        inputs = self.inputs
        printer = ProgressPrinter(inputs.opt_verbose, self.error_name)
        printer.banner(self.name, self.space, inputs.n_cond, inputs.opt_max_iter, inputs.opt_tol)
        printer.note("Compiling the solver (the first iteration can take a while)...")

        start_time = time.time()
        state = self.init_state()
        vars_scaled = self.vars_scaled
        params_struct = self.params_struct

        error_log = [float(getattr(state, 'error', 1.0))]
        time_log = [0.0]
        last_report = time.time()
        status = "max_iter"

        for i in range(inputs.opt_max_iter):
            if self.stop_event.is_set():
                status = "stopped"
                break

            iter_start = time.time()

            # Step delegated to the specific solver
            vars_scaled, state, V_i_0 = self.step(vars_scaled, state, params_struct)

            # Update and rescale params
            params_struct = self.space.from_unit(vars_scaled, params_struct)

            # Keep the solved induced velocity as the warm start for the next step
            V_i_0 = jnp.where(jnp.isfinite(V_i_0), V_i_0, params_struct.V_i_0)
            params_struct = eqx.tree_at(lambda p: p.V_i_0, params_struct, V_i_0)

            error_log.append(float(getattr(state, 'error', 1.0)))
            time_log.append(time.time() - iter_start)

            if error_log[-1] < inputs.opt_tol:
                status = "converged"
                break

            iteration = i + 1
            if iteration == 1 or iteration % inputs.opt_print_every == 0 or time.time() - last_report > 10.0:
                progress = self._progress(iteration, state, params_struct, error_log, start_time, "running")
                printer.report(progress)
                self._notify(progress)
                last_report = time.time()

        final = self._progress(len(error_log) - 1, state, params_struct, error_log, start_time, status)
        printer.summary(final)
        self._notify(final)

        # Geometry is shared by every condition; operating state is kept per condition
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

    def _progress(self, iteration, state, params_struct, error_log, start_time, status) -> Progress:
        # Calculate the per-condition objectives of the current design
        (thrust_terms, power_terms, confidence_terms), (powers, thrusts, _) = evaluate_all_conditions(params_struct)
        as_np = lambda a: np.asarray(a, dtype=float)
        return Progress(
            iteration=iteration,
            max_iter=self.inputs.opt_max_iter,
            cost=float(getattr(state, 'value', np.nan)),
            error=error_log[-1],
            elapsed=time.time() - start_time,
            status=status,
            omega=as_np(params_struct.omega),
            thrust=as_np(thrusts),
            thrust_target=as_np(params_struct.thrust_target),
            power=as_np(powers),
            cost_terms=dict(thrust=as_np(thrust_terms), power=as_np(power_terms), confidence=as_np(confidence_terms)),
            thickness_penalty=float(self.space.thickness_penalty(params_struct)),
            error_log=np.asarray(error_log),
            params_struct=params_struct,
        )

    def _notify(self, progress):
        if self.callback is not None:
            self.callback(progress)

class LBFGSBSolver(BaseSolver):
    name = "L-BFGS-B"
    error_name = "ProjGrad"

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

# Outer-loop solvers selectable through Inputs.opt_solver
SOLVERS = {"L-BFGS-B": LBFGSBSolver}

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

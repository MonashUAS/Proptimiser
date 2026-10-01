import jax
# Set jax debugging
jax.config.update("jax_traceback_filtering", "off")
jax.config.update("jax_check_tracer_leaks", "on")
jax.config.update("jax_disable_jit", False)
jax.config.update("jax_debug_nans", False)
jax.config.update("jax_enable_x64", True)

import dataclasses
import json
import os
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import numpy as np

# Internal Imports
import geometry

PROP_DIR = Path(__file__).resolve().parent / "propellers"
FORMAT_VERSION = 3

# Inputs whose meaning changed in format 3 (Newton BEMT solve, relative-change stopping test): older files take
# the new defaults for these
_RESET_FROM_V2 = ("V_i_0_eps", "V_i_0_max_iter", "opt_tol")

# Inputs fields holding one entry per flight condition
CONDITION_FIELDS = ("thrust_target", "V_x", "V_yz", "omega", "tilt", "solve_omega", "solve_tilt", "objective_weight")

# Propeller arrays saved in the "state" section of a propeller file
STATE_FIELDS = ("c", "beta", "cst", "omega", "tilt", "V_i_0", "error_log", "time_log")


def _array(values, dtype=None):
    return lambda: jnp.array(values, dtype=dtype)


@dataclass(eq=False)
class Inputs:
    '''
    Run configuration: solver settings, discretisation, model selection and flight conditions.
    Every field is saved with the propeller and reaches the physics through Params.cfg.
    '''
    # Outer-loop Optimisation
    opt_solver: str = "SLSQP"       # "SLSQP": exact constraints (recommended) | "L-BFGS-B": penalty weights below
    opt_max_iter: int = 2000
    opt_tol: float = 1e-6           # converged when the objective changes by less than this fraction between iterations
    opt_verbose: int = 1            # 0: start/end summary only, 1: iteration table, 2: + cost terms and jaxopt line-search info
    opt_print_every: int = 5        # iterations between table rows (a row is also printed after 10 s without one)

    # Induced Velocity Settings
    V_i_0_max_iter: int = 50        # Newton iterations (typically 3-7 from a warm start)
    V_i_0_eps: float = 1e-8         # largest non-dimensional BEMT residual accepted (thrust / T_target, torque / (T_target R_tip))
    V_i_0_verbose: int = 0          # 0: silent, 1: one line per BEMT solve, 2: + every Newton iteration and sectional thrust

    # Solver Type
    bemt_type: tuple = (True, True, False) # (V_ia,V_it,V_ir)

    # Discretisation settings
    n_r: int = 8
    n_psi: int = 12

    # Diagnostics
    jax_tracing: bool = False

    # Flight Conditions (one entry per condition)
    thrust_target: jnp.ndarray = dataclasses.field(default_factory=_array([15.9, 60.0]))
    V_x: jnp.ndarray = dataclasses.field(default_factory=_array([18.0, 0.8]))
    V_yz: jnp.ndarray = dataclasses.field(default_factory=_array([3.0, 3.0]))
    omega: jnp.ndarray = dataclasses.field(default_factory=_array([588.0, 650.0]))
    tilt: jnp.ndarray = dataclasses.field(default_factory=_array([0.0, 0.0]))                     #Angular offset of blade (for CSU)
    solve_omega: jnp.ndarray = dataclasses.field(default_factory=_array([0, 0], dtype=bool))      #0: Fixed Omega, 1: Vary Omega
    solve_tilt: jnp.ndarray = dataclasses.field(default_factory=_array([0, 0], dtype=bool))       #0: Fixed Beta, 1: Vary Beta
    objective_weight: jnp.ndarray = dataclasses.field(default_factory=_array([0.9, 0.1]))

    # Motor Information
    I_0: float = 2.0
    R_m: float = 0.037
    kV: float = 220
    P_max: float = 3300

    # Cost Penalties
    # Constraint limits (every solver) and penalty weights (L-BFGS-B only; SLSQP treats the constraints exactly)
    w_thrust: float = 1e4           # weight of the thrust and motor power penalties (large => near-hard constraint)
    thrust_tol: float = 0.01        # relative thrust error allowed around each target
    w_thickness: float = 10.0       # weight of the minimum t/c penalty (BladeSpec.min_tc)
    w_confidence: float = 1.0       # weight of the NeuralFoil low-confidence penalty
    w_smooth: float = 1e-2          # weight of the chord smoothness penalty (all solvers; 0 disables)
    min_confidence: float = 0.9     # lowest load-weighted NeuralFoil confidence over the aerodynamic zone (0 disables)

    # Aerodynamic Models Selection
    inflow_model: int = 0   #0: Simple 1: Pitt-Peters
    coeff_method: int = 3 # 0: Parametric, 1: 2D Interp, 2: 3D Interp, 3: NeuralFoil
    tip_loss_model: int = 0 # 0: No Loss 1: Prandtl
    root_loss_model: int = 0 # 0: No Loss 1: Prandtl
    rot_corr_model: int = 0 # 3D inflow correction model
    high_induction_model: int = 0 # (0: Classical Momentum, 1: Buhl Correction) NOT USEFUL FOR REGULAR FLIGHT

    def __post_init__(self):
        self.opt_verbose, self.V_i_0_verbose = int(self.opt_verbose), int(self.V_i_0_verbose)
        self.opt_print_every = max(1, int(self.opt_print_every))
        self.bemt_type = tuple(bool(b) for b in self.bemt_type)
        if len(self.bemt_type) != 3 or not any(self.bemt_type):
            raise ValueError(f"bemt_type must select at least one of (V_ia, V_it, V_ir), got {self.bemt_type}")

        for name in CONDITION_FIELDS:
            dtype = bool if name.startswith("solve_") else float
            setattr(self, name, jnp.asarray(getattr(self, name), dtype=dtype))

        shapes = {name: getattr(self, name).shape for name in CONDITION_FIELDS}
        if any(len(s) != 1 for s in shapes.values()) or len({s[0] for s in shapes.values()}) != 1:
            raise ValueError(f"Flight-condition fields must be 1D arrays of equal length, got shapes {shapes}")

    @property
    def n_cond(self) -> int:
        return int(self.thrust_target.shape[0])


@dataclass
class BladeSpec:
    '''
    Blade design definition.

    r_foils = [0, r_hub, r_aero, r_mid..., r_tip] (m) splits the blade into zones:
        0      <= r <  r_hub  : structural hub section (stadium), outside the BEMT domain
        r_hub  <= r <  r_aero : blend zone; chord/twist and section shape blend from the hub to the aero zone
        r_aero <= r <= r_tip  : aerodynamic zone; CST sections defined at r_foils[2:] and interpolated linearly
    The BEMT grid spans [r_hub, r_tip], but only the aerodynamic zone is solved and carries force: the blend zone is a
    structural transition (like a root cutout) whose chord and twist follow from the hub and the first aero station.

    freeze holds radial ranges (m) where a design variable keeps its seed value during optimisation,
    e.g. {"c": [(0.05, 0.08)], "beta": [(0.05, 0.08)], "cst": [(0.27, 0.28)]}.
    For "cst" the ranges select r_foils stations; blend-zone chord/twist are always derived, never free.
    '''
    N: int = 2
    r_foils: tuple = (0.0, 0.025, 0.05, 0.15, 0.275)
    airfoils: tuple = ("arad10",)           # seed airfoil for each aero station (one name is broadcast)

    # Hub (structural) section
    hub_chord: float = 0.03
    hub_twist_deg: float = 0.0
    hub_thickness_ratio: float = 0.1
    hub_cst: tuple = None                   # CST of the hub section; fitted to the stadium when None

    # Design variables
    freeze: dict = dataclasses.field(default_factory=dict)
    optimise_cst: bool = True               # requires NeuralFoil (Inputs.coeff_method = 3)
    c_bounds: tuple = (0.001, 0.2)          # chord bounds (m)
    beta_bounds_deg: tuple = (0.0, 50.0)    # twist bounds (deg)
    cst_bound: float = 0.25                 # CST weights may move +/- this far from their seed values
    min_tc: float = 0.06                    # minimum thickness-to-chord ratio of the aero sections

    def __post_init__(self):
        self.N = int(self.N)
        self.r_foils = tuple(float(r) for r in self.r_foils)
        if len(self.r_foils) < 4:
            raise ValueError(f"r_foils needs at least [0, r_hub, r_aero, r_tip], got {self.r_foils}")
        if self.r_foils[0] > self.r_foils[1] or np.any(np.diff(self.r_foils[1:]) <= 0):
            raise ValueError(f"r_foils must be increasing (r_foils[0] may equal r_hub), got {self.r_foils}")

        airfoils = (self.airfoils,) if isinstance(self.airfoils, str) else tuple(self.airfoils)
        if len(airfoils) == 1:
            airfoils = airfoils * self.n_aero
        if len(airfoils) != self.n_aero:
            raise ValueError(f"Expected {self.n_aero} airfoils for the aero stations {self.r_foils[2:]}, got {len(airfoils)}")
        self.airfoils = airfoils

        if self.hub_cst is not None:
            self.hub_cst = tuple(float(w) for w in self.hub_cst)
            if len(self.hub_cst) != geometry.N_CST:
                raise ValueError(f"hub_cst must have {geometry.N_CST} entries, got {len(self.hub_cst)}")

        unknown = set(self.freeze) - {"c", "beta", "cst"}
        if unknown:
            raise ValueError(f"freeze keys must be 'c', 'beta' or 'cst', got {sorted(unknown)}")
        self.freeze = {k: tuple((float(lo), float(hi)) for lo, hi in ranges) for k, ranges in self.freeze.items()}
        for k, ranges in self.freeze.items():
            if any(lo > hi for lo, hi in ranges):
                raise ValueError(f"freeze['{k}'] ranges must be (r_lo, r_hi) with r_lo <= r_hi, got {ranges}")

        self.c_bounds = tuple(float(b) for b in self.c_bounds)
        self.beta_bounds_deg = tuple(float(b) for b in self.beta_bounds_deg)

    @property
    def R_hub(self) -> float:
        return self.r_foils[1]

    @property
    def R_aero(self) -> float:
        return self.r_foils[2]

    @property
    def R_tip(self) -> float:
        return self.r_foils[-1]

    @property
    def r_cst(self) -> tuple:
        '''Radii of the CST sections: the hub section at r_hub, then the aero stations.'''
        return self.r_foils[1:]

    @property
    def n_aero(self) -> int:
        return len(self.r_foils) - 2

    def is_frozen(self, var, r) -> bool:
        return any(lo <= r <= hi for lo, hi in self.freeze.get(var, ()))


@dataclass(frozen=True)
class ModelConfig:
    '''
    Hashable model settings read by the physics. Built from Inputs + BladeSpec and carried
    on Params as a static field, so jit specialises on it.
    '''
    R_hub: float
    R_aero: float                   # inboard edge of the aerodynamic zone: the blend zone inboard of it carries no force
    R_tip: float
    solve_idx: tuple                # induced-velocity components solved for (from Inputs.bemt_type)
    V_i_0_max_iter: int
    V_i_0_eps: float
    V_i_0_verbose: int
    inflow_model: int
    coeff_method: int
    tip_loss_model: int
    root_loss_model: int
    rot_corr_model: int
    high_induction_model: int
    I_0: float
    R_m: float
    kV: float
    P_max: float
    w_thrust: float
    thrust_tol: float
    w_thickness: float
    w_confidence: float
    w_smooth: float
    min_confidence: float

    @classmethod
    def build(cls, inputs: Inputs, blade: BladeSpec) -> "ModelConfig":
        copied = [f.name for f in dataclasses.fields(cls) if f.name not in ("R_hub", "R_aero", "R_tip", "solve_idx")]
        return cls(
            R_hub=blade.R_hub,
            R_aero=blade.R_aero,
            R_tip=blade.R_tip,
            solve_idx=tuple(i for i, solved in enumerate(inputs.bemt_type) if solved),
            **{name: getattr(inputs, name) for name in copied},
        )

    @property
    def solve_rows(self) -> np.ndarray:
        '''Indices of the solved components, for indexing (3, n_r) induced-velocity arrays.'''
        return np.array(self.solve_idx)


class Params(eqx.Module):
    '''
    Parameters for a given blade solution at one flight condition (built by Propeller.condition).
    Ensure that all inputs are 1D arrays upon assignment.

    shape(r, dr, c, beta) = (N_r,)
    shape(psi, dpsi) = (N_psi,)
    shape(cst) = (N_foils + 1, N_CST) [hub section, then aero sections at r_foils[2:]]
    shape(V_i) = (3, N_r, N_psi) [axial, tangential, radial]
    '''
    # Domain of Integration
    r: jnp.ndarray          # radial stations
    dr: jnp.ndarray         # radial spacing (m)
    psi: jnp.ndarray        # azimuthal stations
    dpsi: jnp.ndarray       # azimuthal spacing (m)

    # Geometry & discretization
    N: jnp.ndarray          # number of blades
    c: jnp.ndarray          # chord length at radial stations (m)
    beta: jnp.ndarray       # blade twist angle (rad)
    cst: jnp.ndarray        # CST sections [lower x8, upper x8, LE weight, TE thickness] at r_cst
    r_cst: jnp.ndarray      # radii of the CST sections (r_foils[1:]) (m)

    # Inflow / rotor state
    V_x: jnp.ndarray        # axial inflow velocity (ms^{-1})
    V_yz: jnp.ndarray       # In plane inflow velocity (ms^{-1})

    omega: jnp.ndarray      # rotational speed (rad/s)
    omega_max: jnp.ndarray  # max subsonic omega

    tilt: jnp.ndarray      # tilt angle of propeller (rad)

    # Cost Weights
    objective_weight: jnp.ndarray

    # Target objective
    thrust_target: jnp.ndarray

    V_i_0: jnp.ndarray     # initial guess/storage for V_i_0 (ms^{-1}), [3, n_r]

    # Solver Logging Info
    error_log: jnp.ndarray
    time_log: jnp.ndarray

    # Model settings and airfoil model (static: jit specialises on them)
    cfg: ModelConfig = eqx.field(static=True)
    airfoil_interp: callable = eqx.field(static=True)

    # Aerodynamic Constants
    rho: float = 1.225          # air density (kgm^{-3})
    mu: float = 1.798e-5        # air viscosity (kgm^-1s^-1)
    a: float = 343.0            # speed of sound (m/s)


class Propeller:
    '''
    A propeller design and the run configuration it is analysed with.

    Holds the Inputs, the BladeSpec, the design arrays (c and beta at the BEMT stations, cst at r_foils[2:])
    and the per-condition operating state (omega, tilt, V_i_0). Builds Params for the solver and handles
    all file I/O, one JSON file per propeller in propellers/.
    '''
    def __init__(self, inputs: Inputs, blade: BladeSpec, c, beta, cst, omega=None, tilt=None, V_i_0=None,
                 error_log=None, time_log=None, name=None):
        # Internal Imports
        from airfoil import load_airfoil_data

        if blade.hub_cst is None:
            blade = dataclasses.replace(blade, hub_cst=tuple(geometry.fit_hub_cst(blade.hub_thickness_ratio).tolist()))
        self.inputs = inputs
        self.blade = blade
        self.name = name

        # Build Grids
        self.r, self.dr = geometry.radial_stops(blade.R_tip, blade.R_hub, inputs.n_r)
        self.psi, self.dpsi = geometry.angular_stops(inputs.n_psi)

        # Design arrays
        self.c = _checked("c", c, (inputs.n_r,))
        self.beta = _checked("beta", beta, (inputs.n_r,))
        self.cst = _checked("cst", cst, (blade.n_aero, geometry.N_CST))

        # Operating state per flight condition
        self.omega = _checked("omega", inputs.omega if omega is None else omega, (inputs.n_cond,))
        self.tilt = _checked("tilt", inputs.tilt if tilt is None else tilt, (inputs.n_cond,))
        self.error_log = jnp.atleast_1d(jnp.asarray([1.0] if error_log is None else error_log, dtype=float))
        self.time_log = jnp.atleast_1d(jnp.asarray([0.0] if time_log is None else time_log, dtype=float))

        # Model objects rebuilt from the configuration (never saved)
        self.cfg = ModelConfig.build(inputs, blade)
        self.airfoil_interp = load_airfoil_data(inputs.coeff_method, blade.airfoils[0])

        self.V_i_0 = self._seeded_V_i_0(V_i_0)

    ## Construction
    @classmethod
    def default(cls, inputs: Inputs = None, blade: BladeSpec = None, c: float = 0.02, beta_deg: float = 10.0) -> "Propeller":
        '''
        Constant-chord, constant-twist blade with CST sections seeded from blade.airfoils.

        Inputs:
            inputs (Inputs): run configuration (defaults to Inputs())
            blade (BladeSpec): blade definition (defaults to BladeSpec())
            c (float): The chord length in meters.
            beta_deg (float): The blade twist angle in degrees.
        '''
        inputs = Inputs() if inputs is None else inputs
        blade = BladeSpec() if blade is None else blade
        return cls(
            inputs, blade,
            c=jnp.full((inputs.n_r,), c),
            beta=jnp.full((inputs.n_r,), jnp.radians(beta_deg)),
            cst=geometry.get_cst_weights_from_names(blade.airfoils),
        )

    @classmethod
    def from_apc(cls, filepath, inputs: Inputs = None, blade: BladeSpec = None) -> "Propeller":
        '''
        Chord and twist from an APC .PE0 geometry file, interpolated onto the BEMT grid.
        The blade tip (r_foils[-1]) moves to the APC tip radius.
        '''
        # Internal Imports
        import apcpropanalysis

        inputs = Inputs() if inputs is None else inputs
        blade = BladeSpec() if blade is None else blade

        # Convert inches to meters and degrees to radians to match JAX solver SI units
        apc_geom = apcpropanalysis.parse_apc_pe0(filepath)
        r_apc = np.asarray(apc_geom["r"]) * 0.0254
        c_apc = np.asarray(apc_geom["chord"]) * 0.0254
        beta_apc = np.radians(np.asarray(apc_geom["twist_deg"]))

        r_tip = float(r_apc.max())
        if not np.isclose(r_tip, blade.R_tip):
            warnings.warn(f"Moving the blade tip from r_foils[-1] = {blade.R_tip:.4f} m to the APC tip radius {r_tip:.4f} m")
            blade = dataclasses.replace(blade, r_foils=blade.r_foils[:-1] + (r_tip,))

        r, _ = geometry.radial_stops(blade.R_tip, blade.R_hub, inputs.n_r)
        return cls(
            inputs, blade,
            c=np.interp(r, r_apc, c_apc),
            beta=np.interp(r, r_apc, beta_apc),
            cst=geometry.get_cst_weights_from_names(blade.airfoils),
        )

    @classmethod
    def load(cls, name) -> "Propeller":
        '''
        Loads propellers/<name>.json (or name itself when it is a path).
        '''
        path = prop_path(name)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data, name=path.stem)

    @classmethod
    def from_dict(cls, data: dict, name=None) -> "Propeller":
        version = data.get("format_version")
        if version not in (2, FORMAT_VERSION):
            raise ValueError(f"Unsupported propeller file format_version {version} (expected {FORMAT_VERSION})")

        inputs_data = dict(data["inputs"])
        if version == 2:
            defaults = Inputs()
            reset = [k for k in _RESET_FROM_V2 if k in inputs_data and inputs_data[k] != getattr(defaults, k)]
            for k in reset:
                del inputs_data[k]
            if reset:
                warnings.warn(f"Format 2 propeller file: {', '.join(reset)} reset to the defaults of the Newton BEMT "
                              f"solve and relative-change stopping test")
        if "verbose" in inputs_data:
            # Files saved before the per-solver verbosity levels: the old flag drove the induced velocity solver
            inputs_data.setdefault("V_i_0_verbose", int(inputs_data.pop("verbose")))
        inputs = _dataclass_from_dict(Inputs, inputs_data, "inputs")
        blade = _dataclass_from_dict(BladeSpec, data["blade"], "blade")

        state = data["state"]
        missing = {"c", "beta", "cst"} - set(state)
        if missing:
            raise ValueError(f"Propeller file state is missing {sorted(missing)}")
        unknown = sorted(set(state) - set(STATE_FIELDS))
        if unknown:
            warnings.warn(f"Ignoring unknown state fields in propeller file: {unknown}")
        return cls(inputs, blade, name=name, **{k: state[k] for k in STATE_FIELDS if k in state})

    @classmethod
    def from_legacy_eqx(cls, name, airfoil_name: str = "arad10") -> "Propeller":
        '''
        Migrates prop_params_<name>.eqx from the old positional format.

        The old prop_inputs_<name>.eqx is not read: it never stored flight conditions, and the physics
        always ran on the class defaults rather than the loaded values, so Inputs() defaults (with n_r and
        n_psi from the saved grid) reproduce the old results. The saved radial grid is kept exactly: the hub
        sits at the first saved station with its saved chord and twist, and every section uses airfoil_name
        (the old model used one airfoil for the whole blade).
        '''
        params_path = PROP_DIR / f"prop_params_{name}.eqx"
        leaves = _read_eqx_leaves(params_path)
        layout = _LEGACY_PARAMS_LAYOUTS.get(len(leaves))
        if layout is None:
            raise ValueError(f"{params_path.name} holds {len(leaves)} arrays; known legacy layouts hold {sorted(_LEGACY_PARAMS_LAYOUTS)}")
        p = dict(zip(layout, leaves))

        r, c, beta = p["r"], p["c"], p["beta"]
        inputs = Inputs(n_r=len(r), n_psi=len(p["psi"]))
        warnings.warn(f"{name}: legacy propeller files store no flight conditions; using the Inputs() defaults")

        seed = geometry.kulfan_from_name(airfoil_name)
        r_hub, r_tip = float(r[0]), float(r[-1])
        r_aero = 0.5 * float(r[0] + r[1])
        blade = BladeSpec(
            N=int(np.ravel(p["N"])[0]),
            r_foils=(0.0, r_hub, r_aero, 0.5 * (r_aero + r_tip), r_tip),
            airfoils=(airfoil_name,),
            hub_chord=float(c[0]),
            hub_twist_deg=float(np.degrees(beta[0])),
            hub_cst=tuple(seed.tolist()),
        )

        r_new, _ = geometry.radial_stops(blade.R_tip, blade.R_hub, inputs.n_r)
        prop = cls(
            inputs, blade,
            c=np.interp(r_new, r, c),
            beta=np.interp(r_new, r, beta),
            cst=np.tile(seed, (blade.n_aero, 1)),
            error_log=p["error_log"],
            time_log=p["time_log"],
            name=name,
        )

        # The saved induced velocity was solved for the first flight condition only
        if p["V_i_0"].shape == prop.V_i_0[0].shape and np.all(np.isfinite(p["V_i_0"])):
            prop = prop.with_state(V_i_0=prop.V_i_0.at[0].set(p["V_i_0"]))
        return prop

    ## Derived propellers
    def with_state(self, **arrays) -> "Propeller":
        '''
        Copy with design/operating arrays replaced: any of c, beta, cst, omega, tilt, V_i_0, error_log, time_log.
        '''
        unknown = set(arrays) - set(STATE_FIELDS)
        if unknown:
            raise TypeError(f"with_state() got unknown fields {sorted(unknown)}")
        state = {k: getattr(self, k) for k in STATE_FIELDS}
        state.update(arrays)
        return Propeller(self.inputs, self.blade, name=self.name, **state)

    def with_inputs(self, inputs: Inputs) -> "Propeller":
        '''
        Same blade analysed with new Inputs. Chord and twist are re-gridded if n_r changes, and the
        operating state (omega, tilt, V_i_0) restarts from the new flight conditions.
        '''
        return self._regridded(inputs, self.blade, self.cst, keep_state=False)

    def with_blade(self, blade: BladeSpec) -> "Propeller":
        '''
        Same design on a new BladeSpec (e.g. different r_foils or freeze ranges). Chord and twist are
        re-gridded onto the new BEMT domain; CST sections are kept if the number of aero stations matches,
        otherwise re-seeded from blade.airfoils.
        '''
        if blade.n_aero == self.blade.n_aero:
            cst = self.cst
        else:
            warnings.warn(f"Number of aero stations changed ({self.blade.n_aero} -> {blade.n_aero}); re-seeding CST from {blade.airfoils}")
            cst = geometry.get_cst_weights_from_names(blade.airfoils)
        return self._regridded(self.inputs, blade, cst, keep_state=True)

    def _regridded(self, inputs, blade, cst, keep_state):
        r_new, _ = geometry.radial_stops(blade.R_tip, blade.R_hub, inputs.n_r)
        same_grid = r_new.shape == self.r.shape and bool(jnp.allclose(r_new, self.r))
        c = self.c if same_grid else jnp.interp(r_new, self.r, self.c)
        beta = self.beta if same_grid else jnp.interp(r_new, self.r, self.beta)

        state = {}
        if keep_state:
            state = dict(omega=self.omega, tilt=self.tilt, error_log=self.error_log, time_log=self.time_log)
            if same_grid:
                state["V_i_0"] = self.V_i_0
        return Propeller(inputs, blade, c=c, beta=beta, cst=cst, name=self.name, **state)

    ## Solver Setup
    def condition(self, idx: int = 0) -> Params:
        '''
        Params for flight condition idx.
        '''
        return self._params(idx, self.V_i_0[idx])

    def batched(self) -> Params:
        '''
        Params for every flight condition, stacked along a leading axis for jax.vmap.
        '''
        conditions = [self.condition(idx) for idx in range(self.inputs.n_cond)]
        return jax.tree_util.tree_map(lambda *leaves: jnp.stack(leaves), *conditions)

    def _params(self, idx, V_i_0) -> Params:
        inputs = self.inputs
        return Params(
            r=self.r, dr=self.dr,
            psi=self.psi, dpsi=self.dpsi,
            N=jnp.array([self.blade.N]),
            c=self.c,
            beta=self.beta,
            cst=jnp.concatenate([jnp.asarray(self.blade.hub_cst)[None, :], self.cst]),
            r_cst=jnp.asarray(self.blade.r_cst),
            V_x=inputs.V_x[idx],
            V_yz=inputs.V_yz[idx],
            omega=self.omega[idx],
            omega_max=jnp.asarray(343.0 / self.blade.R_tip),
            tilt=self.tilt[idx],
            objective_weight=inputs.objective_weight[idx],
            thrust_target=inputs.thrust_target[idx],
            V_i_0=V_i_0,
            error_log=self.error_log,
            time_log=self.time_log,
            cfg=self.cfg,
            airfoil_interp=self.airfoil_interp,
        )

    def _seeded_V_i_0(self, V_i_0):
        '''
        Momentum-theory velocity scales as the initial induced velocity guess for each condition.
        A supplied V_i_0 (n_cond, 3, n_r) is kept, except for conditions that are not finite.
        '''
        # Internal Imports
        from propmodel import get_velocity_scales

        n_r, rows = self.inputs.n_r, self.cfg.solve_rows
        seeds = []
        for idx in range(self.inputs.n_cond):
            V_scale = get_velocity_scales(self._params(idx, jnp.zeros((3, n_r))))
            # Introduce zero for non-solved components to maintain shape (3, n_r)
            seeds.append(jnp.zeros((3, n_r)).at[rows].set(jnp.broadcast_to(V_scale, (len(rows), n_r))))
        seed = jnp.stack(seeds)
        if V_i_0 is None:
            return seed

        V_i_0 = _checked("V_i_0", V_i_0, seed.shape)
        bad = ~jnp.all(jnp.isfinite(V_i_0), axis=(1, 2))
        if bool(jnp.any(bad)):
            warnings.warn(f"Non-finite V_i_0 for condition(s) {np.flatnonzero(np.asarray(bad)).tolist()}; re-seeded from momentum theory")
        return jnp.where(bad[:, None, None], seed, V_i_0)

    ## File I/O
    def to_dict(self) -> dict:
        return {
            "format_version": FORMAT_VERSION,
            "saved": datetime.now().isoformat(timespec="seconds"),
            "inputs": _to_json({f.name: getattr(self.inputs, f.name) for f in dataclasses.fields(self.inputs)}),
            "blade": _to_json({f.name: getattr(self.blade, f.name) for f in dataclasses.fields(self.blade)}),
            "state": _to_json({k: getattr(self, k) for k in STATE_FIELDS}),
        }

    def save(self, name=None, overwrite: bool = False) -> Path:
        '''
        Writes the propeller to propellers/<name>.json (or to name itself when it is a path).
        Refuses to replace an existing file unless overwrite=True. Returns the path written.
        '''
        name = self.name if name is None else name
        if not name:
            raise ValueError("No propeller name given")
        path = prop_path(name)
        if path.exists() and not overwrite:
            raise FileExistsError(f"{path} already exists; pass overwrite=True to replace it")

        # Write to a temporary file first so an interrupted save never corrupts an existing propeller
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(_format_json(self.to_dict()) + "\n")
        os.replace(tmp, path)

        self.name = path.stem
        print(f"Propeller successfully written to: {path}")
        return path


def prop_path(name) -> Path:
    '''
    propellers/<name>.json for a bare name; a name with a directory is used as given.
    '''
    path = Path(name)
    if not path.name.endswith(".json"):
        path = path.with_name(path.name + ".json")
    return PROP_DIR / path if path.parent == Path(".") else path


## JSON helpers
def _checked(name, value, shape):
    array = jnp.asarray(value, dtype=float)
    if array.shape != tuple(shape):
        raise ValueError(f"{name} has shape {array.shape}, expected {tuple(shape)}")
    return array

def _to_json(value):
    if isinstance(value, dict):
        return {k: _to_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json(v) for v in value]
    if isinstance(value, (jax.Array, np.ndarray)):
        return np.asarray(value).tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value

def _dataclass_from_dict(cls, data: dict, section: str):
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(set(data) - known)
    if unknown:
        warnings.warn(f"Ignoring unknown {section} fields in propeller file: {unknown}")
    return cls(**{k: v for k, v in data.items() if k in known})

def _format_json(obj, indent=0):
    '''
    JSON with one key per line and numeric rows kept on a single line, so propeller files diff cleanly.
    '''
    pad = " " * indent
    if isinstance(obj, dict):
        if not obj:
            return "{}"
        items = [f"{pad}  {json.dumps(k)}: {_format_json(v, indent + 2)}" for k, v in obj.items()]
        return "{\n" + ",\n".join(items) + f"\n{pad}}}"
    if isinstance(obj, list) and any(isinstance(v, (list, dict)) for v in obj):
        items = [f"{pad}  {_format_json(v, indent + 2)}" for v in obj]
        return "[\n" + ",\n".join(items) + f"\n{pad}]"
    return json.dumps(obj)


## Legacy .eqx reader (migration only)
# Field order of the old params files: equinox serialised array leaves positionally, skipping strings.
_LEGACY_PARAMS_FIELDS = (
    "r", "dr", "psi", "dpsi", "N", "c", "beta", "c_scale", "beta_scale", "V_x", "V_yz", "omega", "omega_max",
    "tilt", "tilt_scale", "objective_thrust", "objective_MIL", "objective_power", "objective_weight",
    "thrust_target", "V_i_0", "error_log", "time_log", "rho", "mu", "a",
)
# Files written before tilt was added hold 24 arrays
_LEGACY_PARAMS_LAYOUTS = {
    26: _LEGACY_PARAMS_FIELDS,
    24: tuple(f for f in _LEGACY_PARAMS_FIELDS if f not in ("tilt", "tilt_scale")),
}

def _read_eqx_leaves(path):
    leaves = []
    with open(path, "rb") as f:
        while True:
            try:
                leaves.append(np.load(f))
            except (EOFError, ValueError):
                break
    return leaves


if __name__ == "__main__":
    import threading
    import queue
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    # Internal Imports
    import proptimiser
    import results

    inputs = Inputs()
    blade = BladeSpec(
        r_foils=(0.0, 0.025, 0.05, 0.15, 0.275),   # [0, r_hub, r_aero, r_mid, r_tip]
        airfoils=("arad10",),
        freeze={},                                 # e.g. {"c": [(0.05, 0.08)]} holds the chord at its seed value there
    )
    prop_init = Propeller.default(inputs=inputs, blade=blade)
    # prop_init = Propeller.load("Currawong_V1").with_blade(blade)

    params_init = prop_init.condition(0)
    print(f"Initialised {blade.n_aero} CST sections at r = {blade.r_foils[2:]} from {blade.airfoils}")

    # Live Plotting Setup
    print("Setting up live visualization canvas...")
    gui_queue = queue.Queue()
    plotter = results.LivePropPlotter(params_init)
    gui_queue.put(params_init)

    # Jax Profiling (optional)
    if inputs.jax_tracing:
        print("Starting Profiler...")
        jax.profiler.start_trace("/tmp/jax_trace")

    thread_results = {}

    def optimization_worker():
        # Run optimization on background thread and pass the queue
        thread_results['prop_opt'] = proptimiser.run_optimization(prop_init, update_queue=gui_queue)

    # Run Optimisation in Background Thread
    opt_thread = threading.Thread(target=optimization_worker, daemon=True)
    opt_thread.start()

    # This automatically polls the queue and refreshes the canvas
    ani = FuncAnimation(
        plotter.fig,
        plotter.update,
        fargs=(gui_queue,),
        interval=200,
        cache_frame_data=False,
        blit=False
    )

    # Hands control to the GUI until the window is closed
    plt.show()

    # Re-join thread to ensure clean closure
    opt_thread.join()
    prop_opt = thread_results.get('prop_opt', prop_init)

    if inputs.jax_tracing:
        jax.block_until_ready(prop_opt.c)
        jax.profiler.stop_trace()
        print("Profiler stopped. Trace saved to /tmp/jax_trace")

    # Save Results/Export Geometry if Required
    user_input = input('Save Propeller? (y/n): ')
    if user_input.lower() == 'y':
        propname = "prop_" + datetime.now().strftime("%Y%m%d-%H%M")
        prop_opt.save(propname)

    print('Proptimisation Complete')

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
import yaml

# Internal Imports
import geometry

ROOT = Path(__file__).resolve().parent
DESIGN_DIR = ROOT / "designs"           # hand design files: <name>.yaml
PROP_DIR = ROOT / "propellers"          # solved state files: <name>.state.json
DESIGN_FORMAT = 1
STATE_FORMAT = 1

# Inputs fields holding one entry per flight condition
CONDITION_FIELDS = ("thrust_target", "V_x", "V_yz", "omega", "tilt", "solve_omega", "solve_tilt", "objective_weight")

# Propeller arrays saved in the "state" section of a propeller file
STATE_FIELDS = ("c", "beta", "cst", "omega", "tilt", "V_i_0", "error_log", "time_log")

# Unit of each state array (nd: dimensionless). beta and tilt are in radians: the solver works in radians.
STATE_UNITS = {
    "c": "m",                   # chord at the BEMT stations, shape (n_r,)
    "beta": "rad",              # twist at the BEMT stations, shape (n_r,)
    "cst": "nd",                # CST weights of the aero sections, shape (n_aero, N_CST)
    "omega": "rad/s",           # rotor speed for each flight condition, shape (n_cond,)
    "tilt": "rad",              # blade tilt for each flight condition, shape (n_cond,)
    "V_i_0": "m/s",             # induced velocity warm start, shape (n_cond, 3, n_r)
    "error_log": "nd",          # relative change of the merit function for each optimiser iteration
    "time_log": "s",            # wall time of each optimiser iteration
}


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
    kV: float = 220.0
    P_max: float = 3300.0

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

    # Sweep and rake: offsets of the pitch axis (the quarter-chord line of the aero zone) from the radial line through
    # the hub centre, at the control radii offset_r, held at the end values outside them and blended smoothly between
    # them (PCHIP). Empty tuples are an unswept blade. An empty offset_y or offset_z means zero offset.
    offset_r: tuple = ()                    # control radii (m), strictly increasing
    offset_y: tuple = ()                    # tangential offset (m): + toward the rotation direction (APC SWEEP), so a backswept tip is negative
    offset_z: tuple = ()                    # axial offset (m): + toward the thrust direction (APC RAKE)

    # Structure: one isotropic material over the CST section outline (defaults follow the APC propeller files)
    mat_density: float = 1700.0             # kg/m^3 (specific gravity 1.7)
    mat_E: float = 19.7e9                   # Young's modulus (Pa)
    mat_G: float = 7.3e9                    # shear modulus (Pa)
    mat_sigma_allow: float = 60e6           # allowable normal stress (Pa), a placeholder: set it from the material
    mat_tau_allow: float = 0.0              # allowable shear stress (Pa); 0 means mat_sigma_allow / sqrt(3)
    fill_factor: float = 1.0                # share of the outline area that is material: 1 solid, < 1 homogenised infill

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

        self.offset_r = tuple(float(v) for v in self.offset_r)
        self.offset_y = tuple(float(v) for v in self.offset_y)
        self.offset_z = tuple(float(v) for v in self.offset_z)
        if self.offset_r:
            if len(self.offset_r) < 2 or np.any(np.diff(self.offset_r) <= 0):
                raise ValueError(f"offset_r needs at least two strictly increasing radii, got {self.offset_r}")
            n = len(self.offset_r)
            self.offset_y = self.offset_y or (0.0,) * n
            self.offset_z = self.offset_z or (0.0,) * n
            if len(self.offset_y) != n or len(self.offset_z) != n:
                raise ValueError(f"offset_y and offset_z need {n} entries to match offset_r, "
                                 f"got {len(self.offset_y)} and {len(self.offset_z)}")
        elif self.offset_y or self.offset_z:
            raise ValueError("offset_y and offset_z need control radii in offset_r")

        if min(self.mat_density, self.mat_E, self.mat_G, self.mat_sigma_allow) <= 0 or self.mat_tau_allow < 0:
            raise ValueError("Material density, E, G and sigma_allow must be positive, and tau_allow cannot be negative")
        if not 0.0 < self.fill_factor <= 1.0:
            raise ValueError(f"fill_factor must be in (0, 1], got {self.fill_factor}")

    @property
    def swept(self) -> bool:
        '''True when any sweep or rake offset is nonzero: the aerodynamics then take the swept section frame.'''
        return any(v != 0.0 for v in self.offset_y + self.offset_z)

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
    swept: bool                     # the blade has sweep or rake: the aerodynamics use the swept section frame
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
        copied = [f.name for f in dataclasses.fields(cls) if f.name not in ("R_hub", "R_aero", "R_tip", "solve_idx", "swept")]
        return cls(
            R_hub=blade.R_hub,
            R_aero=blade.R_aero,
            R_tip=blade.R_tip,
            solve_idx=tuple(i for i, solved in enumerate(inputs.bemt_type) if solved),
            swept=blade.swept,
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
    shape(offsets) = (N_r, 4) [sweep y, rake z, dy/dr, dz/dr]
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
    offsets: jnp.ndarray    # pitch-axis offsets (N_r, 4): [y_sw, z_rk, dy_sw/dr, dz_rk/dr] (m, m, -, -), see geometry.station_offsets

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
    all file I/O: designs/<name>.yaml (Inputs, BladeSpec) and propellers/<name>.state.json (the arrays).
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
        self.offsets = geometry.station_offsets(self.r, blade)

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
    def from_apc(cls, filepath, inputs: Inputs = None, blade: BladeSpec = None, use_apc_sweep: bool = False) -> "Propeller":
        '''
        Chord and twist from an APC .PE0 geometry file, interpolated onto the BEMT grid.
        The blade tip (r_foils[-1]) moves to the APC tip radius.
        use_apc_sweep also takes the file's sweep and rake, as offsets of the pitch axis (geometry.apc_offsets).
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
        if use_apc_sweep:
            offset_r, offset_y, offset_z = geometry.apc_offsets(apc_geom, blade.R_hub, blade.R_aero)
            blade = dataclasses.replace(blade, offset_r=offset_r, offset_y=offset_y, offset_z=offset_z)

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
        Loads the design designs/<name>.yaml and its solved state propellers/<name>.state.json.
        A name with a folder reads <folder>/<name>.yaml and <folder>/<name>.state.json (see design_paths).
        '''
        design_path, state_path = design_paths(name)
        inputs, blade = load_design(design_path)
        return cls(inputs, blade, name=design_path.stem, **load_state(state_path))

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

    def material(self):
        '''
        The blade material as structure.Material, for the structural loads (kept out of Params and ModelConfig, so a
        change of material never recompiles the BEMT).
        '''
        from structure import Material      # Internal Import: structure imports this module

        return Material.from_blade(self.blade)

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
            offsets=self.offsets,
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
    def design_dict(self) -> dict:
        '''
        The design as written to designs/<name>.yaml: Inputs and BladeSpec in SI units, angles in degrees.
        '''
        return design_to_dict(self.inputs, self.blade)

    def state_dict(self) -> dict:
        '''
        The solved state as written to propellers/<name>.state.json (units in STATE_UNITS).
        '''
        return {"format_version": STATE_FORMAT, "units": dict(STATE_UNITS),
                **_to_json({k: getattr(self, k) for k in STATE_FIELDS})}

    def save(self, name=None, overwrite: bool = False) -> Path:
        '''
        Writes designs/<name>.yaml and propellers/<name>.state.json (see design_paths for a name with a folder).
        Refuses to replace an existing file unless overwrite=True. Returns the path of the design file.
        '''
        name = self.name if name is None else name
        if not name:
            raise ValueError("No propeller name given")
        design_path, state_path = design_paths(name)
        for path in (design_path, state_path):
            if path.exists() and not overwrite:
                raise FileExistsError(f"{path} already exists; pass overwrite=True to replace it")

        _write_atomic(design_path, design_yaml(self.inputs, self.blade, design_path.stem))
        _write_atomic(state_path, _format_json(self.state_dict()) + "\n")

        self.name = design_path.stem
        print(f"Propeller written to: {design_path} and {state_path}")
        return design_path


def design_paths(name) -> tuple:
    '''
    (design YAML, state JSON) paths of a propeller.
    A bare name, or a path in designs/ or propellers/, gives designs/<name>.yaml and propellers/<name>.state.json.
    A path in another folder gives <folder>/<name>.yaml and <folder>/<name>.state.json.
    The suffixes .yaml and .state.json are removed from the name.
    '''
    path = Path(name)
    stem = path.name
    for suffix in (".state.json", ".yaml"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    folder = path.parent
    if folder == Path(".") or folder.resolve() in (DESIGN_DIR.resolve(), PROP_DIR.resolve()):
        return DESIGN_DIR / f"{stem}.yaml", PROP_DIR / f"{stem}.state.json"
    return folder / f"{stem}.yaml", folder / f"{stem}.state.json"


## Design YAML
# Inputs fields that hold an angle in radians: the design YAML gives them in degrees, under the key <name>_deg
_INPUT_ANGLES = ("tilt",)

DESIGN_HEADER = """\
# Proptimiser design: {name}
# Units: SI. Lengths m, speeds m/s, omega rad/s, thrust N, power W, I_0 A, R_m ohm, kV rpm/V,
# density kg/m^3, E, G and allowable stresses Pa. Angles in degrees: every key that ends in _deg.
# Dimensionless (nd): weights, tolerances, ratios, CST weights, fill_factor.
# Every key is required. An unknown key is an error. Write floats with a dot (1.0e-8, not 1e-8).
"""


def _design_keys(cls) -> list:
    names = [f.name for f in dataclasses.fields(cls)]
    if cls is Inputs:
        names = [f"{n}_deg" if n in _INPUT_ANGLES else n for n in names]
    return names


def design_to_dict(inputs: Inputs, blade: BladeSpec) -> dict:
    '''
    The design YAML content as plain Python values.
    '''
    inputs_d = {}
    for f in dataclasses.fields(inputs):
        value = getattr(inputs, f.name)
        if f.name in _INPUT_ANGLES:
            inputs_d[f"{f.name}_deg"] = _to_json(np.degrees(np.asarray(value)))
        else:
            inputs_d[f.name] = _to_json(value)
    blade_d = _to_json({f.name: getattr(blade, f.name) for f in dataclasses.fields(blade)})
    return {"format_version": DESIGN_FORMAT, "inputs": inputs_d, "blade": blade_d}


def design_yaml(inputs: Inputs, blade: BladeSpec, name: str = "") -> str:
    '''
    The text of a design YAML file: the units comment block, then the values.
    '''
    body = yaml.safe_dump(design_to_dict(inputs, blade), sort_keys=False, default_flow_style=None, width=1000)
    return DESIGN_HEADER.format(name=name) + body


def design_from_dict(data: dict, source="design") -> tuple:
    '''
    (Inputs, BladeSpec) from the design YAML content. Rejects a missing key, an unknown key and a format_version
    that is not DESIGN_FORMAT.
    '''
    if not isinstance(data, dict):
        raise ValueError(f"{source}: the design file must hold a mapping")
    _check_keys(data, ["format_version", "inputs", "blade"], source)
    if data["format_version"] != DESIGN_FORMAT:
        raise ValueError(f"{source}: format_version {data['format_version']!r} is not supported "
                         f"(expected {DESIGN_FORMAT})")

    sections = {}
    for section, cls in (("inputs", Inputs), ("blade", BladeSpec)):
        values = data[section]
        if not isinstance(values, dict):
            raise ValueError(f"{source}: '{section}' must be a mapping")
        _check_keys(values, _design_keys(cls), f"{source} {section}")
        kwargs = {}
        for f in dataclasses.fields(cls):
            if cls is Inputs and f.name in _INPUT_ANGLES:
                kwargs[f.name] = np.radians(np.asarray(values[f"{f.name}_deg"], dtype=float))
            elif isinstance(f.default, float) and not isinstance(values[f.name], bool):
                kwargs[f.name] = float(values[f.name])
            else:
                kwargs[f.name] = values[f.name]
        sections[section] = cls(**kwargs)
    return sections["inputs"], sections["blade"]


def load_design(path) -> tuple:
    '''
    (Inputs, BladeSpec) from a design YAML file.
    '''
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return design_from_dict(data, path.name)


## State JSON
def load_state(path) -> dict:
    '''
    The solved state arrays (STATE_FIELDS) from a state JSON file. Rejects a missing or unknown key, other units
    and a format_version that is not STATE_FORMAT.
    '''
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    _check_keys(data, ["format_version", "units", *STATE_FIELDS], path.name)
    if data["format_version"] != STATE_FORMAT:
        raise ValueError(f"{path.name}: format_version {data['format_version']!r} is not supported "
                         f"(expected {STATE_FORMAT})")
    if data["units"] != STATE_UNITS:
        raise ValueError(f"{path.name}: units {data['units']} differ from {STATE_UNITS}")
    return {k: data[k] for k in STATE_FIELDS}


## File helpers
def _check_keys(data: dict, expected, source):
    missing = [k for k in expected if k not in data]
    unknown = sorted(set(data) - set(expected))
    if missing or unknown:
        raise ValueError(f"{source}: missing keys {missing}, unknown keys {unknown}")


def _write_atomic(path, text):
    # Write to a temporary file first so an interrupted save never corrupts an existing file
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    os.replace(tmp, path)


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

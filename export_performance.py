"""
Export the BEMT performance of one design as contract A: prop-performance v1 (.h5), for Propalyser.

The layout comes from AeroPlan. The hand copy is docs/prop_performance_spec.md.

The script loads designs/<name>.yaml and propellers/<name>.state.json. It sweeps the axial speed at fixed rotor
speeds and solves the induced velocity at each point. It does not run the optimiser.

The script refuses to write when the git tree is dirty, when no author is known, or when the design files are
not committed. There is no override.

Examples, run from the repository root:
    python export_performance.py currawong_v2_structural --author "A. Person"
    python export_performance.py --apc apc_prop_data/propgeom_data/20x8E-PERF.PE0 \
        --compare-apc apc_prop_data/performance_data/PER3_20x8E.dat
The second example prints a CT and CP comparison with the APC table. It writes no file.
"""
import argparse
import datetime
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")      # no windows: the solver modules import pyplot

import h5py
import numpy as np
import yaml
import jax
import jax.numpy as jnp
import equinox as eqx

import airflow
import init
import propmodel

SCHEMA = "prop-performance"
SCHEMA_VERSION = 1
GENERATOR = "Proptimiser"
ROOT = Path(__file__).resolve().parent
RHO = 1.225                 # air density (kg/m3)

# Columns of /performance: (units, description)
PERFORMANCE_COLUMNS = {
    "J": ("nd", "advance ratio V / (n D), n in rev/s"),
    "rpm": ("rpm", "rotor speed"),
    "V_mps": ("m/s", "axial flight speed (no off-axis inflow)"),
    "CT": ("nd", "thrust coefficient T / (rho n^2 D^4)"),
    "CP": ("nd", "power coefficient P / (rho n^3 D^5), P the shaft power Q omega"),
    "CQ": ("nd", "torque coefficient Q / (rho n^2 D^5)"),
    "eta": ("nd", "propeller efficiency J CT / CP (0 at J = 0)"),
    "tip_mach": ("nd", "rotational tip Mach number omega R_tip / a, a = 343 m/s"),
    "reynolds_75": ("nd", "chord Reynolds number at r = 0.75 R_tip from the solved relative velocity, "
                          "azimuth mean"),
    "converged": ("nd", "true when the BEMT solve converged. CT, CP, CQ, eta and reynolds_75 are NaN when false"),
}
STATION_COLUMNS = {
    "r_m": ("m", "radius from the rotation axis"),
    "chord_m": ("m", "section chord"),
    "twist_deg": ("deg", "section twist (beta) from the rotor plane"),
}
BLADE_FRAME = ("Blade frame (e_r, e_t, e_z), right-handed: e_r radial, e_t the direction of blade motion, e_z the "
               "thrust direction. r_m is the distance from the rotation axis. twist_deg is the angle from the rotor "
               "plane to the section chord line, positive when the leading edge turns toward e_z. The stations are "
               "the BEMT stations from the hub radius to the tip. Sweep and rake offsets are not in this table.")
UNCERTAINTY_ITEMS = ("CT", "CP", "CQ", "eta")
UNCERTAINTY_BASIS = ("The Proptimiser BEMT model gives no uncertainty estimate, and Proptimiser holds no validation "
                     "of this design against measured data. No sigma is given.")


## Solver access

@jax.jit
def _point(params, guess):
    # Solve one point. Return thrust, torque, the scaled BEMT residual, the solved state and the Reynolds number
    # at 0.75 R_tip.
    params = eqx.tree_at(lambda p: p.V_i_0, params, guess)
    V_i, V_i_0 = propmodel.induced_velocity(params)
    thrust = jnp.squeeze(propmodel.thrust(V_i, params))
    torque = jnp.squeeze(propmodel.torque(V_i, params))
    lam = V_i_0[params.cfg.solve_rows] / propmodel.get_velocity_scales(params)
    residual = jnp.abs(propmodel.bemt_residuals(lam, params))
    V_R, _ = airflow.resultantVelocityComponents(V_i, params)
    reynolds = airflow.calc_reynolds(params.rho, jnp.mean(V_R, axis=1), params.c, params.mu)
    reynolds_75 = jnp.interp(0.75 * params.cfg.R_tip, params.r, reynolds)
    return thrust, torque, jnp.max(jnp.where(propmodel.aero_zone(params), residual, 0.0)), V_i_0, reynolds_75


@jax.jit
def _seed(params):
    # The momentum theory guess that Propeller uses as its start.
    scale = propmodel.get_velocity_scales(params)
    seed = jnp.broadcast_to(scale, (scale.shape[0], params.r.shape[0]))
    return jnp.zeros_like(params.V_i_0).at[params.cfg.solve_rows].set(seed)


class Case:
    """One propeller and a cache of solved points. The sweep has no off-axis inflow (V_yz = 0)."""

    def __init__(self, prop):
        self.prop, self.params, self.cfg = prop, prop.condition(0), prop.cfg
        self.R_tip, self.blades = self.cfg.R_tip, prop.blade.N
        self.sound = float(self.params.a)
        self.mu = float(self.params.mu)
        self.design_omegas = [float(w) for w in prop.omega]
        # The target thrust only scales the residuals and the start. A fixed value keeps the test the same at all points.
        self.thrust_ref = float(np.max(prop.inputs.thrust_target))
        self._cache = {}

    def solve(self, omega, vx, rho=RHO, warm=None):
        """Return (thrust N, torque Nm, Reynolds number at 0.75 R, converged, state). Try the default start, then the warm start."""
        key = (omega, vx, rho)
        if key not in self._cache:
            for guess in [None] + ([warm] if warm is not None else []):
                result = self._solve(omega, vx, rho, guess)
                if result[3]:
                    break
            self._cache[key] = result
        return self._cache[key]

    def _solve(self, omega, vx, rho, guess):
        values = tuple(jnp.asarray(v, dtype=float) for v in (omega, vx, 0.0, self.thrust_ref, rho))
        params = eqx.tree_at(lambda p: (p.omega, p.V_x, p.V_yz, p.thrust_target, p.rho), self.params, values)
        thrust, torque, error, state, reynolds_75 = _point(params, _seed(params) if guess is None else guess)
        ok = bool(error <= self.cfg.V_i_0_eps) and bool(np.isfinite(thrust) and np.isfinite(torque))
        return float(thrust), float(torque), float(reynolds_75), ok, state


def load_apc(filepath, n_r=None):
    """A Case from an APC geometry file (propmodel.apc_model, or Propeller.from_apc with n_r radial stations)."""
    if n_r is None:
        prop = propmodel.apc_model(str(filepath), plot=False)[0]
    else:
        prop = init.Propeller.from_apc(str(filepath), inputs=init.Inputs(n_r=n_r))
    return Case(prop)


## Performance table

def table_row(omega, vx, thrust, torque, reynolds_75, converged, R_tip, rho=RHO, sound=343.0):
    """One row of /performance. A point that did not converge keeps J, rpm, V and tip Mach, and NaN elsewhere."""
    n, D = omega / (2 * math.pi), 2 * R_tip
    J = vx / (n * D)
    nan = float("nan")
    row = dict(J=J, rpm=omega * 30 / math.pi, V_mps=vx, CT=nan, CP=nan, CQ=nan, eta=nan,
               tip_mach=omega * R_tip / sound, reynolds_75=nan, converged=bool(converged))
    if converged:
        CT = thrust / (rho * n**2 * D**4)
        CP = torque * omega / (rho * n**3 * D**5)
        row.update(CT=CT, CP=CP, CQ=torque / (rho * n**2 * D**5), reynolds_75=reynolds_75)
        row["eta"] = 0.0 if J == 0 else (J * CT / CP if CP > 0 else nan)
    return row


def performance_table(case, omega_list, vx_list, rho=RHO):
    """Return one row for each omega (rad/s) and V_x (m/s), omega outer, V_x inner."""
    rows = []
    for omega in omega_list:
        warm, first = None, len(rows)
        for vx in vx_list:
            thrust, torque, reynolds_75, ok, state = case.solve(omega, vx, rho, warm)
            warm = state if ok else warm        # start the next point from this solution
            rows.append(table_row(omega, vx, thrust, torque, reynolds_75, ok, case.R_tip, rho, case.sound))
        done = sum(r["converged"] for r in rows[first:])
        print(f"omega {omega:7.2f} rad/s: {done} of {len(vx_list)} points converged", flush=True)
    return rows


def default_omegas(case):
    """The design values of the propeller and 300 to 750 rad/s in steps of 50."""
    return sorted({round(float(w), 9) for w in case.design_omegas + list(range(300, 751, 50))})


def smallest_vx(case, omega_list, rho=RHO, candidates=(0.0, 0.1, 0.2, 0.5, 1.0, 2.0)):
    """The first candidate speed at which the solver converges for every omega."""
    for vx in candidates:
        if all(case.solve(omega, vx, rho)[3] for omega in omega_list):
            return vx
    return candidates[-1]


## Provenance

def _git(repo, *args):
    out = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed in {repo}: {out.stderr.strip()}")
    return out.stdout.strip()


def git_state(repo=ROOT):
    """(commit, dirty) of the git tree. Dirty means a changed, staged or untracked (not ignored) file."""
    return _git(repo, "rev-parse", "HEAD"), bool(_git(repo, "status", "--porcelain", "--untracked-files=normal"))


def git_user_name(repo=ROOT):
    """git config user.name, or None."""
    try:
        return _git(repo, "config", "user.name") or None
    except RuntimeError:
        return None


def resolve_author(author, repo=ROOT):
    """The author: --author, else git config user.name. Refuse when there is none."""
    author = (author or "").strip() or git_user_name(repo)
    if not author:
        raise RuntimeError("No author: give --author with the name of a person, or set git config user.name")
    return author


def require_clean(repo=ROOT):
    """The commit of a clean tree. Refuse a dirty tree."""
    commit, dirty = git_state(repo)
    if dirty:
        raise RuntimeError(f"The git tree in {repo} is dirty: commit or remove the changes before the export")
    return commit


def require_tracked(paths, repo=ROOT):
    """Refuse a design or state file that git does not track."""
    for path in paths:
        try:
            _git(repo, "ls-files", "--error-unmatch", str(Path(path).resolve()))
        except RuntimeError:
            raise RuntimeError(f"{path} is not committed in {repo}") from None


def design_sha256(design_path, state_path):
    """
    sha256 of the canonical form of the design YAML and the state JSON: the JSON text of
    {"design": <YAML content>, "state": <JSON content>} with sorted keys and no spaces. Comments, layout and line
    endings do not change it.
    """
    with open(design_path, encoding="utf-8") as f:
        design = yaml.safe_load(f)
    with open(state_path, encoding="utf-8") as f:
        state = json.load(f)
    canonical = json.dumps({"design": design, "state": state}, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


## Contract A file

def write_prop_performance(path, *, prop_id, design_hash, author, generator, prop, rows, rho=RHO, mu=None,
                           placeholder=False, created_utc=None):
    """
    Write contract A (prop-performance v1). generator is dict(name, commit, dirty). Refuse a dirty generator
    and an empty author. rows come from performance_table.
    """
    if generator.get("dirty", True):
        raise RuntimeError("Refusing to write from a dirty git tree")
    if not (author or "").strip():
        raise RuntimeError("Refusing to write without an author")
    if not rows:
        raise ValueError("No operating points to write")
    mu = float(prop.condition(0).mu) if mu is None else mu
    created_utc = created_utc or datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with h5py.File(tmp, "w") as f:
        f.attrs["schema"] = SCHEMA
        f.attrs["schema_version"] = np.int64(SCHEMA_VERSION)
        f.attrs["prop_id"] = prop_id
        f.attrs["design_sha256"] = design_hash
        f.attrs["created_utc"] = created_utc
        f.attrs["author"] = author.strip()
        f.attrs["placeholder"] = bool(placeholder)

        gen = f.create_group("generator")
        gen.attrs["name"] = generator["name"]
        gen.attrs["commit"] = generator["commit"]
        gen.attrs["dirty"] = False

        geo = f.create_group("geometry")
        geo.attrs["diameter_m"] = 2.0 * float(prop.blade.R_tip)
        geo.attrs["blades"] = np.int64(prop.blade.N)
        geo.attrs["hub_radius_m"] = float(prop.blade.R_hub)
        geo.attrs["frame"] = BLADE_FRAME
        geo.attrs["twist_sign"] = np.int64(1)
        # rotation: the design does not define the direction of rotation, so the attribute is absent
        stations = geo.create_group("stations")
        _columns(stations, STATION_COLUMNS, dict(
            r_m=np.asarray(prop.r, dtype=float),
            chord_m=np.asarray(prop.c, dtype=float),
            twist_deg=np.degrees(np.asarray(prop.beta, dtype=float)),
        ))

        perf = f.create_group("performance")
        _columns(perf, PERFORMANCE_COLUMNS, {
            k: np.array([r[k] for r in rows], dtype=bool if k == "converged" else float) for k in PERFORMANCE_COLUMNS
        })
        J = np.array([r["J"] for r in rows], dtype=float)
        perf.attrs["rho_kgpm3"] = float(rho)
        perf.attrs["nu_m2ps"] = float(mu / rho)
        perf.attrs["J_range"] = np.array([J.min(), J.max()])
        perf.attrs["rpm_values"] = np.array(sorted({r["rpm"] for r in rows}), dtype=float)

        unc = f.create_group("uncertainty")
        for item in UNCERTAINTY_ITEMS:
            group = unc.create_group(item)
            group.attrs["status"] = "unavailable"
            group.attrs["scope"] = "result"
            group.attrs["basis"] = UNCERTAINTY_BASIS
    os.replace(tmp, path)
    return path


def _columns(group, spec, values):
    for name, (units, description) in spec.items():
        ds = group.create_dataset(name, data=values[name])
        ds.attrs["units"] = units
        ds.attrs["description"] = description


def export(name, author=None, out=None, prop_id=None, rho=RHO, omega_list=None, vx_list=None, repo=ROOT):
    """
    Export designs/<name>.yaml with its state as contract A. Return the path of the .h5 file.
    Refuse a dirty tree, a missing author and design files that git does not track.
    """
    design_path, state_path = init.design_paths(name)
    author = resolve_author(author, repo)
    commit = require_clean(repo)
    require_tracked([design_path, state_path], repo)

    prop = init.Propeller.load(name)
    case = Case(prop)
    omegas = default_omegas(case) if omega_list is None else list(omega_list)
    if vx_list is None:
        vx_list = [smallest_vx(case, omegas, rho)] + [float(v) for v in range(2, 31, 2)]
    rows = performance_table(case, omegas, vx_list, rho)

    # The solve takes minutes: check again that nothing changed during it
    if git_state(repo) != (commit, False):
        raise RuntimeError("The git tree changed during the export")

    prop_id = prop_id or design_path.stem
    out = Path(out) if out else repo / "exports" / f"{prop_id}.h5"
    return write_prop_performance(
        out, prop_id=prop_id, design_hash=design_sha256(design_path, state_path), author=author,
        generator=dict(name=GENERATOR, commit=commit, dirty=False), prop=prop, rows=rows, rho=rho, mu=case.mu,
    )


## Comparison with the performance table of APC (prints only)

def read_apc_table(path):
    """Return {rpm: array with the columns J, CT, CP} from an APC PER3 file."""
    blocks, rpm = {}, None
    for line in Path(path).read_text(errors="replace").splitlines():
        if "PROP RPM" in line:
            rpm = int(float(line.split("=")[1]))
            blocks[rpm] = []
            continue
        try:
            values = [float(x) for x in line.split()]
        except ValueError:
            continue                            # a text row
        if rpm is not None and len(values) >= 5:
            blocks[rpm].append([values[1], values[3], values[4]])   # J, Ct, Cp
    return {k: np.array(v) for k, v in blocks.items()}


def compare_apc(case, dat_path, rho=RHO, rpms=(4000, 6000, 8000), j_show=(0.0, 0.2, 0.4), j_max=0.5):
    """
    Compare CT and CP with the APC table. Return (rows at j_show, summary with the RMS relative difference
    over 0 <= J <= j_max).
    """
    apc = read_apc_table(dat_path)
    shown, diffs = [], {"CT": [], "CP": []}
    for rpm in rpms:
        table = apc[rpm][apc[rpm][:, 0] <= j_max]
        n, D = rpm / 60, 2 * case.R_tip
        # Solve at the APC values of J and at the J values to show
        J = np.concatenate([table[:, 0], j_show])
        model = performance_table(case, [rpm * math.pi / 30], list(J * n * D), rho)
        for k, (j, row) in enumerate(zip(J, model)):
            if not row["converged"]:
                continue
            ct_apc, cp_apc = (np.interp(j, apc[rpm][:, 0], apc[rpm][:, c]) for c in (1, 2))
            if k < len(table):                  # a row of the APC table: counts for the RMS
                diffs["CT"].append(row["CT"] / ct_apc - 1)
                diffs["CP"].append(row["CP"] / cp_apc - 1)
            else:                               # a J value to show
                shown.append(dict(rpm=rpm, J=float(j), CT_proptimiser=row["CT"], CT_apc=float(ct_apc),
                                  CT_diff_pct=100 * (row["CT"] / ct_apc - 1), CP_proptimiser=row["CP"],
                                  CP_apc=float(cp_apc), CP_diff_pct=100 * (row["CP"] / cp_apc - 1)))
    both = np.array(diffs["CT"] + diffs["CP"])
    rms = lambda x: float(100 * np.sqrt(np.mean(np.square(x))))
    summary = dict(
        apc_file=Path(dat_path).name, rpm=list(rpms), J_range=[0.0, j_max],
        rms_rel_diff_CT_pct=rms(diffs["CT"]), rms_rel_diff_CP_pct=rms(diffs["CP"]), rms_rel_diff_pct=rms(both),
        mean_rel_diff_CT_pct=float(100 * np.mean(diffs["CT"])), mean_rel_diff_CP_pct=float(100 * np.mean(diffs["CP"])),
        n_points_per_coefficient=len(diffs["CT"]), n_points_pooled=len(both),
    )
    return shown, summary


def print_comparison(shown, summary):
    print("rpm     J      CT_prop  CT_apc   diff%    CP_prop  CP_apc   diff%")
    for r in shown:
        print(f"{r['rpm']:5d}  {r['J']:.1f}  {r['CT_proptimiser']:.5f}  {r['CT_apc']:.5f}  {r['CT_diff_pct']:+6.1f}"
              f"  {r['CP_proptimiser']:.5f}  {r['CP_apc']:.5f}  {r['CP_diff_pct']:+6.1f}")
    print(f"RMS relative difference, J 0 to {summary['J_range'][1]}, rpm {summary['rpm']}:")
    print(f"  CT {summary['rms_rel_diff_CT_pct']:.1f}%   CP {summary['rms_rel_diff_CP_pct']:.1f}%   "
          f"both {summary['rms_rel_diff_pct']:.1f}%   n = {summary['n_points_per_coefficient']} per coefficient, "
          f"{summary['n_points_pooled']} pooled")
    print(f"  mean signed difference: CT {summary['mean_rel_diff_CT_pct']:+.1f}%   CP {summary['mean_rel_diff_CP_pct']:+.1f}%")


## Command line

def main(argv=None):
    parser = argparse.ArgumentParser(description="Export the BEMT performance of a design as prop-performance v1 (.h5).")
    parser.add_argument("design", nargs="?", help="design name (designs/<name>.yaml)")
    parser.add_argument("--author", help="the person who makes the export (default: git config user.name)")
    parser.add_argument("--prop-id", help="prop_id in the file (default: the design name)")
    parser.add_argument("--out", help="output .h5 file (default: exports/<prop_id>.h5)")
    parser.add_argument("--rho", type=float, default=RHO, help="air density in kg/m3")
    parser.add_argument("--apc", help="APC geometry file (.PE0): compare with --compare-apc, write no file")
    parser.add_argument("--compare-apc", help="APC performance file (PER3 .dat)")
    parser.add_argument("--n-r", type=int, help="radial stations of the APC propeller (default: apc_model)")
    args = parser.parse_args(argv)

    if args.apc:
        if args.design or not args.compare_apc:
            parser.error("--apc needs --compare-apc and no design name")
        print_comparison(*compare_apc(load_apc(args.apc, args.n_r), args.compare_apc, args.rho))
        return
    if not args.design:
        parser.error("give a design name")
    try:
        path = export(args.design, args.author, args.out, args.prop_id, args.rho)
    except RuntimeError as e:
        parser.exit(2, f"export refused: {e}\n")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()

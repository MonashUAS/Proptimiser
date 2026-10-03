"""
Export the BEMT performance of one propeller as a CSV table and a YAML sidecar.

The script sweeps the axial speed V_x at fixed rotor speed omega. It solves the induced
velocity at each point. It does not run the optimiser. Other tools read the two files.

The same file runs in two code trees:
    new code   init.Propeller.load(name) reads propellers/<name>.json
    old code   results.loadprop(name) reads the .eqx files of commit 588118a
The code picks the tree from what the imported modules offer.

Examples, run from the repository root:
    python export_performance.py Currawong_V1
    python export_performance.py --apc apc_prop_data/propgeom_data/20x8E-PERF.PE0 \
        --compare-apc apc_prop_data/performance_data/PER3_20x8E.dat
Old code: copy this file into the old tree and run it there with --commit 588118a.
"""
import argparse
import csv
import datetime
import hashlib
import math
import os
import subprocess
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")      # no windows: the solver modules import pyplot

import numpy as np
import yaml
import jax
import jax.numpy as jnp
import equinox as eqx

import init
import propmodel
import proptimiser
import results

VERSION = "1.0"
REPO = "Propcalc-2_prop-er_faster_stronger"
RHO = 1.225                 # air density (kg/m3)
SOUND = 343.0               # speed of sound for the tip Mach number (m/s)
COLUMNS = ["omega_rad_s", "rpm", "V_x_mps", "J", "CT", "CP", "CQ", "thrust_N", "torque_Nm",
           "shaft_power_W", "eta", "tip_mach", "converged"]
SWITCHES = ["inflow_model", "coeff_method", "tip_loss_model", "root_loss_model", "rot_corr_model"]


## Solver access
# One small class for each code tree. Both offer the same few methods.

@jax.jit
def _new_point(params, guess):
    # New code: solve one point and measure the scaled BEMT residual of the solution.
    params = eqx.tree_at(lambda p: p.V_i_0, params, guess)
    V_i, V_i_0 = propmodel.induced_velocity(params)
    thrust = jnp.squeeze(propmodel.thrust(V_i, params))
    torque = jnp.squeeze(propmodel.torque(V_i, params))
    lam = V_i_0[params.cfg.solve_rows] / propmodel.get_velocity_scales(params)
    residual = jnp.abs(propmodel.bemt_residuals(lam, params))
    return thrust, torque, jnp.max(jnp.where(propmodel.aero_zone(params), residual, 0.0)), V_i_0


@jax.jit
def _new_seed(params):
    # New code: the momentum theory guess that Propeller uses as its start.
    scale = propmodel.get_velocity_scales(params)
    seed = jnp.broadcast_to(scale, (scale.shape[0], params.r.shape[0]))
    return jnp.zeros_like(params.V_i_0).at[params.cfg.solve_rows].set(seed)


@jax.jit
def _old_point(params):
    # Old code: solve one point and measure the norm of the thrust residual (the Broyden stop test).
    V_ia, V_ia_0 = propmodel.induced_velocity(params)
    residual = jnp.linalg.norm(propmodel.thrust_residual(V_ia_0, params))
    return propmodel.thrust(V_ia, params), propmodel.torque(V_ia, params), residual, V_ia_0


class _Case:
    """One propeller together with the solver of the imported code."""

    def __init__(self, files, note):
        self.files, self.note, self._cache = files, note, {}

    def solve(self, omega, vx, vyz, rho=RHO, warm=None):
        """Return (thrust N, torque Nm, converged, state). Try the default start, then the warm start."""
        key = (omega, vx, vyz, rho)
        if key not in self._cache:
            for guess in [None] + ([warm] if warm is not None else []):
                result = self._solve(omega, vx, vyz, rho, guess)
                if result[2]:
                    break
            self._cache[key] = result
        return self._cache[key]

    def _point_params(self, omega, vx, vyz, rho, thrust_target=None):
        values = (omega, vx, vyz, self.thrust_ref if thrust_target is None else thrust_target, rho)
        values = tuple(jnp.asarray(v, dtype=float) for v in values)
        return eqx.tree_at(lambda p: (p.omega, p.V_x, p.V_yz, p.thrust_target, p.rho), self.params, values)


class _NewCase(_Case):
    """A propeller of the new code (init.Propeller)."""

    def __init__(self, prop, files, note):
        super().__init__(files, note)
        self.prop, self.params, self.cfg = prop, prop.condition(0), prop.cfg
        self.R_tip, self.blades = self.cfg.R_tip, prop.blade.N
        self.design_omegas = [float(w) for w in prop.omega]
        # The target thrust only scales the residuals and the start. A fixed value keeps the test the same at all points.
        self.thrust_ref = float(np.max(prop.inputs.thrust_target))

    def _solve(self, omega, vx, vyz, rho, guess):
        params = self._point_params(omega, vx, vyz, rho)
        thrust, torque, error, state = _new_point(params, _new_seed(params) if guess is None else guess)
        ok = bool(error <= self.cfg.V_i_0_eps) and bool(np.isfinite(thrust) and np.isfinite(torque))
        return float(thrust), float(torque), ok, state

    def design_points(self):
        prop, _, rows = results.evaluate_conditions(self.prop)
        return [dict(thrust_target_N=row["thrust_target"], thrust_solved_N=row["thrust"],
                     omega_rad_s=row["omega"], V_x_mps=float(prop.inputs.V_x[i]),
                     V_yz_mps=float(prop.inputs.V_yz[i]), shaft_power_W=row["shaft_power"],
                     motor_power_W=row["motor_power"]) for i, row in enumerate(rows)]

    def model(self):
        model = {k: getattr(self.cfg, k) for k in SWITCHES}
        return dict(model, bemt_type=list(self.prop.inputs.bemt_type), n_r=self.prop.inputs.n_r,
                    n_psi=self.prop.inputs.n_psi)

    def motor(self):
        return dict(kv=self.cfg.kV, rm=self.cfg.R_m, i0=self.cfg.I_0)


class _OldCase(_Case):
    """A propeller of the old code (init.Params). The old physics reads the class defaults of init.Inputs."""

    def __init__(self, params, files, note):
        super().__init__(files, note)
        self.params = params
        self.R_tip, self.blades = float(params.r[-1]), int(np.ravel(params.N)[0])
        self.design_omegas = [float(w) for w in init.Inputs.omega]
        self.thrust_ref = float(np.max(init.Inputs.thrust_target))

    def _params(self, omega, vx, vyz, rho, thrust_target, guess):
        params = self._point_params(omega, vx, vyz, rho, thrust_target)
        start = jnp.full(self.params.r.shape, 4.0) if guess is None else guess     # the optimiser start
        return eqx.tree_at(lambda p: p.V_ia_0, params, start)

    def _solve(self, omega, vx, vyz, rho, guess):
        thrust, torque, error, state = _old_point(self._params(omega, vx, vyz, rho, self.thrust_ref, guess))
        ok = bool(error <= init.Inputs.V_ia_0_eps) and bool(np.isfinite(thrust) and np.isfinite(torque))
        return float(thrust), float(torque), ok, state

    def design_points(self):
        points = []
        for i in range(int(init.Inputs.n_cond)):
            target = float(init.Inputs.thrust_target[i])
            params = self._params(float(init.Inputs.omega[i]), float(init.Inputs.V_x[i]),
                                  float(init.Inputs.V_yz[i]), RHO, target, None)
            motor_power, thrust, _, _ = proptimiser.objectives(params)
            V_ia, _ = propmodel.induced_velocity(params)
            points.append(dict(thrust_target_N=target, thrust_solved_N=float(thrust),
                               omega_rad_s=float(params.omega), V_x_mps=float(params.V_x),
                               V_yz_mps=float(params.V_yz),
                               shaft_power_W=float(propmodel.mechanicalPower(V_ia, params)),
                               motor_power_W=float(motor_power)))
        return points

    def model(self):
        model = {k: getattr(init.Inputs, k) for k in SWITCHES}
        return dict(model, n_r=int(self.params.r.shape[0]), n_psi=int(self.params.psi.shape[0]))

    def motor(self):
        return dict(kv=init.Inputs.kV, rm=init.Inputs.R_m, i0=init.Inputs.I_0)


## Loaders
# A loader is a function with no argument. It returns a case.

NEW_CODE = hasattr(init, "Propeller")


def make_case(prop, files, note):
    """Wrap a loaded propeller (init.Propeller in the new code, init.Params in the old code)."""
    return (_NewCase if NEW_CODE else _OldCase)(prop, files, note)


def load_named(name):
    """Load a propeller by name with the loader of the imported code."""
    if NEW_CODE:
        return make_case(init.Propeller.load(name), [Path(init.prop_path(name)).name],
                         "Geometry from the JSON file of the propeller library.")
    _, params = results.loadprop(name)
    return make_case(params, [f"prop_inputs_{name}.eqx", f"prop_params_{name}.eqx"],
                     "Geometry saved by the old code. Read with results.loadprop.")


def load_apc(filepath, n_r=None):
    """
    Build a propeller from an APC geometry file with propmodel.apc_model.
    The option n_r sets the number of radial stations. It needs the new code.
    """
    if n_r is None:
        prop = propmodel.apc_model(str(filepath), plot=False)[0]
    else:
        prop = init.Propeller.from_apc(str(filepath), inputs=init.Inputs(n_r=n_r))
    note = "Chord and twist from the APC file. Default blade and inputs of apc_model."
    if n_r is not None:
        note += f" The number of radial stations is {n_r}."
    return make_case(prop, [Path(filepath).name],
                     note + " The design points are the default flight conditions, not APC design points.")


## Performance table

def table_row(omega, vx, thrust, torque, converged, R_tip, rho=RHO):
    """One CSV row. The coefficients use the shaft power. A point that did not converge gets NaN."""
    n, D = omega / (2 * math.pi), 2 * R_tip
    J = vx / (n * D)
    nan = float("nan")
    row = dict(omega_rad_s=omega, rpm=omega * 30 / math.pi, V_x_mps=vx, J=J, CT=nan, CP=nan, CQ=nan,
               thrust_N=nan, torque_Nm=nan, shaft_power_W=nan, eta=nan,
               tip_mach=omega * R_tip / SOUND, converged=int(converged))
    if converged:
        power = torque * omega
        CP = power / (rho * n**3 * D**5)
        row.update(thrust_N=thrust, torque_Nm=torque, shaft_power_W=power,
                   CT=thrust / (rho * n**2 * D**4), CP=CP, CQ=CP / (2 * math.pi))
        row["eta"] = 0.0 if J == 0 else (J * row["CT"] / CP if CP > 0 else nan)
    return row


def performance_table(prop_loader, omega_list, vx_list, rho=RHO, vyz=0.0):
    """Return one row (a dict with the CSV columns) for each omega and V_x. vyz is the in-plane speed."""
    return _sweep(prop_loader(), omega_list, vx_list, rho, vyz)


def _sweep(case, omega_list, vx_list, rho, vyz):
    rows = []
    for omega in omega_list:
        warm, first = None, len(rows)
        for vx in vx_list:
            thrust, torque, ok, state = case.solve(omega, vx, vyz, rho, warm)
            warm = state if ok else warm        # start the next point from this solution
            rows.append(table_row(omega, vx, thrust, torque, ok, case.R_tip, rho))
        done = sum(r["converged"] for r in rows[first:])
        print(f"omega {omega:7.2f} rad/s: {done} of {len(vx_list)} points converged", flush=True)
    return rows


def default_omegas(case):
    """The design values of the propeller and 300 to 750 rad/s in steps of 50."""
    return sorted({round(float(w), 9) for w in case.design_omegas + list(range(300, 751, 50))})


def smallest_vx(case, omega_list, rho=RHO, vyz=0.0, candidates=(0.0, 0.1, 0.2, 0.5, 1.0, 2.0)):
    """The first candidate speed at which the solver converges for every omega."""
    for vx in candidates:
        if all(case.solve(omega, vx, vyz, rho)[2] for omega in omega_list):
            return vx
    return candidates[-1]


## Files

def write_csv(rows, path):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(COLUMNS)
        writer.writerows([row[c] for c in COLUMNS] for row in rows)


def sha256_of(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_short_hash():
    """The 7-character hash of HEAD in the folder of this file, or None."""
    try:
        out = subprocess.run(["git", "rev-parse", "--short=7", "HEAD"], cwd=Path(__file__).resolve().parent,
                             capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def export(prop_loader, prop_id, commit, out_dir="performance_exports", rho=RHO, vyz=0.0,
           omega_list=None, vx_list=None, producer="MUAS", code_commit=None, note=""):
    """Write <prop_id>__<commit>.perf.csv and .perf.yaml into out_dir. Return both paths."""
    case = prop_loader()
    omegas = default_omegas(case) if omega_list is None else list(omega_list)
    if vx_list is None:
        vx_list = [smallest_vx(case, omegas, rho, vyz)] + [float(v) for v in range(2, 31, 2)]
    rows = _sweep(case, omegas, vx_list, rho, vyz)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{prop_id}__{commit}.perf.csv"
    yaml_path = out_dir / f"{prop_id}__{commit}.perf.yaml"
    write_csv(rows, csv_path)

    # The smallest V_x at which every omega converged
    good = [vx for vx in vx_list if all(r["converged"] for r in rows if r["V_x_mps"] == vx)]
    note = " ".join(filter(None, [case.note, note, f"Solver code from commit {code_commit or commit}."]))
    doc = dict(
        schema="prop-performance", schema_version=1, producer=producer, kind="muas_bemt", prop_id=prop_id,
        source=dict(repository=REPO, commit=commit, files=case.files, note=note),
        D_m=2 * case.R_tip, blades=case.blades, rho=rho,
        model=dict(case.model(), V_yz_mps=vyz),
        design_points=case.design_points(), motor_assumed=case.motor(),
        V_x_min_converged_mps=min(good) if good else None,
        generated=datetime.date.today().isoformat(), exporter=f"{Path(__file__).name} {VERSION}",
        csv_sha256=sha256_of(csv_path),
    )
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(_plain(doc), f, sort_keys=False)
    return csv_path, yaml_path


def _plain(value):
    """Turn numpy numbers into Python numbers so that YAML can write them."""
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, (np.generic, jax.Array)):
        return np.asarray(value).item()
    return value


## Comparison with the performance table of APC

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


def compare_apc(prop_loader, dat_path, out_dir="performance_exports", prop_id="APC", rho=RHO,
                rpms=(4000, 6000, 8000), j_show=(0.0, 0.2, 0.4), j_max=0.5):
    """
    Compare CT and CP with the APC table. Write <prop_id>__apc.compare.csv (CT and CP at j_show)
    and .compare.yaml (the RMS relative difference over 0 <= J <= j_max). Return (rows, summary).
    """
    case, apc = prop_loader(), read_apc_table(dat_path)
    shown, diffs = [], {"CT": [], "CP": []}
    for rpm in rpms:
        table = apc[rpm][apc[rpm][:, 0] <= j_max]
        n, D = rpm / 60, 2 * case.R_tip
        # Solve at the APC values of J and at the J values to show
        J = np.concatenate([table[:, 0], j_show])
        model = performance_table(lambda: case, [rpm * math.pi / 30], list(J * n * D), rho)
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
        n_r=case.model()["n_r"], n_psi=case.model()["n_psi"],
        rms_rel_diff_CT_pct=rms(diffs["CT"]), rms_rel_diff_CP_pct=rms(diffs["CP"]), rms_rel_diff_pct=rms(both),
        mean_rel_diff_CT_pct=float(100 * np.mean(diffs["CT"])), mean_rel_diff_CP_pct=float(100 * np.mean(diffs["CP"])),
        n_points_per_coefficient=len(diffs["CT"]), n_points_pooled=len(both),
    )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{prop_id}__apc.compare.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, list(shown[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(shown)
    with open(out_dir / f"{prop_id}__apc.compare.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(summary, f, sort_keys=False)
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
    parser = argparse.ArgumentParser(description="Export the BEMT performance table of a propeller.")
    parser.add_argument("prop", nargs="?", help="propeller name")
    parser.add_argument("--apc", help="APC geometry file (.PE0) in place of a propeller name")
    parser.add_argument("--compare-apc", help="APC performance file (PER3 .dat). Also write the CT and CP comparison.")
    parser.add_argument("--n-r", type=int, help="radial stations of the APC propeller (default: apc_model)")
    parser.add_argument("--prop-id", help="id in the file names (default: the propeller name)")
    parser.add_argument("--commit", help="7-character commit of the geometry (default: git HEAD)")
    parser.add_argument("--out", default="performance_exports", help="output folder")
    parser.add_argument("--rho", type=float, default=RHO, help="air density in kg/m3")
    parser.add_argument("--note", default="", help="sentence to add to the geometry note in the YAML file")
    parser.add_argument("--vyz", type=float, default=0.0, help="in-plane speed in m/s (0 is a pure axial flow)")
    args = parser.parse_args(argv)
    if bool(args.prop) == bool(args.apc):
        parser.error("give either a propeller name or --apc")

    code_commit = args.commit or git_short_hash()
    if args.apc:
        case = load_apc(args.apc, args.n_r)
        prop_id = args.prop_id or "APC_" + Path(args.apc).stem.split("-")[0]
        commit, producer = "apc", "APC-geometry-via-Proptimiser"
    else:
        case = load_named(args.prop)
        prop_id, commit, producer = args.prop_id or args.prop, code_commit, "MUAS"
        if commit is None:
            parser.error("git does not answer: give --commit")

    loader = lambda: case                       # load once, so the solver compiles once
    csv_path, yaml_path = export(loader, prop_id, commit, args.out, args.rho, args.vyz, producer=producer,
                                 code_commit=code_commit, note=args.note)
    print(f"Wrote {csv_path}\nWrote {yaml_path}")
    if args.compare_apc:
        print_comparison(*compare_apc(loader, args.compare_apc, args.out, prop_id, args.rho))


if __name__ == "__main__":
    main()

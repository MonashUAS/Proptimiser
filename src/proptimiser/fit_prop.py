"""
Fit a propeller with no public geometry to a manufacturer bench table (thrust and shaft power against rpm).

The result is a FITTED model, not a measured one. The blade is a chord and twist family with a few free
parameters. The BEMT of Proptimiser gives thrust and shaft power. A least squares fit with priors on the
shape parameters keeps the blade physical.

The pure functions (read_maker_table, chart_targets, blade_family, residuals, error_summary, figure_of_merit) hold
no solver. fit_propeller and model_curve use the BEMT through export_performance.Case.

Run from the repository root:
    .venv/Scripts/python -m proptimiser.fit_prop
"""
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

G0 = 9.80665                # N per kgf
INCH = 0.0254               # m per inch
RHO = 1.225                 # air density (kg/m3), sea level, same as the exporter

# Names of the free parameters, in the order of the vector theta
PARAM_NAMES = ("c75_over_R", "dtheta_deg", "a_inboard", "b_tip", "pitch_gradient")


@dataclass(frozen=True)
class Target:
    """What the fit follows. Lengths in m, pitch in m."""
    diameter_m: float
    pitch_m: float
    blades: int
    r_hub_m: float
    sigma_thrust: float = 0.03          # relative weight scale of the thrust. The maker values are "theoretical".
    sigma_power: float = 0.03           # relative weight scale of the shaft power. This is a choice, not a measured error.


# Priors of the shape parameters: (value, sigma). c75_over_R has no prior. Hobby props peak at 0.5R to 0.7R.
PRIORS = {
    "dtheta_deg": (0.0, 2.0),           # the quoted pitch holds at 0.75R within 2 deg
    "a_inboard": (0.4, 0.3),
    "b_tip": (0.35, 0.2),
    "pitch_gradient": (0.0, 0.1),       # a constant geometric pitch
}
THETA_START = (0.13, 0.0, 0.4, 0.35, 0.0)
THETA_LOWER = (0.04, -4.0, 0.0, 0.0, -0.4)
THETA_UPPER = (0.30, 4.0, 1.5, 1.5, 0.4)


def read_maker_table(path):
    """
    Read the maker bench table CSV. A line that starts with # is a comment. The columns used are
    rpm, thrust_gf, output_W (shaft power) and torque_Nm. Return a dict of 1-D float arrays.
    """
    with open(path, encoding="utf-8", newline="") as f:
        lines = [line for line in f if line.strip() and not line.lstrip().startswith("#")]
    rows = list(csv.DictReader(lines))
    keys = ("rpm", "thrust_gf", "output_W", "torque_Nm")
    missing = [k for k in keys if not rows or k not in rows[0]]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    return {k: np.array([float(r[k]) for r in rows]) for k in keys}


def chart_targets(table):
    """Return (rpm, thrust in N, shaft power in W). The shaft power is the maker output power."""
    return table["rpm"], table["thrust_gf"] * 1e-3 * G0, table["output_W"]


def shaft_power_check(table):
    """Relative difference of torque * omega from output_W, per row. The table must agree to a rounding error."""
    omega = table["rpm"] * math.pi / 30.0
    return table["torque_Nm"] * omega / table["output_W"] - 1.0


def blade_family(theta, r, target):
    """
    Chord (m) and twist beta (rad) at the radii r (m) for the parameter vector theta.

    chord:  c(x) = c75 (x / 0.75)^a ((1 - x + e) / (0.25 + e))^b, x = r / R, e = 0.02. c75 = c75_over_R R.
    twist:  tan(beta) = P(x) / (2 pi r) + dtheta, with P(x) = P0 (1 + g (x - 0.75)).
    P0 is the quoted geometric pitch. dtheta is an offset angle (deg) from the chord line to the
    quoted pitch line.
    """
    c75_over_R, dtheta_deg, a, b, g = (float(v) for v in theta)
    R = 0.5 * target.diameter_m
    x = np.asarray(r, dtype=float) / R
    eps = 0.02
    chord = c75_over_R * R * (x / 0.75) ** a * ((1.0 - x + eps) / (0.25 + eps)) ** b
    pitch = target.pitch_m * (1.0 + g * (x - 0.75))
    beta = np.arctan(pitch / (2.0 * math.pi * np.asarray(r, dtype=float))) + math.radians(dtheta_deg)
    return chord, beta


def prior_residuals(theta):
    """Normalised distance of the shape parameters from their priors."""
    out = []
    for name, value in zip(PARAM_NAMES, theta):
        if name in PRIORS:
            mean, sigma = PRIORS[name]
            out.append((float(value) - mean) / sigma)
    return np.array(out)


def residuals(thrust_model, power_model, thrust_chart, power_chart, target):
    """Log ratio residuals of thrust and power, each divided by its reading error. Return one 1-D array."""
    rt = np.log(np.asarray(thrust_model) / np.asarray(thrust_chart)) / target.sigma_thrust
    rp = np.log(np.asarray(power_model) / np.asarray(power_chart)) / target.sigma_power
    return np.concatenate([rt, rp])


def power_at_thrust(thrust_model, power_model, thrust_query):
    """
    Model shaft power at the thrust values thrust_query. Interpolate log P against log T along the model curve.
    Outside the curve, extend with the slope of the end segment. The model thrust must increase with the index.
    """
    lt, lp = np.log(np.asarray(thrust_model, dtype=float)), np.log(np.asarray(power_model, dtype=float))
    q = np.log(np.asarray(thrust_query, dtype=float))
    out = np.interp(q, lt, lp)
    lo, hi = q < lt[0], q > lt[-1]
    out = np.where(lo, lp[0] + (q - lt[0]) * (lp[1] - lp[0]) / (lt[1] - lt[0]), out)
    out = np.where(hi, lp[-1] + (q - lt[-1]) * (lp[-1] - lp[-2]) / (lt[-1] - lt[-2]), out)
    return np.exp(out)


def hover_weights(rpm, low=2800.0, high=3800.0, factor=2.0):
    """Residual weight: factor inside the hover range [low, high] rpm, 1 outside."""
    rpm = np.asarray(rpm, dtype=float)
    return np.where((rpm >= low) & (rpm <= high), factor, 1.0)


def equal_thrust_residuals(thrust_model, power_model, thrust_chart, power_chart, target):
    """Log ratio of the model power at the chart thrust to the chart power, over the power reading error."""
    p = power_at_thrust(thrust_model, power_model, thrust_chart)
    return np.log(p / np.asarray(power_chart)) / target.sigma_power


def error_summary(model, chart):
    """Relative errors (model / chart - 1), their RMS and their largest absolute value."""
    err = np.asarray(model, dtype=float) / np.asarray(chart, dtype=float) - 1.0
    return dict(error=err, rms=float(np.sqrt(np.mean(err**2))), max_abs=float(np.max(np.abs(err))))


def gate(thrust_summary, power_summary, rms_limit=0.05, max_limit=0.10):
    """The validation gate: RMS at most 5 % and no point worse than 10 %, for thrust and for power."""
    ok = True
    for s in (thrust_summary, power_summary):
        ok = ok and s["rms"] <= rms_limit and s["max_abs"] <= max_limit
    return bool(ok)


def figure_of_merit(thrust, power, diameter, rho=RHO):
    """Figure of merit T^1.5 / (sqrt(2 rho A) P) for the static thrust T (N) and shaft power P (W)."""
    area = math.pi * diameter**2 / 4.0
    return thrust**1.5 / (math.sqrt(2.0 * rho * area) * power)


def torque_over_thrust_diameter(thrust, power, omega, diameter):
    """Q / (T D) with the torque Q = P / omega."""
    return power / omega / (thrust * diameter)


## BEMT access

def build_propeller(theta, target, inputs, blade_kwargs):
    """A Propeller with the chord and twist of blade_family(theta)."""
    from . import init

    blade = init.BladeSpec(**blade_kwargs)
    r, _ = init.geometry.radial_stops(blade.R_tip, blade.R_hub, inputs.n_r)
    chord, beta = blade_family(theta, np.asarray(r), target)
    r_hub_chord, r_hub_beta = blade_family(theta, np.array([blade.R_hub]), target)
    blade = init.dataclasses.replace(blade, hub_chord=float(r_hub_chord[0]),
                                     hub_twist_deg=float(np.degrees(r_hub_beta[0])))
    return init.Propeller(inputs, blade, c=chord, beta=beta,
                          cst=init.geometry.get_cst_weights_from_names(blade.airfoils))


def model_curve(prop, rpm_list, vx=0.0, rho=RHO):
    """Return (thrust N, shaft power W, converged) at each rpm for one axial speed, from the BEMT."""
    from . import export_performance as ep

    case = ep.Case(prop)
    thrust, power, ok = [], [], []
    for rpm in rpm_list:
        omega = rpm * math.pi / 30.0
        t, q, _, converged, _ = case.solve(omega, vx, rho)
        thrust.append(t)
        power.append(q * omega)
        ok.append(converged)
    return np.array(thrust), np.array(power), np.array(ok)


def fit_propeller(target, chart, inputs, blade_kwargs, theta0=THETA_START, verbose=1):
    """
    Least squares fit of theta to the chart. Return (theta, Propeller, scipy result).
    The residuals are the log error of the thrust against rpm and of the shaft power against rpm (Fit C), each
    over its weight scale and doubled in the hover range 2800 to 3800 rpm, plus the prior terms.
    """
    from scipy.optimize import least_squares

    rpm, thrust_chart, power_chart = chart_targets(chart)

    def fun(theta):
        prop = build_propeller(theta, target, inputs, blade_kwargs)
        thrust, power, ok = model_curve(prop, rpm)
        if not ok.all() or np.any(thrust <= 0) or np.any(power <= 0):
            return np.full(2 * len(rpm) + len(prior_residuals(theta)), 1e3)
        w = hover_weights(rpm)
        rt = np.log(thrust / thrust_chart) / target.sigma_thrust
        rp = np.log(power / power_chart) / target.sigma_power
        return np.concatenate([w * rt, w * rp, prior_residuals(theta)])

    result = least_squares(fun, np.asarray(theta0, dtype=float), bounds=(THETA_LOWER, THETA_UPPER),
                           diff_step=1e-3, x_scale=(0.05, 1.0, 0.3, 0.3, 0.1), verbose=2 if verbose else 0)
    theta = result.x
    return theta, build_propeller(theta, target, inputs, blade_kwargs), result


## The MAD FLUXER PRO 22x6.6 fit

NAME = "MAD_FLUXER_22x6_6"             # Fit C, the default. Fit B is MAD_FLUXER_22x6_6_fitB.
GATE_RPM_MIN = 2490.0                  # the table row at 2492 rpm stands for the 2500 rpm limit
FLUXER_TARGET = Target(diameter_m=22 * INCH, pitch_m=6.6 * INCH, blades=2, r_hub_m=0.025)
HOVER_RPM = 3200.0
FLUXER_NOTE = """\
# FITTED model of the MAD FLUXER PRO 22x6.6 MATT. It is not a measurement and not the maker geometry.
# No blade geometry of this propeller is public. Chord and twist come from the 5-parameter family of
# proptimiser.fit_prop (blade_family), fitted to designs/MAD_FLUXER_22x6_6_maker.csv (maker bench table, shaft power) with the BEMT (Fit C).
# Fitted parameters {params}.
# The sections are {airfoil}, an assumption. The motor keys (I_0, R_tot, kappa, Q0, P_max) are class
# defaults and do not describe the MAD M6C10 IPE V3. The export does not use them. kV 150 is the maker value.
# The design condition is the hover point: 3200 rpm, 22.4 N, V_x = 0.
"""


def fluxer_setup(airfoil="clarky"):
    """(Inputs, blade keyword dict) of the FLUXER design. Settings follow the Currawong V1 design."""
    from . import init

    R = FLUXER_TARGET.diameter_m / 2
    inputs = init.Inputs(
        opt_solver="L-BFGS-B", n_r=25, n_psi=12, thrust_target=[22.4], V_x=[0.0], V_yz=[0.0],
        omega=[HOVER_RPM * math.pi / 30.0], tilt=[0.0], solve_omega=[False], solve_tilt=[False],
        objective_weight=[1.0], kV=150.0, V_bus=0.0, inflow_model=1, coeff_method=3, tip_loss_model=1,
        root_loss_model=1, rot_corr_model=1, V_i_0_max_iter=50)
    blade = dict(N=2, r_foils=(0.0, FLUXER_TARGET.r_hub_m, 0.05, 0.15, R), airfoils=(airfoil,) * 3)
    return inputs, blade


def apc_static_reference(dat_path, rpm):
    """(CT, CP, FoM) of an APC table at J = 0, linear in rpm between the table rows."""
    from .export_performance import read_apc_table

    table = read_apc_table(dat_path)
    rpms = sorted(table)
    ct = np.interp(rpm, rpms, [table[k][0, 1] for k in rpms])
    cp = np.interp(rpm, rpms, [table[k][0, 2] for k in rpms])
    return float(ct), float(cp), float(ct**1.5 / (cp * math.sqrt(math.pi / 2)))


HOVER_THRUST_N = 22.4       # thrust of one rotor in hover


def table_at_thrust(table, thrust_n):
    """Maker shaft power (W) and rpm at the thrust thrust_n (N): log-log interpolation on the maker table."""
    rpm, thrust, power = chart_targets(table)
    return (float(power_at_thrust(thrust, power, [thrust_n])[0]),
            float(np.exp(np.interp(np.log(thrust_n), np.log(thrust), np.log(rpm)))))


def evaluate(prop, table, rho=RHO, hover_thrust_n=HOVER_THRUST_N):
    """
    Compare the model with the maker table. Return a dict with the per-row arrays and the summaries.
    The gate rows are the rows at or above GATE_RPM_MIN. The hover view compares the shaft power at the thrust
    hover_thrust_n (log-log interpolation on the model curve and on the table).
    """
    D = FLUXER_TARGET.diameter_m
    rpm, thrust_t, power_t = chart_targets(table)
    thrust, power, ok = model_curve(prop, rpm)
    p_eq = power_at_thrust(thrust, power, thrust_t)
    g = rpm >= GATE_RPM_MIN
    out = dict(rpm=rpm, thrust_table=thrust_t, thrust_model=thrust, power_table=power_t, power_model=power,
               power_at_table_thrust=p_eq, converged=ok, gate_rows=g,
               fom_table=figure_of_merit(thrust_t, power_t, D, rho), fom_model=figure_of_merit(thrust, power, D, rho))
    out["thrust"] = error_summary(thrust[g], thrust_t[g])
    out["power"] = error_summary(power[g], power_t[g])
    out["equal_thrust"] = error_summary(p_eq[g], power_t[g])
    out["fom_diff"] = error_summary(out["fom_model"][g], out["fom_table"][g])
    p_hover_table, rpm_hover = table_at_thrust(table, hover_thrust_n)
    p_hover_model = float(power_at_thrust(thrust, power, [hover_thrust_n])[0])
    out["hover"] = dict(thrust_n=hover_thrust_n, rpm_table=rpm_hover, power_table=p_hover_table,
                        power_model=p_hover_model, error=p_hover_model / p_hover_table - 1.0,
                        fom_table=float(figure_of_merit(hover_thrust_n, p_hover_table, D, rho)),
                        fom_model=float(figure_of_merit(hover_thrust_n, p_hover_model, D, rho)))
    out["gate"] = bool(gate(out["thrust"], out["power"]) and abs(out["hover"]["error"]) <= 0.07)
    return out


def reynolds_at(prop, rpm, x=0.75, rho=RHO, mu=1.798e-5):
    """Chord Reynolds number and section Mach number of the model at x = r / R, static, at the given rpm."""
    r = np.asarray(prop.r)
    R = float(prop.blade.R_tip)
    c = float(np.interp(x * R, r, np.asarray(prop.c)))
    speed = rpm * math.pi / 30.0 * x * R
    return rho * speed * c / mu, speed / 343.0


def print_evaluation(ev):
    """Print the tables of evaluate()."""
    print("rpm   thrust_tab_N thrust_mod_N  err%   power_tab_W power_mod_W  err%   P_at_Ttab_W  err%   FoM_tab FoM_mod")
    for i, r in enumerate(ev["rpm"]):
        mark = " " if ev["gate_rows"][i] else "*"
        print(f"{r:5.0f}{mark}{ev['thrust_table'][i]:10.2f} {ev['thrust_model'][i]:11.2f} "
              f"{100 * (ev['thrust_model'][i] / ev['thrust_table'][i] - 1):+6.1f} {ev['power_table'][i]:11.1f} "
              f"{ev['power_model'][i]:11.1f} {100 * (ev['power_model'][i] / ev['power_table'][i] - 1):+6.1f} "
              f"{ev['power_at_table_thrust'][i]:11.1f} {100 * (ev['power_at_table_thrust'][i] / ev['power_table'][i] - 1):+6.1f} "
              f"{ev['fom_table'][i]:8.3f} {ev['fom_model'][i]:7.3f}")
    print("(* = outside the gate range)")
    for key, label in (("thrust", "thrust vs rpm"), ("power", "power vs rpm"),
                       ("equal_thrust", "power at equal thrust"), ("fom_diff", "FoM")):
        print(f"{label:24s} RMS {100 * ev[key]['rms']:5.1f} %  max {100 * ev[key]['max_abs']:5.1f} %")
    h = ev["hover"]
    print(f"hover {h['thrust_n']} N: table {h['power_table']:.1f} W at {h['rpm_table']:.0f} rpm, model {h['power_model']:.1f} W, "
          f"error {100 * h['error']:+.1f} %; FoM table {h['fom_table']:.3f}, model {h['fom_model']:.3f}")
    print(f"gate (5 % RMS, 10 % max on thrust and power vs rpm, 7 % power at hover thrust): {'PASS' if ev['gate'] else 'FAIL'}")


def report(prop, table, dat_path, rho=RHO):
    """Print the validation tables, Q/(T D) and the J trend. Return the evaluate() dict."""
    from . import export_performance as ep

    D = FLUXER_TARGET.diameter_m
    ev = evaluate(prop, table, rho)
    print_evaluation(ev)
    print("\nQ/(T D) and the Reynolds number at 0.75 R")
    for r in (HOVER_RPM, float(ev["rpm"][-1])):
        th, pw, _ = model_curve(prop, [r])
        ct, cp, fom_apc = apc_static_reference(dat_path, r)
        re75, mach = reynolds_at(prop, r)
        print(f"rpm {r:6.0f}: model T {th[0]:6.2f} N P {pw[0]:6.1f} W FoM {figure_of_merit(th[0], pw[0], D, rho):.3f} "
              f"Q/(TD) {torque_over_thrust_diameter(th[0], pw[0], r * math.pi / 30.0, D):.4f} Re75 {re75:,.0f} M75 {mach:.2f} "
              f"| APC 22x8 FoM {fom_apc:.3f} Q/(TD) {cp / (2 * math.pi) / ct:.4f}")
    print("\nCT and CP trend with J")
    case = ep.Case(prop)
    js = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    for r in (HOVER_RPM, float(ev["rpm"][-1])):
        n = r / 60.0
        rows = ep.performance_table(case, [r * math.pi / 30.0], [j * n * D for j in js], rho)
        print(f"rpm {r:.0f}: " + "  ".join(f"J {row['J']:.2f} CT {row['CT']:.4f} CP {row['CP']:.4f}" for row in rows))
    return ev


def main(argv=None):
    import argparse
    from . import init

    parser = argparse.ArgumentParser(description="Fit the MAD FLUXER PRO 22x6.6 to the maker bench table (Fit C).")
    parser.add_argument("--table", default="designs/MAD_FLUXER_22x6_6_maker.csv")
    parser.add_argument("--apc-dat", default="apc_prop_data/performance_data/PER3_22x8.dat")
    parser.add_argument("--airfoil", default="clarky", help="airfoil folder name in src/proptimiser/airfoils")
    parser.add_argument("--name", default=NAME, help="name of the saved design (with --save)")
    parser.add_argument("--save", action="store_true", help="write designs/ and propellers/ files (replaces them)")
    parser.add_argument("--no-fit", action="store_true", help="skip the fit: use the start values")
    args = parser.parse_args(argv)

    table = read_maker_table(args.table)
    inputs, blade = fluxer_setup(args.airfoil)
    if args.no_fit:
        theta = np.array(THETA_START)
        prop = build_propeller(theta, FLUXER_TARGET, inputs, blade)
    else:
        theta, prop, result = fit_propeller(FLUXER_TARGET, table, inputs, blade, verbose=0)
        print(f"fit cost {result.cost:.3f} after {result.nfev} model runs")
    params = ", ".join(f"{k} = {v:.4f}" for k, v in zip(PARAM_NAMES, theta))
    print("parameters:", params)
    r = np.asarray(prop.r)
    print("r_m      ", np.round(r[::3], 4))
    print("chord_m  ", np.round(np.asarray(prop.c)[::3], 4))
    print("twist_deg", np.round(np.degrees(np.asarray(prop.beta))[::3], 2))
    report(prop, table, args.apc_dat)
    if args.save:
        design_path = prop.save(args.name, overwrite=True)
        text = design_path.read_text(encoding="utf-8")
        design_path.write_text(FLUXER_NOTE.format(params=params, airfoil=args.airfoil) + text, encoding="utf-8", newline="\n")
        print("wrote", design_path)


if __name__ == "__main__":
    main()

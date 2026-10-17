# Proptimiser

Proptimiser is a propeller design tool. It uses blade element momentum theory (BEMT) in JAX. It does these tasks:

- It optimises the chord, the twist, the airfoil sections and the rotor speed of a blade for one or more flight conditions.
- It calculates the structural loads along the blade and at the hub.
- It exports the blade to Creo as `.ibl` curve files.
- It exports the propeller performance for Propalyser as one file: contract A, `prop-performance` v1 (`.h5`).

![Proptimiser outputs](media/proptimiseroutputs.png)

## Rules

- Proptimiser sends data to Propalyser only, through contract A. AeroPlan owns the contract layout.
  The hand copy of the layout is [docs/prop_performance_spec.md](docs/prop_performance_spec.md).
- The code holds no path to another repository. The code imports no module of another repository.
- Copy data between repositories by hand. Do not use symbolic links.
- There is no backward compatibility. There is no migration and no reader for old file formats.
  A file with an unknown `format_version` stops the load.
- Use SI units. Give the unit in the key name or in the units comment block.

## Installation

Use Python 3.12 and a virtual environment in `.venv`.

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[gui,dev]"      # Windows
.venv/bin/python -m pip install -e ".[gui,dev]"          # Linux or macOS
```

The extra `gui` adds PySide6 and pyvista for the graphical interface. The extra `dev` adds pytest.
Leave out the extras for the library only.
`requirements.txt` holds the exact versions of the tested environment.

## Use as a library

```python
from proptimiser import init, optimiser, results
prop = init.Propeller.load("currawong_v2")
```

The package holds only the airfoil data. Your files stay outside the package.
Designs, solved states and exports are in `designs/`, `propellers/` and `exports/`
inside the folder in the environment variable `PROPTIMISER_HOME`.
When the variable is not set, the folder is the current working directory.
The folder `apc_prop_data/` is also in that folder. It is not part of the package.

For GPU use, install the CUDA version of JAX. Refer to the JAX installation guide.

## Design files

A design has two files with the same name:

| File | Content | Who writes it |
|---|---|---|
| `designs/<name>.yaml` | The design configuration: flight conditions, solver settings, model selection, motor values, blade specification and material | You, or the GUI |
| `propellers/<name>.state.json` | The solved state: chord, twist, CST sections, rotor speed, tilt, induced velocity warm start, optimiser logs | The optimiser, or the GUI |

### Design YAML

The YAML file has three keys: `format_version` (1), `inputs` and `blade`.

- `inputs` holds the fields of the `Inputs` class in `init.py`. Each flight condition is one entry in the
  arrays `thrust_target`, `V_x`, `V_yz`, `omega`, `tilt_deg`, `solve_omega`, `solve_tilt` and `objective_weight`.
- `blade` holds the fields of the `BladeSpec` class in `init.py`.
- The comment block at the top of the file gives the units. Lengths are in m, speeds in m/s and `omega` in rad/s.
  Angles are in degrees: each angle key ends in `_deg`.
- Every key is required. An unknown key stops the load. Write a float with a dot, for example `1.0e-8`.

`r_foils = [0, r_hub, r_aero, r_mid..., r_tip]` (m) divides the blade into three zones:

1. The structural hub, `r < r_hub`. The BEMT does not include it.
2. The blend zone, `r_hub <= r < r_aero`. The chord and the twist blend from the hub values to the first
   aerodynamic station. The BEMT does not solve this zone, and it carries no force.
3. The aerodynamic zone. The CST airfoil sections are at `r_foils[2:]`, with linear interpolation between them.

`freeze` gives radial ranges (m) where the chord, the twist or the CST sections keep their seed values.
`offset_r`, `offset_y` and `offset_z` give the sweep and rake of the pitch axis. `mat_*` and `fill_factor` give
the blade material.

The motor values (`I_0`, `R_tot`, `kV`, `kappa`, `Q0`, `V_bus`, `P_max`) and the off-axis inflow (`V_yz`) stay in the design YAML.
The optimiser uses them for the motor power and the inflow. Contract A does not contain them.

### State JSON

The state file has `format_version` 1, a `units` map and one key for each array:

| Key | Unit | Shape |
|---|---|---|
| `c` | m | (n_r,) |
| `beta` | rad | (n_r,) |
| `cst` | nd | (n_aero, 18) |
| `omega` | rad/s | (n_cond,) |
| `tilt` | rad | (n_cond,) |
| `V_i_0` | m/s | (n_cond, 3, n_r) |
| `error_log` | nd | (iterations,) |
| `time_log` | s | (iterations,) |

### Load and save in Python

```python
from init import Propeller

prop = Propeller.load("currawong_v2")       # designs/currawong_v2.yaml and propellers/currawong_v2.state.json
prop.save("currawong_v3")                   # writes both files, refuses to replace a file without overwrite=True
```

A new design starts from `Propeller.default(inputs, blade)` or `Propeller.from_apc(path)`.
`prop.with_inputs(...)` and `prop.with_blade(...)` use a design with new conditions or new blade zones.

## Run the optimiser

### Graphical interface

```bash
proptimiser-gui                 # start from the default propeller
proptimiser-gui currawong_v2    # open a design
```

The window has three tabs and a log panel:

- **Setup**: edit `Inputs` and `BladeSpec`. Click **Apply** to rebuild the propeller.
- **Design**: evaluate all conditions, show the diagnostics, the structural loads and the planform editor.
  Show the blade in 3D and write the Creo `.ibl` files. Blend saved designs.
- **Optimise**: start and stop the optimiser. The plots update during the run.
  **Use as current design** sends the result to the Design tab.

The **File** menu opens and saves design YAML files (with the state JSON) and imports APC `.PE0` geometry.

### Command line

```bash
python -m proptimiser.init          # optimise the design in the __main__ block of init.py, with live plots
python -m proptimiser.results       # evaluate a saved design, print the structural summary, smooth it, save it
```

Push Ctrl+C one time to stop after the current iteration.

### Optimisation problem

The optimiser sees only dimensionless values:

| | Definition | Meaning |
|---|---|---|
| Objective | sum of `w_i P_i` divided by sum of `w_i P_ideal,i` | Motor power relative to the ideal actuator disc power. `objective_weight` gives the time fractions. |
| Thrust | `abs(T_i / T_i* - 1) <= thrust_tol` | Each condition stays in its thrust band. |
| Motor power | `P_i <= P_max` | The motor limit. |
| Thickness | `t/c >= min_tc` | At each optimised CST section. |
| Confidence | NeuralFoil confidence `>= min_confidence` | Load-weighted mean over the aerodynamic zone. `0` disables it. |

- `SLSQP` (default) applies the constraints exactly. Phase 1 restores feasibility when the seed violates a constraint.
  Phase 2 runs SLSQP with Jacobi scaling from the Hessian diagonal.
- `L-BFGS-B` uses bounds only. The constraints become quadratic penalties (`w_thrust`, `w_thickness`, `w_confidence`).
- The optimiser stops when the objective changes by less than `opt_tol` (relative), or after `opt_max_iter` iterations.

The BEMT solves the induced velocity with damped Newton on each annulus, to `V_i_0_eps`. `jax.lax.custom_root`
gives exact gradients. A bracketed fallback solves an annulus where Newton stops at a minimum that is not a root.

`opt_verbose` (0, 1, 2) and `V_i_0_verbose` (0, 1, 2) set the output of the optimiser and of the BEMT solve.

## Export for Propalyser (contract A)

```bash
proptimiser-export currawong_v2_structural --author "A. Person"
```

The exporter writes `exports/<prop_id>.h5`. The file follows contract A, `prop-performance` v1:

- Root attributes: `schema`, `schema_version`, `prop_id`, `design_sha256`, `created_utc`, `author`, `placeholder`.
- `/generator`: `name`, `commit`, `dirty`.
- `/geometry` and `/geometry/stations`: the diameter, the blade count, the hub radius, the blade frame, and
  `r_m`, `chord_m`, `twist_deg` at the BEMT stations.
- `/performance`: one row for each operating point (`J`, `rpm`, `V_mps`, `CT`, `CP`, `CQ`, `eta`, `tip_mach`,
  `reynolds_75`, `converged`). Each dataset has a `units` attribute.
- `/uncertainty/<item>`: the BEMT gives no uncertainty estimate today. Each item has `status = unavailable` and a basis.

The exporter refuses to write in these conditions. There is no override.

- The git tree is dirty (a changed, staged or untracked file).
- No author is known. Give `--author`, or set `git config user.name`.
- The design YAML or the state JSON is not committed.

To compare the BEMT with an APC table, run this command. It prints the comparison and writes no file.

```bash
proptimiser-export --apc apc_prop_data/propgeom_data/20x8E-PERF.PE0 \
    --compare-apc apc_prop_data/performance_data/PER3_20x8E.dat
```

## Geometry export

`geometry.py` builds the blade from the CST sections along the pitch axis. It writes three Creo `.ibl` files
into `exports/`:

1. The aerodynamic mold (the outer mold line).
2. The internal insert, offset by the laminate thickness.
3. The parting surface along the leading and trailing edges.

The exporters cut each section at constant radius. A swept section is longer by `1 / cos L`, where `L` is the
angle between the span and the radial line.

## Structural loads

`structure.py` calculates the loads along the blade and at the hub for the solved flight conditions. It uses the
stored induced velocity and solves nothing. All functions are safe for `jit`, `vmap` and `grad`.

```python
import results

prop, params_list, rows = results.evaluate_conditions(prop)                # solves the induced velocity
data = results.structural_data(params_list[0], prop.material())
print(results.structural_summary(data, 0))
```

| Part | Method | Result |
|---|---|---|
| Section | Strip integrals of the CST outline | Area, centroid, second moments, torsion constant |
| Inertia | Each blade slice is a rigid body that follows the hub and the pitch change | Centrifugal, Euler, Coriolis and gyroscopic loads, centrifugal twisting moment |
| Statics | Force summation from the tip to the root with 3D position vectors | Axial force, shear, moments and torque at each station and azimuth |
| Hub | Sum of the root loads of the `N` blades by Fourier series over the azimuth | Hub forces, moments and their `N`/rev amplitudes |
| Stress | Axial and bending stress at the outline points, torsion and shear, von Mises | Safety factor against `mat_sigma_allow` and `mat_tau_allow` |

Frames and signs:

- Blade frame `(e_r, e_t, e_z)`, right-handed: `e_r` radial, `e_t` the direction of blade motion, `e_z` the
  thrust direction. The shaft turns at `+omega e_z`.
- Positive twist `beta` lifts the leading edge toward `+e_z`.
- `offset_y` is positive toward the direction of rotation (APC `SWEEP`). `offset_z` is positive toward the thrust
  side (APC `RAKE`).

Limits:

- The blade is rigid. There is no elastic deflection, no fatigue model and no flutter model.
- The shear centre is the centroid. The model has no laminate: use `fill_factor` for an equivalent solid.
- The material allowables are placeholders. Set them from the real material.
- `tilt` does not change the aerodynamics, so it does not change the loads.

## Use a propeller in a CasADi problem

`casadi_prop.PropellerModel` puts a solved propeller in an AeroSandbox (or CasADi) `Opti` problem.
Use it to size a quadcopter, a quadplane or a fixed-wing UAV with a real BEMT propeller model.

```python
import aerosandbox as asb
from casadi_prop import PropellerModel
from init import Propeller

model = PropellerModel.from_propeller(Propeller.load("currawong_v2_solved"), n_psi=1)   # axial flight

opti = asb.Opti()
omega = opti.variable(init_guess=500.0, lower_bound=100.0)                 # rad/s
rotor = model.add_to(opti, omega_rad_s=omega, V_x_m_s=18.0, V_yz_m_s=0.0, thrust_scale_N=5.0)
model.warm_start(opti, rotor, 500.0, 18.0, 0.0, 5.0)                       # optional: start at the JAX solution
opti.subject_to(rotor.thrust_N == 4.0)
opti.minimize(rotor.shaft_power_W)
sol = opti.solve()
```

- The geometry is fixed: chord, twist and sections come from the `Propeller`.
- The flight state is free. Pass numbers or Opti expressions for `omega_rad_s`, `V_x_m_s`, `V_yz_m_s` and the air
  properties `rho_kg_m3`, `mu_Pa_s` and `a_m_s`.
- `add_to` adds the induced velocity ratios as variables and the BEMT residuals as equality constraints.
  IPOPT solves the BEMT together with the vehicle. Call `add_to` once for each rotor and each flight state.
- `thrust_scale_N` scales the unknowns and the residuals. It is not a constraint. Use the thrust you expect.
- `add_to` returns `thrust_N`, `torque_Nm` and `shaft_power_W` as expressions.
- Use `n_psi=1` when `V_yz` is zero. The model is about 9 times faster and gives the same loads.
  Use the default azimuth grid for edgewise flight (a lift rotor in forward flight).
- `warm_start` runs the JAX solver at a numeric flight state and starts the variables there.
  The solve from the default start takes 4 IPOPT iterations in the tests. The warm start takes 0.

### Scale a propeller in a CasADi problem

Pass `scale` to `add_to`, `expressions`, `solve`, `warm_start` and `static_and_axial` to change the size of the blade.
`scale` multiplies every length of the blade: radius, chord, station spacing and the blade section offsets. The twist,
the sections and the pitch ratio stay the same. The radius is `scale * R_tip` of the `Propeller`.

```python
model = PropellerModel.from_propeller(Propeller.load("MAD_FLUXER_22x6_6"), n_psi=1)
opti = asb.Opti()
s = opti.variable(init_guess=1.0, lower_bound=0.6, upper_bound=1.6)         # radius / 0.2794 m
rotor = model.add_to(opti, omega_rad_s=omega, V_x_m_s=0.0, V_yz_m_s=0.0, thrust_scale_N=22.0, scale=s)
```

- `scale` is a number or an Opti expression. The default is 1. With a number, the loads equal the loads of a
  `Propeller` built at the scaled size (`tests/test_casadi_scale.py` checks this to 1e-8).
- The Reynolds number, the tip speed and the Mach number change with the radius, so the model is not a similarity law.
  A scale far from 1 (below 0.6 or above 1.6) leaves the range where the fitted blade is valid. Check the result.
- The tip loss, the Pitt-Peters skew and the Du-Selig correction depend on r/R only. They do not change with `scale`.
- `warm_start` takes `scale` as a number.

Limits:

- The CasADi backend runs the parametric and NeuralFoil section models (`coeff_method` 0 and 3).
  The table models (1 and 2) run on JAX only.
- Chord, twist and sections are constants. The CasADi problem can scale the blade (see below) but cannot change its shape.
- The model gives thrust, torque and power. It does not give the in-plane hub force or the side force.
- The airfoil model is NeuralFoil, trained for Reynolds numbers of about 1e4 to 1e7.
  Check the confidence output (`airfoil.aero_coeffs(..., return_log_confidence=True)`) on JAX for a new design.

The BEMT physics is written against one array namespace (`backend.py`).
`ModelConfig.backend` selects `"jax"` or `"casadi"`. The Newton solver stays on JAX.
`tests/test_backend_parity.py` checks that both backends give the same residuals, loads and coefficients.

## Tests

```bash
.venv/Scripts/python -m pytest -q
```

The full suite takes about 13 minutes.

## References

1. S. Mancini, K. Boorsma, G. Schepers, and F. Savenije, "A comparison of dynamic inflow models for the blade element momentum method," Wind Energ. Sci., vol. 8, no. 2, pp. 193-210, 2023, doi: 10.5194/wes-8-193-2023.
2. W. Zhong, T. Wang, W. Z. Shen, and W. J. Zhu, "A function improving tip loss correction of blade-element momentum theory for wind turbines," Int. J. Sustainable Energy, vol. 43, no. 1, 2024, doi: 10.1080/14786451.2024.2352794.
3. W. Khan and M. Nahon, "A propeller model for general forward flight conditions," Int. J. Intelligent Unmanned Systems, vol. 3, no. 2/3, pp. 72-92, 2015, doi: 10.1108/IJIUS-06-2015-0007.
4. P. Yu, J. Peng, J. Bai, X. Han, and X. Song, "Aeroacoustic and aerodynamic optimization of propeller blades," Chinese J. Aeronautics, vol. 33, no. 3, pp. 826-839, 2020, doi: 10.1016/j.cja.2019.11.005.
5. P. Sharpe, "AeroSandbox: A Differentiable Framework for Aircraft Design Optimization," MIT, 2021. https://hdl.handle.net/1721.1/140023
6. J. V. R. Prasad, Y.-B. Kong, and D. Peters, "Analytical methods for modeling inflow dynamics of a coaxial rotor system."
7. Y. Leng, M. Bronz, T. Jardin, and J.-M. Moschetta, "Comparisons of Different Propeller Wake Models for a Propeller-Wing Combination," 2019, doi: 10.13009/EUCASS2019-147.
8. JAXopt documentation. https://jaxopt.github.io/stable/
9. P. Sharpe and R. J. Hansman, "NeuralFoil: An Airfoil Aerodynamics Analysis Tool Using Physics-Informed Machine Learning," 2025, arXiv:2503.16323.
10. Optax documentation. https://optax.readthedocs.io/
11. D. M. Pitt and D. A. Peters, "Theoretical prediction of dynamic-inflow derivatives," Vertica, vol. 5, no. 1, pp. 21-34, 1981.
12. B. M. Kulfan, "Universal Parametric Geometry Representation Method," J. Aircraft, vol. 45, no. 1, pp. 142-158, 2008, doi: 10.2514/1.29958.

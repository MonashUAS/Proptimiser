## Performance export

`export_performance.py` writes the BEMT performance of one propeller as a CSV table and a YAML file. Other tools read these two files.

The script sweeps the axial speed `V_x` at a fixed rotor speed `omega`. It solves the induced velocity at each point. It does not run the optimiser.

This section belongs in the main README. The folder `performance_exports/` holds the outputs.

### Run it

Run these commands from the repository root with the project environment.

```
python export_performance.py Currawong_V1
python export_performance.py --apc apc_prop_data/propgeom_data/20x8E-PERF.PE0 --compare-apc apc_prop_data/performance_data/PER3_20x8E.dat
```

The first command exports the propeller `propellers/Currawong_V1.json`. The second command exports the APC 20x8E geometry and compares it with the performance table of APC.

The script needs PyYAML. It is not in `requirements.txt`.

### Old code

The geometry tested on the thrust stand exists only in commit `588118a`. The saved `.eqx` files load only with the code of that commit. Use the script on an extract of that commit.

```
mkdir old_tree
git archive 588118a | tar -x -C old_tree
cp export_performance.py old_tree/
cd old_tree
python export_performance.py Currawong_V1 --commit 588118a
```

The same file runs in both trees. It uses `results.loadprop` in the old code and `Propeller.load` in the new code. Run the old tree from its own folder, because the old loader reads `propellers/` from the current folder.

### Files

Each run writes two files. The names are `<prop_id>__<commit>.perf.csv` and `<prop_id>__<commit>.perf.yaml`. The commit has 7 characters. It is the commit of the propeller geometry. The value `apc` marks an APC geometry file.

The CSV file has these columns in this order.

```
omega_rad_s, rpm, V_x_mps, J, CT, CP, CQ, thrust_N, torque_Nm, shaft_power_W, eta, tip_mach, converged
```

- All values use SI units. The rotor speed is `n = omega / (2 pi)` in rev/s and `D` is the diameter.
- `J = V_x / (n D)`, `CT = T / (rho n^2 D^4)` and `CP = P / (rho n^3 D^5)`. `P` is the shaft power.
- `CQ = CP / (2 pi)` and `eta = J CT / CP`. The value `eta` is 0 at `J = 0`. It is NaN where `CP` is zero or negative.
- `tip_mach = omega R_tip / 343`. The air density `rho` is 1.225 kg/m3 unless the YAML file says otherwise.
- `converged` is 1 or 0. A point that does not converge keeps its row. Its coefficients, thrust, torque and power are NaN.

The grid has these values.

- `omega` has the design values of the propeller and 300 to 750 rad/s in steps of 50.
- `V_x` starts at 0 m/s. If the solver does not converge at 0 for every `omega`, it starts at the smallest value that converges. The YAML key `V_x_min_converged_mps` gives that value. The other values are 2, 4 ... 30 m/s.
- The in-plane speed `V_yz` is 0 m/s. The key `model.V_yz_mps` gives it. The design points in the YAML file use the `V_yz` of their flight condition.

The YAML file describes the run. It has the keys `schema`, `schema_version`, `producer`, `kind`, `prop_id`, `source`, `D_m`, `blades`, `rho`, `model`, `design_points`, `motor_assumed`, `V_x_min_converged_mps`, `generated`, `exporter` and `csv_sha256`. The key `csv_sha256` is the hash of the CSV file.

### Convergence

The new code solves the induced velocity with a Newton method. A point converges when the largest scaled residual in the aerodynamic zone is below `V_i_0_eps`.

The old code solves it with a Broyden method. A point converges when the norm of the thrust residual is below `V_ia_0_eps`. The old solver fails at a few points near zero thrust. The script then tries again from the solution of the previous point.

### APC comparison

The option `--compare-apc` compares CT and CP with the performance table of APC. It writes `<prop_id>__apc.compare.csv` and `<prop_id>__apc.compare.yaml`.

The CSV file shows CT and CP at J = 0, 0.2 and 0.4 for 4000, 6000 and 8000 rpm. The YAML file gives the RMS relative difference over J = 0 to 0.5 and the number of points. The script uses each row of the APC table with J up to 0.5. It solves the same J with the BEMT.

The default `apc_model` grid has only 8 radial stations. The comparison changes when the grid has more stations. The option `--n-r 50` uses 50 stations. Give it a new `--prop-id`, so the files keep their own names. The YAML file gives `n_r`.

This is a report. It is not a pass or fail test. The APC geometry file gives only chord and twist. The airfoil, the number of stations and the other inputs are the defaults of `propmodel.apc_model`. For an APC file, the `omega` design values and the design points come from the default flight conditions of `Inputs`. They are not APC design points.

### Test

```
python -m pytest -q tests/test_export_performance.py
```

The test runs offline on a small grid.

# Cleanup to the MUAS data standard

Branch `standard-cleanup`, from `main` at 5389d1f. Date 2026-10-08.

## Test results

| | Result | Time |
|---|---|---|
| Baseline on `main`, as reported before the work | 92 passed, 18 warnings | 780 s |
| Baseline on `main`, run again for this work | 92 passed, 22 warnings | 764 s |
| Final, on `standard-cleanup` (commit 175ab8c) | 104 passed, 21 warnings | 708 s |

The test count changes: the work deletes 5 legacy tests and the 7 old exporter tests. It adds 6 YAML and state
loader tests, 12 contract A exporter tests and 5 compliance tests.

## Line counts

Tracked text files outside `airfoils/`, `apc_prop_data/` and `media/` (.py, .md, .txt, .json, .yaml):
11908 lines before, 11936 after (before CLEANUP.md). Python only: 10942 before, 11165 after (the new tests add
lines). `git diff --shortstat main`: 41 files, 1494 insertions, 2208 deletions. `init.py` 842 to 870,
`export_performance.py` 447 to 445, `README.md` 292 to 243.

## Deleted

| Path or code | Reason |
|---|---|
| `performance_exports/` (13 files, with its README) | Old CSV and YAML exports pinned to commit hashes. Contract A replaces them. |
| `init.Propeller.from_legacy_eqx`, `_LEGACY_PARAMS_*`, `_read_eqx_leaves` | `.eqx` reader. No backward compatibility. |
| Format 2 branch of `Propeller.from_dict` (`_RESET_FROM_V2`), the `verbose` key mapping | Format 2 loader and old-key migration. |
| `Propeller.from_dict`, `Propeller.to_dict`, `prop_path`, `_dataclass_from_dict` | The single-JSON propeller format. Unknown keys gave only a warning. |
| `propellers/*.json` (4 files) | Replaced by `designs/<name>.yaml` and `propellers/<name>.state.json`. |
| `export_performance.py`: `_OldCase`, `_old_point` (Broyden path), `NEW_CODE`, `results.loadprop` path, `--commit`, CSV and YAML writers, `--vyz`, the `.compare` file writers | Old-commit code and the old export format. Off-axis inflow is not in contract A. |
| Tests: `.eqx` migration, format 2 defaults, `verbose` key, unknown fields warn, `Currawong_V1` saved-state Newton test, the old-file half of the offsets and material test | Tied to the removed loaders. |
| README sections: Python-class config, `Save Propeller? (y/n)` JSON flow, "new fields never break old files", `.eqx` conversion, old clone URL, `venv` path | Stale. |

`exports/` and `logs/` were empty and not tracked. Git does not track an empty folder, so nothing was removed.
`/exports/` is now in `.gitignore` (Creo `.ibl` files and contract A `.h5` files go there).

## New files

| File | Content |
|---|---|
| `designs/<name>.yaml` | Hand design configuration: `format_version` 1, `inputs`, `blade`. SI, angles in degrees (`*_deg`), units comment block. Every key required, unknown key rejected. |
| `propellers/<name>.state.json` | Solved state: `format_version` 1, `units` map, `c`, `beta`, `cst`, `omega`, `tilt`, `V_i_0`, `error_log`, `time_log`. |
| `docs/prop_performance_spec.md` | Hand copy of AeroPlan contract A v1 and the uncertainty fields, with the SOURCE line and local notes. |
| `tests/test_compliance.py` | No import of or path to aerolink, aeroplan, propalyser. No legacy loader names. Files at format_version 1. Exporter at schema_version 1. |
| `CLAUDE.md` | The repository rules. |
| `CLEANUP.md` | This file. |

Changed: `init.py` (YAML and state I/O), `export_performance.py` (contract A writer), `app.py` and `gui/` (open and
save YAML designs), `results.py` (a comment only: `propgeomblender` uses `Propeller.load(name)`, which works
unchanged), `requirements.txt` (pinned, PyYAML 6.0.3 and h5py 3.16.0 added), `README.md` (rewritten),
`tests/test_export_performance.py` (rewritten for contract A).

## Units of the state arrays

The code makes these units certain:

| Array | Unit | Evidence in the code |
|---|---|---|
| `beta` | rad | `Params.beta` is "blade twist angle (rad)". `Propeller.default` uses `jnp.radians(beta_deg)`. `from_apc` converts degrees with `np.radians`. `results` plots `jnp.degrees(params.beta)`. |
| `tilt` | rad | The optimiser bound is `np.radians(30.0)`. The GUI column is "Tilt [rad]". |
| `c` | m | `Params.c` "chord length (m)". |
| `omega` | rad/s | `Params.omega` "rotational speed (rad/s)". |
| `V_i_0` | m/s | `Params.V_i_0` "(ms^{-1})". |
| `cst` | nd | Kulfan CST weights. |
| `error_log` | nd | "relative change of the merit function per iteration". |
| `time_log` | s | `time.time()` differences. |

The design YAML gives `tilt` in degrees (`tilt_deg`). The loader converts it to radians.

## Motor values and V_yz

`I_0`, `R_m`, `kV` and `P_max` stay in the design YAML. The optimiser needs them: `proptimiser.objectives` and the
motor power constraint call `propmodel.motorPower` with them. `V_yz` also stays: the BEMT uses the off-axis inflow
(`airflow`, `propmodel`). The exporter never writes them. The contract A sweep sets `V_yz = 0` (axial flow only).

## Conversion of the four designs

A one-off script (not committed) read each old JSON file with the old loader and wrote the new files. A check
then loaded each new pair with the new loader. Every state array is bit-identical to the old file. Every
`Inputs` and `BladeSpec` value equals the value that the old loader gave.

The old loader added values to the older files. The new files keep these values, so the designs behave as before:

- `Currawong_V1` (a format 2 file): the old loader replaced `V_i_0_eps` 0.05, `V_i_0_max_iter` 1000 and `opt_tol`
  1e-4 with the format 3 defaults 1e-8, 50 and 1e-6. The file had no `opt_verbose`, `opt_print_every`,
  `V_i_0_verbose`, `w_thrust`, `thrust_tol`, `w_smooth`, offsets or material, so the loader used the defaults.
  The new YAML holds the values that the old loader used.
- `currawong_v2`: no `w_smooth`, offsets or material in the old file. The new YAML holds the defaults.
- `currawong_v2_solved`, `currawong_v2_structural`: complete files. No added value.

## Diameter conflict (open)

The user says that the flying custom propeller has D = 0.55 m. The design files say otherwise:

| Design | Tip radius `r_foils[-1]` | D |
|---|---|---|
| `Currawong_V1` | 0.254 m | 0.508 m |
| `currawong_v2`, `currawong_v2_solved`, `currawong_v2_structural` | 0.285 m | 0.570 m |

No value was changed. Decide which design is the flying propeller, or add the flown design as a new design file.

## Contract A check export

From the clean tree at commit 175ab8c:
`python export_performance.py currawong_v2_structural` (author from `git config user.name`).

- Output `exports/currawong_v2_structural.h5` (35 kB, not tracked: `/exports/` is ignored).
- `design_sha256` d8f94583aaa0120fbd164b4679b9094a1bd8439aec007dd815ed894a3fcee370.
- `diameter_m` 0.57, `blades` 2, `hub_radius_m` 0.025, `twist_sign` 1, no `rotation`.
- 176 operating points (11 rotor speeds from 2865 to 7162 rpm, 16 speeds each), all converged. J from 0 to 1.10.
- CT from -0.019 to 0.044 (negative at high J), `reynolds_75` from 4.7e4 to 1.2e5.
- `/uncertainty/CT`, `CP`, `CQ`, `eta`: `status = unavailable`.

## Open questions

1. Diameter: which design is the flying D = 0.55 m propeller? See above.
2. `rotation`: no design defines the direction of rotation, so `/geometry` has no `rotation` attribute. Add a
   design key (`rotation: cw_from_behind` or `ccw_from_behind`) when the direction is known.
3. `author`: the default comes from `git config user.name` (`lmca0007` here). Contract A wants a person. Give
   `--author "<name>"` when the git name is not a person's name.
4. Uncertainty: the BEMT gives no estimate, so every item is `unavailable`. A validation against stand data
   (for example the APC comparison in `export_performance.py`) could give a `quantitative` record later.
5. `placeholder` is `false`: the values come from the BEMT of a real design. The structural values are not
   validated (commit 8d23f17 says so). They are not in contract A.
6. `hub_radius_m` of `Currawong_V1` is 0 (`r_foils[1] = 0`). The exporter writes the design value.

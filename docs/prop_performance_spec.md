# Contract A: `prop-performance` v1

SOURCE: AeroPlan design page "Exchange contracts" (`exchange.md`), section A, contract `prop-performance`
schema_version 1. Copied by hand on 2026-10-08. AeroPlan owns this layout. Do not change it here. When AeroPlan
changes the page, copy it again by hand.

The uncertainty fields come from the AeroPlan design page "Uncertainty schema" (`uncertainty.md`).

## Rules for all contracts (copy)

- One `.h5` file (or, for C, a folder with an optional `.h5` wrapper). No other cross-repo file.
- SI units. Every dataset carries an attribute `units` (`nd` for a dimensionless value). An angle column says `deg` or `rad`.
- Every uncertainty item has the fields of the AeroPlan uncertainty schema: `status`, `scope`,
  `basis`, and, when quantitative, `method`, `distribution`, `effect`, and `num_sigmas` or `coverage`.
- `author` is a person, never a program name.
- A writer refuses to write from a dirty git tree. There is no override that hides it.
- A reader refuses a file with an unknown `schema` or `schema_version`. There is no migration and no legacy reader.
- No value is invented. A field that the writer does not know is absent, and the matching reader reports it.
- `generator` is a group `/generator` with attributes `name`, `commit`, `dirty`. Each uncertainty item is a group
  `/uncertainty/<item>` whose attributes are the uncertainty fields. Each scalar result is a group `/results/<name>`.
- Strings are UTF-8 variable-length. Lists of strings (`/warnings`) are 1-D string datasets.

```
Proptimiser --(A: prop-performance .h5)--> Propalyser --(B: propulsion-system .h5)--> AeroPlan --(C: package)--> Aerolink
```

## A. `prop-performance` v1 (Proptimiser to Propalyser) (copy)

Root attributes: `schema = "prop-performance"`, `schema_version = 1`, `prop_id`, `design_sha256` (hash of the design
YAML and the solved state JSON, canonical form), `created_utc`, `author`, `generator` (`name`, `commit`, `dirty = false`),
`placeholder` (bool).

| Path | Content | Units |
|---|---|---|
| `/geometry` attrs | `diameter_m`, `blades`, `hub_radius_m`, `frame` (blade frame text), `twist_sign`, `rotation` (`cw_from_behind`, `ccw_from_behind` or absent) | m |
| `/geometry/stations` | table: `r_m`, `chord_m`, `twist_deg` | as named |
| `/performance` | table, one row for each operating point: `J`, `rpm`, `V_mps`, `CT`, `CP`, `CQ`, `eta`, `tip_mach`, `reynolds_75`, `converged` | `units` attr on each column |
| `/performance` attrs | `rho_kgpm3`, `nu_m2ps`, `J_range`, `rpm_values` | |
| `/uncertainty/<item>` | one group for each item (for example `CT`, `CP`): the uncertainty fields. Optional columns `CT_sigma`, `CP_sigma` in `/performance` | |

Definitions: `J = V/(n D)`, `CT = T/(rho n^2 D^4)`, `CP = P/(rho n^3 D^5)`, `CQ = Q/(rho n^2 D^5)`, `n` in rev/s.
Not in the file: motor values, off-axis inflow (`V_yz`), installation (hub position, thrust axis). AeroPlan holds the installation.

## Uncertainty fields (copy of the field table)

| Field | Meaning | Rule |
|---|---|---|
| `status` | `quantitative`, `qualitative` or `unavailable` | [M&S 33] a, b, c |
| `scope` | `input`, `result` or `derived` | [M&S 29] a, b, c |
| `basis` | The process and the reason | Required, not empty |
| `method` | `repeat_scatter`, `binning`, `grid_study`, `tool_spread`, `sysid_cov`, `measurement_chain`, `vendor_spec`, `monte_carlo`, `engineering` | Required when quantitative |
| `distribution` | `normal`, `uniform` or `interval` | Required when quantitative |
| `effect` | additive, multiplicative, percentage, absolute | Required when quantitative |
| `sigma`, `num_sigmas` | Dataset value (normal) | Needs `sigma` or `per_point_column` |
| `lower`, `upper` | Bounds (uniform, interval) | `lower <= upper` |
| `coverage` | Probability inside the interval | Required for `interval` |
| `per_point_column` | Name of the sigma column | |
| `factors` | Margins and factors of safety, with what they cover | [M&S 29] b |
| `correlations` | Correlation with other variables | ADH form |
| `qualitative_level` | `small`, `medium`, `large` | Required when qualitative |

An `unavailable` record carries no number. It is a valid, honest state.

## How Proptimiser fills the file (local notes, not part of the contract)

The writer is `export_performance.py` (`write_prop_performance`).

- Tables are HDF5 groups with one 1-D dataset for each column. Each dataset has the attributes `units` and
  `description`.
- `design_sha256`: the sha256 of the JSON text of `{"design": <design YAML content>, "state": <state JSON content>}`,
  with sorted keys, the separators `,` and `:`, and ASCII escapes. Comments, layout and line endings do not change it.
- `generator`: `name = "Proptimiser"`, `commit` = the full git hash of HEAD, `dirty = false`.
- `author`: `--author`, else `git config user.name`. The writer refuses when neither gives a name.
- `placeholder = false`: the values come from the BEMT model of the design.
- `/geometry`: `diameter_m = 2 R_tip`, `hub_radius_m = r_foils[1]`, `twist_sign = 1`. `rotation` is absent: the
  design does not define the direction of rotation.
- `/geometry/stations`: the BEMT stations from the hub radius to the tip. `twist_deg` is the solver twist `beta`
  (stored in radians in the state file) converted to degrees.
- `/performance`: rows in the order rpm, then `V_mps`. The sweep has no off-axis inflow. A point that did not
  converge has `converged = false`, and NaN in `CT`, `CP`, `CQ`, `eta` and `reynolds_75`.
- `tip_mach = omega R_tip / a` with `a = 343 m/s` (rotational tip speed only).
- `reynolds_75`: the chord Reynolds number at `r = 0.75 R_tip`, from the solved relative velocity (azimuth mean)
  and the air viscosity `mu = 1.798e-5 kg/(m s)`. `nu_m2ps = mu / rho`.
- `/uncertainty/CT`, `CP`, `CQ`, `eta`: `status = unavailable`, `scope = result`, and a `basis` that says why.
  The BEMT model gives no uncertainty estimate today. The writer gives no sigma and no sigma column.

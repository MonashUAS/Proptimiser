# MAD FLUXER PRO 22x6.6 MATT: fitted BEMT model

This model is FITTED. It is not measured. It does not use the real blade geometry. The maker does not publish it.

## Purpose

The model gives static (J = 0) and low-J coefficients for the quadcopter digital twin.
The motor is the MAD M6C10 IPE V3 KV150. The hover point is 22.4 N per rotor. The maker table gives 3298 rpm for it.

## Data and an error in the first target

`designs/MAD_FLUXER_22x6_6_maker.csv` holds the maker bench table (AMPX 40A ESC, 12S, 48 V, 25 C, sea level).
The source is the MAD M6C10 IPE V3 datasheet image, supplied by the user on 2026-10-17. The maker calls the values
theoretical and for reference. The column `output_W` is shaft power. Torque times omega agrees with it within 0.25 % in every row.

ERROR in the first target (Fit A and Fit B): the old chart file (`MAD_FLUXER_22x6_6_chart.csv`, read by hand from the chart)
used the chart "P" curve as INPUT power and took shaft power as 0.83 times that. The chart "P" curve is OUTPUT (shaft)
power. The old shaft power was 17 % too low. The old file is deleted. Fit A (`_gatefit`) and Fit B (`_fitB`) used it.
Do not use them for new work. Their tables below are kept only as a record, with the maker table as reference.

## Method (Fit C, the default)

1. `fit_prop.blade_family` makes the chord and the twist from 5 parameters (`c75_over_R`, `a_inboard`, `b_tip`,
   `dtheta_deg`, `pitch_gradient`). Twist: `tan(beta) = P / (2 pi r) + dtheta`, P = 6.6 in.
2. `fit_prop.fit_propeller` runs a bounded least squares fit. The residuals are the log ratio of model to table for thrust
   against rpm and for shaft power against rpm, at all 15 rows. Each is divided by 0.03 (a weight scale, not a measured error)
   and doubled from 2800 to 3800 rpm. Four prior terms keep the shape physical: `dtheta_deg` 0 +/- 2, `a_inboard` 0.4 +/- 0.3,
   `b_tip` 0.35 +/- 0.2, `pitch_gradient` 0 +/- 0.1. The chord scale has no prior.
3. The BEMT is the Proptimiser solver with the Currawong V1 model choices (Pitt-Peters inflow, NeuralFoil,
   Prandtl tip and root loss, rotation correction), 25 radial stations.

## Assumptions

- The sections are Clark-Y. This is an assumption. No source confirms it for the FLUXER.
- The hub radius is 0.025 m and the aerodynamic zone starts at 0.05 m. These are not maker values.
- The design YAML motor keys (`I_0`, `R_tot`, `kappa`, `Q0`, `P_max`) are class defaults, not MAD values. Only `kV` 150 is the maker value.
  The export does not use them.
- Air is sea level, rho 1.225 kg/m^3.

## Results against the maker table

Fit C (`MAD_FLUXER_22x6_6`): `c75_over_R` 0.1163, `dtheta_deg` 1.321, `a_inboard` 0.697, `b_tip` 0.081, `pitch_gradient` 0.0098.
The gate rows are 2492 rpm and above. The 2492 rpm row stands for the 2500 rpm limit.

| | Fit B (`_fitB`) | Fit C (default) |
|---|---|---|
| Thrust against rpm: RMS, max | 4.7 %, 6.0 % (high) | 1.1 %, 1.9 % |
| Shaft power against rpm: RMS, max | 1.9 %, 3.7 % (low) | 0.8 %, 1.4 % |
| Power at equal thrust: RMS, max | 7.7 %, 8.8 % (low) | 1.1 %, 1.8 % |
| FoM model against table: RMS, max | 8.5 %, 9.9 % | 1.2 %, 1.9 % |
| Power at 22.4 N | 175.6 W (-8.5 %) | 188.9 W (-1.5 %) |
| Maker power at 22.4 N | 191.8 W at 3298 rpm | 191.8 W at 3298 rpm |

Fit C against the table:

| rpm | Thrust table N | Thrust model N | Error % | Power table W | Power model W | Error % | Power at table thrust W | Error % | FoM table | FoM model |
|---|---|---|---|---|---|---|---|---|---|---|
| 1899 | 7.46 | 7.07 | -5.3 | 39.0 | 36.7 | -5.9 | 39.5 | +1.4 | 0.674 | 0.661 |
| 2234 | 10.15 | 10.03 | -1.2 | 61.4 | 59.4 | -3.2 | 60.4 | -1.6 | 0.679 | 0.689 |
| 2492 | 12.56 | 12.63 | +0.5 | 83.3 | 82.4 | -1.1 | 81.8 | -1.8 | 0.690 | 0.702 |
| 2746 | 15.41 | 15.47 | +0.4 | 110.7 | 110.3 | -0.3 | 109.6 | -1.0 | 0.705 | 0.712 |
| 3019 | 18.67 | 18.82 | +0.8 | 146.6 | 146.8 | +0.1 | 145.1 | -1.0 | 0.710 | 0.718 |
| 3329 | 22.83 | 23.07 | +1.1 | 197.3 | 197.2 | -0.1 | 194.2 | -1.6 | 0.713 | 0.725 |
| 3623 | 26.98 | 27.44 | +1.7 | 252.6 | 254.8 | +0.9 | 248.4 | -1.7 | 0.716 | 0.728 |
| 3886 | 31.18 | 31.68 | +1.6 | 312.4 | 315.1 | +0.9 | 307.7 | -1.5 | 0.719 | 0.730 |
| 4132 | 36.04 | 36.01 | -0.1 | 381.4 | 379.7 | -0.4 | 380.2 | -0.3 | 0.732 | 0.734 |
| 4368 | 39.62 | 40.39 | +1.9 | 443.9 | 449.8 | +1.3 | 437.2 | -1.5 | 0.725 | 0.736 |
| 4576 | 43.92 | 44.44 | +1.2 | 513.6 | 518.4 | +0.9 | 509.5 | -0.8 | 0.731 | 0.737 |
| 4818 | 49.59 | 49.40 | -0.4 | 608.9 | 606.9 | -0.3 | 610.4 | +0.2 | 0.740 | 0.738 |
| 5042 | 54.95 | 54.25 | -1.3 | 706.4 | 697.7 | -1.2 | 711.1 | +0.7 | 0.744 | 0.739 |
| 5276 | 59.62 | 59.57 | -0.1 | 802.8 | 802.0 | -0.1 | 803.0 | 0.0 | 0.740 | 0.740 |
| 5543 | 66.76 | 66.14 | -0.9 | 946.6 | 933.8 | -1.4 | 946.7 | 0.0 | 0.743 | 0.743 |

The table FoM is 0.713 at 3329 rpm (0.71 at 3300 rpm) and 0.743 at 5543 rpm. This agrees with the hand check.

Fit B against the table, per row from 1899 to 5543 rpm:

- Thrust error: +0.4, +4.4, +5.9, +5.6, +5.7, +5.4, +6.0, +5.8, +3.7, +5.6, +4.7, +3.0, +2.0, +3.1, +1.9 %.
- Power error: -4.1, -1.9, -0.4, -0.3, -0.4, -1.1, -0.6, -0.8, -2.2, -0.6, -1.1, -2.4, -3.4, -2.4, -3.7 %.
- Power error at table thrust: -4.6, -7.6, -8.1, -7.7, -8.1, -8.5, -8.8, -8.7, -7.3, -8.3, -7.6, -6.6, -6.2, -6.8, -6.4 %.

FoM of Fit B is 0.71 to 0.79, so it is too high by 0.03 to 0.08.

## Does Clark-Y BEMT reach the table FoM?

Yes. The earlier statement that it stays at 0.77 or above was true only for the wrong target (the wrong target gave FoM 0.88).
Fit C gives FoM 0.66 to 0.74 across the table, and 0.725 at 3329 rpm and 0.743 at 5543 rpm.
Reynolds number at 0.75 R (chord from Fit C): 155,000 at 3200 rpm and 269,000 at 5543 rpm. Mach number at 0.75 R: 0.20 and 0.35.
Fit B had a higher FoM only because it matched a power that was 17 % too low in an equal-thrust view of the wrong data.

Second airfoil, tested once: ARA-D 5 % (`arad05`). Reason: it is a thin section of the ARA-D series. I took it as a plausible thin-propeller section
from memory and I did not check the source online. The fit ran with the same parameters, bounds and priors. Result: thrust 1.9 % RMS and 3.3 % max. Power 1.2 % RMS and 2.5 % max. Power at 22.4 N is -0.8 %. The FoM stays flat at 0.72 from 2746 to 5543 rpm, but the table rises from 0.705 to 0.743. The thrust error is -3.3 % at the top. The geometry loader warned that it found no coordinates for `arad05`. This is no better than Clark-Y. Clark-Y stays. The ARA-D 5 % result is not saved.

## Initial validation gate for the twin

Limits: thrust and shaft power against rpm within 5 % RMS and 10 % max over 2492 to 5543 rpm, and power at equal thrust within 7 % at 22.4 N.

| | Limit | Fit C | Fit B |
|---|---|---|---|
| Thrust against rpm: RMS, max | 5 %, 10 % | 1.1 %, 1.9 % | 4.7 %, 6.0 % |
| Power against rpm: RMS, max | 5 %, 10 % | 0.8 %, 1.4 % | 1.9 %, 3.7 % |
| Power at 22.4 N | 7 % | -1.5 % | -8.5 % |
| Result | | PASS | FAIL (hover power) |

PASS means the model agrees with the maker table. The table is theoretical. The gate is not a check against a thrust stand.
Fit C was fitted to the same 15 rows that it is checked against. The gate does not test the model on new data.
Fit C is weaker at 1899 rpm (thrust -5.3 %, power -5.9 %), outside the gate range.

## Shape check against APC 22x8

| rpm | Fit C FoM | APC 22x8 FoM | Fit C Q/(T D) | APC 22x8 Q/(T D) |
|---|---|---|---|---|
| 3200 | 0.723 | 0.697 | 0.0440 | 0.0461 |
| 5543 | 0.743 | 0.744 | 0.0435 | 0.0439 |

Fit C, 5543 rpm: CT 0.0649, 0.0605, 0.0556, 0.0502, 0.0442, 0.0373, 0.0293 and CP 0.0177, 0.0179, 0.0179, 0.0176, 0.0170,
0.0158, 0.0138 at J = 0, 0.05, ..., 0.30. CT falls to 45 % at J = 0.3.

## Limits

- The model is FITTED. It is not measured. No uncertainty is available: the export gives `unavailable`.
- Only J = 0 data exist. All J > 0 values are BEMT extrapolation. No data check them.
- The maker values are theoretical. Check the model with a thrust stand measurement before you rely on it.
- The fit does not use blade mass (49 g with hardware) or any structure data.
- The model has no motor, no ESC and no airframe interference.

## Files

- `MAD_FLUXER_22x6_6` (design YAML and state JSON): Fit C, the default.
- `MAD_FLUXER_22x6_6_fitB`: Fit B, kept for reference. Fitted to the wrong target.
- `MAD_FLUXER_22x6_6_gatefit`: Fit A, fitted to the wrong target. Not changed.

## How to repeat the fit

```bash
.venv/Scripts/python -m proptimiser.fit_prop --save
```

The command takes about 10 minutes. It replaces the Fit C design YAML and state JSON and prints the tables.
Use `--airfoil <folder name>` to test another airfoil. Without `--save` it writes nothing.

## CasADi call for the twin

```python
from proptimiser import init
from proptimiser.casadi_prop import static_and_axial

prop = init.Propeller.load("MAD_FLUXER_22x6_6")        # reads the two files below
loads = static_and_axial(prop, rpm=3300.0, v_axial=0.0, rho=1.225)
loads.thrust_N, loads.torque_Nm, loads.shaft_power_W, loads.CT, loads.CP, loads.J
```

Files needed: `designs/MAD_FLUXER_22x6_6.yaml` and `propellers/MAD_FLUXER_22x6_6.state.json`. The call runs on the CasADi path
(damped Newton on the BEMT residuals) and does not need JAX. It uses one azimuth point (exact in axial flow). The model is built on the
first call for each Propeller object and kept. Keep one Propeller object for each rotor design. CT and CP use n in rev/s and the propeller diameter.
At 3300 rpm: 22.66 N, 0.556 N m, 192.0 W, CT 0.0627, CP 0.01729 (static). Time: 2.0 s for the first call (builds the model), 0.77 s for
the next call (one Windows machine, one run each).

## How to export (contract A)

The exporter refuses a dirty git tree. Commit the design YAML, the state JSON and the maker table first. Then run
this command from the repository root:

```bash
.venv/Scripts/proptimiser-export MAD_FLUXER_22x6_6 --author "YOUR NAME" --rpm-list 1500,2000,2500,3000,3200,3500,4000,4500,5000,5550 --vx-list 0,0.5,1,1.5,2,3,4,6,8,10,14,18,22,26,30
```

The file is `exports/MAD_FLUXER_22x6_6.h5`. Without `--vx-list` the sweep is V_x = 0 and 2 to 30 m/s in steps of 2 m/s.
Without `--rpm-list` the sweep is 300 to 750 rad/s.

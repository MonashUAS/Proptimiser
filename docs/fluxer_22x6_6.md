# MAD FLUXER PRO 22x6.6 MATT: fitted BEMT model

This model is FITTED. It is not measured. It does not use the real blade geometry. The maker does not publish it.

## Purpose

The model gives static (J = 0) and low-J coefficients for the quadcopter digital twin.
The motor is the MAD M6C10 IPE V3 KV150. The hover point is about 3200 rpm and 22.4 N for each rotor.

## Method

1. `designs/MAD_FLUXER_22x6_6_chart.csv` holds the maker chart (12S, 25 C, sea level, read by hand).
   The reading error is 5 % on thrust and power and 0.02 on efficiency.
   Shaft power is the efficiency times the input power.
2. `fit_prop.blade_family` makes the chord and the twist from 5 parameters:
   - `c75_over_R`: the chord at 0.75 R divided by R.
   - `a_inboard` and `b_tip`: the chord shape, `c ~ x^a (1 - x + 0.02)^b`, `x = r / R`.
   - `dtheta_deg`: an angle offset added to the twist.
   - `pitch_gradient`: the change of geometric pitch along the span. Zero means constant pitch.
   The twist follows `tan(beta) = P / (2 pi r) + dtheta`. `P` is 6.6 in.
3. `fit_prop.fit_propeller` runs a bounded least squares fit (Fit B). The residuals are the log ratio of model to
   chart for thrust against rpm and for the model shaft power at the chart thrust, at the 9 rpm points, each
   divided by its reading error (5 % and 5.6 %) and doubled from 2800 to 3800 rpm. Fit A used thrust and power against rpm. Four prior terms keep the shape physical: `dtheta_deg` 0 +/- 2, `a_inboard` 0.4 +/- 0.3,
   `b_tip` 0.35 +/- 0.2, `pitch_gradient` 0 +/- 0.1. The chord scale has no prior.
4. The BEMT is the Proptimiser solver with the Currawong V1 model choices (Pitt-Peters inflow, NeuralFoil,
   Prandtl tip and root loss, rotation correction), 25 radial stations.

## Assumptions

- The sections are Clark-Y. This is an assumption. It is the airfoil of the Currawong V1 design, and it
  is a common choice for small propellers. No source confirms it for the FLUXER. A thin carbon blade may
  have a thinner section. The effect on the fit was not tested.
- The hub radius is 0.025 m and the aerodynamic zone starts at 0.05 m. These are not maker values.
- The chart is read by hand. The chart implies a figure of merit of 0.88 to 0.89. A real two-blade propeller
  of this size seldom reaches that value. The chart is probably optimistic.
- The design YAML motor keys (`I_0`, `R_tot`, `kappa`, `Q0`, `P_max`) are class defaults. They are not MAD values.
  Only `kV` 150 is the maker value. The export does not use them.
- Air is sea level, rho 1.225 kg/m^3.

## Two fits

| | Fit A (`MAD_FLUXER_22x6_6_gatefit`) | Fit B (`MAD_FLUXER_22x6_6`, default) |
|---|---|---|
| Objective | thrust and power against rpm, equal weight | thrust against rpm, and power at equal thrust; weight 2 for 2800 to 3800 rpm |
| `c75_over_R`, `dtheta_deg` | 0.1004, 2.004 | 0.1028, 2.482 |
| `a_inboard`, `b_tip`, `pitch_gradient` | 0.127, 0.338, 0.007 | 0.0004, 0.314, 0.015 |
| Thrust against rpm: RMS, max | 6.1 %, 7.2 % (about 6 % low) | 1.2 %, 2.6 % |
| Power against rpm: RMS, max | 6.6 %, 11.9 % | 15.9 %, 21.8 % |
| Power at equal thrust: RMS, max | 16.5 %, 22.9 % | 15.2 %, 19.8 % |
| Rpm gate (7 % RMS, 12 % max) | PASS (small margin) | FAIL on power |
| Thrust at 3200 rpm | 20.8 N | 22.2 N (chart need: 22.4 N) |
| FoM at 3200 / 5550 rpm | 0.769 / 0.788 | 0.779 / 0.794 |

Result: no geometry in the family closes the equal-thrust power gap. The gap is a figure of merit limit.
The chart FoM is 0.88 to 0.89. The BEMT with Clark-Y gives 0.77 to 0.79 for both fits. Power at equal thrust stays
about 12 to 20 % high (about 13 % at the hover thrust). The fit can only choose where the error goes:
Fit A puts it in thrust (6 % low) and Fit B puts it in power at equal rpm.
Fit B gives the right thrust against rpm and the right hover rpm. It does not pass the rpm gate.
Use Fit B only with a known power gain of about 0.87 at equal thrust, or wait for a thrust stand measurement.
The chart FoM is probably optimistic, so the gain is not proven.

Fit B against the chart (equal-rpm view and equal-thrust view):

| rpm | Thrust chart N | Thrust model N | Error % | Power chart W | Power model W | Error % | Power at chart thrust W | Error % |
|---|---|---|---|---|---|---|---|---|
| 1950 | 7.85 | 7.94 | +1.2 | 33.2 | 40.4 | +21.8 | 39.8 | +19.8 |
| 2400 | 12.26 | 12.32 | +0.5 | 66.4 | 74.3 | +11.9 | 73.8 | +11.2 |
| 2850 | 17.46 | 17.56 | +0.6 | 107.9 | 123.2 | +14.2 | 122.2 | +13.3 |
| 3300 | 23.34 | 23.65 | +1.3 | 166.0 | 190.1 | +14.5 | 186.5 | +12.3 |
| 3750 | 29.91 | 30.68 | +2.6 | 240.7 | 278.4 | +15.6 | 268.2 | +11.4 |
| 4200 | 38.74 | 38.62 | -0.3 | 332.0 | 391.7 | +18.0 | 393.4 | +18.5 |
| 4650 | 48.05 | 47.51 | -1.1 | 464.8 | 533.2 | +14.7 | 542.4 | +16.7 |
| 5100 | 57.86 | 57.36 | -0.9 | 605.9 | 706.4 | +16.6 | 715.6 | +18.1 |
| 5550 | 67.67 | 68.21 | +0.8 | 806.4 | 915.0 | +13.5 | 904.1 | +12.1 |

Fit A against the chart: thrust 7.33, 11.45, 16.38, 22.16, 28.75, 36.22, 44.58, 53.85, 64.05 N (errors -6.6, -6.6, -6.1,
-5.1, -3.9, -6.5, -7.2, -6.9, -5.3 %). Power 37.2, 68.3, 113.1, 174.3, 255.2, 359.1, 488.9, 647.7, 839.0 W
(errors +11.9, +2.8, +4.8, +5.0, +6.0, +8.2, +5.2, +6.9, +4.0 %).
Power at chart thrust: +22.9, +13.2, +14.8, +13.3, +12.4, +19.5, +17.6, +19.0, +12.9 %.

The data fix mostly two combinations: the solidity and the pitch level. The shape parameters stay near their priors.
Do not read the chord and twist as the real blade.

## Shape check against APC 22x8

| rpm | Fit B FoM | APC 22x8 FoM | Fit B Q/(T D) | APC 22x8 Q/(T D) |
|---|---|---|---|---|
| 3200 | 0.779 | 0.697 | 0.0417 | 0.0461 |
| 5550 | 0.794 | 0.744 | 0.0413 | 0.0439 |

The APC values come from `PER3_22x8.dat` (CT and CP at J = 0, linear in rpm). The APC pitch ratio is 0.36 and the
FLUXER pitch ratio is 0.30. The model has a higher FoM and a lower Q/(T D) than the APC, as a lower pitch gives.
Fit B, 5550 rpm: CT 0.0667, 0.0625, 0.0577, 0.0525, 0.0468, 0.0409, 0.0348 and CP 0.0173, 0.0176, 0.0177, 0.0176,
0.0172, 0.0165, 0.0155 at J = 0, 0.05, ..., 0.30. CT falls to 52 % at J = 0.3 (APC 22x8: 55 %).

## Limits

- The model is FITTED. It is not measured. No uncertainty is available: the export gives `unavailable`.
- Only J = 0 data exist. All J > 0 values are BEMT extrapolation. No data check them.
- Power at equal thrust is 12 to 20 % high in both fits. Do not tune it out with a gain unless a thrust stand
  measurement supports it.
- The fit does not use blade mass (49 g with hardware) or any structure data.
- The model has no motor, no ESC and no airframe interference.

## How to repeat the fit

```bash
.venv/Scripts/python -m proptimiser.fit_prop --save
```

The command takes about 10 minutes. It replaces the Fit B design YAML and state JSON and prints the tables.
Commit both files after that.

## How to export (contract A)

The exporter refuses a dirty git tree. Commit the design YAML, the state JSON and the chart first. Then run
this command from the repository root:

```bash
.venv/Scripts/proptimiser-export MAD_FLUXER_22x6_6 --author "YOUR NAME" --rpm-list 1500,2000,2500,3000,3200,3500,4000,4500,5000,5550 --vx-list 0,0.5,1,1.5,2,3,4,6,8,10,14,18,22,26,30
```

The file is `exports/MAD_FLUXER_22x6_6.h5`. Without `--vx-list` the sweep is V_x = 0 and 2 to 30 m/s in steps of 2 m/s.
That gives only J = 0, 0.067 and 0.134 below J 0.15 at 3200 rpm. Without `--rpm-list` the sweep is 300 to 750 rad/s.

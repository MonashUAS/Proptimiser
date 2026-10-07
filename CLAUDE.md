# Proptimiser rules

Proptimiser is a propeller design tool (JAX BEMT, optimiser, structural loads, Creo `.ibl` export).

1. Proptimiser sends data to Propalyser only, through contract A: `prop-performance` v1 (`.h5`).
   AeroPlan owns the layout. The hand copy is `docs/prop_performance_spec.md`. Do not change the layout here.
2. Do not write a path to another repository. Do not import aeroplan, propalyser or aerolink.
3. Copy data between repositories by hand. Do not use symbolic links.
4. There is no backward compatibility. Do not add a migration, a legacy reader or an "old format" branch.
5. A design is `designs/<name>.yaml` (hand config, SI, angles in degrees, every key required).
   The solved state is `propellers/<name>.state.json` (`format_version` 1, `beta` and `tilt` in rad).
6. Give the unit of each value in the key name or in a units block. Use SI units.
7. The exporter refuses a dirty git tree and a missing author. Do not add an override.
   Do not invent a value or a sigma. Write `unavailable` with a basis when no estimate exists.
8. Use the repository venv: `.venv/Scripts/python`. The full test suite takes about 13 minutes.
   `tests/test_compliance.py` checks rules 2 and 4.
9. Write docs, comments and commit messages in ASD-STE100 style: short active sentences.

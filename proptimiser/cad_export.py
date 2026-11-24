"""
Write the CAD files of a solved design for Creo.

    proptimiser-cad APC_20x10E                  (blade STEP and face table in exports/APC_20x10E/)
    proptimiser-cad APC_20x10E --ibl --out x    (also the three mould .ibl files, in folder x)

The blade STEP is one closed blade solid in the .ibl frame, in mm. The face table (<name>_blade_faces.json) gives the
area and the box of each named face of each blade, and the hub size. Creo places the blades and the hub.
The STEP export needs the cad extra (cadquery-ocp).
"""
import argparse
from pathlib import Path

from . import geometry, init


def export(design, out_dir=None, ibl=False):
    """Write <design>_blade.step and <design>_blade_faces.json, and with ibl the mould .ibl files. Return the paths."""
    prop = init.Propeller.load(design)
    out_dir = Path(out_dir) if out_dir else init.EXPORT_DIR / design
    out_dir.mkdir(parents=True, exist_ok=True)
    solid = geometry.export_propeller_step(prop, out_dir, design, rotor=False)
    paths = [solid["blade"], solid["faces"]]
    if ibl:
        paths += list(geometry.export_standard_ibl(prop, f"{design}_mold", out_dir=out_dir)[0])
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(prog="proptimiser-cad", description="Write the CAD files of a solved design for Creo.")
    parser.add_argument("design", help="design name (designs/<name>.yaml and propellers/<name>.state.json)")
    parser.add_argument("--out", help="output folder (default: exports/<design>)")
    parser.add_argument("--ibl", action="store_true", help="also write the three mould .ibl files")
    args = parser.parse_args(argv)
    for path in export(args.design, args.out, args.ibl):
        print(path)


if __name__ == "__main__":
    main()

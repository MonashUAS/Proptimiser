"""
Folders used by Proptimiser.

AIRFOIL_DIR holds the airfoil data. It is part of the installed package.
HOME holds the user files: designs, solved states and exports. It is the folder in the environment
variable PROPTIMISER_HOME. When the variable is not set, HOME is the current working directory.
"""
import os
from pathlib import Path

AIRFOIL_DIR = Path(__file__).resolve().parent / "airfoils"

HOME = Path(os.environ.get("PROPTIMISER_HOME") or Path.cwd()).resolve()
DESIGN_DIR = HOME / "designs"           # hand design files: <name>.yaml
PROP_DIR = HOME / "propellers"          # solved state files: <name>.state.json
EXPORT_DIR = HOME / "exports"           # generated .ibl, .csv and .h5 files
APC_DIR = HOME / "apc_prop_data"        # APC data. It is not part of the package.

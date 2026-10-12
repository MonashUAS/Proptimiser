"""
CasADi interface of the propeller model.

The BEMT physics (airflow.py, airfoil.py, propmodel.py) runs on CasADi expressions when Params.cfg.backend is
"casadi" (backend.py). The geometry of the propeller is then a set of NumPy constants, and the flight state
(omega, V_x, V_yz, ...) can be CasADi symbols.
"""
import dataclasses

import jax
import numpy as np


def casadi_params(params, **symbols):
    """
    Params with NumPy constants and the CasADi backend, from the Params of one flight condition.
    Each keyword replaces a field with a CasADi symbol or a number, e.g. omega=opti.variable().
    """
    cfg = dataclasses.replace(params.cfg, backend="casadi")
    constants = jax.tree_util.tree_map(np.asarray, dataclasses.replace(params, cfg=cfg))
    return dataclasses.replace(constants, **symbols)

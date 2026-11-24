"""
Optional JAX.

The CasADi path of the BEMT physics (backend.py, casadi_prop.py) runs without JAX. The modules import their JAX names
from here. With JAX installed, these are the real jax, jax.numpy and equinox. Without JAX, jnp is NumPy, jit does
nothing, and eqx.Module is a plain dataclass. Code that needs JAX itself (the Newton solver, the optimiser, the
table interpolators) calls require_jax() first.
"""
import dataclasses
from types import SimpleNamespace

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    import equinox as eqx
    HAS_JAX = True
    jax.config.update("jax_enable_x64", True)
except ImportError:
    HAS_JAX = False
    jax = None
    jnp = np

    class _Module:
        """A dataclass on subclassing, like equinox.Module."""
        def __init_subclass__(cls, **kwargs):
            super().__init_subclass__(**kwargs)
            dataclasses.dataclass(cls)

    eqx = SimpleNamespace(Module=_Module, field=lambda static=False, **kwargs: dataclasses.field(**kwargs))

ARRAY_TYPES = (jax.Array, np.ndarray) if HAS_JAX else (np.ndarray,)


def jit(fn):
    """jax.jit, or the plain function without JAX."""
    return jax.jit(fn) if HAS_JAX else fn


def require_jax(what):
    if not HAS_JAX:
        raise ImportError(f"{what} needs JAX. Install the 'jax' extra: pip install proptimiser[jax]")

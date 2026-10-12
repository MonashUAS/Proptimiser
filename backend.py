"""
Array namespace for the BEMT physics.

The physics functions take an `xp` namespace instead of importing jax.numpy. JAX_OPS is the JAX namespace.
Each name in it is also in the CasADi namespace (added in phase 2), so one function body runs on both.
Control flow that depends on traced values stays out of the physics: use where, not if.
"""
from types import SimpleNamespace

import jax
import jax.numpy as jnp


def _logsumexp2(a, b):
    """Stable log(exp(a) + exp(b)) of two arrays."""
    m = jnp.maximum(a, b)
    return m + jnp.log(jnp.exp(a - m) + jnp.exp(b - m))


JAX_OPS = SimpleNamespace(
    name="jax",
    sin=jnp.sin, cos=jnp.cos, tan=jnp.tan, arctan2=jnp.arctan2, arccos=jnp.arccos,
    sqrt=jnp.sqrt, exp=jnp.exp, log=jnp.log, tanh=jnp.tanh, abs=jnp.abs,
    maximum=jnp.maximum, minimum=jnp.minimum, clip=jnp.clip, where=jnp.where,
    sum=jnp.sum, mean=jnp.mean,
    concatenate=jnp.concatenate, stack=jnp.stack,
    swish=jax.nn.swish,
    log_sigmoid=jax.nn.log_sigmoid,
    logaddexp=_logsumexp2,
)


def ops(cfg):
    """The namespace for ModelConfig cfg."""
    return JAX_OPS

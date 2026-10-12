"""
Array namespace for the BEMT physics.

The physics functions take an `xp` namespace instead of importing jax.numpy. JAX_OPS is the JAX namespace.
CASADI_OPS has the same names, so one function body runs on both. Control flow that depends on traced values stays out
of the physics: use where, not if.

Shape rules that make the two namespaces agree:
    - A field on the flat point layout is a 1D array on JAX and a column (n, 1) on CasADi.
    - A stack of fields (stack, axis 0) is a matrix with one field in each row on both. Take a row with xp.row.
    - CasADi does not broadcast. Expand a per-station value with a constant matrix (xp.matmul), add a row vector to
      a matrix with xp.row_add.
    - CASADI_OPS keeps an expression of constants as NumPy, so constant geometry costs nothing in the CasADi graph.
"""
from types import SimpleNamespace

import numpy as np
import jax
import jax.numpy as jnp


def _logaddexp(a, b):
    """Stable log(exp(a) + exp(b)) of two arrays."""
    m = jnp.maximum(a, b)
    return m + jnp.log(jnp.exp(a - m) + jnp.exp(b - m))


JAX_OPS = SimpleNamespace(
    name="jax",
    sin=jnp.sin, cos=jnp.cos, tan=jnp.tan, arctan2=jnp.arctan2, arccos=jnp.arccos,
    sqrt=jnp.sqrt, exp=jnp.exp, log=jnp.log, tanh=jnp.tanh, abs=jnp.abs,
    maximum=jnp.maximum, minimum=jnp.minimum, clip=jnp.clip, where=jnp.where,
    sum=jnp.sum, mean=jnp.mean, matmul=jnp.matmul,
    concatenate=jnp.concatenate, stack=jnp.stack, column_stack=jnp.column_stack,
    ones_like=jnp.ones_like, zeros_like=jnp.zeros_like,
    vstack_scalars=lambda items: jnp.stack(items)[:, None],   # list of scalars -> column (k, 1)
    row=lambda matrix, i: matrix[i],                          # row i of a stack of fields
    row_add=lambda x, row: x + row,                           # (p, n) + (n,): CasADi does not broadcast
    const=lambda x: x,                                        # a constant array, in the namespace's array type
    swish=jax.nn.swish,
    log_sigmoid=jax.nn.log_sigmoid,
    logaddexp=_logaddexp,
)

_CASADI_OPS = None


def _build_casadi_ops():
    import casadi as cas
    import aerosandbox.numpy as anp

    symbolic = (cas.MX, cas.SX)
    is_sym = lambda *xs: any(isinstance(x, symbolic) for x in xs)
    as_column = lambda x: x if isinstance(x, symbolic) else np.asarray(x, dtype=float).reshape(-1, 1)

    def matmul(a, b):
        if is_sym(a, b):
            return cas.mtimes(a, b)
        return np.matmul(np.asarray(a), np.asarray(b))

    def concatenate(items, axis=1):
        assert axis == 1, "CASADI_OPS joins 2D blocks along axis 1"
        return cas.horzcat(*items) if is_sym(*items) else np.concatenate(items, axis=1)

    def column_stack(items):
        return cas.horzcat(*[as_column(x) for x in items]) if is_sym(*items) else np.column_stack(items)

    def stack(items, axis=0):
        assert axis == 0, "CASADI_OPS stacks fields into rows (axis 0)"
        if not is_sym(*items):
            return np.stack([np.asarray(x) for x in items], axis=0)
        return cas.vertcat(*[x.T if isinstance(x, symbolic) else np.asarray(x, dtype=float).reshape(1, -1) for x in items])

    def vstack_scalars(items):
        return cas.vertcat(*items) if is_sym(*items) else np.array(items, dtype=float)[:, None]

    def row(matrix, i):
        return matrix[i, :].T if is_sym(matrix) else matrix[i]

    def row_add(x, r):
        if is_sym(x):
            return x + cas.repmat(np.asarray(r, dtype=float).reshape(1, -1), x.shape[0], 1)
        return x + r

    def swish(x):
        return x / (1.0 + anp.exp(-x))

    def log_sigmoid(x):
        # -softplus(-x), stable for either sign of x
        return -(anp.maximum(-x, 0.0) + (cas.log1p(anp.exp(-anp.abs(x))) if is_sym(x) else np.log1p(np.exp(-np.abs(x)))))

    def logaddexp(a, b):
        m = anp.maximum(a, b)
        return m + anp.log(anp.exp(a - m) + anp.exp(b - m))

    return SimpleNamespace(
        name="casadi",
        sin=anp.sin, cos=anp.cos, tan=anp.tan, arctan2=anp.arctan2, arccos=anp.arccos,
        sqrt=anp.sqrt, exp=anp.exp, log=anp.log, tanh=anp.tanh, abs=anp.abs,
        maximum=anp.maximum, minimum=anp.minimum, clip=anp.clip, where=anp.where,
        sum=anp.sum, mean=anp.mean, matmul=matmul,
        concatenate=concatenate, stack=stack, column_stack=column_stack,
        ones_like=lambda x: np.ones(x.shape), zeros_like=lambda x: np.zeros(x.shape),
        vstack_scalars=vstack_scalars, row=row, row_add=row_add,
        const=lambda x: np.asarray(x, dtype=float),
        swish=swish, log_sigmoid=log_sigmoid, logaddexp=logaddexp,
    )


def ops(cfg):
    """The namespace for ModelConfig cfg (cfg.backend: "jax" or "casadi")."""
    global _CASADI_OPS
    if cfg.backend == "jax":
        return JAX_OPS
    if cfg.backend == "casadi":
        if _CASADI_OPS is None:
            _CASADI_OPS = _build_casadi_ops()
        return _CASADI_OPS
    raise ValueError(f"Unknown backend {cfg.backend!r}, expected 'jax' or 'casadi'")

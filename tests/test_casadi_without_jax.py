"""The CasADi path of the propeller model runs when JAX is not installed."""
import subprocess
import sys
import textwrap

import pytest

SCRIPT = textwrap.dedent("""
    import sys
    for name in ("jax", "jaxlib", "equinox", "interpax", "jaxopt", "optax"):
        sys.modules[name] = None          # an import of these raises ImportError
    import aerosandbox as asb
    from proptimiser.init import Inputs, Propeller
    from proptimiser.casadi_prop import PropellerModel
    prop = Propeller.default(inputs=Inputs(coeff_method=3, n_r=8, n_psi=1, V_x=[18.0, 0.8], V_yz=[0.0, 0.0]))
    model = PropellerModel.from_propeller(prop)
    T, Q, lam = model.solve(588.0, 18.0, 0.0, 30.0)
    opti = asb.Opti()
    out = model.add_to(opti, 588.0, 18.0, 0.0, 30.0)
    model.warm_start(opti, out, 588.0, 18.0, 0.0, 30.0)
    sol = opti.solve(verbose=False)
    print(T, Q, float(sol(out.thrust_N)), float(sol(out.torque_Nm)))
""")


def test_opti_point_without_jax():
    done = subprocess.run([sys.executable, "-c", SCRIPT], capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, done.stderr[-2000:]
    T, Q, T_opti, Q_opti = (float(v) for v in done.stdout.split()[-4:])
    assert T > 0 and Q > 0
    assert T_opti == pytest.approx(T, rel=1e-6) and Q_opti == pytest.approx(Q, rel=1e-6)


def test_newton_solve_matches_the_jax_solver():
    pytest.importorskip("jax")
    from proptimiser.casadi_prop import PropellerModel
    from proptimiser.init import Inputs, Propeller
    prop = Propeller.default(inputs=Inputs(coeff_method=3, n_r=8, n_psi=12, V_x=[18.0, 0.8], V_yz=[3.0, 3.0]))
    model = PropellerModel.from_propeller(prop)
    for omega, V_x in ((588.0, 18.0), (650.0, 0.8)):
        T, Q, _ = model.solve(omega, V_x, 3.0, 30.0)
        T_jax, Q_jax, _ = model.solve_jax(omega, V_x, 3.0, 30.0)
        assert T == pytest.approx(T_jax, rel=1e-6) and Q == pytest.approx(Q_jax, rel=1e-6)

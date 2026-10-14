"""Motor and ESC chain: synthetic numbers only."""
import numpy as np
import pytest

from proptimiser import init, propmodel

KV, I0, R, KAPPA, Q0, V_BUS = 200.0, 1.5, 0.1, 1.1, 0.02, 40.0
KV_RAD = KV * 2 * np.pi / 60


def _chain(torque, omega, kappa=KAPPA, q0=Q0, r=R):
    i = I0 + (torque + q0) * KV_RAD
    return (omega / (kappa * KV_RAD) + r * i) * i


def test_chain_equations():
    got = float(propmodel.motorPower(0.8, 300.0, I_0=I0, R_tot=R, kV=KV, kappa=KAPPA, Q0=Q0))
    assert got == pytest.approx(_chain(0.8, 300.0), rel=1e-12)


def test_kappa_one_gives_the_first_order_motor():
    torque, omega = 0.8, 300.0
    kt = 60 / (2 * np.pi * KV)
    old = torque * omega + R * (torque / kt + I0) ** 2 + omega * 60 / (2 * np.pi) / KV * I0
    got = float(propmodel.motorPower(torque, omega, I_0=I0, R_tot=R, kV=KV))
    assert got == pytest.approx(old, rel=1e-12)


def test_duty_is_terminal_voltage_over_bus_voltage():
    v = float(propmodel.motor_voltage(0.8, 300.0, I0, R, KV, KAPPA, Q0))
    assert v == pytest.approx(300.0 / (KAPPA * KV_RAD) + R * (I0 + (0.8 + Q0) * KV_RAD), rel=1e-12)
    assert v / V_BUS < 1.0


def test_design_format_1_migrates():
    data = {"format_version": 1, "inputs": {"I_0": 2.0, "R_m": 0.037, "kV": 220.0, "P_max": 1.0}, "blade": {}}
    out = init._migrate_design_1(data)
    assert out["format_version"] == init.DESIGN_FORMAT
    assert list(out["inputs"]) == ["I_0", "R_tot", "kV", "kappa", "Q0", "V_bus", "P_max"]
    assert out["inputs"]["R_tot"] == 0.037 and out["inputs"]["kappa"] == 1.0 and out["inputs"]["V_bus"] == 0.0

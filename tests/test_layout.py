import numpy as np

from proptimiser import init  # noqa: F401  sets float64 before geometry builds arrays
from proptimiser import geometry


def test_point_layout_mean_and_expand():
    r, _ = geometry.radial_stops(0.275, 0.025, 5)
    psi, _ = geometry.angular_stops(6)
    E_r, M_psi, r_pt, psi_pt = geometry.point_layout(r, psi)
    field = np.random.default_rng(0).normal(size=(5, 6))
    np.testing.assert_allclose(M_psi @ field.ravel(), field.mean(axis=1), atol=1e-14)
    np.testing.assert_allclose((E_r @ np.asarray(r)).reshape(5, 6), np.asarray(r)[:, None] * np.ones((1, 6)))
    np.testing.assert_allclose(psi_pt.reshape(5, 6)[2], np.asarray(psi))
    np.testing.assert_allclose(r_pt, E_r @ np.asarray(r))


def test_cst_station_weights_match_interp():
    r, _ = geometry.radial_stops(0.275, 0.025, 8)
    r_cst = np.array([0.025, 0.05, 0.15, 0.275])
    cst = np.random.default_rng(1).normal(size=(4, geometry.N_CST))
    W = geometry.cst_station_weights(r, r_cst)
    np.testing.assert_allclose(W @ cst, np.asarray(geometry.cst_at_stations(r, r_cst, cst)), atol=1e-12)

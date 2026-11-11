import numpy as np
import pytest

from imu_fusion import (G, AttitudeEKF, ImuSpec, accel_tilt, complementary,
                        gyro_integration, simulate)
from imu_fusion.simulate import gravity_body

SPEC = ImuSpec()
DT = 1.0 / SPEC.fs


def rmse_deg(est, truth):
    err = (est - truth + np.pi) % (2 * np.pi) - np.pi
    return np.rad2deg(np.sqrt((err ** 2).mean()))


def make_ekf(**kw):
    return AttitudeEKF(DT, gyro_noise=np.deg2rad(SPEC.gyro_noise_dps),
                       bias_walk=np.deg2rad(SPEC.gyro_bias_walk_dps_rt_s), **kw)


@pytest.mark.parametrize("roll, pitch", [(0, 0), (30, -20), (-75, 10), (120, 45)])
def test_accel_tilt_inverts_gravity_model(roll, pitch):
    phi, theta = np.deg2rad([roll]), np.deg2rad([pitch])
    est = accel_tilt(gravity_body(phi, theta))[0]
    assert np.allclose(est, [phi[0], theta[0]], atol=1e-12)


def test_static_level_accel_reads_plus_g_on_z():
    assert np.allclose(gravity_body(np.zeros(1), np.zeros(1)), [[0, 0, G]])


def test_noise_free_gyro_integration_tracks_truth():
    spec = ImuSpec(gyro_noise_dps=0.0, gyro_bias_dps=(0, 0, 0), gyro_bias_walk_dps_rt_s=0.0)
    data = simulate(duration_s=30, spec=spec, rng=1)
    est = gyro_integration(data.gyro, DT, data.euler_true[0, :2])
    assert rmse_deg(est, data.euler_true[:, :2]) < 0.2


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_ranking_of_estimators(seed):
    data = simulate(spec=SPEC, rng=seed)
    truth = data.euler_true[:, :2]
    accel_only = rmse_deg(accel_tilt(data.accel), truth)
    comp = rmse_deg(complementary(data.gyro, data.accel, DT), truth)
    ekf_plain = rmse_deg(make_ekf(nis_gate=None).run(data.gyro, data.accel)[:, :2], truth)
    ekf = rmse_deg(make_ekf().run(data.gyro, data.accel)[:, :2], truth)
    assert ekf < ekf_plain < comp < accel_only
    assert ekf < 1.0


def test_ekf_estimates_gyro_bias():
    data = simulate(spec=SPEC, rng=4)
    states = make_ekf().run(data.gyro, data.accel)
    tail = slice(-int(10 * SPEC.fs), None)  # last 10 s
    err = np.rad2deg(np.abs(states[tail, 2:] - data.gyro_bias[tail, :2])).mean(axis=0)
    assert np.all(err < 0.25)  # deg/s, from a 1.5 / -1.0 deg/s initial bias


def test_ekf_covariance_stays_symmetric_positive_definite():
    data = simulate(duration_s=20, spec=SPEC, rng=5)
    ekf = make_ekf()
    ekf.run(data.gyro, data.accel)
    assert np.allclose(ekf.P, ekf.P.T, atol=1e-12)
    assert np.all(np.linalg.eigvalsh(0.5 * (ekf.P + ekf.P.T)) > 0)

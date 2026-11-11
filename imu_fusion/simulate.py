"""Synthetic 6-DoF IMU data with a known attitude ground truth.

Conventions: Euler angles roll (phi), pitch (theta), yaw (psi), ZYX order.
The accelerometer measures specific force in the body frame and reads +g on
the z axis when level:  f_b = g * [-sin(theta), sin(phi)cos(theta), cos(phi)cos(theta)] + a_lin_b
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

G = 9.80665


@dataclass(frozen=True)
class ImuSpec:
    fs: float = 200.0                                # sample rate, Hz
    gyro_noise_dps: float = 0.3                      # white noise std per sample, deg/s
    gyro_bias_dps: tuple = (1.5, -1.0, 0.5)          # initial bias, deg/s
    gyro_bias_walk_dps_rt_s: float = 0.02            # bias random walk, deg/s/sqrt(s)
    accel_noise_mps2: float = 0.08                   # white noise std per sample, m/s^2


@dataclass
class ImuData:
    t: np.ndarray            # (N,) s
    euler_true: np.ndarray   # (N, 3) rad: roll, pitch, yaw
    gyro: np.ndarray         # (N, 3) rad/s, measured
    accel: np.ndarray        # (N, 3) m/s^2, measured
    gyro_bias: np.ndarray    # (N, 3) rad/s, true bias
    disturbed: np.ndarray    # (N,) bool, True while linear acceleration is applied


def euler_rates_to_body(euler: np.ndarray, euler_dot: np.ndarray) -> np.ndarray:
    """Body angular rates [p, q, r] from Euler angles and their derivatives."""
    phi, theta = euler[:, 0], euler[:, 1]
    phi_d, theta_d, psi_d = euler_dot.T
    p = phi_d - psi_d * np.sin(theta)
    q = theta_d * np.cos(phi) + psi_d * np.cos(theta) * np.sin(phi)
    r = -theta_d * np.sin(phi) + psi_d * np.cos(theta) * np.cos(phi)
    return np.column_stack([p, q, r])


def gravity_body(phi: np.ndarray, theta: np.ndarray) -> np.ndarray:
    return G * np.column_stack([-np.sin(theta),
                                np.sin(phi) * np.cos(theta),
                                np.cos(phi) * np.cos(theta)])


def _trajectory(t: np.ndarray) -> np.ndarray:
    """Smooth manoeuvring attitude profile, rad."""
    d = np.deg2rad
    roll = d(25) * np.sin(2 * np.pi * 0.08 * t) + d(10) * np.sin(2 * np.pi * 0.31 * t)
    pitch = d(15) * np.sin(2 * np.pi * 0.05 * t + 0.7) + d(6) * np.sin(2 * np.pi * 0.23 * t)
    yaw = d(60) * np.sin(2 * np.pi * 0.03 * t)
    return np.column_stack([roll, pitch, yaw])


def simulate(duration_s: float = 90.0, spec: ImuSpec = ImuSpec(),
             rng: np.random.Generator | int | None = 0) -> ImuData:
    rng = np.random.default_rng(rng)
    n = int(duration_s * spec.fs)
    dt = 1.0 / spec.fs
    t = np.arange(n) * dt

    euler = _trajectory(t)
    euler_dot = np.gradient(euler, dt, axis=0)
    omega = euler_rates_to_body(euler, euler_dot)

    bias0 = np.deg2rad(np.asarray(spec.gyro_bias_dps, dtype=float))
    walk = np.deg2rad(spec.gyro_bias_walk_dps_rt_s) * np.sqrt(dt)
    bias = bias0 + np.cumsum(walk * rng.standard_normal((n, 3)), axis=0)
    gyro = omega + bias + np.deg2rad(spec.gyro_noise_dps) * rng.standard_normal((n, 3))

    # Linear-acceleration episodes (manoeuvres / vibration) that corrupt the
    # gravity reference. This is what breaks accelerometer-only attitude.
    a_lin = np.zeros((n, 3))
    disturbed = np.zeros(n, dtype=bool)
    for start, stop, amp in ((25.0, 33.0, 3.0), (55.0, 62.0, 4.0)):
        idx = (t >= start) & (t < stop)
        tt = t[idx] - start
        a_lin[idx] = amp * np.column_stack([np.sin(2 * np.pi * 0.6 * tt),
                                            np.cos(2 * np.pi * 0.45 * tt),
                                            0.5 * np.sin(2 * np.pi * 0.9 * tt)])
        disturbed |= idx

    accel = (gravity_body(euler[:, 0], euler[:, 1]) + a_lin
             + spec.accel_noise_mps2 * rng.standard_normal((n, 3)))
    return ImuData(t=t, euler_true=euler, gyro=gyro, accel=accel,
                   gyro_bias=bias, disturbed=disturbed)

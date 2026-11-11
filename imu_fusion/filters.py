"""Roll/pitch estimators: gyro integration, accelerometer tilt, complementary
filter and an extended Kalman filter that also estimates gyro bias."""
from __future__ import annotations

import numpy as np

from .simulate import G


def accel_tilt(accel: np.ndarray) -> np.ndarray:
    """Roll and pitch from the gravity direction, (N, 2) rad."""
    ax, ay, az = accel.T
    roll = np.arctan2(ay, az)
    pitch = np.arctan2(-ax, np.hypot(ay, az))
    return np.column_stack([roll, pitch])


def _euler_dot(phi: float, theta: float, w: np.ndarray) -> np.ndarray:
    """Roll and pitch rates from body rates (ZYX Euler kinematics)."""
    p, q, r = w
    sp, cp = np.sin(phi), np.cos(phi)
    return np.array([p + (q * sp + r * cp) * np.tan(theta),
                     q * cp - r * sp])


def _wrap(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi


def gyro_integration(gyro: np.ndarray, dt: float, initial: np.ndarray) -> np.ndarray:
    est = np.empty((len(gyro), 2))
    x = np.asarray(initial, dtype=float).copy()
    for k, w in enumerate(gyro):
        x = x + _euler_dot(x[0], x[1], w) * dt
        est[k] = x
    return est


def complementary(gyro: np.ndarray, accel: np.ndarray, dt: float,
                  tau_s: float = 1.0) -> np.ndarray:
    """First-order complementary filter: gyro at high frequency, accelerometer
    at low frequency, crossover time constant tau_s."""
    alpha = tau_s / (tau_s + dt)
    tilt = accel_tilt(accel)
    est = np.empty((len(gyro), 2))
    x = tilt[0].copy()
    for k in range(len(gyro)):
        pred = x + _euler_dot(x[0], x[1], gyro[k]) * dt
        x = alpha * pred + (1 - alpha) * tilt[k]
        est[k] = x
    return est


class AttitudeEKF:
    """Extended Kalman filter.

    State x = [roll, pitch, b_p, b_q]  (rad, rad, rad/s, rad/s)
    Process: gyro-driven Euler kinematics with bias-corrected rates, random-walk biases.
    Measurement: accelerometer tilt (roll, pitch).

    Adaptive measurement noise: R grows with the deviation of |accel| from g,
    so the filter trusts gravity less while the body is accelerating.
    """

    def __init__(self, dt: float, gyro_noise: float, bias_walk: float,
                 tilt_noise: float = np.deg2rad(1.5), accel_gate_mps2: float = 0.3,
                 nis_gate: float | None = 9.21):
        self.dt = dt
        self.nis_gate = np.inf if nis_gate is None else nis_gate  # chi2, 2 dof, 99 %
        self.x = np.zeros(4)
        self.P = np.diag([np.deg2rad(5) ** 2] * 2 + [np.deg2rad(3) ** 2] * 2)
        self.Q = np.diag([(gyro_noise * dt) ** 2] * 2 + [bias_walk ** 2 * dt] * 2)
        self.R0 = tilt_noise ** 2
        self.gate = accel_gate_mps2
        self.H = np.hstack([np.eye(2), np.zeros((2, 2))])

    def _f(self, x: np.ndarray, w: np.ndarray) -> np.ndarray:
        corrected = np.array([w[0] - x[2], w[1] - x[3], w[2]])
        return np.concatenate([x[:2] + _euler_dot(x[0], x[1], corrected) * self.dt, x[2:]])

    def _jacobian(self, x: np.ndarray, w: np.ndarray) -> np.ndarray:
        phi, theta = x[0], x[1]
        p, q, r = w[0] - x[2], w[1] - x[3], w[2]
        sp, cp, tt = np.sin(phi), np.cos(phi), np.tan(theta)
        sec2 = 1.0 / np.cos(theta) ** 2
        dt = self.dt
        F = np.eye(4)
        F[0, 0] += (q * cp - r * sp) * tt * dt
        F[0, 1] += (q * sp + r * cp) * sec2 * dt
        F[0, 2] = -dt
        F[0, 3] = -sp * tt * dt
        F[1, 0] += (-q * sp - r * cp) * dt
        F[1, 3] = -cp * dt
        return F

    def step(self, w: np.ndarray, f: np.ndarray) -> np.ndarray:
        F = self._jacobian(self.x, w)
        self.x = self._f(self.x, w)
        self.P = F @ self.P @ F.T + self.Q

        z = accel_tilt(f[None, :])[0]
        deviation = abs(np.linalg.norm(f) - G)
        R = np.eye(2) * self.R0 * (1.0 + (deviation / self.gate) ** 2)
        y = np.array([_wrap(z[0] - self.x[0]), z[1] - self.x[1]])
        S = self.H @ self.P @ self.H.T + R
        # Innovation gating: when the normalised innovation squared exceeds the
        # chi-square bound, inflate the covariance so NIS sits at the bound.
        # Catches accelerations that leave |accel| close to g.
        nis = float(y @ np.linalg.solve(S, y))
        if nis > self.nis_gate:
            S = S * (nis / self.nis_gate)
            R = S - self.H @ self.P @ self.H.T  # effective R, used by the Joseph update
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        # Joseph form keeps P symmetric positive definite under rounding.
        I_KH = np.eye(4) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        return self.x.copy()

    def run(self, gyro: np.ndarray, accel: np.ndarray) -> np.ndarray:
        self.x[:2] = accel_tilt(accel[:1])[0]
        return np.array([self.step(w, f) for w, f in zip(gyro, accel)])

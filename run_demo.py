"""Compare four roll/pitch estimators on the same simulated IMU run.

    python run_demo.py            # prints RMSE table, saves figures to docs/
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from imu_fusion import (AttitudeEKF, ImuSpec, accel_tilt, complementary,
                        gyro_integration, simulate)

DOCS = Path(__file__).parent / "docs"
COLORS = {"Gyro only": "tab:gray", "Accel only": "tab:orange",
          "Complementary": "tab:green", "EKF, no NIS gate": "tab:cyan", "EKF": "tab:blue"}


def run(seed: int = 0):
    spec = ImuSpec()
    data = simulate(spec=spec, rng=seed)
    dt = 1.0 / spec.fs
    ekf = AttitudeEKF(dt, gyro_noise=np.deg2rad(spec.gyro_noise_dps),
                      bias_walk=np.deg2rad(spec.gyro_bias_walk_dps_rt_s))
    ekf_states = ekf.run(data.gyro, data.accel)
    ekf_ungated = AttitudeEKF(dt, gyro_noise=np.deg2rad(spec.gyro_noise_dps),
                              bias_walk=np.deg2rad(spec.gyro_bias_walk_dps_rt_s), nis_gate=None)
    estimates = {
        "Gyro only": gyro_integration(data.gyro, dt, data.euler_true[0, :2]),
        "Accel only": accel_tilt(data.accel),
        "Complementary": complementary(data.gyro, data.accel, dt),
        "EKF, no NIS gate": ekf_ungated.run(data.gyro, data.accel)[:, :2],
        "EKF": ekf_states[:, :2],
    }
    return data, estimates, ekf_states


def rmse_deg(est: np.ndarray, truth: np.ndarray, mask=None) -> np.ndarray:
    err = np.rad2deg((est - truth + np.pi) % (2 * np.pi) - np.pi)
    if mask is not None:
        err = err[mask]
    return np.sqrt((err ** 2).mean(axis=0))


def report(data, estimates) -> None:
    truth = data.euler_true[:, :2]
    print(f"{'Method':<17} {'Roll RMSE':>10} {'Pitch RMSE':>11} {'Roll, manoeuvres':>17} {'Pitch, manoeuvres':>18}")
    for name, est in estimates.items():
        a = rmse_deg(est, truth)
        b = rmse_deg(est, truth, data.disturbed)
        print(f"{name:<17} {a[0]:9.2f}° {a[1]:10.2f}° {b[0]:16.2f}° {b[1]:17.2f}°")


def plot(data, estimates, ekf_states) -> None:
    DOCS.mkdir(exist_ok=True)
    t, truth = data.t, np.rad2deg(data.euler_true[:, :2])

    def shade(ax):
        edges = np.flatnonzero(np.diff(data.disturbed.astype(int)))
        for a, b in zip(edges[::2], edges[1::2]):
            ax.axvspan(t[a], t[b], color="tab:red", alpha=0.08, lw=0)

    # 1. Roll and pitch, all methods
    fig, axs = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
    for i, (ax, label) in enumerate(zip(axs, ("Roll", "Pitch"))):
        shade(ax)
        ax.plot(t, truth[:, i], color="black", lw=1.6, label="Truth")
        for name, est in estimates.items():
            if name == "EKF, no NIS gate":
                continue
            ax.plot(t, np.rad2deg(est[:, i]), color=COLORS[name], lw=0.8, alpha=0.85, label=name)
        ax.set_ylabel(f"{label}, deg")
        ax.grid(alpha=0.3)
    axs[0].set_title("Attitude estimates (shaded = linear-acceleration manoeuvres)")
    axs[0].legend(ncol=5, fontsize=8, loc="upper right")
    axs[0].set_ylim(-60, 60)
    axs[1].set_ylim(-45, 45)
    axs[1].set_xlabel("Time, s")
    fig.tight_layout()
    fig.savefig(DOCS / "01_attitude.png", dpi=150)
    plt.close(fig)

    # 2. Error comparison: complementary vs EKF
    fig, axs = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True)
    for i, (ax, label) in enumerate(zip(axs, ("Roll", "Pitch"))):
        shade(ax)
        for name in ("Accel only", "Complementary", "EKF, no NIS gate", "EKF"):
            ax.plot(t, np.rad2deg(estimates[name][:, i]) - truth[:, i],
                    color=COLORS[name], lw=0.8, label=name)
        ax.set_ylabel(f"{label} error, deg")
        ax.set_ylim(-12, 12)
        ax.grid(alpha=0.3)
    axs[0].set_title("Estimation error")
    axs[0].legend(ncol=4, fontsize=8, loc="upper right")
    axs[1].set_xlabel("Time, s")
    fig.tight_layout()
    fig.savefig(DOCS / "02_error.png", dpi=150)
    plt.close(fig)

    # 3. Gyro bias estimation
    fig, ax = plt.subplots(figsize=(11, 3.8))
    for i, (axis, c) in enumerate((("p (x)", "tab:blue"), ("q (y)", "tab:purple"))):
        ax.plot(t, np.rad2deg(data.gyro_bias[:, i]), color=c, lw=1.4, ls="--", label=f"True bias {axis}")
        ax.plot(t, np.rad2deg(ekf_states[:, 2 + i]), color=c, lw=0.9, label=f"EKF estimate {axis}")
    ax.set(title="Gyro bias estimated online by the EKF", xlabel="Time, s", ylabel="Bias, deg/s")
    ax.grid(alpha=0.3)
    ax.legend(ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(DOCS / "03_bias.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    data, estimates, ekf_states = run()
    report(data, estimates)
    plot(data, estimates, ekf_states)
    print(f"\nFigures saved to {DOCS}/")

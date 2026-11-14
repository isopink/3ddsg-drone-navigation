"""Quadrotor parameters and a geometric position/attitude controller (numpy only).

Simplified: the controller outputs total thrust and body torques that are applied
directly to the rigid body (no motor/propeller model, no sensor noise). Good for
checking paths and control logic; it does NOT reproduce Flow deck behavior
(optical flow drift, dark-room divergence).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

G = 9.81


@dataclass(frozen=True)
class QuadrotorParams:
    mass: float = 0.027  # kg (Crazyflie 2.1)
    inertia: tuple[float, float, float] = (1.4e-5, 1.4e-5, 2.17e-5)  # kg m^2
    radius: float = 0.05  # collision sphere, m
    max_thrust: float = 0.6  # N, total
    max_accel_xy: float = 3.0  # m/s^2, limits tilt
    kp: float = 6.0  # position gain (1/s^2)
    kd: float = 5.0  # velocity gain (1/s)
    att_wn: float = 30.0  # attitude loop natural frequency (rad/s)


def _vee(m: np.ndarray) -> np.ndarray:
    return np.array([m[2, 1], m[0, 2], m[1, 0]])


class GeometricController:
    def __init__(self, params: QuadrotorParams) -> None:
        self.p = params
        j = np.array(params.inertia)
        self.j = j
        self.k_r = j * params.att_wn**2
        self.k_w = 2.0 * j * params.att_wn

    def compute(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        rot: np.ndarray,  # body -> world, 3x3
        ang_vel_body: np.ndarray,
        pos_ref: np.ndarray,
        vel_ref: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        """Return (total thrust N along body z, torque N m in the body frame)."""
        p = self.p
        acc = p.kp * (pos_ref - pos) + p.kd * (vel_ref - vel)
        xy = np.linalg.norm(acc[:2])
        if xy > p.max_accel_xy:
            acc[:2] *= p.max_accel_xy / xy
        force = p.mass * (acc + np.array([0.0, 0.0, G]))
        thrust = float(np.clip(force @ rot[:, 2], 0.0, p.max_thrust))

        b3 = force / max(np.linalg.norm(force), 1e-9)
        b2 = np.cross(b3, np.array([1.0, 0.0, 0.0]))  # yaw reference fixed at 0
        b2 /= max(np.linalg.norm(b2), 1e-9)
        b1 = np.cross(b2, b3)
        rot_d = np.column_stack([b1, b2, b3])

        e_r = 0.5 * _vee(rot_d.T @ rot - rot.T @ rot_d)
        torque = (
            -self.k_r * e_r
            - self.k_w * ang_vel_body
            + np.cross(ang_vel_body, self.j * ang_vel_body)
        )
        return thrust, torque

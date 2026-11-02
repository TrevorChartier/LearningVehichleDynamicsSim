import numpy as np
from sim.vehicles.vehicle import BaseVehicle
from sim.backend_dynamics.base_backend_dynamics import BaseBackendDynamics

class KBM(BaseBackendDynamics):
    def __init__(self, vehicle: BaseVehicle):
        self.vehicle = vehicle

    def step(self, state: np.ndarray, action: np.ndarray, dt: float) -> np.ndarray:

        x, y, yaw, v, roll, pitch, w = state
        accel, steering = action

        g = 9.81

        x_next = x + v * np.cos(yaw) * dt
        y_next = y + v * np.sin(yaw) * dt

        yaw_next = (
            yaw
            + (v / self.vehicle.wheelbase)
            * np.tan(steering)
            * dt
        )

        terrain_accel = -g * np.sin(pitch)
        drag_accel = -self.vehicle.drag_coeff * v

        v_next = v + (
            accel
            + terrain_accel
            + drag_accel
        ) * dt

        return np.array(
            [
                x_next,
                y_next,
                yaw_next,
                v_next,
                roll,
                pitch,
                w,
            ],
            dtype=np.float64,
        )
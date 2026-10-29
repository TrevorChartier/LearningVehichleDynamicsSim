import numpy as np
from src.sim.backend_dynamics.base_backend_dynamics import BaseBackendDynamics

class KBM(BaseBackendDynamics):
    def __init__(self, wheelbase: float = 0.33, drag_coeff: float = 0.0):
        self.wheelbase = wheelbase
        self.drag_coeff = drag_coeff

    def step(self, state: np.ndarray, action: np.ndarray, dt: float) -> np.ndarray:

        x, y, yaw, v, roll, pitch, w = state
        accel, steering = action

        g = 9.81
        
        # 1. Kinematic propagation equations
        x_next = x + v * np.cos(yaw) * dt
        y_next = y + v * np.sin(yaw) * dt
        yaw_next = yaw + (v / self.wheelbase) * np.tan(steering) * dt
        v_next = v + (accel - g * np.sin(pitch) - self.drag_coeff * v) * dt
        
        # Note: Roll and pitch are set to 0.0 here because the Simulator 
        # immediately overwrites indices 4 and 5 based on (x_next, y_next).
        return np.array([x_next, y_next, yaw_next, v_next, 0.0, 0.0, w], dtype=np.float64)
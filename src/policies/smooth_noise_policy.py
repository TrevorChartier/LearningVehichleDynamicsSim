import numpy as np

class SmoothNoisePolicy:
    """Generates temporally correlated actions using a sum of sine waves."""
    def __init__(self, dt: float, max_accel: float = 3.0, max_steer: float = 0.5):
        self.dt = dt
        self.max_accel = max_accel
        self.max_steer = max_steer
        self.t = 0.0
        
        # Randomize phase shifts for each rollout
        self.phases = np.random.uniform(0, 2 * np.pi, size=4)
        
    def get_action(self) -> np.ndarray:
        # Low frequency components for sustained maneuvers, high freq for corrections
        accel = self.max_accel * (0.6 * np.sin(0.5 * self.t + self.phases[0]) + 
                                  0.4 * np.sin(1.5 * self.t + self.phases[1]))
        steer = self.max_steer * (0.7 * np.sin(0.3 * self.t + self.phases[2]) + 
                                  0.3 * np.sin(1.2 * self.t + self.phases[3]))
        self.t += self.dt
        return np.array([accel, steer], dtype=np.float64)
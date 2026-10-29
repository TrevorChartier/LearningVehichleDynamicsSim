import numpy as np
from src.sim.backend_dynamics.base_backend_dynamics import BaseBackendDynamics
from src.sim.terrain.world import BaseWorld

class Simulator:
    def __init__(self, backend: BaseBackendDynamics, world: BaseWorld, dt: float = 0.05):
        self.backend = backend
        self.dt = dt
        self.world = world
        self.state = np.zeros(7, dtype=np.float64)  # [x, y, yaw, v, roll, pitch, w]


    def reset(self, initial_state: np.ndarray = None) -> np.ndarray:
        if initial_state is not None:
            self.state = np.array(initial_state, dtype=np.float64)
        else:
            self.state = np.zeros(7, dtype=np.float64)

        pitch, roll = self.world.get_pitch_roll(self.state[0], self.state[1])
        self.state[4] = roll
        self.state[5] = pitch

        return self.state.copy()

    def step(self, action: np.ndarray) -> np.ndarray:
        next_state = self.backend.step(self.state, action, self.dt)

        # Query the world with new position to update new pitch, roll
        x_next, y_next = next_state[0], next_state[1]
        pitch_next, roll_next = self.world.get_pitch_roll(x_next, y_next)   
        next_state[4] = roll_next
        next_state[5] = pitch_next

        self.state = next_state
        return self.state.copy()
from abc import ABC, abstractmethod
import numpy as np

class BaseBackendDynamics(ABC):
    """Abstract interface for all dynamics backends (physics or learned)."""

    @abstractmethod
    def step(
        self, 
        state: np.ndarray, 
        action: np.ndarray, 
        dt: float, 
    ) -> np.ndarray:
        """
        Args:
            state: State array [x, y, yaw, v, roll, pitch, w]
            action: Control input array [acceleration, steering_angle]
            dt: Time step in seconds
        Returns:
            next_state: Array [x, y, yaw, v, roll, pitch, w]
        """
        pass
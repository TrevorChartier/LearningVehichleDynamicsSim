from abc import ABC, abstractmethod
import numpy as np

class BaseWorld(ABC):
    """Abstract Base Class for all terrain/world representations."""
    
    @abstractmethod
    def get_height(self, x: float, y: float) -> float:
        """Returns height z at position (x, y)."""
        pass

    @abstractmethod
    def get_pitch_roll(self, x: float, y: float, eps: float = 1e-3) -> tuple[float, float]:
        """Returns (pitch, roll) in radians at position (x, y)."""
        pass


class FlatWorld(BaseWorld):
    """Default flat terrain (zero pitch and roll everywhere)."""
    
    def get_height(self, x: float, y: float) -> float:
        return 0.0

    def get_pitch_roll(self, x: float, y: float, eps: float = 1e-3) -> tuple[float, float]:
        return 0.0, 0.0


class MathExpressionWorld(BaseWorld):
    """Analytical heightmap defined by a math expression z = f(x, y)."""
    
    def __init__(self, expression: str = "0.3 * sin(0.5 * x) + 0.3 * cos(0.5 * y)"):
        self.expression = expression

    def get_height(self, x: float, y: float) -> float:
        sin, cos, exp, sqrt = np.sin, np.cos, np.exp, np.sqrt
        return float(eval(self.expression, {"x": x, "y": y, "sin": sin, "cos": cos, "exp": exp, "sqrt": sqrt}))

    def get_pitch_roll(self, x: float, y: float, eps: float = 1e-3) -> tuple[float, float]:
        z_center = self.get_height(x, y)
        dz_dx = (self.get_height(x + eps, y) - z_center) / eps
        dz_dy = (self.get_height(x, y + eps) - z_center) / eps
        
        pitch = float(-np.arctan(dz_dx))
        roll = float(np.arctan(dz_dy))
        return pitch, roll


class RandomCirclesWorld(BaseWorld):
    """Procedurally generated terrain with random hill bumps."""
    
    def __init__(self, num_hills: int = 10, max_height: float = 0.5, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.centers = rng.uniform(-10, 10, size=(num_hills, 2))
        self.heights = rng.uniform(0.1, max_height, size=num_hills)
        self.radii = rng.uniform(1.0, 3.0, size=num_hills)

    def get_height(self, x: float, y: float) -> float:
        dists = np.hypot(x - self.centers[:, 0], y - self.centers[:, 1])
        # Gaussian radial basis hill elevation
        height = np.sum(self.heights * np.exp(-(dists ** 2) / (2 * self.radii ** 2)))
        return float(height)

    def get_pitch_roll(self, x: float, y: float, eps: float = 1e-3) -> tuple[float, float]:
        z_center = self.get_height(x, y)
        dz_dx = (self.get_height(x + eps, y) - z_center) / eps
        dz_dy = (self.get_height(x, y + eps) - z_center) / eps
        
        pitch = float(-np.arctan(dz_dx))
        roll = float(np.arctan(dz_dy))
        return pitch, roll
from abc import ABC
from dataclasses import dataclass

class BaseVehicle(ABC):
    """Abstract Base Class for vehicle parameter definitions."""
    pass

@dataclass
class KBMVehicle(BaseVehicle):
    """
    Physical parameters for the Kinematic Bicycle Model.
    """
    wheelbase: float = 0.33
    drag_coeff: float = 0.0
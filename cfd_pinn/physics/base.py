import torch
from abc import ABC, abstractmethod

class PhysicsSystem(ABC):

    @abstractmethod
    def physics_residuals(self, coordinates: torch.Tensor, prediction: torch.Tensor, nut_values: torch.Tensor):
        raise NotImplementedError

    @staticmethod
    def gradient(value: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
        return torch.autograd.grad(
            value, coordinates,
            grad_outputs=torch.ones_like(value),
            create_graph=True,
            retain_graph=True
        )[0]
